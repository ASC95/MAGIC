#!/usr/bin/env python


'''
- This module loads the load shapes that get interpolated:
'''


import os
import re

import numpy as np
import pandas as pd

import config


def smartds_load_shapes(mode):
    '''
    - Load the configured circuit's customers as 15-minute kW shapes
        - mode 'loads': one column per physical customer (split-phase halves merged), kW
        - mode 'pu': one column per unique profile, scaled by a representative customer's kW (the first customer
          seen using that profile)

    :param mode: 'loads' or 'pu'
    :type mode: str
    :return: DataFrame of kW indexed by 15-minute timestamps, dict of load name -> kW scale,
        dict of load name -> Residential/Commercial/Other)
    :rtype: tuple
    '''
    assert mode in ('pu', 'loads')
    dss_path = os.path.join(config.SMARTDS_DATASET_PATH, 'scenarios', config.SMARTDS_SCENARIO, 'opendss',
                            config.SMARTDS_SUBSTATION, config.SMARTDS_CIRCUIT, 'Loads.dss')
    profiles_dir = os.path.join(config.SMARTDS_DATASET_PATH, 'profiles')
    definitions = _parse_dss_loads(dss_path)
    cache = {}
    for d in definitions:
        name = d['profile']
        if name not in cache:
            path = os.path.join(profiles_dir, f'{name}.csv')
            cache[name] = pd.read_csv(path, header=None).iloc[:, 0].to_numpy(dtype=np.float64)
    customers = {}
    for d in definitions:
        key = re.sub(r'_[12]$', '', d['name'])
        pu = cache[d['profile']]
        if key in customers:
            customers[key]['shape'] = customers[key]['shape'] + pu * d['kw']
            customers[key]['kw'] += d['kw']
        else:
            prefix = d['profile'].lower()
            l_type = 'Residential' if prefix.startswith('res_') else \
                     'Commercial' if prefix.startswith('com_') else 'Other'
            customers[key] = {'shape': pu * d['kw'], 'kw': d['kw'], 'profile': d['profile'], 'type': l_type}
    columns = {}
    kw_bases = {}
    load_types = {}
    if mode == 'pu':
        for c in customers.values():
            key = c['profile']
            if key in columns:
                continue
            columns[key] = c['shape']
            kw_bases[key] = c['kw']
            load_types[key] = c['type']
    else:
        for key, c in customers.items():
            columns[key] = c['shape']
            kw_bases[key] = c['kw']
            load_types[key] = c['type']
    n_rows = len(next(iter(columns.values())))
    index = pd.date_range(f'{config.SMARTDS_YEAR}-01-01', periods=n_rows, freq='15min')
    df = pd.DataFrame(columns, index=index)
    return df, kw_bases, load_types


def _parse_dss_loads(dss_path):
    '''
    - Pull (name, kW, yearly profile) out of every New Load line of a Loads.dss

    :param dss_path: the circuit's Loads.dss
    :type dss_path: str
    :rtype: list
    '''
    pattern = re.compile(r'New Load\.(?P<name>[\w]+).*?kW=(?P<kw>[\d.]+).*?yearly=(?P<profile>[\w]+)',
                         re.IGNORECASE)
    definitions = []
    with open(dss_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('!'):
                continue
            match = pattern.search(line)
            if not match:
                raise ValueError(f'unparseable line in {dss_path}: {line}')
            definitions.append({'name': match.group('name'), 'kw': float(match.group('kw')),
                                'profile': match.group('profile')})
    return definitions
