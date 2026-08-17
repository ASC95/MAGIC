#!/usr/bin/env python


'''
- Read the appliance events that shapelet libraries are built from
'''


import os

import numpy as np
import pandas as pd

import config


# - The RAE dataset was recorded in the Vancouver area
LOCAL_TZ = 'America/Vancouver'

_RAE_POWER_FILES = {
    1: ['house1_power_blk1.csv', 'house1_power_blk2.csv'],
    2: ['house2_power_blk1.csv'],
}

def rae_series(house, channels):
    '''
    - Return the summed watts of the given RAE meter as a 1-second Series

    :param house: which RAE house (1 or 2)
    :type house: int
    :param channels: power_blk column names to sum
    :type channels: list
    :return: watts at 1 s
    :rtype: Series
    '''
    assert isinstance(house, int)
    assert isinstance(channels, list)
    frames = [pd.read_csv(os.path.join(config.RAE_DIR, filename),
                          usecols=['unix_ts'] + sorted(set(channels)),
                          dtype={c: np.float32 for c in channels},
                          index_col='unix_ts')
              for filename in _RAE_POWER_FILES[house]]
    df = pd.concat(frames).fillna(0.0)
    values = df[channels].sum(axis=1)
    index = pd.to_datetime(df.index, unit='s', utc=True).tz_convert(LOCAL_TZ)
    return pd.Series(values.to_numpy(dtype=np.float32), index=index, name='+'.join(channels))


def appliance_series(app_cfg):
    '''
    - Load one appliance definition

    :param app_cfg: the appliance's yaml dict (source + rae house/channels)
    :type app_cfg: dict
    :return: series of watts
    :rtype: tuple
    '''
    assert isinstance(app_cfg, dict)
    if app_cfg['source'] == 'rae':
        return rae_series(app_cfg['house'], list(app_cfg['channels'])), 1
    raise ValueError(f"Unknown appliance source: {app_cfg['source']}")


def split_on_gaps(series, dt_seconds):
    '''
    - Split a single load shape containing meter outages into a list of series, where each series is a non-interrupted set of appliance measurements
      containing 0 or more events

    :param series: watts with a DatetimeIndex
    :type series: Series
    :param dt_seconds: the load shape's resolution
    :type dt_seconds: int
    :return: list of contiguous sub-Series, longest stretches included as-is (order preserved)
    :rtype: list
    '''
    assert isinstance(series, pd.Series)
    assert isinstance(dt_seconds, int)
    if len(series) == 0:
        return []
    step = (series.index[1:] - series.index[:-1]) / pd.Timedelta(seconds=1)
    breaks = np.flatnonzero(step.to_numpy() != dt_seconds)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks + 1, [len(series)]))
    return [series.iloc[s:e] for s, e in zip(starts, ends)]


def longest_clean_stretch(series, dt_seconds, max_gap_seconds=60):
    '''
    - Return the longest stretch of a load shape whose internal gaps are all <= max_gap_seconds, with those small
      gaps forward-filled onto a complete index

    :param series: watts with a DatetimeIndex
    :type series: Series
    :param dt_seconds: the load shape's resolution
    :type dt_seconds: int
    :param max_gap_seconds: the largest hole that may be forward-filled over
    :type max_gap_seconds: int
    :return: gap-free watts on a complete dt_seconds grid
    :rtype: Series
    '''
    assert isinstance(series, pd.Series)
    step = (series.index[1:] - series.index[:-1]) / pd.Timedelta(seconds=1)
    big = np.flatnonzero(step.to_numpy() > max_gap_seconds)
    starts = np.concatenate(([0], big + 1))
    ends = np.concatenate((big + 1, [len(series)]))
    spans = ends - starts
    s, e = starts[np.argmax(spans)], ends[np.argmax(spans)]
    stretch = series.iloc[s:e]
    full_index = pd.date_range(stretch.index[0], stretch.index[-1], freq=f'{dt_seconds}s')
    return stretch.reindex(full_index).ffill()
