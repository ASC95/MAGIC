#!/usr/bin/env python


'''
- Build the shapelet library for every active appliance: read the real load shape, cut it into on/off events, save the events as
  outputs/library/<appliance>.npz, and report extraction statistics
- Usage:
    python train.py [--plot] [--appliance NAME ...]

    --plot            also write per-appliance html figures: the raw load shape vs extracted-shapelet timeline 
    --appliance NAME  build only the named appliance(s) 
'''


import argparse
import os
import time

import numpy as np
import pandas as pd

import appliance_data
import config
import plot_utils
import shapelets


def main():
    parser = argparse.ArgumentParser(description='Build appliance shapelet libraries')
    parser.add_argument('--plot', action='store_true')
    parser.add_argument('--appliance', action='append', default=None)
    args = parser.parse_args()
    names = args.appliance if args.appliance else config.ACTIVE_APPLIANCES
    os.makedirs(config.LIBRARY_DIR, exist_ok=True)
    summary_rows = []
    for name in names:
        if name not in config.APPLIANCES:
            raise KeyError(f'{name} is not defined in config/appliances.yaml')
        app_cfg = config.APPLIANCES[name]
        print(f'--- {name} ({app_cfg["source"]}) ---')
        start_time = time.perf_counter()
        series, dt_seconds = appliance_data.appliance_series(app_cfg)
        spans = []
        for stretch in appliance_data.split_on_gaps(series, dt_seconds):
            offset = series.index.get_loc(stretch.index[0])
            spans.extend((offset + s, p) for s, p in shapelets.extract_shapelets(
                stretch.to_numpy(), dt_seconds,
                start_w=app_cfg['start_w'], end_w=app_cfg['end_w'],
                min_seconds=app_cfg['min_seconds'], max_seconds=app_cfg['max_seconds'],
                max_w=app_cfg['max_w']))
        if not spans:
            raise RuntimeError(f'no shapelets extracted for {name} - check its thresholds')
        patterns = [p for _, p in spans]
        stats = {'appliance': name, 'dt_s': dt_seconds, **shapelets.library_stats(patterns, dt_seconds),
                 'seconds': round(time.perf_counter() - start_time, 1)}
        summary_rows.append(stats)
        print(pd.DataFrame([stats]).to_string(index=False))
        shapelets.save_library(os.path.join(config.LIBRARY_DIR, f'{name}.npz'), patterns, dt_seconds)
        if args.plot:
            _write_figures(name, series, dt_seconds, spans, patterns)
    summary = pd.DataFrame(summary_rows)
    summary_path = os.path.join(config.LIBRARY_DIR, 'library_summary.csv')
    if args.appliance and os.path.isfile(summary_path):
        old = pd.read_csv(summary_path)
        summary = pd.concat([old[~old['appliance'].isin(summary['appliance'])], summary], ignore_index=True)
    summary.to_csv(summary_path, index=False)
    print(f'\nLibrary written to {config.LIBRARY_DIR}')
    print(summary.to_string(index=False))


def _write_figures(name, series, dt_seconds, spans, patterns):
    '''
    - Write the two per-appliance diagnostic figures next to the library

    :param name: the appliance name
    :type name: str
    :param series: the full appliance load shape
    :type series: Series
    :param dt_seconds: the load shape resolution
    :type dt_seconds: int
    :param spans: (global start index, pattern) pairs from extraction
    :type spans: list
    :param patterns: the extracted patterns
    :type patterns: list
    :rtype: None
    '''
    figures_dir = os.path.join(config.LIBRARY_DIR, 'figures')
    os.makedirs(figures_dir, exist_ok=True)
    n = min(len(series), config.TRAIN_PLOT_HOURS * 3600 // dt_seconds)
    window = series.iloc[:n]
    silhouette = np.zeros(n, dtype=np.float32)
    for start, pattern in spans:
        if start < n:
            k = min(len(pattern), n - start)
            silhouette[start:start + k] = pattern[:k]
    fig = plot_utils.figure_library_timeline(window.index, window.to_numpy(), silhouette,
                                             f'{name}: load shape vs extracted shapelets')
    fig.write_html(os.path.join(figures_dir, f'{name}_timeline.html'))
    fig = plot_utils.figure_library_samples(patterns, dt_seconds, f'{name}: library samples')
    fig.write_html(os.path.join(figures_dir, f'{name}_samples.html'))


if __name__ == '__main__':
    main()
