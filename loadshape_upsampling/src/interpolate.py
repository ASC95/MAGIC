#!/usr/bin/env python


'''
- Interpolate low-resolution load shapes into high-resolution load shapes by combining real appliance shapelets under an energy budget 

- Usage:
    python interpolate.py [--per-unit] [--load NAME [--load NAME ...]] [--list] [--validate]
        - By default, every physical load in the feeder is interpolated and summed into a higher-resolution load shape for the entire feeder
        - Loads that already have a complete csv in the output directory are skipped. Delete files from the output directory to redo their
          interpolations
            --per-unit: interpolate the per-unit load shapes instead of the physical loads. No feeder-wide load shape is created because summing the
                per-unit load shapes doesn't create a physically meaningful result
            --load: interpolate only the specified load shape(s)
            --list: show what loads are going to be interpolated without actually interpolating
            --validate: validate the upsampling process instead of interpolating
'''


import argparse
import concurrent.futures
import os
import re
import shutil
import time

import numpy as np
import pandas as pd

import appliance_data
import composer
import config
import metrics
import plot_utils
import shapelets
import target_data


def main():
    parser = argparse.ArgumentParser(description='Compose higher-resolution load shapes from appliance shapelets')
    parser.add_argument('--per-unit', action='store_true', dest='per_unit')
    parser.add_argument('--load', action='append', default=None)
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--validate', action='store_true')
    args = parser.parse_args()
    if args.validate:
        _validate(args)
    else:
        _interpolate_smartds(args)


def _build_appliances(output_seconds, exclude_rae_house=None):
    '''
    - Load every active appliance's shapelet library and create a composer.Appliance with patterns resampled to the
      output resolution

    :param output_seconds: the composition's output resolution
    :type output_seconds: int
    :param exclude_rae_house: drop entries sourced from this RAE house
    :type exclude_rae_house: int
    :rtype: list
    '''
    appliances = []
    excluded = []
    for name in config.ACTIVE_APPLIANCES:
        app_def = config.APPLIANCES[name]
        if exclude_rae_house is not None and app_def['source'] == 'rae' \
                and app_def['house'] == exclude_rae_house:
            excluded.append(name)
            continue
        path = os.path.join(config.LIBRARY_DIR, f'{name}.npz')
        if not os.path.isfile(path):
            raise FileNotFoundError(f'{path} not found - run train.py first')
        patterns, dt_seconds = shapelets.load_library(path)
        patterns = shapelets.resample_patterns(patterns, dt_seconds, output_seconds)
        app_cfg = config.APPLIANCES[name]
        appliances.append(composer.Appliance(
            name=name,
            patterns=patterns,
            weight=app_cfg['weight'],
            num_instances=app_cfg['num_instances'],
            amplitude_jitter=app_cfg['amplitude_jitter'],
            valid_daily_windows=[tuple(w) for w in app_cfg['valid_daily_windows']],
            valid_seasons=[tuple(s) for s in app_cfg['valid_seasons']]))
    if excluded:
        print(f'Did not use appliances from truth house {exclude_rae_house}: {", ".join(excluded)}')
    return appliances


def _validate(args):
    '''
    - Reconstruct the main meter of a house from its own binned intervals of that main meter and score the reconstruction
    '''
    truth_source = config.VALIDATE_TRUTH_SOURCE
    truth_house = {'rae_house2_mains': 2, 'rae_house1_mains': 1}[truth_source]
    print(f'Loading truth: {truth_source}...')
    series, truth_dt = appliance_data.rae_series(truth_house, ['mains']), 1
    if config.OUTPUT_SECONDS != truth_dt:
        raise ValueError(f'output_seconds ({config.OUTPUT_SECONDS}) must equal the truth resolution '
                         f'({truth_dt}) for validation')
    series = appliance_data.longest_clean_stretch(series, truth_dt)
    n_keep = min(len(series), config.VALIDATE_WINDOW_DAYS * 86400 // truth_dt)
    series = series.iloc[:n_keep]
    truth = series.to_numpy(dtype=np.float64)
    per = config.CONSTRAINT_SECONDS // config.OUTPUT_SECONDS
    target, trimmed = composer.bin_means(truth, per)
    truth = truth[:len(target) * per]
    print(f'Truth: {len(truth)} samples @ {truth_dt}s from {series.index[0]} '
          f'({len(target)} intervals of {config.CONSTRAINT_SECONDS}s, {trimmed} samples trimmed)')
    appliances = _build_appliances(config.OUTPUT_SECONDS, exclude_rae_house=truth_house)
    start_time = time.perf_counter()
    result = composer.compose(
        target, config.CONSTRAINT_SECONDS, config.OUTPUT_SECONDS, appliances,
        seed=composer.per_load_seed(config.RANDOM_SEED, 'validate'), start_timestamp=series.index[0],
        fill_fraction=config.FILL_FRACTION, base_quantile=config.BASE_QUANTILE,
        base_window_seconds=config.BASE_WINDOW_SECONDS,
        max_consecutive_rejects=config.MAX_CONSECUTIVE_REJECTS, max_events=config.MAX_EVENTS,
        padding_seconds=config.COMPOSE_PADDING_SECONDS, base_floor_w=config.BASE_FLOOR_W)
    elapsed = time.perf_counter() - start_time
    print(f'Composed in {elapsed:.1f}s: {result.diagnostics}')
    out_dir = os.path.join(config.OUTPUTS_DIR, 'validation', truth_source)
    os.makedirs(out_dir, exist_ok=True)
    staircase = np.repeat(target, per)
    table = metrics.reconstruction_metrics(truth, result.total, staircase, config.OUTPUT_SECONDS)
    table.insert(0, 'truth', truth_source)
    table.to_csv(os.path.join(out_dir, 'metrics.csv'), index=False)
    print(table.T.to_string(header=False))
    n_plot = min(len(truth), config.PLOT_WINDOW_DAYS * 86400 // config.OUTPUT_SECONDS)
    index = series.index[:n_plot]
    fig = plot_utils.figure_validation(index, truth[:n_plot], result.total[:n_plot],
                                       f'validate: {truth_source}')
    fig.write_html(os.path.join(out_dir, 'overlay.html'))
    _write_composition_figure(result, appliances, index, staircase, out_dir, 'composition',
                              f'validate: {truth_source} composition', load_name=truth_source, original=truth)
    print(f'Results in {out_dir}')


def _interpolate_smartds(args):
    '''
    - Interpolate the configured SMART-DS circuit's 15-minute shapes to the output resolution
    '''
    df, kw_bases, load_types = target_data.smartds_load_shapes('pu' if args.per_unit else 'loads')
    if args.list:
        for name in df.columns:
            print(f'{name}  ({load_types[name]}, {kw_bases[name]:.2f} kW)')
        print(f'{len(df.columns)} shapes ({"per-unit" if args.per_unit else "physical"})')
        return
    names = _select_names(df.columns, args, strip_split_suffix=not args.per_unit)
    if config.CONSTRAINT_SECONDS < 900:
        raise ValueError('constraint_seconds < 900 cannot be honored: 15-minute input has nothing finer to preserve')
    rebin = config.CONSTRAINT_SECONDS // 900
    out_dir = os.path.join(config.OUTPUTS_DIR, 'application', 'smartds', config.SMARTDS_CIRCUIT)
    os.makedirs(out_dir, exist_ok=True)
    full_run = args.load is None
    aggregate_run = full_run and not args.per_unit
    tasks = []
    done = []
    skipped = 0
    estimated_rows = 0
    for name in names:
        target_w = df[name].to_numpy(dtype=np.float64) * 1000.0
        if rebin > 1:
            target_w, _ = composer.bin_means(target_w, rebin)
        csv_divisor = 1000.0 * kw_bases[name] if args.per_unit else 1000.0
        csv_path = os.path.join(out_dir, f'{name}.csv')
        expected_rows = len(target_w) * (config.CONSTRAINT_SECONDS // config.OUTPUT_SECONDS)
        if _csv_is_complete(csv_path, expected_rows):
            skipped += 1
            if aggregate_run:
                done.append((name, csv_path, csv_divisor))
            continue
        estimated_rows += expected_rows
        original_w = None
        if not full_run and 900 % config.OUTPUT_SECONDS == 0:
            plot_rows = config.PLOT_WINDOW_DAYS * 86400 // 900
            original_w = np.repeat(df[name].to_numpy(dtype=np.float64)[:plot_rows] * 1000.0,
                                   900 // config.OUTPUT_SECONDS)
        tasks.append((name, target_w, df.index[0], out_dir, csv_divisor, config.RANDOM_SEED, not full_run,
                      original_w, config.BASE_FLOOR_W, aggregate_run))
    if aggregate_run and (tasks or done):
        estimated_rows += expected_rows
    if skipped:
        print(f'{skipped} of {len(names)} loads already complete, composing {len(tasks)} '
              f'(delete csvs from the output directory to redo them)')
    # - Abort before composing anything if the output will not fit
    estimated_bytes = estimated_rows * 8
    free_bytes = shutil.disk_usage(out_dir).free
    if estimated_bytes > free_bytes:
        raise SystemExit(f'Estimated output ~{estimated_bytes / 1e9:.1f} GB will not fit in the '
                         f'{free_bytes / 1e9:.1f} GB free at {out_dir}. Free up space or delete existing '
                         'outputs, then re-run')
    feeder_total = None

    def _accumulate(total):
        nonlocal feeder_total
        if total is not None:
            if feeder_total is None:
                feeder_total = np.zeros(len(total), dtype=np.float64)
            feeder_total += total

    workers = max(1, min(config.PARALLEL_WORKERS, len(tasks) + len(done)))
    if workers == 1:
        appliances = _build_appliances(config.OUTPUT_SECONDS) if tasks else None
        for task in tasks:
            result = _compose_and_write(task[0], task[1], appliances, *task[2:-1])
            if task[-1]:
                _accumulate(result.total)
        for entry in done:
            _accumulate(_worker_read(*entry))
    else:
        print(f'Composing {len(tasks)} loads (+{len(done)} read back) across {workers} workers...')
        with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_worker_init) as pool:
            futures = {pool.submit(_worker_compose, *task) for task in tasks}
            futures |= {pool.submit(_worker_read, *entry) for entry in done}
            for future in concurrent.futures.as_completed(futures):
                # - Drop every reference to a future before moving to the next one. A completed future is a ~126 MB array, so keeping all of the futures in
                #   the set grew the parent process until it crashed
                futures.discard(future)
                _accumulate(future.result())
                del future
    if feeder_total is not None:
        _write_feeder_aggregate(df, names, feeder_total, rebin, out_dir)
    print(f'Results in {out_dir}')


def _write_feeder_aggregate(df, names, feeder_total, rebin, out_dir):
    '''
    - Write the upsampled feeder csv and the upsampled feeder html plot

    :param df: the 15-minute kW table
    :type df: DataFrame
    :param names: the composed customer names
    :type names: list
    :param feeder_total: the summed composed watts
    :type feeder_total: ndarray
    :param rebin: how many 900-s readings each constraint interval covers
    :type rebin: int
    :param out_dir: where aggregate.csv / aggregate.html go
    :type out_dir: str
    :rtype: None
    '''
    csv_path = os.path.join(out_dir, 'aggregate.csv')
    pd.Series(feeder_total / 1000.0).to_csv(csv_path, index=False, header=False, float_format='%.2f')
    print(f'Feeder csv: {csv_path}')
    original_sum_w = df[names].sum(axis=1).to_numpy(dtype=np.float64) * 1000.0
    target_sum_w = composer.bin_means(original_sum_w, rebin)[0] if rebin > 1 else original_sum_w
    fig = plot_utils.figure_feeder(
        df.index[0], original_sum_w, target_sum_w, config.CONSTRAINT_SECONDS, feeder_total,
        config.OUTPUT_SECONDS,
        f'{config.SMARTDS_CIRCUIT}: feeder of {len(names)} customers composed at {config.OUTPUT_SECONDS}s',
        config.AGGREGATE_PLOT_MAX_POINTS)
    fig.write_html(os.path.join(out_dir, 'aggregate.html'))
    print(f'Feeder figure: {os.path.join(out_dir, "aggregate.html")}')


def _csv_is_complete(path, expected_rows):
    '''
    - Check if a previously written csv is complete or if the process stopped midway while writing it

    :param path: the <name>.csv to check
    :type path: str
    :param expected_rows: the exact row count a complete file must have
    :type expected_rows: int
    :rtype: bool
    '''
    if not os.path.isfile(path):
        return False
    if os.path.getsize(path) < expected_rows * 5:
        return False
    count = 0
    with open(path, 'rb') as f:
        while chunk := f.read(1 << 22):
            count += chunk.count(b'\n')
    return count == expected_rows


def _worker_read(name, path, csv_divisor):
    '''
    - Read watts from a csv via a worker process

    :rtype: ndarray
    '''
    values = pd.read_csv(path, header=None, dtype=np.float64)[0].to_numpy()
    return values * csv_divisor


_worker_appliances = None


def _worker_init():
    '''
    - Initialize a worker process by having it load the library of appliance shapelets from disk. Each worker must load the appliance shapelets on its own
      because placing appliance events is a CPU-bound task and we can't use threads

    :rtype: None
    '''
    global _worker_appliances
    _worker_appliances = _build_appliances(config.OUTPUT_SECONDS)


def _worker_compose(name, target_w, start_timestamp, out_dir, csv_divisor, seed, make_figure, original_w=None,
                    base_floor_w=None, return_total=False):
    '''
    - Upsample one physical feeder load or one per-unit load with this worker process. When a worker process is finished, it writes the higher-resolution
      load shape to disk

    :param name: the load being composed
    :type name: str
    :param target_w: mean watts per constraint interval
    :type target_w: ndarray
    :param start_timestamp: time of the first sample
    :type start_timestamp: Timestamp
    :param out_dir: where the csvs/figures go
    :type out_dir: str
    :param csv_divisor: divide composed watts by this before writing (1000*kW = pu, 1000 = kW)
    :type csv_divisor: float
    :param seed: the run seed
    :type seed: int
    :param make_figure: whether to write the composition html
    :type make_figure: bool
    :param original_w: the source's own watts at the output resolution
    :type original_w: ndarray
    :param base_floor_w: absolute cap on the base floor
    :type base_floor_w: float
    :param return_total: whether the parent process needs this load's composed watts returned back
    :type return_total: bool
    :return: the load's composed watts at output resolution when return_total is True, else None
    :rtype: ndarray
    '''
    result = _compose_and_write(name, target_w, _worker_appliances, start_timestamp, out_dir, csv_divisor,
                                seed, make_figure, original_w, base_floor_w)
    return result.total if return_total else None


def _compose_and_write(name, target_w, appliances, start_timestamp, out_dir, csv_divisor, seed, make_figure,
                       original_w=None, base_floor_w=None):
    '''
    - Perform the actual upsampling process with the worker process

    :param name: the load being composed
    :type name: str
    :param target_w: mean watts per constraint interval
    :type target_w: ndarray
    :param appliances: composer.Appliance objects at the output resolution
    :type appliances: list
    :param start_timestamp: time of the first sample
    :type start_timestamp: Timestamp
    :param out_dir: where the csvs/figures go
    :type out_dir: str
    :param csv_divisor: divide composed watts by this before writing (1000*kW = pu, 1000 = kW)
    :type csv_divisor: float
    :param seed: the run seed
    :type seed: int
    :param make_figure: whether to write the composition html
    :type make_figure: bool
    :param original_w: the source's own watts
    :type original_w: ndarray
    :param base_floor_w: absolute floor cap for house
    :type base_floor_w: float
    :return: the ComposeResult
    :rtype: ComposeResult
    '''
    start_time = time.perf_counter()
    result = composer.compose(
        target_w, config.CONSTRAINT_SECONDS, config.OUTPUT_SECONDS, appliances,
        seed=composer.per_load_seed(seed, name), start_timestamp=start_timestamp,
        fill_fraction=config.FILL_FRACTION, base_quantile=config.BASE_QUANTILE,
        base_window_seconds=config.BASE_WINDOW_SECONDS,
        max_consecutive_rejects=config.MAX_CONSECUTIVE_REJECTS, max_events=config.MAX_EVENTS,
        padding_seconds=config.COMPOSE_PADDING_SECONDS, base_floor_w=base_floor_w)
    elapsed = time.perf_counter() - start_time
    d = result.diagnostics
    print(f'{name}: {d["n_events"]} events, event share {d["event_energy_share"]:.2f}, '
          f'max interval error {d["max_interval_error_w"]:.2e} W, {elapsed:.1f}s')
    log = result.log.copy()
    if len(log):
        log['start_time'] = start_timestamp + pd.to_timedelta(log['start_idx'] * config.OUTPUT_SECONDS,
                                                              unit='s')
    log.to_csv(os.path.join(out_dir, f'{name}_events.csv'), index=False)
    csv_path = os.path.join(out_dir, f'{name}.csv')
    pd.Series(result.total / csv_divisor).to_csv(csv_path + '.part',
                                                 index=False, header=False, float_format='%.2f')
    os.replace(csv_path + '.part', csv_path)
    if make_figure:
        per = config.CONSTRAINT_SECONDS // config.OUTPUT_SECONDS
        staircase = np.repeat(target_w, per)
        n_plot = min(len(result.total), config.PLOT_WINDOW_DAYS * 86400 // config.OUTPUT_SECONDS)
        index = pd.date_range(start_timestamp, periods=n_plot, freq=f'{config.OUTPUT_SECONDS}s')
        _write_composition_figure(result, appliances, index, staircase, out_dir, name,
                                  f'{name}: composed at {config.OUTPUT_SECONDS}s '
                                  f'(means preserved per {config.CONSTRAINT_SECONDS}s)',
                                  load_name=name, original=original_w)
    return result


def _write_composition_figure(result, appliances, index, staircase, out_dir, stem, title, load_name,
                              original=None):
    '''
    - Write the stacked composition figure for the first len(index) samples of a result

    :rtype: None
    '''
    n = len(index)
    traces = composer.appliance_traces(result.log, appliances, 0, n)
    fig = plot_utils.figure_composition(index, staircase[:n], traces, result.base[:n], result.total[:n],
                                        title, load_name,
                                        original[:n] if original is not None else None)
    fig.write_html(os.path.join(out_dir, f'{stem}.html'))


def _select_names(available, args, strip_split_suffix):
    '''
    - Resolve --load into a list of target names, accounting for _1/_2 spellings for merged customers. No --load = every available name

    :rtype: list
    '''
    if not args.load:
        return list(available)
    names = []
    for raw in args.load:
        name = raw
        if strip_split_suffix and name not in available:
            name = re.sub(r'_[12]$', '', name)
        if name not in available:
            raise SystemExit(f'{raw} not found - use --list to see available names')
        names.append(name)
    return names




if __name__ == '__main__':
    main()
