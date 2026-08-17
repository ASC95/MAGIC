#!/usr/bin/env python


'''
- This module cuts an appliance load shape into shapelets (self-contained on/off events) and stores them as a reusable library
'''


import numpy as np
import pandas as pd


def extract_shapelets(values, dt_seconds, start_w, end_w, min_seconds, max_seconds, max_w):
    '''
    - Cut one gap-free stretch of a load shape into shapelets

    :param values: watts of ONE contiguous stretch
    :type values: ndarray
    :param dt_seconds: the load shape's resolution
    :type dt_seconds: int
    :param start_w: rising threshold that opens an event 
    :type start_w: float
    :param end_w: falling threshold that closes an event
    :type end_w: float
    :param min_seconds: drop events shorter than this
    :type min_seconds: float
    :param max_seconds: drop events longer than this
    :type max_seconds: float
    :param max_w: drop events containing any sample above this
    :type max_w: float
    :return: list of pairs at the load shape's resolution
    :rtype: list
    '''
    assert isinstance(values, np.ndarray)
    if start_w < end_w:
        raise ValueError('start_w must be >= end_w for hysteresis')
    trigger = np.full(len(values), np.nan, dtype=np.float32)
    trigger[values >= start_w] = 1.0
    trigger[values <= end_w] = 0.0
    state = pd.Series(trigger).ffill().fillna(0.0).to_numpy()
    edges = np.diff(np.concatenate(([0.0], state, [0.0])))
    starts = np.flatnonzero(edges == 1.0)
    ends = np.flatnonzero(edges == -1.0)  # exclusive end index of each run
    patterns = []
    for s, e in zip(starts, ends):
        if s == 0 or e == len(values):
            continue
        duration = (e - s) * dt_seconds
        if not (min_seconds <= duration <= max_seconds):
            continue
        pattern = values[s:e]
        if max_w is not None and pattern.max() > max_w:
            continue
        patterns.append((int(s), pattern.astype(np.float32)))
    return patterns


def library_stats(patterns, dt_seconds):
    '''
    - Summarize a pattern list for the training report

    :param patterns: extracted pattern arrays
    :type patterns: list
    :param dt_seconds: the patterns' resolution
    :type dt_seconds: int
    :rtype: dict
    '''
    assert isinstance(patterns, list)
    durations = np.array([len(p) * dt_seconds for p in patterns])
    energies = np.array([p.sum() * dt_seconds / 3600.0 for p in patterns])  # Wh
    peaks = np.array([p.max() for p in patterns])
    return {
        'count': len(patterns),
        'duration_s_p10': np.percentile(durations, 10),
        'duration_s_p50': np.percentile(durations, 50),
        'duration_s_p90': np.percentile(durations, 90),
        'energy_wh_p50': np.percentile(energies, 50),
        'peak_w_p50': np.percentile(peaks, 50),
        'peak_w_p90': np.percentile(peaks, 90),
    }


def save_library(path, patterns, dt_seconds):
    '''
    - Save a pattern list as <name>.npz: values + lengths + dt_seconds

    :param path: the .npz file to write
    :type path: str
    :param patterns: extracted pattern arrays
    :type patterns: list
    :param dt_seconds: the patterns' resolution
    :type dt_seconds: int
    :rtype: None
    '''
    assert isinstance(patterns, list)
    values = np.concatenate(patterns)
    lengths = np.array([len(p) for p in patterns], dtype=np.int64)
    np.savez_compressed(path, values=values.astype(np.float32), lengths=lengths,
                        dt_seconds=np.int64(dt_seconds))


def load_library(path):
    '''
    - Load a library saved by save_library()

    :param path: the .npz file to read
    :type path: str
    :return: (list of float32 pattern arrays, resolution in seconds)
    :rtype: tuple
    '''
    with np.load(path) as data:
        values = data['values']
        lengths = data['lengths']
        dt_seconds = int(data['dt_seconds'])
    bounds = np.concatenate(([0], np.cumsum(lengths)))
    patterns = [values[s:e] for s, e in zip(bounds[:-1], bounds[1:])]
    return patterns, dt_seconds


def resample_patterns(patterns, dt_from, dt_to):
    '''
    - Convert a pattern list from its library resolution to the composition's output resolution

    :param patterns: pattern arrays at dt_from resolution
    :type patterns: list
    :param dt_from: library resolution in seconds
    :type dt_from: int
    :param dt_to: output resolution in seconds
    :type dt_to: int
    :return: pattern arrays at dt_to resolution
    :rtype: list
    '''
    assert isinstance(patterns, list)
    if dt_from == dt_to:
        return patterns
    if dt_to < dt_from:
        if dt_from % dt_to != 0:
            raise ValueError(f'library dt {dt_from}s is not a multiple of output dt {dt_to}s')
        factor = dt_from // dt_to
        return [np.repeat(p, factor) for p in patterns]
    if dt_to % dt_from != 0:
        raise ValueError(f'output dt {dt_to}s is not a multiple of library dt {dt_from}s')
    factor = dt_to // dt_from
    out = []
    for p in patterns:
        n = len(p) // factor
        if n == 0:
            continue
        out.append(p[:n * factor].reshape(n, factor).mean(axis=1).astype(np.float32))
    return out
