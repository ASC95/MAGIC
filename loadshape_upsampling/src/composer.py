#!/usr/bin/env python


'''
- Create a higher-resolution load shape from appliance shapelets plus a smooth base layer, such that the result reproduces the target's mean power over
  every interval
'''


import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd


@dataclass
class Appliance:
    '''
    - One placeable appliance
    '''
    name: str
    patterns: list                       # list of arrays at output resolution
    weight: float = 1.0                  # relative draw probability during placement
    num_instances: int = 1               # max simultaneously-running copies in one composed load
    amplitude_jitter: float = 0.0        # each placement scaled by uniform(1-j, 1+j)
    valid_daily_windows: list = field(default_factory=list)   # [("HH:MM", "HH:MM"), ...]; empty = always
    valid_seasons: list = field(default_factory=list)         # [("MM-DD", "MM-DD"), ...]; empty = all year


@dataclass
class ComposeResult:
    total: np.ndarray        # base + events, flen = n_intervals * (interval_seconds // output_seconds)
    base: np.ndarray         # the smooth mean-preserving remainder layer
    events: np.ndarray       # the summed placed shapelets
    log: pd.DataFrame        # one row per placed event: appliance, start_idx, length, pattern_idx, scale
    diagnostics: dict        # counts, energy shares, exactness check


def per_load_seed(seed, load_name):
    '''
    - Create a per-load seed so different loads get independent event timing from one config seed

    :param seed: the random_seed
    :type seed: int
    :param load_name: the load being composed
    :type load_name: str
    :rtype: int
    '''
    return zlib.crc32(f'{seed}:{load_name}'.encode())


def bin_means(values, factor):
    '''
    - Divide the lower-resolution (e.g. 15-minute) load shape into intervals, then find the mean of each interval

    :param values: the high-resolution samples
    :type values: ndarray
    :param factor: samples per interval
    :type factor: int
    :return: (interval means, number of trailing samples dropped)
    :rtype: tuple
    '''
    assert isinstance(values, np.ndarray)
    n = len(values) // factor
    trimmed = len(values) - n * factor
    means = values[:n * factor].reshape(n, factor).mean(axis=1)
    return means.astype(np.float64), trimmed


def compose(target_means_w, interval_seconds, output_seconds, appliances, seed, start_timestamp=None,
            fill_fraction=0.9, base_quantile=0.05, base_window_seconds=86400,
            max_consecutive_rejects=200, max_events=500000, padding_seconds=0, base_floor_w=None):
    '''
    - Create a high-resolution shape whose mean over every interval equals target_means_w exactly

    :param target_means_w: the coarse target - mean watts per constraint interval
    :type target_means_w: ndarray
    :param interval_seconds: the constraint interval width (e.g. 900 for 15-minute readings)
    :type interval_seconds: int
    :param output_seconds: the output resolution; must divide interval_seconds
    :type output_seconds: int
    :param appliances: the placeable Appliance objects
    :type appliances: list
    :param seed: rng seed (use per_load_seed() when composing many loads)
    :type seed: int
    :param start_timestamp: time of sample 0. This is required only when any appliance has daily/season windows 
    :type start_timestamp: Timestamp
    :param fill_fraction: share of each interval's above-base energy that events may claim
    :type fill_fraction: float
    :param base_quantile: rolling quantile of the target that stands in for the floor
    :type base_quantile: float
    :param base_window_seconds: width of the rolling window behind the floor quantile
    :type base_window_seconds: int
    :param max_consecutive_rejects: consecutive failed placements that retire an appliance
    :type max_consecutive_rejects: int
    :param max_events: hard cap on placed events. Prevents runaway loops
    :type max_events: int
    :param padding_seconds: warm-up/cool-down span beyond each end and trimmed from the returned
        arrays 
    :type padding_seconds: int
    :param base_floor_w: absolute cap on the floor, in watts 
    :type base_floor_w: float
    :rtype: ComposeResult
    '''
    target = np.asarray(target_means_w, dtype=np.float64)
    assert np.all(np.isfinite(target)), 'target means must be finite - fill or trim gaps first'
    if interval_seconds % output_seconds != 0:
        raise ValueError('interval_seconds must be a multiple of output_seconds')
    per = interval_seconds // output_seconds
    n_int_requested = len(target)
    pad_intervals = int(np.ceil(padding_seconds / interval_seconds)) if padding_seconds > 0 else 0
    if pad_intervals:
        target = np.concatenate((np.full(pad_intervals, target[0]), target,
                                 np.full(pad_intervals, target[-1])))
        if start_timestamp is not None:
            # - The time of extended sample 0
            start_timestamp = start_timestamp - pd.Timedelta(seconds=pad_intervals * interval_seconds)
    n_int = len(target)
    n_out = n_int * per
    dt = float(output_seconds)
    rng = np.random.default_rng(seed)

    # - The always-on floor
    window = max(1, round(base_window_seconds / interval_seconds))
    floor = pd.Series(target).rolling(window, center=True, min_periods=1).quantile(base_quantile).to_numpy()
    if base_floor_w is not None:
        floor = np.minimum(floor, base_floor_w)
    floor = np.clip(np.minimum(floor, target), 0.0, None)
    # - The energy budget per interval in watt-seconds
    budget = np.maximum(target - floor, 0.0) * fill_fraction * interval_seconds
    budget_offered = float(budget.sum())
    # - The hard interval ceiling 
    hard_remaining = target * float(interval_seconds)

    # - Per-appliance placement state
    for app in appliances:
        if app.num_instances > 255:
            raise ValueError(f'{app.name}: num_instances > 255 not supported')
    usable = [i for i, app in enumerate(appliances)
              if app.patterns and min(len(p) for p in app.patterns) <= n_out]
    weights = np.array([appliances[i].weight for i in usable], dtype=np.float64)
    consecutive_rejects = {i: 0 for i in usable}
    active_counts = {}
    sec_of_day0 = None
    if start_timestamp is not None:
        sec_of_day0 = float((start_timestamp - start_timestamp.normalize()) / pd.Timedelta(seconds=1))
    windows_min = {i: [_window_minutes(w) for w in appliances[i].valid_daily_windows] for i in usable}
    seasons_md = {i: [(_md_int(a), _md_int(b)) for a, b in appliances[i].valid_seasons] for i in usable}
    for i in usable:
        if (windows_min[i] or seasons_md[i]) and start_timestamp is None:
            raise ValueError(f'{appliances[i].name} has time constraints but no start_timestamp was given')

    events = np.zeros(n_out, dtype=np.float32)
    log_rows = []
    n_attempts = 0

    while usable and len(log_rows) < max_events:
        n_attempts += 1
        probs = weights / weights.sum()
        pick = rng.choice(len(usable), p=probs)
        ai = usable[pick]
        app = appliances[ai]
        pattern_idx = int(rng.integers(len(app.patterns)))
        pattern = app.patterns[pattern_idx]
        length = len(pattern)
        accepted = False
        if length <= n_out:
            start = int(rng.integers(0, n_out - length + 1))
            if _is_valid_time(windows_min[ai], seasons_md[ai], start, start + length - 1, dt, sec_of_day0,
                        start_timestamp):
                counts = active_counts.get(ai)
                if counts is None or counts[start:start + length].max() < app.num_instances:
                    scale = float(rng.uniform(1.0 - app.amplitude_jitter, 1.0 + app.amplitude_jitter)) \
                        if app.amplitude_jitter > 0 else 1.0
                    iv0 = start // per
                    first_cut = (iv0 + 1) * per - start
                    cuts = np.arange(first_cut, length, per)
                    idx = np.concatenate(([0], cuts)) if len(cuts) else np.array([0])
                    contrib = np.add.reduceat(pattern.astype(np.float64), idx) * (dt * scale)
                    span = slice(iv0, iv0 + len(idx))
                    if contrib.sum() <= budget[span].sum() + 1e-6 \
                            and np.all(contrib <= hard_remaining[span] + 1e-6):
                        budget[span] -= contrib
                        hard_remaining[span] -= contrib
                        events[start:start + length] += (pattern * scale).astype(np.float32)
                        if counts is None:
                            counts = active_counts.setdefault(ai, np.zeros(n_out, dtype=np.uint8))
                        counts[start:start + length] += 1
                        log_rows.append((app.name, start, length, pattern_idx, scale))
                        consecutive_rejects[ai] = 0
                        accepted = True
        if not accepted:
            consecutive_rejects[ai] += 1
            if consecutive_rejects[ai] > max_consecutive_rejects:
                keep = [j for j, u in enumerate(usable) if u != ai]
                usable = [usable[j] for j in keep]
                weights = weights[keep]

    # - Add the base layer after the appliances have been placed according to the constraints. The base layer is a staircase
    ev_means = events.reshape(n_int, per).mean(axis=1, dtype=np.float64)
    base_means = np.clip(target - ev_means, 0.0, None)
    base = np.repeat(base_means, per)
    residual = target - (ev_means + base.reshape(n_int, per).mean(axis=1))
    base = np.clip(base + np.repeat(residual, per), 0.0, None)
    total = (base + events.astype(np.float64)).astype(np.float32)
    # - Get rid of the padding at the start and end of the load shape
    offset = pad_intervals * per
    if pad_intervals:
        keep = slice(offset, offset + n_int_requested * per)
        total = total[keep]
        base = base[keep]
        events = events[keep]
        target = target[pad_intervals:pad_intervals + n_int_requested]
        n_int = n_int_requested
        log_rows = [(name, start - offset, length, pattern_idx, scale)
                    for name, start, length, pattern_idx, scale in log_rows
                    if start - offset < n_int * per and start - offset + length > 0]
    achieved = total.astype(np.float64).reshape(n_int, per).mean(axis=1)
    ev_means_kept = events.reshape(n_int, per).mean(axis=1, dtype=np.float64)
    log = pd.DataFrame(log_rows, columns=['appliance', 'start_idx', 'length', 'pattern_idx', 'scale'])
    diagnostics = {
        'n_events': len(log),
        'n_attempts': n_attempts,
        'padding_intervals': pad_intervals,
        'events_per_appliance': log['appliance'].value_counts().to_dict() if len(log) else {},
        'budget_used_fraction': 1.0 - (float(budget.sum()) / max(budget_offered, 1e-9)),
        'event_energy_share': float(ev_means_kept.sum() / max(target.sum(), 1e-9)),
        'max_interval_error_w': float(np.abs(achieved - target).max()),
        'base_min_w': float(base.min()),
    }
    return ComposeResult(total=total, base=base.astype(np.float32), events=events, log=log,
                         diagnostics=diagnostics)


def appliance_traces(log, appliances, window_start, window_end):
    '''
    - Rebuild each appliance's trace for plotting

    :param log: the ComposeResult log
    :type log: DataFrame
    :param appliances: the same Appliance objects used to compose
    :type appliances: list
    :param window_start: first output sample index of the window
    :type window_start: int
    :param window_end: exclusive last output sample index of the window
    :type window_end: int
    :return: dict of appliance name -> float array of len window_end - window_start
    :rtype: dict
    '''
    assert isinstance(log, pd.DataFrame)
    by_name = {app.name: app for app in appliances}
    traces = {app.name: np.zeros(window_end - window_start, dtype=np.float32) for app in appliances}
    overlapping = log[(log['start_idx'] < window_end) & (log['start_idx'] + log['length'] > window_start)]
    for row in overlapping.itertuples(index=False):
        pattern = by_name[row.appliance].patterns[row.pattern_idx] * row.scale
        s = row.start_idx - window_start
        p0 = max(0, -s)
        p1 = min(row.length, window_end - window_start - s)
        traces[row.appliance][s + p0:s + p1] += pattern[p0:p1]
    return traces


def _window_minutes(window):
    '''
    - Convert a ("HH:MM", "HH:MM") pair to integer minutes-of-day

    :param window: the daily window strings
    :type window: list
    :rtype: tuple
    '''
    def minutes(hhmm):
        h, m = hhmm.split(':')
        return int(h) * 60 + int(m)
    return minutes(window[0]), minutes(window[1])


def _md_int(mmdd):
    '''
    - Convert "MM-DD" to the integer MMDD (Dec 1 -> 1201)

    :param mmdd: the month-day string
    :type mmdd: str
    :rtype: int
    '''
    m, d = mmdd.split('-')
    return int(m) * 100 + int(d)


def _is_valid_time(windows, seasons, start_idx, end_idx, dt, sec_of_day0, start_timestamp):
    '''
    - Check an appliance event's placement against the appliance's daily windows and seasons

    :rtype: bool
    '''
    if windows:
        for idx in (start_idx, end_idx):
            minute = int((sec_of_day0 + idx * dt) // 60) % 1440
            ok = False
            for s_min, e_min in windows:
                if s_min <= e_min:
                    ok = ok or (s_min <= minute <= e_min)
                else:
                    # - Overnight window, e.g. 23:00-02:00
                    ok = ok or (minute >= s_min or minute <= e_min)
            if not ok:
                return False
    if seasons:
        ts = start_timestamp + pd.Timedelta(seconds=start_idx * dt)
        md = ts.month * 100 + ts.day
        ok = False
        for s_md, e_md in seasons:
            if s_md <= e_md:
                ok = ok or (s_md <= md <= e_md)
            else:
                # - Wrap-around season, e.g. 12-01 to 01-15
                ok = ok or (md >= s_md or md <= e_md)
        if not ok:
            return False
    return True
