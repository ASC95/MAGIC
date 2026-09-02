'''
The upsampler model turns low-resolution load shapes into high-resolution load shapes. Each uploaded column is rebuilt at the output resolution by placing
real appliance power signatures (shapelets) on top of a staircase base layer such that the mean power of every uploaded reading is preserved exactly. By
default, this model only performs upsampling, not training. The extracted appliance shapelet data for the model is stored in
omf/omf/static/testFiles/upsampler/library. The model can be re-trained via Python in the CLI to generate new extracted shapelets. Re-training is not
available via the web.py interface.
'''


import argparse
import concurrent.futures
import datetime
import glob
import re
import sys
import time
import zlib
from dataclasses import dataclass, field
from itertools import cycle
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
import yaml

from omf.models import __neoMetaModel__
from omf.models.__neoMetaModel__ import *

# Model metadata:
modelName, template = __neoMetaModel__.metadata(__file__)
tooltip = 'Upsample low-resolution load shapes into high-resolution load shapes composed from real appliance power signatures.'
hidden = False
# - Raise an error if there isn't enough space on disk to store the upsampled output
# 	- None: entire disk must have at least 10x the estimated output size of the csv.xz 
#   - <number>: (e.g. 10, or 10 GB). The user's entire model directory (i.e. data/Model/<user>) must not exceed this number at present or after the csv.xz
#     file is written
user_disk_limit = None
# - Where the training data is stored
_upsamplerFilesDir = pJoin(__neoMetaModel__._omfDir, 'static', 'testFiles', 'upsampler')
# - The output data file is named after the user's own upload and its units: <upload>_<units>_at_<output resolution>s.csv.xz
_outputDataFileNameSuffix = '_{units}_at_{output_seconds}s.csv.xz'
# - Use xz compression for output csv. This created the smallest files out of various compression methods
# - Use level 6 compression (default). Levels go from 0 (fastest, least compression) to 9 (slowest, most compression)
_outputCompression = {'method': 'xz', 'preset': 6}
# - What one value of the output csv.xz measures on disk: ~0.44 bytes on realistic composed data.
#   The disk check in work() estimate the output file as rows x columns x this value
_outputXzBytesPerValue = 0.44
# - Don't allow uploads longer than a year + leap day. 1-year load shapes are the standard size. We need to guard against extremely large output files
#   and/or run-times. 366 days creates a different constraint depending on the input resolution:
#	- 366 days = 8,784 hourly rows
#	- 366 days = 17,568 30-minute rows
#	- 366 days = 35,136 15-minute rows
#	- 366 days = 105,408 5-minute rows
#	- 366 days = 527,040 1-minute rows
_maxInputDays = 366
# - Reject uploads wider than 20 load shape columns to prevent excessive file sizes and run-times
#	- 20 columns x 366 days x 1s creates a 280 MB output csv.xz (3.2 GB uncompressed). It takes an hour to write the file and requires 5 GB of RAM
_maxInputColumns = 20
# - The maximum number of points we allow in one plotly graph. A value measures 5 bytes, so 4000000 * 5 = 20 MB + 4 MB plotly = 24 MB size of the HTML
#   file. Anything much larger and the browser gets slow. A 2-day graph window should show all appliances and the upsampled total.
_maxFigureValues = 4_000_000
# - When we need to graph more than _maxFigureValues, we use _figureBandBuckets to group individual values of the composed total load shape into buckets.
#   Each bucket is represented with its max and min value. Those two values from every bucket are used to create a gray band that stretches across the
#   figure. The appliance traces and base layer are removed entirely. We allow up to 200,000 buckets, so the values will be distributed equally within
#   those x-axis buckets
_figureBandBuckets = 200_000

#############################
# - Copied from composer.py #
#############################
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
	total: np.ndarray        # base + events, len = n_intervals * (interval_seconds // output_seconds)
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


##############################
# - Copied from shapelets.py #
##############################
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

###############################
# - Copied from plot_utils.py #
###############################
def _finish(fig, title, xaxis_title='Timestamp', hovermode='x unified'):
	'''
	- Apply the layout every figure shares

	:rtype: Figure
	'''
	fig.update_layout(template='plotly_white', title=title, xaxis_title=xaxis_title, yaxis_title='Power (W)',
		hovermode=hovermode)
	fig.update_yaxes(rangemode='tozero')
	return fig


def figure_composition(start_timestamp, output_seconds, target_staircase, traces_by_appliance, base,
		total, title, load_name=''):
	'''
	- Create the figure for a single load. Contains appliance layers + base, the total, and the coarse
	  target's staircase
	- The uploaded reading repeated at the output resolution IS the coarse target staircase here (the
	  model always preserves means over exactly one uploaded reading), so the package's separate
	  "original" trace would duplicate the staircase and is omitted
	- Every trace shares one uniform time axis, so timestamps are passed as x0 + dx instead of a
	  per-trace timestamp array. Serialized timestamp arrays were ~4x the size of the y data and made
	  multi-day figures balloon to tens of MB; x0 + dx renders identically for evenly spaced samples

	:param start_timestamp: time of the first plotted sample
	:type start_timestamp: Timestamp
	:param output_seconds: seconds between plotted samples
	:type output_seconds: int
	:param target_staircase: the uploaded readings stretched out for the sake of plotting
	:type target_staircase: ndarray
	:param traces_by_appliance: the appliance traces
	:type traces_by_appliance: dict
	:param base: the base layer
	:type base: ndarray
	:param total: the composed total
	:type total: ndarray
	:param title: figure title
	:type title: str
	:param load_name: the load name
	:type load_name: str
	:rtype: Figure
	'''
	prefix = f'{load_name} ' if load_name else ''
	# - x0 is a timestamp and dx is in milliseconds because the x axis type is date
	xkw = dict(x0=start_timestamp.isoformat(), dx=float(output_seconds) * 1000.0)
	fig = go.Figure()
	colors = cycle(pc.qualitative.D3 + pc.qualitative.Plotly)
	# - Base first so it sits at the bottom of the stack
	fig.add_trace(go.Scatter(y=base, mode='lines', name='base', stackgroup='one',
		line=dict(width=0.5, color='lightgray'), **xkw))
	for name in sorted(traces_by_appliance):
		y = traces_by_appliance[name]
		if np.sum(y) <= 0:
			continue
		color = next(colors)
		fig.add_trace(go.Scatter(y=y, mode='lines', name=name, stackgroup='one',
			line=dict(width=0.5, color=color), **xkw))
	fig.add_trace(go.Scatter(y=total, mode='lines', name=f'{prefix}composed total',
		line=dict(color='black', width=1.5), **xkw))
	fig.add_trace(go.Scatter(y=target_staircase, mode='lines', name=f'{prefix}coarse target',
		line=dict(color='red', width=2), **xkw))
	fig.update_xaxes(type='date')
	return _finish(fig, title)


def figure_composition_banded(start_timestamp, output_seconds, constraint_seconds, target_w, total,
		title, load_name=''):
	'''
	- The oversized-window variant of figure_composition: the composed total is drawn as a gray
	  min/max band per bucket of samples instead of point-by-point, and the appliance and base
	  layers are left out entirely - a stacked band per appliance would not be readable, so they
	  are hidden rather than banded. Same technique as the source package's figure_feeder
	- The coarse target is drawn at its own native resolution (one step per uploaded reading), not
	  repeated out to the output resolution, so it stays small at any window

	:param start_timestamp: time of the first plotted sample
	:type start_timestamp: Timestamp
	:param output_seconds: seconds between composed samples
	:type output_seconds: int
	:param constraint_seconds: seconds between uploaded readings
	:type constraint_seconds: int
	:param target_w: the uploaded readings inside the window, in watts
	:type target_w: ndarray
	:param total: the composed total inside the window
	:type total: ndarray
	:param title: figure title
	:type title: str
	:param load_name: the load name
	:type load_name: str
	:rtype: Figure
	'''
	prefix = f'{load_name} ' if load_name else ''
	n = len(total)
	bucket = int(np.ceil(n / _figureBandBuckets))
	starts = np.arange(0, n, bucket)
	mins = np.minimum.reduceat(total, starts)
	maxs = np.maximum.reduceat(total, starts)
	fig = go.Figure()
	# - Band points sit at bucket centers on the same x0/dx date axis figure_composition uses (the
	#   last bucket's center may overshoot the data end by up to half a bucket, which is invisible
	#   at these widths)
	bandX = dict(x0=(start_timestamp + pd.Timedelta(seconds=bucket * output_seconds / 2)).isoformat(),
		dx=float(bucket * output_seconds) * 1000.0)
	label = f'{prefix}composed total (min/max per {bucket * output_seconds} s)'
	fig.add_trace(go.Scatter(y=maxs, mode='lines', name=label, legendgroup='total',
		line=dict(color='rgba(0, 0, 0, 0.6)', width=0.5), **bandX))
	fig.add_trace(go.Scatter(y=mins, mode='lines', name=label, legendgroup='total',
		showlegend=False, fill='tonexty', fillcolor='rgba(0, 0, 0, 0.25)',
		line=dict(color='rgba(0, 0, 0, 0.6)', width=0.5), **bandX))
	# - hv steps hold each reading's value for its whole interval; the final value is repeated once
	#   so the last interval gets its width drawn too
	fig.add_trace(go.Scatter(y=np.concatenate((target_w, target_w[-1:])), mode='lines',
		name=f'{prefix}coarse target', line_shape='hv', line=dict(color='red', width=2),
		x0=start_timestamp.isoformat(), dx=float(constraint_seconds) * 1000.0))
	fig.update_xaxes(type='date')
	return _finish(fig, title)


def figure_library_timeline(index, raw, extracted, title):
	'''
	- Raw appliance load shape vs the extracted-shapelets

	:param index: timestamps of the plotted window
	:type index: DatetimeIndex
	:param raw: the appliance load shape within the window
	:type raw: ndarray
	:param extracted: raw values where a shapelet was extracted
	:type extracted: ndarray
	:param title: figure title
	:type title: str
	:rtype: Figure
	'''
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=index, y=raw, mode='lines', name='recording',
		line=dict(color='blue', width=1)))
	fig.add_trace(go.Scatter(x=index, y=extracted, mode='lines', name='extracted shapelets',
		line=dict(color='red', width=1)))
	return _finish(fig, title)


def figure_library_samples(patterns, dt_seconds, title, max_samples=12):
	'''
	- A handful of library patterns overlaid on a minutes axis, for eyeballing

	:param patterns: the library's pattern arrays
	:type patterns: list
	:param dt_seconds: the library resolution
	:type dt_seconds: int
	:param title: figure title
	:type title: str
	:param max_samples: how many patterns to draw
	:type max_samples: int
	:rtype: Figure
	'''
	fig = go.Figure()
	step = max(1, len(patterns) // max_samples)
	for i, pattern in enumerate(patterns[::step][:max_samples]):
		minutes = np.arange(len(pattern)) * dt_seconds / 60.0
		fig.add_trace(go.Scatter(x=minutes, y=pattern, mode='lines', name=f'#{i * step}',
			line=dict(width=1)))
	return _finish(fig, title, 'Minutes', None)


# - Functions for running the model within the OMF
def _readConfig(modelDir):
	'''
	- Read the model instance's config.yaml and appliances.yaml. Both are copied into the instance at
	  creation
		- The Input Resolution UI element replaces config.yaml's constraint_seconds
		- The Plot Window UI element replaces config.yaml's plot_window_days

	:param modelDir: the model instance folder
	:type modelDir: str
	:return: (config dict, appliance definition dict)
	:rtype: tuple
	'''
	with open(pJoin(modelDir, 'config.yaml')) as f:
		cfg = yaml.safe_load(f)
	with open(pJoin(modelDir, 'appliances.yaml')) as f:
		appCfgs = yaml.safe_load(f)
	return cfg, appCfgs


def _buildAppliances(modelDir, outputSeconds):
	'''
	- Load every active appliance's shapelet library from the instance's library folder and create
	  Appliance objects with patterns resampled to the output resolution

	:param modelDir: the model instance folder
	:type modelDir: str
	:param outputSeconds: the composition's output resolution
	:type outputSeconds: int
	:rtype: list
	'''
	cfg, appCfgs = _readConfig(modelDir)
	appliances = []
	for name in cfg['active_appliances']:
		path = pJoin(modelDir, 'library', f'{name}.npz')
		if not os.path.isfile(path):
			raise FileNotFoundError(f'{path} not found - the library was not staged into this model instance')
		patterns, dtSeconds = load_library(path)
		patterns = resample_patterns(patterns, dtSeconds, outputSeconds)
		a = appCfgs[name]
		appliances.append(Appliance(
			name=name,
			patterns=patterns,
			weight=a['weight'],
			num_instances=a['num_instances'],
			amplitude_jitter=a['amplitude_jitter'],
			valid_daily_windows=[tuple(w) for w in a['valid_daily_windows']],
			valid_seasons=[tuple(s) for s in a['valid_seasons']]))
	return appliances


def _readInputCsv(path):
	'''
	- Read the uploaded csv of load shape columns and decide whether its first row is a header
	- If every cell of the first row parses as a number, the row is data and the columns get named loadshape_1, loadshape_2
	- Every cell must be numeric, gap-free, non-negative, and every column the same length

	:param path: the uploaded csv
	:type path: str
	:return: (DataFrame of float64 shapes, whether a header row was found)
	:rtype: tuple
	'''
	probe = pd.read_csv(path, header=None, nrows=1)
	try:
		probe.iloc[0].astype(np.float64)
		hadHeader = False
	except (ValueError, TypeError):
		hadHeader = True
	df = pd.read_csv(path, header=0 if hadHeader else None)
	if hadHeader:
		df.columns = [str(c).strip() for c in df.columns]
	else:
		df.columns = [f'loadshape_{i + 1}' for i in range(df.shape[1])]
	try:
		df = df.astype(np.float64)
	except (ValueError, TypeError) as e:
		raise ValueError(f'The input csv contains non-numeric data below the first row: {e}')
	if df.shape[0] == 0:
		raise ValueError('The input csv contains no data rows.')
	nanCols = [c for c, hasNan in df.isna().any().items() if hasNan]
	if nanCols:
		raise ValueError(f'Empty or NaN cells found in columns {nanCols}. Every column must be gap-free '
			'and every column must have the same number of rows.')
	negCols = [c for c in df.columns if (df[c] < 0).any()]
	if negCols:
		raise ValueError(f'Negative values found in columns {negCols}. This model upsamples consumption '
			'load shapes, which must be >= 0.')
	return df, hadHeader


def _safeFileNames(names):
	'''
	- Turn the load shape names into filesystem-safe names for the events.csv and composition.html that are output for each column

	:param names: the load shape names
	:type names: list
	:rtype: dict
	'''
	safe = {}
	used = set()
	for name in names:
		stem = re.sub(r'[^\w.-]', '_', str(name)) or 'loadshape'
		candidate = stem
		i = 2
		while candidate in used:
			candidate = f'{stem}_{i}'
			i += 1
		used.add(candidate)
		safe[name] = candidate
	return safe


def _wattsFactors(units, puBaseKw, columnNames):
	'''
	- Determine each column's multiplier from its uploaded units to watts. 

	:param units: 'pu', 'W', 'kW', or 'MW'
	:type units: str
	:param puBaseKw: the p.u. base in kW - one number for every column, or one comma-separated number
		per column. Ignored unless units is 'pu'
	:type puBaseKw: str
	:param columnNames: the load shape names in csv order
	:type columnNames: list
	:return: dict of load shape name -> watts factor
	:rtype: dict
	'''
	if units == 'W':
		return {name: 1.0 for name in columnNames}
	elif units == 'kW':
		return {name: 1000.0 for name in columnNames}
	elif units == 'MW':
		return {name: 1e6 for name in columnNames}
	elif units == 'pu':
		try:
			bases = [float(x) for x in str(puBaseKw).split(',')]
		except ValueError:
			raise ValueError(f'Could not parse p.u. base kW "{puBaseKw}" - enter one number, or one '
				'comma-separated number per column.')
		if len(bases) == 1:
			bases = bases * len(columnNames)
		if len(bases) != len(columnNames):
			raise ValueError(f'{len(bases)} p.u. base kW values were given for {len(columnNames)} '
				'columns - enter one number, or one number per column.')
		if any(b <= 0 for b in bases):
			raise ValueError('Every p.u. base kW must be > 0.')
		return {name: 1000.0 * base for name, base in zip(columnNames, bases)}
	raise ValueError(f'Unknown units "{units}".')


def _composeOne(name, safeName, targetW, appliances, cfg, constraintSeconds, startTimestamp, modelDir,
		wattsFactor):
	'''
	- Upsample one uploaded column: compose it, write its appliance-event log, and write its
	  composition figure

	:param name: the load shape's display name
	:type name: str
	:param safeName: the load shape's filesystem-safe name
	:type safeName: str
	:param targetW: mean watts per uploaded reading
	:type targetW: ndarray
	:param appliances: Appliance objects at the output resolution
	:type appliances: list
	:param cfg: the instance's config.yaml dict
	:type cfg: dict
	:param constraintSeconds: the uploaded csv's resolution
	:type constraintSeconds: int
	:param startTimestamp: time of the first uploaded reading
	:type startTimestamp: Timestamp
	:param modelDir: the model instance folder
	:type modelDir: str
	:param wattsFactor: divide composed watts by this to get back to the uploaded units
	:type wattsFactor: float
	:return: (composed values in the uploaded units as float32, one diagnostics row for the results table)
	:rtype: tuple
	'''
	outputSeconds = int(cfg['output_seconds'])
	startTime = time.perf_counter()
	result = compose(
		targetW, constraintSeconds, outputSeconds, appliances,
		seed=per_load_seed(cfg['random_seed'], name), start_timestamp=startTimestamp,
		fill_fraction=cfg['fill_fraction'], base_quantile=cfg['base_quantile'],
		base_window_seconds=cfg['base_window_seconds'],
		max_consecutive_rejects=cfg['max_consecutive_rejects'], max_events=cfg['max_events'],
		padding_seconds=cfg['compose_padding_seconds'], base_floor_w=cfg['base_floor_w'])
	elapsed = time.perf_counter() - startTime
	d = result.diagnostics
	print(f'{name}: {d["n_events"]} events, event share {d["event_energy_share"]:.2f}, '
		f'max interval error {d["max_interval_error_w"]:.2e} W, {elapsed:.1f}s')
	log = result.log.copy()
	if len(log):
		log['start_time'] = startTimestamp + pd.to_timedelta(log['start_idx'] * outputSeconds, unit='s')
	log.to_csv(pJoin(modelDir, f'{safeName}_events.csv.gz'), index=False, compression='gzip')
	per = constraintSeconds // outputSeconds
	nPlot = min(len(result.total), int(cfg['plot_window_days']) * 86400 // outputSeconds)
	title = f'{name}: composed at {outputSeconds}s (means preserved per {constraintSeconds}s)'
	nTraces = result.log['appliance'].nunique() + 3
	if nPlot * nTraces <= _maxFigureValues:
		staircase = np.repeat(targetW, per)[:nPlot]
		traces = appliance_traces(result.log, appliances, 0, nPlot)
		fig = figure_composition(startTimestamp, outputSeconds, staircase,
			{n: t[:nPlot] for n, t in traces.items()}, result.base[:nPlot], result.total[:nPlot],
			title, load_name=name)
	else:
		fig = figure_composition_banded(startTimestamp, outputSeconds, constraintSeconds,
			targetW[:-(-nPlot // per)], result.total[:nPlot], title + ' - banded for size',
			load_name=name)
	fig.write_html(pJoin(modelDir, f'{safeName}_composition.html'))
	diagRow = {
		'name': name,
		'n_events': d['n_events'],
		'event_energy_share': round(d['event_energy_share'], 3),
		'max_interval_error_w': f'{d["max_interval_error_w"]:.2e}',
		'compose_seconds': round(elapsed, 1),
	}
	return (result.total / np.float32(wattsFactor)).astype(np.float32), diagRow


# - Each worker process builds the appliance list once via the pool initializer, because rebuilding it
#   per column re-reads and re-inflates the whole shapelet library
_workerAppliances = None


def _workerInit(modelDir, outputSeconds):
	'''
	- Load the shapelet library into a worker process

	:rtype: None
	'''
	global _workerAppliances
	_workerAppliances = _buildAppliances(modelDir, outputSeconds)


def _workerCompose(name, safeName, targetW, cfg, constraintSeconds, startTimestamp, modelDir, wattsFactor):
	'''
	- Upsample one uploaded column inside a worker process

	:rtype: tuple
	'''
	return _composeOne(name, safeName, targetW, _workerAppliances, cfg, constraintSeconds,
		startTimestamp, modelDir, wattsFactor)


def _dirBytes(path):
	'''
	- Measure how much disk a directory tree holds: the size of every file under it, summed. Sizes
	  are apparent file sizes (os.path.getsize), matching the sizes the results page reports, not
	  filesystem block usage

	:param path: the directory to measure
	:type path: str
	:return: total bytes
	:rtype: int
	'''
	total = 0
	for dirPath, _, fileNames in os.walk(path):
		for fileName in fileNames:
			try:
				total += os.path.getsize(pJoin(dirPath, fileName))
			except OSError:
				pass
	return total


def work(modelDir, inputDict):
	''' Run the model in its directory. '''
	outData = {}
	cfg, _ = _readConfig(modelDir)
	outputSeconds = int(cfg['output_seconds'])
	constraintSeconds = int(inputDict['inputResolution'])
	if constraintSeconds <= outputSeconds or constraintSeconds % outputSeconds != 0:
		raise ValueError(f'The input resolution ({constraintSeconds}s) must be a coarser multiple of '
			f"config.yaml's output_seconds ({outputSeconds}s).")
	plotDays = str(inputDict.get('plotDays', 'all'))
	cfg['plot_window_days'] = _maxInputDays if plotDays == 'all' else int(plotDays)
	startTimestamp = pd.Timestamp(inputDict['startTimestamp'])
	if pd.isna(startTimestamp):
		raise ValueError('Start Timestamp is required, e.g. 2018-01-01 00:00:00.')
	df, hadHeader = _readInputCsv(pJoin(modelDir, inputDict['inputDataFileName']))
	names = list(df.columns)
	maxRows = _maxInputDays * 86400 // constraintSeconds
	if df.shape[0] > maxRows:
		raise ValueError(f'The upload covers {df.shape[0] * constraintSeconds / 86400:.1f} days '
			f'({df.shape[0]} rows at {constraintSeconds}s per row); the most this model accepts is '
			f'one year = {maxRows} rows at this input resolution. Upload a shorter csv.')
	if len(names) > _maxInputColumns:
		raise ValueError(f'The upload has {len(names)} load shape columns; the most this model '
			f'accepts is {_maxInputColumns}. Split the csv into smaller batches of columns.')
	safeNames = _safeFileNames(names)
	factors = _wattsFactors(inputDict['units'], inputDict.get('puBaseKw', '1.0'), names)
	per = constraintSeconds // outputSeconds
	nOut = df.shape[0] * per
	estimatedXzBytes = int(nOut * len(names) * _outputXzBytesPerValue)
	freeBytes = shutil.disk_usage(modelDir).free
	if user_disk_limit is None:
		# - The drive must have room for 10x the estimated output before composing starts
		if estimatedXzBytes * 10 > freeBytes:
			raise ValueError(f'The estimated ~{estimatedXzBytes / 1e6:.1f} MB output csv.xz needs 10x '
				f'that ({estimatedXzBytes * 10 / 1e9:.2f} GB) free on disk, but {modelDir} has only '
				f'{freeBytes / 1e9:.2f} GB free. Free up space, upload fewer columns, or raise '
				'output_seconds in config.yaml.')
	else:
		# - The user's whole model directory must have less than user_disk_limit content
		limitBytes = float(user_disk_limit) * 1e9
		userDir = str(Path(modelDir).resolve().parent)
		usedBytes = _dirBytes(userDir)
		if usedBytes > limitBytes:
			raise ValueError(f'Your model directory {userDir} holds ~{usedBytes / 1e9:.2f} GB, already '
				f'over the {user_disk_limit} GB disk limit. Delete old model runs before upsampling.')
		if usedBytes + estimatedXzBytes > limitBytes:
			raise ValueError(f'The estimated ~{estimatedXzBytes / 1e6:.1f} MB output csv.xz would push '
				f'your model directory {userDir} (~{usedBytes / 1e9:.2f} GB) over the '
				f'{user_disk_limit} GB disk limit. Delete old model runs, upload fewer columns, or '
				'raise output_seconds in config.yaml.')
		if estimatedXzBytes > freeBytes:
			raise ValueError(f'The estimated ~{estimatedXzBytes / 1e6:.1f} MB output csv.xz will not '
				f'fit in the {freeBytes / 1e9:.2f} GB free at {modelDir}. Free up space, upload fewer '
				'columns, or raise output_seconds in config.yaml.')
	tasks = [(name, safeNames[name], df[name].to_numpy(dtype=np.float64) * factors[name], cfg,
		constraintSeconds, startTimestamp, modelDir, factors[name]) for name in names]
	workers = max(1, min(int(cfg['parallel_workers']), len(tasks)))
	composed = {}
	diagRows = {}
	if workers == 1:
		appliances = _buildAppliances(modelDir, outputSeconds)
		for task in tasks:
			composed[task[0]], diagRows[task[0]] = _composeOne(task[0], task[1], task[2], appliances,
				*task[3:])
	else:
		print(f'Composing {len(tasks)} load shapes across {workers} workers...')
		with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=_workerInit,
				initargs=(modelDir, outputSeconds)) as pool:
			futures = {pool.submit(_workerCompose, *task): task[0] for task in tasks}
			for future in concurrent.futures.as_completed(futures):
				name = futures[future]
				composed[name], diagRows[name] = future.result()
				del futures[future]
	outDf = pd.DataFrame({name: composed.pop(name) for name in names})
	uploadStem = re.sub(r'[^\w.-]', '_', str(Path(inputDict.get('inputUIDisplay') or 'loadshapes').stem)) or 'loadshapes'
	outputFileName = uploadStem + _outputDataFileNameSuffix.format(units=inputDict['units'], output_seconds=outputSeconds)
	for old in glob.glob(pJoin(modelDir, '*' + _outputDataFileNameSuffix.format(units='*', output_seconds='*'))) + \
			glob.glob(pJoin(modelDir, '*_upsampled_to_*s.csv.xz')) + \
			[pJoin(modelDir, 'upsampled_loadshapes.csv.xz')]:
		if os.path.isfile(old):
			os.remove(old)
	outputPath = pJoin(modelDir, outputFileName)
	writeStart = time.perf_counter()
	outDf.to_csv(outputPath, index=False, float_format='%.2f', compression=_outputCompression)
	writeSeconds = time.perf_counter() - writeStart
	# - Everything the results page shows
	outData['loadShapeNames'] = names
	outData['hadHeader'] = hadHeader
	outData['plotWindowDays'] = min(int(cfg['plot_window_days']), -(-nOut * outputSeconds // 86400))
	outData['figureFileNames'] = [f'{safeNames[name]}_composition.html' for name in names]
	outData['diagnosticsHeadings'] = ['Load Shape', 'Events Placed', 'Event Energy Share',
		'Max Interval Error (W)', 'Compose Time (s)']
	outData['diagnosticsValues'] = [list(diagRows[name].values()) for name in names]
	outData['outputFileName'] = outputFileName
	outData['outputRows'] = int(nOut)
	outData['outputColumns'] = len(names)
	outData['outputSeconds'] = outputSeconds
	outData['inputResolution'] = constraintSeconds
	outData['startTimestamp'] = str(startTimestamp)
	outData['endTimestamp'] = str(startTimestamp + pd.Timedelta(seconds=(nOut - 1) * outputSeconds))
	outData['units'] = inputDict['units']
	outData['outputSizeMB'] = round(os.path.getsize(outputPath) / 1e6, 2)
	outData['uncompressedEstimateMB'] = round(nOut * len(names) * 5 / 1e6, 2)
	outData['writeSeconds'] = round(writeSeconds, 1)
	# Stdout/stderr.
	outData['stdout'] = 'Success'
	outData['stderr'] = ''
	return outData


def runtimeEstimate(modelDir):
	''' Estimated runtime of model in minutes. '''
	with open(pJoin(modelDir, 'allInputData.json')) as f:
		inputDict = json.load(f)
	cfg, _ = _readConfig(modelDir)
	df, _ = _readInputCsv(pJoin(modelDir, inputDict['inputDataFileName']))
	outputSeconds = int(cfg['output_seconds'])
	per = max(1, int(inputDict['inputResolution']) // outputSeconds)
	nVals = df.shape[0] * per * df.shape[1]
	plotDays = str(inputDict.get('plotDays', 'all'))
	plotWindowDays = _maxInputDays if plotDays == 'all' else int(plotDays)
	plotVals = min(df.shape[0] * per, plotWindowDays * 86400 // outputSeconds, _maxFigureValues) * df.shape[1]
	return round(max(0.1, (5.0 + nVals * 8e-6 + plotVals * 1.5e-5) / 60.0), 2)


def new(modelDir):
	''' Create a new instance of this model. Returns true on success, false on failure. '''
	defaultInputs = {
		'modelType': modelName,
		'created': str(datetime.datetime.now()),
		'inputDataFileName': 'input_loadShapes.csv',
		'inputUIDisplay': 'input_loadShapes.csv',
		'inputResolution': '900',
		'startTimestamp': '2018-01-01 00:00:00',
		'units': 'kW',
		'puBaseKw': '1.0',
		'plotDays': '7',
	}
	creationCode = __neoMetaModel__.new(modelDir, defaultInputs)
	try:
		for fileName in ['input_loadShapes.csv', 'config.yaml', 'appliances.yaml']:
			shutil.copyfile(pJoin(_upsamplerFilesDir, fileName), pJoin(modelDir, fileName))
		shutil.copytree(pJoin(_upsamplerFilesDir, 'library'), pJoin(modelDir, 'library'),
			dirs_exist_ok=True)
	except:
		return False
	return creationCode


# ---------------------------------------------------------------------------------------------
# - Training code. Shouldn't really ever need to re-run this
# ---------------------------------------------------------------------------------------------


# - The RAE dataset was recorded in the Vancouver area
_RAE_LOCAL_TZ = 'America/Vancouver'

_RAE_POWER_FILES = {
	1: ['house1_power_blk1.csv', 'house1_power_blk2.csv'],
	2: ['house2_power_blk1.csv'],
}


def _raeSeries(raeDir, house, channels):
	'''
	- Return the summed watts of the given RAE meter as a 1-second Series

	:param raeDir: the folder holding the RAE power csvs
	:type raeDir: str
	:param house: which RAE house (1 or 2)
	:type house: int
	:param channels: power_blk column names to sum
	:type channels: list
	:return: watts at 1 s
	:rtype: Series
	'''
	assert isinstance(house, int)
	assert isinstance(channels, list)
	frames = [pd.read_csv(pJoin(raeDir, fileName),
		usecols=['unix_ts'] + sorted(set(channels)),
		dtype={c: np.float32 for c in channels},
		index_col='unix_ts')
		for fileName in _RAE_POWER_FILES[house]]
	df = pd.concat(frames).fillna(0.0)
	values = df[channels].sum(axis=1)
	index = pd.to_datetime(df.index, unit='s', utc=True).tz_convert(_RAE_LOCAL_TZ)
	return pd.Series(values.to_numpy(dtype=np.float32), index=index, name='+'.join(channels))


def _applianceSeries(raeDir, appCfg):
	'''
	- Load one appliance definition

	:param raeDir: the folder holding the RAE power csvs
	:type raeDir: str
	:param appCfg: the appliance's yaml dict (source + rae house/channels)
	:type appCfg: dict
	:return: (series of watts, resolution in seconds)
	:rtype: tuple
	'''
	assert isinstance(appCfg, dict)
	if appCfg['source'] == 'rae':
		return _raeSeries(raeDir, appCfg['house'], list(appCfg['channels'])), 1
	raise ValueError(f'Unknown appliance source: {appCfg["source"]}')


def _splitOnGaps(series, dtSeconds):
	'''
	- Split a single load shape containing meter outages into a list of series, where each series is a
	  non-interrupted set of appliance measurements containing 0 or more events

	:param series: watts with a DatetimeIndex
	:type series: Series
	:param dtSeconds: the load shape's resolution
	:type dtSeconds: int
	:return: list of contiguous sub-Series (order preserved)
	:rtype: list
	'''
	assert isinstance(series, pd.Series)
	assert isinstance(dtSeconds, int)
	if len(series) == 0:
		return []
	step = (series.index[1:] - series.index[:-1]) / pd.Timedelta(seconds=1)
	breaks = np.flatnonzero(step.to_numpy() != dtSeconds)
	starts = np.concatenate(([0], breaks + 1))
	ends = np.concatenate((breaks + 1, [len(series)]))
	return [series.iloc[s:e] for s, e in zip(starts, ends)]


def _train(raeDir, applianceNames=None, plot=False):
	'''
	- Build the shapelet library for every requested appliance: read the real load shape, cut it into
	  on/off events, save the events as static/testFiles/upsampler/library/<appliance>.npz, and report
	  extraction statistics
	- Reads config.yaml and appliances.yaml from static/testFiles/upsampler, so edit those (not a model
	  instance's copies) before retraining

	:param raeDir: the folder holding the RAE power csvs
	:type raeDir: str
	:param applianceNames: build only these appliances. None = every active appliance
	:type applianceNames: list
	:param plot: also write per-appliance diagnostic figures
	:type plot: bool
	:rtype: None
	'''
	with open(pJoin(_upsamplerFilesDir, 'config.yaml')) as f:
		cfg = yaml.safe_load(f)
	with open(pJoin(_upsamplerFilesDir, 'appliances.yaml')) as f:
		appCfgs = yaml.safe_load(f)
	libraryDir = pJoin(_upsamplerFilesDir, 'library')
	os.makedirs(libraryDir, exist_ok=True)
	names = applianceNames if applianceNames else cfg['active_appliances']
	summaryRows = []
	for name in names:
		if name not in appCfgs:
			raise KeyError(f'{name} is not defined in appliances.yaml')
		appCfg = appCfgs[name]
		print(f'--- {name} ({appCfg["source"]}) ---')
		startTime = time.perf_counter()
		series, dtSeconds = _applianceSeries(raeDir, appCfg)
		spans = []
		for stretch in _splitOnGaps(series, dtSeconds):
			offset = series.index.get_loc(stretch.index[0])
			spans.extend((offset + s, p) for s, p in extract_shapelets(
				stretch.to_numpy(), dtSeconds,
				start_w=appCfg['start_w'], end_w=appCfg['end_w'],
				min_seconds=appCfg['min_seconds'], max_seconds=appCfg['max_seconds'],
				max_w=appCfg['max_w']))
		if not spans:
			raise RuntimeError(f'no shapelets extracted for {name} - check its thresholds')
		patterns = [p for _, p in spans]
		stats = {'appliance': name, 'dt_s': dtSeconds, **library_stats(patterns, dtSeconds),
			'seconds': round(time.perf_counter() - startTime, 1)}
		summaryRows.append(stats)
		print(pd.DataFrame([stats]).to_string(index=False))
		save_library(pJoin(libraryDir, f'{name}.npz'), patterns, dtSeconds)
		if plot:
			_writeTrainingFigures(libraryDir, cfg, name, series, dtSeconds, spans, patterns)
	summary = pd.DataFrame(summaryRows)
	summaryPath = pJoin(libraryDir, 'library_summary.csv')
	if applianceNames and os.path.isfile(summaryPath):
		old = pd.read_csv(summaryPath)
		summary = pd.concat([old[~old['appliance'].isin(summary['appliance'])], summary], ignore_index=True)
	summary.to_csv(summaryPath, index=False)
	print(f'\nLibrary written to {libraryDir}')
	print(summary.to_string(index=False))


def _writeTrainingFigures(libraryDir, cfg, name, series, dtSeconds, spans, patterns):
	'''
	- Write the two per-appliance diagnostic figures next to the library

	:param libraryDir: the library folder the figures go under
	:type libraryDir: str
	:param cfg: the config.yaml dict
	:type cfg: dict
	:param name: the appliance name
	:type name: str
	:param series: the full appliance load shape
	:type series: Series
	:param dtSeconds: the load shape resolution
	:type dtSeconds: int
	:param spans: (global start index, pattern) pairs from extraction
	:type spans: list
	:param patterns: the extracted patterns
	:type patterns: list
	:rtype: None
	'''
	figuresDir = pJoin(libraryDir, 'figures')
	os.makedirs(figuresDir, exist_ok=True)
	n = min(len(series), int(cfg['train_plot_hours']) * 3600 // dtSeconds)
	window = series.iloc[:n]
	silhouette = np.zeros(n, dtype=np.float32)
	for start, pattern in spans:
		if start < n:
			k = min(len(pattern), n - start)
			silhouette[start:start + k] = pattern[:k]
	fig = figure_library_timeline(window.index, window.to_numpy(), silhouette,
		f'{name}: load shape vs extracted shapelets')
	fig.write_html(pJoin(figuresDir, f'{name}_timeline.html'))
	fig = figure_library_samples(patterns, dtSeconds, f'{name}: library samples')
	fig.write_html(pJoin(figuresDir, f'{name}_samples.html'))


@neoMetaModel_test_setup
def _tests():
	''' Run this module's local smoke test: create, run, and render a default model instance. '''
	modelLoc = Path(__neoMetaModel__._omfDir, 'data', 'Model', 'admin', 'Automated Testing of ' + modelName)
	# Blow away old test results if necessary.
	try:
		shutil.rmtree(modelLoc)
	except:
		# No previous test results.
		pass
	# Create New.
	new(modelLoc)
	# Pre-run.
	__neoMetaModel__.renderAndShow(modelLoc)
	# Run the model.
	__neoMetaModel__.runForeground(modelLoc)
	# Show the output.
	__neoMetaModel__.renderAndShow(modelLoc)


if __name__ == '__main__':
	if len(sys.argv) > 1 and sys.argv[1] == 'train':
		parser = argparse.ArgumentParser(description='Rebuild the upsampler appliance shapelet library')
		# - Pass the "train" parameter to re-train the model instead of upsampling the synthetic, 15-minute resolution house_A and house_B load shapes
		#	- house_A and house_B are in kilowatts
		parser.add_argument('train')
		# - When training, the directory that contains the RAE power csvs must be specified. That directory must contain house1_power_blk1.csv,
		#   house1_power_blk2.csv, and house2_power_blk1.csv. 
		# - The files stored in omf/omf/static/testFiles/upsampler/library only store extracted shapelets, not the full RAE power csvs
		parser.add_argument('--rae-dir', required=True, dest='rae_dir')
		# - Pass this option one or more times to re-train one or more specific appliances
		parser.add_argument('--appliance', action='append', default=None)
		# - Pass this option to generate plots for appliances that underwent re-training
		parser.add_argument('--plot', action='store_true')
		args = parser.parse_args()
		_train(args.rae_dir, applianceNames=args.appliance, plot=args.plot)
	else:
		_tests()