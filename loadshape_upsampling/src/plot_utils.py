#!/usr/bin/env python


'''
- This module draws the package's plotly figures
'''


import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.colors as pc
from itertools import cycle


def _finish(fig, title, xaxis_title='Timestamp', hovermode='x unified'):
    '''
    - Apply the layout every figure shares

    :rtype: Figure
    '''
    fig.update_layout(template='plotly_white', title=title, xaxis_title=xaxis_title, yaxis_title='Power (W)',
                      hovermode=hovermode)
    fig.update_yaxes(rangemode='tozero')
    return fig


def figure_composition(index, target_staircase, traces_by_appliance, base, total, title, load_name='',
                       original=None):
    '''
    - Create the figure for a single load. Contains appliance layers + base, the total, the coarse target's staircase, and the original higher-resolution
      series for comparison

    :param index: timestamps of the plotted window
    :type index: DatetimeIndex
    :param target_staircase: the binned means stretched out for the sake of plotting
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
    :param original: the original load shape
    :type original: ndarray
    :rtype: Figure
    '''
    prefix = f'{load_name} ' if load_name else ''
    fig = go.Figure()
    colors = cycle(pc.qualitative.D3 + pc.qualitative.Plotly)
    # - Base first so it sits at the bottom of the stack
    fig.add_trace(go.Scatter(x=index, y=base, mode='lines', name='base', stackgroup='one',
                             line=dict(width=0.5, color='lightgray')))
    for name in sorted(traces_by_appliance):
        y = traces_by_appliance[name]
        if np.sum(y) <= 0:
            continue
        color = next(colors)
        fig.add_trace(go.Scatter(x=index, y=y, mode='lines', name=name, stackgroup='one',
                                 line=dict(width=0.5, color=color)))
    fig.add_trace(go.Scatter(x=index, y=total, mode='lines', name=f'{prefix}composed total',
                             line=dict(color='black', width=1.5)))
    if original is not None:
        fig.add_trace(go.Scatter(x=index, y=original, mode='lines', name=f'{prefix}original',
                                 line=dict(color='rgba(31, 119, 180, 0.45)', width=1)))
    fig.add_trace(go.Scatter(x=index, y=target_staircase, mode='lines', name=f'{prefix}coarse target',
                             line=dict(color='red', width=2)))
    return _finish(fig, title)


def figure_feeder(start_timestamp, original_900_w, target_w, constraint_seconds, total, output_seconds, title,
                  max_points):
    '''
    - Create the figure for the entire feeder

    :param start_timestamp: time of sample 0
    :type start_timestamp: Timestamp
    :param original_900_w: the summed original 15-minute watts
    :type original_900_w: ndarray
    :param target_w: the summed coarse-target watts
    :type target_w: ndarray
    :param constraint_seconds: the constraint interval width
    :type constraint_seconds: int
    :param total: the summed composed watts at output resolution
    :type total: ndarray
    :param output_seconds: the output resolution
    :type output_seconds: int
    :param title: figure title
    :type title: str
    :param max_points: threshold for the composed trace
    :type max_points: int
    :rtype: Figure
    '''
    fig = go.Figure()
    n = len(total)
    if n <= max_points:
        index = pd.date_range(start_timestamp, periods=n, freq=f'{output_seconds}s')
        fig.add_trace(go.Scattergl(x=index, y=total, mode='lines', name='feeder composed total',
                                   line=dict(color='black', width=1)))
    else:
        bucket = int(np.ceil(n / (max_points // 2)))
        starts = np.arange(0, n, bucket)
        mins = np.minimum.reduceat(total, starts)
        maxs = np.maximum.reduceat(total, starts)
        centers = np.minimum(starts + bucket // 2, n - 1)
        index = start_timestamp + pd.to_timedelta(centers * output_seconds, unit='s')
        label = f'feeder composed total (min/max per {bucket * output_seconds} s)'
        fig.add_trace(go.Scatter(x=index, y=maxs, mode='lines', name=label, legendgroup='total',
                                 line=dict(color='rgba(0, 0, 0, 0.6)', width=0.5)))
        fig.add_trace(go.Scatter(x=index, y=mins, mode='lines', name=label, legendgroup='total',
                                 showlegend=False, fill='tonexty', fillcolor='rgba(0, 0, 0, 0.25)',
                                 line=dict(color='rgba(0, 0, 0, 0.6)', width=0.5)))
    for values, step_seconds, name, color in (
            (original_900_w, 900, 'feeder original (15-min)', 'rgba(31, 119, 180, 0.8)'),
            (target_w, constraint_seconds, 'feeder coarse target', 'red')):
        x = pd.date_range(start_timestamp, periods=len(values) + 1, freq=f'{step_seconds}s')
        y = np.concatenate((values, values[-1:]))
        fig.add_trace(go.Scatter(x=x, y=y, mode='lines', name=name, line_shape='hv',
                                 line=dict(color=color, width=1.5)))
    return _finish(fig, title)


def figure_validation(index, truth, synthetic, title):
    '''
    - Truth vs reconstruction overlay for a validation window

    :param index: timestamps of the plotted window
    :type index: DatetimeIndex
    :param truth: real watts within the window
    :type truth: ndarray
    :param synthetic: composed watts within the window
    :type synthetic: ndarray
    :param title: figure title
    :type title: str
    :rtype: Figure
    '''
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=index, y=truth, mode='lines', name='truth',
                             line=dict(color='blue', width=1)))
    fig.add_trace(go.Scatter(x=index, y=synthetic, mode='lines', name='synthetic',
                             line=dict(color='orange', width=1)))
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
