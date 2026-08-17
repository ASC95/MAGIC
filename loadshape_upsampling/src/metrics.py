#!/usr/bin/env python


'''
- This module compares a reconstruction against ground truth
'''


import numpy as np
import pandas as pd
from scipy import stats


def reconstruction_metrics(truth, synthetic, staircase, dt_seconds):
    '''
    - Score a synthetic reconstruction against the true high-resolution load shape

    :param truth: the real high-resolution watts
    :type truth: ndarray
    :param synthetic: the composed watts, same length/resolution
    :type synthetic: ndarray
    :param staircase: the forward-filled coarse target at the same resolution used as the pointwise-error reference
    :type staircase: ndarray
    :param dt_seconds: the sample resolution
    :type dt_seconds: int
    :return: one row of metrics
    :rtype: DataFrame
    '''
    truth = np.asarray(truth, dtype=np.float64)
    synthetic = np.asarray(synthetic, dtype=np.float64)
    assert len(truth) == len(synthetic)
    stride = max(1, len(truth) // 200000)
    ks_value = stats.ks_2samp(truth[::stride], synthetic[::stride]).statistic
    step_truth = float(np.std(np.diff(truth)))
    step_synth = float(np.std(np.diff(synthetic)))
    assert truth.mean() > 0 and step_truth > 0
    lag1_truth = float(np.corrcoef(truth[:-1], truth[1:])[0, 1])
    lag1_synth = float(np.corrcoef(synthetic[:-1], synthetic[1:])[0, 1])
    per_day = 86400 // dt_seconds
    n_days = len(truth) // per_day
    if n_days >= 2:
        peak_truth = float(truth[:n_days * per_day].reshape(n_days, per_day).max(axis=1).mean())
        peak_synth = float(synthetic[:n_days * per_day].reshape(n_days, per_day).max(axis=1).mean())
    else:
        peak_truth = float(truth.max())
        peak_synth = float(synthetic.max())
    cvmae = float(np.mean(np.abs(truth - synthetic)) / truth.mean() * 100)
    cvmae_staircase = float(np.mean(np.abs(truth - staircase)) / truth.mean() * 100)
    return pd.DataFrame([{
        'KS value dist': round(ks_value, 4),
        'step std truth (W)': round(step_truth, 2),
        'step std synth (W)': round(step_synth, 2),
        'step std ratio': round(step_synth / step_truth, 3),
        'lag-1 autocorr truth': round(lag1_truth, 4),
        'lag-1 autocorr synth': round(lag1_synth, 4),
        'daily peak truth (W)': round(peak_truth, 1),
        'daily peak synth (W)': round(peak_synth, 1),
        'daily peak ratio': round(peak_synth / peak_truth, 3),
        'CVMAE % (reference)': round(cvmae, 2),
        'CVMAE % staircase': round(cvmae_staircase, 2),
        'mean truth (W)': round(float(truth.mean()), 1),
        'mean synth (W)': round(float(synthetic.mean()), 1),
    }])
