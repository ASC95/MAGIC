import yaml, re
from numba import njit
from datetime import datetime
from pathlib import Path
from itertools import cycle
from dataclasses import dataclass, field
from typing import List, Tuple

import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
import plotly.graph_objects as go
import plotly.colors as pc

import paths
import visualize
from smartds_load_builder import get_smartds_loadshapes
from ampds2_load_builder import get_ampds2_loadshape


@dataclass
class Appliance:
    name: str
    patterns: List[np.array]  # List of patterns (e.g. [Normal_Cycle, Heavy_Cycle])
    num_instances: int        # Physical capacity (e.g. 2 machines)
    # "replacement": Can use the same pattern multiple times.
    # "no_replacement": Each pattern in the list can be used MAX once.
    selection_mode: str = "replacement" 
    # E.g. only allowed to run between 7 PM and 11 PM
    #valid_daily_windows=[("19:00", "23:00")],
    valid_daily_windows: List[Tuple[str, str]] = field(default_factory=list)
    # E.g. only allowed in Winter (Dec 1st to Jan 15th)
    #valid_seasons=[("12-01", "01-15")]
    valid_seasons: List[Tuple[str, str]] = field(default_factory=list)


@njit
def find_best_spot_integrated(residual, pattern, static_mask, active_counts, limit):
    """
    Finds the best spot while checking Time & Concurrency constraints on the fly.
    """
    n_res = len(residual)
    n_pat = len(pattern)
    best_score = -np.inf
    best_idx = -1
    
    # Loop through every possible start time 't'
    for t in range(n_res - n_pat + 1):
        
        # --- 1. Static Time Check (Fastest) ---
        # We need ALL points in static_mask[t : t+L] to be True.
        # Optimization: Usually checking the start or end is enough to fail fast.
        # Here we scan the window strictly.
        time_valid = True
        for i in range(n_pat):
            if not static_mask[t + i]:
                time_valid = False
                break
        if not time_valid:
            continue

        # --- 2. Concurrency Check (Lazy Short-Circuit) ---
        # We only check this if time was valid.
        # We stop immediately if we find usage >= limit.
        concurrency_valid = True
        for i in range(n_pat):
            if active_counts[t + i] >= limit:
                concurrency_valid = False
                break
        if not concurrency_valid:
            continue

        # --- 3. Score Calculation ---
        # If we reached here, the spot is valid. Calculate score.
        current_score = 0.0
        for i in range(n_pat):
            r_val = residual[t + i]
            p_val = pattern[i]
            # Net Error Reduction
            current_score += (abs(r_val) - abs(r_val - p_val))
            
        if current_score > best_score:
            best_score = current_score
            best_idx = t
            
    return best_idx, best_score


def main():
    # - Load appliances
    print(f"Loading interpolation config from {paths.INTERPOLATION_CONFIG_FILE_PATH}...")
    with open(paths.INTERPOLATION_CONFIG_FILE_PATH) as f:
        interpolation_cfg = yaml.safe_load(f)
    appliance_names = interpolation_cfg['active_appliances']
    appliances = []
    for name in appliance_names:

        num_instances = 1
        valid_daily_windows = None
        valid_seasons = None
        #if name == 'dryer':
        #    valid_daily_windows = [("06:00", "08:00")]
        #if name == 'heat_pump':
        #    num_instances = 2
        #    valid_seasons = [('09-01', '01-01')]

        appliances.append(load_appliance_from_vae(
            directory=paths.VAE_SHAPELETS_OUTPUTS_DIR,
            appliance_name=name, 
            num_instances=num_instances,
            selection_mode='replacement',
            valid_daily_windows=valid_daily_windows,
            valid_seasons=valid_seasons))

    evaluate_ampds2_reconstruction(appliances)

    '''
    # - Get SMART-DS data
    dss_file = paths.RDT1262_FILE_PATH
    loadshapes_dir = paths.LOADSHAPES_DIR
    df = get_smartds_loadshapes(dss_file, loadshapes_dir)
    # - Grab a subset of data for visualiation
    df = df['2018-01-03 00:00:00':'2018-01-03 12:45:00']
    # - Scale to W instead of kW
    df *= 1000
    load_name = 'load_p1rlv630'
    target_w_15 = df['Residential', load_name]
    # - Stepwise interpolation to 1-minute intervals
    target_w_1 = forward_fill_load_shape(target_w_15)

    # - Perform interpolation
    max_iterations = 2000
    results_df, results_w, log_df = greedy_interpolation(target_w_1, appliances, max_iterations)

    # - save results_df to file to check consecutive runs
    #results_df.to_csv(paths.OUTPUTS_DIR / 'interpolation_runs' / f'{load_name}_{max_iterations}_{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')

    verify_capacity_constraints_greedy(log_df, appliances)
    accuracy_df = summarize_interpolation_accuracy(target_w_1, results_w)
    print(accuracy_df)
    plot_interpolation_results(target_w_1, results_w, results_df)
    '''


def forward_fill_load_shape(target_15min):
    assert isinstance(target_15min, pd.Series)
    # - Reindex 15-minute interval data to 1-minute data in a forward-fill fashion because each 15-minute interval reading represents the average of
    #   the subsequent 15-minute interval of power consumption (not an instantaneous reading taken every 15 minutes)
    # 1. Define the Start and exact End time you want
    # Assuming your data covers a single day, e.g., '2024-01-01'
    start_time = target_15min.index[0]
    # Explicitly set the end to the last minute of the last day in your data
    last_hour = target_15min.index[-1].hour
    end_time = target_15min.index[-1].replace(hour=last_hour, minute=59, second=0)

    # 2. Create the full 1-minute index
    full_1min_index = pd.date_range(start=start_time, end=end_time, freq='1min')

    # 3. Reindex and Forward Fill
    # This forces the dataframe to conform to the new index.
    # method='ffill' takes the 23:45 value and propagates it to 23:46...23:59.
    target_1min = target_15min.reindex(full_1min_index, method='ffill')
    return target_1min


def load_appliance_from_vae(
    directory: str, 
    appliance_name: str, 
    num_instances: int, 
    selection_mode: str,
    valid_daily_windows: list,
    valid_seasons: list,
) -> Appliance:
    """
    Scans the directory for the VAE shapelet file for the given appliance,
    loads the patterns, and returns an Appliance instance.

    :param directory: Path to the folder containing .npz files
    :param appliance_name: Name of the appliance (e.g., 'fridge')
    :param num_instances: Physical capacity (default: 1)
    :param selection_mode: 'replacement' or 'no_replacement' (default: 'replacement')
    :param kwargs: Additional arguments for valid_daily_windows or valid_seasons
    :return: Appliance instance
    """
    data_dir = Path(directory)
    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    # 1. Find the file (Handle variable epoch numbers)
    # Pattern: vae_shapelets_{name}_epo-{number}.npz
    search_pattern = f"vae_shapelets_{appliance_name}_epo-*.npz"
    found_files = list(data_dir.glob(search_pattern))

    if not found_files:
        raise FileNotFoundError(f"No VAE file found for '{appliance_name}' in {directory} matching pattern '{search_pattern}'")

    # 2. If multiple files exist, pick the one with the highest epoch number
    if len(found_files) > 1:
        # Helper to extract epoch number from filename
        def get_epoch(file_path):
            match = re.search(r"epo-(\d+)", file_path.name)
            return int(match.group(1)) if match else 0
            
        target_file = max(found_files, key=get_epoch)
        print(f"Multiple files found for {appliance_name}. Loading latest: {target_file.name}")
    else:
        target_file = found_files[0]

    # 3. Load the Data
    with np.load(target_file) as data:
        # Assuming the key is 'data' based on standard np.savez usage
        # If your npz uses a different key (like 'arr_0'), change this line.
        if 'data' in data:
            raw_patterns = data['data']
        #elif 'arr_0' in data:
        #    raw_patterns = data['arr_0']
        else:
            raise KeyError(f"Could not find 'data' or 'arr_0' key in {target_file.name}. Keys found: {list(data.keys())}")
    # 4. Process Patterns into List[np.array]
    # VAE outputs are often (N, T, 1) or (N, T). We want a list of 1D arrays.
    # .squeeze() removes dimensions of size 1 (e.g. changing shape (100, 60, 1) -> (100, 60))
    cleaned_patterns = [p.squeeze() for p in raw_patterns]
    # 5. Remove padding
    cleaned_patterns = [p[p > 0] for p in cleaned_patterns]
            
    # 5. Instantiate Appliance
    # Retrieve optional params from kwargs or defaults
    windows = valid_daily_windows
    seasons = valid_seasons

    return Appliance(
        name=appliance_name,
        patterns=cleaned_patterns,
        num_instances=num_instances,
        selection_mode=selection_mode,
        valid_daily_windows=windows,
        valid_seasons=seasons
    )


# --- 1. Helper: Season & Time Constraint Logic ---


def get_md_int(timestamp):
    """Converts a timestamp or string to an integer MMDD (e.g., Dec 1 -> 1201)."""
    # If it's already a pandas Timestamp/DatetimeIndex
    if hasattr(timestamp, 'month'): 
        return timestamp.month * 100 + timestamp.day
    # If it's a string "MM-DD"
    if isinstance(timestamp, str):
        m, d = map(int, timestamp.split('-'))
        return m * 100 + d
    return 0


def check_time_constraints(times: pd.DatetimeIndex, app):
    """
    Generates a boolean mask (True = Valid Start Time) based on Appliance constraints.
    """
    n = len(times)
    # Start assuming everything is allowed, then restrict down.
    final_mask = np.ones(n, dtype=bool)
    
    # --- A. SEASON CHECK ---
    if app.valid_seasons:
        # Start assuming NO days are allowed, then add valid seasons back in.
        season_mask = np.zeros(n, dtype=bool)
        
        # Pre-calculate integer representations for the target index
        # This is fast: (Month * 100) + Day. Ex: Dec 31 -> 1231, Jan 1 -> 101
        times_md = times.month * 100 + times.day
        
        for start_str, end_str in app.valid_seasons:
            s_int = get_md_int(start_str)
            e_int = get_md_int(end_str)
            
            if s_int <= e_int:
                # Standard range (e.g. June to August: 601 to 831)
                # Allow times BETWEEN start and end
                current_range = (times_md >= s_int) & (times_md <= e_int)
            else:
                # Wrap-around range (e.g. Dec to Jan: 1201 to 115)
                # Allow times AFTER start OR BEFORE end
                current_range = (times_md >= s_int) | (times_md <= e_int)
                
            season_mask |= current_range
            
        final_mask &= season_mask

    # --- B. DAILY WINDOW CHECK ---
    if app.valid_daily_windows:
        daily_mask = np.zeros(n, dtype=bool)
        
        # We assume times are sorted, but this works purely on hour/minute comparison
        # To make this fast, we can convert times to "minutes from midnight"
        minutes_of_day = times.hour * 60 + times.minute
        
        for start_str, end_str in app.valid_daily_windows:
            s_ts = pd.Timestamp(start_str)
            e_ts = pd.Timestamp(end_str)
            s_min = s_ts.hour * 60 + s_ts.minute
            e_min = e_ts.hour * 60 + e_ts.minute
            
            if s_min <= e_min:
                current_window = (minutes_of_day >= s_min) & (minutes_of_day <= e_min)
            else:
                # Overnight window (e.g. 23:00 to 02:00)
                current_window = (minutes_of_day >= s_min) | (minutes_of_day <= e_min)
            
            daily_mask |= current_window
            
        final_mask &= daily_mask

    return final_mask


# --- 2. Main Greedy Function (Updated for Class Structure) ---
def greedy_interpolation(
    target_series: pd.Series, 
    appliances: list, 
    max_iterations, 
    error_threshold=10.0,
    seed=None):
    if seed is not None:
        np.random.seed(seed)

    target_values = target_series.values.astype(float)
    times = target_series.index
    n_points = len(target_values)
    total_interpolated = np.zeros(n_points)
    
    # Pre-calc constraints
    static_masks = {app.name: check_time_constraints(times, app) for app in appliances}
    active_counts = {app.name: np.zeros(n_points, dtype=int) for app in appliances}
    used_patterns = {app.name: set() for app in appliances}
    
    # --- CACHE INITIALIZATION ---
    # Stores: {'App_Name': {'score': float, 'idx': int, 'pat_idx': int, 'dirty': True}}
    # 'dirty' means "needs recalculation"
    cache = {app.name: {'dirty': True, 'score': -np.inf} for app in appliances}
    
    log_entries = []
    results_storage = {app.name: [] for app in appliances}
    
    # Helper to check overlap
    def ranges_overlap(start1, len1, start2, len2):
        return max(start1, start2) < min(start1 + len1, start2 + len2)

    print(f"Starting Optimized Interpolation...")

    for it in range(max_iterations):

        # 1. Check Global Convergence
        current_residual = target_values - total_interpolated
        global_error = np.sum(np.abs(current_residual))
        if it % 100 == 0:
            print(f'{it}: global error {global_error}')
        if global_error < error_threshold:
            print(f"Converged at iteration {it}.")
            break

        best_round_move = None 
        best_round_score = -np.inf 
        
        # --- PROPOSAL PHASE (Using Cache) ---
        for app in appliances:
            entry = cache[app.name]
            
            # If dirty, we MUST re-scan/re-select a pattern
            if entry['dirty']:
                # 1. Select Pattern
                n_patterns = len(app.patterns)
                if app.selection_mode == "no_replacement":
                    available = [i for i in range(n_patterns) if i not in used_patterns[app.name]]
                else:
                    available = list(range(n_patterns))
                
                if not available:
                    entry['score'] = -np.inf # Mark as exhausted
                    entry['dirty'] = False
                    continue

                pat_idx = np.random.choice(available)
                pattern = np.array(app.patterns[pat_idx])
                L = len(pattern)
                if L > n_points: continue

                # 2. Build Mask
                best_idx, score = find_best_spot_integrated(
                    current_residual, 
                    pattern, 
                    static_masks[app.name],  # Raw boolean array
                    active_counts[app.name], # Raw int array
                    int(app.num_instances)
                )
                
                # 4. Update Cache
                entry['score'] = score
                entry['idx'] = best_idx
                entry['pat_idx'] = pat_idx
                entry['pattern_len'] = L
                entry['dirty'] = False # Clean for next time
            
            # --- COMPARE AGAINST ROUND LEADER ---
            # Whether we just calculated it OR retrieved it from cache, compare it now.
            if entry['score'] > best_round_score:
                best_round_score = entry['score']
                best_round_move = (app, entry['idx'], entry['pat_idx'], entry['pattern_len'])

        # --- EXECUTE PHASE ---
        # - best_round_score represents the net error reduction. If it is positive, then the net error will decrease by at least 0.1 W as a result of
        #   adding the shapelet
        # - If it looks like there is space to add more appliances in the interpolation output, they probably aren't being added because of the
        #   appliance constraints
        if best_round_move and best_round_score > 0.1:
            winner_app, start_idx, pat_idx, duration = best_round_move
            pattern = winner_app.patterns[pat_idx]
            
            # 1. Update Grid
            total_interpolated[start_idx : start_idx + duration] += pattern
            active_counts[winner_app.name][start_idx : start_idx + duration] += 1
            if winner_app.selection_mode == "no_replacement":
                used_patterns[winner_app.name].add(pat_idx)
                # If we used a pattern, this specific app is definitely 'dirty' next time
                # because its pool of available patterns changed.
                cache[winner_app.name]['dirty'] = True

            # 2. Log
            results_storage[winner_app.name].append((start_idx, pat_idx))
            log_entries.append({
                "Appliance": winner_app.name,
                "Start Time": times[start_idx],
                "Variant": pat_idx,
                "Duration": duration
            })
            
            # --- CACHE INVALIDATION ---
            # This is the "Lazy" magic. We only mark other apps as 'dirty' 
            # if their best spot OVERLAPS with where we just placed the winner.
            
            # The range we just modified:
            winner_start = start_idx
            winner_len = duration
            
            for app in appliances:
                c = cache[app.name]
                if c['dirty']: continue # Already dirty, ignore
                if c['score'] == -np.inf: continue # Already exhausted, ignore
                
                # If the app's cached best spot overlaps the new placement, 
                # the residual there has changed. The cached score is invalid.
                if ranges_overlap(c['idx'], c['pattern_len'], winner_start, winner_len):
                    c['dirty'] = True
                    
        else:
            print("Stalled.")
            break
            
    # --- FINALIZATION ---
    results_dict = {}
    for app in appliances:
        app_total = np.zeros(n_points)
        for start_t, pat_idx in results_storage[app.name]:
            prof = app.patterns[pat_idx]
            app_total[start_t : start_t + len(prof)] += prof
        results_dict[app.name] = app_total
    
    results_df = pd.DataFrame(results_dict, index=target_series.index)
    total_series = pd.Series(total_interpolated, index=target_series.index)
    log_df = pd.DataFrame(log_entries)
    return results_df, total_series, log_df


def verify_capacity_constraints_greedy(log_df: pd.DataFrame, appliances: list):
    """
    Verifies that the number of active instances never exceeds limits.
    Compatible with both Greedy (explicit duration) and MILP (pattern lookup) logs.
    """
    if log_df.empty:
        print("Log is empty. No constraints to check.")
        return True

    print("\n--- Detailed Timeline Reconstruction (Constraint Check) ---")
    
    # 1. Setup & Pre-calculation
    min_time = log_df['Start Time'].min()
    app_map = {app.name: app for app in appliances}
    
    # Calculate durations and global max time
    durations_per_row = []
    end_times_abs = [] # Absolute timestamps for end
    
    for _, row in log_df.iterrows():
        name = row['Appliance']
        start = row['Start Time']
        variant_idx = int(row['Variant']) if 'Variant' in row else 0
        
        # KEY CHANGE: Priority to 'Duration' col (Greedy), fallback to Pattern Len (MILP)
        if 'Duration' in row and pd.notnull(row['Duration']):
            duration = int(row['Duration'])
        elif name in app_map:
            duration = len(app_map[name].patterns[variant_idx])
        else:
            duration = 0 
            
        durations_per_row.append(duration)
        end_times_abs.append(start + pd.Timedelta(minutes=duration))

    # Determine simulation horizon
    max_time_abs = max(end_times_abs) if end_times_abs else min_time
    # +1 minute buffer to handle inclusive endpoints safely
    total_minutes = int((max_time_abs - min_time).total_seconds() / 60) + 5
    
    # 2. Initialize Timelines
    timeline_counts = {app.name: np.zeros(total_minutes, dtype=int) for app in appliances}
    
    # 3. Populate Timeline & Print Details
    print(f"Simulation Range: {min_time} to {max_time_abs}")
    
    for idx, row in log_df.iterrows():
        name = row['Appliance']
        start = row['Start Time']
        duration = durations_per_row[idx]
        
        if name not in timeline_counts: continue
            
        # Convert timestamp to integer index relative to min_time
        start_idx = int((start - min_time).total_seconds() / 60)
        end_idx = start_idx + duration
        
        # Boundary check
        if start_idx < 0: continue
        
        # Populate the timeline array
        # Note: slicing [start:end] is exclusive of end, which creates exactly 'duration' points
        actual_end = min(end_idx, total_minutes)
        timeline_counts[name][start_idx : actual_end] += 1

    print("\n--- Final Constraint Verification ---")
    all_passed = True
    
    for app in appliances:
        counts = timeline_counts[app.name]
        limit = app.num_instances
        max_overlap = np.max(counts) if len(counts) > 0 else 0
        
        if max_overlap > limit:
            print(f"[FAIL] {app.name}: Peak Overlap {max_overlap} > Limit {limit}")
            all_passed = False
            
            # Show where the violation happened
            violation_indices = np.where(counts > limit)[0]
            if len(violation_indices) > 0:
                first_fail_idx = violation_indices[0]
                fail_time = min_time + pd.Timedelta(minutes=int(first_fail_idx))
                print(f"       !!! Limit exceeded starting at {fail_time}")
        else:
            print(f"[PASS] {app.name}: Peak Overlap {max_overlap} <= Limit {limit}")
            
    return all_passed


def summarize_interpolation_accuracy(target_series: pd.Series, result_series: pd.Series):
    """
    Compares the original 15-minute target against the mean of the 
    interpolated 1-minute values for each block.

    Args:
        target_series: The original 15-minute target load (Start timestamps).
        result_series: The final 1-minute interpolated total load.

    Returns:
        pd.DataFrame: A summary table with columns:
                      ['Timestamp', 'Target_kW', 'Interp_Mean_kW', 'Error_kW']
    """
    summary_data = []
    differences = []

    # Iterate through each 15-minute block in the target
    for timestamp, target_val in target_series.items():
        
        # Define the end of this 15-minute block
        end_time = timestamp + pd.Timedelta(minutes=15)
        
        # Slice the 1-minute result series for this specific window
        # We slice [start : end) (exclusive of end time)
        # resulting in exactly 15 one-minute points.
        block_values = result_series[timestamp : end_time - pd.Timedelta(minutes=1)]
        
        if block_values.empty:
            continue
            
        # Calculate the actual mean achieved by the solver
        interp_mean = block_values.mean()
        
        # Calculate error (Positive means Target was higher than Interpolation)
        difference = target_val - interp_mean
        differences.append(difference)
        
        summary_data.append({
            "Timestamp": timestamp,
            "Target (kW)": target_val,
            "Interpolated Mean (kW)": interp_mean,
            "Error (kW)": difference
        })

    # Create DataFrame
    df_summary = pd.DataFrame(summary_data)
    
    # Optional: formatting for cleaner display
    pd.options.display.float_format = '{:.3f}'.format
    
    return df_summary


def calculate_reconstruction_metrics(ground_truth: pd.Series, interpolated: pd.Series) -> pd.DataFrame:
    """
    Computes performance metrics (MAE, CV-MAE) between a ground truth load shape
    and an interpolated load shape.

    Args:
        ground_truth: The original, measured 1-minute data (e.g., AMPds2).
        interpolated: The synthetic 1-minute output from the greedy algorithm.

    Returns:
        pd.DataFrame: A single-row DataFrame containing the metrics.
    """
    # 1. Align Data
    # Ensure we only compare timestamps that exist in both series
    common_index = ground_truth.index.intersection(interpolated.index)
    
    if len(common_index) == 0:
        print("Error: No overlapping timestamps found between ground truth and interpolation.")
        return pd.DataFrame()

    truth_aligned = ground_truth.loc[common_index]
    interp_aligned = interpolated.loc[common_index]

    # 2. Compute MAE (Mean Absolute Error)
    # The average magnitude of the error in Watts
    absolute_error = np.abs(truth_aligned - interp_aligned)
    mae = np.mean(absolute_error)

    # 3. Compute CV(MAE) (Coefficient of Variation of MAE)
    # This normalizes the error relative to the average load.
    # Formula: MAE / Mean(Ground Truth)
    truth_mean = np.mean(truth_aligned)
    
    # Avoid division by zero
    if truth_mean > 1e-9:
        cv_mae = mae / truth_mean
    else:
        cv_mae = np.nan  # Undefined if the mean load is 0

    # 4. Format Results
    metrics = {
        "Metric": ["Performance"],
        "MAE (W)": [mae],
        #"CV(MAE)": [cv_mae],  # Result is a ratio (e.g., 0.15 for 15%)
        "CV(MAE) %": [cv_mae * 100],  # Result is a percentage (e.g., 15.0%)
        "Ground Truth Mean (W)": [truth_mean],
        "Points Evaluated": [len(common_index)]
    }

    return pd.DataFrame(metrics)


def plot_interpolation_results(
    target_series: pd.Series, 
    result_series: pd.Series, 
    results_df: pd.DataFrame
):
    """
    Plots interpolation results using the exact data from the greedy output.
    
    Args:
        target_series: The target load curve (Red Line).
        result_series: The sum of all interpolated appliances (Black Dashed Line).
        results_df: DataFrame where columns are Appliance Names and values are Load.
    """
    fig = go.Figure()
    
    # --- 1. Setup Color Palette ---
    # Combine palettes to ensure enough distinct colors for many appliances
    color_pool = pc.qualitative.D3 + pc.qualitative.Plotly + pc.qualitative.G10
    color_iterator = cycle(color_pool)

    # --- 2. Add Appliance Stacks ---
    # We sort columns to ensure consistent legend order between runs
    for col_name in sorted(results_df.columns):
        profile = results_df[col_name].values
        
        # Only plot if this appliance type has non-zero load
        if np.sum(profile) > 0:
            this_color = next(color_iterator)
            
            fig.add_trace(go.Scatter(
                x=results_df.index,
                y=profile,
                mode='lines',
                name=col_name,
                stackgroup='one', # Enables stacking
                line=dict(width=0.5, color=this_color),
                fillcolor=this_color # Ensures the fill matches the line color
            ))

    # --- 3. Add Reference Lines ---
    
    # Total Interpolated (Black Dashed)
    # This sits on top of the stack. It must match the top edge perfectly.
    fig.add_trace(go.Scatter(
        x=result_series.index,
        y=result_series.values,
        mode='lines',
        name='Total Interpolated',
        line=dict(color='black', width=2.5, dash='solid'),
    ))

    # Target (Red Step Line)
    # We extend the last point to make the step-plot look complete at the end
    if not target_series.empty:
        last_time = target_series.index[-1]
        last_val = target_series.values[-1]
        
        # Estimate frequency (e.g., 15 mins) for the visual extension
        if len(target_series) > 1:
            freq_min = (target_series.index[1] - target_series.index[0]).total_seconds() / 60
        else:
            freq_min = 15 # Default fallback
            
        end_time = last_time + pd.Timedelta(minutes=freq_min)
        
        extension = pd.Series([last_val], index=[end_time])
        target_plot = pd.concat([target_series, extension])

        fig.add_trace(go.Scatter(
            x=target_plot.index,
            y=target_plot.values,
            mode='lines',
            name='Target',
            line=dict(color='red', width=3),
            line_shape='hv' # Horizontal-Vertical step shape
        ))

    # --- 4. Formatting ---
    fig.update_layout(
        title="Interpolation Results: Target vs. Appliance Stack",
        xaxis_title="Time",

        #yaxis_title="Power (kW)",
        yaxis_title="Power (W)",

        hovermode="x unified",
        legend=dict(yanchor="top", y=1, xanchor="left", x=1.02), # Legend outside right
        margin=dict(r=150) # Right margin for legend
    )
    
    # Ensure y-axis starts at 0 (or slightly below if needed)
    fig.update_yaxes(rangemode="tozero")
    
    fig.show()


def compare_runs():
    # - Used to inspect if appliance constraints are being respected
    df_1 = pd.read_csv(paths.OUTPUTS_DIR / 'interpolation_1500_2025-12-29 21:39:03')
    df_2 = pd.read_csv()
    pass


def evaluate_ampds2_reconstruction(appliances: list, max_iterations=2000):
    """
    Runs a full validation experiment:
    1. Loads real 1-minute data (AMPds2).
    2. Downsamples it to 15-minute intervals (simulating utility data).
    3. Runs Greedy Interpolation to reconstruct the 1-minute shape.
    4. Compares the Reconstruction vs. The Original.
    """
    print("\n--- Starting AMPds2 Reconstruction Evaluation ---")

    # 1. Load Ground Truth (1-minute)
    # Note: Requires your ampds2_load_builder module
    df_truth_1min = get_ampds2_loadshape()

    
    # Optional: Slice to a smaller window for faster testing if needed
    # df_truth_1min = df_truth_1min['2018-01-01':'2018-01-07']
    
    # 2. Create Target (15-minute)
    # Downsample to simulate the low-res input we would get from a utility
    target_15min = df_truth_1min.resample('15min', label='left', closed='left').mean()
    
    # - Select a subset for testing
    df_truth_1min = df_truth_1min['2012-04-03 00:00:00':'2012-04-03 12:45:00']
    target_15min = target_15min['2012-04-03 00:00:00':'2012-04-03 12:45:00']

    # 3. Prepare Target for Interpolation (Stepwise 1-minute)
    # This creates the "Red Line" step function your interpolator expects
    target_step_1min = forward_fill_load_shape(target_15min)

    # 4. Run Interpolation
    print(f"Interpolating {len(target_step_1min)} minutes of data...")
    results_df, results_total_1min, log_df = greedy_interpolation(
        target_series=target_step_1min, 
        appliances=appliances, 
        max_iterations=max_iterations
    )

    # 5. Evaluate Performance
    metrics_df = calculate_reconstruction_metrics(
        ground_truth=df_truth_1min, 
        interpolated=results_total_1min
    )
    
    # 6. Display Results
    print("\nEvaluation Results:")
    # Format for cleaner console output
    pd.options.display.float_format = '{:.4f}'.format
    #print(metrics_df.T) # Transpose for vertical readability
    print(metrics_df)
    visualize.save_metrics_table_png(metrics_df, 'ampds2_performance.png')
    
    # 7. Plot Comparison
    # We pass the results_df to your existing plotter
    plot_interpolation_results(target_step_1min, results_total_1min, results_df)
    plot_interpolation_results(df_truth_1min, results_total_1min, results_df)
    
    return metrics_df


if __name__ == '__main__':
    main()
    #compare_runs()