#!/usr/bin/env python


import paths
import pandas as pd


def main():
    # 1. Get the original 1-minute data
    df_1min = get_ampds2_loadshape()
    print(f"Original Shape (1-min): {df_1min.shape}")
    
    # 2. Downsample to 15-minute
    #   - I use left labels because that's what the interpolation code does (i.e. We forward-fill every 15-minute interval measurement so for example
    #     12:00 pm gets forward-filled to 12:14 pm. I just have to be consistent in this function and in interpolation.py
    # We use 'mean' to preserve the average power usage over the interval.
    # label='left':  Uses the start time (12:00) as the label
    # closed='left': Includes 12:00 data, excludes 12:15 data (standard for start-labeled intervals)
    df_15min = df_1min.resample('15min', label='left', closed='left').mean()

    # label='right' and closed='right' are standard for utility metering 
    # (e.g., 00:15 represents the average from 00:00 to 00:15).
    #df_15min = df_1min.resample('15min', label='right', closed='right').mean()
    
    print(f"Downsampled Shape (15-min): {df_15min.shape}")
    print(df_15min.head())


def get_ampds2_loadshape():
    h5_path = paths.AMPDS2_FILE_PATH
    if not h5_path.is_file():
        raise FileNotFoundError(f"AMPds2 file not found at {h5_path}")
    house_meter_id = 'meter1'  # Main aggregated meter
    with pd.HDFStore(h5_path, 'r') as store:
        key = f'/building1/elec/{house_meter_id}'
        df = store.get(key)
    # Ensure index is datetime (crucial for resampling)
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    # AMPds2 typically has a MultiIndex column structure like ('power', 'active')
    # We select just the active power column
    # Use tuple selection to handle MultiIndex gracefully
    df = df[('power', 'active')]
    return df


if __name__ == '__main__':
     main()