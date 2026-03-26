#!/usr/bin/env python
import yaml
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view
from pathlib import Path
from typing import List, Tuple, Dict
import paths
import constants
import timeVAE.data_utils
import timeVAE.vae.vae_utils
import timeVAE.paths


class ShapeletProcessor:

    def __init__(self, appliance_config: Dict):
        self.config = appliance_config
        self.name = appliance_config['name']

        if self.config['shapelet_max_W'] is None:
            self.config['shapelet_max_W'] = np.inf
        if self.config['shapelet_min_len'] is None:
            self.config['shapelet_min_len'] = -np.inf
        if self.config['shapelet_max_len'] is None:
            self.config['shapelet_max_len'] = np.inf
        
        # Validation
        if self.config['shapelet_start_W'] < self.config['shapelet_end_W']:
            raise ValueError("Start threshold must be greater than or equal to end threshold for hysteresis.")

    def get_ampds2_data(self) -> pd.DataFrame:
        """Loads and cleans data from the AMPds2 HDF5 file."""
        meter_id = constants.METER_MAPPING.get(self.name)
        if not meter_id:
            raise ValueError(f"Appliance {self.name} not found in constants.METER_MAPPING")

        h5_path = paths.AMPDS2_FILE_PATH
        if not h5_path.is_file():
            raise FileNotFoundError(f"AMPds2 file not found at {h5_path}")

        with pd.HDFStore(h5_path, 'r') as store:
            key = f'/building1/elec/{meter_id}'
            df = store.get(key)

        # Extract active power and standardize index
        df = df[[('power', 'active')]].copy()
        df.columns = ['power']
        df.index = pd.date_range(start=constants.AMPDS_START_DATE, 
                                 end=constants.AMPDS_END_DATE, 
                                 freq='1min')
        return df

    def extract_shapelets(self, data: pd.DataFrame) -> Tuple[List[np.ndarray], pd.DataFrame]:
        """Extracts shapelets using Hysteresis (Schmitt Trigger) logic."""
        ary_data = data['power'].fillna(0).values
        filtered_shapelet_indexes1 = []
        is_on = False
        start_idx = 0
        shapelet_lens1 = []
        # - The first pass must get the actual length of every shapelet because that's how I find the lengths that exceed one standard deviation above
        #   or below the mean length
        for i, val in enumerate(ary_data):
            if val >= self.config['shapelet_start_W'] and not is_on:
                is_on = True
                start_idx = i
            elif val <= self.config['shapelet_end_W'] and is_on:
                is_on = False
                end_idx = i - 1
                length = end_idx - start_idx + 1 
                # - Record the real length
                shapelet_lens1.append(length)
                # - Filter shapelet indexes based on user-defined length constraints
                # - The purpose of user-defined min_len is to prevent shapelets less than 3 readings long from being used and to select certain parts
                #   of appliance cycles
                # - The purpose of user-defined max_len is to remove ridiculously long shapelets
                if self.config['shapelet_min_len'] <= length <= self.config['shapelet_max_len']:
                    filtered_shapelet_indexes1.append((start_idx, end_idx))
        # - View statistics about the real lengths
        mean_len, std_len = np.mean(shapelet_lens1), np.std(shapelet_lens1)
        print(f'Real min shapelet length: {min(shapelet_lens1)}')
        print(f'Real max shapelet length: {max(shapelet_lens1)}')
        print(f'Real mean shapelet length: {mean_len}')
        print(f'Real std shapelet length: {std_len}')

        # - Filter shapelets based on the standard deviation of lengths
        #   - Actually don't do this. It isn't that useful. Eyeballing is better. This logic is good for inspection however
        #upper = round(mean_len + std_len)
        #lower = round(mean_len - std_len)
        #shapelet_lens2 = []
        #filtered_shapelet_indexes2 = []
        #for t in filtered_shapelet_indexes1:
        #    length = t[1] - t[0] + 1
        #    # - We need to remove these shapelets before training the VAE model
        #    # - These filtered shapelets are used to compute the evaluation metrics
        #    if lower <= length <= upper:
        #        filtered_shapelet_indexes2.append((t[0], t[1]))
        #        shapelet_lens2.append(length)
        ## - View statistics about filtered lengths
        #print(f'Filtered min shapelet length: {min(shapelet_lens2)}')
        #print(f'Real max shapelet length: {max(shapelet_lens2)}')
        #print(f'Real mean shapelet length: {np.mean(shapelet_lens2)}')
        #print(f'Real std shapelet length: {np.std(shapelet_lens2)}')

        # - Get rid of shapelets that exceed my minimum and maximum watt constraints
        filtered_shapelet_indexes3 = []
        for t in filtered_shapelet_indexes1:
            shapelet = ary_data[t[0]:t[1] + 1]
            if all([val <= self.config['shapelet_max_W'] for val in shapelet]):
                filtered_shapelet_indexes3.append((t[0], t[1]))
        shapelets = [ary_data[t[0]:t[1] + 1] for t in filtered_shapelet_indexes3]
        # - Inspect first 50 shapelets
        for i in range(min([len(shapelets), 50])):
            print(f'{i}th mean: {np.mean(shapelets[i])}')
        # Create visualization dataframe (zeroed out except for shapelets)
        viz_data = np.zeros_like(ary_data)
        for s, e in filtered_shapelet_indexes3:
            viz_data[s:e+1] = ary_data[s:e+1]
        df_viz = pd.DataFrame(viz_data, index=data.index, columns=['shapelet_activity'])
        return shapelets, df_viz

    def prepare_training_data(self, shapelets: List[np.ndarray]) -> List[np.ndarray]:
        """Applies sliding window and padding logic for training."""
        processed = shapelets
        
        if self.config.get('sliding_window_length') is not None:
            processed = self._apply_sliding_window(processed, self.config['sliding_window_length'])
            
        return self._pad_shapelets(processed)

    def get_vae_samples(self, sample_size: int, lengths) -> Tuple[List[np.ndarray], pd.DataFrame]:
        """Loads model, samples prior, clips negatives, and resamples length."""
        assert isinstance(lengths, list)

        model_name = f'shapelets_{self.name}_epo-{self.config["timeVAE_epochs"]}'
        model_dir = Path(timeVAE.paths.MODELS_DIR) / model_name
        
        # Load VAE components
        vae = timeVAE.vae.vae_utils.load_vae_model('timeVAE', model_dir)
        scaler = timeVAE.data_utils.load_scaler(model_dir)
        
        # Sample and Inverse Scale
        samples = timeVAE.vae.vae_utils.get_prior_samples(vae, 50000)
        inverse_scaled = timeVAE.data_utils.inverse_transform_data(samples, scaler)
        
        # Physics Correction: Clip negatives to 0 (Load cannot be negative)
        corrected = [np.clip(a, 0, None) for a in inverse_scaled]

        # - Filter bad shapelets that contains values less than the end of all extracted shapelets or greater than the maximum value allowed in
        #   extracted shapelets
        corrected = [s for s in corrected if all(self.config['shapelet_end_W'] <= val <= self.config['shapelet_max_W'] for val in s)]
        if len(corrected) == 0:
            raise Exception('No valid shapelets for the appliance were sampled. Consider loosening constraints.')
        # - Return at most sample_size samples
        corrected = corrected[:sample_size]
        
        # Round and squeeze
        shapelets = [np.round(arr).squeeze().tolist() for arr in corrected]
        
        # Apply Length Distribution (Truncate to match real data lengths)
        shapelets = [s[:np.random.choice(lengths, replace=False)] for s in shapelets]

        # Flatten for DataFrame
        flat_data = [val for sublist in shapelets for val in sublist]

        space_shapelets = True
        if space_shapelets:
            flat_data = [val for sublist in shapelets for val in ([0] * 30 + sublist + [0] * 30)]

        idx = pd.date_range(start=constants.AMPDS_START_DATE, periods=len(flat_data), freq='1min')
        df_prior = pd.DataFrame(flat_data, index=idx, columns=['generated_load'])
        
        return shapelets, df_prior

    def _apply_sliding_window(self, shapelets: List[np.ndarray], sliding_window_length) -> List[np.ndarray]:
        windows = []
        for ary in shapelets:
            if len(ary) - sliding_window_length > self.config['shapelet_min_len']:
                # - Get <sliding_window_length> + 1 of each shapelet, but each one is <sliding_window_length> elements shorter
                v = sliding_window_view(ary, len(ary) - sliding_window_length)
                if any([len(window) < self.config['shapelet_min_len'] for window in v]):
                    raise Exception('Sliding window was too large and created shapelets that are less than shapelet_min_len')
                windows.extend(v)
            else:
                windows.append(ary)
        return windows

    def _pad_shapelets(self, shapelets: List[np.ndarray]) -> List[np.ndarray]:
        if not shapelets: return []
        lengths = np.array([len(x) for x in shapelets])
        max_len = max(lengths)
        padded = []
        for x in shapelets:
            if len(x) < max_len:
                padded.append(np.pad(x, (0, max_len - len(x)), mode='constant'))
            else:
                padded.append(x)
        return padded