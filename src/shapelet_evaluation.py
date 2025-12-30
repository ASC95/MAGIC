#!/usr/bin/env python
import numpy as np
import pandas as pd
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import random
from typing import List, Dict, Tuple

class ComprehensiveEvaluator:
    def __init__(self):
        self.metrics_rows = []
        self.best_worst_matches = {}

    def compute_all_metrics(self, 
                          real_shapelets: List[np.ndarray], 
                          vae_shapelets: List[np.ndarray], 
                          appliance_config: dict,
                          max_samples: int = 100):
        """
        Computes all statistics for the summary table and identifies Best/Worst matches.
        """
        print(f"[{appliance_config['name']}] Computing Comprehensive Metrics...")
        
        # --- 1. Basic Statistics (Mean & STD) ---
        # Calculate Mean/STD for every individual shapelet, then take the average of those
        real_means = [np.mean(s) for s in real_shapelets]
        vae_means = [np.mean(s) for s in vae_shapelets]
        
        real_stds = [np.std(s) for s in real_shapelets]
        vae_stds = [np.std(s) for s in vae_shapelets]
        
        avg_real_mean = np.mean(real_means)
        avg_vae_mean = np.mean(vae_means)
        
        avg_real_std = np.mean(real_stds)
        avg_vae_std = np.mean(vae_stds)

        real_mean_len = np.mean([len(s) for s in real_shapelets])
        vae_mean_len = np.mean([len(s) for s in vae_shapelets])

        # --- 2. DTW Distribution Metrics (Diversity) ---
        # Downsample for O(N^2) operations
        real_sub = self._downsample(real_shapelets, max_samples)
        vae_sub = self._downsample(vae_shapelets, max_samples)
        
        # Real-Real Diversity
        real_dists = []
        for i in range(len(real_sub)):
            for j in range(i + 1, len(real_sub)):
                d, _ = fastdtw(np.array(real_sub[i]).flatten(), np.array(real_sub[j]).flatten())
                real_dists.append(d)
        real_real_dtw = np.mean(real_dists) if real_dists else 0.0

        # Synthetic-Synthetic Diversity
        vae_dists = []
        for i in range(len(vae_sub)):
            for j in range(i + 1, len(vae_sub)):
                d, _ = fastdtw(np.array(vae_sub[i]).flatten(), np.array(vae_sub[j]).flatten())
                vae_dists.append(d)
        syn_syn_dtw = np.mean(vae_dists) if vae_dists else 0.0

        # --- 3. Real-Synthetic DTW (Gap) & Best/Worst Identification ---
        # For this, we compare EVERY synthetic shapelet in the subset to ALL real shapelets in the subset
        # to find its "Nearest Real Neighbor".
        
        min_distances = [] # List of (min_dist, vae_idx, best_real_idx)
        
        for i, syn_item in enumerate(vae_sub):
            best_dist_for_this_syn = float('inf')
            best_real_idx = -1
            
            syn_flat = np.array(syn_item).flatten()
            
            for j, real_item in enumerate(real_sub):
                real_flat = np.array(real_item).flatten()
                d, _ = fastdtw(syn_flat, real_flat)
                
                if d < best_dist_for_this_syn:
                    best_dist_for_this_syn = d
                    best_real_idx = j
            
            min_distances.append((best_dist_for_this_syn, i, best_real_idx))

        # The "Real-Synthetic DTW Distance" is the average of these minimum distances
        # (i.e. How far is the average synthetic shapelet from its closest real cousin?)
        real_syn_dtw = np.mean([x[0] for x in min_distances])

        # Identify Best and Worst
        # Best = Smallest distance to a real neighbor
        # Worst = Largest distance to a real neighbor
        min_distances.sort(key=lambda x: x[0])
        
        best_match = min_distances[0]  # (dist, vae_idx, real_idx)
        worst_match = min_distances[-1]
        
        # Store actual arrays for visualization later
        self.best_worst_matches[appliance_config['name']] = {
            'best_vae': vae_sub[best_match[1]],
            'best_real': real_sub[best_match[2]],
            'best_dist': best_match[0],
            'worst_vae': vae_sub[worst_match[1]],
            'worst_real': real_sub[worst_match[2]],
            'worst_dist': worst_match[0]
        }

        # --- 4. Store Row Data ---
        self.metrics_rows.append({
            'Appliance':        appliance_config['name'],
            'Load Model':       appliance_config['load_model'],
            'Count Real':       f'{len(real_sub):.2f}',
            'Count Syn':        f'{len(vae_sub):.2f}',
            'Mean Len Real':    f'{round(real_mean_len)}',
            'Mean Len Syn':     f'{round(vae_mean_len)}',
            'Real-Real DTW':    f'{round(real_real_dtw, 2):.2f}',
            'Syn-Syn DTW':      f'{round(syn_syn_dtw, 2):.2f}',
            'Real-Syn DTW':     f'{round(real_syn_dtw, 2):.2f}',
            'Mean Val Real':    f'{round(avg_real_mean, 2):.2f}',
            'Mean Val Syn':     f'{round(avg_vae_mean, 2):.2f}',
            #'Diff Mean':        f'{round(abs(avg_real_mean - avg_vae_mean), 2):.2f}',
            'Mean Std Real':    f'{round(avg_real_std, 2):.2f}',
            'Mean Std Syn':     f'{round(avg_vae_std, 2):.2f}',
            #'Diff Std':         f'{round(abs(avg_real_std - avg_vae_std), 2):.2f}'
        })

    def _downsample(self, data_list, n):
        if len(data_list) > n: return random.sample(data_list, n)
        return data_list

    def get_results_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.metrics_rows)

    def get_best_worst_data(self, appliance_name: str) -> Dict:
        return self.best_worst_matches.get(appliance_name, {})