#!/usr/bin/env python
import yaml
import numpy as np
import random
import paths
from pathlib import Path
import timeVAE.paths
import timeVAE.vae_pipeline
from data_processor import ShapeletProcessor
from shapelet_evaluation import ComprehensiveEvaluator
import visualize

def main():
    #SEED = 42
    #np.random.seed(SEED)
    #random.seed(SEED)

    print(f"Loading run config from {paths.TRAINING_CONFIG_FILE_PATH}...")
    with open(paths.TRAINING_CONFIG_FILE_PATH) as f:
        train_cfg = yaml.safe_load(f)

    print(f"Loading appliance definitions from {paths.APPLIANCES_FILE_PATH}...")
    with open(paths.APPLIANCES_FILE_PATH) as f:
        app_full_config = yaml.safe_load(f)

    target_names = train_cfg.get('active_appliances', [])
    should_evaluate = train_cfg.get('run_evaluation', False)

    if not target_names:
        print("No appliances listed in 'active_appliances'. Exiting.")
        return

    # Initialize the new Evaluator
    evaluator = ComprehensiveEvaluator()

    for app_name in target_names:
        if app_name not in app_full_config:
            print(f"Warning: '{app_name}' not found. Skipping.")
            continue

        app_config = app_full_config[app_name]
        print(f"\n--- Processing {app_config['name']} ---")
        
        processor = ShapeletProcessor(app_config)
        
        # 1. Data Prep
        df_raw = processor.get_ampds2_data()
        real_shapelets, df_viz = processor.extract_shapelets(df_raw)
        print(f"Found {len(real_shapelets)} shapelets. Preparing training data...")
        training_data = processor.prepare_training_data(real_shapelets)
        
        if not training_data:
            print(f"No valid shapelets for {app_name}. Skipping.")
            continue

        # 2. Train
        _save_dataset(app_config, training_data)
        model_name = _get_model_name(app_config)
        print(f"Training VAE model: {model_name}")
        timeVAE.vae_pipeline.run_vae_pipeline(model_name, 'timeVAE', app_config['timeVAE_epochs'])

        # 3. Evaluate
        if should_evaluate:
            print(f"Evaluating {app_name}...")
            # Sample synthetic data (matches training size distribution)
            #   - Internally we always ask for 10,000 samples, but filter down to at most <sample size> (e.g. 500)
            vae_shapelets, df_prior = processor.get_vae_samples(500, [len(s) for s in real_shapelets])
            
            # Compute Table Metrics & Best/Worst
            evaluator.compute_all_metrics(real_shapelets, vae_shapelets, app_config, max_samples=min([len(real_shapelets), len(vae_shapelets)]))
            
            # Plot 1: Timeline
            graph_end = train_cfg['graph_end']
            visualize.graph_appliance_timeline(app_config['name'], df_raw, df_viz, df_prior, graph_end)
            
            # Plot 2: Best/Worst Matches
            match_data = evaluator.get_best_worst_data(app_config['name'])
            visualize.plot_best_worst_matches(app_config['name'], match_data)

        # - The vae_shapelets must be padded before saving to create a homogenous multidimensional array
        padded_vae_shapelets = processor._pad_shapelets(vae_shapelets)
        _save_vae_shapelets(app_config, padded_vae_shapelets)

    # 4. Final Table Output
    if should_evaluate:
        df_results = evaluator.get_results_dataframe()
        if not df_results.empty:
            print("\n--- Final Evaluation Table ---")
            print(df_results)
            # Save to PNG
            visualize.save_metrics_table_png(df_results)

def _save_dataset(app_config, shapelets):
    model_name = _get_model_name(app_config)
    data_array = np.array(shapelets)[..., np.newaxis]
    output_path = Path(timeVAE.paths.DATASETS_DIR) / model_name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, data=data_array)

def _save_vae_shapelets(app_config, shapelets):
    model_name = _get_model_name(app_config)
    data_array = np.array(shapelets)[..., np.newaxis]
    output_path = paths.VAE_SHAPELETS_OUTPUTS_DIR / f'vae_{model_name}'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(output_path, data=data_array)

def _get_model_name(app_config):
    return f'shapelets_{app_config["name"]}_epo-{app_config["timeVAE_epochs"]}'

if __name__ == '__main__':
    main()