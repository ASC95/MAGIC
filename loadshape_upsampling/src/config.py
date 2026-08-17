

import os
import yaml


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONFIG_PATH = os.path.join(BASE_DIR, 'config', 'config.yaml')
APPLIANCES_PATH = os.path.join(BASE_DIR, 'config', 'appliances.yaml')
OUTPUTS_DIR = os.path.join(BASE_DIR, 'outputs')
LIBRARY_DIR = os.path.join(OUTPUTS_DIR, 'library')
RAE_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'rae')
SMARTDS_DIR = os.path.join(os.path.dirname(BASE_DIR), 'data', 'smartds')


with open(CONFIG_PATH) as f:
    _cfg = yaml.safe_load(f)


# - Copy each yaml setting onto this module so other modules can write config.<NAME>
ACTIVE_APPLIANCES = _cfg['active_appliances']
OUTPUT_SECONDS = _cfg['output_seconds']
CONSTRAINT_SECONDS = _cfg['constraint_seconds']
FILL_FRACTION = _cfg['fill_fraction']
BASE_QUANTILE = _cfg['base_quantile']
BASE_FLOOR_W = _cfg['base_floor_w']
BASE_WINDOW_SECONDS = _cfg['base_window_seconds']
MAX_CONSECUTIVE_REJECTS = _cfg['max_consecutive_rejects']
MAX_EVENTS = _cfg['max_events']
COMPOSE_PADDING_SECONDS = _cfg['compose_padding_seconds']
PARALLEL_WORKERS = _cfg['parallel_workers']
RANDOM_SEED = _cfg['random_seed']
VALIDATE_TRUTH_SOURCE = _cfg['validate_truth_source']
VALIDATE_WINDOW_DAYS = _cfg['validate_window_days']
SMARTDS_DATASET_PATH = os.path.join(SMARTDS_DIR, *_cfg['smartds_dataset_path'].split('/'))
SMARTDS_SCENARIO = _cfg['smartds_scenario']
SMARTDS_SUBSTATION = _cfg['smartds_substation']
SMARTDS_CIRCUIT = _cfg['smartds_circuit']
SMARTDS_YEAR = _cfg['smartds_year']
PLOT_WINDOW_DAYS = _cfg['plot_window_days']
AGGREGATE_PLOT_MAX_POINTS = _cfg['aggregate_plot_max_points']
TRAIN_PLOT_HOURS = _cfg['train_plot_hours']


with open(APPLIANCES_PATH) as f:
    APPLIANCES = yaml.safe_load(f)


del f, _cfg
