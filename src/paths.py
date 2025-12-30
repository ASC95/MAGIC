#!/usr/bin/env python
from pathlib import Path

# Resolve project root relative to this file
# Assumes structure: project/src/paths.py
ROOT_DIR = Path(__file__).resolve().parent

DATASETS_DIR = ROOT_DIR / 'data'
OUTPUTS_DIR = ROOT_DIR / 'outputs'
VAE_SHAPELETS_OUTPUTS_DIR = OUTPUTS_DIR / 'vae_shapelets'
#SRC_DIR = ROOT_DIR / 'src'
CONFIG_DIR = ROOT_DIR / 'config'

# Config Files
APPLIANCES_FILE_PATH = CONFIG_DIR / 'appliance_config.yaml'
TRAINING_CONFIG_FILE_PATH = CONFIG_DIR / 'training_config.yaml'
INTERPOLATION_CONFIG_FILE_PATH = CONFIG_DIR / 'interpolation_config.yaml'

# Data Files
AMPDS2_FILE_PATH = DATASETS_DIR / 'ampds2' / 'AMPds2.h5'
RDT1262_FILE_PATH = DATASETS_DIR / 'smartds' / '2018' / 'GSO' / 'rural' / 'scenarios' / 'base_timeseries' / 'opendss' / 'rhs2_1247' / 'rhs2_1247--rdt1262' / 'Loads.dss'
RDT1264_FILE_PATH = DATASETS_DIR / 'smartds' / '2018' / 'GSO' / 'rural' / 'scenarios' / 'base_timeseries' / 'opendss' / 'rhs2_1247' / 'rhs2_1247--rdt1264' / 'Loads.dss'
LOADSHAPES_DIR = DATASETS_DIR / 'smartds' / '2018' / 'GSO' / 'rural' / 'profiles'