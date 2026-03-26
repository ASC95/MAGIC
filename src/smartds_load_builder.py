import pandas as pd
import numpy as np
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import paths


def main():
    dss_file = paths.RDT1262_FILE_PATH
    csv_dir = paths.LOADSHAPES_DIR
    df_result = get_smartds_loadshapes(dss_file, csv_dir)


def get_smartds_loadshapes(dss_file_path: str, profiles_dir: str, year: int = 2018, mode: str = 'loads') -> pd.DataFrame:
    """
    Parses a Loads.dss file and a directory of per-unit load profiles to generate 
    actual kW load shapes. 
    
    Features:
    - Merges split-phase loads (suffixed _1/_2).
    - Returns a DataFrame with a 2-level MultiIndex columns: (Load Type, Load Name).
    - Load Type is derived from 'res_' (Residential) or 'com_' (Commercial) prefixes.

    Args:
        dss_file_path (str): Path to the Loads.dss file.
        profiles_dir (str): Directory containing the CSV profile files.
        year (int): The year for the timestamp index (default 2018).

    Returns:
        tuple: (A DataFrame where columns are MultiIndex (Type, Name) and index is 15-minute timestamps, A dict of kW bases)
    """
    dss_path = Path(dss_file_path)
    profiles_path = Path(profiles_dir)
    # 1. Parse the DSS file to extract load definitions
    print(f"Parsing DSS file: {dss_path.name}...")
    load_definitions = _parse_dss_loads(dss_path)
    # 2. Identify unique profiles needed and load them into memory (Cache)
    unique_profile_names = {load['profile'] for load in load_definitions}
    print(f"Loading {len(unique_profile_names)} unique profile shapes...")
    profile_cache = _load_profiles_into_memory(profiles_path, unique_profile_names)
    # 3. Calculate actual kW, Aggregate split-phase loads, and Track Types
    print("Calculating and aggregating load shapes...")
    aggregated_data = {}
    load_types = {}  # Map base_name -> 'Residential' | 'Commercial' | 'Other'
    kw_bases = {}
    # Expected rows: 35040 for non-leap year (365 days * 96 intervals)
    expected_length = 35040 

    for load_def in load_definitions:
        raw_name = load_def['name']
        kw_base = load_def['kw']
        profile_name = load_def['profile']
        # Determine the canonical base name (merging _1 and _2)
        base_name = _get_base_load_name(raw_name)
        # Determine Load Type based on profile prefix
        p_lower = profile_name.lower()
        if p_lower.startswith('res_'):
            l_type = 'Residential'
        elif p_lower.startswith('com_'):
            l_type = 'Commercial'
        else:
            l_type = 'Other'
        # Store the type (Last writer wins, but split-phases should be identical)
        load_types[base_name] = l_type
        if profile_name not in profile_cache:
            print(f"Warning: Profile '{profile_name}' not found for load '{raw_name}'. Skipping.")
            continue
        # Get PU array and calculate actual kW
        pu_array = profile_cache[profile_name]
        # Simple length check
        if len(pu_array) != expected_length:
            pass # Handle mismatches if necessary
        actual_kw_shape = pu_array * kw_base
        # Aggregate (Sum) for split phases
        if base_name in aggregated_data:
            aggregated_data[base_name]['actual_kw_shape'] += actual_kw_shape
            kw_bases[base_name] += kw_base
        else:
            aggregated_data[base_name] = {
                'actual_kw_shape': actual_kw_shape,
                'profile_name': profile_name,
                'load_type': load_types[base_name],
            }
            kw_bases[base_name] = kw_base
    
    # - If we want a DataFrame of loads, revert back to the original dict format
    if mode == 'loads':
        aggregated_data = {k: v['actual_kw_shape'] for k, v in aggregated_data.items()}
    # - If we want a DataFrame of per-unit load shapes, modify "aggregated_data"
    elif mode == 'pu':
        aggregated_data_pu = {}
        load_types_pu = {}
        kw_bases_pu = {}
        for k, d in aggregated_data.items():
            profile_name = d['profile_name']
            if profile_name not in aggregated_data_pu:
                aggregated_data_pu[profile_name] = d['actual_kw_shape']
                load_types_pu[profile_name] = d['load_type']
                kw_bases_pu[profile_name] = kw_bases[k]
        aggregated_data = aggregated_data_pu
        load_types = load_types_pu
        kw_bases = kw_bases_pu

    # 4. Construct Final DataFrame with MultiIndex
    print("Constructing DataFrame...")
    
    # Create Timestamp Index
    date_range = pd.date_range(start=f'{year}-01-01', end=f'{year}-12-31 23:45:00', freq='15min')
    
    # Validation: Align index length with data length
    if aggregated_data:
        data_length = len(next(iter(aggregated_data.values())))
        if len(date_range) != data_length:
            print(f"Notice: Generated time index length ({len(date_range)}) differs from data length ({data_length}). Adjusting index.")
            date_range = date_range[:data_length]

    # Create initial DataFrame (Columns are just names for now)
    df_loads = pd.DataFrame(aggregated_data, index=date_range)
    
    # Create the MultiIndex (Level 0: Type, Level 1: Name)
    # df_loads.columns is currently just the list of base_names
    multi_index_tuples = []
    for col_name in df_loads.columns:
        l_type = load_types.get(col_name, 'Other')
        multi_index_tuples.append((l_type, col_name))
        
    df_loads.columns = pd.MultiIndex.from_tuples(
        multi_index_tuples, 
        names=['Load Type', 'Load Name']
    )
    
    # Optional: Sort columns for cleaner viewing (e.g. all Residential together)
    df_loads = df_loads.sort_index(axis=1)

    print(f"Successfully generated load shapes for {len(df_loads.columns)} unique loads.")
    return df_loads, kw_bases


def _parse_dss_loads(dss_path: Path) -> List[Dict]:
    """
    Parses OpenDSS lines to extract name, kW, and yearly profile.
    """
    load_defs = []
    # Regex captures name, kW, and yearly profile
    pattern = re.compile(
        r"New Load\.(?P<name>[\w_]+).*kW=(?P<kw>[\d\.]+).*yearly=(?P<profile>[\w_]+)", 
        re.IGNORECASE
    )

    with open(dss_path, 'r') as f:
        for line in f:
            if line.strip().startswith('!') or not line.strip():
                continue
            match = pattern.search(line)
            if match:
                load_defs.append({
                    'name': match.group('name'),
                    'kw': float(match.group('kw')),
                    'profile': match.group('profile')
                })
    return load_defs


def _load_profiles_into_memory(profiles_dir: Path, profile_names: set) -> Dict[str, np.ndarray]:
    """
    Reads CSVs for the required profiles.
    Returns a dict: { 'profile_name': numpy_array_of_floats }
    """
    cache = {}
    for pname in profile_names:
        file_path = profiles_dir / f"{pname}.csv"
        
        # Fallback: try without extension
        if not file_path.exists():
            file_path = profiles_dir / pname
            
        if file_path.exists():
            try:
                # Read headerless CSV, coerce to numeric
                data = pd.read_csv(file_path, header=None)
                vals = pd.to_numeric(data.iloc[:, 0], errors='coerce').fillna(0).values
                cache[pname] = vals
            except Exception as e:
                print(f"Error reading profile {pname}: {e}")
        else:
            print(f"Error: Profile file not found for {pname}")
            
    return cache


def _get_base_load_name(load_name: str) -> str:
    """
    Removes _1 or _2 suffix to combine split phases.
    """
    return re.sub(r'_[12]$', '', load_name)


if __name__ == "__main__":
    main()