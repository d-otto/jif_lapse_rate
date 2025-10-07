#!/usr/bin/env python3
"""
merge_intermediate_site_data.py

Script to merge site-level NetCDF files from individual sites into combined
files based on subdirectory structure. This script:

1. Loads data from data/2025/intermediate/pendants/by_site/
2. Groups files by subdirectory (root, intensive/, etc.)
3. Merges datasets using tree-based pairwise merging for efficiency
4. Aligns datasets by finding union of datetimes and reindexing
5. Concatenates along site_id coordinate
6. Saves combined datasets to data/2025/processed/lvl0/

Created: 2025-10-03
"""

import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from jiflr import ROOT


def merge_two_datasets(ds1, ds2):
    """
    Merge two xarray datasets by finding the union of their datetime coordinates,
    reindexing both to the common grid, then concatenating along site_id.
    
    Parameters
    ----------
    ds1, ds2 : xarray.Dataset
        Datasets to merge
        
    Returns
    -------
    xarray.Dataset
        Merged dataset with common datetime grid and concatenated site_id
    """
    # Get all unique datetime values from both datasets
    all_times = []
    if 'datetime' in ds1.dims:
        all_times.extend(ds1.datetime.values)
    if 'datetime' in ds2.dims:
        all_times.extend(ds2.datetime.values)
    
    if not all_times:
        raise ValueError("No datetime coordinate found in datasets")
    
    # Create common time grid
    unique_times = pd.to_datetime(sorted(set(all_times)))
    
    # Reindex both datasets to common time grid
    ds1_reindexed = ds1.reindex(datetime=unique_times, method=None)
    ds2_reindexed = ds2.reindex(datetime=unique_times, method=None)
    
    # Concatenate along site_id dimension
    # Note: site_id should be a coordinate in the site-level files
    merged = xr.concat([ds1_reindexed, ds2_reindexed], dim='site_id')
    
    return merged


def tree_merge_datasets(datasets):
    """
    Merge a list of datasets using a tree-based approach for efficiency.
    Pairs datasets (0+1, 2+3, etc.) and recursively merges until single dataset.
    
    Parameters
    ----------
    datasets : list of xarray.Dataset
        List of datasets to merge
        
    Returns
    -------
    xarray.Dataset
        Single merged dataset
    """
    if len(datasets) == 0:
        return None
    if len(datasets) == 1:
        return datasets[0]
    
    # Pair up datasets and merge
    next_level = []
    for i in range(0, len(datasets), 2):
        if i + 1 < len(datasets):
            # Merge pair
            merged = merge_two_datasets(datasets[i], datasets[i + 1])
            next_level.append(merged)
        else:
            # Odd number of datasets, carry forward the last one
            next_level.append(datasets[i])
    
    # Recursively merge the next level
    return tree_merge_datasets(next_level)


def process_directory(input_dir, output_file):
    """
    Process all NetCDF files in a directory and merge them into a single file.
    
    Parameters
    ----------
    input_dir : Path
        Input directory containing site NetCDF files
    output_file : Path
        Output file path for merged dataset
    """
    print(f"Processing directory: {input_dir}")
    
    # Find all NetCDF files in the directory, excluding hidden files
    nc_files = [f for f in input_dir.glob("*.nc") if not f.name.startswith('.')]
    
    if not nc_files:
        print(f"  No NetCDF files found in {input_dir}")
        return
    
    print(f"  Found {len(nc_files)} files to merge")
    
    # Load all datasets
    datasets = []
    site_names = []
    
    for nc_file in tqdm(nc_files, desc="Loading datasets"):
        try:
            ds = xr.open_dataset(nc_file)
            datasets.append(ds)
            
            # Extract site name from filename or dataset attributes
            if 'site_name' in ds.attrs and ds.attrs['site_name'].strip():
                site_name = ds.attrs['site_name']
            else:
                site_name = nc_file.stem  # Use filename without extension
            
            # Handle empty site names
            if not site_name or site_name.strip() == '':
                site_name = f"unknown_{nc_file.stem}"
            
            site_names.append(site_name)
            
        except Exception as e:
            print(f"  Warning: Failed to load {nc_file}: {e}")
            continue
    
    if not datasets:
        print(f"  No valid datasets loaded from {input_dir}")
        return
    
    # Add site_id coordinate to each dataset if not present
    for i, (ds, site_name) in enumerate(zip(datasets, site_names)):
        if 'site_id' not in ds.coords:
            datasets[i] = ds.assign_coords(site_id=site_name)
    
    # Merge all datasets using tree approach
    print(f"  Merging {len(datasets)} datasets using tree approach...")
    merged_ds = tree_merge_datasets(datasets)
    
    if merged_ds is None:
        print(f"  Failed to merge datasets from {input_dir}")
        return
    
    # Update global attributes
    merged_ds.attrs.update({
        'processing_step': 'site_merged_lvl0',
        'source_directory': str(input_dir),
        'n_sites': len(datasets),
        'site_names': ', '.join(site_names)
    })
    
    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Save merged dataset
    print(f"  Saving merged dataset to: {output_file}")
    merged_ds.to_netcdf(output_file)
    
    # Print summary
    n_sites = len(site_names)
    n_times = len(merged_ds.datetime) if 'datetime' in merged_ds.dims else 0
    print(f"  Successfully merged {n_sites} sites -> {n_times} time points")
    
    # Close datasets to free memory
    for ds in datasets:
        ds.close()
    merged_ds.close()


if __name__ == "__main__":
    print("=== JIFLR Site Data Merging ===\n")
    
    # Define paths
    base_dir = Path(ROOT) / "data" / "2025" / "intermediate" / "pendants" / "by_site"
    output_dir = Path(ROOT) / "data" / "2025" / "processed" / "lvl0"
    
    print(f"Input base directory: {base_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Input directory exists: {base_dir.exists()}\n")
    
    if not base_dir.exists():
        print(f"Error: Input directory {base_dir} does not exist")
        exit(1)
    
    # Process main/root directory
    print("Processing main directory...")
    main_output = output_dir / "lvl0_main.nc"
    process_directory(base_dir, main_output)
    
    # Discover and process all subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]
    
    if subdirs:
        print(f"\nFound {len(subdirs)} subdirectories to process:")
        for subdir in sorted(subdirs):
            print(f"  - {subdir.name}")
        
        for subdir in sorted(subdirs):
            # Check if subdirectory contains any NetCDF files
            nc_files = [f for f in subdir.glob("*.nc") if not f.name.startswith('.')]
            
            if nc_files:
                print(f"\nProcessing {subdir.name} subdirectory...")
                subdir_output = output_dir / f"lvl0_{subdir.name}.nc"
                process_directory(subdir, subdir_output)
            else:
                print(f"\nSkipping {subdir.name} subdirectory (no NetCDF files found)")
    else:
        print("\nNo subdirectories found to process")
    
    print("\n=== Site Data Merging Complete ===")