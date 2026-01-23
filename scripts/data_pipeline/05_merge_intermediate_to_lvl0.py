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

import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from jiflr import ROOT
from jiflr.logging import item, key_value, setup_pipeline_logging, subheader


def _fix_string_coordinate_lengths(ds, other_ds):
    """
    Fix string coordinate lengths to prevent truncation during concatenation.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to fix
    other_ds : xarray.Dataset
        Other dataset to compare string lengths with

    Returns
    -------
    xarray.Dataset
        Dataset with fixed string coordinate lengths
    """
    # Identify string coordinates
    string_coords = []
    for coord_name in ds.coords:
        if coord_name != 'sensor_idx' and ds[coord_name].dtype.kind in ['U', 'S']:
            string_coords.append(coord_name)

    if not string_coords:
        return ds

    # Calculate maximum string length needed for each coordinate
    coord_updates = {}
    for coord_name in string_coords:
        if coord_name in other_ds.coords:
            # Get max length from both datasets
            max_len_ds = max(len(str(val)) for val in ds[coord_name].values)
            max_len_other = max(len(str(val)) for val in other_ds[coord_name].values)
            max_len = max(max_len_ds, max_len_other, 10)  # Minimum 10 chars
        else:
            # Just use current dataset
            max_len = max(max(len(str(val)) for val in ds[coord_name].values), 10)

        # Create new coordinate with adequate string length
        current_values = ds[coord_name].values
        new_dtype = f'U{max_len}'
        new_values = np.array(current_values, dtype=new_dtype)
        coord_updates[coord_name] = (ds[coord_name].dims, new_values)

    if coord_updates:
        ds = ds.assign_coords(coord_updates)

    return ds


def merge_two_datasets(ds1, ds2):
    """
    Merge two xarray datasets with sensor_idx structure by concatenating along sensor_idx dimension.

    Parameters
    ----------
    ds1, ds2 : xarray.Dataset
        Datasets to merge (must have sensor_idx structure)

    Returns
    -------
    xarray.Dataset
        Merged dataset concatenated along sensor_idx dimension
    """
    # Verify both datasets have sensor_idx structure
    if 'sensor_idx' not in ds1.dims:
        raise ValueError("Dataset 1 does not have sensor_idx structure")
    if 'sensor_idx' not in ds2.dims:
        raise ValueError("Dataset 2 does not have sensor_idx structure")

    # Fix string coordinate truncation by ensuring adequate string lengths
    ds1_fixed = _fix_string_coordinate_lengths(ds1, ds2)
    ds2_fixed = _fix_string_coordinate_lengths(ds2, ds1)

    # Simply concatenate along sensor_idx dimension - much simpler!
    try:
        merged_ds = xr.concat([ds1_fixed, ds2_fixed], dim='sensor_idx', data_vars='all', coords='all', join='outer')
        # Fix sensor_idx to be sequential (0, 1, 2, 3...) instead of all zeros
        new_sensor_idx = list(range(len(merged_ds.sensor_idx)))
        merged_ds = merged_ds.assign_coords(sensor_idx=new_sensor_idx)
        return merged_ds
    except Exception as e:
        raise ValueError(f"Failed to concatenate datasets along sensor_idx: {e}")


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


def process_directory(input_dir, output_file, logger):
    """
    Process all NetCDF files in a directory and merge them into a single file.

    Parameters
    ----------
    input_dir : Path
        Input directory containing site NetCDF files
    output_file : Path
        Output file path for merged dataset
    logger : logging.Logger
        Logger instance
    """
    logger.info(subheader(f"Processing: {input_dir.name}"))

    # Find all NetCDF files in the directory, excluding hidden files
    nc_files = [f for f in input_dir.glob("*.nc") if not f.name.startswith('.')]

    if not nc_files:
        logger.info(f"No NetCDF files found in {input_dir}")
        return

    logger.info(f"Found {len(nc_files)} files to merge")

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
            logger.warning(f"Failed to load {nc_file}: {e}")
            continue

    if not datasets:
        logger.warning(f"No valid datasets loaded from {input_dir}")
        return

    # Add site_id coordinate to each dataset if not present
    # Calculate maximum site name length to prevent truncation
    max_site_name_len = max(len(name) for name in site_names) if site_names else 10
    max_site_name_len = max(max_site_name_len, 10)  # Minimum 10 characters

    for i, (ds, site_name) in enumerate(zip(datasets, site_names)):
        if 'site_id' not in ds.coords:
            # Create site_id coordinate with adequate string length
            site_id_dtype = f'U{max_site_name_len}'
            site_id_values = np.array([site_name] * len(ds.sensor_idx), dtype=site_id_dtype)
            datasets[i] = ds.assign_coords(site_id=('sensor_idx', site_id_values))

    # Merge all datasets using tree approach
    logger.info(f"Merging {len(datasets)} datasets using tree approach...")
    merged_ds = tree_merge_datasets(datasets)

    if merged_ds is None:
        logger.error(f"Failed to merge datasets from {input_dir}")
        return

    # Update global attributes for sensor_idx structure
    merged_ds.attrs.update({
        'processing_step': 'site_merged_lvl0_sensor_idx',
        'source_directory': str(input_dir),
        'n_sites': len(datasets),
        'n_sensors': len(merged_ds.sensor_idx) if 'sensor_idx' in merged_ds.dims else 0,
        'site_names': ', '.join(site_names),
        'structure': 'sensor_idx x datetime'
    })

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save merged dataset
    logger.info(key_value("Saving to", str(output_file)))
    merged_ds.to_netcdf(output_file)

    # Print summary
    n_sites = len(site_names)
    n_sensors = len(merged_ds.sensor_idx) if 'sensor_idx' in merged_ds.dims else 0
    datetime_coord = 'datetime' if 'datetime' in merged_ds.dims else 'datetime_utc'
    n_times = len(merged_ds[datetime_coord]) if datetime_coord in merged_ds.dims else 0
    logger.info(f"Successfully merged {n_sites} sites ({n_sensors} sensors) -> {n_times} time points")

    # Close datasets to free memory
    for ds in datasets:
        ds.close()
    merged_ds.close()


def main():
    """Main function to merge site data to lvl0."""
    # Set up logging (appends to pipeline log if running as part of pipeline)
    logger = setup_pipeline_logging(step_number=5, total_steps=6, mode="a")

    # Define paths
    base_dir = Path(ROOT) / "data" / "2025" / "intermediate" / "pendants" / "by_site"
    output_dir = Path(ROOT) / "data" / "2025" / "processed" / "lvl0"

    logger.info(key_value("Input base directory", str(base_dir)))
    logger.info(key_value("Output directory", str(output_dir)))
    logger.info(key_value("Input directory exists", str(base_dir.exists())))

    if not base_dir.exists():
        logger.error(f"Input directory {base_dir} does not exist")
        exit(1)

    # Process main/root directory
    main_output = output_dir / "lvl0_main.nc"
    process_directory(base_dir, main_output, logger)

    # Discover and process all subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]

    if subdirs:
        logger.info(f"Found {len(subdirs)} subdirectories to process:")
        for subdir in sorted(subdirs):
            logger.info(item(subdir.name))

        for subdir in sorted(subdirs):
            # Check if subdirectory contains any NetCDF files
            nc_files = [f for f in subdir.glob("*.nc") if not f.name.startswith('.')]

            if nc_files:
                subdir_output = output_dir / f"lvl0_{subdir.name}.nc"
                process_directory(subdir, subdir_output, logger)
            else:
                logger.info(f"Skipping {subdir.name} subdirectory (no NetCDF files found)")
    else:
        logger.info("No subdirectories found to process")


if __name__ == "__main__":
    main()
