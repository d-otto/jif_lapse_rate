#!/usr/bin/env python3
"""
Process JIFLR level 0 data to level 1.

This script converts processed lvl0 data to lvl1 by:
1. Resampling all data to regular 5-minute intervals using mean aggregation
2. Creating revised versions of individual lvl0 files  
3. Creating a single combined file merging all lvl0 data

Usage:
    python scripts/process_lvl0_to_lvl1.py
"""

import xarray as xr
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import argparse
from datetime import datetime

sys.path.append(str(Path(__file__).parent.parent))
from jiflr import ROOT


def interpolate_to_5min(ds):
    """
    Resample dataset to regular 5-minute intervals.
    
    This function resamples data to 5-minute intervals using mean aggregation,
    which preserves data from sensors that are already on 5-minute schedules
    but at different minute offsets (e.g., :13, :18, :23 vs :05, :10, :15).
    
    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with datetime coordinate
        
    Returns
    -------
    xarray.Dataset
        Dataset resampled to 5-minute intervals
    """
    # Resample to 5-minute intervals using mean aggregation
    # This preserves existing data points that fall on 5-minute boundaries
    # regardless of their minute offset from standard boundaries
    ds_resampled = ds.resample(datetime='5min').mean()
    
    # Update processing step attribute
    attrs = ds.attrs.copy()
    attrs['processing_step'] = 'lvl1_5min_resampled'
    attrs['resampling_method'] = 'mean'
    attrs['time_resolution'] = '5 minutes'
    attrs['processed_timestamp'] = datetime.now().isoformat()
    
    ds_resampled.attrs = attrs
    
    return ds_resampled


def process_individual_file(input_path, output_dir):
    """
    Process a single lvl0 file to lvl1.
    
    Parameters
    ----------
    input_path : Path
        Path to input lvl0 NetCDF file
    output_dir : Path
        Output directory for processed file
        
    Returns
    -------
    xarray.Dataset
        Processed dataset
    """
    print(f"Processing {input_path.name}...")
    
    # Load the dataset
    ds = xr.open_dataset(input_path)
    
    # Resample to 5-minute intervals
    ds_lvl1 = interpolate_to_5min(ds)
    
    # Generate output filename
    output_filename = input_path.name.replace('lvl0_', 'lvl1_')
    output_path = output_dir / output_filename
    
    # Save processed file
    ds_lvl1.to_netcdf(output_path)
    print(f"Saved {output_filename}")
    
    ds.close()
    
    return ds_lvl1


def combine_datasets(datasets, output_path):
    """
    Combine multiple datasets into a single file using sensor_idx concatenation.
    
    Parameters
    ----------
    datasets : list of xarray.Dataset
        List of datasets to combine (should have sensor_idx structure)
    output_path : Path
        Path for combined output file
    """
    print("Combining datasets using sensor_idx concatenation...")
    
    if len(datasets) == 1:
        # Only one dataset, just copy it
        combined = datasets[0].copy()
        print(f"Single dataset with {len(combined.sensor_idx)} sensors")
    else:
        # Multiple datasets - concatenate along sensor_idx dimension
        print(f"Concatenating {len(datasets)} datasets along sensor_idx dimension")
        
        try:
            # Concatenate along sensor_idx dimension
            combined = xr.concat(datasets, dim='sensor_idx', data_vars='all', coords='all')
            # Fix sensor_idx to be sequential (0, 1, 2, 3...) instead of all zeros
            new_sensor_idx = list(range(len(combined.sensor_idx)))
            combined = combined.assign_coords(sensor_idx=new_sensor_idx)
            
            # Get summary statistics
            total_sensors = len(combined.sensor_idx)
            unique_sites = sorted(set(combined.site_id.values)) if 'site_id' in combined.coords else []
            
            print(f"Successfully combined {total_sensors} sensors from {len(unique_sites)} sites")
            print(f"Sites: {unique_sites}")
            
        except Exception as e:
            print(f"Could not concatenate datasets along sensor_idx: {e}")
            print("Trying alternative concatenation method...")
            
            # Fallback: try concatenating with minimal options
            try:
                combined = xr.concat(datasets, dim='sensor_idx')
                # Fix sensor_idx to be sequential (0, 1, 2, 3...) instead of all zeros
                new_sensor_idx = list(range(len(combined.sensor_idx)))
                combined = combined.assign_coords(sensor_idx=new_sensor_idx)
                print(f"Fallback concatenation successful with {len(combined.sensor_idx)} sensors")
            except Exception as e2:
                print(f"Fallback concatenation also failed: {e2}")
                print("Saving first dataset only as combined file")
                combined = datasets[0].copy()
    
    # Update attributes for combined file
    attrs = combined.attrs.copy()
    attrs['title'] = 'Combined JIFLR Level 1 Data'
    attrs['processing_step'] = 'lvl1_combined_sensor_idx'
    attrs['n_source_files'] = len(datasets)
    attrs['combined_timestamp'] = datetime.now().isoformat()
    attrs['structure'] = 'sensor_idx × datetime'
    
    if 'sensor_idx' in combined.dims:
        attrs['n_sensors'] = len(combined.sensor_idx)
    
    combined.attrs = attrs
    
    # Save combined file
    combined.to_netcdf(output_path)
    print(f"Saved combined file: {output_path.name}")
    
    return combined


def main():
    """Main processing function."""
    parser = argparse.ArgumentParser(description='Process JIFLR lvl0 data to lvl1')
    parser.add_argument('--input-dir', type=str, 
                       default='data/2025/processed/lvl0',
                       help='Input directory containing lvl0 files')
    parser.add_argument('--output-dir', type=str,
                       default='data/2025/processed/lvl1', 
                       help='Output directory for lvl1 files')
    
    args = parser.parse_args()
    
    # Set up paths
    input_dir = ROOT / args.input_dir
    output_dir = ROOT / args.output_dir
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Processing lvl0 files from: {input_dir}")
    print(f"Output directory: {output_dir}")
    
    # Find all lvl0 NetCDF files
    lvl0_files = list(input_dir.glob('lvl0_*.nc'))
    
    if not lvl0_files:
        print("No lvl0 files found!")
        return
    
    print(f"Found {len(lvl0_files)} lvl0 files to process")
    
    # Process each file individually
    processed_datasets = []
    
    for lvl0_file in lvl0_files:
        try:
            ds_lvl1 = process_individual_file(lvl0_file, output_dir)
            processed_datasets.append(ds_lvl1)
        except Exception as e:
            print(f"Error processing {lvl0_file.name}: {e}")
            continue
    
    # Create combined file
    if processed_datasets:
        combined_output = output_dir / 'lvl1_combined.nc'
        try:
            combine_datasets(processed_datasets, combined_output)
        except Exception as e:
            print(f"Error creating combined file: {e}")
    
    # Close all datasets
    for ds in processed_datasets:
        ds.close()
    
    print("Processing complete!")


if __name__ == '__main__':
    main()