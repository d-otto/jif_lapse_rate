#!/usr/bin/env python3
"""
load_pendant_data_example.py

Example script demonstrating how to load the pendant sensor dataset.
This script shows the complete workflow from file paths to analysis-ready data,
using the functions available in jiflr.data module.

Based on the patterns used in explore_2025_data.py.

Author: Claude Code
Created: 2025-10-02
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Import core functions from jiflr
from jiflr import ROOT
from jiflr.data import (
    load_all_site_data,
    load_deployment_metadata, 
    read_hobo_pendant,
    read_pendant_dataset,
    load_netcdf_with_masking,
    find_overlapping_period,
    interpolate_to_common_grid
)
# Import utility functions that are needed but missing from data.py
from jiflr.utils import mask_from_csv_deployment_periods, find_matching_csv_filename



print("=== JIFLR Pendant Data Loading Example ===\n")

# ========================================================================
# STEP 1: Define Data Paths
# ========================================================================
print("Step 1: Setting up data paths...")

# Define paths to data directories
processed_dir = Path(ROOT) / "data" / "2025" / "intermediate" / "pendants"
csv_deployment_path = Path(ROOT) / "deployment_periods.csv"

print(f"  Processed NetCDF directory: {processed_dir}")
print(f"  Deployment periods CSV: {csv_deployment_path}")
print(f"  CSV exists: {csv_deployment_path.exists()}")

# Count available NetCDF files
netcdf_files = list(processed_dir.glob("*.nc"))
print(f"  Found {len(netcdf_files)} NetCDF files")


# ========================================================================
# STEP 2: Load All Site Data (Recommended Method)
# ========================================================================
print("Step 2: Loading all site data using load_all_site_data()...")

# Load all data organized by site and sensor height
# Note: CSV masking disabled due to missing function imports in data.py
# We'll demonstrate manual masking later
use_csv_masking = False  # Disabled due to missing imports

site_data = load_all_site_data(
    processed_dir=processed_dir,
    csv_deployment_path=csv_deployment_path,
    use_csv_masking=use_csv_masking,
    required_heights=['1m', '2m']  # Only load 1m and 2m sensors
)

print(f"  Loaded data for {len(site_data)} sites")
print(f"  Sites found: {list(site_data.keys())}")
print(f"  CSV deployment masking: {'Enabled' if use_csv_masking else 'Disabled'}")

# DATA STATE: site_data is now a nested dictionary:
# {site_name: {height: sensor_info_dict}}
# Each sensor_info_dict contains:
# - 'dataset': xarray Dataset with temp_c, intensity_lux variables
# - 'site_name', 'sensor_height', 'sensor_id': metadata
# - 'deployed_mask': boolean array (if CSV masking enabled)

print("  ✓ Site data loaded successfully\n")

# ========================================================================
# STEP 3: Load Deployment Metadata (Optional)
# ========================================================================
print("Step 3: Loading deployment metadata...")

metadata = load_deployment_metadata(csv_deployment_path)

print(f"  Loaded elevations for {len(metadata['elevations'])} sites")
print(f"  Loaded coordinates for {len(metadata['latitudes'])} sites")

# DATA STATE: metadata contains:
# - 'elevations': {site_name: elevation_m}
# - 'latitudes': {site_name: latitude_deg}
# - 'longitudes': {site_name: longitude_deg}

print("  ✓ Metadata loaded successfully\n")

# ========================================================================
# STEP 3.5: Manual Deployment Masking (Alternative Method)
# ========================================================================
print("Step 3.5: Demonstrating manual deployment masking...")

# Show how to apply deployment masking manually since load_all_site_data
# couldn't do it due to missing imports
if csv_deployment_path.exists():
    print("  Applying manual deployment masking to first site...")
    
    for site_name, sensors in list(site_data.items())[:1]:  # Just first site
        for height, sensor_info in sensors.items():
            dataset = sensor_info['dataset']
            sensor_id = sensor_info['sensor_id']
            
            # Get temperature data
            temp_data = dataset['temp_c'].dropna('datetime_utc')
            if len(temp_data) > 0:
                # Find matching CSV filename for deployment periods
                csv_filename = find_matching_csv_filename(
                    sensor_id, site_name, height, csv_deployment_path
                )
                
                if csv_filename:
                    # Apply deployment masking
                    deployed_mask = mask_from_csv_deployment_periods(
                        temp_data.values,
                        temp_data.datetime_utc.values,
                        csv_filename,
                        csv_deployment_path
                    )
                    
                    # Add mask to sensor info
                    sensor_info['deployed_mask'] = deployed_mask
                    deployed_fraction = np.mean(deployed_mask)
                    
                    print(f"    {site_name} {height}: {deployed_fraction:.1%} deployed")
                else:
                    print(f"    {site_name} {height}: No matching CSV found")
else:
    print("  No deployment CSV available for masking")

print("  ✓ Manual deployment masking demonstrated\n")

# ========================================================================
# STEP 4: Demonstrate Data Access Patterns
# ========================================================================
print("Step 4: Demonstrating data access patterns...")

# Example 1: Iterate through all sites and sensors
print("\n  Example 1: Site and sensor inventory")
for site_name, sensors in site_data.items():
    elevation = metadata['elevations'].get(site_name, 'Unknown')
    print(f"    {site_name} (elevation: {elevation}m):")
    
    for height, sensor_info in sensors.items():
        sensor_id = sensor_info['sensor_id']
        dataset = sensor_info['dataset']
        
        # Get temperature data summary
        temp_data = dataset['temp_c'].dropna('datetime_utc')
        if len(temp_data) > 0:
            start_time = temp_data.datetime_utc.min().values
            end_time = temp_data.datetime_utc.max().values
            n_points = len(temp_data)
            
            print(f"      {height} sensor (ID: {sensor_id})")
            # Handle different timestamp formats safely
            try:
                start_str = pd.Timestamp(start_time).strftime('%Y-%m-%d %H:%M')
                end_str = pd.Timestamp(end_time).strftime('%Y-%m-%d %H:%M')
            except:
                start_str = str(start_time)
                end_str = str(end_time)
            print(f"        Period: {start_str} to {end_str}")
            print(f"        Data points: {n_points}")
            
            # Show deployment masking info if available
            if 'deployed_mask' in sensor_info:
                deployed_points = np.sum(sensor_info['deployed_mask'])
                deployment_fraction = deployed_points / len(sensor_info['deployed_mask'])
                print(f"        Deployed data: {deployed_points} points ({deployment_fraction:.1%})")

# Example 2: Access temperature data for a specific site
print("\n  Example 2: Accessing temperature data")

# Find a site with both 1m and 2m sensors
example_site = None
ds_1m = None
ds_2m = None

for site_name, sensors in site_data.items():
    if '1m' in sensors and '2m' in sensors:
        example_site = site_name
        # Get datasets for both heights
        ds_1m = site_data[example_site]['1m']['dataset']
        ds_2m = site_data[example_site]['2m']['dataset']
        break

if example_site and ds_1m is not None and ds_2m is not None:
    print(f"    Using site: {example_site}")
    
    # Extract temperature data
    temp_1m = ds_1m['temp_c'].dropna('datetime_utc')
    temp_2m = ds_2m['temp_c'].dropna('datetime_utc')
    
    print(f"      1m sensor: {len(temp_1m)} temperature points")
    print(f"      2m sensor: {len(temp_2m)} temperature points")
    
    # DATA STATE: temp_1m and temp_2m are xarray DataArrays with:
    # - values: temperature in Celsius
    # - datetime_utc coordinate: pandas datetime index in UTC
    # - attributes: sensor metadata

# Example 3: Find overlapping periods and align data
print("\n  Example 3: Time alignment and overlapping periods")

if example_site and ds_1m is not None and ds_2m is not None:
    # Find overlapping time period
    try:
        start_time, end_time = find_overlapping_period(ds_1m, ds_2m)
    except Exception as e:
        print(f"    Error finding overlapping period: {e}")
        start_time, end_time = None, None
    
    if start_time and end_time:
        overlap_hours = (end_time - start_time).total_seconds() / 3600
        print(f"    Overlapping period: {start_time} to {end_time}")
        print(f"    Duration: {overlap_hours:.1f} hours")
        
        # Create common time grid and interpolate
        common_times = pd.date_range(start_time, end_time, freq='10min')
        valid_times, [temp_1m_aligned, temp_2m_aligned] = interpolate_to_common_grid(
            [ds_1m, ds_2m], common_times
        )
        
        print(f"    Aligned data: {len(valid_times)} time points")
        print(f"    Temperature difference (2m - 1m): {np.mean(temp_2m_aligned - temp_1m_aligned):.2f}°C")
        
        # DATA STATE: temp_1m_aligned and temp_2m_aligned are numpy arrays
        # - Same length as valid_times
        # - Interpolated to common 10-minute grid
        # - Only periods where both sensors have valid data
    else:
        print("    No overlapping period found between 1m and 2m sensors")

print("  ✓ Data access examples completed\n")

# ========================================================================
# STEP 5: Alternative Loading Methods
# ========================================================================
print("Step 5: Demonstrating alternative loading methods...")

# Method 1: Load a single NetCDF file directly
print("\n  Method 1: Direct single file loading")
single_file = netcdf_files[0]
print(f"    Loading: {single_file.name}")

ds_single = read_hobo_pendant(single_file)
print(f"    Variables: {list(ds_single.data_vars)}")
print(f"    Attributes: {ds_single.attrs}")

# DATA STATE: ds_single is an xarray Dataset with:
# - temp_c: temperature data
# - possibly intensity_lux: light intensity data
# - datetime_utc coordinate
# - sensor metadata in attributes



# ========================================================================
# SUMMARY
# ========================================================================
print("=== SUMMARY ===")
print(f"Successfully loaded data from {len(site_data)} sites")
print(f"Total NetCDF files available: {len(netcdf_files)}")
print(f"Deployment masking: {'Enabled' if use_csv_masking else 'Disabled'}")
print("\nData is now ready for analysis!")
print("\nNext steps:")
print("  1. Use site_data dictionary to access temperature time series")
print("  2. Apply deployment masks to filter to field-deployed periods")
print("  3. Use time alignment functions for multi-sensor analysis")
print("  4. Access site metadata for spatial analysis")


