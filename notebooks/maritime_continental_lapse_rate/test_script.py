import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from pathlib import Path
from jiflr import ROOT
import os

plt.style.use("default")

# Create output directory for plots
output_dir = Path(__file__).parent / "output"
output_dir.mkdir(exist_ok=True)

# %% [markdown]
# # Initial Lapse Rate Analysis
#
# Calculate time-varying lapse rates between maritime and continental site groups using lvl1 sensor data. This analysis computes hourly lapse rates over the temporal period where all sites have overlapping data coverage.

# %%
# Load the combined lvl1 data (includes both regular and intensive sites)
lvl1_path = ROOT / "data" / "2025" / "processed" / "lvl1" / "lvl1_combined.nc"

print(f"Attempting to load: {lvl1_path}")
print(f"File exists: {lvl1_path.exists()}")

try:
    ds = xr.open_dataset(lvl1_path)
except Exception as e:
    print(f"ERROR loading dataset: {e}")
    print("\nThis script requires NetCDF dependencies.")
    print("Try running in the conda environment or install netcdf4:")
    print("  conda activate jiflr")
    print("  OR")
    print("  pip install netcdf4")
    exit(1)

print(f"Dataset dimensions: {dict(ds.sizes)}")
print(f"Dataset variables: {list(ds.data_vars)}")
print(f"Available site_ids: {sorted(ds.site_id.values.tolist())}")

# Debug light data availability
if "intensity_lux" in ds.data_vars:
    light_count = ds.intensity_lux.count().values
    print(f"DEBUG: Found intensity_lux with {light_count} total data points")
    # Check which sites have light data
    sites_with_light = []
    for site_id in sorted(ds.site_id.values.tolist()):
        site_mask = ds.site_id == site_id
        site_light_count = ds.intensity_lux.where(site_mask, drop=True).count().values
        if site_light_count > 0:
            sites_with_light.append(f"{site_id}({site_light_count})")
    print(f"DEBUG: Sites with light data: {sites_with_light}")
else:
    print("DEBUG: No intensity_lux variable found in dataset")

# %%
# Configure Butterworth filter for plotting
# For 5-minute data: fs = 1/300 Hz (sampling frequency)
# upper = 1/6 means cutoff at 6 samples = 30 minutes

FILTER_ORDER = 4
FILTER_LOWER = None  # Low-pass filter only
# FILTER_FS = 1 / 300  # Sampling frequency for 5-minute data (1 sample per 300 seconds)
# FILTER_UPPER = 1 / (60 * 60)  # 1 hr cutoff
FILTER_UPPER = 1 / (2 * 60 * 60)  # 2 hr cutoff

# For 15-min data: fs = 1/(300 * 3) Hz
FILTER_FS = 1 / (300 * 3)


# %%
# ============================================================================
# COLOR CONFIGURATION
# ============================================================================
# Configure color scheme for all plots using cmocean tarn colormap
# Maritime: warm colors (higher values in tarn colormap)
# Continental: cool colors (lower values in tarn colormap)
# Difference: neutral grey

# Single color configurations
MARITIME_COLOR = "blue"  # Single maritime color
CONTINENTAL_COLOR = "red"  # Single continental color
DIFFERENCE_COLOR = "purple"  # Color for difference plots

MARITIME_COLORS = [plt.cm.viridis(0.8), plt.cm.viridis(0.5), plt.cm.viridis(0.2)]
CONTINENTAL_COLORS = [plt.cm.magma(0.8), plt.cm.magma(0.6), plt.cm.magma(0.4)]


# Color mapping functions for easier use
def get_maritime_color(index=0):
    """Get maritime color by index, cycling through available colors"""
    return MARITIME_COLORS[index % len(MARITIME_COLORS)]


def get_continental_color(index=0):
    """Get continental color by index, cycling through available colors"""
    return CONTINENTAL_COLORS[index % len(CONTINENTAL_COLORS)]


# %%
# Define site groups based on the available site_ids
# Note: We'll need to map the requested site names to actual site_ids in the dataset

# Maritime sites (as requested): Windward1, Windward2, A04, A05, A06, Divide
# Continental sites (as requested): A07, Lee1, Lee2, A08, A10

# Note: The dataset uses 'Wind' and 'Divi' but we need to separate Wind into Windward1/Windward2
# and rename Divi to Divide for proper site naming
maritime_sites = [
    "Windward1",
    "Windward2",
    "A05",
    "A06",
    "A07",
    "Divide",
]
continental_sites = ["Divide", "Lee1", "Lee2", "A08"]  # A10 removed

print("Requested maritime sites:", maritime_sites)
print(
    "Available maritime sites found:",
    [s for s in maritime_sites if s in ds.site_id.values],
)
print("Requested continental sites:", continental_sites)
print(
    "Available continental sites found:",
    [s for s in continental_sites if s in ds.site_id.values],
)

# Get the union of all sites needed for analysis
all_analysis_sites = maritime_sites + continental_sites
print(f"All analysis sites: {all_analysis_sites}")

# %%
# Filter the dataset to include only our analysis sites and apply sensor filtering
analysis_mask = ds.site_id.isin(all_analysis_sites)
ds_analysis = ds.where(analysis_mask, drop=True)

# For each site, select the best available temperature sensor at 2m height
print("\nApplying sensor filtering to use best available 2m temperature sensors...")
intensive_sites = ["Windward2", "Windward1", "Divide", "Lee1", "Lee2"]

# Group sensors by site and select best 2m sensor for each site
site_sensors = {}
for i, site_id in enumerate(ds_analysis.site_id.values):
    if site_id not in site_sensors:
        site_sensors[site_id] = []

    # Check if this sensor has temperature data and is at 2m
    height = ds_analysis.height.values[i]
    is_2m = height in ["2m", "2.0m"]  # Handle both formats

    # Check if sensor has substantial temperature data
    temp_data = ds_analysis.temp_c.isel(sensor_idx=i)
    valid_temp_count = int(np.sum(~np.isnan(temp_data.values)))

    if is_2m and valid_temp_count > 100:  # Substantial temperature data
        site_sensors[site_id].append(
            {
                "index": i,
                "sensor_id": ds_analysis.sensor_id.values[i],
                "height": height,
                "shielding": ds_analysis.shielding.values[i],
                "sensor_type": ds_analysis.sensor_type.values[i],
                "temp_count": valid_temp_count,
            }
        )

# ============================================================================
# SENSOR PRIORITY CONFIGURATION
# ============================================================================
# Define sensor selection priority (1 = highest priority)
# Each entry: (filter_function, description)
SENSOR_PRIORITIES = [
    (
        lambda s: s["shielding"] == "shielded" and "hobo" in s["sensor_type"],
        "Shielded hobo",
    ),
    (
        lambda s: s["sensor_type"] == "pace" and "temp" in s["sensor_id"],
        "PACE temperature sensor",
    ),
    # NOTE: Removed unshielded hobo - only want shielded temperature sensors
]
# ============================================================================


def select_best_sensor(sensors, priorities):
    """Select the best sensor based on priority list."""
    for filter_func, description in priorities:
        matching_sensors = [s for s in sensors if filter_func(s)]
        if matching_sensors:
            return max(matching_sensors, key=lambda x: x["temp_count"])
    return None


# For each site, select the best sensor based on priority
sensor_filters = [False] * len(ds_analysis.sensor_idx)
selected_sensors = {}

for site_id, sensors in site_sensors.items():
    if not sensors:
        continue

    best_sensor = select_best_sensor(sensors, SENSOR_PRIORITIES)

    if best_sensor:
        sensor_filters[best_sensor["index"]] = True
        selected_sensors[site_id] = best_sensor
        print(
            f"  {site_id}: selected {best_sensor['sensor_id']} "
            f"({best_sensor['height']}, {best_sensor['shielding']}, "
            f"{best_sensor['sensor_type']}) - {best_sensor['temp_count']} obs"
        )

# Identify which sensors to keep for the analysis
print("Identifying which sensors to keep...")

# Get the indices of the carefully selected temperature sensors (already filtered for 2m + shielding)
selected_temp_indices = [sensor["index"] for sensor in selected_sensors.values()]
print(f"  Selected temperature sensor indices: {len(selected_temp_indices)}")

# Find sensors with valuable non-temperature data (wind, pressure, light, humidity) 
# These should be kept regardless of height/shielding
valuable_nontemp_indices = []
nontemp_vars = ['wind_speed_avg', 'wind_speed_peak', 'wind_direction', 'pressure', 
                'intensity_lux', 'humidity_percent']

for i in range(len(ds_analysis.sensor_idx)):
    # Skip 0m sensors entirely - they should not be kept at all
    height = ds_analysis.height.values[i]
    if height in ["0m", "0.0m"]:
        continue
    
    has_valuable_data = False
    
    for var_name in nontemp_vars:
        if var_name in ds_analysis.data_vars:
            var_data = ds_analysis[var_name].isel(sensor_idx=i)
            if not np.isnan(var_data).all().values:
                has_valuable_data = True
                break
    
    if has_valuable_data:
        valuable_nontemp_indices.append(i)

print(f"  Sensors with valuable non-temperature data: {len(valuable_nontemp_indices)}")

# Combine selected temperature sensors with valuable non-temperature sensors
all_needed_indices = selected_temp_indices + valuable_nontemp_indices
final_sensor_indices = sorted(list(set(all_needed_indices)))  # Remove duplicates

print(f"  Total unique sensors to keep: {len(final_sensor_indices)}")
print(f"  Removed sensors: {len(ds_analysis.sensor_idx) - len(final_sensor_indices)}")

# Apply filtering to dataset
ds_analysis = ds_analysis.isel(sensor_idx=final_sensor_indices)

# IMPORTANT: Mask out temperature data from unshielded 2m sensors
# (Keep the sensors for their light/wind/pressure data, but don't use their temperature)
print("Masking temperature data from unshielded 2m sensors...")
masked_temp_sensors = 0

for i in range(len(ds_analysis.sensor_idx)):
    height = ds_analysis.height.values[i]
    shielding = ds_analysis.shielding.values[i]
    sensor_id = ds_analysis.sensor_id.values[i]
    
    # If this is an unshielded 2m sensor, mask out its temperature data
    # EXCEPT for Windward2 which has no shielded sensors available
    if height in ["2m", "2.0m"] and shielding in ["unshielded", "unshield"]:
        site_id = ds_analysis.site_id.values[i]
        if site_id == "Windward2":
            continue  # Don't mask Windward2 temperature data - it's all we have
        
        # Check if it actually has temperature data before masking
        temp_data = ds_analysis.temp_c.isel(sensor_idx=i)
        temp_count = int(np.sum(~np.isnan(temp_data.values)))
        
        if temp_count > 0:
            print(f"  Masking temperature data from {sensor_id} (unshielded 2m, {temp_count:,} obs)")
            ds_analysis.temp_c.values[:, i] = np.nan
            masked_temp_sensors += 1

print(f"  Masked temperature data from {masked_temp_sensors} unshielded 2m sensors")

print(f"Analysis dataset dimensions: {dict(ds_analysis.sizes)}")
print(f"Analysis sites: {sorted(ds_analysis.site_id.values.tolist())}")
print(
    f"Elevation range: {ds_analysis.elevation.min().values:.1f}m to {ds_analysis.elevation.max().values:.1f}m"
)

# Check data availability for each site with sensor details
print("\nData availability by site (with sensor filtering applied):")
for site in sorted(set(ds_analysis.site_id.values)):
    site_data = ds_analysis.where(ds_analysis.site_id == site, drop=True)
    n_sensors = len(site_data.sensor_idx)
    valid_temps = site_data.temp_c.count().values
    total_times = len(ds_analysis.datetime)
    coverage = (valid_temps / total_times) * 100 if total_times > 0 else 0
    elevation = float(site_data.elevation.values[0])

    # Show sensor details for this site
    sensor_details = []
    for i in range(n_sensors):
        height = site_data.height.values[i]
        shielding = site_data.shielding.values[i]
        sensor_type = site_data.sensor_type.values[i]
        sensor_details.append(f"{height}/{shielding}/{sensor_type}")

    print(f"{site}: {n_sensors} sensor(s) [{', '.join(sensor_details)}]")
    print(
        f"    {valid_temps:,}/{total_times:,} observations ({coverage:.1f}%) at {elevation:.0f}m"
    )

# ============================================================================
# COMPREHENSIVE SENSOR DIAGNOSTICS  
# ============================================================================
print("\n" + "="*80)
print("COMPREHENSIVE SENSOR DIAGNOSTICS")  
print("="*80)
print("Site-by-site comparison: BEFORE and AFTER filtering")
print()

# First, let's look at the original dataset (before filtering) for our analysis sites
original_analysis = ds.where(ds.site_id.isin(all_analysis_sites), drop=True)

def get_sensor_description(sensor_data):
    """Create a concise sensor description."""
    sensor_id = sensor_data.sensor_id.values.item()
    height = sensor_data.height.values.item()  
    shielding = sensor_data.shielding.values.item()
    sensor_type = sensor_data.sensor_type.values.item()
    
    # For temperature sensors, show height and shielding
    if "hobo" in sensor_type.lower() or "temp" in sensor_id.lower():
        return f"{sensor_id}_{height}_{shielding}_{sensor_type}"
    else:
        return f"{sensor_id}_{sensor_type}"

def get_data_summary(sensor_data):
    """Get summary of non-zero data counts for a sensor."""
    key_vars = ['temp_c', 'wind_speed_avg', 'wind_speed_peak', 'wind_direction', 
               'pressure', 'intensity_lux', 'humidity_percent']
    data_parts = []
    
    for var_name in key_vars:
        if var_name in sensor_data:
            var_data = sensor_data[var_name]
            if var_data.values.ndim == 0:  # scalar
                non_nan_count = 1 if not np.isnan(var_data.values) else 0
            else:  # array
                non_nan_count = int(np.sum(~np.isnan(var_data.values)))
            
            if non_nan_count > 0:  # Only show variables with data
                data_parts.append(f"{var_name}: {non_nan_count:,}")
    
    return " | ".join(data_parts) if data_parts else "no data"

# Go site by site
for site_id in sorted(all_analysis_sites):
    print(f"\nSite: {site_id}")
    print("="*50)
    
    # Check if site exists in original data
    if site_id not in original_analysis.site_id.values:
        print("  *** SITE NOT FOUND IN DATASET ***")
        continue
    
    # BEFORE filtering
    print("Before filtering:")
    original_site = original_analysis.where(original_analysis.site_id == site_id, drop=True)
    
    if len(original_site.sensor_idx) == 0:
        print("  No sensors found")
    else:
        for i, sensor_idx in enumerate(original_site.sensor_idx.values):
            sensor_data = original_site.isel(sensor_idx=i)
            description = get_sensor_description(sensor_data)
            data_summary = get_data_summary(sensor_data)
            
            print(f"  sensor_idx={sensor_idx}: {description} | {data_summary}")
    
    # Show which sensor was selected for temperature
    if site_id in selected_sensors:
        selected = selected_sensors[site_id]
        print(f"  *** SELECTED for temperature: {selected['sensor_id']} ***")
    else:
        print("  *** NO TEMPERATURE SENSOR SELECTED ***")
    
    print()
    
    # AFTER filtering  
    print("After filtering:")
    if site_id in ds_analysis.site_id.values:
        filtered_site = ds_analysis.where(ds_analysis.site_id == site_id, drop=True)
        
        if len(filtered_site.sensor_idx) == 0:
            print("  *** NO SENSORS REMAINING ***")
        else:
            for i, sensor_idx in enumerate(filtered_site.sensor_idx.values):
                sensor_data = filtered_site.isel(sensor_idx=i)
                description = get_sensor_description(sensor_data)
                data_summary = get_data_summary(sensor_data)
                
                print(f"  sensor_idx={sensor_idx}: {description} | {data_summary}")
    else:
        print("  *** SITE COMPLETELY FILTERED OUT ***")
    
    print()

# Summary
print("\n" + "="*80)
print("FILTERING SUMMARY")
print("="*80)

print("\nTemperature sensor selections:")
for site_id in sorted(selected_sensors.keys()):
    sensor = selected_sensors[site_id]
    print(f"  {site_id}: {sensor['sensor_id']} ({sensor['height']}, {sensor['shielding']}, {sensor['sensor_type']}) - {sensor['temp_count']:,} obs")

missing_selections = []
for site_id in all_analysis_sites:
    if site_id in original_analysis.site_id.values and site_id not in selected_sensors:
        missing_selections.append(site_id)

if missing_selections:
    print(f"\nSites with NO temperature sensor selected: {missing_selections}")
    print("This will cause lapse rate calculation failures!")

print(f"\nSites remaining after filtering: {sorted(ds_analysis.site_id.values.tolist())}")
missing_sites = [s for s in all_analysis_sites if s not in ds_analysis.site_id.values]
if missing_sites:
    print(f"Sites completely filtered out: {missing_sites}")
else:
    print("All analysis sites preserved ✓")

# Summary by site type
print("\n" + "="*80)
print("SUMMARY BY SITE TYPE")
print("="*80)

print("\nMARITIME SITES:")
available_maritime = [s for s in maritime_sites if s in ds_analysis.site_id.values]
for site in available_maritime:
    site_data = ds_analysis.where(ds_analysis.site_id == site, drop=True)
    temp_count = site_data.temp_c.count().values
    print(f"  {site}: {len(site_data.sensor_idx)} sensors, {temp_count:,} temp observations")

print("\nCONTINENTAL SITES:")  
available_continental = [s for s in continental_sites if s in ds_analysis.site_id.values]
for site in available_continental:
    site_data = ds_analysis.where(ds_analysis.site_id == site, drop=True)
    temp_count = site_data.temp_c.count().values
    print(f"  {site}: {len(site_data.sensor_idx)} sensors, {temp_count:,} temp observations")

print("\nMISSING SITES:")
missing_maritime = [s for s in maritime_sites if s not in ds_analysis.site_id.values]
missing_continental = [s for s in continental_sites if s not in ds_analysis.site_id.values]
if missing_maritime:
    print(f"  Missing maritime: {missing_maritime}")
if missing_continental:
    print(f"  Missing continental: {missing_continental}")
if not missing_maritime and not missing_continental:
    print("  None - all requested sites are available")
