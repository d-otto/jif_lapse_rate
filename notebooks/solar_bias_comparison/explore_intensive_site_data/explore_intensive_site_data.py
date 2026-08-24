# %%
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
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
ds = xr.open_dataset(lvl1_path)

print(f"Dataset dimensions: {dict(ds.sizes)}")
print(f"Dataset variables: {list(ds.data_vars)}")
print(f"Available site_ids: {sorted(ds.site_id.values.tolist())}")

ds

#%%


# %%
# Filter the dataset to include only our analysis sites and apply sensor filtering
all_analysis_sites = ["Windward2", "Windward1", "Divide", "Lee1", "Lee2"]

analysis_mask = ds.site_id.isin(all_analysis_sites)
ds_analysis = ds.where(analysis_mask, drop=True)
ds_analysis

#%%


# For all sites, filter to use only sensors at 2m height from intensive sites
intensive_sites = ["Windward2", "Windward1", "Divide", "Lee1", "Lee2"]

# Filter to intensive sites first
intensive_mask = ds_analysis.site_id.isin(intensive_sites)
ds_intensive_all = ds_analysis.where(intensive_mask, drop=True)

print("\\nAll sensors at intensive sites (before height filtering):")
for site in intensive_sites:
    site_all = ds_intensive_all.where(ds_intensive_all.site_id == site, drop=True)
    if len(site_all.sensor_idx) > 0:
        print(f"  {site}:")
        for i in range(len(site_all.sensor_idx)):
            height = site_all.height.values[i]
            shielding = site_all.shielding.values[i]
            sensor_type = site_all.sensor_type.values[i]
            sensor_id = site_all.sensor_id.values[i]
            temp_count = site_all.temp_c.isel(sensor_idx=i).count().values
            print(f"    - {sensor_id} ({height}, {shielding}, {sensor_type}) - {temp_count} temp observations")

# Now filter to 2m height sensors (include both "2m" and "2.0m")
height_mask = (ds_intensive_all.height == "2m") | (ds_intensive_all.height == "2.0m")
ds_analysis = ds_intensive_all.where(height_mask, drop=True)

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

#%%
# Analyze the filtered dataset
print(f"Filtered dataset dimensions: {dict(ds_analysis.sizes)}")
print(f"Temperature data variable shape: {ds_analysis.temp_c.shape}")

# Check how many sensors we have per intensive site
print("\nSensors per intensive site (2m height only):")
for site in intensive_sites:
    site_sensors = ds_analysis.where(ds_analysis.site_id == site, drop=True)
    n_sensors = len(site_sensors.sensor_idx) if len(site_sensors.sensor_idx) > 0 else 0
    
    if n_sensors > 0:
        # Filter to only temperature sensors (those that have meaningful temp_c data)
        temp_sensors = []
        for i in range(n_sensors):
            temp_data = site_sensors.temp_c.isel(sensor_idx=i)
            n_valid_temps = temp_data.count().values
            
            if n_valid_temps > 100:  # Only include sensors with substantial temperature data
                shielding = site_sensors.shielding.values[i]
                sensor_type = site_sensors.sensor_type.values[i]
                sensor_id = site_sensors.sensor_id.values[i]
                temp_sensors.append({
                    'idx': i,
                    'sensor_id': sensor_id,
                    'shielding': shielding,
                    'sensor_type': sensor_type,
                    'n_temps': int(n_valid_temps)
                })
        
        print(f"  {site}: {len(temp_sensors)} temperature sensors")
        for sensor in temp_sensors:
            print(f"    - {sensor['sensor_id']} ({sensor['shielding']}/{sensor['sensor_type']}) - {sensor['n_temps']:,} observations")
    else:
        print(f"  {site}: 0 sensors")

# Create individual comparison plots for each intensive site
for site in intensive_sites:
    site_data = ds_analysis.where(ds_analysis.site_id == site, drop=True)
    n_sensors = len(site_data.sensor_idx) if len(site_data.sensor_idx) > 0 else 0
    
    if n_sensors == 0:
        print(f"\nSkipping {site} - no 2m sensors found")
        continue
    
    # Filter to only temperature sensors (those that have meaningful temp_c data)
    temp_sensors = []
    for i in range(n_sensors):
        temp_data = site_data.temp_c.isel(sensor_idx=i)
        n_valid_temps = temp_data.count().values
        
        if n_valid_temps > 100:  # Only include sensors with substantial temperature data
            temp_sensors.append({
                'idx': i,
                'temp_data': temp_data,
                'sensor_id': site_data.sensor_id.values[i],
                'shielding': site_data.shielding.values[i],
                'sensor_type': site_data.sensor_type.values[i]
            })
    
    if len(temp_sensors) == 0:
        print(f"\nSkipping {site} - no temperature sensors found")
        continue
        
    print(f"\nCreating comparison plot for {site} ({len(temp_sensors)} temperature sensors at 2m)")
    
    # Create figure with 3 rows
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    
    # Plot each temperature sensor's data
    colors = plt.cm.Set1(np.linspace(0, 1, max(len(temp_sensors), 3)))
    
    # Define time periods for each row
    time_periods = [
        (None, None, "Full Time Series"),
        ("2025-06-26", "2025-07-01", "Last 5 Days of June"),
        ("2025-07-17", "2025-07-23", "July 17-22")
    ]
    
    for row_idx, (start_time, end_time, title_suffix) in enumerate(time_periods):
        ax = axes[row_idx]
        
        for i, sensor in enumerate(temp_sensors):
            # Create label with sensor details
            label = f"{sensor['sensor_id']} ({sensor['shielding']}, {sensor['sensor_type']})"
            
            # Filter time period if specified
            if start_time and end_time:
                temp_data_subset = sensor['temp_data'].sel(datetime=slice(start_time, end_time))
            else:
                temp_data_subset = sensor['temp_data']
            
            # Plot temperature timeseries
            temp_data_subset.plot(ax=ax, label=label, color=colors[i], linewidth=1.5)
        
        # Customize plot
        ax.set_title(f"Temperature Comparison at {site} - 2m Sensors ({title_suffix})", fontsize=14, fontweight='bold')
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel("Temperature (°C)", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Adjust layout and save
    plt.tight_layout()
    output_file = output_dir / f"{site}_2m_temperature_comparison_3panel.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    plt.show()
    plt.close()

print("\nCompleted temperature sensor comparison plots for intensive sites.")