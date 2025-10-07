#!/usr/bin/env python3
"""
merge_intermediate_pendant_data.py

Script to merge pendant sensor data from individual sensor files (by_sensor/)
into site-combined files (by_site/). This script:

1. Loads data from data/2025/intermediate/pendants/by_sensor/
2. Groups sensors by site using load_all_pendant_data()
3. Concatenates datasets along sensor_height coordinate
4. Applies deployed masks to filter non-deployment periods
5. Saves combined datasets as {site_name}.nc in by_site/ directory
6. Handles subfolder structure (camp_wx/, intensive/) with individual sites

Created: 2025-10-03
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

from jiflr import ROOT
from jiflr.data import load_all_pendant_data
from jiflr.utils import get_deployment_periods


def merge_site_data(site_data_dict):
    """
    Merge multiple sensor datasets for a site into a single dataset.

    Parameters
    ----------
    site_data_dict : dict
        Dictionary with sensor height as keys and sensor info as values
        (output from load_all_pendant_data for a single site)
        Note: deployment masks should already be applied during data loading

    Returns
    -------
    xarray.Dataset
        Combined dataset with sensor_height as coordinate
    """
    datasets = []
    sensor_heights = []

    for height, sensor_info in site_data_dict.items():
        ds = sensor_info["dataset"].copy()

        # Round datetime coordinates to nearest minute for consistent time alignment
        rounded_datetime = ds.datetime.dt.round("min")
        ds = ds.assign_coords(datetime=rounded_datetime)

        # Remove any duplicates created by rounding (keep first occurrence)
        datetime_values = ds.datetime.values
        u, indices, counts = np.unique(
            datetime_values, return_inverse=True, return_counts=True
        )
        if np.any(counts > 1):
            # Keep only the first occurrence of each unique datetime
            first_occurrence_mask = np.zeros(len(datetime_values), dtype=bool)
            first_occurrence_mask[np.unique(indices, return_index=True)[1]] = True
            ds = ds.isel(datetime=first_occurrence_mask)

        # Preserve sensor metadata in attributes
        ds.attrs.update(
            {
                f"sensor_id_{height}": sensor_info["sensor_id"],
                f"sensor_file_{height}": sensor_info.get("file", "unknown"),
            }
        )

        datasets.append(ds)
        sensor_heights.append(height)

    if not datasets:
        return None

    # Instead of concat, use merge to handle misaligned time coordinates
    # First create a common time grid by finding the union of all time points
    all_times = []
    for ds in datasets:
        all_times.extend(ds.datetime.values)

    # Get unique sorted times
    unique_times = np.unique(sorted(set(all_times)))

    # Create a new combined dataset by reindexing each dataset to the common time grid
    reindexed_datasets = []
    for ds, height in zip(datasets, sensor_heights):
        # Reindex to common time grid
        ds_reindexed = ds.reindex(datetime=unique_times, method=None)
        reindexed_datasets.append(ds_reindexed)

    # Now concatenate along height dimension (all have same time grid)
    combined_ds = xr.concat(reindexed_datasets, dim="height")

    # Update global attributes
    first_sensor = list(site_data_dict.values())[0]
    combined_ds.attrs.update(
        {
            "site_name": first_sensor["site_name"],
            "sensor_type": "hobo pendant",
            "processing_step": "site_combined",
            "sensor_heights": ", ".join(sensor_heights),
            "n_sensors": len(datasets),
        }
    )

    return combined_ds


def create_qc_plots(combined_ds, site_name, output_dir, deployment_periods=None):
    """
    Create quality control plots for a merged site dataset.

    Parameters
    ----------
    combined_ds : xarray.Dataset
        Combined dataset for the site with dimensions (height, datetime)
    site_name : str
        Name of the site for plot titles and filename
    output_dir : Path
        Directory to save QC plots
    deployment_periods : pandas.DataFrame, optional
        DataFrame containing deployment period information for shading
    """
    # Ensure QC plots directory exists
    qc_plots_dir = output_dir / "qc_plots"
    qc_plots_dir.mkdir(parents=True, exist_ok=True)

    # Set up the figure with subplots
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        f"Quality Control Plots - Site {site_name}", fontsize=16, fontweight="bold"
    )

    # Create subplot layout: temperature time series spans full width on top,
    # box plot and summary on bottom
    gs = fig.add_gridspec(2, 2, height_ratios=[2, 1], width_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, :])  # Top row, full width
    ax2 = fig.add_subplot(gs[1, 0])  # Bottom left
    ax4 = fig.add_subplot(gs[1, 1])  # Bottom right

    # Extract data
    temp_data = combined_ds["temp_c"]
    datetime_coords = combined_ds["datetime"]
    height_coords = combined_ds["height"]

    # Color map for different heights
    colors = plt.cm.Set1(np.linspace(0, 1, len(height_coords)))

    # Get deployment periods for this site using standardized function
    site_deployment_periods = []
    if deployment_periods is not None:
        try:
            # Use the standardized deployment period parsing from utils
            csv_deployment_path = (
                Path(ROOT) / "data" / "2025" / "metadata" / "deployment_periods.csv"
            )
            deployment_periods_dict = get_deployment_periods(
                site_name, csv_deployment_path
            )
            planned_periods = deployment_periods_dict.get(site_name, [])

            # Adjust deployment periods based on actual data availability
            if planned_periods:
                actual_data_start = pd.Timestamp(datetime_coords.values[0])
                actual_data_end = pd.Timestamp(datetime_coords.values[-1])

                for planned_start, planned_end in planned_periods:
                    # Use actual data start if it's later than planned start
                    adjusted_start = max(planned_start, actual_data_start)
                    # Use actual data end if it's earlier than planned end
                    adjusted_end = min(planned_end, actual_data_end)

                    # Only include period if there's actual overlap
                    if adjusted_start < adjusted_end:
                        site_deployment_periods.append(
                            (planned_start, planned_end, adjusted_start, adjusted_end)
                        )
        except Exception as e:
            print(
                f"    Warning: Could not load deployment periods for {site_name}: {e}"
            )
            site_deployment_periods = []

    # Plot 1: Time series of temperature by height (full width)
    # ax1 already defined above
    for i, height in enumerate(height_coords.values):
        height_data = temp_data.sel(height=height)
        # Only plot non-NaN values
        valid_mask = ~np.isnan(height_data.values)
        if np.any(valid_mask):
            ax1.plot(
                datetime_coords[valid_mask],
                height_data.values[valid_mask],
                color=colors[i],
                label=f"{height}",
                alpha=0.7,
                linewidth=1,
            )

    # Add deployment period shading to time series plot
    if site_deployment_periods:
        for i, (planned_start, planned_end, adjusted_start, adjusted_end) in enumerate(
            site_deployment_periods
        ):
            ax1.axvspan(
                adjusted_start,
                adjusted_end,
                alpha=0.2,
                color="gray",
                label="Deployment Period" if i == 0 else "",
            )

    ax1.set_xlabel("Date")
    ax1.set_ylabel("Temperature (°C)")
    ax1.set_title("Temperature Time Series by Height")
    ax1.legend(title="Height", bbox_to_anchor=(1.05, 1), loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Temperature distribution by height (box plot)
    # ax2 already defined above
    temp_data_for_box = []
    height_labels = []
    for height in height_coords.values:
        height_data = temp_data.sel(height=height).values
        # Remove NaN values
        height_data_clean = height_data[~np.isnan(height_data)]
        if len(height_data_clean) > 0:
            temp_data_for_box.append(height_data_clean)
            height_labels.append(str(height))

    if temp_data_for_box:
        bp = ax2.boxplot(temp_data_for_box, labels=height_labels, patch_artist=True)
        for patch, color in zip(bp["boxes"], colors[: len(temp_data_for_box)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_xlabel("Height")
    ax2.set_ylabel("Temperature (°C)")
    ax2.set_title("Temperature Distribution by Height")
    ax2.grid(True, alpha=0.3)

    # Plot 3: Summary statistics
    # ax4 already defined above
    ax4.axis("off")  # Turn off axes for text summary

    # Calculate summary statistics
    stats_text = f"Site {site_name} - Data Summary\n"
    stats_text += "=" * 30 + "\n\n"
    stats_text += f"Number of sensors: {len(height_coords)}\n"
    stats_text += f"Heights: {', '.join([str(h) for h in height_coords.values])}\n"
    stats_text += (
        f"Data period: {datetime_coords.values[0]} to {datetime_coords.values[-1]}\n"
    )
    stats_text += f"Total time points: {len(datetime_coords)}\n\n"

    # Add deployment period information
    stats_text += "Deployment Periods:\n"
    stats_text += "-" * 20 + "\n"
    if site_deployment_periods:
        for i, (planned_start, planned_end, adjusted_start, adjusted_end) in enumerate(
            site_deployment_periods
        ):
            stats_text += f"Period {i + 1}:\n"
            stats_text += f"  Start: {adjusted_start.strftime('%Y-%m-%d %H:%M:%S')}\n"
            stats_text += f"  End: {adjusted_end.strftime('%Y-%m-%d %H:%M:%S')}\n"
            duration = adjusted_end - adjusted_start
            stats_text += f"  Duration: {duration.days} days\n\n"
    else:
        stats_text += "No deployment periods found\n\n"

    stats_text += "Temperature Statistics by Height:\n"
    stats_text += "-" * 35 + "\n"

    for height in height_coords.values:
        height_data = temp_data.sel(height=height).values
        valid_data = height_data[~np.isnan(height_data)]

        if len(valid_data) > 0:
            coverage = len(valid_data) / len(height_data) * 100
            stats_text += f"{height}:\n"
            stats_text += f"  Coverage: {coverage:.1f}%\n"
            stats_text += f"  Mean: {np.mean(valid_data):.2f}°C\n"
            stats_text += (
                f"  Range: {np.min(valid_data):.2f} to {np.max(valid_data):.2f}°C\n"
            )
            stats_text += f"  Std: {np.std(valid_data):.2f}°C\n\n"

    ax4.text(
        0.05,
        0.95,
        stats_text,
        transform=ax4.transAxes,
        fontsize=10,
        verticalalignment="top",
        fontfamily="monospace",
    )

    # Adjust layout and save
    plt.tight_layout()

    # Save the plot
    output_file = qc_plots_dir / f"{site_name}_qc.png"
    plt.savefig(output_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"    QC plot saved: {output_file}")


def process_directory(input_dir, output_dir, csv_deployment_path):
    """
    Process all NetCDF files in a directory and merge by site.

    Parameters
    ----------
    input_dir : Path
        Input directory containing individual sensor NetCDF files
    output_dir : Path
        Output directory for combined site files
    csv_deployment_path : Path
        Path to deployment periods CSV
    """
    print(f"Processing directory: {input_dir}")

    # Load deployment periods CSV for QC plotting
    deployment_periods = None
    if csv_deployment_path.exists():
        try:
            deployment_periods = pd.read_csv(csv_deployment_path)
        except Exception as e:
            print(f"Warning: Could not load deployment periods CSV: {e}")

    # Load all site data from input directory
    site_data = load_all_pendant_data(
        processed_dir=input_dir,
        csv_deployment_path=csv_deployment_path,
        use_csv_masking=True,  # Enable deployed masking
        required_heights=None,  # Load all available heights
        drop_events=True,
        drop_light=False,  # Preserve light data
        apply_mask_to_data=True,  # Apply masks during loading to avoid dimension issues
    )

    print(f"Found {len(site_data)} sites")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each site
    for site_name, sensors in tqdm(site_data.items(), desc="Merging sites"):
        print(f"  Processing site: {site_name}")

        # Merge sensors for this site (masks already applied during loading)
        combined_ds = merge_site_data(sensors)

        if combined_ds is None:
            print(f"    Warning: No valid data for site {site_name}")
            continue

        # Create output filename
        output_file = output_dir / f"{site_name}.nc"

        # Save combined dataset
        combined_ds.to_netcdf(output_file)
        print(f"    Saved: {output_file}")

        # Create QC plots for this site
        create_qc_plots(combined_ds, site_name, output_dir, deployment_periods)

        # Print summary
        n_sensors = len(sensors)
        heights = list(sensors.keys())
        data_points = len(combined_ds.datetime) if "datetime" in combined_ds.dims else 0
        print(
            f"    Combined {n_sensors} sensors ({', '.join(heights)}) -> {data_points} time points"
        )


if __name__ == "__main__":
    print("=== JIFLR Pendant Data Merging ===\n")

    # Define paths
    base_dir = Path(ROOT) / "data" / "2025" / "intermediate" / "pendants"
    input_base = base_dir / "by_sensor"
    output_base = base_dir / "by_site"
    csv_deployment_path = (
        Path(ROOT) / "data" / "2025" / "metadata" / "deployment_periods.csv"
    )

    print(f"Input base directory: {input_base}")
    print(f"Output base directory: {output_base}")
    print(f"Deployment CSV: {csv_deployment_path}")
    print(f"CSV exists: {csv_deployment_path.exists()}\n")

    # Process main directory
    main_input = input_base
    main_output = output_base

    print("Processing main directory...")
    process_directory(main_input, main_output, csv_deployment_path)

    # Process subdirectories
    # TODO: Make this procedural
    subdirs = ["camp_wx", "intensive"]

    for subdir in subdirs:
        subdir_input = input_base / subdir
        subdir_output = output_base / subdir

        if subdir_input.exists() and any(subdir_input.glob("*.nc")):
            print(f"\nProcessing {subdir} subdirectory...")
            process_directory(subdir_input, subdir_output, csv_deployment_path)
        else:
            print(f"\nSkipping {subdir} subdirectory (not found or empty)")

    print("\n=== Merging Complete ===")
