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

from jiflr import ROOT
from jiflr.data import load_all_pendant_data
from jiflr.logging import indent, key_value, setup_pipeline_logging, subheader
from jiflr.pipeline import merge_sites
from jiflr.utils import get_deployment_periods


# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Configuration for merging colocated sites
# Maps target site_id -> list of source site_ids to merge
# Set to empty dict {} to disable colocated site merging
COLOCATED_SITES = {
    "G03": ["G03a", "G03b"],
}

# Join method for merging colocated sites:
#   "inner": only overlapping time period (no NaN gaps)
#   "outer": union of all time periods (NaN-filled gaps)
COLOCATED_MERGE_JOIN = "inner"
# =============================================================================


def merge_site_data(site_data_dict):
    """
    Merge multiple sensor datasets for a site into a single dataset using sensor_idx concatenation.

    Uses sensor_idx structure with sensor attributes as coordinates.
    Creates structure: data_vars(sensor_idx, datetime)

    Parameters
    ----------
    site_data_dict : dict
        Dictionary with sensor height as keys and sensor info as values
        (output from load_all_pendant_data for a single site)
        Note: deployment masks should already be applied during data loading
        Note: All input datasets MUST already have sensor_idx structure

    Returns
    -------
    xarray.Dataset
        Combined dataset with sensor_idx structure
    """
    if not site_data_dict:
        return None

    # Collect all datasets to concatenate
    datasets_to_concat = []
    site_name = None

    for height_key, sensor_info in site_data_dict.items():
        ds = sensor_info["dataset"]

        # Track site name
        if site_name is None:
            site_name = sensor_info.get("site_name", "unknown")

        # Verify the dataset has the correct sensor_idx structure
        if "sensor_idx" not in ds.dims:
            raise ValueError(
                f"Dataset for sensor at {height_key} does not have sensor_idx structure. "
                f"All intermediate data must be regenerated with the new structure."
            )

        datasets_to_concat.append(ds)

    if not datasets_to_concat:
        return None

    # Concatenate all datasets along sensor_idx dimension
    try:
        combined_ds = xr.concat(
            datasets_to_concat,
            dim="sensor_idx",
            data_vars="all",
            coords="all",
            join="outer",
        )
        # Fix sensor_idx to be sequential (0, 1, 2, 3...) instead of all zeros
        new_sensor_idx = list(range(len(combined_ds.sensor_idx)))
        combined_ds = combined_ds.assign_coords(sensor_idx=new_sensor_idx)
    except Exception as e:
        raise ValueError(f"Error concatenating datasets with sensor_idx structure: {e}")

    # Add global attributes
    combined_ds.attrs.update(
        {
            "site_name": site_name,
            "sensor_type": "hobo pendant",
            "processing_step": "site_combined_sensor_idx",
            "n_sensors": len(combined_ds.sensor_idx),
            "structure": "sensor_idx x datetime",
        }
    )

    return combined_ds


def _extract_height_from_sensor(ds, height_key):
    """
    Extract height information from sensor dataset, handling various sources.

    Parameters
    ----------
    ds : xarray.Dataset
        Sensor dataset
    height_key : str
        Height key from the site_data_dict (fallback)

    Returns
    -------
    str
        Standardized height string
    """
    # Try sensor_height attribute first
    height = ds.attrs.get("sensor_height", "").strip()

    # If empty, try sensor_config
    if not height:
        height = ds.attrs.get("sensor_config", "").strip()

    # If still empty, use the height_key from dictionary
    if not height:
        height = height_key

    # Standardize height format
    if height and not height.endswith("m"):
        # Handle cases like "0.5", "2", etc.
        try:
            height_num = float(height.replace("m", ""))
            if height_num == int(height_num):
                height = f"{int(height_num)}m"
            else:
                height = f"{height_num}m"
        except (ValueError, TypeError):
            # Keep as-is if can't parse
            pass

    return height if height else "unknown"


def create_qc_plots(
    combined_ds, site_name, output_dir, deployment_periods=None, logger=None
):
    """
    Create quality control plots for a merged site dataset.

    Parameters
    ----------
    combined_ds : xarray.Dataset
        Combined dataset with dimensions (sensor_idx, datetime)
    site_name : str
        Name of the site for plot titles and filename
    output_dir : Path
        Directory to save QC plots
    deployment_periods : pandas.DataFrame, optional
        DataFrame containing deployment period information for shading
    logger : logging.Logger, optional
        Logger instance for output
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

    # Extract data - now using sensor_idx structure
    temp_data = combined_ds["temp_c"]

    # Get the appropriate datetime coordinate
    if "datetime" in combined_ds.coords:
        datetime_coords = combined_ds["datetime"]
    elif "datetime_utc" in combined_ds.coords:
        datetime_coords = combined_ds["datetime_utc"]
    else:
        if logger:
            logger.warning(f"No datetime coordinate found for site {site_name}")
        return

    # Get sensor metadata from coordinates
    n_sensors = len(combined_ds.sensor_idx)
    sensor_info = []

    for i in range(n_sensors):
        height = (
            combined_ds.height.values[i]
            if "height" in combined_ds.coords
            else f"sensor_{i}"
        )
        shielding = (
            combined_ds.shielding.values[i]
            if "shielding" in combined_ds.coords
            else "unknown"
        )
        sensor_id = (
            combined_ds.sensor_id.values[i]
            if "sensor_id" in combined_ds.coords
            else f"unknown_{i}"
        )

        sensor_info.append(
            {
                "height": height,
                "shielding": shielding,
                "sensor_id": sensor_id,
                "index": i,
                "label": f"{height} ({shielding})",
            }
        )

    # Color map for different sensors
    colors = plt.cm.Set1(np.linspace(0, 1, max(n_sensors, 1)))

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
            if logger:
                logger.warning(
                    f"Could not load deployment periods for {site_name}: {e}"
                )
            site_deployment_periods = []

    # Plot 1: Time series of temperature by sensor (full width)
    for i, sensor in enumerate(sensor_info):
        # Select data for this sensor
        sensor_data = temp_data.isel(sensor_idx=sensor["index"])
        label = sensor["label"]

        # Get the time series values
        sensor_values = sensor_data.values

        # Only plot non-NaN values
        valid_mask = ~np.isnan(sensor_values)
        if np.any(valid_mask):
            valid_times = datetime_coords.values[valid_mask]
            valid_data = sensor_values[valid_mask]

            ax1.plot(
                valid_times,
                valid_data,
                color=colors[i],
                label=label,
                alpha=0.7,
                linewidth=1,
            )

    # Check if light data is available and add to second y-axis
    if "intensity_lux" in combined_ds.data_vars:
        # Create second y-axis for light data
        ax1_light = ax1.twinx()

        light_data = combined_ds["intensity_lux"]
        # Use dashed lines and lighter colors for light data
        light_colors = plt.cm.Set2(np.linspace(0, 1, max(n_sensors, 1)))

        for i, sensor in enumerate(sensor_info):
            # Select light data for this sensor
            sensor_light = light_data.isel(sensor_idx=sensor["index"])
            light_label = f"{sensor['label']} light"

            # Get the time series values
            sensor_light_values = sensor_light.values

            # Only plot non-NaN values
            valid_mask = ~np.isnan(sensor_light_values)
            if np.any(valid_mask):
                valid_times = datetime_coords.values[valid_mask]
                valid_light = sensor_light_values[valid_mask]

                ax1_light.plot(
                    valid_times,
                    valid_light,
                    color=light_colors[i],
                    label=light_label,
                    alpha=0.6,
                    linewidth=1,
                    linestyle="--",
                )

        ax1_light.set_ylabel("Light Intensity (lux)", color="orange")
        ax1_light.tick_params(axis="y", labelcolor="orange")

        # Create combined legend for both axes
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax1_light.get_legend_handles_labels()

        # Only create light legend if there's light data
        if lines2:
            ax1.legend(
                lines1 + lines2,
                labels1 + labels2,
                title="Height/Shielding (temp/light)",
                bbox_to_anchor=(1.05, 1),
                loc="upper left",
            )
        else:
            ax1.legend(
                title="Height/Shielding", bbox_to_anchor=(1.05, 1), loc="upper left"
            )
    else:
        ax1.legend(title="Height/Shielding", bbox_to_anchor=(1.05, 1), loc="upper left")

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
    ax1.set_ylabel("Temperature (C)")
    ax1.set_title("Temperature Time Series by Height")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax1.xaxis.set_major_locator(mdates.DayLocator())
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)

    # Plot 2: Temperature distribution by sensor (box plot)
    temp_data_for_box = []
    sensor_labels = []

    for i, sensor in enumerate(sensor_info):
        # Select data for this sensor
        sensor_data = temp_data.isel(sensor_idx=sensor["index"])
        label = f"{sensor['height']}\n({sensor['shielding']})"

        # Get values and remove NaN
        sensor_values = sensor_data.values
        sensor_data_clean = sensor_values[~np.isnan(sensor_values)]

        if len(sensor_data_clean) > 0:
            temp_data_for_box.append(sensor_data_clean)
            sensor_labels.append(label)

    if temp_data_for_box:
        bp = ax2.boxplot(
            temp_data_for_box, tick_labels=sensor_labels, patch_artist=True
        )
        for patch, color in zip(bp["boxes"], colors[: len(temp_data_for_box)]):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

    ax2.set_xlabel("Height/Shielding")
    ax2.set_ylabel("Temperature (C)")
    ax2.set_title("Temperature Distribution by Height and Shielding")
    ax2.grid(True, alpha=0.3)
    # Rotate labels if needed for better readability
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Plot 3: Summary statistics
    # ax4 already defined above
    ax4.axis("off")  # Turn off axes for text summary

    # Calculate summary statistics
    stats_text = f"Site {site_name} - Data Summary\n"
    stats_text += "=" * 30 + "\n\n"
    stats_text += f"Number of sensors: {n_sensors}\n"
    unique_heights = sorted(set([s["height"] for s in sensor_info]))
    unique_shielding = sorted(set([s["shielding"] for s in sensor_info]))
    stats_text += f"Heights: {', '.join([str(h) for h in unique_heights])}\n"
    stats_text += f"Shielding types: {', '.join([str(s) for s in unique_shielding])}\n"

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

    stats_text += "Temperature Statistics by Sensor:\n"
    stats_text += "-" * 35 + "\n"

    for sensor in sensor_info:
        # Select data for this sensor
        sensor_data = temp_data.isel(sensor_idx=sensor["index"])
        sensor_label = sensor["label"]

        # Get values and calculate statistics
        sensor_values = sensor_data.values
        valid_data = sensor_values[~np.isnan(sensor_values)]

        if len(valid_data) > 0:
            coverage = len(valid_data) / len(sensor_values) * 100
            stats_text += f"{sensor_label} (ID: {sensor['sensor_id']}):\n"
            stats_text += f"  Coverage: {coverage:.1f}%\n"
            stats_text += f"  Mean: {np.mean(valid_data):.2f}C\n"
            stats_text += (
                f"  Range: {np.min(valid_data):.2f} to {np.max(valid_data):.2f}C\n"
            )
            stats_text += f"  Std: {np.std(valid_data):.2f}C\n\n"

    # Add light data statistics if available
    if "intensity_lux" in combined_ds.data_vars:
        light_data = combined_ds["intensity_lux"]
        stats_text += "Light Intensity Statistics by Sensor:\n"
        stats_text += "-" * 38 + "\n"

        for sensor in sensor_info:
            # Select light data for this sensor
            sensor_light = light_data.isel(sensor_idx=sensor["index"])
            sensor_label = sensor["label"]

            # Get values and calculate statistics
            sensor_light_values = sensor_light.values
            valid_light = sensor_light_values[~np.isnan(sensor_light_values)]

            if len(valid_light) > 0:
                coverage = len(valid_light) / len(sensor_light_values) * 100
                stats_text += f"{sensor_label} (ID: {sensor['sensor_id']}):\n"
                stats_text += f"  Coverage: {coverage:.1f}%\n"
                stats_text += f"  Mean: {np.mean(valid_light):.1f} lux\n"
                stats_text += f"  Range: {np.min(valid_light):.1f} to {np.max(valid_light):.1f} lux\n"
                stats_text += f"  Std: {np.std(valid_light):.1f} lux\n\n"

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

    if logger:
        logger.info(indent(f"QC plot saved: {output_file.name}", level=2))


def process_directory(input_dir, output_dir, csv_deployment_path, logger):
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
    logger : logging.Logger
        Logger instance
    """
    logger.info(subheader(f"Processing: {input_dir.name}"))

    # Load deployment periods CSV for QC plotting
    deployment_periods = None
    if csv_deployment_path.exists():
        try:
            deployment_periods = pd.read_csv(csv_deployment_path)
        except Exception as e:
            logger.warning(f"Could not load deployment periods CSV: {e}")

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

    logger.info(f"Found {len(site_data)} sites")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each site
    for site_name, sensors in tqdm(site_data.items(), desc="Merging sites"):
        logger.info(indent(f"Processing site: {site_name}"))

        # Merge sensors for this site (masks already applied during loading)
        combined_ds = merge_site_data(sensors)

        if combined_ds is None:
            logger.warning(indent(f"No valid data for site {site_name}", level=2))
            continue

        # Create output filename
        output_file = output_dir / f"{site_name}.nc"

        # Save combined dataset
        combined_ds.to_netcdf(output_file)
        logger.info(indent(f"Saved: {output_file.name}", level=2))

        # Create QC plots for this site
        create_qc_plots(combined_ds, site_name, output_dir, deployment_periods, logger)

        # Print summary
        n_sensors = (
            len(combined_ds.sensor_idx) if "sensor_idx" in combined_ds.dims else 0
        )
        datetime_coord = (
            "datetime" if "datetime" in combined_ds.dims else "datetime_utc"
        )
        data_points = (
            len(combined_ds[datetime_coord])
            if datetime_coord in combined_ds.dims
            else 0
        )
        sensor_heights = (
            [combined_ds.height.values[i] for i in range(n_sensors)]
            if "height" in combined_ds.coords
            else []
        )
        logger.info(
            indent(
                f"Combined {n_sensors} sensors ({', '.join(sensor_heights)}) -> {data_points} time points",
                level=2,
            )
        )


def main():
    """Main function to merge pendant data by site."""
    # Set up logging (appends to pipeline log if running as part of pipeline)
    logger = setup_pipeline_logging(step_number=3, total_steps=6, mode="a")

    # Define paths
    base_dir = Path(ROOT) / "data" / "2025" / "intermediate" / "pendants"
    input_base = base_dir / "by_sensor"
    output_base = base_dir / "by_site"
    csv_deployment_path = (
        Path(ROOT) / "data" / "2025" / "metadata" / "deployment_periods.csv"
    )

    logger.info(key_value("Input base directory", str(input_base)))
    logger.info(key_value("Output base directory", str(output_base)))
    logger.info(key_value("Deployment CSV", str(csv_deployment_path)))
    logger.info(key_value("CSV exists", str(csv_deployment_path.exists())))

    # Process main directory
    main_input = input_base
    main_output = output_base

    process_directory(main_input, main_output, csv_deployment_path, logger)

    # Process colocated site merges if configured
    if COLOCATED_SITES:
        # Load deployment periods CSV for QC plotting
        deployment_periods = None
        if csv_deployment_path.exists():
            try:
                deployment_periods = pd.read_csv(csv_deployment_path)
            except Exception as e:
                logger.warning(f"Could not load deployment periods CSV: {e}")

        for target_site, source_sites in COLOCATED_SITES.items():
            logger.info(
                subheader(
                    f"Merging colocated sites: {', '.join(source_sites)} -> {target_site}"
                )
            )

            # Load source site datasets
            site_datasets = {}
            for source_site in source_sites:
                source_path = main_output / f"{source_site}.nc"
                if source_path.exists():
                    site_datasets[source_site] = xr.open_dataset(source_path)
                    logger.info(
                        indent(f"Loaded {source_site} from {source_path.name}", level=1)
                    )
                else:
                    logger.warning(
                        indent(f"Source file not found: {source_path}", level=1)
                    )

            if len(site_datasets) >= 2:
                # Merge sites
                merged_ds = merge_sites(
                    site_datasets, target_site, join=COLOCATED_MERGE_JOIN
                )

                # Close source datasets before deleting
                for ds in site_datasets.values():
                    ds.close()

                # Save merged dataset
                output_path = main_output / f"{target_site}.nc"
                merged_ds.to_netcdf(output_path)
                logger.info(
                    indent(f"Saved merged dataset: {output_path.name}", level=1)
                )

                # Delete source files
                for source_site in source_sites:
                    source_path = main_output / f"{source_site}.nc"
                    if source_path.exists():
                        source_path.unlink()
                        logger.info(
                            indent(f"Removed source file: {source_path.name}", level=1)
                        )

                    # Also remove source QC plots
                    source_qc = main_output / "qc_plots" / f"{source_site}_qc.png"
                    if source_qc.exists():
                        source_qc.unlink()
                        logger.info(
                            indent(f"Removed source QC plot: {source_qc.name}", level=1)
                        )

                # Create QC plot for merged site
                create_qc_plots(
                    merged_ds, target_site, main_output, deployment_periods, logger
                )
            else:
                logger.warning(
                    f"Not enough source datasets found for {target_site} merge (need >= 2, got {len(site_datasets)})"
                )

    # Process subdirectories
    # TODO: Make this procedural
    subdirs = ["camp_wx", "on_ice_intensive", "off_ice", "on_ice"]

    for subdir in subdirs:
        subdir_input = input_base / subdir
        subdir_output = output_base / subdir

        if subdir_input.exists() and any(subdir_input.glob("*.nc")):
            process_directory(subdir_input, subdir_output, csv_deployment_path, logger)
        else:
            logger.info(f"Skipping {subdir} subdirectory (not found or empty)")


if __name__ == "__main__":
    main()
