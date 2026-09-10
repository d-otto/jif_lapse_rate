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

import argparse
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
import cmocean as cmo

from jiflr import ROOT
from jiflr.data import load_all_pendant_data
from jiflr.logging import indent, key_value, setup_pipeline_logging, subheader
from jiflr.pipeline import merge_sites
from jiflr.utils import apply_deployment_mask, get_deployment_periods


# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Configuration for merging colocated sites
# Maps target site_id -> list of source site_ids to merge
# Set to empty dict {} to disable colocated site merging
COLOCATED_SITES = {
    "G03": ["G03A", "G03B"],
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
    Creates structure: data_vars(sensor_idx, datetime_utc)

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
            "structure": "sensor_idx x datetime_utc",
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


def _sensor_colors(n_sensors):
    """Return evenly spaced categorical colors from a perceptual colormap."""
    return cmo.cm.haline(np.linspace(0.25, 0.75, n_sensors))


def _height_in_metres(height):
    """Return a numeric sensor height, or None when it cannot be parsed."""
    try:
        return float(str(height).lower().replace("m", ""))
    except ValueError:
        return None


def _masked_deployment_intervals(time_values, deployment_periods):
    """Return intervals outside the deployment periods within the data extent."""
    start, end = map(pd.Timestamp, (time_values[0], time_values[-1]))
    deployed = []
    for period_start, period_end in deployment_periods:
        period_start, period_end = max(start, period_start), min(end, period_end)
        if period_start < period_end:
            deployed.append((period_start, period_end))

    masked, cursor = [], start
    for period_start, period_end in sorted(deployed):
        if cursor < period_start:
            masked.append((cursor, period_start))
        cursor = max(cursor, period_end)
    if cursor < end:
        masked.append((cursor, end))
    return masked


def _deployment_time_mask(time_values, deployment_periods):
    """Return True for timestamps inside at least one deployment period."""
    if not deployment_periods:
        return np.ones(len(time_values), dtype=bool)

    timestamps = pd.to_datetime(time_values)
    mask = np.zeros(len(timestamps), dtype=bool)
    for start, end in deployment_periods:
        mask |= (timestamps >= start) & (timestamps <= end)
    return mask


def _add_qc_readout(
    deployment_ax,
    statistics_ax,
    site_name,
    time_values,
    sensor_info,
    temp_values,
    deployment_periods,
):
    """Add compact deployment and per-sensor temperature readouts."""
    heights = ", ".join(sorted({str(sensor["height"]) for sensor in sensor_info}))
    shielding = ", ".join(
        sorted({str(sensor["shielding"]) for sensor in sensor_info})
    )
    start, end = map(pd.Timestamp, (time_values[0], time_values[-1]))
    deployment_lines = [
        f"Site: {site_name}",
        f"Sensors: {len(sensor_info)}",
        f"Heights: {heights}",
        f"Shielding: {shielding}",
        f"Data: {start:%Y-%m-%d} to {end:%Y-%m-%d}",
        f"Time points: {len(time_values)}",
        "",
    ]
    if deployment_periods:
        deployment_lines.append("Deployment periods:")
        for index, (period_start, period_end) in enumerate(deployment_periods, start=1):
            duration = period_end - period_start
            deployment_lines.append(
                f"{index}. {period_start:%Y-%m-%d} to {period_end:%Y-%m-%d}"
            )
            deployment_lines.append(f"   {duration.days} days")
    else:
        deployment_lines.append("Deployment periods: unavailable")

    statistics_lines = []
    for sensor, values in zip(sensor_info, temp_values):
        valid = values[np.isfinite(values)]
        statistics_lines.append(f"{sensor['label']} (ID: {sensor['sensor_id']})")
        if valid.size == 0:
            statistics_lines.append("  No finite temperature observations")
        else:
            statistics_lines.append(
                f"  Coverage: {100 * valid.size / values.size:.1f}%  "
                f"Mean: {np.mean(valid):.2f} °C  SD: {np.std(valid):.2f} °C"
            )
            statistics_lines.append(
                f"  Range: {np.min(valid):.2f} to {np.max(valid):.2f} °C"
            )
        statistics_lines.append("")

    for ax, title, lines in (
        (deployment_ax, "Data and deployment", deployment_lines),
        (statistics_ax, "Temperature statistics", statistics_lines),
    ):
        ax.axis("off")
        ax.set_title(title, loc="left")
        ax.text(
            0,
            0.95,
            "\n".join(lines),
            transform=ax.transAxes,
            va="top",
            fontsize=7.5,
        )


def create_qc_plots(
    combined_ds,
    site_name,
    output_dir,
    deployment_periods=None,
    logger=None,
    csv_deployment_path=None,
    year=None,
):
    """
    Create quality control plots for a merged site dataset.

    Parameters
    ----------
    combined_ds : xarray.Dataset
        Combined dataset with dimensions (sensor_idx, datetime_utc)
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

    if "datetime_utc" not in combined_ds.coords:
        if logger:
            logger.warning(f"No datetime_utc coordinate found for site {site_name}")
        return
    if "temp_c" not in combined_ds.data_vars:
        raise ValueError(f"Site {site_name} has no temp_c data for QC plotting")

    time_values = combined_ds["datetime_utc"].values
    temp_data = combined_ds["temp_c"]

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

    # Deployment periods are used to shade data excluded by the deployment mask.
    site_deployment_periods = []
    if deployment_periods is not None:
        try:
            if csv_deployment_path is None or year is None:
                raise ValueError(
                    "csv_deployment_path and year are required for QC deployment shading"
                )
            site_deployment_periods = get_deployment_periods(
                site_name, csv_deployment_path, year
            ).get(site_name, [])
        except Exception as e:
            if logger:
                logger.warning(
                    f"Could not load deployment periods for {site_name}: {e}"
                )
            site_deployment_periods = []

    sensor_values = [
        temp_data.isel(sensor_idx=sensor["index"]).values
        for sensor in sensor_info
    ]
    if not any(np.isfinite(values).any() for values in sensor_values):
        raise ValueError(
            f"Site {site_name} has no finite temperature values for QC plotting"
        )

    colors = _sensor_colors(n_sensors)
    fig = plt.figure(
        figsize=(16, max(10.5, 8.5 + 0.35 * n_sensors)),
        dpi=200,
        layout="constrained",
    )
    fig.suptitle(f"Temperature QC — Site {site_name}", fontsize=16, fontweight="bold")
    grid = fig.add_gridspec(
        3,
        2,
        height_ratios=[2.5, 1.25, max(1.25, 0.25 * n_sensors)],
    )
    ax_series = fig.add_subplot(grid[0, :])
    ax_difference = fig.add_subplot(grid[1, :], sharex=ax_series)
    ax_histogram = fig.add_subplot(grid[2, 0])
    details_grid = grid[2, 1].subgridspec(1, 2, wspace=0.18)
    ax_deployment = fig.add_subplot(details_grid[0, 0])
    ax_statistics = fig.add_subplot(details_grid[0, 1])

    masked_intervals = (
        _masked_deployment_intervals(time_values, site_deployment_periods)
        if site_deployment_periods
        else []
    )
    for i, (start, end) in enumerate(masked_intervals):
        ax_series.axvspan(
            start,
            end,
            color="0.5",
            alpha=0.2,
            label="Masked outside deployment" if i == 0 else None,
            zorder=0,
        )

    for sensor, values, color in zip(sensor_info, sensor_values, colors):
        ax_series.plot(
            time_values, values, color=color, label=sensor["label"], linewidth=0.65
        )
    ax_series.axhline(0, color="0.15", linewidth=1.5, zorder=1)
    ax_series.set(
        title="Temperature time series", xlabel="Date (UTC)", ylabel="Temperature (°C)"
    )
    ax_series.grid(True, alpha=0.25)
    ax_series.legend(
        loc="upper left", ncol=min(4, n_sensors), fontsize=8, framealpha=0.9
    )
    ax_series.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_series.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax_series.xaxis.get_majorticklabels(), rotation=45, ha="right")

    deployment_mask = _deployment_time_mask(time_values, site_deployment_periods)
    valid_values = [
        values[deployment_mask & np.isfinite(values)] for values in sensor_values
    ]
    if not any(values.size for values in valid_values):
        raise ValueError(
            f"Site {site_name} has no finite temperatures during deployment periods"
        )
    all_values = np.concatenate([values for values in valid_values if values.size])
    bins = np.histogram_bin_edges(all_values, bins="auto")
    for values, color in zip(valid_values, colors):
        if values.size:
            ax_histogram.hist(
                values,
                bins=bins,
                density=True,
                histtype="step",
                color=color,
                linewidth=1.2,
            )
            ax_histogram.axvline(
                np.mean(values), color=color, linestyle="-", linewidth=1
            )
            ax_histogram.axvline(
                np.median(values), color=color, linestyle="--", linewidth=1
            )
    ax_histogram.axvline(0, color="0.15", linewidth=1.5, zorder=0)
    ax_histogram.set(
        title="Temperature distribution", xlabel="Temperature (°C)", ylabel="Density"
    )
    ax_histogram.grid(True, axis="y", alpha=0.25)
    ax_histogram.legend(
        handles=[
            Line2D([], [], color="0.2", linestyle="-", label="Mean"),
            Line2D([], [], color="0.2", linestyle="--", label="Median"),
        ],
        loc="upper left",
        fontsize=7,
        framealpha=0.9,
    )

    height_indices = {
        height: [
            sensor["index"]
            for sensor in sensor_info
            if _height_in_metres(sensor["height"]) == height
        ]
        for height in (1.0, 2.0)
    }
    if height_indices[1.0] and height_indices[2.0]:
        one_m_values = temp_data.isel(sensor_idx=height_indices[1.0]).values
        two_m_values = temp_data.isel(sensor_idx=height_indices[2.0]).values
        overlapping_times = np.isfinite(one_m_values).any(axis=0) & np.isfinite(
            two_m_values
        ).any(axis=0)
        if overlapping_times.any():
            one_m = np.nanmean(
                one_m_values[:, overlapping_times], axis=0
            )
            two_m = np.nanmean(
                two_m_values[:, overlapping_times], axis=0
            )
            ax_difference.plot(
                time_values[overlapping_times],
                two_m - one_m,
                color=cmo.cm.balance(0.2),
                linewidth=0.65,
            )
        else:
            ax_difference.text(
                0.5,
                0.5,
                "No overlapping 1 m and 2 m observations",
                ha="center",
                va="center",
                transform=ax_difference.transAxes,
            )
    else:
        ax_difference.text(
            0.5,
            0.5,
            "Requires both 1 m and 2 m sensors",
            ha="center",
            va="center",
            transform=ax_difference.transAxes,
        )
    ax_difference.axhline(0, color="0.15", linewidth=1.5, zorder=0)
    ax_difference.set(
        title="2 m − 1 m temperature difference",
        xlabel="Date (UTC)",
        ylabel="Difference (°C)",
    )
    ax_difference.grid(True, alpha=0.25)
    ax_difference.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_difference.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax_difference.xaxis.get_majorticklabels(), rotation=45, ha="right")

    _add_qc_readout(
        ax_deployment,
        ax_statistics,
        site_name,
        time_values,
        sensor_info,
        [values[deployment_mask] for values in sensor_values],
        site_deployment_periods,
    )

    # Save the plot
    output_file = qc_plots_dir / f"{site_name}_qc.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if logger:
        logger.info(indent(f"QC plot saved: {output_file.name}", level=2))


def process_directory(input_dir, output_dir, csv_deployment_path, year, logger):
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

    # Generate QC plots immediately before deployment masking is applied.
    site_data = load_all_pendant_data(
        processed_dir=input_dir,
        csv_deployment_path=csv_deployment_path,
        year=year,
        use_csv_masking=True,
        required_heights=None,  # Load all available heights
        drop_events=True,
        drop_light=False,  # Preserve light data
        apply_mask_to_data=False,
    )

    logger.info(f"Found {len(site_data)} sites")

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each site
    for site_name, sensors in tqdm(site_data.items(), desc="Merging sites"):
        logger.info(indent(f"Processing site: {site_name}"))

        pre_mask_combined_ds = merge_site_data(sensors)

        if pre_mask_combined_ds is None:
            logger.warning(indent(f"No valid data for site {site_name}", level=2))
            continue

        # Show observations that the following deployment-mask operation removes.
        create_qc_plots(
            pre_mask_combined_ds,
            site_name,
            output_dir,
            deployment_periods,
            logger,
            csv_deployment_path=csv_deployment_path,
            year=year,
        )

        try:
            combined_ds = apply_deployment_mask(
                pre_mask_combined_ds,
                site_name,
                csv_deployment_path,
                year,
                ignore_missing=True,
            )
        except Exception as error:
            logger.warning(
                indent(
                    f"Could not apply deployment mask for {site_name}: {error}. "
                    "Saving unmasked data.",
                    level=2,
                )
            )
            combined_ds = pre_mask_combined_ds

        # Create output filename
        output_file = output_dir / f"{site_name}.nc"

        # Save combined dataset
        combined_ds.to_netcdf(output_file)
        logger.info(indent(f"Saved: {output_file.name}", level=2))

        # Print summary
        n_sensors = (
            len(combined_ds.sensor_idx) if "sensor_idx" in combined_ds.dims else 0
        )
        datetime_coord = "datetime_utc"
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
    parser = argparse.ArgumentParser(description="Merge pendant data by site for one field season")
    parser.add_argument("--year", required=True, type=int, help="Field season to process")
    args = parser.parse_args()
    year = args.year
    # Set up logging (appends to pipeline log if running as part of pipeline)
    logger = setup_pipeline_logging(step_number=3, total_steps=7, mode="a")

    # Define paths
    base_dir = Path(ROOT) / "data" / str(year) / "intermediate" / "pendants"
    input_base = base_dir / "by_sensor"
    output_base = base_dir / "by_site"
    csv_deployment_path = (
        Path(ROOT) / "data" / str(year) / "metadata" / "deployment_periods.csv"
    )

    logger.info(key_value("Input base directory", str(input_base)))
    logger.info(key_value("Output base directory", str(output_base)))
    logger.info(key_value("Deployment CSV", str(csv_deployment_path)))
    logger.info(key_value("CSV exists", str(csv_deployment_path.exists())))

    # Process main directory
    main_input = input_base
    main_output = output_base

    process_directory(main_input, main_output, csv_deployment_path, year, logger)

    # Process subdirectories
    # TODO: Make this procedural
    subdirs = ["camp_wx", "on_ice_intensive", "off_ice", "on_ice"]

    for subdir in subdirs:
        subdir_input = input_base / subdir
        subdir_output = output_base / subdir

        if subdir_input.exists() and any(subdir_input.glob("*.nc")):
            process_directory(subdir_input, subdir_output, csv_deployment_path, year, logger)
        else:
            logger.info(f"Skipping {subdir} subdirectory (not found or empty)")

    # Process colocated site merges if configured.
    # Runs after all subdirectories are processed so source files are present.
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

            # Search output_base and all subdirs for source files
            site_datasets = {}
            site_paths = {}
            for source_site in source_sites:
                for candidate in output_base.rglob(f"{source_site}.nc"):
                    site_datasets[source_site] = xr.open_dataset(candidate)
                    site_paths[source_site] = candidate
                    logger.info(
                        indent(f"Loaded {source_site} from {candidate.relative_to(output_base)}", level=1)
                    )
                    break
                else:
                    logger.warning(
                        indent(f"Source file not found: {source_site}.nc", level=1)
                    )

            if len(site_datasets) >= 2:
                # All source files must be in the same directory; use that as output dir
                source_dirs = {p.parent for p in site_paths.values()}
                if len(source_dirs) > 1:
                    logger.warning(
                        f"Source files for {target_site} are in different directories: {source_dirs}. Using first."
                    )
                output_dir = next(iter(source_dirs))

                # Merge sites
                merged_ds = merge_sites(
                    site_datasets, target_site, join=COLOCATED_MERGE_JOIN
                )

                # Close source datasets before deleting
                for ds in site_datasets.values():
                    ds.close()

                # Save merged dataset
                output_path = output_dir / f"{target_site}.nc"
                merged_ds.to_netcdf(output_path)
                logger.info(
                    indent(f"Saved merged dataset: {output_path.name}", level=1)
                )

                # Delete source files
                for source_site, source_path in site_paths.items():
                    if source_path.exists():
                        source_path.unlink()
                        logger.info(
                            indent(f"Removed source file: {source_path.name}", level=1)
                        )

                    # Also remove source QC plots
                    source_qc = source_path.parent / "qc_plots" / f"{source_site}_qc.png"
                    if source_qc.exists():
                        source_qc.unlink()
                        logger.info(
                            indent(f"Removed source QC plot: {source_qc.name}", level=1)
                        )

                # Create QC plot for merged site
                create_qc_plots(
                    merged_ds, target_site, output_dir, deployment_periods, logger,
                    csv_deployment_path=csv_deployment_path, year=year,
                )
            else:
                logger.warning(
                    f"Not enough source datasets found for {target_site} merge (need >= 2, got {len(site_datasets)})"
                )


if __name__ == "__main__":
    main()
