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
from jiflr.deployment_manifest import (
    DeploymentManifest,
    apply_manifest_deployment_mask,
    load_deployment_manifest,
)
from jiflr.logging import indent, key_value, setup_pipeline_logging, subheader
from jiflr.pipeline import merge_sites
from jiflr.qc_plots import to_anchorage_time


# =============================================================================
# USER CONFIGURATION
# =============================================================================
# Hard-coded exception: average the two matching 1 m G03 pendants in 2025.
# Other sensors, including both 2 m B01 generations, remain separate.
# Keys are (year, canonical site_id); values are the exact logger serials.
SITE_SENSOR_AVERAGES = {
    (2025, "G03"): ("10568620", "10383462"),
}

# Old site labels are accepted only as generated output names to remove after
# the canonical site file is saved. By-sensor inputs must use the new manifest.
LEGACY_SITE_ALIASES = {
    (2025, "G03"): ("G03a", "G03b"),
    (2026, "B01"): ("B01a", "B01b"),
}
# =============================================================================


def merge_site_data(site_data_dict):
    """
    Merge multiple sensor datasets for a site into a single dataset using sensor_idx concatenation.

    Uses sensor_idx structure with sensor attributes as coordinates.
    Creates structure: data_vars(sensor_idx, datetime_utc)

    Parameters
    ----------
    site_data_dict : dict
        Dictionary with logger serials as keys and sensor info as values
        (output from load_all_pendant_data for a single site)
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


def merge_configured_site_sensors(
    site_name: str,
    site_data_dict: dict,
    manifest: DeploymentManifest,
    year: int,
) -> xr.Dataset:
    """Apply an explicit same-height average after masking each logger."""
    serials_to_average = SITE_SENSOR_AVERAGES[(year, site_name)]
    if len(serials_to_average) < 2 or len(set(serials_to_average)) != len(serials_to_average):
        raise ValueError(f"Invalid sensor average configured for {year} {site_name}")

    actual_serials = set(site_data_dict)
    missing_serials = set(serials_to_average) - actual_serials
    if missing_serials:
        raise ValueError(
            f"Cannot average {year} {site_name}: missing logger serials "
            f"{sorted(missing_serials)}"
        )

    heights_by_serial = {}
    for serial, sensor_info in site_data_dict.items():
        ds = sensor_info["dataset"]
        if ds.sizes.get("sensor_idx") != 1:
            raise ValueError(f"Expected one sensor for logger {serial} at {year} {site_name}")
        if str(ds["sensor_id"].item()) != serial or str(ds["site_id"].item()) != site_name:
            raise ValueError(f"Sensor metadata does not match {year} {site_name} logger {serial}")
        heights_by_serial[serial] = str(ds["height"].item())

    averaged_heights = {heights_by_serial[serial] for serial in serials_to_average}
    if len(averaged_heights) != 1:
        raise ValueError(f"Cannot average {year} {site_name}: logger heights differ")
    average_height = averaged_heights.pop()
    serials_at_height = {
        serial for serial, height in heights_by_serial.items() if height == average_height
    }
    if serials_at_height != set(serials_to_average):
        raise ValueError(
            f"Cannot average {year} {site_name} at {average_height}: "
            f"found logger serials {sorted(serials_at_height)}"
        )

    for coordinate in ("sensor_type", "sensor_generation", "shielding"):
        values = {
            str(site_data_dict[serial]["dataset"][coordinate].item())
            for serial in serials_to_average
        }
        if len(values) != 1:
            raise ValueError(
                f"Cannot average {year} {site_name}: {coordinate} differs across loggers"
            )

    for height in set(heights_by_serial.values()) - {average_height}:
        serials = [serial for serial, value in heights_by_serial.items() if value == height]
        if len(serials) > 1:
            raise ValueError(
                f"No average configured for {year} {site_name} at {height}: {serials}"
            )

    ordered_serials = [
        *serials_to_average,
        *sorted(actual_serials - set(serials_to_average)),
    ]
    masked_datasets = {
        serial: apply_manifest_deployment_mask(
            site_data_dict[serial]["dataset"], manifest
        )
        for serial in ordered_serials
    }
    return merge_sites(masked_datasets, site_name, join="inner")


def remove_legacy_site_outputs(
    output_dir: Path, year: int, site_name: str, logger
) -> None:
    """Remove obsolete by-site files only after saving their replacement."""
    aliases = {
        alias.casefold()
        for alias in LEGACY_SITE_ALIASES.get((year, site_name), ())
    }
    if not aliases:
        return

    for directory, suffix in ((output_dir, ".nc"), (output_dir / "qc_plots", "_qc.png")):
        if not directory.exists():
            continue
        for path in directory.iterdir():
            name = path.name.removesuffix(suffix)
            if path.name.endswith(suffix) and name.casefold() in aliases:
                path.unlink()
                logger.info(indent(f"Removed obsolete site output: {path.name}", level=2))


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


def _histogram_bins(values, bin_width):
    """Return fixed-width histogram bins spanning all finite values."""
    finite_values = np.asarray(values)[np.isfinite(values)]
    if finite_values.size == 0:
        raise ValueError("Cannot create histogram bins without finite values")

    lower = np.floor(finite_values.min() / bin_width) * bin_width
    upper = np.ceil(finite_values.max() / bin_width) * bin_width
    if lower == upper:
        upper += bin_width
    return np.arange(lower, upper + bin_width * 0.5, bin_width)


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
    histogram_ds=None,
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
    deployment_periods : dict, optional
        Optional precomputed UTC intervals keyed by site ID for shading.
    logger : logging.Logger, optional
        Logger instance for output
    histogram_ds : xarray.Dataset, optional
        Dataset with non-deployment observations masked. When provided, its
        temperature data are used for the distribution plots while the time
        series continues to show all source observations.
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

    utc_time_values = combined_ds["datetime_utc"].values
    time_values = to_anchorage_time(utc_time_values)
    temp_data = combined_ds["temp_c"]
    histogram_temp_data = (
        histogram_ds["temp_c"] if histogram_ds is not None else temp_data
    )
    if histogram_temp_data.shape != temp_data.shape:
        raise ValueError(
            "Histogram dataset temp_c must have the same shape as the QC dataset"
        )

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
                "label": f"{height} ({shielding}; {sensor_id})",
            }
        )

    # Optional deployment periods are used only for QC shading. Processing
    # itself uses the per-instrument manifest below.
    site_deployment_periods = []
    if deployment_periods is not None:
        site_deployment_periods = deployment_periods.get(site_name, [])

    sensor_values = [
        temp_data.isel(sensor_idx=sensor["index"]).values
        for sensor in sensor_info
    ]
    if not any(np.isfinite(values).any() for values in sensor_values):
        raise ValueError(
            f"Site {site_name} has no finite temperature values for QC plotting"
        )
    plotted_time_mask = np.any(np.isfinite(np.asarray(sensor_values)), axis=0)
    plotted_time_values = time_values[plotted_time_mask]
    time_limits = (plotted_time_values[0], plotted_time_values[-1])

    colors = _sensor_colors(n_sensors)
    fig = plt.figure(
        figsize=(16, max(12, 10 + 0.35 * n_sensors)),
        dpi=200,
        layout="constrained",
    )
    fig.suptitle(f"Temperature QC — Site {site_name}", fontsize=16, fontweight="bold")
    grid = fig.add_gridspec(
        4,
        2,
        height_ratios=[2.5, 1.25, 1.25, max(1.25, 0.25 * n_sensors)],
    )
    ax_series = fig.add_subplot(grid[0, :])
    ax_difference = fig.add_subplot(grid[1, :], sharex=ax_series)
    ax_histogram = fig.add_subplot(grid[2, 0])
    ax_difference_histogram = fig.add_subplot(grid[2, 1])
    details_grid = grid[3, :].subgridspec(1, 2, wspace=0.18)
    ax_deployment = fig.add_subplot(details_grid[0, 0])
    ax_statistics = fig.add_subplot(details_grid[0, 1])

    masked_intervals = (
        _masked_deployment_intervals(utc_time_values, site_deployment_periods)
        if site_deployment_periods
        else []
    )
    for i, (start, end) in enumerate(masked_intervals):
        ax_series.axvspan(
            to_anchorage_time([start])[0],
            to_anchorage_time([end])[0],
            color="0.5",
            alpha=0.2,
            label="Masked outside deployment" if i == 0 else None,
            zorder=0,
        )

    for sensor, color in zip(sensor_info, colors):
        sensor_data = temp_data.isel(sensor_idx=sensor["index"]).dropna(
            dim="datetime_utc"
        )
        ax_series.plot(
            to_anchorage_time(sensor_data["datetime_utc"].values),
            sensor_data.values,
            color=color,
            label=sensor["label"],
            linewidth=0.65,
        )
    ax_series.axhline(0, color="0.15", linewidth=1.5, zorder=1)
    ax_series.set(
        title="Temperature time series",
        xlabel="Date (Alaska time)",
        ylabel="Temperature (°C)",
    )
    ax_series.grid(True, alpha=0.25)
    ax_series.legend(
        loc="upper left", ncol=min(4, n_sensors), fontsize=8, framealpha=0.9
    )
    ax_series.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_series.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax_series.xaxis.get_majorticklabels(), rotation=45, ha="right")

    deployment_mask = _deployment_time_mask(utc_time_values, site_deployment_periods)
    histogram_sensor_values = [
        histogram_temp_data.isel(sensor_idx=sensor["index"]).values
        for sensor in sensor_info
    ]
    valid_values = [
        values[deployment_mask & np.isfinite(values)]
        for values in histogram_sensor_values
    ]
    if not any(values.size for values in valid_values):
        raise ValueError(
            f"Site {site_name} has no finite temperatures during deployment periods"
    )
    all_values = np.concatenate([values for values in valid_values if values.size])
    bins = _histogram_bins(all_values, bin_width=0.25)
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
    one_to_one_height_comparison = all(
        len(height_indices[height]) == 1 for height in (1.0, 2.0)
    )
    if one_to_one_height_comparison:
        one_m = temp_data.isel(sensor_idx=height_indices[1.0][0]).values
        two_m = temp_data.isel(sensor_idx=height_indices[2.0][0]).values
        difference_values = two_m - one_m
        overlapping_times = np.isfinite(difference_values)
        if overlapping_times.any():
            ax_difference.plot(
                time_values[overlapping_times],
                difference_values[overlapping_times],
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
            "Requires one sensor at 1 m and one at 2 m",
            ha="center",
            va="center",
            transform=ax_difference.transAxes,
        )
    ax_difference.axhline(0, color="0.15", linewidth=1.5, zorder=0)
    ax_difference.set(
        title="2 m − 1 m temperature difference",
        xlabel="Date (Alaska time)",
        ylabel="Difference (°C)",
    )
    ax_difference.grid(True, alpha=0.25)
    ax_difference.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    ax_difference.xaxis.set_major_locator(mdates.DayLocator(interval=2))
    plt.setp(ax_difference.xaxis.get_majorticklabels(), rotation=45, ha="right")
    ax_series.set_xlim(time_limits)

    if one_to_one_height_comparison:
        deployment_one_m = histogram_temp_data.isel(
            sensor_idx=height_indices[1.0][0]
        ).values
        deployment_two_m = histogram_temp_data.isel(
            sensor_idx=height_indices[2.0][0]
        ).values
        deployment_difference = deployment_two_m - deployment_one_m
        deployment_difference = deployment_difference[
            deployment_mask & np.isfinite(deployment_difference)
        ]
        if deployment_difference.size:
            ax_difference_histogram.hist(
                deployment_difference,
                bins=_histogram_bins(deployment_difference, bin_width=0.1),
                density=True,
                color=cmo.cm.balance(0.2),
                alpha=0.8,
            )
            ax_difference_histogram.axvline(
                np.mean(deployment_difference),
                color="0.2",
                linestyle="-",
                linewidth=1,
                label="Mean",
            )
            ax_difference_histogram.axvline(
                np.median(deployment_difference),
                color="0.2",
                linestyle="--",
                linewidth=1,
                label="Median",
            )
            ax_difference_histogram.legend(
                loc="upper left", fontsize=7, framealpha=0.9
            )
        else:
            ax_difference_histogram.text(
                0.5,
                0.5,
                "No overlapping 1 m and 2 m deployment observations",
                ha="center",
                va="center",
                transform=ax_difference_histogram.transAxes,
            )
    else:
        ax_difference_histogram.text(
            0.5,
            0.5,
            "Requires one sensor at 1 m and one at 2 m",
            ha="center",
            va="center",
            transform=ax_difference_histogram.transAxes,
        )
    ax_difference_histogram.axvline(0, color="0.15", linewidth=1.5, zorder=0)
    ax_difference_histogram.set(
        title="2 m − 1 m difference distribution",
        xlabel="Difference (°C)",
        ylabel="Density",
    )
    ax_difference_histogram.grid(True, axis="y", alpha=0.25)

    _add_qc_readout(
        ax_deployment,
        ax_statistics,
        site_name,
        time_values,
        sensor_info,
        [values[deployment_mask] for values in histogram_sensor_values],
        site_deployment_periods,
    )

    # Save the plot
    output_file = qc_plots_dir / f"{site_name}_qc.png"
    fig.savefig(output_file, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if logger:
        logger.info(indent(f"QC plot saved: {output_file.name}", level=2))


def process_directory(
    input_dir: Path,
    output_dir: Path,
    manifest: DeploymentManifest,
    year: int,
    logger,
):
    """
    Process all NetCDF files in a directory and merge by site.

    Parameters
    ----------
    input_dir : Path
        Input directory containing individual sensor NetCDF files
    output_dir : Path
        Output directory for combined site files
    manifest : DeploymentManifest
        Per-season deployment metadata used to mask each individual sensor.
    logger : logging.Logger
        Logger instance
    """
    logger.info(subheader(f"Processing: {input_dir.name}"))

    # Generate QC plots immediately before deployment masking is applied.
    site_data = load_all_pendant_data(
        processed_dir=input_dir,
        year=year,
        manifest=manifest,
        use_csv_masking=True,
        required_heights=None,  # Load all available heights
        drop_events=True,
        drop_light=False,  # Preserve light data
        apply_mask_to_data=False,
    )

    logger.info(f"Found {len(site_data)} sites")

    legacy_input_names = {
        alias.casefold()
        for (rule_year, _), aliases in LEGACY_SITE_ALIASES.items()
        if rule_year == year
        for alias in aliases
    }
    stale_sites = [name for name in site_data if name.casefold() in legacy_input_names]
    if stale_sites:
        raise ValueError(
            f"By-sensor data still uses obsolete site IDs {sorted(stale_sites)}; "
            "regenerate pendant by-sensor data from the updated deployment manifest"
        )

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process each site
    for site_name, sensors in tqdm(site_data.items(), desc="Merging sites"):
        logger.info(indent(f"Processing site: {site_name}"))

        pre_mask_combined_ds = merge_site_data(sensors)

        if pre_mask_combined_ds is None:
            logger.warning(indent(f"No valid data for site {site_name}", level=2))
            continue

        deployment_combined_ds = apply_manifest_deployment_mask(
            pre_mask_combined_ds, manifest
        )

        # Show observations that the deployment-mask operation removes while
        # calculating distribution plots from deployment observations only.
        create_qc_plots(
            pre_mask_combined_ds,
            site_name,
            output_dir,
            None,
            logger,
            histogram_ds=deployment_combined_ds,
        )

        if (year, site_name) in SITE_SENSOR_AVERAGES:
            combined_ds = merge_configured_site_sensors(
                site_name, sensors, manifest, year
            )
            create_qc_plots(combined_ds, site_name, output_dir, logger=logger)
        else:
            combined_ds = deployment_combined_ds

        # Create output filename
        output_file = output_dir / f"{site_name}.nc"

        # Save combined dataset
        combined_ds.to_netcdf(output_file)
        logger.info(indent(f"Saved: {output_file.name}", level=2))
        remove_legacy_site_outputs(output_dir, year, site_name, logger)

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
    logger = setup_pipeline_logging(step_number=4, total_steps=8, mode="a")

    # Define paths
    base_dir = Path(ROOT) / "data" / str(year) / "intermediate" / "pendants"
    input_base = base_dir / "by_sensor"
    output_base = base_dir / "by_site"
    manifest_path = Path(ROOT) / "data" / str(year) / "metadata" / "deployment_manifest.csv"
    manifest = load_deployment_manifest(manifest_path)

    logger.info(key_value("Input base directory", str(input_base)))
    logger.info(key_value("Output base directory", str(output_base)))
    logger.info(key_value("Deployment manifest", str(manifest_path)))

    # Process main directory
    main_input = input_base
    main_output = output_base

    process_directory(main_input, main_output, manifest, year, logger)

    # Process subdirectories
    # TODO: Make this procedural
    subdirs = ["camp_wx", "on_ice_intensive", "off_ice", "on_ice"]

    for subdir in subdirs:
        subdir_input = input_base / subdir
        subdir_output = output_base / subdir

        if subdir_input.exists() and any(subdir_input.glob("*.nc")):
            process_directory(subdir_input, subdir_output, manifest, year, logger)
        else:
            logger.info(f"Skipping {subdir} subdirectory (not found or empty)")


if __name__ == "__main__":
    main()
