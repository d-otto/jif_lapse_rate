#!/usr/bin/env python3
"""
qc_plots.py

QC (Quality Control) plotting utilities for the JIFLR data pipeline.
Creates multi-row time series plots for data variables with one row per sensor.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np


def _get_sorted_sensor_indices(ds):
    """
    Get sensor indices sorted by site_id and elevation (descending).

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with sensor_idx dimension and metadata coordinates

    Returns
    -------
    list
        Sorted list of sensor indices
    """
    sensor_info = []
    for i in range(len(ds.sensor_idx)):
        site_id = str(ds.site_id.values[i]) if "site_id" in ds.coords else ""
        elevation = float(ds.elevation.values[i]) if "elevation" in ds.coords else 0.0
        height = str(ds.height.values[i]) if "height" in ds.coords else ""
        sensor_info.append((i, site_id, elevation, height))

    # Sort by site_id (ascending), elevation (descending), height (descending)
    sensor_info.sort(key=lambda x: (x[1], -x[2], x[3]), reverse=False)
    return [info[0] for info in sensor_info]


def _get_row_label(ds, sensor_idx):
    """
    Get the row label for a sensor.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with sensor metadata coordinates
    sensor_idx : int
        Index of the sensor

    Returns
    -------
    str
        Label like "A01 2m (S)" or "B02 1m (U)"
    """
    site_id = str(ds.site_id.values[sensor_idx]) if "site_id" in ds.coords else "?"
    height = str(ds.height.values[sensor_idx]) if "height" in ds.coords else ""
    shielding = str(ds.shielding.values[sensor_idx]) if "shielding" in ds.coords else ""

    # Abbreviate shielding
    if shielding.lower().startswith("shield"):
        shield_abbr = "(S)"
    elif shielding.lower().startswith("unshield"):
        shield_abbr = "(U)"
    else:
        shield_abbr = ""

    parts = [site_id]
    if height:
        parts.append(height)
    if shield_abbr:
        parts.append(shield_abbr)

    return " ".join(parts)


def create_sensor_qc_plot(ds, var_name, output_path, title=None, logger=None):
    """
    Create a QC plot for a single variable with one row per sensor.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with sensor_idx dimension
    var_name : str
        Name of the data variable to plot
    output_path : Path
        Output file path for the plot
    title : str, optional
        Title for the plot
    logger : logging.Logger, optional
        Logger instance for status messages
    """
    if var_name not in ds.data_vars:
        if logger:
            logger.warning(f"Variable {var_name} not found in dataset, skipping")
        return

    # Get sorted sensor indices
    sorted_indices = _get_sorted_sensor_indices(ds)

    # Filter out sensors with all-NaN data for this variable
    valid_indices = []
    for sensor_idx in sorted_indices:
        data = ds[var_name].isel(sensor_idx=sensor_idx).values
        if not np.all(np.isnan(data)):
            valid_indices.append(sensor_idx)

    n_sensors = len(valid_indices)

    if n_sensors == 0:
        if logger:
            logger.warning(f"No valid data for {var_name}, skipping")
        return

    # Calculate figure size
    row_height = 0.6  # inches per row
    fig_height = max(
        min(n_sensors * row_height, 50), 16
    )  # cap at 50 inches, but set the minimum at 16
    fig_width = 16

    # Create figure with subplots
    fig, axes = plt.subplots(
        n_sensors,
        1,
        figsize=(fig_width, fig_height),
        sharex=True,
        sharey=True,
        squeeze=False,
        gridspec_kw={"hspace": 0},
    )
    axes = axes.flatten()

    # Get datetime coordinate
    datetime_coord = "datetime" if "datetime" in ds.coords else "datetime_utc"
    time_values = ds[datetime_coord].values
    step = 1

    # Plot each sensor (only those with valid data)
    for ax_idx, sensor_idx in enumerate(valid_indices):
        ax = axes[ax_idx]

        # Get data for this sensor (handles both dimension orders)
        data = ds[var_name].isel(sensor_idx=sensor_idx).values

        # Plot all points
        ax.plot(time_values[::step], data[::step], linewidth=0.5, alpha=0.8)

        # Set row label
        label = _get_row_label(ds, sensor_idx)
        ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center")
        ax.yaxis.set_label_coords(-0.02, 0.5)

        # Style axes
        ax.tick_params(axis="y", labelsize=6)
        ax.grid(True, which="both", alpha=0.3, linewidth=0.5)

        # Hide x-axis labels except for bottom plot
        if ax_idx < n_sensors - 1:
            ax.tick_params(axis="x", labelbottom=False)

        # Hide bottom and top spine
        ax.spines["bottom"].set_visible(True)
        ax.spines["top"].set_visible(False)

    # Format x-axis on bottom plot
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
    axes[-1].xaxis.set_major_locator(mdates.DayLocator(interval=7))
    axes[-1].xaxis.set_minor_locator(mdates.DayLocator(interval=1))
    axes[-1].tick_params(axis="x", labelsize=8, rotation=45)
    axes[-1].set_xlabel("Date", fontsize=10)
    axes[-1].spines["bottom"].set_visible(True)

    # restore top spine on first subplot
    axes[0].spines["top"].set_visible(True)

    # Set title
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    else:
        fig.suptitle(f"QC Plot: {var_name}", fontsize=12, fontweight="bold")

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(hspace=0)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if logger:
        logger.info(f"    Saved QC plot: {output_path.name}")


def create_all_qc_plots(ds, output_dir, filename_prefix, logger=None):
    """
    Create QC plots for all data variables in a dataset.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset with sensor_idx dimension and data variables
    output_dir : Path
        Output directory (qc_plots subfolder will be created)
    filename_prefix : str
        Prefix for output filenames (e.g., 'lvl0_on_ice')
    logger : logging.Logger, optional
        Logger instance for status messages
    """
    output_dir = Path(output_dir)
    qc_dir = output_dir / "qc_plots"
    qc_dir.mkdir(parents=True, exist_ok=True)

    if logger:
        logger.info(f"  Creating QC plots in {qc_dir}")

    # Create a plot for each data variable
    for var_name in ds.data_vars:
        output_path = qc_dir / f"{filename_prefix}_{var_name}_qc.png"
        title = f"{filename_prefix}: {var_name}"
        create_sensor_qc_plot(ds, var_name, output_path, title=title, logger=logger)
