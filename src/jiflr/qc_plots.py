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


def _get_row_label(ds, sensor_idx, include_shielding=False):
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

    shield_abbr = ""
    if include_shielding:
        if shielding.lower().startswith("shield"):
            shield_abbr = "(S)"
        elif shielding.lower().startswith("unshield"):
            shield_abbr = "(U)"

    parts = [site_id]
    if height:
        parts.append(height)
    if shield_abbr:
        parts.append(shield_abbr)

    return " ".join(parts)


def create_sensor_qc_plot(ds, var_name, output_path, title=None, logger=None):
    """
    Create a QC plot for a single numeric variable with one row per sensor.

    Each row contains a time series, histogram, and statistical summary in
    columns with a 3:1:1 width ratio.

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

    if "datetime_utc" not in ds.coords:
        raise ValueError("QC plotting requires a datetime_utc coordinate")
    if "sensor_idx" not in ds[var_name].dims:
        raise ValueError(f"QC variable {var_name} must be indexed by sensor_idx")
    if not np.issubdtype(ds[var_name].dtype, np.number):
        raise ValueError(f"QC variable {var_name} must have a numeric dtype")

    # Get sorted sensor indices
    sorted_indices = _get_sorted_sensor_indices(ds)

    # Filter out sensors with all-NaN data for this variable
    valid_indices = []
    for sensor_idx in sorted_indices:
        data = ds[var_name].isel(sensor_idx=sensor_idx).values
        if np.any(np.isfinite(data)):
            valid_indices.append(sensor_idx)

    n_sensors = len(valid_indices)

    if n_sensors == 0:
        if logger:
            logger.warning(f"No valid data for {var_name}, skipping")
        return

    # Create one time-series, histogram, and statistics row per sensor.
    fig, axes = plt.subplots(
        n_sensors,
        3,
        figsize=(20, max(min(1.4 * n_sensors, 36), 4.0)),
        layout="constrained",
        squeeze=False,
        gridspec_kw={"width_ratios": [3, 1, 1]},
    )
    fig.set_constrained_layout_pads(h_pad=0.04, hspace=0.02, w_pad=0.04, wspace=0.05)
    time_values = ds["datetime_utc"].values
    units = ds[var_name].attrs.get("units", "")
    value_label = f"{var_name} ({units})" if units else var_name

    # Plot each sensor with its own distribution and summary.
    for ax_idx, sensor_idx in enumerate(valid_indices):
        series_ax, histogram_ax, statistics_ax = axes[ax_idx]
        if ax_idx > 0:
            series_ax.sharex(axes[0, 0])
        data = ds[var_name].isel(sensor_idx=sensor_idx).values
        valid = data[np.isfinite(data)]

        label = _get_row_label(
            ds, sensor_idx, include_shielding=var_name == "temp_c"
        )
        series_ax.plot(time_values, data, color="tab:blue", linewidth=0.65)
        series_ax.set_title(label, loc="left", fontsize=10, fontweight="bold")
        series_ax.set_ylabel(value_label, fontsize=8)
        series_ax.grid(True, alpha=0.25)
        series_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        series_ax.tick_params(axis="x", labelsize=7, rotation=0)
        series_ax.tick_params(axis="y", labelsize=7)
        if ax_idx == n_sensors - 1:
            series_ax.set_xlabel("Date (UTC)", fontsize=8)
        else:
            series_ax.tick_params(axis="x", labelbottom=False)

        histogram_ax.hist(valid, bins="auto", color="tab:blue", alpha=0.8)
        histogram_ax.axvline(np.mean(valid), color="tab:red", linewidth=1, label="Mean")
        histogram_ax.axvline(
            np.median(valid), color="tab:orange", linestyle="--", linewidth=1, label="Median"
        )
        histogram_ax.set_title("Distribution", fontsize=9)
        histogram_ax.set_xlabel(units, fontsize=8)
        histogram_ax.tick_params(axis="both", labelsize=7)
        histogram_ax.grid(True, axis="y", alpha=0.25)
        histogram_ax.legend(fontsize=6, frameon=False)

        statistics_ax.axis("off")
        statistics_ax.set_title("Statistics", loc="left", fontsize=9)
        statistics_ax.text(
            0,
            0.95,
            "\n".join(
                (
                    f"Mean: {np.mean(valid):.3g}",
                    f"Median: {np.median(valid):.3g}",
                    f"Std dev: {np.std(valid):.3g}",
                    f"Range: {np.min(valid):.3g} to {np.max(valid):.3g}",
                )
            ),
            transform=statistics_ax.transAxes,
            va="top",
            fontsize=8,
        )

    # Set title
    if title:
        fig.suptitle(title, fontsize=12, fontweight="bold")
    else:
        fig.suptitle(f"QC Plot: {var_name}", fontsize=12, fontweight="bold")

    
    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if logger:
        logger.info(f"    Saved QC plot: {output_path.name}")


def _pace_channel_label(ds, var_name, sensor_idx):
    """Return a concise label for one populated Pace data channel."""
    sensor_id = (
        str(ds["sensor_id"].sel(sensor_idx=sensor_idx).item())
        if "sensor_id" in ds.coords
        else f"sensor {sensor_idx}"
    )
    height = (
        str(ds["height"].sel(sensor_idx=sensor_idx).item())
        if "height" in ds.coords
        else ""
    )
    return " — ".join(part for part in (var_name, height, sensor_id) if part)


def create_pace_qc_plot(ds, output_dir, filename_prefix, logger=None):
    """Create a channel-by-channel QC plot for one Pace logger dataset.

    Each populated variable and sensor combination is represented by one row.
    Rows contain a time series, histogram, and statistical summary in columns
    with a 3:1:1 width ratio.
    """
    if "datetime_utc" not in ds.coords:
        raise ValueError("Pace QC plotting requires a datetime_utc coordinate")
    if "sensor_idx" not in ds.dims:
        raise ValueError("Pace QC plotting requires a sensor_idx dimension")

    channels = []
    for var_name, data_array in ds.data_vars.items():
        if "sensor_idx" not in data_array.dims:
            continue
        for sensor_idx in ds.sensor_idx.values:
            values = data_array.sel(sensor_idx=sensor_idx).values
            if np.any(np.isfinite(values)):
                channels.append((var_name, sensor_idx, np.asarray(values)))

    if not channels:
        raise ValueError("Pace dataset has no finite channel values for QC plotting")

    output_dir = Path(output_dir)
    output_path = output_dir / "qc_plots" / f"{filename_prefix}_qc.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_channels = len(channels)
    fig, axes = plt.subplots(
        n_channels,
        3,
        figsize=(20, max(min(1.4 * n_channels, 36), 4.0)),
        layout="constrained",
        squeeze=False,
        gridspec_kw={"width_ratios": [3, 1, 1]},
    )
    fig.set_constrained_layout_pads(h_pad=0.04, hspace=0.02, w_pad=0.04, wspace=0.05)
    time_values = ds["datetime_utc"].values

    for row, (var_name, sensor_idx, values) in enumerate(channels):
        series_ax, histogram_ax, statistics_ax = axes[row]
        valid = values[np.isfinite(values)]
        label = _pace_channel_label(ds, var_name, sensor_idx)
        units = ds[var_name].attrs.get("units", "")
        value_label = f"{var_name} ({units})" if units else var_name

        series_ax.plot(time_values, values, color="tab:blue", linewidth=0.65)
        series_ax.set_title(label, loc="left", fontsize=10, fontweight="bold")
        series_ax.set_ylabel(value_label, fontsize=8)
        series_ax.grid(True, alpha=0.25)
        series_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        series_ax.tick_params(axis="x", labelsize=7, rotation=0)
        series_ax.tick_params(axis="y", labelsize=7)
        if row == n_channels - 1:
            series_ax.set_xlabel("Date (UTC)", fontsize=8)
        else:
            series_ax.tick_params(axis="x", labelbottom=False)

        histogram_ax.hist(valid, bins="auto", color="tab:blue", alpha=0.8)
        histogram_ax.axvline(np.mean(valid), color="tab:red", linewidth=1, label="Mean")
        histogram_ax.axvline(
            np.median(valid), color="tab:orange", linestyle="--", linewidth=1, label="Median"
        )
        histogram_ax.set_title("Distribution", fontsize=9)
        histogram_ax.set_xlabel(units, fontsize=8)
        histogram_ax.tick_params(axis="both", labelsize=7)
        histogram_ax.grid(True, axis="y", alpha=0.25)
        histogram_ax.legend(fontsize=6, frameon=False)

        statistics_ax.axis("off")
        statistics_ax.set_title("Statistics", loc="left", fontsize=9)
        statistics_ax.text(
            0,
            0.95,
            "\n".join(
                (
                    f"Mean: {np.mean(valid):.3g}",
                    f"Median: {np.median(valid):.3g}",
                    f"Std dev: {np.std(valid):.3g}",
                    f"Range: {np.min(valid):.3g} to {np.max(valid):.3g}",
                )
            ),
            transform=statistics_ax.transAxes,
            va="top",
            fontsize=8,
        )

    fig.suptitle(f"Pace QC — {filename_prefix}", fontsize=14, fontweight="bold")
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if logger:
        logger.info(f"    Saved Pace QC plot: {output_path.name}")

    return output_path


def create_wind_masking_qc_plots(
    ds,
    output_dir,
    filename_prefix,
    wind_speed_threshold,
    wind_speed_var="wind_speed_avg",
    wind_direction_var="wind_direction",
    logger=None,
):
    """Create one two-panel wind masking QC plot for each site.

    The plots are intentionally made before wind-direction masking is applied.
    This retains the direction values that will be removed, so the highlighted
    points directly show the effect of the masking rule.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset containing wind speed and direction with a ``datetime_utc``
        coordinate.
    output_dir : str or pathlib.Path
        Directory where the ``qc_plots`` directory will be created.
    filename_prefix : str
        Prefix for output filenames.
    wind_speed_threshold : float
        Wind speeds below this value are highlighted as masked.
    wind_speed_var : str, optional
        Name of the wind speed data variable.
    wind_direction_var : str, optional
        Name of the wind direction data variable.
    logger : logging.Logger, optional
        Logger for status messages.
    """
    if "datetime_utc" not in ds.coords:
        raise ValueError("Wind masking QC plotting requires a datetime_utc coordinate")
    if "site_id" not in ds.coords:
        raise ValueError("Wind masking QC plotting requires a site_id coordinate")
    if wind_speed_var not in ds.data_vars or wind_direction_var not in ds.data_vars:
        return

    qc_dir = Path(output_dir) / "qc_plots"
    qc_dir.mkdir(parents=True, exist_ok=True)
    time_values = ds["datetime_utc"].values

    speed_sensors = {}
    for sensor_idx in ds.sensor_idx.values:
        speed = ds[wind_speed_var].sel(sensor_idx=sensor_idx).values
        if np.any(np.isfinite(speed)):
            site_id = str(ds.site_id.sel(sensor_idx=sensor_idx).item())
            if site_id in speed_sensors:
                raise ValueError(
                    f"Multiple wind speed sensors found for site {site_id}; "
                    "cannot create an unambiguous wind masking QC plot"
                )
            speed_sensors[site_id] = sensor_idx

    direction_sensors = {}
    for sensor_idx in ds.sensor_idx.values:
        direction = ds[wind_direction_var].sel(sensor_idx=sensor_idx).values
        if np.any(np.isfinite(direction)):
            site_id = str(ds.site_id.sel(sensor_idx=sensor_idx).item())
            if site_id in direction_sensors:
                raise ValueError(
                    f"Multiple wind direction sensors found for site {site_id}; "
                    "cannot create an unambiguous wind masking QC plot"
                )
            direction_sensors[site_id] = sensor_idx

    for site_id, direction_idx in direction_sensors.items():
        direction = ds[wind_direction_var].sel(sensor_idx=direction_idx).values
        if site_id not in speed_sensors:
            if logger:
                logger.warning(
                    f"Site {site_id} has wind direction data but no matching wind speed data; "
                    "skipping wind masking QC plot"
                )
            continue

        speed_idx = speed_sensors[site_id]
        speed = ds[wind_speed_var].sel(sensor_idx=speed_idx).values
        low_speed = np.isfinite(speed) & (speed < wind_speed_threshold)
        masked_direction = low_speed & np.isfinite(direction)
        retained_direction = np.where(masked_direction, np.nan, direction)

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        speed_ax, direction_ax = axes

        speed_ax.plot(time_values, speed, color="tab:blue", linewidth=0.7, label="Wind speed")
        speed_ax.scatter(
            time_values[low_speed],
            speed[low_speed],
            color="tab:red",
            s=8,
            zorder=3,
            label="Masked by threshold",
        )
        speed_ax.axhline(
            wind_speed_threshold,
            color="tab:red",
            linestyle="--",
            linewidth=1,
            label=f"Mask threshold ({wind_speed_threshold:g} m/s)",
        )
        speed_ax.set_ylabel("Wind speed (m/s)")
        speed_ax.grid(True, alpha=0.3)
        speed_ax.legend(loc="upper right")

        direction_ax.plot(
            time_values,
            retained_direction,
            color="tab:blue",
            linewidth=0.7,
            label="Wind direction retained",
        )
        direction_ax.scatter(
            time_values[masked_direction],
            direction[masked_direction],
            color="tab:red",
            s=8,
            zorder=3,
            label="Values masked for low speed",
        )
        direction_ax.set_ylabel("Wind direction (degrees)")
        direction_ax.set_ylim(0, 360)
        direction_ax.set_yticks([0, 90, 180, 270, 360])
        direction_ax.grid(True, alpha=0.3)
        direction_ax.legend(loc="upper right")

        direction_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        direction_ax.tick_params(axis="x", rotation=45)
        direction_ax.set_xlabel("Date (UTC)")
        fig.suptitle(
            f"{filename_prefix}: {site_id} wind direction speed masking",
            fontweight="bold",
        )
        fig.tight_layout()

        output_path = qc_dir / f"{filename_prefix}_{site_id}_wind_masking_qc.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        if logger:
            logger.info(f"    Saved wind masking QC plot: {output_path.name}")


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

    # Only measurement variables can have numeric distributions and statistics.
    # This excludes auxiliary timestamp/event arrays such as ``datetime``.
    numeric_variables = [
        var_name
        for var_name, data_array in ds.data_vars.items()
        if "sensor_idx" in data_array.dims and np.issubdtype(data_array.dtype, np.number)
    ]
    for var_name in numeric_variables:
        output_path = qc_dir / f"{filename_prefix}_{var_name}_qc.png"
        title = f"{filename_prefix}: {var_name}"
        create_sensor_qc_plot(ds, var_name, output_path, title=title, logger=logger)
