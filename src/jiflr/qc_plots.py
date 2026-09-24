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
import pandas as pd
import xarray as xr


ANCHORAGE_TIMEZONE = "America/Anchorage"
QC_PLOT_DPI = 200
QC_TIME_SERIES_LINEWIDTH = 1.5
QC_FLAG_SUFFIX = "_qc_flag"


def to_anchorage_time(time_values):
    """Convert UTC timestamps to naive local timestamps for QC plot axes."""
    timestamps = pd.DatetimeIndex(time_values)
    if timestamps.tz is None:
        timestamps = timestamps.tz_localize("UTC")
    return timestamps.tz_convert(ANCHORAGE_TIMEZONE).tz_localize(None)


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
        Label like "A01 2m (rmyoung)" or "B02 1m (S) (hobo)"
    """
    site_id = str(ds.site_id.values[sensor_idx]) if "site_id" in ds.coords else "?"
    height = str(ds.height.values[sensor_idx]) if "height" in ds.coords else ""
    shielding = str(ds.shielding.values[sensor_idx]) if "shielding" in ds.coords else ""
    sensor_type = (
        str(ds.sensor_type.values[sensor_idx]).strip()
        if "sensor_type" in ds.coords
        else ""
    )

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

    label = " ".join(parts)
    if sensor_type and sensor_type.casefold() != "nan":
        label = f"{label} ({sensor_type})"
    return label


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
        figsize=(20, max(1.4 * n_sensors, 4.0)),
        layout="constrained",
        squeeze=False,
        gridspec_kw={"width_ratios": [3, 1, 1]},
    )
    fig.set_constrained_layout_pads(h_pad=0.04, hspace=0.02, w_pad=0.04, wspace=0.05)
    data_array = ds[var_name]
    time_values = to_anchorage_time(ds["datetime_utc"].values)
    units = ds[var_name].attrs.get("units", "")
    value_label = f"{var_name} ({units})" if units else var_name
    plotted_data = data_array.dropna(dim="datetime_utc", how="all")
    plotted_times = to_anchorage_time(plotted_data["datetime_utc"].values)
    time_limits = (plotted_times[0], plotted_times[-1])
    flag_name = f"{var_name}{QC_FLAG_SUFFIX}"
    has_qc_flags = flag_name in ds.data_vars
    flag_data = None
    if has_qc_flags:
        flag_data = ds[flag_name]
        if set(flag_data.dims) != set(data_array.dims):
            raise ValueError(
                f"QC flag {flag_name} must use the same named dimensions as {var_name}"
            )
        flag_data = flag_data.transpose(*data_array.dims)

    # Plot each sensor with its own distribution and summary.
    for ax_idx, sensor_idx in enumerate(valid_indices):
        series_ax, histogram_ax, statistics_ax = axes[ax_idx]
        data = ds[var_name].isel(sensor_idx=sensor_idx).values
        finite = np.isfinite(data)
        valid = data[finite]
        flagged = np.zeros(data.shape, dtype=bool)
        if has_qc_flags:
            flags = flag_data.isel(sensor_idx=sensor_idx).values
            flagged = finite & (flags != 0)

        label = _get_row_label(ds, sensor_idx, include_shielding=var_name == "temp_c")
        series_ax.plot(
            time_values[finite],
            valid,
            color="tab:blue",
            linewidth=QC_TIME_SERIES_LINEWIDTH,
        )
        if np.any(flagged):
            series_ax.scatter(
                time_values[flagged],
                data[flagged],
                color="tab:red",
                edgecolors="none",
                s=12,
                zorder=3,
                label="Flagged",
            )
            series_ax.legend(loc="best", fontsize=6, frameon=False)
        series_ax.set_title(label, loc="left", fontsize=10, fontweight="bold")
        series_ax.set_xlim(time_limits)
        series_ax.grid(True, alpha=0.25)
        series_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        series_ax.tick_params(axis="x", labelsize=7, rotation=0)
        series_ax.tick_params(axis="y", labelsize=7)
        if ax_idx == n_sensors - 1:
            series_ax.set_xlabel("Date (Alaska time)", fontsize=8)

        histogram_ax.hist(valid, bins="auto", color="tab:blue", alpha=0.8)
        histogram_ax.axvline(np.mean(valid), color="tab:red", linewidth=1, label="Mean")
        histogram_ax.axvline(
            np.median(valid),
            color="tab:orange",
            linestyle="--",
            linewidth=1,
            label="Median",
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
    fig.supylabel(value_label, fontsize=9)

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    # A 300 DPI export makes multi-sensor plots exceed common image viewers'
    # maximum renderable pixel count.  At 200 DPI, the time-series traces
    # remain legible while the generated PNGs can be opened reliably.
    fig.savefig(output_path, dpi=QC_PLOT_DPI, bbox_inches="tight", facecolor="white")
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


def create_pace_qc_plot(
    ds, output_dir, filename_prefix, logger=None, logger_name="Pace"
):
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
        if "sensor_idx" not in data_array.dims or not np.issubdtype(
            data_array.dtype, np.number
        ):
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
    time_values = to_anchorage_time(ds["datetime_utc"].values)
    plotted_times = time_values[
        np.any([np.isfinite(values) for _, _, values in channels], axis=0)
    ]
    time_limits = (plotted_times[0], plotted_times[-1])

    for row, (var_name, sensor_idx, values) in enumerate(channels):
        series_ax, histogram_ax, statistics_ax = axes[row]
        finite = np.isfinite(values)
        valid = values[finite]
        label = _pace_channel_label(ds, var_name, sensor_idx)
        units = ds[var_name].attrs.get("units", "")
        value_label = f"{var_name} ({units})" if units else var_name

        series_ax.plot(
            time_values[finite],
            valid,
            color="tab:blue",
            linewidth=QC_TIME_SERIES_LINEWIDTH,
        )
        series_ax.set_title(label, loc="left", fontsize=10, fontweight="bold")
        series_ax.set_ylabel(value_label, fontsize=8)
        series_ax.set_xlim(time_limits)
        series_ax.grid(True, alpha=0.25)
        series_ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
        series_ax.tick_params(axis="x", labelsize=7, rotation=0)
        series_ax.tick_params(axis="y", labelsize=7)
        if row == n_channels - 1:
            series_ax.set_xlabel("Date (Alaska time)", fontsize=8)
        else:
            series_ax.tick_params(axis="x", labelbottom=False)

        histogram_ax.hist(valid, bins="auto", color="tab:blue", alpha=0.8)
        histogram_ax.axvline(np.mean(valid), color="tab:red", linewidth=1, label="Mean")
        histogram_ax.axvline(
            np.median(valid),
            color="tab:orange",
            linestyle="--",
            linewidth=1,
            label="Median",
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

    fig.suptitle(
        f"{logger_name} QC — {filename_prefix}", fontsize=14, fontweight="bold"
    )
    fig.savefig(output_path, dpi=QC_PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    if logger:
        logger.info(f"    Saved Pace QC plot: {output_path.name}")

    return output_path


def create_rmyoung_wind_qc_plot(ds, output_dir, filename_prefix, logger=None):
    """Plot RM Young WindVector fields and derived east/north components."""
    required = ("wind_speed_avg", "wind_direction", "wind_direction_std")
    missing = [name for name in required if name not in ds]
    if missing:
        raise ValueError(f"RM Young wind QC is missing variables: {missing}")

    def channel_values(name):
        values = ds[name].values
        populated = np.flatnonzero(np.any(np.isfinite(values), axis=1))
        if len(populated) != 1:
            raise ValueError(
                f"RM Young wind QC expected exactly one populated {name} channel; found {len(populated)}"
            )
        return values[populated[0]]

    speed = channel_values("wind_speed_avg")
    direction = channel_values("wind_direction")
    direction_std = channel_values("wind_direction_std")
    direction_radians = np.deg2rad(direction)
    eastward = -speed * np.sin(direction_radians)
    northward = -speed * np.cos(direction_radians)
    rows = (
        (speed, "WVc(1): mean speed (m/s)"),
        (direction, "WVc(2): unit-vector mean direction (degrees from)"),
        (direction_std, "WVc(3): direction standard deviation (degrees)"),
        (eastward, "Derived eastward component (m/s)"),
        (northward, "Derived northward component (m/s)"),
    )
    time_values = to_anchorage_time(ds["datetime_utc"].values)
    plotted_times = time_values[
        np.any([np.isfinite(values) for values, _ in rows], axis=0)
    ]
    time_limits = (plotted_times[0], plotted_times[-1])
    fig, axes = plt.subplots(
        len(rows), 1, figsize=(16, 11), sharex=True, layout="constrained"
    )
    for axis, (values, label) in zip(axes, rows, strict=True):
        axis.plot(time_values, values, linewidth=0.6)
        axis.set_ylabel(label, fontsize=8)
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    axes[0].set_xlim(time_limits)
    axes[-1].set_xlabel("Date (Alaska time)")
    fig.suptitle(
        f"R. M. Young wind QC — {filename_prefix}", fontsize=14, fontweight="bold"
    )
    output_path = Path(output_dir) / "qc_plots" / f"{filename_prefix}_wind_qc.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=QC_PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if logger:
        logger.info(f"    Saved RM Young wind QC plot: {output_path.name}")
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
    if "sensor_type" not in ds.coords:
        if logger:
            logger.warning(
                "Wind masking QC plotting requires a sensor_type coordinate; skipping plots"
            )
        return

    def is_pace_sensor(sensor_idx):
        sensor_type = str(ds["sensor_type"].sel(sensor_idx=sensor_idx).item())
        return sensor_type.lower().startswith("pace")

    qc_dir = Path(output_dir) / "qc_plots"
    qc_dir.mkdir(parents=True, exist_ok=True)
    time_values = to_anchorage_time(ds["datetime_utc"].values)

    speed_sensors = {}
    for sensor_idx in ds.sensor_idx.values:
        if not is_pace_sensor(sensor_idx):
            continue
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
        if not is_pace_sensor(sensor_idx):
            continue
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
        retained_direction = np.isfinite(direction) & ~masked_direction
        plotted_times = time_values[np.isfinite(speed) | np.isfinite(direction)]
        time_limits = (plotted_times[0], plotted_times[-1])

        fig, axes = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
        speed_ax, direction_ax = axes

        speed_ax.plot(
            time_values[np.isfinite(speed)],
            speed[np.isfinite(speed)],
            color="tab:blue",
            linewidth=0.7,
            label="Wind speed",
        )
        speed_ax.scatter(
            time_values[low_speed],
            speed[low_speed],
            color="tab:red",
            s=8,
            zorder=3,
            label="Low-speed observations",
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
            time_values[retained_direction],
            direction[retained_direction],
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
        direction_ax.set_xlim(time_limits)
        direction_ax.grid(True, alpha=0.3)
        direction_ax.legend(loc="upper right")

        direction_ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
        direction_ax.tick_params(axis="x", rotation=45)
        direction_ax.set_xlabel("Date (Alaska time)")
        fig.suptitle(
            f"{filename_prefix}: {site_id} wind direction masking",
            fontweight="bold",
        )
        fig.tight_layout()

        output_path = qc_dir / f"{filename_prefix}_{site_id}_wind_masking_qc.png"
        fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
        plt.close(fig)

        if logger:
            logger.info(f"    Saved wind masking QC plot: {output_path.name}")


def _pressure_filter_sensor_label(ds, sensor_idx):
    """Return a concise identifier for one pressure sensor row."""
    parts = [f"sensor_idx={sensor_idx}"]
    for coordinate in ("site_id", "sensor_id", "sensor_type"):
        if coordinate in ds.coords:
            parts.append(f"{coordinate}={ds[coordinate].sel(sensor_idx=sensor_idx).item()}")
    return ", ".join(parts)


def _pressure_filter_filename_part(value):
    """Return a filename-safe version of a sensor metadata value."""
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in str(value))


def create_pressure_filter_diagnostic(
    before_filter_ds,
    after_filter_ds,
    sensor_idx,
    output_path,
    overview_pressure=None,
    overview_filtered_pressure=None,
    logger=None,
):
    """Write a pressure filter diagnostic with overview, filter, and wind rows.

    ``before_filter_ds`` must be the Level 1 five-minute resampled dataset and
    ``after_filter_ds`` the same dataset after signal processing but before QC
    masking. The comparison therefore isolates only the Butterworth filter.
    ``overview_pressure`` optionally provides a full-record pressure series
    when the two datasets contain only a selected detail window.
    ``overview_filtered_pressure`` supplies its filtered counterpart.
    """
    for dataset_name, dataset in (
        ("before_filter_ds", before_filter_ds),
        ("after_filter_ds", after_filter_ds),
    ):
        if "pressure" not in dataset:
            raise ValueError(f"{dataset_name} has no pressure variable")
        if "datetime_utc" not in dataset.coords:
            raise ValueError(f"{dataset_name} has no datetime_utc coordinate")
        if "sensor_idx" not in dataset["pressure"].dims:
            raise ValueError(f"{dataset_name} pressure must be indexed by sensor_idx")

    before = before_filter_ds["pressure"].sel(sensor_idx=sensor_idx)
    after = after_filter_ds["pressure"].sel(sensor_idx=sensor_idx)
    if not np.isfinite(before.values).any():
        raise ValueError(f"Pressure sensor_idx {sensor_idx} contains no finite values")
    removed = before - after
    time_values = to_anchorage_time(before["datetime_utc"].values)
    site_id = (
        str(before_filter_ds["site_id"].sel(sensor_idx=sensor_idx).item())
        if "site_id" in before_filter_ds.coords
        else None
    )

    fig, axes = plt.subplots(5, 1, figsize=(16, 16), layout="constrained")
    detail_axes = axes[1:]
    for axis in detail_axes[1:]:
        axis.sharex(detail_axes[0])

    overview = before if overview_pressure is None else overview_pressure
    overview_time_values = to_anchorage_time(overview["datetime_utc"].values)
    overview_finite = np.isfinite(overview.values)
    axes[0].plot(
        overview_time_values[overview_finite],
        overview.values[overview_finite],
        color="0.45",
        linewidth=0.6,
        label="Entire pressure series",
    )
    overview_filtered = after if overview_filtered_pressure is None else overview_filtered_pressure
    overview_filtered_time_values = to_anchorage_time(
        overview_filtered["datetime_utc"].values
    )
    overview_filtered_finite = np.isfinite(overview_filtered.values)
    axes[0].plot(
        overview_filtered_time_values[overview_filtered_finite],
        overview_filtered.values[overview_filtered_finite],
        color="tab:red",
        linewidth=0.8,
        label="Filtered pressure",
    )
    if overview_pressure is not None:
        axes[0].axvspan(time_values[0], time_values[-1], color="tab:blue", alpha=0.12, label="Detail window")
    axes[0].set_title("Entire pressure time series")
    axes[0].legend(loc="best")

    detail_axes[0].plot(time_values, before.values, color="0.45", linewidth=0.8, label="Before filter")
    detail_axes[0].plot(time_values, after.values, color="tab:red", linewidth=1.0, label="After filter")
    pressure_units = before.attrs.get("units", "")
    pressure_label = f"Pressure ({pressure_units})" if pressure_units else "Pressure"
    axes[0].set_ylabel(pressure_label)
    detail_axes[0].set_ylabel(pressure_label)
    detail_axes[0].set_title("Pressure series: selected detail window")
    detail_axes[0].legend(loc="best")

    detail_axes[1].axhline(0, color="black", linewidth=0.8)
    detail_axes[1].plot(time_values, removed.values, color="tab:blue", linewidth=0.8)
    detail_axes[1].set_ylabel("Before − after")
    detail_axes[1].set_title("Component removed by filter")

    wind_axis = detail_axes[2]

    def plot_site_wind(variable, label_prefix, color):
        """Plot every finite wind channel of one type at the pressure site."""
        wind_indices = []
        if variable in before_filter_ds.data_vars and site_id is not None:
            for candidate_idx in before_filter_ds.sensor_idx.values:
                candidate_site = str(
                    before_filter_ds["site_id"].sel(sensor_idx=candidate_idx).item()
                )
                wind_speed = before_filter_ds[variable].sel(sensor_idx=candidate_idx)
                if candidate_site == site_id and np.isfinite(wind_speed.values).any():
                    wind_indices.append(candidate_idx)
        for wind_idx in wind_indices:
            wind_speed = before_filter_ds[variable].sel(sensor_idx=wind_idx)
            finite = np.isfinite(wind_speed.values)
            wind_axis.plot(
                time_values[finite],
                wind_speed.values[finite],
                color=color,
                linewidth=0.8,
                label=(
                    f"{label_prefix}: "
                    f"{_pressure_filter_sensor_label(before_filter_ds, wind_idx)}"
                ),
            )
        return wind_indices

    average_indices = plot_site_wind("wind_speed_avg", "Average", "tab:blue")
    maximum_indices = plot_site_wind("wind_speed_max", "Maximum", "tab:orange")
    if average_indices or maximum_indices:
        wind_axis.legend(loc="best", fontsize=7)
    else:
        site_text = site_id if site_id is not None else "this sensor"
        wind_axis.text(
            0.5,
            0.5,
            f"No finite wind-speed series at {site_text}",
            ha="center",
            va="center",
            transform=wind_axis.transAxes,
        )
    wind_axis.set_ylabel("Wind speed (m/s)")
    wind_axis.set_title("Average and maximum wind speed at pressure-sensor site")

    direction_axis = detail_axes[3]
    direction_indices = []
    if "wind_direction" in before_filter_ds.data_vars and site_id is not None:
        for candidate_idx in before_filter_ds.sensor_idx.values:
            candidate_site = str(
                before_filter_ds["site_id"].sel(sensor_idx=candidate_idx).item()
            )
            direction = before_filter_ds["wind_direction"].sel(sensor_idx=candidate_idx)
            if candidate_site == site_id and np.isfinite(direction.values).any():
                direction_indices.append(candidate_idx)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for color_index, direction_idx in enumerate(direction_indices):
        direction = before_filter_ds["wind_direction"].sel(sensor_idx=direction_idx)
        finite = np.isfinite(direction.values)
        direction_axis.plot(
            time_values[finite],
            direction.values[finite],
            color=colors[color_index % len(colors)],
            linewidth=0.8,
            label=_pressure_filter_sensor_label(before_filter_ds, direction_idx),
        )
    if direction_indices:
        direction_axis.legend(loc="best", fontsize=7)
    else:
        site_text = site_id if site_id is not None else "this sensor"
        direction_axis.text(
            0.5,
            0.5,
            f"No finite wind_direction series at {site_text}",
            ha="center",
            va="center",
            transform=direction_axis.transAxes,
        )
    direction_axis.set_ylabel("Wind direction (degrees)")
    direction_axis.set_ylim(0, 360)
    direction_axis.set_yticks([0, 90, 180, 270, 360])
    direction_axis.set_xlabel("Date (Alaska time)")
    direction_axis.set_title("Wind direction at pressure-sensor site")

    for axis in axes:
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
    cutoff_seconds = after.attrs.get("signal_processing_filter_cutoff_period_seconds")
    order = after.attrs.get("signal_processing_filter_order")
    cutoff_text = (
        f", cutoff period {float(cutoff_seconds) / 60:g} minutes"
        if cutoff_seconds is not None
        else ""
    )
    order_text = f", order {order}" if order is not None else ""
    fig.suptitle(
        f"Pressure filter diagnostic: {_pressure_filter_sensor_label(before_filter_ds, sensor_idx)}\n"
        f"Butterworth low-pass{order_text}{cutoff_text}",
        fontweight="bold",
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=QC_PLOT_DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    if logger:
        logger.info(f"    Saved pressure filter diagnostic: {output_path.name}")
    return output_path


def create_pressure_filter_diagnostics(
    before_filter_ds,
    after_filter_ds,
    output_dir,
    filename_prefix,
    logger=None,
):
    """Write one filter diagnostic for each populated PACE pressure series."""
    if "pressure" not in before_filter_ds.data_vars:
        return []
    if "sensor_type" not in before_filter_ds.coords:
        raise ValueError("Pressure filter diagnostics require a sensor_type coordinate")
    if "site_id" not in before_filter_ds.coords:
        raise ValueError("Pressure filter diagnostics require a site_id coordinate")

    output_dir = Path(output_dir) / "qc_plots"
    output_paths = []
    for sensor_idx in before_filter_ds.sensor_idx.values:
        sensor_type = str(before_filter_ds["sensor_type"].sel(sensor_idx=sensor_idx).item())
        pressure = before_filter_ds["pressure"].sel(sensor_idx=sensor_idx)
        if not sensor_type.casefold().startswith("pace") or not np.isfinite(pressure.values).any():
            continue
        site_id = before_filter_ds["site_id"].sel(sensor_idx=sensor_idx).item()
        sensor_id = (
            before_filter_ds["sensor_id"].sel(sensor_idx=sensor_idx).item()
            if "sensor_id" in before_filter_ds.coords
            else sensor_idx
        )
        output_path = output_dir / (
            f"{filename_prefix}_{_pressure_filter_filename_part(site_id)}_"
            f"{_pressure_filter_filename_part(sensor_id)}_pressure_filter_diagnostic.png"
        )
        output_paths.append(
            create_pressure_filter_diagnostic(
                before_filter_ds,
                after_filter_ds,
                sensor_idx,
                output_path,
                logger=logger,
            )
        )
    return output_paths


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
        Prefix for output filenames (e.g., 'lvl0_on_ice_standard')
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
        if not var_name.endswith("_qc_flag")
        and "sensor_idx" in data_array.dims
        and np.issubdtype(data_array.dtype, np.number)
    ]
    for var_name in numeric_variables:
        output_path = qc_dir / f"{filename_prefix}_{var_name}_qc.png"
        title = f"{filename_prefix}: {var_name}"
        create_sensor_qc_plot(ds, var_name, output_path, title=title, logger=logger)
