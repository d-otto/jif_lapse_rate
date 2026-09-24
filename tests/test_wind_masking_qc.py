"""Tests for wind-direction masking quality-control plots."""

import numpy as np
import pandas as pd
import xarray as xr

import jiflr.qc_plots as qc_plots


def test_create_wind_masking_qc_plot(tmp_path):
    """Create a two-row QC plot with low-speed direction values highlighted."""
    times = pd.date_range("2026-06-14", periods=4, freq="5min")
    ds = xr.Dataset(
        {
            "wind_speed_avg": (
                ("sensor_idx", "datetime_utc"),
                [[np.nan, np.nan, np.nan, np.nan], [0.2, 0.5, 0.7, 0.3]],
            ),
            "wind_direction": (
                ("sensor_idx", "datetime_utc"),
                [[45.0, 90.0, 135.0, 180.0], [np.nan, np.nan, np.nan, np.nan]],
            ),
        },
        coords={
            "sensor_idx": [0, 1],
            "datetime_utc": times,
            "site_id": ("sensor_idx", ["A01", "A01"]),
            "sensor_type": ("sensor_idx", ["pace", "pace"]),
        },
    )

    qc_plots.create_wind_masking_qc_plots(
        ds,
        output_dir=tmp_path,
        filename_prefix="lvl1_test",
        wind_speed_threshold=0.5,
    )

    output_path = tmp_path / "qc_plots" / "lvl1_test_A01_wind_masking_qc.png"
    assert output_path.exists()
    assert output_path.stat().st_size > 0


def test_wind_masking_qc_plot_draws_retained_data_lines(tmp_path, monkeypatch):
    """Both retained-data lines have gaps where the masking rule applies."""
    times = pd.date_range("2026-06-14", periods=4, freq="5min")
    ds = xr.Dataset(
        {
            "wind_speed_avg": (("sensor_idx", "datetime_utc"), [[0.2, 0.6, 0.7, 0.3]]),
            "wind_direction": (("sensor_idx", "datetime_utc"), [[45.0, 90.0, 135.0, 180.0]]),
        },
        coords={
            "sensor_idx": [0],
            "datetime_utc": times,
            "site_id": ("sensor_idx", ["A01"]),
            "sensor_type": ("sensor_idx", ["pace"]),
        },
    )
    plotted_lines = {"speed": [], "direction": []}
    original_subplots = qc_plots.plt.subplots

    def record_subplots(*args, **kwargs):
        fig, axes = original_subplots(*args, **kwargs)
        for name, axis in zip(plotted_lines, axes):
            original_plot = axis.plot

            def record_plot(*plot_args, _name=name, _plot=original_plot, **plot_kwargs):
                plotted_lines[_name].append((plot_args, plot_kwargs))
                return _plot(*plot_args, **plot_kwargs)

            axis.plot = record_plot
        return fig, axes

    monkeypatch.setattr(qc_plots.plt, "subplots", record_subplots)

    qc_plots.create_wind_masking_qc_plots(
        ds,
        output_dir=tmp_path,
        filename_prefix="lvl1_test",
        wind_speed_threshold=0.5,
    )

    assert len(plotted_lines["speed"]) == 1
    speed_values = plotted_lines["speed"][0][0][1]
    np.testing.assert_allclose(speed_values, [0.2, 0.6, 0.7, 0.3])
    assert plotted_lines["speed"][0][1]["label"] == "Wind speed"

    assert len(plotted_lines["direction"]) == 1
    direction_values = plotted_lines["direction"][0][0][1]
    np.testing.assert_allclose(direction_values, [90.0, 135.0])
    assert plotted_lines["direction"][0][1]["label"] == "Wind direction retained"


def test_wind_masking_qc_plot_limits_axis_to_speed_and_direction_data(
    tmp_path, monkeypatch
):
    times = pd.date_range("2026-06-14", periods=6, freq="5min")
    ds = xr.Dataset(
        {
            "wind_speed_avg": (
                ("sensor_idx", "datetime_utc"),
                [[np.nan, 0.2, np.nan, np.nan, np.nan, np.nan]],
            ),
            "wind_direction": (
                ("sensor_idx", "datetime_utc"),
                [[np.nan, np.nan, np.nan, np.nan, 180.0, np.nan]],
            ),
        },
        coords={
            "sensor_idx": [0],
            "datetime_utc": times,
            "site_id": ("sensor_idx", ["A01"]),
            "sensor_type": ("sensor_idx", ["pace"]),
        },
    )
    x_limits = []
    original_set_xlim = qc_plots.plt.Axes.set_xlim

    def record_set_xlim(axis, *args, **kwargs):
        x_limits.append(args)
        return original_set_xlim(axis, *args, **kwargs)

    monkeypatch.setattr(qc_plots.plt.Axes, "set_xlim", record_set_xlim)

    qc_plots.create_wind_masking_qc_plots(
        ds,
        output_dir=tmp_path,
        filename_prefix="lvl1_test",
        wind_speed_threshold=0.5,
    )

    expected = tuple(qc_plots.to_anchorage_time([times[1], times[4]]))
    timestamp_limits = [
        args[0]
        for args in x_limits
        if len(args) == 1 and isinstance(args[0], tuple)
    ]
    assert timestamp_limits == [expected]
