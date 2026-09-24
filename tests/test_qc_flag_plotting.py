"""Tests for displaying Level 0 QC flags in sensor plots."""

import numpy as np
import pandas as pd
import xarray as xr

import jiflr.qc_plots as qc_plots


def test_sensor_qc_plot_marks_flagged_values_in_red(tmp_path, monkeypatch):
    dataset = xr.Dataset(
        {
            "temp_c": (("datetime_utc", "sensor_idx"), [[1.0], [2.0], [3.0]]),
            "temp_c_qc_flag": (("sensor_idx", "datetime_utc"), [[0, 1, 0]]),
        },
        coords={
            "sensor_idx": [0],
            "datetime_utc": pd.date_range("2026-06-01", periods=3, freq="min"),
            "site_id": ("sensor_idx", ["A01"]),
        },
    )
    scatter_calls = []
    original_scatter = qc_plots.plt.Axes.scatter

    def record_scatter(axis, *args, **kwargs):
        scatter_calls.append((args, kwargs))
        return original_scatter(axis, *args, **kwargs)

    monkeypatch.setattr(qc_plots.plt.Axes, "scatter", record_scatter)

    output_path = tmp_path / "temp_qc.png"
    qc_plots.create_sensor_qc_plot(dataset, "temp_c", output_path)

    assert output_path.exists()
    assert len(scatter_calls) == 1
    np.testing.assert_allclose(scatter_calls[0][0][1], [2.0])
    assert scatter_calls[0][1]["label"] == "Flagged"


def test_sensor_qc_plot_limits_date_axis_to_plotted_observations(tmp_path, monkeypatch):
    times = pd.date_range("2026-06-01", periods=6, freq="h")
    dataset = xr.Dataset(
        {
            "temp_c": (
                ("sensor_idx", "datetime_utc"),
                [[np.nan, 1.0, 2.0, np.nan, np.nan, np.nan], [np.nan] * 4 + [3.0, np.nan]],
            )
        },
        coords={
            "sensor_idx": [0, 1],
            "datetime_utc": times,
            "site_id": ("sensor_idx", ["A01", "A02"]),
        },
    )
    x_limits = []
    original_set_xlim = qc_plots.plt.Axes.set_xlim

    def record_set_xlim(axis, *args, **kwargs):
        x_limits.append(args)
        return original_set_xlim(axis, *args, **kwargs)

    monkeypatch.setattr(qc_plots.plt.Axes, "set_xlim", record_set_xlim)

    qc_plots.create_sensor_qc_plot(dataset, "temp_c", tmp_path / "temp_qc.png")

    expected = tuple(qc_plots.to_anchorage_time([times[1], times[4]]))
    timestamp_limits = [
        args[0]
        for args in x_limits
        if len(args) == 1 and isinstance(args[0], tuple)
    ]
    assert timestamp_limits == [expected, expected]
