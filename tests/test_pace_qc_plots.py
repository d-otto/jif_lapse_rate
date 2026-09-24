import numpy as np
import pandas as pd
import xarray as xr

import jiflr.qc_plots as qc_plots


def test_create_pace_qc_plot_creates_one_row_per_populated_channel(tmp_path, monkeypatch):
    time = pd.date_range("2026-06-01", periods=6, freq="h")
    dataset = xr.Dataset(
        {
            "temp_c": (
                ("sensor_idx", "datetime_utc"),
                [[1.0, 2.0, np.nan, 3.0, 4.0, 5.0], [np.nan] * 6],
            ),
            "wind_speed_avg": (
                ("sensor_idx", "datetime_utc"),
                [[np.nan] * 6, [0.2, 0.4, 0.3, 0.7, 0.6, 0.5]],
            ),
        },
        coords={
            "sensor_idx": [0, 1],
            "datetime_utc": time,
            "sensor_id": ("sensor_idx", ["pace_1m_temperature", "pace_2m_wind"]),
            "height": ("sensor_idx", ["1m", "2m"]),
        },
    )
    captured = {}
    original_subplots = qc_plots.plt.subplots

    def record_subplots(*args, **kwargs):
        captured["shape"] = args[:2]
        captured["width_ratios"] = kwargs["gridspec_kw"]["width_ratios"]
        return original_subplots(*args, **kwargs)

    monkeypatch.setattr(qc_plots.plt, "subplots", record_subplots)

    output_path = qc_plots.create_pace_qc_plot(dataset, tmp_path, "A01")

    assert output_path == tmp_path / "qc_plots" / "A01_qc.png"
    assert output_path.exists()
    assert captured["shape"] == (2, 3)
    assert captured["width_ratios"] == [3, 1, 1]


def test_create_pace_qc_plot_rejects_datasets_without_finite_channels(tmp_path):
    dataset = xr.Dataset(
        {"temp_c": (("sensor_idx", "datetime_utc"), [[np.nan, np.nan]])},
        coords={
            "sensor_idx": [0],
            "datetime_utc": pd.date_range("2026-06-01", periods=2, freq="h"),
        },
    )

    try:
        qc_plots.create_pace_qc_plot(dataset, tmp_path, "A01")
    except ValueError as error:
        assert str(error) == "Pace dataset has no finite channel values for QC plotting"
    else:
        raise AssertionError("Expected Pace QC plotting to reject all-NaN data")


def test_pace_qc_plot_uses_the_union_of_channel_observation_times(tmp_path, monkeypatch):
    time = pd.date_range("2026-06-01", periods=5, freq="h")
    dataset = xr.Dataset(
        {
            "temp_c": (("sensor_idx", "datetime_utc"), [[np.nan, 1.0, np.nan, np.nan, np.nan]]),
            "wind_speed_avg": (("sensor_idx", "datetime_utc"), [[np.nan, np.nan, np.nan, 0.5, np.nan]]),
        },
        coords={"sensor_idx": [0], "datetime_utc": time},
    )
    x_limits = []
    original_set_xlim = qc_plots.plt.Axes.set_xlim

    def record_set_xlim(axis, *args, **kwargs):
        x_limits.append(args)
        return original_set_xlim(axis, *args, **kwargs)

    monkeypatch.setattr(qc_plots.plt.Axes, "set_xlim", record_set_xlim)

    qc_plots.create_pace_qc_plot(dataset, tmp_path, "A01")

    expected = tuple(qc_plots.to_anchorage_time([time[1], time[3]]))
    timestamp_limits = [
        args[0]
        for args in x_limits
        if len(args) == 1 and isinstance(args[0], tuple)
    ]
    assert timestamp_limits == [expected, expected]


def test_create_all_qc_plots_uses_channel_layout_and_skips_datetime_data(
    tmp_path, monkeypatch
):
    time = pd.date_range("2026-06-01", periods=3, freq="h")
    dataset = xr.Dataset(
        {
            "temp_c": (("sensor_idx", "datetime_utc"), [[1.0, 2.0, 3.0]]),
            "datetime": (("sensor_idx", "datetime_utc"), [[time[0], time[1], time[2]]]),
        },
        coords={
            "sensor_idx": [0],
            "datetime_utc": time,
            "site_id": ("sensor_idx", ["A01"]),
        },
    )
    captured = {}
    original_subplots = qc_plots.plt.subplots

    def record_subplots(*args, **kwargs):
        captured["shape"] = args[:2]
        captured["width_ratios"] = kwargs["gridspec_kw"]["width_ratios"]
        return original_subplots(*args, **kwargs)

    monkeypatch.setattr(qc_plots.plt, "subplots", record_subplots)

    qc_plots.create_all_qc_plots(dataset, tmp_path, "lvl1_on_ice_standard")

    assert (tmp_path / "qc_plots" / "lvl1_on_ice_standard_temp_c_qc.png").exists()
    assert not (tmp_path / "qc_plots" / "lvl1_on_ice_standard_datetime_qc.png").exists()
    assert captured["shape"] == (1, 3)
    assert captured["width_ratios"] == [3, 1, 1]
