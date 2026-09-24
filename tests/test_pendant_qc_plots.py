import importlib.util
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import xarray as xr
from PIL import Image


SCRIPT_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "data_pipeline"
    / "04_merge_raw_pendants_by_site.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("pendant_merge", SCRIPT_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_temperature_qc_plot_uses_200_dpi_and_supports_height_difference(tmp_path):
    module = _load_module()
    time = pd.date_range("2026-06-01", periods=48, freq="h")
    temperatures = np.array(
        [np.linspace(-2, 4, len(time)), np.linspace(-1, 5, len(time))]
    )
    dataset = xr.Dataset(
        {"temp_c": (("sensor_idx", "datetime_utc"), temperatures)},
        coords={
            "sensor_idx": [0, 1],
            "datetime_utc": time,
            "height": ("sensor_idx", ["1m", "2m"]),
            "shielding": ("sensor_idx", ["shielded", "shielded"]),
            "sensor_id": ("sensor_idx", ["one", "two"]),
        },
    )

    module.create_qc_plots(dataset, "A01", tmp_path)

    output_path = tmp_path / "qc_plots" / "A01_qc.png"
    assert output_path.exists()
    with Image.open(output_path) as image:
        assert image.info["dpi"] == (199.9996, 199.9996)


def test_temperature_qc_plot_ignores_empty_height_mean_warnings(tmp_path):
    module = _load_module()
    time = pd.date_range("2026-06-01", periods=4, freq="h")
    temperatures = np.array(
        [[np.nan, 1.0, 2.0, 3.0], [np.nan, 2.0, 3.0, 4.0]]
    )
    dataset = xr.Dataset(
        {"temp_c": (("sensor_idx", "datetime_utc"), temperatures)},
        coords={
            "sensor_idx": [0, 1],
            "datetime_utc": time,
            "height": ("sensor_idx", ["1m", "2m"]),
            "shielding": ("sensor_idx", ["shielded", "shielded"]),
            "sensor_id": ("sensor_idx", ["one", "two"]),
        },
    )

    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        module.create_qc_plots(dataset, "A01", tmp_path)


def test_temperature_qc_plot_uses_each_sensor_finite_time_grid(tmp_path, monkeypatch):
    module = _load_module()
    time = pd.date_range("2026-06-01", periods=6, freq="min")
    dataset = xr.Dataset(
        {
            "temp_c": (
                ("sensor_idx", "datetime_utc"),
                [[1.0, np.nan, 3.0, np.nan, 5.0, np.nan], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0]],
            )
        },
        coords={
            "sensor_idx": [0, 1],
            "datetime_utc": time,
            "height": ("sensor_idx", ["2m", "2m"]),
            "shielding": ("sensor_idx", ["unshielded", "shielded"]),
            "sensor_id": ("sensor_idx", ["sparse", "continuous"]),
        },
    )
    plot_calls = []
    original_plot = module.plt.Axes.plot

    def record_plot(axis, x_values, y_values, *args, **kwargs):
        if kwargs.get("label", "").startswith("2m"):
            plot_calls.append((np.asarray(x_values), np.asarray(y_values)))
        return original_plot(axis, x_values, y_values, *args, **kwargs)

    monkeypatch.setattr(module.plt.Axes, "plot", record_plot)

    module.create_qc_plots(dataset, "A01", tmp_path)

    assert len(plot_calls) == 2
    np.testing.assert_array_equal(
        plot_calls[0][0], module.to_anchorage_time(time[[0, 2, 4]])
    )
    np.testing.assert_allclose(plot_calls[0][1], [1.0, 3.0, 5.0])
    np.testing.assert_array_equal(plot_calls[1][0], module.to_anchorage_time(time))
    np.testing.assert_allclose(plot_calls[1][1], [0.0, 1.0, 2.0, 3.0, 4.0, 5.0])


def test_temperature_qc_histograms_use_deployment_data_and_fixed_bin_widths(
    tmp_path, monkeypatch
):
    module = _load_module()
    time = pd.date_range("2026-06-01", periods=4, freq="h")
    dataset = xr.Dataset(
        {
            "temp_c": (
                ("sensor_idx", "datetime_utc"),
                [[-10.0, 1.0, 2.0, 20.0], [-9.0, 2.0, 3.0, 21.0]],
            )
        },
        coords={
            "sensor_idx": [0, 1],
            "datetime_utc": time,
            "height": ("sensor_idx", ["1m", "2m"]),
            "shielding": ("sensor_idx", ["shielded", "shielded"]),
            "sensor_id": ("sensor_idx", ["one", "two"]),
        },
    )
    deployment_dataset = dataset.copy(deep=True)
    deployment_dataset["temp_c"] = deployment_dataset["temp_c"].where(
        xr.DataArray(
            [[False, True, True, False], [False, True, True, False]],
            dims=("sensor_idx", "datetime_utc"),
            coords={
                "sensor_idx": dataset.sensor_idx,
                "datetime_utc": dataset.datetime_utc,
            },
        )
    )

    histogram_calls = []
    original_hist = module.plt.Axes.hist

    def record_histogram(axis, values, *args, **kwargs):
        histogram_calls.append((np.asarray(values), np.asarray(kwargs["bins"])))
        return original_hist(axis, values, *args, **kwargs)

    monkeypatch.setattr(module.plt.Axes, "hist", record_histogram)

    module.create_qc_plots(
        dataset,
        "A01",
        tmp_path,
        histogram_ds=deployment_dataset,
    )

    assert len(histogram_calls) == 3
    np.testing.assert_allclose(histogram_calls[0][0], [1.0, 2.0])
    np.testing.assert_allclose(histogram_calls[1][0], [2.0, 3.0])
    np.testing.assert_allclose(np.diff(histogram_calls[0][1]), 0.25)
    np.testing.assert_allclose(np.diff(histogram_calls[1][1]), 0.25)
    np.testing.assert_allclose(histogram_calls[2][0], [1.0, 1.0])
    np.testing.assert_allclose(np.diff(histogram_calls[2][1]), 0.1)


def test_masked_deployment_intervals_exclude_deployed_time():
    module = _load_module()
    time = pd.date_range("2026-06-01", periods=10, freq="h")

    intervals = module._masked_deployment_intervals(time, [(time[2], time[7])])

    assert intervals == [(time[0], time[2]), (time[7], time[-1])]


def test_sensor_colors_use_quarter_and_three_quarter_colormap_values():
    module = _load_module()

    colors = module._sensor_colors(2)

    np.testing.assert_allclose(colors, module.cmo.cm.haline([0.25, 0.75]))
