"""Tests for wind-speed-maximum naming and method metadata."""

import importlib.util
import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from jiflr.pipeline import (
    PACE_WIND_SPEED_MAX_METHOD,
    RM_YOUNG_WIND_SPEED_MAX_METHOD,
    _clean_pace_column_names,
)


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "data_pipeline" / "05_add_pendants_to_intensive.py"
SPEC = importlib.util.spec_from_file_location("add_pendants_to_intensive", SCRIPT_PATH)
add_pendants = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(add_pendants)


def test_pace_peak_column_uses_the_shared_maximum_name() -> None:
    frame = pd.DataFrame({"WindSpdPeak2s_ms": [4.2]})

    result = _clean_pace_column_names(frame, metadata={})

    assert list(result) == ["wind_speed_max"]


def _maximum_dataset(site_id: str, sensor_type: str, method: str) -> xr.Dataset:
    return xr.Dataset(
        {
            "wind_speed_max": (
                ("sensor_idx", "datetime_utc"),
                [[4.2]],
                {"method": method},
            )
        },
        coords={
            "sensor_idx": [0],
            "datetime_utc": pd.to_datetime(["2026-06-01T00:00:00"]),
            "site_id": ("sensor_idx", [site_id]),
            "height": ("sensor_idx", ["2m"]),
            "shielding": ("sensor_idx", ["unshielded"]),
            "sensor_type": ("sensor_idx", [sensor_type]),
            "sensor_generation": ("sensor_idx", [f"{sensor_type}_logger"]),
            "sensor_id": ("sensor_idx", [f"{site_id}_{sensor_type}"]),
        },
    )


def test_seasonal_pace_rmyoung_merge_records_both_methods() -> None:
    pace = _maximum_dataset("A01", "pace", PACE_WIND_SPEED_MAX_METHOD)
    rmyoung = _maximum_dataset("A02", "rmyoung", RM_YOUNG_WIND_SPEED_MAX_METHOD)

    result = add_pendants.combine_datasets(
        {"A01": pace},
        {"A02": rmyoung},
        {},
        pd.DatetimeIndex(["2026-06-01T00:00:00"]),
        {"A01": "A01"},
        2026,
        logging.getLogger("test"),
    )

    assert result["wind_speed_max"].attrs["method"] == (
        "Pace: Maximum 2 second mean; RM Young: Maximum of 1 minute samples"
    )


def test_load_rmyoung_data_removes_only_rows_emptied_by_diagnostic_drop(tmp_path) -> None:
    times = pd.date_range("2026-06-01", periods=2, freq="min")
    source = xr.Dataset(
        {
            "wind_speed_avg": (("sensor_idx", "datetime_utc"), [[1.0, 2.0], [float("nan"), float("nan")], [float("nan"), float("nan")]]),
            "temp_c": (("sensor_idx", "datetime_utc"), [[float("nan"), float("nan")], [3.0, 4.0], [float("nan"), float("nan")]]),
            "battery_voltage": (("sensor_idx", "datetime_utc"), [[float("nan"), float("nan")], [float("nan"), float("nan")], [12.0, 12.0]]),
        },
        coords={
            "sensor_idx": [0, 1, 2],
            "datetime_utc": times,
            "site_id": ("sensor_idx", ["A01", "A01", "A01"]),
            "sensor_id": ("sensor_idx", ["logger_wind", "logger_temp", "logger_battery"]),
        },
    )
    source.to_netcdf(tmp_path / "logger_A01.nc")

    result = add_pendants.load_rmyoung_data(tmp_path, logging.getLogger("test"))["A01"]

    assert result.sizes["sensor_idx"] == 2
    assert result.sensor_id.values.tolist() == ["logger_wind", "logger_temp"]
    assert result["wind_speed_avg"].isel(sensor_idx=0).values.tolist() == [1.0, 2.0]
    assert result["temp_c"].isel(sensor_idx=1).values.tolist() == [3.0, 4.0]
