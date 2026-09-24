"""Regression tests for flag-first Level 0 quality control."""

import importlib.util
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from jiflr.pipeline import NoiseQCSpec, apply_noise_qc


SCRIPTS_DIRECTORY = Path(__file__).resolve().parents[1] / "scripts" / "data_pipeline"


def _load_script_module(module_name, filename):
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS_DIRECTORY / filename)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


lvl0 = _load_script_module("lvl0_filters", "06_merge_intermediate_to_lvl0.py")
lvl1 = _load_script_module("lvl1_processing", "07_lvl0_to_lvl1.py")


def _dataset():
    times = pd.date_range("2026-06-01T00:00:00", periods=6, freq="min")
    return xr.Dataset(
        {
            "temp_c": (("sensor_idx", "datetime_utc"), [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]),
            "wind_speed_avg": (
                ("sensor_idx", "datetime_utc"),
                [[60, 10, 20, 30, 40, 50], [5, 10, 20, 30, 40, 55]],
            ),
        },
        coords={
            "sensor_idx": [0, 1],
            "datetime_utc": times,
            "site_id": ("sensor_idx", ["A01", "B01"]),
            "sensor_id": ("sensor_idx", ["sensor-a", "sensor-b"]),
        },
    )


def test_filters_add_variable_specific_flags_without_masking_lvl0_data():
    periods = [
        {
            "row_number": 2,
            "start": pd.Timestamp("2026-06-01T00:01:00"),
            "end": pd.Timestamp("2026-06-01T00:02:00"),
            "site_id": "A01",
            "sensor_id": "",
            "variable": "temp_c",
            "notes": "Field inspection found a shaded sensor.",
        }
    ]

    result, counts = lvl0.apply_lvl0_filters(_dataset(), periods, year=2026)

    assert result["temp_c_qc_flag"].values.tolist() == [
        [0, lvl0.MANUAL_MASK_PERIOD_BIT, lvl0.MANUAL_MASK_PERIOD_BIT, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
    ]
    assert result["wind_speed_avg_qc_flag"].values.tolist() == [
        [lvl0.WIND_SPEED_OVER_50_M_S_BIT, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, lvl0.WIND_SPEED_OVER_50_M_S_BIT],
    ]
    assert counts["manual_mask_periods"] == {2: 2}
    assert counts["wind_speed_over_50_m_s"] == 2
    np.testing.assert_allclose(result["temp_c"].values, _dataset()["temp_c"].values)
    assert result["temp_c_qc_flag"].attrs["flag_meanings"] == lvl0.QC_FLAG_MEANINGS


def test_2025_skips_wind_speed_and_rainfall_filters():
    result, counts = lvl0.apply_lvl0_filters(_dataset(), [], year=2025)

    assert counts == {"manual_mask_periods": {}, "wind_direction_low_speed": 0, "pace_pressure_noise_candidates": 0, "wind_speed_noise_candidates": 0}
    assert not result["wind_speed_avg_qc_flag"].any()
    assert result.attrs["qc_filter_order"] == (
        "manual_mask_periods, wind_direction_low_speed, pace_pressure_noise_candidates, "
        "wind_speed_noise_candidates"
    )


def test_unconfigured_year_runs_only_filters_without_year_restrictions():
    result, counts = lvl0.apply_lvl0_filters(_dataset(), [], year=2027)

    assert counts == {"manual_mask_periods": {}, "pace_pressure_noise_candidates": 0, "wind_speed_noise_candidates": 0}
    assert not result["wind_speed_avg_qc_flag"].any()
    assert result.attrs["qc_filter_order"] == (
        "manual_mask_periods, pace_pressure_noise_candidates, wind_speed_noise_candidates"
    )


def test_mask_period_csv_requires_exact_schema_and_converts_utc_offsets(tmp_path):
    csv_path = tmp_path / "lvl0_mask_periods.csv"
    csv_path.write_text(
        "start_datetime_utc,end_datetime_utc,site_id,sensor_id,variable,notes\n"
        "2026-06-01T00:00:00-08:00,2026-06-01T00:05:00-08:00,A01,,,Clock issue\n",
        encoding="utf-8",
    )

    periods = lvl0._load_mask_periods(csv_path)

    assert periods[0]["start"] == pd.Timestamp("2026-06-01T08:00:00")
    assert periods[0]["end"] == pd.Timestamp("2026-06-01T08:05:00")

    csv_path.write_text("start_datetime_utc,end_datetime_utc,notes\n", encoding="utf-8")
    with pytest.raises(ValueError, match="must use these columns"):
        lvl0._load_mask_periods(csv_path)


def test_lvl1_resampling_preserves_all_reason_bits_and_masks_only_flagged_measurements():
    period = {
        "row_number": 2,
        "start": pd.Timestamp("2026-06-01T00:00:00"),
        "end": pd.Timestamp("2026-06-01T00:00:00"),
        "site_id": "A01",
        "sensor_id": "sensor-a",
        "variable": "wind_speed_avg",
        "notes": "Manual check of the implausible value.",
    }
    flagged_lvl0, _ = lvl0.apply_lvl0_filters(_dataset(), [period], year=2026)

    result = lvl1.process_to_5min(flagged_lvl0)

    assert result["wind_speed_avg_qc_flag"].sel(sensor_idx=0).values[0] == 3
    assert np.isnan(result["wind_speed_avg"].sel(sensor_idx=0).values[0])
    assert result["wind_speed_avg_qc_flag"].sel(sensor_idx=1).values[1] == 2
    assert np.isnan(result["wind_speed_avg"].sel(sensor_idx=1).values[1])
    assert result["temp_c"].sel(sensor_idx=0).values[0] == 3
    assert result["temp_c_qc_flag"].dims == result["temp_c"].dims


def test_rainfall_is_flagged_outside_a04_and_a17_then_masked_in_lvl1():
    times = pd.date_range("2026-06-01T00:00:00", periods=2, freq="min")
    dataset = xr.Dataset(
        {
            "rainfall_mm": (
                ("sensor_idx", "datetime_utc"),
                [[1.0, 2.0], [3.0, 4.0], [5.0, np.nan]],
            )
        },
        coords={
            "sensor_idx": [0, 1, 2],
            "datetime_utc": times,
            "site_id": ("sensor_idx", ["A04", "A17", "A03"]),
            "sensor_id": ("sensor_idx", ["rain-a04", "rain-a17", "rain-a03"]),
        },
    )

    flagged_lvl0, counts = lvl0.apply_lvl0_filters(dataset, [], year=2026)
    lvl1_result = lvl1.process_to_5min(flagged_lvl0)

    assert flagged_lvl0["rainfall_mm_qc_flag"].values.tolist() == [
        [0, 0],
        [0, 0],
        [lvl0.RAINFALL_OUTSIDE_ALLOWED_SITES_BIT, 0],
    ]
    assert counts["rainfall_site_allowlist"] == 1
    np.testing.assert_allclose(
        flagged_lvl0["rainfall_mm"].sel(sensor_idx=2).values, [5.0, np.nan]
    )
    assert lvl1_result["rainfall_mm"].sel(sensor_idx=0).values[0] == 1.5
    assert lvl1_result["rainfall_mm"].sel(sensor_idx=1).values[0] == 3.5
    assert np.isnan(lvl1_result["rainfall_mm"].sel(sensor_idx=2).values[0])


@pytest.mark.parametrize("year", [2025, 2026])
def test_low_speed_wind_direction_is_flagged_in_lvl0_then_masked_in_lvl1(year):
    times = pd.date_range(f"{year}-06-01T00:00:00", periods=3, freq="min")
    dataset = xr.Dataset(
        {
            "wind_speed_avg": (
                ("sensor_idx", "datetime_utc"),
                [[np.nan, np.nan, np.nan], [0.2, 0.7, 0.3], [0.1, 0.2, 0.3]],
            ),
            "wind_direction": (
                ("sensor_idx", "datetime_utc"),
                [[10.0, 20.0, 30.0], [np.nan, np.nan, np.nan], [40.0, 50.0, 60.0]],
            ),
        },
        coords={
            "sensor_idx": [0, 1, 2],
            "datetime_utc": times,
            "site_id": ("sensor_idx", ["A01", "A01", "B01"]),
            "sensor_id": ("sensor_idx", ["direction-a01", "speed-a01", "other-b01"]),
            "sensor_type": ("sensor_idx", ["pace", "pace", "other wind"]),
        },
    )

    flagged_lvl0, counts = lvl0.apply_lvl0_filters(dataset, [], year=year)
    lvl1_result = lvl1.process_to_5min(flagged_lvl0)

    assert flagged_lvl0["wind_direction_qc_flag"].values.tolist() == [
        [lvl0.WIND_DIRECTION_LOW_SPEED_BIT, 0, lvl0.WIND_DIRECTION_LOW_SPEED_BIT],
        [0, 0, 0],
        [0, 0, 0],
    ]
    assert counts["wind_direction_low_speed"] == 2
    np.testing.assert_allclose(
        flagged_lvl0["wind_direction"].sel(sensor_idx=0).values, [10.0, 20.0, 30.0]
    )
    assert np.isnan(lvl1_result["wind_direction"].sel(sensor_idx=0).values[0])
    assert lvl1_result["wind_direction"].sel(sensor_idx=2).values[0] == 50.0


def test_lvl0_flags_pace_pressure_hampel_candidates_without_masking_values():
    times = pd.date_range("2026-06-01T00:00:00", periods=25, freq="5min")
    pressure = np.full(len(times), 80.0)
    pressure[10] = 95.0
    dataset = xr.Dataset(
        {
            "pressure": (
                ("sensor_idx", "datetime_utc"),
                [pressure, pressure],
            )
        },
        coords={
            "sensor_idx": [0, 1],
            "datetime_utc": times,
            "sensor_type": ("sensor_idx", ["pace", "other"]),
        },
    )

    result, counts = lvl0.apply_lvl0_filters(dataset, [], year=2026)

    assert counts["pace_pressure_noise_candidates"] == 1
    assert result["pressure_qc_flag"].sel(sensor_idx=0).values[10] == (
        lvl0.PRESSURE_NOISE_CANDIDATE_QC_BIT
    )
    assert not result["pressure_qc_flag"].sel(sensor_idx=1).any()
    np.testing.assert_allclose(result["pressure"].values, dataset["pressure"].values)


def test_lvl1_masks_short_pace_pressure_candidate_and_keeps_qc_flag():
    times = pd.date_range("2026-06-01T00:00:00", periods=25, freq="5min")
    expected_pressure = 80.0 + 0.1 * np.arange(len(times))
    observed_pressure = expected_pressure.copy()
    observed_pressure[10] = 95.0
    flags = np.zeros((2, len(times)), dtype=np.uint32)
    flags[0, 10] = lvl0.PRESSURE_NOISE_CANDIDATE_QC_BIT
    dataset = xr.Dataset(
        {
            "pressure": (
                ("sensor_idx", "datetime_utc"),
                [observed_pressure, observed_pressure],
            ),
            "pressure_qc_flag": (("sensor_idx", "datetime_utc"), flags),
        },
        coords={
            "sensor_idx": [0, 1],
            "datetime_utc": times,
            "sensor_type": ("sensor_idx", ["pace", "other"]),
        },
    )

    result = lvl1.process_to_5min(dataset)

    assert np.isnan(result["pressure"].sel(sensor_idx=0).values[10])
    assert result["pressure_qc_flag"].sel(sensor_idx=0).values[10] == (
        lvl0.PRESSURE_NOISE_CANDIDATE_QC_BIT
    )
    assert "interpolated_qc_flag" not in result
    np.testing.assert_allclose(
        result["pressure"].sel(sensor_idx=1).values, observed_pressure
    )


def test_lvl1_leaves_overlong_pace_pressure_candidate_run_masked():
    times = pd.date_range("2026-06-01T00:00:00", periods=110, freq="5min")
    pressure = 80.0 + 0.1 * np.arange(len(times))
    flags = np.zeros((1, len(times)), dtype=np.uint32)
    flags[0, 4:102] = lvl0.PRESSURE_NOISE_CANDIDATE_QC_BIT
    dataset = xr.Dataset(
        {
            "pressure": (("sensor_idx", "datetime_utc"), [pressure]),
            "pressure_qc_flag": (("sensor_idx", "datetime_utc"), flags),
        },
        coords={
            "sensor_idx": [0],
            "datetime_utc": times,
            "sensor_type": ("sensor_idx", ["pace"]),
        },
    )

    result = lvl1.process_to_5min(dataset)

    assert np.isnan(result["pressure"].sel(sensor_idx=0).values[4:102]).all()
    assert np.all(
        result["pressure_qc_flag"].sel(sensor_idx=0).values[4:102]
        == lvl0.PRESSURE_NOISE_CANDIDATE_QC_BIT
    )
    assert "interpolated_qc_flag" not in result


def test_wind_speed_noise_flags_both_channels_then_masks_outliers():
    times = pd.date_range("2026-06-01", periods=25, freq="5min")
    avg = np.full(25, 5.0)
    maximum = np.full(25, 8.0)
    avg[10] = 20.0
    maximum[12] = 23.0
    dataset = xr.Dataset(
        {
            "wind_speed_avg": (("sensor_idx", "datetime_utc"), [avg]),
            "wind_speed_max": (("sensor_idx", "datetime_utc"), [maximum]),
        },
        coords={"sensor_idx": [0], "datetime_utc": times},
    )

    flagged, counts = lvl0.apply_lvl0_filters(dataset, [], year=2025)
    masked = lvl1.process_to_5min(flagged)

    assert counts["wind_speed_noise_candidates"] == 2
    assert flagged["wind_speed_avg_qc_flag"].values[0, 10] == 32
    assert flagged["wind_speed_max_qc_flag"].values[0, 12] == 32
    np.testing.assert_allclose(flagged["wind_speed_avg"].values[0], avg)
    assert np.isnan(masked["wind_speed_avg"].sel(sensor_idx=0).values[10])
    assert np.isnan(masked["wind_speed_max"].sel(sensor_idx=0).values[12])
    assert masked["wind_speed_avg_qc_flag"].sel(sensor_idx=0).values[10] == 32
    assert masked["wind_speed_max_qc_flag"].sel(sensor_idx=0).values[12] == 32
    assert "wind_speed_avg_interpolated_qc_flag" not in masked
    assert "wind_speed_max_interpolated_qc_flag" not in masked


def test_noise_qc_uses_original_timestamps_and_flags_only_the_spike():
    times = pd.date_range("2026-06-01", periods=25, freq="min")
    speed = np.full(25, 5.0)
    speed[11] = 20.0
    dataset = xr.Dataset(
        {"wind_speed_avg": (("sensor_idx", "datetime_utc"), [speed])},
        coords={"sensor_idx": [0], "datetime_utc": times},
    )

    result, counts = lvl0.apply_lvl0_filters(dataset, [], year=2025)

    assert counts["wind_speed_noise_candidates"] == 1
    assert result["wind_speed_avg_qc_flag"].values[0, 11] == 32
    assert result["wind_speed_avg_qc_flag"].values[0, 10] == 0
    assert result["wind_speed_avg_qc_flag"].values[0, 12] == 0


def test_noise_qc_spec_can_target_another_measurement():
    times = pd.date_range("2026-06-01", periods=25, freq="min")
    temperature = np.full(25, 2.0)
    temperature[10] = 15.0
    dataset = xr.Dataset(
        {
            "temp_c": (("sensor_idx", "datetime_utc"), [temperature]),
            "temp_c_qc_flag": (
                ("sensor_idx", "datetime_utc"), np.zeros((1, 25), dtype=np.uint32)
            ),
        },
        coords={"sensor_idx": [0], "datetime_utc": times},
    )
    spec = NoiseQCSpec(
        name="temperature_noise", variables=("temp_c",), flag_bit=64,
        absolute_floor=5.0, floor_unit="degc", window="720min",
        min_periods=7, mad_multiplier=0.25,
    )

    result, count = apply_noise_qc(dataset, spec=spec)

    assert count == 1
    assert result["temp_c_qc_flag"].values[0, 10] == 64
    assert result.attrs["temperature_noise_qc_absolute_floor_degc"] == 5.0


def test_wind_speed_noise_masks_long_runs_and_preserves_overlapping_flags():
    times = pd.date_range("2026-06-01", periods=110, freq="5min")
    speed = np.full(110, 5.0)
    flags = np.zeros((1, 110), dtype=np.uint32)
    flags[0, 4:102] = lvl0.WIND_SPEED_NOISE_CANDIDATE_QC_BIT
    flags[0, 10] |= lvl0.MANUAL_MASK_PERIOD_BIT
    dataset = xr.Dataset(
        {
            "wind_speed_avg": (("sensor_idx", "datetime_utc"), [speed]),
            "wind_speed_avg_qc_flag": (("sensor_idx", "datetime_utc"), flags),
        },
        coords={"sensor_idx": [0], "datetime_utc": times},
    )

    result = lvl1.process_to_5min(dataset)

    assert np.isnan(result["wind_speed_avg"].sel(sensor_idx=0).values[4:102]).all()
    np.testing.assert_array_equal(result["wind_speed_avg_qc_flag"].sel(sensor_idx=0).values, flags[0])
    assert "wind_speed_avg_interpolated_qc_flag" not in result
