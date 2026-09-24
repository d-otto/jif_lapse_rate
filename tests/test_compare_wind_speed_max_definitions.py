"""Tests for the Pace/RM Young maximum-wind comparison script."""

import importlib.util
from pathlib import Path

import numpy as np
import xarray as xr


SCRIPT_PATH = Path(__file__).parents[1] / "scripts" / "compare_wind_speed_max_definitions.py"
SPEC = importlib.util.spec_from_file_location("wind_speed_comparison", SCRIPT_PATH)
comparison = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(comparison)


def _dataset(maximum_name: str, maximum_values: list[float]) -> xr.Dataset:
    time = np.array(["2026-06-01T00:00:00", "2026-06-01T00:05:00"], dtype="datetime64[ns]")
    return xr.Dataset(
        {
            "wind_speed_avg": (("sensor_idx", "datetime_utc"), [[4.0, 5.0], [np.nan, np.nan]]),
            maximum_name: (("sensor_idx", "datetime_utc"), [[np.nan, np.nan], maximum_values]),
        },
        coords={
            "sensor_idx": [0, 1], "datetime_utc": time,
            "site_id": ("sensor_idx", ["A01", "A01"]), "height": ("sensor_idx", ["2m", "2m"]),
            "shielding": ("sensor_idx", ["unshielded", "unshielded"]),
            "sensor_type": ("sensor_idx", ["pace", "pace"]),
            "sensor_id": ("sensor_idx", ["mean", "maximum"]),
        },
    )


def test_comparison_writes_source_and_site_statistics(tmp_path: Path) -> None:
    pace_dir = tmp_path / "pace"
    rmyoung_dir = tmp_path / "rmyoung"
    output_dir = tmp_path / "output"
    pace_dir.mkdir()
    rmyoung_dir.mkdir()
    _dataset("wind_speed_peak", [4.5, 6.0]).to_netcdf(pace_dir / "pace.nc")
    _dataset("wind_speed_max", [5.0, 7.0]).to_netcdf(rmyoung_dir / "rmyoung.nc")

    comparison.main([
        "--year", "2026", "--pace-dir", str(pace_dir), "--rmyoung-dir", str(rmyoung_dir),
        "--output-dir", str(output_dir), "--site-pair", "A01:A01", "--exclude-pace-zero-wind",
    ])

    records = comparison.pd.read_csv(output_dir / "wind_speed_max_comparison_records.csv")
    assert set(records["source"]) == {"pace", "rmyoung"}
    assert set(records["maximum_definition"]) == {
        "two-second peak", "maximum of one-minute subsamples in five-minute interval",
    }
    assert (output_dir / "wind_speed_max_comparison.png").is_file()
    assert (output_dir / "wind_speed_max_comparison_by_mean_speed_bin.csv").is_file()
    assert (output_dir / "wind_speed_distribution_pace_A01_vs_rmyoung_A01_pace_avg_nonzero.png").is_file()


def test_site_pair_differences_match_nearest_five_minute_intervals() -> None:
    pace = comparison.pd.DataFrame(
        {
            "datetime_utc": comparison.pd.to_datetime(["2026-06-01T00:04:59"]),
            "wind_speed_avg": [4.0], "wind_speed_max": [6.0],
        }
    )
    rmyoung = comparison.pd.DataFrame(
        {
            "datetime_utc": comparison.pd.to_datetime(["2026-06-01T00:05:00"]),
            "wind_speed_avg": [3.5], "wind_speed_max": [4.5],
        }
    )

    matched = comparison._match_site_pair_observations(pace, rmyoung)

    assert len(matched) == 1
    assert matched["wind_speed_avg_pace"].iloc[0] - matched["wind_speed_avg_rmyoung"].iloc[0] == 0.5
