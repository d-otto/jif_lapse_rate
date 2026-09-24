"""Regression tests for cross-season pipeline behavior."""

from pathlib import Path

import pandas as pd
import pytest
import xarray as xr

from jiflr.data import unstack_sensor_idx
from jiflr.pipeline import (
    MetadataPaths,
    clean_hobo_pendants,
    ensure_season_year_coordinate,
    merge_lvl1_all_years,
)
from jiflr.utils import (
    convert_utc_time_to_local,
    deployment_mask,
    get_deployment_periods,
)


def _season_dataset(year: int, observed_site: str) -> xr.Dataset:
    times = pd.date_range(f"{year}-07-15 12:00", periods=2, freq="5min")
    return xr.Dataset(
        {"temp_c": (("sensor_idx", "datetime_utc"), [[1.0, 2.0]])},
        coords={
            "sensor_idx": [0],
            "datetime_utc": times,
            "year": ("sensor_idx", [year]),
            "site_id": ("sensor_idx", [observed_site]),
            "height": ("sensor_idx", ["1m"]),
            "shielding": ("sensor_idx", ["shielded"]),
            "sensor_type": ("sensor_idx", ["hobo pendant"]),
        },
    )


def test_deployment_periods_filter_to_requested_year(tmp_path: Path) -> None:
    metadata = tmp_path / "deployment_periods.csv"
    metadata.write_text(
        "year,site,deploy_date,deploy_time,pickup_date,pickup_time\n"
        "2025,A01,7/15/25,12:00,7/15/25,12:10\n"
        "2026,A01,7/15/26,12:00,7/15/26,12:10\n"
    )

    periods = get_deployment_periods("A01", metadata, 2026)

    assert periods["A01"] == [(pd.Timestamp("2026-07-15 20:00"), pd.Timestamp("2026-07-15 20:10"))]


def test_deployment_periods_convert_anchorage_time_to_utc_across_dst(tmp_path: Path) -> None:
    metadata = tmp_path / "deployment_periods.csv"
    metadata.write_text(
        "year,site,deploy_date,deploy_time,pickup_date,pickup_time\n"
        "2026,SUMMER,6/13/26,17:22,6/13/26,17:27\n"
        "2026,WINTER,12/13/26,17:22,12/13/26,17:27\n"
    )

    summer = get_deployment_periods("SUMMER", metadata, 2026)["SUMMER"]
    winter = get_deployment_periods("WINTER", metadata, 2026)["WINTER"]

    assert summer == [(pd.Timestamp("2026-06-14 01:22"), pd.Timestamp("2026-06-14 01:27"))]
    assert winter == [(pd.Timestamp("2026-12-14 02:22"), pd.Timestamp("2026-12-14 02:27"))]

    assert convert_utc_time_to_local(summer[0][0]) == pd.Timestamp("2026-06-13 17:22")
    assert convert_utc_time_to_local(winter[0][0]) == pd.Timestamp("2026-12-13 17:22")


def test_deployment_mask_uses_utc_clock(tmp_path: Path) -> None:
    metadata = tmp_path / "deployment_periods.csv"
    metadata.write_text(
        "year,site,deploy_date,deploy_time,pickup_date,pickup_time\n"
        "2026,A01,6/13/26,17:22,6/13/26,17:27\n"
    )
    dataset = xr.Dataset(
        coords={
            "datetime_utc": pd.DatetimeIndex(
                ["2026-06-14 01:20", "2026-06-14 01:25", "2026-06-14 01:30"]
            )
        }
    )

    mask = deployment_mask(dataset, "A01", metadata, 2026)

    assert mask.tolist() == [False, True, False]


def test_pendant_cleaning_retains_utc_coordinate(tmp_path: Path) -> None:
    source = tmp_path / "A01 1m 2026-08-26 13_07_30 PDT (Data AKDT).csv"
    source.write_text(
        "Date Time,Temp (°F) LGR S/N: 10568633 c:1,End Of File LGR S/N: 10568633\n"
        "06/13/26 17:20:00,51.07,\n",
        encoding="utf-8",
    )

    clean_hobo_pendants(source, tmp_path, year=2026)

    output = next(tmp_path.glob("*.nc"))
    with xr.open_dataset(output) as dataset:
        assert "datetime_utc" in dataset.coords
        assert "datetime" not in dataset.coords
        assert str(dataset.datetime_utc.values[0]) == "2026-06-14T01:20:00.000000000"
        assert dataset.attrs["time_coverage_timezone"] == "UTC"


def test_pendant_cleaning_skips_identical_duplicate_export(tmp_path: Path) -> None:
    original = tmp_path / "A01 1m 2026-08-26 13_07_30 PDT (Data AKDT).csv"
    duplicate = tmp_path / "A01-I_1m.csv"
    contents = (
        "Date Time,Temp (°F) LGR S/N: 10568633 c:1,End Of File LGR S/N: 10568633\n"
        "06/13/26 17:20:00,51.07,\n"
    )
    original.write_text(contents, encoding="utf-8")
    duplicate.write_text(contents, encoding="utf-8")

    clean_hobo_pendants([original, duplicate], tmp_path, year=2026)

    assert len(list(tmp_path.glob("*.nc"))) == 1


def test_pendant_cleaning_requires_timezone_in_unique_export(tmp_path: Path) -> None:
    source = tmp_path / "A01-I_1m.csv"
    source.write_text(
        "Date Time,Temp (°F) LGR S/N: 10568633 c:1,End Of File LGR S/N: 10568633\n"
        "06/13/26 17:20:00,51.07,\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Could not determine the logger timezone"):
        clean_hobo_pendants(source, tmp_path, year=2026)


def test_metadata_paths_are_year_specific(tmp_path: Path) -> None:
    paths = MetadataPaths(2026, tmp_path / "data")

    assert paths.directory == tmp_path / "data" / "2026" / "metadata"
    assert paths.data_inventory == paths.directory / "data_inventory.xlsx"
    assert paths.deployment_periods == paths.directory / "deployment_periods.csv"


def test_year_coordinate_is_added_to_legacy_seasonal_data() -> None:
    legacy = _season_dataset(2025, "A01").drop_vars("year")

    normalized = ensure_season_year_coordinate(legacy, 2025, source_name="legacy.nc")

    assert normalized.year.values.tolist() == [2025]


def test_unstack_sensor_idx_keeps_equivalent_sites_from_both_years() -> None:
    combined = xr.concat(
        [
            _season_dataset(2025, "A01"),
            _season_dataset(2026, "A01"),
        ],
        dim="sensor_idx",
        join="outer",
    ).assign_coords(sensor_idx=[0, 1])

    unstacked = unstack_sensor_idx(combined)

    assert set(unstacked.year.values.tolist()) == {2025, 2026}
    assert unstacked.temp_c.sizes["year"] == 2


def test_merge_lvl1_all_years_preserves_observed_and_canonical_sites(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    for year, observed_site in [(2025, "A01"), (2026, "A26")]:
        lvl1_dir = data_root / str(year) / "processed" / "lvl1"
        lvl1_dir.mkdir(parents=True)
        _season_dataset(year, observed_site).to_netcdf(lvl1_dir / "lvl1_on_ice_standard.nc")

    merge_lvl1_all_years([2025, 2026], data_root)

    output = xr.open_dataset(
        data_root / "all_years" / "processed" / "lvl1" / "lvl1_on_ice_standard_all_years.nc"
    )
    assert output.year.values.tolist() == [2025, 2026]
    assert output.site_id.values.tolist() == ["A01", "A26"]
    assert output.canonical_site_id.values.tolist() == ["A01", "A01"]
    assert output.attrs["n_sensors"] == 2
    assert output.attrs["n_sites"] == 2
    assert output.attrs["source_years"] == "2025, 2026"
    assert output.attrs["time_coverage_start"] == "2025-07-15T12:00:00Z"
    assert output.attrs["time_coverage_end"] == "2026-07-15T12:05:00Z"
    assert output["height"].values.tolist() == ["1m", "1m"]


def test_merge_lvl1_all_years_allows_a_category_absent_in_one_season(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    for year in (2025, 2026):
        directory = data_root / str(year) / "processed" / "lvl1"
        directory.mkdir(parents=True)
        _season_dataset(year, "A01").to_netcdf(directory / "lvl1_on_ice_standard.nc")
    _season_dataset(2025, "Lee1").to_netcdf(
        data_root / "2025" / "processed" / "lvl1" / "lvl1_off_ice.nc"
    )

    outputs = merge_lvl1_all_years([2025, 2026], data_root)

    assert {path.name for path in outputs} == {
        "lvl1_off_ice_all_years.nc",
        "lvl1_on_ice_standard_all_years.nc",
    }
    off_ice = xr.open_dataset(
        data_root / "all_years" / "processed" / "lvl1" / "lvl1_off_ice_all_years.nc"
    )
    assert off_ice.year.values.tolist() == [2025]
