"""Metadata on finished Level products stays accurate after a NetCDF round trip."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from jiflr.netcdf_metadata import QC_FLAG_MEANINGS, apply_product_metadata


def _dataset() -> xr.Dataset:
    return xr.Dataset(
        {
            "temp_c": (
                ("sensor_idx", "datetime_utc"),
                [[1.0, 2.0], [3.0, 4.0]],
                {"sensor_id": "stale source sensor", "units": "wrong"},
            ),
            "temp_c_qc_flag": (
                ("sensor_idx", "datetime_utc"),
                np.array([[0, 1], [0, 0]], dtype=np.uint32),
            ),
        },
        coords={
            "sensor_idx": [0, 1],
            "datetime_utc": pd.date_range("2026-07-01", periods=2, freq="5min"),
            "sensor_id": ("sensor_idx", ["pendant-1", "rmyoung-1"]),
            "sensor_type": ("sensor_idx", ["hobo pendant", "rmyoung"]),
            "site_id": ("sensor_idx", ["A01", "A02"]),
            "height": ("sensor_idx", ["1m", "2m"]),
            "elevation": ("sensor_idx", [1000.0, 1100.0]),
            "latitude": ("sensor_idx", [59.0, 59.1]),
            "longitude": ("sensor_idx", [-135.0, -135.1]),
        },
        attrs={"sensor_id": "stale source sensor", "n_sensors": 1},
    )


@pytest.mark.parametrize("level", ["lvl0", "lvl1"])
def test_product_metadata_replaces_source_attrs_and_preserves_string_height(tmp_path, level):
    result = apply_product_metadata(
        _dataset(), level=level, product="on_ice_intensive"
    )
    path = tmp_path / f"{level}.nc"
    result.to_netcdf(path)

    with xr.open_dataset(path) as written:
        assert "sensor_id" not in written.attrs
        assert written.attrs["n_sensors"] == 2
        assert written.attrs["n_sites"] == 2
        assert written.attrs["time_coverage_start"] == "2026-07-01T00:00:00Z"
        assert written["height"].values.tolist() == ["1m", "2m"]
        assert "units" not in written["height"].attrs
        assert "standard_name" not in written["elevation"].attrs
        assert written["latitude"].attrs["units"] == "degrees_north"
        assert written["temp_c"].attrs["units"] == "degC"
        assert "sensor_id" not in written["temp_c"].attrs
        assert written["temp_c"].attrs["ancillary_variables"] == "temp_c_qc_flag"
        assert written["temp_c_qc_flag"].attrs["flag_meanings"] == QC_FLAG_MEANINGS
        assert written["temp_c_qc_flag"].attrs["flag_masks"].tolist() == [
            1, 2, 4, 8, 16, 32
        ]
        if level == "lvl0":
            assert "cell_methods" not in written["temp_c"].attrs
            assert "point measurements" in written["temp_c"].attrs["comment"]
        else:
            assert written["temp_c"].attrs["cell_methods"] == "datetime_utc: mean"
            assert "five one-minute samples" in written["temp_c"].attrs["comment"]
            assert written.datetime_utc.attrs["bounds"] == "datetime_utc_bounds"
            assert written.datetime_utc_bounds.dims == ("datetime_utc", "bounds")
            np.testing.assert_array_equal(
                written.datetime_utc_bounds.values[:, 1],
                written.datetime_utc.values + np.timedelta64(5, "m"),
            )
            assert written.attrs["time_coverage_resolution"] == "PT5M"
        assert written.attrs["geospatial_lat_min"] == 59.0
        assert written.attrs["geospatial_lon_max"] == -135.0


def test_product_metadata_fails_for_unknown_measurement():
    dataset = _dataset()
    dataset["unknown_measurement"] = dataset["temp_c"]

    with pytest.raises(ValueError, match="unknown_measurement"):
        apply_product_metadata(dataset, level="lvl0", product="on_ice_intensive")


def test_level0_point_temperature_without_rm_young():
    dataset = _dataset().isel(sensor_idx=[0])

    result = apply_product_metadata(
        dataset, level="lvl0", product="on_ice_standard"
    )

    assert result["temp_c"].attrs["cell_methods"] == "datetime_utc: point"


def test_metadata_does_not_mutate_input_or_measurements():
    dataset = _dataset()
    original = dataset.copy(deep=True)
    result = apply_product_metadata(dataset, level="lvl1", product="on_ice_intensive")
    xr.testing.assert_identical(dataset, original)
    np.testing.assert_array_equal(result.temp_c.values, original.temp_c.values)


@pytest.mark.parametrize("times,match", [
    (["2026-07-01", "2026-07-01"], "unique"),
    (["2026-07-01T00:01", "2026-07-01T00:06"], "bin starts"),
])
def test_invalid_level1_time_labels_fail(times, match):
    dataset = _dataset().assign_coords(datetime_utc=pd.to_datetime(times))
    with pytest.raises(ValueError, match=match):
        apply_product_metadata(dataset, level="lvl1", product="on_ice_intensive")


def test_level1_methods_describe_actual_aggregation_and_limitations():
    dataset = _dataset()
    for name in ("wind_direction", "wind_speed_max", "rainfall_mm"):
        dataset[name] = dataset.temp_c.copy()
    result = apply_product_metadata(dataset, level="lvl1", product="on_ice_intensive")
    for name in ("wind_direction", "wind_speed_max", "rainfall_mm"):
        assert result[name].attrs["cell_methods"] == "datetime_utc: mean"
    assert "not circular" in result.wind_direction.attrs["comment"]
    assert "undercatch" in result.rainfall_mm.attrs["comment"]


def test_geospatial_extent_ignores_missing_positions():
    dataset = _dataset().assign_coords(
        latitude=("sensor_idx", [np.nan, 59.1]),
        longitude=("sensor_idx", [np.nan, np.nan]),
    )
    result = apply_product_metadata(dataset, level="lvl0", product="on_ice_intensive")
    assert result.attrs["geospatial_lat_min"] == 59.1
    assert "geospatial_lon_min" not in result.attrs


def test_bounds_rebuilt_after_cross_season_concatenation(tmp_path):
    seasonal = []
    for year in (2025, 2026):
        dataset = _dataset().drop_vars("temp_c_qc_flag").assign_coords(
            datetime_utc=pd.date_range(f"{year}-07-01", periods=2, freq="5min")
        )
        seasonal.append(apply_product_metadata(dataset, level="lvl1", product="off_ice"))
    merged = xr.concat(seasonal, dim="sensor_idx", data_vars="all", coords="all", join="outer")
    result = apply_product_metadata(
        merged, level="lvl1", product="off_ice", source_years=(2025, 2026)
    )
    path = tmp_path / "all_years.nc"
    result.to_netcdf(path)
    with xr.open_dataset(path) as written:
        assert written.datetime_utc_bounds.dims == ("datetime_utc", "bounds")
        assert written.datetime_utc_bounds.shape == (4, 2)
