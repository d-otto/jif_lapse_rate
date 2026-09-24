"""Tests for compact, lossless Level NetCDF storage."""

import h5py
import numpy as np
import pandas as pd
import xarray as xr

from jiflr.pipeline import create_netcdf_encoding


def test_create_netcdf_encoding_uses_gzip_sensor_oriented_chunks(tmp_path):
    dataset = xr.Dataset(
        {
            "temp_c": (("sensor_idx", "datetime_utc"), np.ones((2, 3))),
            "site_elevation": (("sensor_idx",), [100.0, 200.0]),
        },
        coords={
            "sensor_idx": [0, 1],
            "datetime_utc": pd.date_range("2026-06-01", periods=3, freq="min"),
        },
    )
    output = tmp_path / "level.nc"

    dataset.to_netcdf(output, encoding=create_netcdf_encoding(dataset))

    with h5py.File(output) as netcdf:
        assert netcdf["temp_c"].compression == "gzip"
        assert netcdf["temp_c"].chunks == (1, 3)
        assert netcdf["site_elevation"].compression is None


def test_create_netcdf_encoding_preserves_strings_added_by_concat(tmp_path):
    def dataset(sensor_type: str, shielding: str) -> xr.Dataset:
        return xr.Dataset(
            {"temp_c": (("sensor_idx", "datetime_utc"), np.ones((1, 1)))},
            coords={
                "sensor_idx": [0],
                "datetime_utc": pd.date_range("2026-06-01", periods=1, freq="min"),
                "sensor_type": ("sensor_idx", [sensor_type]),
                "shielding": ("sensor_idx", [shielding]),
            },
        )

    pace_path = tmp_path / "pace.nc"
    rmyoung_path = tmp_path / "rmyoung.nc"
    output = tmp_path / "merged.nc"
    dataset("pace", "shielded").to_netcdf(pace_path)
    dataset("rmyoung", "unshielded").to_netcdf(rmyoung_path)

    pace = xr.open_dataset(pace_path).load()
    rmyoung = xr.open_dataset(rmyoung_path).load()
    merged = xr.concat([pace, rmyoung], dim="sensor_idx", data_vars="all", coords="all")
    merged.to_netcdf(output, encoding=create_netcdf_encoding(merged))

    result = xr.open_dataset(output).load()
    assert result["sensor_type"].values.tolist() == ["pace", "rmyoung"]
    assert result["shielding"].values.tolist() == ["shielded", "unshielded"]
