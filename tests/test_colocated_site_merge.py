from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from jiflr.pipeline import merge_sites


def _site_dataset(site_id: str, heights: list[str], values: list[float]) -> xr.Dataset:
    times = pd.date_range("2025-06-01", periods=3, freq="h")
    sensor_count = len(heights)
    return xr.Dataset(
        {
            "temp_c": (
                ("sensor_idx", "datetime_utc"),
                np.asarray(values, dtype=float).reshape(sensor_count, 1)
                * np.ones((1, len(times))),
            )
        },
        coords={
            "sensor_idx": np.arange(sensor_count),
            "datetime_utc": times,
            "site_id": ("sensor_idx", [site_id] * sensor_count),
            "sensor_id": (
                "sensor_idx",
                [f"{site_id}-{index}" for index in range(sensor_count)],
            ),
            "height": ("sensor_idx", heights),
            "shielding": ("sensor_idx", ["shielded"] * sensor_count),
            "sensor_type": ("sensor_idx", ["hobo"] * sensor_count),
            "sensor_generation": ("sensor_idx", ["hobo"] * sensor_count),
            "year": ("sensor_idx", [2025] * sensor_count),
            "site_type": ("sensor_idx", ["on_ice_standard"] * sensor_count),
            "processing_group": ("sensor_idx", ["on_ice"] * sensor_count),
            "elevation": ("sensor_idx", [1000.0] * sensor_count),
            "latitude": ("sensor_idx", [58.0] * sensor_count),
            "longitude": ("sensor_idx", [-134.0] * sensor_count),
        },
    )


def test_colocated_merge_preserves_manifest_metadata() -> None:
    merged = merge_sites(
        {
            "G03A": _site_dataset("G03A", ["1m", "2m"], [1.0, 2.0]),
            "G03B": _site_dataset("G03B", ["1m"], [3.0]),
        },
        "G03",
    )

    assert merged["site_id"].values.tolist() == ["G03", "G03"]
    assert merged["year"].values.tolist() == [2025, 2025]
    assert merged["site_type"].values.tolist() == ["on_ice_standard"] * 2
    assert merged["processing_group"].values.tolist() == ["on_ice"] * 2
