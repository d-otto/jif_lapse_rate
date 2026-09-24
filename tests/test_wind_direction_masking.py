"""Tests for PACE-specific wind-direction masking."""

import numpy as np
import pandas as pd
import xarray as xr

from jiflr.pipeline import mask_wind_direction_by_speed


def test_masks_only_pace_wind_direction_and_preserves_wind_speed():
    """Low PACE speeds mask direction without changing any speed observations."""
    times = pd.date_range("2026-06-14", periods=3, freq="5min")
    speed_values = np.array(
        [[np.nan, np.nan, np.nan], [0.2, 0.7, 0.3], [np.nan, np.nan, np.nan], [0.1, 0.2, 0.3]]
    )
    ds = xr.Dataset(
        {
            "wind_speed_avg": (("sensor_idx", "datetime_utc"), speed_values),
            "wind_direction": (
                ("sensor_idx", "datetime_utc"),
                [[10.0, 20.0, 30.0], [np.nan, np.nan, np.nan], [40.0, 50.0, 60.0], [np.nan, np.nan, np.nan]],
            ),
        },
        coords={
            "sensor_idx": [0, 1, 2, 3],
            "datetime_utc": times,
            "site_id": ("sensor_idx", ["A01", "A01", "B01", "B01"]),
            "sensor_type": ("sensor_idx", ["pace", "pace", "other wind", "other wind"]),
        },
    )

    result = mask_wind_direction_by_speed(ds, wind_speed_threshold=0.5)

    np.testing.assert_allclose(
        result["wind_direction"].sel(sensor_idx=0).values,
        [np.nan, 20.0, np.nan],
        equal_nan=True,
    )
    np.testing.assert_allclose(result["wind_direction"].sel(sensor_idx=2).values, [40.0, 50.0, 60.0])
    np.testing.assert_allclose(result["wind_speed_avg"].values, speed_values, equal_nan=True)
    np.testing.assert_allclose(ds["wind_direction"].sel(sensor_idx=0).values, [10.0, 20.0, 30.0])
    assert result["wind_direction"].attrs["wind_speed_masking_sensor_type"] == "pace"
