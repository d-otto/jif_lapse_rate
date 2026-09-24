from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from jiflr.data import load_all_pendant_data
from jiflr.deployment_manifest import (
    MANIFEST_COLUMNS,
    apply_manifest_deployment_mask,
    load_deployment_manifest,
)


def _write_manifest(path: Path, records: list[dict[str, object]]) -> Path:
    frame = pd.DataFrame(records, columns=MANIFEST_COLUMNS).fillna("")
    frame.to_csv(path, index=False)
    return path


def _record(**overrides: object) -> dict[str, object]:
    record = {column: "" for column in MANIFEST_COLUMNS}
    record.update(
        {
            "site_id": "A01",
            "site_type": "on_ice_standard",
            "processing_group": "on_ice",
            "instrument_type": "hobo_temp",
            "logger_serial": "1001",
            "height_m": "2",
            "shielding": "shielded",
        }
    )
    record.update(overrides)
    return record


@pytest.mark.parametrize(
    ("start", "end", "expected"),
    [
        ("2026-06-01T01:00:00Z", "2026-06-01T02:00:00Z", [False, True, True, False]),
        ("2026-06-01T01:00:00Z", "", [False, True, True, True]),
        ("", "2026-06-01T02:00:00Z", [True, True, True, False]),
        ("", "", [True, True, True, True]),
    ],
)
def test_manifest_masks_blank_bounds_with_dataset_limits(
    tmp_path: Path, start: str, end: str, expected: list[bool]
) -> None:
    manifest = load_deployment_manifest(
        _write_manifest(tmp_path / "deployment_manifest.csv", [_record(deployed_at_utc=start, retrieved_at_utc=end)])
    )
    times = pd.date_range("2026-06-01", periods=4, freq="h")
    dataset = xr.Dataset(
        {"temp_c": (("sensor_idx", "datetime_utc"), np.arange(4, dtype=float).reshape(1, -1))},
        coords={
            "sensor_idx": [0],
            "datetime_utc": times,
            "sensor_id": ("sensor_idx", ["1001"]),
        },
    )

    masked = apply_manifest_deployment_mask(dataset, manifest)

    assert np.isfinite(masked["temp_c"].values[0]).tolist() == expected


def test_unknown_serial_is_not_masked_or_dropped(tmp_path: Path) -> None:
    manifest = load_deployment_manifest(
        _write_manifest(
            tmp_path / "deployment_manifest.csv",
            [_record(deployed_at_utc="2026-06-01T01:00:00Z", retrieved_at_utc="2026-06-01T01:00:00Z")],
        )
    )
    times = pd.date_range("2026-06-01", periods=3, freq="h")
    dataset = xr.Dataset(
        {"temp_c": (("sensor_idx", "datetime_utc"), np.ones((2, 3)))},
        coords={
            "sensor_idx": [0, 1],
            "datetime_utc": times,
            "sensor_id": ("sensor_idx", ["1001", "unknown-serial"]),
        },
    )

    masked = apply_manifest_deployment_mask(dataset, manifest)

    assert np.isfinite(masked["temp_c"].values[0]).tolist() == [False, True, False]
    assert np.isfinite(masked["temp_c"].values[1]).all()


def test_pendant_loader_keeps_a15_light_and_temperature_sensors(tmp_path: Path) -> None:
    times = pd.date_range("2026-06-01", periods=10, freq="5min")
    for serial, shielding, include_light in (
        ("10568620", "unshielded", True),
        ("22133634", "shielded", False),
    ):
        variables = {"temp_c": (("sensor_idx", "datetime_utc"), np.ones((1, len(times))))}
        if include_light:
            variables["intensity_lux"] = (
                ("sensor_idx", "datetime_utc"), np.full((1, len(times)), 100.0)
            )
        dataset = xr.Dataset(
            variables,
            coords={
                "sensor_idx": [0],
                "datetime_utc": times,
                "sensor_id": ("sensor_idx", [serial]),
                "site_id": ("sensor_idx", ["A15"]),
                "height": ("sensor_idx", ["2m"]),
                "shielding": ("sensor_idx", [shielding]),
            },
        )
        dataset.to_netcdf(tmp_path / f"{serial}.nc")

    loaded = load_all_pendant_data(tmp_path, year=2026)

    assert set(loaded["A15"]) == {"10568620", "22133634"}
