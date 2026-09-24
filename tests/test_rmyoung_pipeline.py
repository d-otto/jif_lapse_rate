"""Tests for the R. M. Young intermediate processing path."""

import importlib.util
import logging
from pathlib import Path

import pandas as pd
import xarray as xr

from jiflr.pipeline import (
    RM_YOUNG_COLUMNS,
    RM_YOUNG_WIND_SPEED_MAX_METHOD,
    clean_rmyoung_loggers,
)


def test_clean_rmyoung_loggers_converts_alaska_time_and_preserves_diagnostics(tmp_path):
    """Weather tables become manifest-tagged UTC intermediate datasets."""
    raw_dir = tmp_path / "raw" / "rmyoung" / "A15"
    raw_dir.mkdir(parents=True)
    weather_path = raw_dir / "A-15-RM_Weather.dat"
    columns = ["TIMESTAMP", "RECORD", *RM_YOUNG_COLUMNS, "WindSpeed_ms_TMx"]
    weather_path.write_text(
        '"TOA5","A-15-RM","CR350","11711","CR350.1.8.1","CPU:test","1","Weather"\n'
        + ",".join(f'"{column}"' for column in columns)
        + "\n"
        + ",".join('""' for _ in columns)
        + "\n"
        + ",".join('""' for _ in columns)
        + "\n"
        + "\n".join(
            ",".join(
                str(value)
                for value in [
                    timestamp,
                    record,
                    5.0,
                    180.0,
                    20.0,
                    6.0,
                    0.4,
                    2.0,
                    0.1,
                    80.0,
                    1.0,
                    900.0,
                    0.2,
                    1,
                    0.2,
                    7.0,
                    12.5,
                    0,
                    0,
                    timestamp,
                ]
            )
            for timestamp, record in (("2026-06-25 01:00:00", 0), ("2026-06-25 01:05:00", 1))
        )
        + "\n"
    )
    manifest_path = tmp_path / "deployment_manifest.csv"
    pd.DataFrame(
        [{
            "site_id": "A15", "site_type": "intensive", "processing_group": "on_ice_intensive",
            "instrument_type": "rmyoung_logger", "logger_serial": "11711", "sensor_serial": "",
            "height_m": 2, "shielding": "unshielded", "deployed_at_utc": "2026-06-25T09:00:00Z",
            "retrieved_at_utc": "2026-06-25T09:05:00Z", "latitude": 58.9, "longitude": -134.1,
            "elevation_m": 1700, "wind_direction_offset_deg": "", "status": "retrieved", "notes": "",
        }]
    ).to_csv(manifest_path, index=False)

    outputs = clean_rmyoung_loggers(raw_dir.parent, tmp_path / "intermediate", manifest_path, 2026)

    assert outputs == [tmp_path / "intermediate" / "11711_A15.nc"]
    with xr.open_dataset(outputs[0]) as ds:
        assert str(ds.datetime_utc.values[0]) == "2026-06-25T09:00:00.000000000"
        assert {name for name, _, _ in RM_YOUNG_COLUMNS.values()} <= set(ds.data_vars)
        assert "wind_speed_max_time" in ds
        assert ds["wind_speed_max"].attrs["method"] == RM_YOUNG_WIND_SPEED_MAX_METHOD
        assert set(ds.site_id.values) == {"A15"}
        assert set(ds.sensor_type.values) == {"rmyoung"}

    script = Path(__file__).resolve().parents[1] / "scripts" / "data_pipeline" / "05_add_pendants_to_intensive.py"
    spec = importlib.util.spec_from_file_location("intensive_merge", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    loaded = module.load_rmyoung_data(
        tmp_path / "intermediate", logging.getLogger(__name__)
    )["A15"]
    assert loaded["pressure"].attrs["units"] == "kPa"
    assert float(loaded["pressure"].max()) == 90.0
