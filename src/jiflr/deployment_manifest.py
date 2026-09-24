"""Machine-readable deployment metadata for one JIFLR field season."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import xarray as xr


MANIFEST_COLUMNS = (
    "site_id",
    "site_type",
    "processing_group",
    "instrument_type",
    "logger_serial",
    "sensor_serial",
    "height_m",
    "shielding",
    "deployed_at_utc",
    "retrieved_at_utc",
    "latitude",
    "longitude",
    "elevation_m",
    "wind_direction_offset_deg",
    "status",
    "notes",
)

INSTRUMENT_TYPES = {
    "hobo_temp",
    "hobo_temp_light",
    "pace_logger",
    "rmyoung_logger",
}


def _optional_text(value: object) -> Optional[str]:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip()
    return text or None


def _optional_float(value: object, *, field: str, row_number: int) -> Optional[float]:
    if value is None or pd.isna(value):
        return None
    try:
        return float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Deployment manifest row {row_number} has invalid {field}: {value!r}"
        ) from error


def _optional_timestamp(value: object, *, field: str, row_number: int) -> Optional[pd.Timestamp]:
    if value is None or pd.isna(value):
        return None
    parsed = pd.to_datetime(value, utc=True, errors="coerce")
    if pd.isna(parsed):
        raise ValueError(
            f"Deployment manifest row {row_number} has invalid {field}: {value!r}"
        )
    return pd.Timestamp(parsed).tz_localize(None)


@dataclass(frozen=True)
class DeploymentRecord:
    """One physical instrument deployed during a field season."""

    site_id: str
    site_type: str
    processing_group: str
    instrument_type: str
    logger_serial: str
    sensor_serial: Optional[str]
    height_m: Optional[float]
    shielding: Optional[str]
    deployed_at_utc: Optional[pd.Timestamp]
    retrieved_at_utc: Optional[pd.Timestamp]
    latitude: Optional[float]
    longitude: Optional[float]
    elevation_m: Optional[float]
    wind_direction_offset_deg: Optional[float]
    status: Optional[str]
    notes: Optional[str]

    @property
    def height(self) -> str:
        """Return the project's standard height label, or an explicit unknown."""
        if self.height_m is None:
            return "unknown"
        if self.height_m.is_integer():
            return f"{int(self.height_m)}m"
        return f"{self.height_m:g}m"


class DeploymentManifest:
    """Validated, serial-number-indexed deployment records for one season."""

    def __init__(self, records: list[DeploymentRecord], path: Path):
        self.records = records
        self.path = path
        self._records_by_serial = {
            record.logger_serial.casefold(): record for record in records
        }

    def lookup(self, logger_serial: str) -> Optional[DeploymentRecord]:
        """Return metadata for a logger serial, or ``None`` when it is unregistered."""
        return self._records_by_serial.get(str(logger_serial).strip().casefold())

    def wind_direction_offsets(self) -> dict[str, float]:
        """Return nonblank wind offsets indexed by case-insensitive site ID."""
        offsets = {}
        for record in self.records:
            if record.wind_direction_offset_deg is not None:
                offsets[record.site_id.casefold()] = record.wind_direction_offset_deg
        return offsets


def load_deployment_manifest(path: Path) -> DeploymentManifest:
    """Load a per-season deployment manifest and validate its stable schema.

    Blank deployment timestamps are valid. They are interpreted only when a
    dataset is masked, using that dataset's first or final timestamp.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Deployment manifest does not exist: {path}")

    frame = pd.read_csv(path, dtype="string", keep_default_na=False).replace("", pd.NA)
    missing_columns = [column for column in MANIFEST_COLUMNS if column not in frame.columns]
    if missing_columns:
        raise ValueError(
            f"Deployment manifest {path} is missing required columns: {missing_columns}"
        )

    records = []
    serial_rows: dict[str, int] = {}
    for index, row in frame.iterrows():
        row_number = index + 2
        required = {
            column: _optional_text(row[column])
            for column in (
                "site_id",
                "site_type",
                "processing_group",
                "instrument_type",
                "logger_serial",
            )
        }
        absent = [column for column, value in required.items() if value is None]
        if absent:
            raise ValueError(
                f"Deployment manifest row {row_number} is missing required values: {absent}"
            )
        if required["instrument_type"] not in INSTRUMENT_TYPES:
            raise ValueError(
                f"Deployment manifest row {row_number} has unsupported instrument_type "
                f"{required['instrument_type']!r}"
            )

        serial_key = required["logger_serial"].casefold()
        if serial_key in serial_rows:
            raise ValueError(
                f"Deployment manifest assigns logger serial {required['logger_serial']!r} "
                f"more than once (rows {serial_rows[serial_key]} and {row_number})"
            )
        serial_rows[serial_key] = row_number

        shielding = _optional_text(row["shielding"])
        if shielding not in {None, "shielded", "unshielded"}:
            raise ValueError(
                f"Deployment manifest row {row_number} has invalid shielding {shielding!r}"
            )

        records.append(
            DeploymentRecord(
                **required,
                sensor_serial=_optional_text(row["sensor_serial"]),
                height_m=_optional_float(row["height_m"], field="height_m", row_number=row_number),
                shielding=shielding,
                deployed_at_utc=_optional_timestamp(
                    row["deployed_at_utc"], field="deployed_at_utc", row_number=row_number
                ),
                retrieved_at_utc=_optional_timestamp(
                    row["retrieved_at_utc"], field="retrieved_at_utc", row_number=row_number
                ),
                latitude=_optional_float(row["latitude"], field="latitude", row_number=row_number),
                longitude=_optional_float(row["longitude"], field="longitude", row_number=row_number),
                elevation_m=_optional_float(row["elevation_m"], field="elevation_m", row_number=row_number),
                wind_direction_offset_deg=_optional_float(
                    row["wind_direction_offset_deg"],
                    field="wind_direction_offset_deg",
                    row_number=row_number,
                ),
                status=_optional_text(row["status"]),
                notes=_optional_text(row["notes"]),
            )
        )
    return DeploymentManifest(records, path)


def deployment_time_mask(
    datetime_values: np.ndarray, record: Optional[DeploymentRecord]
) -> np.ndarray:
    """Return a mask using manifest bounds and dataset limits for blank bounds."""
    times = pd.to_datetime(datetime_values)
    if len(times) == 0 or record is None:
        return np.ones(len(times), dtype=bool)
    start = record.deployed_at_utc if record.deployed_at_utc is not None else times[0]
    end = record.retrieved_at_utc if record.retrieved_at_utc is not None else times[-1]
    return np.asarray((times >= start) & (times <= end), dtype=bool)


def apply_manifest_deployment_mask(ds: xr.Dataset, manifest: DeploymentManifest) -> xr.Dataset:
    """Mask each sensor with its own manifest interval, preserving unknown serials."""
    if "datetime_utc" not in ds.coords or "sensor_idx" not in ds.dims:
        raise ValueError("Manifest deployment masking requires sensor_idx and datetime_utc")
    if "sensor_id" not in ds.coords:
        raise ValueError("Manifest deployment masking requires a sensor_id coordinate")

    times = ds["datetime_utc"].values
    masks = [
        deployment_time_mask(times, manifest.lookup(serial))
        for serial in ds["sensor_id"].values
    ]
    mask = xr.DataArray(
        np.asarray(masks),
        dims=("sensor_idx", "datetime_utc"),
        coords={"sensor_idx": ds["sensor_idx"], "datetime_utc": ds["datetime_utc"]},
    )
    masked = ds.copy()
    for variable in ds.data_vars:
        if {"sensor_idx", "datetime_utc"}.issubset(ds[variable].dims):
            masked[variable] = ds[variable].where(mask)
    return masked


def apply_record_deployment_mask(ds: xr.Dataset, record: Optional[DeploymentRecord]) -> xr.Dataset:
    """Mask every time-indexed variable from one logger using its manifest row."""
    if "datetime_utc" not in ds.coords:
        raise ValueError("Manifest deployment masking requires a datetime_utc coordinate")
    mask = deployment_time_mask(ds["datetime_utc"].values, record)
    masked = ds.copy()
    for variable in ds.data_vars:
        if "datetime_utc" in ds[variable].dims:
            masked[variable] = ds[variable].where(mask)
    return masked


def apply_record_metadata(
    ds: xr.Dataset, record: DeploymentRecord, *, include_configuration: bool = True
) -> xr.Dataset:
    """Assign deployment metadata to every sensor channel from one logger.

    Pace and RM Young loggers contain channels at multiple heights, so callers
    can preserve their parser-derived height and shielding configuration.
    """
    count = ds.sizes["sensor_idx"]
    coordinates = {
        "site_id": ("sensor_idx", [record.site_id] * count),
        "site_type": ("sensor_idx", [record.site_type] * count),
        "processing_group": ("sensor_idx", [record.processing_group] * count),
        "elevation": ("sensor_idx", [record.elevation_m if record.elevation_m is not None else np.nan] * count),
        "latitude": ("sensor_idx", [record.latitude if record.latitude is not None else np.nan] * count),
        "longitude": ("sensor_idx", [record.longitude if record.longitude is not None else np.nan] * count),
    }
    if include_configuration:
        coordinates.update(
            {
                "height": ("sensor_idx", [record.height] * count),
                "shielding": ("sensor_idx", [record.shielding or "unknown"] * count),
            }
        )
    updated = ds.assign_coords(coordinates)
    updated.attrs["deployment_metadata_source"] = "manifest"
    return updated
