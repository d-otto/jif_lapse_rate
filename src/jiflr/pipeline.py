# -*- coding: utf-8 -*-
"""
pipeline.py

Data processing workflows for converting raw CSV data to NetCDF format.

Author: drotto
Created: 2025-01-20
Project: jif_lapse_rate
"""

import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional
import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import re
import warnings
from zoneinfo import ZoneInfo

from jiflr import ROOT
from jiflr.deployment_manifest import (
    DeploymentManifest,
    apply_record_deployment_mask,
    apply_record_metadata,
    load_deployment_manifest,
)
from jiflr.netcdf_metadata import (
    MEASUREMENT_ATTRS,
    WIND_SPEED_MAX_METHODS,
    apply_product_metadata,
)

# Module-level logger for pipeline operations
_logger = logging.getLogger("jiflr.pipeline")

PACE_WIND_SPEED_MAX_METHOD = WIND_SPEED_MAX_METHODS["pace"][1]
RM_YOUNG_WIND_SPEED_MAX_METHOD = WIND_SPEED_MAX_METHODS["rmyoung"][1]
NETCDF_COMPRESSION_LEVEL = 5
NETCDF_TIME_CHUNK_SIZE = 4096


@dataclass(frozen=True)
class NoiseQCSpec:
    """Settings for one group of measurements sharing a noise QC rule."""

    name: str
    variables: tuple[str, ...]
    flag_bit: int
    absolute_floor: float
    floor_unit: str
    window: str
    min_periods: int
    mad_multiplier: float
    sensor_type_prefixes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.name or not self.variables or not self.floor_unit:
            raise ValueError("Noise QC requires a name, variables, and a floor unit")
        if not 0 < self.flag_bit <= np.iinfo(np.uint32).max or self.flag_bit & (self.flag_bit - 1):
            raise ValueError("Noise QC flag_bit must be one uint32 bit")
        if pd.Timedelta(self.window) <= pd.Timedelta(0):
            raise ValueError("Noise QC window must be a positive time duration")
        if self.min_periods < 1 or self.mad_multiplier <= 0 or self.absolute_floor <= 0:
            raise ValueError("Noise QC periods, multiplier, and floor must be positive")


def rolling_hampel_candidates(
    series: pd.Series,
    *,
    window: str,
    min_periods: int,
    mad_multiplier: float,
    absolute_floor: float,
) -> pd.Series:
    """Find outliers using a centered, time-based rolling median and MAD."""
    if not isinstance(series.index, pd.DatetimeIndex):
        raise TypeError("Noise QC requires a DatetimeIndex")
    if not series.index.is_monotonic_increasing:
        raise ValueError("Noise QC requires timestamps sorted in ascending order")
    if min_periods < 1:
        raise ValueError("min_periods must be at least one")
    if mad_multiplier <= 0 or absolute_floor <= 0:
        raise ValueError("mad_multiplier and absolute_floor must be positive")

    baseline = series.rolling(window, center=True, min_periods=min_periods).median()
    residual = (series - baseline).abs()
    mad = residual.rolling(window, center=True, min_periods=min_periods).median()
    threshold = np.maximum(absolute_floor, mad_multiplier * 1.4826 * mad)
    return series.notna() & baseline.notna() & (residual > threshold)


def apply_noise_qc(ds: xr.Dataset, *, spec: NoiseQCSpec) -> tuple[xr.Dataset, int]:
    """Flag raw observations selected by a noise rule without changing values."""
    variables = [name for name in spec.variables if name in ds.data_vars]
    if not variables:
        return ds, 0
    if spec.sensor_type_prefixes and "sensor_type" not in ds.coords:
        raise ValueError("Noise QC sensor selection requires a sensor_type coordinate")

    time_index = pd.DatetimeIndex(ds["datetime_utc"].values)
    if not time_index.is_monotonic_increasing:
        raise ValueError("Noise QC requires sorted datetime_utc values")

    affected = 0
    sensor_type_prefixes = tuple(value.casefold() for value in spec.sensor_type_prefixes)
    for variable in variables:
        flag_name = f"{variable}_qc_flag"
        if flag_name not in ds:
            raise ValueError(f"Noise QC requires {flag_name}")
        for sensor_idx in ds["sensor_idx"].values:
            if sensor_type_prefixes:
                sensor_type = str(ds["sensor_type"].sel(sensor_idx=sensor_idx).item())
                if not sensor_type.casefold().startswith(sensor_type_prefixes):
                    continue
            values = ds[variable].sel(sensor_idx=sensor_idx)
            if not values.notnull().any():
                continue
            series = pd.Series(values.values, index=time_index)
            candidates = rolling_hampel_candidates(
                series,
                window=spec.window,
                min_periods=spec.min_periods,
                mad_multiplier=spec.mad_multiplier,
                absolute_floor=spec.absolute_floor,
            )
            affected += int(candidates.sum())
            mask = xr.DataArray(
                candidates.to_numpy(),
                dims=("datetime_utc",),
                coords={"datetime_utc": ds["datetime_utc"]},
            )
            ds[flag_name].loc[dict(sensor_idx=sensor_idx)] = (
                ds[flag_name].sel(sensor_idx=sensor_idx)
                | xr.where(mask, np.uint32(spec.flag_bit), np.uint32(0))
            )

    prefix = f"{spec.name}_qc"
    ds.attrs.update(
        {
            f"{prefix}_method": "centered_rolling_median_hampel",
            f"{prefix}_rolling_window": spec.window,
            f"{prefix}_min_periods": spec.min_periods,
            f"{prefix}_mad_multiplier": spec.mad_multiplier,
            f"{prefix}_absolute_floor_{spec.floor_unit}": spec.absolute_floor,
        }
    )
    return ds, affected


def create_netcdf_encoding(dataset: xr.Dataset) -> dict[str, dict[str, object]]:
    """Return lossless NetCDF encoding for Level-product datasets.

    Level products use ``sensor_idx × datetime_utc`` arrays that are often
    sparse.  Sensor-oriented chunks preserve efficient single-sensor reads,
    while gzip and shuffle make fill-value-heavy chunks compact on disk.
    String coordinates use NetCDF character arrays so that values introduced
    during a merge are not truncated by a stale string-width encoding inherited
    from an input file.
    """
    if "sensor_idx" not in dataset.sizes or "datetime_utc" not in dataset.sizes:
        raise ValueError(
            "Compressed Level NetCDF output requires sensor_idx and datetime_utc dimensions"
        )

    n_sensors = dataset.sizes["sensor_idx"]
    n_times = dataset.sizes["datetime_utc"]
    if n_sensors == 0 or n_times == 0:
        raise ValueError(
            "Compressed Level NetCDF output requires non-empty sensor_idx and datetime_utc dimensions"
        )

    chunksizes = (1, min(NETCDF_TIME_CHUNK_SIZE, n_times))
    encoding = {
        name: {
            "zlib": True,
            "complevel": NETCDF_COMPRESSION_LEVEL,
            "shuffle": True,
            "chunksizes": chunksizes,
        }
        for name, variable in dataset.variables.items()
        if variable.dims == ("sensor_idx", "datetime_utc")
    }
    encoding.update(
        {
            name: {"dtype": "S1"}
            for name, variable in dataset.coords.items()
            if variable.dtype.kind in {"U", "S"}
        }
    )
    return encoding


@dataclass(frozen=True)
class MetadataPaths:
    """Metadata locations for one field season."""

    year: int
    data_root: Path = ROOT / "data"

    def __post_init__(self) -> None:
        if not isinstance(self.year, int) or not 1900 <= self.year <= 2100:
            raise ValueError(f"year must be an integer between 1900 and 2100; got {self.year!r}")

    @property
    def directory(self) -> Path:
        return self.data_root / str(self.year) / "metadata"

    @property
    def deployment_periods(self) -> Path:
        return self.directory / "deployment_periods.csv"

    @property
    def data_inventory(self) -> Path:
        return self.directory / "data_inventory.xlsx"

    @property
    def deployment_manifest(self) -> Path:
        return self.directory / "deployment_manifest.csv"


def read_season_metadata(path: Path, year: int, *, source_name: str) -> pd.DataFrame:
    """Read metadata and return validated, nonblank records for one field season."""
    if not path.exists():
        raise FileNotFoundError(f"{source_name} metadata does not exist: {path}")
    frame = pd.read_excel(path) if path.suffix.lower() in {".xlsx", ".xls"} else pd.read_csv(path)
    frame = frame.dropna(how="all").copy()
    if "year" not in frame.columns:
        raise ValueError(f"{source_name} metadata must contain a 'year' column: {path}")
    parsed_year = pd.to_numeric(frame["year"], errors="coerce")
    if parsed_year.isna().any() or (parsed_year % 1 != 0).any():
        bad_rows = frame.index[parsed_year.isna() | (parsed_year % 1 != 0)].tolist()
        raise ValueError(
            f"{source_name} metadata has missing or non-integral year values in rows {bad_rows}: {path}"
        )
    frame["year"] = parsed_year.astype(int)
    season_frame = frame.loc[frame["year"] == year].copy()
    if season_frame.empty:
        raise ValueError(f"{source_name} metadata contains no records for year {year}: {path}")
    return season_frame


def ensure_season_year_coordinate(
    ds: xr.Dataset, year: int, *, source_name: str
) -> xr.Dataset:
    """Attach a requested season to legacy data and reject conflicting metadata."""
    if "sensor_idx" not in ds.dims:
        raise ValueError(f"{source_name} does not have a sensor_idx dimension")
    if "year" not in ds.coords:
        return ds.assign_coords(
            year=("sensor_idx", np.full(ds.sizes["sensor_idx"], year, dtype=int))
        )
    if ds["year"].dims != ("sensor_idx",):
        raise ValueError(
            f"{source_name} has a year coordinate that is not indexed by sensor_idx"
        )
    observed_years = set(ds["year"].values.tolist())
    if observed_years != {year}:
        raise ValueError(
            f"{source_name} contains year values {sorted(observed_years)}; "
            f"expected only {year}"
        )
    return ds


# Mapping: processing year -> normalized observed site ID -> canonical site ID.
# The entries are fictional and exist only to test the synthetic 2026 season.
# Add verified field mappings before building a real cross-season data product.
SITE_ASSOCIATIONS: dict[int, dict[str, str]] = {
    2026: {"a26": "A01", "ridge2026": "Lee1"},
}


def canonical_site_id(site_id: str, year: int) -> str:
    """Return the stable site identity for an observed seasonal site ID."""
    observed = str(site_id).strip()
    if not observed:
        raise ValueError("site_id cannot be empty when resolving a canonical site")
    return SITE_ASSOCIATIONS.get(year, {}).get(observed.casefold(), observed)


def standardize_height(height: str) -> str:
    """Standardize height string to consistent format (e.g., '2.0m' -> '2m').

    Parameters
    ----------
    height : str
        Height string in various formats (e.g., "2.0m", "2m", "0.5m")

    Returns
    -------
    str
        Standardized height string with integer format for whole numbers
    """
    if not height:
        return height
    height_str = str(height).strip()
    if height_str.endswith("m"):
        height_str = height_str[:-1]
    try:
        height_val = float(height_str)
        if height_val == int(height_val):
            return f"{int(height_val)}m"
        else:
            return f"{height_val}m"
    except (ValueError, TypeError):
        return height


def standardize_shielding(shielding: str) -> str:
    """Standardize shielding string (e.g., 'unshield' -> 'unshielded').

    Parameters
    ----------
    shielding : str
        Shielding string in various formats

    Returns
    -------
    str
        Standardized shielding string ('shielded' or 'unshielded')
    """
    if not shielding:
        return shielding
    shielding_lower = str(shielding).strip().lower()
    if shielding_lower in ("unshield", "unshielded"):
        return "unshielded"
    elif shielding_lower in ("shield", "shielded"):
        return "shielded"
    return shielding


def convert_utc_to_offset(
    datetime_utc: xr.DataArray, utc_offset_hours: float, name: str = "datetime"
) -> xr.DataArray:
    """
    Convert UTC datetime DataArray to local time with specified UTC offset.

    This function takes a UTC datetime coordinate/variable and converts it to
    local time by applying the specified UTC offset. The result can be assigned
    directly to a dataset coordinate using pandas-like syntax.

    Parameters
    ----------
    datetime_utc : xr.DataArray
        Input datetime array in UTC. Should contain datetime64 values.
    utc_offset_hours : float
        UTC offset in hours. Negative values for locations west of UTC.
        Examples: -9.0 for AKST, -8.0 for AKDT, -5.0 for EST, +1.0 for CET
    name : str, optional
        Name for the returned DataArray (default: "datetime")

    Returns
    -------
    xr.DataArray
        New DataArray with local time values, ready for direct assignment

    Examples
    --------
    Convert UTC to Alaska Standard Time:
    >>> ds["datetime"] = convert_utc_to_offset(ds["datetime_utc"], utc_offset_hours=-9.0)

    Convert to custom coordinate name:
    >>> ds["local_time"] = convert_utc_to_offset(ds["datetime_utc"], utc_offset_hours=-8.0, name="local_time")

    Raises
    ------
    ValueError
        If datetime_utc is not a datetime type or contains invalid values
    TypeError
        If utc_offset_hours is not a number
    """

    # Convert offset hours to numpy timedelta64
    offset_timedelta = np.timedelta64(int(utc_offset_hours * 3600), "s")

    # Apply offset to get local time
    local_datetime = datetime_utc + offset_timedelta

    # Create new DataArray with specified name and preserve attributes
    local_da = xr.DataArray(
        local_datetime.values,
        dims=datetime_utc.dims,
        coords={
            dim: datetime_utc.coords[dim]
            for dim in datetime_utc.dims
            if dim in datetime_utc.coords
        },
        attrs=datetime_utc.attrs.copy(),
        name=name,
    )

    # Add offset information to attributes
    local_da.attrs.update(
        {
            "utc_offset_hours": utc_offset_hours,
            "timezone_info": f"UTC{utc_offset_hours:+.1f}",
            "converted_from": "UTC",
        }
    )

    return local_da


def replace_utc_datetime_coord(
    ds: xr.Dataset,
    utc_offset_hours: float,
    new_coord_name: str = "datetime",
    old_coord_name: str = "datetime_utc",
    add_cf_attributes: bool = True,
) -> xr.Dataset:
    """
    Replace UTC datetime coordinate with local time coordinate.

    This convenience function performs the complete workflow of converting a UTC
    datetime coordinate to local time, swapping the dimension, and removing the
    old coordinate. This is the most common use case for datetime conversion.

    Parameters
    ----------
    ds : xr.Dataset
        Input dataset with UTC datetime coordinate
    utc_offset_hours : float
        UTC offset in hours. Negative values for locations west of UTC.
        Examples: -9.0 for AKST, -8.0 for AKDT, -5.0 for EST, +1.0 for CET
    new_coord_name : str, optional
        Name for the new local time coordinate (default: "datetime")
    old_coord_name : str, optional
        Name of the existing UTC coordinate to replace (default: "datetime_utc")
    add_cf_attributes : bool, optional
        Whether to add CF-compliant timezone attributes (default: True)

    Returns
    -------
    xr.Dataset
        Dataset with UTC coordinate replaced by local time coordinate

    Examples
    --------
    Convert to Alaska Standard Time:
    >>> ds_local = replace_utc_datetime_coord(ds, utc_offset_hours=-9.0)

    Custom coordinate names:
    >>> ds_local = replace_utc_datetime_coord(
    ...     ds, utc_offset_hours=-8.0,
    ...     new_coord_name="local_time",
    ...     old_coord_name="utc_time"
    ... )

    Raises
    ------
    KeyError
        If old_coord_name is not found in the dataset
    ValueError
        If the coordinate is not a datetime type
    """
    # Check that old coordinate exists
    if old_coord_name not in ds.coords:
        raise KeyError(
            f"Coordinate '{old_coord_name}' not found in dataset. Available coordinates: {list(ds.coords.keys())}"
        )

    # Convert UTC datetime to local time
    ds_new = ds.copy()
    ds_new[new_coord_name] = convert_utc_to_offset(
        ds[old_coord_name], utc_offset_hours, name=new_coord_name
    )

    # Swap dimensions if the old coordinate is a dimension
    if old_coord_name in ds.dims:
        ds_new = ds_new.swap_dims({old_coord_name: new_coord_name})

    # Remove the old coordinate
    ds_new = ds_new.drop_vars(old_coord_name)

    # Add CF-compliant timezone attributes if requested
    if add_cf_attributes:
        # Add standard CF time coordinate attributes (calendar handled by xarray CF encoder)
        ds_new[new_coord_name].attrs.update(
            {
                "standard_name": "time",
                "axis": "T",
                "long_name": f"time with UTC offset {utc_offset_hours:+.1f} hours",
            }
        )

        # Create timezone identifier string
        if utc_offset_hours == -9.0:
            tz_name = "Alaska Standard Time (AKST)"
            tz_iana = "America/Anchorage"
        elif utc_offset_hours == -8.0:
            tz_name = "Alaska Daylight Time (AKDT)"
            tz_iana = "America/Anchorage"
        else:
            tz_name = f"UTC{utc_offset_hours:+.1f}"
            tz_iana = f"UTC{utc_offset_hours:+.1f}"

        # Add dataset-level timezone metadata
        if len(ds_new[new_coord_name]) > 0:
            start_time = pd.Timestamp(ds_new[new_coord_name].values[0])
            end_time = pd.Timestamp(ds_new[new_coord_name].values[-1])

            # Format as ISO 8601 with timezone offset
            offset_str = f"{int(utc_offset_hours):+03d}:00"
            start_iso = start_time.strftime(f"%Y-%m-%dT%H:%M:%S{offset_str}")
            end_iso = end_time.strftime(f"%Y-%m-%dT%H:%M:%S{offset_str}")

            ds_new.attrs.update(
                {
                    "time_coverage_start": start_iso,
                    "time_coverage_end": end_iso,
                    "time_coverage_timezone": tz_iana,
                    "time_coverage_timezone_name": tz_name,
                }
            )

    return ds_new


def load_deployment_metadata(csv_path: Path, year: int) -> Dict[str, Dict[str, float]]:
    """
    Load deployment metadata (elevations, coordinates) from deployment CSV.

    Parameters
    ----------
    csv_path : Path
        Path to deployment_periods.csv file

    Returns
    -------
    dict
        Dictionary with keys 'elevations', 'latitudes', 'longitudes'
        Each containing {site_name: value} mappings

    Example
    -------
    >>> metadata = load_deployment_metadata(Path("deployment_periods.csv"))
    >>> elevation = metadata['elevations']['A01']
    """
    metadata = {"elevations": {}, "latitudes": {}, "longitudes": {}}

    if not csv_path.exists():
        return metadata

    try:
        df = read_season_metadata(csv_path, year, source_name="deployment periods")

        for _, row in df.iterrows():
            site = row.get("site")
            if pd.notna(site):
                if pd.notna(row.get("site_elevation")):
                    metadata["elevations"][site] = float(row["site_elevation"])
                if pd.notna(row.get("latitude")):
                    metadata["latitudes"][site] = float(row["latitude"])
                if pd.notna(row.get("longitude")):
                    metadata["longitudes"][site] = float(row["longitude"])

    except Exception as e:
        _logger.warning(f"Error loading deployment metadata: {e}")

    return metadata


def _populate_spatial_metadata(
    ds: "xr.Dataset", deployment_metadata_path: Path, year: int
) -> "xr.Dataset":
    """
    Populate elevation, latitude, and longitude coordinates from deployment metadata.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with site_id coordinate containing site names
    deployment_metadata_path : Path
        Path to deployment_periods.csv file

    Returns
    -------
    xr.Dataset
        Dataset with populated spatial coordinates
    """
    # Load deployment metadata
    metadata = load_deployment_metadata(deployment_metadata_path, year)

    if not metadata or not metadata.get("elevations"):
        _logger.warning(f"No elevation data found in {deployment_metadata_path}")
        return ds

    # Create copies of coordinate arrays to modify
    new_elevations = list(ds.elevation.values)
    new_latitudes = list(ds.latitude.values)
    new_longitudes = list(ds.longitude.values)

    # Create case-insensitive lookup dictionaries
    elevation_lookup = {k.lower(): v for k, v in metadata["elevations"].items()}
    latitude_lookup = {k.lower(): v for k, v in metadata.get("latitudes", {}).items()}
    longitude_lookup = {k.lower(): v for k, v in metadata.get("longitudes", {}).items()}

    # Populate spatial metadata for each sensor
    for i, site_id in enumerate(ds.site_id.values):
        site_name = str(site_id).strip()

        # Skip if site_name is empty or unknown
        if not site_name or site_name.lower() in ["", "unknown", "nan"]:
            continue

        # Convert to lowercase for case-insensitive lookup
        site_name_lower = site_name.lower()

        # Look up elevation
        if site_name_lower in elevation_lookup:
            new_elevations[i] = float(elevation_lookup[site_name_lower])

        # Look up latitude (if available)
        if site_name_lower in latitude_lookup:
            new_latitudes[i] = float(latitude_lookup[site_name_lower])

        # Look up longitude (if available)
        if site_name_lower in longitude_lookup:
            new_longitudes[i] = float(longitude_lookup[site_name_lower])

    # Update coordinates in dataset
    ds = ds.assign_coords(
        {
            "elevation": ("sensor_idx", new_elevations),
            "latitude": ("sensor_idx", new_latitudes),
            "longitude": ("sensor_idx", new_longitudes),
        }
    )

    # Count how many sites were populated
    n_populated = sum(1 for elev in new_elevations if not np.isnan(elev))
    n_total = len(new_elevations)

    # Only log if elevation data is not found for any of the sensors
    if n_populated < n_total:
        _logger.info(
            f"Populated elevation data for {n_populated}/{n_total} sensors from deployment metadata for {site_name}"
        )

    return ds


def _file_checksum(path: Path) -> str:
    """Return the SHA-256 checksum of a raw export without loading it all at once."""
    digest = hashlib.sha256()
    with path.open("rb") as raw_export:
        for chunk in iter(lambda: raw_export.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clean_hobo_pendants(
    ps: list[Path] | Path,
    dir_out: Path,
    manifest_path: Optional[Path] = None,
    year: Optional[int] = None,
):
    """
    Reads data exported from HOBOware and HOBOconnect and outputs it as netcdf.

    Parameters
    ----------
    ps : list[Path] | Path
        Path or list of paths to CSV files exported from HOBOware/HOBOconnect
    dir_out : Path
        Output directory for NetCDF files
    manifest_path : Path, optional
        Path to the per-season deployment manifest. Raw files whose serial is
        absent from this manifest are retained with explicit ``unknown``
        metadata rather than being discarded or coerced to shielded.
    """

    if year is None:
        raise ValueError("year is required when cleaning pendant data")

    # make sure ps is always a list
    if isinstance(ps, list) is False:
        ps = [ps]

    manifest = load_deployment_manifest(manifest_path) if manifest_path else None

    raw_exports_by_checksum: dict[str, Path] = {}

    for p in tqdm(ps):
        checksum = _file_checksum(p)
        original_export = raw_exports_by_checksum.get(checksum)
        if original_export is not None:
            _logger.warning(
                "Skipping duplicate HOBO export %s; it is identical to %s",
                p,
                original_export,
            )
            continue
        raw_exports_by_checksum[checksum] = p

        # First read: extract metadata from plot title
        df_meta = pd.read_csv(p, nrows=1)

        # Extract site metadata from filename
        site_name = None
        sensor_height = None
        sensor_config = None

        filename_parts = p.name.split(" ")
        if len(filename_parts) >= 2:
            # Check if first part looks like a site name (e.g., A01, B03, G03, Lee1, Lee2, Divide, Windward1)
            if (
                re.match(r"^[A-Z]\d+[a-z]?$", filename_parts[0])
                or re.match(r"^[A-Z]+\d*$", filename_parts[0])
                or filename_parts[0]
                in ["Lee1", "Lee2", "Divide", "Windward1", "Windward2"]
            ):
                site_name = filename_parts[0]
                # Check for height/configuration in second part
                if filename_parts[1] in ["1m", "2m"]:
                    sensor_height = filename_parts[1]
                elif filename_parts[1] == "unshielded":
                    sensor_config = "unshielded"
                elif filename_parts[1] == "WX":
                    sensor_config = "weather_station"
                else:
                    # Try to capture any other configuration info
                    sensor_config = filename_parts[1]

            # Handle multi-part processing for patterns like "Lee2 2m unshielded"
            if len(filename_parts) >= 3:
                # Check if we have height and config info
                if (
                    filename_parts[1] in ["1m", "2m"]
                    and filename_parts[2] == "unshielded"
                ):
                    sensor_height = filename_parts[1]
                    sensor_config = "unshielded"

        # Second read: get actual data with proper column headers
        df = pd.read_csv(p)

        # Remove # column if present
        if df.columns[0] == "#":
            df = df.drop(columns="#")

        # Extract serial number from column headers
        sn = None
        for col in df.columns:
            match = re.search(r"S/N:\s*(\d+)", col)
            if match:
                sn = match.group(1)
                break
        if sn is None:
            # Raise exception, would need to implement finding it some other way
            raise ValueError("Serial number not found in column headers")

        # Detect sensor generation based on presence of "Button Down" column
        button_down_cols = [col for col in df.columns if col.startswith("Button Down")]
        if button_down_cols:
            sensor_generation = "new"
        else:
            sensor_generation = "old"

        # Detect timezone from filename
        tz = None
        filename = p.name
        tz_match = re.search(r"\(Data ([A-Z]+)\)", filename)
        if tz_match:
            tz_code = tz_match.group(1)
            # Map timezone codes to ZoneInfo-compatible names
            if tz_code == "AKDT":
                tz = "America/Anchorage"
            elif tz_code == "AKST":
                tz = "America/Anchorage"
            elif tz_code == "PDT":
                tz = "America/Los_Angeles"
            else:
                tz = tz_code  # Use as-is for other timezone codes

        # Detect temperature unit
        temp_col = None
        temp_unit = None
        for col in df.columns:
            if "temp" in col.lower():
                temp_col = col
                if "°F" in col or "temp, °f" in col.lower():
                    temp_unit = "F"
                else:
                    temp_unit = "C"
                break

        # Find light column
        light_col = None
        for col in df.columns:
            if col.startswith("Intensity"):
                light_col = col
                break

        # Find event columns
        event_cols = []
        for col in df.columns:
            if any(
                event in col.lower()
                for event in [
                    "button",
                    "host",
                    "coupler",
                    "stopped",
                    "bad battery",
                    "end of file",
                ]
            ):
                event_cols.append(col)

        # Create column mapping
        col_map = {"Date Time": "datetime"}
        if temp_col:
            col_map[temp_col] = "temp_c"
        if light_col:
            col_map[light_col] = "intensity_lux"

        # Select and rename columns
        available_cols = [col for col in col_map.keys() if col in df.columns]
        df_clean = df[available_cols + event_cols].copy()
        df_clean = df_clean.rename(columns=col_map)

        # Convert numeric columns
        numeric_cols = ["temp_c"]
        if "intensity_lux" in df_clean.columns:
            numeric_cols.append("intensity_lux")
        for col in numeric_cols:
            if col in df_clean.columns:
                df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")

        # Convert temperature from Fahrenheit if needed
        if temp_col and temp_unit == "F":
            df_clean["temp_c"] = (df_clean["temp_c"] - 32) * 5 / 9
            temp_unit = "C"

        # Combine event columns
        # TODO: add more events here? Missing coupler ones, but prob not useful.
        if event_cols:
            event_flags = []
            for _, row in df_clean.iterrows():
                flags = []
                for col in event_cols:
                    if pd.notna(row[col]) & (row[col] != ""):
                        if "button down" in col.lower():
                            flags.append("BD")
                        elif "button up" in col.lower():
                            flags.append("BU")
                        elif "host" in col.lower():
                            flags.append("H")
                        elif "end of file" in col.lower():
                            flags.append("E")
                        elif "bad battery" in col.lower():
                            flags.append("BB")
                event_flags.append(",".join(flags) if flags else "")
            df_clean["events"] = event_flags
            df_clean = df_clean.drop(columns=event_cols)

        # Drop event-only rows (no temperature data)
        # These are rows logged for button presses, host connections, etc.
        if "temp_c" in df_clean.columns:
            df_clean = df_clean.dropna(subset=["temp_c"])

        # Convert datetime with timezone if available
        df_clean["datetime"] = pd.to_datetime(
            df_clean["datetime"], format="%m/%d/%y %H:%M:%S"
        )

        if tz is None:
            raise ValueError(
                f"Could not determine the logger timezone from {p.name!r}. "
                "Expected a filename suffix such as '(Data AKDT)'. "
                "Rename the export to include its logger timezone before processing it."
            )

        # Localize to the timezone, convert to UTC, then remove timezone info
        df_clean["datetime_utc"] = (
            df_clean["datetime"]
            .dt.tz_localize(ZoneInfo(tz))
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)  # Remove timezone info, keeping UTC time
        )

        # Convert to xarray Dataset using sensor_idx structure
        ds = xr.Dataset.from_dataframe(df_clean.set_index("datetime_utc"))

        # Prefer the manifest. Unknown serials remain processable and must not
        # inherit a false shielding classification from filename fallbacks.
        record = manifest.lookup(sn) if manifest else None
        if record is not None:
            site_name = record.site_id
            sensor_height = record.height
            shielding_status = record.shielding or "unknown"
        else:
            shielding_status = (
                "unshielded" if sensor_config == "unshielded" else "unknown"
            )
            if manifest is not None:
                warnings.warn(
                    f"Logger serial {sn} from {p.name} is not in the deployment manifest; "
                    "keeping it with filename-derived metadata."
                )

        # Create sensor_idx dimension (single sensor = index 0)
        sensor_idx = 0

        # Expand all data variables to include sensor_idx dimension
        data_vars_with_sensor_idx = {}
        for var_name, var_data in ds.data_vars.items():
            # Add sensor_idx as first dimension: (sensor_idx, datetime)
            expanded_data = var_data.expand_dims("sensor_idx", axis=0)
            data_vars_with_sensor_idx[var_name] = expanded_data

        # Create new dataset with sensor_idx structure
        ds = xr.Dataset(
            data_vars_with_sensor_idx,
            coords={
                "sensor_idx": [sensor_idx],
                "datetime_utc": ds.datetime_utc,
                # Sensor attributes as coordinates indexed by sensor_idx
                "sensor_id": ("sensor_idx", [sn]),
                "site_id": ("sensor_idx", [site_name if site_name else "Unknown"]),
                "year": ("sensor_idx", [year]),
                "height": (
                    "sensor_idx",
                    [
                        standardize_height(
                            sensor_height
                            if sensor_height
                            else (sensor_config if sensor_config else "")
                        )
                    ],
                ),
                "shielding": ("sensor_idx", [standardize_shielding(shielding_status)]),
                "sensor_type": ("sensor_idx", ["hobo pendant"]),
                "sensor_generation": ("sensor_idx", [sensor_generation]),
                # Placeholder coordinates for spatial data (to be filled from deployment metadata)
                "elevation": ("sensor_idx", [np.nan]),
                "latitude": ("sensor_idx", [np.nan]),
                "longitude": ("sensor_idx", [np.nan]),
            },
        )
        ds["datetime_utc"].attrs.update(
            {
                "standard_name": "time",
                "axis": "T",
                "long_name": "UTC time",
                "timezone": "UTC",
            }
        )

        # Add metadata
        attr_dict = {
            "sensor_type": "hobo pendant",
            "sensor_generation": sensor_generation,
            "sensor_id": sn,
            "tz": tz if tz else "local",
            "temp_unit_original": temp_unit if temp_unit else "C",
            "site_name": site_name if site_name else "",
            "sensor_height": sensor_height if sensor_height else "",
            "sensor_config": sensor_config if sensor_config else "",
            "shielding": shielding_status,
            "year": year,
            "time_coordinate": "datetime_utc",
            "time_coverage_timezone": "UTC",
            "structure": "sensor_idx × datetime",
        }

        # Add attributes to variables
        for var in ds.data_vars:
            ds[var].attrs = attr_dict.copy()

        # Also add metadata to dataset attributes for easy access
        ds.attrs.update(attr_dict)

        # Output to netcdf
        start_time = df_clean["datetime_utc"].iloc[0]
        end_time = df_clean["datetime_utc"].iloc[-1]

        if record is not None:
            ds = apply_record_metadata(ds, record)
        else:
            ds.attrs["deployment_metadata_source"] = "filename"

        fname = f"{sn}_{start_time.strftime('%Y%m%dT%H%M')}_{end_time.strftime('%Y%m%dT%H%M')}.nc"
        pout = dir_out / fname
        ds.to_netcdf(pout)

    return None


def clean_pace_loggers(
    file_paths: list[Path] | Path,
    dir_out: Path,
    convert_to_local_tz: bool = False,
    utc_offset_hours: float = -9.0,
    manifest_path: Optional[Path] = None,
    year: Optional[int] = None,
):
    """
    Parse Pace logger data files from intensive monitoring sites and convert to NetCDF.

    Parameters
    ----------
    file_paths : list[Path] | Path
        Path or list of paths to Pace logger .txt files
    dir_out : Path
        Output directory for NetCDF files
    convert_to_local_tz : bool, optional
        If True, convert from UTC storage to local timezone (default: False)
    utc_offset_hours : float, optional
        UTC offset in hours for local timezone conversion (default: -9.0 for AKST)
    manifest_path : Path, optional
        Path to the per-season deployment manifest.

    Notes
    -----
    This function parses Pace logger files that contain meteorological data from
    intensive monitoring sites. The files contain multiple temperature sensors
    at different heights (50cm, 100cm, 150cm, 200cm) as well as wind, pressure,
    and humidity measurements.

    Height assignments:
    - Temperature sensors: heights extracted from channel descriptions
    - Pressure data: assigned to 1m height
    - Other meteorological data: assigned to 2m height

    The output xarray dataset has dimensions:
    - time: measurement timestamps
    - height: sensor heights in meters
    - site_id: site identifier from filename or label

    Variable names are CF-compliant with units stored as attributes.
    """

    if year is None:
        raise ValueError("year is required when cleaning Pace logger data")
    manifest = load_deployment_manifest(manifest_path) if manifest_path else None

    # Ensure file_paths is always a list
    if isinstance(file_paths, Path):
        file_paths = [file_paths]

    for file_path in tqdm(file_paths, desc="Processing Pace logger files"):
        _process_single_pace_file(
            file_path,
            dir_out,
            convert_to_local_tz,
            utc_offset_hours,
            manifest,
            year,
        )


def merge_lvl1_all_years(years: List[int], data_root: Path = ROOT / "data") -> List[Path]:
    """Merge matching Level 1 products into all-years datasets.

    Seasonal products retain observed ``site_id`` values. This is the first
    point where ``canonical_site_id`` is added, preserving the provenance of a
    renamed site while supplying a stable cross-season identity.
    """
    years = sorted(set(years))
    if len(years) < 2:
        raise ValueError("at least two distinct years are required for an all-years merge")

    files_by_name: Dict[str, List[tuple[int, Path]]] = {}
    for year in years:
        lvl1_dir = data_root / str(year) / "processed" / "lvl1"
        if not lvl1_dir.exists():
            raise FileNotFoundError(f"Level 1 directory does not exist for {year}: {lvl1_dir}")
        for path in lvl1_dir.glob("lvl1_*.nc"):
            if path.name.endswith("_all_years.nc"):
                continue
            files_by_name.setdefault(path.name, []).append((year, path))

    output_dir = data_root / "all_years" / "processed" / "lvl1"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = []

    for filename, season_paths in sorted(files_by_name.items()):
        datasets = []
        for year, path in sorted(season_paths):
            opened = xr.open_dataset(path)
            ds = opened.load()
            opened.close()
            if "year" not in ds.coords or ds["year"].dims != ("sensor_idx",):
                raise ValueError(
                    f"{path} has no sensor-indexed year coordinate; regenerate season {year} first"
                )
            if set(ds["year"].values.tolist()) != {year}:
                raise ValueError(f"{path} contains records outside field season {year}")
            canonical_ids = [canonical_site_id(site, year) for site in ds.site_id.values]
            ds = ds.assign_coords(canonical_site_id=("sensor_idx", canonical_ids))
            datasets.append(ds)

        combined = xr.concat(datasets, dim="sensor_idx", data_vars="all", coords="all", join="outer")
        combined = combined.assign_coords(sensor_idx=np.arange(combined.sizes["sensor_idx"]))
        combined.attrs.update(
            {
                "processing_step": "lvl1_all_years_combined",
                "source_years": ", ".join(
                    str(year) for year, _ in sorted(season_paths)
                ),
                "structure": "sensor_idx x datetime",
            }
        )
        category = filename.removeprefix("lvl1_").removesuffix(".nc")
        output_path = output_dir / f"lvl1_{category}_all_years.nc"
        combined = apply_product_metadata(
            combined,
            level="lvl1",
            product=category,
            source_years=tuple(year for year, _ in season_paths),
        )
        combined.to_netcdf(output_path)
        output_paths.append(output_path)

    if not output_paths:
        raise ValueError(f"No Level 1 datasets found for years {years}")
    return output_paths


def _parse_pace_header(lines: list[str]) -> dict:
    """Parse Pace logger file header to extract metadata and channel definitions."""
    metadata = {
        "serial_number": None,
        "label": None,
        "channels": {},
        "channel_order": [],
        "logging_start": None,
        "logging_stop": None,
        "log_interval": None,
        "battery_voltage": None,
    }

    for line in lines:
        line = line.strip()

        # Extract serial number
        if "Serial #:" in line:
            match = re.search(r"Serial #:\s*([A-Z0-9]+)", line)
            if match:
                metadata["serial_number"] = match.group(1)

        # Extract label (site name)
        elif line.startswith("Label:"):
            metadata["label"] = line.split(":", 1)[1].strip()

        # Extract logging start time
        elif line.startswith("Start Logging:"):
            metadata["logging_start"] = line.split(":", 1)[1].strip()

        # Extract logging stop time
        elif line.startswith("Stop Logging:"):
            metadata["logging_stop"] = line.split(":", 1)[1].strip()

        # Extract log interval
        elif line.startswith("Log Interval:"):
            metadata["log_interval"] = line.split(":", 1)[1].strip()

        # Extract battery voltage
        elif "Battery Voltage:" in line:
            match = re.search(r"Battery Voltage:\s*([\d.]+)V", line)
            if match:
                metadata["battery_voltage"] = float(match.group(1))

        # Parse channel definitions
        elif line.startswith(
            (
                "Ch1:",
                "Ch2:",
                "Ch3:",
                "Ch4:",
                "Ch5:",
                "Ch6:",
                "Ch7:",
                "Ch8:",
                "ChX:",
                "ChY:",
                "ChZ:",
            )
        ):
            ch_num = line.split(":")[0]
            # Look for the description line that follows
            continue

        elif line.startswith(
            (
                "Ch1 Descr:",
                "Ch2 Descr:",
                "Ch3 Descr:",
                "Ch4 Descr:",
                "Ch5 Descr:",
                "Ch6 Descr:",
                "Ch7 Descr:",
                "Ch8 Descr:",
                "ChX Descr:",
                "ChY Descr:",
                "ChZ Descr:",
            )
        ):
            ch_num = line.split(" ")[0]
            description = line.split(":", 1)[1].strip()
            if description:  # Only add channels with descriptions
                metadata["channels"][ch_num] = description

        # Stop parsing when we reach the data section
        elif "Date Time," in line and "Ch1," in line:
            # This is the header line, extract channel order
            metadata["channel_order"] = line.split(",")
            break

    return metadata


def _find_data_start(lines: list[str]) -> int:
    """Find the line number where actual CSV data starts."""
    for i, line in enumerate(lines):
        if line.strip().startswith("Data Set"):
            # Data starts on the next line
            return i + 1
    raise ValueError("Could not find 'Data Set' marker in file")


def _parse_pace_data(
    file_path: Path, data_start_idx: int, metadata: dict
) -> pd.DataFrame:
    """Parse the CSV data section of the Pace logger file."""

    # Find line number where "Data Set 1" appears
    with open(file_path, "r") as f:
        lines = f.readlines()

    data_set_line = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Data Set"):
            data_set_line = i
            break

    if data_set_line is None:
        raise ValueError("Could not find 'Data Set' line in file")

    # Header line is 2 lines before "Data Set 1"
    header_line_idx = data_set_line - 2

    # Read CSV starting from header line (without parsing dates yet)
    df = pd.read_csv(file_path, skiprows=header_line_idx)

    # Drop the "Data Set 1" line (row 0) and the checksum line (last row)
    if len(df) >= 2:
        df = df.drop([0, len(df) - 1]).reset_index(drop=True)
    elif len(df) >= 1:
        df = df.drop([0]).reset_index(drop=True)

    # Now parse dates on the cleaned dataframe and rename to datetime
    df["Date Time"] = pd.to_datetime(df["Date Time"])
    df = df.rename(columns={"Date Time": "datetime"})

    # Clean column names and map to CF-compliant names
    df = _clean_pace_column_names(df, metadata)

    return df


def _clean_pace_column_names(df: pd.DataFrame, metadata: dict) -> pd.DataFrame:
    """Clean column names and create CF-compliant variable names."""

    # Create mapping from original column names to clean names
    column_mapping = {}

    for col in df.columns:
        if col == "Date Time":
            column_mapping[col] = "datetime"
        elif "WindDir" in col:
            column_mapping[col] = "wind_direction"
        elif "P_kpa" in col:
            column_mapping[col] = "pressure"
        elif "RH" in col:
            column_mapping[col] = "relative_humidity"
        elif "WindSpd" in col and "Avg" in col:
            column_mapping[col] = "wind_speed_avg"
        elif "WindSpd" in col and "Peak" in col:
            column_mapping[col] = "wind_speed_max"
        elif "Ta_" in col and "cm_c" in col:
            # Extract height from temperature column name
            height_match = re.search(r"Ta_(\d+)cm_c", col)
            if height_match:
                height_cm = int(height_match.group(1))
                column_mapping[col] = f"temp_c_{height_cm}cm"
        else:
            # Keep other columns as-is but clean them
            column_mapping[col] = col.strip()

    # Rename columns
    df_clean = df.rename(columns=column_mapping)

    return df_clean


def _create_pace_dataset(
    df: pd.DataFrame, metadata: dict, file_path: Path, year: int
) -> xr.Dataset:
    """Create xarray dataset from parsed Pace logger data using sensor_idx structure."""

    # Extract site name from filename (not from file label which can be incorrect)
    site_name = file_path.stem

    # Convert datetime to UTC (assuming input is in local time)
    df["datetime_utc"] = (
        pd.to_datetime(df["datetime"])
        .dt.tz_localize(ZoneInfo("America/Anchorage"))
        .dt.tz_convert("UTC")
        .dt.tz_localize(None)
    )

    # Separate temperature and non-temperature variables
    temp_cols = [col for col in df.columns if col.startswith("temp_c_")]
    other_cols = [
        col for col in df.columns if not col.startswith(("temp_c_", "datetime"))
    ]

    # Create sensor entries for each height that has data
    sensors = []

    # Process temperature variables to create sensor entries
    for col in temp_cols:
        height_match = re.search(r"temp_c_(\d+)cm", col)
        if height_match:
            height_cm = int(height_match.group(1))
            height_m = height_cm / 100.0  # Convert to meters
            height_str = standardize_height(f"{height_m}m")

            sensors.append(
                {
                    "height": height_str,
                    "sensor_type": "pace",
                    "sensor_generation": "pace_logger",
                    "shielding": "unshielded",  # Pace sensors are unshielded
                    "variable_type": "temperature",
                    "data": df[col].values,
                }
            )

    # Process other meteorological variables
    for col in other_cols:
        if col == "pressure":
            height_str = "1m"  # Pressure at 1m
        else:
            height_str = "2m"  # Other meteorological variables at 2m

        sensors.append(
            {
                "height": height_str,
                "sensor_type": "pace",
                "sensor_generation": "pace_logger",
                "shielding": "unshielded",
                "variable_type": col,
                "data": df[col].values,
            }
        )

    # If no sensors found, create a minimal dataset
    if not sensors:
        return None

    # Create sensor_idx coordinate and sensor attribute coordinates
    n_sensors = len(sensors)
    sensor_indices = list(range(n_sensors))

    # Extract serial number from metadata
    serial_number = metadata.get("serial_number", "unknown")

    # Create data variables dictionary - each variable gets its own array
    data_vars = {}

    # Initialize coordinate arrays
    site_ids = []
    heights = []
    shielding_types = []
    sensor_types = []
    sensor_generations = []
    sensor_ids = []
    elevations = []
    latitudes = []
    longitudes = []

    # Create arrays for each variable type
    temp_array = np.full((n_sensors, len(df)), np.nan)

    # Create arrays for other variables
    other_var_arrays = {}
    for col in other_cols:
        other_var_arrays[col] = np.full((n_sensors, len(df)), np.nan)

    # Fill in data and coordinates
    for i, sensor in enumerate(sensors):
        # Fill coordinate arrays
        site_ids.append(site_name)
        heights.append(sensor["height"])
        shielding_types.append(sensor["shielding"])
        sensor_types.append(sensor["sensor_type"])
        sensor_generations.append(sensor["sensor_generation"])
        sensor_ids.append(
            f"{serial_number}_{sensor['height']}_{sensor['variable_type']}"
        )
        elevations.append(np.nan)  # To be filled from deployment metadata
        latitudes.append(np.nan)
        longitudes.append(np.nan)

        # Fill data arrays
        if sensor["variable_type"] == "temperature":
            temp_array[i, :] = sensor["data"]
        elif sensor["variable_type"] in other_var_arrays:
            other_var_arrays[sensor["variable_type"]][i, :] = sensor["data"]

    # Create data variables
    data_vars["temp_c"] = (["sensor_idx", "datetime_utc"], temp_array)

    for var_name, var_array in other_var_arrays.items():
        data_vars[var_name] = (["sensor_idx", "datetime_utc"], var_array)

    # Create coordinates
    coords = {
        "sensor_idx": sensor_indices,
        "datetime_utc": df["datetime_utc"],
        # Sensor attributes as coordinates
        "site_id": ("sensor_idx", site_ids),
        "year": ("sensor_idx", [year] * n_sensors),
        "height": ("sensor_idx", heights),
        "shielding": ("sensor_idx", shielding_types),
        "sensor_type": ("sensor_idx", sensor_types),
        "sensor_generation": ("sensor_idx", sensor_generations),
        "sensor_id": ("sensor_idx", sensor_ids),
        "elevation": ("sensor_idx", elevations),
        "latitude": ("sensor_idx", latitudes),
        "longitude": ("sensor_idx", longitudes),
    }

    # Create dataset
    ds = xr.Dataset(data_vars, coords=coords)
    ds["datetime_utc"].attrs.update(
        {
            "standard_name": "time",
            "axis": "T",
            "long_name": "UTC time",
            "timezone": "UTC",
        }
    )

    # Add CF-compliant attributes
    _add_pace_attributes(ds, metadata)

    # Add structure information
    ds.attrs["structure"] = "sensor_idx × datetime"
    ds.attrs["time_coordinate"] = "datetime_utc"
    ds.attrs["time_coverage_timezone"] = "UTC"
    ds.attrs["n_sensors"] = n_sensors

    return ds


def _add_pace_attributes(ds: xr.Dataset, metadata: dict):
    """Add CF-compliant attributes to variables and dataset."""

    # Dataset-level attributes (convert None to empty string for NetCDF compatibility)
    ds.attrs.update(
        {
            "sensor_type": "pace logger",
            "serial_number": metadata.get("serial_number") or "unknown",
            "site_label": metadata.get("label") or "",
            "logging_start": metadata.get("logging_start") or "",
            "logging_stop": metadata.get("logging_stop") or "",
            "log_interval": metadata.get("log_interval") or "",
            "battery_voltage": metadata.get("battery_voltage") or 0.0,
            "institution": "JIFLR Project",
            "source": "Pace Scientific XR5-SE-M data logger",
        }
    )

    for name, attributes in MEASUREMENT_ATTRS.items():
        if name in ds.data_vars:
            ds[name].attrs.update(attributes)
    if "wind_speed_max" in ds.data_vars:
        ds["wind_speed_max"].attrs["method"] = PACE_WIND_SPEED_MAX_METHOD


def apply_wind_direction_correction(
    ds: xr.Dataset, offsets: Dict[str, float], logger: logging.Logger
) -> xr.Dataset:
    """Apply azimuthal offset corrections to wind_direction variable.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with wind_direction(sensor_idx, datetime) and site_id coordinate
    offsets : dict
        Dictionary mapping site names (lowercase) to offset degrees
    logger : logging.Logger
        Logger for diagnostic output

    Returns
    -------
    xr.Dataset
        Dataset with corrected wind_direction values
    """
    # Check if wind_direction variable exists
    if "wind_direction" not in ds.data_vars:
        logger.debug("No wind_direction variable found in dataset, skipping correction")
        return ds

    # Create a copy to avoid modifying original
    ds = ds.copy(deep=True)

    # Iterate over each sensor_idx
    for idx in ds.sensor_idx.values:
        # Get wind direction data for this sensor
        wind_dir = ds.wind_direction.sel(sensor_idx=idx)

        # Skip this sensor_idx if all wind_direction values are NaN
        if wind_dir.isnull().all():
            continue

        # Get site_id for this sensor
        site_id = ds.site_id.sel(sensor_idx=idx).values.item()

        # Missing deployment offsets are zero: this still normalizes directions.
        site_id_lower = str(site_id).lower()
        offset = offsets.get(site_id_lower, 0.0)

        # Calculate initial statistics (skipna=True)
        initial_median = float(wind_dir.median(skipna=True).values)
        initial_min = float(wind_dir.min(skipna=True).values)
        initial_max = float(wind_dir.max(skipna=True).values)

        # Apply correction and wrap to [0, 360) range
        corrected_wind_dir = (wind_dir + offset) % 360

        # Calculate final statistics
        final_median = float(corrected_wind_dir.median(skipna=True).values)
        final_min = float(corrected_wind_dir.min(skipna=True).values)
        final_max = float(corrected_wind_dir.max(skipna=True).values)

        # Update dataset
        ds.wind_direction.loc[dict(sensor_idx=idx)] = corrected_wind_dir

        # Log the correction
        logger.info(
            f"Site {site_id} (sensor_idx={idx}): Applied wind direction correction\n"
            f"  Initial - median: {initial_median:.1f}°, min: {initial_min:.1f}°, max: {initial_max:.1f}°\n"
            f"  Offset: {offset:+.1f}°\n"
            f"  Final - median: {final_median:.1f}°, min: {final_min:.1f}°, max: {final_max:.1f}°"
        )

    # Update wind_direction attributes to document correction
    ds.wind_direction.attrs.update(
        {
            "correction_applied": "true",
            "correction_description": "Azimuthal offset correction applied based on deployment metadata",
            "correction_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
        }
    )

    return ds


def mask_wind_direction_by_speed(
    ds: xr.Dataset,
    wind_speed_threshold: float = 1.0,
    wind_speed_var: str = "wind_speed_avg",
    wind_direction_var: str = "wind_direction",
    sensor_types: tuple[str, ...] = ("pace",),
    logger: Optional[logging.Logger] = None,
) -> xr.Dataset:
    """Mask wind direction values where wind speed is below threshold.

    Wind direction measurements become unreliable at low wind speeds due to
    sensor limitations and flow variability. This function sets wind direction
    to NaN where wind speed is below a specified threshold (default: 1.0 m/s).

    This function applies to the requested meteorological sensor types. It handles the case where PACE data splits wind_direction and
    wind_speed_avg into separate sensor_idx entries (e.g., EM54054_2m_wind_direction
    and EM54054_2m_wind_speed_avg as separate sensors). It matches sensors by
    finding corresponding speed sensors for each direction sensor based on site_id.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with wind_direction and wind_speed_avg variables
    wind_speed_threshold : float, optional
        Minimum wind speed in m/s below which direction is masked (default: 1.0)
    wind_speed_var : str, optional
        Name of wind speed variable (default: 'wind_speed_avg')
    wind_direction_var : str, optional
        Name of wind direction variable (default: 'wind_direction')
    sensor_types : tuple of str, optional
        Sensor types whose directions should be masked. Defaults to ``("pace",)``
        for backwards-compatible PACE-only behavior.
    logger : logging.Logger, optional
        Logger for diagnostic output

    Returns
    -------
    xr.Dataset
        Dataset with masked wind_direction values

    Raises
    ------
    ValueError
        If wind_direction or wind_speed variables are not in dataset
    """
    # Validate that required variables exist
    if wind_direction_var not in ds.data_vars:
        raise ValueError(f"Variable '{wind_direction_var}' not found in dataset")

    if wind_speed_var not in ds.data_vars:
        raise ValueError(f"Variable '{wind_speed_var}' not found in dataset")

    # Create a copy to avoid modifying the caller's dataset.
    ds = ds.copy(deep=True)

    if "sensor_type" not in ds.coords:
        if logger:
            logger.warning(
                "Wind direction masking requires a sensor_type coordinate; skipping masking"
            )
        return ds

    requested_types = {sensor_type.casefold() for sensor_type in sensor_types}
    if not requested_types:
        raise ValueError("sensor_types must contain at least one sensor type")

    def is_requested_sensor(sensor_idx):
        sensor_type = str(ds["sensor_type"].sel(sensor_idx=sensor_idx).item())
        return sensor_type.casefold() in requested_types

    # Track total statistics across all sensors
    total_masked = 0
    total_points = 0
    n_sensors_masked = 0

    # Build a mapping of site_id -> sensor_idx for sensors with wind_speed data
    speed_sensor_map = {}
    for idx in ds.sensor_idx.values:
        if not is_requested_sensor(idx):
            continue
        wind_speed = ds[wind_speed_var].sel(sensor_idx=idx)
        if (~wind_speed.isnull()).sum() > 0:  # Has wind speed data
            site_id = str(ds.site_id.sel(sensor_idx=idx).values.item())
            speed_sensor_map[site_id] = idx

    # Iterate over each sensor_idx to find sensors with wind_direction
    for idx in ds.sensor_idx.values:
        if not is_requested_sensor(idx):
            continue
        wind_dir = ds[wind_direction_var].sel(sensor_idx=idx)

        # Skip this sensor if wind_direction is all NaN (no wind data)
        if wind_dir.isnull().all():
            continue

        # Get site_id for this direction sensor
        site_id = str(ds.site_id.sel(sensor_idx=idx).values.item())

        # Find corresponding speed sensor
        if site_id not in speed_sensor_map:
            if logger:
                logger.warning(
                    f"Site {site_id} (sensor_idx={idx}): Has wind_direction data but no corresponding "
                    f"wind_speed_avg sensor found, skipping masking"
                )
            continue

        # Get wind speed from the corresponding sensor
        speed_idx = speed_sensor_map[site_id]
        wind_speed = ds[wind_speed_var].sel(sensor_idx=speed_idx)

        # Create mask for low wind speeds (including NaN wind speeds)
        # Need to align wind_speed with wind_dir timeline (they should match but be safe)
        low_speed_mask = wind_speed < wind_speed_threshold

        # Count points before masking
        n_total = int((~wind_dir.isnull()).sum().values)
        n_to_mask = int((low_speed_mask & ~wind_dir.isnull()).sum().values)

        # Apply mask: set wind direction to NaN where wind speed is low
        masked_wind_dir = wind_dir.where(~low_speed_mask)

        # Update dataset
        ds[wind_direction_var].loc[dict(sensor_idx=idx)] = masked_wind_dir

        # Update statistics
        total_masked += n_to_mask
        total_points += n_total
        n_sensors_masked += 1

        # Log the masking for this sensor
        if logger:
            percentage = 100.0 * n_to_mask / n_total if n_total > 0 else 0.0
            logger.info(
                f"Site {site_id} (dir_idx={idx}, speed_idx={speed_idx}): "
                f"Masked {n_to_mask}/{n_total} points ({percentage:.1f}%) "
                f"where {wind_speed_var} < {wind_speed_threshold} m/s"
            )

    # Log overall statistics
    if logger:
        if total_points > 0:
            overall_percentage = 100.0 * total_masked / total_points
            logger.info(
                f"Overall wind direction masking: {total_masked}/{total_points} points "
                f"({overall_percentage:.1f}%) masked across {n_sensors_masked} sensors"
            )
        else:
            logger.info("No wind direction sensors found with valid data")

    if n_sensors_masked:
        # Only wind direction is modified. Wind speed remains unchanged.
        ds[wind_direction_var].attrs.update(
            {
                "wind_speed_masking_applied": "true",
                "wind_speed_threshold_m_s": wind_speed_threshold,
                "wind_speed_source_variable": wind_speed_var,
                "wind_speed_masking_sensor_type": ",".join(sorted(requested_types)),
                "masking_description": (
                    "Meteorological-station wind direction masked to NaN where "
                    f"{wind_speed_var} < {wind_speed_threshold} m/s"
                ),
                "masking_date": pd.Timestamp.now().strftime("%Y-%m-%d"),
            }
        )

    return ds


def _save_pace_netcdf(
    ds: xr.Dataset, dir_out: Path, metadata: dict, time_coord_name: str, site_name: str
):
    """Save xarray dataset as NetCDF file with appropriate filename."""

    # Get time range for filename
    time_data = ds[time_coord_name]
    start_time = pd.Timestamp(time_data.values[0])
    end_time = pd.Timestamp(time_data.values[-1])

    # Create filename using provided site_name (from filename, not metadata label)
    serial_number = metadata.get("serial_number", "unknown")

    filename = f"{serial_number}_{site_name}_{start_time.strftime('%Y%m%dT%H%M')}_{end_time.strftime('%Y%m%dT%H%M')}.nc"
    output_path = dir_out / filename

    # Save to NetCDF
    ds.to_netcdf(output_path)
    _logger.info(f"Saved: {output_path}")


def _process_single_pace_file(
    file_path: Path,
    dir_out: Path,
    convert_to_local_tz: bool,
    utc_offset_hours: float,
    manifest: Optional[DeploymentManifest] = None,
    year: Optional[int] = None,
):
    """Process a single Pace logger file and save as NetCDF."""

    # Read and parse the file
    with open(file_path, "r") as f:
        lines = f.readlines()

    # Extract metadata from header
    metadata = _parse_pace_header(lines)

    # Find the data section and parse CSV data
    data_start_idx = _find_data_start(lines)
    df = _parse_pace_data(file_path, data_start_idx, metadata)

    # Create xarray dataset
    if year is None:
        raise ValueError("year is required when processing a Pace logger file")
    ds = _create_pace_dataset(df, metadata, file_path, year)

    record = manifest.lookup(metadata["serial_number"]) if manifest else None
    if record is None:
        _logger.warning(
            "Pace logger %s is not in the deployment manifest; retaining filename-derived metadata.",
            metadata["serial_number"],
        )
    else:
        ds = apply_record_metadata(ds, record, include_configuration=False)
        ds = apply_record_deployment_mask(ds, record)

    offsets = manifest.wind_direction_offsets() if manifest else {}
    ds = apply_wind_direction_correction(ds, offsets, _logger)

    # Create an inspection plot while each logger's original channels are still
    # separate. Later pipeline stages merge Pace and pendant sensors by site.
    from jiflr.qc_plots import create_pace_qc_plot

    site_name = file_path.stem
    create_pace_qc_plot(ds, dir_out, filename_prefix=site_name, logger=_logger)

    # Preserve the legacy local-time export option after plotting. QC plots
    # consistently use the raw UTC timestamps used by the pipeline.
    if convert_to_local_tz:
        ds = replace_utc_datetime_coord(
            ds, utc_offset_hours=utc_offset_hours, add_cf_attributes=True
        )
        time_coord_name = "datetime"
    else:
        time_coord_name = "datetime_utc"

    # Generate output filename and save
    # Use the filename rather than the logger label, which can be incorrect.
    _save_pace_netcdf(ds, dir_out, metadata, time_coord_name, site_name)


def merge_sites(
    site_datasets: dict[str, xr.Dataset],
    target_site_id: str,
    join: str = "inner",
) -> xr.Dataset:
    """
    Merge multiple colocated site datasets into a single site.

    Sensors at the same height are averaged. Sensors at unique heights
    are preserved as-is.

    Parameters
    ----------
    site_datasets : dict[str, xr.Dataset]
        Mapping of source labels to datasets, such as logger serials for
        colocated sensors at one site.
    target_site_id : str
        Site ID for merged result, e.g. "G03"
    join : str
        "inner" (default): only overlapping time period
        "outer": union of all time periods (NaN-filled gaps)

    Returns
    -------
    xr.Dataset
        Merged dataset with sensor_idx structure
    """
    if not site_datasets:
        raise ValueError("site_datasets cannot be empty")

    if len(site_datasets) < 2:
        raise ValueError("merge_sites requires at least 2 datasets to merge")

    source_sites = list(site_datasets.keys())
    _logger.info(f"Merging sites: {', '.join(source_sites)} -> {target_site_id}")

    # Collect all sensors with their source site and metadata
    all_sensors = []
    for source_site, ds in site_datasets.items():
        if "sensor_idx" not in ds.dims:
            raise ValueError(
                f"Dataset for {source_site} does not have sensor_idx structure"
            )

        for i in range(len(ds.sensor_idx)):
            sensor_ds = ds.isel(sensor_idx=i)
            height = (
                str(sensor_ds.height.values)
                if "height" in sensor_ds.coords
                else "unknown"
            )
            sensor_id = (
                str(sensor_ds.sensor_id.values)
                if "sensor_id" in sensor_ds.coords
                else f"unknown_{i}"
            )

            all_sensors.append(
                {
                    "source_site": source_site,
                    "height": height,
                    "sensor_id": sensor_id,
                    "dataset": sensor_ds,
                }
            )

    # Group sensors by height
    sensors_by_height = {}
    for sensor in all_sensors:
        height = sensor["height"]
        if height not in sensors_by_height:
            sensors_by_height[height] = []
        sensors_by_height[height].append(sensor)

    # Process each height group
    processed_datasets = []

    for height, sensors in sensors_by_height.items():
        if len(sensors) == 1:
            # Single sensor at this height - preserve as-is, update site_id
            _logger.info(
                f"  Preserving single sensor at height {height} from {sensors[0]['source_site']}"
            )
            sensor_ds = sensors[0]["dataset"]

            # Create dataset with updated site_id
            processed_ds = _create_single_sensor_dataset(sensor_ds, target_site_id)
            processed_datasets.append(processed_ds)
        else:
            # Multiple sensors at this height - average them
            _logger.info(f"  Averaging {len(sensors)} sensors at height {height}:")

            # Log statistics for each sensor being averaged
            sensor_stats = []
            for sensor in sensors:
                sensor_ds = sensor["dataset"]
                temp_values = sensor_ds["temp_c"].values
                valid_mask = ~np.isnan(temp_values)
                valid_temps = temp_values[valid_mask]

                if len(valid_temps) > 0:
                    stats = {
                        "sensor_id": sensor["sensor_id"],
                        "source_site": sensor["source_site"],
                        "mean": np.mean(valid_temps),
                        "std": np.std(valid_temps),
                        "n": len(valid_temps),
                    }
                    sensor_stats.append(stats)
                    _logger.info(
                        f"    {sensor['source_site']} sensor {sensor['sensor_id']}: "
                        f"mean={stats['mean']:.2f}C, std={stats['std']:.2f}C, n={stats['n']}"
                    )

            # Calculate correlation if we have exactly 2 sensors
            if len(sensors) == 2:
                _log_sensor_correlation(sensors)

            # Average the sensors
            averaged_ds = _average_sensors_at_height(
                sensors, target_site_id, height, join
            )
            processed_datasets.append(averaged_ds)

    # Concatenate all processed datasets along sensor_idx
    if not processed_datasets:
        raise ValueError("No valid data after processing")

    merged_ds = xr.concat(
        processed_datasets,
        dim="sensor_idx",
        data_vars="all",
        coords="all",
        join="outer",
    )

    # Fix sensor_idx to be sequential
    new_sensor_idx = list(range(len(merged_ds.sensor_idx)))
    merged_ds = merged_ds.assign_coords(sensor_idx=new_sensor_idx)

    # Update global attributes
    merged_ds.attrs.update(
        {
            "site_name": target_site_id,
            "merged_from": ", ".join(source_sites),
            "merge_join": join,
            "n_sensors": len(merged_ds.sensor_idx),
            "structure": "sensor_idx x datetime_utc",
        }
    )

    _logger.info(f"  Merged result: {len(merged_ds.sensor_idx)} sensors")

    return merged_ds


_COLOCATED_METADATA_COORDINATES = ("year", "site_type", "processing_group")


def _colocated_sensor_metadata(sensor_ds: xr.DataArray) -> dict[str, object]:
    """Return metadata that must survive a colocated-site merge."""
    missing = [
        coordinate
        for coordinate in _COLOCATED_METADATA_COORDINATES
        if coordinate not in sensor_ds.coords
    ]
    if missing:
        raise ValueError(
            "Colocated-site merge requires source sensor metadata coordinates: "
            f"{missing}"
        )
    return {
        coordinate: sensor_ds[coordinate].item()
        for coordinate in _COLOCATED_METADATA_COORDINATES
    }


def _shared_colocated_sensor_metadata(sensors: list[dict]) -> dict[str, object]:
    """Return shared metadata and reject incompatible colocated sensors."""
    metadata = _colocated_sensor_metadata(sensors[0]["dataset"])
    for sensor in sensors[1:]:
        candidate = _colocated_sensor_metadata(sensor["dataset"])
        mismatched = [
            coordinate
            for coordinate in _COLOCATED_METADATA_COORDINATES
            if candidate[coordinate] != metadata[coordinate]
        ]
        if mismatched:
            raise ValueError(
                "Cannot average colocated sensors with conflicting metadata "
                f"coordinates: {mismatched}"
            )
    return metadata


def _create_single_sensor_dataset(
    sensor_ds: xr.DataArray, target_site_id: str
) -> xr.Dataset:
    """
    Create a dataset from a single sensor with updated site_id.

    Parameters
    ----------
    sensor_ds : xr.DataArray
        Single sensor dataset (already indexed by sensor_idx)
    target_site_id : str
        New site ID to assign

    Returns
    -------
    xr.Dataset
        Dataset with sensor_idx dimension restored and site_id updated
    """
    if "datetime_utc" not in sensor_ds.coords:
        raise ValueError("Sensor dataset does not have a datetime_utc coordinate")
    datetime_coord = "datetime_utc"
    metadata = _colocated_sensor_metadata(sensor_ds)

    # Build data variables
    data_vars = {}
    if "temp_c" in sensor_ds:
        data_vars["temp_c"] = (
            ["sensor_idx", datetime_coord],
            sensor_ds["temp_c"].values.reshape(1, -1),
        )
    if "intensity_lux" in sensor_ds:
        data_vars["intensity_lux"] = (
            ["sensor_idx", datetime_coord],
            sensor_ds["intensity_lux"].values.reshape(1, -1),
        )

    # Build coordinates
    coords = {
        "sensor_idx": [0],
        datetime_coord: sensor_ds[datetime_coord].values,
        "site_id": ("sensor_idx", [target_site_id]),
        "height": (
            "sensor_idx",
            [str(sensor_ds.height.values) if "height" in sensor_ds.coords else ""],
        ),
        "shielding": (
            "sensor_idx",
            [
                str(sensor_ds.shielding.values)
                if "shielding" in sensor_ds.coords
                else ""
            ],
        ),
        "sensor_type": (
            "sensor_idx",
            [
                str(sensor_ds.sensor_type.values)
                if "sensor_type" in sensor_ds.coords
                else ""
            ],
        ),
        "sensor_id": (
            "sensor_idx",
            [
                str(sensor_ds.sensor_id.values)
                if "sensor_id" in sensor_ds.coords
                else ""
            ],
        ),
        "sensor_generation": (
            "sensor_idx",
            [
                str(sensor_ds.sensor_generation.values)
                if "sensor_generation" in sensor_ds.coords
                else ""
            ],
        ),
        "elevation": (
            "sensor_idx",
            [
                float(sensor_ds.elevation.values)
                if "elevation" in sensor_ds.coords
                else np.nan
            ],
        ),
        "latitude": (
            "sensor_idx",
            [
                float(sensor_ds.latitude.values)
                if "latitude" in sensor_ds.coords
                else np.nan
            ],
        ),
        "longitude": (
            "sensor_idx",
            [
                float(sensor_ds.longitude.values)
                if "longitude" in sensor_ds.coords
                else np.nan
            ],
        ),
    }
    coords.update(
        {
            coordinate: ("sensor_idx", [value])
            for coordinate, value in metadata.items()
        }
    )

    return xr.Dataset(data_vars, coords=coords)


RM_YOUNG_COLUMNS = {
    "WindSpeed_ms_WVc(1)": ("wind_speed_avg", "m s-1", "Mean horizontal wind speed"),
    "WindSpeed_ms_WVc(2)": ("wind_direction", "degrees", "Unit-vector mean wind direction"),
    "WindSpeed_ms_WVc(3)": ("wind_direction_std", "degrees", "Standard deviation of wind direction"),
    "WindSpeed_ms_Max": ("wind_speed_max", "m s-1", "Maximum wind speed during output interval"),
    "WindSpeed_ms_Std": ("wind_speed_std", "m s-1", "Standard deviation of wind speed"),
    "Temperature_C_Avg": ("temp_c", "degrees_Celsius", "Air temperature"),
    "Temperature_C_Std": ("temp_c_std", "degrees_Celsius", "Standard deviation of air temperature"),
    "RelHumidity_pct_Avg": ("relative_humidity", "%", "Relative humidity"),
    "RelHumidity_pct_Std": ("relative_humidity_std", "%", "Standard deviation of relative humidity"),
    "Pressure_hPa_Avg": ("pressure", "hPa", "Atmospheric pressure"),
    "Pressure_hPa_Std": ("pressure_std", "hPa", "Standard deviation of atmospheric pressure"),
    "RainTips": ("rain_tips", "count", "Rain-gauge tips during output interval"),
    "Rainfall_mm": ("rainfall_mm", "mm", "Rainfall during output interval"),
    "LoggerTemp_C_Avg": ("logger_temp_c", "degrees_Celsius", "Datalogger temperature"),
    "BattV": ("battery_voltage", "V", "Datalogger battery voltage"),
    "WindStatusCode": ("wind_status_code", "1", "RM Young wind status code"),
    "StatusCode": ("status_code", "1", "Datalogger status code"),
}


def clean_rmyoung_loggers(
    raw_dir: Path,
    dir_out: Path,
    deployment_manifest_path: Path,
    year: int,
) -> list[Path]:
    """Convert Campbell TOA5 R. M. Young Weather tables to intermediate NetCDF.

    Raw timestamps are Alaska local time and are converted to timezone-naive
    UTC in ``datetime_utc``.  Each Weather table must identify a CR350 serial
    that appears exactly once in the deployment manifest.
    """
    raw_dir = Path(raw_dir)
    dir_out = Path(dir_out)
    manifest = load_deployment_manifest(deployment_manifest_path)
    weather_files = sorted(raw_dir.glob("*/*_Weather.dat"))
    if not weather_files:
        raise FileNotFoundError(f"No RM Young Weather tables found below {raw_dir}")
    dir_out.mkdir(parents=True, exist_ok=True)

    outputs = []
    for weather_file in weather_files:
        header = pd.read_csv(weather_file, header=None, nrows=1).iloc[0]
        if len(header) < 4 or str(header.iloc[0]) != "TOA5":
            raise ValueError(f"{weather_file} is not a Campbell TOA5 table")
        logger_serial = str(header.iloc[3]).strip()
        record = manifest.lookup(logger_serial)
        if record is None:
            raise ValueError(f"No deployment-manifest record for RM Young logger {logger_serial}")
        if record.instrument_type != "rmyoung_logger":
            raise ValueError(
                f"Manifest serial {logger_serial} has instrument_type {record.instrument_type!r}, "
                "expected 'rmyoung_logger'"
            )

        frame = pd.read_csv(weather_file, skiprows=[0, 2, 3], na_values=["NAN"])
        required = {"TIMESTAMP", "WindSpeed_ms_TMx", *RM_YOUNG_COLUMNS}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{weather_file} is missing expected columns: {missing}")
        local_time = pd.to_datetime(frame["TIMESTAMP"], errors="raise")
        if local_time.dt.tz is not None:
            raise ValueError(f"RM Young timestamps must be timezone-naive local timestamps: {weather_file}")
        if not local_time.is_monotonic_increasing or local_time.duplicated().any():
            raise ValueError(f"RM Young timestamps must be strictly increasing and unique: {weather_file}")
        datetime_utc = local_time.dt.tz_localize(ZoneInfo("America/Anchorage")).dt.tz_convert("UTC").dt.tz_localize(None)

        sensors = list(RM_YOUNG_COLUMNS.values())
        n_sensors = len(sensors)
        n_times = len(frame)
        data_vars = {}
        for index, (raw_name, (name, units, long_name)) in enumerate(RM_YOUNG_COLUMNS.items()):
            values = pd.to_numeric(frame[raw_name], errors="raise").to_numpy(dtype=float)
            channel = np.full((n_sensors, n_times), np.nan)
            channel[index] = values
            attrs = {"units": units, "long_name": long_name}
            if name == "wind_speed_max":
                attrs["method"] = RM_YOUNG_WIND_SPEED_MAX_METHOD
            data_vars[name] = (("sensor_idx", "datetime_utc"), channel, attrs)
        max_time_local = pd.to_datetime(frame["WindSpeed_ms_TMx"], errors="raise")
        data_vars["wind_speed_max_time"] = (
            ("datetime_utc",),
            max_time_local.dt.tz_localize(ZoneInfo("America/Anchorage")).dt.tz_convert("UTC").dt.tz_localize(None).to_numpy(),
            {"long_name": "Timestamp of maximum wind speed during output interval", "timezone": "UTC"},
        )
        names = [name for name, _, _ in sensors]
        ds = xr.Dataset(
            data_vars,
            coords={
                "sensor_idx": np.arange(n_sensors),
                "datetime_utc": datetime_utc,
                "site_id": ("sensor_idx", [record.site_id] * n_sensors),
                "year": ("sensor_idx", [year] * n_sensors),
                "height": ("sensor_idx", [record.height] * n_sensors),
                "shielding": ("sensor_idx", [record.shielding or "unknown"] * n_sensors),
                "sensor_type": ("sensor_idx", ["rmyoung"] * n_sensors),
                "sensor_generation": ("sensor_idx", ["rmyoung_logger"] * n_sensors),
                "sensor_id": ("sensor_idx", [f"{logger_serial}_{name}" for name in names]),
            },
            attrs={
                "sensor_type": "R. M. Young ResponseONE-Pro weather transmitter",
                "logger_serial": logger_serial,
                "source": str(weather_file),
                "raw_timestamp_timezone": "America/Anchorage",
                "time_coordinate": "datetime_utc",
                "time_coverage_timezone": "UTC",
                "structure": "sensor_idx x datetime_utc",
            },
        )
        ds = apply_record_metadata(ds, record, include_configuration=False)
        ds = apply_record_deployment_mask(ds, record)
        ds["datetime_utc"].attrs.update({"standard_name": "time", "axis": "T", "timezone": "UTC"})
        output_path = dir_out / f"{logger_serial}_{record.site_id}.nc"
        ds.to_netcdf(output_path)
        from jiflr.qc_plots import create_pace_qc_plot, create_rmyoung_wind_qc_plot

        create_pace_qc_plot(ds, dir_out, record.site_id, logger_name="R. M. Young")
        create_rmyoung_wind_qc_plot(ds, dir_out, record.site_id)
        outputs.append(output_path)
    return outputs


def _log_sensor_correlation(sensors: list) -> None:
    """Log correlation coefficient between two sensors."""
    ds1 = sensors[0]["dataset"]
    ds2 = sensors[1]["dataset"]

    if "datetime_utc" not in ds1.coords or "datetime_utc" not in ds2.coords:
        raise ValueError("Sensor correlation requires datetime_utc coordinates")
    datetime_coord = "datetime_utc"

    # Align time arrays
    common_times = np.intersect1d(
        ds1[datetime_coord].values, ds2[datetime_coord].values
    )

    if len(common_times) > 10:
        # Get temperature values at common times
        temp1 = ds1["temp_c"].sel({datetime_coord: common_times}).values
        temp2 = ds2["temp_c"].sel({datetime_coord: common_times}).values

        # Remove NaN pairs
        valid_mask = ~(np.isnan(temp1) | np.isnan(temp2))
        temp1_valid = temp1[valid_mask]
        temp2_valid = temp2[valid_mask]

        if len(temp1_valid) > 10:
            correlation = np.corrcoef(temp1_valid, temp2_valid)[0, 1]
            _logger.info(f"    Correlation coefficient: {correlation:.3f}")


def _average_sensors_at_height(
    sensors: list,
    target_site_id: str,
    height: str,
    join: str,
) -> xr.Dataset:
    """
    Average multiple sensors at the same height into a single sensor.

    Parameters
    ----------
    sensors : list
        List of sensor dictionaries with 'dataset' keys
    target_site_id : str
        Site ID for the result
    height : str
        Height of the sensors
    join : str
        "inner" or "outer" for time alignment

    Returns
    -------
    xr.Dataset
        Averaged sensor as dataset with sensor_idx structure
    """
    first_ds = sensors[0]["dataset"]
    if "datetime_utc" not in first_ds.coords:
        raise ValueError("Sensor averaging requires datetime_utc coordinates")
    datetime_coord = "datetime_utc"

    # Collect all time arrays
    all_times = [s["dataset"][datetime_coord].values for s in sensors]

    # Determine common time range based on join type
    if join == "inner":
        # Find intersection of all time arrays
        common_times = all_times[0]
        for times in all_times[1:]:
            common_times = np.intersect1d(common_times, times)
        target_times = np.sort(common_times)
    else:  # outer
        # Find union of all time arrays
        target_times = np.unique(np.concatenate(all_times))
        target_times = np.sort(target_times)

    if len(target_times) == 0:
        raise ValueError(f"No overlapping time points for sensors at height {height}")

    # Collect aligned temperature and light data
    temp_arrays = []
    light_arrays = []

    for sensor in sensors:
        sensor_ds = sensor["dataset"]

        # Create aligned arrays
        temp_aligned = np.full(len(target_times), np.nan)
        light_aligned = (
            np.full(len(target_times), np.nan) if "intensity_lux" in sensor_ds else None
        )

        # Get sensor times and find indices in target times
        sensor_times = sensor_ds[datetime_coord].values

        for i, t in enumerate(target_times):
            idx = np.where(sensor_times == t)[0]
            if len(idx) > 0:
                temp_aligned[i] = sensor_ds["temp_c"].values[idx[0]]
                if light_aligned is not None and "intensity_lux" in sensor_ds:
                    light_aligned[i] = sensor_ds["intensity_lux"].values[idx[0]]

        temp_arrays.append(temp_aligned)
        if light_aligned is not None:
            light_arrays.append(light_aligned)

    # Average the arrays (nanmean handles NaN values)
    temp_averaged = np.nanmean(np.array(temp_arrays), axis=0)
    light_averaged = (
        np.nanmean(np.array(light_arrays), axis=0) if light_arrays else None
    )

    # Build the output dataset
    data_vars = {
        "temp_c": (["sensor_idx", datetime_coord], temp_averaged.reshape(1, -1)),
    }
    if light_averaged is not None:
        data_vars["intensity_lux"] = (
            ["sensor_idx", datetime_coord],
            light_averaged.reshape(1, -1),
        )

    # Create merged sensor_id from all contributing sensors
    merged_sensor_ids = "+".join([s["sensor_id"] for s in sensors])

    # Get metadata from first sensor (they should be similar)
    first_sensor_ds = sensors[0]["dataset"]
    metadata = _shared_colocated_sensor_metadata(sensors)

    coords = {
        "sensor_idx": [0],
        datetime_coord: target_times,
        "site_id": ("sensor_idx", [target_site_id]),
        "height": ("sensor_idx", [height]),
        "shielding": (
            "sensor_idx",
            [
                str(first_sensor_ds.shielding.values)
                if "shielding" in first_sensor_ds.coords
                else ""
            ],
        ),
        "sensor_type": (
            "sensor_idx",
            [
                str(first_sensor_ds.sensor_type.values)
                if "sensor_type" in first_sensor_ds.coords
                else ""
            ],
        ),
        "sensor_id": ("sensor_idx", [merged_sensor_ids]),
        "sensor_generation": (
            "sensor_idx",
            [
                str(first_sensor_ds.sensor_generation.values)
                if "sensor_generation" in first_sensor_ds.coords
                else ""
            ],
        ),
        "elevation": (
            "sensor_idx",
            [
                float(first_sensor_ds.elevation.values)
                if "elevation" in first_sensor_ds.coords
                else np.nan
            ],
        ),
        "latitude": (
            "sensor_idx",
            [
                float(first_sensor_ds.latitude.values)
                if "latitude" in first_sensor_ds.coords
                else np.nan
            ],
        ),
        "longitude": (
            "sensor_idx",
            [
                float(first_sensor_ds.longitude.values)
                if "longitude" in first_sensor_ds.coords
                else np.nan
            ],
        ),
    }
    coords.update(
        {
            coordinate: ("sensor_idx", [value])
            for coordinate, value in metadata.items()
        }
    )

    return xr.Dataset(data_vars, coords=coords)
