# -*- coding: utf-8 -*-
"""
data.py

High-level data access, analysis helpers, and utilities for JIFLR sensor data.

Author: drotto
Created: 6/13/24 @ 16:39
Project: jif_lapse_rate
"""

from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
from tqdm import tqdm
import warnings

from jiflr import ROOT
from jiflr.utils import get_deployment_periods, filter_by_deployment_periods


# Site groupings for analysis
SITE_GROUPS = {
    "all_except_B": [
        "A01",
        "A02",
        "A03",
        "A04",
        "A05",
        "A06",
        "A07",
        "A08",
        "A10",
        "C02",
        "D01",
        "D02",
        "D04",
        "E01",
        "E03",
        "E04",
        "F03",
        "F05",
        "F06",
        "G01",
        "G02",
        "G03",
        "G04",
    ],
    "D_and_E": ["D01", "D02", "D04", "E01", "E03", "E04"],
    "A_and_intensive": [
        "A01",
        "A02",
        "A03",
        "A04",
        "A05",
        "A06",
        "A07",
        "A08",
        "A10",
        "22038776",
        "22038777",
        "22038778",
        "22038779",
        "22038781",
        "22133649",
        "22133654",
        "22133658",
        "22133662",
    ],
    "G_and_A02": ["G01", "G02", "G03", "G04", "A02"],
    "F_and_Windward1": ["F03", "F05", "F06", "Windward1"],
}


def select_sensors(ds: xr.Dataset, **coords) -> xr.Dataset:
    """
    Filter dataset to sensors matching coordinate criteria.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset with sensor_idx dimension
    **coords : keyword arguments
        Coordinate filters (e.g., site_id='A02', height=['1m', '2m'])

    Returns
    -------
    xr.Dataset
        Dataset with subset of sensors (still has sensor_idx dimension)

    Examples
    --------
    Get all sensors at a specific site:
    >>> site_data = select_sensors(ds, site_id='A02')

    Get sensors at specific heights:
    >>> height_data = select_sensors(ds, height=['1m', '2m'])

    Get specific sensor configuration:
    >>> sensor = select_sensors(ds, site_id='A01', height='2m', shielding='shielded')
    """
    if not coords:
        return ds

    mask = True

    for coord_name, coord_values in coords.items():
        coord_data = ds.coords[coord_name]

        if isinstance(coord_values, (list, tuple)):
            coord_mask = coord_data.isin(coord_values)
        else:
            coord_mask = coord_data == coord_values

        mask = mask & coord_mask

    if hasattr(mask, "compute"):
        mask = mask.compute()

    return ds.isel(sensor_idx=mask)


def unstack_sensor_idx(
    ds: xr.Dataset,
    coords: Optional[List[str]] = None,
    fill_value: Optional[float] = None,
    sparse: Optional[bool] = False,
    squeeze: Optional[bool] = True,
) -> xr.Dataset:
    """
    Convert sensor_idx dimension to multi-index structure for convenient selection.

    Parameters
    ----------
    ds : xr.Dataset
        xarray Dataset with sensor_idx dimension
    coords : list of str, optional
        Coordinate names to use for multi-index (default: ['site_id', 'height', 'shielding'])
    fill_value : float, optional
        Value to use for missing combinations (default: NaN)
    sparse : bool, optional
        Whether to unstack to a sparse array (default: False)

    Returns
    -------
    xr.Dataset
        Dataset with separate dimensions for each coordinate

    Examples
    --------
    Unstack with default coordinates:
    >>> unstacked = unstack_sensor_idx(ds)
    >>> temp_data = unstacked.temp_c.sel(site_id='A01', height='2m', shielding='shielded')

    Unstack with custom coordinates:
    >>> unstacked = unstack_sensor_idx(ds, coords=['site_id', 'height'])
    >>> temp_data = unstacked.temp_c.sel(site_id='A01', height='2m')
    """
    if coords is None:
        coords = ["site_id", "height", "shielding"]

    ds_indexed = ds.set_index(sensor_idx=coords)
    ds_unstacked = ds_indexed.unstack(
        "sensor_idx", fill_value=fill_value, sparse=sparse
    )
    if squeeze:
        ds_unstacked = ds_unstacked.squeeze()
        # Only sortby coords that still exist
        coords = [c for c in coords if c in ds_unstacked.dims]

    return ds_unstacked.sortby(coords)


def load_netcdf_with_masking(
    nc_file: Path,
    csv_deployment_path: Path,
    use_csv_masking: bool = True,
    apply_mask_to_data: bool = False,
) -> Optional[Dict[str, Any]]:
    """
    Load a single NetCDF file with optional deployment masking.

    Parameters
    ----------
    nc_file : Path
        Path to NetCDF file
    csv_deployment_path : Path
        Path to deployment_periods.csv
    use_csv_masking : bool
        Whether to apply CSV-based deployment masking
    apply_mask_to_data : bool
        If True, return only deployed data (reduces dataset)
        If False, return full dataset with deployed_mask array

    Returns
    -------
    dict or None
        Dictionary containing:
        - 'dataset': xarray Dataset
        - 'site_name': str
        - 'sensor_height': str
        - 'sensor_id': str
        - 'file': str (filename)
        - 'deployed_mask': np.ndarray (if use_csv_masking=True and apply_mask_to_data=False)
        - 'times': np.ndarray (if apply_mask_to_data=True, only deployed times)
        - 'temps': np.ndarray (if apply_mask_to_data=True, only deployed temps)

    Returns None if file cannot be loaded or has insufficient data
    """
    try:
        ds = xr.open_dataset(nc_file)

        # Extract metadata from sensor_idx structure (REQUIRED)
        if "sensor_idx" not in ds.dims:
            raise ValueError(
                f"File {nc_file.name} does not have sensor_idx structure. "
                f"All intermediate data must be regenerated with the new structure."
            )

        if len(ds.sensor_idx) == 0:
            return None

        # Extract from coordinates (single sensor per file)
        sensor_idx = 0
        site_name = (
            ds.site_id.values[sensor_idx] if "site_id" in ds.coords else "Unknown"
        )
        sensor_height = (
            ds.height.values[sensor_idx] if "height" in ds.coords else "Unknown"
        )
        sensor_id = (
            ds.sensor_id.values[sensor_idx] if "sensor_id" in ds.coords else "Unknown"
        )

        # Get temperature data with sensor_idx structure: (sensor_idx, datetime)
        datetime_coord = "datetime" if "datetime" in ds.coords else "datetime_utc"
        temp_data = ds["temp_c"].isel(sensor_idx=0).dropna(datetime_coord)

        if len(temp_data) < 10:  # Require minimum data
            return None

        result = {
            "dataset": ds,
            "site_name": site_name,
            "sensor_height": sensor_height,
            "sensor_id": sensor_id,
            "file": nc_file.name,
        }

        # Apply deployment masking if requested
        if use_csv_masking:
            try:
                # Get deployment periods for this site
                deployment_periods_dict = get_deployment_periods(
                    site_name, csv_deployment_path
                )
                deployment_periods = deployment_periods_dict.get(site_name, [])

                if deployment_periods:
                    if apply_mask_to_data:
                        # Apply mask to entire dataset (all data variables)
                        ds_filtered = filter_by_deployment_periods(
                            ds, deployment_periods, return_mask=False
                        )
                        result["dataset"] = ds_filtered
                    else:
                        # Keep full dataset with mask
                        deployed_mask = filter_by_deployment_periods(
                            ds, deployment_periods, return_mask=True
                        )
                        result["deployed_mask"] = deployed_mask
                else:
                    # If no deployment periods found, assume all data is deployed
                    deployed_mask = np.ones(len(ds[datetime_coord]), dtype=bool)
                    if not apply_mask_to_data:
                        result["deployed_mask"] = deployed_mask
            except Exception as e:
                # If deployment period lookup fails, warn and assume all data is deployed
                warnings.warn(
                    f"Failed to get deployment periods for site {site_name}: {e}. Assuming all data is deployed."
                )
                deployed_mask = np.ones(len(ds[datetime_coord]), dtype=bool)
                if not apply_mask_to_data:
                    result["deployed_mask"] = deployed_mask

        return result

    except Exception as e:
        print(f"Error loading {nc_file}: {e}")
        return None


def load_all_pendant_data(
    processed_dir: Path,
    csv_deployment_path: Path,
    use_csv_masking: bool = True,
    required_heights: Optional[List[str]] = None,
    drop_events: bool = True,
    drop_light: bool = False,
    apply_mask_to_data: bool = False,
) -> Dict[str, Dict[str, Dict]]:
    """
    Load all NetCDF files organized by site and sensor height.

    Parameters
    ----------
    processed_dir : Path
        Directory containing processed NetCDF files
    csv_deployment_path : Path
        Path to deployment_periods.csv
    use_csv_masking : bool
        Whether to apply CSV-based deployment masking
    required_heights : list of str, optional
        If provided, only load sensors at these heights (e.g., ['1m', '2m'])
    drop_events : bool, optional
        Whether to drop the 'events' data variable from loaded datasets (default: True)
    drop_light : bool, optional
        Whether to drop the 'intensity_lux' data variable from loaded datasets (default: False)
    apply_mask_to_data : bool, optional
        If True, apply deployment masks during loading and return filtered datasets.
        If False, return full datasets with separate deployed_mask arrays (default: False)

    Returns
    -------
    dict
        Nested dictionary: {site_name: {height: sensor_info_dict}}

    Example
    -------
    >>> site_data = load_all_pendant_data(processed_dir, csv_path)
    >>> site_data['A01']['2m']['dataset']  # Access 2m sensor at A01
    """
    netcdf_files = list(processed_dir.glob("*.nc"))
    site_data = {}

    for nc_file in tqdm(netcdf_files, desc="Loading NetCDF files"):
        sensor_info = load_netcdf_with_masking(
            nc_file,
            csv_deployment_path,
            use_csv_masking,
            apply_mask_to_data=apply_mask_to_data,
        )

        if sensor_info is None:
            continue

        site_name = sensor_info["site_name"]
        sensor_height = sensor_info["sensor_height"]

        # Filter by required heights if specified
        if required_heights and sensor_height not in required_heights:
            continue

        # Create site key
        if site_name == "Unknown":
            site_key = f"Sensor_{sensor_info['sensor_id']}"
        else:
            site_key = site_name

        # Initialize nested structure
        if site_key not in site_data:
            site_data[site_key] = {}

        # Drop events variable if requested
        if drop_events and "dataset" in sensor_info:
            ds = sensor_info["dataset"]
            if "events" in ds.data_vars:
                sensor_info["dataset"] = ds.drop_vars("events")

        # Drop light variable if requested
        if drop_light and "dataset" in sensor_info:
            ds = sensor_info["dataset"]
            if "intensity_lux" in ds.data_vars:
                sensor_info["dataset"] = ds.drop_vars("intensity_lux")

        # Create combined key to distinguish sensors at same height with different shielding
        ds = sensor_info["dataset"]
        shielding = ds.shielding.values[0] if "shielding" in ds.coords else "unknown"
        combined_key = f"{sensor_height}_{shielding}" if sensor_height else shielding

        site_data[site_key][combined_key] = sensor_info

    return site_data


def extract_site_prefix(site_name: str) -> str | None:
    """
    Extract letter prefix from site name.

    Parameters
    ----------
    site_name : str
        Site name (e.g., 'A01', 'TKG4', 'B02')

    Returns
    -------
    str | None
        Letter prefix (e.g., 'A', 'TKG', 'B')
        None if no alphabetic prefix found
    """

    if not site_name or not site_name[0].isalpha():
        return None

    prefix = ""
    for char in site_name:
        if char.isalpha():
            prefix += char
        else:
            break

    return prefix


def group_by_site_name(netcdf_data: List[Dict]) -> Dict[str, List[Dict]]:
    """
    Group sensor data by site name.

    Parameters
    ----------
    netcdf_data : list of dict
        List of sensor info dictionaries from load_netcdf_with_masking

    Returns
    -------
    dict
        {site_name: [sensor_info_dicts]} mapping
    """
    site_data = {}

    for sensor_info in netcdf_data:
        site_name = sensor_info["site_name"]

        if site_name == "Unknown":
            site_key = f"Sensor_{sensor_info['sensor_id']}"
        else:
            site_key = site_name

        if site_key not in site_data:
            site_data[site_key] = []

        site_data[site_key].append(sensor_info)

    return site_data


def group_by_site_prefix(
    sites_dict: Dict[str, Any], site_elevations: Dict[str, float], min_sites: int = 2
) -> Dict[str, Dict]:
    """
    Group sites by their letter prefix (e.g., all 'A' sites together).

    Parameters
    ----------
    sites_dict : dict
        Dictionary with site names as keys
    site_elevations : dict
        {site_name: elevation} mapping
    min_sites : int
        Minimum number of sites required to form a group

    Returns
    -------
    dict
        {prefix: {'sites': [sorted_site_list], 'elevations': {site: elev}}}
        Sites are sorted by elevation (ascending)

    Example
    -------
    >>> groups = group_by_site_prefix(sites, elevations, min_sites=2)
    >>> groups['A']  # All sites starting with 'A'
    {'sites': ['A01', 'A03'], 'elevations': {'A01': 1200, 'A03': 1500}}
    """
    groups = {}

    for site_name in sites_dict.keys():
        prefix = extract_site_prefix(site_name)

        if prefix is None:
            continue

        if prefix not in groups:
            groups[prefix] = {"sites": [], "elevations": {}}

        groups[prefix]["sites"].append(site_name)
        groups[prefix]["elevations"][site_name] = site_elevations.get(site_name, 0)

    # Filter groups with minimum sites and sort by elevation
    filtered_groups = {}
    for prefix, group_info in groups.items():
        if len(group_info["sites"]) >= min_sites:
            # Sort sites by elevation
            sorted_sites = sorted(
                group_info["sites"], key=lambda x: group_info["elevations"][x]
            )
            group_info["sites"] = sorted_sites
            filtered_groups[prefix] = group_info

    return filtered_groups


# ============================================================================
# TIME ALIGNMENT FUNCTIONS
# ============================================================================


def find_overlapping_period(
    *datasets,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Timestamp]]:
    """
    Find overlapping time period across multiple xarray datasets.

    Parameters
    ----------
    *datasets : xarray.Dataset
        Variable number of xarray datasets with 'temp_c' variable

    Returns
    -------
    tuple of (start_time, end_time) or (None, None)
        pd.Timestamp for start and end of overlapping period
        Returns (None, None) if no overlap exists

    Example
    -------
    >>> start, end = find_overlapping_period(ds1, ds2, ds3)
    >>> if start and end:
    ...     print(f"Overlap: {start} to {end}")
    """
    if not datasets:
        return None, None

    start_times = []
    end_times = []

    for ds in datasets:
        try:
            temp_data = ds["temp_c"].dropna("datetime")
            start_times.append(temp_data.datetime.min().values)
            end_times.append(temp_data.datetime.max().values)
        except (KeyError, AttributeError):
            return None, None

    start_time = pd.Timestamp(max(start_times))
    end_time = pd.Timestamp(min(end_times))

    if start_time >= end_time:
        return None, None

    return start_time, end_time


def create_common_time_grid(
    start_time: pd.Timestamp, end_time: pd.Timestamp, freq: str = "10min"
) -> pd.DatetimeIndex:
    """
    Create a common time grid for interpolation.

    Parameters
    ----------
    start_time : pd.Timestamp
        Start of time range
    end_time : pd.Timestamp
        End of time range
    freq : str
        Pandas frequency string (e.g., '10min', '1H', '3H')

    Returns
    -------
    pd.DatetimeIndex
        Common time grid
    """
    return pd.date_range(start_time, end_time, freq=freq)


def interpolate_to_common_grid(
    datasets: List[xr.Dataset], common_times: pd.DatetimeIndex
) -> Tuple[pd.DatetimeIndex, List[np.ndarray]]:
    """
    Interpolate multiple datasets to a common time grid.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        Datasets to interpolate (must have 'temp_c' variable)
    common_times : pd.DatetimeIndex
        Common time grid

    Returns
    -------
    tuple of (valid_times, valid_temp_arrays)
        valid_times : pd.DatetimeIndex
            Times where all datasets have valid (non-NaN) data
        valid_temp_arrays : list of np.ndarray
            Temperature arrays for each dataset at valid times

    Example
    -------
    >>> datasets = [ds_1m, ds_2m]
    >>> times, [temp_1m, temp_2m] = interpolate_to_common_grid(datasets, common_times)
    """
    interp_arrays = []

    # Interpolate each dataset
    for ds in datasets:
        temp_data = ds["temp_c"].dropna("datetime")
        temp_interp = temp_data.interp(datetime=common_times, method="linear")
        interp_arrays.append(temp_interp.values)

    # Find valid mask (all datasets have non-NaN values)
    valid_mask = ~np.any(np.isnan(interp_arrays), axis=0)

    if not np.any(valid_mask):
        return pd.DatetimeIndex([]), []

    # Return only valid times and data
    valid_times = common_times[valid_mask]
    valid_arrays = [arr[valid_mask] for arr in interp_arrays]

    return valid_times, valid_arrays


def resample_to_hourly(
    times: np.ndarray, temp_arrays: List[np.ndarray], deployed_mask: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[List[np.ndarray]], Optional[np.ndarray]]:
    """
    Resample temperature data to 1-hour averages (deployed periods only).

    Parameters
    ----------
    times : np.ndarray
        Array of datetime values
    temp_arrays : list of np.ndarray
        List of temperature arrays to resample
    deployed_mask : np.ndarray of bool
        Boolean mask indicating deployed periods

    Returns
    -------
    tuple of (times_hourly, temp_arrays_hourly, deployed_mask_hourly)
        Returns (None, None, None) if insufficient data

    Example
    -------
    >>> times_hr, [temp1_hr, temp2_hr], mask_hr = resample_to_hourly(
    ...     times, [temp1, temp2], deployed_mask
    ... )
    """
    # Build DataFrame with all data
    data_dict = {"datetime": times, "deployed": deployed_mask}
    for i, arr in enumerate(temp_arrays):
        data_dict[f"temp_{i}"] = arr

    df = pd.DataFrame(data_dict)
    df.set_index("datetime", inplace=True)

    # Filter to deployed periods only
    df_deployed = df[df["deployed"]].drop(columns=["deployed"])

    if len(df_deployed) == 0:
        return None, None, None

    # Resample to hourly
    df_hourly = df_deployed.resample("1H").mean().dropna()

    if len(df_hourly) == 0:
        return None, None, None

    # Extract results
    times_hourly = df_hourly.index.values
    temp_arrays_hourly = [
        df_hourly[f"temp_{i}"].values for i in range(len(temp_arrays))
    ]
    deployed_mask_hourly = np.ones(len(times_hourly), dtype=bool)

    return times_hourly, temp_arrays_hourly, deployed_mask_hourly


# ============================================================================
# DEPLOYMENT MASKING FUNCTIONS
# ============================================================================


def apply_combined_deployment_mask(
    datasets: List[xr.Dataset],
    sensor_infos: List[Dict],
    times_valid: np.ndarray,
    temp_arrays: List[np.ndarray],
    csv_deployment_path: Path,
) -> np.ndarray:
    """
    Apply deployment masking to multiple sensors (all must be deployed).

    Parameters
    ----------
    datasets : list of xarray.Dataset
        Datasets for each sensor
    sensor_infos : list of dict
        Sensor info dicts (must have 'file' key for CSV matching)
    times_valid : np.ndarray
        Valid time points
    temp_arrays : list of np.ndarray
        Temperature arrays for each sensor
    csv_deployment_path : Path
        Path to deployment_periods.csv

    Returns
    -------
    np.ndarray of bool
        Combined deployment mask (True where ALL sensors are deployed)

    Example
    -------
    >>> mask = apply_combined_deployment_mask(
    ...     [ds1_1m, ds1_2m, ds2_1m, ds2_2m],
    ...     [info1_1m, info1_2m, info2_1m, info2_2m],
    ...     times, temps, csv_path
    ... )
    """
    deployed_masks = []

    for ds, sensor_info, temp_array in zip(datasets, sensor_infos, temp_arrays):
        site_name = ds.attrs.get("site_name", "Unknown")
        height = ds.attrs.get("sensor_height", "Unknown")

        try:
            # Get deployment periods for this site
            deployment_periods_dict = get_deployment_periods(
                site_name, csv_deployment_path
            )
            deployment_periods = deployment_periods_dict.get(site_name, [])

            if deployment_periods:
                # Create a temporary DataArray for masking
                temp_da = xr.DataArray(
                    temp_array,
                    coords={"datetime": pd.to_datetime(times_valid)},
                    dims=["datetime"],
                )
                deployed = filter_by_deployment_periods(
                    temp_da, deployment_periods, return_mask=True
                )
                deployed_masks.append(deployed)
            else:
                # If no deployment periods found, assume all deployed
                deployed_masks.append(np.ones(len(times_valid), dtype=bool))
        except Exception as e:
            # If deployment period lookup fails, warn and assume all deployed
            warnings.warn(
                f"Failed to get deployment periods for site {site_name}: {e}. Assuming all data is deployed."
            )
            deployed_masks.append(np.ones(len(times_valid), dtype=bool))

    # Combine masks: all sensors must be deployed
    if deployed_masks:
        combined_mask = np.all(deployed_masks, axis=0)
    else:
        combined_mask = np.ones(len(times_valid), dtype=bool)

    return combined_mask


def mask_site_data(
    site_data: Dict[str, Dict], csv_deployment_path: Path
) -> Dict[str, Dict]:
    """
    Apply deployment masking to all sensors in site_data dictionary.

    Adds 'deployed_mask' key to each sensor info dict.

    Parameters
    ----------
    site_data : dict
        Nested dict: {site_name: {height: sensor_info}}
    csv_deployment_path : Path
        Path to deployment_periods.csv

    Returns
    -------
    dict
        Same structure with 'deployed_mask' added to each sensor

    Example
    -------
    >>> site_data = load_all_pendant_data(processed_dir, csv_path, use_csv_masking=False)
    >>> site_data = mask_site_data(site_data, csv_path)
    >>> mask = site_data['A01']['2m']['deployed_mask']
    """
    for site_name, sensors in site_data.items():
        for height, sensor_info in sensors.items():
            if "dataset" not in sensor_info:
                continue

            ds = sensor_info["dataset"]
            temp_data = ds["temp_c"].dropna("datetime")

            site_name_attr = ds.attrs.get("site_name", "Unknown")

            try:
                # Get deployment periods for this site
                deployment_periods_dict = get_deployment_periods(
                    site_name_attr, csv_deployment_path
                )
                deployment_periods = deployment_periods_dict.get(site_name_attr, [])

                if deployment_periods:
                    deployed_mask = filter_by_deployment_periods(
                        temp_data, deployment_periods, return_mask=True
                    )
                    sensor_info["deployed_mask"] = deployed_mask
            except Exception as e:
                # If deployment period lookup fails, warn and skip masking
                warnings.warn(
                    f"Failed to get deployment periods for site {site_name_attr}: {e}. Skipping masking for this sensor."
                )

    return site_data


# ============================================================================
# TIMEZONE UTILITIES
# ============================================================================


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


# ============================================================================
# DATA LOADING AND MERGING
# ============================================================================


def load_and_merge_lvl0_data(
    lvl0_main_path: Optional[Path] = None,
    lvl0_intensive_path: Optional[Path] = None,
    time_slice: Optional[slice] = None,
    drop_conflicting_sites: bool = True,
) -> xr.Dataset:
    """
    Load and merge lvl0_main and lvl0_combined_intensive NetCDF files.

    This function loads both level 0 processed datasets and merges them into a single
    xarray Dataset with consistent dimensions. The main dataset contains regular
    monitoring sites with pendant sensors, while the intensive dataset contains
    fewer sites but with additional meteorological variables from pace loggers.

    Parameters
    ----------
    lvl0_main_path : Path, optional
        Path to lvl0_main.nc file. If None, uses default project path.
    lvl0_intensive_path : Path, optional
        Path to lvl0_combined_intensive.nc file. If None, uses default project path.
    time_slice : slice, optional
        Time slice to apply to both datasets for memory efficiency.
    drop_conflicting_sites : bool, optional
        If True, removes sites that appear in both datasets to avoid conflicts.
        Keeps the intensive version when conflicts occur (default: True).

    Returns
    -------
    xr.Dataset
        Merged dataset with harmonized dimensions:
        - site_id: All unique sites from both datasets
        - height: All unique heights from both datasets
        - datetime: Overlapping time period from both datasets
        - Variables from both datasets, with NaN for missing combinations

    Raises
    ------
    FileNotFoundError
        If either input file cannot be found
    ValueError
        If datasets cannot be merged due to incompatible structures

    Examples
    --------
    Load and merge with default paths:
    >>> merged_ds = load_and_merge_lvl0_data()

    Load specific time period:
    >>> merged_ds = load_and_merge_lvl0_data(time_slice=slice('2025-06-01', '2025-08-01'))

    Keep conflicting sites from both datasets:
    >>> merged_ds = load_and_merge_lvl0_data(drop_conflicting_sites=False)
    """

    # Set default paths if not provided
    if lvl0_main_path is None:
        lvl0_main_path = Path(ROOT) / "data/2025/processed/lvl0/lvl0_main.nc"
    if lvl0_intensive_path is None:
        lvl0_intensive_path = (
            Path(ROOT) / "data/2025/processed/lvl0/lvl0_combined_intensive.nc"
        )

    # Check that files exist
    if not lvl0_main_path.exists():
        raise FileNotFoundError(f"Main dataset not found: {lvl0_main_path}")
    if not lvl0_intensive_path.exists():
        raise FileNotFoundError(f"Intensive dataset not found: {lvl0_intensive_path}")

    # Load datasets
    print(f"Loading main dataset: {lvl0_main_path}")
    ds_main = xr.open_dataset(lvl0_main_path)

    print(f"Loading intensive dataset: {lvl0_intensive_path}")
    ds_intensive = xr.open_dataset(lvl0_intensive_path)

    # Apply time slice if provided
    if time_slice is not None:
        print(f"Applying time slice: {time_slice}")
        ds_main = ds_main.sel(datetime=time_slice)
        ds_intensive = ds_intensive.sel(datetime=time_slice)

    # Harmonize coordinate names and structures
    print("Harmonizing dataset structures...")

    # Rename 'shielded' coordinate to 'shielding' in intensive dataset for consistency
    if "shielded" in ds_intensive.coords:
        ds_intensive = ds_intensive.rename({"shielded": "shielding"})

    # Handle overlapping sites
    main_sites = set(ds_main.site_id.values)
    intensive_sites = set(ds_intensive.site_id.values)
    overlapping_sites = main_sites.intersection(intensive_sites)

    if overlapping_sites and drop_conflicting_sites:
        print(f"Removing overlapping sites from main dataset: {overlapping_sites}")
        # Keep only non-overlapping sites in main dataset
        non_overlapping_main_sites = [
            site for site in ds_main.site_id.values if site not in overlapping_sites
        ]
        if non_overlapping_main_sites:
            ds_main = ds_main.sel(site_id=non_overlapping_main_sites)
        else:
            # If all main sites overlap, create empty dataset with same structure
            ds_main = ds_main.isel(site_id=slice(0, 0))

    # With the new sensor_idx structure, merging is much simpler
    # Both datasets should now have dimensions: (sensor_idx, datetime)
    # Sensor attributes are stored as coordinates indexed by sensor_idx

    # The datasets can be concatenated directly along the sensor_idx dimension
    ds_main_modified = ds_main.copy()
    ds_intensive_modified = ds_intensive.copy()

    # Find common time period
    time_main = pd.to_datetime(ds_main_modified.datetime.values)
    time_intensive = pd.to_datetime(ds_intensive_modified.datetime.values)

    # Convert datetime coordinates to comparable format
    if hasattr(ds_main_modified.datetime, "values"):
        time_main_range = (time_main.min(), time_main.max())
    if hasattr(ds_intensive_modified.datetime, "values"):
        time_intensive_range = (time_intensive.min(), time_intensive.max())

    print(f"Main dataset time range: {time_main_range[0]} to {time_main_range[1]}")
    print(
        f"Intensive dataset time range: {time_intensive_range[0]} to {time_intensive_range[1]}"
    )

    # Find overlapping time period
    overlap_start = max(time_main_range[0], time_intensive_range[0])
    overlap_end = min(time_main_range[1], time_intensive_range[1])

    if overlap_start >= overlap_end:
        warnings.warn("No overlapping time period found between datasets")
        # Use full time range from both datasets
    else:
        print(f"Overlapping time period: {overlap_start} to {overlap_end}")

    # Merge the datasets using sensor_idx concatenation
    print("Merging datasets...")
    try:
        # With sensor_idx structure, simply concatenate along the sensor_idx dimension
        merged_ds = xr.concat(
            [ds_main_modified, ds_intensive_modified],
            dim="sensor_idx",
            data_vars="all",
            coords="all",
        )

        # Add metadata about the merge
        merged_ds.attrs.update(
            {
                "merged_from": "lvl0_main.nc and lvl0_combined_intensive.nc",
                "merge_timestamp": pd.Timestamp.now().isoformat(),
                "n_main_sensors": len(ds_main.sensor_idx)
                if len(ds_main.sensor_idx) > 0
                else 0,
                "n_intensive_sensors": len(ds_intensive.sensor_idx),
                "total_sensors": len(merged_ds.sensor_idx),
                "overlapping_sites_removed": list(overlapping_sites)
                if drop_conflicting_sites
                else [],
                "drop_conflicting_sites": drop_conflicting_sites,
                "structure": "sensor_idx × datetime",
            }
        )

        print(f"Successfully merged datasets:")
        print(
            f"  - Main dataset sensors: {len(ds_main.sensor_idx) if len(ds_main.sensor_idx) > 0 else 0}"
        )
        print(f"  - Intensive dataset sensors: {len(ds_intensive.sensor_idx)}")
        print(f"  - Total merged sensors: {len(merged_ds.sensor_idx)}")
        print(f"  - Data variables: {list(merged_ds.data_vars.keys())}")

        return merged_ds

    except Exception as e:
        raise ValueError(f"Failed to merge datasets: {e}")
