# -*- coding: utf-8 -*-
"""
data.py

Description.

Author: drotto
Created: 6/13/24 @ 16:39
Project: jif_lapse_rate
"""

from typing import Dict, List, Tuple, Optional, Any

import pandas as pd
import geopandas as gpd
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
from pathlib import Path
from tqdm import tqdm
import scipy as sci
import re
import warnings
from zoneinfo import ZoneInfo

from jiflr import ROOT
from jiflr.utils import get_deployment_periods, filter_by_deployment_periods

plt.style.use("default")


def clean_hobo_pendants(
    ps: list[Path] | Path,
    dir_out: Path,
    convert_to_local_tz: bool = False,
    utc_offset_hours: float = -9.0,
    data_inventory_path: Optional[Path] = None,
):
    """
    Reads data exported from HOBOware and HOBOconnect and outputs it as netcdf.

    Parameters
    ----------
    ps : list[Path] | Path
        Path or list of paths to CSV files exported from HOBOware/HOBOconnect
    dir_out : Path
        Output directory for NetCDF files
    convert_to_local_tz : bool, optional
        If True, convert from UTC storage to local timezone (default: False)
    utc_offset_hours : float, optional
        UTC offset in hours for local timezone conversion (default: -9.0 for AKST)
    data_inventory_path : Path, optional
        Path to data inventory Excel file containing shielding information
    """

    # make sure ps is always a list
    if isinstance(ps, list) is False:
        ps = [ps]

    # Load shielding information if data inventory provided
    shielding_info = {}
    if data_inventory_path and data_inventory_path.exists():
        try:
            import pandas as pd
            inventory_df = pd.read_excel(data_inventory_path)
            # Create mapping from sensor_id to shielding status
            if 'sensor_id' in inventory_df.columns and 'shielding' in inventory_df.columns:
                for _, row in inventory_df.iterrows():
                    sensor_id = str(row['sensor_id'])
                    shielding = row['shielding']
                    if pd.notna(shielding) and str(shielding).lower() == 'unshielded':
                        shielding_info[sensor_id] = 'unshielded'
                    # All other sensors default to 'shielded'
        except Exception as e:
            warnings.warn(f"Could not load data inventory: {e}")

    for p in tqdm(ps):
        # First read: extract metadata from plot title
        df_meta = pd.read_csv(p, nrows=1)

        # Extract site metadata from filename
        site_name = None
        sensor_height = None
        sensor_config = None

        filename_parts = p.name.split(" ")
        if len(filename_parts) >= 2:
            # Check if first part looks like a site name (e.g., A01, B03, C17)
            if re.match(r"^[A-Z]\d+$", filename_parts[0]) or re.match(
                r"^[A-Z]+\d*$", filename_parts[0]
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
            if "light" in col.lower() or f"#{sn}" in col:
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

        # Convert datetime with timezone if available
        df_clean["datetime"] = pd.to_datetime(
            df_clean["datetime"], format="%m/%d/%y %H:%M:%S"
        )

        # Localize to the timezone, convert to UTC, then remove timezone info
        df_clean["datetime_utc"] = (
            df_clean["datetime"]
            .dt.tz_localize(ZoneInfo(tz))
            .dt.tz_convert("UTC")
            .dt.tz_localize(None)  # Remove timezone info, keeping UTC time
        )

        # Convert to xarray Dataset
        ds = xr.Dataset.from_dataframe(df_clean.set_index("datetime_utc"))

        # Get shielding status for this sensor (default to 'shielded')
        shielding_status = shielding_info.get(sn, "shielded")

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
        }

        ds.coords["sensor_id"] = sn

        # Add site metadata as scalar coordinates
        if site_name:
            ds.coords["site_id"] = site_name
        if sensor_height:
            ds.coords["height"] = sensor_height
        elif sensor_config:
            # Use sensor_config as height if no explicit height (e.g., 'unshielded')
            ds.coords["height"] = sensor_config
        
        # Add shielding as a coordinate dimension
        ds.coords["shielding"] = shielding_status

        # Add attributes to variables
        for var in ds.data_vars:
            ds[var].attrs = attr_dict.copy()

        # Also add metadata to dataset attributes for easy access
        ds.attrs.update(attr_dict)

        # Convert to local timezone if requested
        if convert_to_local_tz:
            ds = replace_utc_datetime_coord(
                ds, utc_offset_hours=utc_offset_hours, add_cf_attributes=True
            )
            time_coord_name = "datetime"
        else:
            time_coord_name = "datetime_utc"

        # Output to netcdf
        start_time = (
            df_clean["datetime_utc"].iloc[0]
            if not convert_to_local_tz
            else ds[time_coord_name].values[0]
        )
        end_time = (
            df_clean["datetime_utc"].iloc[-1]
            if not convert_to_local_tz
            else ds[time_coord_name].values[-1]
        )

        # Format filename timestamps consistently
        if convert_to_local_tz:
            start_time = pd.Timestamp(start_time)
            end_time = pd.Timestamp(end_time)

        fname = f"{sn}_{start_time.strftime('%Y%m%dT%H%M')}_{end_time.strftime('%Y%m%dT%H%M')}.nc"
        pout = dir_out / fname
        ds.to_netcdf(pout)

    return None


def load_deployment_metadata(csv_path: Path) -> Dict[str, Dict[str, float]]:
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
        df = pd.read_csv(csv_path)

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
        print(f"Error loading deployment metadata: {e}")

    return metadata


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

        # Extract metadata
        site_name = ds.attrs.get("site_name", "Unknown")
        sensor_height = ds.attrs.get("sensor_height", "Unknown")
        sensor_id = ds.attrs.get("sensor_id", "Unknown")

        # Get temperature data
        temp_data = ds["temp_c"].dropna("datetime")

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
                    deployed_mask = np.ones(len(ds.datetime), dtype=bool)
                    if not apply_mask_to_data:
                        result["deployed_mask"] = deployed_mask
            except Exception as e:
                # If deployment period lookup fails, warn and assume all data is deployed
                warnings.warn(f"Failed to get deployment periods for site {site_name}: {e}. Assuming all data is deployed.")
                deployed_mask = np.ones(len(ds.datetime), dtype=bool)
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

        site_data[site_key][sensor_height] = sensor_info

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
                    coords={'datetime': pd.to_datetime(times_valid)}, 
                    dims=['datetime']
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
            warnings.warn(f"Failed to get deployment periods for site {site_name}: {e}. Assuming all data is deployed.")
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
                warnings.warn(f"Failed to get deployment periods for site {site_name_attr}: {e}. Skipping masking for this sensor.")

    return site_data


def read_hobo_pendant(p: Path):
    """
    Read a single HOBO pendant netcdf file
    Returns xarray Dataset with proper coordinates
    """
    return xr.open_dataset(p)


def read_pendant_dataset(ps: list[Path]):
    """
    Read multiple HOBO pendant netcdf files and concatenate them efficiently
    Uses xarray's open_mfdataset for automatic concatenation
    """
    if not ps:
        return None

    # Use xarray's built-in multi-file dataset loading
    mfds = xr.open_mfdataset(
        ps, combine="nested", concat_dim="sensor_id", decode_times=True
    )

    return mfds


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


def clean_pace_loggers(
    file_paths: list[Path] | Path,
    dir_out: Path,
    convert_to_local_tz: bool = False,
    utc_offset_hours: float = -9.0,
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
    
    # Ensure file_paths is always a list
    if isinstance(file_paths, Path):
        file_paths = [file_paths]
    
    for file_path in tqdm(file_paths, desc="Processing Pace logger files"):
        _process_single_pace_file(file_path, dir_out, convert_to_local_tz, utc_offset_hours)


def _parse_pace_header(lines: list[str]) -> dict:
    """Parse Pace logger file header to extract metadata and channel definitions."""
    metadata = {
        'serial_number': None,
        'label': None,
        'channels': {},
        'channel_order': [],
        'logging_start': None,
        'logging_stop': None,
        'log_interval': None,
        'battery_voltage': None
    }
    
    for line in lines:
        line = line.strip()
        
        # Extract serial number
        if "Serial #:" in line:
            match = re.search(r"Serial #:\s*([A-Z0-9]+)", line)
            if match:
                metadata['serial_number'] = match.group(1)
        
        # Extract label (site name)
        elif line.startswith("Label:"):
            metadata['label'] = line.split(":", 1)[1].strip()
        
        # Extract logging start time
        elif line.startswith("Start Logging:"):
            metadata['logging_start'] = line.split(":", 1)[1].strip()
        
        # Extract logging stop time  
        elif line.startswith("Stop Logging:"):
            metadata['logging_stop'] = line.split(":", 1)[1].strip()
        
        # Extract log interval
        elif line.startswith("Log Interval:"):
            metadata['log_interval'] = line.split(":", 1)[1].strip()
        
        # Extract battery voltage
        elif "Battery Voltage:" in line:
            match = re.search(r"Battery Voltage:\s*([\d.]+)V", line)
            if match:
                metadata['battery_voltage'] = float(match.group(1))
        
        # Parse channel definitions
        elif line.startswith(("Ch1:", "Ch2:", "Ch3:", "Ch4:", "Ch5:", "Ch6:", "Ch7:", "Ch8:", "ChX:", "ChY:", "ChZ:")):
            ch_num = line.split(":")[0]
            # Look for the description line that follows
            continue
        
        elif line.startswith(("Ch1 Descr:", "Ch2 Descr:", "Ch3 Descr:", "Ch4 Descr:", 
                             "Ch5 Descr:", "Ch6 Descr:", "Ch7 Descr:", "Ch8 Descr:",
                             "ChX Descr:", "ChY Descr:", "ChZ Descr:")):
            ch_num = line.split(" ")[0]
            description = line.split(":", 1)[1].strip()
            if description:  # Only add channels with descriptions
                metadata['channels'][ch_num] = description
        
        # Stop parsing when we reach the data section
        elif "Date Time," in line and "Ch1," in line:
            # This is the header line, extract channel order
            metadata['channel_order'] = line.split(",")
            break
    
    return metadata


def _find_data_start(lines: list[str]) -> int:
    """Find the line number where actual CSV data starts."""
    for i, line in enumerate(lines):
        if line.strip().startswith("Data Set"):
            # Data starts on the next line
            return i + 1
    raise ValueError("Could not find 'Data Set' marker in file")


def _parse_pace_data(file_path: Path, data_start_idx: int, metadata: dict) -> pd.DataFrame:
    """Parse the CSV data section of the Pace logger file."""
    
    # Find line number where "Data Set 1" appears
    with open(file_path, 'r') as f:
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
        df = df.drop([0, len(df)-1]).reset_index(drop=True)
    elif len(df) >= 1:
        df = df.drop([0]).reset_index(drop=True)
    
    # Now parse dates on the cleaned dataframe and rename to datetime
    df['Date Time'] = pd.to_datetime(df['Date Time'])
    df = df.rename(columns={'Date Time': 'datetime'})
    
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
            column_mapping[col] = "wind_speed_peak"
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


def _create_pace_dataset(df: pd.DataFrame, metadata: dict, file_path: Path) -> xr.Dataset:
    """Create xarray dataset from parsed Pace logger data."""
    
    # Extract site name from label or filename
    site_name = metadata.get('label', file_path.stem).strip()
    
    # Convert datetime to UTC (assuming input is in local time)
    df['datetime_utc'] = pd.to_datetime(df['datetime']).dt.tz_localize(ZoneInfo("America/Anchorage")).dt.tz_convert("UTC").dt.tz_localize(None)
    
    # Separate temperature and non-temperature variables
    temp_cols = [col for col in df.columns if col.startswith('temp_c_')]
    other_cols = [col for col in df.columns if not col.startswith(('temp_c_', 'datetime'))]
    
    # Process temperature variables to extract heights
    temp_data = {}
    height_values = []
    
    for col in temp_cols:
        height_match = re.search(r"temp_c_(\d+)cm", col)
        if height_match:
            height_cm = int(height_match.group(1))
            height_m = height_cm / 100.0  # Convert to meters
            height_values.append(height_m)
            temp_data[height_m] = df[col].values
    
    # Add heights for other variables
    other_var_heights = {}
    for col in other_cols:
        if col == 'pressure':
            # Pressure at 1m
            other_var_heights[col] = 1.0
        else:
            # Other meteorological variables at 2m  
            other_var_heights[col] = 2.0
        
        # Add to height list if not already present
        if other_var_heights[col] not in height_values:
            height_values.append(other_var_heights[col])
    
    # Sort all heights
    height_values = sorted(set(height_values))
    
    # Create data variables dictionary
    data_vars = {}
    
    # Create temperature array with all heights
    if temp_data:
        temp_array = np.full((len(df), len(height_values)), np.nan)
        for i, height in enumerate(height_values):
            if height in temp_data:
                temp_array[:, i] = temp_data[height]
        data_vars['temp_c'] = (['datetime_utc', 'height'], temp_array)
    
    # Create arrays for other variables
    for col in other_cols:
        var_height = other_var_heights[col]
        var_array = np.full((len(df), len(height_values)), np.nan)
        
        # Find index of this variable's height
        height_idx = height_values.index(var_height)
        var_array[:, height_idx] = df[col].values
        
        data_vars[col] = (['datetime_utc', 'height'], var_array)
    
    # Create coordinates
    coords = {
        'datetime_utc': df['datetime_utc'],
        'height': height_values,
        'site_id': site_name
    }
    
    # Create dataset
    ds = xr.Dataset(data_vars, coords=coords)
    
    # Add CF-compliant attributes
    _add_pace_attributes(ds, metadata)
    
    return ds


def _add_pace_attributes(ds: xr.Dataset, metadata: dict):
    """Add CF-compliant attributes to variables and dataset."""
    
    # Dataset-level attributes (convert None to empty string for NetCDF compatibility)
    ds.attrs.update({
        'sensor_type': 'pace logger',
        'serial_number': metadata.get('serial_number') or 'unknown',
        'site_label': metadata.get('label') or '',
        'logging_start': metadata.get('logging_start') or '',
        'logging_stop': metadata.get('logging_stop') or '',
        'log_interval': metadata.get('log_interval') or '',
        'battery_voltage': metadata.get('battery_voltage') or 0.0,
        'institution': 'JIFLR Project',
        'source': 'Pace Scientific XR5-SE-M data logger'
    })
    
    # Variable-specific attributes
    if 'temp_c' in ds.data_vars:
        ds['temp_c'].attrs.update({
            'units': 'degrees_Celsius',
            'long_name': 'Air temperature',
            'standard_name': 'air_temperature'
        })
    
    if 'wind_direction' in ds.data_vars:
        ds['wind_direction'].attrs.update({
            'units': 'degrees',
            'long_name': 'Wind direction',
            'standard_name': 'wind_from_direction'
        })
    
    if 'pressure' in ds.data_vars:
        ds['pressure'].attrs.update({
            'units': 'kPa',
            'long_name': 'Atmospheric pressure',
            'standard_name': 'air_pressure'
        })
    
    if 'relative_humidity' in ds.data_vars:
        ds['relative_humidity'].attrs.update({
            'units': 'percent',
            'long_name': 'Relative humidity',
            'standard_name': 'relative_humidity'
        })
    
    if 'wind_speed_avg' in ds.data_vars:
        ds['wind_speed_avg'].attrs.update({
            'units': 'm s-1',
            'long_name': 'Average wind speed',
            'standard_name': 'wind_speed'
        })
    
    if 'wind_speed_peak' in ds.data_vars:
        ds['wind_speed_peak'].attrs.update({
            'units': 'm s-1',
            'long_name': 'Peak wind speed (2 second)',
            'standard_name': 'wind_speed_of_gust'
        })


def _save_pace_netcdf(ds: xr.Dataset, dir_out: Path, metadata: dict, time_coord_name: str):
    """Save xarray dataset as NetCDF file with appropriate filename."""
    
    # Get time range for filename
    time_data = ds[time_coord_name]
    start_time = pd.Timestamp(time_data.values[0])
    end_time = pd.Timestamp(time_data.values[-1])
    
    # Create filename
    serial_number = metadata.get('serial_number', 'unknown')
    site_name = metadata.get('label', 'unknown').replace(' ', '_').replace('-', '_')
    
    filename = f"{serial_number}_{site_name}_{start_time.strftime('%Y%m%dT%H%M')}_{end_time.strftime('%Y%m%dT%H%M')}.nc"
    output_path = dir_out / filename
    
    # Save to NetCDF
    ds.to_netcdf(output_path)
    print(f"Saved: {output_path}")


def _process_single_pace_file(
    file_path: Path,
    dir_out: Path, 
    convert_to_local_tz: bool,
    utc_offset_hours: float
):
    """Process a single Pace logger file and save as NetCDF."""
    
    # Read and parse the file
    with open(file_path, 'r') as f:
        lines = f.readlines()
    
    # Extract metadata from header
    metadata = _parse_pace_header(lines)
    
    # Find the data section and parse CSV data
    data_start_idx = _find_data_start(lines)
    df = _parse_pace_data(file_path, data_start_idx, metadata)
    
    # Create xarray dataset
    ds = _create_pace_dataset(df, metadata, file_path)
    
    # Convert timezone if requested
    if convert_to_local_tz:
        ds = replace_utc_datetime_coord(
            ds, utc_offset_hours=utc_offset_hours, add_cf_attributes=True
        )
        time_coord_name = "datetime"
    else:
        time_coord_name = "datetime_utc"
    
    # Generate output filename and save
    _save_pace_netcdf(ds, dir_out, metadata, time_coord_name)


def read_rgi(product: str, v=7):
    product = product.upper()
    p = Path(ROOT, f"data/external/rgi7/RGI2000-v7.0-{product}-01_alaska")
    rgi = gpd.read_file(p)

    return rgi
