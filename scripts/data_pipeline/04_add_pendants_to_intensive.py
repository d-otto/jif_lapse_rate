#!/usr/bin/env python3
"""
combine_pace_pendant_data.py

Script to combine intermediate pace data with by-site pendant data from the
intensive folder. This script creates a unified dataset that includes both
pace meteorological data (unshielded) and pendant temperature/light data
(shielded) with consistent coordinate structure.

The resulting combined dataset goes in processed/lvl0/ with coordinates:
- height: Combined heights from both data sources
- shielded: Two levels ["shielded", "unshielded"]
- datetime: Common time grid
- site_id: Combined site names

Only temperature data has the 'shielded' dimension. Other variables (wind,
pressure, etc.) from pace data are 3D without shielding dimension.

Created: 2025-10-09
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Dict, List

from jiflr import ROOT
from jiflr.data import replace_utc_datetime_coord
from jiflr.logging import indent, key_value, setup_pipeline_logging, subheader


def load_pace_data(pace_dir: Path, logger) -> Dict[str, xr.Dataset]:
    """
    Load all pace data files from the intermediate/pace directory.

    Parameters
    ----------
    pace_dir : Path
        Directory containing pace NetCDF files
    logger : logging.Logger
        Logger instance

    Returns
    -------
    Dict[str, xr.Dataset]
        Dictionary mapping site names to pace datasets
    """
    pace_data = {}

    if not pace_dir.exists():
        logger.warning(f"Pace directory not found: {pace_dir}")
        return pace_data

    pace_files = list(pace_dir.glob("*.nc"))
    logger.info(f"Found {len(pace_files)} pace files")

    for pace_file in pace_files:
        try:
            ds = xr.open_dataset(pace_file)

            # Extract site name from site_id variable or filename
            if "site_id" in ds.data_vars:
                site_name = str(ds.site_id.values)
                # Convert dashes to numbers (e.g., "Lee-1" -> "Lee1", "Windward-1" -> "Windward1")
                site_name = site_name.replace("-", "")
            else:
                # Extract from filename if site_id not available
                # Parse filename like "EM54051_Windward_1_..." to get "Windward1"
                filename_parts = pace_file.stem.split("_")
                if len(filename_parts) >= 3 and not filename_parts[2].startswith("20"):
                    # Case: EM54051_Windward_1_... -> "Windward1"
                    site_name = (
                        filename_parts[1] + filename_parts[2]
                    )  # "Windward" + "1"
                else:
                    # Case: EM54053_Divide_20250619... -> "Divide"
                    site_name = filename_parts[1]  # Just "Divide"

            # Skip if we already have this site (handle duplicates)
            if site_name in pace_data:
                logger.warning(
                    f"Duplicate pace data for site {site_name}, skipping {pace_file.name}"
                )
                continue

            pace_data[site_name] = ds
            logger.info(indent(f"Loaded pace data for site: {site_name}"))

        except Exception as e:
            logger.warning(f"Failed to load pace file {pace_file}: {e}")

    return pace_data


def load_pendant_data(pendant_dir: Path, logger) -> Dict[str, xr.Dataset]:
    """
    Load all pendant data files from the intensive by_site directory.

    Parameters
    ----------
    pendant_dir : Path
        Directory containing pendant NetCDF files (by_site/intensive)
    logger : logging.Logger
        Logger instance

    Returns
    -------
    Dict[str, xr.Dataset]
        Dictionary mapping site names to pendant datasets
    """
    pendant_data = {}

    if not pendant_dir.exists():
        logger.warning(f"Pendant directory not found: {pendant_dir}")
        return pendant_data

    pendant_files = list(pendant_dir.glob("*.nc"))
    logger.info(f"Found {len(pendant_files)} pendant files")

    for pendant_file in pendant_files:
        try:
            ds = xr.open_dataset(pendant_file)

            # Extract site name from global attributes or filename
            if "site_name" in ds.attrs:
                site_name = str(ds.attrs["site_name"]).strip()
            else:
                site_name = pendant_file.stem

            # Skip files with empty or invalid site names
            if not site_name or site_name.isspace():
                logger.warning(f"Skipping file with empty site name: {pendant_file}")
                continue

            pendant_data[site_name] = ds
            logger.info(indent(f"Loaded pendant data for site: {site_name}"))

        except Exception as e:
            logger.warning(f"Failed to load pendant file {pendant_file}: {e}")

    return pendant_data


# TODO: Move this section to the prior pace processing step
def convert_pace_datetime_coords(
    pace_data: Dict[str, xr.Dataset],
    logger,
) -> Dict[str, xr.Dataset]:
    """
    Convert pace data datetime coordinates from UTC to local time and rename.

    This function will eventually be moved to the pace data processing script.
    For now, it's included here for testing purposes.

    This bundles together the time shifting (-9 UTC) and coordinate renaming
    (datetime_utc -> datetime) steps.

    Parameters
    ----------
    pace_data : Dict[str, xr.Dataset]
        Dictionary of pace datasets with datetime_utc coordinates
    logger : logging.Logger
        Logger instance

    Returns
    -------
    Dict[str, xr.Dataset]
        Dictionary of pace datasets with datetime coordinates in local time
    """
    logger.info("TODO: Converting pace datetime coordinates (will be moved to prior step)")

    converted_data = {}

    for site_name, ds in pace_data.items():
        try:
            if "datetime_utc" in ds.coords:
                # Bundle time shifting and coordinate renaming together
                # Convert UTC to local time (-9 hours) and rename coordinate from datetime_utc to datetime
                ds_converted = replace_utc_datetime_coord(
                    ds,
                    utc_offset_hours=-9.0,
                    new_coord_name="datetime",
                    old_coord_name="datetime_utc",
                    add_cf_attributes=True,
                )
                converted_data[site_name] = ds_converted
                logger.info(
                    indent(f"Converted datetime_utc -> datetime with -9hr offset for site: {site_name}")
                )
            else:
                # Already has datetime coordinate
                converted_data[site_name] = ds
                logger.info(indent(f"Site {site_name} already has datetime coordinate"))

        except Exception as e:
            logger.warning(f"Failed to convert datetime for {site_name}: {e}")
            converted_data[site_name] = ds  # Use original if conversion fails

    return converted_data


def standardize_height_formats(heights: List) -> List[str]:
    """
    Standardize height values to consistent string format (e.g., "0.5m", "1m", "2m").

    Parameters
    ----------
    heights : List
        List of height values in various formats

    Returns
    -------
    List[str]
        Standardized height strings
    """
    standardized = []

    for height in heights:
        height_str = str(height)

        # Remove 'm' suffix if present
        if height_str.endswith("m"):
            height_str = height_str[:-1]

        try:
            height_val = float(height_str)
            if height_val == int(height_val):
                standardized.append(f"{int(height_val)}m")
            else:
                standardized.append(f"{height_val}m")
        except (ValueError, TypeError):
            # Keep original if can't parse
            standardized.append(str(height))

    return sorted(set(standardized))


def map_site_names(pace_sites: List[str], pendant_sites: List[str]) -> Dict[str, str]:
    """
    Create mapping between pace site names and pendant site names.

    Parameters
    ----------
    pace_sites : List[str]
        List of pace site names
    pendant_sites : List[str]
        List of pendant site names

    Returns
    -------
    Dict[str, str]
        Mapping from pace site names to standardized site names
    """
    site_mapping = {}

    # Known mappings (can be expanded as needed)
    known_mappings = {
        "Windward1": "Windward1",
        "Windward2": "Windward2",
        "Lee1": "Lee1",
        "Lee2": "Lee2",
        "Divide": "Divide",
        "A02": "A02",
    }

    # Map pace sites to standard names
    for pace_site in pace_sites:
        # Try exact match first
        if pace_site in known_mappings:
            site_mapping[pace_site] = known_mappings[pace_site]
        else:
            # Try fuzzy matching
            for standard_name in known_mappings.values():
                if pace_site.lower().replace("-", "").replace(
                    "_", ""
                ) == standard_name.lower().replace("-", "").replace("_", ""):
                    site_mapping[pace_site] = standard_name
                    break
            else:
                # Use pace site name as-is if no mapping found
                site_mapping[pace_site] = pace_site

    return site_mapping


def create_common_datetime_grid(
    pace_data: Dict[str, xr.Dataset], pendant_data: Dict[str, xr.Dataset], logger
) -> pd.DatetimeIndex:
    """
    Create a common datetime grid that spans all data from both sources.

    Parameters
    ----------
    pace_data : Dict[str, xr.Dataset]
        Pace datasets
    pendant_data : Dict[str, xr.Dataset]
        Pendant datasets
    logger : logging.Logger
        Logger instance

    Returns
    -------
    pd.DatetimeIndex
        Common datetime grid at minute resolution
    """
    all_times = []

    # Collect all datetime values from pace data
    for ds in pace_data.values():
        if "datetime" in ds.coords:
            times = pd.to_datetime(ds.datetime.values)
            all_times.extend(times)

    # Collect all datetime values from pendant data
    for ds in pendant_data.values():
        if "datetime" in ds.coords:
            times = pd.to_datetime(ds.datetime.values)
            all_times.extend(times)

    if not all_times:
        raise ValueError("No datetime coordinates found in any dataset")

    # Round to nearest minute and get unique times
    all_times_rounded = [pd.Timestamp(t).round("min") for t in all_times]
    unique_times = pd.DatetimeIndex(sorted(set(all_times_rounded)))

    logger.info(
        f"Created common datetime grid: {unique_times[0]} to {unique_times[-1]} ({len(unique_times)} points)"
    )

    return unique_times


def combine_datasets(
    pace_data: Dict[str, xr.Dataset],
    pendant_data: Dict[str, xr.Dataset],
    common_datetime: pd.DatetimeIndex,
    site_mapping: Dict[str, str],
    logger,
) -> xr.Dataset:
    """
    Combine pace and pendant datasets using sensor_idx concatenation.

    Much simpler approach: just concatenate all datasets along sensor_idx dimension.

    Parameters
    ----------
    pace_data : Dict[str, xr.Dataset]
        Pace datasets (already using sensor_idx structure)
    pendant_data : Dict[str, xr.Dataset]
        Pendant datasets (already using sensor_idx structure)
    common_datetime : pd.DatetimeIndex
        Common datetime grid
    site_mapping : Dict[str, str]
        Mapping from pace site names to standard site names
    logger : logging.Logger
        Logger instance

    Returns
    -------
    xr.Dataset
        Combined dataset with sensor_idx structure
    """

    # Collect all datasets to concatenate
    datasets_to_concat = []

    # Add pace datasets
    for pace_site, ds in pace_data.items():
        # Standardize site names in pace data if needed
        standard_site = site_mapping.get(pace_site, pace_site)

        # Update site_id coordinate to use standard site name
        if "site_id" in ds.coords and standard_site != pace_site:
            # Update the site_id coordinate values
            new_site_ids = [standard_site if site == pace_site else site for site in ds.site_id.values]
            ds = ds.assign_coords(site_id=('sensor_idx', new_site_ids))

        # Reindex to common datetime grid
        datetime_coord = "datetime" if "datetime" in ds.coords else "datetime_utc"
        ds_rounded = ds.assign_coords({datetime_coord: ds[datetime_coord].dt.round("min")})

        # Remove duplicate times (keep first occurrence)
        _, unique_idx = np.unique(ds_rounded[datetime_coord].values, return_index=True)
        ds_unique = ds_rounded.isel(**{datetime_coord: unique_idx})

        ds_reindexed = ds_unique.reindex({datetime_coord: common_datetime}, method=None)

        datasets_to_concat.append(ds_reindexed)
        logger.info(indent(f"Added pace dataset for site: {standard_site} ({len(ds.sensor_idx)} sensors)"))

    # Add pendant datasets
    for pendant_site, ds in pendant_data.items():
        # Reindex to common datetime grid
        datetime_coord = "datetime" if "datetime" in ds.coords else "datetime_utc"
        ds_rounded = ds.assign_coords({datetime_coord: ds[datetime_coord].dt.round("min")})

        # Remove duplicate times (keep first occurrence)
        _, unique_idx = np.unique(ds_rounded[datetime_coord].values, return_index=True)
        ds_unique = ds_rounded.isel(**{datetime_coord: unique_idx})

        ds_reindexed = ds_unique.reindex({datetime_coord: common_datetime}, method=None)

        datasets_to_concat.append(ds_reindexed)
        logger.info(indent(f"Added pendant dataset for site: {pendant_site} ({len(ds.sensor_idx)} sensors)"))

    if not datasets_to_concat:
        raise ValueError("No datasets to concatenate")

    logger.info(f"Combining {len(datasets_to_concat)} datasets along sensor_idx dimension...")

    # Concatenate all datasets along sensor_idx dimension
    try:
        combined_ds = xr.concat(datasets_to_concat, dim="sensor_idx", data_vars='all', coords='all')
        # Fix sensor_idx to be sequential (0, 1, 2, 3...) instead of all zeros
        new_sensor_idx = list(range(len(combined_ds.sensor_idx)))
        combined_ds = combined_ds.assign_coords(sensor_idx=new_sensor_idx)
    except Exception as e:
        logger.warning(f"Error concatenating datasets: {e}")
        # Try with minimal options if full concatenation fails
        combined_ds = xr.concat(datasets_to_concat, dim="sensor_idx")
        # Fix sensor_idx to be sequential (0, 1, 2, 3...) instead of all zeros
        new_sensor_idx = list(range(len(combined_ds.sensor_idx)))
        combined_ds = combined_ds.assign_coords(sensor_idx=new_sensor_idx)

    # Get unique sites for summary
    unique_sites = sorted(set(combined_ds.site_id.values))
    n_sensors = len(combined_ds.sensor_idx)

    logger.info(f"Successfully combined {n_sensors} sensors from {len(unique_sites)} sites:")
    logger.info(indent(f"Sites: {unique_sites}"))
    logger.info(indent(f"Time points: {len(common_datetime)}"))

    # Add global attributes
    combined_ds.attrs.update(
        {
            "title": "Combined JIFLR Intensive Site Data (Pace + Pendant)",
            "source": "Combined from pace and pendant sensor data",
            "processing_step": "lvl0_combined_sensor_idx",
            "institution": "JIFLR Project",
            "structure": "sensor_idx x datetime",
            "n_sensors": n_sensors,
            "n_sites": len(unique_sites),
            "site_names": ", ".join(unique_sites),
            "sensor_type_info": "pace=unshielded meteorological station, pendant=shielded/unshielded temperature sensors",
            "data_conflict_resolution": "sensor_type coordinate separates pace vs pendant measurements",
            "created_by": "combine_pace_pendant_data.py",
        }
    )

    return combined_ds


def main():
    """Main function to combine pace and pendant data."""
    # Set up logging (appends to pipeline log if running as part of pipeline)
    logger = setup_pipeline_logging(step_number=4, total_steps=6, mode="a")

    # Define paths
    base_dir = Path(ROOT) / "data" / "2025"
    pace_dir = base_dir / "intermediate" / "pace"
    pendant_dir = base_dir / "intermediate" / "pendants" / "by_site" / "intensive"
    output_dir = base_dir / "processed" / "lvl0"

    logger.info(key_value("Pace data directory", str(pace_dir)))
    logger.info(key_value("Pendant data directory", str(pendant_dir)))
    logger.info(key_value("Output directory", str(output_dir)))

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    logger.info(subheader("1. Loading pace data"))
    pace_data = load_pace_data(pace_dir, logger)

    logger.info(subheader("2. Loading pendant data"))
    pendant_data = load_pendant_data(pendant_dir, logger)

    if not pace_data and not pendant_data:
        logger.error("No data found in either pace or pendant directories")
        return

    # Convert pace datetime coordinates (TODO: move to prior step)
    if pace_data:
        logger.info(subheader("3. Converting pace datetime coordinates"))
        pace_data = convert_pace_datetime_coords(pace_data, logger)

    # Create site name mapping
    logger.info(subheader("4. Creating site name mapping"))
    pace_sites = list(pace_data.keys())
    pendant_sites = list(pendant_data.keys())
    site_mapping = map_site_names(pace_sites, pendant_sites)
    logger.info(f"Site mapping: {site_mapping}")

    # Create common datetime grid
    logger.info(subheader("5. Creating common datetime grid"))
    common_datetime = create_common_datetime_grid(pace_data, pendant_data, logger)

    # Combine datasets
    logger.info(subheader("6. Combining datasets"))
    combined_ds = combine_datasets(
        pace_data, pendant_data, common_datetime, site_mapping, logger
    )

    # Save combined dataset
    output_file = output_dir / "lvl0_intensive.nc"
    logger.info(subheader("7. Saving combined dataset"))
    logger.info(key_value("Output file", str(output_file)))
    combined_ds.to_netcdf(output_file)

    # Print summary
    logger.info(subheader("Summary"))
    logger.info(key_value("Dimensions", str(dict(combined_ds.dims))))
    logger.info(key_value("Variables", str(list(combined_ds.data_vars.keys()))))
    logger.info(key_value("Total sensors", str(len(combined_ds.sensor_idx))))

    # Get unique values from sensor coordinates
    unique_sites = sorted(set(combined_ds.site_id.values))
    unique_heights = sorted(set(combined_ds.height.values))
    unique_shielding = sorted(set(combined_ds.shielding.values))
    unique_sensor_types = sorted(set(combined_ds.sensor_type.values))

    logger.info(key_value("Unique sites", str(unique_sites)))
    logger.info(key_value("Unique heights", str(unique_heights)))
    logger.info(key_value("Unique shielding levels", str(unique_shielding)))
    logger.info(key_value("Unique sensor types", str(unique_sensor_types)))

    datetime_coord = "datetime" if "datetime" in combined_ds.coords else "datetime_utc"
    logger.info(
        key_value("Time range", f"{combined_ds[datetime_coord].values[0]} to {combined_ds[datetime_coord].values[-1]}")
    )

    logger.info("Data structure uses sensor_idx with sensor attributes as coordinates")
    logger.info(indent("- Each sensor has site_id, height, shielding, sensor_type attributes"))
    logger.info(indent("- Data dimensions: (sensor_idx, datetime)"))
    logger.info(indent("- Use ds.where() or groupby operations for analysis"))


if __name__ == "__main__":
    main()
