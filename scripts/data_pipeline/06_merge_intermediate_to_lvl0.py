#!/usr/bin/env python3
"""
merge_intermediate_site_data.py

Script to merge site-level NetCDF files from individual sites into combined
files based on subdirectory structure. This script:

1. Loads data from data/2025/intermediate/pendants/by_site/
2. Groups files by subdirectory (root, intensive/, etc.)
3. Merges datasets using tree-based pairwise merging for efficiency
4. Aligns datasets by finding union of datetimes and reindexing
5. Concatenates along site_id coordinate
6. Saves combined datasets to data/2025/processed/lvl0/

Created: 2025-10-03
"""

import argparse
import csv
import hashlib
from functools import partial
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path
from tqdm import tqdm

from jiflr import ROOT
from jiflr.logging import indent, item, key_value, setup_pipeline_logging, subheader
from jiflr.netcdf_metadata import (
    QC_FLAG_COMMENT,
    QC_FLAG_MASKS,
    QC_FLAG_MEANINGS,
    apply_product_metadata,
)
from jiflr.pipeline import (
    NoiseQCSpec,
    apply_noise_qc,
    create_netcdf_encoding,
    ensure_season_year_coordinate,
)
from jiflr.qc_plots import create_all_qc_plots


QC_FLAG_SUFFIX = "_qc_flag"
MANUAL_MASK_PERIOD_BIT = np.uint32(1)
WIND_SPEED_OVER_50_M_S_BIT = np.uint32(2)
RAINFALL_OUTSIDE_ALLOWED_SITES_BIT = np.uint32(4)
WIND_DIRECTION_LOW_SPEED_BIT = np.uint32(8)
PRESSURE_NOISE_CANDIDATE_QC_BIT = np.uint32(16)
WIND_SPEED_NOISE_CANDIDATE_QC_BIT = np.uint32(32)
PRESSURE_NOISE_QC = NoiseQCSpec(
    name="pressure_noise",
    variables=("pressure",),
    flag_bit=int(PRESSURE_NOISE_CANDIDATE_QC_BIT),
    absolute_floor=0.125,
    floor_unit="kpa",
    window="720min",
    min_periods=7,
    mad_multiplier=0.25,
    sensor_type_prefixes=("pace",),
)
WIND_SPEED_NOISE_QC = NoiseQCSpec(
    name="wind_speed_noise",
    variables=("wind_speed_avg", "wind_speed_max"),
    flag_bit=int(WIND_SPEED_NOISE_CANDIDATE_QC_BIT),
    absolute_floor=10.0,
    floor_unit="m_s",
    window="720min",
    min_periods=7,
    mad_multiplier=0.25,
)
RAINFALL_ALLOWED_SITES = {"A04", "A17"}
WIND_DIRECTION_SENSOR_TYPES_2025 = {"pace", "rmyoung"}
WIND_DIRECTION_SENSOR_TYPES_2026 = {"pace"}
WIND_DIRECTION_SPEED_THRESHOLD_M_S = 0.5
MASK_PERIOD_COLUMNS = (
    "start_datetime_utc",
    "end_datetime_utc",
    "site_id",
    "sensor_id",
    "variable",
    "notes",
)


def _measurement_variables(ds):
    """Return numeric measurements that can receive time-indexed QC flags."""
    return [
        name
        for name, data_array in ds.data_vars.items()
        if not name.endswith(QC_FLAG_SUFFIX)
        and {"sensor_idx", "datetime_utc"}.issubset(data_array.dims)
        and np.issubdtype(data_array.dtype, np.number)
    ]


def _qc_flag_name(variable):
    """Return the QC flag variable name associated with *variable*."""
    return f"{variable}{QC_FLAG_SUFFIX}"


def _initialize_qc_flags(ds):
    """Reset this pipeline's per-measurement QC flags for a fresh lvl0 run."""
    for variable in _measurement_variables(ds):
        flag_name = _qc_flag_name(variable)
        flags = xr.zeros_like(ds[variable], dtype=np.uint32)
        flags.attrs = {
            "long_name": f"Quality-control flags for {variable}",
            "standard_name": "quality_flag",
            "flag_masks": QC_FLAG_MASKS,
            "flag_meanings": QC_FLAG_MEANINGS,
            "valid_min": np.uint32(0),
            "comment": QC_FLAG_COMMENT,
        }
        ds[flag_name] = flags
    return ds


def _load_mask_periods(csv_path):
    """Load and strictly validate the manual Level 0 mask-period CSV."""
    if not csv_path.exists():
        return []

    with csv_path.open(newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None:
            raise ValueError(f"Mask-period CSV is empty: {csv_path}")
        if tuple(reader.fieldnames) != MASK_PERIOD_COLUMNS:
            raise ValueError(
                f"Mask-period CSV must use these columns in this order: "
                f"{', '.join(MASK_PERIOD_COLUMNS)}. Found: {', '.join(reader.fieldnames)}"
            )
        rows = list(reader)

    periods = []
    for row_number, row in enumerate(rows, start=2):
        start_value = (row["start_datetime_utc"] or "").strip()
        end_value = (row["end_datetime_utc"] or "").strip()
        notes = (row["notes"] or "").strip()
        if not start_value:
            raise ValueError(f"Mask-period CSV row {row_number} requires a start datetime")
        if not notes:
            raise ValueError(f"Mask-period CSV row {row_number} requires notes")
        try:
            start = pd.Timestamp(start_value)
            end = pd.Timestamp(end_value) if end_value else None
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"Mask-period CSV row {row_number} has an invalid UTC datetime"
            ) from error
        if pd.isna(start) or (end is not None and pd.isna(end)):
            raise ValueError(
                f"Mask-period CSV row {row_number} has an invalid UTC datetime"
            )
        if start.tzinfo is not None:
            start = start.tz_convert("UTC").tz_localize(None)
        if end is not None and end.tzinfo is not None:
            end = end.tz_convert("UTC").tz_localize(None)
        if end is not None and start > end:
            raise ValueError(
                f"Mask-period CSV row {row_number} starts after it ends: {start} > {end}"
            )
        periods.append(
            {
                "row_number": row_number,
                "start": start,
                "end": end,
                "site_id": (row["site_id"] or "").strip(),
                "sensor_id": (row["sensor_id"] or "").strip(),
                "variable": (row["variable"] or "").strip(),
                "notes": notes,
            }
        )
    return periods


def _target_sensor_mask(ds, period):
    """Return the sensors selected by a manual period, or ``None`` if absent."""
    selected = np.ones(ds.sizes["sensor_idx"], dtype=bool)
    for coordinate in ("site_id", "sensor_id"):
        target = period[coordinate]
        if not target:
            continue
        if coordinate not in ds.coords:
            raise ValueError(
                f"Mask-period CSV row {period['row_number']} targets {coordinate}={target!r}, "
                f"but the dataset has no {coordinate} coordinate"
            )
        selected &= np.asarray(ds[coordinate].values).astype(str) == target
    if not selected.any():
        return None
    return xr.DataArray(
        selected,
        dims=("sensor_idx",),
        coords={"sensor_idx": ds["sensor_idx"]},
    )


def _apply_manual_mask_periods(ds, periods):
    """Set the manual-mask bit for every CSV interval that targets this dataset."""
    matches_by_row = {period["row_number"]: 0 for period in periods}
    variables = _measurement_variables(ds)
    time_values = pd.DatetimeIndex(ds["datetime_utc"].values)

    for period in periods:
        sensor_mask = _target_sensor_mask(ds, period)
        if sensor_mask is None:
            continue
        if period["variable"]:
            if period["variable"] not in variables:
                raise ValueError(
                    f"Mask-period CSV row {period['row_number']} targets unsupported variable "
                    f"{period['variable']!r}"
                )
            target_variables = [period["variable"]]
        else:
            target_variables = variables

        in_period = time_values >= period["start"]
        if period["end"] is not None:
            in_period &= time_values <= period["end"]
        time_mask = xr.DataArray(
            in_period,
            dims=("datetime_utc",),
            coords={"datetime_utc": ds["datetime_utc"]},
        )
        mask = sensor_mask & time_mask
        n_values = int(mask.sum().item()) * len(target_variables)
        if n_values == 0:
            continue
        matches_by_row[period["row_number"]] += n_values
        for variable in target_variables:
            flag_name = _qc_flag_name(variable)
            ds[flag_name] = ds[flag_name] | xr.where(
                mask, MANUAL_MASK_PERIOD_BIT, np.uint32(0)
            )
    return ds, matches_by_row


def _apply_wind_speed_over_50_m_s(ds):
    """Set the plausibility bit for all wind-speed measurements above 50 m/s."""
    affected = 0
    for variable in _measurement_variables(ds):
        if not variable.startswith("wind_speed_"):
            continue
        exceedance = ds[variable] > 50.0
        affected += int(exceedance.sum().item())
        flag_name = _qc_flag_name(variable)
        ds[flag_name] = ds[flag_name] | xr.where(
            exceedance, WIND_SPEED_OVER_50_M_S_BIT, np.uint32(0)
        )
    return ds, affected


def _apply_rainfall_site_allowlist(ds):
    """Flag rainfall at every site other than A04 and A17."""
    rainfall_variables = [
        variable for variable in _measurement_variables(ds) if variable.startswith("rainfall_")
    ]
    if not rainfall_variables:
        return ds, 0
    if "site_id" not in ds.coords:
        raise ValueError("Rainfall filtering requires a site_id coordinate")

    excluded_sites = xr.DataArray(
        ~np.isin(ds["site_id"].values.astype(str), sorted(RAINFALL_ALLOWED_SITES)),
        dims=("sensor_idx",),
        coords={"sensor_idx": ds["sensor_idx"]},
    )
    for variable in rainfall_variables:
        observed_rainfall = ds[variable].notnull()
        invalid_rainfall = excluded_sites & observed_rainfall
        flag_name = _qc_flag_name(variable)
        ds[flag_name] = ds[flag_name] | xr.where(
            invalid_rainfall, RAINFALL_OUTSIDE_ALLOWED_SITES_BIT, np.uint32(0)
        )
    return ds, sum(
        int((excluded_sites & ds[variable].notnull()).sum().item())
        for variable in rainfall_variables
    )


def _apply_wind_direction_low_speed(ds, sensor_types):
    """Flag selected wind sensor directions below the wind-speed threshold."""
    if "wind_direction" not in ds.data_vars or "wind_speed_avg" not in ds.data_vars:
        return ds, 0
    if "sensor_type" not in ds.coords or "site_id" not in ds.coords:
        raise ValueError(
            "Wind-direction low-speed filtering requires sensor_type and site_id coordinates"
        )

    def is_wind_sensor(sensor_idx):
        sensor_type = str(ds["sensor_type"].sel(sensor_idx=sensor_idx).item()).casefold()
        return sensor_type in sensor_types

    speed_sensor_by_site = {}
    for sensor_idx in ds["sensor_idx"].values:
        if not is_wind_sensor(sensor_idx):
            continue
        speed = ds["wind_speed_avg"].sel(sensor_idx=sensor_idx)
        if speed.notnull().any():
            site_id = str(ds["site_id"].sel(sensor_idx=sensor_idx).item())
            speed_sensor_by_site[site_id] = sensor_idx

    affected = 0
    for sensor_idx in ds["sensor_idx"].values:
        if not is_wind_sensor(sensor_idx):
            continue
        direction = ds["wind_direction"].sel(sensor_idx=sensor_idx)
        if direction.isnull().all():
            continue
        site_id = str(ds["site_id"].sel(sensor_idx=sensor_idx).item())
        speed_sensor_idx = speed_sensor_by_site.get(site_id)
        if speed_sensor_idx is None:
            continue
        speed = ds["wind_speed_avg"].sel(sensor_idx=speed_sensor_idx)
        low_speed_direction = (speed < WIND_DIRECTION_SPEED_THRESHOLD_M_S) & direction.notnull()
        affected += int(low_speed_direction.sum().item())
        ds[_qc_flag_name("wind_direction")].loc[dict(sensor_idx=sensor_idx)] = (
            ds[_qc_flag_name("wind_direction")].sel(sensor_idx=sensor_idx)
            | xr.where(low_speed_direction, WIND_DIRECTION_LOW_SPEED_BIT, np.uint32(0))
        )
    return ds, affected


def _apply_wind_direction_low_speed_2025(ds):
    """Apply the 2025 low-speed wind-direction filter to Pace and RM Young."""
    return _apply_wind_direction_low_speed(ds, WIND_DIRECTION_SENSOR_TYPES_2025)


def _apply_wind_direction_low_speed_2026(ds):
    """Apply the 2026 low-speed wind-direction filter to Pace only."""
    return _apply_wind_direction_low_speed(ds, WIND_DIRECTION_SENSOR_TYPES_2026)


# New filters belong here, in execution order. Each filter only adds its own bit.
# ``None`` applies to every season; a set limits a filter to specific seasons.
LVL0_FILTERS = (
    ("manual_mask_periods", None, _apply_manual_mask_periods),
    ("wind_speed_over_50_m_s", {2026}, _apply_wind_speed_over_50_m_s),
    ("rainfall_site_allowlist", {2026}, _apply_rainfall_site_allowlist),
    ("wind_direction_low_speed", {2025}, _apply_wind_direction_low_speed_2025),
    ("wind_direction_low_speed", {2026}, _apply_wind_direction_low_speed_2026),
    ("pace_pressure_noise_candidates", None, partial(apply_noise_qc, spec=PRESSURE_NOISE_QC)),
    ("wind_speed_noise_candidates", None, partial(apply_noise_qc, spec=WIND_SPEED_NOISE_QC)),
)


def _filters_for_year(year):
    """Return the ordered Level 0 QC filters applicable to *year*."""
    return tuple(
        (filter_name, filter_function)
        for filter_name, years, filter_function in LVL0_FILTERS
        if years is None or year in years
    )


def apply_lvl0_filters(ds, periods, year):
    """Apply the ordered Level 0 QC filters for *year* and return their counts."""
    ds = _initialize_qc_flags(ds.copy())
    flag_attrs = {
        _qc_flag_name(variable): ds[_qc_flag_name(variable)].attrs.copy()
        for variable in _measurement_variables(ds)
    }
    counts = {}
    active_filters = _filters_for_year(year)
    for filter_name, filter_function in active_filters:
        if filter_name == "manual_mask_periods":
            ds, counts[filter_name] = filter_function(ds, periods)
        else:
            ds, counts[filter_name] = filter_function(ds)
    for flag_name, attrs in flag_attrs.items():
        ds[flag_name].attrs = attrs
    ds.attrs.update(
        {
            "qc_filter_order": ", ".join(name for name, _ in active_filters),
            "qc_flag_meanings": QC_FLAG_MEANINGS,
            "qc_flags_mask_lvl1_data": "true",
        }
    )
    return ds, counts


def apply_filters_to_lvl0_outputs(output_dir, mask_periods_path, year, logger):
    """Apply the QC filters applicable to one season's Level 0 products."""
    periods = _load_mask_periods(mask_periods_path)
    checksum = (
        hashlib.sha256(mask_periods_path.read_bytes()).hexdigest() if mask_periods_path.exists() else None
    )
    row_matches = {period["row_number"]: 0 for period in periods}
    output_paths = sorted(output_dir.glob("lvl0_*.nc"))
    if not output_paths:
        raise FileNotFoundError(f"No Level 0 products found for filtering in {output_dir}")

    logger.info(subheader("Applying Level 0 QC filters"))
    logger.info(key_value("Manual mask-period CSV", str(mask_periods_path)))
    for output_path in output_paths:
        ds = xr.load_dataset(output_path)
        filtered, counts = apply_lvl0_filters(ds, periods, year)
        if checksum is not None:
            filtered.attrs["qc_manual_mask_periods_csv"] = str(mask_periods_path)
            filtered.attrs["qc_manual_mask_periods_sha256"] = checksum
        for row_number, count in counts["manual_mask_periods"].items():
            row_matches[row_number] += count
        filtered = apply_product_metadata(
            filtered, level="lvl0", product=output_path.stem.removeprefix("lvl0_")
        )
        filtered.to_netcdf(
            output_path,
            mode="w",
            encoding=create_netcdf_encoding(filtered),
        )
        create_all_qc_plots(
            ds=filtered,
            output_dir=output_dir,
            filename_prefix=output_path.stem,
            logger=logger,
        )
        if "wind_speed_over_50_m_s" in counts:
            logger.info(
                indent(
                    f"{output_path.name}: {counts['wind_speed_over_50_m_s']} wind-speed values "
                    "flagged above 50 m/s",
                    level=2,
                )
            )
        if "rainfall_site_allowlist" in counts:
            logger.info(
                indent(
                    f"{output_path.name}: {counts['rainfall_site_allowlist']} rainfall values "
                    "flagged outside A04 and A17",
                    level=2,
                )
            )
        if "wind_direction_low_speed" in counts:
            logger.info(
                indent(
                    f"{output_path.name}: {counts['wind_direction_low_speed']} wind-direction values "
                    "flagged below 0.5 m/s",
                    level=2,
                )
            )
        if "pace_pressure_noise_candidates" in counts:
            logger.info(
                indent(
                    f"{output_path.name}: {counts['pace_pressure_noise_candidates']} PACE pressure "
                    "noise candidates flagged",
                    level=2,
                )
            )
        if "wind_speed_noise_candidates" in counts:
            logger.info(
                indent(
                    f"{output_path.name}: {counts['wind_speed_noise_candidates']} "
                    "average/maximum wind speed noise candidates flagged",
                    level=2,
                )
            )

    unmatched = [
        row_number for row_number, count in row_matches.items() if count == 0
    ]
    if unmatched:
        raise ValueError(
            "Mask-period CSV rows did not match any Level 0 measurement: "
            + ", ".join(str(row_number) for row_number in unmatched)
        )


def _fix_string_coordinate_lengths(ds, other_ds):
    """
    Fix string coordinate lengths to prevent truncation during concatenation.

    Parameters
    ----------
    ds : xarray.Dataset
        Dataset to fix
    other_ds : xarray.Dataset
        Other dataset to compare string lengths with

    Returns
    -------
    xarray.Dataset
        Dataset with fixed string coordinate lengths
    """
    # Identify string coordinates
    string_coords = []
    for coord_name in ds.coords:
        if coord_name != 'sensor_idx' and ds[coord_name].dtype.kind in ['U', 'S']:
            string_coords.append(coord_name)

    if not string_coords:
        return ds

    # Calculate maximum string length needed for each coordinate
    coord_updates = {}
    for coord_name in string_coords:
        if coord_name in other_ds.coords:
            # Get max length from both datasets
            max_len_ds = max(len(str(val)) for val in ds[coord_name].values)
            max_len_other = max(len(str(val)) for val in other_ds[coord_name].values)
            max_len = max(max_len_ds, max_len_other, 10)  # Minimum 10 chars
        else:
            # Just use current dataset
            max_len = max(max(len(str(val)) for val in ds[coord_name].values), 10)

        # Create new coordinate with adequate string length
        current_values = ds[coord_name].values
        new_dtype = f'U{max_len}'
        new_values = np.array(current_values, dtype=new_dtype)
        coord_updates[coord_name] = (ds[coord_name].dims, new_values)

    if coord_updates:
        ds = ds.assign_coords(coord_updates)

    return ds


def merge_two_datasets(ds1, ds2):
    """
    Merge two xarray datasets with sensor_idx structure by concatenating along sensor_idx dimension.

    Parameters
    ----------
    ds1, ds2 : xarray.Dataset
        Datasets to merge (must have sensor_idx structure)

    Returns
    -------
    xarray.Dataset
        Merged dataset concatenated along sensor_idx dimension
    """
    # Verify both datasets have sensor_idx structure
    if 'sensor_idx' not in ds1.dims:
        raise ValueError("Dataset 1 does not have sensor_idx structure")
    if 'sensor_idx' not in ds2.dims:
        raise ValueError("Dataset 2 does not have sensor_idx structure")

    # Fix string coordinate truncation by ensuring adequate string lengths
    ds1_fixed = _fix_string_coordinate_lengths(ds1, ds2)
    ds2_fixed = _fix_string_coordinate_lengths(ds2, ds1)

    # Simply concatenate along sensor_idx dimension - much simpler!
    try:
        merged_ds = xr.concat([ds1_fixed, ds2_fixed], dim='sensor_idx', data_vars='all', coords='all', join='outer')
        # Fix sensor_idx to be sequential (0, 1, 2, 3...) instead of all zeros
        new_sensor_idx = list(range(len(merged_ds.sensor_idx)))
        merged_ds = merged_ds.assign_coords(sensor_idx=new_sensor_idx)
        return merged_ds
    except Exception as e:
        raise ValueError(f"Failed to concatenate datasets along sensor_idx: {e}")


def tree_merge_datasets(datasets):
    """
    Merge a list of datasets using a tree-based approach for efficiency.
    Pairs datasets (0+1, 2+3, etc.) and recursively merges until single dataset.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        List of datasets to merge

    Returns
    -------
    xarray.Dataset
        Single merged dataset
    """
    if len(datasets) == 0:
        return None
    if len(datasets) == 1:
        return datasets[0]

    # Pair up datasets and merge
    next_level = []
    for i in range(0, len(datasets), 2):
        if i + 1 < len(datasets):
            # Merge pair
            merged = merge_two_datasets(datasets[i], datasets[i + 1])
            next_level.append(merged)
        else:
            # Odd number of datasets, carry forward the last one
            next_level.append(datasets[i])

    # Recursively merge the next level
    return tree_merge_datasets(next_level)


def process_directory(input_dir, output_file, year, logger):
    """
    Process all NetCDF files in a directory and merge them into a single file.

    Parameters
    ----------
    input_dir : Path
        Input directory containing site NetCDF files
    output_file : Path
        Output file path for merged dataset
    year : int
        Field season represented by every input file in this directory.
    logger : logging.Logger
        Logger instance
    """
    logger.info(subheader(f"Processing: {input_dir.name}"))

    # Find all NetCDF files in the directory, excluding hidden files
    nc_files = [f for f in input_dir.glob("*.nc") if not f.name.startswith('.')]

    if not nc_files:
        logger.info(f"No NetCDF files found in {input_dir}")
        return

    logger.info(f"Found {len(nc_files)} files to merge")

    # Load all datasets
    datasets = []
    site_names = []

    for nc_file in tqdm(nc_files, desc="Loading datasets"):
        try:
            ds = xr.open_dataset(nc_file)
            ds = ensure_season_year_coordinate(ds, year, source_name=str(nc_file))
            datasets.append(ds)

            # Extract site name from filename or dataset attributes
            if 'site_name' in ds.attrs and ds.attrs['site_name'].strip():
                site_name = ds.attrs['site_name']
            else:
                site_name = nc_file.stem  # Use filename without extension

            # Handle empty site names
            if not site_name or site_name.strip() == '':
                site_name = f"unknown_{nc_file.stem}"

            site_names.append(site_name)

        except Exception as e:
            logger.warning(f"Failed to load {nc_file}: {e}")
            continue

    if not datasets:
        logger.warning(f"No valid datasets loaded from {input_dir}")
        return

    # Add site_id coordinate to each dataset if not present
    # Calculate maximum site name length to prevent truncation
    max_site_name_len = max(len(name) for name in site_names) if site_names else 10
    max_site_name_len = max(max_site_name_len, 10)  # Minimum 10 characters

    for i, (ds, site_name) in enumerate(zip(datasets, site_names)):
        if 'site_id' not in ds.coords:
            # Create site_id coordinate with adequate string length
            site_id_dtype = f'U{max_site_name_len}'
            site_id_values = np.array([site_name] * len(ds.sensor_idx), dtype=site_id_dtype)
            datasets[i] = ds.assign_coords(site_id=('sensor_idx', site_id_values))

    # Merge all datasets using tree approach
    logger.info(f"Merging {len(datasets)} datasets using tree approach...")
    merged_ds = tree_merge_datasets(datasets)

    if merged_ds is None:
        logger.error(f"Failed to merge datasets from {input_dir}")
        return

    # Update global attributes for sensor_idx structure
    merged_ds.attrs.update({
        'processing_step': 'site_merged_lvl0_sensor_idx',
        'source_directory': str(input_dir),
        'n_sites': len(datasets),
        'n_sensors': len(merged_ds.sensor_idx) if 'sensor_idx' in merged_ds.dims else 0,
        'site_names': ', '.join(site_names),
        'structure': 'sensor_idx x datetime_utc'
    })
    merged_ds = apply_product_metadata(
        merged_ds, level="lvl0", product=output_file.stem.removeprefix("lvl0_")
    )

    # Ensure output directory exists
    output_file.parent.mkdir(parents=True, exist_ok=True)

    # Save merged dataset
    logger.info(key_value("Saving to", str(output_file)))
    merged_ds.to_netcdf(output_file, encoding=create_netcdf_encoding(merged_ds))

    # Create QC plots
    create_all_qc_plots(
        ds=merged_ds,
        output_dir=output_file.parent,
        filename_prefix=output_file.stem,
        logger=logger,
    )

    # Print summary
    n_sites = len(site_names)
    n_sensors = len(merged_ds.sensor_idx) if 'sensor_idx' in merged_ds.dims else 0
    n_times = len(merged_ds["datetime_utc"])
    logger.info(f"Successfully merged {n_sites} sites ({n_sensors} sensors) -> {n_times} time points")

    # Close datasets to free memory
    for ds in datasets:
        ds.close()
    merged_ds.close()


def main():
    """Main function to merge site data to lvl0."""
    parser = argparse.ArgumentParser(description="Merge site data to level 0 for one field season")
    parser.add_argument("--year", required=True, type=int, help="Field season to process")
    args = parser.parse_args()
    year = args.year
    # Set up logging (appends to pipeline log if running as part of pipeline)
    logger = setup_pipeline_logging(step_number=6, total_steps=8, mode="a")

    # Define paths
    base_dir = Path(ROOT) / "data" / str(year) / "intermediate" / "pendants" / "by_site"
    output_dir = Path(ROOT) / "data" / str(year) / "processed" / "lvl0"

    logger.info(key_value("Input base directory", str(base_dir)))
    logger.info(key_value("Output directory", str(output_dir)))
    logger.info(key_value("Input directory exists", str(base_dir.exists())))

    if not base_dir.exists():
        logger.error(f"Input directory {base_dir} does not exist")
        exit(1)

    # Process any files directly under by_site as standard on-ice sites.
    standard_output = output_dir / "lvl0_on_ice_standard.nc"
    process_directory(base_dir, standard_output, year, logger)

    # Output name mapping for subdirectories
    OUTPUT_NAME_MAP = {
        "on_ice": "on_ice_standard",
        "off_ice": "off_ice",
        # camp_wx stays as-is (no mapping needed)
    }

    # Discover and process all subdirectories
    subdirs = [d for d in base_dir.iterdir() if d.is_dir()]

    if subdirs:
        logger.info(f"Found {len(subdirs)} subdirectories:")
        for subdir in sorted(subdirs):
            logger.info(item(subdir.name))
            if subdir.name == "on_ice_intensive":
                logger.info(
                    "  Skipped in step 06: step 05 creates the combined Pace and pendant "
                    "Level 0 dataset."
                )

        for subdir in sorted(subdirs):
            if subdir.name == "on_ice_intensive":
                continue
            # Check if subdirectory contains any NetCDF files
            nc_files = [f for f in subdir.glob("*.nc") if not f.name.startswith('.')]

            if nc_files:
                output_name = OUTPUT_NAME_MAP.get(subdir.name, subdir.name)
                subdir_output = output_dir / f"lvl0_{output_name}.nc"
                process_directory(subdir, subdir_output, year, logger)
            else:
                logger.info(f"Skipping {subdir.name} subdirectory (no NetCDF files found)")
    else:
        logger.info("No subdirectories found to process")

    apply_filters_to_lvl0_outputs(
        output_dir=output_dir,
        mask_periods_path=Path(ROOT)
        / "data"
        / str(year)
        / "metadata"
        / "lvl0_mask_periods.csv",
        year=year,
        logger=logger,
    )


if __name__ == "__main__":
    main()
