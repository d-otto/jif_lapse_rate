#!/usr/bin/env python3
"""
Process JIFLR level 0 data to level 1.

This script converts processed lvl0 data to lvl1 by:
1. Resampling all data to regular 5-minute intervals using mean aggregation
2. Creating one Level 1 file for each Level 0 product while preserving its product name

Usage:
    python scripts/data_pipeline/process_lvl0_to_lvl1.py
"""

import argparse
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import xarray as xr

from jiflr import ROOT
from jiflr.logging import indent, key_value, setup_pipeline_logging, subheader
from jiflr.netcdf_metadata import apply_product_metadata
from jiflr.pipeline import create_netcdf_encoding, ensure_season_year_coordinate
from jiflr.qc_plots import create_all_qc_plots
from jiflr.utils import butterworth_filter


QC_FLAG_SUFFIX = "_qc_flag"
PRESSURE_LOWPASS_CUTOFF_PERIOD_MINUTES = 480
PRESSURE_LOWPASS_CUTOFF_PERIOD_SECONDS = PRESSURE_LOWPASS_CUTOFF_PERIOD_MINUTES * 60
LEVEL1_SAMPLE_PERIOD_SECONDS = 5 * 60
PRESSURE_LOWPASS_ORDER = 5


def _qc_flag_variables(ds):
    """Return the time-indexed Level 0 QC flag variables in *ds*."""
    return [
        name
        for name, data_array in ds.data_vars.items()
        if name.endswith(QC_FLAG_SUFFIX)
        and {"sensor_idx", "datetime_utc"}.issubset(data_array.dims)
        and data_array.attrs.get("qc_masks_measurement", "true") == "true"
    ]


def _bitwise_or_reduce(values, axis=None, **_):
    """Aggregate QC bitfields without discarding any reason bit."""
    return np.bitwise_or.reduce(values, axis=axis)


def apply_qc_flag_masks(ds):
    """Mask each measurement whose paired QC bitfield is nonzero."""
    result = ds.copy()
    for flag_name in _qc_flag_variables(ds):
        variable = flag_name.removesuffix(QC_FLAG_SUFFIX)
        if variable not in result.data_vars:
            raise ValueError(
                f"QC flag {flag_name} has no corresponding measurement variable {variable}"
            )
        result[variable] = result[variable].where(result[flag_name] == 0)
    return result


def apply_signal_processing_filters(
    ds,
    *,
    pressure_lowpass_cutoff_period_seconds=PRESSURE_LOWPASS_CUTOFF_PERIOD_SECONDS,
    pressure_lowpass_order=PRESSURE_LOWPASS_ORDER,
):
    """Apply the legacy Butterworth pressure diagnostic filter.

    This function supports ``plot_pressure_filter_diagnostic.py`` only. The
    Level 1 production masks Level 0 noise candidates without interpolation.
    """
    if "pressure" not in ds.data_vars:
        return ds
    if "sensor_type" not in ds.coords:
        raise ValueError(
            "Cannot filter pressure: dataset has pressure but no sensor_type coordinate"
        )
    if not {"sensor_idx", "datetime_utc"}.issubset(ds["pressure"].dims):
        raise ValueError(
            "Cannot filter pressure: pressure must have sensor_idx and datetime_utc dimensions"
        )
    if ds["sensor_type"].dims != ("sensor_idx",):
        raise ValueError(
            "Cannot filter pressure: sensor_type must be indexed by sensor_idx"
        )

    if pressure_lowpass_cutoff_period_seconds <= 0:
        raise ValueError("Pressure low-pass cutoff period must be positive")
    if pressure_lowpass_order < 1:
        raise ValueError("Pressure low-pass filter order must be at least one")

    result = ds.copy()
    pressure = result["pressure"]
    pressure_by_sensor = pressure.transpose("sensor_idx", "datetime_utc")
    filtered_values = pressure_by_sensor.values.copy()
    sampling_frequency = 1 / LEVEL1_SAMPLE_PERIOD_SECONDS
    cutoff_frequency = 1 / pressure_lowpass_cutoff_period_seconds

    for position, sensor_type in enumerate(result["sensor_type"].values):
        pressure_values = filtered_values[position]
        is_pace_sensor = str(sensor_type).casefold().startswith("pace")
        if is_pace_sensor and not np.all(np.isnan(pressure_values)):
            filtered_values[position] = butterworth_filter(
                pressure_values,
                fs=sampling_frequency,
                order=pressure_lowpass_order,
                upper=cutoff_frequency,
            )

    filtered_pressure = xr.DataArray(
        filtered_values,
        coords=pressure_by_sensor.coords,
        dims=pressure_by_sensor.dims,
        attrs=pressure.attrs.copy(),
    ).transpose(*pressure.dims)
    filtered_pressure.attrs.update(
        {
            "signal_processing_filter": "Butterworth low-pass (zero-phase)",
            "signal_processing_filter_method": "scipy.signal.sosfiltfilt",
            "signal_processing_filter_order": pressure_lowpass_order,
            "signal_processing_filter_sensor_type": "pace",
            "signal_processing_filter_cutoff_period_seconds": (
                pressure_lowpass_cutoff_period_seconds
            ),
        }
    )
    result["pressure"] = filtered_pressure
    return result


def resample_to_5min(ds):
    """
    Resample dataset to regular 5-minute intervals.

    This function resamples data to 5-minute intervals using mean aggregation,
    which preserves data from sensors that are already on 5-minute schedules
    but at different minute offsets (e.g., :13, :18, :23 vs :05, :10, :15).

    Parameters
    ----------
    ds : xarray.Dataset
        Input dataset with a UTC datetime_utc coordinate

    Returns
    -------
    xarray.Dataset
        Dataset resampled to 5-minute intervals
    """
    # Resample to 5-minute intervals using mean aggregation
    # This preserves existing data points that fall on 5-minute boundaries
    # regardless of their minute offset from standard boundaries
    if "datetime_utc" not in ds.coords:
        raise ValueError("Level 0 dataset does not have a datetime_utc coordinate")
    qc_flag_variables = _qc_flag_variables(ds)
    measurements = ds.drop_vars(qc_flag_variables)
    ds_resampled = measurements.resample(datetime_utc="5min").mean()
    for flag_name in qc_flag_variables:
        variable_name = flag_name.removesuffix(QC_FLAG_SUFFIX)
        if variable_name not in ds_resampled.data_vars:
            raise ValueError(
                f"QC flag {flag_name} has no resampled measurement {variable_name}"
            )
        flags = (
            ds[flag_name]
            .astype(np.uint32)
            .resample(datetime_utc="5min")
            .reduce(_bitwise_or_reduce)
            # A bin with no source timestamps has no QC reason bits.
            .fillna(0)
            .astype(np.uint32)
        )
        flags = flags.transpose(*ds_resampled[variable_name].dims)
        flags.attrs = ds[flag_name].attrs.copy()
        ds_resampled[flag_name] = flags

    # Update processing step attribute
    attrs = ds.attrs.copy()
    attrs["processing_step"] = "lvl1_5min_resampled"
    attrs["resampling_method"] = "mean"
    attrs["time_resolution"] = "5 minutes"
    attrs["processed_timestamp"] = datetime.now(timezone.utc).isoformat()

    ds_resampled.attrs = attrs

    return ds_resampled


def process_to_5min(
    ds,
):
    """Resample Level 0 data and mask every QC-flagged measurement."""
    return apply_qc_flag_masks(resample_to_5min(ds))


def process_individual_file(
    input_path,
    output_dir,
    year,
    logger,
):
    """
    Process a single lvl0 file to lvl1.

    Parameters
    ----------
    input_path : Path
        Path to input lvl0 NetCDF file
    output_dir : Path
        Output directory for processed file
    year : int
        Field season represented by this input file.
    logger : logging.Logger
        Logger instance

    Returns
    -------
    xarray.Dataset
        Processed dataset
    """
    logger.info(indent(f"Processing {input_path.name}..."))

    # Load the dataset
    ds = xr.open_dataset(input_path)
    ds = ensure_season_year_coordinate(ds, year, source_name=str(input_path))

    ds_lvl1 = process_to_5min(ds)

    # Generate output filename
    output_filename = input_path.name.replace("lvl0_", "lvl1_")
    output_path = output_dir / output_filename
    ds_lvl1 = apply_product_metadata(
        ds_lvl1, level="lvl1", product=output_path.stem.removeprefix("lvl1_")
    )

    # Save processed file
    ds_lvl1.to_netcdf(output_path, encoding=create_netcdf_encoding(ds_lvl1))
    logger.info(indent(f"Saved {output_filename}", level=2))

    # Create QC plots
    create_all_qc_plots(
        ds=ds_lvl1,
        output_dir=output_dir,
        filename_prefix=output_filename.replace(".nc", ""),
        logger=logger,
    )
    ds.close()

    return ds_lvl1


def combine_datasets(datasets, output_path, logger):
    """
    Combine multiple datasets into a single file using sensor_idx concatenation.

    Parameters
    ----------
    datasets : list of xarray.Dataset
        List of datasets to combine (should have sensor_idx structure)
    output_path : Path
        Path for combined output file
    logger : logging.Logger
        Logger instance
    """
    logger.info("Combining datasets using sensor_idx concatenation...")

    if len(datasets) == 1:
        # Only one dataset, just copy it
        combined = datasets[0].copy()
        logger.info(indent(f"Single dataset with {len(combined.sensor_idx)} sensors"))
    else:
        # Multiple datasets - concatenate along sensor_idx dimension
        logger.info(
            indent(f"Concatenating {len(datasets)} datasets along sensor_idx dimension")
        )

        try:
            # Concatenate along sensor_idx dimension
            combined = xr.concat(
                datasets, dim="sensor_idx", data_vars="all", coords="all", join="outer"
            )
            # Fix sensor_idx to be sequential (0, 1, 2, 3...) instead of all zeros
            new_sensor_idx = list(range(len(combined.sensor_idx)))
            combined = combined.assign_coords(sensor_idx=new_sensor_idx)

            # Get summary statistics
            total_sensors = len(combined.sensor_idx)
            unique_sites = (
                sorted(set(combined.site_id.values))
                if "site_id" in combined.coords
                else []
            )

            logger.info(
                indent(
                    f"Successfully combined {total_sensors} sensors from {len(unique_sites)} sites"
                )
            )
            logger.info(indent(f"Sites: {unique_sites}"))

        except Exception as e:
            logger.warning(f"Could not concatenate datasets along sensor_idx: {e}")
            logger.info("Trying alternative concatenation method...")

            # Fallback: try concatenating with minimal options
            try:
                combined = xr.concat(datasets, dim="sensor_idx")
                # Fix sensor_idx to be sequential (0, 1, 2, 3...) instead of all zeros
                new_sensor_idx = list(range(len(combined.sensor_idx)))
                combined = combined.assign_coords(sensor_idx=new_sensor_idx)
                logger.info(
                    indent(
                        f"Fallback concatenation successful with {len(combined.sensor_idx)} sensors"
                    )
                )
            except Exception as e2:
                logger.error(f"Fallback concatenation also failed: {e2}")
                logger.info("Saving first dataset only as combined file")
                combined = datasets[0].copy()

    # Update attributes for combined file
    attrs = combined.attrs.copy()
    attrs["title"] = "Combined JIFLR Level 1 Data"
    attrs["processing_step"] = "lvl1_combined_sensor_idx"
    attrs["n_source_files"] = len(datasets)
    attrs["combined_timestamp"] = datetime.now().isoformat()
    attrs["structure"] = "sensor_idx x datetime_utc"

    if "sensor_idx" in combined.dims:
        attrs["n_sensors"] = len(combined.sensor_idx)

    combined.attrs = attrs
    combined = apply_product_metadata(combined, level="lvl1", product="combined")

    # Save combined file
    combined.to_netcdf(output_path, encoding=create_netcdf_encoding(combined))
    logger.info(key_value("Saved combined file", output_path.name))

    return combined


def main():
    """Main processing function."""
    # Set up logging (appends to pipeline log if running as part of pipeline)
    logger = setup_pipeline_logging(step_number=7, total_steps=8, mode="a")

    parser = argparse.ArgumentParser(description="Process JIFLR lvl0 data to lvl1")
    parser.add_argument(
        "--year",
        type=int,
        required=True,
        help="Field season to process",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="Input directory containing lvl0 files",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for lvl1 files",
    )
    args = parser.parse_args()

    # Set up paths
    input_dir = (
        ROOT / args.input_dir
        if args.input_dir
        else ROOT / "data" / str(args.year) / "processed" / "lvl0"
    )
    output_dir = (
        ROOT / args.output_dir
        if args.output_dir
        else ROOT / "data" / str(args.year) / "processed" / "lvl1"
    )

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(key_value("Input directory", str(input_dir)))
    logger.info(key_value("Output directory", str(output_dir)))

    # Find all lvl0 NetCDF files
    lvl0_files = list(input_dir.glob("lvl0_*.nc"))

    if not lvl0_files:
        logger.warning("No lvl0 files found!")
        return

    logger.info(f"Found {len(lvl0_files)} lvl0 files to process")

    # Process each file individually
    logger.info(subheader("Processing individual files"))
    processed_datasets = []

    for lvl0_file in lvl0_files:
        try:
            ds_lvl1 = process_individual_file(
                lvl0_file,
                output_dir,
                args.year,
                logger,
            )
            processed_datasets.append(ds_lvl1)
        except Exception as e:
            raise RuntimeError(f"Error processing {lvl0_file.name}: {e}") from e

    # Close all datasets
    for ds in processed_datasets:
        ds.close()

    logger.info(subheader("Summary"))
    logger.info(key_value("Files processed", str(len(processed_datasets))))
    logger.info(key_value("Output directory", str(output_dir)))


if __name__ == "__main__":
    main()
