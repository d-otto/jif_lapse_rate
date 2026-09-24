#!/usr/bin/env python3
"""
process_raw_pendant_data.py

Process raw HOBO pendant CSV exports to NetCDF format.
Run this script when new raw CSV files are added to data/{year}/raw/pendants

This script:
1. Finds all CSV files in the directory
2. Converts them to standardized NetCDF format using clean_hobo_pendants()
3. Saves processed files to data/{year}/intermediate/pendants/

Created: 2025-10-01
"""

import argparse
from pathlib import Path
from typing import Union

from jiflr import ROOT
from jiflr.logging import item, key_value, setup_pipeline_logging, subheader
from jiflr.pipeline import clean_hobo_pendants


def main(
    raw_dir: Union[str, Path],
    output_base_dir: Union[str, Path],
    year: str = "2025",
    force: bool = False,
    manifest_path: Union[str, Path, None] = None,
) -> None:
    """
    Process raw HOBO pendant CSV files to NetCDF format.

    This function recursively searches for CSV files in the raw directory and its
    subdirectories, then processes them using the clean_hobo_pendants() function
    to create standardized NetCDF files with proper metadata and variable naming.

    Parameters
    ----------
    raw_dir : Union[str, Path]
        Path to directory containing raw CSV files exported from HOBOware/HOBOconnect.
        The function will search this directory and all subdirectories for *.csv files.
    output_base_dir : Union[str, Path]
        Base directory where processed NetCDF files will be saved. The directory
        structure from raw_dir will be preserved in the output.
    year : str, optional
        Year of data being processed, used for informational output (default: "2025").
    force : bool, optional
        If True, reprocess all files even if outputs already exist. Currently not
        implemented but reserved for future use (default: False).
    manifest_path : Union[str, Path, None], optional
        Path to the per-season machine-readable deployment manifest.

    Returns
    -------
    None

    Notes
    -----
    The function maintains the directory structure of the input when creating outputs.
    For example, if raw_dir contains subdirectories 'site1/' and 'site2/', the
    corresponding NetCDF files will be saved to 'output_base_dir/site1/' and
    'output_base_dir/site2/' respectively.
    """
    # Set up logging (appends to pipeline log if running as part of pipeline)
    logger = setup_pipeline_logging(step_number=3, total_steps=8, mode="a")

    # Convert paths to Path objects
    raw_dir = Path(raw_dir)
    output_base_dir = Path(output_base_dir)
    if manifest_path:
        manifest_path = Path(manifest_path)

    logger.info(key_value("Year", year))
    logger.info(key_value("Raw export directory", str(raw_dir)))
    logger.info(key_value("Output base directory", str(output_base_dir)))
    if manifest_path:
        logger.info(key_value("Deployment manifest", str(manifest_path)))

    # Find all subdirectories plus the root directory
    subdirs = [d for d in raw_dir.rglob("*") if d.is_dir()]
    subdirs.insert(0, raw_dir)  # Include root directory

    # init counters
    total_csv_files = 0
    total_processed = 0

    for subdir in subdirs:
        # Get CSV files in this specific directory (not recursive)
        csv_files = sorted(subdir.glob("*.csv"))

        if not csv_files:
            continue

        # Calculate relative path from raw_dir
        rel_path = subdir.relative_to(raw_dir)
        output_dir = output_base_dir / rel_path

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Log what we're processing
        if rel_path == Path("."):
            logger.info(subheader("Processing root directory"))
        else:
            logger.info(subheader(f"Processing subdirectory: {rel_path}/"))

        logger.info(f"Found {len(csv_files)} CSV files")

        total_csv_files += len(csv_files)

        clean_hobo_pendants(
            csv_files,
            output_dir,
            manifest_path=manifest_path,
            year=int(year),
        )

        # Count output files
        nc_files = sorted(output_dir.glob("*.nc"))
        output_location = (
            output_dir.relative_to(output_base_dir) if rel_path != Path(".") else "root"
        )
        logger.info(f"Created {len(nc_files)} NetCDF files in UTC in {output_location}")

        total_processed += len(nc_files)

    logger.info(subheader("Summary"))
    logger.info(key_value("Total CSV files found", str(total_csv_files)))
    logger.info(key_value("Total NetCDF files created", str(total_processed)))

    # Show final directory structure
    logger.info(subheader("Output directory structure"))
    for nc_file in sorted(output_base_dir.rglob("*.nc")):
        rel_path = nc_file.relative_to(output_base_dir)
        size_mb = nc_file.stat().st_size / (1024 * 1024)
        logger.info(item(f"{rel_path} ({size_mb:.2f} MB)"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Clean pendant data for one field season"
    )
    parser.add_argument(
        "--year", required=True, type=int, help="Field season to process"
    )
    args = parser.parse_args()
    year = args.year

    # Define paths
    raw_dir = Path(ROOT) / "data" / str(year) / "raw" / "pendants" / "exported"
    output_base_dir = (
        Path(ROOT) / "data" / str(year) / "intermediate" / "pendants" / "by_sensor"
    )
    manifest_path = (
        Path(ROOT) / "data" / str(year) / "metadata" / "deployment_manifest.csv"
    )

    main(
        raw_dir,
        output_base_dir,
        year=str(year),
        force=True,
        manifest_path=manifest_path,
    )
