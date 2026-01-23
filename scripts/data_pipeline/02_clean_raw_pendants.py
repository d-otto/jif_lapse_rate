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
    convert_to_local_tz: bool = True,
    utc_offset_hours: float = -9.0,
    data_inventory_path: Union[str, Path, None] = None,
    deployment_metadata_path: Union[str, Path, None] = None,
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
    convert_to_local_tz : bool, optional
        If True, convert from UTC storage to local timezone (default: True).
    utc_offset_hours : float, optional
        UTC offset in hours for local timezone conversion (default: -9.0 for AKST).
    data_inventory_path : Union[str, Path, None], optional
        Path to data inventory Excel file containing shielding information (default: None).
    deployment_metadata_path : Union[str, Path, None], optional
        Path to deployment_periods.csv file containing site elevations and coordinates (default: None).

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
    logger = setup_pipeline_logging(step_number=2, total_steps=6, mode="a")

    # Convert paths to Path objects
    raw_dir = Path(raw_dir)
    output_base_dir = Path(output_base_dir)
    if data_inventory_path:
        data_inventory_path = Path(data_inventory_path)
    if deployment_metadata_path:
        deployment_metadata_path = Path(deployment_metadata_path)

    logger.info(key_value("Year", year))
    logger.info(key_value("Raw export directory", str(raw_dir)))
    logger.info(key_value("Output base directory", str(output_base_dir)))
    if data_inventory_path:
        logger.info(key_value("Data inventory", str(data_inventory_path)))
    if deployment_metadata_path:
        logger.info(key_value("Deployment metadata", str(deployment_metadata_path)))

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

        # Process CSV files for this subdirectory with optional timezone conversion
        if convert_to_local_tz:
            logger.info(f"Converting to local timezone (UTC{utc_offset_hours:+.1f})")

        clean_hobo_pendants(
            csv_files,
            output_dir,
            convert_to_local_tz=convert_to_local_tz,
            utc_offset_hours=utc_offset_hours,
            data_inventory_path=data_inventory_path,
            deployment_metadata_path=deployment_metadata_path,
        )

        # Count output files
        nc_files = sorted(output_dir.glob("*.nc"))
        tz_info = (
            f" with timezone UTC{utc_offset_hours:+.1f}"
            if convert_to_local_tz
            else " in UTC"
        )
        output_location = (
            output_dir.relative_to(output_base_dir)
            if rel_path != Path(".")
            else "root"
        )
        logger.info(f"Created {len(nc_files)} NetCDF files{tz_info} in {output_location}")

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
    year = "2025"

    # Define paths
    raw_dir = Path(ROOT) / "data" / year / "raw" / "pendants" / "exported"
    output_base_dir = (
        Path(ROOT) / "data" / year / "intermediate" / "pendants" / "by_sensor"
    )
    data_inventory_path = Path(ROOT) / "data" / year / "metadata" / "data_inventory.xlsx"
    deployment_metadata_path = (
        Path(ROOT) / "data" / year / "metadata" / "deployment_periods.csv"
    )

    main(
        raw_dir,
        output_base_dir,
        year="2025",
        force=True,
        data_inventory_path=data_inventory_path,
        deployment_metadata_path=deployment_metadata_path,
    )
