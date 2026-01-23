#!/usr/bin/env python3
"""
Clean raw Pace logger data and convert to NetCDF format.

This script processes raw Pace logger data files from intensive monitoring sites,
parsing the complex file format and converting them to standardized NetCDF files
with CF-compliant metadata.
"""

import sys
import traceback

from jiflr import ROOT
from jiflr.logging import item, key_value, setup_pipeline_logging, subheader
from jiflr.pipeline import clean_pace_loggers


def main():
    """Process all raw Pace logger files."""
    # Set up logging (appends to pipeline log if running as part of pipeline)
    logger = setup_pipeline_logging(step_number=1, total_steps=6, mode="a")

    # Define paths using ROOT from jiflr
    raw_data_dir = ROOT / "data" / "2025" / "raw" / "pace"
    output_dir = ROOT / "data" / "2025" / "intermediate" / "pace"
    deployment_metadata_path = ROOT / "data" / "2025" / "metadata" / "deployment_periods.csv"

    logger.info(key_value("Input directory", str(raw_data_dir)))
    logger.info(key_value("Output directory", str(output_dir)))
    logger.info(key_value("Deployment metadata", str(deployment_metadata_path)))

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find all .txt files in the raw data directory
    pace_files = list(raw_data_dir.glob("*.txt"))

    if not pace_files:
        logger.warning("No .txt files found in pace raw data directory")
        logger.warning(f"Searched in: {raw_data_dir}")
        return

    logger.info(subheader("Input files"))
    logger.info(f"Found {len(pace_files)} Pace logger files:")
    for f in pace_files:
        logger.info(item(f.name))

    try:
        # Process all files
        logger.info(subheader("Processing"))
        clean_pace_loggers(
            pace_files,
            output_dir,
            convert_to_local_tz=False,
            deployment_metadata_path=deployment_metadata_path,
        )

        logger.info(subheader("Output files"))
        output_files = list(output_dir.glob("*.nc"))
        logger.info(f"Created {len(output_files)} NetCDF files:")
        for f in sorted(output_files):
            logger.info(item(f.name))

    except Exception as e:
        logger.error(f"Error processing files: {e}")
        logger.debug(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()
