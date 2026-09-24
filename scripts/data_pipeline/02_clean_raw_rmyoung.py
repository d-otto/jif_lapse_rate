#!/usr/bin/env python3
"""Clean raw R. M. Young logger Weather tables for one field season."""

import argparse
import sys
import traceback

from jiflr import ROOT
from jiflr.logging import item, key_value, setup_pipeline_logging, subheader
from jiflr.pipeline import clean_rmyoung_loggers


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean RM Young logger data for one field season")
    parser.add_argument("--year", required=True, type=int, help="Field season to process")
    args = parser.parse_args()
    year = args.year
    logger = setup_pipeline_logging(step_number=2, total_steps=8, mode="a")
    raw_dir = ROOT / "data" / str(year) / "raw" / "rmyoung"
    output_dir = ROOT / "data" / str(year) / "intermediate" / "rmyoung"
    manifest_path = ROOT / "data" / str(year) / "metadata" / "deployment_manifest.csv"
    weather_files = sorted(raw_dir.glob("*/*_Weather.dat"))
    logger.info(key_value("Input directory", str(raw_dir)))
    logger.info(key_value("Output directory", str(output_dir)))
    logger.info(key_value("Deployment manifest", str(manifest_path)))
    if not weather_files:
        logger.warning(f"No *_Weather.dat files found below {raw_dir}")
        return
    logger.info(subheader("Input files"))
    for path in weather_files:
        logger.info(item(str(path.relative_to(raw_dir))))
    try:
        outputs = clean_rmyoung_loggers(raw_dir, output_dir, manifest_path, year)
    except Exception as error:
        logger.error(f"Error processing RM Young files: {error}")
        logger.debug(traceback.format_exc())
        sys.exit(1)
    logger.info(subheader("Output files"))
    for path in outputs:
        logger.info(item(path.name))


if __name__ == "__main__":
    main()
