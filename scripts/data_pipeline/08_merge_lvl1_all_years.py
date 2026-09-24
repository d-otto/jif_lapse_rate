#!/usr/bin/env python3
"""Merge seasonal Level 1 products into all-years Level 1 datasets."""

import argparse
import sys
import traceback

from jiflr import ROOT
from jiflr.logging import item, key_value, setup_pipeline_logging, subheader
from jiflr.pipeline import merge_lvl1_all_years


def main() -> None:
    """Create all-years Level 1 products from complete seasonal datasets."""
    parser = argparse.ArgumentParser(description="Merge Level 1 datasets across field seasons")
    parser.add_argument(
        "--years",
        required=True,
        type=int,
        nargs="+",
        help="Field seasons to merge, for example: --years 2025 2026",
    )
    args = parser.parse_args()
    years = sorted(set(args.years))

    logger = setup_pipeline_logging(step_number=8, total_steps=8, mode="a")
    logger.info(key_value("Years", ", ".join(str(year) for year in years)))
    logger.info(key_value("Output directory", str(ROOT / "data" / "all_years" / "processed" / "lvl1")))

    try:
        output_paths = merge_lvl1_all_years(years, ROOT / "data")
    except Exception as error:
        logger.error(f"Failed to merge Level 1 seasons: {error}")
        logger.debug(traceback.format_exc())
        sys.exit(1)

    logger.info(subheader("All-years outputs"))
    for output_path in output_paths:
        logger.info(item(output_path.name))


if __name__ == "__main__":
    main()
