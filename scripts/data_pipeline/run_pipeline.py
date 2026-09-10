#!/usr/bin/env python3
"""
Simple script to run the complete JIFLR data processing pipeline.

This script executes six seasonal steps and, optionally, a seventh all-years
merge step:
1. Clean raw Pace data
2. Clean raw pendant data
3. Merge pendant data by site
4. Combine Pace and pendant data
5. Merge site data to lvl0
6. Process lvl0 to lvl1
7. Merge selected seasonal lvl1 datasets into all-years products

Usage:
    python scripts/data_pipeline/run_pipeline.py
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from jiflr.logging import (
    LOG_PATH_ENV_VAR,
    footer,
    header,
    key_value,
    pipeline_footer,
    pipeline_header,
    setup_pipeline_logging,
)

# Get script directory for running other scripts
SCRIPT_DIR = Path(__file__).parent
LOG_FILE = SCRIPT_DIR / "pipeline.log"

# Define pipeline steps
STEPS = [
    (1, "01_clean_raw_pace.py", "Clean raw Pace data"),
    (2, "02_clean_raw_pendants.py", "Clean raw pendant data"),
    (3, "03_merge_raw_pendants_by_site.py", "Merge pendant data by site"),
    (4, "04_add_pendants_to_intensive.py", "Add pendants to intensive sites"),
    (5, "05_merge_intermediate_to_lvl0.py", "Merge intermediate to lvl0"),
    (6, "06_lvl0_to_lvl1.py", "Process lvl0 to lvl1"),
    (7, "07_merge_lvl1_all_years.py", "Merge Level 1 data across years"),
]


def run_step(step_num, script_name, description, total_steps, logger, env, year, all_years):
    """Run a pipeline step and handle errors."""
    logger.info("")
    logger.info(header(description, step_number=step_num, total_steps=total_steps))

    script_path = SCRIPT_DIR / script_name
    logger.info(key_value("Script", str(script_path)))

    try:
        if script_name == "07_merge_lvl1_all_years.py":
            command = [sys.executable, str(script_path), "--years", *(str(value) for value in all_years)]
        else:
            command = [sys.executable, str(script_path), "--year", str(year)]
        subprocess.run(
            command,
            check=True,
            env=env,
        )
        logger.info(footer(success=True))
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"Exit code: {e.returncode}")
        logger.info(footer(success=False))
        return False


def main():
    """Run the complete pipeline."""
    parser = argparse.ArgumentParser(description="Run the JIFLR pipeline for one field season")
    parser.add_argument("--year", required=True, type=int, help="Field season to process")
    parser.add_argument(
        "--all-years",
        type=int,
        nargs="+",
        help="Run step 07 after seasonal processing, merging these Level 1 years",
    )
    args = parser.parse_args()
    total_steps = len(STEPS)

    # Initialize logging with overwrite mode (fresh start)
    logger = setup_pipeline_logging(
        log_file=LOG_FILE,
        mode="w",  # Overwrite existing log
    )

    # Print pipeline header
    logger.info(pipeline_header(log_file=LOG_FILE))

    # Set up environment for subprocess scripts
    env = os.environ.copy()
    env[LOG_PATH_ENV_VAR] = str(LOG_FILE)

    # Run each step
    all_success = True
    for step_num, script_name, description in STEPS:
        if script_name == "07_merge_lvl1_all_years.py" and not args.all_years:
            continue
        success = run_step(
            step_num, script_name, description, total_steps, logger, env,
            args.year, args.all_years,
        )
        if not success:
            logger.error(f"\nPipeline failed at step {step_num}")
            all_success = False
            break

    # Print pipeline footer
    logger.info(pipeline_footer(success=all_success))

    if not all_success:
        sys.exit(1)


if __name__ == "__main__":
    main()
