#!/usr/bin/env python3
"""
Simple script to run the complete JIFLR data processing pipeline.

This script executes all 6 pipeline steps in sequence:
1. Clean raw Pace data
2. Clean raw pendant data
3. Merge pendant data by site
4. Combine Pace and pendant data
5. Merge site data to lvl0
6. Process lvl0 to lvl1

Usage:
    python scripts/data_pipeline/run_pipeline.py
"""

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
]


def run_step(step_num, script_name, description, total_steps, logger, env):
    """Run a pipeline step and handle errors."""
    logger.info("")
    logger.info(header(description, step_number=step_num, total_steps=total_steps))

    script_path = SCRIPT_DIR / script_name
    logger.info(key_value("Script", str(script_path)))

    try:
        subprocess.run(
            [sys.executable, str(script_path)],
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
        success = run_step(step_num, script_name, description, total_steps, logger, env)
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
