# Scripts

This directory contains executable workflow and processing scripts for the JIFLR project.

## Purpose

Scripts here are **executable workflows** - they're meant to be run directly to process data, generate outputs, or execute complete analyses. These are distinct from:
- `src/jiflr/scripts/` - Package utility scripts (e.g., data download helpers)
- `notebooks/` - Exploratory analysis and visualization
- `analyses/` - Formal analysis workflows with outputs

## Available Scripts

### process_raw_pendant_data.py

Processes raw HOBO pendant CSV exports to NetCDF format.

**When to run:** After adding new CSV files to `data/{year}/raw_exported/`

**Usage:**
```bash
# Process current year (2025)
python scripts/process_raw_pendant_data.py

# Process specific year
python scripts/process_raw_pendant_data.py --year 2024

# Force reprocessing of all files
python scripts/process_raw_pendant_data.py --force
```

**Input:** CSV files from `data/{year}/raw_exported/`
**Output:** NetCDF files to `data/{year}/intermediate/pendants/`

## Adding New Scripts

When adding new processing scripts here, consider:
1. Is this a reusable workflow that processes data?
2. Does it have clear inputs and outputs?
3. Should it be run when new data arrives?

If yes → put it here
If it's exploratory → use `notebooks/`
If it's a formal analysis → use `analyses/`
If it's a package utility → use `src/jiflr/scripts/`
