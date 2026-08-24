#!/usr/bin/env python
"""
Example script showing how to add wind_dir_offset_deg column to deployment_periods.csv

This is a one-time setup. After running this, you can edit the CSV directly to adjust offsets.
"""

import pandas as pd
from pathlib import Path

# Path to deployment_periods.csv
csv_path = Path("data/2025/metadata/deployment_periods.csv")

# Read existing CSV
df = pd.read_csv(csv_path)

# Check if column already exists
if 'wind_dir_offset_deg' in df.columns:
    print(f"✓ Column 'wind_dir_offset_deg' already exists in {csv_path}")
    print("\nCurrent offsets:")
    print(df[['site', 'wind_dir_offset_deg']].to_string(index=False))
else:
    # Add the column with NaN (empty) values by default
    df['wind_dir_offset_deg'] = pd.NA

    # Example: Set offsets for intensive sites
    # Modify these values based on your field calibration data
    example_offsets = {
        'Lee1': 15.0,      # Example: 15° clockwise correction
        'Lee2': 10.5,      # Example: 10.5° clockwise correction
        'Windward1': -20.0, # Example: 20° counterclockwise correction
        'Windward2': 15.0,  # Example: 15° clockwise correction
        'Divide': pd.NA,    # Example: No correction needed (leave as NA)
    }

    # Apply the example offsets
    for site, offset in example_offsets.items():
        mask = df['site'] == site
        if mask.any():
            df.loc[mask, 'wind_dir_offset_deg'] = offset
            print(f"Set {site} offset to {offset}°")

    # Save the updated CSV
    df.to_csv(csv_path, index=False)
    print(f"\n✓ Added 'wind_dir_offset_deg' column to {csv_path}")
    print("\nNOTE: These are EXAMPLE values. Edit the CSV file to set actual offsets")
    print("based on your field calibration data.")

print("\nNext steps:")
print("1. Edit data/2025/metadata/deployment_periods.csv to set actual offset values")
print("2. Run: uv run scripts/data_pipeline/01_clean_raw_pace.py")
print("3. Check logs for correction messages")
