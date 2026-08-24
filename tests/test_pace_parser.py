#!/usr/bin/env python3
"""
Quick test script for the Pace logger parsing function.
"""

from pathlib import Path
import sys

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from jiflr.pipeline import clean_pace_loggers

# Define paths
pace_data_dir = Path(__file__).parent / "data" / "2025" / "raw" / "pace"
output_dir = Path(__file__).parent / "data" / "2025" / "processed" / "pace"

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

# Get list of .txt files
pace_files = list(pace_data_dir.glob("*.txt"))

if not pace_files:
    print("No .txt files found in pace data directory")
    

print(f"Found {len(pace_files)} Pace logger files:")
for f in pace_files:
    print(f"  - {f.name}")

# Test with all files
print(f"\nTesting all files:")

try:
    clean_pace_loggers(pace_files, output_dir, convert_to_local_tz=False)
    print("✓ Successfully processed all Pace logger files!")
    
    # List output files
    output_files = list(output_dir.glob("*.nc"))
    print(f"Output files created: {len(output_files)}")
    for f in output_files:
        print(f"  - {f.name}")
        
except Exception as e:
    print(f"✗ Error processing files: {e}")
    import traceback
    traceback.print_exc()