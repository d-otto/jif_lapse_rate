#!/usr/bin/env python3
"""
Clean raw Pace logger data and convert to NetCDF format.

This script processes raw Pace logger data files from intensive monitoring sites,
parsing the complex file format and converting them to standardized NetCDF files
with CF-compliant metadata.
"""

from pathlib import Path
import sys

# Add the src directory to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from jiflr.data import clean_pace_loggers

def main():
    """Process all raw Pace logger files."""
    
    # Define paths
    raw_data_dir = Path(__file__).parent / "data" / "2025" / "raw" / "pace"
    output_dir = Path(__file__).parent / "data" / "2025" / "intermediate" / "pace"
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all .txt files in the raw data directory
    pace_files = list(raw_data_dir.glob("*.txt"))
    
    if not pace_files:
        print("No .txt files found in pace raw data directory")
        print(f"Searched in: {raw_data_dir}")
        return
    
    print(f"Found {len(pace_files)} Pace logger files:")
    for f in pace_files:
        print(f"  - {f.name}")
    
    print(f"\nProcessing files...")
    print(f"Input directory: {raw_data_dir}")
    print(f"Output directory: {output_dir}")
    
    try:
        # Process all files
        clean_pace_loggers(pace_files, output_dir, convert_to_local_tz=False)
        
        print("\n✓ Successfully processed all Pace logger files!")
        
        # List output files
        output_files = list(output_dir.glob("*.nc"))
        print(f"\nOutput files created ({len(output_files)}):")
        for f in sorted(output_files):
            print(f"  - {f.name}")
            
    except Exception as e:
        print(f"\n✗ Error processing files: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()