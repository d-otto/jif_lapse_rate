#!/usr/bin/env python3
"""
Test script for the updated metadata parsing functionality
"""

from pathlib import Path
import tempfile
import sys
sys.path.append('/Users/drotto/src/jiflr/src')

from jiflr.pipeline import clean_hobo_pendants
from jiflr.io import read_pendant_dataset

def test_metadata_parsing():
    """Test the metadata parsing with a 2025 raw_exported file"""
    
    # Test file path
    test_file = Path('/Users/drotto/src/jiflr/data/2025/raw_exported/A01 1m 2025-07-27 10_22_50 AKDT (Data AKDT).csv')
    
    if not test_file.exists():
        print(f"Test file not found: {test_file}")
        return
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print(f"Testing metadata parsing with: {test_file.name}")
        
        # Test clean_hobo_pendants
        try:
            clean_hobo_pendants(test_file, temp_path, year=2025)
            print("✓ clean_hobo_pendants completed successfully")
            
            # Check the output file
            output_files = list(temp_path.glob('*.nc'))
            if output_files:
                output_file = output_files[0]
                print(f"✓ Created output file: {output_file.name}")
                
                # Test read_hobo_pendants
                datasets = read_hobo_pendants([output_file])
                print("✓ read_hobo_pendants completed successfully")
                
                # Check metadata
                print("\nDataset attributes:")
                for key, value in datasets.attrs.items():
                    print(f"  {key}: {value}")
                
                print("\nVariable attributes for 'temp_c':")
                if 'temp_c' in datasets.data_vars:
                    for key, value in datasets['temp_c'].attrs.items():
                        print(f"  {key}: {value}")
                
                print("\nCoordinates:")
                for coord in datasets.coords:
                    print(f"  {coord}: {datasets.coords[coord].values}")
                    
            else:
                print("✗ No output file created")
                
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_metadata_parsing()
