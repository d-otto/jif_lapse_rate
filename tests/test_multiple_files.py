#!/usr/bin/env python3
"""
Test script for multiple file types
"""

from pathlib import Path
import tempfile
import sys
sys.path.append('/Users/drotto/src/jiflr/src')

from jiflr.pipeline import clean_hobo_pendants
from jiflr.io import read_pendant_dataset

def test_multiple_files():
    """Test the metadata parsing with different file types"""
    
    test_files = [
        Path('/Users/drotto/src/jiflr/data/2025/raw_exported/A01 1m 2025-07-27 10_22_50 AKDT (Data AKDT).csv'),
        Path('/Users/drotto/src/jiflr/data/2025/raw_exported/A02 unshielded 2025-07-09 23_16_13 AKDT (Data AKDT).csv'),
        Path('/Users/drotto/src/jiflr/data/2025/raw_exported/B01 2m 2025-07-27 09_58_55 AKDT (Data AKDT).csv')
    ]
    
    # Filter to existing files
    existing_files = [f for f in test_files if f.exists()]
    
    if not existing_files:
        print("No test files found")
        return
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        print(f"Testing metadata parsing with {len(existing_files)} files")
        
        # Test clean_hobo_pendants on all files
        try:
            clean_hobo_pendants(existing_files, temp_path)
            print("✓ clean_hobo_pendants completed successfully for all files")
            
            # Check output files
            output_files = list(temp_path.glob('*.nc'))
            print(f"✓ Created {len(output_files)} output files")
            
            if output_files:
                # Test read_hobo_pendants with all files
                datasets = read_hobo_pendants(output_files)
                print("✓ read_hobo_pendants completed successfully")
                
                print(f"\nCombined dataset shape: {datasets.dims}")
                
                print("\nSite metadata for all sensors:")
                site_metadata = datasets.attrs.get('site_metadata', {})
                for sensor_id, metadata in site_metadata.items():
                    print(f"  {sensor_id}: {metadata}")
                
                print("\nCoordinates:")
                for coord in ['sensor_id', 'site_name', 'sensor_height', 'sensor_config']:
                    if coord in datasets.coords:
                        values = datasets.coords[coord].values
                        if hasattr(values, '__len__') and len(values) > 10:
                            print(f"  {coord}: {values[:3]}... ({len(values)} total)")
                        else:
                            print(f"  {coord}: {values}")
                    
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_multiple_files()