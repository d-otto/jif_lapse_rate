#!/usr/bin/env python3
"""
Test script for the new function structure
"""

from pathlib import Path
import tempfile
import sys
sys.path.append('/Users/drotto/src/jiflr/src')

from jiflr.pipeline import clean_hobo_pendants
from jiflr.io import read_hobo_pendant, read_pendant_dataset

def test_new_structure():
    """Test the new single and multi-file reading functions"""
    
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
        
        print(f"Testing new structure with {len(existing_files)} files")
        
        # Step 1: Convert CSV to NetCDF
        try:
            clean_hobo_pendants(existing_files, temp_path, year=2025)
            print("✓ clean_hobo_pendants completed successfully")
            
            output_files = list(temp_path.glob('*.nc'))
            print(f"✓ Created {len(output_files)} output files")
            
            if not output_files:
                print("✗ No output files created")
                return
            
            # Step 2: Test single file reading
            print("\n--- Testing single file reading ---")
            single_ds = read_hobo_pendant(output_files[0])
            print(f"✓ read_hobo_pendant completed for {output_files[0].name}")
            print(f"  Dimensions: {dict(single_ds.dims)}")
            print(f"  Coordinates: {list(single_ds.coords.keys())}")
            
            # Check scalar coordinates
            if 'site_id' in single_ds.coords:
                print(f"  site_id: {single_ds.coords['site_id'].values}")
            if 'height' in single_ds.coords:
                print(f"  height: {single_ds.coords['height'].values}")
            if 'sensor_id' in single_ds.coords:
                print(f"  sensor_id: {single_ds.coords['sensor_id'].values}")
            
            # Step 3: Test multi-file reading
            print("\n--- Testing multi-file reading ---")
            multi_ds = read_pendant_dataset(output_files)
            print(f"✓ read_pendant_dataset completed for {len(output_files)} files")
            print(f"  Dimensions: {dict(multi_ds.dims)}")
            print(f"  Coordinates: {list(multi_ds.coords.keys())}")
            
            # Check concatenated coordinates
            if 'site_id' in multi_ds.coords:
                print(f"  site_id: {multi_ds.coords['site_id'].values}")
            if 'height' in multi_ds.coords:
                print(f"  height: {multi_ds.coords['height'].values}")
            if 'sensor_id' in multi_ds.coords:
                print(f"  sensor_id: {multi_ds.coords['sensor_id'].values}")
            
            # Test data access patterns
            print("\n--- Testing data access patterns ---")
            if 'temp_c' in multi_ds.data_vars:
                print(f"  Temperature data shape: {multi_ds['temp_c'].shape}")
                
                # Try site-based selection if possible
                if 'site_id' in multi_ds.coords:
                    unique_sites = multi_ds.coords['site_id'].values
                    if len(unique_sites) > 0:
                        first_site = unique_sites[0] if hasattr(unique_sites, '__getitem__') else unique_sites
                        try:
                            site_data = multi_ds.where(multi_ds.site_id == first_site, drop=True)
                            print(f"  ✓ Can filter by site_id: {first_site}")
                        except Exception as e:
                            print(f"  ⚠ Site filtering issue: {e}")
                            
        except Exception as e:
            print(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    test_new_structure()
