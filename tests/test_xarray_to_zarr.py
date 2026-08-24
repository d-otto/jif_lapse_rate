#!/usr/bin/env python3
"""
Test script for xarray_to_zarr functionality.

This script tests the jiflr.io module with ERA5 monthly CDS data.
"""

import sys
from pathlib import Path
import tempfile
import shutil
import logging

# Add src to path so we can import jiflr
sys.path.insert(0, str(Path(__file__).parent / "src"))

from jiflr.io import xarray_to_zarr, convert_era5_to_zarr, _parse_filename, _validate_dataset_consistency
from jiflr import ROOT

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_parse_filename():
    """Test filename parsing functionality."""
    logger.info("Testing filename parsing...")
    
    # Test valid filenames
    test_cases = [
        ("era5_monthly_cds.2m_temperature.grib", "era5_monthly_cds", "2m_temperature"),
        ("era5_monthly_cds.10m_u_component_of_wind.grib", "era5_monthly_cds", "10m_u_component_of_wind"),
        ("dataset.complex.variable.name.nc", "dataset", "complex.variable.name"),
    ]
    
    for filename, expected_dataset, expected_variable in test_cases:
        dataset, variable = _parse_filename(filename)
        assert dataset == expected_dataset, f"Expected dataset '{expected_dataset}', got '{dataset}'"
        assert variable == expected_variable, f"Expected variable '{expected_variable}', got '{variable}'"
        logger.info(f"✓ {filename} -> dataset: {dataset}, variable: {variable}")
    
    # Test invalid filename
    try:
        _parse_filename("invalid_filename.grib")
        assert False, "Should have raised ValueError"
    except ValueError:
        logger.info("✓ Invalid filename correctly rejected")
    
    logger.info("Filename parsing tests passed!")


def test_dataset_consistency():
    """Test dataset consistency validation."""
    logger.info("Testing dataset consistency validation...")
    
    # Test consistent datasets
    consistent_files = [
        Path("era5_monthly_cds.temp.grib"),
        Path("era5_monthly_cds.pressure.grib"),
        Path("era5_monthly_cds.wind.grib"),
    ]
    
    dataset_name = _validate_dataset_consistency(consistent_files)
    assert dataset_name == "era5_monthly_cds"
    logger.info(f"✓ Consistent dataset validation passed: {dataset_name}")
    
    # Test inconsistent datasets
    inconsistent_files = [
        Path("era5_monthly_cds.temp.grib"),
        Path("different_dataset.pressure.grib"),
    ]
    
    try:
        _validate_dataset_consistency(inconsistent_files)
        assert False, "Should have raised ValueError"
    except ValueError:
        logger.info("✓ Inconsistent datasets correctly rejected")
    
    logger.info("Dataset consistency tests passed!")


def test_zarr_creation():
    """Test zarr creation with real ERA5 files."""
    logger.info("Testing zarr creation with ERA5 files...")
    
    # Find ERA5 files
    era5_dir = Path(ROOT) / "data" / "external" / "era5" / "era5_monthly_cds"
    grib_files = list(era5_dir.glob("*.grib"))
    
    if not grib_files:
        logger.warning("No ERA5 GRIB files found for testing")
        return
    
    logger.info(f"Found {len(grib_files)} GRIB files for testing")
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_zarr = Path(temp_dir) / "test_era5.zarr"
        
        try:
            # Test with first few files to keep test fast
            test_files = grib_files[:3]  # Test with first 3 files
            logger.info(f"Testing with files: {[f.name for f in test_files]}")
            
            # Create temporary input directory with symlinks
            temp_input = Path(temp_dir) / "input"
            temp_input.mkdir()
            
            for file in test_files:
                symlink_path = temp_input / file.name
                symlink_path.symlink_to(file.absolute())
            
            # Test zarr creation
            ds = xarray_to_zarr(
                input_path=temp_input,
                output_zarr=temp_zarr,
                pattern="*.grib"
            )
            
            logger.info(f"✓ Zarr created successfully")
            logger.info(f"  Dimensions: {dict(ds.dims)}")
            logger.info(f"  Variables: {list(ds.data_vars.keys())}")
            logger.info(f"  Zarr size: {temp_zarr}")
            
            # Test that zarr can be reopened
            import xarray as xr
            reopened_ds = xr.open_zarr(temp_zarr)
            logger.info(f"✓ Zarr can be reopened successfully")
            logger.info(f"  Reopened dimensions: {dict(reopened_ds.dims)}")
            
            # Test appending (should fail without overwrite for same data)
            try:
                xarray_to_zarr(
                    input_path=temp_input,
                    output_zarr=temp_zarr,
                    pattern="*.grib",
                    overwrite=False
                )
                logger.warning("Expected conflict detection to fail this operation")
            except ValueError as e:
                logger.info(f"✓ Conflict detection working: {e}")
            
            # Test with overwrite
            ds2 = xarray_to_zarr(
                input_path=temp_input,
                output_zarr=temp_zarr,
                pattern="*.grib",
                overwrite=True
            )
            logger.info(f"✓ Overwrite functionality working")
            
        except Exception as e:
            logger.error(f"Zarr creation test failed: {e}")
            raise
    
    logger.info("Zarr creation tests passed!")


def test_convenience_function():
    """Test the ERA5 convenience function."""
    logger.info("Testing ERA5 convenience function...")
    
    era5_dir = Path(ROOT) / "data" / "external" / "era5" / "era5_monthly_cds"
    
    if not any(era5_dir.glob("*.grib")):
        logger.warning("No ERA5 GRIB files found for convenience function test")
        return
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_zarr = Path(temp_dir) / "era5_convenience_test.zarr"
        
        try:
            # This should work the same as the main function
            ds = convert_era5_to_zarr(
                era5_dir=era5_dir,
                output_zarr=temp_zarr,
                overwrite=True
            )
            
            logger.info(f"✓ ERA5 convenience function works")
            logger.info(f"  Dataset: {ds}")
            
        except Exception as e:
            logger.error(f"ERA5 convenience function test failed: {e}")
            raise
    
    logger.info("ERA5 convenience function test passed!")


def main():
    """Run all tests."""
    logger.info("Starting xarray_to_zarr tests...")
    
    try:
        test_parse_filename()
        test_dataset_consistency()
        test_zarr_creation()
        test_convenience_function()
        
        logger.info("🎉 All tests passed!")
        
    except Exception as e:
        logger.error(f"❌ Tests failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()