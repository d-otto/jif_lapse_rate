"""
Input/Output functions for the JIFLR project.

This module provides functions for converting xarray-compatible files 
(NetCDF, GRIB) to Zarr format for efficient access and analysis.
"""

import xarray as xr
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Union
import warnings
import logging
from dask.distributed import LocalCluster, Client
import zarr

# Set up logging
logger = logging.getLogger(__name__)


def _parse_filename(filepath: Union[str, Path]) -> Tuple[str, str]:
    """
    Parse dataset name and variable name from file path.
    
    Expected format: {dataset_name}.{variable_name}.{extension}
    
    Args:
        filepath: Path to the file
        
    Returns:
        Tuple of (dataset_name, variable_name)
        
    Raises:
        ValueError: If filename doesn't follow expected format
    """
    filepath = Path(filepath)
    stem = filepath.stem  # filename without extension
    
    parts = stem.split('.')
    if len(parts) < 2:
        raise ValueError(
            f"Filename '{filepath.name}' doesn't follow expected format: "
            "{{dataset_name}}.{{variable_name}}.{{extension}}"
        )
    
    dataset_name = parts[0]
    variable_name = '.'.join(parts[1:])  # Join remaining parts in case variable has dots
    
    return dataset_name, variable_name


def _validate_dataset_consistency(filepaths: List[Path]) -> str:
    """
    Validate that all files belong to the same dataset.
    
    Args:
        filepaths: List of file paths
        
    Returns:
        The common dataset name
        
    Raises:
        ValueError: If files have different dataset names
    """
    if not filepaths:
        raise ValueError("No files provided")
    
    dataset_names = set()
    for filepath in filepaths:
        dataset_name, _ = _parse_filename(filepath)
        dataset_names.add(dataset_name)
    
    if len(dataset_names) > 1:
        raise ValueError(
            f"Found files from multiple datasets: {sorted(dataset_names)}. "
            "All files must belong to the same dataset."
        )
    
    return dataset_names.pop()


def _get_optimal_chunks(ds: xr.Dataset) -> Dict[str, Any]:
    """
    Determine optimal chunking strategy based on dataset characteristics.
    
    Args:
        ds: xarray Dataset
        
    Returns:
        Dictionary of dimension names to chunk sizes
    """
    chunks = {}
    
    for dim, size in ds.dims.items():
        if dim in ['time', 'step']:
            # For time dimensions, use ~2 weeks of data or the full size if smaller
            chunks[dim] = min(24 * 14, size)
        elif dim in ['latitude', 'lat']:
            # Use full latitude dimension for efficient spatial operations
            chunks[dim] = -1
        elif dim in ['longitude', 'lon']:
            # Use full longitude dimension for efficient spatial operations  
            chunks[dim] = -1
        elif dim in ['level', 'plev', 'pressure_level']:
            # For pressure levels, chunk by single level for memory efficiency
            chunks[dim] = 1
        else:
            # For unknown dimensions, use a reasonable default
            chunks[dim] = min(100, size)
    
    return chunks


def _detect_file_engine(filepath: Path) -> str:
    """
    Detect the appropriate xarray engine for a file.
    
    Args:
        filepath: Path to the file
        
    Returns:
        Engine name for xarray.open_dataset()
        
    Raises:
        ImportError: If required engine is not available
    """
    suffix = filepath.suffix.lower()
    if suffix == '.nc':
        return 'netcdf4'
    elif suffix == '.grib':
        # Check if cfgrib is available
        try:
            import cfgrib
            return 'cfgrib'
        except ImportError:
            raise ImportError(
                "cfgrib is required to read GRIB files. "
                "Install it with: pip install cfgrib"
            )
    else:
        # Default to netcdf4 and let xarray figure it out
        logger.warning(f"Unknown file extension '{suffix}', defaulting to netcdf4 engine")
        return 'netcdf4'


def _check_zarr_conflicts(
    new_ds: xr.Dataset, 
    zarr_path: Path, 
    overwrite: bool = False
) -> bool:
    """
    Check for conflicts when appending to existing zarr store.
    
    Args:
        new_ds: New dataset to append
        zarr_path: Path to existing zarr store
        overwrite: Whether to allow overwriting conflicting data
        
    Returns:
        True if safe to proceed, False if conflicts exist and overwrite=False
    """
    if not zarr_path.exists():
        return True  # No existing store, safe to create
    
    try:
        existing_ds = xr.open_zarr(zarr_path)
    except Exception as e:
        logger.warning(f"Could not read existing zarr store: {e}")
        return overwrite
    
    # Check for overlapping time periods
    if 'time' in new_ds.dims and 'time' in existing_ds.dims:
        new_times = set(pd.to_datetime(new_ds.time.values))
        existing_times = set(pd.to_datetime(existing_ds.time.values))
        overlapping_times = new_times & existing_times
        
        if overlapping_times:
            if overwrite:
                logger.warning(
                    f"Found {len(overlapping_times)} overlapping time steps. "
                    "Will overwrite existing data."
                )
                return True
            else:
                logger.warning(
                    f"Found {len(overlapping_times)} overlapping time steps. "
                    "Set overwrite=True to overwrite existing data."
                )
                return False
    
    # Check for overlapping variables
    new_vars = set(new_ds.data_vars.keys())
    existing_vars = set(existing_ds.data_vars.keys())
    overlapping_vars = new_vars & existing_vars
    
    if overlapping_vars:
        if overwrite:
            logger.warning(
                f"Found overlapping variables: {sorted(overlapping_vars)}. "
                "Will overwrite existing data."
            )
            return True
        else:
            logger.warning(
                f"Found overlapping variables: {sorted(overlapping_vars)}. "
                "Set overwrite=True to overwrite existing data."
            )
            return False
    
    return True


def xarray_to_zarr(
    input_path: Union[str, Path],
    output_zarr: Union[str, Path],
    pattern: str = "*",
    overwrite: bool = False,
    chunks: Optional[Dict[str, Any]] = None,
    dask_client: Optional[Client] = None,
    temp_zarr: Optional[Union[str, Path]] = None
) -> xr.Dataset:
    """
    Convert xarray-compatible files to Zarr format.
    
    This function treats the zarr store like a database:
    - Creates new zarr if it doesn't exist
    - Appends non-conflicting data if zarr exists
    - Warns about conflicts unless overwrite=True
    
    Args:
        input_path: Directory containing input files
        output_zarr: Path for output zarr store
        pattern: Glob pattern for file selection (default: "*")
        overwrite: Whether to overwrite conflicting data (default: False)
        chunks: Custom chunking dict (default: uses optimized chunks)
        dask_client: Optional existing Dask client
        temp_zarr: Optional temporary zarr store for rechunking
        
    Returns:
        The created/updated xarray Dataset
        
    Raises:
        ValueError: If files don't follow naming convention or have mismatched datasets
        FileNotFoundError: If no matching files found
    """
    input_path = Path(input_path)
    output_zarr = Path(output_zarr)
    
    # Find all matching files
    if input_path.is_file():
        files = [input_path]
    else:
        files = sorted(input_path.glob(pattern))
        files = [f for f in files if f.is_file()]
    
    if not files:
        raise FileNotFoundError(f"No files found matching pattern '{pattern}' in {input_path}")
    
    logger.info(f"Found {len(files)} files to process")
    
    # Validate dataset consistency
    dataset_name = _validate_dataset_consistency(files)
    logger.info(f"Processing dataset: {dataset_name}")
    
    # Default chunking strategy - will be determined after loading the data
    use_optimal_chunks = chunks is None
    
    # Group files by engine type and validate them
    engine_groups = {}
    skipped_files = []
    
    for file in files:
        try:
            engine = _detect_file_engine(file)
            
            # Quick validation: try to open the file to check if it's readable
            if engine == 'cfgrib':
                try:
                    # Quick test read to validate GRIB file
                    test_ds = xr.open_dataset(file, engine=engine)
                    test_ds.close()
                except Exception as e:
                    logger.warning(f"Skipping corrupted file {file.name}: {e}")
                    skipped_files.append(file)
                    continue
            
            if engine not in engine_groups:
                engine_groups[engine] = []
            engine_groups[engine].append(file)
            
        except Exception as e:
            logger.warning(f"Skipping problematic file {file.name}: {e}")
            skipped_files.append(file)
    
    if skipped_files:
        logger.warning(f"Skipped {len(skipped_files)} corrupted/problematic files")
    
    if not engine_groups:
        raise FileNotFoundError("No valid files found after filtering corrupted files")
    
    # Process each engine group separately
    datasets = []
    
    for engine, file_group in engine_groups.items():
        logger.info(f"Processing {len(file_group)} files with {engine} engine")
        
        try:
            # Open all files as single dataset
            # Use override compat mode to handle conflicting coordinate values
            ds = xr.open_mfdataset(
                file_group,
                parallel=True,
                engine=engine,
                chunks=None,  # Will rechunk after loading
                compat='override',  # Handle conflicting coordinates
                join='outer',       # Keep all time periods
            )
            
            # Determine chunking strategy
            if use_optimal_chunks:
                # Auto-detect optimal chunks based on dataset
                dataset_chunks = _get_optimal_chunks(ds)
                logger.info(f"Using optimal chunking strategy: {dataset_chunks}")
            else:
                # Apply user-specified chunks, filtered to available dimensions
                available_dims = set(ds.dims.keys())
                dataset_chunks = {dim: size for dim, size in chunks.items() if dim in available_dims}
                
                # Log chunking info
                skipped_dims = set(chunks.keys()) - available_dims
                logger.info(f"Dataset dimensions: {sorted(available_dims)}")
                logger.info(f"Applying user chunks: {dataset_chunks}")
                if skipped_dims:
                    logger.info(f"Skipped chunk dimensions (not in dataset): {sorted(skipped_dims)}")
            
            ds = ds.chunk(dataset_chunks)
            datasets.append(ds)
            
        except Exception as e:
            logger.error(f"Failed to open files with {engine} engine: {e}")
            raise
    
    # Combine datasets if multiple engines were used
    if len(datasets) == 1:
        combined_ds = datasets[0]
    else:
        logger.info("Combining datasets from different engines")
        combined_ds = xr.merge(datasets)
    
    # Check for conflicts if zarr already exists
    if not _check_zarr_conflicts(combined_ds, output_zarr, overwrite):
        raise ValueError(
            "Conflicts detected with existing zarr store. "
            "Set overwrite=True to overwrite existing data."
        )
    
    # Create encoding that matches chunk sizes
    # Use actual chunks from the dataset, not the original chunk specification
    encoding = {}
    for var in combined_ds.data_vars:
        if hasattr(combined_ds[var].data, 'chunks'):
            # Get actual chunk sizes from dask array
            var_chunks = combined_ds[var].data.chunks
            # Convert to tuple of chunk sizes (take first chunk size for each dim)
            var_chunks = tuple(chunk[0] for chunk in var_chunks)
        else:
            # Fallback to dataset sizes
            var_chunks = tuple(combined_ds.sizes[dim] for dim in combined_ds[var].dims)
        encoding[var] = {"chunks": var_chunks}
    
    # Determine write mode
    mode = "w" if overwrite or not output_zarr.exists() else "a"
    
    # Write to zarr
    logger.info(f"Writing to zarr store: {output_zarr}")
    combined_ds.to_zarr(
        output_zarr, 
        mode=mode, 
        consolidated=True, 
        encoding=encoding
    )
    
    logger.info(f"Successfully created/updated zarr store: {output_zarr}")
    
    return combined_ds


def convert_era5_to_zarr(
    era5_dir: Union[str, Path],
    output_zarr: Union[str, Path],
    overwrite: bool = False
) -> xr.Dataset:
    """
    Convenience function to convert ERA5 monthly CDS files to zarr.
    
    Args:
        era5_dir: Directory containing ERA5 .grib files
        output_zarr: Path for output zarr store
        overwrite: Whether to overwrite existing zarr store
        
    Returns:
        The created xarray Dataset
    """
    return xarray_to_zarr(
        input_path=era5_dir,
        output_zarr=output_zarr,
        pattern="*.grib",
        overwrite=overwrite
    )