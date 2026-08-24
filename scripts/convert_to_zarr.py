#!/usr/bin/env python3
"""
Convert xarray-compatible files to Zarr format.

This script provides a command-line interface for converting NetCDF and GRIB
files to Zarr format using the jiflr.io module. The zarr store is treated
like a database for efficient data access.
"""

import argparse
import sys
import logging
from pathlib import Path
from typing import Dict, Any, Optional

# Add src to path so we can import jiflr
script_dir = Path(__file__).parent
project_root = script_dir.parent
sys.path.insert(0, str(project_root / "src"))

from jiflr.io import xarray_to_zarr, convert_era5_to_zarr


def setup_logging(verbose: bool = False):
    """Set up logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_chunks(chunk_str: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Parse chunk specification from string.
    
    Format: "dim1:size1,dim2:size2" or "dim1:size1 dim2:size2"
    Use -1 for full dimension size.
    
    Example: "time:336,latitude:-1,longitude:-1"
    """
    if not chunk_str:
        return None
    
    chunks = {}
    
    # Handle both comma and space separators
    chunk_str = chunk_str.replace(' ', ',')
    pairs = chunk_str.split(',')
    
    for pair in pairs:
        if ':' not in pair:
            continue
        
        dim, size_str = pair.split(':', 1)
        dim = dim.strip()
        size_str = size_str.strip()
        
        if size_str == '-1':
            chunks[dim] = -1
        else:
            try:
                chunks[dim] = int(size_str)
            except ValueError:
                raise argparse.ArgumentTypeError(
                    f"Invalid chunk size '{size_str}' for dimension '{dim}'. "
                    "Use integer values or -1 for full dimension."
                )
    
    return chunks


def main():
    """Main command-line interface."""
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert all GRIB files in a directory
  %(prog)s /path/to/era5_files output.zarr --pattern "*.grib"
  
  # Convert with custom chunking
  %(prog)s /path/to/files output.zarr --chunks "time:336,latitude:-1,longitude:-1"
  
  # Overwrite existing zarr store
  %(prog)s /path/to/files output.zarr --overwrite
  
  # ERA5 convenience command
  %(prog)s /path/to/era5_monthly_cds output.zarr --era5
  
  # Process specific files with pattern
  %(prog)s /path/to/files output.zarr --pattern "era5_monthly_cds.2m_*.grib"
        """
    )
    
    parser.add_argument(
        'input_path',
        type=Path,
        help='Input directory containing files to convert, or single file path'
    )
    
    parser.add_argument(
        'output_zarr',
        type=Path,
        help='Output path for zarr store'
    )
    
    parser.add_argument(
        '--pattern',
        default='*',
        help='Glob pattern for file selection (default: "*")'
    )
    
    parser.add_argument(
        '--overwrite',
        action='store_true',
        help='Overwrite conflicting data in existing zarr store'
    )
    
    parser.add_argument(
        '--chunks',
        type=parse_chunks,
        help='Custom chunking specification (e.g., "time:336,latitude:-1,longitude:-1")'
    )
    
    parser.add_argument(
        '--era5',
        action='store_true',
        help='Use ERA5 convenience function (sets pattern="*.grib" and optimized defaults)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show what would be processed without actually converting'
    )
    
    args = parser.parse_args()
    
    # Set up logging
    setup_logging(args.verbose)
    logger = logging.getLogger(__name__)
    
    # Validate input path
    if not args.input_path.exists():
        logger.error(f"Input path does not exist: {args.input_path}")
        sys.exit(1)
    
    # Dry run: just show what would be processed
    if args.dry_run:
        logger.info("DRY RUN MODE - no files will be converted")
        
        if args.input_path.is_file():
            files = [args.input_path]
        else:
            files = sorted(args.input_path.glob(args.pattern))
            files = [f for f in files if f.is_file()]
        
        logger.info(f"Would process {len(files)} files:")
        for i, file in enumerate(files, 1):
            logger.info(f"  {i:3d}. {file.name}")
        
        if not files:
            logger.warning(f"No files found matching pattern '{args.pattern}'")
        
        logger.info(f"Output zarr: {args.output_zarr}")
        logger.info(f"Overwrite: {args.overwrite}")
        logger.info(f"Custom chunks: {args.chunks}")
        return
    
    try:
        # Choose appropriate function
        if args.era5:
            logger.info("Using ERA5 convenience function")
            if args.chunks:
                logger.warning("Custom chunks ignored when using --era5 flag")
            
            ds = convert_era5_to_zarr(
                era5_dir=args.input_path,
                output_zarr=args.output_zarr,
                overwrite=args.overwrite
            )
        else:
            logger.info("Using general xarray_to_zarr function")
            
            ds = xarray_to_zarr(
                input_path=args.input_path,
                output_zarr=args.output_zarr,
                pattern=args.pattern,
                overwrite=args.overwrite,
                chunks=args.chunks
            )
        
        # Report success
        logger.info("🎉 Conversion completed successfully!")
        logger.info(f"Dataset info:")
        logger.info(f"  Dimensions: {dict(ds.dims)}")
        logger.info(f"  Variables: {len(ds.data_vars)} ({', '.join(list(ds.data_vars.keys())[:5])}{'...' if len(ds.data_vars) > 5 else ''})")
        logger.info(f"  Zarr store: {args.output_zarr}")
        
        # Show zarr store size if possible
        try:
            import zarr
            store = zarr.open(args.output_zarr)
            nbytes = sum(arr.nbytes for arr in store.arrays()[1])
            size_mb = nbytes / (1024 * 1024)
            logger.info(f"  Store size: {size_mb:.1f} MB")
        except Exception:
            pass  # Size calculation is optional
        
    except Exception as e:
        logger.error(f"❌ Conversion failed: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()