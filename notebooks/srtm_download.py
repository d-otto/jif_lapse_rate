"""
SRTM DEM Download Script

Downloads SRTM Digital Elevation Models from OpenTopography API for specified geographic areas.

Author: Claude Code
Created: 2025-08-20
Project: jiflr
"""

import requests
from pathlib import Path
from jiflr import ROOT

try:
    from api_keys import OPENTOPOGRAPHY_API_KEY
except ImportError:
    raise ImportError("API key file not found. Please create api_keys.py with your OpenTopography API key.")


def download_srtm_dem(lat_min, lat_max, lon_min, lon_max, demtype="SRTMGL1", output_dir=None):
    """
    Download SRTM DEM data from OpenTopography API for a specified bounding box.
    
    Parameters:
    -----------
    lat_min : float
        Minimum latitude (south bound)
    lat_max : float  
        Maximum latitude (north bound)
    lon_min : float
        Minimum longitude (west bound)
    lon_max : float
        Maximum longitude (east bound)
    demtype : str, optional
        DEM dataset type. Default is "SRTMGL1" (1-arcsecond SRTM)
    output_dir : Path or str, optional
        Output directory. Default is data/external/
        
    Returns:
    --------
    Path
        Path to downloaded GeoTIFF file
        
    Example:
    --------
    >>> # Download SRTM data for Juneau Icefield area
    >>> dem_path = download_srtm_dem(58.0, 59.0, -135.0, -133.0)
    """
    
    # Set default output directory
    if output_dir is None:
        output_dir = ROOT / "data" / "external"
    else:
        output_dir = Path(output_dir)
    
    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Construct API URL
    base_url = "https://portal.opentopography.org/API/globaldem"
    params = {
        "demtype": demtype,
        "south": lat_min,
        "north": lat_max, 
        "west": lon_min,
        "east": lon_max,
        "outputFormat": "GTiff",
        "API_Key": OPENTOPOGRAPHY_API_KEY
    }
    
    # Generate filename
    filename = f"srtm_{lat_min}_{lat_max}_{lon_min}_{lon_max}.tif"
    output_path = output_dir / filename
    
    # Download DEM
    response = requests.get(base_url, params=params)
    response.raise_for_status()
    
    # Save to file
    with open(output_path, 'wb') as f:
        f.write(response.content)
        
    return output_path