import pandas as pd
from pathlib import Path
import re
from jiflr import ROOT

def merge_weather_data():
    """
    Process weather station data from multiple CSV files:
    1. Find all CSV files in the specified directory
    2. Identify hourly and daily data files
    3. Extract camp information and standardize identifiers
    4. Merge data by type (hourly/daily)
    5. Write merged data to output CSV files
    """
    # Define the mapping of elevations to camp numbers based on the provided table
    camp_info = {
        "C-10": {"Elevation": 1196, "Latitude": 59.57564, "Longitude": -133.70982},
        "C-17": {"Elevation": 1285, "Latitude": 58.64721, "Longitude": -134.20635},
        "C-26": {"Elevation": 1435, "Latitude": 58.36737, "Longitude": -134.36638},
        "C-9":  {"Elevation": 1555, "Latitude": 59.01664, "Longitude": -134.12104},
        "C-29": {"Elevation": 1618, "Latitude": 58.712433, "Longitude": -134.182217},
        "C-18": {"Elevation": 1705, "Latitude": 59.34211, "Longitude": -134.10221},
        "C-8":  {"Elevation": 2050, "Latitude": 58.83497, "Longitude": -134.27643},
        "C-25": {"Elevation": 2121, "Latitude": 58.8067, "Longitude": -134.13613}
    }
    
    # Create reverse mapping from elevation to camp ID
    elevation_to_camp = {info["Elevation"]: camp for camp, info in camp_info.items()}
    
    # Path to the root directory containing all CSV files
    root_dir = Path(ROOT, "data/external/JIRP_AWS_Stations/juneauIceField_weather_v1.0/LVL2")
    
    # Lists to store dataframes by type
    hourly_dfs = []
    daily_dfs = []
    
    # Find all CSV files in the directory (recursively)
    csv_files = list(root_dir.glob("**/*.csv"))
    
    print(f"Found {len(csv_files)} CSV files")
    
    for file_path in csv_files:
        file_name = file_path.name
        
        # Determine if the file contains hourly or daily data
        is_hourly = "hourly" in file_name.lower()
        is_daily = "daily" in file_name.lower()
        
        if not (is_hourly or is_daily):
            print(f"Skipping file {file_name} - unable to determine if hourly or daily data")
            continue
        
        try:
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Extract camp identifier from filename
            camp_id = extract_camp_id(file_name, elevation_to_camp)
            if camp_id:
                # Add standardized camp identifier column
                df["Camp"] = camp_id
                
                # Add additional camp metadata
                df["Elevation_m"] = camp_info[camp_id]["Elevation"]
                df["Latitude"] = camp_info[camp_id]["Latitude"]
                df["Longitude"] = camp_info[camp_id]["Longitude"]
            else:
                print(f"Warning: Could not extract camp ID from {file_name}")
                continue  # Skip files where camp ID can't be determined
            
            # Store in the appropriate list
            if is_hourly:
                hourly_dfs.append(df)
                print(f"Added hourly data from {file_name} ({camp_id}) - {len(df)} rows")
            elif is_daily:
                daily_dfs.append(df)
                print(f"Added daily data from {file_name} ({camp_id}) - {len(df)} rows")
                
        except Exception as e:
            print(f"Error processing file {file_name}: {e}")
    
    # Merge all hourly data
    if hourly_dfs:
        merged_hourly = pd.concat(hourly_dfs, ignore_index=True)
        # Write to disk
        output_hourly_path = Path("merged_hourly_weather_data.csv")
        merged_hourly.to_csv(output_hourly_path, index=False)
        print(f"Created merged hourly file with {len(merged_hourly)} rows at {output_hourly_path.absolute()}")
    else:
        print("No hourly data found")
    
    # Merge all daily data
    if daily_dfs:
        merged_daily = pd.concat(daily_dfs, ignore_index=True)
        # Write to disk
        output_daily_path = Path("merged_daily_weather_data.csv")
        merged_daily.to_csv(output_daily_path, index=False)
        print(f"Created merged daily file with {len(merged_daily)} rows at {output_daily_path.absolute()}")
    else:
        print("No daily data found")
        
def extract_camp_id(filename, elevation_to_camp):
    """
    Extract the standardized camp ID from the filename.
    Handles two patterns:
    1. Files with camp numbers (e.g., 'juneauicefieldCamp10AWS_hourly_LVL2.csv')
    2. Files with elevation (e.g., 'juneauicefield2121_daily_LVL2.csv')
    
    Args:
        filename: Name of the CSV file
        elevation_to_camp: Dictionary mapping elevations to camp IDs
        
    Returns:
        Standardized camp ID (e.g., 'C-10') or None if not found
    """
    # Pattern 1: juneauicefieldCamp10AWS_hourly_LVL2.csv
    camp_pattern = re.search(r"Camp(\d+)", filename, re.IGNORECASE)
    if camp_pattern:
        camp_num = camp_pattern.group(1)
        camp_id = f"C-{camp_num}"
        return camp_id
    
    # Pattern 2: juneauicefield2121_daily_LVL2.csv (where 2121 is elevation)
    elev_pattern = re.search(r"juneauicefield(\d+)_", filename)
    if elev_pattern:
        try:
            elevation = int(elev_pattern.group(1))
            if elevation in elevation_to_camp:
                return elevation_to_camp[elevation]
        except (ValueError, TypeError):
            pass
    
    return None

if __name__ == "__main__":
    merge_weather_data()