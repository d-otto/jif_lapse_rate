#!/usr/bin/env python3
"""
JIRP Weather Station Data Processor

This script processes the JIRP Temperature Record data from multiple research stations,
standardizes the data, and creates hourly and daily aggregated datasets.

Usage:
    python jirp_weather_processor.py

Output:
    - processed_data/jirp_hourly_weather_data.csv
    - processed_data/jirp_daily_weather_data.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
import re
import logging
from datetime import datetime, timedelta
import os
from jiflr import ROOT

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def load_camp_reference_data():
    """Load the reference data for the research station camps."""
    # Reference data provided in the original request
    data = """Location,Elevation,Latitude,Longitude,NWIS ID
C-10,1196,59.57564,-133.70982,NA
C-17,1285,58.64721,-134.20635,NA
C-26,1435,58.36737,-134.36638,NA
C-9,1555,59.01664,-134.12104,NA
C-29,1618,58.712433,-134.182217,NA
C-18,1705,59.34211,-134.10221,NA
C-8,2050,58.83497,-134.27643,NA
C-25,2121,58.8067,-134.13613,NA
C-30,674,59.57544893154394, -133.70967902899886,NA"""
    
    # Create a DataFrame from the CSV text
    import io
    df = pd.read_csv(io.StringIO(data))
    
    # Standardize column names
    df.columns = [col.lower().strip().replace(' ', '_') for col in df.columns]
    
    # Strip whitespace from location values
    df['location'] = df['location'].str.strip()
    
    # Convert NA to NaN
    df = df.replace('NA', np.nan)
    
    # Log the locations for debugging
    logging.info(f"Reference data locations: {df['location'].tolist()}")
    
    return df

def extract_camp_id(sheet_name):
    """Extract the camp ID from the sheet name (format: 'Camp XX')."""
    match = re.search(r'Camp\s*(\d+)', sheet_name, re.IGNORECASE)
    if match:
        camp_number = match.group(1)
        return f"C-{camp_number}"
    return None

def parse_day_month_hour(day_month_hour):
    """Parse the 'Day-Month-Hour' column (format: 'MM/DD H:MM')."""
    try:
        # Check if day_month_hour is already a datetime object
        if isinstance(day_month_hour, datetime):
            return day_month_hour
            
        # Convert to string if it's not already
        day_month_hour = str(day_month_hour).strip()
        
        # Skip summary rows with labels like "Minimum", "Maximum", etc.
        summary_labels = ["minimum", "maximum", "mean", "average", "sum", "std", "count"]
        if day_month_hour.lower() in summary_labels:
            return None
        
        # Skip any row that doesn't match expected format
        if not re.match(r'\d+/\d+\s+\d+', day_month_hour) and not re.match(r'\d+/\d+\s+\d+:\d+', day_month_hour):
            return None
        
        # Extract components
        parts = day_month_hour.split()
        if len(parts) != 2:
            return None
            
        day_month, hour = parts
        month, day = map(int, day_month.split('/'))  # Format is MM/DD
        
        # Parse hour, handling both H:MM and just H formats
        if ':' in hour:
            hour_val = int(hour.split(':')[0])
        else:
            hour_val = int(hour)
        
        # Create a datetime with a placeholder year (will be replaced later)
        # Using 2000 as placeholder since it's a leap year
        dt = datetime(2000, month, day, hour_val)
        return dt
    except Exception as e:
        logging.error(f"Error parsing date-time value '{day_month_hour}': {str(e)}")
        return None

def process_camp_sheet(sheet_data, sheet_name, camps_reference, max_rows=8785):
    """Process a single camp sheet."""
    """Process a single camp sheet."""
    logging.info(f"Processing sheet: {sheet_name}")
    
    # Extract camp ID from sheet name
    camp_id = extract_camp_id(sheet_name)
    if not camp_id:
        logging.warning(f"Could not extract camp ID from sheet name: {sheet_name}")
        return None
    
    # Limit rows to specified maximum (to exclude summary statistics)
    if len(sheet_data) > max_rows:
        sheet_data = sheet_data.iloc[:max_rows]
    
    # Check for expected columns
    required_cols = ['Serial-Day', 'Day-Month-Hour']
    if not all(col in sheet_data.columns for col in required_cols):
        logging.warning(f"Sheet {sheet_name} missing required columns. Found: {sheet_data.columns.tolist()}")
        return None
    
    # Identify year columns (all columns except the first two)
    year_columns = [col for col in sheet_data.columns if col not in required_cols]
    if not year_columns:
        logging.warning(f"No year columns found in sheet {sheet_name}")
        return None
    
    # Convert to long format
    logging.info(f"Converting data to long format with {len(year_columns)} year columns")
    
    # Melt the dataframe to convert from wide to long format
    df_long = pd.melt(
        sheet_data,
        id_vars=required_cols,
        value_vars=year_columns,
        var_name='year',
        value_name='temperature'
    )
    
    # Filter out rows with missing temperature
    df_long = df_long.dropna(subset=['temperature'])
    
    # Try to convert year column to numeric if not already
    if not pd.api.types.is_numeric_dtype(df_long['year']):
        # Extract year from column name if it's a string like "2018" or contains year
        df_long['year'] = df_long['year'].apply(
            lambda x: int(re.search(r'(\d{4})', str(x)).group(1)) if isinstance(x, str) and re.search(r'(\d{4})', str(x)) else x
        )
        df_long['year'] = pd.to_numeric(df_long['year'], errors='coerce')
    
    # Parse the Day-Month-Hour column
    df_long['datetime_base'] = df_long['Day-Month-Hour'].apply(parse_day_month_hour)
    
    # Drop rows with invalid datetime_base
    df_long = df_long.dropna(subset=['datetime_base'])
    
    # Combine year with the datetime_base to get full datetime
    def combine_year_with_date(row):
        try:
            if pd.isna(row['datetime_base']) or pd.isna(row['year']):
                return None
            year_val = int(row['year'])
            return row['datetime_base'].replace(year=year_val)
        except Exception as e:
            logging.error(f"Error combining year with date: {str(e)}")
            return None
            
    df_long['datetime'] = df_long.apply(combine_year_with_date, axis=1)
    
    # Add camp information
    df_long['camp_id'] = camp_id
    
    # Add reference data
    camp_info = camps_reference[camps_reference['location'] == camp_id]
    if not camp_info.empty:
        df_long['elevation'] = camp_info['elevation'].values[0]
        df_long['latitude'] = camp_info['latitude'].values[0]
        df_long['longitude'] = camp_info['longitude'].values[0]
        df_long['nwis_id'] = camp_info['nwis_id'].values[0]
    else:
        logging.warning(f"Camp {camp_id} not found in reference data")
    
    # Clean up temperature values if needed
    df_long['temperature'] = pd.to_numeric(df_long['temperature'], errors='coerce')
    df_long = df_long.dropna(subset=['temperature'])
    
    # Select only relevant columns
    relevant_cols = ['datetime', 'camp_id', 'temperature', 'elevation', 'latitude', 'longitude', 'nwis_id']
    relevant_cols = [col for col in relevant_cols if col in df_long.columns]
    
    df_clean = df_long[relevant_cols]
    
    logging.info(f"Processed {len(df_clean)} temperature records for camp {camp_id}")
    
    return df_clean

def process_weather_data(file_path, camps_reference):
    """Process the weather station data from the Excel file."""
    logging.info(f"Reading Excel file: {file_path}")
    
    # Read the Excel file without loading the data
    try:
        xl = pd.ExcelFile(file_path)
        sheet_names = xl.sheet_names
        
        logging.info(f"Found {len(sheet_names)} sheets: {sheet_names}")
    except Exception as e:
        logging.error(f"Error opening Excel file: {e}")
        return None, None
    
    # Filter for sheets that match the camp naming pattern
    camp_sheets = [name for name in sheet_names if re.search(r'Camp\s*\d+', name, re.IGNORECASE)]
    
    if not camp_sheets:
        logging.error("No sheets found with camp data (format: 'Camp XX')")
        return None, None
    
    logging.info(f"Found {len(camp_sheets)} camp data sheets: {camp_sheets}")
    
    # Process each camp sheet
    all_data = []
    
    for sheet_name in camp_sheets:
        try:
            # Read the sheet data
            sheet_data = pd.read_excel(file_path, sheet_name=sheet_name)
            
            # Skip empty sheets
            if sheet_data.empty:
                logging.warning(f"Sheet {sheet_name} is empty. Skipping.")
                continue
            
            # Process the camp sheet
            processed_data = process_camp_sheet(sheet_data, sheet_name, camps_reference)
            
            if processed_data is not None:
                all_data.append(processed_data)
        except Exception as e:
            logging.error(f"Error processing sheet {sheet_name}: {e}")
    
    if not all_data:
        logging.error("No data was successfully processed")
        return None, None
    
    # Combine all data
    logging.info("Combining all processed data")
    
    combined_hourly = pd.concat(all_data, ignore_index=True)
    
    # Check for and handle duplicate timestamps
    duplicates = combined_hourly.duplicated(subset=['datetime', 'camp_id'], keep=False)
    if duplicates.any():
        logging.warning(f"Found {duplicates.sum()} duplicate timestamp records")
        # For duplicates, take the average temperature
        combined_hourly = combined_hourly.groupby(['datetime', 'camp_id'], as_index=False).agg({
            'temperature': 'mean',
            'elevation': 'first',
            'latitude': 'first',
            'longitude': 'first',
            'nwis_id': 'first'
        })
    
    # Sort by datetime and camp_id
    combined_hourly = combined_hourly.sort_values(['datetime', 'camp_id'])
    
    # Create daily aggregation
    logging.info("Creating daily aggregates")
    
    combined_hourly['date'] = combined_hourly['datetime'].dt.date
    
    # Define aggregation functions
    agg_funcs = {
        'temperature': ['min', 'max', 'mean', 'count', 'std']
    }
    
    # Add optional columns to aggregation
    optional_cols = ['elevation', 'latitude', 'longitude', 'nwis_id']
    for col in optional_cols:
        if col in combined_hourly.columns:
            agg_funcs[col] = 'first'
    
    # Group by date and camp_id to create daily aggregates
    combined_daily = combined_hourly.groupby(['date', 'camp_id']).agg(agg_funcs)
    
    # Flatten the multi-index columns
    combined_daily.columns = ['_'.join(col).strip('_') for col in combined_daily.columns.values]
    
    # Reset the index to convert date and camp_id to columns
    combined_daily = combined_daily.reset_index()
    
    # Rename the count column to make it clearer
    combined_daily = combined_daily.rename(columns={'temperature_count': 'hourly_observations'})
    
    logging.info(f"Processed {len(combined_hourly)} hourly records and {len(combined_daily)} daily records")
    
    return combined_hourly, combined_daily

def main():
    """Main function to load, process, and save the weather station data."""
    logging.info("Starting to process the weather station data")
    
    # Import the ROOT path from jiflr
    from jiflr import ROOT
    
    # Define the path to the input file
    file_path = Path(ROOT, "data/external/mcgee_wx/JIRP Temperature Record (Main Camps).xlsx")
    
    # Load the camp reference data
    camps_reference = load_camp_reference_data()
    
    # Process the data
    hourly_data, daily_data = process_weather_data(file_path, camps_reference)
    
    if hourly_data is not None and daily_data is not None:
        # Create output directory if it doesn't exist
        output_dir = Path("processed_data")
        output_dir.mkdir(exist_ok=True)
        
        # Save to CSV
        hourly_output_path = output_dir / "mcgee_jirp_hourly_weather_data.csv"
        daily_output_path = output_dir / "mcgee_jirp_daily_weather_data.csv"
        
        hourly_data.to_csv(hourly_output_path, index=False)
        daily_data.to_csv(daily_output_path, index=False)
        
        logging.info(f"Hourly data saved to: {hourly_output_path}")
        logging.info(f"Daily data saved to: {daily_output_path}")
        
        logging.info(f"Summary:")
        logging.info(f"  - Total hourly records: {len(hourly_data)}")
        logging.info(f"  - Total daily records: {len(daily_data)}")
        logging.info(f"  - Camps included: {hourly_data['camp_id'].nunique()}")
        logging.info(f"  - Date range: {hourly_data['datetime'].min()} to {hourly_data['datetime'].max()}")
    else:
        logging.error("Failed to process the data")

if __name__ == "__main__":
    main()