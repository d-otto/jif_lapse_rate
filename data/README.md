# JIFLR Data Organization Structure

This document describes the standardized data organization structure for the Juneau Icefield Lapse Rate (JIFLR) project. This structure should be followed for all field seasons to ensure consistency and support automated data processing.

## Directory Structure

Each field season follows this standardized structure:

```
data/
├── 2025/
│   ├── metadata/
│   |   ├── field_notes/           # Photos, field notebooks, documentation
│   │   ├── site_info.xlsx
│   │   ├── sensor_deployments.xlsx
│   │   └── data_inventory.xlsx
│   ├── raw/                     # Original sensor data files
│   │   ├── pendants/              # Temperature sensor data
│   │   │   ├── hobo/                # Raw .hobo files
│   │   │   │   ├── camp_wx/           # JIRP camp weather station data
│   │   │   │   ├── intensive/         # Intensive site data
│   │   │   │   └── on_ice/            # Standard measurement sites
│   │   │   │   └── off_ice/            # Standard measurement sites
│   │   │   └── exported/            # CSV files exported from HOBOware
│   │   │       └── [same structure as above]
│   │   └── pace/                  # Pace Scientific sensor data (if applicable)
│   ├── intermediate/            # Intermediate steps in the data processing pipeline
│   ├── processed/               # Analysis-ready data
│   │   └── lvl0/                  # Cleaned and combined data, trimmed to deployment periods 

├── 2024/
│   └── [same structure as above]
├── 2023/
│   └── [same structure as above]
├── 2022/
│   └── [same structure as above]
└── external/                   # External datasets (ArcticDEM, ERA5, etc.)
```

## Metadata Files

Each year's `metadata/` directory contains three files that track different aspects of the field season. Some of these are meant to be primarily machine-readable for use in data processing/analysis. Others are meant primarily for human reading and record keeping:

### 1. site_info.xlsx
**Purpose**: Static reference information about each site location

- **site_id**: Standardized site code (e.g., A01, B02, C17)
- **site_name**: Descriptive name if applicable (e.g., Heather_high, C17_WX)
- **latitude**: Decimal degrees (e.g., 58.8338)
- **longitude**: Decimal degrees (e.g., -134.2870)
- **elevation_m**: Site elevation in meters (e.g., 982)
- **transect**: Study area/transect name (e.g., A-transect, G-transect)
- **site_type**: Category of site (e.g., lapse_rate, weather_station, intensive)
- **access_notes**: How to reach the site (e.g., "Via C18 traverse")
- **established_date**: When site was first used (e.g., 2024-06-20)

### 2. sensor_deployments.xlsx
**Purpose**: Track sensor deployment/retrieval events and field activities

- **site_id**: Links to site_info.xlsx (e.g., A01)
- **sensor_id**: Physical sensor serial number (e.g., 22133645)
- **sensor_height**: Height above ground (e.g., 1m, 2m, unshielded)
- **deploy_datetime**: Deployment time in YYYY-MM-DD HH:MM TZ format (e.g., 2025-06-20 15:18:00 AKDT)
- **pickup_datetime**: Retrieval time in YYYY-MM-DD HH:MM TZ format (e.g., 2025-07-27 10:22:50 AKDT)
- **deploy_notes**: Field conditions during deployment (e.g., "Clear weather, firm snow")
- **pickup_notes**: Sensor condition, data quality (e.g., "Sensor functional, good data")
- **responsible_person**: Who deployed/retrieved (e.g., "D. Otto")
- **field_photos**: Photo filenames taken (e.g., "IMG_8710.jpeg, IMG_8711.jpeg")

### 3. data_inventory.xlsx
**Purpose**: Track what data products exist for each site/deployment

- **site_id**: Links to site_info.xlsx (e.g., A01)
- **sensor_height**: Height specification (e.g., 1m)
- **raw_file_exists**: Boolean - .hobo file retrieved (e.g., TRUE)
- **csv_file_exists**: Boolean - exported from HOBOware (e.g., TRUE)
- **processed_file_exists**: Boolean - NetCDF created (e.g., TRUE)
- **csv_filename**: Full CSV filename (e.g., "A01 1m 2025-07-27 10_22_50 AKDT (Data AKDT).csv")
- **processed_filename**: NetCDF filename (e.g., "22133645_20250620T1518_20250727T1023.nc")
- **data_quality**: Qualitative assessment (e.g., Good, Fair, Poor, Issues)
- **data_start_date**: First valid data point (e.g., 2025-06-20)
- **data_end_date**: Last valid data point (e.g., 2025-07-27)
- **site_photos_exist**: Boolean - site documentation photos (e.g., TRUE)
- **field_notes_exist**: Boolean - written field notes (e.g., TRUE)

## Data Processing Workflow

### Data Export
- Export .hobo files and archive in `raw/hobo`
- Export to CSV using Hoboware (NOT HoboConnect)
- Place CSV files in `raw/exported` or appropriate subdirectory
- Follow naming conventions above

### Metadata Entry
- Update `sensor_deployments.xlsx` with deployment/retrieval details
- Update `data_inventory.xlsx` to track file existence
- Ensure `site_info.xlsx` has current site information

### Data Processing

Run scripts in the following order to move new or updated data through the pipeline. Note that running these scripts will reprocess all the data.

1. `clean_raw_pendant_data.py`: Converted exported .csv's to netCDF files, organized by sensor serial number
2. `merge_intermediate_pendant_data.py`: Groups by-sensor data into netCDF files for each site and trims data to the deployment period in `deployment_periods.csv`
3. `merge_intermediate_site_data.py`: Combines by-site data into files by category (e.g., standard sites, intensive sites, camp wx)

### Quality Control

TODO
