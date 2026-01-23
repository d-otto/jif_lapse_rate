# JIFLR Data Processing Pipeline

This directory contains the complete data processing pipeline for the JIFLR (Juneau Icefield Lapse Rate) project. The scripts transform raw sensor data through multiple processing levels to create standardized, analysis-ready datasets.

## Pipeline Overview

The data processing pipeline consists of 6 main scripts that process sensor data from raw exports to final analysis-ready datasets. The data processing follows two parallel tracks that eventually merge:

**Pendant Data Track:**
1. `clean_raw_pendant_data.py` - Processes individual sensor CSV files to NetCDF
2. `merge_intermediate_pendant_data.py` - Combines sensors by site/height
3. `merge_intermediate_site_data.py` - Creates site-level datasets

**Pace Data Track:**
1. `clean_raw_pace_data.py` - Processes pace logger data to NetCDF

**Combined Processing:**
1. `combine_pace_pendant_data.py` - Merges pendant and pace data
2. `process_lvl0_to_lvl1.py` - Further processing to create lvl1 data

```
Raw Data → Intermediate Processing → Level 0 → Level 1
    ↓              ↓                   ↓         ↓
[Pace .txt]   [Individual       [Combined    [5-min
[Pendant       sensor NetCDF]   datasets]    resampled]
 .csv]
```

**Key Architecture**: All processed data uses the **sensor_idx structure**. Data dimensions are `(sensor_idx, datetime)` with sensor metadata stored as coordinates.

## Processing Levels

- **Raw**: Unprocessed sensor exports (CSV, TXT, HOBO files)
- **Intermediate**: Individual sensor NetCDF files with sensor_idx structure
- **Level 0**: Combined datasets with standardized coordinates  
- **Level 1**: Resampled to regular 5-minute intervals

## Scripts

### 1. clean_raw_pace_data.py

**Purpose**: Convert raw Pace logger data files to NetCDF format

**Input**: `data/2025/raw/pace/*.txt`  
**Output**: `data/2025/intermediate/pace/*.nc`

**Usage**:
```bash
python scripts/data_pipeline/clean_raw_pace_data.py
```

**Description**: Processes Pace meteorological station data (wind, temperature, pressure) exported as text files. Creates individual NetCDF files with sensor_idx structure and CF-compliant metadata.

### 2. clean_raw_pendant_data.py  

**Purpose**: Convert raw HOBO pendant CSV exports to NetCDF format

**Input**: `data/2025/raw/pendants/exported/**/*.csv`  
**Output**: `data/2025/intermediate/pendants/by_sensor/**/*.nc`

**Usage**:
```bash
python scripts/data_pipeline/clean_raw_pendant_data.py
```

**Description**: Processes HOBO pendant temperature and light sensors exported from HOBOware/HOBOconnect. Handles both old and new generation sensors with different CSV formats. Preserves directory structure and applies timezone conversion.

**Features**:
- Recursive processing of subdirectories
- Automatic shielding detection from data inventory
- Deployment period masking
- Timezone conversion (UTC to local AKST)

### 3. merge_intermediate_pendant_data.py

**Purpose**: Merge individual pendant sensors into site-based files

**Input**: `data/2025/intermediate/pendants/by_sensor/**/*.nc`  
**Output**: `data/2025/intermediate/pendants/by_site/**/*.nc`

**Usage**:
```bash
python scripts/data_pipeline/merge_intermediate_pendant_data.py
```

**Description**: Groups pendant sensors by site and concatenates them along the sensor_idx dimension. Creates one file per site with all sensors for that location. Applies deployment masks to filter non-deployment periods.

**Features**:
- Site-based grouping with sensor_idx concatenation
- Quality control plot generation
- Deployment period filtering
- Handles subdirectory structure (camp_wx/, intensive/)

### 4. combine_pace_pendant_data.py

**Purpose**: Combine Pace and pendant data into unified intensive site dataset

**Input**: 
- `data/2025/intermediate/pace/*.nc`
- `data/2025/intermediate/pendants/by_site/intensive/*.nc`

**Output**: `data/2025/processed/lvl0/lvl0_intensive.nc`

**Usage**:
```bash
python scripts/data_pipeline/combine_pace_pendant_data.py
```

**Description**: Creates a single dataset combining meteorological data from Pace stations with temperature data from pendant sensors. Uses sensor_idx concatenation to merge different sensor types into one unified structure.

**Features**:
- Site name standardization and mapping
- Common datetime grid creation
- Mixed sensor type handling (pace + pendant)
- Comprehensive dataset attributes

### 5. merge_intermediate_site_data.py

**Purpose**: Merge site-level files into combined datasets by directory structure

**Input**: `data/2025/intermediate/pendants/by_site/**/*.nc`  
**Output**: `data/2025/processed/lvl0/lvl0_*.nc`

**Usage**:
```bash
python scripts/data_pipeline/merge_intermediate_site_data.py
```

**Description**: Combines multiple site files into larger datasets organized by subdirectory. Uses tree-based merging for memory efficiency with large datasets.

**Outputs**:
- `lvl0_main.nc` - Main directory sites
- `lvl0_intensive.nc` - Intensive monitoring sites  
- `lvl0_camp_wx.nc` - Camp weather station data

**Features**:
- Tree-based pairwise merging for efficiency
- String coordinate length fixing
- Automatic subdirectory discovery

### 6. process_lvl0_to_lvl1.py

**Purpose**: Resample level 0 data to regular 5-minute intervals

**Input**: `data/2025/processed/lvl0/lvl0_*.nc`  
**Output**: `data/2025/processed/lvl1/lvl1_*.nc`

**Usage**:
```bash
python scripts/data_pipeline/process_lvl0_to_lvl1.py

# With custom directories
python scripts/data_pipeline/process_lvl0_to_lvl1.py --input-dir data/2024/processed/lvl0 --output-dir data/2024/processed/lvl1
```

**Description**: Resamples irregular sensor data to standardized 5-minute intervals using mean aggregation. Creates both individual resampled files and a single combined dataset.

**Features**:
- Mean aggregation resampling  
- Preserves existing 5-minute data regardless of offset
- Creates individual and combined output files
- Sensor_idx structure preservation

## Data Structure

All processed data uses the **sensor_idx structure**:

**Dimensions**: `(sensor_idx, datetime)`

**Sensor Metadata** (as coordinates):
- `site_id(sensor_idx)`: Site identifier (e.g., 'A01', 'Windward1')
- `height(sensor_idx)`: Sensor height (e.g., '1m', '2m')  
- `shielding(sensor_idx)`: Shielding type ('shielded', 'unshielded')
- `sensor_type(sensor_idx)`: Sensor type ('hobo pendant', 'pace')
- `sensor_id(sensor_idx)`: Unique sensor identifier

**Analysis Patterns**:
```python
# Get all sensors at a site
site_data = ds.where(ds.site_id == 'A01', drop=True)

# Get sensors at specific height
height_data = ds.where(ds.height == '2m', drop=True)

# Site-based aggregations
site_means = ds.groupby('site_id').mean()
```

## Pipeline Execution

To run the complete pipeline:

```bash
# 1. Process raw data to intermediate
python scripts/data_pipeline/clean_raw_pace_data.py
python scripts/data_pipeline/clean_raw_pendant_data.py

# 2. Merge to site level
python scripts/data_pipeline/merge_intermediate_pendant_data.py

# 3. Combine different sensor types and create lvl0
python scripts/data_pipeline/combine_pace_pendant_data.py
python scripts/data_pipeline/merge_intermediate_site_data.py

# 4. Create lvl1 resampled data
python scripts/data_pipeline/process_lvl0_to_lvl1.py
```

## Requirements

- Python 3.8+
- xarray, pandas, numpy
- pathlib, tqdm
- matplotlib (for QC plots)
- JIFLR package installed in development mode: `pip install -e .`

## Directory Structure

Expected directory structure:
```
data/2025/
├── raw/
│   ├── pace/*.txt                    # Raw Pace logger files
│   └── pendants/exported/            # Raw pendant CSV exports
├── intermediate/
│   ├── pace/*.nc                     # Processed Pace data
│   └── pendants/
│       ├── by_sensor/                # Individual sensor files
│       └── by_site/                  # Site-combined files
├── processed/
│   ├── lvl0/*.nc                     # Level 0 combined datasets
│   └── lvl1/*.nc                     # Level 1 resampled data
└── metadata/
    ├── data_inventory.xlsx           # Sensor shielding information
    └── deployment_periods.csv        # Deployment periods for masking
```

## Breaking Change Notice

**October 2025**: The data structure was refactored from sparse dimensions to efficient sensor_idx structure. This is an intentional breaking change with no backward compatibility. All intermediate and processed data files must be regenerated using these scripts.

## Quality Control

The `merge_intermediate_pendant_data.py` script automatically generates QC plots for each site showing:
- Temperature time series by sensor
- Distribution box plots  
- Summary statistics
- Deployment period shading

Plots are saved to `qc_plots/` subdirectories within the output directories.