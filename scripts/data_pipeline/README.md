# JIFLR Data Processing Pipeline

This directory contains the complete data processing pipeline for the JIFLR (Juneau Icefield Lapse Rate) project. The scripts transform raw sensor data through multiple processing levels to create standardized, analysis-ready datasets.

See the [Level product guide](../../data/LEVEL_PRODUCTS.md) for user-facing
variable definitions and the [release assessment](../../data/RELEASE_READINESS.md)
for outstanding publication requirements.

## Pipeline Overview

The data processing pipeline consists of seven seasonal scripts plus an eighth
cross-season merge script. The seasonal steps process sensor data from raw
exports to final analysis-ready datasets; step 08 merges the resulting Level 1
files across years.

**Pendant Data Track:**
1. `03_clean_raw_pendants.py` - Processes individual sensor CSV files to NetCDF
2. `04_merge_raw_pendants_by_site.py` - Combines sensors by site/height
3. `06_merge_intermediate_to_lvl0.py` - Creates site-level Level 0 datasets

**Pace Data Track:**
1. `01_clean_raw_pace.py` - Processes Pace logger data to NetCDF

**RM Young Data Track:**
1. `02_clean_raw_rmyoung.py` - Processes Campbell TOA5 RM Young Weather
   tables to NetCDF, retaining within-interval diagnostics and creating QC plots

**Combined Processing:**
1. `05_add_pendants_to_intensive.py` - Merges pendant, Pace, and RM Young data
2. `07_lvl0_to_lvl1.py` - Further processing to create Level 1 data
3. `08_merge_lvl1_all_years.py` - Merges selected seasonal Level 1 datasets

```
Raw Data → Intermediate Processing → Level 0 → Level 1
    ↓              ↓                   ↓         ↓
[Pace .txt]   [Individual       [Combined    [5-min
[Pendant       sensor NetCDF]   datasets]    resampled]
 .csv]
```

**Key Architecture**: All processed data uses the **sensor_idx structure**. Data dimensions are `(sensor_idx, datetime_utc)` with sensor metadata stored as coordinates. `datetime_utc` is the canonical, timezone-naive UTC coordinate.

## Multi-season processing

Every pipeline command requires an explicit field-season year. Process each
season independently, then create the analysis-ready cross-season outputs after
all requested seasons have been regenerated:

```bash
uv run python scripts/data_pipeline/run_pipeline.py --year 2025
uv run python scripts/data_pipeline/run_pipeline.py --year 2026
uv run python scripts/data_pipeline/08_merge_lvl1_all_years.py --years 2025 2026
```

To resume a seasonal pipeline run from a numbered script, pass `--from-step`.
For example, this runs steps 4 through 7 for the 2026 field season:

```bash
uv run python scripts/data_pipeline/run_pipeline.py --year 2026 --from-step 4
```

Step 08 is optional and requires `--all-years`; to run only that cross-season
step, use `--from-step 8 --all-years 2025 2026`.

Step 08 writes all-years Level 1 files beneath `data/all_years/processed/lvl1/`,
for example `lvl1_on_ice_standard_all_years.nc` and
`lvl1_on_ice_intensive_all_years.nc`. Seasonal files remain within their own
`data/YYYY/processed/lvl1/` directories. Each sensor retains its observed
`site_id`; the all-years products additionally contain:

- `year(sensor_idx)`: field season of that sensor record.
- `canonical_site_id(sensor_idx)`: stable identity used to associate equivalent
  sites whose observed names changed between seasons.

Canonical associations are explicit in `src/jiflr/pipeline.py`. Add a
verified mapping before processing real renamed sites; unmatched sites retain
their observed ID as their canonical ID.

### Product filename migration

Processed filenames now distinguish `on_ice_standard` from
`on_ice_intensive`. Existing NetCDF files are not renamed or removed by the
pipeline. Before regenerating a season, archive its existing Level 0 and Level
1 products so the scripts do not also pick up files with the previous names.
After regenerating the seasons, rerun step 08 to refresh the all-years products.

## Processing Levels

- **Raw**: Unprocessed sensor exports (CSV, TXT, HOBO files)
- **Intermediate**: Individual sensor NetCDF files with sensor_idx structure
- **Level 0**: Combined datasets with standardized coordinates  
- **Level 1**: Resampled to regular 5-minute intervals

## Scripts

### 1. 01_clean_raw_pace.py

**Purpose**: Convert raw Pace logger data files to NetCDF format

**Input**: `data/2025/raw/pace/*.txt`  
**Output**: `data/2025/intermediate/pace/*.nc`

**Usage**:
```bash
uv run python scripts/data_pipeline/01_clean_raw_pace.py --year 2026
```

**Description**: Processes Pace meteorological station data (wind, temperature, pressure) exported as text files. Creates individual NetCDF files with sensor_idx structure and descriptive metadata. CF conformance has not yet been validated.

### 2. 02_clean_raw_rmyoung.py

**Purpose**: Convert raw R. M. Young Weather tables to intermediate NetCDF

**Input**: `data/YYYY/raw/rmyoung/*/*_Weather.dat`
**Output**: `data/YYYY/intermediate/rmyoung/*.nc`

**Usage**:
```bash
uv run python scripts/data_pipeline/02_clean_raw_rmyoung.py --year 2026
```

Raw timestamps are interpreted as `America/Anchorage` and stored as canonical
UTC. The per-season deployment manifest must contain one `rmyoung_logger`
record for each CR350 serial number. Intermediate files retain wind and
meteorological standard deviations, status codes, battery voltage, rain tips,
and maximum-wind timestamps. Level 0 retains only the analysis variables.

### 3. 03_clean_raw_pendants.py

**Purpose**: Convert raw HOBO pendant CSV exports to NetCDF format

**Input**: `data/2025/raw/pendants/exported/**/*.csv`  
**Output**: `data/2025/intermediate/pendants/by_sensor/**/*.nc`

**Usage**:
```bash
uv run python scripts/data_pipeline/03_clean_raw_pendants.py --year 2026
```

**Description**: Processes HOBO pendant temperature and light sensors exported from HOBOware/HOBOconnect. Handles both old and new generation sensors with different CSV formats, preserves directory structure, and converts each logger's stated timezone to UTC.

**Features**:
- Recursive processing of subdirectories
- Automatic shielding detection from data inventory
- Deployment period masking
- Timezone conversion from each logger's stated timezone to canonical UTC

### 4. 04_merge_raw_pendants_by_site.py

**Purpose**: Merge individual pendant sensors into site-based files

**Input**: `data/2025/intermediate/pendants/by_sensor/**/*.nc`  
**Output**: `data/2025/intermediate/pendants/by_site/**/*.nc`

**Usage**:
```bash
uv run python scripts/data_pipeline/04_merge_raw_pendants_by_site.py --year 2026
```

**Description**: Groups pendant sensors by site and concatenates them along the sensor_idx dimension. Creates one file per site with all sensors for that location. Applies deployment masks to filter non-deployment periods.

The hard-coded exception in this script averages the two 1 m G03 pendants
from 2025 by logger serial. The 2026 G03 records and both 2 m B01 sensor
generations remain separate.

**Features**:
- Site-based grouping with sensor_idx concatenation
- Quality control plot generation
- Deployment period filtering
- Handles the `camp_wx/`, `on_ice/`, `on_ice_intensive/`, and `off_ice/`
  intermediate groups. The `on_ice/` source group becomes the
  `on_ice_standard` processed product.

### 5. 05_add_pendants_to_intensive.py

**Purpose**: Combine Pace and pendant data into unified intensive site dataset

**Input**: 
- `data/2025/intermediate/pace/*.nc`
- `data/2025/intermediate/rmyoung/*.nc`
- `data/2025/intermediate/pendants/by_site/on_ice_intensive/*.nc`

**Output**: `data/2025/processed/lvl0/lvl0_on_ice_intensive.nc`

**Usage**:
```bash
uv run python scripts/data_pipeline/05_add_pendants_to_intensive.py --year 2026
```

**Description**: Creates a single on-ice intensive dataset combining meteorological data from Pace and RM Young stations with pendant sensor data. Uses sensor_idx concatenation to merge different sensor types into one unified structure.

**Features**:
- Site name standardization and mapping
- Common UTC datetime grid creation
- Mixed sensor type handling (pace + pendant)
- RM Young weather station integration
- Comprehensive dataset attributes

### 6. 06_merge_intermediate_to_lvl0.py

**Purpose**: Merge site-level files into combined datasets by directory structure

**Input**: `data/2025/intermediate/pendants/by_site/**/*.nc`  
**Output**: `data/2025/processed/lvl0/lvl0_*.nc`

**Usage**:
```bash
uv run python scripts/data_pipeline/06_merge_intermediate_to_lvl0.py --year 2026
```

**Description**: Combines multiple site files into larger datasets organized by intermediate group. The `on_ice/` group is named `on_ice_standard` in processed filenames. Uses tree-based merging for memory efficiency with large datasets, then applies every Level 0 quality-control filter in the documented order to every `lvl0_*.nc` product, including the on-ice intensive product created in step 05.

**Outputs**:
- `lvl0_on_ice_standard.nc` - Standard on-ice monitoring sites
- `lvl0_on_ice_intensive.nc` - On-ice intensive monitoring sites
- `lvl0_off_ice.nc` - Off-ice monitoring sites
- `lvl0_camp_wx.nc` - Camp weather station data, when present

**Features**:
- Tree-based pairwise merging for efficiency
- String coordinate length fixing
- Automatic subdirectory discovery
- Sequential, flag-first Level 0 quality control

#### Level 0 quality-control filters

Each season's `data/YYYY/metadata/lvl0_mask_periods.csv` is the pipeline input
for manually identified bad periods. The CSV may contain only its header when
there are no manual exclusions. It must use this exact header and column order:

```csv
start_datetime_utc,end_datetime_utc,site_id,sensor_id,variable,notes
```

`start_datetime_utc` and `notes` are required. `end_datetime_utc` is optional;
leave it blank to mask from the start time through the end of the time series.
Provided timestamps are inclusive UTC ISO 8601 datetimes. `site_id`,
`sensor_id`, and `variable` are optional exact-match selectors; leave a
selector blank to apply the row to all values of that selector. A `variable`
must name a numeric, time-indexed Level 0 measurement. Each row must match at
least one Level 0 measurement or processing fails, which prevents silently
ignored exclusions.

Step 06 runs the applicable filters in this order. The wind-speed plausibility
and rainfall allowlist filters apply only to 2026; the low-speed direction
filter applies to 2025 and 2026. Other listed filters apply to every season.
The seasonal `qc_filter_order` attribute records the filters actually run.

1. CSV manual-mask periods. The `notes` column records the scientific or
   operational reason for each exclusion.
2. Wind-speed plausibility. Every `wind_speed_*` value greater than 50 m/s is
   flagged.
3. Rainfall site allowlist. Every `rainfall_*` value outside sites `A04` and
   `A17` is flagged.
4. Low-speed wind direction. In 2025, Pace and RM Young `wind_direction`
   values are flagged when their same-site `wind_speed_avg` is below 0.5 m/s.
   In 2026, this filter applies to Pace `wind_direction` values only.
5. PACE pressure noise candidates. A centered 720-minute rolling-median
   Hampel rule flags pressure values whose residual exceeds both 0.125 kPa and
   `0.25 × (1.4826 × local MAD)`. At least seven samples are required in each
   rolling calculation.
6. Average and maximum wind speed noise candidates. The same centered
   720-minute rule flags `wind_speed_avg` and `wind_speed_max` separately when
   their residual exceeds both 10 m/s and `0.25 × (1.4826 × local MAD)`.
   The threshold can be explored in
   `notebooks/wind_speed_rolling_median_experiment.ipynb`.

The filters create a `uint32` `*_qc_flag` data variable alongside each
affected measurement. Flags are bitfields: `1` is `manual_mask_period` and
`2` is `wind_speed_exceeds_50_m_s`; `4` is
`rainfall_outside_allowed_sites`; `8` is `wind_direction_low_speed`. Filters
only add bits, so overlapping reasons are retained. `16` is
`pressure_noise_candidate`; `32` is `wind_speed_noise_candidate`. The source
CSV checksum is recorded in each seasonal Level 0 product when the CSV exists;
the release must include the corresponding CSV. Level 0 QC plots
show flagged measurements as red points over the blue measurement series.
Level 0 retains the observed pressure and wind speed values; Level 1 masks
flagged values to NaN.

### 7. 07_lvl0_to_lvl1.py

**Purpose**: Resample level 0 data to regular 5-minute intervals

**Input**: `data/2025/processed/lvl0/lvl0_*.nc`  
**Output**: `data/2025/processed/lvl1/lvl1_*.nc`

**Usage**:
```bash
uv run python scripts/data_pipeline/07_lvl0_to_lvl1.py --year 2026
```

**Description**: Resamples each Level 0 product to standardized 5-minute intervals using mean aggregation. Each output keeps the same product suffix, for example `lvl0_on_ice_standard.nc` becomes `lvl1_on_ice_standard.nc` and `lvl0_on_ice_intensive.nc` becomes `lvl1_on_ice_intensive.nc`.

**Features**:
- Mean aggregation resampling  
- Preserves existing 5-minute data regardless of offset
- Creates one Level 1 file per Level 0 product
- Sensor_idx structure preservation
- Preserves QC reason bits with a bitwise OR in each five-minute bin, then
  masks a Level 1 measurement whenever its paired masking `*_qc_flag` is
  nonzero
- Masks all flagged PACE pressure, average wind speed, and maximum wind speed
  outliers to NaN, regardless of run length; the Level 0 QC bits remain set
- Labels bins by their UTC start, with `datetime_utc_bounds` specifying
  `[start, start + 5 minutes)`. A single flagged input masks the entire bin.
- Does not interpolate gaps or apply the legacy diagnostic pressure filter

**Wind-direction masking**: Level 0 flags 2025 Pace and RM Young, and 2026
Pace, `wind_direction` observations where the matched same-site
`wind_speed_avg` is below 0.5 m/s. Level 1 then masks the flagged direction
values. Wind speed QC is applied independently using the rules above.

### 8. 08_merge_lvl1_all_years.py

**Purpose**: Combine fully processed Level 1 files from multiple seasons.

**Input**: `data/YYYY/processed/lvl1/lvl1_*.nc` for every requested year
**Output**: `data/all_years/processed/lvl1/lvl1_*_all_years.nc`

**Usage**:
```bash
uv run python scripts/data_pipeline/08_merge_lvl1_all_years.py --years 2025 2026
```

**Description**: Preserves each seasonal `site_id`, adds a
`canonical_site_id` from the explicit associations in `jiflr.pipeline`, and
concatenates available sensors from all requested years. A category absent for
one season (for example, 2026 off-ice data) yields an all-years file containing
the available seasons. Invalid or missing year coordinates still fail loudly.

## Data Structure

All processed data uses the **sensor_idx structure**:

**Dimensions**: `(sensor_idx, datetime_utc)`

**Sensor Metadata** (as coordinates):
- `site_id(sensor_idx)`: Site identifier (e.g., 'A01', 'Windward1')
- `height(sensor_idx)`: Sensor height (e.g., '1m', '2m')  
- `shielding(sensor_idx)`: Shielding type ('shielded', 'unshielded')
- `sensor_type(sensor_idx)`: Sensor type ('hobo pendant', 'pace')
- `sensor_id(sensor_idx)`: Unique sensor identifier
- `year(sensor_idx)`: Field season
- `canonical_site_id(sensor_idx)`: Stable cross-season site identity (all-years outputs only)

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
uv run python scripts/data_pipeline/01_clean_raw_pace.py --year 2026
uv run python scripts/data_pipeline/03_clean_raw_pendants.py --year 2026

# 2. Merge to site level
uv run python scripts/data_pipeline/04_merge_raw_pendants_by_site.py --year 2026

# 3. Combine different sensor types and create lvl0
uv run python scripts/data_pipeline/05_add_pendants_to_intensive.py --year 2026
uv run python scripts/data_pipeline/06_merge_intermediate_to_lvl0.py --year 2026

# 4. Create lvl1 resampled data
uv run python scripts/data_pipeline/07_lvl0_to_lvl1.py --year 2026

# 5. Build all-years Level 1 products after each season is complete
uv run python scripts/data_pipeline/08_merge_lvl1_all_years.py --years 2025 2026
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
    ├── deployment_manifest.csv       # Per-season machine-readable pipeline metadata
    ├── data_inventory.xlsx           # Human-readable sensor inventory
    └── deployment_periods.csv        # Human-readable field deployment record
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
