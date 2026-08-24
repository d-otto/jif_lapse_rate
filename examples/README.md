# JIFLR Examples

This folder contains example notebooks and scripts demonstrating how to work with JIFLR project data.

## Contents

### pendant_data_analysis_example.ipynb
A comprehensive Jupyter notebook showing how to:
- Load and explore processed pendant sensor data
- Create temperature time series plots
- Analyze temperature lapse rates between heights
- Compare temperatures across multiple sites
- Work with light intensity data (when available)
- Perform data selection and filtering
- Assess data quality and coverage

## Data Structure

The processed JIFLR data uses a clean dimensional structure:

```
Dimensions:
- site_id: Measurement sites (A01, A02, B01, etc.)
- height: Sensor heights (1m, 2m, etc.)
- shielding: Sensor shielding type (shielded, unshielded)
- datetime: Time dimension

Variables:
- temp_c(site_id, height, shielding, datetime): Temperature in °C
- intensity_lux(site_id, height, shielding, datetime): Light intensity in lux
```

## Getting Started

1. Ensure you have the required packages installed:
   ```bash
   pip install numpy pandas xarray matplotlib seaborn
   ```

2. Open the example notebook:
   ```bash
   jupyter notebook pendant_data_analysis_example.ipynb
   ```

3. Follow the examples to learn the data access patterns

## Data Selection Examples

```python
import xarray as xr

# Load the data
ds = xr.open_dataset("../data/2025/processed/lvl0/lvl0_main.nc")

# Select all 2m sensors
data_2m = ds.sel(height="2m")

# Select specific sites
sites_abc = ds.sel(site_id=["A01", "A02", "B01"])

# Select time period
july_data = ds.sel(datetime=slice("2025-07-01", "2025-07-31"))

# Calculate lapse rate
lapse_rate = ds.sel(height="2m") - ds.sel(height="1m")
```

## Contributing

Feel free to add more example notebooks or scripts that demonstrate other analysis techniques!