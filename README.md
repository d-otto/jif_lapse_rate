### Basic Data Access
```python
# Get all sensors at a site
site_data = ds.where(ds.site_id == 'A01', drop=True)

# Get sensors at specific height
height_data = ds.where(ds.height == '2m', drop=True)

# Get specific sensor
sensor = ds.where((ds.site_id == 'A01') & (ds.height == '2m') & (ds.shielding == 'shielded'), drop=True)
```

### Aggregation Patterns
```python
# Site means using groupby (recommended for complex aggregations)
site_means = ds.groupby('site_id').mean()

# Manual aggregation for custom operations
site_means = []
for site in np.unique(ds.site_id.values):
    site_data = ds.where(ds.site_id == site, drop=True)
    mean_temp = site_data.temp_c.mean()
    site_means.append(mean_temp)
```

### Analysis Examples
```python
# Temperature lapse rate analysis
temps = ds.temp_c.mean(dim='datetime', skipna=True)
elevations = ds.elevation
# Now you have temps and elevations aligned by sensor_idx

# Height effect analysis  
temp_1m = ds.where(ds.height == '1m', drop=True).temp_c.mean()
temp_2m = ds.where(ds.height == '2m', drop=True).temp_c.mean()
height_effect = temp_2m - temp_1m
```