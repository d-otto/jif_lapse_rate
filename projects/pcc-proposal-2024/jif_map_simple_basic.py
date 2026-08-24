import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as snd
import geopandas as gpd
import rioxarray as riox
from pathlib import Path
from datetime import datetime
from shapely.geometry import Polygon

plt.style.use('default')
plt.ioff()

# Define map extent
lonW = -136
lonE = -133.5
latS = 58.25
latN = 60.25

# Load DEM
p = Path("/Users/drotto/src/jiflr/data/external/arcticDEM_32.tif")
print(f"Loading DEM from {p}")
dem = riox.open_rasterio(p, chunks={"x":1000, "y":1000}, mask_and_scale=True).sel(band=1)
dem = dem.rio.clip_box(
    minx=lonW,
    miny=latS,
    maxx=lonE,
    maxy=latN,
    crs="EPSG:4326",
)
dem.values = np.where(dem.values < 0, np.nan, dem.values)

# Fill NaN values using nearest neighbor
def nn_fill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    arr[mask] = arr[np.nonzero(mask)[0], idx[mask]]
    return arr

dem.values = nn_fill(dem.values)
print(f"DEM loaded with shape: {dem.shape}")

# Create a simple JIF boundary
print("Creating simplified JIF boundary...")
jif_coords = [
    (-134.8, 58.6), (-134.2, 58.5), (-133.9, 58.8), (-133.8, 59.1),
    (-134.0, 59.4), (-134.3, 59.6), (-134.6, 59.8), (-134.8, 59.6),
    (-135.0, 59.3), (-135.1, 58.9), (-134.8, 58.6)
]
jif_geom = Polygon(jif_coords)
jif = gpd.GeoDataFrame([{'geometry': jif_geom}], crs="EPSG:4326")

# Create mask for JIF area using rasterio
print("Creating JIF mask...")
try:
    from rasterio.features import rasterize
    from rasterio.transform import from_bounds
    
    # Create a transform for the DEM
    transform = from_bounds(dem.x.min(), dem.y.min(), dem.x.max(), dem.y.max(), 
                           dem.sizes['x'], dem.sizes['y'])
    
    # Rasterize the JIF geometry
    jif_mask = rasterize([jif_geom], out_shape=dem.shape, transform=transform, fill=0, default_value=1)
    jif_mask = jif_mask.astype(bool)
    print(f"JIF mask created with {np.sum(jif_mask)} pixels inside JIF")
    
    # If rasterization failed, use bounding box
    if np.sum(jif_mask) == 0:
        print("Rasterization returned empty mask, using bounding box fallback...")
        x_mask = (dem.x >= -135.0) & (dem.x <= -133.8)
        y_mask = (dem.y >= 58.7) & (dem.y <= 59.5)
        jif_mask = np.outer(y_mask, x_mask)
        print(f"Fallback JIF mask created with {np.sum(jif_mask)} pixels inside JIF")
    
except Exception as e:
    print(f"Error creating JIF mask: {e}")
    print("Using simple bounding box mask instead...")
    # Simple fallback: use approximate bounding box
    x_mask = (dem.x >= -135.0) & (dem.x <= -133.8)
    y_mask = (dem.y >= 58.7) & (dem.y <= 59.5)
    jif_mask = np.outer(y_mask, x_mask)

# Define greyscale colormaps
vmin = 0
vmax = 2500

# Off-ice colormap: dark grey to medium grey (never reaching white)
off_ice_colors = [(0, "#333333"),           # dark grey at low elevation
                  (750/vmax, "#555555"),     # medium-dark grey
                  (1250/vmax, "#777777"),    # medium grey
                  (2000/vmax, "#999999"),    # light-medium grey
                  (1, "#AAAAAA")]            # lighter grey (but not white)

# On-ice colormap: light grey to white
on_ice_colors = [(0, "#CCCCCC"),            # light grey at low elevation
                 (0.5, "#DDDDDD"),          # lighter grey
                 (1, "#FFFFFF")]            # white at high elevation

# Water color: single dark grey
water_color = "#555555"

off_ice_cmap = mpl.colors.LinearSegmentedColormap.from_list('off_ice_grey', off_ice_colors)
on_ice_cmap = mpl.colors.LinearSegmentedColormap.from_list('on_ice_grey', on_ice_colors)

norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)

# Create hillshade for terrain relief
print("Creating hillshade...")
ls = mpl.colors.LightSource(azdeg=30, altdeg=40)
dem_smooth = snd.gaussian_filter(dem.values, sigma=4)

# Create separate colormaps for on-ice and off-ice areas
dem_off_ice = np.where(jif_mask, np.nan, dem.values)
dem_on_ice = np.where(jif_mask, dem.values, np.nan)

# Create hillshade for off-ice terrain
off_ice_rgb = off_ice_cmap(norm(dem_off_ice))
off_ice_hillshade = ls.shade_rgb(off_ice_rgb, elevation=dem_smooth, blend_mode='soft', vert_exag=1.0)

# Create hillshade for on-ice terrain
on_ice_rgb = on_ice_cmap(norm(dem_on_ice))
on_ice_hillshade = ls.shade_rgb(on_ice_rgb, elevation=dem_smooth, blend_mode='soft', vert_exag=1.0)

print("Hillshades created successfully")

# Set up figure - use simple matplotlib without cartopy
fig, ax = plt.subplots(figsize=(10, 8), dpi=300)

# Plot off-ice hillshaded topography
extent = [dem.x.min(), dem.x.max(), dem.y.min(), dem.y.max()]
im1 = ax.imshow(
    off_ice_hillshade,
    extent=extent,
    origin='lower',
    zorder=1.0,
    aspect='equal'
)

# Plot on-ice hillshaded topography
im2 = ax.imshow(
    on_ice_hillshade,
    extent=extent,
    origin='lower',
    zorder=1.5,
    aspect='equal'
)

# Add water mask for low elevation areas
water_mask = dem.values < 10  # Areas below 10m elevation
water_overlay = np.ma.masked_where(~water_mask, np.ones_like(dem.values))

if np.any(~water_overlay.mask):
    ax.imshow(
        water_overlay,
        extent=extent,
        origin='lower',
        cmap=mpl.colors.ListedColormap([water_color]),
        zorder=1.3,
        aspect='equal'
    )

# Plot JIF boundary with thick black outline
jif_x, jif_y = jif_geom.exterior.xy
ax.plot(jif_x, jif_y, color='black', linewidth=3, zorder=2.0)

# Set the map extent
ax.set_xlim(lonW+0.1, lonE-0.1)
ax.set_ylim(latS, latN)

# Remove axes for clean appearance
ax.set_xticks([])
ax.set_yticks([])
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Save figure
output_filename = f'jif-map-simple_{datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.02, dpi=300)
print(f"Saved figure to {output_filename}")

# Also save as SVG for vector graphics
output_svg = f'jif-map-simple_{datetime.now().strftime("%Y%m%d-%H%M%S")}.svg'
plt.savefig(output_svg, bbox_inches='tight', pad_inches=0.02)
print(f"Saved figure to {output_svg}")

plt.close(fig)
print("Script completed successfully!")