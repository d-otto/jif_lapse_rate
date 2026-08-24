import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as snd
import geopandas as gpd
import rioxarray as riox
import cartopy.crs as ccrs
from pathlib import Path
from datetime import datetime
from shapely.geometry import Polygon

plt.style.use('default')
plt.ioff()

mpl.rcParams['axes.linewidth'] = 0.1

# Define map extent
lonW = -136
lonE = -133.5
latS = 58.25
latN = 60.25

# Load DEM
p = Path("/Users/drotto/src/jiflr/data/external/arcticDEM_32.tif")
print(f"Loading DEM from {p}")
dem = riox.open_rasterio(p, chunks={"x":1000, "y":1000}, mask_and_scale=True).sel(band=1)
dem = dem.rio.reproject(ccrs.PlateCarree())
dem = dem.rio.clip_box(
    minx=lonW,
    miny=latS,
    maxx=lonE,
    maxy=latN,
    crs=ccrs.PlateCarree(),
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

# Create a simple JIF boundary for this simplified version
# Using approximate coordinates for the Juneau Icefield
print("Creating simplified JIF boundary...")
jif_coords = [
    (-134.8, 58.6), (-134.2, 58.5), (-133.9, 58.8), (-133.8, 59.1),
    (-134.0, 59.4), (-134.3, 59.6), (-134.6, 59.8), (-134.8, 59.6),
    (-135.0, 59.3), (-135.1, 58.9), (-134.8, 58.6)
]
jif_geom = Polygon(jif_coords)
jif = gpd.GeoDataFrame([{'geometry': jif_geom}], crs=ccrs.PlateCarree())

# Clip the DEM using the JIF polygon
try:
    jif_dem = dem.rio.clip(jif.geometry)
    print("JIF DEM clipped successfully")
except Exception as e:
    print(f"Warning: Could not clip JIF DEM: {e}")
    jif_dem = dem  # Use full DEM as fallback

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
ls = mpl.colors.LightSource(azdeg=30, altdeg=40)
dem_smooth = snd.gaussian_filter(dem.values, sigma=4)

# Create hillshade for off-ice terrain
off_ice_rgb = off_ice_cmap(norm(dem.values))
off_ice_hillshade = ls.shade_rgb(off_ice_rgb, elevation=dem_smooth, blend_mode='soft', vert_exag=1.0)

# Create hillshade for on-ice terrain
try:
    on_ice_rgb = on_ice_cmap(norm(jif_dem.values))
    on_ice_hillshade = ls.shade_rgb(on_ice_rgb, elevation=snd.gaussian_filter(jif_dem.values, sigma=4),
                                     blend_mode='soft', vert_exag=1.0)
    has_jif_hillshade = True
    print("JIF hillshade created successfully")
except Exception as e:
    print(f"Warning: Could not create JIF hillshade: {e}")
    has_jif_hillshade = False

# Set up figure with simple PlateCarree projection
fig = plt.figure(figsize=(8, 6), dpi=300, layout='constrained')
mapax = fig.add_subplot(111, projection=ccrs.PlateCarree())
mapax.set_extent([lonW+0.1, lonE-0.1, latS, latN], crs=ccrs.PlateCarree())

# Plot off-ice hillshaded topography
mapax.imshow(
    off_ice_hillshade,
    origin='lower',
    extent=[dem.x.min(), dem.x.max(), dem.y.max(), dem.y.min()],
    transform=ccrs.PlateCarree(),
    zorder=1.0
)

# Add a simple water mask by setting low elevation areas to water color
water_mask = dem.values < 10  # Areas below 10m elevation
water_overlay = np.ones_like(dem.values) * np.nan
water_overlay[water_mask] = 1

if np.any(~np.isnan(water_overlay)):
    mapax.imshow(
        water_overlay,
        origin='lower',
        extent=[dem.x.min(), dem.x.max(), dem.y.max(), dem.y.min()],
        transform=ccrs.PlateCarree(),
        cmap=mpl.colors.ListedColormap([water_color]),
        zorder=1.5
    )

# Plot on-ice hillshaded topography for JIF
if has_jif_hillshade:
    jif_bounds = tuple(np.array(jif_dem.rio.bounds())[[0, 2, 3, 1]])
    mapax.imshow(
        on_ice_hillshade,
        origin='lower',
        extent=jif_bounds,
        transform=ccrs.PlateCarree(),
        zorder=1.7
    )

# Plot JIF boundary with thick black outline
jif.plot(
    ax=mapax,
    facecolor='none',
    edgecolor='black',
    lw=3,
    transform=ccrs.PlateCarree(),
    zorder=1.9
)

# Remove axes and ticks for clean appearance
mapax.set_xticks([])
mapax.set_yticks([])
mapax.spines['top'].set_visible(False)
mapax.spines['right'].set_visible(False)
mapax.spines['bottom'].set_visible(False)
mapax.spines['left'].set_visible(False)

# Save figure
output_filename = f'jif-map-simple_{datetime.now().strftime("%Y%m%d-%H%M%S")}.png'
plt.savefig(output_filename, bbox_inches='tight', pad_inches=0.02)
print(f"Saved figure to {output_filename}")

# Also save as SVG for vector graphics
output_svg = f'jif-map-simple_{datetime.now().strftime("%Y%m%d-%H%M%S")}.svg'
plt.savefig(output_svg, bbox_inches='tight', pad_inches=0.02)
print(f"Saved figure to {output_svg}")

plt.close(fig)
print("Script completed successfully!")