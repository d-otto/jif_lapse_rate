import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import scipy.ndimage as snd
import geopandas as gpd
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import xarray as xr
import rioxarray as riox
from datetime import datetime
from shapely.geometry import Polygon
import textalloc as ta
import cmocean as cm

from jiflr import ROOT
from jiflr.data import select_sensors, merge_sensor_datasets

plt.style.use("default")
plt.ioff()

plt.rcParams.update(
    {
        "font.size": 8,  # base size — most things inherit from this
        "axes.labelsize": 8,  # x/y axis labels
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
        "axes.titlesize": 8,
    }
)


###################################################################################
# Map extent and projection
###################################################################################

lonW = -135
lonE = -133.5
latS = 58.35
latN = 59.15

extent = [lonW, lonE, latS, latN]
extent_polygon = Polygon([(lonW, latS), (lonW, latN), (lonE, latN), (lonE, latS)])

cLat = (latN + latS) / 2
cLon = (lonW + lonE) / 2
proj = ccrs.LambertConformal(central_longitude=cLon, central_latitude=cLat)


###################################################################################
# Load and prepare DEM
###################################################################################

dem = riox.open_rasterio(
    ROOT / "data/external/arcticDEM/arcticDEM_32.tif",
    chunks={"x": 1000, "y": 1000},
    mask_and_scale=True,
).sel(band=1)
dem = dem.rio.reproject(ccrs.PlateCarree())
dem = dem.rio.clip_box(
    minx=lonW, miny=latS, maxx=lonE, maxy=latN, crs=ccrs.PlateCarree()
)
dem.values = np.where(dem.values < 0, np.nan, dem.values)


def nn_fill(arr):
    mask = np.isnan(arr)
    idx = np.where(~mask, np.arange(mask.shape[1]), 0)
    np.maximum.accumulate(idx, axis=1, out=idx)
    arr[mask] = arr[np.nonzero(mask)[0], idx[mask]]
    return arr


dem.values = nn_fill(dem.values)

vmin_dem = 0
vmax_dem = 2500
terrain_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "terrain_map", [(0, "#C0C0C0"), (1, "#FAFAFA")]
)
norm_dem = mpl.colors.Normalize(vmin=vmin_dem, vmax=vmax_dem)
ls = mpl.colors.LightSource(azdeg=30, altdeg=40)
hillshade = ls.shade_rgb(
    terrain_cmap(norm_dem(dem.values)),
    elevation=snd.gaussian_filter(dem.values, sigma=4),
    blend_mode="soft",
    vert_exag=1.0,
)


###################################################################################
# Load vector data (glacier boundaries, ocean, lakes)
###################################################################################

rgi = gpd.read_file(ROOT / "data/external/rgi7/RGI2000-v7.0-G-01_alaska")
rgic = gpd.read_file(ROOT / "data/external/rgi7/RGI2000-v7.0-C-01_alaska")
rgic = rgic.sort_values(by="area_km2", ascending=False)

jif = rgic.iloc[[5]].to_crs(ccrs.PlateCarree())
rgi = rgi.cx[lonW:lonE, latS:latN].to_crs(crs=ccrs.PlateCarree())
rgi_jif = rgi.clip(jif)
rgi_other = rgi.loc[[idx for idx in rgi.index if idx not in rgi_jif.index]]

jif_dem = dem.rio.clip(jif.geometry)
dem_icefree = dem.rio.clip(rgi.geometry, invert=True)

ocean = gpd.GeoSeries(
    cfeature.GSHHSFeature(scale="full", levels=[1]).intersecting_geometries(extent)
)
lakes = gpd.GeoSeries(
    cfeature.GSHHSFeature(scale="full", levels=[2]).intersecting_geometries(extent)
)
islands = gpd.GeoSeries(
    cfeature.GSHHSFeature(scale="full", levels=[3]).intersecting_geometries(extent)
)
ocean = gpd.GeoSeries(extent_polygon).symmetric_difference(ocean.union_all())
lakes = lakes.difference(islands.union_all())
ocean_color = lake_color = "#E8F3F8"


###################################################################################
# Load sensor data
###################################################################################

# Load pendant data
ds_pendants = xr.open_dataset(ROOT / "data/2025/processed/lvl1/lvl1_on_ice.nc")
ds_pendants = select_sensors(ds_pendants, height="1m", shielding="shielded")
# Drop sensors missing coordinates (e.g. G03A, G03B)
ds_pendants = ds_pendants.isel(sensor_idx=~np.isnan(ds_pendants.latitude.values))

# Load intensive site data and filter to shielded pendants
ds_intensive = xr.open_dataset(
    ROOT / "data/2025/processed/lvl1/lvl1_on_ice_intensive.nc"
)
ds_intensive = select_sensors(ds_intensive, height="1m" , shielding="shielded")

# Merge datasets
ds = merge_sensor_datasets(ds_pendants, ds_intensive)

# Build site metadata from sensor coordinates
site_ids = ds.site_id.values
site_meta = {
    str(sid): {
        "lat": float(ds.latitude.isel(sensor_idx=i).values),
        "lon": float(ds.longitude.isel(sensor_idx=i).values),
        "elev": int(ds.elevation.isel(sensor_idx=i).values),
    }
    for i, sid in enumerate(site_ids)
}

# resample to 15 min
ds = ds.resample(datetime="1h").mean()
ds = ds.dropna(dim="datetime", how="all", subset=["temp_c"])


###################################################################################
# Temperature colormap (consistent across all frames)
###################################################################################


class SplitSymLogNorm(mpl.colors.Normalize):
    def __init__(self, linthresh, vmin, vmax, linscale=1.0):
        super().__init__(vmin=vmin, vmax=vmax)
        self.linthresh = linthresh
        self.linscale = linscale
        self._neg_norm = mpl.colors.SymLogNorm(
            linthresh, linscale=linscale, vmin=vmin, vmax=0
        )
        self._pos_norm = mpl.colors.SymLogNorm(
            linthresh, linscale=linscale, vmin=0, vmax=vmax
        )

    def __call__(self, value, clip=None):
        value = np.asarray(value, dtype=float)
        result = np.empty_like(value)

        neg = value < 0
        pos = value >= 0

        if neg.any():
            result[neg] = self._neg_norm(value[neg]) * 0.5
        if pos.any():
            result[pos] = 0.5 + self._pos_norm(value[pos]) * 0.5

        if np.ma.is_masked(value):
            result = np.ma.array(result, mask=np.ma.getmask(value))

        return result

    def inverse(self, value):
        value = np.asarray(value, dtype=float)
        result = np.empty_like(value)

        neg = value < 0.5
        pos = value >= 0.5

        if neg.any():
            # [0, 0.5] -> [vmin, 0]: undo the *0.5 scaling, then invert SymLogNorm
            result[neg] = self._neg_norm.inverse(value[neg] * 2.0)
        if pos.any():
            # [0.5, 1.0] -> [0, vmax]: undo the 0.5+ offset and *0.5 scaling
            result[pos] = self._pos_norm.inverse((value[pos] - 0.5) * 2.0)

        return result


vmin_temp = -5
vmax_temp = 20

n = 256
thermal_colors = mpl.cm.rainbow(np.linspace(0, 1, n))
sub0_colors = cm.cm.gray(np.linspace(0, 1, n))
combined = np.vstack([sub0_colors, thermal_colors])
temp_cmap = mpl.colors.LinearSegmentedColormap.from_list("ice_thermal", combined)

# temp_norm = SplitSymLogNorm(linthresh=1, linscale=0.5, vmin=vmin_temp, vmax=vmax_temp)
temp_norm = mpl.colors.TwoSlopeNorm(vcenter=0, vmin=vmin_temp, vmax=vmax_temp)
temp_sm = mpl.cm.ScalarMappable(cmap=temp_cmap, norm=temp_norm)
temp_sm.set_array([])


###################################################################################
# Build static map
###################################################################################

lat_mid = (latN + latS) / 2
aspect_ratio = (lonE - lonW) * np.cos(np.radians(lat_mid)) / (latN - latS)
width = 3.25
fig = plt.figure(
    figsize=(width, width / aspect_ratio), dpi=300, layout="constrained", linewidth=0.25
)
mapax = fig.add_subplot(111, projection=proj)
mapax.set_extent([lonW, lonE, latS, latN], crs=ccrs.PlateCarree())

mapax.imshow(
    hillshade,
    origin="lower",
    extent=[
        dem.x.min().item(),
        dem.x.max().item(),
        dem.y.max().item(),
        dem.y.min().item(),
    ],
    transform=ccrs.PlateCarree(),
)

ocean.plot(ax=mapax, fc=ocean_color, ec="black", lw=0.25, transform=ccrs.PlateCarree())
lakes.plot(ax=mapax, fc=lake_color, ec="black", lw=0.25, transform=ccrs.PlateCarree())

rgi_other.plot(
    ax=mapax,
    legend=False,
    transform=ccrs.PlateCarree(),
    color="#FFF",
    lw=0.1,
    ec="#4B4B4B",
    zorder=1.9,
)

jif.plot(
    ax=mapax,
    legend=False,
    lw=0.25,
    # ec="#c0c0c0",
    ec="#6B95B9",
    facecolor="none",
    transform=ccrs.PlateCarree(),
    zorder=1.9,
    marker="",
)

jif_bounds = tuple(np.array(jif_dem.rio.bounds())[[0, 2, 3, 1]])
# jif_cmap = mpl.colors.LinearSegmentedColormap.from_list(
#     "jif_cmap", plt.cm.gray_r(np.linspace(0.15, 0, 256))
# )
jif_cmap = mpl.colors.LinearSegmentedColormap.from_list(
    "jif_cmap", ["#DBE8F4", "#ffffff"]
)

cs_fill = mapax.contourf(
    jif_dem,
    cmap=jif_cmap,
    zorder=1.8,
    extent=jif_bounds,
    levels=np.arange(vmin_dem, vmax_dem - 200, 100),
    transform=ccrs.PlateCarree(),
)

cs_lines = mapax.contour(
    jif_dem,
    zorder=1.9,
    extent=jif_bounds,
    levels=cs_fill.levels,  # reuse levels from contourf
    colors="#6B95B9",
    linewidths=0.1,
    transform=ccrs.PlateCarree(),
)
mapax.clabel(
    cs_lines,
    cs_lines.levels[::2],
    inline=True,
    inline_spacing=2,
    zorder=1.95,
    fontsize=1.5,
    # fontweight="bold",
    fmt="%d",
)

mapax.gridlines(
    draw_labels=["bottom", "right"],
    linewidth=0.1,
    color="black",
    alpha=1,
    linestyle="-",
    xlocs=mpl.ticker.MultipleLocator(1),
    ylocs=mpl.ticker.MultipleLocator(0.5),
    x_inline=False,
    y_inline=False,
    xlabel_style={"fontsize": "4", "rotation": 0, "ha": "center"},
    ylabel_style={"fontsize": "4", "rotation": 270, "ha": "center"},
    xpadding=4,
    ypadding=2.5,
    rotate_labels=90,
)
mapax.gridlines(
    draw_labels=False,
    linewidth=0.1,
    color="black",
    alpha=0.25,
    linestyle="-",
    xlocs=mpl.ticker.MultipleLocator(0.25),
    ylocs=mpl.ticker.MultipleLocator(0.25),
)


###################################################################################
# Permanent site labels (site id + elevation, non-animated)
###################################################################################

# for sid, meta in site_meta.items():
#     ta.allocate_text(
#         fig,
#         mapax,
#         meta["lon"],
#         meta["lat"],
#         f"{sid} ({meta['elev']}m)",
#         fontsize=2.5,
#         ha="left",
#         va="bottom",
#         transform=ccrs.PlateCarree(),
#         zorder=7,
#         color="black",
#         clip_on=True,
#     )

lons = [meta["lon"] for meta in site_meta.values()]
lats = [meta["lat"] for meta in site_meta.values()]
labels = [f"{sid} ({meta['elev']}m)" for sid, meta in site_meta.items()]

# ta.allocate(
#     mapax,
#     lons,
#     lats,
#     labels,
#     x_scatter=lons,
#     y_scatter=lats,
#     textsize=2.5,
#     textcolor="black",
#     fontweight="bold",
#     linecolor="k",
#     linewidth=0.25,
#     transform=ccrs.PlateCarree(),
#     avoid_label_lines_overlap=True,
#     avoid_crossing_label_lines=True,
#     # direction="southeast",
# )


###################################################################################
# Timestamp annotation (updated each frame)
###################################################################################

ts_text = mapax.text(
    0.02,
    0.02,
    "",
    transform=mapax.transAxes,
    fontsize=5,
    ha="left",
    va="bottom",
    zorder=8,
    color="black",
    bbox=dict(fc="white", ec="none", alpha=0.7, pad=1),
)

cb = fig.colorbar(
    temp_sm,
    ax=mapax,
    orientation="horizontal",
    fraction=0.025,
    pad=0.01,
    aspect=35,
)
cb.ax.tick_params(labelsize=4)  # tick label size
cb.set_label("Temperature (°C)", fontsize=4, labelpad=2)  # colorbar label size
# Or set ticks manually for full control:
# cb.set_ticks([-5, -2, -1, 0, 1, 2, 5, 10, 25])
# cb.set_ticklabels(["-5", "-2", "-1", "0", "1", "2", "5", "10", "25"])

mapax.set_title(
    "Juneau Icefield, Summer 2025", loc="left", fontweight="bold", fontsize=4, pad=2
)
mapax.set_title(
    "1 m Ta [°C], resampled to hourly mean",
    loc="right",
    fontweight="normal",
    fontsize=4,
    pad=2,
)

###################################################################################
# Animation frame function (blit-compatible)
###################################################################################

frame_artists = []


def update_frame(t_idx):
    for artist in frame_artists:
        artist.remove()
    frame_artists.clear()

    temp_slice = ds.temp_c.isel(datetime=t_idx).values  # shape: (sensor_idx,)
    t = ds.datetime.values[t_idx]

    for i, sid in enumerate(site_ids):
        temp_val = float(temp_slice[i])
        if np.isnan(temp_val):
            continue
        meta = site_meta[str(sid)]
        color = temp_cmap(temp_norm(temp_val))

        sc = mapax.scatter(
            meta["lon"],
            meta["lat"],
            color=color,
            s=6,
            zorder=5,
            transform=ccrs.PlateCarree(),
            edgecolors="black",
            linewidths=0.3,
        )
        ann = mapax.annotate(
            f"{temp_val:.1f}",
            xy=(meta["lon"], meta["lat"]),
            xytext=(2, -1),
            textcoords="offset points",
            fontsize=3,
            fontweight="bold",
            zorder=6,
            color=color if temp_val > 0 else "#000",
            xycoords=ccrs.PlateCarree()._as_mpl_transform(mapax),
            clip_on=True,
        )
        frame_artists.extend([sc, ann])
        # frame_artists.extend([sc])
        # frame_artists.extend([ann])

    ts_text.set_text(pd.Timestamp(t).strftime("%Y-%m-%d %H:%M AKDT"))
    return frame_artists + [ts_text]


###################################################################################
# Single frame output
# To animate: from matplotlib.animation import FuncAnimation
#   anim = FuncAnimation(fig, update_frame, frames=len(ds.datetime), blit=True, interval=100)
#   anim.save("jif_animation.mp4", fps=10, dpi=150)
###################################################################################

# # Find first timestep with at least one valid temperature value
# first_valid = int(np.argmax(ds.temp_c.notnull().any(dim="sensor_idx").values))
# # update_frame(first_valid)
# update_frame(300)

# plt.savefig(
#     f"jif_map_temp_anim_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png",
#     bbox_inches="tight",
# )

from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm


frame_idxs = range(0, len(ds.datetime))
# frame_idxs = range(len(ds.datetime) - 100, len(ds.datetime))

writer = FFMpegWriter(
    fps=12,
    bitrate=8000,
    extra_args=["-vcodec", "libx264", "-crf", "18"],
)


with tqdm(total=len(frame_idxs), desc="Saving animation") as pbar:
    anim = FuncAnimation(fig, update_frame, frames=frame_idxs, blit=True)
    anim.save(
        f"jif_animation_{datetime.now().strftime('%Y%m%d-%H%M%S')}.mp4",
        writer=writer,
        dpi=600,
        progress_callback=lambda i, n: pbar.update(1),
    )
