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
from jiflr.plot import jif_map

###################################################################################
# Load sensor data
###################################################################################


# Load pendant data
ds_pendants = xr.open_dataset(ROOT / "data/2025/processed/lvl1/lvl1_on_ice.nc")
ds_pendants = select_sensors(ds_pendants, height="2m", shielding="shielded")
# Drop sensors missing coordinates (e.g. G03A, G03B)
ds_pendants = ds_pendants.isel(sensor_idx=~np.isnan(ds_pendants.latitude.values))

# Load intensive site data and filter to shielded pendants
ds_intensive = xr.open_dataset(
    ROOT / "data/2025/processed/lvl1/lvl1_on_ice_intensive.nc"
)
ds_intensive = select_sensors(ds_intensive, height="2m", shielding="shielded")

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
da = ds.temp_c.resample(datetime="1h").mean(skipna=True)
da = da.dropna(dim="datetime", how="all")

# Remove the mean temperature at every time period
da_mean = da.mean(dim="sensor_idx")
# T_mean = np.mean(da)
da = ds.temp_c - da_mean


###################################################################################
# Create base map
###################################################################################

fig, mapax = jif_map()


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
# thermal_colors = mpl.cm.rainbow(np.linspace(0, 1, n))
# sub0_colors = cm.cm.gray(np.linspace(0, 1, n))
# combined = np.vstack([sub0_colors, thermal_colors])
# temp_cmap = mpl.colors.LinearSegmentedColormap.from_list("ice_thermal", combined)
temp_cmap = mpl.cm.RdYlBu

# temp_norm = SplitSymLogNorm(linthresh=1, linscale=0.5, vmin=vmin_temp, vmax=vmax_temp)

temp_norm = mpl.colors.CenteredNorm(vcenter=0, halfrange=5, clip=True)
temp_sm = mpl.cm.ScalarMappable(cmap=temp_cmap, norm=temp_norm)
temp_sm.set_array([])

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

tmean_text = mapax.text(
    0.98,
    0.02,
    "",
    transform=mapax.transAxes,
    fontsize=5,
    ha="right",
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
    "2 m Ta [°C], resampled to hourly mean",
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

    temp_slice = da.isel(datetime=t_idx).values  # shape: (sensor_idx,)
    t = da.datetime.values[t_idx]

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
            color="#000",
            xycoords=ccrs.PlateCarree()._as_mpl_transform(mapax),
            clip_on=True,
        )
        frame_artists.extend([sc, ann])
        # frame_artists.extend([sc])
        # frame_artists.extend([ann])

    ts_text.set_text(pd.Timestamp(t).strftime("%Y-%m-%d %H:%M AKDT"))
    tmean_text.set_text(f"Tmean = {da_mean[t_idx].item():.1f} °C")
    # tmean_text.set_text(f"Tmean = {T_mean:.1f} °C")

    return frame_artists + [ts_text, tmean_text]


###################################################################################
# Single frame output
# To animate: from matplotlib.animation import FuncAnimation
#   anim = FuncAnimation(fig, update_frame, frames=len(ds.datetime), blit=True, interval=100)
#   anim.save("jif_animation.mp4", fps=10, dpi=150)
###################################################################################

# update_frame(700)

# plt.savefig(
#     f"jif_map_temp_anim_{datetime.now().strftime('%Y%m%d-%H%M%S')}.png",
#     bbox_inches="tight", dpi=600
# )

from matplotlib.animation import FuncAnimation, FFMpegWriter
from tqdm import tqdm


frame_idxs = range(0, len(da.datetime))
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
