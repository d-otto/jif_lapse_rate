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


def jif_map():

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
    # Build static map
    ###################################################################################

    lat_mid = (latN + latS) / 2
    aspect_ratio = (lonE - lonW) * np.cos(np.radians(lat_mid)) / (latN - latS)
    width = 3.25
    fig = plt.figure(
        figsize=(width, width / aspect_ratio),
        dpi=300,
        layout="constrained",
        linewidth=0.25,
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

    ocean.plot(
        ax=mapax, fc=ocean_color, ec="black", lw=0.25, transform=ccrs.PlateCarree()
    )
    lakes.plot(
        ax=mapax, fc=lake_color, ec="black", lw=0.25, transform=ccrs.PlateCarree()
    )

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

    return fig, mapax
