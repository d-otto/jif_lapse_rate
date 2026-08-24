#!/usr/bin/env python3
"""
plotting.py

Standardized plotting utilities for the JIFLR project.
Provides consistent styling, formatting, and common plot types for temperature sensor analysis.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
from matplotlib.dates import DateFormatter, DayLocator, HourLocator
import numpy as np
import pandas as pd
import xarray as xr
from typing import List, Optional, Dict, Any, Tuple
import warnings

# Standard styling configuration
STYLE_CONFIG = {
    'figure': {
        'figsize': (12, 8),
        'dpi': 150,
        'facecolor': 'white'
    },
    'axes': {
        'grid': True,
        'grid_alpha': 0.3,
        'grid_linewidth': 0.5,
        'spines_to_hide': ['top', 'right'],
        'minor_grid': True,
        'minor_grid_alpha': 0.2
    },
    'temperature': {
        'major_tick_interval': 5,
        'minor_tick_interval': 1,
        'ylabel': 'Temperature (°C)',
        'reference_line_0c': True
    },
    'datetime': {
        'major_locator': DayLocator(),
        'minor_locator': HourLocator(interval=12),
        'formatter': DateFormatter("%b %d"),
        'xlabel': 'Date',
        'rotation': 45
    },
    'colors': {
        'maritime': ['#2E8B57', '#4682B4', '#6495ED'],  # Sea green, steel blue, cornflower blue
        'continental': ['#CD853F', '#D2691E', '#A0522D'],  # Sandy brown, chocolate, sienna
        'default': ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
        'reference_line': '#888888'
    }
}

def apply_plot_styling(ax, style_type='default', **kwargs):
    """
    Apply standardized styling to a matplotlib axis.
    
    Parameters:
    -----------
    ax : matplotlib.axes.Axes
        The axis to style
    style_type : str
        Type of styling to apply ('default', 'temperature', 'datetime')
    **kwargs : dict
        Override any default styling parameters
    """
    # Get configuration with overrides
    config = STYLE_CONFIG['axes'].copy()
    config.update(kwargs)
    
    # Apply basic styling
    if config.get('grid', True):
        ax.grid(True, alpha=config.get('grid_alpha', 0.3), 
               linewidth=config.get('grid_linewidth', 0.5))
    
    # Apply minor grid if requested
    if config.get('minor_grid', True):
        ax.grid(True, which='minor', alpha=config.get('minor_grid_alpha', 0.2))
        ax.minorticks_on()
    
    # Hide spines
    spines_to_hide = config.get('spines_to_hide', ['top', 'right'])
    for spine in spines_to_hide:
        if spine in ax.spines:
            ax.spines[spine].set_visible(False)
    
    # Apply style-specific formatting
    if style_type == 'temperature':
        _apply_temperature_styling(ax, **kwargs)
    elif style_type == 'datetime':
        _apply_datetime_styling(ax, **kwargs)

def _apply_temperature_styling(ax, **kwargs):
    """Apply temperature-specific axis styling."""
    temp_config = STYLE_CONFIG['temperature'].copy()
    temp_config.update(kwargs)
    
    # Set temperature axis formatting
    major_interval = temp_config.get('major_tick_interval', 5)
    minor_interval = temp_config.get('minor_tick_interval', 1)
    
    ax.yaxis.set_major_locator(MultipleLocator(major_interval))
    ax.yaxis.set_minor_locator(MultipleLocator(minor_interval))
    
    # Set ylabel if not already set
    if not ax.get_ylabel():
        ax.set_ylabel(temp_config.get('ylabel', 'Temperature (°C)'))
    
    # Add 0°C reference line if requested
    if temp_config.get('reference_line_0c', True):
        ax.axhline(y=0, color=STYLE_CONFIG['colors']['reference_line'], 
                  linestyle='--', alpha=0.7, linewidth=1)

def _apply_datetime_styling(ax, **kwargs):
    """Apply datetime-specific axis styling."""
    dt_config = STYLE_CONFIG['datetime'].copy()
    dt_config.update(kwargs)
    
    # Set datetime formatting
    if 'major_locator' in dt_config:
        ax.xaxis.set_major_locator(dt_config['major_locator'])
    if 'minor_locator' in dt_config:
        ax.xaxis.set_minor_locator(dt_config['minor_locator'])
    if 'formatter' in dt_config:
        ax.xaxis.set_major_formatter(dt_config['formatter'])
    
    # Set xlabel if not already set
    if not ax.get_xlabel():
        ax.set_xlabel(dt_config.get('xlabel', 'Date'))
    
    # Apply rotation
    rotation = dt_config.get('rotation', 45)
    if rotation:
        ax.tick_params(axis='x', rotation=rotation)

def create_standard_figure(nrows=1, ncols=1, figsize=None, **kwargs):
    """
    Create a standardized figure with consistent styling.
    
    Parameters:
    -----------
    nrows, ncols : int
        Number of subplot rows and columns
    figsize : tuple, optional
        Figure size, defaults to STYLE_CONFIG values
    **kwargs : dict
        Additional arguments passed to plt.subplots()
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    if figsize is None:
        base_figsize = STYLE_CONFIG['figure']['figsize']
        figsize = (base_figsize[0] * ncols, base_figsize[1] * nrows)
    
    fig_config = STYLE_CONFIG['figure'].copy()
    fig_config.update(kwargs)
    fig_config['figsize'] = figsize
    
    return plt.subplots(nrows, ncols, **fig_config)

def plot_temperature_timeseries(data, ax=None, site_name="", mask_data=True, 
                               deployment_periods=None, **plot_kwargs):
    """
    Create a standardized temperature time series plot.
    
    Parameters:
    -----------
    data : xarray.Dataset or pandas.DataFrame
        Temperature data to plot
    ax : matplotlib.axes.Axes, optional
        Axis to plot on, creates new if None
    site_name : str
        Site name for title
    mask_data : bool
        Whether to apply deployment period masking
    deployment_periods : DataFrame, optional
        Deployment periods for masking
    **plot_kwargs : dict
        Additional plotting arguments
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = create_standard_figure()
    
    # Handle different data types
    if isinstance(data, xr.Dataset):
        if 'temp_c' in data.variables:
            temp_data = data['temp_c']
            time_data = data.coords['datetime']
        else:
            raise ValueError("Expected 'temp_c' variable in xarray Dataset")
    elif isinstance(data, pd.DataFrame):
        if 'temp_c' in data.columns:
            temp_data = data['temp_c']
            time_data = data.index
        else:
            raise ValueError("Expected 'temp_c' column in DataFrame")
    else:
        raise ValueError("Data must be xarray.Dataset or pandas.DataFrame")
    
    # Plot the data
    default_kwargs = {'alpha': 0.8, 'linewidth': 1}
    default_kwargs.update(plot_kwargs)
    
    ax.plot(time_data, temp_data, **default_kwargs)
    
    # Apply styling
    apply_plot_styling(ax, style_type='temperature')
    apply_plot_styling(ax, style_type='datetime')
    
    # Set title
    if site_name:
        ax.set_title(f'{site_name} Temperature Data', fontweight='bold')
    
    return ax

def plot_site_comparison(site_data_dict, figsize=None, titles=None):
    """
    Create a multi-panel comparison plot for multiple sites.
    
    Parameters:
    -----------
    site_data_dict : dict
        Dictionary mapping site names to data
    figsize : tuple, optional
        Figure size
    titles : list, optional
        Custom titles for each subplot
    
    Returns:
    --------
    fig, axes : matplotlib figure and axes
    """
    n_sites = len(site_data_dict)
    nrows = int(np.ceil(n_sites / 2)) if n_sites > 1 else 1
    ncols = 2 if n_sites > 1 else 1
    
    fig, axes = create_standard_figure(nrows, ncols, figsize=figsize)
    
    if n_sites == 1:
        axes = [axes]
    elif nrows == 1:
        axes = list(axes)
    else:
        axes = axes.flatten()
    
    # Create color palette
    colors = STYLE_CONFIG['colors']['default']
    
    for i, (site_name, data) in enumerate(site_data_dict.items()):
        ax = axes[i]
        color = colors[i % len(colors)]
        
        title = titles[i] if titles and i < len(titles) else site_name
        plot_temperature_timeseries(data, ax=ax, site_name=title, color=color)
    
    # Hide extra subplots
    for i in range(n_sites, len(axes)):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    return fig, axes

def create_lapse_rate_plot(elevation_data, temp_data, ax=None, site_name="", **kwargs):
    """
    Create a standardized lapse rate (temperature vs elevation) plot.
    
    Parameters:
    -----------
    elevation_data : array-like
        Elevation values
    temp_data : array-like
        Temperature values
    ax : matplotlib.axes.Axes, optional
        Axis to plot on
    site_name : str
        Site name for title
    **kwargs : dict
        Additional plotting arguments
    
    Returns:
    --------
    ax : matplotlib.axes.Axes
    """
    if ax is None:
        fig, ax = create_standard_figure()
    
    # Create scatter plot
    default_kwargs = {'alpha': 0.6, 's': 20}
    default_kwargs.update(kwargs)
    
    ax.scatter(elevation_data, temp_data, **default_kwargs)
    
    # Apply styling
    apply_plot_styling(ax, style_type='temperature')
    
    # Set labels
    ax.set_xlabel('Elevation (m)')
    ax.set_title(f'{site_name} Temperature vs Elevation', fontweight='bold')
    
    return ax

def format_axes_list(axes_list, style_types=None, **kwargs):
    """
    Apply consistent formatting to a list of axes.
    
    Parameters:
    -----------
    axes_list : list
        List of matplotlib.axes.Axes objects
    style_types : list, optional
        List of style types for each axis
    **kwargs : dict
        Styling parameters to apply to all axes
    """
    if style_types is None:
        style_types = ['default'] * len(axes_list)
    
    for i, ax in enumerate(axes_list):
        style_type = style_types[i] if i < len(style_types) else 'default'
        apply_plot_styling(ax, style_type=style_type, **kwargs)

def save_publication_figure(fig, filename, output_dir=None, dpi=300, formats=['png', 'svg']):
    """
    Save figure in publication-ready formats.
    
    Parameters:
    -----------
    fig : matplotlib.figure.Figure
        Figure to save
    filename : str
        Base filename (without extension)
    output_dir : str or Path, optional
        Output directory
    dpi : int
        Resolution for raster formats
    formats : list
        File formats to save
    """
    from pathlib import Path
    
    if output_dir is None:
        output_dir = Path.cwd()
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for fmt in formats:
        filepath = output_dir / f"{filename}.{fmt}"
        fig.savefig(filepath, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"Saved: {filepath}")

# Convenience function for common plot configurations
def setup_temperature_datetime_axes(axes_list):
    """Apply temperature + datetime styling to a list of axes."""
    for ax in axes_list:
        apply_plot_styling(ax, style_type='temperature')
        apply_plot_styling(ax, style_type='datetime')

def setup_basic_axes(axes_list):
    """Apply basic styling to a list of axes."""
    for ax in axes_list:
        apply_plot_styling(ax, style_type='default')

# Color palette functions
def get_color_palette(palette_name='default', n_colors=None):
    """Get a color palette for plotting."""
    palette = STYLE_CONFIG['colors'].get(palette_name, STYLE_CONFIG['colors']['default'])
    
    if n_colors is None:
        return palette
    
    # Extend or truncate palette as needed
    if n_colors <= len(palette):
        return palette[:n_colors]
    else:
        # Repeat palette to get required number of colors
        repeats = int(np.ceil(n_colors / len(palette)))
        extended_palette = (palette * repeats)[:n_colors]
        return extended_palette