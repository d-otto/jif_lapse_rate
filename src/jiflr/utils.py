# -*- coding: utf-8 -*-
"""
utils.py

Utility functions for the JIFLR project.

Author: drotto
Created: 2025-07-27
Project: jif_lapse_rate
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from scipy.ndimage import binary_opening, binary_closing
import warnings
from typing import Union, Optional, Tuple, List, Dict
from pathlib import Path
import xarray as xr


def guess_deployment_period(
    timeseries: Union[pd.Series, np.ndarray],
    datetime_index: Union[pd.DatetimeIndex, np.ndarray],
    method: str = 'variance',
    window_hours: float = 1.0,
    chunk_hours: float = 6.0,
    autocorr_max_lag_hours: float = 2.0,
    min_deployment_hours: float = 3.0,
    gap_fill_hours: float = 6.0,
    threshold: Optional[float] = None,
    plot: bool = False,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> np.ndarray:
    """
    Guess sensor deployment periods based on time series smoothness.
    
    Smoother time series are assumed to be from non-deployed periods (indoor storage),
    while more variable time series indicate field deployment.
    
    Parameters
    ----------
    timeseries : pd.Series or np.ndarray
        Temperature time series data
    datetime_index : pd.DatetimeIndex or np.ndarray
        Datetime index corresponding to timeseries
    method : str, default 'variance'
        Detection method: 'variance', 'wosa', 'autocorr', or 'gradient'
    window_hours : float, default 1.0
        Rolling window size in hours for variance/gradient methods
    chunk_hours : float, default 6.0
        Chunk length in hours for WOSA method
    autocorr_max_lag_hours : float, default 2.0
        Maximum lag in hours for autocorrelation analysis
    min_deployment_hours : float, default 3.0
        Minimum deployment period length in hours
    gap_fill_hours : float, default 6.0
        Maximum gap size to fill between adjacent deployed periods in hours
    threshold : float, optional
        Absolute threshold for deployment detection in method units:
        - variance: temperature standard deviation (°C)
        - wosa: spectral power density
        - autocorr: inverted autocorrelation (0-1)
        - gradient: gradient standard deviation (°C/timestep)
        If None, uses method-specific defaults
    plot : bool, default False
        Whether to plot the results
    ax : matplotlib.Axes, optional
        Axes to plot on. If None and plot=True, creates new figure
    **kwargs
        Additional plotting arguments
        
    Returns
    -------
    np.ndarray
        Boolean array where True indicates likely deployment periods
        
    Examples
    --------
    >>> import pandas as pd
    >>> import numpy as np
    >>> from jiflr.utils import guess_deployment_period
    >>> 
    >>> # Create synthetic temperature data
    >>> dates = pd.date_range('2025-01-01', periods=1000, freq='10min')
    >>> # Smooth indoor period followed by variable outdoor period
    >>> temp = np.concatenate([
    ...     20 + 0.1 * np.random.randn(500),  # Indoor: low variability
    ...     15 + 5 * np.sin(np.arange(500) * 2*np.pi/144) + np.random.randn(500)  # Field: high variability
    ... ])
    >>> 
    >>> deployed = guess_deployment_period(temp, dates, method='variance', plot=True)
    """
    
    # Convert inputs to pandas for easier handling
    if isinstance(timeseries, np.ndarray):
        timeseries = pd.Series(timeseries, index=datetime_index)
    elif not isinstance(timeseries.index, pd.DatetimeIndex):
        timeseries.index = pd.to_datetime(datetime_index)
    
    # Remove NaN values
    timeseries_clean = timeseries.dropna()
    if len(timeseries_clean) == 0:
        warnings.warn("All timeseries values are NaN")
        return np.zeros(len(timeseries), dtype=bool)
    
    if len(timeseries_clean) < 10:
        warnings.warn("Time series too short for reliable deployment detection")
        return np.ones(len(timeseries), dtype=bool)  # Assume all deployed
    
    # Calculate detection metric based on method
    try:
        if method == 'variance':
            metric = _calculate_variance_metric(timeseries_clean, window_hours)
        elif method == 'wosa':
            metric = _calculate_wosa_metric(timeseries_clean, chunk_hours)
        elif method == 'autocorr':
            metric = _calculate_autocorr_metric(timeseries_clean, window_hours, autocorr_max_lag_hours)
        elif method == 'gradient':
            metric = _calculate_gradient_metric(timeseries_clean, window_hours)
        else:
            raise ValueError(f"Unknown method: {method}. Choose from 'variance', 'wosa', 'autocorr', 'gradient'")
    except Exception as e:
        warnings.warn(f"Error calculating {method} metric: {e}. Returning all-deployed mask.")
        return np.ones(len(timeseries), dtype=bool)
    
    # Apply threshold to create boolean mask
    if threshold is None:
        # Use method-specific default thresholds
        if method == 'variance':
            threshold_value = 0.5  # 0.5°C standard deviation
        elif method == 'wosa':
            threshold_value = 0.3  # 0.3°C RMS of high-frequency variability
        elif method == 'autocorr':
            threshold_value = 0.3  # 0.3 inverted autocorrelation
        elif method == 'gradient':
            threshold_value = 0.1  # 0.1°C/timestep gradient std
    else:
        threshold_value = threshold
    
    deployed_mask = metric > threshold_value
    
    # Apply morphological cleaning
    deployed_clean = _clean_deployment_mask(
        deployed_mask, 
        timeseries_clean.index, 
        min_deployment_hours,
        gap_fill_hours
    )
    
    # Map back to original index (handle NaN values)
    result = np.zeros(len(timeseries), dtype=bool)
    result[timeseries.notna()] = deployed_clean
    
    # Plot if requested
    if plot:
        _plot_deployment_detection(
            timeseries, 
            result, 
            metric=metric,
            method=method,
            threshold_value=threshold_value,
            ax=ax,
            **kwargs
        )
    
    return result


def _calculate_variance_metric(ts: pd.Series, window_hours: float) -> pd.Series:
    """Calculate rolling standard deviation metric."""
    window_size = pd.Timedelta(hours=window_hours)
    return ts.rolling(window=window_size, center=True).std()


def _calculate_wosa_metric(ts: pd.Series, chunk_hours: float) -> pd.Series:
    """Calculate WOSA (Welch's method) spectral power metric."""
    # Determine sampling frequency
    dt_series = ts.index.to_series().diff()
    dt_median = dt_series.median()
    
    # Handle case where datetime diff returns NaN or invalid values
    if pd.isna(dt_median):
        warnings.warn("Invalid datetime index detected, using default 10-minute sampling")
        dt = 600.0  # 10 minutes default
    else:
        try:
            dt = dt_median.total_seconds()
            if dt <= 0:
                warnings.warn("Invalid time step detected, using default 10-minute sampling")
                dt = 600.0
        except (AttributeError, TypeError):
            # dt_median might already be in seconds (float/int)
            try:
                dt = float(dt_median)
                if dt <= 0:
                    warnings.warn("Invalid time step detected, using default 10-minute sampling")
                    dt = 600.0
            except (ValueError, TypeError):
                warnings.warn("Could not determine time step, using default 10-minute sampling")
                dt = 600.0
    
    fs = 1.0 / dt
    
    # Calculate chunk size in samples
    chunk_samples = int(chunk_hours * 3600 / dt)
    chunk_samples = min(chunk_samples, len(ts) // 4)  # Ensure at least 4 chunks
    
    if chunk_samples < 10:
        warnings.warn("Time series too short for WOSA analysis")
        return pd.Series(np.ones(len(ts)), index=ts.index)
    
    # Calculate overlap
    overlap_samples = chunk_samples // 2
    
    # Calculate step size for sliding window
    step_samples = chunk_samples - overlap_samples
    
    # Initialize result array
    metric_values = np.full(len(ts), np.nan)
    
    # Slide window through time series
    for i in range(0, len(ts) - chunk_samples + 1, step_samples):
        chunk = ts.iloc[i:i + chunk_samples].values
        
        # Remove trend
        chunk_detrended = signal.detrend(chunk)
        
        # Calculate power spectral density
        freqs, psd = signal.welch(
            chunk_detrended, 
            fs=fs, 
            nperseg=len(chunk_detrended)//4,
            noverlap=len(chunk_detrended)//8
        )
        
        # Focus on high-frequency content (variability)
        # Use frequencies above 1/hour (exclude very low frequencies)
        freq_threshold = 1.0 / 3600.0  # 1/hour in Hz
        high_freq_mask = freqs > freq_threshold
        
        if np.any(high_freq_mask):
            # Use RMS of high-frequency power as metric
            # This gives units of °C (temperature variability)
            high_freq_power = np.sum(psd[high_freq_mask])
            metric_value = np.sqrt(high_freq_power)  # RMS gives °C units
        else:
            # Fall back to total power if no high frequencies
            total_power = np.sum(psd[1:])  # Skip DC
            metric_value = np.sqrt(total_power)
        
        # Assign to center of window
        center_idx = i + chunk_samples // 2
        if center_idx < len(metric_values):
            metric_values[center_idx] = metric_value
    
    # Forward fill and backward fill to handle edges
    metric_series = pd.Series(metric_values, index=ts.index)
    metric_series = metric_series.ffill().bfill()
    
    return metric_series


def _calculate_autocorr_metric(ts: pd.Series, window_hours: float, max_lag_hours: float) -> pd.Series:
    """Calculate autocorrelation-based metric."""
    window_size = pd.Timedelta(hours=window_hours)
    
    # Determine sampling frequency and max lag in samples
    dt_series = ts.index.to_series().diff()
    dt_median = dt_series.median()
    
    # Handle case where datetime diff returns NaN or invalid values
    if pd.isna(dt_median):
        warnings.warn("Invalid datetime index detected, using default 10-minute sampling")
        dt = 600.0  # 10 minutes default
    else:
        try:
            dt = dt_median.total_seconds()
            if dt <= 0:
                warnings.warn("Invalid time step detected, using default 10-minute sampling")
                dt = 600.0
        except (AttributeError, TypeError):
            # dt_median might already be in seconds (float/int)
            try:
                dt = float(dt_median)
                if dt <= 0:
                    warnings.warn("Invalid time step detected, using default 10-minute sampling")
                    dt = 600.0
            except (ValueError, TypeError):
                warnings.warn("Could not determine time step, using default 10-minute sampling")
                dt = 600.0
    
    max_lag_samples = int(max_lag_hours * 3600 / dt)
    
    def autocorr_metric(x):
        if len(x) < max_lag_samples * 2:
            return np.nan
        
        # Remove mean
        x_centered = x - np.mean(x)
        
        # Calculate autocorrelation using numpy correlate
        autocorr = np.correlate(x_centered, x_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]  # Take positive lags only
        
        # Normalize by zero-lag value
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        else:
            return np.nan
        
        # Use mean autocorrelation at short lags as metric
        # Lower autocorrelation = more random = likely deployed
        if len(autocorr) > max_lag_samples:
            short_lag_autocorr = np.mean(autocorr[1:max_lag_samples+1])
            return 1.0 - short_lag_autocorr  # Invert so higher = more deployed
        else:
            return np.nan
    
    return ts.rolling(window=window_size, center=True).apply(autocorr_metric, raw=True)


def _calculate_gradient_metric(ts: pd.Series, window_hours: float) -> pd.Series:
    """Calculate temperature gradient variability metric."""
    # Calculate temperature gradients
    gradients = ts.diff().abs()
    
    # Calculate rolling standard deviation of gradients
    window_size = pd.Timedelta(hours=window_hours)
    return gradients.rolling(window=window_size, center=True).std()


def _clean_deployment_mask(
    mask: pd.Series, 
    datetime_index: pd.DatetimeIndex, 
    min_hours: float,
    gap_fill_hours: float
) -> np.ndarray:
    """Clean deployment mask using morphological operations and gap filling."""
    # Convert to numpy array
    mask_array = mask.values.astype(bool)
    
    # Calculate sample sizes for different time periods
    dt_series = datetime_index.to_series().diff()
    dt_median = dt_series.median()
    
    # Handle datetime issues
    if pd.isna(dt_median):
        dt = 600.0  # 10 minutes default
    else:
        try:
            dt = dt_median.total_seconds()
            if dt <= 0:
                dt = 600.0
        except (AttributeError, TypeError):
            try:
                dt = float(dt_median)
                if dt <= 0:
                    dt = 600.0
            except (ValueError, TypeError):
                dt = 600.0
    
    min_samples = max(1, int(min_hours * 3600 / dt))
    gap_fill_samples = max(1, int(gap_fill_hours * 3600 / dt))
    
    # Step 1: Fill gaps between deployed periods (gap filling)
    if gap_fill_hours > 0:
        gap_fill_element = np.ones(gap_fill_samples)
        mask_gap_filled = binary_closing(mask_array, structure=gap_fill_element)
    else:
        mask_gap_filled = mask_array.copy()
    
    # Step 2: Remove small isolated deployed regions (minimum deployment period)
    if min_hours > 0:
        min_period_element = np.ones(min_samples)
        mask_cleaned = binary_opening(mask_gap_filled, structure=min_period_element)
    else:
        mask_cleaned = mask_gap_filled
    
    return mask_cleaned


def _plot_deployment_detection(
    timeseries: pd.Series,
    deployed_mask: np.ndarray,
    metric: Optional[pd.Series] = None,
    method: str = 'variance',
    threshold_value: float = 0.0,
    ax: Optional[plt.Axes] = None,
    **kwargs
) -> None:
    """Plot time series with deployment period detection."""
    if ax is None:
        fig, axes = plt.subplots(2 if metric is not None else 1, 1, 
                                figsize=(12, 8 if metric is not None else 6), 
                                sharex=True)
        if metric is not None:
            ax_ts, ax_metric = axes
        else:
            ax_ts = axes
    else:
        ax_ts = ax
        ax_metric = None
    
    # Plot time series
    ax_ts.plot(timeseries.index, timeseries.values, 'b-', alpha=0.7, linewidth=0.8, label='Temperature')
    
    # Highlight non-deployed periods
    non_deployed = ~deployed_mask
    if np.any(non_deployed):
        # Find continuous non-deployed periods
        non_deployed_diff = np.diff(np.concatenate(([False], non_deployed, [False])).astype(int))
        starts = np.where(non_deployed_diff == 1)[0]
        ends = np.where(non_deployed_diff == -1)[0]
        
        for start, end in zip(starts, ends):
            ax_ts.axvspan(
                timeseries.index[start], 
                timeseries.index[end-1] if end < len(timeseries.index) else timeseries.index[-1],
                alpha=0.3, 
                color='red', 
                label='Likely not deployed' if start == starts[0] else ''
            )
    
    ax_ts.set_ylabel('Temperature (°C)')
    ax_ts.set_title(f'Deployment Period Detection ({method.upper()} method)')
    ax_ts.legend()
    ax_ts.grid(True, alpha=0.3)
    
    # Plot metric if provided
    if metric is not None and ax_metric is not None:
        ax_metric.plot(metric.index, metric.values, 'g-', alpha=0.8, linewidth=1)
        ax_metric.set_ylabel(f'{method.title()} Metric')
        ax_metric.set_xlabel('Time')
        ax_metric.grid(True, alpha=0.3)
        
        # Add threshold line
        ax_metric.axhline(threshold_value, color='r', linestyle='--', alpha=0.7, label=f'Threshold ({threshold_value:.3f})')
        ax_metric.legend()
    else:
        ax_ts.set_xlabel('Time')
    
    plt.tight_layout()


def butterworth_filter(
    data: np.ndarray,
    fs: float,
    order: int = 4,
    lower: Optional[float] = None,
    upper: Optional[float] = None
) -> np.ndarray:
    """
    Apply Butterworth filter to data using zero-phase filtering.

    This function provides low-pass, high-pass, or band-pass filtering using
    a Butterworth filter design. It uses second-order sections (SOS) for
    numerical stability and sosfiltfilt for zero-phase distortion.
    
    NOTE: This does not account for the shift in cutoff frequency from filtfilt.

    Parameters
    ----------
    data : np.ndarray
        1D array of data to filter
    fs : float
        Sampling frequency in Hz (e.g., for 5-minute data: fs = 1/300)
    order : int, default 4
        Filter order (higher = steeper roll-off)
    lower : float, optional
        Lower cutoff frequency in Hz (None for low-pass filter)
        For band-pass, this is the lower cutoff
    upper : float, optional
        Upper cutoff frequency in Hz (None for high-pass filter)
        For band-pass, this is the upper cutoff

    Returns
    -------
    np.ndarray
        Filtered data with same shape as input

    Notes
    -----
    - Cutoff frequencies are specified in Hz (cycles per second)
    - For time-series with irregular sampling, resample to regular intervals first
    - Uses sosfiltfilt for zero-phase filtering (no time shift)
    - For 5-minute data (300 seconds), fs = 1/300 Hz
    - Example: upper=1/6 with 5-min data means 6 samples = 30 minutes cutoff

    Examples
    --------
    >>> # Low-pass filter: remove fluctuations faster than 30 minutes
    >>> # For 5-minute data: fs = 1/300 Hz, upper = 1/6 samples = 30 min period
    >>> filtered = butterworth_filter(data, fs=1/300, order=4, upper=1/6)

    >>> # High-pass filter: remove slow trends
    >>> filtered = butterworth_filter(data, fs=1/300, order=4, lower=1/100)

    >>> # Band-pass filter
    >>> filtered = butterworth_filter(data, fs=1/300, order=4, lower=1/100, upper=1/6)
    """
    # Validate inputs
    if lower is None and upper is None:
        raise ValueError("Must specify at least one of 'lower' or 'upper' cutoff frequency")

    # Handle NaN values
    data_copy = data.copy()
    nan_mask = np.isnan(data_copy)

    if np.all(nan_mask):
        warnings.warn("All data values are NaN, returning input unchanged")
        return data_copy

    # If there are some NaNs, interpolate them for filtering
    # (sosfiltfilt cannot handle NaN values)
    if np.any(nan_mask):
        valid_indices = np.where(~nan_mask)[0]
        data_interpolated = np.interp(
            np.arange(len(data_copy)),
            valid_indices,
            data_copy[valid_indices]
        )
    else:
        data_interpolated = data_copy

    # Determine filter type and design filter
    if lower is None:
        # Low-pass filter
        btype = 'low'
        Wn = upper
    elif upper is None:
        # High-pass filter
        btype = 'high'
        Wn = lower
    else:
        # Band-pass filter
        btype = 'band'
        Wn = [lower, upper]

    # Design Butterworth filter using second-order sections (SOS)
    # SOS is more numerically stable than transfer function (b, a)
    sos = signal.butter(order, Wn, btype=btype, fs=fs, output='sos')

    # Apply zero-phase filtering (forward and backward pass)
    filtered_data = signal.sosfiltfilt(sos, data_interpolated)

    # Restore NaN values at their original positions
    if np.any(nan_mask):
        filtered_data[nan_mask] = np.nan

    return filtered_data


def get_deployment_periods(
    site_id: Union[str, List[str]],
    csv_path: Union[str, Path]
) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Get deployment periods for one or more sites from CSV file.
    
    Parameters
    ----------
    site_id : str or list of str
        Site identifier(s) to look up deployment periods for
    csv_path : str or Path
        Path to CSV file containing deployment periods
        
    Returns
    -------
    dict
        Dict {site: [(start, end), ...]} for all requested sites
        
    Examples
    --------
    >>> # Single site
    >>> periods = get_deployment_periods('A01', 'deployment_periods.csv')
    >>> # {'A01': [(Timestamp('2025-01-01'), Timestamp('2025-07-19')), ...]}
    
    >>> # Multiple sites
    >>> periods = get_deployment_periods(['A01', 'B02'], 'deployment_periods.csv')
    >>> # {'A01': [(start, end), ...], 'B02': [(start, end), ...]}
    """
    # Load deployment periods CSV
    df_periods = pd.read_csv(csv_path)
    
    # Convert to list if single site
    site_list = [site_id] if isinstance(site_id, str) else site_id
    
    # Result container
    result = {}
    
    for site in site_list:
        # Filter by site
        site_mask = df_periods['site'] == site
        site_periods = df_periods[site_mask]
        
        # Parse deployment periods for this site
        periods = []
        for _, row in site_periods.iterrows():
            deploy_date = row.get('deploy_date', '')
            deploy_time = row.get('deploy_time', '00:00:00')
            pickup_date = row.get('pickup_date', '')
            pickup_time = row.get('pickup_time', '23:59:59')
            
            # Handle missing dates by skipping this period
            if pd.isna(deploy_date) or deploy_date == '' or pd.isna(pickup_date) or pickup_date == '':
                continue
                
            # Handle missing times
            if pd.isna(deploy_time) or deploy_time == '':
                deploy_time = '00:00:00'
            if pd.isna(pickup_time) or pickup_time == '':
                pickup_time = '23:59:59'
            
            # Combine date and time strings
            deploy_datetime_str = f"{deploy_date} {deploy_time}"
            pickup_datetime_str = f"{pickup_date} {pickup_time}"
            
            # Parse to timestamps
            deploy_datetime = pd.to_datetime(deploy_datetime_str)
            pickup_datetime = pd.to_datetime(pickup_datetime_str)
            
            periods.append((deploy_datetime, pickup_datetime))
        
        result[site] = periods
    
    return result


def filter_by_deployment_periods(
    data: Union[xr.DataArray, xr.Dataset],
    deployment_periods: List[Tuple[pd.Timestamp, pd.Timestamp]],
    return_mask: bool = False,
    fill_value: Optional[float] = np.nan
) -> Union[xr.DataArray, xr.Dataset, np.ndarray]:
    """
    Filter data to deployment periods or return mask.
    
    Parameters
    ----------
    data : xr.DataArray or xr.Dataset
        Data array/dataset with datetime coordinate (typically temperature data)
    deployment_periods : list of tuple
        List of (start_datetime, end_datetime) tuples defining deployment periods
    return_mask : bool, default False
        If True, return boolean mask array
        If False, return filtered DataArray/Dataset with non-deployment periods masked
    fill_value : float, optional
        Value to use for masked (non-deployment) periods when return_mask=False
        Default is np.nan
        
    Returns
    -------
    xr.DataArray, xr.Dataset, or np.ndarray
        If return_mask=True: boolean numpy array where True indicates deployment period
        If return_mask=False: DataArray/Dataset with non-deployment periods set to fill_value
        
    Examples
    --------
    >>> import xarray as xr
    >>> import pandas as pd
    >>> 
    >>> # Create sample data
    >>> dates = pd.date_range('2025-01-01', '2025-12-31', freq='1H')
    >>> temp_data = xr.DataArray(np.random.randn(len(dates)), coords={'datetime': dates}, dims=['datetime'])
    >>> 
    >>> # Get deployment periods
    >>> periods = get_deployment_periods('A01', 'deployment_periods.csv')
    >>> 
    >>> # Get mask
    >>> mask = filter_by_deployment_periods(temp_data, periods, return_mask=True)
    >>> 
    >>> # Get filtered data
    >>> filtered_data = filter_by_deployment_periods(temp_data, periods, return_mask=False)
    """
    # Extract datetime coordinate
    datetime_coord = data.coords['datetime']
    datetime_values = pd.to_datetime(datetime_coord.values)
    
    # Initialize mask as False (not deployed)
    deployment_mask = np.zeros(len(datetime_values), dtype=bool)
    
    # Apply each deployment period (union of all periods)
    for start_time, end_time in deployment_periods:
        period_mask = (datetime_values >= start_time) & (datetime_values <= end_time)
        deployment_mask |= period_mask
    
    if return_mask:
        return deployment_mask
    else:
        # Create filtered data
        filtered_data = data.copy()
        
        if isinstance(data, xr.DataArray):
            filtered_data.values[~deployment_mask] = fill_value
        elif isinstance(data, xr.Dataset):
            # Apply mask to all data variables that have datetime dimension
            for var_name in data.data_vars:
                if 'datetime' in data[var_name].dims:
                    filtered_data[var_name] = filtered_data[var_name].where(deployment_mask, fill_value)
        
        return filtered_data


