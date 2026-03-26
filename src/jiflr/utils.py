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
import datetime
from scipy import stats

from astral import LocationInfo
from astral.sun import sun


def guess_deployment_period(
    timeseries: Union[pd.Series, np.ndarray],
    datetime_index: Union[pd.DatetimeIndex, np.ndarray],
    method: str = "variance",
    window_hours: float = 1.0,
    chunk_hours: float = 6.0,
    autocorr_max_lag_hours: float = 2.0,
    min_deployment_hours: float = 3.0,
    gap_fill_hours: float = 6.0,
    threshold: Optional[float] = None,
    plot: bool = False,
    ax: Optional[plt.Axes] = None,
    **kwargs,
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
        if method == "variance":
            metric = _calculate_variance_metric(timeseries_clean, window_hours)
        elif method == "wosa":
            metric = _calculate_wosa_metric(timeseries_clean, chunk_hours)
        elif method == "autocorr":
            metric = _calculate_autocorr_metric(
                timeseries_clean, window_hours, autocorr_max_lag_hours
            )
        elif method == "gradient":
            metric = _calculate_gradient_metric(timeseries_clean, window_hours)
        else:
            raise ValueError(
                f"Unknown method: {method}. Choose from 'variance', 'wosa', 'autocorr', 'gradient'"
            )
    except Exception as e:
        warnings.warn(
            f"Error calculating {method} metric: {e}. Returning all-deployed mask."
        )
        return np.ones(len(timeseries), dtype=bool)

    # Apply threshold to create boolean mask
    if threshold is None:
        # Use method-specific default thresholds
        if method == "variance":
            threshold_value = 0.5  # 0.5°C standard deviation
        elif method == "wosa":
            threshold_value = 0.3  # 0.3°C RMS of high-frequency variability
        elif method == "autocorr":
            threshold_value = 0.3  # 0.3 inverted autocorrelation
        elif method == "gradient":
            threshold_value = 0.1  # 0.1°C/timestep gradient std
    else:
        threshold_value = threshold

    deployed_mask = metric > threshold_value

    # Apply morphological cleaning
    deployed_clean = _clean_deployment_mask(
        deployed_mask, timeseries_clean.index, min_deployment_hours, gap_fill_hours
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
            **kwargs,
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
        warnings.warn(
            "Invalid datetime index detected, using default 10-minute sampling"
        )
        dt = 600.0  # 10 minutes default
    else:
        try:
            dt = dt_median.total_seconds()
            if dt <= 0:
                warnings.warn(
                    "Invalid time step detected, using default 10-minute sampling"
                )
                dt = 600.0
        except (AttributeError, TypeError):
            # dt_median might already be in seconds (float/int)
            try:
                dt = float(dt_median)
                if dt <= 0:
                    warnings.warn(
                        "Invalid time step detected, using default 10-minute sampling"
                    )
                    dt = 600.0
            except (ValueError, TypeError):
                warnings.warn(
                    "Could not determine time step, using default 10-minute sampling"
                )
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
        chunk = ts.iloc[i : i + chunk_samples].values

        # Remove trend
        chunk_detrended = signal.detrend(chunk)

        # Calculate power spectral density
        freqs, psd = signal.welch(
            chunk_detrended,
            fs=fs,
            nperseg=len(chunk_detrended) // 4,
            noverlap=len(chunk_detrended) // 8,
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


def _calculate_autocorr_metric(
    ts: pd.Series, window_hours: float, max_lag_hours: float
) -> pd.Series:
    """Calculate autocorrelation-based metric."""
    window_size = pd.Timedelta(hours=window_hours)

    # Determine sampling frequency and max lag in samples
    dt_series = ts.index.to_series().diff()
    dt_median = dt_series.median()

    # Handle case where datetime diff returns NaN or invalid values
    if pd.isna(dt_median):
        warnings.warn(
            "Invalid datetime index detected, using default 10-minute sampling"
        )
        dt = 600.0  # 10 minutes default
    else:
        try:
            dt = dt_median.total_seconds()
            if dt <= 0:
                warnings.warn(
                    "Invalid time step detected, using default 10-minute sampling"
                )
                dt = 600.0
        except (AttributeError, TypeError):
            # dt_median might already be in seconds (float/int)
            try:
                dt = float(dt_median)
                if dt <= 0:
                    warnings.warn(
                        "Invalid time step detected, using default 10-minute sampling"
                    )
                    dt = 600.0
            except (ValueError, TypeError):
                warnings.warn(
                    "Could not determine time step, using default 10-minute sampling"
                )
                dt = 600.0

    max_lag_samples = int(max_lag_hours * 3600 / dt)

    def autocorr_metric(x):
        if len(x) < max_lag_samples * 2:
            return np.nan

        # Remove mean
        x_centered = x - np.mean(x)

        # Calculate autocorrelation using numpy correlate
        autocorr = np.correlate(x_centered, x_centered, mode="full")
        autocorr = autocorr[len(autocorr) // 2 :]  # Take positive lags only

        # Normalize by zero-lag value
        if autocorr[0] > 0:
            autocorr = autocorr / autocorr[0]
        else:
            return np.nan

        # Use mean autocorrelation at short lags as metric
        # Lower autocorrelation = more random = likely deployed
        if len(autocorr) > max_lag_samples:
            short_lag_autocorr = np.mean(autocorr[1 : max_lag_samples + 1])
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
    gap_fill_hours: float,
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
    method: str = "variance",
    threshold_value: float = 0.0,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> None:
    """Plot time series with deployment period detection."""
    if ax is None:
        fig, axes = plt.subplots(
            2 if metric is not None else 1,
            1,
            figsize=(12, 8 if metric is not None else 6),
            sharex=True,
        )
        if metric is not None:
            ax_ts, ax_metric = axes
        else:
            ax_ts = axes
    else:
        ax_ts = ax
        ax_metric = None

    # Plot time series
    ax_ts.plot(
        timeseries.index,
        timeseries.values,
        "b-",
        alpha=0.7,
        linewidth=0.8,
        label="Temperature",
    )

    # Highlight non-deployed periods
    non_deployed = ~deployed_mask
    if np.any(non_deployed):
        # Find continuous non-deployed periods
        non_deployed_diff = np.diff(
            np.concatenate(([False], non_deployed, [False])).astype(int)
        )
        starts = np.where(non_deployed_diff == 1)[0]
        ends = np.where(non_deployed_diff == -1)[0]

        for start, end in zip(starts, ends):
            ax_ts.axvspan(
                timeseries.index[start],
                timeseries.index[end - 1]
                if end < len(timeseries.index)
                else timeseries.index[-1],
                alpha=0.3,
                color="red",
                label="Likely not deployed" if start == starts[0] else "",
            )

    ax_ts.set_ylabel("Temperature (°C)")
    ax_ts.set_title(f"Deployment Period Detection ({method.upper()} method)")
    ax_ts.legend()
    ax_ts.grid(True, alpha=0.3)

    # Plot metric if provided
    if metric is not None and ax_metric is not None:
        ax_metric.plot(metric.index, metric.values, "g-", alpha=0.8, linewidth=1)
        ax_metric.set_ylabel(f"{method.title()} Metric")
        ax_metric.set_xlabel("Time")
        ax_metric.grid(True, alpha=0.3)

        # Add threshold line
        ax_metric.axhline(
            threshold_value,
            color="r",
            linestyle="--",
            alpha=0.7,
            label=f"Threshold ({threshold_value:.3f})",
        )
        ax_metric.legend()
    else:
        ax_ts.set_xlabel("Time")

    plt.tight_layout()


def butterworth_filter(
    data: np.ndarray,
    fs: float,
    order: int = 4,
    lower: Optional[float] = None,
    upper: Optional[float] = None,
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
        raise ValueError(
            "Must specify at least one of 'lower' or 'upper' cutoff frequency"
        )

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
            np.arange(len(data_copy)), valid_indices, data_copy[valid_indices]
        )
    else:
        data_interpolated = data_copy

    # Determine filter type and design filter
    if lower is None:
        # Low-pass filter
        btype = "low"
        Wn = upper
    elif upper is None:
        # High-pass filter
        btype = "high"
        Wn = lower
    else:
        # Band-pass filter
        btype = "band"
        Wn = [lower, upper]

    # Design Butterworth filter using second-order sections (SOS)
    # SOS is more numerically stable than transfer function (b, a)
    sos = signal.butter(order, Wn, btype=btype, fs=fs, output="sos")

    # Apply zero-phase filtering (forward and backward pass)
    filtered_data = signal.sosfiltfilt(sos, data_interpolated)

    # Restore NaN values at their original positions
    if np.any(nan_mask):
        filtered_data[nan_mask] = np.nan

    return filtered_data


def get_deployment_periods(
    site_id: Union[str, List[str]],
    csv_path: Union[str, Path],
    default_start: Optional[pd.Timestamp] = None,
    logger: Optional["logging.Logger"] = None,
) -> Dict[str, List[Tuple[pd.Timestamp, pd.Timestamp]]]:
    """
    Get deployment periods for one or more sites from CSV file.

    Parameters
    ----------
    site_id : str or list of str
        Site identifier(s) to look up deployment periods for
    csv_path : str or Path
        Path to CSV file containing deployment periods
    default_start : pd.Timestamp, optional
        Default start time to use when deploy_date is missing in CSV.
        If None, rows with missing deploy_date are skipped.
    logger : logging.Logger, optional
        Logger for warning messages when default_start is used.

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
        # Filter by site (case-insensitive matching)
        site_mask = df_periods["site"].str.lower() == site.lower()
        site_periods = df_periods[site_mask]

        # Parse deployment periods for this site
        periods = []
        for _, row in site_periods.iterrows():
            deploy_date = row.get("deploy_date", "")
            deploy_time = row.get("deploy_time", "00:00:00")
            pickup_date = row.get("pickup_date", "")
            pickup_time = row.get("pickup_time", "23:59:59")

            # Handle missing pickup_date - skip row (can't determine end)
            if pd.isna(pickup_date) or pickup_date == "":
                continue

            # Handle missing deploy_date - use default_start if provided
            if pd.isna(deploy_date) or deploy_date == "":
                if default_start is not None:
                    deploy_datetime = default_start
                    if logger:
                        logger.warning(
                            f"Site {site}: deploy_date missing, using data start time "
                            f"({default_start.strftime('%Y-%m-%d %H:%M')}) as deployment start"
                        )
                else:
                    continue  # Skip if no default_start provided (original behavior)
            else:
                # Handle missing deploy time
                if pd.isna(deploy_time) or deploy_time == "":
                    deploy_time = "00:00:00"
                # Combine date and time strings and parse
                deploy_datetime_str = f"{deploy_date} {deploy_time}"
                deploy_datetime = pd.to_datetime(deploy_datetime_str)

            # Handle missing pickup time
            if pd.isna(pickup_time) or pickup_time == "":
                pickup_time = "23:59:59"

            # Parse pickup datetime
            pickup_datetime_str = f"{pickup_date} {pickup_time}"
            pickup_datetime = pd.to_datetime(pickup_datetime_str)

            periods.append((deploy_datetime, pickup_datetime))

        result[site] = periods

    return result


def get_wind_dir_offsets(csv_path: Union[str, Path]) -> Dict[str, float]:
    """Load wind direction offset corrections from deployment metadata CSV.

    Reads the wind_dir_offset_deg column from deployment_periods.csv and returns
    a dictionary mapping site names to offset degrees. Sites with NaN or missing
    offsets are not included in the returned dictionary.

    Parameters
    ----------
    csv_path : str or Path
        Path to deployment_periods.csv file

    Returns
    -------
    dict
        Dictionary mapping site names (lowercase) to offset degrees.
        Empty dict if CSV missing, column missing, or no valid offsets.

    Examples
    --------
    >>> offsets = get_wind_dir_offsets('data/2025/metadata/deployment_periods.csv')
    >>> offsets
    {'lee1': 15.0, 'windward1': -20.0}
    """
    from typing import Dict

    csv_path = Path(csv_path)

    # Return empty dict if file doesn't exist
    if not csv_path.exists():
        return {}

    try:
        # Read CSV
        df = pd.read_csv(csv_path)

        # Check if wind_dir_offset_deg column exists
        if "wind_dir_offset_deg" not in df.columns:
            return {}

        # Extract non-NaN offsets
        offsets = {}
        for _, row in df.iterrows():
            site = row.get("site", None)
            offset = row.get("wind_dir_offset_deg", None)

            # Skip if site or offset is missing/NaN
            if pd.isna(site) or pd.isna(offset):
                continue

            # Store with lowercase site name for case-insensitive matching
            offsets[str(site).lower()] = float(offset)

        return offsets

    except Exception:
        # Return empty dict on any error (file read, parsing, etc.)
        return {}


def deployment_mask(
    ds: xr.Dataset,
    site_id: str,
    csv_path: Path,
    ignore_missing: bool = False,
) -> np.ndarray:
    """
    Return a boolean array aligned to ds's datetime coordinate.

    True where the timestamp falls within a deployment period for site_id.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a 'datetime' or 'datetime_utc' coordinate.
    site_id : str
        Site identifier to look up in the deployment periods CSV.
    csv_path : Path
        Path to deployment_periods.csv.
    ignore_missing : bool, default False
        If True, return all-True mask when no periods are found.
        If False, raise ValueError when no periods are found.

    Returns
    -------
    np.ndarray
        Boolean array of length matching ds's datetime coordinate.

    Raises
    ------
    ValueError
        If no deployment periods are found and ignore_missing is False.
    """
    datetime_coord_name = "datetime" if "datetime" in ds.coords else "datetime_utc"
    datetime_values = pd.to_datetime(ds.coords[datetime_coord_name].values)

    periods_dict = get_deployment_periods(site_id, csv_path)
    periods = periods_dict.get(site_id, [])

    if not periods:
        if ignore_missing:
            return np.ones(len(datetime_values), dtype=bool)
        raise ValueError(
            f"No deployment periods found for site '{site_id}' in {csv_path}"
        )

    mask = np.zeros(len(datetime_values), dtype=bool)
    for start_time, end_time in periods:
        mask |= (datetime_values >= start_time) & (datetime_values <= end_time)
    return mask


def apply_deployment_mask(
    ds: xr.Dataset,
    site_id: str,
    csv_path: Path,
    ignore_missing: bool = False,
) -> xr.Dataset:
    """
    Return dataset with out-of-deployment values set to NaN.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset with a 'datetime' or 'datetime_utc' coordinate.
    site_id : str
        Site identifier to look up in the deployment periods CSV.
    csv_path : Path
        Path to deployment_periods.csv.
    ignore_missing : bool, default False
        If True, return ds unchanged when no periods are found.
        If False, raise ValueError when no periods are found.

    Returns
    -------
    xr.Dataset
        Dataset with non-deployment periods set to NaN for all data variables
        that have the datetime coordinate as a dimension.

    Raises
    ------
    ValueError
        If no deployment periods are found and ignore_missing is False.
    """
    datetime_coord_name = "datetime" if "datetime" in ds.coords else "datetime_utc"
    mask = deployment_mask(ds, site_id, csv_path, ignore_missing=ignore_missing)

    ds_out = ds.copy()
    for var_name in ds.data_vars:
        if datetime_coord_name in ds[var_name].dims:
            ds_out[var_name] = ds_out[var_name].where(mask)
    return ds_out


# Function to create a GeoDataFrame from GeoJSON features
def create_gdf_from_features(features, feature_class=None, folder_id=None):

    # Filter features by class and/or folder ID if specified
    if feature_class:
        features = [
            f for f in features if f["properties"].get("class") == feature_class
        ]
    if folder_id:
        features = [f for f in features if f["properties"].get("folderId") == folder_id]

    if not features:
        return None

    # Extract geometries and properties
    geometries = []
    properties = []

    for feature in features:
        if feature["geometry"] is not None:
            geom_type = feature["geometry"]["type"]
            coords = feature["geometry"]["coordinates"]

            if geom_type == "Point":
                geometry = Point(coords[0], coords[1])
            elif geom_type == "LineString":
                geometry = LineString(coords)
            elif geom_type == "Polygon":
                geometry = Polygon(coords[0])
            else:
                continue

            geometries.append(geometry)
            properties.append(feature["properties"])

    if not geometries:
        return None

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs="EPSG:4326")
    return gdf


def get_sunrise_sunset(
    dates: pd.DatetimeIndex, location: LocationInfo | None = None
) -> pd.DataFrame:
    """
    Generate a DataFrame of sunrise/sunset times for each date in `dates`.

    Returns a DataFrame indexed by date with columns 'sunrise' and 'sunset'.
    Timezone of output matches timezone of input `dates`; if `dates` is
    timezone-naive, output is UTC.

    If location is not provided, defaults to SE Alaska site (58.75°N, 134.25°W).
    """
    if location is None:
        location = LocationInfo(
            name="SE Alaska Site",
            region="Alaska",
            timezone="US/Alaska",
            latitude=58.75,
            longitude=-134.25,
        )

    input_tz = dates.tzinfo

    records = []
    for d in dates:
        s = sun(location.observer, date=d, tzinfo=location.timezone)
        sunrise = s["sunrise"].astimezone(datetime.timezone.utc)
        sunset = s["sunset"].astimezone(datetime.timezone.utc)
        if input_tz is not None:
            sunrise = sunrise.astimezone(input_tz)
            sunset = sunset.astimezone(input_tz)
        records.append(
            {
                "date": pd.Timestamp(d).date(),
                "sunrise": sunrise,
                "sunset": sunset,
            }
        )
    return pd.DataFrame(records).set_index("date")





def calculate_lapse_rate_timestep(
    temp_data: np.ndarray,
    elevation_data: np.ndarray,
    site_mask: np.ndarray,
) -> tuple[float, float]:
    """Calculate the temperature lapse rate at a single timestep via linear regression.

    Fits a linear model of temperature vs. elevation for the subset of sensors
    selected by ``site_mask``, excluding any sensors with NaN values. Requires
    at least two valid data points; returns ``(nan, nan)`` otherwise.

    Parameters
    ----------
    temp_data:
        Temperature values at this timestep, indexed by ``sensor_idx``. Shape: ``(n_sensors,)``.
    elevation_data:
        Elevation values for each sensor. Shape: ``(n_sensors,)``.
    site_mask:
        Boolean mask selecting which sensors to include in the regression. Shape: ``(n_sensors,)``.

    Returns
    -------
    tuple[float, float]
        ``(slope, intercept)`` — lapse rate in °C/m and intercept in °C.
    """
    masked_temp = temp_data[site_mask]
    masked_elev = elevation_data[site_mask]

    valid_idx = ~(np.isnan(masked_temp) | np.isnan(masked_elev))

    if valid_idx.sum() < 2:
        return np.nan, np.nan

    result = stats.linregress(masked_elev[valid_idx], masked_temp[valid_idx])
    return float(result.slope), float(result.intercept)


def lapse_rate_ufunc(
    temp_data: np.ndarray,
    elevation_data: np.ndarray,
    site_mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Wrapper for calculate_lapse_rate_timestep compatible with xr.apply_ufunc.

    Iterates over the time dimension of temperature data and computes the
    lapse rate slope and intercept for each timestep using a site mask to
    filter valid locations.

    Args:
        temp_data: Temperature array of shape (n_times, n_sites).
        elevation_data: 1D elevation array of shape (n_sites,).
        site_mask: Boolean array of shape (n_sites,) indicating valid sites.

    Returns:
        A tuple of (slopes, intercepts), each a 1D array of shape (n_times,)
        containing the lapse rate slope and intercept for each timestep.
        Values are NaN where the lapse rate could not be calculated.
    """
    n_times = temp_data.shape[0]
    slopes = np.full(n_times, np.nan)
    intercepts = np.full(n_times, np.nan)

    for t in range(n_times):
        slopes[t], intercepts[t] = calculate_lapse_rate_timestep(
            temp_data[t, :], elevation_data, site_mask
        )

    return slopes, intercepts
