"""
pdd_utils.py
------------
Core utilities for positive degree day (PDD) calculations and analysis.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Internal constants
# ---------------------------------------------------------------------------
_JULY_1_DOY = 182  # day-of-year of July 1 (non-leap year)
_STEPS_PER_YEAR_DAILY = 365


def datetimes_to_step_indices(
    datetimes: pd.DatetimeIndex,
    n_steps: int,
) -> np.ndarray:
    """
    Convert a DatetimeIndex to integer indices into a pdd() output array
    of length *n_steps*, anchored so that index 0 == July 1 00:00.

    Parameters
    ----------
    datetimes : pd.DatetimeIndex
        Timestamps to convert. Any regular frequency is supported.
    n_steps : int
        Length of the pdd() array these indices will address
        (e.g. 365 for daily, 35_040 for 15-min).

    Returns
    -------
    np.ndarray of int, shape (len(datetimes),)
        Indices in [0, n_steps).
    """
    dti = pd.DatetimeIndex(datetimes)

    # Fractional day-of-year within the annual cycle, anchored to July 1
    frac_doy = dti.day_of_year - 1 + dti.hour / 24.0 + dti.minute / 1440.0
    frac_from_jul1 = (frac_doy - _JULY_1_DOY) % _STEPS_PER_YEAR_DAILY

    # Scale to n_steps
    indices = (frac_from_jul1 / _STEPS_PER_YEAR_DAILY * n_steps).astype(int) % n_steps
    return indices


def pdd_from_datetimes(
    datetimes: pd.DatetimeIndex,
    T_ma: float,
    T_mj: float,
    sigma: float,
    sum_result: bool = True,
) -> float | np.ndarray:
    """
    Theoretical PDD for a given elevation over an arbitrary set of
    datetimes, using the sinusoidal temperature model.

    The temporal resolution is inferred automatically from the DatetimeIndex
    frequency (or the median gap between consecutive timestamps if freq is
    unset), so the caller never needs to compute n_steps or scaling factors.

    Parameters
    ----------
    datetimes : pd.DatetimeIndex
        Timestamps at which to evaluate the model.  Must be a regular series
        (uniform spacing).
    T_ma : float
        Mean annual temperature at this elevation (°C).
    T_mj : float
        Mean July temperature at this elevation (°C).
    sigma : float
        Standard deviation of the annual temperature cycle (°C).
    sum_result : bool, optional
        If True (default) return a single scalar (total degree-days).
        If False return a per-timestep array before summing so callers can
        apply additional masks.

    Returns
    -------
    float or np.ndarray
        Total PDD in degree-days (sum_result=True), or per-step
        contributions scaled to degree-days (sum_result=False).
    """
    dti = pd.DatetimeIndex(datetimes)

    # ── infer step duration in fractional days ──────────────────────────────
    if len(dti) < 2:
        raise ValueError("datetimes must contain at least two timestamps.")

    if dti.freq is not None:
        step_days = dti.freq.nanos / 1e9 / 86_400.0
    else:
        median_ns = np.median(np.diff(dti.asi8))
        step_days = median_ns / 1e9 / 86_400.0

    # Number of steps needed to represent one full year at this resolution
    n_steps = round(_STEPS_PER_YEAR_DAILY / step_days)

    # ── call the sinusoidal model at full-year resolution ───────────────────
    annual_array = pdd(T_ma, T_mj, sigma, n_steps=n_steps)  # shape (n_steps,)

    # ── select the relevant steps and scale to degree-days ──────────────────
    idx = datetimes_to_step_indices(dti, n_steps)
    per_step = annual_array[idx] * step_days  # each element now in degree-days

    return per_step.sum() if sum_result else per_step


def sum_observed_pdds(da: xr.DataArray) -> pd.Series:
    """
    Compute observed PDD from a DataArray of temperatures by summing
    positive values and scaling each timestep to degree-days.

    The temporal resolution is inferred from the datetime_utc coordinate,
    so the function works for any regular sampling frequency.

    Parameters
    ----------
    da : xr.DataArray
        Temperature array with dimensions (datetime_utc, sensor_idx).
        Must have a 'site_id' coordinate on sensor_idx and a regular
        datetime_utc coordinate.

    Returns
    -------
    pd.Series
        PDD in degree-days, indexed by site_id.
    """
    dti = pd.DatetimeIndex(da.datetime_utc.values)

    if dti.freq is not None:
        step_days = dti.freq.nanos / 1e9 / 86_400.0
    else:
        median_ns = float(np.median(np.diff(dti.asi8)))
        step_days = median_ns / 1e9 / 86_400.0

    pdd_vals = da.where(da > 0, 0).sum(dim="datetime_utc") * step_days

    return pd.Series(
        pdd_vals.values,
        index=da.site_id.values,
        name="pdd",
    )


def get_theoretical_pdd(
    da_period: xr.DataArray,
    t_ma_0: float,
    t_mj_0: float,
    lapse_t_ma: float,
    lapse_t_mj: float,
    sigma: float,
) -> pd.DataFrame:
    """
    Compute theoretical PDD for each site over the overlapping period of
    valid data in *da_period*, using the sinusoidal temperature model.

    Internally delegates to pdd_from_datetimes, so temporal resolution is
    inferred automatically and no manual index arithmetic is required.

    Parameters
    ----------
    da_period : xr.DataArray
        Temperature array with dimensions (datetime_utc, sensor_idx).
        Must have 'site_id' and 'elevation' coordinates on sensor_idx.
    t_ma_0 : float
        Mean annual temperature at reference elevation (°C).
    t_mj_0 : float
        Mean July temperature at reference elevation (°C).
    lapse_t_ma : float
        Lapse rate applied to t_ma_0 (°C/m).
    lapse_t_mj : float
        Lapse rate applied to t_mj_0 (°C/m).
    sigma : float
        Standard deviation of the annual temperature cycle (°C).

    Returns
    -------
    df : pd.DataFrame
        Columns ['pdd_theoretical', 'elevation'], indexed by site_id.
    start : pd.Timestamp
        First datetime of the overlapping valid period.
    end : pd.Timestamp
        Last datetime of the overlapping valid period.
    n_days : float
        Length of the overlapping period in days.
    """
    has_data = da_period.notnull().all(dim="sensor_idx")
    da_overlap = da_period.sel(datetime_utc=has_data)
    overlap_dti = pd.DatetimeIndex(da_overlap.datetime_utc.values)

    elev = (
        da_period.to_dataframe(name="temp_c")
        .reset_index()[["site_id", "elevation"]]
        .drop_duplicates("site_id")
        .set_index("site_id")["elevation"]
    )

    results = {
        site_id: pdd_from_datetimes(
            overlap_dti,
            T_ma=t_ma_0 + lapse_t_ma * elevation,
            T_mj=t_mj_0 + lapse_t_mj * elevation,
            sigma=sigma,
        )
        for site_id, elevation in elev.items()
    }

    df = pd.DataFrame.from_dict(results, orient="index", columns=["pdd_theoretical"])
    df.index.name = "site_id"
    df = df.join(elev)
    return df


def erfc_approx(x):
    """
    Approximation of the complementary error function for numba compatibility.
    Uses Abramowitz and Stegun approximation (equation 7.1.26).

    Parameters:
    -----------
    x : float or array
        Input value(s)

    Returns:
    --------
    float or array
        Approximation of erfc(x)
    """
    # Constants for the approximation
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429
    p = 0.3275911

    # Handle scalar input
    if np.isscalar(x):
        if x < 0:
            return 2.0 - erfc_approx(-x)

        # Abramowitz and Stegun approximation
        t = 1.0 / (1.0 + p * x)
        return t * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5)))) * np.exp(-x * x)

    # Handle array input
    result = np.zeros_like(x)
    for i in range(len(x)):
        if x[i] < 0:
            result[i] = 2.0 - erfc_approx(-x[i])
        else:
            t = 1.0 / (1.0 + p * x[i])
            result[i] = (
                t
                * (a1 + t * (a2 + t * (a3 + t * (a4 + t * a5))))
                * np.exp(-x[i] * x[i])
            )

    return result


def pdd(
    T_ma,
    T_mj,
    sigma,
    A=1.0,
    n_steps=365,
):
    """
    Calculate positive degree days using numba-compatible implementation.

    Parameters:
    -----------
    T_ma : float
        Mean annual surface-air temperature (°C)
    T_mj : float
        Mean July (January) surface-air temperature (°C)
    sigma : float
        Standard deviation of temperature from annual cycle (°C)
    A : float, optional
        Period length in years (default: 1.0)
    n_steps : int, optional
        Number of time steps for integration (default: 365)

    Returns:
    --------
    float
        Positive degree days
    """
    dt = A / n_steps

    # Initialize sum
    pdds = []

    # Integration loop
    for i in range(n_steps):
        t = i * dt

        # Annual temperature cycle (sinusoidal)
        T_ac = T_ma + (T_mj - T_ma) * np.cos(2 * np.pi * t / A)

        # Calculate the integrand from equation (6)
        term1 = sigma / np.sqrt(2 * np.pi) * np.exp(-(T_ac**2) / (2 * sigma**2))
        term2 = T_ac / 2 * erfc_approx(-T_ac / (np.sqrt(2) * sigma))

        integrand = term1 + term2

        # Add to sum (rectangular rule for simplicity in numba)
        pdds.append(integrand)

    return np.array(pdds)
