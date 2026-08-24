#!/usr/bin/env python3
"""
test_deployment_periods.py

Tests for deployment period functions.
"""

import pandas as pd
import numpy as np
import xarray as xr
from pathlib import Path
import warnings

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from jiflr import ROOT
from jiflr.utils import get_deployment_periods, deployment_mask, apply_deployment_mask


def _make_dataset(dates):
    """Helper: create a minimal xr.Dataset with a datetime coordinate."""
    return xr.Dataset(
        {"temp_c": (["datetime"], np.random.randn(len(dates)))},
        coords={"datetime": dates},
    )


def test_get_deployment_periods():
    """Test the get_deployment_periods function."""
    print("=" * 60)
    print("TESTING get_deployment_periods()")
    print("=" * 60)

    csv_path = Path(ROOT) / "data" / "2025" / "metadata" / "deployment_periods.csv"

    if not csv_path.exists():
        print(f"[SKIP] CSV file not found: {csv_path}")
        return True

    # Single site
    periods = get_deployment_periods("A01", csv_path)
    assert isinstance(periods, dict), "Expected dict"
    assert "A01" in periods, "A01 not found in results"
    print(f"A01 has {len(periods['A01'])} deployment periods")

    # Multiple sites
    multi = get_deployment_periods(["A01", "B02"], csv_path)
    assert "A01" in multi and "B02" in multi

    # Non-existent site returns empty list
    fake = get_deployment_periods("FAKE01", csv_path)
    assert fake.get("FAKE01", []) == []

    print("[PASS] get_deployment_periods() tests passed!")
    return True


def test_deployment_mask():
    """Test deployment_mask with synthetic data."""
    print("\n" + "=" * 60)
    print("TESTING deployment_mask()")
    print("=" * 60)

    csv_path = Path(ROOT) / "data" / "2025" / "metadata" / "deployment_periods.csv"
    if not csv_path.exists():
        print(f"[SKIP] CSV file not found: {csv_path}")
        return True

    # Build a dataset spanning the whole year
    dates = pd.date_range("2025-01-01", "2025-12-31", freq="1h")
    ds = _make_dataset(dates)

    # deployment_mask raises on missing site
    try:
        deployment_mask(ds, "FAKE_SITE_XYZ", csv_path, ignore_missing=False)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    # ignore_missing returns all-True
    mask = deployment_mask(ds, "FAKE_SITE_XYZ", csv_path, ignore_missing=True)
    assert mask.all(), "ignore_missing should return all-True mask"

    # Real site produces a valid boolean array
    mask = deployment_mask(ds, "A01", csv_path, ignore_missing=True)
    assert isinstance(mask, np.ndarray)
    assert mask.dtype == bool
    assert mask.shape == (len(dates),)
    print(f"A01 deployed: {mask.sum()} / {len(mask)} timesteps")

    print("[PASS] deployment_mask() tests passed!")
    return True


def test_apply_deployment_mask():
    """Test apply_deployment_mask with synthetic data."""
    print("\n" + "=" * 60)
    print("TESTING apply_deployment_mask()")
    print("=" * 60)

    csv_path = Path(ROOT) / "data" / "2025" / "metadata" / "deployment_periods.csv"
    if not csv_path.exists():
        print(f"[SKIP] CSV file not found: {csv_path}")
        return True

    dates = pd.date_range("2025-01-01", "2025-12-31", freq="1h")
    ds = _make_dataset(dates)

    # ignore_missing=False raises on missing site
    try:
        apply_deployment_mask(ds, "FAKE_SITE_XYZ", csv_path, ignore_missing=False)
        raise AssertionError("Should have raised ValueError")
    except ValueError:
        pass

    # ignore_missing=True returns dataset unchanged
    ds_out = apply_deployment_mask(ds, "FAKE_SITE_XYZ", csv_path, ignore_missing=True)
    assert isinstance(ds_out, xr.Dataset)

    # Real site: out-of-deployment values become NaN
    mask = deployment_mask(ds, "A01", csv_path, ignore_missing=True)
    ds_masked = apply_deployment_mask(ds, "A01", csv_path, ignore_missing=True)
    temp_out = ds_masked["temp_c"].values
    n_nan = np.isnan(temp_out).sum()
    n_non_deployed = (~mask).sum()
    assert n_nan == n_non_deployed, (
        f"Expected {n_non_deployed} NaN values, got {n_nan}"
    )
    print(f"A01: {n_non_deployed} out-of-deployment values set to NaN")

    print("[PASS] apply_deployment_mask() tests passed!")
    return True


def main():
    warnings.filterwarnings("ignore")
    results = [
        test_get_deployment_periods(),
        test_deployment_mask(),
        test_apply_deployment_mask(),
    ]
    if all(results):
        print("\nALL TESTS PASSED!")
        return True
    print("\nSOME TESTS FAILED!")
    return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
