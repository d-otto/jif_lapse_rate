"""Metadata for the processed JIFLR NetCDF products.

Keep descriptions here rather than on individual pipeline writers. Call
``apply_product_metadata`` after all merges and processing, just before writing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr


QC_FLAG_MASKS = np.array([1, 2, 4, 8, 16, 32], dtype=np.uint32)
QC_FLAG_MEANINGS = (
    "manual_mask_period wind_speed_exceeds_50_m_s rainfall_outside_allowed_sites "
    "wind_direction_low_speed pressure_noise_candidate wind_speed_noise_candidate"
)
QC_FLAG_COMMENT = (
    "Bit 1 marks a CSV-defined manual mask period. Bit 2 marks wind speed "
    "greater than 50 m/s. Bit 4 marks rainfall outside A04 and A17. Bit 8 "
    "marks wind direction at wind speeds below 0.5 m/s. Bit 16 marks a PACE "
    "pressure Hampel noise candidate. Bit 32 marks an average or maximum wind "
    "speed Hampel noise candidate. A nonzero flag masks the corresponding "
    "Level 1 measurement. Mask values are powers of two and may be combined. "
    "Zero means no listed condition was flagged, not that an observation exists "
    "or that every possible quality check was performed. Filter applicability "
    "depends on field season; consult qc_filter_order in the seasonal product."
)

MEASUREMENT_ATTRS = {
    "temp_c": {
        "standard_name": "air_temperature",
        "long_name": "Air temperature",
        "units": "degC",
    },
    "intensity_lux": {"long_name": "Illuminance", "units": "lux"},
    "pressure": {
        "standard_name": "air_pressure",
        "long_name": "Air pressure",
        "units": "kPa",
    },
    "relative_humidity": {
        "standard_name": "relative_humidity",
        "long_name": "Relative humidity",
        "units": "percent",
    },
    "wind_direction": {
        "standard_name": "wind_from_direction",
        "long_name": "Wind direction",
        "units": "degree",
    },
    "wind_speed_avg": {
        "standard_name": "wind_speed",
        "long_name": "Average wind speed reported by logger",
        "units": "m s-1",
    },
    "wind_speed_max": {
        "standard_name": "wind_speed",
        "long_name": "Maximum wind speed reported by logger",
        "units": "m s-1",
    },
    "rainfall_mm": {
        "long_name": "Rainfall reported during logger output interval",
        "units": "mm",
    },
}

WIND_SPEED_MAX_METHODS = {
    "pace": ("Pace", "Maximum 2 second mean"),
    "rmyoung": ("RM Young", "Maximum of 1 minute samples"),
}


def wind_speed_max_method(sensor_types: set[str]) -> str:
    """Describe logger maximum methods represented by the given sensors."""
    normalized = {sensor_type.casefold() for sensor_type in sensor_types}
    return "; ".join(
        f"{label}: {method}"
        for sensor_type, (label, method) in WIND_SPEED_MAX_METHODS.items()
        if any(value.startswith(sensor_type) for value in normalized)
    )

COORDINATE_ATTRS = {
    "sensor_idx": {
        "long_name": "Sensor record index within this file",
        "comment": "Not a persistent instrument identifier; use sensor_id and year.",
    },
    "datetime_utc": {"standard_name": "time", "axis": "T", "long_name": "UTC time"},
    "latitude": {"standard_name": "latitude", "units": "degrees_north"},
    "longitude": {"standard_name": "longitude", "units": "degrees_east"},
    "elevation": {
        "long_name": "Approximate site elevation",
        "units": "m",
        "comment": (
            "Approximate site value from deployment metadata. Vertical reference "
            "and uncertainty have not been established; do not use for precise "
            "vertical positioning."
        ),
    },
    "height": {
        "long_name": "Nominal sensor height above the local surface",
        "comment": "String deployment label, such as 1m; not a surveyed vertical coordinate.",
    },
    "sensor_id": {"long_name": "Sensor identifier"},
    "site_id": {"long_name": "Observed site identifier"},
    "canonical_site_id": {"long_name": "Cross-season site identifier"},
    "sensor_type": {"long_name": "Sensor type"},
    "sensor_generation": {"long_name": "Sensor generation"},
    "shielding": {"long_name": "Sensor shielding classification"},
    "site_type": {"long_name": "Site type"},
    "processing_group": {"long_name": "Processing group"},
    "year": {"long_name": "Field season year"},
}

PRODUCT_TITLES = {
    "on_ice_intensive": "JIFLR on-ice intensive observations",
    "on_ice_standard": "JIFLR on-ice standard observations",
    "off_ice": "JIFLR off-ice observations",
}

PRESERVED_GLOBAL_ATTRS = (
    "deployment_metadata_source",
    "processing_step",
    "qc_filter_order",
    "qc_flag_meanings",
    "qc_flags_mask_lvl1_data",
    "qc_manual_mask_periods_sha256",
    "pressure_noise_qc_method",
    "pressure_noise_qc_rolling_window",
    "pressure_noise_qc_min_periods",
    "pressure_noise_qc_mad_multiplier",
    "pressure_noise_qc_absolute_floor_kpa",
    "wind_speed_noise_qc_method",
    "wind_speed_noise_qc_rolling_window",
    "wind_speed_noise_qc_min_periods",
    "wind_speed_noise_qc_mad_multiplier",
    "wind_speed_noise_qc_absolute_floor_m_s",
    "processed_timestamp",
)


def _utc_text(value: np.datetime64) -> str:
    """Format a timezone-naive UTC coordinate as an ISO 8601 UTC timestamp."""
    return pd.Timestamp(value).isoformat() + "Z"


def apply_product_metadata(
    dataset: xr.Dataset, *, level: str, product: str, source_years: tuple[int, ...] = ()
) -> xr.Dataset:
    """Apply shared metadata to a finished Level 0 or Level 1 product.

    This function does not alter measurements or the string-valued ``height``
    coordinate. It discards source-sensor attributes inherited by concatenation.
    """
    if level not in {"lvl0", "lvl1"}:
        raise ValueError(f"Unsupported processing level: {level}")
    if not product:
        raise ValueError("Product name is required")
    if "sensor_idx" not in dataset.sizes or "datetime_utc" not in dataset.coords:
        raise ValueError("Processed metadata requires sensor_idx and datetime_utc")
    if dataset.sizes["sensor_idx"] == 0 or dataset.sizes["datetime_utc"] == 0:
        raise ValueError("Processed metadata requires nonempty sensor and time axes")
    times = pd.DatetimeIndex(dataset["datetime_utc"].values)
    if times.hasnans or not times.is_monotonic_increasing or not times.is_unique:
        raise ValueError("datetime_utc must be nonmissing, sorted, and unique")
    if level == "lvl1" and not times.equals(times.floor("5min")):
        raise ValueError("Level 1 datetime_utc must label five-minute bin starts")

    result = dataset.copy(deep=False)
    original_globals = dataset.attrs
    preserved = ("processing_step",) if source_years else PRESERVED_GLOBAL_ATTRS
    attrs = {key: original_globals[key] for key in preserved if key in original_globals}
    title = PRODUCT_TITLES.get(product, f"JIFLR {product.replace('_', ' ')} observations")
    attrs.update(
        {
            "title": f"{title}, Level {level[-1]}",
            "institution": "JIFLR Project",
            "project": "Juneau Icefield Lapse Rate (JIFLR)",
            "summary": (
                "In situ meteorological observations from the Juneau Icefield "
                "Lapse Rate project. Sensor records share a UTC time axis; "
                "instrument and deployment information are sensor-indexed coordinates. "
                + (
                    "Level 0 retains measurements with separate quality-control flags."
                    if level == "lvl0" else
                    "Level 1 contains five-minute arithmetic means, masked wherever "
                    "any contributing Level 0 observation has a masking QC flag. "
                    "No interpolation or pressure low-pass filtering is applied."
                )
            ),
            "source": "In situ field observations from the listed sensor types",
            "processing_level": level,
            "time_coordinate": "datetime_utc",
            "time_coverage_timezone": "UTC",
            "time_coverage_start": _utc_text(times[0]),
            "time_coverage_end": _utc_text(times[-1]),
            "n_sensors": dataset.sizes["sensor_idx"],
        }
    )
    if "site_id" in dataset.coords:
        attrs["n_sites"] = len(set(dataset["site_id"].values.astype(str)))
    if source_years:
        attrs["source_years"] = ", ".join(str(year) for year in source_years)
    for coordinate, abbreviation in (("latitude", "lat"), ("longitude", "lon")):
        if coordinate in dataset.coords:
            values = np.asarray(dataset[coordinate].values, dtype=float)
            finite = values[np.isfinite(values)]
            if finite.size:
                attrs[f"geospatial_{abbreviation}_min"] = float(finite.min())
                attrs[f"geospatial_{abbreviation}_max"] = float(finite.max())
                attrs[f"geospatial_{abbreviation}_units"] = COORDINATE_ATTRS[coordinate]["units"]
    if level == "lvl1":
        attrs["resampling_method"] = "mean of available observations in 5-minute bins"
        attrs["time_resolution"] = "5 minutes"
        attrs["time_coverage_resolution"] = "PT5M"
        attrs["qc_aggregation"] = (
            "Bitwise OR of source flags in each bin; any nonzero masking flag "
            "excludes the entire bin, including otherwise unflagged observations."
        )
    result.attrs = attrs

    for name, definitions in COORDINATE_ATTRS.items():
        if name in result.coords:
            result[name].attrs = definitions.copy()

    for name in result.data_vars:
        variable = result[name]
        if name == "datetime_utc_bounds":
            continue  # Rebuilt below after concatenation, without a sensor dimension.
        if name.endswith("_qc_flag"):
            if variable.dtype != np.dtype("uint32"):
                raise ValueError(f"{name} must be uint32, got {variable.dtype}")
            flag_attrs = {
                "long_name": f"Quality-control flags for {name.removesuffix('_qc_flag')}",
                "standard_name": "quality_flag",
                "flag_masks": QC_FLAG_MASKS.copy(),
                "flag_meanings": QC_FLAG_MEANINGS,
                "valid_min": np.uint32(0),
                "comment": QC_FLAG_COMMENT,
            }
            if variable.attrs.get("qc_masks_measurement") == "false":
                flag_attrs = variable.attrs.copy()
            result[name].attrs = flag_attrs
            continue
        if name == "datetime":
            result[name].attrs = {
                "long_name": "Legacy source timestamp",
                "comment": (
                    "Source time zones and Level 1 aggregation of this variable "
                    "are not standardized. Use datetime_utc for analysis."
                ),
            }
            continue
        if name not in MEASUREMENT_ATTRS:
            if {"sensor_idx", "datetime_utc"}.issubset(variable.dims):
                raise ValueError(f"No processed-product metadata defined for {name}")
            # Nonmeasurement diagnostics retain their own metadata.
            continue
        measurement_attrs = MEASUREMENT_ATTRS[name].copy()
        if level == "lvl1":
            measurement_attrs["cell_methods"] = "datetime_utc: mean"
        flag_names = [
            flag_name for flag_name in (f"{name}_qc_flag", f"{name}_interpolated_qc_flag")
            if flag_name in result.data_vars
        ]
        if name == "pressure" and "interpolated_qc_flag" in result.data_vars:
            flag_names.append("interpolated_qc_flag")
        if flag_names:
            measurement_attrs["ancillary_variables"] = " ".join(flag_names)
        if name == "temp_c":
            if level == "lvl0":
                if "sensor_type" in result.coords and not any(
                    str(value).casefold().startswith("rmyoung")
                    for value in result["sensor_type"].values
                ):
                    measurement_attrs["cell_methods"] = "datetime_utc: point"
                measurement_attrs["comment"] = (
                    "Pace and HOBO pendant values are point measurements. RM Young "
                    "values are logger averages of five one-minute samples. "
                    "Use sensor_type to distinguish sources."
                )
            else:
                measurement_attrs["cell_methods"] = "datetime_utc: mean"
                measurement_attrs["comment"] = (
                    "Arithmetic mean of available temperature observations in "
                    "each five-minute bin; a bin may contain one observation. "
                    "RM Young inputs are logger averages of five one-minute samples."
                )
        elif name == "wind_speed_max" and "sensor_type" in result.coords:
            method = wind_speed_max_method(
                {str(value) for value in result["sensor_type"].values}
            )
            if method:
                measurement_attrs["method"] = method
                if level == "lvl1":
                    measurement_attrs["comment"] = (
                        "Level 1 takes the arithmetic mean of available "
                        "logger maxima in each five-minute bin."
                    )
        elif level == "lvl1":
            measurement_attrs["comment"] = (
                "Level 1 currently takes the arithmetic mean of available "
                "logger values in each five-minute bin. Interpret this together "
                "with the variable's original logger interval definition."
            )
        if name == "rainfall_mm":
            measurement_attrs["comment"] = (
                measurement_attrs.get("comment", "") + " "
                "Gauge undercatch and unmeasured mist make amounts unreliable; "
                "use as an indicator of significant precipitation occurrence."
            ).strip()
        if name == "wind_direction" and level == "lvl1":
            measurement_attrs["comment"] += (
                " This is an arithmetic, not circular, mean; values spanning "
                "north (0/360 degrees) can produce misleading directions."
            )
        result[name].attrs = measurement_attrs
    if level == "lvl1":
        result["datetime_utc"].attrs.update({
            "bounds": "datetime_utc_bounds",
            "comment": "Start of a left-closed, right-open five-minute UTC bin.",
        })
        result["datetime_utc_bounds"] = xr.DataArray(
            np.column_stack((times.values, (times + pd.Timedelta(minutes=5)).values)),
            dims=("datetime_utc", "bounds"),
            attrs={"long_name": "Five-minute aggregation interval boundaries"},
        )
        # Use identical encodings so boundary values have the time axis's units.
        for name in ("datetime_utc", "datetime_utc_bounds"):
            result[name].encoding.update({
                "units": "seconds since 1970-01-01 00:00:00",
                "calendar": "proleptic_gregorian",
            })
    return result
