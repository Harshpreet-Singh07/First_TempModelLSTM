"""
Reduce a 2026 ERA5 2 m temperature GRIB to the same domain-mean hourly series
as 2025, so the LSTM can be scored against real observations.

Expects a GRIB downloaded from the CDS with the *same domain* as the 2025 file:

    "area": [57, 74.25, 41.75, 95]        # [North, West, South, East]

Note those are EAST longitudes. A request written as [57, -95, 41.75, -74.25]
returns northern North America instead, and the resulting series is not
comparable with the 2025 training data at all - so the longitudes are checked
here rather than silently averaged.

Usage:
    python scripts/era5_2026.py                     # newest GRIB in data/raw/
    python scripts/era5_2026.py path/to/file.grib

Writes data/processed/domain_mean_hourly_2026.csv (valid_time, t2m in deg C).
"""

import sys
from pathlib import Path

import pandas as pd
import xarray as xr

from era5_series import area_weighted_mean, RAW, PROCESSED, GRIB as GRIB_2025

OUT = PROCESSED / "domain_mean_hourly_2026.csv"

# the 2025 training domain - anything materially different is not comparable
LAT_RANGE = (41.75, 57.0)
LON_RANGE = (74.25, 95.0)


def find_grib(argv):
    """The GRIB given on the command line, or the newest one that isn't 2025's."""
    if len(argv) > 1:
        p = Path(argv[1]).expanduser().resolve()
        if not p.exists():
            raise SystemExit(f"no such file: {p}")
        return p

    candidates = [p for p in RAW.glob("*.grib") if p.resolve() != GRIB_2025.resolve()]
    if not candidates:
        raise SystemExit(
            f"no 2026 GRIB found in {RAW}\n"
            f"download 2m_temperature for 2026 with area [57, 74.25, 41.75, 95] "
            f"and drop it there, or pass the path as an argument."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def check_domain(ds):
    """Fail loudly if the file is not the 2025 domain."""
    lat, lon = ds["latitude"].values, ds["longitude"].values
    lo_lat, hi_lat = float(lat.min()), float(lat.max())
    lo_lon, hi_lon = float(lon.min()), float(lon.max())
    print(f"  domain: {lo_lat:.2f}-{hi_lat:.2f} N, {lo_lon:.2f}-{hi_lon:.2f} E "
          f"({ds.sizes['latitude']} x {ds.sizes['longitude']})")

    off = (abs(lo_lat - LAT_RANGE[0]) > 1 or abs(hi_lat - LAT_RANGE[1]) > 1
           or abs(lo_lon - LON_RANGE[0]) > 1 or abs(hi_lon - LON_RANGE[1]) > 1)
    if off:
        raise SystemExit(
            f"\nthis GRIB covers {lo_lat:.2f}-{hi_lat:.2f} N, "
            f"{lo_lon:.2f}-{hi_lon:.2f} E, but the model was trained on "
            f"{LAT_RANGE[0]}-{LAT_RANGE[1]} N, {LON_RANGE[0]}-{LON_RANGE[1]} E.\n"
            f"Negative longitudes in the CDS request are the usual cause - "
            f"use area [57, 74.25, 41.75, 95]."
        )


def build(grib):
    print("Reading", grib)
    ds = xr.open_dataset(
        grib, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"stepType": "instant"}, "indexpath": ""},
    )
    check_domain(ds)

    print("  reducing t2m ...", flush=True)
    t2m = area_weighted_mean(ds["t2m"]).values.astype("float64") - 273.15
    idx = pd.DatetimeIndex(ds["valid_time"].values)
    ds.close()

    s = pd.Series(t2m, index=idx, name="t2m").sort_index()
    s = s[~s.index.duplicated()]
    s = s[s.index.year == 2026]
    s.index.name = "valid_time"
    return s.to_frame()


if __name__ == "__main__":
    df = build(find_grib(sys.argv))
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, float_format="%.6f")

    full = pd.date_range(df.index[0], df.index[-1], freq="h")
    missing = len(full) - len(df)
    print(f"\nrows: {len(df)}   range: {df.index[0]} .. {df.index[-1]}")
    if missing:
        print(f"gaps: {missing} missing hours in that range (the app interpolates)")
    print(f"t2m : mean {df['t2m'].mean():.2f}, min {df['t2m'].min():.2f}, "
          f"max {df['t2m'].max():.2f} deg C")
    print("Cached ->", OUT.resolve())
