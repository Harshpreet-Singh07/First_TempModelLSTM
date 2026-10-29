"""
Build the area-weighted domain-mean hourly series from the 2025 ERA5 GRIB file.

Source : 185bc0671d5cc50980a767e3a9f6cbe5.grib
Domain : 41.75-57.0 N, 74.25-95.0 E (Central Asia)   grid 62 x 84
Period : 2025-01-01 00:00 .. 2025-12-31 23:00 (8760 hourly steps)

Same philosophy as the earlier December-2025 scripts: every variable is reduced
to a single cos(latitude)-weighted domain mean per hour, then plotted.

The GRIB read is slow (~440 MB), so results are cached to
output/domain_mean_hourly_2025.csv and reused by the plotting scripts.

Units after reduction:
  t2m, d2m  deg C      u10, v10, wspd  m/s      tp  mm/hr
  t2m_adv   deg C/hr   (temperature advection by the 10 m wind, see below)

reduce_grib() does the same reduction for any year's GRIB; scripts/era5_years.py
uses it to build the multi-year record the V3 LSTM trains on.
"""

import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

# All paths are anchored to the project root, so the scripts can be run from
# anywhere (python scripts/stats_2025.py, or from inside scripts/).
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGURES = ROOT / "output" / "figures"
TABLES = ROOT / "output" / "tables"

GRIB = RAW / "185bc0671d5cc50980a767e3a9f6cbe5.grib"
CACHE = PROCESSED / "domain_mean_hourly_2025.csv"

YEAR_START = "2025-01-01 00:00"
YEAR_END = "2025-12-31 23:00"

# the 2025 domain - every other year must match it to be comparable
LAT_RANGE = (41.75, 57.0)
LON_RANGE = (74.25, 95.0)
CDS_AREA = [57, 74.25, 41.75, 95]       # [N, W, S, E] - EAST longitudes


def area_weighted_mean(da):
    """Domain mean weighted by cos(latitude)."""
    w = np.cos(np.deg2rad(da["latitude"]))
    return da.weighted(w).mean(dim=("latitude", "longitude"))


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
            f"use area {CDS_AREA}."
        )


def _open(step_type, grib=GRIB):
    return xr.open_dataset(
        grib, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"stepType": step_type}, "indexpath": ""},
    )


# --------------------------------------------------------------- advection
R_EARTH = 6.371e6        # m


def temperature_advection(t, u, v, lat, lon):
    """-(u dT/dx + v dT/dy) per grid cell, in deg C per hour.

    How fast the wind is carrying warmer or colder air into each cell: positive
    when the wind blows from warm air towards cold (that cell warms). This is the
    "wind moves heat from one place to another" term, which a single domain-mean
    temperature cannot see on its own - it needs the gradient across the grid.

    t, u, v: (time, lat, lon) arrays; lat, lon: 1-D coordinates in degrees.
    np.gradient takes the coordinates directly, so the descending ERA5 latitude
    axis needs no flipping, and the edges fall back to one-sided differences.
    """
    phi = np.deg2rad(lat)
    dt_dx = np.gradient(t, np.deg2rad(lon), axis=2) / (R_EARTH * np.cos(phi))[None, :, None]
    dt_dy = np.gradient(t, phi, axis=1) / R_EARTH
    return -(u * dt_dx + v * dt_dy) * 3600.0


def mean_advection(ds, chunk=744):
    """Domain mean of temperature_advection, a month of hours at a time.

    Same cos(latitude) weighting as area_weighted_mean, done in numpy so a full
    year never sits in memory as float64 gradients at once.
    """
    lat, lon = ds["latitude"].values, ds["longitude"].values
    w = np.cos(np.deg2rad(lat))[None, :, None]
    norm = w.sum() * len(lon)
    out = []
    for i in range(0, ds.sizes["time"], chunk):
        sl = {"time": slice(i, i + chunk)}
        adv = temperature_advection(ds["t2m"].isel(sl).values.astype("float64"),
                                    ds["u10"].isel(sl).values.astype("float64"),
                                    ds["v10"].isel(sl).values.astype("float64"), lat, lon)
        out.append((adv * w).sum(axis=(1, 2)) / norm)
    return np.concatenate(out)


# -------------------------------------------------------------- reduction
def reduce_grib(grib):
    """Read one ERA5 GRIB and return its hourly domain means as a DataFrame.

    Reduces whichever of t2m, d2m, u10, v10, tp the file holds (t2m is
    required), plus wind speed and t2m_adv when both wind components are there.
    Rows are every valid_time in the file; callers trim to the year they want.
    """
    print("Reading", grib)
    ds_i = _open("instant", grib)
    check_domain(ds_i)
    if "t2m" not in ds_i:
        raise SystemExit(f"{grib} has no 2 m temperature (t2m) - nothing to reduce")
    idx = pd.DatetimeIndex(ds_i["valid_time"].values)

    cols = {}
    for name, conv in [("t2m", lambda x: x - 273.15),
                       ("d2m", lambda x: x - 273.15),
                       ("u10", lambda x: x),
                       ("v10", lambda x: x)]:
        if name not in ds_i:
            continue
        print(f"  reducing {name} ...", flush=True)
        cols[name] = conv(area_weighted_mean(ds_i[name]).values.astype("float64"))

    if "u10" in cols and "v10" in cols:
        # true area-mean wind speed: average the speed, not the components
        print("  reducing wind speed ...", flush=True)
        spd = np.sqrt(ds_i["u10"] ** 2 + ds_i["v10"] ** 2)
        cols["wspd"] = area_weighted_mean(spd).values.astype("float64")
        # magnitude of the mean vector - the quantity the December figures plotted
        cols["wspd_vec"] = np.sqrt(cols["u10"] ** 2 + cols["v10"] ** 2)

        print("  computing temperature advection ...", flush=True)
        cols["t2m_adv"] = mean_advection(ds_i)

    inst = pd.DataFrame(cols, index=idx)
    ds_i.close()

    # ---- precipitation: (time, step) forecast blocks -> flat hourly series.
    # Each field is already a 1-hour accumulation ending at its valid_time,
    # so the blocks only need flattening, not differencing.
    try:
        ds_a = _open("accum", grib)
    except Exception:              # cfgrib raises a variety of errors on no match
        ds_a = None
    if ds_a is not None and "tp" in ds_a:
        print("  reducing tp ...", flush=True)
        tp = area_weighted_mean(ds_a["tp"]).values.astype("float64") * 1000.0  # m -> mm
        vt = pd.DatetimeIndex(ds_a["valid_time"].values.ravel())
        prec = pd.Series(tp.ravel(), index=vt).sort_index()
        prec = prec[~prec.index.duplicated()]
        inst = inst.join(prec.rename("tp"), how="left")
    if ds_a is not None:
        ds_a.close()

    # the column order the 2025 CSV has always had, with the new column last
    order = ["t2m", "d2m", "u10", "v10", "wspd", "wspd_vec", "tp", "t2m_adv"]
    df = inst[[c for c in order if c in inst]].sort_index()
    df = df[~df.index.duplicated()]
    df.index.name = "valid_time"
    return df


def build():
    """Read the GRIB and return a DataFrame of hourly domain means for 2025."""
    return reduce_grib(GRIB).loc[YEAR_START:YEAR_END]


def load(refresh=False):
    """Return the cached series, building it from the GRIB if needed."""
    if CACHE.exists() and not refresh:
        return pd.read_csv(CACHE, index_col="valid_time", parse_dates=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    print("Building domain-mean series from", GRIB)
    df = build()
    df.to_csv(CACHE, float_format="%.6f")
    print("Cached ->", CACHE.resolve())
    return df


# ------------------------------------------------------------------ shared style
DOMAIN = "41.8-57.0N, 74.2-95.0E"
PERIOD = "01 Jan 2025 - 31 Dec 2025"
SUPTITLE = f"ERA5 domain-mean ({DOMAIN})  |  {PERIOD}"

C = {"temp": "#d1495b", "dew": "#2e86ab", "wind": "#3a7d44",
     "u10": "#3a7d44", "v10": "#2a6f97", "precip": "#5b3a99", "rh": "#e07a1f",
     "kp": "#4a4e69"}


def rh(temp_c, dew_c):
    """Relative humidity from temperature and dewpoint (Magnus formula)."""
    a, b = 17.625, 243.04
    return 100 * np.exp((a * dew_c) / (b + dew_c) - (a * temp_c) / (b + temp_c))


if __name__ == "__main__":
    d = load(refresh=True)
    print("\nrows:", len(d), " range:", d.index[0], "..", d.index[-1])
    print("missing values per column:\n", d.isna().sum())
    print("\n", d.describe().round(3))
