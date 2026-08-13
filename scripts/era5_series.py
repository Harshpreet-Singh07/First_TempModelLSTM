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


def area_weighted_mean(da):
    """Domain mean weighted by cos(latitude)."""
    w = np.cos(np.deg2rad(da["latitude"]))
    return da.weighted(w).mean(dim=("latitude", "longitude"))


def _open(step_type):
    return xr.open_dataset(
        GRIB, engine="cfgrib",
        backend_kwargs={"filter_by_keys": {"stepType": step_type}, "indexpath": ""},
    )


def build():
    """Read the GRIB and return a DataFrame of hourly domain means for 2025."""
    ds_i = _open("instant")
    idx = pd.DatetimeIndex(ds_i["valid_time"].values)

    cols = {}
    for name, conv in [("t2m", lambda x: x - 273.15),
                       ("d2m", lambda x: x - 273.15),
                       ("u10", lambda x: x),
                       ("v10", lambda x: x)]:
        print(f"  reducing {name} ...", flush=True)
        cols[name] = conv(area_weighted_mean(ds_i[name]).values.astype("float64"))

    # true area-mean wind speed: average the speed, not the components
    print("  reducing wind speed ...", flush=True)
    spd = np.sqrt(ds_i["u10"] ** 2 + ds_i["v10"] ** 2)
    cols["wspd"] = area_weighted_mean(spd).values.astype("float64")
    # magnitude of the mean vector - the quantity the December figures plotted
    cols["wspd_vec"] = np.sqrt(cols["u10"] ** 2 + cols["v10"] ** 2)

    inst = pd.DataFrame(cols, index=idx)
    ds_i.close()

    # ---- precipitation: (time, step) forecast blocks -> flat hourly series.
    # Each field is already a 1-hour accumulation ending at its valid_time,
    # so the blocks only need flattening, not differencing.
    print("  reducing tp ...", flush=True)
    ds_a = _open("accum")
    tp = area_weighted_mean(ds_a["tp"]).values.astype("float64") * 1000.0  # m -> mm
    vt = pd.DatetimeIndex(ds_a["valid_time"].values.ravel())
    ds_a.close()
    prec = pd.Series(tp.ravel(), index=vt).sort_index()
    prec = prec[~prec.index.duplicated()]

    df = inst.join(prec.rename("tp"), how="left")
    df = df.loc[YEAR_START:YEAR_END]
    df.index.name = "valid_time"
    return df


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
