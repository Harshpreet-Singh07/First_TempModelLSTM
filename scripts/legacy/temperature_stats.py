"""
Descriptive statistics for the ERA5 2 m temperature (t2m) dataset.

Two views are reported:
  A) domain-mean time series  - the cos(lat)-weighted area mean per hour
                                (same series plotted in plot_timeseries.py)
  B) all raw grid points      - every (time, lat, lon) value, unweighted

Skewness is the sample (adjusted Fisher-Pearson) coefficient, G1.
Quartiles use the linear-interpolation method (numpy default, = R type 7).
"""

import numpy as np
import xarray as xr
from pathlib import Path

DATA = Path("382c46a9358c879403b826b3bb39ad87")
INSTANT = DATA / "data_stream-oper_stepType-instant.nc"


def skewness(x):
    """Sample skewness G1 (adjusted Fisher-Pearson, as in Excel's SKEW)."""
    x = np.asarray(x, dtype=np.float64)
    n = x.size
    m = x.mean()
    s = x.std(ddof=1)
    g1 = ((x - m) ** 3).sum() / (n * s ** 3)          # population/biased g1
    G1 = g1 * np.sqrt(n * (n - 1)) / (n - 2)          # sample-adjusted
    return g1, G1


def report(label, x, unit="deg C"):
    x = np.asarray(x, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    g1, G1 = skewness(x)
    print(f"\n{label}")
    print("-" * len(label))
    print(f"  n (values)          : {x.size:,}")
    print(f"  Mean                : {x.mean():8.3f} {unit}")
    print(f"  Standard deviation  : {x.std(ddof=1):8.3f} {unit}   (sample, ddof=1)")
    print(f"    population sd     : {x.std(ddof=0):8.3f} {unit}   (ddof=0)")
    print(f"  Minimum             : {x.min():8.3f} {unit}")
    print(f"  Maximum             : {x.max():8.3f} {unit}")
    print(f"  Range               : {x.max() - x.min():8.3f} {unit}")
    print(f"  Skewness            : {G1:8.4f}          (sample G1)")
    print(f"    biased g1         : {g1:8.4f}")
    print(f"  Q1 (25th pct)       : {q1:8.3f} {unit}")
    print(f"  Q2 (median, 50th)   : {q2:8.3f} {unit}")
    print(f"  Q3 (75th pct)       : {q3:8.3f} {unit}")
    print(f"  IQR (Q3 - Q1)       : {q3 - q1:8.3f} {unit}")


ds = xr.open_dataset(INSTANT)
t2m = ds["t2m"]

# A) area-weighted domain-mean series (deg C)
w = np.cos(np.deg2rad(t2m["latitude"]))
series = (t2m.weighted(w).mean(dim=("latitude", "longitude")) - 273.15).values

# B) every grid point (deg C)
grid = t2m.values - 273.15

print("=" * 62)
print("ERA5 2 m temperature (t2m) - descriptive statistics")
print("=" * 62)
print(f"file   : {INSTANT}")
print(f"shape  : {dict(t2m.sizes)}")
print(f"period : {str(ds.valid_time.values[0])[:16]} .. {str(ds.valid_time.values[-1])[:16]}")

report("A) Domain-mean hourly time series (cos-lat weighted)", series)
report("B) All raw grid points (time x lat x lon, unweighted)", grid)
report("C) All raw grid points, as stored in the file (Kelvin)", t2m.values, unit="K")

ds.close()
