"""
Descriptive statistics for every ERA5 variable, calendar year 2025.

Computed on the area-weighted (cos-latitude) domain-mean hourly series -
the same series the individual figures plot - over
41.75-57.0 N, 74.25-95.0 E, n = 8760 hourly values.

  t2m  2 m temperature          deg C
  u10  10 m U wind component    m/s
  v10  10 m V wind component    m/s
  tp   total precipitation      mm per hour   (daily totals also reported)
  d2m  2 m dewpoint temperature deg C

Skewness is the sample (adjusted Fisher-Pearson) coefficient G1, matching
Excel's SKEW.  Quartiles use linear interpolation (numpy default = R type 7).
Standard deviation is the sample SD (ddof = 1).

Output: printed table + output/summary_statistics_2025.csv
"""

import numpy as np
import pandas as pd
from pathlib import Path

from era5_series import load, TABLES as OUT
from omni_series import load_kp

OUT.mkdir(parents=True, exist_ok=True)

ORDER = ["Standard Deviation", "Minimum", "Maximum", "Skewness",
         "First Quartile", "Second Quartile/Median", "Third Quartile", "Mean"]


def skewness(x):
    """Sample skewness G1 (adjusted Fisher-Pearson, as in Excel's SKEW)."""
    n = x.size
    s = x.std(ddof=1)
    g1 = ((x - x.mean()) ** 3).sum() / (n * s ** 3)
    return g1 * np.sqrt(n * (n - 1)) / (n - 2)


def stats(x):
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    q1, q2, q3 = np.percentile(x, [25, 50, 75])
    return {
        "Standard Deviation": x.std(ddof=1),
        "Minimum": x.min(),
        "Maximum": x.max(),
        "Skewness": skewness(x),
        "First Quartile": q1,
        "Second Quartile/Median": q2,
        "Third Quartile": q3,
        "Mean": x.mean(),
    }


df = load()
kp = load_kp()

SERIES = [
    ("Temperature, 2025", "t2m", "deg C", df["t2m"].to_numpy()),
    ("Wind Speed U-Component, 2025", "u10", "m/s", df["u10"].to_numpy()),
    ("Wind Speed V-Component, 2025", "v10", "m/s", df["v10"].to_numpy()),
    ("Total Precipitation, 2025", "tp", "mm/hour", df["tp"].to_numpy()),
    ("Total Precipitation, 2025 (daily totals)", "tp_daily", "mm/day",
     df["tp"].resample("1D").sum().to_numpy()),
    ("Dewpoint Temperature, 2025", "d2m", "deg C", df["d2m"].to_numpy()),
    # Kp is 3-hourly, repeated across each block; the hourly and 3-hourly
    # series have identical distributions, so these statistics are unaffected.
    ("Kp Index, 2025", "kp", "dimensionless", kp.to_numpy()),
]

rows = []
for label, key, unit, values in SERIES:
    s = stats(values)
    for stat_name in ORDER:
        rows.append({
            "Variable Type": label,
            "Statistical Average Type": stat_name,
            "Value": s[stat_name],
            "Unit": "" if stat_name == "Skewness" else unit,
            "n": values.size,
        })

table = pd.DataFrame(rows)
table.to_csv(OUT / "summary_statistics_2025.csv", index=False, float_format="%.4f")

# ------------------------------------------------------------------ printed form
for label, key, unit, values in SERIES:
    s = stats(values)
    print(f"\n{label}   [{unit}]   n = {values.size:,}")
    print("-" * 58)
    for stat_name in ORDER:
        u = "" if stat_name == "Skewness" else f" {unit}"
        print(f"  {stat_name:<24} {s[stat_name]:>10.3f}{u}")

print("\nSaved:", (OUT / "summary_statistics_2025.csv").resolve())
