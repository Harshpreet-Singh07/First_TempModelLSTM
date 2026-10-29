"""
Reduce a 2026 ERA5 GRIB to the same domain-mean hourly series as 2025, so the
LSTM can be scored against real observations.

Expects a GRIB downloaded from the CDS with the *same domain* as the 2025 file:

    "area": [57, 74.25, 41.75, 95]        # [North, West, South, East]

Note those are EAST longitudes. A request written as [57, -95, 41.75, -74.25]
returns northern North America instead, and the resulting series is not
comparable with the 2025 training data at all - so the longitudes are checked
rather than silently averaged.

Which variables to download:
  - the V2 model (t2m) needs only 2m_temperature
  - the V3 model also needs 2m_dewpoint_temperature and both 10 m wind
    components (it computes temperature advection from them)
Whatever the file holds is reduced; a V3 forecast simply refuses days whose
inputs are missing. scripts/era5_years.py --years 2026 downloads all of them.

Usage:
    python scripts/era5_2026.py                     # newest GRIB in data/raw/
    python scripts/era5_2026.py path/to/file.grib

Writes data/processed/domain_mean_hourly_2026.csv (valid_time, t2m in deg C,
plus d2m, u10, v10, wspd, wspd_vec, tp, t2m_adv when the GRIB has them).
"""

import sys
from pathlib import Path

import pandas as pd

from era5_series import RAW, PROCESSED, GRIB as GRIB_2025, CDS_AREA, reduce_grib

OUT = PROCESSED / "domain_mean_hourly_2026.csv"


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
            f"download 2026 from the CDS with area {CDS_AREA} "
            f"and drop it there, or pass the path as an argument."
        )
    return max(candidates, key=lambda p: p.stat().st_mtime)


def build(grib):
    df = reduce_grib(grib)
    return df[df.index.year == 2026]


if __name__ == "__main__":
    df = build(find_grib(sys.argv))
    if df.empty:
        raise SystemExit("that GRIB has no 2026 hours in it")
    PROCESSED.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, float_format="%.6f")

    full = pd.date_range(df.index[0], df.index[-1], freq="h")
    missing = len(full) - len(df)
    print(f"\nrows: {len(df)}   range: {df.index[0]} .. {df.index[-1]}")
    print(f"columns: {', '.join(df.columns)}")
    if missing:
        print(f"gaps: {missing} missing hours in that range (the app interpolates)")
    print(f"t2m : mean {df['t2m'].mean():.2f}, min {df['t2m'].min():.2f}, "
          f"max {df['t2m'].max():.2f} deg C")
    print("Cached ->", OUT.resolve())
