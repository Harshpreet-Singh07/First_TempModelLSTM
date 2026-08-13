"""
OMNI2 Kp index loader (planetary geomagnetic activity index), 2025.

Source: NASA/GSFC OMNIWeb subsetted hourly file
        data/OMNI_data/omni2_qFVli8Rtt7.lst
        columns: YEAR (I4), DOY (I4), Hour (I3), Kp index (I3)

OMNI stores Kp multiplied by 10 and quantised to thirds, so the raw integers
run 0, 3, 7, 10, 13, 17, ... 90 (documented as "3+ = 33, 6- = 57").  Dividing
by 10 recovers the familiar 0-9 Kp scale: 0.0, 0.3, 0.7, 1.0, ...

Kp is a THREE-HOURLY index.  OMNI repeats each value across the three hours of
its block, so the 8760-row hourly file carries only 2920 independent values.
That matters for any statistic that assumes independent samples - the hourly
and 3-hourly series have identical distributions (each value appears exactly
three times), so means, quantiles and moments are unchanged, but the effective
sample size is a third of the nominal one.

Kp is dimensionless.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LST = ROOT / "data" / "OMNI_data" / "omni2_qFVli8Rtt7.lst"

FILL = 99          # OMNI fill value for a missing Kp


def load_kp(path=LST):
    """Return the hourly Kp index for 2025 as a pandas Series (0-9 scale)."""
    raw = pd.read_csv(path, sep=r"\s+", header=None,
                      names=["year", "doy", "hour", "kp_raw"])

    index = (pd.to_datetime(raw["year"].astype(str)
                            + raw["doy"].astype(str).str.zfill(3), format="%Y%j")
             + pd.to_timedelta(raw["hour"], unit="h"))

    kp = raw["kp_raw"].astype("float64")
    kp[kp >= FILL] = np.nan            # missing hours, if any
    kp = kp / 10.0

    s = pd.Series(kp.to_numpy(), index=pd.DatetimeIndex(index), name="kp")
    return s.sort_index()


def three_hourly(s):
    """Collapse the repeated hourly values back to the native 3-hourly series."""
    return s.iloc[::3]


if __name__ == "__main__":
    s = load_kp()
    print(f"rows        : {len(s):,}   {s.index[0]} .. {s.index[-1]}")
    print(f"missing     : {int(s.isna().sum())}")
    print(f"range       : {s.min():.1f} .. {s.max():.1f}")
    print(f"native 3-hly: {len(three_hourly(s)):,} values")
    print("\nstorm levels (NOAA G-scale, Kp >= 5):")
    for lo, name in [(5, "G1 minor"), (6, "G2 moderate"), (7, "G3 strong"),
                     (8, "G4 severe"), (9, "G5 extreme")]:
        hrs = int((s >= lo).sum())
        print(f"  Kp >= {lo} ({name:11s}): {hrs:5,} hours  ({100 * hrs / len(s):5.2f}%)")
