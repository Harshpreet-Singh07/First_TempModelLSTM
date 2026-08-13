"""
ERA5 individual time-series graphs (calendar year 2025, Central Asia).

One figure per variable, one variable per figure - nothing is combined,
paired or derived from two variables. All five share an identical format:

  t2m  - 2 m temperature            (deg C)
  d2m  - 2 m dewpoint temperature   (deg C)
  u10  - 10 m U wind component      (m/s)
  v10  - 10 m V wind component      (m/s)
  tp   - total precipitation        (mm)

Each series is an area-weighted (cos-latitude) domain mean over
41.75-57.0 N, 74.25-95.0 E, hourly for all of 2025.

Figures are saved to ./output/.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

from era5_series import load, DOMAIN, PERIOD, C, FIGURES as OUT
from omni_series import load_kp

OUT.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

df = load()
df["kp"] = load_kp().reindex(df.index)     # OMNI2 Kp, same hourly index
t = df.index

# Each entry: one variable, its own figure. No pairing, no derived quantities.
plots = [
    dict(n=1, key="t2m", title="2 m temperature",
         ylabel="Temperature (deg C)", color=C["temp"]),
    dict(n=2, key="d2m", title="2 m dewpoint temperature",
         ylabel="Dewpoint (deg C)", color=C["dew"]),
    dict(n=3, key="u10", title="10 m U wind component",
         ylabel="U wind (m/s)", color=C["u10"]),
    dict(n=4, key="v10", title="10 m V wind component",
         ylabel="V wind (m/s)", color=C["v10"]),
    dict(n=5, key="tp", title="Total precipitation",
         ylabel="Precipitation (mm/day)", color=C["precip"]),
    # not ERA5: OMNI2 planetary geomagnetic activity index, same hourly grid
    dict(n=10, key="kp", title="Kp index (OMNI2)",
         ylabel="Kp (0-9)", color=C["kp"]),
]


# ---------------------------------------------------------- identical formatting
def make_figure(p):
    y = np.asarray(df[p["key"]].values, dtype=float)
    fig, ax = plt.subplots(figsize=(11, 4.5))

    if p["key"] == "tp":
        # a year of hourly bars is unreadable; daily totals carry the same signal
        daily = pd.Series(y, index=t).resample("1D").sum()
        ax.bar(daily.index, daily.values, width=1.0, color=p["color"])
    else:
        # faint hourly trace (8760 points) under a bold 24 h rolling mean
        ax.plot(t, y, color=p["color"], lw=0.5, alpha=0.25)
        smooth = pd.Series(y, index=t).rolling(24, center=True, min_periods=1).mean()
        ax.plot(t, smooth.values, color=p["color"], lw=1.4)
        if p["key"] != "kp":
            ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)

    if p["key"] == "kp":
        # a planetary index, not an ERA5 field: no domain, and no domain mean
        ax.set_title(f"{p['title']} - planetary geomagnetic activity")
        ax.set_ylim(0, 9)
        ax.axhline(5, color="k", lw=0.8, ls="--", alpha=0.6)   # G1 storm threshold
        ax.text(t[24], 5.15, "Kp 5 = G1 storm", fontsize=8, color="0.25")
    else:
        ax.set_title(f"ERA5 {p['title']} - domain mean")

    ax.set_ylabel(p["ylabel"])
    ax.set_xlabel("Month (2025)")

    # identical time axis on every figure
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    ax.set_xlim(t[0], t[-1])

    banner = ("OMNI2 / NASA GSFC - global index" if p["key"] == "kp" else DOMAIN)
    fig.suptitle(f"{banner}  |  {PERIOD}", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT / f"{p['n']:02d}_{p['key']}_timeseries.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, y


print("Saved one figure per variable:")
for p in plots:
    out, y = make_figure(p)
    print(f"  {p['key']:4s} -> {out}   "
          f"(mean {np.mean(y):.3f}, min {np.min(y):.3f}, max {np.max(y):.3f})")

print(f"\nprecipitation year total: {df['tp'].sum():.1f} mm")
