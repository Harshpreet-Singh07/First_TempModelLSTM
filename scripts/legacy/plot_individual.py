"""
ERA5 individual time-series graphs (December 2025, northern North America).

One figure per variable, all in an identical format:
  t2m  - 2 m temperature
  u10  - 10 m U wind component
  v10  - 10 m V wind component
  tp   - total precipitation
  d2m  - 2 m dewpoint temperature

Each series is an area-weighted (cos-latitude) domain mean.
Figures are saved to ./output/individual/.
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA = Path("382c46a9358c879403b826b3bb39ad87")
INSTANT = DATA / "data_stream-oper_stepType-instant.nc"
ACCUM = DATA / "data_stream-oper_stepType-accum.nc"
OUT = Path("output") / "individual"
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


def area_weighted_mean(da):
    """Domain mean weighted by cos(latitude)."""
    w = np.cos(np.deg2rad(da["latitude"]))
    return da.weighted(w).mean(dim=("latitude", "longitude"))


# ------------------------------------------------------------------- load data
ds_i = xr.open_dataset(INSTANT)
ds_a = xr.open_dataset(ACCUM)
t = ds_i["valid_time"].values

period = (pd.to_datetime(t[0]).strftime("%d %b %Y")
          + " - " + pd.to_datetime(t[-1]).strftime("%d %b %Y"))
domain = (f"{float(ds_i.latitude.min()):.1f}-{float(ds_i.latitude.max()):.1f}N, "
          f"{float(ds_i.longitude.min()):.1f} to {float(ds_i.longitude.max()):.1f}E")

# Each entry: series (domain mean), title, y-axis label, colour
plots = [
    dict(key="t2m",
         data=area_weighted_mean(ds_i["t2m"]) - 273.15,
         title="2 m temperature", ylabel="Temperature (deg C)", color="#d1495b"),
    dict(key="u10",
         data=area_weighted_mean(ds_i["u10"]),
         title="10 m U wind component", ylabel="U wind (m/s)", color="#3a7d44"),
    dict(key="v10",
         data=area_weighted_mean(ds_i["v10"]),
         title="10 m V wind component", ylabel="V wind (m/s)", color="#2a6f97"),
    dict(key="tp",
         data=area_weighted_mean(ds_a["tp"]) * 1000.0,
         title="Total precipitation", ylabel="Precipitation (mm/hr)", color="#5b3a99"),
    dict(key="d2m",
         data=area_weighted_mean(ds_i["d2m"]) - 273.15,
         title="2 m dewpoint temperature", ylabel="Dewpoint (deg C)", color="#2e86ab"),
]

# ---------------------------------------------------------- identical formatting
def make_figure(p):
    y = np.asarray(p["data"].values, dtype=float)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(t, y, color=p["color"], lw=1.2)
    ax.set_title(f"ERA5 {p['title']} - domain mean")
    ax.set_ylabel(p["ylabel"])
    ax.set_xlabel("Date (2025)")

    # reference line at zero for variables that cross it
    if p["key"] in ("t2m", "d2m", "u10", "v10"):
        ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.5)

    # identical time axis on every figure
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    ax.set_xlim(t[0], t[-1])

    fig.suptitle(f"{domain}  |  {period}", fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = OUT / f"{p['key']}_timeseries.png"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    return out, y


print("Saved individual figures:")
for p in plots:
    out, y = make_figure(p)
    print(f"  {p['key']:4s} -> {out}   (mean {np.mean(y):.3f}, min {np.min(y):.3f}, max {np.max(y):.3f})")

ds_i.close()
ds_a.close()
