"""
ERA5 time-series visualisation (December 2025, northern North America domain).

Computes area-weighted (cos-latitude) domain-mean time series for each variable
and saves matplotlib figures to ./output/.

Variables:
  instant file: t2m, d2m (K)  |  u10, v10 (m/s)
  accum   file: tp (m, hourly total precipitation)
"""

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

DATA = Path("382c46a9358c879403b826b3bb39ad87")
INSTANT = DATA / "data_stream-oper_stepType-instant.nc"
ACCUM = DATA / "data_stream-oper_stepType-instant.nc".replace("instant", "accum")
ACCUM = DATA / "data_stream-oper_stepType-accum.nc"
OUT = Path("output")
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
    "figure.autolayout": True,
})


def area_weighted_mean(da):
    """Domain mean weighted by cos(latitude)."""
    w = np.cos(np.deg2rad(da["latitude"]))
    return da.weighted(w).mean(dim=("latitude", "longitude"))


def fmt_time_axis(ax):
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=3))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    for lbl in ax.get_xticklabels():
        lbl.set_rotation(0)


# ---------------------------------------------------------------- load & reduce
ds_i = xr.open_dataset(INSTANT)
ds_a = xr.open_dataset(ACCUM)

t = ds_i["valid_time"].values

t2m = area_weighted_mean(ds_i["t2m"]) - 273.15   # -> deg C
d2m = area_weighted_mean(ds_i["d2m"]) - 273.15   # -> deg C
u10 = area_weighted_mean(ds_i["u10"])
v10 = area_weighted_mean(ds_i["v10"])
wspd = np.sqrt(u10**2 + v10**2)                  # scalar mean wind speed
tp_mm = area_weighted_mean(ds_a["tp"]) * 1000.0  # m -> mm (per hour)

domain = (f"{float(ds_i.latitude.min()):.1f}-{float(ds_i.latitude.max()):.1f}"
          f"N, {float(ds_i.longitude.min()):.1f} to {float(ds_i.longitude.max()):.1f}E")
period = (pd.to_datetime(t[0]).strftime("%d %b %Y")
          + " - " + pd.to_datetime(t[-1]).strftime("%d %b %Y"))
suptitle = f"ERA5 domain-mean ({domain})  |  {period}"

C = {"temp": "#d1495b", "dew": "#2e86ab", "wind": "#3a7d44", "precip": "#5b3a99"}

# ---------------------------------------------------------- 1) combined overview
fig, axes = plt.subplots(4, 1, figsize=(11, 12), sharex=True)

axes[0].plot(t, t2m, color=C["temp"], lw=1.2)
axes[0].plot(t, d2m, color=C["dew"], lw=1.0, alpha=0.8, label="dewpoint")
axes[0].set_ylabel("Temperature (deg C)")
axes[0].set_title("2 m temperature (red) & dewpoint (blue)")
axes[0].axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)

axes[1].plot(t, wspd, color=C["wind"], lw=1.1)
axes[1].set_ylabel("Wind speed (m/s)")
axes[1].set_title("10 m wind speed  sqrt(u^2 + v^2)")

axes[2].bar(t, tp_mm, width=0.04, color=C["precip"], align="center")
axes[2].set_ylabel("Precip (mm/hr)")
axes[2].set_title("Total precipitation (hourly)")

# relative humidity from t2m/d2m (Magnus) as a bonus panel
def rh(temp_c, dew_c):
    a, b = 17.625, 243.04
    return 100 * np.exp((a * dew_c) / (b + dew_c) - (a * temp_c) / (b + temp_c))
axes[3].plot(t, rh(t2m.values, d2m.values), color="#e07a1f", lw=1.1)
axes[3].set_ylabel("RH (%)")
axes[3].set_title("Relative humidity (from t2m & d2m, Magnus formula)")
axes[3].set_ylim(0, 100)

fmt_time_axis(axes[3])
axes[3].set_xlabel("Date (2025)")
fig.suptitle(suptitle, y=1.005, fontsize=13, fontweight="bold")
fig.savefig(OUT / "01_overview_all_variables.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------ 2) temperature & dewpoint solo
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(t, t2m, color=C["temp"], lw=1.3, label="2 m temperature")
ax.plot(t, d2m, color=C["dew"], lw=1.1, label="2 m dewpoint")
ax.fill_between(t, d2m, t2m, color="grey", alpha=0.12)
ax.axhline(0, color="k", lw=0.7, ls="--", alpha=0.6)
ax.set_ylabel("Temperature (deg C)")
ax.set_xlabel("Date (2025)")
ax.set_title("ERA5 2 m temperature & dewpoint - domain mean")
ax.legend(loc="upper right")
fmt_time_axis(ax)
fig.savefig(OUT / "02_temperature_dewpoint.png", bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------ 3) wind speed solo
fig, ax = plt.subplots(figsize=(11, 5))
ax.plot(t, wspd, color=C["wind"], lw=1.2)
ax.set_ylabel("Wind speed (m/s)")
ax.set_xlabel("Date (2025)")
ax.set_title("ERA5 10 m wind speed - domain mean")
fmt_time_axis(ax)
fig.savefig(OUT / "03_wind_speed.png", bbox_inches="tight")
plt.close(fig)

# --------------------------------------------- 4) precipitation hourly + daily
tp_series = pd.Series(tp_mm.values, index=pd.to_datetime(t))
tp_daily = tp_series.resample("1D").sum()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8))
ax1.bar(tp_series.index, tp_series.values, width=0.04, color=C["precip"])
ax1.set_ylabel("mm / hour")
ax1.set_title("Total precipitation - hourly (domain mean)")
fmt_time_axis(ax1)

ax2.bar(tp_daily.index, tp_daily.values, width=0.7, color=C["precip"], alpha=0.85)
ax2.set_ylabel("mm / day")
ax2.set_xlabel("Date (2025)")
ax2.set_title("Total precipitation - daily total (domain mean)")
fmt_time_axis(ax2)
fig.suptitle(suptitle, y=1.005, fontsize=13, fontweight="bold")
fig.savefig(OUT / "04_precipitation.png", bbox_inches="tight")
plt.close(fig)

# ----------------------------------------------------------------- summary stats
print("Saved figures to", OUT.resolve())
print("\n--- Domain-mean summary (Dec 2025) ---")
print(f"2 m temp   : mean {float(t2m.mean()):6.2f}  min {float(t2m.min()):6.2f}  max {float(t2m.max()):6.2f}  deg C")
print(f"2 m dewpt  : mean {float(d2m.mean()):6.2f}  min {float(d2m.min()):6.2f}  max {float(d2m.max()):6.2f}  deg C")
print(f"wind speed : mean {float(wspd.mean()):6.2f}  min {float(wspd.min()):6.2f}  max {float(wspd.max()):6.2f}  m/s")
print(f"precip     : hourly mean {float(tp_mm.mean()):.4f} mm  |  month total {float(tp_series.sum()):.1f} mm")

ds_i.close()
ds_a.close()
