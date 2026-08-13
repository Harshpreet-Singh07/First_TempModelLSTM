"""
Spatial maps of ERA5 2 m temperature, in the style of the Copernicus /
Climate Data Store daily-mean temperature figures.

Every other figure in this project collapses the grid to a single
domain-mean number per hour.  This one does the opposite: it keeps the
full 62 x 84 grid and shows one map, so the spatial structure that the
domain mean averages away is visible.

Domain is 41.75-57.0 N, 74.25-95.0 E (Central Asia) - a regional window,
not the globe, because that is the extent of the source GRIB.

Outputs
  output/figures/08_map_t2m_<date>.png   daily mean for one date
  output/figures/09_map_t2m_annual.png   annual mean + seasonal panels

Run
  python scripts/plot_map.py                    # default 2025-07-01
  python scripts/plot_map.py --date 2025-01-15
"""

import argparse

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from era5_series import GRIB, FIGURES

FIGURES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.titleweight": "bold",
})

# Approximation of the Copernicus 2 m temperature palette:
# deep purple -> blue -> cyan -> green -> yellow -> orange -> dark red,
# spanning -40 .. +40 deg C so figures are comparable across dates.
CDS_COLORS = [
    (0.000, "#3b0f70"), (0.100, "#2b1a9e"), (0.200, "#1f4fd8"),
    (0.300, "#1f9ad8"), (0.400, "#3fd0c9"), (0.500, "#8fe36b"),
    (0.560, "#b8e830"), (0.625, "#e8e020"), (0.750, "#f59a1e"),
    (0.875, "#e03b1e"), (1.000, "#7f0d0d"),
]
CDS_CMAP = LinearSegmentedColormap.from_list("cds_t2m", CDS_COLORS)
VMIN, VMAX = -40, 40

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAVE_CARTOPY = True
except ImportError:
    HAVE_CARTOPY = False


def open_t2m():
    ds = xr.open_dataset(GRIB, engine="cfgrib", backend_kwargs={
        "filter_by_keys": {"stepType": "instant"}, "indexpath": ""})
    return ds


def basemap(ax, lon, lat):
    """Coastlines, borders and lakes if cartopy has its data; graticule always."""
    if HAVE_CARTOPY:
        try:
            ax.add_feature(cfeature.COASTLINE.with_scale("50m"), lw=0.7, edgecolor="0.15")
            ax.add_feature(cfeature.BORDERS.with_scale("50m"), lw=0.5, edgecolor="0.3")
            ax.add_feature(cfeature.LAKES.with_scale("50m"), lw=0.4,
                           facecolor="none", edgecolor="0.2")
        except Exception as exc:            # no cached Natural Earth data, offline
            print(f"  (map features unavailable: {type(exc).__name__}; drawing bare grid)")
        gl = ax.gridlines(draw_labels=True, lw=0.4, color="0.5", alpha=0.5, ls=":")
        gl.top_labels = gl.right_labels = False
    else:
        ax.set_xlabel("Longitude (deg E)")
        ax.set_ylabel("Latitude (deg N)")
        ax.grid(lw=0.4, color="0.5", alpha=0.5, ls=":")


def make_ax(fig, *spec):
    """add_subplot with the cartopy projection attached when available."""
    if HAVE_CARTOPY:
        ax = fig.add_subplot(*spec, projection=ccrs.PlateCarree())
        ax.set_extent(EXTENT, crs=ccrs.PlateCarree())
        return ax
    return fig.add_subplot(*spec)


def draw(ax, lon, lat, field, title):
    kw = {"cmap": CDS_CMAP, "vmin": VMIN, "vmax": VMAX, "shading": "auto"}
    if HAVE_CARTOPY:
        kw["transform"] = ccrs.PlateCarree()
    im = ax.pcolormesh(lon, lat, field, **kw)
    basemap(ax, lon, lat)
    # An explicit y is required: cartopy's gridliner drives the auto title
    # position to inf when draw_labels=True, which throws the title off-canvas.
    ax.set_title(title, y=1.03)
    return im


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--date", default="2025-07-01", help="date for the daily-mean map")
    a = p.parse_args()

    ds = open_t2m()
    lat = ds["latitude"].values
    lon = ds["longitude"].values
    t2m = ds["t2m"]
    times = pd.DatetimeIndex(ds["valid_time"].values)

    global EXTENT
    EXTENT = [lon.min(), lon.max(), lat.min(), lat.max()]
    print(f"grid {lat.size} x {lon.size}   domain {EXTENT}")
    print(f"cartopy: {'yes' if HAVE_CARTOPY else 'no (plain lat/lon axes)'}")

    # ---------------------------------------------------- 1) one day, daily mean
    day = pd.Timestamp(a.date)
    sel = np.flatnonzero((times >= day) & (times < day + pd.Timedelta(days=1)))
    if sel.size == 0:
        raise SystemExit(f"{a.date} is outside the file ({times[0]} .. {times[-1]})")
    field = t2m.values[sel].mean(axis=0) - 273.15

    fig = plt.figure(figsize=(9, 6.4))
    ax = make_ax(fig, 1, 1, 1)
    im = draw(ax, lon, lat, field,
              f"ERA5-Single-levels, daily mean 2 metre temperature\n{day:%Y-%m-%d} (UTC)")
    cb = fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.06,
                      shrink=0.85, extend="both")
    cb.set_label("2 metre temperature (deg C)")
    cb.set_ticks(np.arange(VMIN, VMAX + 1, 10))
    out = FIGURES / f"08_map_t2m_{day:%Y%m%d}.png"
    # NB: no bbox_inches="tight" here - it collapses the figure to the colorbar
    # when cartopy's gridliner has draw_labels=True.
    fig.savefig(out)
    plt.close(fig)
    print(f"  {day:%Y-%m-%d}: min {field.min():6.2f}  mean {field.mean():6.2f}  "
          f"max {field.max():6.2f} deg C  -> {out.name}")

    # ------------------------------------------- 2) annual mean + seasonal panels
    seasons = {"DJF": [12, 1, 2], "MAM": [3, 4, 5], "JJA": [6, 7, 8], "SON": [9, 10, 11]}
    fig = plt.figure(figsize=(13, 9))
    annual = t2m.values.mean(axis=0) - 273.15

    ax = make_ax(fig, 2, 3, 1)
    im = draw(ax, lon, lat, annual, "Annual mean 2025")
    print(f"  annual   : min {annual.min():6.2f}  mean {annual.mean():6.2f}  "
          f"max {annual.max():6.2f} deg C")

    for pos, (name, months) in zip([3, 4, 5, 6], seasons.items()):
        m = np.isin(times.month, months)
        f = t2m.values[m].mean(axis=0) - 273.15
        ax = make_ax(fig, 2, 3, pos)
        draw(ax, lon, lat, f, f"{name} mean")
        print(f"  {name}      : min {f.min():6.2f}  mean {f.mean():6.2f}  "
              f"max {f.max():6.2f} deg C")

    cb = fig.colorbar(im, ax=fig.axes, orientation="horizontal", pad=0.05,
                      shrink=0.5, extend="both")
    cb.set_label("2 metre temperature (deg C)")
    cb.set_ticks(np.arange(VMIN, VMAX + 1, 10))
    fig.suptitle("ERA5 2 m temperature - annual and seasonal means, 2025\n"
                 f"{lat.min():.2f}-{lat.max():.2f}N, {lon.min():.2f}-{lon.max():.2f}E",
                 fontsize=13, fontweight="bold")
    out = FIGURES / "09_map_t2m_annual.png"
    fig.savefig(out)
    plt.close(fig)
    print(f"  -> {out.name}")

    ds.close()


if __name__ == "__main__":
    main()
