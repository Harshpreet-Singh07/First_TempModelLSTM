# Visualizing ERA5

Analysis and visualisation of ERA5 reanalysis data for **calendar year 2025**
over a Central Asian domain (41.75-57.0 N, 74.25-95.0 E), plus setup for
generative modelling with NVlabs/LSGM.

## Layout

```
data/
  raw/        185bc0671d5cc50980a767e3a9f6cbe5.grib   source GRIB (440 MB)
  processed/  domain_mean_hourly_2025.csv             cached domain-mean series
scripts/
  era5_series.py            GRIB -> cos-lat weighted domain means (+ shared style)
  plot_individual_2025.py   one figure per variable
  plot_correlation.py       Pearson + Spearman correlation matrices
  stats_2025.py             descriptive statistics for every variable
  era5_2026.py              same reduction for a 2026 t2m GRIB (for lstm/)
  legacy/                   December-2025 North America scripts (see note below)
output/
  figures/    01-05 per-variable time series, 06-07 correlation matrices
  tables/     summary statistics, correlation matrices (CSV)
lsgm/         LSGM training setup - see lsgm/README.md
lstm/         Keras LSTM t2m forecaster + Streamlit UI - see lstm/README.md
```

## Running

All scripts anchor their paths to the project root, so they work from anywhere:

```bash
python scripts/stats_2025.py
python scripts/plot_individual_2025.py
python scripts/plot_correlation.py
```

The first run reads the 440 MB GRIB and caches the domain-mean series to
`data/processed/`; later runs read the cache. Force a rebuild with:

```bash
python scripts/era5_series.py
```

Requires `numpy pandas xarray matplotlib cfgrib eccodes`.

## The data

| | |
|---|---|
| source | ERA5, GRIB edition 1, ECMWF |
| period | 2025-01-01 00:00 - 2025-12-31 23:00 (8,760 hourly steps) |
| domain | 41.75-57.0 N, 74.25-95.0 E, 62 x 84 grid |
| variables | `t2m`, `d2m` (K), `u10`, `v10` (m/s), `tp` (m, hourly accumulation) |

Every figure and statistic is computed on the **cos(latitude)-weighted domain
mean** - one value per hour, averaged over all 5,208 grid cells. Statistics
therefore describe the domain average, not the spread across the map: individual
grid cells reach well beyond the domain-mean minima and maxima.

`tp` is stored as `(time, step)` forecast blocks. Each field is already a 1-hour
accumulation ending at its `valid_time`, so the blocks are flattened, not
differenced - the values are not cumulative along `step`.

## Note on `scripts/legacy/`

These four scripts target an earlier December-2025 dataset over northern North
America (NetCDF, 165 x 354 grid). **That data is no longer present in the
project**, so the scripts will not run as-is. They are kept for reference only.
