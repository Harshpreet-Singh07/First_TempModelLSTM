# Visualizing ERA5

Analysis and visualisation of ERA5 reanalysis data for **calendar year 2025**
over a Central Asian domain (41.75-57.0 N, 74.25-95.0 E), plus setup for
generative modelling with NVlabs/LSGM.

## Layout

```
data/
  raw/        185bc0671d5cc50980a767e3a9f6cbe5.grib   source GRIB (440 MB)
  processed/  domain_mean_hourly_<year>.csv           cached domain-mean series, one per year
scripts/
  era5_series.py            GRIB -> cos-lat weighted domain means (+ shared style)
  plot_individual_2025.py   one figure per variable
  plot_correlation.py       Pearson + Spearman correlation matrices
  stats_2025.py             descriptive statistics for every variable
  era5_years.py             download + reduce many years from the CDS (V3 training data)
  era5_2026.py              same reduction for a 2026 GRIB (for lstm/)
  legacy/                   December-2025 North America scripts (see note below)
output/
  figures/    01-05 per-variable time series, 06-07 correlation matrices
  tables/     summary statistics, correlation matrices (CSV)
lsgm/         LSGM training setup - see lsgm/README.md
lstm/         Keras LSTM t2m forecaster, Colab notebook, Streamlit UI - see lstm/README.md
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

## Try the LSTM forecaster (Gradio)

Needs Python 3.10+ and nothing else: the trained weights and cached CSVs are in
the repo, and the app does not use TensorFlow or the GRIB files.

```bash
git clone <repo-url>
cd <repo-folder>
pip install gradio pandas numpy plotly
python app.py
```

Open http://127.0.0.1:7860 in a browser (it runs on your own machine, not online).
Enter a first day (YYYY-MM-DD, from 2025-01-04 onward, or from 2006 once the
20-year record is in `data/processed/`) and a length of 1-7 days; the app plots
the LSTM forecast against ERA5 and scores it versus persistence. The **Model**
dropdown switches between `t2m` (V2, temperature only) and `v3` once trained.
Stop it with Ctrl+C.

### Updating the model or the observed data

The app reads files from disk once, at startup, so after any change below
**restart `python app.py`** (or push to the Space, which restarts itself).

**New model weights** (needs `pip install tensorflow`):

The V3 model trains on 20 years of data, which is easiest on Google Colab: open
[`lstm/colab_train.ipynb`](lstm/colab_train.ipynb) in Colab and run it top to
bottom. It downloads the data, trains, and hands back a zip to unzip into this
folder. See [lstm/README.md](lstm/README.md#v3-more-data-more-inputs) for what V3 changes.

Locally, give each model a name with `--name` and the previous one is kept:

```bash
python lstm/train_lstm.py --name big128 --units 128 --epochs 100
```

This writes `lstm/models/lstm_big128.keras`, `scaler_big128.json` and
`lstm_big128_weights.npz` (plus plots in `lstm/output/big128/`); the `.npz` the
apps read is exported and checked against Keras at the end of every run. Both
apps list every model in `lstm/models/` in a **Model** dropdown, so there is
nothing to point or edit - restart the app and pick the new name. The default
name is `v3`; `t2m` is the V2 model, and training under that name **overwrites** it.

No code change is needed for the same architecture (one LSTM layer + Dense,
`--horizon 1`); anything else needs `lstm/lstm_numpy.py` updated too.

**New observed ERA5 data** (needs `pip install cdsapi xarray cfgrib eccodes`
and a CDS API key in `~/.cdsapirc`):

```bash
python scripts/era5_years.py --years 2026          # complete months so far
python scripts/era5_years.py --years 2006-2025     # the V3 training record
```

That downloads each year from the Copernicus CDS and writes
`data/processed/domain_mean_hourly_<year>.csv`. To use a GRIB you downloaded
yourself instead:

1. Download hourly `2m_temperature` - plus `2m_dewpoint_temperature`,
   `10m_u_component_of_wind` and `10m_v_component_of_wind` for the V3 model -
   from the Copernicus CDS with `area = [57, 74.25, 41.75, 95]`
   (N, W, S, E - **east** longitudes, same domain as 2025).
2. Put the GRIB in `data/raw/` (it is gitignored).
3. Run `python scripts/era5_2026.py` (or pass the GRIB path as an argument).
   It writes `data/processed/domain_mean_hourly_2026.csv` and refuses a file with
   the wrong domain. Rebuilding replaces the whole file and keeps only 2026 rows, so download the
   full year-to-date, not just the new days.

The app picks the CSV up automatically: new days become forecastable and are
scored against the observations.

**Deploying to the Hugging Face Space:** copy whichever files changed into the
Space repo, keeping the same folder layout, then commit and push:

| Changed | Copy |
|---|---|
| model | `lstm/models/lstm_<name>_weights.npz`, `lstm/models/scaler_<name>.json`, and optionally `lstm/output/<name>/metrics.csv` (`lstm/output/metrics.csv` for `t2m`) for the test-set accuracy table |
| data | `data/processed/domain_mean_hourly_<year>.csv` for each year added |

```bash
git add . && git commit -m "Update model/data" && git push
```

A Streamlit version of the same UI: `pip install -r requirements.txt` then
`streamlit run lstm/app.py`.

## The data

| | |
|---|---|
| source | ERA5, GRIB edition 1, ECMWF |
| period | 2025-01-01 00:00 - 2025-12-31 23:00 (8,760 hourly steps) |
| domain | 41.75-57.0 N, 74.25-95.0 E, 62 x 84 grid |
| variables | `t2m`, `d2m` (K), `u10`, `v10` (m/s), `tp` (m, hourly accumulation) |
| derived | `t2m_adv` - temperature advection by the 10 m wind, deg C per hour |

Every figure and statistic is computed on the **cos(latitude)-weighted domain
mean** - one value per hour, averaged over all 5,208 grid cells. Statistics
therefore describe the domain average, not the spread across the map: individual
grid cells reach well beyond the domain-mean minima and maxima.

`t2m_adv` is `-(u dT/dx + v dT/dy)` computed on every grid cell, then averaged
the same way: the rate at which the wind carries warmer or colder air in. It is
the one input that needs the full grid, which is why it is computed when the
GRIB is reduced rather than from the domain means. Its 2025 mean is negative
(-0.15 deg C/h), probably in part because 2 m temperature gradients include
terrain height, so air crossing the domain's mountains reads as "cold
advection". The model learns from how it varies, not its level.

`tp` is stored as `(time, step)` forecast blocks. Each field is already a 1-hour
accumulation ending at its `valid_time`, so the blocks are flattened, not
differenced - the values are not cumulative along `step`.

## Note on `scripts/legacy/`

These four scripts target an earlier December-2025 dataset over northern North
America (NetCDF, 165 x 354 grid). **That data is no longer present in the
project**, so the scripts will not run as-is. They are kept for reference only.
