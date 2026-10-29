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
Enter a first day (YYYY-MM-DD, from 2025-01-04 onward) and a length of 1-7 days;
the app plots the LSTM forecast against ERA5 and scores it versus persistence.
Stop it with Ctrl+C.

### Updating the model or the observed data

The app reads files from disk once, at startup, so after any change below
**restart `python app.py`** (or push to the Space, which restarts itself).

**New model weights** (needs `pip install tensorflow`):

Give each model a name with `--name` and the previous one is kept:

```bash
python lstm/train_lstm.py --name big128 --units 128 --epochs 100
python lstm/lstm_numpy.py big128   # exports the .npz the app actually reads
```

This writes `lstm/models/lstm_big128.keras`, `scaler_big128.json` and
`lstm_big128_weights.npz` (plus plots in `lstm/output/big128/`). The Gradio app
lists every model in `lstm/models/` in its **Model** dropdown, so there is nothing
to point or edit - restart the app and pick the new name. Omitting `--name` uses
the name `t2m`, which **overwrites** the original model.

No code change is needed for the same architecture (one LSTM layer + Dense(1),
`--horizon 1`); anything else needs `lstm/lstm_numpy.py` updated too. The Streamlit
app always uses `t2m`.

**New observed ERA5 data** (needs `pip install xarray cfgrib eccodes`):

1. Download hourly `2m_temperature` from the Copernicus CDS with
   `area = [57, 74.25, 41.75, 95]` (N, W, S, E - **east** longitudes, same domain as 2025).
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
| data | `data/processed/domain_mean_hourly_2026.csv` |

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
