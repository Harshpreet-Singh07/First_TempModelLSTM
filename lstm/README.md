# LSTM temperature model

A small Keras/TensorFlow LSTM that forecasts the ERA5 domain-mean 2 m
temperature (`t2m`) an hour at a time. Self-contained: it reads the cached CSVs
in `data/processed/`, so it needs no GRIB, `xarray` or `cfgrib`.

Two versions ship side by side, and both apps let you pick:

| | V2 - `t2m` | V3 - `v3` |
|---|---|---|
| data | 2025 only | 2006-2025 (20 years) |
| inputs | temperature | temperature, its last-hour change, dewpoint, 10 m wind, wind-driven temperature advection, time of day/year |
| predicts | change in temperature | change in every input variable (so it can run on its own output) |
| split | first 70 / 15 / 15 % of 2025's hours | whole years 70 / 15 / 15: train 2006-2019, val 2020-2022, test 2023-2025 |
| trained | locally | on Google Colab, `colab_train.ipynb` |

The V2 code is kept on the `LSTM_Model_V2` branch. The V3 model files appear
in `models/` once the Colab run's zip is unzipped into the repo; until then the
apps run V2 only.

```
lstm/
  features.py      the model inputs, built the same way for training and forecasting
  train_lstm.py    build windows, train, evaluate, export, save everything
  lstm_numpy.py    export the weights + run the forward pass without TensorFlow
  forecast.py      inference: seed a window, roll the model forward a day
  app.py           Streamlit UI - the record, a day forecast, the actuals
  colab_train.ipynb  download 20 years + train V3 on Google Colab
  models/          lstm_<name>.keras, lstm_<name>_weights.npz, scaler_<name>.json
  output/          V2 figures + metrics.csv; output/<name>/ for every other model
```

**TensorFlow trains the model; it does not run it.** `forecast.py` prefers the
exported numpy weights and only falls back to Keras if the `.npz` is missing, so
the app (and the deployment) never imports TensorFlow. `train_lstm.py`
re-exports at the end of every run and checks the numpy forward pass against
Keras; `python lstm/lstm_numpy.py <name>` does the same by hand.

## Running

```bash
python lstm/train_lstm.py                          # V3 on every CSV present, name v3
python lstm/train_lstm.py --years 2006-2025        # leave out a partial 2026
python lstm/train_lstm.py --resume --epochs 30     # continue lstm_v3.keras
python lstm/train_lstm.py --name uni --features t2m        # temperature only
python lstm/train_lstm.py --name h24 --horizon 24          # 24 h ahead
```

Flags: `--name --features --years --val-start --test-start --resume --lookback
--horizon --units --input-dropout --dropout --lr --epochs --batch --patience
--no-delta`. A run only writes files under its own `--name`.

Requires `tensorflow` plus the `numpy pandas matplotlib` the rest of the project
already uses. Training data is `data/processed/domain_mean_hourly_<year>.csv`;
build the 20-year record with `python scripts/era5_years.py` (or on Colab). With
only the 2025 CSV present, the split falls back to V2's 70 / 15 / 15 of the hours.

## V3: more data, more inputs

**20 years of data.** `scripts/era5_years.py` downloads 2006-2025 from the CDS,
one request per year, and reduces each to a yearly CSV. The split moves to whole
calendar years (70 / 15 / 15 of the years), so every split holds every season;
V2's test set was November-December only. The split is saved in the scaler JSON,
and `--resume` reuses it, so continuing training in a new Colab session cannot
train on the test years. `metrics.csv` also scores V3 on V2's exact test window
(7 Nov - 31 Dec 2025), so the two versions compare like for like.

**Wind moves heat.** When the wind blows from warm air towards cold, the place
downwind warms. A domain mean cannot see that by itself, so `t2m_adv`
(`-(u dT/dx + v dT/dy)`, deg C per hour) is computed on the full grid when the
GRIB is reduced, then averaged like every other variable. It goes in alongside
`u10`, `v10` and dewpoint `d2m`.

**Deltas in and out.** V2 already predicted the change from the last hour rather
than the temperature itself: "no change" is the most common answer an hour
ahead, and an output of zero means exactly that. V3 keeps that and also feeds
the last hour's change in (`t2m_d1`). It predicts the change of *every* input
variable, not just temperature, because a 24-168 h forecast runs on its own
output and needs next hour's wind as much as next hour's temperature. Each
output is scaled by the spread of its own changes so they weigh equally in the loss.

**Input dropout.** In training, each input value - one feature at one hour - is
zeroed independently with probability `--input-dropout` (0.1), so every feature
is sometimes missing but no feature is ever removed. The alternatives that look
similar, `SpatialDropout1D` and `LSTM(dropout=...)`, reuse one mask across the
whole window and drop an entire feature for that sample. Separately, training
now stops with an error if any year is missing an input column. Before, those
hours would have quietly produced no windows, so the model would train as if
that feature, or that year, had never existed.

**Colab.** `colab_train.ipynb` clones the repo, keeps data and checkpoints in
Google Drive, downloads and reduces the years (skipping finished ones), trains
with a checkpoint every improved epoch, and downloads a zip of the model and data
to unzip into this folder.

## What V2 does

| | |
|---|---|
| input | last `lookback` hourly temperatures (default 72), one feature |
| output | temperature `horizon` hours ahead (default 1) |
| target | the **change** from the last input hour, added back on afterwards |
| split | chronological 70 / 15 / 15 - train Jan-Sep, val Sep-Nov, test Nov-Dec |
| scaling | standardised with the **training** mean/std only |
| network | `LSTM(64)` -> `Dropout(0.1)` -> `Dense(1)`, Adam, MSE loss |
| stopping | early stopping on val loss (patience 8, best weights restored) |

Each split is windowed independently, so no training hour ever appears inside a
validation or test window. Windows are built as a strided index view rather than
copied, which keeps the whole dataset in a few MB.

Predicting the change rather than the absolute value matters more than the
architecture here. Asked for the absolute temperature (`--no-delta`), the same
network scores MAE 0.741 deg C at 1 h - 69% *worse* than persistence, because
16k parameters have to reproduce the temperature level from scratch. Predicting
the departure from the last observed hour drops that to 0.474 deg C.

## Reading the numbers

`output/metrics.csv` reports MAE, RMSE, bias and R^2 in **deg C** for the LSTM
and for a **persistence** baseline (`prediction = last observed hour`), on the
held-out Nov-Dec test split:

| horizon | model | MAE | RMSE | R^2 |
|---|---|---|---|---|
| 1 h | LSTM | 0.474 | **0.724** | **0.970** |
| 1 h | persistence | **0.439** | 0.826 | 0.961 |
| 24 h | LSTM | 1.988 | 2.413 | 0.667 |
| 24 h | persistence | **1.904** | **2.358** | **0.682** |

So: this model **does not beat persistence**, it ties it. At 1 h it wins on RMSE
and R^2 (it clips the large errors persistence makes during fast swings) and
loses slightly on MAE, which is what training on MSE buys you. At 24 h it is
marginally behind on everything, and the prediction figure shows why - it has
learned to reproduce the diurnal cycle with a lag rather than anticipate it.

That is the expected result, not a bug. A cos-latitude domain mean over 5,208
grid cells is an extremely smooth, strongly autocorrelated series, and both
1 h and 24 h persistence are hard baselines on it. Beating them needs
information the model does not have: calendar features (hour-of-day, day-of-year
as sin/cos pairs), the other ERA5 variables (`d2m`, `u10`, `v10`, `tp`), or more
than one year of training data. Those are the next steps, not bigger layers.

The test split is Nov-Dec only - early winter, one domain, one year. It measures
this model on this series, not general forecast skill.

## Reusing the trained model

```python
import json, numpy as np, keras

model  = keras.models.load_model("lstm/models/lstm_t2m.keras")
scaler = json.load(open("lstm/models/scaler_t2m.json"))

# last `lookback` hourly temperatures in deg C, oldest first
z = (np.asarray(recent_c) - scaler["mean_c"]) / scaler["std_c"]
out = model.predict(z.reshape(1, scaler["lookback"], 1))[0, 0]

if scaler["delta"]:          # model returned a change, not a temperature
    out += z[-1]
pred_c = out * scaler["std_c"] + scaler["mean_c"]
```

The scaler JSON records the lookback/horizon, the `delta` flag and the split
dates the model was trained with. Both steps above are silent failures if you
skip them - wrong scaling or a forgotten `+= z[-1]` produces plausible-looking
numbers that are simply wrong - so always load the JSON alongside the weights.

`forecast.py` already does all of this:

```bash
python lstm/forecast.py 2026-03-14      # 24 hourly steps, printed with actuals
```

## The UI

```bash
streamlit run lstm/app.py
```

Pick any day the record covers - all of 2025, plus 2026 once that CSV exists -
and a forecast length of 1 to 7 days. You get the full observed record with the
forecast window marked, then the window itself as **LSTM forecast vs ERA5
observed vs persistence**, with MAE / RMSE / bias underneath and the hourly
table downloadable as CSV. There is no run button: changing the date or the
length recomputes immediately.

Forecasting 2025 days is the useful demo, because every hour has an observation
to score against. A note on what the persistence baseline means here: it holds
the 00:00 value flat for the whole window, so over 24 hours it is a much weaker
opponent than the 1-hour-ahead persistence in the metrics table above. On
1 July 2025 the LSTM scores MAE 0.63 deg C against persistence's 4.03, because
the model reproduces the diurnal cycle and a flat line cannot. Both comparisons
are fair; they just answer different questions.

The forecast is issued at **00:00 of the chosen day** and rolled out 24 hours,
one hour per step, each step fed the previous prediction. Two modes, and the app
says which one it used:

- **warm start** - the 72 hours before that day are observed, so the model
  starts from reality. A genuine 24 h forecast; this is the mode worth scoring.
- **free running** - those hours are not observed, so the model is seeded at the
  last real observation and fed its own output until it reaches the day. Error
  compounds every step. Capped at 60 days, and flagged in red, because past a
  few days it shows what the model decays towards, not what the weather did.

With no 2026 file present, *every* day is free-running from 31 Dec 2025 - so
1 January is informative and 1 June is not.

## Deploying to Streamlit Community Cloud

The repo is already prepared: `requirements.txt` at the root, the 440 MB GRIB
gitignored, and the two files the app actually loads (the 2025 CSV and
`lstm/models/`) committed - so a clone runs without any raw data.

Note the Python version: Community Cloud currently builds on **3.14**, which has
no TensorFlow wheels. That is why the app runs on exported numpy weights - with
`tensorflow-cpu` in `requirements.txt` the build fails at dependency resolution,
whatever Python version you pin.

1. Create an empty repo at <https://github.com/new>. **Public** - the free tier
   cannot deploy from a private repo. Everything committed becomes visible:
   the ERA5 series, the figures, the trained model. ERA5 is Copernicus-licensed
   and free to redistribute with attribution, so this is fine, but it is a
   deliberate choice rather than an accident.

2. Push:

   ```bash
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```

3. At <https://share.streamlit.io> → **New app** → pick the repo, branch `main`,
   and set **Main file path** to `lstm/app.py`. Under *Advanced*, choose
   **Python 3.12** (TensorFlow does not publish wheels for every version).
   Deploy.

The first build takes several minutes - `tensorflow-cpu` is a large wheel.

Two things to expect:

- **The 2026 data is not there until you push it.** `data/processed/` is
  committed, so once `era5_2026.py` has written the 2026 CSV, `git add` and push
  it or the hosted app will free-run exactly like a fresh clone.
- **The weights must be committed, not just the .keras file.** The app loads
  `lstm/models/lstm_t2m_weights.npz` (68 KB). If you retrain and forget to
  re-export, the deployment silently keeps serving the old network.

## 2026 observations

The app scores forecasts against `data/processed/domain_mean_hourly_2026.csv`,
which does not ship with the repo. To build it, download 2026 `2m_temperature`
from the Copernicus CDS **using the same domain as 2025**:

```python
"area": [57, 74.25, 41.75, 95]        # [North, West, South, East], EAST longitudes
```

Put the GRIB in `data/raw/` and run:

```bash
python scripts/era5_2026.py            # picks up the newest GRIB in data/raw/
```

That script refuses files from a different domain rather than averaging them:
writing the area with negative longitudes (`[57, -95, 41.75, -74.25]`) silently
returns northern North America, which shares nothing with the series this model
was trained on. ERA5 runs about five days behind real time, so the current month
comes back partial - the loader handles that and the app just stops at the last
observed hour.
