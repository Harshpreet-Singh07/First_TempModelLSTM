# LSTM temperature model

A small Keras/TensorFlow LSTM that forecasts the ERA5 domain-mean 2 m
temperature (`t2m`) from its own recent history. Self-contained: it reads the
cached CSV in `data/processed/`, so it needs no GRIB, `xarray` or `cfgrib`.

```
lstm/
  train_lstm.py    build windows, train, evaluate, save everything
  forecast.py      inference: seed a window, roll the model forward a day
  app.py           Streamlit UI - the record, a day forecast, the actuals
  models/          lstm_t2m.keras + scaler_t2m.json   (written by the script)
  output/          loss curve, prediction figure, metrics.csv
```

## Running

```bash
python lstm/train_lstm.py                 # 72 h in -> 1 h ahead, ~3 min on CPU
python lstm/train_lstm.py --horizon 24    # 24 h ahead
python lstm/train_lstm.py --units 128 --epochs 100 --lookback 168
python lstm/train_lstm.py --no-delta      # absolute-temperature target
```

Flags: `--lookback --horizon --units --dropout --lr --epochs --batch --patience
--no-delta`. Every run overwrites `models/` and `output/` - the files currently
there are from the `--horizon 24` run.

Requires `tensorflow` (installed) plus the `numpy pandas matplotlib` the rest of
the project already uses. If `data/processed/domain_mean_hourly_2025.csv` is
missing, build it once with `python scripts/era5_series.py`.

## What it does

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

A sidebar date picker for any day in 2026, the full observed record with that
day marked, and the day itself as **LSTM forecast vs ERA5 observed vs
persistence**, with MAE / RMSE / bias underneath. The hourly table is
downloadable as CSV.

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
