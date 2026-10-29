"""
Inference helpers for the trained LSTMs - shared by both apps and usable alone.

Two ways to forecast a day:

  warm start   the hours immediately before that day are observed, so the
               model is seeded with reality and rolled forward. This is a real
               forecast issued at 00:00 and is the only mode worth scoring.

  free running the seed is not available (no observations near that day), so
               the model is seeded with the last observed hours and fed its
               own output for as many hours as it takes to reach the target.
               Errors compound every step; past a few days this says nothing
               about the weather, only about what the model decays towards.
               Capped at FREE_RUN_CAP_DAYS.

Models predict the *change* per hour (see train_lstm.py --no-delta) of every
variable they read, so a rollout step is: predict the changes -> add them to the
last hour -> rebuild the inputs (lstm/features.py) -> shift the window. The V2
model (t2m) reads only temperature; V3 models also read humidity and wind, and
predict those too, since they are not known in advance.

Standalone use:
    python lstm/forecast.py 2025-07-01 [days] [model]
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
MODELS_DIR = HERE / "models"
PROCESSED = ROOT / "data" / "processed"

sys.path.insert(0, str(HERE))            # lstm_numpy.py, features.py sit next to this file
import features as feat                                  # noqa: E402
from lstm_numpy import paths as model_files   # noqa: E402  (keras, npz, scaler json)


def list_models():
    """Names of every model that has a scaler plus weights (npz or keras)."""
    names = []
    for scaler in sorted(MODELS_DIR.glob("scaler_*.json")):
        name = scaler.stem.removeprefix("scaler_")
        keras_p, npz_p, _ = model_files(name)
        if npz_p.exists() or keras_p.exists():
            names.append(name)
    return names


CSV_2025 = PROCESSED / "domain_mean_hourly_2025.csv"
CSV_2026 = PROCESSED / "domain_mean_hourly_2026.csv"

HOURS_PER_DAY = 24
FREE_RUN_CAP_DAYS = 60
GAP_FILL_HOURS = 6        # interpolate holes up to this long; longer ones stay missing


# ---------------------------------------------------------------------- data
def load_history():
    """Observed hourly domain means, every year in data/processed/.

    One row per hour from the first to the last observation. Short gaps are
    interpolated; anything longer, and any column a year's CSV does not have
    (a t2m-only 2026 file has no wind), stays NaN - forecast_day() then refuses
    or free-runs past those hours rather than inventing them.
    """
    paths = sorted(PROCESSED.glob("domain_mean_hourly_*.csv"))
    if not paths:
        raise FileNotFoundError(
            f"no domain_mean_hourly_<year>.csv in {PROCESSED} - build it with: "
            f"python scripts/era5_series.py")

    df = pd.concat(pd.read_csv(p, index_col="valid_time", parse_dates=True) for p in paths)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="h")).interpolate(
        limit=GAP_FILL_HOURS, limit_area="inside")
    df.index.name = "valid_time"

    obs_2026 = df["t2m"][df.index.year == 2026].dropna()
    return df, {
        "has_2026": len(obs_2026) > 0,
        "last_2026": obs_2026.index[-1] if len(obs_2026) else None,
        "n_2026": len(obs_2026),
        "years": sorted(df["t2m"].dropna().index.year.unique()),
    }


# --------------------------------------------------------------------- model
def _v2_scaler(sc):
    """A V2 (temperature-only) scaler in the V3 layout, so one code path runs both."""
    delta = sc.get("delta", False)
    return dict(sc, features=["t2m"], outputs=["t2m"],
                mean=[sc["mean_c"]], std=[sc["std_c"]],
                target_mean=[0.0 if delta else sc["mean_c"]],
                target_scale=[sc["std_c"]], delta=delta)


class Forecaster:
    """The trained network plus the inputs and scaling it was trained with.

    Prefers the exported numpy weights and only falls back to Keras if they are
    missing, so the deployed app never imports TensorFlow - Streamlit Community
    Cloud runs Python 3.14, which has no TensorFlow wheels. The two backends are
    verified equal to ~1e-7 by lstm/lstm_numpy.py.
    """

    def __init__(self, name="t2m"):
        """`name` picks the model files: lstm_<name>.keras, scaler_<name>.json,
        lstm_<name>_weights.npz in lstm/models/ (see train_lstm.py --name)."""
        self.name = name
        model_path, weights_path, scaler_path = model_files(name)
        if not scaler_path.exists():
            raise FileNotFoundError(
                f"no trained model '{name}' in {scaler_path.parent} - "
                f"run: python lstm/train_lstm.py --name {name}")
        self.scaler = json.loads(scaler_path.read_text())
        sc = self.scaler if "features" in self.scaler else _v2_scaler(self.scaler)

        if weights_path.exists():
            from lstm_numpy import NumpyLSTM
            net = NumpyLSTM(weights_path)
            self.backend = "numpy"
            self._forward = lambda x: net(x)[0]
        elif model_path.exists():
            import keras                              # slow import, do it late
            model = keras.models.load_model(model_path)
            self.backend = "keras"
            self._forward = lambda x: np.asarray(model(x, training=False))[0]
        else:
            raise FileNotFoundError(
                f"no weights in {model_path.parent} - "
                f"run: python lstm/train_lstm.py --name {name} && "
                f"python lstm/lstm_numpy.py {name}")

        self.features, self.outputs = sc["features"], sc["outputs"]
        self.mean, self.std = np.array(sc["mean"]), np.array(sc["std"])
        self.t_mean, self.t_scale = np.array(sc["target_mean"]), np.array(sc["target_scale"])
        self.lookback = sc["lookback"]
        self.horizon = sc["horizon"]
        self.delta = sc["delta"]
        # hours of observations a forecast needs before it starts
        self.seed_hours = self.lookback + feat.extra_hours(self.features)

    def describe(self):
        """One line on what the model reads and what it was trained on."""
        sc = self.scaler
        inputs = ", ".join(f for f in self.features if f not in feat.CALENDAR)
        if any(f in feat.CALENDAR for f in self.features):
            inputs += " + time of day/year"
        period = (f"trained {sc['train_start'][:10]} to {sc['train_end'][:10]}"
                  if "train_start" in sc else f"trained to {sc['train_end'][:10]}")
        return f"{self.lookback} h of {inputs} in; {period}"

    def seed_ok(self, history, end):
        """True if every input hour of a window ending at `end` is observed."""
        start = end - pd.Timedelta(hours=self.seed_hours - 1)
        if start < history.index[0] or end > history.index[-1]:
            return False
        if any(c not in history for c in self.outputs):
            return False
        return bool(history.loc[start:end, self.outputs].notna().all().all())

    def last_seedable(self, history):
        """The last hour every input of this model is observed."""
        if any(c not in history for c in self.outputs):
            return None
        ok = history[self.outputs].notna().all(axis=1)
        return ok[ok].index[-1] if ok.any() else None

    def rollout(self, seed, steps):
        """Feed the model its own output `steps` times. Returns t2m in deg C.

        seed: DataFrame of the observed hours before the forecast (at least
        seed_hours of them, hourly, holding every column in self.outputs).
        """
        if len(seed) < self.seed_hours:
            raise ValueError(f"seed must be {self.seed_hours} hours, got {len(seed)}")
        if self.horizon != 1:
            raise ValueError(
                f"this model was trained with horizon={self.horizon}; an hourly "
                f"rollout needs horizon=1 (retrain with: python lstm/train_lstm.py)")

        seed = seed.iloc[-self.seed_hours:]
        k = self.seed_hours
        times = pd.date_range(seed.index[0], periods=k + steps, freq="h")
        base = np.empty((k + steps, len(self.outputs)))
        base[:k] = seed[self.outputs].to_numpy("float64")

        for i in range(k, k + steps):
            lo = i - k
            raw = feat.build_array(times[lo:i], base[lo:i], self.outputs, self.features)
            z = ((raw[-self.lookback:] - self.mean) / self.std).astype("float32")
            step = self._forward(z[None])
            base[i] = (base[i - 1] if self.delta else 0) + self.t_mean + step * self.t_scale
        return base[k:, 0]                            # t2m is always output 0


# ------------------------------------------------------------------ forecast
def forecast_day(fc, history, day, days=1):
    """Predict `days` x 24 hours from 00:00 on `day`, warm-started if possible.

    Returns a dict with an hourly DataFrame (predicted / actual / persistence),
    the mode used, and how far the forecast had to free-run.
    """
    day = pd.Timestamp(day).normalize()
    steps = int(days) * HOURS_PER_DAY
    target = pd.date_range(day, periods=steps, freq="h")
    seed_end = day - pd.Timedelta(hours=1)              # last hour before the day
    last_ok = fc.last_seedable(history)

    if fc.seed_ok(history, seed_end):
        # every seed hour is observed: a genuine forecast issued at 00:00
        pred = fc.rollout(history.loc[:seed_end], steps)
        mode, free_hours, issued = "warm", 0, seed_end

    elif last_ok is not None and day > last_ok and fc.seed_ok(history, last_ok):
        # beyond the model's inputs: seed with the last full window, then run on
        seed_end = last_ok
        free_hours = int((day - seed_end) / pd.Timedelta(hours=1)) - 1
        if free_hours > FREE_RUN_CAP_DAYS * HOURS_PER_DAY:
            return {
                "error": (
                    f"{day.date()} is {free_hours / 24:.0f} days past the last hour "
                    f"model '{fc.name}' has inputs for ({seed_end.date()}). "
                    f"Free-running the model that far produces a decayed curve, not "
                    f"a forecast, so it is capped at {FREE_RUN_CAP_DAYS} days. Build "
                    f"the observations for that period (scripts/era5_years.py or "
                    f"scripts/era5_2026.py) to forecast this day properly."),
            }
        pred = fc.rollout(history.loc[:seed_end], free_hours + steps)[-steps:]
        mode, issued = "free", seed_end

    elif seed_end - pd.Timedelta(hours=fc.seed_hours - 1) < history.index[0]:
        return {
            "error": (
                f"{day.date()} is too close to the start of the record - the "
                f"forecast needs {fc.seed_hours} observed hours before it, and the "
                f"data starts {history.index[0]:%d %b %Y}."),
        }
    else:
        return {
            "error": (
                f"model '{fc.name}' needs {', '.join(fc.outputs)} for the "
                f"{fc.seed_hours} hours before {day.date()}, and some of them are "
                f"not in the data. Rebuild that year with every variable "
                f"(python scripts/era5_years.py --years {day.year} --force)."),
        }

    actual = history["t2m"].reindex(target)
    persistence = np.full(steps, history.at[issued, "t2m"])

    df = pd.DataFrame(
        {"predicted": pred, "persistence": persistence, "actual": actual.values},
        index=target)
    df.index.name = "valid_time"

    scored = df.dropna(subset=["actual"])
    return {
        "frame": df,
        "mode": mode,
        "issued": issued,
        "free_hours": free_hours,
        "steps": steps,
        "n_scored": len(scored),          # partial days still score what exists
        "scores": _scores(scored) if len(scored) else None,
    }


def _scores(df):
    actual = df["actual"].values
    sst = float(np.sum((actual - actual.mean()) ** 2))
    out = {}
    for col in ("predicted", "persistence"):
        err = df[col].values - actual
        out[col] = {
            "mae": float(np.mean(np.abs(err))),
            "rmse": float(np.sqrt(np.mean(err ** 2))),
            "bias": float(np.mean(err)),
            "max_err": float(np.max(np.abs(err))),
            # share of variance explained; undefined for a flat observed series
            "r2": float(1 - np.sum(err ** 2) / sst) if sst > 0 else float("nan"),
            # "accuracy" for a temperature forecast: hours within a tolerance
            "within_1": float(np.mean(np.abs(err) <= 1.0)),
            "within_2": float(np.mean(np.abs(err) <= 2.0)),
        }
    return out


if __name__ == "__main__":
    day = sys.argv[1] if len(sys.argv) > 1 else "2025-07-01"
    n_days = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    hist, meta = load_history()
    print(f"history: {hist.index[0]} .. {hist.index[-1]}  "
          f"({meta['n_2026']} hours of 2026)")

    fc = Forecaster(sys.argv[3] if len(sys.argv) > 3 else "t2m")
    print(f"model: {fc.name}  backend: {fc.backend}  ({fc.describe()})")
    res = forecast_day(fc, hist, day, n_days)
    if "error" in res:
        raise SystemExit(res["error"])

    print(f"\n{day}  mode={res['mode']}  issued={res['issued']}"
          + (f"  free-run {res['free_hours']} h" if res["free_hours"] else ""))
    print(res["frame"].round(2).to_string())
    if res["scores"]:
        for name, d in res["scores"].items():
            print(f"{name:12s} MAE {d['mae']:.3f}  RMSE {d['rmse']:.3f}  "
                  f"bias {d['bias']:+.3f} deg C")
