"""
Inference helpers for the trained t2m LSTM - shared by app.py and usable alone.

Two ways to forecast a day:

  warm start   the 72 hours immediately before that day are observed, so the
               model is seeded with reality and rolled forward 24 hours. This
               is a real 24 h forecast issued at 00:00 and is the only mode
               worth scoring.

  free running the seed is not available (no observations near that day), so
               the model is seeded with the last observed 72 hours and fed its
               own output for as many hours as it takes to reach the target.
               Errors compound every step; past a few days this says nothing
               about the weather, only about what the model decays towards.
               Capped at FREE_RUN_CAP_DAYS.

The model predicts the *change* per hour (see train_lstm.py --no-delta), so a
rollout step is: predict delta -> add to the last value -> shift the window.

Standalone use:
    python lstm/forecast.py 2026-03-14
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

sys.path.insert(0, str(HERE))            # lstm_numpy.py sits next to this file
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


CSV_2025 = ROOT / "data" / "processed" / "domain_mean_hourly_2025.csv"
CSV_2026 = ROOT / "data" / "processed" / "domain_mean_hourly_2026.csv"

HOURS_PER_DAY = 24
FREE_RUN_CAP_DAYS = 60


# ---------------------------------------------------------------------- data
def _read(path):
    df = pd.read_csv(path, index_col="valid_time", parse_dates=True)
    return df["t2m"].astype("float64").sort_index()


def load_history():
    """Observed hourly t2m (deg C), 2025 plus 2026 if it has been built.

    Returned on a gap-free hourly index: missing hours are interpolated, since
    a rollout window cannot have holes in it.
    """
    if not CSV_2025.exists():
        raise FileNotFoundError(
            f"{CSV_2025} not found - build it with: python scripts/era5_series.py")

    parts = [_read(CSV_2025)]
    has_2026 = CSV_2026.exists()
    if has_2026:
        parts.append(_read(CSV_2026))

    s = pd.concat(parts)
    s = s[~s.index.duplicated(keep="last")]
    s = s.reindex(pd.date_range(s.index[0], s.index[-1], freq="h")).interpolate(
        limit_direction="both")
    s.index.name = "valid_time"

    obs_2026 = s[s.index.year == 2026]
    return s, {
        "has_2026": has_2026 and len(obs_2026) > 0,
        "last_2026": obs_2026.index[-1] if len(obs_2026) else None,
        "n_2026": len(obs_2026),
    }


# --------------------------------------------------------------------- model
class Forecaster:
    """The trained network plus the scaling it was trained with.

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

        if weights_path.exists():
            from lstm_numpy import NumpyLSTM
            net = NumpyLSTM(weights_path)
            self.backend = "numpy"
            self._forward = lambda x: float(net(x)[0])
        elif model_path.exists():
            import keras                              # slow import, do it late
            model = keras.models.load_model(model_path)
            self.backend = "keras"
            self._forward = lambda x: float(model(x, training=False)[0, 0])
        else:
            raise FileNotFoundError(
                f"no weights in {model_path.parent} - "
                f"run: python lstm/train_lstm.py --name {name} && "
                f"python lstm/lstm_numpy.py {name}")

        self.mu = self.scaler["mean_c"]
        self.sd = self.scaler["std_c"]
        self.lookback = self.scaler["lookback"]
        self.horizon = self.scaler["horizon"]
        self.delta = self.scaler.get("delta", False)

    def rollout(self, seed_c, steps):
        """Feed the model its own output `steps` times. Returns deg C."""
        if len(seed_c) != self.lookback:
            raise ValueError(f"seed must be {self.lookback} hours, got {len(seed_c)}")
        if self.horizon != 1:
            raise ValueError(
                f"this model was trained with horizon={self.horizon}; an hourly "
                f"rollout needs horizon=1 (retrain with: python lstm/train_lstm.py)")

        z = ((np.asarray(seed_c, dtype="float64") - self.mu) / self.sd).astype("float32")
        out = np.empty(steps, dtype="float64")
        for i in range(steps):
            step = self._forward(z[None, :, None])
            nxt = step + z[-1] if self.delta else step
            out[i] = nxt
            z = np.append(z[1:], np.float32(nxt))
        return out * self.sd + self.mu


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
    seed_start = seed_end - pd.Timedelta(hours=fc.lookback - 1)

    if seed_start >= history.index[0] and seed_end <= history.index[-1]:
        # every seed hour is observed: a genuine forecast issued at 00:00
        seed = history.loc[seed_start:seed_end].values
        pred = fc.rollout(seed, steps)
        mode, free_hours, issued = "warm", 0, seed_end

    elif day > history.index[-1]:
        # beyond the observations: seed with the last real window, then run on
        seed_end = history.index[-1]
        seed = history.iloc[-fc.lookback:].values
        free_hours = int((day - seed_end) / pd.Timedelta(hours=1)) - 1
        if free_hours > FREE_RUN_CAP_DAYS * HOURS_PER_DAY:
            return {
                "error": (
                    f"{day.date()} is {free_hours / 24:.0f} days past the last "
                    f"observation ({seed_end.date()}). Free-running the model that "
                    f"far produces a decayed curve, not a forecast, so it is capped "
                    f"at {FREE_RUN_CAP_DAYS} days. Build the 2026 observations "
                    f"(python scripts/era5_2026.py) to forecast this day properly."),
            }
        pred = fc.rollout(seed, free_hours + steps)[-steps:]
        mode, issued = "free", seed_end

    else:
        return {
            "error": (
                f"{day.date()} is too close to the start of the record - the "
                f"forecast needs {fc.lookback} observed hours before it, and the "
                f"data starts {history.index[0]:%d %b %Y}."),
        }

    actual = history.reindex(target)
    persistence = np.full(steps, history.loc[issued])

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
    print(f"model: {fc.name}  backend: {fc.backend}")
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
