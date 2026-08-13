"""
Train a simple LSTM on the ERA5 domain-mean 2 m temperature series (2025).

Input  : data/processed/domain_mean_hourly_2025.csv  (column t2m, deg C, hourly)
Task   : given the last LOOKBACK hours of temperature, predict the value
         HORIZON hours ahead.
Split  : chronological - first 70% train, next 15% validation, last 15% test.
         No shuffling across the split boundaries, so the test set is genuinely
         "the future" relative to everything the model saw.
Scaling: mean/std of the *training* window only, applied to all three splits.
Target : by default the model predicts the *change* from the last input hour and
         that change is added back on - so predicting zero reproduces
         persistence, and the LSTM only has to learn the departure from it.
         --no-delta switches to predicting the absolute temperature.

Every score is reported in deg C next to a persistence baseline
(y_hat = last observed value), which is the number an hourly temperature model
actually has to beat.

Artifacts:
  models/lstm_t2m.keras      trained model
  models/scaler_t2m.json     train mean/std + window settings (needed to reuse it)
  output/01_loss_curve.png   train/val loss per epoch
  output/02_test_predictions.png
  output/metrics.csv

Usage:
  python lstm/train_lstm.py                      # 72 h in -> 1 h ahead
  python lstm/train_lstm.py --horizon 24         # 24 h ahead (much harder)
  python lstm/train_lstm.py --units 128 --epochs 100
  python lstm/train_lstm.py --no-delta            # absolute-temperature target
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")   # hide oneDNN/CUDA chatter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Paths are anchored to the project root, like the scripts/ modules, so this
# runs from anywhere: python lstm/train_lstm.py
ROOT = Path(__file__).resolve().parents[1]
CSV = ROOT / "data" / "processed" / "domain_mean_hourly_2025.csv"
MODELS = Path(__file__).resolve().parent / "models"
OUT = Path(__file__).resolve().parent / "output"

TARGET = "t2m"
TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
SEED = 7


# ---------------------------------------------------------------------- data
def load_series():
    """Hourly domain-mean 2 m temperature (deg C) as a pandas Series."""
    if not CSV.exists():
        raise SystemExit(
            f"{CSV} not found - build it first with:\n"
            f"    python scripts/era5_series.py"
        )
    df = pd.read_csv(CSV, index_col="valid_time", parse_dates=True)
    s = df[TARGET].astype("float64").sort_index()
    if s.isna().any():
        s = s.interpolate(limit_direction="both")   # keep the hourly grid intact
    return s


def make_windows(values, lookback, horizon):
    """Sliding windows: X[i] = values[i : i+lookback], y[i] = value horizon ahead.

    Built with a strided view, so the 8k-sample dataset costs no real memory.
    """
    n = len(values) - lookback - horizon + 1
    if n <= 0:
        raise ValueError("series too short for this lookback/horizon")
    idx = np.arange(lookback)[None, :] + np.arange(n)[:, None]
    X = values[idx][..., None]                       # (n, lookback, 1 feature)
    y = values[lookback + horizon - 1:]              # aligned target
    return X.astype("float32"), y.astype("float32")


def split_series(s):
    """Chronological train/val/test cut of the raw series."""
    n = len(s)
    i_train = int(n * TRAIN_FRAC)
    i_val = int(n * (TRAIN_FRAC + VAL_FRAC))
    return s.iloc[:i_train], s.iloc[i_train:i_val], s.iloc[i_val:]


def windows_for(part, mu, sd, lookback, horizon, delta):
    """Scale one split with the *training* statistics, then window it.

    The lookback tail of the previous split is not carried over: each split is
    windowed on its own, so no training hour ever leaks into a test window.

    With delta=True the target is the *change* from the last input hour rather
    than the absolute temperature. Same network, much easier target: the model
    no longer has to carry the temperature level through its hidden state, and
    predicting zero reproduces persistence exactly.
    """
    z = (part.values - mu) / sd
    X, y = make_windows(z, lookback, horizon)
    if delta:
        y = y - X[:, -1, 0]
    times = part.index[lookback + horizon - 1:]      # valid_time of each target
    return X, y, times


# --------------------------------------------------------------------- model
def build_model(lookback, units, dropout, lr):
    model = keras.Sequential([
        keras.Input(shape=(lookback, 1)),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(1),
    ], name="lstm_t2m")
    model.compile(optimizer=keras.optimizers.Adam(lr), loss="mse", metrics=["mae"])
    return model


def scores(true_c, pred_c):
    err = pred_c - true_c
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "bias": float(np.mean(err)),
        "r2": float(1 - np.sum(err ** 2) / np.sum((true_c - true_c.mean()) ** 2)),
    }


# --------------------------------------------------------------------- plots
def plot_history(hist, path):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(np.arange(1, len(hist["loss"]) + 1), hist["loss"], label="train")
    ax.plot(np.arange(1, len(hist["val_loss"]) + 1), hist["val_loss"], label="validation")
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (MSE, standardised units)")
    ax.set_title("LSTM training history", fontweight="bold")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_predictions(times, true_c, pred_c, base_c, m, mb, args, path):
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    ax.plot(times, true_c, color="#d1495b", lw=1.0, label="ERA5 observed")
    ax.plot(times, pred_c, color="#2a6f97", lw=1.0, label="LSTM prediction")
    ax.set_ylabel("2 m temperature (deg C)")
    ax.set_title(
        f"LSTM {args.horizon} h-ahead forecast, held-out test period  |  "
        f"MAE {m['mae']:.3f} deg C (persistence {mb['mae']:.3f})",
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    axr.plot(times, pred_c - true_c, color="#4a4e69", lw=0.8)
    axr.axhline(0, color="k", lw=0.7, ls="--", alpha=0.6)
    axr.set_ylabel("Error (deg C)")
    axr.set_xlabel("Date (2025)")
    axr.grid(alpha=0.3)
    axr.xaxis.set_major_locator(mdates.AutoDateLocator())
    axr.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    axr.set_xlim(times[0], times[-1])

    fig.suptitle(
        f"ERA5 domain mean (41.8-57.0N, 74.2-95.0E)  |  "
        f"lookback {args.lookback} h, horizon {args.horizon} h",
        fontsize=10, y=0.98,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main(args):
    keras.utils.set_random_seed(SEED)

    s = load_series()
    tr, va, te = split_series(s)
    mu, sd = float(tr.values.mean()), float(tr.values.std())

    Xtr, ytr, _ = windows_for(tr, mu, sd, args.lookback, args.horizon, args.delta)
    Xva, yva, _ = windows_for(va, mu, sd, args.lookback, args.horizon, args.delta)
    Xte, yte, tte = windows_for(te, mu, sd, args.lookback, args.horizon, args.delta)

    print(f"series : {len(s)} hours, {s.index[0]} .. {s.index[-1]}")
    print(f"windows: train {len(Xtr)}, val {len(Xva)}, test {len(Xte)}  "
          f"(lookback {args.lookback} h -> +{args.horizon} h)")
    print(f"target : {'change from last input hour' if args.delta else 'absolute temperature'}")
    print(f"scaler : mean {mu:.3f} deg C, std {sd:.3f} deg C (train split only)\n")

    model = build_model(args.lookback, args.units, args.dropout, args.lr)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, args.patience // 3),
            min_lr=1e-5),
    ]
    hist = model.fit(
        Xtr, ytr, validation_data=(Xva, yva),
        epochs=args.epochs, batch_size=args.batch,
        callbacks=callbacks, shuffle=True, verbose=2,
    )

    # ---- evaluate in deg C, against persistence (repeat the last observed hour)
    last_z = Xte[:, -1, 0]
    out = model.predict(Xte, verbose=0).ravel()
    pred_z, true_z = (out + last_z, yte + last_z) if args.delta else (out, yte)
    pred_c, true_c, base_c = pred_z * sd + mu, true_z * sd + mu, last_z * sd + mu

    m, mb = scores(true_c, pred_c), scores(true_c, base_c)
    skill = 1 - m["mae"] / mb["mae"]

    MODELS.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)

    model.save(MODELS / "lstm_t2m.keras")
    (MODELS / "scaler_t2m.json").write_text(json.dumps({
        "mean_c": mu, "std_c": sd, "lookback": args.lookback,
        "horizon": args.horizon, "target": TARGET, "delta": bool(args.delta),
        "train_end": str(tr.index[-1]), "test_start": str(te.index[0]),
    }, indent=2))

    pd.DataFrame([dict(model="lstm", **m), dict(model="persistence", **mb)]).to_csv(
        OUT / "metrics.csv", index=False, float_format="%.4f")

    plot_history(hist.history, OUT / "01_loss_curve.png")
    plot_predictions(tte, true_c, pred_c, base_c, m, mb, args,
                     OUT / "02_test_predictions.png")

    print(f"\ntest period: {tte[0]} .. {tte[-1]}  ({len(tte)} hours)")
    print(f"{'':12s} {'MAE':>8s} {'RMSE':>8s} {'bias':>8s} {'R2':>8s}")
    for name, d in (("LSTM", m), ("persistence", mb)):
        print(f"{name:12s} {d['mae']:8.3f} {d['rmse']:8.3f} "
              f"{d['bias']:8.3f} {d['r2']:8.3f}")
    print(f"\nskill vs persistence (MAE): {skill:+.1%}")
    print(f"saved -> {MODELS.resolve()}\n         {OUT.resolve()}")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--lookback", type=int, default=72, help="input hours (default 72)")
    p.add_argument("--horizon", type=int, default=1, help="hours ahead (default 1)")
    p.add_argument("--units", type=int, default=64, help="LSTM units (default 64)")
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--patience", type=int, default=8, help="early-stopping patience")
    p.add_argument("--no-delta", dest="delta", action="store_false",
                   help="predict the absolute temperature instead of the change")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
