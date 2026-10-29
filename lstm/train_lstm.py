"""
Train the LSTM on the ERA5 domain-mean hourly record (V3: many years, many inputs).

Input  : data/processed/domain_mean_hourly_<year>.csv, every year present
         (build them with scripts/era5_years.py - 2006-2025 by default)
Inputs : temperature, its last-hour change, dewpoint, the 10 m wind, the heat
         the wind carries in (t2m_adv) and time of day / year - see
         lstm/features.py. --features picks a different set.
Task   : given the last LOOKBACK hours, predict HORIZON hours ahead.
Target : the *change* of every input variable from the last input hour, so
         predicting zero reproduces persistence and "no change" - the most
         common answer an hour ahead - is the easiest thing to say. Temperature
         is the one that is scored; the rest are predicted so a multi-hour
         forecast can feed them back in. --no-delta predicts absolute values.
Split  : chronological, by whole calendar years, 70 / 15 / 15 - with 2006-2025
         that is train 2006-2019, validation 2020-2022, test 2023-2025. Whole
         years keep every season in every split; with under three years of data
         it falls back to cutting the hours 70 / 15 / 15 like V2. The split is
         saved with the model, and --resume reuses it.
Scaling: mean/std of the *training* split only, per feature.

Every score is reported in deg C next to a persistence baseline
(y_hat = last observed value), which is the number an hourly temperature model
actually has to beat.

Artifacts (for --name v3):
  models/lstm_v3.keras            trained model (also the per-epoch checkpoint)
  models/scaler_v3.json           features, scaling, split - needed to reuse it
  models/lstm_v3_weights.npz      numpy export the apps run on (checked vs Keras)
  output/v3/01_loss_curve.png     train/val loss per epoch
  output/v3/02_test_predictions.png
  output/v3/metrics.csv

Usage:
  python lstm/train_lstm.py                              # V3 defaults, name v3
  python lstm/train_lstm.py --years 2006-2025            # leave a partial 2026 out
  python lstm/train_lstm.py --resume --epochs 30         # continue lstm_v3.keras
  python lstm/train_lstm.py --name big --units 128 --epochs 100
  python lstm/train_lstm.py --name uni --features t2m    # temperature only, like V2
"""

import argparse
import json
import os
import sys
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

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import features as feat                                  # noqa: E402
from lstm_numpy import export, verify                    # noqa: E402

# Paths are anchored to the project root, like the scripts/ modules, so this
# runs from anywhere: python lstm/train_lstm.py
ROOT = HERE.parent
PROCESSED = ROOT / "data" / "processed"
MODELS = HERE / "models"
OUT = HERE / "output"

TRAIN_FRAC, VAL_FRAC = 0.70, 0.15
SEED = 7


# ---------------------------------------------------------------------- data
def load_frame(columns, years=None):
    """Every yearly CSV, joined on one gap-free hourly index.

    Missing hours become NaN rows (short gaps are interpolated) and any window
    touching one is skipped later - nothing is stitched across a gap.
    """
    paths = sorted(PROCESSED.glob("domain_mean_hourly_*.csv"))
    if years:
        paths = [p for p in paths if int(p.stem.rsplit("_", 1)[1]) in years]
    if not paths:
        raise SystemExit(
            f"no domain_mean_hourly_<year>.csv in {PROCESSED} - build them with:\n"
            f"    python scripts/era5_years.py")

    df = pd.concat(pd.read_csv(p, index_col="valid_time", parse_dates=True) for p in paths)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    df = df.reindex(pd.date_range(df.index[0], df.index[-1], freq="h"))
    df.index.name = "valid_time"
    check_coverage(df, columns)
    return df[columns].interpolate(limit=6, limit_area="inside")


def check_coverage(df, columns):
    """Refuse to train if a year is missing a whole input column.

    Otherwise those hours would quietly produce no windows and the model would
    train as if that year - or that feature - never existed. A year built from
    a t2m-only GRIB (the original 2026 instructions) is the usual cause.
    """
    absent = [c for c in columns if c not in df]
    if absent:
        raise SystemExit(f"no CSV has column(s) {absent} - rebuild with "
                         f"python scripts/era5_years.py --force")
    have = df[columns].notna().groupby(df.index.year).mean()
    have = have[df["t2m"].notna().groupby(df.index.year).any()]     # years with data
    holes = [(y, c) for y in have.index for c in columns if have.at[y, c] < 0.5]
    if holes:
        lines = "\n".join(f"  {y}: {c} is {have.at[y, c]:.0%} present" for y, c in holes)
        raise SystemExit(
            f"some years are missing inputs this model needs:\n{lines}\n"
            f"rebuild them (python scripts/era5_years.py --years <year> --force) "
            f"or leave them out with --years.")


def split_bounds(df, val_start=None, test_start=None):
    """(validation start, test start). Whole years if there are three or more."""
    if val_start and test_start:
        return pd.Timestamp(val_start), pd.Timestamp(test_start)
    years = sorted(df["t2m"].dropna().index.year.unique())
    if len(years) >= 3:
        hold = max(1, round(len(years) * (1 - TRAIN_FRAC) / 2))
        return (pd.Timestamp(f"{years[-2 * hold]}-01-01"),
                pd.Timestamp(f"{years[-hold]}-01-01"))
    n = len(df)                                   # one or two years: cut the hours
    return df.index[int(n * TRAIN_FRAC)], df.index[int(n * (TRAIN_FRAC + VAL_FRAC))]


def fit_scaler(train, features, outputs, horizon, delta):
    """Per-feature mean/std, and the scale of each predicted quantity, from train only.

    With delta the target is a change over `horizon` hours, scaled by the spread
    of those changes - so every output weighs about the same in the loss, rather
    than the wind's changes swamping the temperature's.
    """
    F = feat.build(train, features)
    mean, std = np.nanmean(F, axis=0), np.nanstd(F, axis=0)
    std[std == 0] = 1.0

    B = train[outputs].to_numpy("float64")
    if delta:
        d = B[horizon:] - B[:-horizon]
        t_mean, t_scale = np.zeros(len(outputs)), np.nanstd(d, axis=0)
    else:
        t_mean, t_scale = np.nanmean(B, axis=0), np.nanstd(B, axis=0)
    return {"mean": mean.tolist(), "std": std.tolist(),
            "target_mean": t_mean.tolist(), "target_scale": t_scale.tolist()}


def windows_for(part, sc, lookback, horizon, delta):
    """Scale one split with the *training* statistics, then window it.

    The lookback tail of the previous split is not carried over: each split is
    windowed on its own, so no training hour ever leaks into a test window.
    Windows that touch a missing hour are dropped.

    Returns X (n, lookback, features), Y (n, outputs), the valid_time of each
    target, and the last input / true target temperature in deg C.
    """
    features, outputs = sc["features"], sc["outputs"]
    Z = (feat.build(part, features) - np.array(sc["mean"])) / np.array(sc["std"])
    B = part[outputs].to_numpy("float64")

    n = len(part) - lookback - horizon + 1
    if n <= 0:
        raise ValueError(f"a split of {len(part)} hours is too short for "
                         f"lookback {lookback} + horizon {horizon}")
    bad = np.concatenate([[0], np.cumsum(~np.isfinite(Z).all(axis=1))])
    start = np.arange(n)
    last = start + lookback - 1                      # last input hour
    tgt = last + horizon
    ok = ((bad[last + 1] - bad[start]) == 0) & np.isfinite(B[tgt]).all(1) \
        & np.isfinite(B[last]).all(1)
    start, last, tgt = start[ok], last[ok], tgt[ok]

    X = Z[start[:, None] + np.arange(lookback)[None, :]].astype("float32")
    Y = (B[tgt] - (B[last] if delta else 0) - np.array(sc["target_mean"])) \
        / np.array(sc["target_scale"])
    return X, Y.astype("float32"), part.index[tgt], B[last, 0], B[tgt, 0]


# --------------------------------------------------------------------- model
def build_model(lookback, n_features, n_outputs, units, input_dropout, dropout, lr):
    model = keras.Sequential([
        keras.Input(shape=(lookback, n_features)),
        # Each input value - one feature at one hour - is zeroed on its own with
        # probability input_dropout, so every feature is sometimes missing but
        # none ever disappears. SpatialDropout1D or LSTM(dropout=...) would
        # instead reuse one mask across all hours of a window, dropping a whole
        # feature for that sample - and with a single input, the entire series.
        layers.Dropout(input_dropout),
        layers.LSTM(units),
        layers.Dropout(dropout),
        layers.Dense(n_outputs),
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


def plot_predictions(times, true_c, pred_c, base_c, m, mb, sc, path):
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(11, 6), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08},
    )
    lw = 1.0 if len(times) < 5000 else 0.4
    ax.plot(times, true_c, color="#d1495b", lw=lw, label="ERA5 observed")
    ax.plot(times, pred_c, color="#2a6f97", lw=lw, label="LSTM prediction")
    ax.set_ylabel("2 m temperature (deg C)")
    ax.set_title(
        f"LSTM {sc['horizon']} h-ahead forecast, held-out test period  |  "
        f"MAE {m['mae']:.3f} deg C (persistence {mb['mae']:.3f})",
        fontweight="bold",
    )
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right")

    axr.plot(times, pred_c - true_c, color="#4a4e69", lw=0.6)
    axr.axhline(0, color="k", lw=0.7, ls="--", alpha=0.6)
    axr.set_ylabel("Error (deg C)")
    axr.set_xlabel("Date (UTC)")
    axr.grid(alpha=0.3)
    loc = mdates.AutoDateLocator()
    axr.xaxis.set_major_locator(loc)
    axr.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    axr.set_xlim(times[0], times[-1])

    fig.suptitle(
        f"ERA5 domain mean (41.8-57.0N, 74.2-95.0E)  |  "
        f"lookback {sc['lookback']} h, horizon {sc['horizon']} h  |  "
        f"{len(sc['features'])} inputs",
        fontsize=10, y=0.98,
    )
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------- main
def main(args):
    keras.utils.set_random_seed(SEED)
    model_path = MODELS / f"lstm_{args.name}.keras"
    scaler_path = MODELS / f"scaler_{args.name}.json"

    if args.resume:
        # same features, scaling and split as the run being continued - a
        # different split would let the model train on its own test years
        if not (model_path.exists() and scaler_path.exists()):
            raise SystemExit(f"nothing to resume: {model_path.name} / {scaler_path.name} "
                             f"not found in {MODELS}")
        sc = json.loads(scaler_path.read_text())
        if "features" not in sc:
            raise SystemExit(f"'{args.name}' is a V2 model; --resume needs a V3 scaler")
        df = load_frame(sc["outputs"], args.years)
        val_start, test_start = pd.Timestamp(sc["val_start"]), pd.Timestamp(sc["test_start"])
        model = keras.models.load_model(model_path)
        print(f"resuming '{args.name}' (best val loss so far "
              f"{sc.get('best_val_loss', float('nan')):.5f})")
    else:
        features = args.features.split(",")
        outputs = feat.outputs_for(features)
        df = load_frame(outputs, args.years)
        val_start, test_start = split_bounds(df, args.val_start, args.test_start)
        sc = {"version": 3, "features": features, "outputs": outputs,
              "lookback": args.lookback, "horizon": args.horizon,
              "delta": bool(args.delta), "target": "t2m"}
        sc.update(fit_scaler(df[df.index < val_start], features, outputs,
                             args.horizon, args.delta))
        model = build_model(args.lookback, len(features), len(outputs), args.units,
                            args.input_dropout, args.dropout, args.lr)

    tr = df[df.index < val_start]
    va = df[(df.index >= val_start) & (df.index < test_start)]
    te = df[df.index >= test_start]
    lookback, horizon, delta = sc["lookback"], sc["horizon"], sc["delta"]

    Xtr, Ytr, _, _, _ = windows_for(tr, sc, lookback, horizon, delta)
    Xva, Yva, _, _, _ = windows_for(va, sc, lookback, horizon, delta)
    Xte, Yte, tte, last_c, true_c = windows_for(te, sc, lookback, horizon, delta)
    # one column per feature, or a feature was lost on the way in
    assert Xtr.shape[-1] == len(sc["features"]) and Ytr.shape[-1] == len(sc["outputs"])

    sc.update({"train_start": str(tr.index[0]), "train_end": str(tr.index[-1]),
               "val_start": str(va.index[0]), "test_start": str(te.index[0]),
               "test_end": str(te.index[-1])})

    print(f"record : {len(df):,} hours, {df.index[0]} .. {df.index[-1]}")
    print(f"split  : train {tr.index[0]:%Y-%m-%d} .. {tr.index[-1]:%Y-%m-%d}, "
          f"val {va.index[0]:%Y-%m-%d} .. {va.index[-1]:%Y-%m-%d}, "
          f"test {te.index[0]:%Y-%m-%d} .. {te.index[-1]:%Y-%m-%d}")
    print(f"windows: train {len(Xtr):,}, val {len(Xva):,}, test {len(Xte):,}  "
          f"(lookback {lookback} h -> +{horizon} h)")
    print(f"inputs : {', '.join(sc['features'])}")
    print(f"outputs: {'change in ' if delta else ''}{', '.join(sc['outputs'])}\n")
    model.summary()

    # write the scaler before training so every checkpoint is usable on its own -
    # a Colab session that disconnects mid-run can be picked up with --resume
    MODELS.mkdir(parents=True, exist_ok=True)
    scaler_path.write_text(json.dumps(sc, indent=2))

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=args.patience, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=max(2, args.patience // 3),
            min_lr=1e-5),
        keras.callbacks.ModelCheckpoint(
            model_path, monitor="val_loss", save_best_only=True,
            initial_value_threshold=sc.get("best_val_loss")),
    ]
    hist = model.fit(
        Xtr, Ytr, validation_data=(Xva, Yva),
        epochs=args.epochs, batch_size=args.batch,
        callbacks=callbacks, shuffle=True, verbose=2,
    )

    # ---- evaluate temperature in deg C, against persistence (last observed hour)
    out = model.predict(Xte, verbose=0, batch_size=4096)[:, 0]
    pred_c = out * sc["target_scale"][0] + sc["target_mean"][0] + (last_c if delta else 0)
    m, mb = scores(true_c, pred_c), scores(true_c, last_c)
    skill = 1 - m["mae"] / mb["mae"]

    # the default V2 name keeps the original file names; any other name gets its
    # own output folder so earlier models are never overwritten
    out_dir = OUT if args.name == "t2m" else OUT / args.name
    out_dir.mkdir(parents=True, exist_ok=True)

    best = float(np.min(hist.history["val_loss"]))
    sc["best_val_loss"] = min(best, sc.get("best_val_loss", np.inf))
    model.save(model_path)
    scaler_path.write_text(json.dumps(sc, indent=2))

    rows = [dict(model="lstm", **m), dict(model="persistence", **mb)]
    # the V2 test window (Nov-Dec 2025), so the two versions compare like for like
    v2_path = MODELS / "scaler_t2m.json"
    if args.name != "t2m" and v2_path.exists():
        v2_start = pd.Timestamp(json.loads(v2_path.read_text())["test_start"])
        sel = (tte >= v2_start) & (tte < pd.Timestamp("2026-01-01"))
        if sel.any():
            rows += [dict(model="lstm (V2 test window)", **scores(true_c[sel], pred_c[sel])),
                     dict(model="persistence (V2 test window)",
                          **scores(true_c[sel], last_c[sel]))]
    pd.DataFrame(rows).to_csv(out_dir / "metrics.csv", index=False, float_format="%.4f")

    plot_history(hist.history, out_dir / "01_loss_curve.png")
    plot_predictions(tte, true_c, pred_c, last_c, m, mb, sc,
                     out_dir / "02_test_predictions.png")

    print(f"\ntest period: {tte[0]} .. {tte[-1]}  ({len(tte):,} hours)")
    print(f"{'':30s} {'MAE':>8s} {'RMSE':>8s} {'bias':>8s} {'R2':>8s}")
    for r in rows:
        print(f"{r['model']:30s} {r['mae']:8.3f} {r['rmse']:8.3f} "
              f"{r['bias']:8.3f} {r['r2']:8.3f}")
    print(f"\nskill vs persistence (MAE): {skill:+.1%}")
    print(f"saved model '{args.name}' -> {MODELS.resolve()}\n         {out_dir.resolve()}\n")

    # the apps run the numpy export, so refresh it now rather than leave a stale one
    verify(export(args.name), args.name)


def parse_years(text):
    if not text:
        return None
    if "-" in text:
        a, b = (int(x) for x in text.split("-"))
        return set(range(a, b + 1))
    return {int(y) for y in text.split(",")}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--name", default="v3",
                   help="model name; files are lstm_<name>.keras etc. (default v3; "
                        "t2m is the V2 model the apps ship with)")
    p.add_argument("--features", default=",".join(feat.DEFAULT_FEATURES),
                   help="comma-separated inputs, see lstm/features.py "
                        f"(default {','.join(feat.DEFAULT_FEATURES)})")
    p.add_argument("--years", type=parse_years, default=None,
                   help="years to use, e.g. 2006-2025 (default: every CSV present)")
    p.add_argument("--val-start", help="first validation hour, e.g. 2020-01-01 "
                                       "(with --test-start; default: whole-year 70/15/15)")
    p.add_argument("--test-start", help="first test hour, e.g. 2023-01-01")
    p.add_argument("--resume", action="store_true",
                   help="continue training lstm_<name>.keras with its saved split and scaling")
    p.add_argument("--lookback", type=int, default=72, help="input hours (default 72)")
    p.add_argument("--horizon", type=int, default=1, help="hours ahead (default 1)")
    p.add_argument("--units", type=int, default=64, help="LSTM units (default 64)")
    p.add_argument("--input-dropout", type=float, default=0.1,
                   help="chance each input value is zeroed in training (default 0.1)")
    p.add_argument("--dropout", type=float, default=0.1,
                   help="dropout on the LSTM output (default 0.1)")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--patience", type=int, default=8, help="early-stopping patience")
    p.add_argument("--no-delta", dest="delta", action="store_false",
                   help="predict absolute values instead of the change")
    return p.parse_args()


if __name__ == "__main__":
    main(parse_args())
