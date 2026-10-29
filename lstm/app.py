"""
Streamlit UI for the ERA5 2 m temperature LSTM.

    streamlit run lstm/app.py

Shows the observed domain-mean record, then forecasts any day in it (or up to
60 days past it) for 1-7 days, hourly steps issued at 00:00, and scores the
forecast against the observations. Pick the V2 (t2m) or V3 model in the sidebar.
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # forecast.py sits here

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from forecast import (FREE_RUN_CAP_DAYS, Forecaster, forecast_day, list_models,
                      load_history, CSV_2026)

DOMAIN = "41.8-57.0N, 74.2-95.0E"
C_OBS = "#d1495b"       # same palette as the scripts/ figures
C_PRED = "#2a6f97"
C_PERS = "#8d99ae"

st.set_page_config(page_title="ERA5 t2m LSTM", page_icon="🌡️", layout="wide")
plt.rcParams.update({"axes.grid": True, "grid.alpha": 0.3, "font.size": 9})


# ------------------------------------------------------------------- loading
@st.cache_data(show_spinner="Loading observations ...")
def get_history():
    return load_history()


@st.cache_resource(show_spinner="Loading model ...")
def get_forecaster(name):
    return Forecaster(name)


@st.cache_data(show_spinner=False)
def get_forecast(name, day, days, _cache_key):
    """_cache_key changes when the model or the 2026 file does."""
    hist, _ = get_history()
    return forecast_day(get_forecaster(name), hist, day, days)


# -------------------------------------------------------------------- charts
def record_chart(t2m, day, n_days):
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(t2m.index, t2m.values, color=C_OBS, lw=0.4, alpha=0.25)
    ax.plot(t2m.index, t2m.rolling(24, center=True, min_periods=1).mean().values,
            color=C_OBS, lw=1.1, label="observed (24 h mean)")
    ax.axvspan(pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=n_days),
               color=C_PRED, alpha=0.35, label="forecast window")
    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_ylabel("2 m temperature (deg C)")
    ax.set_xlim(t2m.index[0], max(t2m.index[-1], pd.Timestamp(day) + pd.Timedelta(days=1)))
    loc = mdates.AutoDateLocator()          # months for one year, years for twenty
    ax.xaxis.set_major_locator(loc)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(loc))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def day_chart(df, day, mode, n_days):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    marker = dict(marker="o", ms=3) if n_days == 1 else {}
    ax.plot(df.index, df["persistence"], color=C_PERS, lw=1.0, ls=":",
            label="persistence (last observed hour)")
    if df["actual"].notna().any():
        ax.plot(df.index, df["actual"], color=C_OBS, lw=2.0, label="ERA5 observed",
                **marker)
    ax.plot(df.index, df["predicted"], color=C_PRED, lw=2.0, ls="--",
            label="LSTM forecast", **marker)

    span = (f"{pd.Timestamp(day):%A %d %B %Y}" if n_days == 1 else
            f"{pd.Timestamp(day):%d %b} - {df.index[-1]:%d %b %Y}")
    ax.set_title(f"{span} - {len(df)} hourly steps issued at 00:00"
                 + ("" if mode == "warm" else "  (free-running)"), fontweight="bold")
    ax.set_ylabel("2 m temperature (deg C)")

    if n_days == 1:
        ax.set_xlabel("Hour (UTC)")
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    else:
        ax.set_xlabel("Date (UTC)")
        ax.xaxis.set_major_locator(mdates.DayLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b"))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------- app
st.title("ERA5 2 m temperature - LSTM day forecast")
st.caption(f"Domain mean {DOMAIN} · hourly ERA5 · the last few days in, "
           f"one hour predicted at a time, fed back in")

models = list_models()
if not models:
    st.error("No models in lstm/models/ - run: python lstm/train_lstm.py")
    st.stop()

# ---- sidebar: what day, and what the app currently knows
with st.sidebar:
    st.header("Forecast a day")
    # v3 reads humidity and wind too; t2m is the V2 temperature-only model
    default = next((m for m in ("v3", "t2m") if m in models), models[0])
    model_name = st.selectbox("Model", models, index=models.index(default))

try:
    hist, meta = get_history()
    fc = get_forecaster(model_name)
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

# the forecast cache must notice a retrained model or a new 2026 file
cache_key = (fc.scaler.get("train_end"), fc.horizon, fc.delta, meta["n_2026"])

with st.sidebar:
    st.caption(fc.describe())

    last_obs = hist.index[-1]
    # Any day with `lookback` observed hours in front of it can be forecast and
    # scored - that is most of the record, not just 2026. Days past the last
    # observation are allowed too, but only the model's own output carries the
    # forecast there, so forecast_day() caps and flags those.
    min_day = (hist.index[0] + pd.Timedelta(hours=fc.seed_hours)).date()
    max_day = (last_obs + pd.Timedelta(days=FREE_RUN_CAP_DAYS)).date()
    default_day = (last_obs - pd.Timedelta(days=1)).date() if meta["has_2026"] \
        else dt.date(2025, 7, 1)

    if "day" not in st.session_state:
        st.session_state.day = default_day

    day = st.date_input("First day to forecast", key="day",
                        min_value=min_day, max_value=max_day, format="YYYY-MM-DD")
    n_days = st.slider("Forecast length (days)", 1, 7, 1,
                       help="Each hour is predicted from the previous one, so "
                            "error compounds the further out you go.")

    if st.button("🎲 Random observed day", width="stretch"):
        # any day that is fully observed, so the forecast can be scored
        span = max((last_obs - pd.Timedelta(days=n_days)).date() - min_day,
                   dt.timedelta(days=1))
        st.session_state.day = min_day + dt.timedelta(
            days=int(np.random.randint(span.days)))
        st.rerun()

    st.divider()
    st.subheader("Data")
    st.write(f"Observations to **{last_obs:%Y-%m-%d %H:%M}**")
    if meta["has_2026"]:
        st.success(f"{meta['n_2026']:,} hours of 2026 loaded")
    else:
        st.warning("No 2026 observations yet")
        st.caption(
            "Run `python scripts/era5_years.py --years 2026` (or download a "
            "GRIB with `area = [57, 74.25, 41.75, 95]` into `data/raw/` and run "
            "`python scripts/era5_2026.py`). Until then the model can only "
            "free-run forward from the last observation and there is nothing to "
            "score it against.")
    st.caption(f"Expected file: `{CSV_2026.relative_to(CSV_2026.parents[2])}`")

# ---- the record, with the chosen day marked
st.subheader("Observed record")
st.pyplot(record_chart(hist["t2m"], day, n_days), width="stretch")
plt.close("all")

# ---- the forecast for that day
st.subheader("Forecast vs actual")
with st.spinner("Rolling the model forward ..."):
    res = get_forecast(model_name, pd.Timestamp(day), n_days, cache_key)

if "error" in res:
    st.error(res["error"])
    st.stop()

df, mode = res["frame"], res["mode"]

if mode == "warm":
    st.info(f"**Warm start.** Seeded with the {fc.lookback} observed hours ending "
            f"{res['issued']:%d %b %Y %H:%M}, then run {res['steps']} steps. "
            f"This is a real {res['steps']} h forecast.")
else:
    st.warning(f"**Free-running.** No inputs for this model near {day}, so it was "
               f"seeded at {res['issued']:%d %b %Y %H:%M} and fed its own output "
               f"for {res['free_hours']:,} hours to reach this day. Errors compound "
               f"at every step - read this as model behaviour, not a forecast.")

st.pyplot(day_chart(df, day, mode, n_days), width="stretch")
plt.close("all")

if res["scores"]:
    s, p = res["scores"]["predicted"], res["scores"]["persistence"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{s['mae']:.2f} °C",
              f"{p['mae'] - s['mae']:+.2f} vs persistence", delta_color="normal")
    c2.metric("RMSE", f"{s['rmse']:.2f} °C", f"{p['rmse'] - s['rmse']:+.2f} vs persistence")
    c3.metric("Bias", f"{s['bias']:+.2f} °C")
    c4.metric("Peak error",
              f"{df['predicted'].max() - df['actual'].max():+.2f} °C",
              help="Forecast warmest hour minus observed warmest hour")
    scored = "" if res["n_scored"] == res["steps"] else \
        f" Only {res['n_scored']} of {res['steps']} hours are observed, so the "\
        f"scores cover that part."
    st.caption("Deltas are persistence minus LSTM: positive means the LSTM was "
               "closer to the observations." + scored)
else:
    st.info("None of these hours are observed, so there is nothing to score the "
            "forecast against.")

with st.expander("Hourly values"):
    st.dataframe(df.round(2), width="stretch")
    st.download_button("Download CSV", df.to_csv().encode(),
                       file_name=f"forecast_{pd.Timestamp(day):%Y%m%d}.csv",
                       mime="text/csv")
