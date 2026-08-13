"""
Streamlit UI for the ERA5 2 m temperature LSTM.

    streamlit run lstm/app.py

Shows the observed domain-mean record, then forecasts any single day of 2026
(24 hourly steps issued at 00:00) and scores it against the observations once
data/processed/domain_mean_hourly_2026.csv exists.
"""

import datetime as dt
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # forecast.py sits here

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from forecast import (FREE_RUN_CAP_DAYS, Forecaster, forecast_day, load_history,
                      CSV_2026)

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
def get_forecaster():
    return Forecaster()


@st.cache_data(show_spinner=False)
def get_forecast(day, _cache_key):
    """_cache_key changes when the model or the 2026 file does."""
    hist, _ = get_history()
    return forecast_day(get_forecaster(), hist, day)


# -------------------------------------------------------------------- charts
def record_chart(hist, day):
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.plot(hist.index, hist.values, color=C_OBS, lw=0.4, alpha=0.25)
    ax.plot(hist.index, hist.rolling(24, center=True, min_periods=1).mean().values,
            color=C_OBS, lw=1.1, label="observed (24 h mean)")
    ax.axvspan(pd.Timestamp(day), pd.Timestamp(day) + pd.Timedelta(days=1),
               color=C_PRED, alpha=0.35, label="selected day")
    ax.axhline(0, color="k", lw=0.6, ls="--", alpha=0.5)
    ax.set_ylabel("2 m temperature (deg C)")
    ax.set_xlim(hist.index[0], max(hist.index[-1], pd.Timestamp(day) + pd.Timedelta(days=1)))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def day_chart(df, day, mode):
    fig, ax = plt.subplots(figsize=(12, 4.2))
    ax.plot(df.index, df["persistence"], color=C_PERS, lw=1.0, ls=":",
            label="persistence (last observed hour)")
    if df["actual"].notna().any():
        ax.plot(df.index, df["actual"], color=C_OBS, lw=2.0, marker="o", ms=3,
                label="ERA5 observed")
    ax.plot(df.index, df["predicted"], color=C_PRED, lw=2.0, ls="--", marker="s",
            ms=3, label="LSTM forecast")
    ax.set_title(f"{pd.Timestamp(day):%A %d %B %Y} - 24 hourly steps issued at 00:00"
                 + ("" if mode == "warm" else "  (free-running)"), fontweight="bold")
    ax.set_ylabel("2 m temperature (deg C)")
    ax.set_xlabel("Hour (UTC)")
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------- app
st.title("ERA5 2 m temperature - LSTM day forecast")
st.caption(f"Domain mean {DOMAIN} · trained on 2025 hourly ERA5 · "
           f"72 h in, 1 h ahead, rolled out 24 times")

try:
    hist, meta = get_history()
    fc = get_forecaster()
except (FileNotFoundError, ValueError) as e:
    st.error(str(e))
    st.stop()

# the forecast cache must notice a retrained model or a new 2026 file
cache_key = (fc.scaler.get("train_end"), fc.horizon, fc.delta, meta["n_2026"])

# ---- sidebar: what day, and what the app currently knows
with st.sidebar:
    st.header("Forecast a day")

    last_obs = hist.index[-1]
    # days past the last observation are allowed, but only the model's own
    # output carries the forecast there - forecast_day() caps and flags it
    max_day = (last_obs + pd.Timedelta(days=FREE_RUN_CAP_DAYS)).date()
    default_day = (last_obs - pd.Timedelta(days=1)).date() if meta["has_2026"] \
        else dt.date(2026, 1, 1)

    day = st.date_input("Day in 2026", value=default_day,
                        min_value=dt.date(2026, 1, 1), max_value=max_day,
                        format="YYYY-MM-DD")

    st.divider()
    st.subheader("Data")
    st.write(f"Observations to **{last_obs:%Y-%m-%d %H:%M}**")
    if meta["has_2026"]:
        st.success(f"{meta['n_2026']:,} hours of 2026 loaded")
    else:
        st.warning("No 2026 observations yet")
        st.caption(
            "Download 2026 `2m_temperature` from the CDS with "
            "`area = [57, 74.25, 41.75, 95]`, put the GRIB in `data/raw/`, then "
            "run `python scripts/era5_2026.py`. Until then the model can only "
            "free-run forward from 31 Dec 2025 and there is nothing to score it "
            "against.")
    st.caption(f"Expected file: `{CSV_2026.relative_to(CSV_2026.parents[2])}`")

# ---- the record, with the chosen day marked
st.subheader("Observed record")
st.pyplot(record_chart(hist, day), width="stretch")
plt.close("all")

# ---- the forecast for that day
st.subheader("Forecast vs actual")
with st.spinner("Rolling the model forward ..."):
    res = get_forecast(pd.Timestamp(day), cache_key)

if "error" in res:
    st.error(res["error"])
    st.stop()

df, mode = res["frame"], res["mode"]

if mode == "warm":
    st.info(f"**Warm start.** Seeded with the {fc.lookback} observed hours ending "
            f"{res['issued']:%d %b %H:%M}, then run 24 steps. This is a real "
            f"24 h forecast.")
else:
    st.warning(f"**Free-running.** No observations near {day}, so the model was "
               f"seeded at {res['issued']:%d %b %Y %H:%M} and fed its own output "
               f"for {res['free_hours']:,} hours to reach this day. Errors compound "
               f"at every step - read this as model behaviour, not a forecast.")

st.pyplot(day_chart(df, day, mode), width="stretch")
plt.close("all")

if res["scores"]:
    s, p = res["scores"]["predicted"], res["scores"]["persistence"]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE", f"{s['mae']:.2f} °C",
              f"{p['mae'] - s['mae']:+.2f} vs persistence", delta_color="normal")
    c2.metric("RMSE", f"{s['rmse']:.2f} °C", f"{p['rmse'] - s['rmse']:+.2f} vs persistence")
    c3.metric("Bias", f"{s['bias']:+.2f} °C")
    c4.metric("Warmest hour error",
              f"{df['predicted'].max() - df['actual'].max():+.2f} °C")
    st.caption("Deltas are persistence minus LSTM: positive means the LSTM was "
               "closer to the observations that day.")
elif mode == "warm":
    st.info("This day is not fully observed yet, so there is nothing to score "
            "the forecast against.")

with st.expander("Hourly values"):
    st.dataframe(df.round(2), width="stretch")
    st.download_button("Download CSV", df.to_csv().encode(),
                       file_name=f"forecast_{pd.Timestamp(day):%Y%m%d}.csv",
                       mime="text/csv")
