"""
Gradio UI for the ERA5 2 m temperature LSTM (Hugging Face Spaces entry point).

    pip install gradio
    python app.py

Same model and forecasting code as the Streamlit app (lstm/app.py): pick a first
day, get 24-168 hourly steps issued at 00:00, scored against ERA5 where observed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "lstm"))   # forecast.py

import gradio as gr
import pandas as pd
import plotly.graph_objects as go

from forecast import (FREE_RUN_CAP_DAYS, Forecaster, forecast_day, list_models,
                      load_history)

DOMAIN = "41.8-57.0N, 74.2-95.0E"
C_OBS, C_PRED, C_PERS = "#d1495b", "#2a6f97", "#8d99ae"   # same palette as lstm/app.py

hist, meta = load_history()

# every model in lstm/models/ (scaler_<name>.json + weights): t2m is V2, v3 the
# multi-variable model once it has been trained, and the default when present.
# Models load on first use and are kept, so switching is instant after that.
MODELS = list_models()
if not MODELS:
    raise SystemExit("no models in lstm/models/ - run: python lstm/train_lstm.py")
DEFAULT_MODEL = next((m for m in ("v3", "t2m") if m in MODELS), MODELS[0])
_loaded = {}


def get_model(name):
    if name not in _loaded:
        _loaded[name] = Forecaster(name)
    return _loaded[name]


OUTPUT_DIR = Path(__file__).resolve().parent / "lstm" / "output"

# (key, label, how to format, True if higher is better)
STATS = [
    ("mae", "MAE (°C)", "{:.2f}", False),
    ("rmse", "RMSE (°C)", "{:.2f}", False),
    ("bias", "Bias (°C, + = too warm)", "{:+.2f}", None),
    ("max_err", "Largest error (°C)", "{:.2f}", False),
    ("r2", "R² (variance explained)", "{:.3f}", True),
    ("within_1", "Hours within ±1 °C", "{:.0%}", True),
    ("within_2", "Hours within ±2 °C", "{:.0%}", True),
]


def stats_table(scores):
    """LSTM vs persistence for this forecast, one row per statistic."""
    s, p = scores["predicted"], scores["persistence"]
    rows = []
    for key, label, fmt, higher in STATS:
        a, b = s[key], p[key]
        if higher is None:              # bias: closer to zero wins
            a_wins = abs(a) < abs(b)
        else:
            a_wins = a > b if higher else a < b
        better = "–" if pd.isna(a) or pd.isna(b) or a == b else \
            ("LSTM" if a_wins else "persistence")
        fmt_ = lambda v: "–" if pd.isna(v) else fmt.format(v)
        rows.append([label, fmt_(a), fmt_(b), better])
    skill = 1 - s["mae"] / p["mae"] if p["mae"] > 0 else float("nan")
    rows.append(["Skill vs persistence (MAE)", "–" if pd.isna(skill) else f"{skill:+.0%}",
                 "0%", "LSTM" if skill > 0 else "persistence"])
    return pd.DataFrame(rows, columns=["Statistic", "LSTM", "Persistence", "Better"])


def test_metrics(name):
    """Held-out test-set scores written by train_lstm.py, if they were kept."""
    path = OUTPUT_DIR / ("metrics.csv" if name == "t2m" else f"{name}/metrics.csv")
    if not path.exists():
        return pd.DataFrame({"Note": [f"No training metrics found for '{name}' "
                                      f"({path.relative_to(OUTPUT_DIR.parents[1]).as_posix()})"]})
    df = pd.read_csv(path).set_index("model").T
    df.index = [{"mae": "MAE (°C)", "rmse": "RMSE (°C)", "bias": "Bias (°C)",
                 "r2": "R²"}.get(i, i) for i in df.index]
    return df.round(3).rename(columns={"lstm": "LSTM", "persistence": "Persistence"}) \
        .rename_axis("Statistic").reset_index()


MIN_DAY = (hist.index[0] + pd.Timedelta(hours=get_model(DEFAULT_MODEL).seed_hours)).date()
LAST_OBS = hist.index[-1]
MAX_DAY = (LAST_OBS + pd.Timedelta(days=FREE_RUN_CAP_DAYS)).date()
DEFAULT_DAY = "2025-07-01"


def chart(df, day, mode, n_days):
    """Interactive Plotly chart: hover any point to read its time and value."""
    mk = "lines+markers" if n_days == 1 else "lines"
    hover = "%{x|%d %b %H:%M}<br>%{y:.2f} °C"
    fig = go.Figure()
    fig.add_scatter(x=df.index, y=df["persistence"], name="persistence (last observed hour)",
                    mode="lines", line=dict(color=C_PERS, width=1.5, dash="dot"),
                    hovertemplate=hover)
    if df["actual"].notna().any():
        fig.add_scatter(x=df.index, y=df["actual"], name="ERA5 observed", mode=mk,
                        line=dict(color=C_OBS, width=2.5), marker=dict(size=5),
                        hovertemplate=hover)
    fig.add_scatter(x=df.index, y=df["predicted"], name="LSTM forecast", mode=mk,
                    line=dict(color=C_PRED, width=2.5, dash="dash"), marker=dict(size=5),
                    hovertemplate=hover)
    fig.update_layout(
        title=f"{day:%d %b %Y} - {len(df)} hourly steps issued at 00:00"
              + ("" if mode == "warm" else "  (free-running)"),
        yaxis_title="2 m temperature (deg C)",
        xaxis_title="Time (UTC)",
        hovermode="x unified",
        legend=dict(orientation="h", y=-0.2),
        margin=dict(t=50, b=40),
        template="plotly_white",
    )
    return fig


def run(day_text, n_days, model_name=DEFAULT_MODEL):
    try:
        day = pd.Timestamp(str(day_text).strip()).normalize()
    except (ValueError, TypeError):
        raise gr.Error("Enter the date as YYYY-MM-DD, e.g. 2025-07-01.")
    n_days = int(n_days)
    fc = get_model(model_name)

    res = forecast_day(fc, hist, day, n_days)
    if "error" in res:
        raise gr.Error(res["error"])

    df, mode = res["frame"], res["mode"]
    status = f"Model '{fc.name}': {fc.describe()}.\n"
    if mode == "warm":
        status += (f"Warm start: seeded with the {fc.lookback} observed hours ending "
                   f"{res['issued']:%d %b %Y %H:%M}, then run {res['steps']} steps.")
    else:
        status += (f"Free-running: no inputs for this model near {day:%Y-%m-%d}, so it "
                   f"fed on its own output for {res['free_hours']:,} h. Errors compound "
                   f"- read this as model behaviour, not a forecast.")

    heading = (f"### Scoring: model **{fc.name}**, {day:%d %b %Y}, "
               f"{n_days} day{'s' if n_days > 1 else ''} "
               f"({'warm start' if mode == 'warm' else 'free-running'})\n")
    if res["scores"]:
        status += f"\n{res['n_scored']} of {res['steps']} hours observed - see the Statistics tab."
        heading += f"{res['n_scored']} of {res['steps']} hours have observations to score against."
        stats = stats_table(res["scores"])
    else:
        status += "\nNone of these hours are observed, so there is nothing to score against."
        heading += "None of these hours are observed."
        stats = pd.DataFrame({"Note": ["No observations in this window."]})

    fig = chart(df, day, mode, n_days)
    table = df.round(2).reset_index()
    table["valid_time"] = table["valid_time"].dt.strftime("%Y-%m-%d %H:%M")
    return fig, status, table, heading, stats, test_metrics(fc.name)


with gr.Blocks(title="ERA5 t2m LSTM") as demo:
    gr.Markdown(
        f"# ERA5 2 m temperature - LSTM day forecast\n"
        f"Domain mean {DOMAIN} · hourly ERA5 · the last few days in, one hour "
        f"predicted at a time and fed back in. **t2m** (V2) reads temperature only; "
        f"**v3** also reads humidity, wind and the heat the wind carries in.\n\n"
        f"Observations run to **{LAST_OBS:%Y-%m-%d %H:%M}**. Forecastable days: "
        f"{MIN_DAY} to {MAX_DAY} (days past the observations are free-running).")
    with gr.Row():
        day_in = gr.Textbox(DEFAULT_DAY, label="First day to forecast (YYYY-MM-DD)")
        days_in = gr.Slider(1, 7, value=1, step=1, label="Forecast length (days)")
        model_in = gr.Dropdown(MODELS, value=DEFAULT_MODEL, label="Model")
        btn = gr.Button("Forecast", variant="primary")
    # the controls above drive both tabs: one forecast, two views of it
    with gr.Tabs():
        with gr.Tab("Forecast"):
            plot_out = gr.Plot(label="Forecast vs actual")
            status_out = gr.Textbox(label="Mode", lines=4)
            with gr.Accordion("Hourly values", open=False):
                table_out = gr.Dataframe()

        with gr.Tab("Statistics"):
            heading_out = gr.Markdown()
            stats_out = gr.Dataframe(label="Accuracy of this forecast vs persistence "
                                           "(last observed hour held flat)")
            test_out = gr.Dataframe(
                label="Model test-set accuracy (from training: 1 h ahead, held-out test period)")
            gr.Markdown(
                "The two tables answer different questions. The first scores the "
                "whole rollout you picked, where persistence cannot follow the "
                "day-night cycle. The second is one hour ahead, where temperature "
                "barely changes and persistence is hard to beat.")

    inputs = [day_in, days_in, model_in]
    outputs = [plot_out, status_out, table_out, heading_out, stats_out, test_out]
    btn.click(run, inputs, outputs)
    model_in.change(run, inputs, outputs)
    demo.load(run, inputs, outputs)

if __name__ == "__main__":
    demo.launch()
