"""
Input features for the LSTM - shared by training (train_lstm.py) and inference
(forecast.py), so the two can never build them differently.

A model is described by its `features`, saved in its scaler JSON. Each one is:

  a CSV column        t2m, d2m, u10, v10, t2m_adv, ... (data/processed/*.csv)
  <column>_d1         that column's change over the last hour, e.g. t2m_d1
  calendar            hour_sin, hour_cos (time of day), doy_sin, doy_cos (time of year)

The model predicts the next-hour change of every CSV column it reads (its
`outputs`, see outputs_for). A multi-hour forecast feeds its own predictions
back in, so any input that is not known in advance - the wind, the advection -
has to be predicted along with the temperature. Calendar features are known for
any hour, so they are not.

The V2 model is the special case features = ["t2m"].
"""

import numpy as np

CALENDAR = ("hour_sin", "hour_cos", "doy_sin", "doy_cos")
DIFF = "_d1"

# V3 default: temperature, its last-hour change, humidity, the wind, the heat the
# wind carries in (t2m_adv, scripts/era5_series.py), and where in the day/year we are
DEFAULT_FEATURES = ["t2m", "t2m_d1", "d2m", "u10", "v10", "t2m_adv",
                    "hour_sin", "hour_cos", "doy_sin", "doy_cos"]


def outputs_for(features):
    """CSV columns the features are built from, t2m first - what the model predicts."""
    cols = []
    for f in features:
        if f in CALENDAR:
            continue
        c = f.removesuffix(DIFF)
        if c not in cols:
            cols.append(c)
    if "t2m" not in cols:
        raise ValueError(f"features {features} do not include t2m, the forecast target")
    cols.remove("t2m")
    return ["t2m"] + cols


def extra_hours(features):
    """Hours needed before the window: an hourly change needs the hour before it."""
    return 1 if any(f.endswith(DIFF) for f in features) else 0


def build_array(times, base, columns, features):
    """Raw (unscaled) features, shape (hours, len(features)).

    times   hourly DatetimeIndex, gap-free (missing hours are NaN rows in base)
    base    (hours, len(columns)) array of CSV values
    Rows whose inputs are missing come out NaN - and the first row of every _d1
    feature always does - so callers skip any window that touches one.
    """
    col = {c: base[:, i] for i, c in enumerate(columns)}
    hour = np.asarray(times.hour, dtype="float64")
    doy = (np.asarray(times.dayofyear, dtype="float64") - 1 + hour / 24) / 365.25

    out = np.empty((len(times), len(features)))
    for j, f in enumerate(features):
        if f == "hour_sin":
            out[:, j] = np.sin(2 * np.pi * hour / 24)
        elif f == "hour_cos":
            out[:, j] = np.cos(2 * np.pi * hour / 24)
        elif f == "doy_sin":
            out[:, j] = np.sin(2 * np.pi * doy)
        elif f == "doy_cos":
            out[:, j] = np.cos(2 * np.pi * doy)
        elif f.endswith(DIFF):
            v = col[f.removesuffix(DIFF)]
            out[0, j] = np.nan
            out[1:, j] = v[1:] - v[:-1]
        else:
            out[:, j] = col[f]
    return out


def build(frame, features):
    """build_array for a DataFrame holding outputs_for(features) as columns."""
    columns = outputs_for(features)
    missing = [c for c in columns if c not in frame]
    if missing:
        raise KeyError(f"features need column(s) {missing}, which the data does not have")
    return build_array(frame.index, frame[columns].to_numpy("float64"), columns, features)
