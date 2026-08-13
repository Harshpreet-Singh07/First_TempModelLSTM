"""
The trained LSTM as ~40 lines of numpy, so the deployed app needs no TensorFlow.

The network is one LSTM cell (64 units) plus a Dense(1) head - 16,961
parameters. A forward pass is four matrix multiplies and a handful of
elementwise ops per timestep, which numpy does perfectly well. Training still
uses Keras (train_lstm.py); this is inference only.

Why it exists: Streamlit Community Cloud builds on Python 3.14, and TensorFlow
publishes no wheels for it - so the hosted app cannot import TensorFlow at all.
Exported weights sidestep that, and drop the deploy from ~600 MB to numpy.

    python lstm/lstm_numpy.py        # export weights from the .keras model,
                                     # then verify numpy == Keras

Weight layout, as Keras stores it:
    kernel            (n_features, 4 * units)     input -> gates
    recurrent_kernel  (units,      4 * units)     state -> gates
    bias              (4 * units,)
Gates are concatenated in the order i, f, c, o. With unit_forget_bias the +1 on
the forget gate is already baked into `bias`, so inference needs no special case.
"""

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "models" / "lstm_t2m.keras"
WEIGHTS = HERE / "models" / "lstm_t2m_weights.npz"


def _sigmoid(x):
    # exp of a large positive x overflows; the algebraically identical branch
    # below keeps both tails finite
    out = np.empty_like(x)
    pos, neg = x >= 0, x < 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[neg])
    out[neg] = ex / (1.0 + ex)
    return out


class NumpyLSTM:
    """LSTM(units) -> Dense(1), forward pass only. Dropout is a no-op here."""

    def __init__(self, path=WEIGHTS):
        if not path.exists():
            raise FileNotFoundError(
                f"{path} not found - run: python lstm/lstm_numpy.py")
        w = np.load(path)
        self.W = w["kernel"].astype("float64")            # (features, 4u)
        self.U = w["recurrent_kernel"].astype("float64")  # (units,    4u)
        self.b = w["bias"].astype("float64")              # (4u,)
        self.Wd = w["dense_kernel"].astype("float64")     # (units, 1)
        self.bd = w["dense_bias"].astype("float64")       # (1,)
        self.units = self.U.shape[0]

    def __call__(self, x):
        """x: (batch, timesteps, features) -> (batch,) predictions."""
        x = np.asarray(x, dtype="float64")
        if x.ndim == 2:
            x = x[None, ...]
        n, steps, _ = x.shape
        u = self.units

        h = np.zeros((n, u))
        c = np.zeros((n, u))
        for t in range(steps):
            z = x[:, t, :] @ self.W + h @ self.U + self.b
            i = _sigmoid(z[:, :u])
            f = _sigmoid(z[:, u:2 * u])
            g = np.tanh(z[:, 2 * u:3 * u])
            o = _sigmoid(z[:, 3 * u:])
            c = f * c + i * g
            h = o * np.tanh(c)

        return (h @ self.Wd + self.bd).ravel()


# ------------------------------------------------------------------- export
def export(model_path=MODEL_PATH, out=WEIGHTS):
    """Pull the weights out of the .keras file into a small npz."""
    import keras                                   # only needed to export

    model = keras.models.load_model(model_path)
    lstm = next(l for l in model.layers if l.__class__.__name__ == "LSTM")
    dense = next(l for l in model.layers if l.__class__.__name__ == "Dense")

    if lstm.activation.__name__ != "tanh" or \
            lstm.recurrent_activation.__name__ != "sigmoid":
        raise ValueError(
            f"this exporter assumes tanh/sigmoid activations, but the layer uses "
            f"{lstm.activation.__name__}/{lstm.recurrent_activation.__name__}")

    kernel, recurrent, bias = lstm.get_weights()
    dk, db = dense.get_weights()
    np.savez(out, kernel=kernel, recurrent_kernel=recurrent, bias=bias,
             dense_kernel=dk, dense_bias=db)
    print(f"exported {kernel.size + recurrent.size + bias.size + dk.size + db.size:,}"
          f" parameters -> {out}  ({out.stat().st_size / 1024:.0f} KB)")
    return model


def verify(model, tol=1e-5, n=64):
    """numpy and Keras must agree on random windows, or the export is wrong."""
    scaler = json.loads((HERE / "models" / "scaler_t2m.json").read_text())
    rng = np.random.default_rng(0)
    x = rng.normal(size=(n, scaler["lookback"], 1)).astype("float32")

    ours = NumpyLSTM()(x)
    theirs = model.predict(x, verbose=0).ravel()
    diff = float(np.max(np.abs(ours - theirs)))

    print(f"max |numpy - keras| over {n} random windows: {diff:.2e}")
    if diff > tol:
        raise SystemExit(f"MISMATCH: {diff:.2e} > {tol:.0e} - do not deploy this")
    print("match within tolerance")


if __name__ == "__main__":
    verify(export())
