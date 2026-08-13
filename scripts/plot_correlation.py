"""
Correlation matrices for the ERA5 2025 domain-mean series.

Two methods, one figure each:
  Pearson  - linear association between the raw values
  Spearman - the same formula applied to the RANKS of the values, so it
             measures monotonic association and is insensitive to outliers
             and to any curvature in the relationship

Variables are the five plotted individually, all area-weighted
(cos-latitude) domain means over 41.75-57.0 N, 74.25-95.0 E:

  t2m  d2m  u10  v10  tp

Each matrix is computed on the hourly series (n = 8760) and, for comparison,
on daily means (n = 365).  Hourly weather data is strongly autocorrelated,
so these values describe how the variables co-vary across the year - they
are not independent samples and should not be fed into a naive p-value.

Outputs
  output/06_pearson_correlation_matrix.png    output/pearson_correlation_matrix.csv
  output/07_spearman_correlation_matrix.png   output/spearman_correlation_matrix.csv
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from era5_series import load, DOMAIN, PERIOD, FIGURES, TABLES
from omni_series import load_kp

FIGURES.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})

VARS = ["t2m", "d2m", "u10", "v10", "tp", "kp"]
LABELS = {"t2m": "t2m", "d2m": "d2m", "u10": "u10",
          "v10": "v10", "tp": "tp", "kp": "Kp"}
LEGEND = ("t2m = 2 m temperature   |   d2m = 2 m dewpoint   |   "
          "u10 = 10 m U wind   |   v10 = 10 m V wind   |   tp = total precipitation\n"
          "Kp = OMNI2 planetary geomagnetic index (global, 3-hourly, "
          "repeated across each block)")

METHODS = {
    "pearson": ("06", "Pearson", "linear association"),
    "spearman": ("07", "Spearman", "rank / monotonic association"),
}

df = load()
df["kp"] = load_kp().reindex(df.index)
df = df[VARS]
daily = df.resample("1D").mean()


def draw(ax, m, title):
    """Annotated heatmap of a correlation matrix, diverging about zero."""
    im = ax.imshow(m.values, cmap="RdBu_r", vmin=-1, vmax=1)
    n = len(m)
    ax.set_xticks(range(n), [LABELS[c] for c in m.columns], fontsize=11)
    ax.set_yticks(range(n), [LABELS[c] for c in m.index], fontsize=11)
    ax.set_xticks(np.arange(n + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(n + 1) - 0.5, minor=True)
    ax.grid(which="minor", color="w", lw=1.5)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)

    for i in range(n):
        for j in range(n):
            r = m.values[i, j]
            # white text on the saturated corners, black in the pale middle
            ax.text(j, i, f"{r:.2f}", ha="center", va="center", fontsize=11,
                    fontweight="bold" if i != j else "normal",
                    color="white" if abs(r) > 0.55 else "black")
    ax.set_title(title)
    return im


def build(method):
    num, name, blurb = METHODS[method]
    corr = df.corr(method=method)
    corr_daily = daily.corr(method=method)

    fig, axes = plt.subplots(1, 2, figsize=(13, 6.0))
    fig.subplots_adjust(wspace=0.18)
    im = draw(axes[0], corr, f"Hourly series  (n = {len(df):,})")
    draw(axes[1], corr_daily, f"Daily means  (n = {len(daily):,})")

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, pad=0.02)
    cbar.set_label(f"{name} correlation", fontsize=11)
    cbar.set_ticks([-1, -0.5, 0, 0.5, 1])

    fig.suptitle(f"ERA5 domain-mean {name} correlation ({blurb})  |  "
                 f"{DOMAIN}  |  {PERIOD}", fontsize=13, fontweight="bold", y=0.97)
    fig.text(0.5, 0.02, LEGEND, ha="center", fontsize=9.5, color="#444444")

    png = FIGURES / f"{num}_{method}_correlation_matrix.png"
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    corr.to_csv(TABLES / f"{method}_correlation_matrix.csv", float_format="%.4f")

    print(f"\n{name} correlation - hourly domain means (n = {len(df):,})")
    print(corr.round(3).to_string())
    print(f"\n{name} correlation - daily domain means (n = {len(daily):,})")
    print(corr_daily.round(3).to_string())
    print("Saved:", png)
    return corr


mats = {m: build(m) for m in METHODS}

# ------------------------------------------------- where the two methods differ
pear, spear = mats["pearson"], mats["spearman"]
mask = np.triu(np.ones(pear.shape), k=1).astype(bool)
p = pear.where(mask).stack().dropna()
s = spear.where(mask).stack().dropna()
diff = (s - p).sort_values(key=np.abs, ascending=False)

print("\nPearson vs Spearman (hourly), ordered by |difference|:")
print(f"  {'pair':<14}{'pearson':>9}{'spearman':>10}{'diff':>9}")
for (a, b), dv in diff.items():
    print(f"  {a + ' vs ' + b:<14}{p[(a, b)]:>9.3f}{s[(a, b)]:>10.3f}{dv:>+9.3f}")
