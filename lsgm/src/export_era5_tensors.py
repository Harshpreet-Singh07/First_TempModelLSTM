"""
Export the ERA5 GRIB into image-like tensors for NVlabs/LSGM.

LSGM (and the NVAE backbone it builds on) consumes [channels, height, width]
tensors, so each hourly ERA5 snapshot becomes one "image":

    channels = 5   t2m, d2m, u10, v10, tp   (in this fixed order)
    height   = latitude   (62 native)
    width    = longitude  (84 native)
    samples  = 8760       (one per hour of 2025)

Two things have to happen before the data is usable:

1. SPATIAL SIZE.  The native grid is 62 x 84.  NVAE-style encoders halve the
   resolution at every scale, so the side lengths must be divisible by a
   reasonably large power of two - 62 and 84 are not (62 = 2*31, 84 = 4*21).
   --mode crop  centre-crops to --size (default 56 x 80, both divisible by 8)
   --mode pad   edge-pads to --size instead, inventing no interior values
   Cropping discards real data at the edges; padding fabricates none but does
   feed the model a border of duplicated rows.  Cropping is the default.

2. VALUE RANGE.  The five variables live on wildly different scales
   (t2m ~ -24..31 C, tp ~ 0..0.4 mm/hr).  Each channel is mapped to [0, 1]
   independently using robust percentile limits, so no single channel
   dominates the reconstruction loss.  Precipitation is log1p-compressed
   first because it is strongly right-skewed (skewness +1.68).

   The exact constants are written to normalisation.json so that generated
   samples can be mapped back to physical units.

Output (default): lsgm/data/era5_2025.npy   float32 [N, 5, H, W]
                  lsgm/data/normalisation.json
                  lsgm/data/timestamps.csv

Run:
    python lsgm/src/export_era5_tensors.py            # full year
    python lsgm/src/export_era5_tensors.py --limit 48 # quick smoke test
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))

GRIB = ROOT / "data" / "raw" / "185bc0671d5cc50980a767e3a9f6cbe5.grib"
OUTDIR = ROOT / "lsgm" / "data"

CHANNELS = ["t2m", "d2m", "u10", "v10", "tp"]
LOG_CHANNELS = {"tp"}          # right-skewed -> log1p before scaling
KELVIN_CHANNELS = {"t2m", "d2m"}


def open_instant():
    return xr.open_dataset(GRIB, engine="cfgrib", backend_kwargs={
        "filter_by_keys": {"stepType": "instant"}, "indexpath": ""})


def open_accum():
    return xr.open_dataset(GRIB, engine="cfgrib", backend_kwargs={
        "filter_by_keys": {"stepType": "accum"}, "indexpath": ""})


def fit_grid(a, size, mode):
    """Centre-crop or edge-pad the trailing two dims of `a` to `size`."""
    th, tw = size
    h, w = a.shape[-2:]
    if mode == "crop":
        if th > h or tw > w:
            raise SystemExit(f"--size {th}x{tw} exceeds the native grid {h}x{w}; use --mode pad")
        t, l = (h - th) // 2, (w - tw) // 2
        return a[..., t:t + th, l:l + tw]
    if th < h or tw < w:
        raise SystemExit(f"--size {th}x{tw} is smaller than the native grid {h}x{w}; use --mode crop")
    pt, pl = (th - h) // 2, (tw - w) // 2
    pad = [(0, 0)] * (a.ndim - 2) + [(pt, th - h - pt), (pl, tw - w - pl)]
    return np.pad(a, pad, mode="edge")


def build_precip_index(ds_a):
    """tp arrives as (time, step) forecast blocks; map each valid_time to its slot."""
    vt = pd.DatetimeIndex(ds_a["valid_time"].values.ravel())
    n_step = ds_a.sizes["step"]
    flat = np.arange(vt.size)
    return pd.Series(flat, index=vt).groupby(level=0).first(), n_step


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="56x80", help="target HxW (default 56x80)")
    p.add_argument("--mode", choices=["crop", "pad"], default="crop")
    p.add_argument("--limit", type=int, default=None, help="only export the first N hours")
    p.add_argument("--out", default=None, help="output .npy path")
    p.add_argument("--chunk", type=int, default=744, help="hours read per pass")
    p.add_argument("--lo", type=float, default=0.1, help="lower percentile for scaling")
    p.add_argument("--hi", type=float, default=99.9, help="upper percentile for scaling")
    a = p.parse_args()

    th, tw = (int(v) for v in a.size.lower().split("x"))
    OUTDIR.mkdir(parents=True, exist_ok=True)
    out_path = Path(a.out) if a.out else OUTDIR / "era5_2025.npy"

    ds_i, ds_a = open_instant(), open_accum()
    times = pd.DatetimeIndex(ds_i["valid_time"].values)
    keep = (times >= "2025-01-01") & (times <= "2025-12-31 23:00")
    idx = np.flatnonzero(keep)
    if a.limit:
        idx = idx[:a.limit]
    times = times[idx]
    n = idx.size

    tp_slot, _ = build_precip_index(ds_a)
    tp_flat_shape = ds_a["tp"].shape          # (time, step, lat, lon)

    print(f"grid {ds_i.sizes['latitude']}x{ds_i.sizes['longitude']} -> {th}x{tw} ({a.mode})")
    print(f"samples: {n:,}   channels: {CHANNELS}")

    arr = np.lib.format.open_memmap(out_path, mode="w+", dtype=np.float32,
                                    shape=(n, len(CHANNELS), th, tw))

    # ---- pass 1: read, convert units, fit the grid
    for s in range(0, n, a.chunk):
        e = min(s + a.chunk, n)
        sel = idx[s:e]
        for c, name in enumerate(CHANNELS):
            if name == "tp":
                slots = tp_slot.reindex(times[s:e]).to_numpy()
                if np.isnan(slots).any():
                    raise SystemExit("tp is missing hours that the instant fields cover")
                ti, si = np.divmod(slots.astype(int), tp_flat_shape[1])
                block = ds_a["tp"].values[ti, si] * 1000.0        # m -> mm
            else:
                block = ds_i[name].values[sel]
                if name in KELVIN_CHANNELS:
                    block = block - 273.15                        # K -> deg C
            arr[s:e, c] = fit_grid(block.astype(np.float32), (th, tw), a.mode)
        print(f"  hours {s:>5}-{e:>5} done", flush=True)

    ds_i.close()
    ds_a.close()

    # ---- pass 2: per-channel transform + scaling to [0, 1]
    stats = {}
    for c, name in enumerate(CHANNELS):
        ch = arr[:, c]
        if name in LOG_CHANNELS:
            np.log1p(np.clip(ch, 0, None), out=ch)
        lo = float(np.percentile(ch, a.lo))
        hi = float(np.percentile(ch, a.hi))
        if hi <= lo:
            raise SystemExit(f"degenerate range for {name}: {lo} .. {hi}")
        np.clip((ch - lo) / (hi - lo), 0.0, 1.0, out=ch)
        stats[name] = {
            "index": c, "lo": lo, "hi": hi,
            "log1p": name in LOG_CHANNELS,
            "kelvin_offset": 273.15 if name in KELVIN_CHANNELS else 0.0,
            "units": "deg C" if name in KELVIN_CHANNELS else ("mm/hour" if name == "tp" else "m/s"),
        }
        print(f"  {name:4s} scaled from [{lo:.4f}, {hi:.4f}] "
              f"-> mean {float(ch.mean()):.3f}, std {float(ch.std()):.3f}")

    arr.flush()

    meta = {
        "channels": CHANNELS, "height": th, "width": tw, "samples": int(n),
        "mode": a.mode, "percentiles": [a.lo, a.hi],
        "source": GRIB.name, "array": out_path.name,
        "note": "values are in [0,1]; invert with x*(hi-lo)+lo, then expm1 if log1p",
        "stats": stats,
    }
    (OUTDIR / "normalisation.json").write_text(json.dumps(meta, indent=2))
    pd.Series(times, name="valid_time").to_csv(OUTDIR / "timestamps.csv", index=False)

    print(f"\nwrote {out_path}  ({out_path.stat().st_size / 1e6:.1f} MB)")
    print(f"wrote {OUTDIR / 'normalisation.json'}")
    print(f"wrote {OUTDIR / 'timestamps.csv'}")


if __name__ == "__main__":
    main()
