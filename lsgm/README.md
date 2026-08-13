# LSGM on ERA5 2025

Setup for training [NVlabs/LSGM](https://github.com/NVlabs/LSGM) (Score-based
Generative Modeling in Latent Space) on the ERA5 fields in this project.

Nothing here clones the repo, installs anything, or starts training - those are
your calls to make. What is prepared is the part that is specific to this data:
turning the GRIB into tensors LSGM can read, and the `Dataset` class to register.

## Layout

```
lsgm/
  src/export_era5_tensors.py   GRIB  -> [N, 5, H, W] float32 array in [0,1]
  src/era5_dataset.py          PyTorch Dataset + registration snippet
  data/                        exported tensors + normalisation.json
  models/                      VAE and diffusion checkpoints (.pt)
  notebooks/                   exploration
  output/                      generated samples, decoded back to physical units
```

The LSGM repo itself should be cloned **outside** this folder (or into
`lsgm/LSGM/`), and left as an unmodified checkout apart from the two edits in
step 2.

## The data, as LSGM sees it

Each hour of 2025 becomes one "image":

| | |
|---|---|
| samples | 8,760 (hourly, 1 Jan - 31 Dec 2025) |
| channels | 5 - `t2m`, `d2m`, `u10`, `v10`, `tp` (fixed order) |
| native grid | 62 lat x 84 lon |
| exported grid | 56 x 80 (centre-cropped; see below) |
| value range | [0, 1] per channel |

### Two decisions worth reviewing before you commit to a run

**1. Spatial size.** The native grid is 62 x 84. NVAE-style encoders halve the
resolution at each scale, so side lengths need to divide by a decent power of
two - and 62 = 2x31, 84 = 4x21 do not. The export defaults to a centre crop to
**56 x 80** (both divisible by 8), which discards ~13% of the domain area at the
edges. `--mode pad --size 64x96` keeps every real value instead, at the cost of
feeding the model a border of duplicated rows that it will learn to reproduce.
Neither is free; the crop is the default only because fabricated borders tend to
show up in generated samples.

**2. Per-channel scaling.** The variables span very different ranges (`t2m`
-24..31 C, `tp` 0..0.4 mm/hr). Each channel is mapped to [0, 1] independently
using 0.1/99.9 percentile limits, so no channel dominates the reconstruction
loss. `tp` is `log1p`-compressed first - it is strongly right-skewed
(skewness +1.68), and without it the model sees a near-constant zero field with
rare spikes. All constants are stored in `data/normalisation.json`, and
`ERA5Dataset.to_physical()` inverts them.

A caveat that no amount of preprocessing fixes: **8,760 samples is a small
dataset** for a latent diffusion model, and consecutive hours are nearly
identical, so the effective sample count is far lower than the nominal one.
Expect to lean on the VAE being small and on heavy regularisation, and treat
early results sceptically.

## Steps

Script names (`train_vae.py`, `train_vada.py`) and the flags `--data`,
`--dataset`, `--root`, `--save`, `--vae_checkpoint` are verified against the
upstream README. Everything specific to ERA5 - the `era5` dataset name, the
5-channel handling, the registration snippet - is this project's own and is not
upstream.

### 1. Clone and install

```bash
git clone https://github.com/NVlabs/LSGM.git
cd LSGM
pip install -r requirements.txt
```

**Read this before budgeting time.** The upstream README specifies
**Python 3.8 with PyTorch 1.8.0**, and the repo dates from 2021. PyTorch 1.8
has no build for Python 3.12 (the interpreter this project's analysis scripts
use), so LSGM needs its own environment - conda with Python 3.8, or a container.
Do not try to share an environment between the two halves of this project.

The published configs assume **2-8 V100 GPUs per node**. A single consumer GPU
or a free Colab tier will need the batch size and channel counts reduced, which
changes results.

NVIDIA Apex is listed as **optional**, and specifically for a faster fused Adam
optimizer - not as a requirement for mixed-precision training. Skip it if the
build fights you; you lose some speed, not correctness.

### 2. Export the tensors and register the dataset

```bash
python lsgm/src/export_era5_tensors.py                 # full year, ~785 MB
python lsgm/src/export_era5_tensors.py --limit 48      # quick smoke test
python lsgm/src/export_era5_tensors.py --mode pad --size 64x96   # keep all data
```

Then copy `src/era5_dataset.py` into the clone as `LSGM/util/era5_dataset.py`
and add the branch to `util/datasets.py`. Run the file directly to print the
snippet:

```bash
python lsgm/src/era5_dataset.py
```

**Caveat:** the upstream README does not document custom-dataset registration
at all - it points at the NVAE repository for data preparation. The snippet in
`era5_dataset.py` is inferred from how the built-in datasets are wired, so treat
it as a starting point and expect to adapt it to the code you actually find in
`util/datasets.py`. It is the least certain part of this setup.

Note that ERA5 has **5 channels, not 3** - anywhere the repo assumes RGB needs
the channel count threaded through.

### 3. Train the VAE backbone

Builds the latent space the diffusion model will operate in. Outputs a `.pt`
checkpoint.

```bash
python train_vae.py --data <repo-root>/lsgm/data --dataset era5 \
    --root lsgm/models --save vae_era5 --num_channels 5
```

### 4. Train the latent diffusion model

Pass the VAE checkpoint from step 3:

```bash
python train_vada.py --data <repo-root>/lsgm/data --dataset era5 \
    --vae_checkpoint lsgm/models/vae_era5/checkpoint.pt \
    --root lsgm/models --save vada_era5
```

### 5. Generate and decode

Sampling produces latents, which the trained VAE decoder expands back to
5 x 56 x 80 fields in [0, 1]. Map them to physical units with:

```python
from era5_dataset import ERA5Dataset
ds = ERA5Dataset("lsgm/data", split="all")
physical = ds.to_physical(sample)   # deg C, m/s, mm/hour
```

Worth checking on the output before trusting it: whether `d2m <= t2m` holds
everywhere (dewpoint above temperature is unphysical), whether `tp >= 0`, and
whether the wind fields stay spatially coherent. Nothing in the training
objective enforces any of these, so they are a useful honesty check on samples.
