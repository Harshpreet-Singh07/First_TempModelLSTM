"""
Custom PyTorch Dataset for the ERA5 2025 tensors, for use with NVlabs/LSGM.

Copy this file into the cloned LSGM repo (e.g. LSGM/util/era5_dataset.py) and
register it in util/datasets.py - see REGISTRATION at the bottom of this file.

The array produced by export_era5_tensors.py is a float32 memmap of shape
[N, 5, H, W] already scaled to [0, 1], so __getitem__ is a cheap slice with no
per-sample normalisation work.  LSGM's training loop expects (image, label)
pairs; these fields are unlabelled, so a constant 0 is returned as the label.

Channel order is fixed: t2m, d2m, u10, v10, tp.
"""

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


class ERA5Dataset(Dataset):
    """ERA5 hourly fields as [C, H, W] tensors in [0, 1].

    Args:
        root:      directory holding era5_2025.npy + normalisation.json
        split:     'train' | 'valid' | 'all'
        valid_frac: fraction of samples held out at the END of the year, so the
                    split is chronological rather than random - a random split
                    would leak neighbouring hours across the boundary, and
                    consecutive hours are almost identical.
        augment:   random horizontal flip.  Off by default: these are physical
                   fields on a geographic grid, and mirroring longitude
                   reverses the sign of the u wind component, so a naive flip
                   produces states that violate the data's own physics.
    """

    def __init__(self, root, split="train", valid_frac=0.1, augment=False):
        root = Path(root)
        self.meta = json.loads((root / "normalisation.json").read_text())
        self.data = np.load(root / self.meta["array"], mmap_mode="r")

        n = self.data.shape[0]
        cut = int(round(n * (1.0 - valid_frac)))
        if split == "train":
            self.lo, self.hi = 0, cut
        elif split == "valid":
            self.lo, self.hi = cut, n
        elif split == "all":
            self.lo, self.hi = 0, n
        else:
            raise ValueError(f"unknown split {split!r}")

        self.split = split
        self.augment = augment
        self.channels = self.meta["channels"]

    def __len__(self):
        return self.hi - self.lo

    def __getitem__(self, i):
        x = np.asarray(self.data[self.lo + i], dtype=np.float32)  # [C, H, W]
        if self.augment and np.random.rand() < 0.5:
            x = x[:, :, ::-1].copy()
            u = self.channels.index("u10")
            x[u] = 1.0 - x[u]          # mirrored longitude flips the u sign
        return torch.from_numpy(x), 0

    # ---- inverse transform, for turning generated samples back into weather
    def to_physical(self, x):
        """Map a [C, H, W] tensor in [0, 1] back to physical units."""
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
        out = np.empty_like(x, dtype=np.float32)
        for name in self.channels:
            s = self.meta["stats"][name]
            v = x[s["index"]] * (s["hi"] - s["lo"]) + s["lo"]
            if s["log1p"]:
                v = np.expm1(v)
            out[s["index"]] = v
        return out


REGISTRATION = r"""
Register in the cloned repo's util/datasets.py
---------------------------------------------
1. At the top of util/datasets.py:

       from util.era5_dataset import ERA5Dataset

2. Inside get_loaders_eval(dataset, args), add a branch alongside the
   existing cifar10 / celeba / mnist cases:

       elif dataset == 'era5':
           resize = 56               # must match --size used at export time
           num_classes = 1
           train_data = ERA5Dataset(args.data, split='train')
           valid_data = ERA5Dataset(args.data, split='valid')

3. Anywhere the code switches on the number of input channels, ERA5 has 5
   (not 3).  Pass --num_channels 5 if the config exposes it, otherwise set
   num_input_channels for the 'era5' branch.

4. Point --data at lsgm/data when launching training.
"""

if __name__ == "__main__":
    import sys
    root = sys.argv[1] if len(sys.argv) > 1 else Path(__file__).resolve().parents[1] / "data"
    ds = ERA5Dataset(root, split="all")
    x, y = ds[0]
    print(f"samples: {len(ds):,}   sample shape: {tuple(x.shape)}   dtype: {x.dtype}")
    print(f"range: {float(x.min()):.3f} .. {float(x.max()):.3f}   label: {y}")
    print(f"channels: {ds.channels}")
    print("\nphysical units of sample 0, per channel (min / mean / max):")
    phys = ds.to_physical(x)
    for name in ds.channels:
        i = ds.meta["stats"][name]["index"]
        u = ds.meta["stats"][name]["units"]
        print(f"  {name:4s} {phys[i].min():8.2f} {phys[i].mean():8.2f} {phys[i].max():8.2f}  {u}")
    print(REGISTRATION)
