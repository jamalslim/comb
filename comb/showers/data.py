"""
Calorimeter shower profiles with exactly the splits of the CoMB paper.

  X       all showers (d = 12: 47 682 x 12; d = 25: 47 682 x 25)
  subset  6 000 showers drawn with default_rng(123)
  train / test   sklearn train_test_split(test_size=0.2, random_state=42)
                 -> 4 800 / 1 200
  fit / val      800 of the 4 800 training showers drawn with default_rng(8)
                 (CoMB: seed + 1 with seed = 7) -> 4 000 / 800
  unseen         the 41 682 showers outside the 6 000-shower subset, split
                 (default_rng(2026)) into a validation set of 8 000 (model
                 selection for every model trained in this study) and a test
                 set of 33 682 (all reported metrics)

`load("d12")` reproduces CoMB's Y_tr, Y_te and the token-level fit/val split
row for row (checked in scripts/test_data.py against the comb package).
"""
from __future__ import annotations

import os
import pathlib

import numpy as np
from sklearn.model_selection import train_test_split

DATA_DIR = pathlib.Path(os.environ.get("COMB_DATA", pathlib.Path(__file__).resolve().parent.parent / "data"))
FILES = {"d12": "cal_shower_img_12q.npy", "d25": "cal_shower_img_25q.npy", "ds1": "cal_ds1_photons_65GeV.npy"}


def load(tag: str, subset_n: int = 6000, subset_seed: int = 123, test_size: float = 0.2,
         split_seed: int = 42, val_n: int = 800, val_seed: int = 8, v8_n: int = 8000, v8_seed: int = 2026):
    X = np.load(DATA_DIR / FILES[tag]).astype(np.float64)
    used = np.random.default_rng(subset_seed).choice(len(X), size=min(subset_n, len(X)), replace=False)
    Xs = X[used]
    tr, te = train_test_split(np.arange(len(Xs)), test_size=test_size, random_state=split_seed)
    Y_tr, Y_te = Xs[tr], Xs[te]
    vi = np.random.default_rng(val_seed).choice(len(Y_tr), size=val_n, replace=False)
    vmask = np.zeros(len(Y_tr), bool)
    vmask[vi] = True
    unseen = np.ones(len(X), bool)
    unseen[used] = False
    Y_un = X[unseen]
    perm = np.random.default_rng(v8_seed).permutation(len(Y_un))
    v8_n = min(v8_n, len(Y_un) // 2)   # Dataset 1: 4000 unseen -> 2000 validation, 2000 unseen test
    return dict(Y_tr=Y_tr, Y_te=Y_te, Y_fit=Y_tr[~vmask], Y_val=Y_tr[vmask], Y_un=Y_un,
                Y_v8=Y_un[perm[:v8_n]], Y_test=Y_un[perm[v8_n:]])
