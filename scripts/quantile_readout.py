#!/usr/bin/env python3
"""Set the readout of chosen coordinates by the quantile rule, keeping everything else (circuit included) unchanged.

    python scripts/quantile_readout.py --run outputs/runs/ds1_comb_chi4_s0_hw12.npz --coords 0 --out outputs/runs/ds1_comb_chi4_s0_hw12q.npz

Given the circuit's exact token probabilities p_j(c) of coordinate j, the edges of bin c are placed at the data
quantiles of the cumulative probability (the readout initialization of training). For one coordinate this is a
monotone change of that coordinate alone, so its rank correlations with all others are kept. Use it only on a few
leading coordinates: applied to many coordinates at once it breaks the co-adaptation of the readouts (on Dataset 1,
coordinate 0 alone helps; coordinates 0-3 or more break the model).
"""
import argparse, copy, pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import comb_ds1 as cd                                                             # noqa: E402
from comb.showers import zoo                                                              # noqa: E402
ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--run", required=True)
ap.add_argument("--coords", default="0")
ap.add_argument("--out", required=True)
a = ap.parse_args()
cd.need_shower_project(a.run)
D = cd.load_data()
m0, s0 = D["Y_tr"].mean(0), D["Y_tr"].std(0)
model, spec, extra = zoo.load(a.run)
b = model.base
th = [np.array(x, copy=True) for x in (b.theta_v, b.theta_w)]
q = copy.deepcopy(model)
q.init_readout(D["Y_fit"], m=m0, s=s0)
coords = [int(c) for c in a.coords.split(",")]
for j in coords:
    assert np.isclose(q.base.m[j], b.m[j]) and np.isclose(q.base.s[j], b.s[j]), "standardization must not change"
    b.lo[j], b.hi[j], b.h[j] = q.base.lo[j], q.base.hi[j], q.base.h[j].copy()
assert all(np.array_equal(x, y) for x, y in zip(th, (b.theta_v, b.theta_w))), "the circuit must not change"
zoo.save(a.out, model, spec, dict(extra, readout_quantile_coordinates=coords))
print(f"quantile readout on coordinates {coords}; circuit unchanged -> {a.out}")
