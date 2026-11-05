#!/usr/bin/env python3
"""The Kraus form and the superoperator form of the exact characteristic function are the same function:
values and reverse-mode gradients must agree to rounding, and the gradient must match finite differences."""
import sys, pathlib, numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from comb.showers.model import CoMB
rng = np.random.default_rng(3)
ok = True
for n_m, n_w, cur in ((1, 2, "binary"), (2, 3, "binary"), (3, 2, "binary"), (2, 3, "hamming")):
    m = CoMB(6, n_m=n_m, n_w=n_w, n_sub=1, L_v=2, L_w=1, seed=1, current=cur)
    m.h = rng.normal(0, 0.5, m.h.shape)
    om = rng.normal(0, 0.8, (7, 6))
    g_out = rng.normal(size=7) + 1j * rng.normal(size=7)
    m.cf_path = "superoperator"
    p0, g0 = m.characteristic_function(om, grad_out=g_out)
    m.cf_path = "kraus"
    p1, g1 = m.characteristic_function(om, grad_out=g_out)
    q1 = m.characteristic_function(om)
    e = max(np.abs(p1 - p0).max(), np.abs(g1 - g0).max(), np.abs(q1 - p0).max())
    ok &= e < 1e-12
    print(f"n_m={n_m} n_w={n_w} {cur:7}  max difference {e:.1e}")
m = CoMB(5, n_m=2, n_w=3, n_sub=1, L_v=2, L_w=1, seed=2, current="binary")
m.cf_path = "kraus"
m.h = rng.normal(0, 0.5, m.h.shape)
om = rng.normal(0, 0.8, (5, 5))
g_out = rng.normal(size=5) + 1j * rng.normal(size=5)
L = lambda: float(np.real(np.conj(g_out) * m.characteristic_function(om)).sum())
p = m.get()
_, g = m.characteristic_function(om, grad_out=g_out)
for idx in (3, len(p) - 4):
    q = p.copy()
    q[idx] += 1e-6
    m.set(q)
    Lp = L()
    q[idx] -= 2e-6
    m.set(q)
    Lm = L()
    m.set(p)
    fd = (Lp - Lm) / 2e-6
    ok &= abs(fd - g[idx]) < 1e-7
    print(f"parameter {idx}: finite difference {fd:+.9f}, reverse mode {g[idx]:+.9f}")
print("ALL PASS" if ok else "FAILED")
sys.exit(0 if ok else 1)
