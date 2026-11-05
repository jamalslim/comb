#!/usr/bin/env python3
"""
Training infrastructure tests.
  1. A run interrupted at a checkpoint and resumed is bit-identical to an
     uninterrupted run (parameters, history, selected model).
  2. The per-step gradient used in training is the exact gradient of the
     per-step loss (finite differences through the whole training objective,
     including the PCA decoder).
  3. Model save/load round trip (comb.zoo) reproduces the characteristic
     function exactly.
"""
import os, pathlib, sys, tempfile
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from comb.showers import zoo
from comb.showers.mmd import Spectrum, empirical_cf
from comb.showers.train import train

FAILS = []
def report(name, err, tol):
    ok = err <= tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:60s} {err:.2e} (tol {tol:.0e})", flush=True)
    if not ok: FAILS.append(name)

rng = np.random.default_rng(0)
d = 4
Y = rng.normal(size=(1500, d)) @ rng.normal(size=(d, d)) + np.abs(rng.normal(size=(1500, d)))
Yv = rng.normal(size=(500, d)) @ rng.normal(size=(d, d))
spec = Spectrum(d, [1.0, 2.0])
m0, s0 = Y.mean(0), Y.std(0)
ms = dict(model="comb", basis="pca", n_m=1, n_w=2, n_sub=2, L_v=2, L_w=1, dephase=0, current="hamming", entangler="cz", seed=3)
hs = dict(model="hmm", basis="pixel", chi=3, K=5, seed=1)
with tempfile.TemporaryDirectory() as td:
    for sp in (ms, hs):
        kw = dict(steps=40, R=32, lr=0.05, lr_readout=0.02, warmup=5, eval_every=10, R_val=64, seed=sp["seed"], m=m0, s=s0)
        a = zoo.build(sp, d)
        ra = train(a, Y, Yv, spec, **kw)
        ck = os.path.join(td, f"ck_{sp['model']}.pkl")
        b = zoo.build(sp, d)
        try:
            train(b, Y, Yv, spec, ckpt=ck, ckpt_every=10, _interrupt_at=23, **kw)   # killed at step 23
            print("  FAIL no interruption")
            FAILS.append("interrupt")
        except KeyboardInterrupt:
            pass
        c = zoo.build(sp, d)
        rc = train(c, Y, Yv, spec, ckpt=ck, ckpt_every=10, **kw)
        # the interrupted run left a checkpoint at step 20; the resumed run continues from there
        report(f"resume == uninterrupted, final params ({sp['model']})", float(np.max(np.abs(ra['final_params'] - rc['final_params']))), 0.0)
        report(f"resume == uninterrupted, selected params ({sp['model']})", float(np.max(np.abs(ra['params'] - rc['params']))), 0.0)
        report(f"resume == uninterrupted, val history ({sp['model']})", float(np.nanmax(np.abs(np.array(ra['hist']['val']) - np.array(rc['hist']['val'])))), 0.0)
        report(f"resume == uninterrupted, loss history ({sp['model']})", float(np.nanmax(np.abs(np.array(ra['hist']['loss'][1:]) - np.array(rc['hist']['loss'][1:])))), 0.0)
        # save / load round trip
        a.set(ra["params"])
        path = os.path.join(td, f"m_{sp['model']}.npz")
        zoo.save(path, a, sp, dict(x=1))
        a2, sp2, ex = zoo.load(path)
        om = rng.normal(size=(7, d))
        report(f"zoo save/load round trip ({sp['model']})", float(np.max(np.abs(a.characteristic_function(om) - a2.characteristic_function(om)))), 1e-15)
    # exact gradient of the per-step loss (PCA-decoded CoMB)
    a = zoo.build(ms, d)
    a.init_readout(Y, m=m0, s=s0)
    Z = (Y - m0) / s0
    om = spec.sample(16, rng)
    php = empirical_cf(Z, om)
    def loss(p):
        a.set(p)
        return float(np.mean(np.abs(a.characteristic_function(om) - php) ** 2))
    p0 = a.get()
    a.set(p0)
    q = a.characteristic_function(om)
    _, g = a.characteristic_function(om, grad_out=2.0 * (q - php) / len(om))
    fd = np.zeros_like(p0)
    eps = 1e-6
    for i in range(p0.size):
        e = np.zeros_like(p0)
        e[i] = eps
        fd[i] = (loss(p0 + e) - loss(p0 - e)) / (2 * eps)
    a.set(p0)
    report("per-step loss gradient == finite differences", float(np.max(np.abs(g - fd)) / np.max(np.abs(fd))), 1e-6)
print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
