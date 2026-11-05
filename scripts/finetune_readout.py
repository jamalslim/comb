#!/usr/bin/env python3
"""Fine-tune only the readouts of a trained shower model with the axis-resolved MMD; the circuit stays unchanged.

    python scripts/finetune_readout.py --run outputs/runs/ds1_comb_chi4_s0.npz --steps 1200

The trained readouts pile probability into a few narrow bins (sharp peaks, e.g. the double peaks of the layer-1 and
layer-2 energies), because the Gaussian-mixture kernel of the training loss is far wider than the spread of the data
along any single principal axis. This script adds, with weight --axis-share, frequencies along the --axis-k leading
principal axes (comb_ds1.add_axis_frequencies), and optimizes the readouts only. The instruments are frozen, so their
gradient is exactly zero and Adam leaves every circuit parameter exactly where it was: any device run of the circuit
remains valid, and its device-started showers can be recompleted with the new readouts. The final parameters are kept
(the best-validation rule of the original training would score only the original kernel). Writes <run>_ft.npz.
"""
import argparse, json, pathlib, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import comb_ds1 as cd                                                             # noqa: E402
from comb.showers import zoo                                                              # noqa: E402
from comb.showers.mmd import Spectrum, median_sigma, cf_sum_and_sq                        # noqa: E402
from comb.showers.train import Adam, lr_at                                                # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--steps", type=int, default=1200)
    ap.add_argument("--axis-share", type=float, default=0.2)
    ap.add_argument("--axis-k", type=int, default=16)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--R", type=int, default=256)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--full", action="store_true", help="fine-tune the circuit too (changes the circuit: a device run of the old circuit no longer applies)")
    ap.add_argument("--lr-circuit", type=float, default=0.01, help="learning rate of the circuit angles with --full")
    ap.add_argument("--freeze-readout", default="", help="comma-separated coordinates whose readout is kept exactly (e.g. 0)")
    ap.add_argument("--out", default=None, help="output run file (default: derived from --run)")
    ap.add_argument("--freeze-rounds", type=int, default=0, help="with --full: keep the circuit of the first N rounds exactly (e.g. 12, the rounds run on the device)")
    a = ap.parse_args()
    t0 = time.time()
    cd.need_shower_project(a.run)
    D = cd.load_data()
    model, spec, extra = zoo.load(a.run)
    m0, s0 = D["Y_tr"].mean(0), D["Y_tr"].std(0)
    Z_fit = (D["Y_fit"] - m0) / s0
    Z_val = (D["Y_v8"] - m0) / s0
    pool = np.load(cd.DATA / "ds1_pool.npz")
    pool = cd.add_axis_frequencies((pool["om"], pool["phi"]), Z_fit, k=a.axis_k, share=a.axis_share)
    sig = median_sigma(Z_fit)
    kspec = Spectrum(Z_fit.shape[1], [sig / 2, sig, 2 * sig])
    om_val = kspec.sample(4096, np.random.default_rng(10_000))
    phi_val, sq_val = cf_sum_and_sq(Z_val, om_val)
    def val_mmd():                                                                # the paper's validation MMD^2 (original kernel)
        q = model.characteristic_function(om_val)
        return float(np.mean(np.abs(q) ** 2 - 2.0 * np.real(q * phi_val.conj()) + sq_val))
    nc = model.n_circuit_params()
    p = model.get()
    circuit0 = p[:nc].copy()
    if not a.full: model.freeze_instruments()
    lrs = np.where(np.arange(p.size) < nc, a.lr_circuit / a.lr if a.full else 0.0, 1.0)
    frozen = np.zeros(p.size, bool)
    if a.freeze_rounds > 0:                       # find the parameters of the first rounds by perturbing them (no layout assumed)
        b = model.base
        for name in ("theta_v", "theta_w"):
            arr = getattr(b, name)
            keep = arr.copy()
            arr[:a.freeze_rounds] += 1.0
            frozen |= model.get() != p
            arr[:] = keep
        assert np.array_equal(model.get(), p)
        frozen[nc:] = False
        lrs[frozen] = 0.0
        print(f"circuit of the first {a.freeze_rounds} rounds frozen: {frozen.sum()} of {nc} circuit parameters", flush=True)
    if a.freeze_readout:                          # the readout parameters of the listed coordinates, found by perturbation
        b = model.base
        coords = [int(x) for x in a.freeze_readout.split(",")]
        before = frozen.sum()
        for name in ("h", "lo", "hi"):
            arr = getattr(b, name)
            keep = arr.copy()
            arr[coords] += 0.5
            frozen |= model.get() != p
            arr[:] = keep
        assert np.array_equal(model.get(), p)
        lrs[frozen] = 0.0
        print(f"readout of coordinates {coords} frozen: {frozen.sum() - before} parameters", flush=True)
    p_start = p.copy()
    adam = Adam(p.size)
    rng = np.random.default_rng(a.seed)
    hist = []
    print(f"fine-tuning {'all ' + str(p.size) + ' parameters (circuit and readout)' if a.full else 'the ' + str(p.size - nc) + ' readout parameters'} of {a.run}; "
          f"pool {len(pool[0])} frequencies ({100 * a.axis_share:.0f}% along {a.axis_k} axes)", flush=True)
    for step in range(a.steps + 1):
        model.set(p)
        if step % 100 == 0:
            v = val_mmd()
            hist.append((step, v))
            print(f"step {step:5d}  validation MMD^2 {v:.4e}  ({time.time() - t0:.0f} s)", flush=True)
        if step == a.steps:
            break
        idx = rng.choice(len(pool[0]), size=a.R, replace=False)
        om, php = pool[0][idx], pool[1][idx]
        diff = model.characteristic_function(om) - php
        _, g = model.characteristic_function(om, grad_out=2.0 * diff / a.R)
        p = adam.step(p, g, lr_at(step, a.steps, a.lr, 20) * lrs)
    if not a.full: assert np.array_equal(p[:nc], circuit0), "the circuit must not change"
    if frozen.any(): assert np.array_equal(p[frozen], p_start[frozen]), "the frozen parameters must not change"
    model.set(p)
    model.unfreeze_instruments()
    out = a.out or str(a.run).replace(".npz", ("_ftc" if a.full else "_ft") + (f"_hw{a.freeze_rounds}" if a.freeze_rounds else "") + ".npz")
    zoo.save(out, model, spec, dict(extra, finetune=dict(steps=a.steps, axis_share=a.axis_share, axis_k=a.axis_k, lr=a.lr, val=hist)))
    print(f"done in {(time.time() - t0) / 60:.1f} min; " + ("circuit fine-tuned too" if a.full else "circuit unchanged") + f" -> {out}", flush=True)


if __name__ == "__main__":
    main()
