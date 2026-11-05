#!/usr/bin/env python3
"""How much of its memory does a CoMB model actually use?

    python maxwell/memory_usage.py ds1_comb_chi4_s0_ly ds1_comb_chi8_s0_ly ds1_comb_chi8_s1_ly ...

For each run: the model's memory is followed along N token histories sampled from the model itself, up to rounds T.
  levels  = effective rank of the average memory state, exp(von Neumann entropy): how many of the chi levels carry weight
            (at most chi: 4 with two memory qubits, 8 with three);
  states  = number of independent memory states that different histories reach (numerical rank of their span):
            at most chi for a classical memory, chi^2 for a coherent one (16 or 64).
A model that ignores part of its memory shows levels and states no larger than a smaller memory would."""
import pathlib, sys
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comb.showers import zoo                                                              # noqa: E402

def usage(model, rounds=(10, 50, 200), n=400, seed=0):
    b = model.base
    chi = b.chi
    rng = np.random.default_rng(seed)
    out = {}
    lam = [np.asarray(b.pixel_instrument(beta)[0]) for beta in range(max(rounds))]
    D = lam[0].shape[1]
    coherent = D == chi * chi
    rho = np.zeros((n, D), complex)
    rho[:, 0] = 1.0                                # |0><0| (or classical state 0)
    for beta in range(max(rounds)):
        new = np.einsum("kij,nj->nki", lam[beta], rho)                              # (n, K, D): unnormalized next states
        if coherent: p = np.real(new.reshape(n, -1, chi, chi).trace(axis1=2, axis2=3))
        else:        p = np.real(new.sum(axis=2))
        p = np.clip(p, 0, None)
        p /= p.sum(1, keepdims=True)
        c = np.array([rng.choice(p.shape[1], p=pi) for pi in p])
        rho = new[np.arange(n), c] / p[np.arange(n), c][:, None]
        if beta + 1 in rounds:
            if coherent:
                R = rho.reshape(n, chi, chi)
                avg = R.mean(0)
                ev = np.clip(np.linalg.eigvalsh((avg + avg.conj().T) / 2), 1e-15, None)
                X = np.concatenate([R.reshape(n, -1).real, R.reshape(n, -1).imag], 1)
            else:
                avg = rho.mean(0).real
                ev = np.clip(avg / avg.sum(), 1e-15, None)
                X = rho.real
            ev = ev / ev.sum()
            levels = float(np.exp(-(ev * np.log(ev)).sum()))
            sv = np.linalg.svd(X - 0 * X.mean(0), compute_uv=False)
            states = int((sv > 1e-3 * sv[0]).sum())
            out[beta + 1] = (levels, states)
    return chi, coherent, out

if __name__ == "__main__":
    for name in sys.argv[1:]:
        model = zoo.load(str(ROOT / "outputs/runs" / f"{name}.npz"))[0]
        chi, coh, out = usage(model)
        print(f"{name:28s} chi={chi}: " + "   ".join(f"round {r}: levels {lv:.1f} of {chi}, states {st} of {chi * chi if coh else chi}" for r, (lv, st) in out.items()))
