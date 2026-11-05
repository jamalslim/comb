#!/usr/bin/env python3
"""Tests of comb/kpoint.py: exact window and pair laws against brute-force enumeration of every
token sequence, gradients against central finite differences (coherent CoMB, reduced dephased
twin, classical HMM), zero loss and gradient at a self-match, and the kernel's positivity."""
import sys, pathlib, itertools
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comb.digits.model import CoMB, ClassicalHMM
from comb.digits import kpoint as kp

rng = np.random.default_rng(0)


def brute_force_laws(model):
    """Joint law of all token sequences by enumeration (tiny models only)."""
    d, K = model.d, model.K
    lams = [model.pixel_instrument(b)[0] for b in range(d)]
    ell = np.zeros(model._v0.size)
    ell[model._tr_idx] = 1.0
    P = np.zeros((K,) * d)
    for seq in itertools.product(range(K), repeat=d):
        v = model._v0.astype(np.complex128)
        for b, c in enumerate(seq):
            v = lams[b][c] @ v
        P[seq] = np.real(ell @ v)
    return P


def laws_from_joint(P, stats):
    d, K = P.ndim, P.shape[0]
    W = []
    for k in range(1, stats["kmax"] + 1):
        Wk = np.zeros((d - k + 1,) + (K,) * k)
        for t in range(d - k + 1):
            axes = tuple(a for a in range(d) if not (t <= a < t + k))
            Wk[t] = P.sum(axis=axes)
        W.append(Wk)
    pairs = np.zeros((len(stats["pair_t"]), K, K))
    for n, (t, s) in enumerate(zip(stats["pair_t"], stats["pair_s"])):
        axes = tuple(a for a in range(d) if a not in (t, s))
        pairs[n] = P.sum(axis=axes)
    return W, pairs


def make(kind, d=4, seed=1):
    if kind == "comb":
        return CoMB(d, n_m=1, n_w=2, n_sub=1, L_v=2, L_w=1, seed=seed, current="binary")
    if kind == "dephased":
        return CoMB(d, n_m=1, n_w=2, n_sub=1, L_v=2, L_w=1, seed=seed, current="binary", dephase=True)
    if kind == "hmm":
        return ClassicalHMM(d, chi=2, K=4, seed=seed)
    raise ValueError(kind)


def test_laws():
    for kind in ("comb", "dephased", "hmm"):
        m = make(kind)
        fake = dict(d=m.d, K=m.K, kmax=3, lags=[2, 3], N=1, W=None, pairs=None)
        tI, sI = kp.pair_index(m.d, [2, 3])
        fake.update(pair_t=tI, pair_s=sI)
        sm, _ = kp.model_statistics(m, fake)
        Pj = brute_force_laws(m)
        assert abs(Pj.sum() - 1) < 1e-10, "joint law does not sum to one"
        W, pairs = laws_from_joint(Pj, fake)
        errs = [np.abs(a - b).max() for a, b in zip(sm["W"], W)] + [np.abs(sm["pairs"] - pairs).max()]
        assert max(errs) < 1e-12, (kind, errs)
        print(f"  laws vs brute force ({kind:8}): max error {max(errs):.1e}")


def test_gradients():
    for kind in ("comb", "dephased", "hmm"):
        m = make(kind, seed=3)
        data = make(kind, seed=7)                                         # a different model as the data
        C = data.sample_counts(4000, np.random.default_rng(5))
        stats = kp.data_statistics(C, m.K, kmax=3, lags=[2, 3])
        G = kp.token_kernel(m.K, "delta") + 0.3 * np.ones((m.K, m.K))      # a non-trivial PSD kernel
        w = (1.0, 0.7, 0.5, 1.3)
        loss, vals, g = kp.kpoint_loss(m, stats, G, w, grad=True)
        p0 = m.get()
        n_inst = m.n_inst_params()
        fd = np.zeros_like(g)
        idx = rng.choice(n_inst, size=min(25, n_inst), replace=False)
        for i in idx:
            h = 1e-5
            pp = p0.copy()
            pp[i] += h
            m.set(pp)
            lp = kp.kpoint_loss(m, stats, G, w, grad=False)[0]
            pm = p0.copy()
            pm[i] -= h
            m.set(pm)
            lm = kp.kpoint_loss(m, stats, G, w, grad=False)[0]
            fd[i] = (lp - lm) / (2 * h)
        m.set(p0)
        err = np.abs(g[idx] - fd[idx]).max() / max(np.abs(fd[idx]).max(), 1e-12)
        assert err < 1e-5, (kind, err)
        assert np.all(g[n_inst:] == 0), "readout parameters must get zero gradient"
        print(f"  gradient vs finite differences ({kind:8}): relative error {err:.1e}, loss {loss:.3e}")


def test_self_match_and_kernel():
    m = make("comb", seed=11)
    fake = dict(d=m.d, K=m.K, kmax=3, lags=[2, 3])
    tI, sI = kp.pair_index(m.d, [2, 3])
    fake.update(pair_t=tI, pair_s=sI)
    sm, _ = kp.model_statistics(m, fake)
    G = kp.token_kernel(m.K, "delta")
    loss, vals, g = kp.kpoint_loss(m, sm, G, (1, 1, 1, 1), grad=True)
    assert abs(loss) < 1e-14 and np.abs(g).max() < 1e-12, (loss, np.abs(g).max())
    print(f"  self-match: loss {loss:.1e}, max |grad| {np.abs(g).max():.1e}")
    G16 = kp.token_kernel(16, "stroke", kappa=2.0, rho_pen=0.3)
    ev = np.linalg.eigvalsh(G16)
    assert ev.min() > -1e-12 and abs(G16.diagonal() - 1).max() < 1e-12
    print(f"  stroke kernel: 16 x 16, eigenvalues in [{ev.min():.3f}, {ev.max():.3f}], adjacent-direction similarity {G16[0, 1]:.3f}")
    null = kp.independent_statistics(sm)
    assert abs(null["W"][1][0].sum() - 1) < 1e-12 and abs(null["pairs"][0].sum() - 1) < 1e-12
    print("  null statistics normalised")


def test_training_decreases():
    teacher = CoMB(6, n_m=1, n_w=2, n_sub=1, L_v=2, L_w=1, seed=21, current="binary")
    C = teacher.sample_counts(20000, np.random.default_rng(1))
    st = kp.data_statistics(C[:16000], 4, kmax=3)
    sv = kp.data_statistics(C[16000:], 4, kmax=3)
    student = CoMB(6, n_m=1, n_w=2, n_sub=1, L_v=2, L_w=1, seed=99, current="binary")
    G = kp.token_kernel(4, "delta")
    l0 = kp.kpoint_loss(student, sv, G, (1, 1, 1, 1), grad=False)[0]
    res = kp.train_kpoint(student, st, sv, G, steps=60, lr=0.1, warmup=5, val_every=20, log=None)
    assert res["best_val"] < 0.5 * l0, (l0, res["best_val"])
    print(f"  training: validation loss {l0:.3e} -> {res['best_val']:.3e} in 60 steps")


if __name__ == "__main__":
    test_laws()
    test_gradients()
    test_self_match_and_kernel()
    test_training_decreases()
    print("all kpoint tests passed")
