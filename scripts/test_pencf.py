#!/usr/bin/env python3
"""Tests of comb/pencf.py: exact model CF against brute-force enumeration of every token sequence, gradient
against central finite differences (CoMB, dephased twin, classical HMM, CoMB with environment qubits),
the factorised null CF against direct enumeration, and positions against an explicit pen walk."""
import sys, pathlib, itertools
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comb.digits.model import CoMB, ClassicalHMM
from comb.digits import pencf, kpoint as kp

rng = np.random.default_rng(0)


def make(kind, d=4, seed=1):
    if kind == "comb":
        return CoMB(d, n_m=1, n_w=4, n_sub=1, L_v=2, L_w=1, seed=seed, current="binary")
    if kind == "dephased":
        return CoMB(d, n_m=1, n_w=4, n_sub=1, L_v=2, L_w=1, seed=seed, current="binary", dephase=True)
    if kind == "comb_env":
        return CoMB(d, n_m=1, n_w=5, n_sub=1, L_v=2, L_w=1, seed=seed, current="binary_env:1")
    if kind == "hmm":
        return ClassicalHMM(d, chi=2, K=16, seed=seed)


def joint(model):
    d, K = model.d, model.K
    lams = [model.pixel_instrument(b)[0] for b in range(d)]
    ell = np.zeros(model._v0.size)
    ell[model._tr_idx] = 1.0
    seqs = np.array(list(itertools.product(range(K), repeat=d)))
    P = np.zeros(len(seqs))
    for n, seq in enumerate(seqs):
        v = model._v0.astype(np.complex128)
        for b, c in enumerate(seq):
            v = lams[b][c] @ v
        P[n] = np.real(ell @ v)
    return seqs, P


def test_exact_cf():
    for kind in ("comb", "dephased", "comb_env", "hmm"):
        m = make(kind)
        assert m.K == 16
        pc = pencf.PathCF(rng.integers(0, 16, size=(50, m.d)), m.d, 16, every=2)
        om = pc.frequencies(5, rng)
        phi, _ = pc.model_cf(m, pc.weights(om))
        seqs, P = joint(m)
        assert abs(P.sum() - 1) < 1e-10
        X = pencf.positions(seqs, pc.taus)
        brute = (P[:, None] * np.exp(1j * X @ om.T)).sum(0)
        err = np.abs(phi - brute).max()
        assert err < 1e-12, (kind, err)
        print(f"  CF vs brute force over 16^4 sequences ({kind:8}): max error {err:.1e}")


def test_gradient():
    for kind in ("comb", "dephased", "comb_env", "hmm"):
        m = make(kind, seed=3)
        data = make(kind, seed=8)
        T = data.sample_counts(3000, np.random.default_rng(2))
        pc = pencf.PathCF(T, m.d, 16, every=2)
        om = pc.frequencies(64, np.random.default_rng(4))
        X = pencf.positions(T, pc.taus)
        L, g = pc.loss(m, X, om, grad=True)
        p0 = m.get()
        n_inst = m.n_inst_params()
        idx = rng.choice(n_inst, size=min(20, n_inst), replace=False)
        fd = np.zeros(len(idx))
        for k, i in enumerate(idx):
            h = 1e-5
            pp = p0.copy()
            pp[i] += h
            m.set(pp)
            lp = pc.loss(m, X, om, grad=False)[0]
            pm = p0.copy()
            pm[i] -= h
            m.set(pm)
            lm = pc.loss(m, X, om, grad=False)[0]
            fd[k] = (lp - lm) / (2 * h)
        m.set(p0)
        err = np.abs(g[idx] - fd).max() / max(np.abs(fd).max(), 1e-12)
        assert err < 1e-5, (kind, err)
        print(f"  gradient vs finite differences ({kind:8}): relative error {err:.1e}")


def test_null_and_positions():
    d, K = 4, 16
    p1 = rng.dirichlet(np.ones(K), size=d)
    pc = pencf.PathCF(rng.integers(0, K, size=(50, d)), d, K, every=2)
    om = pc.frequencies(6, rng)
    W = pc.weights(om)
    seqs = np.array(list(itertools.product(range(K), repeat=d)))
    P = np.prod(p1[np.arange(d), seqs], axis=1)
    brute = (P[:, None] * np.exp(1j * pencf.positions(seqs, pc.taus) @ om.T)).sum(0)
    err = np.abs(pencf.PathCF.null_cf(p1, W) - brute).max()
    assert err < 1e-12, err
    tok = np.array([[0, 0, 2, 9, 12, 6]])
    pos = np.zeros(2)
    walk = []
    for c in tok[0]:
        pos = pos + pencf.STEP[c % 8]
        walk.append(pos.copy())
    assert np.allclose(pencf.positions(tok, [2, 4, 6]).reshape(3, 2), np.array(walk)[[1, 3, 5]])
    print(f"  null CF vs brute force: max error {err:.1e}; positions match an explicit pen walk")


if __name__ == "__main__":
    test_exact_cf()
    test_gradient()
    test_null_and_positions()
    print("all path-CF tests passed")
