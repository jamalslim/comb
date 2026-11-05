#!/usr/bin/env python3
"""Tests of comb/hqmm.py: isometry and trace preservation, laws against brute-force enumeration, gradients of
the k-point and path losses against finite differences, sampling against exact one-point laws, and a numerical
check of Proposition 4: a classical HMM written as an isometry with chi^2 environment levels reproduces every
window law, pair law and path characteristic function of that HMM exactly."""
import sys, pathlib, itertools
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comb.digits.hqmm import GeneralHQMM
from comb.digits.model import ClassicalHMM
from comb.digits import kpoint as kp, pencf

rng = np.random.default_rng(0)


def layout(d, K, lags=(2, 3)):
    s = dict(d=d, K=K, kmax=3, lags=list(lags))
    tI, sI = kp.pair_index(d, lags)
    s.update(pair_t=tI, pair_s=sI)
    return s


def test_isometry_and_laws():
    m = GeneralHQMM(4, chi=2, K=4, E=3, seed=1)
    for b in range(m.d):
        V, _ = m._isometry(b)
        assert np.abs(V.conj().T @ V - np.eye(2)).max() < 1e-12
        lam, _ = m.pixel_instrument(b)
        T = lam.sum(0)
        rho = rng.normal(size=(2, 2)) + 1j * rng.normal(size=(2, 2))
        rho = rho @ rho.conj().T
        out = (T @ rho.reshape(-1)).reshape(2, 2)
        assert abs(np.trace(out) - np.trace(rho)) < 1e-12
    st = layout(m.d, m.K)
    sm, _ = kp.model_statistics(m, st)
    lams = [m.pixel_instrument(b)[0] for b in range(m.d)]
    ell = np.zeros(4, complex)
    ell[m._tr_idx] = 1
    P = np.zeros((4,) * 4)
    for seq in itertools.product(range(4), repeat=4):
        v = m._v0.copy()
        for b, c in enumerate(seq):
            v = lams[b][c] @ v
        P[seq] = np.real(ell @ v)
    assert abs(P.sum() - 1) < 1e-12
    err = max(np.abs(sm["W"][0] - np.stack([P.sum(axis=tuple(a for a in range(4) if a != t)) for t in range(4)])).max(),
              np.abs(sm["W"][1][0] - P.sum(axis=(2, 3))).max(), np.abs(sm["pairs"][0] - P.sum(axis=(1, 3))).max())
    assert err < 1e-12, err
    print(f"  isometry, trace preservation and laws vs brute force: max error {err:.1e}")


def test_gradients():
    teacher = GeneralHQMM(5, chi=2, K=16, E=2, seed=7)
    T = teacher.sample_counts(3000, np.random.default_rng(3))
    m = GeneralHQMM(5, chi=2, K=16, E=2, seed=9)
    st = kp.data_statistics(T, 16, kmax=3, lags=[3, 4])
    G = kp.token_kernel(16, "stroke")
    pc = pencf.PathCF(T, 5, 16, every=1)
    om = pc.frequencies(32, np.random.default_rng(5))
    X = pencf.positions(T, pc.taus)
    def total(grad):
        l1, _, g1 = kp.kpoint_loss(m, st, G, (1, 1, 1, 1), grad=grad)
        l2, g2 = pc.loss(m, X, om, grad=grad)
        return l1 + l2, (None if not grad else g1 + g2)
    L, g = total(True)
    p0 = m.get()
    idx = rng.choice(m.n_inst_params(), 24, replace=False)
    fd = []
    for i in idx:
        h = 1e-6
        pp = p0.copy()
        pp[i] += h
        m.set(pp)
        lp = total(False)[0]
        pm = p0.copy()
        pm[i] -= h
        m.set(pm)
        lm = total(False)[0]
        fd.append((lp - lm) / (2 * h))
    m.set(p0)
    fd = np.array(fd)
    err = np.abs(g[idx] - fd).max() / np.abs(fd).max()
    assert err < 1e-5, err
    print(f"  gradient of k-point + path loss through the polar factor vs finite differences: relative error {err:.1e}")


def test_sampling():
    m = GeneralHQMM(6, chi=2, K=4, E=2, seed=4)
    C = m.sample_counts(60000, np.random.default_rng(1))
    sm, _ = kp.model_statistics(m, layout(6, 4))
    emp = np.stack([np.bincount(C[:, t], minlength=4) / len(C) for t in range(6)])
    err = np.abs(emp - sm["W"][0]).max()
    assert err < 0.01, err
    print(f"  sampled one-point laws vs exact (60000 samples): max deviation {err:.4f}")


def test_proposition4():
    d, chi, K = 5, 3, 16
    h = ClassicalHMM(d, chi=chi, K=K, seed=11, init_scale=1.5)
    q = GeneralHQMM(d, chi=chi, K=K, E=chi * chi, seed=0)
    for b in range(d):
        T = h._softmax0(h.tl[b])
        E = h._softmax0(h.el[b])
        V = np.zeros((K, chi * chi, chi, chi))                       # rows (c, e = (i, j), j), column i
        for c in range(K):
            for i in range(chi):
                for j in range(chi):
                    V[c, i * chi + j, j, i] = np.sqrt(T[j, i] * E[c, j])
        q.are[b] = V.reshape(-1, chi)
        q.aim[b] = 0.0
    st = layout(d, K, lags=(3, 4))
    sh, _ = kp.model_statistics(h, st)
    sq, _ = kp.model_statistics(q, st)
    err = max(max(np.abs(a - b).max() for a, b in zip(sh["W"], sq["W"])), np.abs(sh["pairs"] - sq["pairs"]).max())
    pc = pencf.PathCF(rng.integers(0, K, (50, d)), d, K, every=1)
    W = pc.weights(pc.frequencies(16, rng))
    errc = np.abs(pc.model_cf(h, W)[0] - pc.model_cf(q, W)[0]).max()
    assert err < 1e-12 and errc < 1e-12, (err, errc)
    print(f"  Proposition 4: HMM (chi=3, K=16) as an isometry with 9 environment levels: window/pair laws differ by {err:.1e}, path CF by {errc:.1e}")


if __name__ == "__main__":
    test_isometry_and_laws()
    test_gradients()
    test_sampling()
    test_proposition4()
    print("all general-HQMM tests passed")


def test_dephased_twin():
    """Dephased general memory: laws equal the coherent model with the memory dephased after every round
    (checked by enumeration), gradients match finite differences, sampling matches the exact laws."""
    import itertools
    q = GeneralHQMM(4, chi=2, K=4, E=2, seed=5, dephase=True)
    qc = GeneralHQMM(4, chi=2, K=4, E=2, seed=5)
    lams = [qc.pixel_instrument(b)[0] for b in range(4)]
    diag = np.arange(2) * 3
    P = np.zeros((4,) * 4)
    for seq in itertools.product(range(4), repeat=4):
        v = qc._v0.copy()
        for b, c in enumerate(seq):
            v = lams[b][c] @ v
            keep = np.zeros_like(v)
            keep[diag] = v[diag]
            v = keep
        P[seq] = np.real(v[diag].sum())
    sm, _ = kp.model_statistics(q, layout(4, 4))
    err = np.abs(sm["W"][1][0] - P.sum(axis=(2, 3))).max()
    assert err < 1e-12, err
    teacher = GeneralHQMM(5, chi=2, K=16, E=2, seed=7)
    T = teacher.sample_counts(2000, np.random.default_rng(3))
    m = GeneralHQMM(5, chi=2, K=16, E=2, seed=9, dephase=True)
    st = kp.data_statistics(T, 16, kmax=3, lags=[3, 4])
    G = kp.token_kernel(16, "stroke")
    L, _, g = kp.kpoint_loss(m, st, G, (1, 1, 1, 1))
    p0 = m.get()
    idx = rng.choice(m.n_inst_params(), 16, replace=False)
    fd = []
    for i in idx:
        h = 1e-6
        pp = p0.copy()
        pp[i] += h
        m.set(pp)
        lp = kp.kpoint_loss(m, st, G, (1, 1, 1, 1), grad=False)[0]
        pm = p0.copy()
        pm[i] -= h
        m.set(pm)
        lm = kp.kpoint_loss(m, st, G, (1, 1, 1, 1), grad=False)[0]
        fd.append((lp - lm) / (2 * h))
    m.set(p0)
    fd = np.array(fd)
    gerr = np.abs(g[idx] - fd).max() / np.abs(fd).max()
    assert gerr < 1e-5, gerr
    C = m.sample_counts(60000, np.random.default_rng(2))
    sm2, _ = kp.model_statistics(m, layout(5, 16))
    serr = np.abs(np.stack([np.bincount(C[:, t], minlength=16) / len(C) for t in range(5)]) - sm2["W"][0]).max()
    assert serr < 0.01, serr
    print(f"  dephased general memory: laws vs dephased enumeration {err:.1e}, gradient {gerr:.1e}, sampling {serr:.4f}")


if __name__ == "__main__":
    test_dephased_twin()
