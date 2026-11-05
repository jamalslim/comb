#!/usr/bin/env python3
"""Tests of comb/noisytree.py.

1. zero noise reproduces TreeCoMB exactly (coherent: instruments; dephased: all laws);
2. the batched noisy instruments equal an independent reference that simulates every node one at a time with
   comb.noise.noisy_circuit (tested module) and does reset, readout confusion, feed-forward on the recorded bit,
   and idling by hand on density matrices, for every token sequence (coherent and dephased);
3. every token map is completely positive (Choi matrix PSD) and a round preserves the trace;
4. gradients of the k-point + path loss under noise agree with central finite differences (coherent, dephased);
5. exact sampling of recorded tokens reproduces the exact one-point and pair laws.
"""
import sys, pathlib, itertools
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comb.digits.treecomb import TreeCoMB
from comb.digits.noisytree import NoisyTreeCoMB
from comb.digits.noise import NoiseModel, noisy_circuit, _amp_damp_dephase
from comb.digits import kpoint as kp, pencf

NZ = NoiseModel(p1=3e-3, p2=2e-2, p_ro=3e-2, p_reset=2e-2, t_round=2.0, T1=60.0, T2=40.0, name="test")
rng = np.random.default_rng(0)


def layout(d, K, lags=(2, 3)):
    s = dict(d=d, K=K, kmax=3, lags=list(lags))
    tI, sI = kp.pair_index(d, list(lags))
    s.update(pair_t=tI, pair_s=sI)
    return s


def joint_from_instruments(m):
    lam, _ = m.instruments_all()
    ell = np.zeros(m._v0.size)
    ell[m._tr_idx] = 1.0
    P = {}
    for seq in itertools.product(range(m.K), repeat=m.d):
        v = m._v0.copy()
        for t, o in enumerate(seq):
            v = lam[t, o] @ v
        P[seq] = float(np.real(ell @ v))
    return P


def joint_reference(m):
    nz = m.noise
    c = m.chi
    gamma, deph = m._idle()
    P = {}
    dg = (lambda r: np.diag(np.diag(r))) if m.dephase else (lambda r: r)
    for seq in itertools.product(range(m.K), repeat=m.d):
        rho = np.zeros((c, c), dtype=np.complex128)
        rho[0, 0] = 1.0
        for t, o in enumerate(seq):
            if m.cw is not None:
                rho = noisy_circuit(m.cw, m.theta_w[t], rho[None], nz.p1, nz.p2)[0]
            rho = dg(rho)
            p = 0
            for l in range(m.n_bits):
                r = (o >> (m.n_bits - 1 - l)) & 1                                      # recorded bit = emitted bit
                full = np.kron(rho, np.diag([1 - nz.p_reset, nz.p_reset]))
                out = noisy_circuit(m.cn, m.theta_n[t, 2 ** l - 1 + p], full[None], nz.p1, nz.p2)[0].reshape(c, 2, c, 2)
                new = sum(m._conf[r, b] * out[:, b, :, b] for b in (0, 1))
                for q in range(m.n_m):
                    new = _amp_damp_dephase(new[None], q, m.n_m, gamma, deph)[0]
                rho = dg(new)
                p = 2 * p + r                                            # feed-forward on the recorded bit
        P[seq] = float(np.real(np.trace(rho)))
    return P


def test_zero_noise():
    for deph in (False, True):
        t = TreeCoMB(4, n_m=1, n_bits=2, L_node=2, L_w=1, seed=3, dephase=deph)
        z = NoisyTreeCoMB.from_model(t, NoiseModel())
        a, _ = kp.model_statistics(t, layout(4, 4))
        b, _ = kp.model_statistics(z, layout(4, 4))
        err = max(max(np.abs(x - y).max() for x, y in zip(a["W"], b["W"])), np.abs(a["pairs"] - b["pairs"]).max())
        if not deph:
            err = max(err, np.abs(t.instruments_all()[0] - z.instruments_all()[0]).max())
        assert err < 1e-13, err
        print(f"  zero noise reproduces TreeCoMB ({'dephased' if deph else 'coherent'}): {err:.1e}")


def test_reference():
    for deph in (False, True):
        m = NoisyTreeCoMB.from_model(TreeCoMB(2, n_m=1, n_bits=2, L_node=2, L_w=1, seed=5, dephase=deph), NZ)
        A, B = joint_from_instruments(m), joint_reference(m)
        err = max(abs(A[k] - B[k]) for k in A)
        tot = sum(A.values())
        assert err < 1e-13 and abs(tot - 1) < 1e-12, (err, tot)
        print(f"  noisy instruments vs node-by-node density-matrix reference ({'dephased' if deph else 'coherent'}): {err:.1e}")


def test_cptp():
    m = NoisyTreeCoMB.from_model(TreeCoMB(3, n_m=2, n_bits=2, L_node=2, L_w=1, seed=7), NZ)
    lam, _ = m.instruments_all()
    c = m.chi
    ell = np.zeros(c * c)
    ell[m._tr_idx] = 1.0
    tp = np.abs(ell @ lam.sum(1) - ell[None, :]).max()
    worst = min(np.linalg.eigvalsh(lam[t, o].reshape(c, c, c, c).transpose(2, 0, 3, 1).reshape(c * c, c * c)).min()
                for t in range(3) for o in range(m.K))
    assert tp < 1e-12 and worst > -1e-12, (tp, worst)
    print(f"  trace preservation {tp:.1e}; smallest Choi eigenvalue of any token map {worst:.1e} (completely positive)")


def test_gradients():
    for deph in (False, True):
        teacher = TreeCoMB(4, n_m=1, n_bits=4, L_node=2, L_w=1, seed=11)
        T = teacher.sample_counts(3000, np.random.default_rng(1))
        m = NoisyTreeCoMB.from_model(TreeCoMB(4, n_m=1, n_bits=4, L_node=2, L_w=1, seed=13, dephase=deph), NZ)
        st = kp.data_statistics(T, 16, kmax=3, lags=[3])
        G = kp.token_kernel(16, "stroke")
        pc = pencf.PathCF(T, 4, 16, every=1)
        om = pc.frequencies(24, np.random.default_rng(2))
        X = pencf.positions(T, pc.taus)
        def total(grad):
            l1, _, g1 = kp.kpoint_loss(m, st, G, (1, 1, 1, 1), grad=grad)
            l2, g2 = pc.loss(m, X, om, grad=grad)
            return l1 + l2, (g1 + g2 if grad else None)
        L, g = total(True)
        p0 = m.get()
        idx = rng.choice(m.n_inst_params(), 20, replace=False)
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
        assert err < 1e-6, err
        print(f"  gradient under noise vs finite differences ({'dephased' if deph else 'coherent'}): {err:.1e}")


def test_sampling():
    m = NoisyTreeCoMB.from_model(TreeCoMB(5, n_m=2, n_bits=2, L_node=2, L_w=1, seed=17), NZ)
    C = m.sample_counts(60000, np.random.default_rng(3))
    sm, _ = kp.model_statistics(m, layout(5, 4))
    e1 = np.abs(np.stack([np.bincount(C[:, t], minlength=4) / len(C) for t in range(5)]) - sm["W"][0]).max()
    e2 = np.abs(np.bincount(C[:, 0] * 4 + C[:, 2], minlength=16).reshape(4, 4) / len(C) - sm["pairs"][0]).max()
    assert e1 < 0.01 and e2 < 0.01, (e1, e2)
    print(f"  sampling of recorded tokens under noise vs exact laws (60000 samples): one-point {e1:.4f}, pair {e2:.4f}")


if __name__ == "__main__":
    test_zero_noise()
    test_reference()
    test_cptp()
    test_gradients()
    test_sampling()
    print("all noisy tree-CoMB tests passed")
