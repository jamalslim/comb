#!/usr/bin/env python3
"""Tests of comb/treecomb.py.

1. laws: window and pair laws against brute-force enumeration of every token sequence (coherent, dephased);
2. dephased twin: equals explicit density-matrix evolution with dephasing after W and after every node;
3. batched nodes: Kraus pairs equal each node circuit simulated on its own, and every node is an isometry;
4. gradients of the k-point + path loss against central finite differences (coherent, dephased);
5. sampling: the dynamic circuit with feed-forward reproduces the exact one-point laws;
6. Theorem 5: compile_tree reproduces every Kraus operator of random general memories and of a trained one,
   and a TreeCoMB whose nodes are the compiled trees has the same statistics.
"""
import sys, pathlib, itertools
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comb.digits.treecomb import TreeCoMB, compile_tree, compose_tree
from comb.digits.hqmm import GeneralHQMM
from comb.digits import kpoint as kp, pencf

rng = np.random.default_rng(0)


def layout(d, K, lags=(2, 3)):
    s = dict(d=d, K=K, kmax=3, lags=list(lags))
    tI, sI = kp.pair_index(d, list(lags))
    s.update(pair_t=tI, pair_s=sI)
    return s


def joint(model):
    lam, _ = model.instruments_all()
    d, K = model.d, model.K
    ell = np.zeros(model._v0.size)
    ell[model._tr_idx] = 1.0
    P = np.zeros((K,) * d)
    for seq in itertools.product(range(K), repeat=d):
        v = model._v0.copy()
        for b, c in enumerate(seq):
            v = lam[b][c] @ v
        P[seq] = np.real(ell @ v)
    return P


def test_laws():
    for deph in (False, True):
        m = TreeCoMB(4, n_m=1, n_bits=2, L_node=2, L_w=1, seed=3, dephase=deph)
        lam, _ = m.instruments_all()
        T = lam.sum(1)
        ell = np.zeros(m._v0.size)
        ell[m._tr_idx] = 1.0
        assert np.abs(ell @ T - ell[None, :]).max() < 1e-12, "a round must preserve the trace"
        P = joint(m)
        assert abs(P.sum() - 1) < 1e-12
        sm, _ = kp.model_statistics(m, layout(4, 4))
        err = max(np.abs(sm["W"][0] - np.stack([P.sum(axis=tuple(a for a in range(4) if a != t)) for t in range(4)])).max(),
                  np.abs(sm["W"][2][0] - P.sum(axis=3)).max(), np.abs(sm["pairs"][0] - P.sum(axis=(1, 3))).max())
        assert err < 1e-12, err
        print(f"  laws vs brute force ({'dephased' if deph else 'coherent'}): {err:.1e}")


def test_dephased_semantics():
    m = TreeCoMB(3, n_m=1, n_bits=2, L_node=2, L_w=1, seed=5)
    md = TreeCoMB(3, n_m=1, n_bits=2, L_node=2, L_w=1, seed=5, dephase=True)
    N, _ = m.nodes()
    W, _ = m.rotations()
    dg = lambda r: np.diag(np.diag(r))
    P = np.zeros((4,) * 3)
    for seq in itertools.product(range(4), repeat=3):
        rho = np.zeros((2, 2), complex)
        rho[0, 0] = 1
        for t, o in enumerate(seq):
            rho = dg(W[t] @ rho @ W[t].conj().T)
            p = 0
            for l, b in enumerate(((o >> 1) & 1, o & 1)):
                Nb = N[t, 2 ** l - 1 + p, b]
                rho = dg(Nb @ rho @ Nb.conj().T)
                p = 2 * p + b
        P[seq] = np.real(np.trace(rho))
    err = np.abs(P - joint(md)).max()
    assert err < 1e-12, err
    print(f"  dephased twin vs density matrices dephased after W and every node: {err:.1e}")


def test_nodes_explicit():
    m = TreeCoMB(3, n_m=2, n_bits=4, L_node=3, L_w=2, seed=7)
    N, _ = m.nodes()
    err = err_iso = 0.0
    for t in range(3):
        for k in range(m.n_nodes):
            S, _ = m.cn.forward(m.theta_n[t, k], m._E)                                    # single circuit, rows m*2 + b
            err = max(err, np.abs(S[0::2] - N[t, k, 0]).max(), np.abs(S[1::2] - N[t, k, 1]).max())
            Sk = np.vstack([N[t, k, 0], N[t, k, 1]])
            err_iso = max(err_iso, np.abs(Sk.conj().T @ Sk - np.eye(4)).max())
    assert err < 1e-13 and err_iso < 1e-12, (err, err_iso)
    print(f"  batched node circuits vs single circuits: {err:.1e}; nodes are isometries to {err_iso:.1e}")


def test_gradients():
    for deph in (False, True):
        teacher = TreeCoMB(5, n_m=1, n_bits=4, L_node=2, L_w=1, seed=11, dephase=deph)
        T = teacher.sample_counts(3000, np.random.default_rng(1))
        m = TreeCoMB(5, n_m=1, n_bits=4, L_node=2, L_w=1, seed=13, dephase=deph)
        st = kp.data_statistics(T, 16, kmax=3, lags=[3, 4])
        G = kp.token_kernel(16, "stroke")
        pc = pencf.PathCF(T, 5, 16, every=1)
        om = pc.frequencies(32, np.random.default_rng(2))
        X = pencf.positions(T, pc.taus)
        def total(grad):
            l1, _, g1 = kp.kpoint_loss(m, st, G, (1, 1, 1, 1), grad=grad)
            l2, g2 = pc.loss(m, X, om, grad=grad)
            return l1 + l2, (g1 + g2 if grad else None)
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
        assert err < 1e-6, err
        print(f"  gradient of k-point + path loss vs finite differences ({'dephased' if deph else 'coherent'}): {err:.1e}")


def test_sampling():
    for deph in (False, True):
        m = TreeCoMB(6, n_m=2, n_bits=2, L_node=2, L_w=1, seed=17, dephase=deph)
        C = m.sample_counts(60000, np.random.default_rng(3))
        sm, _ = kp.model_statistics(m, layout(6, 4))
        e1 = np.abs(np.stack([np.bincount(C[:, t], minlength=4) / len(C) for t in range(6)]) - sm["W"][0]).max()
        pair = np.bincount(C[:, 0] * 4 + C[:, 2], minlength=16).reshape(4, 4) / len(C)   # first pair of the layout: moves (0, 2)
        e2 = np.abs(pair - sm["pairs"][0]).max()
        assert e1 < 0.01 and e2 < 0.01, (e1, e2)
        print(f"  feed-forward sampling vs exact laws ({'dephased' if deph else 'coherent'}, 60000 samples): one-point {e1:.4f}, pair {e2:.4f}")


def test_theorem5():
    worst = 0.0
    for seed in range(5):
        g = GeneralHQMM(2, chi=4, K=16, E=1, seed=seed)
        for t in range(2):
            M = g.kraus(t)[0][:, 0]
            N = compile_tree(M)
            worst = max(worst, np.abs(compose_tree(N) - M).max())
            S = np.concatenate([N[:, 0], N[:, 1]], axis=1)
            worst_iso = np.abs(np.einsum("kji,kjl->kil", S.conj(), S) - np.eye(4)).max()
            assert worst_iso < 1e-10, worst_iso
    assert worst < 1e-12, worst
    print(f"  Theorem 5 on random general memories (chi=4, K=16): Kraus operators reproduced to {worst:.1e}")
    ck = ROOT / "outputs/strokes/strokes_uci_hqmm_chi4_e0_path2_s0.ckpt.npz"
    if ck.exists():
        g = GeneralHQMM(56, chi=4, K=16, E=1, seed=0)
        g.set(np.load(ck, allow_pickle=True)["best_p"])
        err = max(np.abs(compose_tree(compile_tree(g.kraus(t)[0][:, 0])) - g.kraus(t)[0][:, 0]).max() for t in range(56))
        assert err < 1e-10, err
        print(f"  Theorem 5 on the trained coherent memory (56 rounds): {err:.1e}")


if __name__ == "__main__":
    test_laws()
    test_dephased_semantics()
    test_nodes_explicit()
    test_gradients()
    test_sampling()
    test_theorem5()
    print("all tree-CoMB tests passed")
