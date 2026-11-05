#!/usr/bin/env python3
"""
Tests of the coherent memory register (every piece against an independent
reference).

  1. Circuit reverse mode vs central finite differences (CZ and RZZ).
  2. kron_grad vs finite differences.
  3. Count-resolved instrument (dynamic programme) vs brute-force enumeration
     of all work qubit-outcome sequences (both counters, coherent and dephased);
     trace preservation of sum_c Lambda_c.
  4. Instrument VJP vs finite differences.
  5. CF weights vs numerical quadrature of int_0^1 exp(i w (g(c+u)-m)/s) du.
  6. Characteristic function vs explicit sum over all count vectors.
  7. Characteristic function vs Monte-Carlo average over sampled trajectories
     (coherent and dephased), and count marginals vs sampled counts.
  8. Gradient of the characteristic function (all parameter groups) vs finite
     differences (coherent, dephased, RZZ, binary counter).

    python scripts/test_comb.py
"""
from __future__ import annotations

import itertools
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from comb.showers.circuit import Circuit, interleaved_edges   # noqa: E402
from comb.showers.model import CoMB, ClassicalHMM, kron_grad   # noqa: E402

FAILS = []


def report(name, err, tol):
    ok = err < tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:58s} {err:.2e} (tol {tol:.0e})", flush=True)
    if not ok:
        FAILS.append(name)


def cplx(rng, *shape):
    return rng.normal(size=shape) + 1j * rng.normal(size=shape)


def fd_real(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        g.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return g


def relerr(a, b):
    return float(np.max(np.abs(a - b)) / max(np.max(np.abs(b)), 1e-12))


# ---------------------------------------------------------------------------
def test_circuit():
    print("1. circuit reverse mode")
    rng = np.random.default_rng(1)
    for ent in ("cz", "rzz"):
        c = Circuit(3, 2, interleaved_edges(1, 2), ent)
        th = rng.normal(size=c.n_params)
        E = cplx(rng, 8, 3)
        C = cplx(rng, 8, 3)
        Bw = rng.random((8, 3))

        def loss(t):
            V, _ = c.forward(t, E)
            return np.real(np.sum(C.conj() * V)) + np.sum(Bw * np.abs(V) ** 2)

        V, tape = c.forward(th, E)
        g = c.vjp(tape, C + 2 * Bw * V)
        report(f"circuit vjp ({ent}, {c.n_params} params)", relerr(g, fd_real(loss, th)), 1e-7)
        U = c.unitary(th)
        report(f"circuit unitarity ({ent})", float(np.max(np.abs(U.conj().T @ U - np.eye(8)))), 1e-12)


def test_kron_grad():
    print("2. kron_grad")
    rng = np.random.default_rng(2)
    M = cplx(rng, 3, 3)
    C = cplx(rng, 9, 9)
    Bw = rng.random((9, 9))

    def loss_c(Mc):
        S = np.kron(Mc, Mc.conj())
        return np.real(np.sum(C.conj() * S)) + np.sum(Bw * np.abs(S) ** 2)

    S = np.kron(M, M.conj())
    g = kron_grad(C + 2 * Bw * S, M)
    gre = fd_real(lambda x: loss_c(x + 1j * M.imag), M.real)
    gim = fd_real(lambda x: loss_c(M.real + 1j * x), M.imag)
    report("kron_grad", relerr(g, gre + 1j * gim), 1e-7)


# ---------------------------------------------------------------------------
def brute_instrument(m, beta):
    """Lambda_c by enumerating every outcome sequence of the n_sub rounds."""
    c2 = m.chi ** 2
    Vb, _ = m.cv.forward(m.theta_v[beta], m._E)
    Ms = [Vb[x::m.A, :] for x in range(m.A)]
    W = m.cw.unitary(m.theta_w[beta])
    D = np.zeros((c2, c2))
    D[m._diag_idx, m._diag_idx] = 1.0
    Dm = D if m.dephase else np.eye(c2)
    lam = np.zeros((m.K, c2, c2), dtype=np.complex128)
    S0 = Dm @ np.kron(W, W.conj()) @ Dm
    for seq in itertools.product(range(m.A), repeat=m.n_sub):
        S = S0
        for x in seq:
            S = Dm @ np.kron(Ms[x], Ms[x].conj()) @ Dm @ S
        lam[sum(m.phi[x] for x in seq)] += S
    return lam


def test_instrument():
    print("3. count-resolved instrument vs brute-force enumeration (both engines, both dephased representations)")
    for (n_m, n_w, n_sub, cur, deph) in [(1, 2, 3, "hamming", False), (2, 1, 4, "hamming", False),
                                          (1, 2, 3, "binary", False), (2, 1, 3, "hamming", True),
                                          (1, 2, 2, "binary", True), (2, 1, 7, "hamming", False)]:
        for engine in ("dp", "fft"):
            for mode in (("full", "reduced") if deph else ("full",)):
                m = CoMB(2, n_m=n_m, n_w=n_w, n_sub=n_sub, L_v=2, L_w=1, seed=3, dephase=deph, current=cur,
                        engine=engine, dephase_mode=mode)
                lam, _ = m.pixel_instrument(1)
                mf = CoMB(2, n_m=n_m, n_w=n_w, n_sub=n_sub, L_v=2, L_w=1, seed=3, dephase=deph, current=cur,
                         dephase_mode="full")
                ref = brute_instrument(mf, 1)
                if m.reduced:
                    ref = ref[:, mf._diag_idx][:, :, mf._diag_idx]
                tag = f"chi={m.chi} A={m.A} n_sub={n_sub} {cur}{' dephased-' + mode if deph else ''} {engine}"
                report(f"instrument == enumeration ({tag})", relerr(lam, ref), 1e-12)
                tp = lam.sum(axis=0)[m._tr_idx, :].sum(axis=0)
                e = np.zeros(len(m._v0))
                e[m._tr_idx] = 1.0
                report(f"trace preservation ({tag})", float(np.max(np.abs(tp - e))), 1e-12)


def test_instrument_vjp():
    print("4. instrument VJP (both engines, both dephased representations)")
    rng = np.random.default_rng(4)
    cases = [(2, 1, 3, "hamming", False, "cz", "dp", "full"), (2, 1, 3, "hamming", False, "cz", "fft", "full"),
             (1, 2, 2, "hamming", True, "cz", "dp", "full"), (1, 2, 2, "hamming", True, "cz", "fft", "reduced"),
             (2, 1, 5, "hamming", True, "cz", "fft", "reduced"), (1, 2, 2, "binary", False, "rzz", "fft", "full"),
             (2, 1, 6, "hamming", False, "cz", "fft", "full")]
    for (n_m, n_w, n_sub, cur, deph, ent, engine, mode) in cases:
        m = CoMB(2, n_m=n_m, n_w=n_w, n_sub=n_sub, L_v=2, L_w=2, seed=5, dephase=deph, current=cur, entangler=ent,
                engine=engine, dephase_mode=mode)
        Dv = len(m._v0)
        C = cplx(rng, m.K, Dv, Dv)
        beta = 1
        lam, tape = m.pixel_instrument(beta, keep_tape=True)
        g_v, g_w = m.pixel_instrument_vjp(beta, tape, C)

        def loss_v(t):
            old = m.theta_v[beta].copy()
            m.theta_v[beta] = t
            out = np.real(np.sum(C.conj() * m.pixel_instrument(beta)[0]))
            m.theta_v[beta] = old
            return out

        def loss_w(t):
            old = m.theta_w[beta].copy()
            m.theta_w[beta] = t
            out = np.real(np.sum(C.conj() * m.pixel_instrument(beta)[0]))
            m.theta_w[beta] = old
            return out

        tag = f"chi={m.chi} A={m.A} n_sub={n_sub} {cur}{' dephased-' + mode if deph else ''} {ent} {engine}"
        report(f"d/dtheta_v ({tag})", relerr(g_v, fd_real(loss_v, m.theta_v[beta].copy())), 1e-6)
        report(f"d/dtheta_w ({tag})", relerr(g_w, fd_real(loss_w, m.theta_w[beta].copy())), 1e-6)


# ---------------------------------------------------------------------------
def test_cf_weights():
    print("5. CF weights vs quadrature")
    rng = np.random.default_rng(6)
    m = CoMB(3, n_m=1, n_w=1, n_sub=5, seed=7)
    m.lo = rng.normal(size=3) - 2.0
    m.hi = m.lo + 1.0 + rng.random(3) * 3
    m.h = rng.normal(size=m.h.shape)
    m.m = rng.normal(size=3)
    m.s = 0.5 + rng.random(3)
    om = rng.normal(0, 2.0, size=7)
    w = m.cf_weights(2, om)[0]
    delta, left, _ = m.readout(2)
    u = (np.arange(4000) + 0.5) / 4000                         # midpoint rule
    ref = np.exp(1j * om[:, None, None] * (left[None, :, None] + delta[None, :, None] * u[None, None, :])).mean(axis=2)
    # and the readout really is y = m + s (left_c + delta_c u)
    c = rng.integers(0, m.K, 50)
    uu = rng.random(50)
    report("apply_readout", float(np.max(np.abs((m.apply_readout(2, c, uu) - m.m[2]) / m.s[2]
                                                 - (left[c] + delta[c] * uu)))), 1e-12)
    report("w(c, omega) closed form", float(np.max(np.abs(w - ref))), 1e-6)


def random_readout(m, rng):
    m.lo = rng.normal(0, 0.3, size=m.d) - 2.0
    m.hi = m.lo + 2.0 + rng.random(m.d)
    m.h = rng.normal(0.0, 0.7, size=m.h.shape)
    m.m = rng.normal(0, 0.3, size=m.d)
    m.s = 0.4 + rng.random(m.d)


def test_cf_enumeration():
    print("6. characteristic function vs explicit sum over count vectors")
    rng = np.random.default_rng(8)
    for deph in (False, True):
        m = CoMB(3, n_m=1, n_w=1, n_sub=2, L_v=2, L_w=1, seed=9, dephase=deph)
        random_readout(m, rng)
        om = rng.normal(0, 1.5, size=(5, 3))
        phi = m.characteristic_function(om)
        lams = [m.pixel_instrument(b)[0] for b in range(m.d)]
        Ws = [m.cf_weights(b, om[:, b])[0] for b in range(m.d)]
        ref = np.zeros(5, dtype=np.complex128)
        ptot = 0.0
        for cs in itertools.product(range(m.K), repeat=m.d):
            v = m._v0.astype(np.complex128)
            for b, c in enumerate(cs):
                v = lams[b][c] @ v
            p = np.real(v[m._tr_idx].sum())
            ptot += p
            ref += p * np.prod([Ws[b][:, c] for b, c in enumerate(cs)], axis=0)
        report(f"Phi == sum_c P(c) prod w ({'dephased' if deph else 'coherent'})", float(np.max(np.abs(phi - ref))), 1e-13)
        report(f"sum_c P(c) == 1 ({'dephased' if deph else 'coherent'})", abs(ptot - 1.0), 1e-13)
        report(f"Phi(0) == 1 ({'dephased' if deph else 'coherent'})",
               float(abs(m.characteristic_function(np.zeros((1, 3)))[0] - 1)), 1e-13)


def test_cf_montecarlo():
    print("7. characteristic function and count marginals vs sampled trajectories")
    rng = np.random.default_rng(10)
    n = 200_000
    for (n_m, n_w, n_sub, cur, deph) in [(2, 1, 4, "hamming", False), (2, 1, 4, "hamming", True),
                                          (1, 2, 3, "binary", False)]:
        m = CoMB(4, n_m=n_m, n_w=n_w, n_sub=n_sub, L_v=2, L_w=2, seed=11, dephase=deph, current=cur)
        random_readout(m, rng)
        om = rng.normal(0, 1.0, size=(16, 4))
        phi = m.characteristic_function(om)
        Y = m.sample(n, np.random.default_rng(12))
        Z = (Y - m.m) / m.s
        mc = np.exp(1j * Z @ om.T).mean(axis=0)
        z = float(np.max(np.abs(phi - mc)) * np.sqrt(n))       # O(1) if correct
        tag = f"chi={m.chi} {cur}{' dephased' if deph else ''}"
        report(f"sqrt(n) max|Phi - Phi_MC| ({tag})", z, 4.5)
        cnt = m.sample_counts(n, np.random.default_rng(13))
        pc = m.count_marginals()
        emp = np.stack([np.bincount(cnt[:, b], minlength=m.K) / n for b in range(m.d)])
        zc = float(np.max(np.abs(pc - emp) / np.sqrt(pc * (1 - pc) / n + 1e-12)))
        report(f"count marginals, max |z| ({tag})", zc, 5.0)
        report(f"count marginals normalized ({tag})", float(np.max(np.abs(pc.sum(1) - 1))), 1e-12)


def test_cf_gradient():
    print("8. gradient of the characteristic function vs finite differences")
    rng = np.random.default_rng(14)
    for (n_m, n_w, n_sub, cur, deph, ent, engine) in [(1, 1, 3, "hamming", False, "cz", "fft"), (2, 1, 2, "hamming", False, "cz", "fft"),
                                                       (2, 1, 3, "hamming", False, "cz", "dp"), (1, 2, 2, "hamming", True, "cz", "fft"),
                                                       (2, 1, 4, "hamming", True, "cz", "fft"), (1, 2, 2, "binary", False, "rzz", "fft")]:
        m = CoMB(3, n_m=n_m, n_w=n_w, n_sub=n_sub, L_v=2, L_w=1, seed=15, dephase=deph, current=cur, entangler=ent,
                engine=engine)
        random_readout(m, rng)
        om = rng.normal(0, 1.2, size=(6, 3))
        G = cplx(rng, 6)
        phi, g = m.characteristic_function(om, grad_out=G)
        p0 = m.get()

        def loss(p):
            m.set(p)
            out = np.real(np.sum(G.conj() * m.characteristic_function(om)))
            m.set(p0)
            return out

        fd = fd_real(loss, p0.copy())
        nv, nw = m.theta_v.size, m.theta_w.size
        tag = f"chi={m.chi} A={m.A} {cur}{' dephased' if deph else ''} {ent} {engine}"
        for name, sl in (("theta_v", slice(0, nv)), ("theta_w", slice(nv, nv + nw)), ("h", slice(nv + nw, None))):
            report(f"dPhi/d{name} ({tag})", relerr(g[sl], fd[sl]), 1e-6)


def test_hmm():
    print("9. classical HMM reference (same CF code path)")
    rng = np.random.default_rng(16)
    h = ClassicalHMM(3, chi=3, K=4, seed=17)
    random_readout(h, rng)
    om = rng.normal(0, 1.5, size=(5, 3))
    phi = h.characteristic_function(om)
    lams = [h.pixel_instrument(b)[0] for b in range(h.d)]
    Ws = [h.cf_weights(b, om[:, b])[0] for b in range(h.d)]
    ref = np.zeros(5, dtype=np.complex128)
    for cs in itertools.product(range(h.K), repeat=h.d):
        v = h._v0.astype(complex)
        for b, c in enumerate(cs):
            v = lams[b][c] @ v
        ref += v.sum().real * np.prod([Ws[b][:, c] for b, c in enumerate(cs)], axis=0)
    report("HMM: Phi == sum_c P(c) prod w", float(np.max(np.abs(phi - ref))), 1e-13)
    n = 200_000
    Y = h.sample(n, np.random.default_rng(18))
    mc = np.exp(1j * ((Y - h.m) / h.s) @ om.T).mean(axis=0)
    report("HMM: sqrt(n) max|Phi - Phi_MC|", float(np.max(np.abs(phi - mc)) * np.sqrt(n)), 4.5)
    G = cplx(rng, 5)
    _, g = h.characteristic_function(om, grad_out=G)
    p0 = h.get()

    def loss(p):
        h.set(p)
        out = np.real(np.sum(G.conj() * h.characteristic_function(om)))
        h.set(p0)
        return out

    report("HMM: gradient (all parameters)", relerr(g, fd_real(loss, p0.copy())), 1e-6)


if __name__ == "__main__":
    test_circuit()
    test_kron_grad()
    test_instrument()
    test_instrument_vjp()
    test_cf_weights()
    test_cf_enumeration()
    test_cf_montecarlo()
    test_cf_gradient()
    test_hmm()
    print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
    sys.exit(1 if FAILS else 0)
