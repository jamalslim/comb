#!/usr/bin/env python3
"""Tests of the exact noise model (comb.noise) against independent constructions."""
import itertools, pathlib, sys
import numpy as np
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
from comb.showers.model import CoMB
from comb.showers.noise import NoiseModel, NoisyCoMB, _depolarize, _apply_1q, _amp_damp_dephase
from comb.showers.circuit import ry

FAILS = []
def report(name, err, tol):
    ok = err < tol
    print(f"  {'ok  ' if ok else 'FAIL'} {name:60s} {err:.2e} (tol {tol:.0e})", flush=True)
    if not ok: FAILS.append(name)

P = [np.eye(2), np.array([[0, 1], [1, 0]]), np.array([[0, -1j], [1j, 0]]), np.diag([1, -1])]
def embed(ops, n):     # ops: dict qubit -> 2x2
    out = np.array([[1.0 + 0j]])
    for q in range(n):
        out = np.kron(out, ops.get(q, np.eye(2)))
    return out
def rand_rho(D, B, rng):
    X = rng.normal(size=(B, D, D)) + 1j * rng.normal(size=(B, D, D))
    r = X @ X.conj().transpose(0, 2, 1)
    return r / np.trace(r, axis1=1, axis2=2)[:, None, None]

rng = np.random.default_rng(0)
n = 3
D = 8
rho = rand_rho(D, 4, rng)
for qs in ([1], [0, 2], [2, 1]):
    p = 0.13
    ref = (1 - p) * rho
    k = len(qs)
    for paulis in itertools.product(range(4), repeat=k):
        if all(x == 0 for x in paulis): continue
        Pm = embed({q: P[a] for q, a in zip(qs, paulis)}, n)
        ref = ref + p / (4 ** k - 1) * Pm @ rho @ Pm.conj().T
    report(f"depolarizing on {qs} == explicit Pauli sum", np.abs(_depolarize(rho, qs, p, n) - ref).max(), 1e-13)
G = ry(0.7) @ np.diag([np.exp(0.3j), 1])
for q in range(n):
    U = embed({q: G}, n)
    report(f"single-qubit gate on qubit {q}", np.abs(_apply_1q(rho, G, q, n) - U @ rho @ U.conj().T).max(), 1e-13)
gam, deph = 0.2, 0.9
K0 = np.array([[1, 0], [0, np.sqrt(1 - gam)]])
K1 = np.array([[0, np.sqrt(gam)], [0, 0]])
for q in range(n):
    a, b = embed({q: K0}, n), embed({q: K1}, n)
    ref = a @ rho @ a.conj().T + b @ rho @ b.conj().T
    Z = embed({q: P[3]}, n)
    pz = (1 - deph) / 2            # extra dephasing: coherence factor deph
    ref = (1 - pz) * ref + pz * Z @ ref @ Z
    report(f"amplitude damping + dephasing on qubit {q}", np.abs(_amp_damp_dephase(rho, q, n, gam, deph) - ref).max(), 1e-13)

# NoisyCoMB with zero noise == CoMB; trace preservation with noise
for (n_m, n_w, n_sub) in [(2, 1, 4), (1, 2, 3), (2, 3, 2)]:
    m = CoMB(3, n_m=n_m, n_w=n_w, n_sub=n_sub, L_v=2, L_w=2, seed=4)
    z = NoisyCoMB.from_model(m, NoiseModel())
    report(f"zero noise == CoMB (chi={m.chi}, A={m.A})", np.abs(z.pixel_instrument(1)[0] - m.pixel_instrument(1)[0]).max(), 1e-13)
    nz = NoiseModel(p1=1e-3, p2=2e-2, p_ro=0.03, p_reset=0.02, t_round=4.0, T1=100.0, T2=80.0)
    y = NoisyCoMB.from_model(m, nz)
    lam = y.pixel_instrument(2)[0]
    tp = lam.sum(axis=0)[y._tr_idx, :].sum(axis=0)
    e = np.zeros(len(y._v0))
    e[y._tr_idx] = 1.0
    report(f"noisy instrument trace preserving (chi={m.chi}, A={m.A})", np.abs(tp - e).max(), 1e-13)
    om = rng.normal(size=(5, 3))
    report(f"noisy Phi(0) == 1 (chi={m.chi}, A={m.A})", abs(y.characteristic_function(np.zeros((1, 3)))[0] - 1), 1e-13)
    pc = y.count_marginals()
    report(f"noisy count marginals >= 0 and normalized (chi={m.chi})", max(np.abs(pc.sum(1) - 1).max(), -min(pc.min(), 0)), 1e-12)

# readout error alone, one round: recorded counts = true counts through the flip channel
m = CoMB(2, n_m=1, n_w=3, n_sub=1, L_v=2, L_w=1, seed=7)
pr = 0.07
y = NoisyCoMB.from_model(m, NoiseModel(p_ro=pr))
p_true = m.count_marginals()[0]          # distribution of the number of ones among 3 work qubits
# flip channel on Hamming weight: each of the w ones flips to 0 w.p. pr, each zero to one w.p. pr
Cw = np.zeros((4, 4))
for w in range(4):
    for a in range(w + 1):               # a ones survive
        for b in range(3 - w + 1):       # b zeros flip to one
            from math import comb
            Cw[a + b, w] += comb(w, a) * (1 - pr) ** a * pr ** (w - a) * comb(3 - w, b) * pr ** b * (1 - pr) ** (3 - w - b)
report("readout error == binomial flip channel on counts", np.abs(y.count_marginals()[0] - Cw @ p_true).max(), 1e-13)
# exact gradient of the noisy model (noise-aware training) vs finite differences
def fd_real(f, x, eps=1e-6):
    g = np.zeros_like(x)
    for i in range(x.size):
        xp, xm = x.copy(), x.copy()
        xp.flat[i] += eps
        xm.flat[i] -= eps
        g.flat[i] = (f(xp) - f(xm)) / (2 * eps)
    return g
for (n_m, n_w, n_sub, cur, ent) in [(1, 2, 2, "binary", "cz"), (2, 1, 3, "hamming", "rzz")]:
    m = CoMB(3, n_m=n_m, n_w=n_w, n_sub=n_sub, L_v=2, L_w=2, seed=11, current=cur, entangler=ent)
    nz = NoiseModel(p1=3e-3, p2=2e-2, p_ro=0.04, p_reset=0.03, t_round=3.0, T1=60.0, T2=50.0)
    y = NoisyCoMB.from_model(m, nz)
    y.lo = rng.normal(0, 0.3, 3) - 2
    y.hi = y.lo + 3.0
    y.h = rng.normal(0, 0.5, y.h.shape)
    om = rng.normal(size=(5, 3))
    Gc = rng.normal(size=5) + 1j * rng.normal(size=5)
    _, g = y.characteristic_function(om, grad_out=Gc)
    p0 = y.get()
    def loss(p):
        y.set(p)
        out = np.real(np.sum(Gc.conj() * y.characteristic_function(om)))
        y.set(p0)
        return out
    fd = fd_real(loss, p0.copy())
    nv, nw = y.theta_v.size, y.theta_w.size
    for nm, sl in (("theta_v", slice(0, nv)), ("theta_w", slice(nv, nv + nw)), ("readout", slice(nv + nw, None))):
        report(f"noisy gradient d/d{nm} (chi={y.chi}, A={y.A}, {cur}, {ent})",
               float(np.max(np.abs(g[sl] - fd[sl])) / max(np.max(np.abs(fd[sl])), 1e-12)), 1e-6)
print("\nALL PASS" if not FAILS else f"\nFAILED: {FAILS}")
sys.exit(1 if FAILS else 0)
