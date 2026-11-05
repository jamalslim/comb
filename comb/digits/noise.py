"""
Exact noisy instruments for the coherent memory register (deployment study).

Noise model per round (all channels exact, density-matrix level):
  * every rotation gate is followed by single-qubit depolarizing noise p1 on
    its qubit, every CZ / RZZ by two-qubit depolarizing noise p2 on its pair;
  * work qubits are reset imperfectly: each starts in |1> with probability p_reset;
  * each work qubit readout is flipped with probability p_ro (the recorded
    outcome, not the true one, enters the counter);
  * during the measure-and-reset window of duration t_round the memory
    idles: amplitude damping gamma = 1 - exp(-t/T1) and pure dephasing with
    1/T_phi = 1/T2 - 1/(2 T1) on every memory qubit.
The memory unitary W of every pixel gets the same gate noise.

Depolarizing noise on a set S of k qubits with probability p (uniform
non-identity Pauli) is applied as
    rho -> (1 - lam) rho + lam Tr_S(rho) (x) 1_S / 2^k,   lam = p 4^k / (4^k - 1).

`NoisyCoMB` evaluates the noisy model's exact characteristic function, count
marginals and MMD with the same code paths as the noiseless model, and its
exact gradient (reverse mode through the noisy density-matrix circuits), so
the model can be trained under the device noise model ("noise-aware
training").  With every parameter zero it reproduces CoMB to machine
precision; gradients are checked against finite differences
(scripts/test_noise.py).
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np

from .circuit import ry
from .model import CoMB, counting_fft, counting_fft_vjp, counting_dp, counting_dp_vjp


@dataclass
class NoiseModel:
    p1: float = 0.0
    p2: float = 0.0
    p_ro: float = 0.0
    p_reset: float = 0.0
    t_round: float = 0.0          # microseconds
    T1: float = np.inf            # microseconds
    T2: float = np.inf
    name: str = "noiseless"

    def scaled(self, lam: float, name: str | None = None):
        """All error probabilities and idle rates multiplied by lam."""
        return NoiseModel(p1=self.p1 * lam, p2=self.p2 * lam, p_ro=self.p_ro * lam, p_reset=self.p_reset * lam,
                          t_round=self.t_round * lam, T1=self.T1, T2=self.T2, name=name or f"{self.name} x{lam:g}")

    def as_dict(self):
        return {k: (float(v) if isinstance(v, (int, float, np.floating)) else v) for k, v in asdict(self).items()}


# ---------------------------------------------------------------------------
# batched density-matrix operations on n qubits; rho has shape (B, D, D)

def _apply_1q(rho, G, q, n):
    B, D = rho.shape[0], rho.shape[1]
    L, R = 2 ** q, 2 ** (n - q - 1)
    r = rho.reshape(B, L, 2, R, D)
    r = np.einsum("ab,xlbrd->xlard", G, r).reshape(B, D, D)
    r = r.reshape(B, D, L, 2, R)
    r = np.einsum("ab,xdlbr->xdlar", G.conj(), r).reshape(B, D, D)
    return r


def _apply_diag(rho, g):
    return rho * (g[:, None] * g.conj()[None, :])[None]


def _depolarize(rho, qubits, p, n):
    if p <= 0:
        return rho
    S = sorted(qubits)
    k = len(S)
    lam = p * 4 ** k / (4 ** k - 1)
    B, D = rho.shape[0], rho.shape[1]
    t = rho.reshape((B,) + (2,) * (2 * n))
    red = t
    for q in reversed(S):                           # partial trace, highest qubit first
        nc = (red.ndim - 1) // 2
        red = np.trace(red, axis1=1 + q, axis2=1 + nc + q)
    full = np.zeros_like(t)                         # red (x) 1_S
    for bits in range(2 ** k):
        idx = [slice(None)] * (1 + 2 * n)
        for j, q in enumerate(S):
            b = (bits >> (k - 1 - j)) & 1
            idx[1 + q] = b
            idx[1 + n + q] = b
        full[tuple(idx)] = red
    return (1 - lam) * rho + lam * full.reshape(B, D, D) / 2 ** k


def _amp_damp_dephase(rho, q, n, gamma, dephase):
    """Amplitude damping gamma and extra coherence factor `dephase` on qubit q."""
    if gamma <= 0 and dephase >= 1:
        return rho
    B, D = rho.shape[0], rho.shape[1]
    L, R = 2 ** q, 2 ** (n - q - 1)
    t = rho.reshape(B, L, 2, R, L, 2, R)
    out = t.copy()
    s = np.sqrt(1 - gamma) * dephase
    out[:, :, 0, :, :, 1, :] = t[:, :, 0, :, :, 1, :] * s
    out[:, :, 1, :, :, 0, :] = t[:, :, 1, :, :, 0, :] * s
    out[:, :, 1, :, :, 1, :] = t[:, :, 1, :, :, 1, :] * (1 - gamma)
    out[:, :, 0, :, :, 0, :] = t[:, :, 0, :, :, 0, :] + gamma * t[:, :, 1, :, :, 1, :]
    return out.reshape(B, D, D)


def _right_1q(rho, M, q, n):
    """rho (I (x) M_q): right multiplication by a single-qubit matrix on qubit q."""
    B, D = rho.shape[0], rho.shape[1]
    L, R = 2 ** q, 2 ** (n - q - 1)
    r = rho.reshape(B, D, L, 2, R)
    return np.einsum("ba,xdlbr->xdlar", M, r).reshape(B, D, D)


def _left_1q(rho, M, q, n):
    B, D = rho.shape[0], rho.shape[1]
    L, R = 2 ** q, 2 ** (n - q - 1)
    r = rho.reshape(B, L, 2, R, D)
    return np.einsum("ab,xlbrd->xlard", M, r).reshape(B, D, D)


def _amp_damp_dephase_adj(G, q, n, gamma, dephase):
    """Adjoint (Hilbert-Schmidt) of _amp_damp_dephase, for reverse mode."""
    if gamma <= 0 and dephase >= 1:
        return G
    B, D = G.shape[0], G.shape[1]
    L, R = 2 ** q, 2 ** (n - q - 1)
    t = G.reshape(B, L, 2, R, L, 2, R)
    out = t.copy()
    s = np.sqrt(1 - gamma) * dephase
    out[:, :, 0, :, :, 1, :] = t[:, :, 0, :, :, 1, :] * s
    out[:, :, 1, :, :, 0, :] = t[:, :, 1, :, :, 0, :] * s
    out[:, :, 1, :, :, 1, :] = gamma * t[:, :, 0, :, :, 0, :] + (1 - gamma) * t[:, :, 1, :, :, 1, :]
    return out.reshape(B, D, D)


def noisy_circuit_tape(circ, theta, rho, p1, p2):
    """noisy_circuit with a tape: for every gate, its output (before the noise)."""
    n = circ.n
    tape = []
    for op in circ.ops:
        kind, qs, pidx = op
        if kind == "ry":
            rho = _apply_1q(rho, ry(theta[pidx]), qs[0], n)
            tape.append((op, rho))
            rho = _depolarize(rho, [qs[0]], p1, n)
        elif kind == "rz":
            rho = _apply_diag(rho, np.exp(-0.5j * theta[pidx] * circ._zd[qs[0]]))
            tape.append((op, rho))
            rho = _depolarize(rho, [qs[0]], p1, n)
        elif kind == "cz":
            za, zb = circ._zd[qs[0]], circ._zd[qs[1]]
            rho = _apply_diag(rho, np.where((za < 0) & (zb < 0), -1.0, 1.0).astype(np.complex128))
            tape.append((op, None))
            rho = _depolarize(rho, list(qs), p2, n)
        elif kind == "rzz":
            rho = _apply_diag(rho, np.exp(-0.5j * theta[pidx] * circ._zd[qs[0]] * circ._zd[qs[1]]))
            tape.append((op, rho))
            rho = _depolarize(rho, list(qs), p2, n)
        else:
            raise ValueError(kind)
    return rho, tape


def noisy_circuit_vjp(circ, theta, tape, G, p1, p2):
    """Gradient wrt theta of a real loss with complex gradient G wrt the output
    density tensors.  Every gate is rho' = U rho U^dag with U = exp(-i t P / 2),
    so dL/dt = (1/2) Im sum conj(G') (P rho' - rho' P), G' the gradient at the
    gate output; depolarizing channels are self-adjoint."""
    n = circ.n
    grad = np.zeros(circ.n_params)
    _Y = np.array([[0, -1j], [1j, 0]])
    for op, out in reversed(tape):
        kind, qs, pidx = op
        G = _depolarize(G, list(qs), p2 if kind in ("cz", "rzz") else p1, n)     # adjoint of the noise
        if kind == "ry":
            comm = _left_1q(out, _Y, qs[0], n) - _right_1q(out, _Y, qs[0], n)
            grad[pidx] = 0.5 * np.imag(np.sum(G.conj() * comm))
            U = ry(theta[pidx])
            G = _apply_1q(G, U.conj().T, qs[0], n)
        elif kind == "rz":
            z = circ._zd[qs[0]]
            grad[pidx] = 0.5 * np.imag(np.sum(G.conj() * ((z[:, None] - z[None, :])[None] * out)))
            G = _apply_diag(G, np.exp(0.5j * theta[pidx] * z))
        elif kind == "cz":
            za, zb = circ._zd[qs[0]], circ._zd[qs[1]]
            G = _apply_diag(G, np.where((za < 0) & (zb < 0), -1.0, 1.0).astype(np.complex128))
        elif kind == "rzz":
            z = circ._zd[qs[0]] * circ._zd[qs[1]]
            grad[pidx] = 0.5 * np.imag(np.sum(G.conj() * ((z[:, None] - z[None, :])[None] * out)))
            G = _apply_diag(G, np.exp(0.5j * theta[pidx] * z))
    return grad


def noisy_circuit(circ, theta, rho, p1, p2):
    """Apply circuit `circ` (comb.circuit.Circuit) with gate noise to rho (B, D, D)."""
    n = circ.n
    for op in circ.ops:
        kind, qs, pidx = op
        if kind == "ry":
            rho = _apply_1q(rho, ry(theta[pidx]), qs[0], n)
            rho = _depolarize(rho, [qs[0]], p1, n)
        elif kind == "rz":
            rho = _apply_diag(rho, np.exp(-0.5j * theta[pidx] * circ._zd[qs[0]]))
            rho = _depolarize(rho, [qs[0]], p1, n)
        elif kind == "cz":
            za, zb = circ._zd[qs[0]], circ._zd[qs[1]]
            rho = _apply_diag(rho, np.where((za < 0) & (zb < 0), -1.0, 1.0).astype(np.complex128))
            rho = _depolarize(rho, list(qs), p2, n)
        elif kind == "rzz":
            rho = _apply_diag(rho, np.exp(-0.5j * theta[pidx] * circ._zd[qs[0]] * circ._zd[qs[1]]))
            rho = _depolarize(rho, list(qs), p2, n)
        else:
            raise ValueError(kind)
    return rho


class NoisyCoMB(CoMB):
    """CoMB evaluated with an exact noise model (forward only)."""

    def __init__(self, *args, noise: NoiseModel = NoiseModel(), **kw):
        super().__init__(*args, **kw)
        if self.dephase:
            raise ValueError("NoisyCoMB is for the coherent model")
        self.noise = noise

    def _batched_ok(self):
        return False                 # the batched path of CoMB is noiseless; NoisyCoMB always goes round by round

    @classmethod
    def from_model(cls, model: CoMB, noise: NoiseModel):
        new = cls.__new__(cls)
        new.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in model.__dict__.items()})
        new.noise = noise
        return new

    def _idle(self):
        nz = self.noise
        if nz.t_round <= 0:
            return 0.0, 1.0
        gamma = 1 - np.exp(-nz.t_round / nz.T1) if np.isfinite(nz.T1) else 0.0
        rate_phi = (1 / nz.T2 if np.isfinite(nz.T2) else 0.0) - (0.5 / nz.T1 if np.isfinite(nz.T1) else 0.0)
        rate_phi = max(rate_phi, 0.0)
        return gamma, float(np.exp(-nz.t_round * rate_phi))

    def _inputs(self):
        chi, A = self.chi, self.A
        D = chi * A
        sig = np.array([1.0])
        for _ in range(self.n_w):
            sig = np.kron(sig, np.array([1 - self.noise.p_reset, self.noise.p_reset]))
        rho = np.zeros((chi * chi, D, D), dtype=np.complex128)
        for i in range(chi):
            for j in range(chi):
                for x in range(A):
                    if sig[x] > 0:
                        rho[i * chi + j, i * A + x, j * A + x] = sig[x]
        rw = np.zeros((chi * chi, chi, chi), dtype=np.complex128)
        for i in range(chi):
            for j in range(chi):
                rw[i * chi + j, i, j] = 1.0
        return rho, rw

    def _confusion(self):
        A, nz = self.A, self.noise
        Cs = np.zeros((self.smax + 1, A))                   # P(recorded increment s | true x)
        for xp in range(A):
            for x in range(A):
                diff = bin(xp ^ x).count("1")
                Cs[self.phi[xp], x] += nz.p_ro ** diff * (1 - nz.p_ro) ** (self.n_w - diff)
        return Cs

    def _elementary(self, beta, keep=False):
        nz, chi, A = self.noise, self.chi, self.A
        rho, rw = self._inputs()
        if keep:
            rho, tv = noisy_circuit_tape(self.cv, self.theta_v[beta], rho, nz.p1, nz.p2)
        else:
            rho = noisy_circuit(self.cv, self.theta_v[beta], rho, nz.p1, nz.p2)
        Cs = self._confusion()
        gamma, deph = self._idle()
        L = np.zeros((self.smax + 1, chi * chi, chi * chi), dtype=np.complex128)
        for x in range(A):
            blk = rho[:, x::A, x::A]                         # (chi^2 inputs, chi, chi)
            if gamma > 0 or deph < 1:
                for q in range(self.n_m):
                    blk = _amp_damp_dephase(blk, q, self.n_m, gamma, deph)
            cols = blk.reshape(chi * chi, chi * chi).T       # column b = vec(output for input b)
            for s in range(self.smax + 1):
                if Cs[s, x] > 0:
                    L[s] += Cs[s, x] * cols
        if keep:
            rw, tw = noisy_circuit_tape(self.cw, self.theta_w[beta], rw, nz.p1, nz.p2)
        else:
            rw = noisy_circuit(self.cw, self.theta_w[beta], rw, nz.p1, nz.p2)
        S0 = rw.reshape(chi * chi, chi * chi).T
        return L, S0, (dict(tv=tv, tw=tw, Cs=Cs, gamma=gamma, deph=deph) if keep else None)

    def pixel_instrument(self, beta, keep_tape=False):
        L, S0, _ = self._elementary(beta, keep=False)
        if self.engine == "fft":
            lam, ktape = counting_fft(L, S0, self.n_sub, self.K)
        else:
            lam, ktape = counting_dp(L, S0, self.n_sub, self.smax, self.K)
        # the circuit tapes are recomputed in the VJP (bounded memory)
        return lam, (dict(L=L, S0=S0, ktape=ktape) if keep_tape else None)

    def pixel_instrument_vjp(self, beta, tape, G_lam):
        nz, chi, A = self.noise, self.chi, self.A
        if self.engine == "fft":
            G_L, G_S0 = counting_fft_vjp(tape["ktape"], G_lam)
        else:
            G_L, G_S0 = counting_dp_vjp(tape["L"], tape["ktape"], G_lam, self.n_sub, self.smax)
        _, _, ct = self._elementary(beta, keep=True)
        Cs, gamma, deph = ct["Cs"], ct["gamma"], ct["deph"]
        D = chi * A
        G_rho = np.zeros((chi * chi, D, D), dtype=np.complex128)
        for x in range(A):
            Gc = np.zeros((chi * chi, chi * chi), dtype=np.complex128)
            for s in range(self.smax + 1):
                if Cs[s, x] > 0:
                    Gc += Cs[s, x] * G_L[s]
            G_blk = Gc.T.reshape(chi * chi, chi, chi)
            if gamma > 0 or deph < 1:
                for q in reversed(range(self.n_m)):
                    G_blk = _amp_damp_dephase_adj(G_blk, q, self.n_m, gamma, deph)
            G_rho[:, x::A, x::A] += G_blk
        g_v = noisy_circuit_vjp(self.cv, self.theta_v[beta], ct["tv"], G_rho, nz.p1, nz.p2)
        G_rw = G_S0.T.reshape(chi * chi, chi, chi)
        g_w = noisy_circuit_vjp(self.cw, self.theta_w[beta], ct["tw"], G_rw, nz.p1, nz.p2)
        return g_v, g_w

    def sample_counts(self, n, rng):
        raise NotImplementedError("use count_marginals / characteristic_function (exact)")
