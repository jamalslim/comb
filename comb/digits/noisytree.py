"""Exact noisy measurement-tree CoMB: a deployment study without hardware.

Same noise model as comb/noise.py (NoiseModel), applied to every node of the tree:
  * gate noise: every rotation is followed by single-qubit depolarizing noise p1 on its qubit, every CZ by
    two-qubit depolarizing noise p2 on its pair (in the node circuits and in the memory rotation W);
  * imperfect reset: the work qubit starts each node in |1> with probability p_reset;
  * readout: the recorded bit is flipped with probability p_ro, and the recorded bit is both the emitted bit
    and the bit that feed-forward uses to choose the next node;
  * idling: during each node's measure-and-reset window of duration t_round, every memory qubit undergoes
    amplitude damping gamma = 1 - exp(-t/T1) and pure dephasing at rate 1/T2 - 1/(2 T1).
The tree has log2(K) windows per token where monolithic CoMB has one; this module makes that trade-off exact.

Every node becomes a two-outcome instrument {S_{t,l,p,r}} on vec(rho_memory), indexed by the recorded bit r.
Composing them along the recorded path gives the instrument of the token. Everything is linear, so the exact
losses of comb/kpoint.py and comb/pencf.py and their exact gradients apply unchanged ("noise-aware
training"). dephase=True keeps the memory diagonal after W and after every node (the dephased twin under the
same noise). The density matrices of all rounds, nodes and basis inputs are propagated in one batched pass.
"""
import numpy as np

from .noise import NoiseModel, _depolarize, _amp_damp_dephase, _amp_damp_dephase_adj, _left_1q, _right_1q
from .treecomb import TreeCoMB

_Ypauli = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)


# ------------------------------------------------------------------------------ batched noisy circuits (per-group angles)
def _ry_batch(th):
    c, s = np.cos(th / 2), np.sin(th / 2)
    G = np.zeros((len(th), 2, 2), dtype=np.complex128)
    G[:, 0, 0] = c
    G[:, 0, 1] = -s
    G[:, 1, 0] = s
    G[:, 1, 1] = c
    return G


def _conj_1q_batch(rho, Gm, q, n):
    """rho -> G rho G^dag on qubit q, a different 2 x 2 matrix G for every density matrix of the batch."""
    X, D = rho.shape[0], rho.shape[1]
    L, R = 2 ** q, 2 ** (n - q - 1)
    r = np.einsum("xab,xlbrd->xlard", Gm, rho.reshape(X, L, 2, R, D)).reshape(X, D, D)
    return np.einsum("xab,xdlbr->xdlar", Gm.conj(), r.reshape(X, D, L, 2, R)).reshape(X, D, D)


def noisy_circuit_batched(circ, thetas, rho, B, p1, p2):
    """Noisy circuit applied to rho (G*B, D, D), group-major; group g uses the angles thetas[g].
    Returns the output and a tape (gate outputs before their noise)."""
    n = circ.n
    tape = []
    for op in circ.ops:
        kind, qs, p = op
        if kind == "ry":
            Gm = np.repeat(_ry_batch(thetas[:, p]), B, axis=0)
            rho = _conj_1q_batch(rho, Gm, qs[0], n)
            tape.append((op, rho, Gm))
            rho = _depolarize(rho, [qs[0]], p1, n)
        elif kind == "rz":
            dg = np.repeat(np.exp(-0.5j * thetas[:, p][:, None] * circ._zd[qs[0]][None, :]), B, axis=0)
            rho = rho * (dg[:, :, None] * dg.conj()[:, None, :])
            tape.append((op, rho, dg))
            rho = _depolarize(rho, [qs[0]], p1, n)
        elif kind == "cz":
            za, zb = circ._zd[qs[0]], circ._zd[qs[1]]
            dg = np.where((za < 0) & (zb < 0), -1.0, 1.0)
            rho = rho * (dg[:, None] * dg[None, :])[None]
            tape.append((op, None, dg))
            rho = _depolarize(rho, list(qs), p2, n)
        else:
            raise ValueError(f"noisy batched circuits support ry, rz, cz (got {kind})")
    return rho, tape


def noisy_circuit_batched_vjp(circ, thetas, tape, G, B, p1, p2):
    """Gradient wrt thetas (G, P) of a real loss with complex gradient G wrt the output density matrices."""
    n = circ.n
    Gn = thetas.shape[0]
    grad = np.zeros_like(thetas, dtype=np.float64)
    for op, out, aux in reversed(tape):
        kind, qs, p = op
        G = _depolarize(G, list(qs), p2 if kind == "cz" else p1, n)          # depolarizing channels are self-adjoint
        if kind == "ry":
            comm = _left_1q(out, _Ypauli, qs[0], n) - _right_1q(out, _Ypauli, qs[0], n)
            grad[:, p] = 0.5 * np.imag((G.conj() * comm).reshape(Gn, -1).sum(1))
            G = _conj_1q_batch(G, np.conj(np.swapaxes(aux, 1, 2)), qs[0], n)
        elif kind == "rz":
            z = circ._zd[qs[0]]
            grad[:, p] = 0.5 * np.imag((G.conj() * ((z[:, None] - z[None, :])[None] * out)).reshape(Gn, -1).sum(1))
            da = aux.conj()
            G = G * (da[:, :, None] * da.conj()[:, None, :])
        else:
            G = G * (aux[:, None] * aux[None, :])[None]
    return grad


# ------------------------------------------------------------------------------ model
class NoisyTreeCoMB(TreeCoMB):
    def __init__(self, *args, noise: NoiseModel = NoiseModel(), **kw):
        super().__init__(*args, **kw)
        self.noise = noise
        self._setup_noise()

    @classmethod
    def from_model(cls, model: TreeCoMB, noise: NoiseModel, dephase=None):
        new = cls.__new__(cls)
        new.__dict__.update({k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in model.__dict__.items()})
        new.noise = noise
        if dephase is not None:
            new.dephase = bool(dephase)
        new._setup_noise()
        return new

    def _setup_noise(self):
        c, pr = self.chi, self.noise.p_reset
        self._v0 = np.zeros(c * c, dtype=np.complex128)
        self._v0[0] = 1.0       # full density-matrix representation
        self._tr_idx = np.arange(c) * (c + 1)
        self._pd = np.zeros(c * c)
        self._pd[self._tr_idx] = 1.0                 # projector onto diagonal entries
        rin = np.zeros((c * c, 2 * c, 2 * c), dtype=np.complex128)               # E_ij (x) reset state of the work qubit
        rw = np.zeros((c * c, c, c), dtype=np.complex128)
        for i in range(c):
            for j in range(c):
                rin[i * c + j, 2 * i, 2 * j] = 1 - pr
                rin[i * c + j, 2 * i + 1, 2 * j + 1] = pr
                rw[i * c + j, i, j] = 1.0
        self._rin, self._rw = rin, rw
        p = self.noise.p_ro
        self._conf = np.array([[1 - p, p], [p, 1 - p]])                          # [recorded, true]

    def _idle(self):
        nz = self.noise
        if nz.t_round <= 0:
            return 0.0, 1.0
        gamma = 1 - np.exp(-nz.t_round / nz.T1) if np.isfinite(nz.T1) else 0.0
        rate = max((1 / nz.T2 if np.isfinite(nz.T2) else 0.0) - (0.5 / nz.T1 if np.isfinite(nz.T1) else 0.0), 0.0)
        return gamma, float(np.exp(-nz.t_round * rate))

    # ------------------------------------------------------------------ noisy instruments and adjoint
    def instruments_all(self, keep_tape=False):
        d, c, nn, nz = self.d, self.chi, self.n_nodes, self.noise
        D = Bn = c * c
        X = d * nn * Bn
        out, tn = noisy_circuit_batched(self.cn, self.theta_n.reshape(d * nn, -1), np.tile(self._rin, (d * nn, 1, 1)), Bn, nz.p1, nz.p2)
        blocks = out.reshape(X, c, 2, c, 2)
        blk = np.stack([blocks[:, :, b, :, b] for b in (0, 1)], axis=1)                  # (X, true bit, c, c)
        rec = np.einsum("rb,xbij->xrij", self._conf, blk)                                 # (X, recorded bit, c, c)
        gamma, deph = self._idle()
        if gamma > 0 or deph < 1:
            r2 = rec.reshape(-1, c, c)
            for q in range(self.n_m):
                r2 = _amp_damp_dephase(r2, q, self.n_m, gamma, deph)
            rec = r2.reshape(X, 2, c, c)
        S = rec.reshape(d * nn, Bn, 2, D).transpose(0, 2, 3, 1).reshape(d, nn, 2, D, D)  # columns = vec(out(E_ij))
        if self.cw is not None:
            rwo, tw = noisy_circuit_batched(self.cw, self.theta_w, np.tile(self._rw, (d, 1, 1)), Bn, nz.p1, nz.p2)
            SW = rwo.reshape(d, Bn, D).transpose(0, 2, 1)
        else:
            SW, tw = np.repeat(np.eye(D, dtype=np.complex128)[None], d, axis=0), None
        if self.dephase:
            Pm = self._pd[:, None] * self._pd[None, :]
            S = S * Pm
            SW = SW * Pm
        P = [SW[:, None]]
        for l in range(self.n_bits):
            lo = 2 ** l - 1
            P.append(np.matmul(S[:, lo:lo + 2 ** l], P[-1][:, :, None]).reshape(d, 2 ** (l + 1), D, D))
        tape = dict(S=S, SW=SW, P=P, tn=tn, tw=tw, gamma=gamma, deph=deph) if keep_tape else None
        return P[-1], tape

    def instruments_all_vjp(self, tape, G_lam):
        d, c, nn, nz = self.d, self.chi, self.n_nodes, self.noise
        D = Bn = c * c
        X = d * nn * Bn
        S, SW, P = tape["S"], tape["SW"], tape["P"]
        G = G_lam
        G_S = np.zeros_like(S)
        for l in range(self.n_bits - 1, -1, -1):
            lo = 2 ** l - 1
            Gc = G.reshape(d, 2 ** l, 2, D, D)
            G_S[:, lo:lo + 2 ** l] = np.matmul(Gc, np.conj(np.swapaxes(P[l], -1, -2))[:, :, None])
            G = np.matmul(np.conj(np.swapaxes(S[:, lo:lo + 2 ** l], -1, -2)), Gc).sum(axis=2)
        G_SW = G[:, 0]
        if self.dephase:
            Pm = self._pd[:, None] * self._pd[None, :]
            G_S = G_S * Pm
            G_SW = G_SW * Pm
        G_rec = G_S.reshape(d * nn, 2, D, Bn).transpose(0, 3, 1, 2).reshape(X, 2, c, c)
        gamma, deph = tape["gamma"], tape["deph"]
        if gamma > 0 or deph < 1:
            g2 = G_rec.reshape(-1, c, c)
            for q in range(self.n_m - 1, -1, -1):
                g2 = _amp_damp_dephase_adj(g2, q, self.n_m, gamma, deph)
            G_rec = g2.reshape(X, 2, c, c)
        G_blk = np.einsum("rb,xrij->xbij", self._conf, G_rec)
        G_out = np.zeros((X, c, 2, c, 2), dtype=np.complex128)
        for b in (0, 1):
            G_out[:, :, b, :, b] = G_blk[:, b]
        g_n = noisy_circuit_batched_vjp(self.cn, self.theta_n.reshape(d * nn, -1), tape["tn"], G_out.reshape(X, 2 * c, 2 * c),
                                        Bn, nz.p1, nz.p2).reshape(self.theta_n.shape)
        if self.cw is not None:
            G_rwo = G_SW.transpose(0, 2, 1).reshape(d * Bn, c, c)
            g_w = noisy_circuit_batched_vjp(self.cw, self.theta_w, tape["tw"], G_rwo, Bn, nz.p1, nz.p2)
        else:
            g_w = np.zeros_like(self.theta_w)
        return [g_n, g_w]

    # ------------------------------------------------------------------ exact sampling of the recorded tokens
    def sample_counts(self, n, rng):
        d, c = self.d, self.chi
        _, tape = self.instruments_all(keep_tape=True)
        S, SW = tape["S"], tape["SW"]
        v = np.zeros((n, c * c), dtype=np.complex128)
        v[:, 0] = 1.0
        counts = np.zeros((n, d), dtype=np.int64)
        rows = np.arange(n)
        for t in range(d):
            v = v @ SW[t].T
            p = np.zeros(n, dtype=np.int64)
            for l in range(self.n_bits):
                amp = np.einsum("nrij,nj->nri", S[t, 2 ** l - 1 + p], v)                       # (n, recorded bit, D)
                pr = np.maximum(np.real(amp[:, :, self._tr_idx].sum(-1)), 0.0)
                r = (rng.random(n) * pr.sum(1) > pr[:, 0]).astype(np.int64)
                v = amp[rows, r] / pr[rows, r][:, None]
                p = 2 * p + r
            counts[:, t] = p
        return counts
