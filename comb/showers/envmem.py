"""
The general coherent memory with an environment.

Each round is an arbitrary isometry V from the memory (dimension chi) into memory x work x environment
(dimension chi * K * E), parameterized by an unconstrained complex matrix X through the polar map
    V = X (X^dag X)^(-1/2),
so V is exactly an isometry for every X. The (c, e) block of V is the Kraus operator M_{c,e} of token c and
environment state e, and token c acts on the memory as Lambda_c(rho) = sum_e M_{c,e} rho M_{c,e}^dag.

With E >= chi^2 the model contains every classical chi-state hidden Markov model with K tokens (Proposition 2 of
the paper), so at its optimum it can never be worse than the best classical memory of its size. With E = 1 it is the
environment-free general memory, a Born machine. The dephased twin keeps only the diagonal of the memory, a classical
chain with transition-emission probabilities sum_e |<i'|M_{c,e}|i>|^2. For E >= chi^2 that is exactly the general
classical HMM class, in the same parameterization and with the same optimizer as the coherent model.
"""
from __future__ import annotations
import numpy as np
from .model import CountChain, _sample_columns


class EnvMemory(CountChain):
    inst_names = ("xr", "xi")

    def __init__(self, d: int, chi: int, K: int = 8, E: int | None = None, seed: int = 0, dephase: bool = False,
                 init_scale: float = 1.0):
        self.d, self.chi, self.K = int(d), int(chi), int(K)
        self.E = int(E) if E else self.chi ** 2
        self.dephase = bool(dephase)
        rng = np.random.default_rng(seed)
        shape = (self.d, self.K, self.E, self.chi, self.chi)
        self.xr = rng.normal(0.0, init_scale, shape)
        self.xi = rng.normal(0.0, init_scale, shape)
        self._init_readout_params()
        self._frozen = None
        c = self.chi
        self._diag_idx = np.arange(c) * (c + 1)
        if self.dephase:
            self._v0 = np.zeros(c)
            self._v0[0] = 1.0
            self._tr_idx = np.arange(c)
        else:
            self._v0 = np.zeros(c * c)
            self._v0[0] = 1.0                 # vec(|0><0|)
            self._tr_idx = self._diag_idx

    # ------------------------------------------------------------------ round instrument
    def _kraus(self, beta):
        X = self.xr[beta] + 1j * self.xi[beta]                            # (K, E, chi, chi)
        S = np.einsum("keji,kejl->il", X.conj(), X)                       # sum over blocks of X^dag X
        s, Q = np.linalg.eigh(S)
        s = np.clip(s, 1e-300, None)
        A = (Q * s ** -0.5) @ Q.conj().T                                  # S^(-1/2)
        return X @ A, (X, s, Q, A)

    def kraus(self, beta):
        """Kraus operators M[c, e] (chi x chi) of round beta."""
        return self._kraus(beta)[0]

    def pixel_instrument(self, beta, keep_tape=False):
        M, t = self._kraus(beta)
        if self.dephase:
            lam = (np.abs(M) ** 2).sum(axis=1).astype(np.complex128)     # lam[c, i', i]
        else:
            c = self.chi
            lam = np.einsum("keij,kelm->kiljm", M, M.conj()).reshape(self.K, c * c, c * c)   # sum_e M (x) conj(M)
        return lam, (dict(M=M, t=t) if keep_tape else None)

    def pixel_instrument_vjp(self, beta, tape, G_lam):
        M = tape["M"]
        X, s, Q, A = tape["t"]
        c = self.chi
        if self.dephase:
            G_M = 2.0 * np.real(G_lam)[:, None, :, :] * M                 # through |M|^2
        else:
            G4 = G_lam.reshape(self.K, c, c, c, c)                        # G4[k, i, l, j, m] for S[(i,l),(j,m)] = M[i,j] conj(M[l,m])
            G_M = np.einsum("kiljm,kelm->keij", G4, M) + np.einsum("kiljm,keij->kelm", G4.conj(), M)
        # polar map V = X A, A = S^(-1/2), S = sum_b X_b^dag X_b; complex convention G = dL/dRe + i dL/dIm
        G_X = G_M @ A
        B = np.einsum("keji,kejl->il", G_M.conj(), X)                     # sum_b G_b^dag X_b
        f = s ** -0.5
        ds = s[:, None] - s[None, :]
        near = np.abs(ds) <= 1e-10 * max(1.0, float(s.max()))
        F = np.where(near, (-0.5 * s ** -1.5)[:, None] * np.ones_like(ds), (f[:, None] - f[None, :]) / np.where(near, 1.0, ds))
        Km = Q @ ((Q.conj().T @ B @ Q) * F) @ Q.conj().T
        G_X = G_X + X @ (Km + Km.conj().T)
        return np.real(G_X), np.imag(G_X)

    # ------------------------------------------------------------------ sampling
    def sample_counts(self, n, rng):
        c, K = self.chi, self.K
        counts = np.zeros((n, self.d), dtype=np.int64)
        if self.dephase:
            state = np.zeros(n, dtype=np.int64)
            for beta in range(self.d):
                P = np.real(self.pixel_instrument(beta)[0])                 # P[x, i', i]
                Pflat = P.transpose(1, 0, 2).reshape(c * K, c)             # joint (i', x) given i
                j = _sample_columns(Pflat, state, rng)
                state, x = j // K, j % K
                counts[:, beta] = x
            return counts
        v = np.zeros((n, c * c), dtype=np.complex128)
        v[:, 0] = 1.0
        for beta in range(self.d):
            lam = self.pixel_instrument(beta)[0]
            q = np.einsum("kab,nb->nka", lam, v)                             # (n, K, chi^2)
            raw = np.clip(np.real(q[:, :, self._diag_idx].sum(axis=2)), 0.0, None)
            p = raw / raw.sum(axis=1, keepdims=True)
            x = np.minimum((rng.random(n)[:, None] > np.cumsum(p, axis=1)).sum(axis=1), K - 1)
            v = q[np.arange(n), x] / raw[np.arange(n), x][:, None]
            counts[:, beta] = x
        return counts


# ---------------------------------------------------------------------- exact embeddings
def embed_hmm(env: EnvMemory, hmm):
    """Write a trained classical HMM exactly into the environment memory (Proposition 2): token c and environment
    state e = (i', i) get the Kraus operator sqrt(E[c, i'] T[i', i]) |i'><i|. These never create a coherence, so the
    coherent model and its dephased twin both reproduce the HMM exactly."""
    c = env.chi
    if env.chi != hmm.chi or env.K != hmm.K or env.E < c * c:
        raise ValueError("the HMM needs the same chi and K, and an environment with E >= chi^2")
    env.xr[:] = 0.0
    env.xi[:] = 0.0
    for beta in range(env.d):
        T = hmm._softmax0(hmm.tl[beta])
        Em = hmm._softmax0(hmm.el[beta])
        for i2 in range(c):
            for i in range(c):
                env.xr[beta, :, i2 * c + i, i2, i] = np.sqrt(Em[:, i2] * T[i2, i])


def embed_circuit(env: EnvMemory, comb):
    """Write a trained coherent CoMB circuit (one work outcome per token) exactly into the environment memory: its
    Kraus operators (block of V) @ W go into environment slot 0, all other slots are zero."""
    if comb.dephase or getattr(comb, "dense", False) or comb.n_sub != 1 or comb.A != env.K or comb.chi != env.chi:
        raise ValueError("only a coherent, circuit-based CoMB with one round per value and K tokens can be embedded")
    env.xr[:] = 0.0
    env.xi[:] = 0.0
    for beta in range(env.d):
        Vb, _ = comb.cv.forward(comb.theta_v[beta], comb._E)
        W = comb.cw.unitary(comb.theta_w[beta])
        for x in range(comb.A):
            M = Vb[x::comb.A, :] @ W
            env.xr[beta, int(comb.phi[x]), 0] = M.real
            env.xi[beta, int(comb.phi[x]), 0] = M.imag
