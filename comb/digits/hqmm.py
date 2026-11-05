"""General coherent memory (HQMM): the instrument of every round comes from an unrestricted isometry

    V : C^chi -> C^K (x) C^E (x) C^chi,     M_{c,e} = (<c, e| (x) 1) V,     Lambda_c(rho) = sum_e M_{c,e} rho M_{c,e}^dag,

with K emitted tokens and E environment levels (measured, not emitted).  Any unitary on memory + work qubits
acting on |work = 0> gives such an isometry and every isometry arises this way, so this is CoMB with an
arbitrarily deep V circuit: the largest class a register with this memory and work size can reach.  With
E >= chi^2 it contains every classical chi-state HMM with K tokens (STROKES_METHOD.md, Proposition 4).

Parameterisation: V = A (A^dag A)^{-1/2} (polar factor) of a free complex matrix A of shape (K E chi, chi).
The gradient through the polar factor is exact (eigendecomposition of the chi x chi Gram matrix, divided
differences); checked against finite differences in scripts/test_hqmm.py.
Package conventions: vec(rho) row-major, Lambda as (K, chi^2, chi^2) acting by kron(M, conj M),
complex gradients G = dL/dRe + i dL/dIm.
"""
import numpy as np

from .model import CountChain


class GeneralHQMM(CountChain):
    inst_names = ("are", "aim")

    def __init__(self, d, chi, K, E=1, seed=0, init_scale=1.0, dephase=False):
        self.d, self.chi, self.K, self.E = int(d), int(chi), int(K), int(E)
        rng = np.random.default_rng(seed)
        n = self.K * self.E * self.chi
        self.are = rng.normal(0, init_scale, (self.d, n, self.chi))
        self.aim = rng.normal(0, init_scale, (self.d, n, self.chi))
        self._init_readout_params()
        c = self.chi
        self.dephase = bool(dephase)
        if self.dephase:                     # dephased twin: same parameters, memory kept diagonal (classical, chi states)
            self._v0 = np.zeros(c, dtype=np.complex128)
            self._v0[0] = 1.0
            self._tr_idx = np.arange(c)
        else:
            self._v0 = np.zeros(c * c, dtype=np.complex128)
            self._v0[0] = 1.0       # vec(|0><0|)
            self._tr_idx = np.arange(c) * (c + 1)

    # ------------------------------------------------------------------ isometry and its adjoint
    def _isometry(self, beta):
        A = self.are[beta] + 1j * self.aim[beta]
        S = A.conj().T @ A
        s, U = np.linalg.eigh(S)
        s = np.maximum(s, 1e-12)
        Minv = (U * s ** -0.5) @ U.conj().T                                          # S^{-1/2}
        return A @ Minv, (A, U, s, Minv)

    def _isometry_vjp(self, cache, G_V):
        A, U, s, Minv = cache
        f = s ** -0.5
        ds = s[:, None] - s[None, :]
        with np.errstate(divide="ignore", invalid="ignore"):
            F = np.where(np.abs(ds) > 1e-12, (f[:, None] - f[None, :]) / ds, 0.0)
        F[np.diag_indices_from(F)] = -0.5 * s ** -1.5
        H = U.conj().T @ (A.conj().T @ G_V) @ U
        KS = U @ (F * H) @ U.conj().T
        return G_V @ Minv + A @ (KS + KS.conj().T)

    def kraus(self, beta):
        V, cache = self._isometry(beta)
        return V.reshape(self.K, self.E, self.chi, self.chi), cache

    # ------------------------------------------------------------------ instrument
    def pixel_instrument(self, beta, keep_tape=False):
        Ms, cache = self.kraus(beta)
        c = self.chi
        if self.dephase:                     # lam[c, j, i] = sum_e |M_ce[j, i]|^2: transition-emission probabilities
            lam = (np.abs(Ms) ** 2).sum(1).astype(np.complex128)
            return lam, (dict(Ms=Ms, cache=cache) if keep_tape else None)
        lam = np.einsum("ceij,cekl->cikjl", Ms, Ms.conj()).reshape(self.K, c * c, c * c)
        return lam, (dict(Ms=Ms, cache=cache) if keep_tape else None)

    def pixel_instrument_vjp(self, beta, tape, G_lam):
        c = self.chi
        Ms = tape["Ms"]
        if self.dephase:
            G_M = 2.0 * np.real(G_lam)[:, None, :, :] * Ms
        else:
            G4 = G_lam.reshape(self.K, c, c, c, c)                                   # [c, i, k, j, l]
            G_M = np.einsum("aikjl,aekl->aeij", G4, Ms) + np.einsum("aikjl,aeij->aekl", G4.conj(), Ms)
        G_A = self._isometry_vjp(tape["cache"], G_M.reshape(-1, c))
        return np.real(G_A), np.imag(G_A)

    # ------------------------------------------------------------------ sampling
    def sample_counts(self, n, rng):
        c = self.chi
        psi = np.zeros((n, c), dtype=np.complex128)
        psi[:, 0] = 1.0
        counts = np.zeros((n, self.d), dtype=np.int64)
        if self.dephase:
            state = np.zeros(n, dtype=np.int64)
            for beta in range(self.d):
                Ms, _ = self.kraus(beta)
                P = (np.abs(Ms) ** 2).transpose(2, 0, 1, 3).reshape(c * self.K * self.E, c)   # rows (j, c, e), column i
                cum = np.cumsum(P[:, state], axis=0)
                cum /= cum[-1:]
                k = np.minimum((rng.random(n)[None, :] > cum).sum(0), P.shape[0] - 1)
                state = k // (self.K * self.E)
                counts[:, beta] = (k % (self.K * self.E)) // self.E
            return counts
        for beta in range(self.d):
            Ms, _ = self.kraus(beta)
            amp = np.einsum("ceij,nj->ncei", Ms, psi).reshape(n, self.K * self.E, c)
            p = np.real((amp * amp.conj()).sum(-1))
            p /= p.sum(1, keepdims=True)
            j = np.minimum((rng.random(n)[:, None] > np.cumsum(p, axis=1)).sum(1), self.K * self.E - 1)
            psi = amp[np.arange(n), j]
            psi /= np.linalg.norm(psi, axis=1, keepdims=True)
            counts[:, beta] = j // self.E
        return counts
