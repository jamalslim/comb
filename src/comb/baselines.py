"""
Classical baselines for the memory-matched comparison.

The scientific claim under test in CoMB is a MEMORY separation, not a
speed separation: a quantum sequential generator with an n_m-qubit
coherent memory (memory dimension chi = 2^{n_m}) versus the best
classical sequential generator with the same memory dimension. The
canonical classical competitor is a hidden Markov model with chi hidden
states; because the calorimeter blocks are position-dependent, the fair
version is the INHOMOGENEOUS HMM (per-position transitions/emissions),
trained by EM with multiple restarts and validation-NLL model selection.

Also provided:
  * IndependentBlocks — product of per-block empirical marginals; the
    zero-memory floor every conditioning mechanism must beat.
  * EmpiricalJoint — add-alpha-smoothed histogram over all K^B
    sequences; at small K^B this is the "cheating" upper reference that
    shows when the token problem is too easy to be informative.

Parameter counts (free parameters, for the fairness table):
  HMM(chi):        (chi-1) + (B-1) chi (chi-1) + B chi (K-1)
  CoMB(n_m):     2 L (n_m + n_w)   [shared across blocks]
The quantum model is drastically SMALLER in trained parameters at equal
memory; state that explicitly rather than letting a referee find it.
"""

from __future__ import annotations

import numpy as np

_TINY = 1e-300


# ----------------------------------------------------------------------
# The control that makes the memory ablation mean something. This model
# has the same tokenizer and the same emission and samples each block
# independently, so it is the correlation error a model with no memory
# MUST produce. The no-memory quantum model lands on top of it (0.366 vs
# 0.365), which is the point.
class IndependentBlocks:
    """Product of per-block empirical token marginals (add-1 smoothing)."""

    def __init__(self, K: int):
        self.K = int(K)
        self.P = None                          # (B, K)

    def fit(self, tokens: np.ndarray) -> "IndependentBlocks":
        tokens = np.asarray(tokens, np.int64)
        n, B = tokens.shape
        P = np.ones((B, self.K))               # add-1
        for b in range(B):
            np.add.at(P[b], tokens[:, b], 1.0)
        self.P = P / P.sum(axis=1, keepdims=True)
        return self

    def logp(self, tokens: np.ndarray) -> np.ndarray:
        tokens = np.asarray(tokens, np.int64)
        n, B = tokens.shape
        lp = np.zeros(n)
        for b in range(B):
            lp += np.log(self.P[b][tokens[:, b]])
        return lp

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        B = self.P.shape[0]
        out = np.zeros((n, B), dtype=np.int64)
        for b in range(B):
            out[:, b] = rng.choice(self.K, size=n, p=self.P[b])
        return out


# ----------------------------------------------------------------------
class EmpiricalJoint:
    """Smoothed full-histogram over K^B sequences (small K^B only)."""

    def __init__(self, K: int, B: int, alpha: float = 0.5):
        self.K, self.B, self.alpha = int(K), int(B), float(alpha)
        self.p = None

    def _flat(self, tokens):
        f = np.zeros(len(tokens), dtype=np.int64)
        for b in range(self.B):
            f = f * self.K + tokens[:, b]
        return f

    def fit(self, tokens: np.ndarray) -> "EmpiricalJoint":
        counts = np.full(self.K ** self.B, self.alpha)
        np.add.at(counts, self._flat(np.asarray(tokens, np.int64)), 1.0)
        self.p = counts / counts.sum()
        return self

    def logp(self, tokens: np.ndarray) -> np.ndarray:
        return np.log(self.p[self._flat(np.asarray(tokens, np.int64))])

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        flat = rng.choice(self.K ** self.B, size=n, p=self.p)
        out = np.zeros((n, self.B), dtype=np.int64)
        for b in range(self.B - 1, -1, -1):
            out[:, b] = flat % self.K
            flat //= self.K
        return out


# ----------------------------------------------------------------------
class InhomogeneousHMM:
    """
    chi hidden states; per-position transition T[b] (b=0..B-2) and
    emission E[b] (b=0..B-1); trained by EM (scaled forward-backward).
    """

    def __init__(self, chi: int, K: int, B: int, seed: int = 0):
        self.chi, self.K, self.B = int(chi), int(K), int(B)
        self.rng = np.random.default_rng(seed)
        self.pi = None      # (chi,)
        self.T = None       # (B-1, chi, chi)
        self.E = None       # (B, chi, K)

    def n_params(self) -> int:
        c, K, B = self.chi, self.K, self.B
        return (c - 1) + (B - 1) * c * (c - 1) + B * c * (K - 1)

    def _init(self):
        rd = self.rng
        self.pi = rd.dirichlet(np.ones(self.chi))
        self.T = rd.dirichlet(np.ones(self.chi),
                              size=(self.B - 1, self.chi))
        self.E = rd.dirichlet(np.ones(self.K),
                              size=(self.B, self.chi))

    # -------------------------- forward-backward ----------------------
    def _fb(self, tokens):
        n, B = tokens.shape
        c = self.chi
        alpha = np.zeros((B, n, c))
        beta = np.zeros((B, n, c))
        scale = np.zeros((B, n))
        e0 = self.E[0][:, tokens[:, 0]].T                  # (n, c)
        a = self.pi[None, :] * e0
        scale[0] = np.maximum(a.sum(1), _TINY)
        alpha[0] = a / scale[0][:, None]
        for b in range(1, B):
            eb = self.E[b][:, tokens[:, b]].T
            a = (alpha[b - 1] @ self.T[b - 1]) * eb
            scale[b] = np.maximum(a.sum(1), _TINY)
            alpha[b] = a / scale[b][:, None]
        beta[B - 1] = 1.0
        for b in range(B - 2, -1, -1):
            eb1 = self.E[b + 1][:, tokens[:, b + 1]].T
            beta[b] = ((beta[b + 1] * eb1) @ self.T[b].T)
            beta[b] /= scale[b + 1][:, None]
        ll = np.log(scale).sum(0)                          # (n,)
        return alpha, beta, scale, ll

    def fit(self, tokens: np.ndarray, tokens_val: np.ndarray,
            n_restarts: int = 8, n_iter: int = 200,
            tol: float = 1e-7, verbose: bool = False) -> "InhomogeneousHMM":
        tokens = np.asarray(tokens, np.int64)
        tokens_val = np.asarray(tokens_val, np.int64)
        best = (-np.inf, None)
        for r in range(n_restarts):
            self._init()
            prev = -np.inf
            for it in range(n_iter):
                alpha, beta, scale, ll = self._fb(tokens)
                mean_ll = float(ll.mean())
                gamma = alpha * beta
                gamma /= np.maximum(gamma.sum(2, keepdims=True), _TINY)
                # M-step
                new_pi = gamma[0].mean(0)
                new_T = np.zeros_like(self.T)
                for b in range(self.B - 1):
                    eb1 = self.E[b + 1][:, tokens[:, b + 1]].T
                    xi = (alpha[b][:, :, None] * self.T[b][None, :, :]
                          * (eb1 * beta[b + 1])[:, None, :])
                    xi /= np.maximum(
                        xi.sum((1, 2), keepdims=True), _TINY)
                    Tb = xi.sum(0)
                    new_T[b] = Tb / np.maximum(
                        Tb.sum(1, keepdims=True), _TINY)
                new_E = np.zeros_like(self.E)
                for b in range(self.B):
                    for k in range(self.K):
                        mask = tokens[:, b] == k
                        if mask.any():
                            new_E[b][:, k] = gamma[b][mask].sum(0)
                    new_E[b] += 1e-6
                    new_E[b] /= new_E[b].sum(1, keepdims=True)
                self.pi, self.T, self.E = new_pi, new_T, new_E
                if mean_ll - prev < tol and it > 10:
                    break
                prev = mean_ll
            val_ll = float(self.logp(tokens_val).mean())
            if verbose:
                print(f"    HMM restart {r}: train_ll={prev:.5f} "
                      f"val_ll={val_ll:.5f}")
            if val_ll > best[0]:
                best = (val_ll, (self.pi.copy(), self.T.copy(),
                                 self.E.copy()))
        self.pi, self.T, self.E = best[1]
        return self

    def logp(self, tokens: np.ndarray) -> np.ndarray:
        return self._fb(np.asarray(tokens, np.int64))[3]

    def sample(self, n: int, rng: np.random.Generator) -> np.ndarray:
        out = np.zeros((n, self.B), dtype=np.int64)
        h = np.array([rng.choice(self.chi, p=self.pi) for _ in range(n)])
        for b in range(self.B):
            probs = self.E[b][h]                          # (n, K)
            cum = np.cumsum(probs, axis=1)
            u = rng.random(n)[:, None]
            out[:, b] = np.minimum((u > cum).sum(1), self.K - 1)
            if b < self.B - 1:
                tp = self.T[b][h]
                cum = np.cumsum(tp, axis=1)
                u = rng.random(n)[:, None]
                h = np.minimum((u > cum).sum(1), self.chi - 1)
        return out


# --- conditional-distribution interfaces for the NLL-free protocol ---

def _independent_conditionals(self, tokens):
    import numpy as _np
    n = len(tokens)
    return _np.tile(self.P[None, :, :], (n, 1, 1))
IndependentBlocks.conditionals = _independent_conditionals


def _hmm_conditionals(self, tokens):
    import numpy as _np
    tokens = _np.asarray(tokens, _np.int64)
    n, B = tokens.shape
    K = self.E[0].shape[1]
    Q = _np.zeros((n, B, K))
    alpha = _np.tile(self.pi, (n, 1))
    for b in range(B):
        Q[:, b, :] = alpha @ self.E[b]
        e = self.E[b][:, tokens[:, b]].T
        a = alpha * e
        a /= _np.maximum(a.sum(1, keepdims=True), 1e-300)
        if b < B - 1:
            alpha = a @ self.T[b]
    return Q
InhomogeneousHMM.conditionals = _hmm_conditionals


def _ej_conditionals(self, tokens):
    import numpy as _np
    tokens = _np.asarray(tokens, _np.int64)
    n, B = tokens.shape
    K = int(round(len(self.p) ** (1.0 / B)))
    pj = self.p.reshape([K] * B)
    Q = _np.zeros((n, B, K))
    flat = _np.zeros(n, dtype=_np.int64)
    for b in range(B):
        marg = pj.reshape(K ** b, K, -1).sum(axis=2)
        Q[:, b, :] = marg[flat] / _np.maximum(
            marg[flat].sum(1, keepdims=True), 1e-300)
        flat = flat * K + tokens[:, b]
    return Q
EmpiricalJoint.conditionals = _ej_conditionals
