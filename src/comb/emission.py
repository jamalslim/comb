"""
Causal value-conditioned emission (de-tokenizer v2) for CoMB.

Generative factorization (STRICTLY CAUSAL -> valid joint; no copula, no
post-hoc correlation installation):

    p(y) = sum_x p_theta(x)  p(y_1|x_1)  prod_{b>1} p(y_b | y_{b-1}, x_b, x_{b-1})

with p(y_b | y_{b-1}, x_b, x_{b-1}) a per-token-pair ridge linear-Gaussian
    y_b = W_{x_{b-1},x_b} [y_{b-1}; 1] + eps,   eps ~ N(0, Sigma_{x_{b-1},x_b}).

Rationale: with a single-token emission the residuals of adjacent blocks
are independent, which structurally decorrelates the pixels straddling
every block boundary (the 'seam', +0.22 at b=3 even with ORACLE tokens).
Conditioning the emission on the neighbor's realized value stitches the
boundary. All long-range (>1 block) structure still flows exclusively
through the quantum token chain; the emission only carries the local
boundary coupling that the block quantization discards. Division of
labor is stated openly in the evaluation.

Sparse pairs (< min_n training rows) fall back to the marginal per-token
Gaussian (no value coupling) for statistical robustness.
"""
from __future__ import annotations
import numpy as np


class CausalPairEmission:
    def __init__(self, blocks, K, min_n=10, shrink=0.15, lam=1e-3,
                 clip_nonneg=True):
        self.blocks = list(blocks); self.K = int(K)
        self.min_n = int(min_n); self.shrink = float(shrink)
        self.lam = float(lam); self.clip = bool(clip_nonneg)
        self.E = {}

    def _gauss(self, Xk, bb):
        mu = Xk.mean(0)
        C = np.atleast_2d(np.cov(Xk, rowvar=False)) if len(Xk) > 1 \
            else np.zeros((bb, bb))
        t = max(np.trace(C) / bb, 1e-12)
        C = (1 - self.shrink) * C + self.shrink * t * np.eye(bb)
        return mu, np.linalg.cholesky(C + 1e-12 * np.eye(bb))

    def fit(self, Y, T):
        Y = np.asarray(Y, float); T = np.asarray(T, np.int64)
        for bi, (s, bb) in enumerate(self.blocks):
            for k in range(self.K):
                if bi == 0:
                    Xk = Y[T[:, 0] == k][:, s:s + bb]
                    if len(Xk) == 0: Xk = np.zeros((1, bb))
                    mu, L = self._gauss(Xk, bb)
                    self.E[(0, k, None)] = (None, mu, L)
                    continue
                sp, bp = self.blocks[bi - 1]
                for kp in range(self.K):
                    m = (T[:, bi] == k) & (T[:, bi - 1] == kp)
                    if m.sum() < self.min_n:
                        Xk = Y[T[:, bi] == k][:, s:s + bb]
                        if len(Xk) == 0: Xk = np.zeros((1, bb))
                        mu, L = self._gauss(Xk, bb)
                        self.E[(bi, k, kp)] = (None, mu, L)
                        continue
                    Yp = Y[m][:, sp:sp + bp]; Yc = Y[m][:, s:s + bb]
                    Z = np.hstack([Yp, np.ones((m.sum(), 1))])
                    W = np.linalg.solve(Z.T @ Z + self.lam * np.eye(bp + 1),
                                        Z.T @ Yc)
                    _, L = self._gauss(Yc - Z @ W, bb)
                    self.E[(bi, k, kp)] = (W, None, L)
        return self

    def detokenize(self, T, rng):
        T = np.asarray(T, np.int64); n = len(T)
        d = sum(bb for _, bb in self.blocks)
        Y = np.zeros((n, d))
        for bi, (s, bb) in enumerate(self.blocks):
            if bi == 0:
                for k in range(self.K):
                    idx = np.where(T[:, 0] == k)[0]
                    if idx.size == 0: continue
                    _, mu, L = self.E[(0, k, None)]
                    Y[idx, s:s + bb] = mu + rng.normal(size=(idx.size, bb)) @ L.T
                continue
            sp, bp = self.blocks[bi - 1]
            for k in range(self.K):
                for kp in range(self.K):
                    idx = np.where((T[:, bi] == k) & (T[:, bi - 1] == kp))[0]
                    if idx.size == 0: continue
                    W, mu, L = self.E[(bi, k, kp)]
                    eps = rng.normal(size=(idx.size, bb)) @ L.T
                    if W is None:
                        Y[idx, s:s + bb] = mu + eps
                    else:
                        Z = np.hstack([Y[idx, sp:sp + bp],
                                       np.ones((idx.size, 1))])
                        Y[idx, s:s + bb] = Z @ W + eps
        return np.maximum(Y, 0.0) if self.clip else Y
