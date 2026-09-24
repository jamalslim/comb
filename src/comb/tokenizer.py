"""
Block tokenizer / de-tokenizer for CoMB.

Each b-pixel block is vector-quantized into one of K = 2^{n_w} codewords
by a per-block-position k-means codebook fitted on training data. The
quantum model then owns the JOINT distribution over token sequences
(i.e., the inter-block correlation structure); marginal fine detail is
restored classically by the de-tokenizer. This is the division of labor
already anticipated by the CoMB paper's K = 2^r residual bank.

De-tokenizer modes:
  * "gaussian" (default): per-(block, token) full-covariance Gaussian
    with shrinkage, clipped to the physical support (y >= 0). Smooth,
    avoids literal resampling of training rows.
  * "pool": draw a training block uniformly from the token's pool
    (empirical; higher fidelity, but is resampling — the memorization
    check in evalx.py must accompany it).

The QUANTIZATION FLOOR matters: applying the de-tokenizer to the true
test-set tokens ("oracle tokens") gives the best pixel-level metrics any
sequence model can achieve through this tokenizer. Every pixel-level
table must report that floor.
"""

from __future__ import annotations

import numpy as np
from sklearn.cluster import KMeans


class BlockTokenizer:
    def __init__(self, blocks, K: int, seed: int = 0,
                 detok_mode: str = "gaussian", shrink: float = 0.10,
                 clip_nonneg: bool = True):
        self.blocks = list(blocks)
        self.K = int(K)
        self.seed = int(seed)
        # detok_mode is load bearing, and the default is NOT what the
        # paper uses. "neighbor" draws each block conditioned on the
        # previous block's realised values, which carries the local
        # coupling across the seam that block quantisation throws away.
        # "gaussian" drops that and the cross-block correlation error
        # degrades by roughly a factor four, while the score and the
        # marginals stay put -- so the failure is silent. Always pass it
        # explicitly; config.py does.
        self.detok_mode = str(detok_mode)
        self.shrink = float(shrink)
        self.clip = bool(clip_nonneg)
        self.codebooks = []      # per block: KMeans
        self.pools = []          # per block: list of arrays (token pools)
        self.gauss = []          # per block: list of (mu, chol)

    # ------------------------------------------------------------------
    def fit(self, Y: np.ndarray) -> "BlockTokenizer":
        Y = np.asarray(Y, np.float64)
        for bi, (s, b) in enumerate(self.blocks):
            Xb = Y[:, s:s + b]
            km = KMeans(n_clusters=self.K, n_init=10,
                        random_state=self.seed + bi)
            labels = km.fit_predict(Xb)
            # relabel clusters by ascending mean energy so that token ids
            # are comparable across blocks (purely cosmetic/stable)
            order = np.argsort(km.cluster_centers_.sum(axis=1))
            remap = np.empty(self.K, dtype=np.int64)
            remap[order] = np.arange(self.K)
            km.cluster_centers_ = km.cluster_centers_[order]
            labels = remap[labels]
            self.codebooks.append(km)
            pools, gaussians = [], []
            for k in range(self.K):
                Xk = Xb[labels == k]
                if len(Xk) == 0:                       # degenerate pool
                    Xk = km.cluster_centers_[k][None, :]
                pools.append(Xk.copy())
                mu = Xk.mean(axis=0)
                if len(Xk) > 1:
                    C = np.cov(Xk, rowvar=False)
                    C = np.atleast_2d(C)
                else:
                    C = np.zeros((b, b))
                tr = max(np.trace(C) / b, 1e-12)
                C = (1 - self.shrink) * C + self.shrink * tr * np.eye(b)
                gaussians.append((mu, np.linalg.cholesky(
                    C + 1e-12 * np.eye(b))))
            self.pools.append(pools)
            self.gauss.append(gaussians)
        if self.detok_mode == "causal":
            # The paper's emission: y_b ~ p(y_b | y_{b-1}, x_b, x_{b-1}).
            from .emission import CausalPairEmission
            self.causal = CausalPairEmission(self.blocks, self.K).fit(
                Y, self.tokenize(Y))
        if self.detok_mode == "neighbor":
            self._fit_neighbor(Y, self.tokenize(Y))
        if self.detok_mode == "gaussian_corr":
            self._fit_residual_corr(Y, self.tokenize(Y))
        return self

    def _oh(self, t):
        o = np.zeros((len(t), self.K)); o[np.arange(len(t)), t] = 1.0
        return o

    def _neighbor_feats(self, T):
        cols = [np.ones((len(T), 1))]
        Bn = len(self.blocks)
        for bi in range(Bn):
            cols.append(self._oh(T[:, bi]))
        for bi in range(Bn - 1):                 # adjacent-block interactions
            cols.append((self._oh(T[:, bi])[:, :, None] *
                         self._oh(T[:, bi + 1])[:, None, :]
                         ).reshape(len(T), -1))
        return np.concatenate(cols, axis=1)

    def _fit_neighbor(self, Y, T, ridge=5.0):
        """Declared de-tokenizer: conditional pixel mean = ridge regression
        on own + adjacent-block tokens; residuals drawn from their joint
        covariance. Fit on data before any model exists; identical for all
        models. Recovers cross-block residual covariance that per-block
        independent sampling discards. NO correlation loss is used."""
        F = self._neighbor_feats(T)
        d = sum(b for _, b in self.blocks)
        self.nb_W = np.linalg.solve(F.T @ F + ridge * np.eye(F.shape[1]),
                                    F.T @ Y)
        R = Y - F @ self.nb_W
        cov = np.cov(R.T) + 1e-9 * np.eye(d)
        self.nb_chol = np.linalg.cholesky(cov)

    def _fit_residual_corr(self, Y, T):
        """Global correlation matrix of standardized residuals
        z_i = (y_i - mu_i(token)) / sigma_i(token), fitted on TRAINING
        data before any sequence model exists; captures the cross-block
        residual correlations that block-independent sampling discards.
        Shrunk toward identity; per-token means and marginal widths are
        untouched."""
        d = sum(b for _, b in self.blocks)
        Z = np.zeros((len(Y), d))
        for bi, (s0, b) in enumerate(self.blocks):
            mu = np.stack([self.gauss[bi][k][0] for k in range(self.K)])
            sd = np.stack([np.sqrt(np.maximum(
                np.diag(self.gauss[bi][k][1] @ self.gauss[bi][k][1].T),
                1e-12)) for k in range(self.K)])
            t = T[:, bi]
            Z[:, s0:s0 + b] = (Y[:, s0:s0 + b] - mu[t]) / sd[t]
        C = np.corrcoef(Z.T)
        C = 0.9 * C + 0.1 * np.eye(d)
        self.resid_chol = np.linalg.cholesky(C + 1e-10 * np.eye(d))

    # ------------------------------------------------------------------
    def tokenize(self, Y: np.ndarray) -> np.ndarray:
        Y = np.asarray(Y, np.float64)
        n = Y.shape[0]
        T = np.zeros((n, len(self.blocks)), dtype=np.int64)
        for bi, (s, b) in enumerate(self.blocks):
            km = self.codebooks[bi]
            D2 = ((Y[:, s:s + b][:, None, :]
                   - km.cluster_centers_[None, :, :]) ** 2).sum(-1)
            T[:, bi] = np.argmin(D2, axis=1)
        return T

    # ------------------------------------------------------------------
    def detokenize(self, tokens: np.ndarray,
                   rng: np.random.Generator) -> np.ndarray:
        tokens = np.asarray(tokens, np.int64)
        n = tokens.shape[0]
        d = sum(b for _, b in self.blocks)
        Y = np.zeros((n, d))
        if self.detok_mode == "causal":
            return self.causal.detokenize(tokens, rng)
        if self.detok_mode == "neighbor":
            mu = self._neighbor_feats(tokens) @ self.nb_W
            Y = mu + rng.standard_normal((n, d)) @ self.nb_chol.T
            if self.clip:
                Y = np.maximum(Y, 0.0)
            return Y
        if self.detok_mode == "gaussian_corr":
            Zc = rng.standard_normal((n, d)) @ self.resid_chol.T
            for bi, (s0, b) in enumerate(self.blocks):
                mu = np.stack([self.gauss[bi][k][0]
                               for k in range(self.K)])
                sd = np.stack([np.sqrt(np.maximum(np.diag(
                    self.gauss[bi][k][1] @ self.gauss[bi][k][1].T),
                    1e-12)) for k in range(self.K)])
                t = tokens[:, bi]
                Y[:, s0:s0 + b] = mu[t] + sd[t] * Zc[:, s0:s0 + b]
            if self.clip:
                Y = np.maximum(Y, 0.0)
            return Y
        for bi, (s, b) in enumerate(self.blocks):
            for k in range(self.K):
                idx = np.where(tokens[:, bi] == k)[0]
                if idx.size == 0:
                    continue
                if self.detok_mode == "pool":
                    pool = self.pools[bi][k]
                    j = rng.integers(0, len(pool), size=idx.size)
                    Y[idx, s:s + b] = pool[j]
                else:
                    mu, Lc = self.gauss[bi][k]
                    z = rng.normal(size=(idx.size, b))
                    Y[idx, s:s + b] = mu[None, :] + z @ Lc.T
        if self.clip:
            Y = np.maximum(Y, 0.0)
        return Y
