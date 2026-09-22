"""
MMD training for CoMB token sequences.

Motivation. NLL requires evaluating p_theta(x) at data sequences, which
costs ~1/p shots per sequence on hardware and is therefore
simulation-only. Squared maximum mean discrepancy needs only SAMPLES
from model and data, which CoMB produces natively (one shot = one
token sequence). This module provides:

  1. TokenKernel — a kernel on token sequences. Default: embed each
     token to its codebook centroid (physics-aware), concatenate over
     blocks, multi-bandwidth RBF with the bandwidth family FIXED at
     initialization (the per-minibatch median heuristic of the original
     CoMB code made the loss a moving target — audited v3 lesson; do
     not reintroduce it). A Hamming product kernel is available as an
     ablation.

  2. ExactMMD — exact MMD^2 between the model law q_theta (enumerated
     over all K^B sequences) and the empirical data law p_hat, with the
     exact gradient
         MMD^2 = (q - p_hat)^T K (q - p_hat),
         d/dtheta_k MMD^2 = 2 (K (q - p_hat))^T (dq/dtheta_k),
     using the adjoint probability Jacobian of
     SequentialBornModel.probs_and_jacobian. This isolates the KERNEL's
     inductive bias from sampling noise — the registered question of
     the MMD study — and is feasible whenever K^B is enumerable
     (4096 at the d=12 configuration).

  3. sampled_mmd_grad — the hardware-compatible estimator: the
     Liu-Wang parameter-shift form
         d/dtheta_{beta,k} MMD^2 =
             [E_{x~q+,x'~q} k - E_{x~q-,x'~q} k]
           - [E_{x~q+,y~p} k - E_{x~q-,y~p} k],
     with q± the model sampled with occurrence (beta,k) shifted by
     ±pi/2. Exact in expectation by Proposition 1 of the manuscript
     (each parameter enters one rotation per occurrence; projectors are
     theta-free under free-running sampling as well, since the shift
     acts on the gate, not on the instrument structure). Verified
     against ExactMMD statistically in test T7.

References: Gretton et al., JMLR 13 (2012); Liu & Wang, PRA 98,
062324 (2018) [differentiable Born-machine MMD]; CoMB v3 audit
(bandwidth freezing).
"""

from __future__ import annotations

from math import pi

import numpy as np

_TINY = 1e-30


# ----------------------------------------------------------------------
class TokenKernel:
    """Kernel on token sequences of shape (n, B), tokens in [0, K)."""

    def __init__(self, kind: str = "embedded", sigma_mults=(0.5, 1.0, 2.0),
                 hamming_lambda: float = 0.25, corr_features: bool = False,
                 corr_weight: float = 2.0):
        """corr_features: append pairwise cross-block interaction
        features (products of block energies) to the embedding, making
        cross-block correlations first-class citizens of the MMD.
        All features are standardized on the training reference before
        the bandwidths are fixed; corr_weight scales the standardized
        interaction features relative to the marginal ones."""
        self.kind = str(kind)
        self.sigma_mults = tuple(sigma_mults)
        self.lam = float(hamming_lambda)
        self.corr = bool(corr_features)
        self.cw = float(corr_weight)
        self.centroids = None       # list over blocks: (K, b)
        self.sigmas = None
        self._mu = None
        self._sd = None

    # -- embedding -----------------------------------------------------
    def _embed_raw(self, T: np.ndarray) -> np.ndarray:
        T = np.asarray(T, np.int64)
        B = T.shape[1]
        base = np.concatenate(
            [self.centroids[b][T[:, b]] for b in range(B)], axis=1)
        if not self.corr:
            return base
        # full cross-block component products: for every block pair
        # (a, b), every centroid component of a times every component
        # of b -- these are exactly the pixel-pair covariances across
        # block boundaries that define cross-block correlation.
        cb = [self.centroids[b][T[:, b]] for b in range(B)]   # (n, b_a)
        feats = [base]
        for a in range(B):
            for b in range(a + 1, B):
                feats.append(np.einsum("ni,nj->nij", cb[a], cb[b]
                                       ).reshape(len(T), -1))
        return np.concatenate(feats, axis=1)

    def _embed(self, T: np.ndarray) -> np.ndarray:
        Z = self._embed_raw(T)
        if self._mu is not None:
            Z = (Z - self._mu) / self._sd
            if self.corr:
                d_base = sum(c.shape[1] for c in self.centroids)
                Z[:, d_base:] *= self.cw
        return Z

    def fit(self, tokenizer, T_ref: np.ndarray,
            cap: int = 2000, seed: int = 0) -> "TokenKernel":
        """Fix centroids from the tokenizer codebooks and the bandwidth
        family from the median pairwise distance of embedded reference
        (training) sequences. Called ONCE; never re-fit per batch."""
        self.centroids = [km.cluster_centers_.copy()
                          for km in tokenizer.codebooks]
        if self.kind == "embedded":
            Zr = self._embed_raw(np.asarray(T_ref, np.int64))
            self._mu = Zr.mean(axis=0)
            self._sd = np.maximum(Zr.std(axis=0), 1e-9)
            Z = self._embed(np.asarray(T_ref, np.int64))
            if len(Z) > cap:
                idx = np.random.default_rng(seed).choice(
                    len(Z), size=cap, replace=False)
                Z = Z[idx]
            D2 = self._sqdist(Z, Z)
            iu = np.triu_indices(len(Z), k=1)
            med = np.median(D2[iu][D2[iu] > 0])
            s0 = float(np.sqrt(0.5 * med) + 1e-12)
            self.sigmas = [m * s0 for m in self.sigma_mults]
        return self

    @staticmethod
    def _sqdist(A, B):
        AA = (A * A).sum(1)[:, None]
        BB = (B * B).sum(1)[None, :]
        return np.maximum(0.0, AA + BB - 2.0 * A @ B.T)

    def gram(self, T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
        T1 = np.asarray(T1, np.int64)
        T2 = np.asarray(T2, np.int64)
        if self.kind == "hamming":
            K = np.ones((len(T1), len(T2)))
            for b in range(T1.shape[1]):
                eq = (T1[:, b][:, None] == T2[:, b][None, :])
                K *= np.where(eq, 1.0, self.lam)
            return K
        Z1, Z2 = self._embed(T1), self._embed(T2)
        D2 = self._sqdist(Z1, Z2)
        K = np.zeros_like(D2)
        for s in self.sigmas:
            K += np.exp(-D2 / (2.0 * s * s))
        return K


# ----------------------------------------------------------------------
def enumerate_sequences(K: int, B: int) -> np.ndarray:
    seqs = np.stack(np.meshgrid(*[np.arange(K)] * B, indexing="ij"),
                    axis=-1).reshape(-1, B)
    return seqs.astype(np.int64)


def empirical_hist(tokens: np.ndarray, K: int, B: int) -> np.ndarray:
    """Empirical distribution over all K^B symbol sequences.

    Only usable where K^B is small enough to enumerate (4096 at d = 12).
    It is the target of the joint objective of Eq. (7) and of the exact
    gradient that the estimator audit compares against.
    """
    flat = np.zeros(len(tokens), dtype=np.int64)
    for b in range(B):
        flat = flat * K + np.asarray(tokens, np.int64)[:, b]
    h = np.zeros(K ** B)
    np.add.at(h, flat, 1.0)
    return h / max(1, len(tokens))


class ExactMMD:
    """Exact MMD^2(q_theta, p_hat) and its exact theta-gradient via the
    enumerated model law and the adjoint probability Jacobian."""

    def __init__(self, model, kernel: TokenKernel):
        self.model = model
        self.seqs = enumerate_sequences(model.K, model.B)
        # full Gram over all sequences, computed once
        self.Kmat = kernel.gram(self.seqs, self.seqs)

    def loss(self, p_hat: np.ndarray) -> float:
        q = np.exp(self.model.logp(self.seqs))
        r = q - p_hat
        return float(r @ (self.Kmat @ r))

    def loss_of_dist(self, q: np.ndarray, p_hat: np.ndarray) -> float:
        r = np.asarray(q) - np.asarray(p_hat)
        return float(r @ (self.Kmat @ r))

    def loss_and_grad(self, p_hat: np.ndarray, chunk: int = 1024):
        q, J = self.model.probs_and_jacobian(self.seqs, chunk=chunk)
        r = q - p_hat
        w = 2.0 * (self.Kmat @ r)              # (M,)
        loss = float(r @ (self.Kmat @ r))
        grad = J @ w                            # (p_theta,)
        return loss, grad, q


# ----------------------------------------------------------------------
def sampled_mmd_grad(model, kernel: TokenKernel, data_tokens: np.ndarray,
                     n_model: int = 256, n_shift: int = 256,
                     rng: np.random.Generator | None = None):
    """
    Hardware-compatible MMD^2 gradient by the per-occurrence
    parameter-shift rule with fresh samples at every shifted setting.
    Implemented for per-block theta (each parameter has exactly one
    occurrence, in its own block); for shared theta, sum occurrences.
    Returns (mmd2_estimate, grad). Cost: (2 p_theta + 1) sampling runs.
    """
    rng = rng or np.random.default_rng(0)
    if model.share:
        raise NotImplementedError(
            "sampled estimator implemented for per-block theta; "
            "sum occurrences for shared theta")
    X0 = model.sample(n_model, rng)
    Kxx = kernel.gram(X0, X0)
    Kxy = kernel.gram(X0, data_tokens)
    Kyy = kernel.gram(data_tokens, data_tokens)
    n, m = len(X0), len(data_tokens)
    mmd2 = (Kxx.sum() / (n * n) - 2 * Kxy.sum() / (n * m)
            + Kyy.sum() / (m * m))
    grad = np.zeros(model.p_theta)
    th0 = model.theta.copy()
    for k in range(model.p_theta):
        g_k = 0.0
        for sgn in (+1.0, -1.0):
            model.theta = th0.copy()
            model.theta[k] += sgn * 0.5 * pi
            Xs = model.sample(n_shift, rng)
            t_qq = kernel.gram(Xs, X0).mean()
            t_qp = kernel.gram(Xs, data_tokens).mean()
            g_k += sgn * (t_qq - t_qp)
        grad[k] = g_k
    model.theta = th0
    return float(mmd2), grad


# Per-block Gram matrices, K x K each. This is the whole reason the
# blockwise objective scales: the joint-MMD alternative needs a Gram over
# all K^B sequences, which is 4096 entries at d=12 and 3.8e22 at d=25.
# Here it is B matrices of 64 entries, whatever d is.
def block_grams(tokenizer, sigma_mults=(0.5, 1.0, 2.0)):
    """Per-block PD Grams for the blockwise conditional kernel score.
    Block-beta tokens are embedded by their codebook centroids; the
    bandwidth family is the median pairwise centroid distance times
    sigma_mults, FIXED at construction (never re-fit per batch)."""
    grams = []
    for km in tokenizer.codebooks:
        C = km.cluster_centers_
        D2 = ((C[:, None, :] - C[None, :, :]) ** 2).sum(-1)
        iu = np.triu_indices(len(C), k=1)
        med = np.median(D2[iu][D2[iu] > 0])
        s0 = float(np.sqrt(0.5 * med) + 1e-12)
        K = np.zeros_like(D2)
        for mlt in sigma_mults:
            sig = mlt * s0
            K += np.exp(-D2 / (2.0 * sig * sig))
        grams.append(K)
    return grams


def score_of_conditionals(Q, tokens, K_blocks):
    """Blockwise conditional kernel score for a model given as explicit
    per-sample conditionals Q (n, B, K) -- used to evaluate classical
    baselines under the same strictly proper metric as CoMB."""
    tokens = np.asarray(tokens, np.int64)
    n, B = tokens.shape
    L = 0.0
    for b in range(B):
        q = Q[:, b, :]
        x = tokens[:, b]
        Kb = np.asarray(K_blocks[b], np.float64)
        Kq = q @ Kb
        L += float((np.einsum("nk,nk->n", q, Kq)
                    - 2.0 * Kq[np.arange(n), x]
                    + np.diag(Kb)[x]).mean())
    return L


def reinforce_mmd_grad(model, kernel: TokenKernel, data_tokens,
                       n_model: int = 1024, rng=None):
    """
    Enumeration-free exact-score sampled gradient of
    MMD^2(q_theta, p_hat):  with witness
        f(x) = E_{x'~q}k(x,x') - E_{y~p_hat}k(x,y),
    d MMD^2/d theta = 2 E_{x~q}[ (f(x) - b) d log q(x)/d theta ],
    b = E_q[f] (baseline; exact control variate since E[dlogq]=0).
    Uses the adjoint per-sample probability Jacobian for d log q
    (exact), so the only stochasticity is the model batch. Scales to
    any K (no K^B enumeration anywhere). Returns (mmd2_est, grad).
    """
    rng = rng or np.random.default_rng(0)
    X = model.sample(n_model, rng)
    D = np.asarray(data_tokens, np.int64)
    # chunked row-means of the Grams (never materialize n x n)
    fxx = np.zeros(n_model)
    fxy = np.zeros(n_model)
    for s0 in range(0, n_model, 2048):
        Xc = X[s0:s0 + 2048]
        fxx[s0:s0 + 2048] = kernel.gram(Xc, X).mean(axis=1)
        fxy[s0:s0 + 2048] = kernel.gram(Xc, D).mean(axis=1)
    f = fxx - fxy                                    # witness at samples
    Kyy_mean = float(sum(kernel.gram(D[s0:s0 + 2048], D).sum()
                         for s0 in range(0, len(D), 2048))
                     / (len(D) * len(D)))
    mmd2 = float(fxx.mean() - 2 * fxy.mean() + Kyy_mean)
    p, J = model.probs_and_jacobian(X)               # J: (p_theta, n)
    dlogq = J / np.maximum(p, 1e-300)[None, :]
    w = 2.0 * (f - f.mean())
    grad = (dlogq * w[None, :]).mean(axis=1)
    return mmd2, grad
