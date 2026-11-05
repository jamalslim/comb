"""
Maximum mean discrepancy with a Gaussian-mixture kernel on standardized
pixels, in its two equivalent forms.

Kernel (z = standardized pixel vector, d coordinates):

    k(z, z') = (1/S) sum_s exp(-|z - z'|^2 / (2 sigma_s^2)),

with spectral measure Lambda = (1/S) sum_s N(0, sigma_s^-2 I) (Bochner), so

    MMD^2(P, Q) = E_{omega ~ Lambda} |Phi_P(omega) - Phi_Q(omega)|^2.

* Training / exact evaluation (model side exact): Phi_Q from the model's
  characteristic function, Phi_P from data; frequencies are orthogonal random
  features (Yu et al., NeurIPS 2016): blocks of d orthogonal directions with
  chi_d-distributed norms -- every omega is still exactly N(0, sigma^-2 I), so
  the estimate is unbiased, with lower variance than i.i.d. frequencies.
* The data term |Phi_P|^2 is estimated without bias by the U-statistic
  (|sum_n e^{i omega.z_n}|^2 - N) / (N (N - 1)).
* Sample form (any model): the unbiased two-sample U-statistic of Gretton et
  al. (JMLR 2012) with the same kernel.
"""
from __future__ import annotations

import numpy as np


class Spectrum:
    """Gaussian-mixture kernel and its spectral measure."""

    def __init__(self, d: int, sigmas):
        self.d = int(d)
        self.sigmas = np.asarray(sigmas, dtype=np.float64)

    def sample(self, R: int, rng, orthogonal: bool = True):
        """R frequencies (R, d), equally split between the mixture components."""
        S, d = len(self.sigmas), self.d
        per = [R // S + (1 if s < R % S else 0) for s in range(S)]
        out = []
        for s, n in enumerate(per):
            if orthogonal:
                blocks = []
                while sum(len(b) for b in blocks) < n:
                    Q, Rm = np.linalg.qr(rng.normal(size=(d, d)))
                    Q = Q * np.sign(np.diag(Rm))[None, :]              # Haar
                    norms = np.sqrt(rng.chisquare(d, size=d))
                    blocks.append((Q * norms[None, :]).T)              # rows: directions
                om = np.concatenate(blocks)[:n]
            else:
                om = rng.normal(size=(n, d))
            out.append(om / self.sigmas[s])
        return np.concatenate(out)

    def kernel(self, A, B):
        d2 = (A * A).sum(1)[:, None] + (B * B).sum(1)[None, :] - 2.0 * A @ B.T
        d2 = np.maximum(d2, 0.0)
        return np.mean([np.exp(-d2 / (2.0 * s * s)) for s in self.sigmas], axis=0)


def empirical_cf(Z, om, max_elems: int = 1 << 22):
    """mean_n exp(i om . z_n) for om (R, d); Z (N, d).  Works in blocks of at
    most max_elems (data x frequency) entries (64 MB of complex128)."""
    out = np.zeros(len(om), dtype=np.complex128)
    fc = max(1, min(len(om), max_elems // 256))
    rc = max(1, max_elems // fc)
    for f in range(0, len(om), fc):
        o = om[f:f + fc].T
        for s in range(0, len(Z), rc):
            out[f:f + fc] += np.exp(1j * (Z[s:s + rc] @ o)).sum(axis=0)
    return out / len(Z)


def cf_sum_and_sq(Z, om):
    """S = sum_n e^{i om.z_n} and the unbiased estimate of |Phi_P|^2."""
    S = empirical_cf(Z, om) * len(Z)
    N = len(Z)
    return S / N, (np.abs(S) ** 2 - N) / (N * (N - 1.0))


def mmd2_exact_model(phi_q, Z, om):
    """Unbiased (over data) estimate of E_omega |Phi_Q - Phi_P|^2 with Phi_Q exact,
    averaged over the given frequencies; also returns the per-frequency terms."""
    phi_p, sq_p = cf_sum_and_sq(Z, om)
    terms = np.abs(phi_q) ** 2 - 2.0 * np.real(phi_q * phi_p.conj()) + sq_p
    return float(terms.mean()), terms


def mmd2_u(spec: Spectrum, X, Y, chunk: int = 2048):
    """Unbiased two-sample MMD^2 (Gretton et al. 2012) between samples X and Y."""
    def total(P, Q, same):
        t, diag = 0.0, 0.0
        for s in range(0, len(P), chunk):
            K = spec.kernel(P[s:s + chunk], Q)
            t += K.sum()
            if same:
                diag += np.trace(K[:, s:s + chunk])
        return t, diag
    n, m = len(X), len(Y)
    kxx, dx = total(X, X, True)
    kyy, dy = total(Y, Y, True)
    kxy, _ = total(X, Y, False)
    return float((kxx - dx) / (n * (n - 1)) + (kyy - dy) / (m * (m - 1)) - 2.0 * kxy / (n * m))


def median_sigma(Z, n: int = 4000, seed: int = 0):
    rng = np.random.default_rng(seed)
    S = Z[rng.choice(len(Z), size=min(n, len(Z)), replace=False)]
    d2 = (S * S).sum(1)[:, None] + (S * S).sum(1)[None, :] - 2 * S @ S.T
    iu = np.triu_indices(len(S), 1)
    return float(np.sqrt(np.median(np.maximum(d2[iu], 0)) / 2.0))
