"""
Honest evaluation for CoMB and baselines. Rules encoded here follow
the v3 audit: (1) there is NO copula or post-hoc correlation-installing
stage anywhere in this pipeline; (2) every pixel-level table carries the
train-vs-test statistical floor AND the tokenizer's quantization floor
(oracle test tokens through the same de-tokenizer); (3) a
nearest-neighbor memorization check accompanies every sample set.
"""

from __future__ import annotations

import numpy as np

# Metrics used throughout the paper. All of them are computed from
# GENERATED images, never from the model's internal state, so the same
# functions apply unchanged to simulator output and to device tokens.
from scipy.stats import wasserstein_distance


def corr_nan_safe(M):
    with np.errstate(all="ignore"):
        C = np.corrcoef(np.asarray(M, np.float64), rowvar=False)
    return np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)


def corr_errors(C_ref, C, blocks):
    d = C_ref.shape[0]
    blk = np.empty(d, dtype=np.int64)
    for bi, (s, b) in enumerate(blocks):
        blk[s:s + b] = bi
    ii, jj = np.indices((d, d))
    off = ii != jj
    err = np.abs(C_ref - C)
    return dict(
        off=float(err[off].mean()),
        within=float(err[(blk[ii] == blk[jj]) & off].mean()),
        cross=float(err[(blk[ii] != blk[jj]) & off].mean()),
    )


def w1_mean(A, B):
    d = A.shape[1]
    return float(np.mean([wasserstein_distance(A[:, j], B[:, j])
                          for j in range(d)]))


def nn_dist(A, B, chunk=512):
    A = np.asarray(A, np.float64)
    B = np.asarray(B, np.float64)
    out = np.empty(len(A))
    BB = (B * B).sum(1)
    for s in range(0, len(A), chunk):
        a = A[s:s + chunk]
        D2 = (a * a).sum(1)[:, None] + BB[None, :] - 2 * a @ B.T
        out[s:s + chunk] = np.sqrt(np.maximum(D2, 0)).min(1)
    return out


def seam(Y, blocks):
    """Within- minus across-boundary adjacent correlation (the paper's
    "seam" column). For every pair of neighbouring cells (j, j+1), take
    the correlation of the generated images; average separately over pairs
    inside one block and pairs that straddle a block boundary, and return
    the difference. A smooth image scores 0. A model that loses the
    dependence between blocks scores clearly positive, because its
    boundary pairs decorrelate while its within-block pairs do not.
    """
    C = corr_nan_safe(Y)
    d = C.shape[0]
    blk = np.concatenate([[i] * n for i, (_, n) in enumerate(blocks)])[:d]
    within = [C[j, j + 1] for j in range(d - 1) if blk[j] == blk[j + 1]]
    across = [C[j, j + 1] for j in range(d - 1) if blk[j] != blk[j + 1]]
    return float(np.mean(within) - np.mean(across))


def pixel_report(name, Y_te, Y, C_te, blocks, Y_tr, nn_ref_median):
    c = corr_errors(C_te, corr_nan_safe(Y), blocks)
    E_te, E = Y_te.sum(1), Y.sum(1)
    nnm = float(np.median(nn_dist(Y, Y_tr)))
    row = dict(
        name=name,
        w1=w1_mean(Y_te, Y),
        corr_off=c["off"], corr_within=c["within"], corr_cross=c["cross"],
        w1_E=float(wasserstein_distance(E_te, E)),
        sigmaE_ratio=float(E.std() / E_te.std()),
        seam=seam(Y, blocks),
        nn_median=nnm,
        nn_flag=bool(nnm < 0.5 * nn_ref_median),
    )
    print(f"  {name:26s} W1={row['w1']:.5f} corr_off={row['corr_off']:.4f} "
          f"cross={row['corr_cross']:.4f} W1(E)={row['w1_E']:.4f} seam={row['seam']:+.3f} "
          f"sE_ratio={row['sigmaE_ratio']:.3f} NNmed={row['nn_median']:.4f}"
          f"{'  << MEMORIZATION FLAG' if row['nn_flag'] else ''}")
    return row


