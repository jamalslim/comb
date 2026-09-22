"""
Batched pure-state simulator with mid-circuit measurement and reset.

This is the exact-simulation backend for CoMB (the sequential Born
machine with coherent quantum memory). States are stored as complex128
arrays of shape (n,) + (2,)*nq, one pure state per batch row. Qubit q
lives on axis 1+q. Qubits [0, n_m) are the MEMORY register (never
measured mid-chain); qubits [n_m, n_m+n_w) are the WORK register,
measured and reset once per autoregressive block.

Design notes
------------
* Rotation gates accept per-sample angles (shape (n,)) because the
  encoding angles are conditioned on each sample's prefix. Variational
  angles are scalars shared across the batch.
* Because the state is pure and work-register measurement outcomes are
  recorded, conditioning on an outcome keeps the state pure: the
  post-measurement joint state is |phi_mem> (x) |x>, and resetting the
  work register is the relabeling |x> -> |0...0>. No density matrices
  are ever needed for the teacher-forced or free-running paths.
* All operations are O(n 2^nq); for the CoMB demonstration
  (n_m + n_w = 6, dim 64) a full autoregressive chain on a batch of
  thousands of samples costs milliseconds.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-300


# ----------------------------------------------------------------------
# state construction
# ----------------------------------------------------------------------

# The state is stored as (n_samples,) + (2,)*nq rather than
# (n_samples, 2**nq). That costs nothing and buys a lot: a gate on qubit q
# is a moveaxis plus a 2x2 contraction, with no index arithmetic to get
# wrong. Qubit 0 is axis 1. Memory qubits come first, work qubits last,
# which is why reshaping to (n, 2**n_m, 2**n_w) splits the two registers
# cleanly everywhere in model.py.
def init_state(n: int, nq: int) -> np.ndarray:
    """|0...0> for each of n batch rows; shape (n,) + (2,)*nq."""
    psi = np.zeros((n,) + (2,) * nq, dtype=np.complex128)
    psi[(slice(None),) + (0,) * nq] = 1.0
    return psi


# ----------------------------------------------------------------------
# gates
# ----------------------------------------------------------------------

def _bcast(theta, n_batch_axes_after: int):
    """Reshape a (n,) angle array for broadcasting over trailing axes."""
    th = np.asarray(theta, np.float64)
    if th.ndim == 0:
        return th, th
    sh = (th.shape[0],) + (1,) * n_batch_axes_after
    return th.reshape(sh), th.reshape(sh)


def apply_ry(psi: np.ndarray, theta, q: int) -> np.ndarray:
    """RY(theta) on qubit q; theta scalar or per-sample (n,)."""
    nq = psi.ndim - 1
    psi = np.moveaxis(psi, 1 + q, -1)
    th, _ = _bcast(theta, nq - 1)
    c = np.cos(th / 2.0)
    s = np.sin(th / 2.0)
    a0 = psi[..., 0]
    a1 = psi[..., 1]
    out = np.empty_like(psi)
    out[..., 0] = c * a0 - s * a1
    out[..., 1] = s * a0 + c * a1
    return np.moveaxis(out, -1, 1 + q)


def apply_rz(psi: np.ndarray, theta, q: int) -> np.ndarray:
    """RZ(theta) on qubit q; theta scalar or per-sample (n,)."""
    nq = psi.ndim - 1
    psi = np.moveaxis(psi, 1 + q, -1)
    th, _ = _bcast(theta, nq - 1)
    ph_m = np.exp(-0.5j * th)
    ph_p = np.exp(+0.5j * th)
    out = np.empty_like(psi)
    out[..., 0] = ph_m * psi[..., 0]
    out[..., 1] = ph_p * psi[..., 1]
    return np.moveaxis(out, -1, 1 + q)


def apply_rzz(psi: np.ndarray, theta, q1: int, q2: int) -> np.ndarray:
    """RZZ(theta) = exp(-i theta/2 Z@Z) on qubits q1,q2 (diagonal)."""
    nq = psi.ndim - 1
    psi = np.moveaxis(psi, 1 + q1, -1)
    psi = np.moveaxis(psi, 1 + q2 if q2 < q1 else q2, -2) if False else psi
    # simpler: build parity phase without axis gymnastics
    psi = np.moveaxis(psi, -1, 1 + q1)          # undo
    out = psi.copy()
    ph_m = np.exp(-0.5j * theta)                 # even parity
    ph_p = np.exp(+0.5j * theta)                 # odd parity
    for b1 in (0, 1):
        for b2 in (0, 1):
            idx = [slice(None)] * out.ndim
            idx[1 + q1] = b1
            idx[1 + q2] = b2
            out[tuple(idx)] = out[tuple(idx)] * (ph_m if b1 == b2 else ph_p)
    return out


def _two_qubit_view(psi, q1, q2):
    """Move qubit axes q1,q2 to the last two positions; return view+undo."""
    p = np.moveaxis(psi, 1 + q1, -1)
    ax2 = 1 + q2 - (1 if q2 > q1 else 0)
    p = np.moveaxis(p, ax2, -2)          # order: [..., q2, q1]
    return p


def _two_qubit_restore(p, q1, q2):
    ax2 = 1 + q2 - (1 if q2 > q1 else 0)
    p = np.moveaxis(p, -2, ax2)
    return np.moveaxis(p, -1, 1 + q1)


def apply_rxx(psi: np.ndarray, theta, q1: int, q2: int) -> np.ndarray:
    """RXX(theta) = exp(-i theta/2 X@X)."""
    p = _two_qubit_view(psi, q1, q2).copy()
    c, s = np.cos(0.5 * theta), np.sin(0.5 * theta)
    a00, a01 = p[..., 0, 0].copy(), p[..., 0, 1].copy()
    a10, a11 = p[..., 1, 0].copy(), p[..., 1, 1].copy()
    p[..., 0, 0] = c * a00 - 1j * s * a11
    p[..., 1, 1] = c * a11 - 1j * s * a00
    p[..., 0, 1] = c * a01 - 1j * s * a10
    p[..., 1, 0] = c * a10 - 1j * s * a01
    return _two_qubit_restore(p, q1, q2)


def apply_ryy(psi: np.ndarray, theta, q1: int, q2: int) -> np.ndarray:
    """RYY(theta) = exp(-i theta/2 Y@Y)."""
    p = _two_qubit_view(psi, q1, q2).copy()
    c, s = np.cos(0.5 * theta), np.sin(0.5 * theta)
    a00, a01 = p[..., 0, 0].copy(), p[..., 0, 1].copy()
    a10, a11 = p[..., 1, 0].copy(), p[..., 1, 1].copy()
    p[..., 0, 0] = c * a00 + 1j * s * a11
    p[..., 1, 1] = c * a11 + 1j * s * a00
    p[..., 0, 1] = c * a01 - 1j * s * a10
    p[..., 1, 0] = c * a10 - 1j * s * a01
    return _two_qubit_restore(p, q1, q2)


def apply_xx(psi: np.ndarray, q1: int, q2: int) -> np.ndarray:
    """Generator X@X."""
    p = _two_qubit_view(psi, q1, q2).copy()
    a00, a01 = p[..., 0, 0].copy(), p[..., 0, 1].copy()
    a10, a11 = p[..., 1, 0].copy(), p[..., 1, 1].copy()
    p[..., 0, 0], p[..., 1, 1] = a11, a00
    p[..., 0, 1], p[..., 1, 0] = a10, a01
    return _two_qubit_restore(p, q1, q2)


def apply_yy(psi: np.ndarray, q1: int, q2: int) -> np.ndarray:
    """Generator Y@Y."""
    p = _two_qubit_view(psi, q1, q2).copy()
    a00, a01 = p[..., 0, 0].copy(), p[..., 0, 1].copy()
    a10, a11 = p[..., 1, 0].copy(), p[..., 1, 1].copy()
    p[..., 0, 0], p[..., 1, 1] = -a11, -a00
    p[..., 0, 1], p[..., 1, 0] = a10, a01
    return _two_qubit_restore(p, q1, q2)


def apply_zz(psi: np.ndarray, q1: int, q2: int) -> np.ndarray:
    """Apply the generator Z@Z on qubits q1,q2 (sign by parity)."""
    out = psi.copy()
    for b1 in (0, 1):
        for b2 in (0, 1):
            if b1 != b2:
                idx = [slice(None)] * out.ndim
                idx[1 + q1] = b1
                idx[1 + q2] = b2
                out[tuple(idx)] *= -1.0
    return out


# CZ is diagonal, so this is a sign flip on one quadrant rather than a
# matrix multiply. Worth keeping separate from the rotations: it is by far
# the most-called gate in the entangler layer.
def apply_cz(psi: np.ndarray, q1: int, q2: int) -> np.ndarray:
    """CZ between qubits q1, q2 (in place on a copy)."""
    out = psi.copy()
    idx = [slice(None)] * out.ndim
    idx[1 + q1] = 1
    idx[1 + q2] = 1
    out[tuple(idx)] *= -1.0
    return out


# ----------------------------------------------------------------------
# work-register measurement primitives
# ----------------------------------------------------------------------

def work_probs(psi: np.ndarray, n_m: int, n_w: int) -> np.ndarray:
    """
    Born probabilities of the 2^n_w computational outcomes on the work
    register, tracing the memory register. Shape (n, 2^n_w).
    Assumes the state is normalized; rows then sum to 1 up to fp error.
    """
    n = psi.shape[0]
    psir = psi.reshape(n, 2 ** n_m, 2 ** n_w)
    return np.einsum("nmw,nmw->nw", psir, psir.conj()).real


def project_and_reset(psi: np.ndarray, outcomes: np.ndarray,
                      n_m: int, n_w: int):
    """
    Project the work register onto |outcomes>, renormalize, and reset the
    work register to |0...0>.

    Returns
    -------
    psi_new : (n,)+(2,)*(n_m+n_w) normalized post-measurement states
    p       : (n,) Born probability of each recorded outcome
    """
    n = psi.shape[0]
    psir = psi.reshape(n, 2 ** n_m, 2 ** n_w)
    mem = psir[np.arange(n), :, outcomes]                      # (n, 2^n_m)
    p = np.einsum("nm,nm->n", mem, mem.conj()).real
    mem = mem / np.sqrt(np.maximum(p, _EPS))[:, None]
    new = np.zeros((n, 2 ** n_m, 2 ** n_w), dtype=np.complex128)
    new[:, :, 0] = mem
    return new.reshape((n,) + (2,) * (n_m + n_w)), p


def sample_outcomes(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Vectorized categorical sampling per row of a (n, K) prob matrix."""
    p = np.maximum(probs, 0.0)
    p = p / np.maximum(p.sum(axis=1, keepdims=True), _EPS)
    cum = np.cumsum(p, axis=1)
    u = rng.random(p.shape[0])[:, None]
    return np.minimum((u > cum).sum(axis=1), p.shape[1] - 1).astype(np.int64)
