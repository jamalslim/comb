"""
CoMB: sequential Born machine with coherent quantum memory.

Model class
-----------
Registers: n_m memory qubits (never measured mid-chain) + n_w work
qubits (measured and reset once per block). One shared parameterized
block unitary U(theta, a_beta) on all n_q = n_m + n_w qubits is applied
per autoregressive step; measuring the work register in the
computational basis EMITS the block token x_beta in {0,...,2^n_w - 1}.

The induced distribution over token sequences is that of a hidden
quantum Markov model / locally purified sequential Born machine:

    p_theta(x_1,...,x_B) = || A_{x_B}(a_B) ... A_{x_1}(a_1) |0>_mem ||^2,

with Kraus operators A_x(a) = ( <x|_work (x) I_mem ) U(theta, a)
( |0>_work (x) I_mem ) acting on the 2^{n_m}-dimensional memory space.
The classically-matched competitor at equal memory is an (inhomogeneous)
HMM with 2^{n_m} hidden states (see baselines.py). The register size is
independent of the image dimension d; d only sets the chain length B,
preserving the CoMB progressive-generation principle.

Block circuit (depth L, matching the CoMB gate alphabet):
    per layer:  RY(pi * a_k) on work qubit k        (encoding re-upload)
                RZ(theta) RY(theta) on every qubit  (variational)
                CZ ring over all n_q qubits         (memory <-> work)

Conditioning channels
---------------------
* quantum memory (always on when n_m > 0): the only channel that can
  carry coherence between blocks;
* positional encoding: fixed random per-block bias in the angles;
* optional classical prefix channel ("+prefix"): the empirical token
  histogram of the prefix, mapped through a fixed random projection into
  the encoding angles. This is the token-level analogue of the original
  CoMB sketch. Under teacher forcing it is computed from data tokens, so
  the exact parameter-shift rule below remains valid (angles do not
  depend on theta). NOTE: with the prefix channel on, hardware sampling
  requires angle feed-forward (dynamic circuits); the memory-only
  configuration runs on hardware with plain mid-circuit measure+reset.

Exact gradients
---------------
Every variational parameter enters one RZ or RY per block (generator
eigenvalues +-1/2), and the surrounding operations (unitaries and the
fixed teacher-forced projectors) are linear CP maps, so the joint
probability is sinusoidal in each *occurrence* (block beta, parameter k)
and the two-term shift rule

    d p / d theta_{beta,k} = [ p(+pi/2 at (beta,k)) - p(-pi/2) ] / 2

is exact. Since theta is shared across blocks, d p / d theta_k is the
sum over occurrences. Suffix caching makes the cost
p_theta * B * (B+1) block applications per gradient instead of
2 B^2 p_theta. This is the standard treatment of parameter sharing on
hardware as well (shift one occurrence at a time).
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import Optional

import numpy as np

from .engine import (init_state, apply_ry, apply_rz, apply_cz,
                     apply_rzz, apply_zz, apply_rxx, apply_ryy,
                     apply_xx, apply_yy,
                     work_probs, project_and_reset, sample_outcomes)

_TINY = 1e-30


@dataclass
class CoMBQSpec:
    n_mem: int = 3            # memory qubits; chi = 2^n_mem
    n_work: int = 3           # work qubits;  K   = 2^n_work tokens/block
    depth: int = 2            # layers per block
    n_blocks: int = 4         # B (set by d and block size)
    use_prefix: bool = False  # classical token-histogram channel
    share_theta: bool = True  # CoMB-style parameter sharing across blocks;
    entangler: str = "ring"   # "ring" | "line"; line = heavy-hex native (no SWAPs)
    trainable_entangler: bool = False   # CZ ring -> parameterized RZZ
    entangler_gates: str = "zz"         # "zz" | "xyz" (RZZ+RYY+RXX)
    mw_dense: bool = False              # all memory-work edges added
    n_cond: int = 0     # external conditioning inputs (e.g. log10 E_inc)
                              # False = per-block theta (the faithful
                              # analogue of CoMB's per-block ridge W_beta)
    seed: int = 7


class SequentialBornModel:
    def __init__(self, spec: CoMBQSpec):
        self.spec = spec
        self.n_m = int(spec.n_mem)
        self.n_w = int(spec.n_work)
        self.nq = self.n_m + self.n_w
        self.L = int(spec.depth)
        self.B = int(spec.n_blocks)
        self.K = 2 ** self.n_w
        self.share = bool(spec.share_theta)
        edges = [(q, q + 1) for q in range(self.nq - 1)]
        if self.nq > 2 and spec.entangler == "ring":
            edges.append((self.nq - 1, 0))
        if spec.mw_dense:
            ring = set(map(frozenset, edges))
            for mq in range(spec.n_mem):
                for wq in range(spec.n_mem, self.nq):
                    if frozenset((mq, wq)) not in ring:
                        edges.append((mq, wq))
        self.edges = edges
        g_per_edge = 3 if spec.entangler_gates == "xyz" else 1
        self.p_per_block = self.L * (2 * self.nq +
            (g_per_edge * len(edges) if spec.trainable_entangler else 0))
        self.p_theta = (self.p_per_block if self.share
                        else self.B * self.p_per_block)
        rng = np.random.default_rng(spec.seed)
        self.theta = rng.normal(0.0, 0.4, size=self.p_theta)
        # fixed positional biases and (optional) prefix projection
        self.pos = rng.uniform(-1.2, 1.2, size=(self.B, self.n_w))
        self.Cproj = (0.9 * rng.normal(size=(self.n_w, self.K))
                      if spec.use_prefix else None)
        self.Wcond = (0.9 * rng.normal(size=(self.n_w, spec.n_cond))
                      if spec.n_cond > 0 else None)

    # ------------------------------------------------------------------
    # conditioning
    # ------------------------------------------------------------------
    def _angles(self, beta: int, prefix_hist: Optional[np.ndarray],
                n: int, cond: Optional[np.ndarray] = None) -> np.ndarray:
        """Encoding angles in (0,1), shape (n, n_w)."""
        z = np.broadcast_to(self.pos[beta][None, :], (n, self.n_w)).copy()
        if self.spec.n_cond > 0 and cond is not None:
            z = z + cond @ self.Wcond.T
        if self.spec.use_prefix and prefix_hist is not None and beta > 0:
            z = z + prefix_hist @ self.Cproj.T
        return 1.0 / (1.0 + np.exp(-z))

    # ------------------------------------------------------------------
    # block unitary
    # ------------------------------------------------------------------
    def _apply_block(self, psi: np.ndarray, angles: np.ndarray,
                     theta_b: np.ndarray) -> np.ndarray:
        """theta_b: (p_theta,) shared, or (n, p_theta) per-sample (the
        stacked parameter-shift path exploits the per-sample case)."""
        per_sample = (np.ndim(theta_b) == 2)
        t = 0
        for _layer in range(self.L):
            for k in range(self.n_w):                     # encode on work
                psi = apply_ry(psi, pi * angles[:, k], self.n_m + k)
            for q in range(self.nq):                      # variational
                th_rz = theta_b[:, t] if per_sample else theta_b[t]
                psi = apply_rz(psi, th_rz, q); t += 1
                th_ry = theta_b[:, t] if per_sample else theta_b[t]
                psi = apply_ry(psi, th_ry, q); t += 1
            if self.spec.trainable_entangler:              # entanglers
                kinds = ((apply_rzz, apply_ryy, apply_rxx)
                         if self.spec.entangler_gates == "xyz"
                         else (apply_rzz,))
                for (qa, qb) in self.edges:
                    for fn in kinds:
                        th = theta_b[:, t] if per_sample else theta_b[t]
                        psi = fn(psi, th, qa, qb); t += 1
            else:
                for (qa, qb) in self.edges:
                    psi = apply_cz(psi, qa, qb)
        return psi

    # ------------------------------------------------------------------
    # teacher-forced likelihood (with optional per-block state cache)
    # ------------------------------------------------------------------
    def logp(self, tokens: np.ndarray,
             cond: Optional[np.ndarray] = None,
             theta_blocks: Optional[np.ndarray] = None,
             return_cache: bool = False):
        """
        Joint log p(x_1..x_B) per sample under teacher forcing.

        theta_blocks : (B, p_theta) per-block override (parameter-shift
                       needs per-occurrence shifts); default tiles theta.
        return_cache : also return, per block beta, the state at the
                       START of block beta, the cumulative logp of the
                       prefix < beta, and the prefix histogram — enabling
                       O(B - beta) suffix re-evaluation.
        """
        tokens = np.asarray(tokens, np.int64)
        n, B = tokens.shape
        assert B == self.B
        tb = (np.stack([self._theta_block(b) for b in range(B)])
              if theta_blocks is None
              else np.asarray(theta_blocks, np.float64))
        psi = init_state(n, self.nq)
        logp = np.zeros(n)
        hist = np.zeros((n, self.K))
        cache = [] if return_cache else None
        for beta in range(B):
            if return_cache:
                cache.append((psi, logp.copy(), hist.copy()))
            a = self._angles(beta, hist / max(1, beta), n,
                             cond=cond)
            psi = self._apply_block(psi, a, tb[beta])
            probs = work_probs(psi, self.n_m, self.n_w)
            x = tokens[:, beta]
            logp = logp + np.log(np.maximum(probs[np.arange(n), x], _TINY))
            psi, _ = project_and_reset(psi, x, self.n_m, self.n_w)
            hist[np.arange(n), x] += 1.0
        return (logp, cache) if return_cache else logp

    def logp_suffix(self, tokens: np.ndarray, cache_entry, beta0: int,
                    theta_beta0: np.ndarray,
                    cond: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Re-evaluate joint logp when only block beta0's parameters change:
        prefix < beta0 is read from the cache, block beta0 uses
        theta_beta0, blocks > beta0 use the shared self.theta.
        """
        tokens = np.asarray(tokens, np.int64)
        n, B = tokens.shape
        psi, logp, hist = cache_entry
        logp = logp.copy()
        hist = hist.copy()
        for beta in range(beta0, B):
            th_b = (theta_beta0 if beta == beta0
                    else self._theta_block(beta))
            a = self._angles(beta, hist / max(1, beta), n,
                             cond=cond)
            psi = self._apply_block(psi, a, th_b)
            probs = work_probs(psi, self.n_m, self.n_w)
            x = tokens[:, beta]
            logp = logp + np.log(np.maximum(probs[np.arange(n), x], _TINY))
            psi, _ = project_and_reset(psi, x, self.n_m, self.n_w)
            hist[np.arange(n), x] += 1.0
        return logp

    # ------------------------------------------------------------------
    # exact NLL gradient by per-occurrence parameter shift
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # adjoint-state exact gradient (default; O(1) chain passes)
    # ------------------------------------------------------------------
    def _theta_offset(self, beta: int) -> int:
        return 0 if self.share else beta * self.p_per_block

    def _theta_block(self, beta: int) -> np.ndarray:
        off = self._theta_offset(beta)
        return self.theta[off:off + self.p_per_block]

    def _block_ops(self, angles: np.ndarray, beta: int):
        """Op list for one block: ('ry'|'rz', qubit, angle, param_idx)
        or ('cz', q1, q2, None). param_idx is a GLOBAL index into
        self.theta (occurrence accumulation under sharing is automatic);
        encoding gates carry param_idx=None."""
        ops = []
        t = self._theta_offset(beta)
        for _layer in range(self.L):
            for k in range(self.n_w):
                ops.append(("ry", self.n_m + k, pi * angles[:, k], None))
            for q in range(self.nq):
                ops.append(("rz", q, None, t)); t += 1
                ops.append(("ry", q, None, t)); t += 1
            if self.spec.trainable_entangler:
                kinds = (("rzz", "ryy", "rxx")
                         if self.spec.entangler_gates == "xyz"
                         else ("rzz",))
                for (qa, qb) in self.edges:
                    for kk in kinds:
                        ops.append((kk, qa, qb, t)); t += 1
            else:
                for (qa, qb) in self.edges:
                    ops.append(("cz", qa, qb, None))
        return ops

    def _apply_op(self, psi, op, sign=+1.0):
        kind, a, b, pidx = op
        if kind == "cz":
            return apply_cz(psi, a, b)
        if kind == "rzz":
            return apply_rzz(psi, sign * self.theta[pidx], a, b)
        if kind == "ryy":
            return apply_ryy(psi, sign * self.theta[pidx], a, b)
        if kind == "rxx":
            return apply_rxx(psi, sign * self.theta[pidx], a, b)
        ang = (self.theta[pidx] if pidx is not None else b)
        ang = sign * ang
        return apply_ry(psi, ang, a) if kind == "ry" \
            else apply_rz(psi, ang, a)

    @staticmethod
    def _apply_pauli(psi, kind, q):
        """Apply the rotation generator (Z for rz, Y for ry) to qubit q."""
        psi = np.moveaxis(psi.copy(), 1 + q, -1)
        if kind == "rz":
            psi[..., 1] *= -1.0
        else:  # Y
            a0 = psi[..., 0].copy()
            psi[..., 0] = -1j * psi[..., 1]
            psi[..., 1] = 1j * a0
        return np.moveaxis(psi, -1, 1 + q)

    def probs_and_jacobian(self, tokens: np.ndarray,
                           chunk: int = 1024,
                           cond: Optional[np.ndarray] = None):
        """
        Exact per-sequence joint probabilities and their full Jacobian,
            p_i = p_theta(tokens_i),   J[k, i] = d p_i / d theta_k,
        by the checkpointed adjoint-state method (see nll_and_grad
        docstring). Chunked over sequences to bound the within-block
        forward-state memory. This is the shared core for both the NLL
        gradient and the exact-MMD gradient (mmd.py).
        """
        tokens = np.asarray(tokens, np.int64)
        N = tokens.shape[0]
        if N > chunk:
            ps, Js = [], []
            for s in range(0, N, chunk):
                p_c, J_c = self.probs_and_jacobian(tokens[s:s + chunk],
                                                   chunk=chunk)
                ps.append(p_c); Js.append(J_c)
            return np.concatenate(ps), np.concatenate(Js, axis=1)
        n, B = tokens.shape
        # ---- forward (subnormalized), storing block checkpoints ----
        psi = init_state(n, self.nq)
        hist = np.zeros((n, self.K))
        ckpt, block_ops, block_x = [], [], []
        for beta in range(B):
            a = self._angles(beta, hist / max(1, beta), n,
                             cond=cond)
            ops = self._block_ops(a, beta)
            ckpt.append(psi)
            block_ops.append(ops)
            x = tokens[:, beta]
            block_x.append(x)
            for op in ops:
                psi = self._apply_op(psi, op)
            # subnormalized measure-and-reset: |0><x| on work register
            psir = psi.reshape(n, 2 ** self.n_m, 2 ** self.n_w)
            mem = psir[np.arange(n), :, x]
            new = np.zeros_like(psir)
            new[:, :, 0] = mem
            psi = new.reshape(psi.shape)
            hist[np.arange(n), x] += 1.0
        chi = psi
        chif = chi.reshape(chi.shape[0], -1)
        p = np.einsum("nk,nk->n", chif, chif.conj()).real
        # ---- reverse sweep ----
        lam = chi
        dp = np.zeros((self.p_theta, n))
        for beta in range(B - 1, -1, -1):
            x = block_x[beta]
            # lambda <- M_beta^dagger lambda = |x><0| lambda
            lr = lam.reshape(n, 2 ** self.n_m, 2 ** self.n_w)
            mem = lr[:, :, 0]
            new = np.zeros_like(lr)
            new[np.arange(n), :, x] = mem
            lam = new.reshape(lam.shape)
            # recompute within-block forward states from the checkpoint
            ops = block_ops[beta]
            states = [ckpt[beta]]
            for op in ops:
                states.append(self._apply_op(states[-1], op))
            # reverse over gates
            for j in range(len(ops) - 1, -1, -1):
                kind, q, q2_, pidx = ops[j]
                if pidx is not None:
                    if kind == "rzz":
                        sp = apply_zz(states[j + 1], q, q2_)
                    elif kind == "ryy":
                        sp = apply_yy(states[j + 1], q, q2_)
                    elif kind == "rxx":
                        sp = apply_xx(states[j + 1], q, q2_)
                    else:
                        sp = self._apply_pauli(states[j + 1], kind, q)
                    z = np.einsum(
                        "nk,nk->n", lam.conj().reshape(n, -1),
                        sp.reshape(n, -1))
                    dp[pidx] += z.imag
                if kind != "cz":
                    lam = self._apply_op(lam, ops[j], sign=-1.0)
                else:
                    lam = self._apply_op(lam, ops[j])
        return p, dp

    def nll_and_grad(self, tokens: np.ndarray, chunk: int = 1024):
        """
        Exact NLL gradient by the adjoint-state method with block
        checkpoints. The teacher-forced joint probability is
        p = <chi|chi> with |chi> = M_B U_B ... M_1 U_1 |0>, where M_beta
        = |0><x_beta| on the work register (linear, theta-free). For a
        rotation gate G(theta) = exp(-i theta S / 2),
            dp/dtheta = 2 Re <lambda| (-i S / 2) |psi_after_G>
                      = Im <lambda| S |psi_after_G>,
        summed over the shared parameter's occurrences, with lambda the
        adjoint state (all later operations, daggered, applied to chi).
        M_beta is not invertible, so the reverse sweep recomputes the
        within-block forward states from a per-block checkpoint. Cost:
        ~3 chain-equivalents, independent of p_theta. Verified against
        the per-occurrence parameter-shift rule and finite differences
        in tests_selfaudit.py (T3/T3b/T3c).
        """
        p, dp = self.probs_and_jacobian(tokens, chunk=chunk)
        p_safe = np.maximum(p, _TINY)
        nll = float(-np.mean(np.log(p_safe)))
        grad = -(dp / p_safe[None, :]).mean(axis=1)
        return nll, grad

    def nll_and_grad_shift(self, tokens: np.ndarray,
                           cond=None):
        """
        Returns (nll, grad) with nll = -mean_i log p_i and the exact
        gradient d nll / d theta by the per-occurrence shift rule.

        Vectorization: for each block beta, all 2*p_theta shifted
        evaluations are stacked into one batch of size 2*p_theta*n and
        the suffix beta..B-1 is simulated once, using per-sample
        variational angles for block beta (shifted) and shared angles
        for the later blocks (unshifted). This collapses the Python
        gate-dispatch overhead by a factor 2*p_theta.
        """
        if not self.share:
            raise NotImplementedError(
                "shift-rule path implemented for shared theta only; "
                "use nll_and_grad (adjoint) for per-block theta")
        tokens = np.asarray(tokens, np.int64)
        n = tokens.shape[0]
        logp0, cache = self.logp(tokens, return_cache=True)
        p0 = np.exp(logp0)
        inv_p0 = 1.0 / np.maximum(p0, _TINY)
        nsh = 2 * self.p_theta
        # per-copy theta for the shifted block: copy 2k is +shift on
        # parameter k, copy 2k+1 is -shift on parameter k
        theta_copies = np.tile(self.theta, (nsh, 1))
        for k in range(self.p_theta):
            theta_copies[2 * k, k] += 0.5 * pi
            theta_copies[2 * k + 1, k] -= 0.5 * pi
        grad = np.zeros(self.p_theta)
        for beta in range(self.B):
            psi_c, logp_c, hist_c = cache[beta]
            psi = np.tile(psi_c, (nsh,) + (1,) * (psi_c.ndim - 1))
            logp = np.tile(logp_c, nsh)
            hist = np.tile(hist_c, (nsh, 1))
            toks = np.tile(tokens, (nsh, 1))
            th_beta = np.repeat(theta_copies, n, axis=0)   # (nsh*n, p)
            N = nsh * n
            for b2 in range(beta, self.B):
                a = self._angles(b2, hist / max(1, b2), N, cond=cond)
                th_b = th_beta if b2 == beta else self.theta
                psi = self._apply_block(psi, a, th_b)
                probs = work_probs(psi, self.n_m, self.n_w)
                x = toks[:, b2]
                logp = logp + np.log(
                    np.maximum(probs[np.arange(N), x], _TINY))
                psi, _ = project_and_reset(psi, x, self.n_m, self.n_w)
                hist[np.arange(N), x] += 1.0
            lp = logp.reshape(nsh, n)
            dp = 0.5 * (np.exp(lp[0::2]) - np.exp(lp[1::2]))  # (p, n)
            grad += -(dp * inv_p0[None, :]).mean(axis=1)
        return float(-np.mean(logp0)), grad


    # ------------------------------------------------------------------
    # blockwise conditional kernel score (the "CoMB-fitting" MMD)
    # ------------------------------------------------------------------
    # The adjoint sweep below is the subtle part of this file. Two things
    # to keep straight if you modify it:
    #
    #   1. The convention is lambda = dL/dpsi*, NOT dL/dpsi. Every sign in
    #      the gradient accumulation follows from that choice; flip it and
    #      the gradient comes out negated, which finite differences will
    #      catch immediately and nothing else will.
    #
    #   2. The loss is scale-invariant in psi (q = u/s divides the norm
    #      out), so the radial component of lambda vanishes and we can
    #      divide by the stored norms without a projection term. That is
    #      why the backward pass looks simpler than it has any right to.
    #
    # If you touch this, run the finite-difference check before anything
    # else. It is cheap and it has caught every sign error so far.
    def blockwise_score_and_grad(self, tokens: np.ndarray, K_blocks,
             cond: Optional[np.ndarray] = None,
                                 chunk: int = 1024):
        """
        Teacher-forced blockwise conditional kernel score
            L = (1/nB) sum_i sum_beta MMD^2_{K_beta}( q_theta(.|x_<beta^i),
                                                      delta_{x_beta^i} ),
        with per-block PD Grams K_beta. Since
            E_{x~p}[ MMD^2(q, delta_x) ] = MMD^2(q, p) + const(p),
        this is a STRICTLY PROPER scoring rule (PD Gram): its population
        minimizer is q = p conditional-by-conditional, hence the joint
        --- the correlations are never measured by the loss; they are
        produced by the conditioning chain. This is the quantum-native
        form of CoMB's blockwise teacher-forced MMD (Algorithm 1), and
        unlike the joint MMD it needs no K^B enumeration: cost is
        linear in B (fully scalable in image dimension d).

        Gradient: reverse-mode through the UNNORMALIZED teacher-forced
        chain. With u_beta(x) = p(x_<beta, x) = psi^dag Pi_x psi at the
        pre-measurement state, s_beta = sum_x u_beta(x), and
        q = u / s, classical calculus gives dL/du, and the adjoint sweep
        of nll_and_grad is generalized by INJECTING
        (dL/du_beta(x)) Pi_x psi_beta^pre into lambda at each block
        before back-propagating through that block's gates (the
        convention lambda = dL/dpsi^* matches the existing
        p = <chi|chi> case where lambda = chi). Verified vs central
        finite differences (test T8).

        Returns (loss, grad, per_block_mean_scores).
        """
        tokens = np.asarray(tokens, np.int64)
        N = tokens.shape[0]
        if N > chunk:
            tot_l, tot_g = 0.0, np.zeros(self.p_theta)
            tot_pb = np.zeros(self.B)
            for s in range(0, N, chunk):
                l_c, g_c, pb_c = self.blockwise_score_and_grad(
                    tokens[s:s + chunk], K_blocks, chunk=chunk)
                w = (min(N, s + chunk) - s) / N
                tot_l += w * l_c
                tot_g += w * g_c
                tot_pb += w * pb_c
            return tot_l, tot_g, tot_pb
        n, B = tokens.shape
        Kb = [np.asarray(K_blocks[b] if not isinstance(K_blocks, np.ndarray)
                         else K_blocks, np.float64) for b in range(B)]
        # ---------------- forward (unnormalized), with records --------
        psi = init_state(n, self.nq)
        hist = np.zeros((n, self.K))
        ckpt, block_ops, block_x = [], [], []
        u_all, dLdu_all = [], []
        loss = 0.0
        pb = np.zeros(B)
        for beta in range(B):
            a = self._angles(beta, hist / max(1, beta), n,
                             cond=cond)
            ops = self._block_ops(a, beta)
            ckpt.append(psi)
            block_ops.append(ops)
            x = tokens[:, beta]
            block_x.append(x)
            for op in ops:
                psi = self._apply_op(psi, op)
            psir = psi.reshape(n, 2 ** self.n_m, 2 ** self.n_w)
            u = np.einsum("nmx,nmx->nx", psir, psir.conj()).real  # (n,K)
            s = np.maximum(u.sum(axis=1), _TINY)                  # (n,)
            q = u / s[:, None]
            Kq = q @ Kb[beta]                                     # (n,K)
            Kxs = Kb[beta][x]                                     # (n,K)
            diagK = np.diag(Kb[beta])[x]                          # (n,)
            l_i = np.einsum("nk,nk->n", q, Kq) - 2.0 * Kq[
                np.arange(n), x] + diagK
            pb[beta] = float(l_i.mean())
            loss += pb[beta]
            # dL_i/dq = 2Kq - 2K[:,x];  dL/du_x = [g_x - q.g]/s
            g_q = 2.0 * (Kq - Kxs)
            qdotg = np.einsum("nk,nk->n", q, g_q)
            dLdu = (g_q - qdotg[:, None]) / s[:, None] / (n * 1.0)
            u_all.append(None)
            dLdu_all.append(dLdu)
            # continue chain: unnormalized projection onto data token
            mem = psir[np.arange(n), :, x]
            new = np.zeros_like(psir)
            new[:, :, 0] = mem
            psi = new.reshape(psi.shape)
            hist[np.arange(n), x] += 1.0
        loss = float(loss)
        # ---------------- reverse sweep with injections ---------------
        lam = np.zeros_like(psi)
        dtheta = np.zeros(self.p_theta)
        for beta in range(B - 1, -1, -1):
            x = block_x[beta]
            # back through M_beta^{data}: |x><0| relabel of lambda
            lr = lam.reshape(n, 2 ** self.n_m, 2 ** self.n_w)
            mem = lr[:, :, 0]
            new = np.zeros_like(lr)
            new[np.arange(n), :, x] = mem
            lam = new.reshape(lam.shape)
            # recompute within-block forward states from checkpoint
            ops = block_ops[beta]
            states = [ckpt[beta]]
            for op in ops:
                states.append(self._apply_op(states[-1], op))
            # inject dL/dpsi^* from this block's score:
            #   sum_x (dL/du_x) Pi_x psi_pre
            psir = states[-1].reshape(n, 2 ** self.n_m, 2 ** self.n_w)
            inj = psir * dLdu_all[beta][:, None, :]
            lam = lam + inj.reshape(lam.shape)
            # back through the gates, accumulating parameter grads
            for j in range(len(ops) - 1, -1, -1):
                kind, q_, q2_, pidx = ops[j]
                if pidx is not None:
                    if kind == "rzz":
                        sp = apply_zz(states[j + 1], q_, q2_)
                    elif kind == "ryy":
                        sp = apply_yy(states[j + 1], q_, q2_)
                    elif kind == "rxx":
                        sp = apply_xx(states[j + 1], q_, q2_)
                    else:
                        sp = self._apply_pauli(states[j + 1], kind, q_)
                    z = np.einsum("nk,nk->n",
                                  lam.conj().reshape(n, -1),
                                  sp.reshape(n, -1))
                    dtheta[pidx] += float(z.imag.sum())
                if kind != "cz":
                    lam = self._apply_op(lam, ops[j], sign=-1.0)
                else:
                    lam = self._apply_op(lam, ops[j])
        return loss, dtheta, pb


    def blockwise_score(self, tokens: np.ndarray, K_blocks,
             cond: Optional[np.ndarray] = None,
                        chunk: int = 2048) -> float:
        """Forward-only blockwise conditional kernel score (no gradient;
        ~3x cheaper than blockwise_score_and_grad for monitoring)."""
        tokens = np.asarray(tokens, np.int64)
        N = tokens.shape[0]
        if N > chunk:
            return float(sum(
                self.blockwise_score(tokens[s:s + chunk], K_blocks)
                * (min(N, s + chunk) - s) / N
                for s in range(0, N, chunk)))
        n, B = tokens.shape
        Kb = [np.asarray(K_blocks[b], np.float64) for b in range(B)]
        psi = init_state(n, self.nq)
        hist = np.zeros((n, self.K))
        loss = 0.0
        for beta in range(B):
            a = self._angles(beta, hist / max(1, beta), n,
                             cond=cond)
            for op in self._block_ops(a, beta):
                psi = self._apply_op(psi, op)
            psir = psi.reshape(n, 2 ** self.n_m, 2 ** self.n_w)
            u = np.einsum("nmx,nmx->nx", psir, psir.conj()).real
            s = np.maximum(u.sum(axis=1), _TINY)
            q = u / s[:, None]
            x = tokens[:, beta]
            Kq = q @ Kb[beta]
            loss += float((np.einsum("nk,nk->n", q, Kq)
                           - 2.0 * Kq[np.arange(n), x]
                           + np.diag(Kb[beta])[x]).mean())
            mem = psir[np.arange(n), :, x]
            new = np.zeros_like(psir)
            new[:, :, 0] = mem
            psi = new.reshape(psi.shape)
            hist[np.arange(n), x] += 1.0
        return loss

    # ------------------------------------------------------------------
    # free-running generation
    # ------------------------------------------------------------------
    def sample(self, n: int, rng: np.random.Generator,
               cond: Optional[np.ndarray] = None) -> np.ndarray:
        """One physical rollout per sample: measure work register per
        block; the outcome IS the token. Returns (n, B) tokens."""
        psi = init_state(n, self.nq)
        hist = np.zeros((n, self.K))
        out = np.zeros((n, self.B), dtype=np.int64)
        for beta in range(self.B):
            a = self._angles(beta, hist / max(1, beta), n,
                             cond=cond)
            psi = self._apply_block(psi, a, self._theta_block(beta))
            probs = work_probs(psi, self.n_m, self.n_w)
            x = sample_outcomes(probs, rng)
            out[:, beta] = x
            psi, _ = project_and_reset(psi, x, self.n_m, self.n_w)
            hist[np.arange(n), x] += 1.0
        return out

    # ------------------------------------------------------------------
    # exact enumeration (validation only; K^B sequences)
    # ------------------------------------------------------------------
    def full_distribution(self, cond: Optional[np.ndarray] = None) -> np.ndarray:
        """Exact p over all K^B sequences (small B only). Used by the
        test suite to verify instrument completeness (sum = 1) and
        sampler consistency."""
        seqs = np.stack(np.meshgrid(
            *[np.arange(self.K)] * self.B, indexing="ij"),
            axis=-1).reshape(-1, self.B)
        lp = self.logp(seqs)
        return np.exp(lp), seqs
