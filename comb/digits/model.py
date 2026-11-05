"""
CoMB: the coherent memory Born machine.

A chi-dimensional coherent memory (n_m qubits) generates a d-dimensional
continuous sample one coordinate ("pixel") at a time.  For pixel beta:

  1. the memory evolves unitarily, rho -> W_beta rho W_beta^dag;
  2. n_sub collision rounds: n_w fresh work qubits in |0>, a joint unitary V_beta
     on memory + work qubits, the work qubits are measured in the computational
     basis and reset; the number of work qubits found in |1> is the number of
     "clicks" (quantum jumps) of the round;
  3. the pixel value is y_beta = g_beta(c_beta + u), where c_beta is the total
     number of clicks in the window (0 .. n_w n_sub), u ~ U(0, 1), and
     g_beta is a learned monotone piecewise-linear readout map.

One execution of the dynamic circuit (n_m + n_w qubits, d n_sub rounds of
mid-circuit measurement and reset) gives one continuous-valued sample.  The
memory is never measured; its conditional state is a quantum trajectory and
carries all the dependence between pixels.

Exact counting statistics.  With Kraus operators M_x = <x|_anc V |0>_anc and
superoperators L_s = sum_{x : |x| = s} M_x (x) conj(M_x) (row-major vec), the
count-resolved instrument of a pixel is obtained by the dynamic programme

    Lambda^(0)_0 = W (x) conj(W),
    Lambda^(r)_c = sum_s L_s Lambda^(r-1)_{c-s},   r = 1 .. n_sub,

i.e. Lambda_c is the coefficient of z^c in (sum_s z^s L_s)^{n_sub} (W (x) conj W)
-- the counting-field ("tilted") generator of full counting statistics.

Characteristic function.  For frequencies omega (standardized coordinates),

    Phi(omega) = E exp(i omega . y') = vec(I)^T  A_d ... A_1 vec(rho_0),
    A_beta = sum_c w_beta(c, omega_beta) Lambda_beta,c,
    w_beta(c, w) = int_0^1 exp(i w (g_beta(c+u) - m_beta)/s_beta) du
                 = exp(i w (mid_c - m)/s) sinc(w delta_c / (2 s)),

with mid_c, delta_c the midpoint and width of the readout bin of count c.  The
cost is O(d (n_sub K chi^6 + R K chi^4)) for R frequencies -- linear in d, no
enumeration of the K^d outcomes, no teacher forcing: Phi is the characteristic
function of the free-running generator.

Dephased twin.  dephase=True inserts complete dephasing of the memory (in the
computational basis) after W and after every round: the memory becomes a
classical chi-state Markov chain that modulates the click statistics (a
Markov-modulated counting process) with the same circuits and readout.

All gradients are exact (reverse mode, written out by hand) and checked
against finite differences in scripts/test_comb.py.
"""
from __future__ import annotations

import numpy as np

from .circuit import Circuit, BatchedCircuit, interleaved_edges, line_edges


def softplus(x):
    return np.logaddexp(0.0, x)


def softplus_inv(y):
    y = np.maximum(y, 1e-12)
    return y + np.log(-np.expm1(-y))


def sigmoid(x):
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def sinc(x):
    out = np.ones_like(x)
    nz = np.abs(x) > 1e-6
    out[nz] = np.sin(x[nz]) / x[nz]
    out[~nz] = 1.0 - x[~nz] ** 2 / 6.0
    return out


def dsinc(x):
    out = np.empty_like(x)
    nz = np.abs(x) > 1e-4
    out[nz] = (x[nz] * np.cos(x[nz]) - np.sin(x[nz])) / x[nz] ** 2
    out[~nz] = -x[~nz] / 3.0 + x[~nz] ** 3 / 30.0
    return out


def kron_grad(G, M):
    """Gradient wrt M of a real loss through S = M (x) conj(M), given G = dL/dS
    (complex convention G = dL/dRe S + i dL/dIm S)."""
    c = M.shape[0]
    G4 = G.reshape(c, c, c, c)                    # [i, k, j, l] for S[(i,k),(j,l)] = M[i,j] conj(M[k,l])
    return np.einsum("ikjl,kl->ij", G4, M) + np.einsum("ikjl,ij->kl", G4.conj(), M)


# ---------------------------------------------------------------------------
# Count-resolved composition of a window of n_sub identical rounds.
#   Lambda_c = [z^c] L(z)^{n_sub} S0,   L(z) = sum_s z^s L_s,   c = 0 .. K-1,
# with K - 1 = smax * n_sub the largest count.  Two exact engines:
#   dp   the convolution recursion Lambda^(r)_c = sum_s L_s Lambda^(r-1)_{c-s};
#   fft  evaluation of L(z)^{n_sub} S0 at the K-th roots of unity (binary
#        exponentiation) and the inverse DFT -- exact because the generating
#        polynomial has degree K - 1 (no aliasing); about 2x fewer matrix
#        products for the windows used here.
# Reverse mode for both; complex-gradient convention G = dL/dRe + i dL/dIm.

def counting_dp(L, S0, n_sub, smax, K):
    D = S0.shape[0]
    lam = np.zeros((K, D, D), dtype=np.complex128)
    lam[0] = S0
    hist = []
    for r in range(n_sub):
        hist.append(lam)
        new = np.zeros_like(lam)
        top = r * smax                                                  # max count so far
        for s in range(smax + 1):
            new[s:top + s + 1] += np.matmul(L[s], lam[:top + 1])
        lam = new
    return lam, hist


def counting_dp_vjp(L, hist, G_lam, n_sub, smax):
    G_L = np.zeros_like(L)
    G = G_lam
    for r in range(n_sub - 1, -1, -1):
        prev = hist[r]
        top = r * smax
        G_prev = np.zeros_like(prev)
        for s in range(smax + 1):
            Gs = G[s:top + s + 1]
            G_L[s] += np.tensordot(Gs, prev[:top + 1].conj(), axes=([0, 2], [0, 2]))
            G_prev[:top + 1] += np.matmul(L[s].conj().T, Gs)
        G = G_prev
    return G_L, G[0]


def _binpow(X, n):
    """X^n (batched) by binary exponentiation, with a tape for reverse mode."""
    mats, ops = [X], []
    res, base, e = None, 0, n
    while True:
        if e & 1:
            if res is None:
                res = base
            else:
                mats.append(np.matmul(mats[res], mats[base]))
                ops.append((len(mats) - 1, res, base))
                res = len(mats) - 1
        e >>= 1
        if not e:
            break
        mats.append(np.matmul(mats[base], mats[base]))
        ops.append((len(mats) - 1, base, base))
        base = len(mats) - 1
    return mats[res], (mats, ops, res)


def _binpow_vjp(tape, G_P):
    mats, ops, res = tape
    G = {res: G_P}
    for o, a, b in reversed(ops):
        Go = G.pop(o, None)
        if Go is None:
            continue
        G[a] = G.get(a, 0) + np.matmul(Go, mats[b].conj().transpose(0, 2, 1))
        G[b] = G.get(b, 0) + np.matmul(mats[a].conj().transpose(0, 2, 1), Go)
    return G.get(0, np.zeros_like(mats[0]))


def counting_fft(L, S0, n_sub, K):
    smax = L.shape[0] - 1
    z = np.exp(2j * np.pi * np.arange(K) / K)
    zp = z[:, None] ** np.arange(smax + 1)[None, :]                     # (K, smax+1)
    S, D = L.shape[0], L.shape[1]
    Lz = (zp @ L.reshape(S, -1)).reshape(K, D, D)
    P, ptape = _binpow(Lz, n_sub)
    T = np.matmul(P, S0)                                                # (K, D, D)
    F = z[None, :] ** (-np.arange(K)[:, None]) / K                      # F[c, k] = z_k^-c / K
    lam = (F @ T.reshape(K, -1)).reshape(K, D, D)
    return lam, dict(zp=zp, P=P, S0=S0, F=F, ptape=ptape)


def counting_fft_vjp(tape, G_lam):
    zp, P, S0, F = tape["zp"], tape["P"], tape["S0"], tape["F"]
    K, D = P.shape[0], P.shape[1]
    G_T = (F.conj().T @ G_lam.reshape(K, -1)).reshape(K, D, D)
    G_S0 = P.reshape(K * D, D).conj().T @ G_T.reshape(K * D, D)        # sum_k P_k^dag G_T_k
    G_P = np.matmul(G_T, S0.conj().T)
    G_Lz = _binpow_vjp(tape["ptape"], G_P)
    G_L = (zp.conj().T @ G_Lz.reshape(K, -1)).reshape(zp.shape[1], D, D)
    return G_L, G_S0


class CountChain:
    """A memory that emits, for every pixel beta, an integer count c_beta in
    0..K-1 through a count-resolved instrument {Lambda_beta,c} (linear maps on
    the memory's state vector, summing to a trace-preserving map), followed by
    the learned monotone readout y_beta = m + s * g_beta(c_beta + u_beta).

    Subclasses define the memory (state vector _v0, normalization functional
    _tr_idx), the instrument (pixel_instrument / pixel_instrument_vjp, with
    parameters inst_names) and sampling (sample_counts).  This class gives the
    exact characteristic function, its gradient, count marginals, the readout
    and its initialization, identically for the quantum model and the
    classical references.
    """

    # ------------------------------------------------------------------ parameters
    inst_names: tuple = ()

    def get(self):
        return np.concatenate([getattr(self, n).ravel() for n in self.inst_names] + [self.h.ravel()])

    def set(self, p):
        p = np.asarray(p, dtype=np.float64)
        i = 0
        for name in self.inst_names + ("h",):
            arr = getattr(self, name)
            n = arr.size
            setattr(self, name, p[i:i + n].reshape(arr.shape).copy())
            i += n

    @property
    def n_params(self):
        return self.get().size

    def n_inst_params(self):
        return sum(getattr(self, n).size for n in self.inst_names)

    n_circuit_params = n_inst_params

    def _init_readout_params(self):
        self.h = np.zeros((self.d, self.K))
        self.lo = np.full(self.d, -3.0)
        self.hi = np.full(self.d, 3.0)   # support of g (fixed)
        self.m = np.zeros(self.d)
        self.s = np.ones(self.d)                # standardization of y
        self.margin = 0.05

    # ------------------------------------------------------------------ readout map
    # The readout works in standardized units, y = m + s * z, and maps [0, K]
    # onto a fixed support [lo, hi] (the training range extended by 5% on
    # each side):  z = g(c + u) = left_c + delta_c u,
    #   delta = (hi - lo) softmax(h),   left_c = lo + sum_{j<c} delta_j.
    # The model can therefore never place mass outside the support observed in
    # training (an unbounded readout lets rare counts map to arbitrarily
    # distant values, which a bounded kernel hardly penalizes).
    def readout(self, beta):
        """Bin widths delta (K), left edges and midpoints of g_beta (standardized units)."""
        hb = self.h[beta]
        e = np.exp(hb - hb.max())
        delta = (self.hi[beta] - self.lo[beta]) * (e / e.sum())
        left = self.lo[beta] + np.concatenate([[0.0], np.cumsum(delta)[:-1]])
        return delta, left, left + 0.5 * delta

    def apply_readout(self, beta, c, u):
        delta, left, _ = self.readout(beta)
        return self.m[beta] + self.s[beta] * (left[c] + delta[c] * u)

    # ------------------------------------------------------------------ characteristic function
    def cf_weights(self, beta, om):
        """w (R, K) for frequencies om (R,) of pixel beta (standardized coordinates):
        w(c, om) = int_0^1 exp(i om g(c + u)) du = exp(i om mid_c) sinc(om delta_c / 2)."""
        delta, left, mid = self.readout(beta)
        sc = om[:, None]
        x = 0.5 * sc * delta[None, :]
        ph = np.exp(1j * sc * mid[None, :])
        return ph * sinc(x), ph, x, sc

    def cf_weights_vjp(self, beta, G_w, cache):
        """Gradient wrt h[beta] from G_w (R, K)."""
        ph, x, sc = cache
        w = ph * sinc(x)
        dmid = np.real(np.sum(G_w.conj() * (1j * sc * w), axis=0))         # dL/dmid_c
        ddel_direct = np.real(np.sum(G_w.conj() * (ph * dsinc(x) * 0.5 * sc), axis=0))
        # mid_c = lo + sum_{j<c} delta_j + delta_c / 2
        suffix = np.cumsum(dmid[::-1])[::-1]                                # sum_{c >= j} dmid_c
        dd = ddel_direct + 0.5 * dmid + (suffix - dmid)                     # dL/ddelta_j
        hb = self.h[beta]
        e = np.exp(hb - hb.max())
        sm = e / e.sum()
        return (self.hi[beta] - self.lo[beta]) * sm * (dd - np.dot(sm, dd))   # softmax Jacobian

    _frozen = None

    def freeze_instruments(self):
        """Cache every pixel's instrument (for readout-only optimization: the
        instruments do not depend on the readout parameters)."""
        self._frozen = [self.pixel_instrument(b)[0] for b in range(self.d)]

    def unfreeze_instruments(self):
        self._frozen = None

    def instruments_all(self, keep_tape=False):
        """Instruments of every round, stacked (d, K, D, D), and per-round tapes (default: one round at a time)."""
        out = [self.pixel_instrument(b, keep_tape=keep_tape) for b in range(self.d)]
        return np.stack([o[0] for o in out]), [o[1] for o in out]

    def instruments_all_vjp(self, tapes, G_lam):
        """Gradients wrt every instrument parameter array from G_lam (d, K, D, D)."""
        grads = [np.zeros_like(getattr(self, n)) for n in self.inst_names]
        for b in range(self.d):
            for arr, g in zip(grads, self.pixel_instrument_vjp(b, tapes[b], G_lam[b])):
                arr[b] = g
        return grads

    def characteristic_function(self, omega, grad_out=None, inst_grad=True):
        """Phi(omega) for omega (R, d) in standardized coordinates.  If grad_out
        (complex (R,), dL/dRe Phi + i dL/dIm Phi) is given, also return the
        gradient of L wrt all parameters (flat, same order as get()); with
        inst_grad=False (or frozen instruments) only the readout part of the
        gradient is computed and the instrument part is zero."""
        R = omega.shape[0]
        c2 = self._v0.size
        v = np.tile(self._v0.astype(np.complex128), (R, 1))                  # initial memory state
        need = grad_out is not None
        inst_grad = inst_grad and need and self._frozen is None
        tapes = []
        for beta in range(self.d):
            if self._frozen is not None:
                lam, tape = self._frozen[beta], None
            else:
                lam, tape = self.pixel_instrument(beta, keep_tape=inst_grad)
            w, ph, x, sc = self.cf_weights(beta, omega[:, beta])
            U = np.matmul(lam, v.T)                                             # (K, c2, R)
            v_new = (U * w.T[:, None, :]).sum(axis=0).T
            if need:
                tapes.append((lam, tape, w, (ph, x, sc), v, U))
            v = v_new
        phi = v[:, self._tr_idx].sum(axis=1)
        if not need:
            return phi
        g_inst = [np.zeros_like(getattr(self, n)) for n in self.inst_names]
        gh = np.zeros((self.d, self.K))
        G_v = np.zeros((R, c2), dtype=np.complex128)
        G_v[:, self._tr_idx] = grad_out[:, None]
        for beta in range(self.d - 1, -1, -1):
            lam, tape, w, cache, v_prev, U = tapes[beta]
            G_w = (U.conj() * G_v.T[None, :, :]).sum(axis=1).T
            G_U = w.conj().T[:, None, :] * G_v.T[None, :, :]                     # (K, c2, R)
            G_v_next = np.matmul(lam.conj().transpose(0, 2, 1), G_U).sum(axis=0).T   # (R, c2)
            if inst_grad:
                G_lam = np.matmul(G_U, v_prev.conj())                           # (K, c2, c2)
                for arr, g in zip(g_inst, self.pixel_instrument_vjp(beta, tape, G_lam)):
                    arr[beta] = g
            G_v = G_v_next
            gh[beta] = self.cf_weights_vjp(beta, G_w, cache)
        grad = np.concatenate([g.ravel() for g in g_inst] + [gh.ravel()])
        return phi, grad

    # ------------------------------------------------------------------ exact marginals / sampling
    def count_marginals(self):
        """Exact marginal distribution of the count of every pixel (d, K)."""
        rho = self._v0.astype(np.complex128)
        out = np.zeros((self.d, self.K))
        for beta in range(self.d):
            lam, _ = self.pixel_instrument(beta)
            parts = np.einsum("cij,j->ci", lam, rho)
            out[beta] = np.real(parts[:, self._tr_idx].sum(axis=1))
            rho = parts.sum(axis=0)
        return np.clip(out, 0, None)

    def sample(self, n, rng):
        cnt = self.sample_counts(n, rng)
        u = rng.random(cnt.shape)
        y = np.empty(cnt.shape)
        for beta in range(self.d):
            y[:, beta] = self.apply_readout(beta, cnt[:, beta], u[:, beta])
        return y

    # ------------------------------------------------------------------ initialization of the readout
    def init_readout(self, Y, m=None, s=None):
        """Set the standardization (m, s) (default: from Y) and match every
        pixel's readout to the data marginal given the model's current exact
        count marginal (quantile matching): bin c of pixel beta covers the
        data quantiles [F_c-1, F_c] of the count CDF.  The support [lo, hi]
        is the data range extended by `margin` of its length on each side."""
        self.m = Y.mean(axis=0) if m is None else np.asarray(m, float)
        self.s = Y.std(axis=0) + 1e-12 if s is None else np.asarray(s, float)
        Z = (Y - self.m) / self.s
        pc = self.count_marginals()
        for beta in range(self.d):
            zmin, zmax = Z[:, beta].min(), Z[:, beta].max()
            span = zmax - zmin
            if span < 1e-6: span = 1e-6            # constant coordinate (e.g. a PCA component of zero variance): give it a tiny support
            self.lo[beta], self.hi[beta] = zmin - self.margin * span, zmax + self.margin * span
            cdf = np.concatenate([[0.0], np.cumsum(pc[beta])])
            cdf = np.clip(cdf / cdf[-1], 0, 1)
            edges = np.quantile(Z[:, beta], cdf)
            edges[0], edges[-1] = self.lo[beta], self.hi[beta]
            widths = np.maximum(np.diff(edges), 1e-6 * (self.hi[beta] - self.lo[beta]))   # no zero widths: log(width) must be finite
            self.h[beta] = np.log(widths)



class CoMB(CountChain):
    """The coherent memory register (see the module docstring)."""
    inst_names = ("theta_v", "theta_w")

    def __init__(self, d: int, n_m: int = 1, n_w: int = 1, n_sub: int = 15,
                 L_v: int = 3, L_w: int = 2, seed: int = 0, dephase: bool = False,
                 entangler: str = "cz", init_scale: float = 0.6, current: str = "hamming",
                 engine: str = "fft", dephase_mode: str = "reduced"):
        self.d, self.n_m, self.n_w, self.n_sub = int(d), int(n_m), int(n_w), int(n_sub)
        self.engine = engine                    # "fft" (default) or "dp": same instrument, checked in tests
        self.chi, self.A = 2 ** self.n_m, 2 ** self.n_w
        # per-round increment phi(x) of the counter for work qubit outcome x:
        #   "hamming": number of work qubits found in |1>  (photon-counting current)
        #   "binary":  the outcome read as a binary number (weighted current;
        #              n_sub = 1 recovers a sequential Born machine with A levels)
        self.current = current
        if current == "hamming":
            self.phi = np.array([bin(x).count("1") for x in range(self.A)])
        elif current == "binary":
            self.phi = np.arange(self.A)
        elif current.startswith("binary_env:"):
            # the last n_e work qubits are environment: measured and reset every round but not emitted;
            # the token is the outcome of the first n_w - n_e qubits read as a binary number.  Each token
            # then has 2^n_e Kraus operators (a mixed instrument); with 2^n_e >= chi^2 the coherent
            # memory contains every classical chi-state HMM with the same emissions.
            n_e = int(current.split(":")[1])
            if not 0 < n_e < self.n_w:
                raise ValueError("need 0 < n_e < n_w")
            self.phi = np.arange(self.A) >> n_e
        else:
            raise ValueError(current)
        self.smax = int(self.phi.max())
        self.K = self.smax * self.n_sub + 1                # count levels per pixel
        self.dephase = bool(dephase)
        self.cv = Circuit(self.n_m + self.n_w, L_v, interleaved_edges(self.n_m, self.n_w), entangler)
        self.cw = Circuit(self.n_m, L_w, line_edges(self.n_m), entangler)
        rng = np.random.default_rng(seed)
        self.theta_v = rng.normal(0, init_scale, (self.d, self.cv.n_params))
        self.theta_w = rng.normal(0, init_scale, (self.d, self.cw.n_params))
        self._init_readout_params()
        c = self.chi
        self._diag_idx = np.arange(c) * (c + 1)                      # vec indices of diagonal entries
        self._E = np.zeros((self.cv.D, c), dtype=np.complex128)      # input columns |j_mem, 0_anc>
        self._E[np.arange(c) * self.A, np.arange(c)] = 1.0
        # dephased twin: by default the memory is represented by its diagonal
        # (a chi-state classical chain, chi x chi maps); dephase_mode="full"
        # keeps the dephased superoperators (identical model, used in tests)
        self.reduced = self.dephase and dephase_mode == "reduced"
        if self.reduced:
            self._v0 = np.zeros(c)
            self._v0[0] = 1.0
            self._tr_idx = np.arange(c)
        else:
            self._v0 = np.zeros(c * c)
            self._v0[0] = 1.0           # vec(|0><0|)
            self._tr_idx = self._diag_idx

    # ------------------------------------------------------------------ instruments
    def _dephase_super(self, S):
        """Complete dephasing of the memory before and after the superoperator S."""
        keep = np.zeros(S.shape[0], bool)
        keep[self._diag_idx] = True
        S = S.copy()
        S[~keep, :] = 0.0
        S[:, ~keep] = 0.0
        return S

    def _elementary(self, beta):
        """Per-round maps L_s (s = 0..smax) and the memory map S_W of pixel beta.
        Coherent: superoperators on vec(rho) (chi^2 x chi^2).  Dephased,
        reduced: transition matrices on the diagonal (chi x chi),
        L_s[i', i] = sum_{phi(x) = s} |<i'|M_x|i>|^2, S_W[i', i] = |W_i'i|^2.
        Dephased, full: the dephased superoperators (reference for tests)."""
        Vb, tape_v = self.cv.forward(self.theta_v[beta], self._E)          # (D, chi)
        Ms = [Vb[x::self.A, :] for x in range(self.A)]                      # Kraus (chi, chi)
        W, tape_w = self.cw.forward(self.theta_w[beta], np.eye(self.chi))
        if self.reduced:
            L = np.zeros((self.smax + 1, self.chi, self.chi), dtype=np.complex128)
            for x, M in enumerate(Ms):
                L[self.phi[x]] += np.abs(M) ** 2
            S0 = (np.abs(W) ** 2).astype(np.complex128)
        else:
            c2 = self.chi ** 2
            L = np.zeros((self.smax + 1, c2, c2), dtype=np.complex128)
            for x, M in enumerate(Ms):
                L[self.phi[x]] += np.kron(M, M.conj())
            S0 = np.kron(W, W.conj())
            if self.dephase:
                L = np.stack([self._dephase_super(Ls) for Ls in L])
                S0 = self._dephase_super(S0)
        return L, S0, dict(Vb=Vb, tape_v=tape_v, Ms=Ms, W=W, tape_w=tape_w)

    # ------------------------------------------------------------------ Kraus form of the exact characteristic function
    # Phi(omega) = Tr[ A_d ... A_1 (rho_0) ],  A_beta(rho) = sum_x w_beta(phi(x), omega_beta) M_x W rho W^dag M_x^dag.
    # Same closed form as the superoperator path, evaluated by applying the Kraus operators to the chi x chi memory:
    # cost (2A + 2) chi^3 per frequency and round instead of A chi^4, and no chi^2 x chi^2 matrices are ever formed.
    # Used for one round per value (n_sub = 1) and a coherent memory; the superoperator path remains the reference.
    def _kraus_round(self, beta):
        Vb, tape_v = self.cv.forward(self.theta_v[beta], self._E)                     # (chi*A, chi)
        Ms = Vb.reshape(self.chi, self.A, self.chi).transpose(1, 0, 2)                 # Ms[x] = (I x <x|) V (I x |0>)
        W, tape_w = self.cw.forward(self.theta_w[beta], np.eye(self.chi))
        return Ms, W, tape_v, tape_w

    def _cf_kraus(self, omega, grad_out=None, inst_grad=True, weight_fn=None):
        # weight_fn(beta) -> (R, K) complex weights replaces the bin readout (used by the patch readout); omega then
        # only fixes R, and the readout parameters get no gradient.
        R, c = omega.shape[0], self.chi
        rho = np.zeros((R, c, c), dtype=np.complex128)
        rho[:, 0, 0] = 1.0              # |0><0|
        need = grad_out is not None
        inst_grad = inst_grad and need
        tapes = []
        for beta in range(self.d):
            Ms, W, tv, tw = self._kraus_round(beta)
            MsH = Ms.conj().transpose(0, 2, 1)
            if weight_fn is None: w, ph, x, sc = self.cf_weights(beta, omega[:, beta])   # (R, K) readout weights
            else: w, ph, x, sc = weight_fn(beta), None, None, None
            wx = w[:, self.phi]                                                          # weight of each outcome x
            rt = W @ rho @ W.conj().T
            if self.dephase: rt = rt * np.eye(c)[None]                                  # dephased twin: also after W
            Y = Ms[None] @ rt[:, None] @ MsH[None]                                       # (R, A, c, c)
            new = np.einsum("ra,raij->rij", wx, Y)
            if self.dephase: new = new * np.eye(c)[None]                                # dephased twin: keep populations only
            if need:
                tapes.append((Ms, W, tv, tw, wx, (ph, x, sc), rho, rt))
            rho = new
        phi = np.trace(rho, axis1=1, axis2=2)
        if not need:
            return phi
        g_v, g_w, gh = np.zeros_like(self.theta_v), np.zeros_like(self.theta_w), np.zeros((self.d, self.K))
        onehot = np.zeros((self.A, self.K))
        onehot[np.arange(self.A), self.phi] = 1.0
        G = np.zeros((R, c, c), dtype=np.complex128)
        G[:, np.arange(c), np.arange(c)] = grad_out[:, None]
        for beta in range(self.d - 1, -1, -1):                                          # reverse mode, one pass
            Ms, W, tv, tw, wx, cache, rho_prev, rt = tapes[beta]
            if self.dephase: G = G * np.eye(c)[None]                                    # adjoint of the diagonal projection
            MsH = Ms.conj().transpose(0, 2, 1)
            Y = Ms[None] @ rt[:, None] @ MsH[None]
            if weight_fn is None: gh[beta] = self.cf_weights_vjp(beta, np.einsum("raij,rij->ra", Y.conj(), G) @ onehot, cache)
            G_Y = wx.conj()[:, :, None, None] * G[:, None]                              # (R, A, c, c)
            G_rt = (MsH[None] @ G_Y @ Ms[None]).sum(axis=1)
            if self.dephase: G_rt = G_rt * np.eye(c)[None]
            if inst_grad:
                G_M = (G_Y @ (Ms[None] @ rt.conj().transpose(0, 2, 1)[:, None])
                       + G_Y.conj().transpose(0, 1, 3, 2) @ (Ms[None] @ rt[:, None])).sum(axis=0)          # (A, c, c)
                G_W = (G_rt @ (W[None] @ rho_prev.conj().transpose(0, 2, 1))
                       + G_rt.conj().transpose(0, 2, 1) @ (W[None] @ rho_prev)).sum(axis=0)
                g_v[beta] = self.cv.vjp(tv, G_M.transpose(1, 0, 2).reshape(c * self.A, c))
                g_w[beta] = self.cw.vjp(tw, G_W)
            G = W.conj().T[None] @ G_rt @ W[None]
        return phi, np.concatenate([g_v.ravel(), g_w.ravel(), gh.ravel()])

    cf_path = "auto"      # "auto": Kraus form from chi = 8 up (faster, chi^3), superoperators below; or "kraus" / "superoperator"

    def characteristic_function(self, omega, grad_out=None, inst_grad=True):
        kraus_ok = self.n_sub == 1 and not self.dephase and self._frozen is None
        use_kraus = kraus_ok and (self.cf_path == "kraus" or (self.cf_path == "auto" and self.chi >= 8))
        if not use_kraus:
            return super().characteristic_function(omega, grad_out=grad_out, inst_grad=inst_grad)
        if grad_out is not None:
            return self._cf_kraus(omega, grad_out=grad_out, inst_grad=inst_grad)
        step = max(1, int(2 ** 22 // (self.A * self.chi ** 2)))                         # bounded memory for large R
        return np.concatenate([self._cf_kraus(omega[i:i + step]) for i in range(0, omega.shape[0], step)])

    def pixel_instrument(self, beta, keep_tape=False):
        """Count-resolved instrument Lambda_c (K, D, D) of pixel beta
        (D = chi^2, or chi for the reduced dephased model)."""
        L, S0, ctape = self._elementary(beta)
        if self.engine == "fft":
            lam, ktape = counting_fft(L, S0, self.n_sub, self.K)
        else:
            lam, ktape = counting_dp(L, S0, self.n_sub, self.smax, self.K)
        tape = dict(ctape, L=L, S0=S0, ktape=ktape) if keep_tape else None
        return lam, tape

    def pixel_instrument_vjp(self, beta, tape, G_lam):
        """Gradients wrt theta_v[beta], theta_w[beta] from G_lam (K, D, D)."""
        L, S0 = tape["L"], tape["S0"]
        if self.engine == "fft":
            G_L, G_S0 = counting_fft_vjp(tape["ktape"], G_lam)
        else:
            G_L, G_S0 = counting_dp_vjp(L, tape["ktape"], G_lam, self.n_sub, self.smax)
        G_V = np.zeros_like(tape["Vb"])
        if self.reduced:
            for x, M in enumerate(tape["Ms"]):
                G_V[x::self.A, :] += 2.0 * np.real(G_L[self.phi[x]]) * M
            G_W = 2.0 * np.real(G_S0) * tape["W"]
        else:
            if self.dephase:
                G_L = np.stack([self._dephase_super(g) for g in G_L])
                G_S0 = self._dephase_super(G_S0)
            for x, M in enumerate(tape["Ms"]):
                G_V[x::self.A, :] += kron_grad(G_L[self.phi[x]], M)
            G_W = kron_grad(G_S0, tape["W"])
        g_v = self.cv.vjp(tape["tape_v"], G_V)
        g_w = self.cw.vjp(tape["tape_w"], G_W)
        return g_v, g_w

    def _batched_ok(self):
        return self.n_sub == 1 and (not self.dephase or self.reduced)

    def instruments_all(self, keep_tape=False):
        """All rounds at once (n_sub = 1): Lambda_c = L_c S0 exactly, the circuits applied to every round's angles in
        one pass.  Falls back to the per-round path otherwise."""
        if not self._batched_ok():
            return CountChain.instruments_all(self, keep_tape)
        if not hasattr(self, "_bcv"):
            self._bcv, self._bcw = BatchedCircuit(self.cv), BatchedCircuit(self.cw)
        d, c, A = self.d, self.chi, self.A
        V, tv = self._bcv.forward(self.theta_v, self._E)                                 # (d, chi*A, chi)
        W, tw = self._bcw.forward(self.theta_w, np.eye(c))                               # (d, chi, chi)
        Ms = V.reshape(d, c, A, c).transpose(0, 2, 1, 3)                                  # Ms[:, x] = V[:, x::A, :]
        onehot = np.zeros((A, self.smax + 1))
        onehot[np.arange(A), self.phi] = 1.0
        sum_x = lambda X: np.matmul(onehot.T, X.reshape(d, A, -1)).reshape((d, self.smax + 1) + X.shape[2:])
        if self.reduced:
            L = sum_x((np.abs(Ms) ** 2).astype(np.complex128))
            S0 = (np.abs(W) ** 2).astype(np.complex128)
        else:                                           # kron(M, conj M)[(i,k), (j,l)] = M[i,j] conj(M[k,l])
            KK = (Ms[:, :, :, None, :, None] * Ms.conj()[:, :, None, :, None, :]).reshape(d, A, c * c, c * c)
            L = sum_x(KK)
            S0 = (W[:, :, None, :, None] * W.conj()[:, None, :, None, :]).reshape(d, c * c, c * c)
        lam = np.matmul(L, S0[:, None])
        return lam, (dict(Ms=Ms, W=W, L=L, S0=S0, tv=tv, tw=tw) if keep_tape else None)

    def instruments_all_vjp(self, tape, G_lam):
        if not self._batched_ok():
            return CountChain.instruments_all_vjp(self, tape, G_lam)
        d, c, A = self.d, self.chi, self.A
        Ms, W, L, S0 = tape["Ms"], tape["W"], tape["L"], tape["S0"]
        G_L = np.matmul(G_lam, np.conj(np.swapaxes(S0, 1, 2))[:, None])                 # G_lam S0^dag
        G_S0 = np.matmul(np.conj(np.swapaxes(L, 2, 3)), G_lam).sum(axis=1)                # sum_s L_s^dag G_lam_s
        G_Lx = G_L[:, self.phi]                                                          # (d, A, D, D)
        if self.reduced:
            G_M = 2.0 * np.real(G_Lx) * Ms
            G_W = 2.0 * np.real(G_S0) * W
        else:
            G6 = G_Lx.reshape(d * A, c, c, c, c)                                          # [., i, k, j, l]
            Mf = Ms.reshape(d * A, c * c, 1)
            t1 = np.matmul(G6.transpose(0, 1, 3, 2, 4).reshape(d * A, c * c, c * c), Mf)  # sum_kl G[i,k,j,l] M[k,l]
            t2 = np.matmul(G6.conj().transpose(0, 2, 4, 1, 3).reshape(d * A, c * c, c * c), Mf)  # sum_ij conj G M[i,j]
            G_M = (t1 + t2).reshape(d, A, c, c)
            G4s = G_S0.reshape(d, c, c, c, c)
            Wf = W.reshape(d, c * c, 1)
            G_W = (np.matmul(G4s.transpose(0, 1, 3, 2, 4).reshape(d, c * c, c * c), Wf)
                   + np.matmul(G4s.conj().transpose(0, 2, 4, 1, 3).reshape(d, c * c, c * c), Wf)).reshape(d, c, c)
        G_V = G_M.transpose(0, 2, 1, 3).reshape(d, c * A, c)
        return [self._bcv.vjp(tape["tv"], G_V), self._bcw.vjp(tape["tw"], G_W)]

    def sample_counts(self, n, rng):
        """Simulate n executions of the dynamic circuit; returns counts (n, d)."""
        c = self.chi
        counts = np.zeros((n, self.d), dtype=np.int64)
        if self.dephase:
            state = np.zeros(n, dtype=np.int64)                                 # classical memory state
        else:
            psi = np.zeros((n, c), dtype=np.complex128)
            psi[:, 0] = 1.0
        for beta in range(self.d):
            Vb, _ = self.cv.forward(self.theta_v[beta], self._E)
            Ms = np.stack([Vb[x::self.A, :] for x in range(self.A)])           # (A, chi, chi)
            W = self.cw.unitary(self.theta_w[beta])
            if self.dephase:
                Pw = np.abs(W) ** 2                                             # Pw[i', i]
                state = _sample_columns(Pw, state, rng)
                P = np.abs(Ms) ** 2                                             # P[x, i', i]
                Pflat = P.transpose(1, 0, 2).reshape(c * self.A, c)            # joint (i', x) given i
                for _ in range(self.n_sub):
                    j = _sample_columns(Pflat, state, rng)
                    state, x = j // self.A, j % self.A
                    counts[:, beta] += self.phi[x]
            else:
                psi = psi @ W.T
                for _ in range(self.n_sub):
                    amp = np.einsum("xij,nj->nxi", Ms, psi)                     # (n, A, chi)
                    p = np.real(np.einsum("nxi,nxi->nx", amp, amp.conj()))
                    p /= p.sum(axis=1, keepdims=True)
                    x = (rng.random(n)[:, None] > np.cumsum(p, axis=1)).sum(axis=1)
                    x = np.minimum(x, self.A - 1)
                    psi = amp[np.arange(n), x]
                    psi /= np.linalg.norm(psi, axis=1, keepdims=True)
                    counts[:, beta] += self.phi[x]
        return counts



class ClassicalHMM(CountChain):
    """Reference: a general chi-state hidden Markov model with K-level count
    emissions and the same readout and training.  Per pixel, the hidden state
    moves with a column-stochastic T_beta = softmax(tl_beta, axis=0) and emits
    c with probability E_beta[c, i'] = softmax(el_beta, axis=0)[c, i'] given the
    new state i'.  Lambda_c = diag(E_beta[c]) T_beta acts on probability vectors.
    Every chi-state model with K-level emissions is of this form, so this is
    the most expressive classical memory of size chi (the dephased CoMB is a
    special case with circuit-parameterized T and E)."""
    inst_names = ("tl", "el")

    def __init__(self, d: int, chi: int, K: int, seed: int = 0, init_scale: float = 1.0):
        self.d, self.chi, self.K = int(d), int(chi), int(K)
        rng = np.random.default_rng(seed)
        self.tl = rng.normal(0, init_scale, (self.d, self.chi, self.chi))
        self.el = rng.normal(0, init_scale, (self.d, self.K, self.chi))
        self._init_readout_params()
        self._v0 = np.zeros(self.chi)
        self._v0[0] = 1.0
        self._tr_idx = np.arange(self.chi)

    @staticmethod
    def _softmax0(x):
        e = np.exp(x - x.max(axis=0, keepdims=True))
        return e / e.sum(axis=0, keepdims=True)

    def pixel_instrument(self, beta, keep_tape=False):
        T = self._softmax0(self.tl[beta])
        E = self._softmax0(self.el[beta])
        lam = (E[:, :, None] * T[None, :, :]).astype(np.complex128)       # (K, chi, chi)
        return lam, (dict(T=T, E=E) if keep_tape else None)

    def pixel_instrument_vjp(self, beta, tape, G_lam):
        T, E = tape["T"], tape["E"]
        G = np.real(G_lam)
        G_T = np.einsum("cij,ci->ij", G, E)
        G_E = np.einsum("cij,ij->ci", G, T)
        g_tl = T * (G_T - (T * G_T).sum(axis=0, keepdims=True))
        g_el = E * (G_E - (E * G_E).sum(axis=0, keepdims=True))
        return g_tl, g_el

    def sample_counts(self, n, rng):
        state = np.zeros(n, dtype=np.int64)
        counts = np.zeros((n, self.d), dtype=np.int64)
        for beta in range(self.d):
            T = self._softmax0(self.tl[beta])
            E = self._softmax0(self.el[beta])
            state = _sample_columns(T, state, rng)
            counts[:, beta] = _sample_columns(E, state, rng)
        return counts


def _sample_columns(P, state, rng):
    """Sample a new index for every column-state: P[new, old] (columns sum to 1)."""
    cum = np.cumsum(P[:, state], axis=0)                                        # (n_new, n)
    cum /= cum[-1:, :]
    r = rng.random(state.shape[0])
    return np.minimum((r[None, :] > cum).sum(axis=0), P.shape[0] - 1)


class LinearDecoded:
    """A count-chain generator of latent coordinates followed by a fixed
    affine decoder:  x = A z + b,  y = m + s * x, where z is the base model's
    standardized output and x are the standardized pixels in which the kernel
    lives.  With A, b from principal components of the training data, the
    base model generates principal-component scores in order of decreasing
    variance.

    Phi_x(omega) = exp(i omega . b) Phi_z(A^T omega): the exact characteristic
    function and its gradient pass through unchanged (frequencies are mapped,
    the phase is a constant), so training is still the exact CF-MMD in pixel
    space.  The decoder has no trainable parameters."""

    def __init__(self, base, basis="pca", n_out=None):
        # base.d latent coordinates (the first base.d principal components) decoded into n_out pixels;
        # n_out = None keeps the square decoder (as many coordinates as pixels)
        self.base, self.basis = base, basis
        self.k = base.d
        self.d = base.d if n_out is None else int(n_out)
        self.m = np.zeros(self.d)
        self.s = np.ones(self.d)
        self.A = np.eye(self.d, self.k)
        self.b = np.zeros(self.d)
        self.P = np.eye(self.d, self.k)
        self.mu = np.zeros(self.d)

    # parameters are the base model's
    def get(self):
        return self.base.get()

    def set(self, p):
        self.base.set(p)

    @property
    def n_params(self):
        return self.base.n_params

    def n_inst_params(self):
        return self.base.n_inst_params()

    n_circuit_params = n_inst_params

    @property
    def K(self):
        return self.base.K

    def init_readout(self, Y, m=None, s=None):
        self.m = Y.mean(axis=0) if m is None else np.asarray(m, float)
        self.s = Y.std(axis=0) + 1e-12 if s is None else np.asarray(s, float)
        X = (Y - self.m) / self.s
        self.mu = X.mean(axis=0)
        if self.basis == "pca":
            ev, P = np.linalg.eigh(np.cov(X.T))
            P = P[:, ::-1] * np.sign(P[:, ::-1].sum(axis=0, keepdims=True) + 1e-300)
            self.P = P[:, :self.k]                     # keep the k directions of largest variance
        U = (X - self.mu) @ self.P
        self.base.init_readout(U)
        self._update_affine()

    def _update_affine(self):
        self.A = self.P * self.base.s[None, :]
        self.b = self.mu + self.P @ self.base.m

    def set_decoder(self, m, s, mu, P, base_m, base_s):
        self.m, self.s, self.mu, self.P = np.asarray(m), np.asarray(s), np.asarray(mu), np.asarray(P)
        self.base.m, self.base.s = np.asarray(base_m), np.asarray(base_s)
        self._update_affine()

    def characteristic_function(self, omega, grad_out=None, **kw):
        ph = np.exp(1j * omega @ self.b)
        om_z = omega @ self.A
        if grad_out is None:
            return ph * self.base.characteristic_function(om_z)
        phi, g = self.base.characteristic_function(om_z, grad_out=np.conj(ph) * grad_out, **kw)
        return ph * phi, g

    def freeze_instruments(self):
        self.base.freeze_instruments()

    def unfreeze_instruments(self):
        self.base.unfreeze_instruments()

    def sample(self, n, rng):
        Z = (self.base.sample(n, rng) - self.base.m) / self.base.s
        return self.m + self.s * (Z @ self.A.T + self.b)
