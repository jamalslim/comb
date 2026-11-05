"""Exact path MMD: the law of the pen's positions at chosen times, through its characteristic function.

The pen position after tau moves is X_tau = sum_{t < tau} v(c_t), with v(c) the unit grid step of token c
(the pen moves whether it is up or down).  Stack the positions at observation times tau_1 < ... < tau_m into
X in R^{2m}.  Every position is a sum over all earlier moves, so the law of X carries correlations of every
order between the moves -- exactly what windows of length <= 3 and pairs cannot see (height, width,
closure, winding).

With a Gaussian kernel on X (mixture of bandwidths), Bochner gives MMD^2 = E_omega |Phi_P(omega) - Phi_Q(omega)|^2,
and for a CountChain model the characteristic function is exact:

    Phi_model(omega) = E exp(i sum_j omega_j . X_tau_j) = E exp(i sum_t Omega_t . v(c_t))
                     = tr[ A_{d-1} ... A_0 (rho_0) ],   A_t = sum_c exp(i Omega_t . v(c)) Lambda_{t,c},
    Omega_t = sum_{j : tau_j > t} omega_j,

the same tilted-map product as the Dataset 1 readout, with a lattice weight instead of a sinc.  Gradient by
reverse mode through the product, then pixel_instrument_vjp (package convention G = dL/dRe + i dL/dIm).
For independent tokens with the data's one-point laws the characteristic function factorises, which gives the
no-dependence reference used to normalise the term.
"""
import numpy as np

from .strokes import DIRS

STEP = DIRS.astype(float)                                      # (8, 2): (drow, dcol) of direction c % 8


def token_steps(K):
    return STEP[np.arange(K) % 8]                               # (K, 2)


def positions(T, taus):
    """Stacked pen positions (N, 2m) after tau moves, tau in taus (1..d)."""
    T = np.asarray(T, dtype=np.int64)
    X = np.cumsum(STEP[T % 8], axis=1)                          # position after move t+1 at index t
    return X[:, np.asarray(taus) - 1, :].reshape(len(T), -1)


class PathCF:
    def __init__(self, T_fit, d, K, every=4, rel_bandwidths=(0.5, 1.0, 2.0), seed=0):
        self.d, self.K = int(d), int(K)
        self.taus = np.arange(every, d + 1, every)
        X = positions(T_fit, self.taus)
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=min(len(X), 1000), replace=False)
        Y = X[idx]
        D2 = ((Y[:, None, :] - Y[None, :, :]) ** 2).sum(-1)
        med = float(np.sqrt(np.median(D2[np.triu_indices(len(Y), 1)]))) if len(Y) > 1 else 0.0
        self.sigma = med if med > 0 else 1.0                       # degenerate fit data: unit bandwidth
        self.bw = self.sigma * np.asarray(rel_bandwidths)
        self.V = token_steps(K)

    # ------------------------------------------------------------------ frequencies and weights
    def frequencies(self, R, rng):
        """omega (R, 2m) from the mixture of Gaussian spectral measures."""
        s = self.bw[rng.integers(0, len(self.bw), size=R)]
        return rng.normal(size=(R, 2 * len(self.taus))) / s[:, None]

    def weights(self, omega):
        """W[r, t, c] = exp(i Omega_t . v(c)),  Omega_t = sum_{j: tau_j > t} omega_j."""
        R, m = omega.shape[0], len(self.taus)
        om = omega.reshape(R, m, 2)
        Om = np.zeros((R, self.d, 2))
        for j, tau in enumerate(self.taus):
            Om[:, :tau, :] += om[:, j, None, :]
        return np.exp(1j * np.einsum("rtk,ck->rtc", Om, self.V))

    @staticmethod
    def empirical_cf(X, omega, chunk=4096):
        out = np.zeros(omega.shape[0], dtype=np.complex128)
        for i in range(0, len(X), chunk):
            out += np.exp(1j * X[i:i + chunk] @ omega.T).sum(0)
        return out / len(X)

    @staticmethod
    def null_cf(p1, W):
        """Independent tokens with one-point laws p1 (d, K): prod_t sum_c p1[t, c] W[:, t, c]."""
        return np.prod(np.einsum("tc,rtc->rt", p1, W), axis=1)

    # ------------------------------------------------------------------ exact model CF and gradient
    def _forward(self, model, W, keep):
        R, d = W.shape[0], model.d
        lam_all, itape = model.instruments_all(keep_tape=keep)
        v = np.tile(model._v0.astype(np.complex128), (R, 1))
        steps = []
        for t in range(d):
            U = np.matmul(lam_all[t], v.T)                                              # (K, D, R)
            v_new = (U * W[:, t, :].T[:, None, :]).sum(axis=0).T
            if keep:
                steps.append((v, U))
            v = v_new
        return v[:, model._tr_idx].sum(axis=1), ((lam_all, itape, steps) if keep else None)

    def _backward(self, model, W, tapes, grad_out):
        lam_all, itape, steps = tapes
        D = lam_all.shape[-1]
        G_v = np.zeros((W.shape[0], D), dtype=np.complex128)
        G_v[:, model._tr_idx] = grad_out[:, None]
        G_lam = np.zeros_like(lam_all)
        for t in range(model.d - 1, -1, -1):
            v_prev, U = steps[t]
            G_U = W[:, t, :].conj().T[:, None, :] * G_v.T[None, :, :]                  # (K, D, R)
            G_v = np.matmul(lam_all[t].conj().transpose(0, 2, 1), G_U).sum(axis=0).T
            G_lam[t] = np.matmul(G_U, v_prev.conj())                                    # (K, D, D)
        g_inst = model.instruments_all_vjp(itape, G_lam)
        return np.concatenate([g.ravel() for g in g_inst] + [np.zeros(model.h.size)])

    def model_cf(self, model, W, grad_out=None):
        phi, tapes = self._forward(model, W, keep=grad_out is not None)
        return (phi, None) if grad_out is None else (phi, self._backward(model, W, tapes, grad_out))

    # ------------------------------------------------------------------ loss
    def loss(self, model, X_data, omega, grad=True):
        """mean_r |Phi_data - Phi_model|^2 at the given frequencies, and its exact gradient."""
        W = self.weights(omega)
        ph = self.empirical_cf(X_data, omega)
        phi, tapes = self._forward(model, W, keep=grad)
        L = float(np.mean(np.abs(ph - phi) ** 2))
        if not grad:
            return L, None
        return L, self._backward(model, W, tapes, 2.0 * (phi - ph) / len(phi))

    def reference(self, X_data, p1, omega):
        """The same quantity for independent tokens with one-point laws p1 (the no-dependence reference)."""
        W = self.weights(omega)
        return float(np.mean(np.abs(self.empirical_cf(X_data, omega) - self.null_cf(p1, W)) ** 2))
