"""
Evaluation on the test showers (the 33 682 unseen showers outside the
validation set; never used for training or model selection), with the
training kernel and with physics metrics.

  mmd_exact     E_omega |Phi_Q - Phi_P|^2 with Phi_Q the model's exact
                characteristic function and the data term unbiased, over a
                fixed set of R_eval frequencies (seed 20 000; same for every
                model).  Available for every count-chain model and for the
                analytic references.
  mmd_samples   the unbiased two-sample statistic (Gretton et al. 2012), with
                the same kernel, between n model samples and n unseen showers
                (fixed subset).  Available for every model, including CoMB.
  sem           standard error of mmd_exact over frequencies (the data side is
                fixed); mmd_samples has its own null spread, reported by
                `sample_null`.
  physics       correlation-matrix error (mean |C_model - C_data| off the
                diagonal), mean per-pixel W1, W1 of the total energy, sigma_E
                ratio and energy shift (in units of the data sigma_E), all
                against the unseen showers, from n_phys model samples.
"""
from __future__ import annotations

import os

import numpy as np
from scipy.stats import wasserstein_distance

from .mmd import Spectrum, cf_sum_and_sq, mmd2_u, median_sigma


class Evaluator:
    def __init__(self, D, R_eval=16384, n_samp=10000, n_phys=20000, cache=None, seed=20000):
        self.D = D
        Y_tr = D["Y_tr"]
        self.d = Y_tr.shape[1]
        self.m0, self.s0 = Y_tr.mean(0), Y_tr.std(0)
        self.sig = median_sigma(self.std(Y_tr))
        self.spec = Spectrum(self.d, [self.sig / 2, self.sig, 2 * self.sig])
        self.om = self.spec.sample(R_eval, np.random.default_rng(seed))
        Zu = self.std(D["Y_test"])
        fp = np.array([len(Zu), Zu.sum(), (Zu ** 2).sum()])              # fingerprint of the test data
        z = np.load(cache) if (cache is not None and os.path.exists(cache)) else None
        if z is not None and z["om"].shape == self.om.shape and np.allclose(z["om"], self.om) \
                and "fp" in z.files and np.allclose(z["fp"], fp):
            self.phi_un, self.sq_un = z["phi"], z["sq"]
        else:
            self.phi_un, self.sq_un = cf_sum_and_sq(Zu, self.om)
            if cache is not None:
                tmp = f"{cache}.tmp{os.getpid()}.npz"
                np.savez(tmp, om=self.om, phi=self.phi_un, sq=self.sq_un, fp=fp)
                os.replace(tmp, cache)
        rng = np.random.default_rng(seed + 1)
        self.n_samp, self.n_phys = n_samp, n_phys
        self.ref_idx = rng.choice(len(Zu), size=n_samp, replace=False)
        self.ref = Zu[self.ref_idx]
        self.Y_un = D["Y_test"]
        self.C_un = np.corrcoef(self.Y_un.T)
        self.E_un = self.Y_un.sum(1)
        self.off = ~np.eye(self.d, dtype=bool)

    def std(self, Y):
        return (Y - self.m0) / self.s0

    # ------------------------------------------------------------------ MMD
    def mmd_exact_from_phi(self, phi_q):
        t = np.abs(phi_q) ** 2 - 2.0 * np.real(phi_q * self.phi_un.conj()) + self.sq_un
        return float(t.mean()), float(t.std() / np.sqrt(len(t)))

    def mmd_exact(self, model, chunk=2048):
        """Model side exact.  `model` must have characteristic_function in the
        standardized pixel coordinates of this evaluator (m0, s0)."""
        assert np.allclose(model.m, self.m0) and np.allclose(model.s, self.s0)
        phi = np.concatenate([model.characteristic_function(self.om[i:i + chunk])
                              for i in range(0, len(self.om), chunk)])
        return self.mmd_exact_from_phi(phi)

    def mmd_samples(self, Y):
        return mmd2_u(self.spec, self.std(Y[:self.n_samp]), self.ref)

    def sample_null(self, reps=5, seed=7):
        """Spread of mmd_samples when the 'model' is a set of test showers
        disjoint from the reference set (the null distribution of the estimator)."""
        rng = np.random.default_rng(seed)
        Zu = self.std(self.Y_un)
        pool = np.setdiff1d(np.arange(len(Zu)), self.ref_idx)
        out = []
        for _ in range(reps):
            idx = rng.choice(pool, size=self.n_samp, replace=False)
            out.append(mmd2_u(self.spec, Zu[idx], self.ref))
        return float(np.mean(out)), float(np.std(out))

    # ------------------------------------------------------------------ physics
    def physics(self, Y):
        C = np.corrcoef(Y.T)
        E = Y.sum(1)
        return dict(corr_err=float(np.abs(C - self.C_un)[self.off].mean()),
                    W1_pix=float(np.mean([wasserstein_distance(self.Y_un[:, j], Y[:, j]) for j in range(self.d)])),
                    W1_E=float(wasserstein_distance(self.E_un, E)),
                    sigmaE_ratio=float(E.std() / self.E_un.std()),
                    E_shift=float((E.mean() - self.E_un.mean()) / self.E_un.std()))

    def full(self, model=None, sampler=None, seed=12):
        """All metrics; `model` (count chain with exact CF) and/or `sampler`
        (callable n, rng -> samples in pixel units)."""
        out = {}
        if model is not None:
            out["mmd_exact"], out["mmd_exact_sem"] = self.mmd_exact(model)
            sampler = sampler or model.sample
        Y = sampler(max(self.n_samp, self.n_phys), np.random.default_rng(seed))
        out["mmd_samples"] = self.mmd_samples(Y)
        out.update(self.physics(Y[:self.n_phys]))
        return out
