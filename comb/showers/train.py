"""
Training by exact characteristic-function MMD.

Per step: R frequencies omega_r ~ Lambda (orthogonal random features of the
Gaussian-mixture kernel), the exact model characteristic function Phi_Q and
its gradient (one forward + one reverse sweep through the instrument chain),
the empirical characteristic function of the training data Phi_P, and

    loss = (1/R) sum_r |Phi_Q(omega_r) - Phi_P(omega_r)|^2,
    dloss/dPhi_Q (complex convention) = 2 (Phi_Q - Phi_P) / R.

The gradient is an unbiased estimate of the gradient of MMD^2_k(P_N, Q) (P_N:
the training sample); no model samples, no likelihood.  Model selection uses
the validation MMD^2 with the model side exact and a fixed frequency set.

Adam with separate learning rates for circuit angles and readout parameters,
linear warm-up and cosine decay.  The whole optimizer state (parameters, Adam
moments, RNG state, best-so-far, history) is checkpointed atomically and a
resumed run is bit-identical to an uninterrupted one (scripts/test_train.py).
"""
from __future__ import annotations

import os
import pickle
import time

import numpy as np

from .mmd import Spectrum, empirical_cf, cf_sum_and_sq


class Adam:
    def __init__(self, n, b1=0.9, b2=0.999, eps=1e-8):
        self.m = np.zeros(n)
        self.v = np.zeros(n)
        self.t = 0
        self.b1, self.b2, self.eps = b1, b2, eps

    def step(self, p, g, lr):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return p - lr * mh / (np.sqrt(vh) + self.eps)


def lr_at(step, steps, base, warmup, floor=0.05):
    if step < warmup:
        return base * (step + 1) / warmup
    x = (step - warmup) / max(1, steps - warmup)
    return base * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * x)))


def _atomic_pickle(obj, path):
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def frequency_pool(spec: Spectrum, Z_fit, P=32768, seed=12345, cache=None):
    """A fixed pool of P frequencies (orthogonal random features) and the
    empirical characteristic function of the training data on it.  Training
    draws R frequencies from the pool at every step, i.e. it minimizes the MMD
    of the random-Fourier-feature kernel (1/P) sum_j cos(omega_j.(z - z')),
    which approximates the Gaussian-mixture kernel to O(P^-1/2).  The pool is
    shared by every model trained on the same data (same seed)."""
    if cache is not None and os.path.exists(cache):
        z = np.load(cache)
        if z["om"].shape == (P, spec.d) and np.allclose(z["sigmas"], spec.sigmas):
            return z["om"], z["phi"]
    om = spec.sample(P, np.random.default_rng(seed))
    phi = empirical_cf(Z_fit, om)
    if cache is not None:
        tmp = f"{cache}.tmp{os.getpid()}.npz"
        np.savez(tmp, om=om, phi=phi, sigmas=spec.sigmas)
        os.replace(tmp, cache)
    return om, phi


def train(model, Y_fit, Y_val, spec: Spectrum, steps=3000, R=256, lr=0.03, lr_readout=0.01,
          warmup=50, eval_every=25, R_val=4096, seed=0, ckpt=None, ckpt_every=100,
          log=None, init_readout=True, m=None, s=None, pool=None, _interrupt_at=None):
    """Train `model` (any CountChain) on Y_fit; select on Y_val.  `pool` = (om,
    phi_data) from frequency_pool (default: fresh i.i.d. ORF frequencies every
    step and the data CF recomputed).  Returns a dict with the best parameters
    and the history."""
    if init_readout:
        model.init_readout(Y_fit, m=m, s=s)
    Z_fit = (Y_fit - model.m) / model.s
    Z_val = (Y_val - model.m) / model.s
    om_val = spec.sample(R_val, np.random.default_rng(10_000 + seed))
    phi_val, sq_val = cf_sum_and_sq(Z_val, om_val)

    def val_mmd():
        q = model.characteristic_function(om_val)
        return float(np.mean(np.abs(q) ** 2 - 2.0 * np.real(q * phi_val.conj()) + sq_val))

    nc = model.n_circuit_params()
    p = model.get()
    lrs = np.where(np.arange(p.size) < nc, 1.0, lr_readout / lr)
    state = dict(step=0, p=p, adam=Adam(p.size), rng=np.random.default_rng(seed).bit_generator.state,
                 best=(np.inf, -1, p.copy()), hist=dict(step=[], loss=[], val=[], time=[]), m=model.m, s=model.s,
                 elapsed=0.0)
    if ckpt is not None and os.path.exists(ckpt):
        with open(ckpt, "rb") as f:
            state = pickle.load(f)
        model.m, model.s = state["m"], state["s"]
        if log:
            log(f"[resume] step {state['step']}")
    rng = np.random.default_rng()
    rng.bit_generator.state = state["rng"]
    p, adam = state["p"], state["adam"]
    t0 = time.time() - state["elapsed"]
    run_loss = state.get("run_loss", [])
    for step in range(state["step"], steps):
        if _interrupt_at is not None and step == _interrupt_at:
            raise KeyboardInterrupt(f"simulated interruption at step {step}")      # tests only
        model.set(p)
        if step % eval_every == 0:
            val = val_mmd()
            state["hist"]["step"].append(step)
            state["hist"]["val"].append(val)
            state["hist"]["loss"].append(float(np.mean(run_loss)) if run_loss else np.nan)
            state["hist"]["time"].append(time.time() - t0)
            run_loss = []
            if val < state["best"][0]:
                state["best"] = (val, step, p.copy())
            if log:
                log(f"step {step:5d}  loss {state['hist']['loss'][-1]:.3e}  val {val:.4e}  best {state['best'][0]:.4e}@{state['best'][1]}  {time.time() - t0:.0f}s")
        if pool is None:
            om = spec.sample(R, rng)
            php = empirical_cf(Z_fit, om)
        else:
            idx = rng.choice(len(pool[0]), size=R, replace=False)
            om, php = pool[0][idx], pool[1][idx]
        phq = model.characteristic_function(om)
        diff = phq - php
        run_loss.append(float(np.mean(np.abs(diff) ** 2)))
        _, g = model.characteristic_function(om, grad_out=2.0 * diff / R)
        p = adam.step(p, g, lr_at(step, steps, lr, warmup) * lrs)
        state["step"] = step + 1
        if ckpt is not None and (step + 1) % ckpt_every == 0:
            state.update(p=p, adam=adam, rng=rng.bit_generator.state, elapsed=time.time() - t0, run_loss=list(run_loss))
            _atomic_pickle(state, ckpt)
    # final evaluation
    model.set(p)
    val = val_mmd()
    state["hist"]["step"].append(steps)
    state["hist"]["val"].append(val)
    state["hist"]["loss"].append(float(np.mean(run_loss)) if run_loss else np.nan)
    state["hist"]["time"].append(time.time() - t0)
    if val < state["best"][0]:
        state["best"] = (val, steps, p.copy())
    state.update(p=p, adam=adam, rng=rng.bit_generator.state, elapsed=time.time() - t0)
    if ckpt is not None:
        _atomic_pickle(state, ckpt)
    model.set(state["best"][2])
    return dict(best_val=float(state["best"][0]), best_step=int(state["best"][1]), params=state["best"][2],
                final_params=p, hist=state["hist"], m=model.m, s=model.s, time=time.time() - t0)
