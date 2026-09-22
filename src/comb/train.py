"""
Training for CoMB: the likelihood-free blockwise conditional kernel
score (train_blockscore), plus the Adam optimizer and learning-rate
schedule it uses.

Objective (train_blockscore): teacher-forced, strictly proper blockwise
MMD; no likelihood is computed anywhere. Optimizer: Adam with linear
warmup + cosine decay, gradient clipping, best-validation-theta
selection, optional early stopping (patience).

The loss is a true likelihood, so train/val NLL are directly comparable
with every classical baseline in baselines.py, in nats per image.
"""

from __future__ import annotations

import time

import numpy as np


class AdamOpt:
    # Plain Adam. Nothing exotic, but the schedule matters: the blockwise
    # score is very stiff in the first few epochs, so a short linear
    # warmup avoids throwing the parameters somewhere useless before the
    # gradient has settled.
    def __init__(self, n_params, lr=0.05, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m = np.zeros(n_params)
        self.v = np.zeros(n_params)
        self.t = 0

    def step(self, g):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * g
        self.v = self.b2 * self.v + (1 - self.b2) * g * g
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return self.lr * mh / (np.sqrt(vh) + self.eps)


def _sched(step, total, base, warmup, floor=0.1):
    # Linear warmup then cosine decay to `floor * base`. The floor is not
    # zero on purpose: the last epochs still need to move, and a rate that
    # decays to nothing just freezes whatever the model happened to be
    # doing at epoch ~0.9 * total.
    if step < warmup:
        return base * (step + 1) / warmup
    prog = min(1.0, (step - warmup) / max(1, total - warmup))
    return base * (floor + (1 - floor) * 0.5 * (1 + np.cos(np.pi * prog)))


def train_blockscore(model, K_blocks, tokens_tr, tokens_val,
                     epochs=100, batch=1024, lr=0.08, warmup=5,
                     clip=10.0, seed=0, verbose=True, patience=None,
                     cond_tr=None, cond_val=None, schedule="cosine"):
    """
    NLL-free training: minimize the teacher-forced blockwise
    conditional kernel score (strictly proper for PD Grams; see
    SequentialBornModel.blockwise_score_and_grad). Model selection by
    the validation score. No likelihood is computed anywhere.
    """
    import time as _time
    rng = np.random.default_rng(seed)
    n = tokens_tr.shape[0]
    opt = AdamOpt(model.p_theta, lr=lr)
    hist = dict(train_score=[], val_score=[], grad_norm=[], lr=[],
                per_block=[])
    best = (np.inf, model.theta.copy())
    t0 = _time.time()
    for ep in range(epochs):
        idx = rng.choice(n, size=min(batch, n), replace=False)
        loss, g, pb = model.blockwise_score_and_grad(
            tokens_tr[idx], K_blocks,
            cond=(cond_tr[idx] if cond_tr is not None else None))
        gn = float(np.linalg.norm(g))
        if gn > clip:
            g = g * (clip / gn)
        opt.lr = (_sched(ep, epochs, lr, warmup) if schedule == "cosine"
                  else lr * min(1.0, (ep + 1) / max(warmup, 1)))
        model.theta = model.theta - opt.step(g)
        val = model.blockwise_score(tokens_val, K_blocks,
                                     cond=cond_val)
        hist["train_score"].append(float(loss))
        hist["val_score"].append(float(val))
        hist["grad_norm"].append(gn)
        hist["lr"].append(float(opt.lr))
        hist["per_block"].append(pb.tolist())
        if val < best[0] - 1e-6:
            best = (val, model.theta.copy())
            since_best = 0
        else:
            since_best = locals().get("since_best", 0) + 1
        if patience is not None and since_best >= patience:
            if verbose:
                print(f"  early stop at ep {ep + 1} "
                      f"(no val improvement in {patience} epochs)")
            break
        if verbose and (ep < 3 or (ep + 1) % max(1, epochs // 10) == 0):
            print(f"  ep {ep + 1:3d}/{epochs} score={loss:.6f} "
                  f"val={val:.6f} ||g||={gn:.4f} "
                  f"[{_time.time() - t0:.0f}s]")
    model.theta = best[1]
    hist["best_val_score"] = float(best[0])
    return hist
