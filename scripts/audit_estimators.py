#!/usr/bin/env python3
"""
Audit the two sample-based gradient estimators against the exact gradient.

Neither estimator is used to train any model in the paper; training uses
the exact adjoint gradient. They exist because the exact gradient needs
the full distribution over all K^B symbol sequences, which is only
possible when K^B is small. Beyond that, the objective has to be
estimated from samples, and these are the two ways to do it:

    parameter shift   sampled_mmd_grad     each parameter enters exactly
                                           one rotation per block, so a
                                           +/- pi/2 shift gives its
                                           derivative exactly, in
                                           expectation
    score function    reinforce_mmd_grad   the gradient of an expectation
                                           written as an expectation of
                                           (witness) x (grad log q)

Both are unbiased, so the check is simple: average each one over many
independent draws and the result must approach the exact gradient, with
the error falling roughly as 1/sqrt(draws). A biased estimator would
level off instead.

At d = 12 the sequence space is 8^4 = 4096, small enough to compute the
exact gradient by enumeration, which is what makes the comparison
possible at all.

    python scripts/audit_estimators.py
    python scripts/audit_estimators.py --draws 8,32,128
"""
import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np

from comb import d12_config, SequentialBornModel, CoMBQSpec, TokenKernel
from comb.mmd import ExactMMD, empirical_hist, sampled_mmd_grad, reinforce_mmd_grad
from _pipeline import load_all

ap = argparse.ArgumentParser()
ap.add_argument("--draws", default="4,16,64",
                help="numbers of independent draws to average, comma separated")
ap.add_argument("--depth", type=int, default=2)
ap.add_argument("--seed", type=int, default=3)
a = ap.parse_args()
draws = [int(x) for x in a.draws.split(",")]

cfg = d12_config()
(Y_tr, Y_te, blocks, tok, T_tr, T_te, T_fit, T_val) = load_all(cfg)
B, K = len(blocks), 2 ** cfg.tokenizer.n_work

model = SequentialBornModel(CoMBQSpec(
    n_mem=3, n_work=cfg.tokenizer.n_work, depth=a.depth, n_blocks=B,
    use_prefix=False, share_theta=False, seed=cfg.data.seed))

# An untrained point, on purpose: at a trained optimum the gradient is
# close to zero and every estimator looks accurate for the wrong reason.
model.theta = np.random.default_rng(a.seed).normal(0.0, 0.6, model.p_theta)

kernel = TokenKernel()
kernel.fit(tok, T_fit[:2000])
data = T_fit[:2000]

# The reference. Enumerates all K^B sequences, which is why this audit
# only runs at d = 12.
exact = ExactMMD(model, kernel)
loss_ex, g_ex, _ = exact.loss_and_grad(empirical_hist(data, K, B))
norm_ex = np.linalg.norm(g_ex)

print(f"d=12  B={B}  K={K}  K^B={K**B}  parameters={model.p_theta}")
print(f"exact:  MMD^2 = {loss_ex:.5f}   |grad| = {norm_ex:.5f}\n")
print(f"{'estimator':16} {'draws':>6} {'rel. error':>11} {'cosine':>8} {'time':>7}")


def audit(name, one_draw):
    rng = np.random.default_rng(a.seed + 1)
    acc = np.zeros_like(g_ex)
    done, errs = 0, []
    t0 = time.time()
    for target in draws:
        while done < target:
            _, g = one_draw(rng)
            acc += g
            done += 1
        mean = acc / done
        err = np.linalg.norm(mean - g_ex) / norm_ex
        cos = float(mean @ g_ex / (np.linalg.norm(mean) * norm_ex))
        errs.append(err)
        print(f"{name:16} {done:>6} {err:>11.4f} {cos:>8.4f} {time.time()-t0:>6.0f}s")
    return errs


errs_ps = audit("parameter shift",
                lambda r: sampled_mmd_grad(model, kernel, data, rng=r))
print()
errs_sf = audit("score function",
                lambda r: reinforce_mmd_grad(model, kernel, data, rng=r))

# Unbiased means the error keeps falling as draws are added. Compare the
# drop against what 1/sqrt(draws) predicts between the first and last row.
expected = np.sqrt(draws[-1] / draws[0])
print(f"\nfrom {draws[0]} to {draws[-1]} draws, 1/sqrt(draws) predicts the error "
      f"falls by {expected:.1f}x")
ok = True
for name, e in (("parameter shift", errs_ps), ("score function", errs_sf)):
    fall = e[0] / e[-1]
    verdict = "consistent with unbiased" if fall > 0.5 * expected else "NOT FALLING -- check for bias"
    ok &= fall > 0.5 * expected
    print(f"  {name:16} fell {fall:.1f}x   {verdict}")
sys.exit(0 if ok else 1)
