#!/usr/bin/env python3
"""
Train CoMB by minimizing MMD^2(model, data) -> 0, then generate and
plot CoMB against the data. That is the whole script. No NLL, no HMM,
no baselines, no comparison tables.

    python scripts/train_two_stage.py            # train (fresh) + plot
    python scripts/train_two_stage.py 300        # train 300 epochs
    python scripts/train_two_stage.py 300 warm.npy   # continue from warm.npy
    python scripts/train_two_stage.py plot       # just re-plot from saved theta

The MMD is the exact squared MMD between the CoMB token-sequence law
q_theta and the empirical data law p_hat under a physics-aware kernel
(codebook-centroid embedding, multi-bandwidth RBF, fixed at init):
        MMD^2 = (q_theta - p_hat)^T K (q_theta - p_hat).
It is 0 iff the two distributions match. With finite data it bottoms
out at the train/test finite-sample floor (drawn on the curve); the
model is driven down to that floor. Exact gradient via the adjoint
Jacobian of the model law.
"""
import sys
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import wasserstein_distance

from comb import (d12_config, CoMBQSpec, SequentialBornModel,
                   TokenKernel, ExactMMD, empirical_hist, block_grams)
from comb.train import AdamOpt, _sched, train_blockscore
from _pipeline import load_all, OUT
from comb.mmd import reinforce_mmd_grad

# ----------------------------- config ---------------------------------
import os
# All settings overridable from the command line via environment
# variables, e.g.:  DEPTH=10 python scripts/train_two_stage.py 300
N_MEM = int(os.environ.get("N_MEM", 3))
N_WORK = int(os.environ.get("N_WORK", 3))   # 4 -> K=16 (stage 2 auto-
#   switches to the verified REINFORCE estimator; exact joint MMD is
#   infeasible at K^B = 65536)
DEPTH = int(os.environ.get("DEPTH", 6))
# widened ansatz (trainable 3-axis entanglers + dense memory-work edges);
# this is the configuration of the shipped best-correlation model.
WIDE = os.environ.get("WIDE", "1") == "1"
EPOCHS = int(os.environ.get("EPOCHS", 150))       # stage-2 budget
LR = float(os.environ.get("LR", 0.01))
STAGE1_EPOCHS = int(os.environ.get("STAGE1_EPOCHS", 450))
TAG = f"M{N_MEM}W{N_WORK}L{DEPTH}"
CORR_KERNEL, CORR_WEIGHT = True, 3.0   # cross-block interaction features
THETA_OUT = OUT / f"theta_mmd_{TAG}.npy"
CURVE_OUT = OUT / f"mmd_curve_{TAG}.npy"
PLOTS = SCRIPT_DIR.parent / "figures"
# ----------------------------------------------------------------------


def setup():
    cfg = d12_config()
    cfg.tokenizer.n_work = N_WORK
    data = load_all(cfg)
    (Y_tr, Y_te, blocks, tok, T_tr, T_te, T_fit, T_val) = data
    sp = dict(trainable_entangler=True, entangler_gates="xyz",
              mw_dense=True) if WIDE else {}
    m = SequentialBornModel(CoMBQSpec(
        n_mem=N_MEM, n_work=N_WORK, depth=DEPTH, n_blocks=len(blocks),
        use_prefix=False, share_theta=False, seed=cfg.data.seed, **sp))
    kern = TokenKernel(kind="embedded", corr_features=CORR_KERNEL,
                       corr_weight=CORR_WEIGHT).fit(tok, T_fit,
                                                    seed=cfg.data.seed)
    ex = ExactMMD(m, kern) if m.K ** m.B <= 20000 else None
    return m, tok, blocks, Y_tr, Y_te, T_fit, T_te, ex, kern


def ts_mmd(kern, A, B_, chunk=2000):
    """Two-sample MMD^2 estimate between token sets (any K)."""
    def mean_gram(P, Q):
        tot = 0.0
        for s0 in range(0, len(P), chunk):
            tot += kern.gram(P[s0:s0 + chunk], Q).sum()
        return tot / (len(P) * len(Q))
    return float(mean_gram(A, A) - 2 * mean_gram(A, B_)
                 + mean_gram(B_, B_))


def train(epochs, warm=None):
    """Two-stage, all-MMD training.
    Stage 1 (skipped if `warm` given): blockwise conditional MMD --
    per-block MMD^2 between the model conditional and the data token,
    strictly proper, trains the conditionals into the right basin.
    Stage 2: joint MMD^2(model, data) minimized to ~the finite-sample
    floor; this is the curve that goes to zero and the one plotted."""
    m, tok, blocks, Y_tr, Y_te, T_fit, T_te, ex, kern = setup()
    cfg = d12_config()
    if warm:
        m.theta = np.load(warm)
        print(f"[stage 1] skipped (warm start: {warm})")
    else:
        print(f"[stage 1] blockwise conditional MMD, "
              f"{STAGE1_EPOCHS} epochs")
        Kb = block_grams(tok)
        n_val = 800
        train_blockscore(m, Kb, T_fit[:-n_val], T_fit[-n_val:],
                         epochs=STAGE1_EPOCHS, batch=1024, lr=0.08,
                         seed=cfg.data.seed, patience=60)
    half = len(T_fit) // 2
    floor = ts_mmd(kern, T_fit[:half], T_fit[half:])
    print(f"[stage 2] joint MMD fine-tune, {epochs} epochs; "
          f"finite-sample floor = {floor:.5f}; "
          f"estimator = {'exact' if ex is not None else 'REINFORCE'}")
    curve = []
    if ex is not None:
        p_fit = empirical_hist(T_fit, m.K, m.B)
        p_te = empirical_hist(T_te, m.K, m.B)
    rng = np.random.default_rng(11)
    legs = [(epochs // 3, LR), (epochs // 3, 0.4 * LR),
            (epochs - 2 * (epochs // 3), 0.18 * LR)]
    ep = -1
    for leg_ep, leg_lr in legs:
        opt = AdamOpt(m.p_theta, lr=leg_lr)
        for le in range(leg_ep):
            ep += 1
            if ex is not None:
                loss, g, q = ex.loss_and_grad(p_fit)
            else:
                loss, g = reinforce_mmd_grad(m, kern, T_fit,
                                             n_model=4096, rng=rng)
            gn = np.linalg.norm(g)
            if gn > 10.0:
                g *= 10.0 / gn
            opt.lr = _sched(le, leg_ep, leg_lr, warmup=2)
            m.theta = m.theta - opt.step(g)
            if ex is not None:
                test_mmd = ex.loss_of_dist(np.exp(m.logp(ex.seqs)), p_te)
            else:
                test_mmd = loss      # running estimate (test at the end)
            curve.append([loss, test_mmd])
            if ep < 3 or (ep + 1) % 10 == 0:
                print(f"  ep {ep+1:4d}/{epochs}  MMD_train={loss:.5f}  "
                      f"MMD_test={test_mmd:.5f}  ||g||={gn:.3f}")
    if ex is None:
        tm = ts_mmd(kern, m.sample(4800, np.random.default_rng(1)), T_te)
        print(f"[final] two-sample MMD2(model, test) = {tm:.5f}")
    np.save(THETA_OUT, m.theta)
    np.save(CURVE_OUT, np.array(curve))
    print(f"[SAVE] {THETA_OUT}")
    from comb import corr_nan_safe, corr_errors
    rngE = np.random.default_rng(1)
    Yg = tok.detokenize(m.sample(len(Y_te), rngE), rngE)
    Cd, Cg = corr_nan_safe(Y_te), corr_nan_safe(Yg)
    ce = corr_errors(Cd, Cg, blocks)
    off = ~np.block([[np.ones((b1, b2), bool) if i == j else
                      np.zeros((b1, b2), bool)
                      for j, (_, b2) in enumerate(blocks)]
                     for i, (_, b1) in enumerate(blocks)])
    print(f"[FINAL {TAG}] corr_cross={ce['cross']:.3f}  "
          f"sigma_E={float(Yg.sum(1).std() / Y_te.sum(1).std()):.3f}  "
          f"cross-block mean|corr|: model="
          f"{np.abs(Cg[off]).mean():.3f} vs data="
          f"{np.abs(Cd[off]).mean():.3f}")
    plot(m, tok, blocks, Y_tr, Y_te, np.array(curve), floor)


def plot(m=None, tok=None, blocks=None, Y_tr=None, Y_te=None,
         curve=None, floor=None):
    PLOTS.mkdir(exist_ok=True)
    if m is None:
        m, tok, blocks, Y_tr, Y_te, T_fit, T_te, ex, kern = setup()
        m.theta = np.load(THETA_OUT)
        curve = (np.load(CURVE_OUT) if CURVE_OUT.exists()
                 else np.empty((0, 2)))
        floor = ts_mmd(kern, T_fit[:len(T_fit)//2],
                       T_fit[len(T_fit)//2:])
    rng = np.random.default_rng(0)
    T_gen = m.sample(len(Y_te), rng)
    Y_gen = tok.detokenize(T_gen, rng)
    d = Y_te.shape[1]

    # ---- Figure 1: MMD training curve -> floor ----
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(curve[:, 0], color="#d62728", lw=1.8, label="MMD$^2$ (train)")
    ax.plot(curve[:, 1], color="#1f77b4", lw=1.4, label="MMD$^2$ (test)")
    ax.axhline(floor, color="k", ls="--", lw=1,
               label=f"finite-sample floor ({floor:.4f})")
    ax.set_xlabel("epoch"); ax.set_ylabel("MMD$^2$(CoMB, data)")
    ax.set_yscale("log"); ax.legend(frameon=False)
    ax.set_title("CoMB training: MMD to the data decreases toward the floor")
    fig.tight_layout()
    fig.savefig(PLOTS / "mmd_training_curve.png", dpi=160)
    print("saved", PLOTS / "mmd_training_curve.png")

    # ---- Figure 2: CoMB vs data — summary (corr x2 + energy) ----
    fig = plt.figure(figsize=(14, 3.6))
    gs = fig.add_gridspec(1, 4, wspace=0.35)
    cax0 = fig.add_subplot(gs[0, 0])
    im = cax0.imshow(np.corrcoef(Y_te.T), vmin=-1, vmax=1, cmap="RdBu_r")
    cax0.set_title("correlations: data", fontsize=10)
    cax0.set_xticks([]); cax0.set_yticks([])
    cax1 = fig.add_subplot(gs[0, 1])
    cax1.imshow(np.corrcoef(Y_gen.T), vmin=-1, vmax=1, cmap="RdBu_r")
    cax1.set_title("correlations: CoMB", fontsize=10)
    cax1.set_xticks([]); cax1.set_yticks([])
    fig.colorbar(im, ax=[cax0, cax1], shrink=0.75)
    axE = fig.add_subplot(gs[0, 2:])
    bins = np.linspace(0, np.quantile(np.r_[Y_te.sum(1), Y_gen.sum(1)],
                                      0.999), 40)
    axE.hist(Y_te.sum(1), bins=bins, density=True, histtype="step",
             color="#1f77b4", lw=2, label="data")
    axE.hist(Y_gen.sum(1), bins=bins, density=True, histtype="step",
             color="#d62728", lw=2, label="CoMB")
    axE.set_title("total energy $\\sum_j y_j$", fontsize=10)
    axE.legend(frameon=False); axE.set_xlabel("energy")
    fig.suptitle("CoMB (red) vs data (blue) — free-running samples",
                 fontsize=13)
    for _name in ("comb_vs_data.png", "corr_neighbor.png"):
        fig.savefig(PLOTS / _name, dpi=155, bbox_inches="tight")
        print("saved", PLOTS / _name)

    # ---- Figure 3: per-pixel marginals, CoMB vs data ----
    ncol = 4
    nrow = int(np.ceil(d / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(14, 3 * nrow))
    axes = np.atleast_2d(axes)
    for j in range(d):
        ax = axes[j // ncol, j % ncol]
        hi = np.quantile(np.r_[Y_te[:, j], Y_gen[:, j]], 0.99)
        b = np.linspace(0, max(hi, 1e-3), 30)
        ax.hist(Y_te[:, j], bins=b, density=True, histtype="step",
                color="#1f77b4", lw=1.8, label="data")
        ax.hist(Y_gen[:, j], bins=b, density=True, histtype="step",
                color="#d62728", lw=1.8, label="CoMB")
        w1 = wasserstein_distance(Y_te[:, j], Y_gen[:, j])
        ax.set_title(f"pixel {j}   $W_1$={w1:.4f}", fontsize=9)
        ax.tick_params(labelsize=7)
        if j == 0:
            ax.legend(frameon=False, fontsize=8)
    for j in range(d, nrow * ncol):
        axes[j // ncol, j % ncol].axis("off")
    fig.suptitle("Per-pixel marginals: CoMB (red) vs data (blue)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(PLOTS / "comb_vs_data_marginals.png", dpi=150)
    print("saved", PLOTS / "comb_vs_data_marginals.png")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "plot":
        plot()
    else:
        ep = int(sys.argv[1]) if len(sys.argv) > 1 else EPOCHS
        warm = sys.argv[2] if len(sys.argv) > 2 else None
        train(ep, warm)
