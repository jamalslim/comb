#!/usr/bin/env python3
"""
Publication figures, ROOT style: each panel a separate vector file
(PDF + PNG), no in-plot titles, ticks inside on all four sides, data
as black points with statistical errors, model as red step histogram,
ratio strips (Data/CoMB with propagated errors) on the total-energy
and per-pixel figures.

    python scripts/plot_paper.py

Outputs -> figures/:
    corr_data.{pdf,png}      correlation heatmap, data
    corr_comb.{pdf,png}      correlation heatmap, CoMB flagship
    energy_sum.{pdf,png}     total energy + Data/CoMB ratio strip
    marg_00..11.{pdf,png}    per-pixel PDFs + ratio strips
"""
import sys
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from comb import d12_config, corr_nan_safe
from _pipeline import load_all, MODELS

OUT = PROJECT_ROOT / "figures"
OUT.mkdir(parents=True, exist_ok=True)
FLAGSHIP = "CoMB M3C0-L6-E450-P60"

ROOT_RC = {
    "font.family": "sans-serif",
    "font.size": 15,
    "axes.linewidth": 1.2,
    "axes.labelsize": 16,
    "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True,
    "xtick.major.size": 7, "ytick.major.size": 7,
    "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
    "xtick.minor.visible": True, "ytick.minor.visible": True,
    "xtick.major.width": 1.1, "ytick.major.width": 1.1,
    "legend.frameon": False,
    "errorbar.capsize": 0,
    "savefig.bbox": "tight",
}
plt.rcParams.update(ROOT_RC)

cfg = d12_config()
(Y_tr, Y_te, blocks, tok, T_tr, T_te, T_fit, T_val) = load_all(cfg)
d = Y_te.shape[1]
z = np.load(MODELS / f"{FLAGSHIP}.npz", allow_pickle=True)
Y_q = tok.detokenize(z["tokens"], np.random.default_rng(1))


def save(fig, stem):
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"{stem}.{ext}", dpi=200)
    plt.close(fig)
    print("saved", OUT / f"{stem}.pdf")


# ---------------------------------------------------------- heatmaps
def heatmap(M, stem):
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r",
                   interpolation="nearest")
    for edge in np.cumsum([b for _, b in blocks])[:-1]:
        ax.axhline(edge - 0.5, color="k", lw=0.8, alpha=0.6)
        ax.axvline(edge - 0.5, color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel("calorimeter cell")
    ax.set_ylabel("calorimeter cell")
    ax.set_xticks(range(0, d, 3))
    ax.set_yticks(range(0, d, 3))
    ax.minorticks_off()
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\rho$")
    save(fig, stem)


heatmap(corr_nan_safe(Y_te), "corr_data")
heatmap(corr_nan_safe(Y_q), "corr_comb")

# The hardware panel. Tokens measured on the device, rendered through the
# same tokenizer and emission as everything else, so the only difference
# from corr_comb is what the device did to them.
_hw = np.load(PROJECT_ROOT / "outputs" / "ibm_run_M3W3L2.npz", allow_pickle=True)
heatmap(corr_nan_safe(tok.detokenize(_hw["tokens"].astype(np.int64),
                                     np.random.default_rng(1))), "corr_ibm")


# --------------------------------------------- hist + ratio machinery
def hist_ratio(x_data, x_model, stem, xlabel, nbins=30, qhi=0.997,
               annotate=None):
    """ROOT-style: data = black points with sqrt(N) errors, CoMB = red
    step; lower panel Data/CoMB with propagated errors."""
    hi = float(np.quantile(np.concatenate([x_data, x_model]), qhi))
    lo = float(min(x_data.min(), x_model.min(), 0.0))
    bins = np.linspace(lo, max(hi, 1e-3), nbins + 1)
    c = 0.5 * (bins[1:] + bins[:-1])
    wbin = bins[1] - bins[0]
    n_d, _ = np.histogram(x_data, bins=bins)
    n_m, _ = np.histogram(x_model, bins=bins)
    Nd, Nm = len(x_data), len(x_model)
    h_d = n_d / (Nd * wbin)
    e_d = np.sqrt(np.maximum(n_d, 0)) / (Nd * wbin)
    h_m = n_m / (Nm * wbin)
    e_m = np.sqrt(np.maximum(n_m, 0)) / (Nm * wbin)
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(5.6, 5.2), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))
    mskm = n_m > 0
    ax.errorbar(c[mskm], h_m[mskm], yerr=e_m[mskm], fmt="s", ms=4.5,
                color="#d62728", lw=1.2, label="CoMB")
    msk = n_d > 0
    ax.errorbar(c[msk], h_d[msk], yerr=e_d[msk], fmt="o", ms=4.5,
                color="k", lw=1.2, label="data")
    ax.set_ylabel("density")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles[::-1], labels[::-1], loc="best")
    ax.tick_params(labelbottom=False)
    if annotate:
        ax.text(0.05, 0.95, annotate, transform=ax.transAxes,
                ha="left", va="top", fontsize=14)
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(n_m > 0, h_d / np.where(h_m > 0, h_m, np.nan),
                     np.nan)
        rerr = r * np.sqrt(1.0 / np.maximum(n_d, 1)
                           + 1.0 / np.maximum(n_m, 1))
    axr.errorbar(c, r, yerr=rerr, fmt="o", ms=3.6, color="k", lw=1.0)
    axr.axhline(1.0, color="#d62728", lw=1.2)
    axr.set_ylim(0.35, 1.65)
    axr.set_ylabel("data / CoMB", fontsize=13)
    axr.set_xlabel(xlabel)
    save(fig, stem)


# total energy, separate
hist_ratio(Y_te.sum(1), Y_q.sum(1), "energy_sum",
           r"$E_{\mathrm{sum}} = \sum_j y_j$", nbins=32)

# per-pixel marginals, each separate, no titles
for j in range(d):
    hist_ratio(Y_te[:, j], Y_q[:, j], f"marg_{j:02d}",
               f"cell {j} intensity", nbins=28,
               annotate=f"cell {j}")
print("done:", len(list(OUT.glob('*.pdf'))), "PDF files")
