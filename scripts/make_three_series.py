#!/usr/bin/env python3
"""Three-series figures: MC data / noiseless simulator / IBM hardware.

Produces, in figures/:
    marg3_00..11.{pdf,png}   per-cell intensity, 3 series + dual ratio panel
    energy3.{pdf,png}        total energy E=sum_j y_j, 3 series + ratios

The simulator series is rendered from the EXACT theta the hardware job
ran (recorded inside the run file), so simulator and hardware are the
same circuit and the only difference between the curves is device noise.
Do not substitute another theta: that would compare hardware-of-theta_1
against simulator-of-theta_2 under a caption claiming they match.

    python scripts/make_three_series.py [run_file.npz]
"""
import sys, pathlib
SD = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SD)); sys.path.insert(0, str(SD.parent / "src"))
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from comb import d12_config, SequentialBornModel, CoMBQSpec
from _pipeline import load_all, OUT

try:
    import mplhep as hep
    hep.style.use(hep.style.ROOT)
except ImportError:
    plt.rcParams.update({
        "font.family": "sans-serif", "font.size": 15, "axes.linewidth": 1.2,
        "axes.labelsize": 16, "xtick.direction": "in", "ytick.direction": "in",
        "xtick.top": True, "ytick.right": True, "xtick.major.size": 7,
        "ytick.major.size": 7, "xtick.minor.visible": True,
        "ytick.minor.visible": True, "legend.frameon": False,
        "savefig.bbox": "tight"})
BLACK, RED, BLUE = "black", "#e42536", "#3f90da"
FIG = SD.parent / "figures"


def dens(v, bins):
    n, _ = np.histogram(v, bins=bins)
    w = np.diff(bins); t = max(n.sum(), 1)
    return n / (t * w), np.sqrt(n) / (t * w)


def three(vd, vs, vh, name, xlabel, panel, nbins=28):
    lo = min(vd.min(), vs.min(), vh.min()); hi = max(vd.max(), vs.max(), vh.max())
    bins = np.linspace(lo, hi, nbins + 1); ctr = .5 * (bins[:-1] + bins[1:])
    fig, (ax, axr) = plt.subplots(2, 1, figsize=(5.0, 5.0), sharex=True,
                                  gridspec_kw=dict(height_ratios=[3, 1], hspace=0.05))
    ds, es = dens(vs, bins); dh, eh = dens(vh, bins); dd, ed = dens(vd, bins)
    ax.stairs(ds, bins, color=RED, lw=1.6, label="simulator")
    ax.stairs(dh, bins, color=BLUE, lw=1.6, label="IBM hardware")
    ax.errorbar(ctr, dd, yerr=ed, fmt="o", ms=4.5, color=BLACK, lw=1.2, label="MC data")
    ax.set_ylabel("density"); ax.set_ylim(bottom=0)
    ax.text(0.05, 0.9, panel, transform=ax.transAxes, ha="left", va="top", fontsize=13)
    ax.legend(loc="upper right", fontsize=11)
    for dc, ec, col, mk in ((ds, es, RED, "s"), (dh, eh, BLUE, "^")):
        with np.errstate(divide="ignore", invalid="ignore"):
            r = dc / dd
            re = r * np.sqrt((ec / np.maximum(dc, 1e-30)) ** 2
                             + (ed / np.maximum(dd, 1e-30)) ** 2)
        m = np.isfinite(r) & (dd > 0)
        axr.errorbar(ctr[m], r[m], yerr=re[m], fmt=mk, ms=3.5, color=col, lw=1.0)
    axr.axhline(1.0, color=BLACK, lw=1.0); axr.set_ylim(0.4, 1.6)
    axr.set_ylabel("model/data", fontsize=11); axr.set_xlabel(xlabel)
    FIG.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", dpi=165)
    plt.close(fig)


def main():
    run = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / "ibm_run_M3W3L2.npz"
    if not run.exists():
        sys.exit(f"ERROR: hardware run file not found: {run}\n"
                 f"       Run: python scripts/run_ibm.py submit <backend>")
    hw = np.load(run, allow_pickle=True)
    theta_file = str(hw["theta_file"]); depth = int(hw["depth"])
    job = str(hw["job_id"]) if "job_id" in hw.files else "(no job id)"
    T_ibm = hw["tokens"].astype(np.int64)
    print(f"[hw] {run.name}  job={job}  theta={theta_file}  depth={depth} "
          f" {len(T_ibm)} sequences")

    tpath = OUT / theta_file
    if not tpath.exists():
        sys.exit(
            f"ERROR: the hardware run used '{theta_file}', which is missing from\n"
            f"       {OUT}\n"
            f"       This file is the provenance of job {job}: the simulator curve\n"
            f"       must be rendered from the SAME theta the device ran.\n"
            f"       Regenerate it (deterministic, ~1 min):\n"
            f"           python scripts/run_d12.py quantum "
            f"{theta_file[6:-4]}\n"
            f"       Do NOT substitute a different checkpoint.")

    cfg = d12_config()
    (Y_tr, Y_te, blocks, tok, T_tr, T_te, T_fit, T_val) = load_all(cfg)
    d = Y_te.shape[1]
    Y_ibm = tok.detokenize(T_ibm, np.random.default_rng(1))
    m = SequentialBornModel(CoMBQSpec(
        n_mem=3, n_work=cfg.tokenizer.n_work, depth=depth, n_blocks=len(blocks),
        use_prefix=False, share_theta=False, seed=cfg.data.seed))
    th = np.load(tpath)
    if th.size != m.p_theta:
        sys.exit(f"ERROR: {theta_file} has {th.size} params but the depth-{depth} "
                 f"model needs {m.p_theta}.")
    m.theta = th
    Y_sim = tok.detokenize(m.sample(len(Y_te), np.random.default_rng(cfg.data.seed)),
                           np.random.default_rng(1))

    for j in range(d):
        three(Y_te[:, j], Y_sim[:, j], Y_ibm[:, j], f"marg3_{j:02d}",
              f"pixel {j}", f"pixel {j}")
    three(Y_te.sum(1), Y_sim.sum(1), Y_ibm.sum(1), "energy3",
          r"$E=\sum_j y_j$", "total energy", nbins=32)

    def st(Y):
        E = Y.sum(1); return E.mean(), E.std()
    (md, sd), (ms, ss), (mi, si) = st(Y_te), st(Y_sim), st(Y_ibm)
    print(f"[energy] data {md:.3f}+-{sd:.3f} | sim {ms:.3f} "
          f"({(ms-md)/sd:+.2f}sigma, x{ss/sd:.2f}) | ibm {mi:.3f} "
          f"({(mi-md)/sd:+.2f}sigma, x{si/sd:.2f})")
    print(f"[ok] wrote {d} marg3_* + energy3 to {FIG}")


if __name__ == "__main__":
    main()
