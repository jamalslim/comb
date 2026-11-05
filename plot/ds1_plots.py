#!/usr/bin/env python3
"""Every Dataset 1 figure of the paper, from the outputs of the comb package.

    cd comb
    python plot/ds1_plots.py                        # figures -> outputs/paper_figures/
    python plot/ds1_plots.py --out ../comb_paper/figures --style CMS

This folder (plot/) sits inside the comb folder; the data are read from the comb folder around it.
Reads (all written by comb_ds1.py and scripts/hardware_run.py):
  outputs/samples/_test.npz              Geant4 test showers                        (comb_ds1.py sample)
  outputs/samples/ds1_*.npz              generated showers of every model, and the device-started showers
                                         <run>_<backend>.npz                        (hardware_run.py complete)
  outputs/hardware/*.samples.npz         the device record: counts, TV per round, exact count laws
  outputs/runs/<run>.json                training histories
Every plot is a single plot in its own file, without a title, drawn at the size it is printed (a third of the
page unless noted): ds1_corr_{geant4,sim,device}.pdf, ds1_energy_{total,layer0,layer1,layer2,layer3,layer12}.pdf,
ds1_marginal_01..12.pdf (twelve voxels; the voxel is named on the axis), ds1_radial_layer{0,1,2,3,12}.pdf, ds1_hardware_tv.pdf,
ds1_hardware_counts_{first,last}.pdf, and ds1_training.pdf (one column). Plots whose inputs are missing are skipped.
Choose the runs with --sim, --hmm and --device (file stems in outputs/samples); by default the CoMB run
ds1_comb_chi4_s0, the first HMM run and the first device-started run are used.
"""
import argparse, json, pathlib, sys
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from matplotlib.ticker import LogLocator, NullFormatter

HERE = pathlib.Path(__file__).resolve().parent
COMB = HERE.parent             # plot/ lives inside the comb folder
sys.path.insert(0, str(HERE))
import hepstyle as hs                                                         # noqa: E402
plt = hs.plt

LAYERS, ANG, RAD = [8, 160, 190, 5, 5], [1, 10, 10, 1, 1], [8, 16, 19, 5, 5]   # Dataset 1 photons
EDGES = np.concatenate([[0], np.cumsum(LAYERS)])
LNAMES = ["layer 0", "layer 1", "layer 2", "layer 3", "layer 12"]


# ----------------------------------------------------------------------------------------------- inputs
def find_runs(samples, sim, hmm, device):
    """(label, showers, color, linestyle) for the simulated CoMB, the HMM and the device-started showers."""
    files = {p.stem: p for p in samples.glob("ds1_*.npz")}
    def spec(p):
        try:
            return json.loads(str(np.load(p, allow_pickle=True)["spec"]))
        except Exception:
            return {}
    pick = lambda name, cond: name if name in files else next((n for n in sorted(files) if cond(n, spec(files[n]))), None)
    is_dev = lambda n: any(b in n for b in ("ibm", "kingston", "device", "quantinuum", "ionq"))
    s = pick(sim, lambda n, sp: not is_dev(n) and sp.get("model") == "comb" and not sp.get("dephase"))
    h = pick(hmm, lambda n, sp: sp.get("model") == "hmm")
    d = pick(device, lambda n, sp: is_dev(n))
    out = {}
    if s:
        out["sim"] = ("CoMB, simulation", np.load(files[s])["showers"], hs.COLORS["comb"], "-", s)
    if h:
        out["hmm"] = ("HMM, $\\chi=4$", np.load(files[h])["showers"], hs.COLORS["hmm"], "--", h)
    if d:
        out["device"] = ("CoMB, device-started", np.load(files[d])["showers"], hs.COLORS["device"], "-.", d)
    return out


def check_same_layout(S_te, runs):
    """Stop if the Geant4 test showers and a model's showers are not in the same voxel order (for example a _test.npz
    written by another version of comb_ds1.py): every figure would compare different voxels. The mean shower of a
    trained model follows Geant4 voxel by voxel (correlation about 0.99), so a low correlation means a mismatch."""
    g = S_te.mean(0)
    for k, v in runs.items():
        r = np.corrcoef(g, v[1].mean(0))[0, 1]
        if not np.isfinite(r) or r < 0.9:
            sys.exit(f"STOP: the Geant4 test showers (outputs/samples/_test.npz) and the {k} showers ({v[4]}) do not have the same\n"
                     f"voxel layout (correlation of their mean showers {r:.2f}, expected about 0.99). Regenerate _test.npz with the\n"
                     f"same comb_ds1.py that wrote the model samples:  python scripts/comb_ds1.py sample --run outputs/runs/{v[4]}.npz")
    print("  voxel layout check: Geant4 and every model agree (" + ", ".join(f"{k} {np.corrcoef(g, v[1].mean(0))[0, 1]:.3f}" for k, v in runs.items()) + ")")


# ----------------------------------------------------------------------------------------------- correlations
def fig_corr(S_te, runs, out):
    """One plot per source: Geant4, CoMB in simulation, device-started CoMB (a third of the page each)."""
    panels = [("geant4", S_te)] + [(k, runs[k][1]) for k in ("sim", "device") if k in runs]
    for tag, S in panels:
        fig, ax = plt.subplots(figsize=(hs.THIRD, hs.THIRD * 0.86))
        C = np.nan_to_num(np.corrcoef(S.T))
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
        for e in EDGES[1:-1]:
            ax.axhline(e - 0.5, color="k", lw=0.3)
            ax.axvline(e - 0.5, color="k", lw=0.3)
        ax.set_xlabel("voxel")
        ax.set_ylabel("voxel")
        ax.minorticks_off()
        ax.set_xticks([0, 150, 300])
        ax.set_yticks([0, 150, 300])
        ax.tick_params(labelsize=7.5)
        cb = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.02)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("Pearson correlation", fontsize=8, loc="center")
        fig.get_layout_engine().set(w_pad=0.08)
        hs.save(fig, out / f"ds1_corr_{tag}.pdf")
    print(f"  ds1_corr_*.pdf: {len(panels)} plots")


# ----------------------------------------------------------------------------------------------- energies
def _bins(x_all, n=30, pct=(0.5, 99.5)):
    lo, hi = np.percentile(x_all, pct)
    return np.linspace(lo, hi if hi > lo else lo + 1, n + 1)


def _hist_plot(path, xd, models, xl, bins, logy=False, legend=False):
    fig, ax, axr = hs.ratio_plot(hs.THIRD, 2.35)
    hs.hist_with_ratio(ax, axr, xd, models, bins, logy=logy)
    axr.set_xlabel(xl, fontsize=8)
    ax.set_ylabel("density", fontsize=8)
    axr.set_ylabel("ratio", fontsize=8)
    for a_ in (ax, axr):
        a_.tick_params(labelsize=7)
    if legend:                                                                # headroom so the legend sits above the data
        lo, hi = ax.get_ylim()
        ax.set_ylim(lo, hi * 30 if logy else hi * 1.75)
        ax.legend(loc="upper left", fontsize=6.5, handlelength=1.5)
    hs.save(fig, path)


def fig_energies(S_te, runs, out):
    """One plot per quantity: total energy and the energy of each layer, with ratio panels."""
    models = [runs[k] for k in ("sim", "hmm", "device") if k in runs]
    quantities = [("total", "total energy [GeV]", lambda S: S.sum(1) / 1e3)] + \
                 [(f"layer{nm.split()[-1]}", f"{nm} energy [GeV]", (lambda S, i=i: S[:, EDGES[i]:EDGES[i + 1]].sum(1) / 1e3)) for i, nm in enumerate(LNAMES)]
    for k, (tag, xl, f) in enumerate(quantities):
        xd = f(S_te)
        xs = [f(m[1]) for m in models]
        _hist_plot(out / f"ds1_energy_{tag}.pdf", xd, [(x, m[0], m[2], m[3]) for x, m in zip(xs, models)], xl, _bins(np.concatenate([xd] + xs)), legend=(k == 0))
    print(f"  ds1_energy_*.pdf: {len(quantities)} plots, {len(models)} models")


# ----------------------------------------------------------------------------------------------- voxel marginals
def representative_voxels(S_te, per_layer=(0.3, 0.6, 0.9), n_max=12):
    means = S_te.mean(0)
    pick = []
    for i in range(len(LAYERS)):
        idx = np.arange(EDGES[i], EDGES[i + 1])
        order = idx[np.argsort(means[idx])]
        pick += [int(order[int(q * (len(order) - 1))]) for q in (per_layer if len(order) >= 3 else (0.5,))]
    return pick[:n_max]


def fig_marginals(S_te, runs, out):
    """One plot per voxel, twelve voxels across the detector, with ratio panels."""
    models = [runs[k] for k in ("sim", "hmm", "device") if k in runs]
    vox = representative_voxels(S_te)
    for k, v in enumerate(vox):
        li = int(np.searchsorted(EDGES, v, side="right") - 1)
        xd = S_te[:, v]
        xs = [m[1][:, v] for m in models]
        _hist_plot(out / f"ds1_marginal_{k + 1:02d}.pdf", xd, [(x, m[0], m[2], m[3]) for x, m in zip(xs, models)],
                   f"{LNAMES[li]}, voxel {v} [MeV]", _bins(np.concatenate([xd] + xs), n=25), logy=True, legend=(k == 0))
    print(f"  ds1_marginal_01..{len(vox):02d}.pdf: voxels {vox}")


# ----------------------------------------------------------------------------------------------- radial profiles
def fig_radial(S_te, runs, out):
    """One plot per layer: mean energy per voxel against the radial bin (averaged over angle), with ratio panels."""
    models = [runs[k] for k in ("sim", "hmm", "device") if k in runs]
    prof = lambda S, li: S[:, EDGES[li]:EDGES[li + 1]].reshape(len(S), RAD[li], ANG[li]).mean(axis=(0, 2))
    err = lambda S, li: S[:, EDGES[li]:EDGES[li + 1]].reshape(len(S), RAD[li], ANG[li]).mean(axis=2).std(0) / np.sqrt(len(S))
    for li, nm in enumerate(LNAMES):
        fig, ax, axr = hs.ratio_plot(hs.THIRD, 2.35)
        r = np.arange(RAD[li])
        pd_ = prof(S_te, li)
        ed = err(S_te, li)
        for lab, S, col, ls, _ in models:
            pm = prof(S, li)
            ax.plot(r, pm, ls, color=col, lw=1.2, label=lab, drawstyle="steps-mid")
            with np.errstate(divide="ignore", invalid="ignore"):
                axr.plot(r, np.where(pd_ > 0, pm / pd_, np.nan), ls, color=col, lw=1.0, drawstyle="steps-mid")
        ax.errorbar(r, pd_, yerr=ed, fmt="o", ms=2.6, color="k", elinewidth=0.7, label="Geant4", zorder=5)
        with np.errstate(divide="ignore", invalid="ignore"):
            axr.errorbar(r, np.ones_like(r, dtype=float), yerr=np.where(pd_ > 0, ed / pd_, np.nan), fmt="none", ecolor="0.55", elinewidth=0.6)
        ax.set_yscale("log")
        ax.tick_params(labelbottom=False, labelsize=7)
        axr.tick_params(labelsize=7)
        axr.axhline(1, color="k", lw=0.6)
        axr.set_ylim(0.5, 1.5)
        axr.set_xlabel(f"{nm}: radial bin", fontsize=8)
        ax.set_ylabel("mean energy [MeV]", fontsize=8)
        axr.set_ylabel("ratio", fontsize=8)
        if li == 0:                                                           # headroom so the legend sits above the profile
            lo, hi = ax.get_ylim()
            ax.set_ylim(lo, hi * 40)
            ax.legend(loc="upper right", fontsize=6.5, handlelength=1.5)
        hs.save(fig, out / f"ds1_radial_layer{nm.split()[-1]}.pdf")
    print(f"  ds1_radial_layer*.pdf: {len(LNAMES)} plots")


# ----------------------------------------------------------------------------------------------- hardware
def fig_hardware(record, out, rng=np.random.default_rng(0)):
    """Three plots of the device run: TV per round, and the count distributions of the first and last rounds."""
    rec = np.load(record, allow_pickle=True)
    C = rec["counts"]
    tv = rec["tv"]
    q = rec["exact_marginals"]
    n, k = C.shape
    K = q.shape[1]
    floor = np.sqrt(K / (2 * np.pi * n))
    tv_n = np.array([0.5 * np.abs(np.bincount(rng.choice(K, n, p=q[j] / q[j].sum()), minlength=K) / n - q[j]).sum() for j in range(k)])
    backend = str(rec["backend"]) if "backend" in rec.files else "device"
    fig, ax = plt.subplots(figsize=(hs.THIRD, 2.2))
    x = np.arange(1, k + 1)
    ax.plot(x, tv, "o-", color=hs.COLORS["device"], ms=3.5, label=backend)
    ax.plot(x, tv_n, "s", color="0.45", ms=3, mfc="white", label=f"exact-model sample")
    ax.axhline(floor, color="0.3", ls=":", lw=0.8, label="shot-noise floor")
    ax.set_xlabel("round", fontsize=8)
    ax.set_ylabel("total-variation distance", fontsize=8)
    ax.set_xticks([1, 4, 8, 12])
    ax.minorticks_off()
    ax.set_ylim(0, 1.75 * tv.max())
    ax.tick_params(labelsize=7)
    ax.legend(loc="upper left", fontsize=6.5)
    hs.save(fig, out / "ds1_hardware_tv.pdf")
    for tag, j in (("first", 0), ("last", k - 1)):
        fig, a = plt.subplots(figsize=(hs.THIRD, 2.2))
        emp = np.bincount(C[:, j], minlength=K) / n
        a.bar(np.arange(K), q[j], color=hs.COLORS["comb"], alpha=0.35, width=0.8, label="exact model")
        a.errorbar(np.arange(K), emp, yerr=np.sqrt(emp * (1 - emp) / n), fmt="o", color=hs.COLORS["device"], ms=3, elinewidth=0.7, label=backend)
        a.set_xticks(range(K))
        a.minorticks_off()
        a.set_xlabel(f"count in round {j + 1}", fontsize=8)
        a.set_ylabel("probability", fontsize=8)
        a.set_ylim(0, 1.45 * max(q[j].max(), emp.max()))
        a.tick_params(labelsize=7)
        a.legend(loc="upper left", fontsize=6.5)
        hs.save(fig, out / f"ds1_hardware_counts_{tag}.pdf")
    print(f"  ds1_hardware_*.pdf: 3 plots; TV first {tv[0]:.3f}, last {tv[-1]:.3f}, mean {tv.mean():.3f}, floor {floor:.3f}")


# ----------------------------------------------------------------------------------------------- training
def fig_training(runs_dir, names, out):
    fig, ax = plt.subplots(figsize=(hs.COL, 2.1))
    n = 0
    for name, lab, col, ls in names:
        f = runs_dir / f"{name}.json"
        if f.exists():
            h = json.load(open(f))["hist"]
            ax.plot(h["step"][1:], h["val"][1:], ls, color=col, label=lab)
            n += 1
    if n == 0:
        plt.close(fig)
        print("  ds1_training.pdf: no training records found; skipped")
        return
    ax.set_yscale("log")
    ax.set_xlabel("optimizer step")
    ax.set_ylabel("validation MMD$^2$ (exact)")
    ax.legend(loc="upper right")
    hs.save(fig, out / "ds1_training.pdf")
    print(f"  ds1_training.pdf: {n} runs")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comb", default=str(COMB), help="the comb folder (default: the folder that contains plot/)")
    ap.add_argument("--out", default=str(COMB / "outputs" / "paper_figures"))
    ap.add_argument("--style", default="ROOT")
    ap.add_argument("--sim", default="ds1_comb_chi4_s0")
    ap.add_argument("--hmm", default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--record", default=None, help="device record (*.samples.npz); default: the first in outputs/hardware")
    ap.add_argument("--no-device", action="store_true", help="leave out the device-started showers and the hardware figure (e.g. for a model that was not run on the device)")
    a = ap.parse_args()
    root = pathlib.Path(a.comb).resolve()
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"data from {root}\nfigures to {out.resolve()}")
    print(f"style: {'mplhep ' + a.style if hs.use(a.style) else 'matplotlib fallback (pip install mplhep for the HEP style)'}")
    samples = root / "outputs" / "samples"
    if (samples / "_test.npz").exists():
        S_te = np.load(samples / "_test.npz")["showers"]
        runs = find_runs(samples, a.sim, a.hmm, a.device)
        if a.no_device: runs.pop("device", None)
        check_same_layout(S_te, runs)
        print(f"  Geant4 test showers: {S_te.shape}; runs: " + ", ".join(f"{k}={v[4]}" for k, v in runs.items()))
        fig_corr(S_te, runs, out)
        fig_energies(S_te, runs, out)
        fig_marginals(S_te, runs, out)
        fig_radial(S_te, runs, out)
    else:
        print("  no outputs/samples/_test.npz: run  python scripts/comb_ds1.py sample  first; shower figures skipped")
    rec = pathlib.Path(a.record) if a.record else next(iter(sorted((root / "outputs" / "hardware").glob("*.samples.npz"))), None)
    if a.no_device:
        print("  --no-device: device-started showers and hardware figure left out")
    elif rec and rec.exists():
        fig_hardware(rec, out)
    else:
        print("  no device record (outputs/hardware/*.samples.npz); hardware figure skipped")
    hmm_name = a.hmm or next((p.stem for p in sorted((root / "outputs" / "runs").glob("ds1_hmm*.json"))), None)
    fig_training(root / "outputs" / "runs", [(a.sim, "CoMB, $\\chi=4$", hs.COLORS["comb"], "-")] + ([(hmm_name, "HMM, $\\chi=4$", hs.COLORS["hmm"], "--")] if hmm_name else []), out)


if __name__ == "__main__":
    main()
