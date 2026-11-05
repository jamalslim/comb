#!/usr/bin/env python3
"""Plot the seed study from every run (maxwell/seed_runs.csv, written by maxwell/summarize_seeds.py).

    python maxwell/plot_results.py --style ROOT --exclude "CoMB chi=8" "dephased twin chi=8"

Writes two figures in the style of the other Dataset 1 figures (comb_ds1.py plot):
  outputs/figs/ds1_memory_full.pdf      held-out test MMD^2 against memory size: every run as a dot, means with standard
                                        errors, the classical HMM curve, CoMB and its dephased twin for every variant, and
                                        the equivalent classical memory of each CoMB group
  outputs/figs/ds1_seed_comparison.pdf  every run of CoMB, its dephased twin and the HMM at chi = 4 and at chi = 8, with
                                        means, standard errors, and the differences in percent and in standard errors
Options: --exclude leaves groups out (names as printed by summarize_seeds.py); --chi8 picks the CoMB and twin variant for
the chi = 8 panel ("nested", the default, or "" for the original wiring); --style is the mplhep style.
"""
import argparse, csv, pathlib, re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--style", default="ROOT", help="mplhep style: ROOT, ATLAS, CMS, LHCb2, ALICE")
ap.add_argument("--exclude", nargs="*", default=[], help='groups to leave out, e.g. "CoMB chi=8" "dephased twin chi=8"')
ap.add_argument("--chi8", default="nested", help='CoMB and twin variant in the chi = 8 panel: "nested" (default) or "" (original wiring)')
ap.add_argument("--jitter-seed", type=int, default=0)
a = ap.parse_args()
try:
    import mplhep as hep
    plt.style.use(getattr(hep.style, a.style))
except Exception:
    print("  (mplhep not installed: pip install mplhep for the HEP style)")
plt.rcParams.update({"font.size": 14, "legend.fontsize": 14, "legend.loc": "best", "figure.dpi": 350})

# ------------------------------------------------------------------------------------------------ every run
G = {}
for r in csv.DictReader(open(ROOT / "maxwell/seed_runs.csv")):
    if r["group"] not in a.exclude and r["test"] not in ("", None):
        G.setdefault(r["group"], []).append(1e4 * float(r["test"]))
def parse(g):                                     # "CoMB nested L=18 chi=8" -> ("CoMB", "nested L=18", 8)
    m = re.match(r"^(CoMB|dephased twin|HMM)((?: dense)?(?: nested)?(?: L=\d+)?) chi=(\d+)$", g)
    return (m.group(1), m.group(2).strip(), int(m.group(3))) if m else None
info = {g: parse(g) for g in G if parse(g)}
mean = {g: float(np.mean(v)) for g, v in G.items()}
se = {g: (float(np.std(v, ddof=1) / np.sqrt(len(v))) if len(v) > 1 else 0.0) for g, v in G.items()}
hmm = sorted((c, g) for g, (k, v, c) in info.items() if k == "HMM")
hx = np.array([c for c, _ in hmm])
hm = np.array([mean[g] for _, g in hmm])
he = np.array([se[g] for _, g in hmm])

def chi_equiv(m):                                  # where the HMM means cross m (log MMD^2 against log chi)
    for (x0, g0), (x1, g1) in zip(hmm[:-1], hmm[1:]):
        if (mean[g0] - m) * (mean[g1] - m) <= 0:
            f = (np.log(mean[g0]) - np.log(m)) / (np.log(mean[g0]) - np.log(mean[g1]))
            return float(np.exp(np.log(x0) + f * (np.log(x1) - np.log(x0))))
    return None

def label(kind, variant):
    v = variant.replace("L=", "").replace("nested", "nested wiring")
    v = (v + " layers") if "L=" in variant else v
    return ("CoMB" if kind == "CoMB" else "dephased twin") + (f", {v}" if variant else "")

MARK = {"": "o", "nested": "s", "L=18": "*", "nested L=18": "P"}
rng = np.random.default_rng(a.jitter_seed)

# ------------------------------------------------------------------------------------------------ figure 1: memory curve, every run
fig, ax = plt.subplots(figsize=(6, 4))
for g, (k, v, c) in info.items():                                            # single runs, slightly jittered in chi
    off = 1.0 if k == "HMM" else (0.97 if k == "CoMB" else 1.03) * (1.0 if not v else (0.97 if k == "CoMB" else 1.03))
    x = c * off * np.exp(rng.normal(0, 0.012, len(G[g])))
    if k == "dephased twin":
        ax.scatter(x, G[g], s=7, facecolors="none", edgecolors="tab:blue", linewidths=0.5, alpha=0.35, zorder=2)
    else:
        ax.scatter(x, G[g], s=6, color="tab:green" if k == "HMM" else "tab:blue", alpha=0.25, linewidths=0, zorder=2)
ax.errorbar(hx, hm, yerr=he, fmt="s-", color="tab:green", ms=6, lw=1.5, capsize=3, label=r"HMM($\chi$)", zorder=4)
n_lab = 0
for g, (k, v, c) in sorted(info.items(), key=lambda t: (t[1][0], t[1][1], t[1][2])):
    if k == "HMM":
        continue
    off = (0.97 if k == "CoMB" else 1.03) * (1.0 if not v else (0.97 if k == "CoMB" else 1.03))
    first = not any(l.get_label() == label(k, v) for l in ax.get_lines() + ax.collections)
    if k == "CoMB":
        ax.errorbar([c * off], [mean[g]], yerr=[se[g]], fmt=MARK.get(v, "o"), color="tab:blue", ms=7 if not v else 8, capsize=3,
                    label=label(k, v) if first else None, zorder=6)
        eq = chi_equiv(mean[g])
        if eq:
            ax.plot([c * off, eq], [mean[g], mean[g]], color="tab:blue", ls=":", lw=1.2)
            ax.plot([eq, eq], [min(hm.min(), mean[g]) * 0.9, mean[g]], color="0.45", ls=":", lw=1.2)
            ax.annotate(rf"$\chi_{{\rm equiv}}\approx{eq:.1f}$", (eq, mean[g]), xytext=(6, 8 if n_lab % 2 == 0 else -20), textcoords="offset points", fontsize=12)
            n_lab += 1
    else:
        eb = ax.errorbar([c * off], [mean[g]], yerr=[se[g]], fmt="D" if not v else "P", mfc="none", color="tab:blue", ms=6, capsize=3,
                         label=label(k, v) if first else None, zorder=6)
        eb[-1][0].set_linestyle("--")
xs = sorted({c for (_, _, c) in info.values()})
ax.set_xscale("log")
ax.set_xticks(xs)
ax.set_xticklabels([str(x) for x in xs])
ax.minorticks_off()
allv = np.concatenate([np.array(v) for v in G.values()])
ax.set_xlim(min(xs) * 0.85, max(xs) * 1.15)
ax.set_ylim(allv.min() * 0.92, allv.max() + 0.55 * (allv.max() - allv.min()))   # headroom for the legend
ax.set_xlabel(r"memory size $\chi$")
ax.set_ylabel(r"test MMD$^2$ ($10^{-4}$)")
h, l = ax.get_legend_handles_labels()
h.append(Line2D([], [], ls="none", marker="o", ms=3, color="0.5", alpha=0.5))
l.append("single runs")
ax.legend(h, l, loc="upper center", ncol=2, fontsize=10, columnspacing=1.0, handlelength=1.4, frameon=False)
out = ROOT / "outputs/figs"
out.mkdir(parents=True, exist_ok=True)
fig.savefig(out / "ds1_memory_full.pdf", bbox_inches="tight")
plt.close(fig)
print(f"written: {out / 'ds1_memory_full.pdf'}")

# ------------------------------------------------------------------------------------------------ figure 2: CoMB, twin and HMM run by run
v8 = (a.chi8 + " ") if a.chi8 else ""
panels = [("chi = 4", ["CoMB chi=4", "dephased twin chi=4", "HMM chi=4"]),
          ("chi = 8" + ((", " + a.chi8.replace("nested", "nested wiring").replace("L=", "") + (" layers" if "L=" in a.chi8 else "")) if a.chi8 else ""),
           [f"CoMB {v8}chi=8", f"dephased twin {v8}chi=8", "HMM chi=8"])]
panels = [(t, gs) for t, gs in panels if all(g in G for g in gs)]
if panels:
    fig, axes = plt.subplots(1, len(panels), figsize=(5.2 * len(panels), 4.4), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (title, gs) in zip(axes, panels):
        for i, g in enumerate(gs):
            col = "tab:green" if g.startswith("HMM") else "tab:blue"
            y = np.array(G[g])
            x = i + rng.uniform(-0.18, 0.18, len(y))
            if g.startswith("dephased"):
                ax.scatter(x, y, s=14, facecolors="none", edgecolors=col, linewidths=0.8, alpha=0.6)
            else:
                ax.scatter(x, y, s=12, color=col, alpha=0.55, linewidths=0)
            ax.errorbar([i], [mean[g]], yerr=[se[g]], fmt="_", ms=26, mew=2.2, color="k", capsize=6, zorder=5)
        top = max(max(G[g]) for g in gs)
        for i, g in enumerate(gs[1:], start=1):                               # difference to CoMB, in % and standard errors
            d = mean[g] - mean[gs[0]]
            s_d = np.sqrt(se[g] ** 2 + se[gs[0]] ** 2)
            z = d / s_d if s_d > 0 else float("nan")
            ax.annotate(f"{100 * d / mean[g]:+.0f}%\n{z:+.1f} SE", (i, top * 1.02), ha="center", va="bottom", fontsize=11)
        ax.set_xticks(range(len(gs)))
        ax.set_xticklabels(["CoMB", "dephased\ntwin", "HMM"])
        ax.set_xlim(-0.6, len(gs) - 0.4)
        ax.set_title(title.replace("chi", r"$\chi$"), fontsize=14)
    axes[0].set_ylabel(r"test MMD$^2$ ($10^{-4}$)")
    lo = min(min(G[g]) for _, gs in panels for g in gs)
    hi = max(max(G[g]) for _, gs in panels for g in gs)
    axes[0].set_ylim(lo * 0.92, hi * 1.18)
    fig.text(0.5, -0.02, "differences: other model minus CoMB, relative to the other model; positive means CoMB is better", ha="center", fontsize=10)
    fig.savefig(out / "ds1_seed_comparison.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"written: {out / 'ds1_seed_comparison.pdf'}")

print("\nheld-out test MMD^2 (1e-4), every run:")
for g in sorted(G, key=lambda g: (info.get(g, ("", "", 0))[0] != "CoMB", g)):
    eq = chi_equiv(mean[g]) if info.get(g, ("",))[0] == "CoMB" else None
    print(f"  {g:28s} n={len(G[g]):3d}  mean {mean[g]:.3f} +- {se[g]:.3f}  range {min(G[g]):.3f} to {max(G[g]):.3f}" + (f"  chi_equiv {eq:.1f}" if eq else ""))
