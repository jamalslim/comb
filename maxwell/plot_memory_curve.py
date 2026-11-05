#!/usr/bin/env python3
"""Shower memory-compression figure in the style of the other Dataset 1 figures (comb_ds1.py plot).

    python maxwell/plot_memory_curve.py [--style ROOT]     # reads maxwell/seed_summary.json -> outputs/figs/ds1_memory_compression.pdf

Held-out test MMD^2 against memory size: classical HMMs at every chi that was run (mean +- standard error over seeds),
CoMB and its dephased twin at every size that was run, and for each CoMB size its level (dotted) and its equivalent
classical memory. Same mplhep style, font sizes, colours (CoMB tab:blue, HMM tab:green, twin dashed), figure size and
saving as comb_ds1.py plot. Reads both the current summary format and the earlier one (CoMB at chi = 4 only)."""
import argparse, json, pathlib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = pathlib.Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--style", default="ROOT", help="mplhep style: ROOT, ATLAS, CMS, LHCb2, ALICE")
ap.add_argument("--exclude", nargs="*", default=[], help='groups to leave out, e.g. --exclude "CoMB chi=8" "dephased twin chi=8"')
a = ap.parse_args()
try:
    import mplhep as hep
    plt.style.use(getattr(hep.style, a.style))
except Exception:
    print("  (mplhep not installed: pip install mplhep for the HEP style)")
plt.rcParams.update({"font.size": 14, "legend.fontsize": 14, "legend.loc": "best", "figure.dpi": 350})

S = json.load(open(ROOT / "maxwell/seed_summary.json"))
for g in a.exclude: S.pop(g, None)
if "CoMB" in S:                                                     # earlier format: CoMB and twin at chi = 4 only
    S["CoMB chi=4"] = S.pop("CoMB")
    if "dephased twin" in S: S["dephased twin chi=4"] = S.pop("dephased twin")
    eq = S.get("equivalent classical memory")
    S["equivalent classical memory"] = {"4": eq} if not isinstance(eq, dict) else eq
import re
def pick(kind, depth=None):                       # groups like "CoMB chi=8" (5 layers) or "CoMB L=18 chi=8"
    out = []
    for k, v in S.items():
        m = re.match(rf"^{kind}((?: dense)?(?: nested)?(?: L=\d+)?) chi=(\d+)$", k)
        if m and isinstance(v, dict) and "test" in v and (m.group(1).strip() or None) == depth:
            out.append((int(m.group(2)), np.array(v["test"]) * 1e4))
    return sorted(out)
hmm = pick("HMM")
depths = sorted({(re.match(r"^CoMB((?: dense)?(?: nested)?(?: L=\d+)?) chi=", k).group(1).strip() or None) for k in S if re.match(r"^CoMB((?: dense)?(?: nested)?(?: L=\d+)?) chi=", k)}, key=lambda d: (d or ""))
hx = np.array([x for x, _ in hmm])
hm = np.array([v[0] for _, v in hmm])
he = np.array([v[1] for _, v in hmm])
fig, ax = plt.subplots(figsize=(6, 4))
ax.errorbar(hx, hm, yerr=he, fmt="s-", color="tab:green", ms=6, lw=1.5, capsize=3, label=r"HMM($\chi$)")
eqs = S.get("equivalent classical memory") or {}
MK = {None: ("o", 7), "nested": ("s", 7), "L=18": ("*", 11), "nested L=18": ("*", 11), "dense": ("^", 9), "dense nested": ("^", 9)}
MT = {None: "D", "nested": "d", "L=18": "P", "nested L=18": "P", "dense": "v", "dense nested": "v"}
def lab(kind, d):
    if d is None: return kind
    parts = (["general"] if "dense" in d else []) + (["nested"] if "nested" in d else []) + ([d.split("L=")[1] + " layers"] if "L=" in d else [])
    return kind + ", " + ", ".join(parts)
comb_all, eq_marks = [], []
for di, d in enumerate(depths):
    comb, twin = pick("CoMB", d), pick("dephased twin", d)
    comb_all += comb + twin
    mk, ms = MK.get(d, ("o", 7))
    for k, (x, (m, e)) in enumerate(comb):
        xo = x * (0.975 - 0.03 * di)
        ax.errorbar([xo], [m], yerr=[e], fmt=mk, color="tab:blue", ms=ms, capsize=3, label=lab("CoMB", d) if k == 0 else None, zorder=5)
        key = (f"{d} " if d else "") + f"chi={x}"
        eq = eqs.get(key) if isinstance(eqs, dict) else None
        if eq is None and isinstance(eqs, dict) and d is None: eq = eqs.get(str(x))
        if eq:
            ax.plot([xo, eq], [m, m], color="tab:blue", ls=":", lw=1.1)
            eq_marks.append((eq, m))
    for k, (x, (m, e)) in enumerate(twin):
        eb = ax.errorbar([x * (1.025 + 0.03 * di)], [m], yerr=[e], fmt=MT.get(d, "D"), mfc="none", color="tab:blue", ms=6, capsize=3,
                         label=lab("dephased twin", d) if k == 0 else None)
        eb[-1][0].set_linestyle("--")
xs = sorted(set(hx.tolist()) | {x for x, _ in comb_all})
ax.set_xscale("log")
ax.set_xticks(xs)
ax.set_xticklabels([str(x) for x in xs])
ax.minorticks_off()
allm = np.concatenate([hm, [v[0] for _, v in comb_all]])
ymin = allm.min() * 0.85
ax.set_xlim(min(xs) * 0.85, max(xs) * 1.15)
ax.set_ylim(ymin, allm.max() * 1.08)
for i, (eq, m) in enumerate(sorted(eq_marks)):          # equivalent classical memory: dotted drop to a label, staggered in height
    yl = ymin + (0.015 + 0.075 * (i % 2)) * (allm.max() - ymin)
    ax.plot([eq, eq], [yl + 0.055 * (allm.max() - ymin), m], color="0.45", ls=":", lw=1.1)
    ax.text(eq, yl, f"{eq:.1f}", ha="center", va="bottom", fontsize=10, color="0.25")
ax.set_xlabel(r"memory size $\chi$")
ax.set_ylabel(r"test MMD$^2$ ($10^{-4}$)")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=11, frameon=False)
out = ROOT / "outputs/figs/ds1_memory_compression.pdf"
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out, bbox_inches="tight")
plt.close(fig)
print(f"written: {out}")
