#!/usr/bin/env python3
"""Equivalent classical memory: how many classical HMM states match the 4-level coherent memory.

    python scripts/memory_equiv.py --data uci

Reads every evaluated classical HMM trained with the combined loss (strokes_<data>_hmm_chi<c>_path2_s<seed>) and
the general coherent memory with chi = 4 (strokes_<data>_hqmm_chi4_e4_path2_s<seed>). For the validation loss and
for the held-out k-point loss separately, it interpolates the HMM curve linearly in log(chi) and reports the
classical memory size at which the HMM matches the coherent memory's mean. A result "chi_equiv = 8" reads: a
4-level coherent memory does what an 8-state classical memory does, on this data and this loss. Writes
outputs/strokes/MEMORY_<data>.md and .json.
"""
import argparse, json, pathlib, re, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT / "outputs" / "strokes"


def values(pattern):
    out = {}
    for f in sorted(OUT.glob("*.metrics.json")):
        m = re.fullmatch(pattern, f.name[:-len(".metrics.json")])
        if not m:
            continue
        run = json.load(open(str(f).replace(".metrics.json", ".json")))
        met = json.load(open(f))
        out.setdefault(int(m.group(1)) if m.groups() else 0, []).append((run["best_val"], met["kmmd"]["model"]["loss"]))
    return out


def crossing(chis, curve, target):
    """chi at which the (decreasing) curve reaches target, linear in log chi; None outside the measured range."""
    for i in range(1, len(chis)):
        a, b = curve[i - 1], curve[i]
        if (a - target) * (b - target) <= 0 and a != b:
            x = np.log(chis[i - 1]) + (target - a) / (b - a) * (np.log(chis[i]) - np.log(chis[i - 1]))
            return float(np.exp(x))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="uci")
    a = ap.parse_args()
    hmm = values(rf"strokes_{a.data}_hmm_chi(\d+)_path2_s\d+")
    coh = values(rf"strokes_{a.data}_hqmm_chi4_e4_path2_s\d+").get(0, [])
    if len(hmm) < 2 or not coh:
        raise SystemExit("need classical HMM runs at two or more memory sizes and the coherent memory (stages decisive and memory)")
    chis = sorted(hmm)
    v = np.array([np.mean([x[0] for x in hmm[c]]) for c in chis])
    k = np.array([np.mean([x[1] for x in hmm[c]]) for c in chis])
    tv, tk = np.mean([x[0] for x in coh]), np.mean([x[1] for x in coh])
    ev, ek = crossing(chis, v, tv), crossing(chis, k, tk)
    def say(e, curve, t):
        if e is not None:
            return f"{e:.1f} classical states"
        return f"more than {chis[-1]} (better than every classical size measured)" if t < curve.min() else f"fewer than {chis[0]}"
    md = [f"# Equivalent classical memory ({a.data})", "",
          f"General coherent memory, chi = 4 ({len(coh)} seeds): validation {tv:.3f}, held-out k-point loss {tk:.3f}.", "",
          "| classical HMM chi | seeds | validation loss | held-out k-point loss |", "|---|---|---|---|"]
    md += [f"| {c} | {len(hmm[c])} | {v[i]:.3f} | {k[i]:.3f} |" for i, c in enumerate(chis)]
    md += ["", f"Equivalent classical memory of the 4-level coherent memory: {say(ev, v, tv)} on the validation loss, "
           f"{say(ek, k, tk)} on the held-out k-point loss (interpolated linearly in log chi)."]
    (OUT / f"MEMORY_{a.data}.md").write_text("\n".join(md) + "\n")
    json.dump(dict(chis=chis, hmm_val=v.tolist(), hmm_kpoint=k.tolist(), coherent_val=tv, coherent_kpoint=tk, chi_equiv_val=ev, chi_equiv_kpoint=ek),
              open(OUT / f"MEMORY_{a.data}.json", "w"), indent=1)
    print("\n".join(md))


if __name__ == "__main__":
    main()
