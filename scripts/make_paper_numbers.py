#!/usr/bin/env python3
"""Fill every result of the CoMB paper from the run outputs (digits from comb2, Dataset 1 from its metrics file).

    python scripts/make_paper_numbers.py --comb2 /path/to/digits/folder          # digits (UCI and MNIST)
    python scripts/make_paper_numbers.py --comb2 . --ds1 outputs/samples/ds1_comb_chi4_s0.metrics.json

--comb2 is the folder that contains scripts/comb_strokes.py, data/ and outputs/strokes/ (default: the current folder).
--ds1 is the Dataset 1 metrics file written by  python scripts/comb_ds1.py sample ; a folder is searched for it, and without
--ds1 the script looks in <comb2>/outputs/samples/ by itself.

Reads comb2/outputs/strokes (metrics, training records, MEMORY_*.json, DEPLOY_*.json, CLASSES_*.json) and
comb2/data (sequence lengths), for every dataset that has been run. Writes into outputs/paper/:
  numbers.tex                                   all macros used in the text
  decisive_*.tex circuits_*.tex deploy_*.tex classes.tex   table rows
The figures are drawn separately by comb_plots/make_all.py.
Anything not run stays a red "?" in the PDF. The sentence on whether every coherent run beats every other
run is written from the data either way, and a warning is printed if it does not hold. Rerun after new runs.
"""
import argparse, json, pathlib, shutil, subprocess, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parents[1] / "outputs" / "paper"
PEND = r"\pending"


def fmt_pm(x, nd=3):
    x = np.asarray(x, dtype=float)
    return f"{x.mean():.{nd}f}" if len(x) == 1 else f"{x.mean():.{nd}f}\\pm{x.std(ddof=1):.{nd}f}"


def parts(m):
    """held-out objective of one run: (total, local k-point part, global path part)"""
    sp = m["spec"]
    k = m["kmmd"]["model"]["loss"]
    p = sp.get("path", 0) * m["path_mmd"]["model"] / sp["path_reference"] if sp.get("path", 0) > 0 and m.get("path_mmd") else 0.0
    return k + p, k, p


def load(out, name):
    m, r = out / f"{name}.metrics.json", out / f"{name}.json"
    if not (m.exists() and r.exists()):
        return None
    return json.load(open(m)), json.load(open(r))


def dataset(out, data_dir, data, P, macros, frags, figs, src_figs):
    prefix = f"strokes_{data}_"
    seeds = sorted({int(p.name.split("_s")[-1].split(".")[0]) for p in out.glob(f"{prefix}hqmm_chi4_e4_path2_s*.metrics.json")})
    groups = {"coh": "hqmm_chi4_e4_path2", "deph": "hqmm_dephased_chi4_e4_path2", "hmm": "hmm_chi4_path2"}
    runs = {g: [x for x in (load(out, f"{prefix}{tag}_s{s}") for s in range(10)) if x] for g, tag in groups.items()}
    pure = load(out, f"{prefix}hqmm_chi4_e0_path2_s0")
    labels = {"coh": "coherent memory, 16 env.\\ levels", "deph": "same memory, dephased", "hmm": "classical HMM"}
    rows, res_rows = [], []                                         # res_rows: the combined results table (memories, then circuits)
    for g in ("coh", "deph", "hmm"):
        if runs[g]:
            v = [r["best_val"] for _, r in runs[g]]
            PT = np.array([parts(m) for m, _ in runs[g]])
            a = [m["auc"] for m, _ in runs[g]]
            rows.append(f"{labels[g]} & {len(v)} & ${fmt_pm(v)}$ & ${fmt_pm(PT[:, 0])}$ & ${fmt_pm(PT[:, 1])}$ & ${fmt_pm(PT[:, 2])}$ & {np.mean(a):.3f}\\\\")
            res_rows.append(f"{labels[g]} & ${fmt_pm(PT[:, 0])}$ & ${fmt_pm(PT[:, 1])}$ & ${fmt_pm(PT[:, 2])}$ & {np.mean(a):.3f}\\\\")
    if pure:
        m, r = pure
        t_, k_, p_ = parts(m)
        rows.append(f"coherent memory, no env.\\ levels & 1 & ${r['best_val']:.3f}$ & ${t_:.3f}$ & ${k_:.3f}$ & ${p_:.3f}$ & {m['auc']:.3f}\\\\")
        res_rows.append(f"coherent memory, no env.\\ levels & ${t_:.3f}$ & ${k_:.3f}$ & ${p_:.3f}$ & {m['auc']:.3f}\\\\")
    if rows:
        frags[f"decisive_{data}"] = "\n".join(rows)
    if runs["coh"] and (runs["deph"] or runs["hmm"]):
        kc = [parts(m)[0] for m, _ in runs["coh"]]                          # the full held-out objective
        ko = [parts(m)[0] for g in ("deph", "hmm") for m, _ in runs[g]]
        holds = max(kc) < min(ko)
        macros[f"{P}BeatSentence"] = ("every coherent run beats every dephased and every classical run on held-out data" if holds
                                     else "the coherent runs do not beat every dephased and classical run on held-out data")
        if not holds:
            print(f"  WARNING ({data}): not every coherent run beats every other run; the text says so")
        macros[f"{P}Seeds"] = str(min(len(runs[g]) for g in groups if runs[g]))
        macros[f"{P}CohWorst"] = f"{max(kc):.3f}"
        macros[f"{P}OtherBest"] = f"{min(ko):.3f}"
        for g, key in (("deph", "GainDeph"), ("hmm", "GainHmm")):
            if runs[g]:
                kg = np.mean([parts(m)[0] for m, _ in runs[g]])
                macros[f"{P}{key}"] = f"{100 * (kg - np.mean(kc)) / np.mean(kc):.0f}\\%"
    hm = {c: [parts(x[0])[0] for x in (load(out, f"{prefix}hmm_chi{c}_path2_s{s_}") for s_ in range(10)) if x] for c in (4, 6, 8, 12, 16)}
    hm = {c: v for c, v in hm.items() if v}
    if len(hm) >= 2 and runs["coh"]:                                  # equivalent classical memory on the held-out objective
        cs_ = sorted(hm)
        cv = [np.mean(hm[c]) for c in cs_]
        target = np.mean([parts(m)[0] for m, _ in runs["coh"]])
        eq = None
        for i in range(1, len(cs_)):
            if (cv[i - 1] - target) * (cv[i] - target) <= 0 and cv[i] != cv[i - 1]:
                eq = float(np.exp(np.log(cs_[i - 1]) + (target - cv[i - 1]) / (cv[i] - cv[i - 1]) * (np.log(cs_[i]) - np.log(cs_[i - 1]))))
                break
        macros[f"{P}ChiEquiv"] = f"{eq:.1f}" if eq else (f"more than {cs_[-1]}" if target < min(cv) else f"fewer than {cs_[0]}")
    ef = out / f"ERRORS_{data}.json"                                  # paired statistical errors (jackknife over the test digits)
    if ef.exists():
        E = json.load(open(ef))
        R = E["runs"]
        se = lambda v: float(np.sqrt((len(v) - 1) / len(v) * ((np.asarray(v) - np.mean(v)) ** 2).sum()))
        jk = lambda tags: np.mean([R[prefix + t_]["total"] for t_ in tags], axis=0) if all(prefix + t_ in R for t_ in tags) else None
        val = lambda tags: np.mean([parts(load(out, prefix + t_)[0])[0] for t_ in tags])
        sd = {g: [f"{tag}_s{s_}" for s_ in range(10) if (prefix + f"{tag}_s{s_}") in R] for g, tag in groups.items()}
        if sd["coh"] and jk(sd["coh"]) is not None:
            jc = jk(sd["coh"])
            vc = val(sd["coh"])
            macros[f"{P}CohMean"] = f"{vc:.3f}"
            macros[f"{P}CommonSE"] = f"{se(jc):.2f}"
            for g, key in (("deph", "DiffDeph"), ("hmm", "DiffHmm")):
                if sd[g]:
                    macros[f"{P}{key}"] = f"{val(sd[g]) - vc:.3f}\\pm{se(jk(sd[g]) - jc):.3f}"
        for a_, b_, key in (("comb_chi4_path2_s0", "dephased_chi4_path2_s0", "DiffComb"), ("tree_chi4_L4_path2_s0", "tree_dephased_chi4_L4_path2_s0", "DiffTreeFour"),
                            ("tree_chi4_L6_path2_s0", "tree_dephased_chi4_L6_path2_s0", "DiffTreeSix")):
            if prefix + a_ in R and prefix + b_ in R:
                d_ = val([a_]) - val([b_])
                macros[f"{P}{key}"] = f"{d_:+.3f}\\pm{se(np.array(R[prefix + a_]['total']) - np.array(R[prefix + b_]['total'])):.3f}"
    mem = out / f"MEMORY_{data}.json"
    if False and mem.exists():
        j = json.load(open(mem))
        e = j.get("chi_equiv_val")
        macros[f"{P}ChiEquiv"] = f"{e:.1f}" if e is not None else (f"more than {max(j['chis'])}" if j["coherent_val"] < min(j["hmm_val"]) else f"fewer than {min(j['chis'])}")
    circ = [("comb_chi4_path2_s0", "CoMB circuit, 4 layers", 6), ("dephased_chi4_path2_s0", "\\quad its dephased twin", 6),
            ("tree_chi4_L4_path2_s0", "tree-CoMB, node depth 4", 3), ("tree_dephased_chi4_L4_path2_s0", "\\quad its dephased twin", 3),
            ("tree_chi4_L6_path2_s0", "tree-CoMB, node depth 6", 3), ("tree_dephased_chi4_L6_path2_s0", "\\quad its dephased twin", 3)]
    crows, res_c = [], []
    for tag, lab, q in circ:
        x = load(out, prefix + tag)
        if x:
            t_, k_, p_ = parts(x[0])
            crows.append(f"{lab} & {q} & ${t_:.3f}$ & ${k_:.3f}$ & ${p_:.3f}$ & {x[0]['auc']:.3f}\\\\")
            res_c.append(f"{lab} & ${t_:.3f}$ & ${k_:.3f}$ & ${p_:.3f}$ & {x[0]['auc']:.3f}\\\\")
    if crows:
        frags[f"circuits_{data}"] = "\n".join(crows)
    if res_rows or res_c:                                              # one table: memories (mean +- sd over seeds), then circuits (seed 0)
        frags[f"results_{data}"] = "\n".join((["\\multicolumn{5}{l}{\\emph{memories, $\\chi=4$}}\\\\"] + res_rows if res_rows else [])
                                              + (["\\multicolumn{5}{l}{\\emph{circuits of fixed depth}}\\\\"] + res_c if res_c else []))
    dep = out / f"DEPLOY_{data}.json"
    if dep.exists():
        j = json.load(open(dep))
        drows = []
        for tag, lab, _ in circ:
            r = j["runs"].get(prefix + tag)
            if r:
                res = r["results"]
                ion = next(k for k in res if k.startswith("trapped"))
                drows.append(f"{lab} & {res['noiseless']['total']:.3f} & {res[ion]['total']:.3f} & {res['SC x1']['total']:.3f} & {res['SC x4']['total']:.3f}\\\\")
        if drows:
            frags[f"deploy_{data}"] = "\n".join(drows)
        t = j.get("thresholds", {}).get(prefix + "tree_chi4_L4_path2_s0")
        if t:
            macros[f"{P}LambdaStar"] = "more than 8" if t["lambda_star"] is None else f"{t['lambda_star']:.2f}"
    cls = out / f"CLASSES_{data}_tree_chi4_L4_path2_s0.json"
    if cls.exists():
        j = json.load(open(cls))
        macros[f"{P}DigitsPass"] = str(sum(r["passed"] for r in j["digits"].values()))
        if "mixture" in j:
            mx = j["mixture"]
            macros[f"{P}MixAUC"] = f"{mx['auc']:.3f}"
            macros[f"{P}AUCfloor"] = f"{mx['auc_floor']:.3f}"
            macros[f"{P}Ceil"] = f"{100 * mx['ceiling']:.1f}\\%"
            macros[f"{P}MixIntended"] = f"{100 * mx['intended']:.0f}\\%"
        figs[data] = {int(k): (r["intended"], r["ceiling"]) for k, r in j["digits"].items()}
        src = src_figs / f"strokes_{data}_tree_chi4_L4_path2_s0_classes.pdf"
        if src.exists():
            shutil.copy(src, HERE / "figures" / f"classes_{data}.pdf")
    npz = data_dir / f"strokes_{data}.npz"
    if npz.exists():
        z = np.load(npz, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        macros[f"{P}d"] = str(meta["d"])
        macros[f"{P}N"] = f"{sum(len(z[f'labels_{s}']) for s in ('train', 'val', 'test')):,}".replace(",", "\\,")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comb2", default=".", help="folder with the digits runs: contains comb_strokes.py, data/ and outputs/strokes/ (default: .)")
    ap.add_argument("--ds1", help="Dataset 1 metrics file (outputs/samples/ds1_comb_chi4_s0.metrics.json), or a folder to look in")
    a = ap.parse_args()
    root = pathlib.Path(a.comb2).expanduser().absolute()               # as written: symbolic links are not followed
    if not (root / "outputs" / "strokes").is_dir():                  # forgive a path one level too deep or too high
        cand = next((c for c in (root.parent, root / "comb2", root.parent / "comb2") if (c / "outputs" / "strokes").is_dir()), None)
        if cand is None:
            raise SystemExit(f"--comb2 {a.comb2}: no outputs/strokes/ there. Give the folder that contains comb_strokes.py, data/ and outputs/strokes/.")
        print(f"  --comb2 {a.comb2}: using {cand} (the folder that contains outputs/strokes/)")
        root = cand
    ds1 = None
    if a.ds1:
        d = pathlib.Path(a.ds1).expanduser()
        if d.is_file():
            ds1 = d
        else:
            look = [d] if d.is_dir() else []
            look += [root / "outputs" / "samples", root.parent / "comb" / "outputs" / "samples"]
            hits = [f for L in look if L.is_dir() for f in sorted(L.rglob("ds1_comb*metrics.json"))]
            if hits:
                ds1 = hits[0]
                print(f"  --ds1 {a.ds1}: using {ds1}")
            else:
                print(f"  --ds1 {a.ds1}: no Dataset 1 metrics file (ds1_comb*.metrics.json) found there or in outputs/samples/;"
                      " it is written by  python scripts/comb_ds1.py sample . The Dataset 1 values stay pending.")
    else:
        auto = sorted((root / "outputs" / "samples").glob("ds1_comb*metrics.json")) if (root / "outputs" / "samples").is_dir() else []
        if auto:
            ds1 = auto[0]
            print(f"  Dataset 1 metrics found: {ds1}")
    out, data_dir, src_figs = root / "outputs/strokes", root / "data", root / "outputs/figs"
    (HERE / "figures").mkdir(parents=True, exist_ok=True)
    macros, frags, figs = {}, {}, {}
    for data, P in (("uci", "U"), ("mnist", "M")):
        dataset(out, data_dir, data, P, macros, frags, figs, src_figs)
    if figs:
        rows = []
        main = "mnist" if "mnist" in figs else "uci"                     # the per-digit table shows the main digits dataset
        for k in range(10):
            cell = lambda d: (f"{100 * figs[d][k][0]:.1f}\\% & {100 * figs[d][k][1]:.1f}\\%" if d in figs and k in figs[d] else f"{PEND} & {PEND}")
            rows.append(f"{k} & {cell(main)}\\\\")
        frags["classes"] = "\n".join(rows)
    if ds1:                        # the same values as paper_single/make_numbers.py, from `python scripts/comb_ds1.py sample`
        m = json.load(open(ds1))
        r = m["rows"]["model"]
        shape = m.get("shape_w1", {})
        macros.update(CoMBWdense=f"{r['W1_dense']:.3f}", CoMBWsparse=f"{r['W1_sparse']:.3f}", CoMBZeros=f"{100 * r['zeros']:.1f}",
                      CoMBSpearman=f"{r['spearman']:.3f}", CoMBKappa=f"{100 * m['kappa']:.0f}", CoMBEspread=f"{r['E_spread']:.2f}",
                      CoMBAUC=f"{m['auc']:.2f}", FloorAUC=f"{m['auc_floor']:.2f}")
        if shape:
            macros.update(CoMBShapeMax=f"{max(v[0] for v in shape.values()):.2f}", FloorShapeMax=f"{max(v[1] for v in shape.values()):.2f}")
    (HERE / "numbers.tex").write_text("% written by make_paper_numbers.py\n" + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())))
    for name, body in frags.items():
        (HERE / f"{name}.tex").write_text(body + "\n")
    print(f"  wrote numbers.tex ({len(macros)} values) and {len(frags)} table fragments: {', '.join(sorted(frags))}")
    print("  still pending: " + ", ".join(sorted(set(n for n in ("decisive_uci", "decisive_mnist", "circuits_uci", "circuits_mnist", "deploy_uci", "deploy_mnist", "classes")) - set(frags))) if len(frags) < 7 else "  all tables filled")


if __name__ == "__main__":
    main()
