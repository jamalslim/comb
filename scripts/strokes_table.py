#!/usr/bin/env python3
"""Collect every evaluated stroke run into one table (markdown and LaTeX), straight from the metrics files.

    python scripts/strokes_table.py --data uci      # writes outputs/strokes/RESULTS_<data>.md and .tex
"""
import argparse, json, pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "strokes"


def label(spec):
    name = {"comb": "CoMB", "dephased": "dephased twin", "hmm": "classical HMM", "hqmm": "general coherent memory", "hqmm_dephased": "general memory, dephased",
            "tree": "tree-CoMB", "tree_dephased": "tree-CoMB, dephased"}[spec["model"]]
    extra = []
    if spec["model"].startswith("tree"):
        extra.append(f"1 work qubit, node depth {spec.get('L_node', 4)}")
    if spec["model"].startswith("hqmm"):
        extra.append(f"any circuit, {2 ** spec.get('n_env', 0)} env. levels")
    elif spec.get("n_env"):
        extra.append(f"{spec['n_env']} env. qubits, L={spec['L_v']}")
    extra.append("k-point + path" if spec.get("path", 0) > 0 else "k-point")
    return f"{name}, chi={spec['chi']} ({'; '.join(extra)})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="uci")
    a = ap.parse_args()
    rows = []
    for f in sorted(OUT.glob(f"strokes_{a.data}_*.metrics.json")):
        m = json.load(open(f))
        run = json.load(open(str(f).replace(".metrics.json", ".json")))
        k, g, kap = m["kmmd"], m["geometry"], m["kappa"]
        rows.append(dict(label=label(m["spec"]), seed=m["spec"]["seed"], params=m["spec"].get("n_params", 0), val=run["best_val"],
                         kloss=k["model"]["loss"], knull=k["null"]["loss"], kfloor=k["floor"]["loss"],
                         w2=kap["window2"], w3=kap["window3"], pairs=kap["pairs"], path=kap.get("path", float("nan")),
                         height=g["height"]["model"], width=g["width"]["model"], closure=g["closure distance"]["model"],
                         winding=g["winding"]["model"], turning=g["total turning"]["model"],
                         floor_geo=max(v["floor"] for v in g.values()),
                         auc=m["auc"], auc_floor=m["auc_floor"], conf=m["digits"]["model"]["confidence"],
                         conf_test=m["digits"]["test"]["confidence"], H=m["digits"]["model"]["class_entropy_bits"],
                         H_test=m["digits"]["test"]["class_entropy_bits"], source=m["spec"]["source"]))
    if not rows:
        raise SystemExit("no evaluated runs")
    r0 = rows[0]
    md = [f"# Stroke-CoMB results: {r0['source']}", "",
          f"Held-out k-point loss: no dependence {r0['knull']:.3f}, train-vs-test floor {r0['kfloor']:.3f}. "
          f"Two-sample AUC floor {r0['auc_floor']:.3f}. Test digits: classifier confidence {r0['conf_test']:.2f}, class entropy {r0['H_test']:.2f} bits. "
          f"Geometry errors are W1/sd against held-out digits; the train-vs-test floor is at most {r0['floor_geo']:.2f} for every descriptor.", "",
          "| model | params | k-point loss | kappa 2 / 3 / pairs / path | height | width | closure | winding | turning | AUC | confidence | classes (bits) |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        md.append(f"| {r['label']}, seed {r['seed']} | {r['params']} | {r['kloss']:.3f} | {100*r['w2']:.0f}% / {100*r['w3']:.0f}% / {100*r['pairs']:.0f}% / {100*r['path']:.0f}% | "
                  f"{r['height']:.2f} | {r['width']:.2f} | {r['closure']:.2f} | {r['winding']:.2f} | {r['turning']:.2f} | {r['auc']:.3f} | {r['conf']:.2f} | {r['H']:.2f} |")
    # mean and spread over seeds for every configuration with more than one evaluated seed
    import numpy as np
    groups = {}
    for r in rows:
        groups.setdefault(r["label"], []).append(r)
    multi = {k: v for k, v in groups.items() if len(v) > 1}
    if multi:
        md += ["", "## Mean +- sd over seeds", "",
               "| model | seeds | validation loss | k-point loss | kappa 2 / 3 / pairs / path | height | turning | AUC |", "|---|---|---|---|---|---|---|---|"]
        f = lambda v, key, sc=1.0, nd=3: f"{sc * np.mean([x[key] for x in v]):.{nd}f} +- {sc * np.std([x[key] for x in v], ddof=1):.{nd}f}"
        for k, v in sorted(multi.items(), key=lambda kv: np.mean([x["val"] for x in kv[1]])):
            md.append(f"| {k} | {len(v)} | {f(v, 'val')} | {f(v, 'kloss')} | {f(v, 'w2', 100, 0)} / {f(v, 'w3', 100, 0)} / {f(v, 'pairs', 100, 0)} / {f(v, 'path', 100, 0)} | "
                      f"{f(v, 'height', 1, 2)} | {f(v, 'turning', 1, 2)} | {f(v, 'auc')} |")
    (OUT / f"RESULTS_{a.data}.md").write_text("\n".join(md) + "\n")
    tex = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
           r"model & params & k-point & $\kappa_2/\kappa_3/\kappa_{\rm pairs}/\kappa_{\rm path}$ & height & turning & AUC & conf. & classes \\", r"\midrule"]
    for r in rows:
        tex.append(f"{r['label'].replace('chi=', '$\\chi=$')} & {r['params']} & {r['kloss']:.3f} & {100*r['w2']:.0f}/{100*r['w3']:.0f}/{100*r['pairs']:.0f}/{100*r['path']:.0f}\\% & "
                   f"{r['height']:.2f} & {r['turning']:.2f} & {r['auc']:.3f} & {r['conf']:.2f} & {r['H']:.2f} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    (OUT / f"RESULTS_{a.data}.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
