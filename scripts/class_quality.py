#!/usr/bin/env python3
"""Does CoMB generate the dataset?  Quality gate for class-conditional CoMB (one model per digit).

    python scripts/class_quality.py --data mnist --tag tree_chi4_L4_path2 --seed 0

Collects the runs strokes_<data>_<tag>_digit<k>_s<seed>, k = 0..9 (trained with comb_strokes.py --digit k), and
reports, straight from samples of the trained models:

  * per digit: the fraction of samples that a classifier trained on real digits (round-trip renders) reads as the
    intended digit, next to the same fraction for real held-out digits of that class (the ceiling the stroke
    representation allows), and the mean classifier confidence;
  * the unconditional generator = the exact mixture "draw a label with the test-set class frequencies, then
    sample that digit's model": two-sample AUC against real held-out renders (0.5 = indistinguishable, floor =
    train vs test renders) and the entropy of the predicted classes (3.32 bits = all ten digits equally often);
  * a figure with one row per digit: real held-out digits on the left, generated digits on the right.

Criterion (stated, adjustable with --target): a digit counts as generated recognisably when the classifier reads
the intended digit in at least target x ceiling of its samples (default target 0.8).
Writes outputs/strokes/CLASSES_<data>_<tag>_s<seed>.md/.json and outputs/figs/strokes_<data>_<tag>_s<seed>_classes.pdf.
"""
import argparse, json, pathlib, pickle, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import comb_strokes as cs                                                    # noqa: E402
from comb.digits import strokes as st                                               # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="mnist")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--target", type=float, default=0.8)
    a = ap.parse_args()
    z, meta = cs.load_prepared(a.data)
    clf = pickle.load(open(cs.DATA / f"strokes_{a.data}_classifier.pkl", "rb"))
    rng = np.random.default_rng(7)
    freq = np.bincount(z["labels_test"], minlength=10) / len(z["labels_test"])
    rows, samples, real = {}, {}, {}
    for k in range(10):
        name = f"strokes_{a.data}_{a.tag}_digit{k}_s{a.seed}"
        if not (cs.OUT / f"{name}.npz").exists():
            print(f"  digit {k}: no trained run {name}")
            continue
        model, spec = cs.load_run(name)
        T = model.sample_counts(a.n, rng).astype(np.int8)
        samples[k] = T
        R = st.render_many(T).reshape(-1, 784)
        P = clf.predict_proba(R)
        Tr = z["tokens_test"][z["labels_test"] == k]
        real[k] = Tr
        Rr = st.render_many(Tr).reshape(-1, 784)
        acc, ceil = float((P.argmax(1) == k).mean()), float((clf.predict(Rr) == k).mean())
        rows[k] = dict(run=name, intended=acc, ceiling=ceil, ratio=acc / max(ceil, 1e-9), confidence=float(P.max(1).mean()),
                       passed=bool(acc >= a.target * ceil), n=len(T), n_real=len(Tr))
        print(f"  digit {k}: {100 * acc:5.1f}% read as {k} (real digits {100 * ceil:5.1f}%), confidence {rows[k]['confidence']:.2f}"
              f"  {'PASS' if rows[k]['passed'] else 'FAIL'}", flush=True)
    if not rows:
        raise SystemExit("no per-digit runs found; train them with comb_strokes.py train --digit k")
    out = dict(data=a.data, tag=a.tag, seed=a.seed, target=a.target, digits=rows)
    if len(rows) == 10:                                                       # the unconditional mixture
        n_k = np.round(freq * a.n).astype(int)
        Tm = np.concatenate([samples[k][:n_k[k]] for k in range(10)])
        Rm = st.render_many(Tm).reshape(-1, 784)
        Rt = st.render_many(z["tokens_test"][:len(Tm)]).reshape(-1, 784)
        Rtr = st.render_many(z["tokens_train"][:len(Tm)]).reshape(-1, 784)
        lab = clf.predict(Rm)
        f = np.bincount(lab, minlength=10) / len(lab)
        out["mixture"] = dict(auc=cs.two_sample_auc(Rm, Rt), auc_floor=cs.two_sample_auc(Rtr, Rt),
                              class_entropy_bits=float(-(f[f > 0] * np.log2(f[f > 0])).sum()),
                              intended=float(np.mean([rows[k]["intended"] for k in range(10)])),
                              ceiling=float(np.mean([rows[k]["ceiling"] for k in range(10)])))
    tag = f"{a.data}_{a.tag}_s{a.seed}"
    json.dump(out, open(cs.OUT / f"CLASSES_{tag}.json", "w"), indent=1)
    md = [f"# Does CoMB generate {a.data.upper()}?  ({a.tag}, seed {a.seed})", "",
          f"Classifier trained on real digits (round-trip renders). Criterion: intended-digit rate >= {a.target:g} x the rate for real digits.", "",
          "| digit | read as intended (generated) | read as intended (real) | ratio | confidence | verdict |", "|---|---|---|---|---|---|"]
    for k, r in sorted(rows.items()):
        md.append(f"| {k} | {100 * r['intended']:.1f}% | {100 * r['ceiling']:.1f}% | {r['ratio']:.2f} | {r['confidence']:.2f} | {'pass' if r['passed'] else 'fail'} |")
    if "mixture" in out:
        m = out["mixture"]
        md += ["", f"Unconditional mixture: {100 * m['intended']:.1f}% of samples read as the intended digit (real digits {100 * m['ceiling']:.1f}%); "
               f"two-sample AUC {m['auc']:.3f} (floor {m['auc_floor']:.3f}; 0.5 = indistinguishable); class entropy {m['class_entropy_bits']:.2f} bits (3.32 = balanced)."]
    md += ["", f"Digits passing: {sum(r['passed'] for r in rows.values())} of {len(rows)}."]
    (cs.OUT / f"CLASSES_{tag}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    nr, ng = 4, 10
    ks = sorted(rows)
    grid = np.ones((len(ks) * 29, (nr + ng) * 29 + 10))
    for i, k in enumerate(ks):
        for j, img in enumerate(st.render_many(real[k][:nr])):
            grid[i * 29:i * 29 + 28, j * 29:j * 29 + 28] = 1 - img
        for j, img in enumerate(st.render_many(samples[k][:ng])):
            x0 = nr * 29 + 10 + j * 29
            grid[i * 29:i * 29 + 28, x0:x0 + 28] = 1 - img
    fig, ax = plt.subplots(figsize=(7.2, 0.52 * len(ks) + 0.4))
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
    ax.axis("off")
    ax.set_title(f"left: real held-out digits | right: CoMB samples, one model per digit ({a.tag})", fontsize=8)
    figs = ROOT / "outputs" / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / f"strokes_{tag}_classes.pdf", bbox_inches="tight", dpi=200)
    fig.savefig(figs / f"strokes_{tag}_classes.png", bbox_inches="tight", dpi=110)


if __name__ == "__main__":
    main()
