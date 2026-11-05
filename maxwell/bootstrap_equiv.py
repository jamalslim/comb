"""
Error bars on the equivalent classical memory (the number of classical states a coherent memory is worth).

Reads maxwell/seed_runs.csv (written by summarize_seeds.py: one row per run with its group and held-out test MMD^2)
and resamples the seeds with replacement, independently for the coherent model and for every classical size of the
matching curve (same training budget and same number of levels). For each resample it recomputes the means and the
equivalent memory with exactly the rule of summarize_seeds.py: log-linear interpolation on the classical curve between
the two sizes that bracket the coherent model's mean. It reports the value of the original means, the median and the
68% and 95% intervals of the resamples, and how often the equivalent memory exceeds the coherent model's own size chi.

    python maxwell/bootstrap_equiv.py                      # every coherent group
    python maxwell/bootstrap_equiv.py --resamples 20000 --groups "CoMB conv chi=4" "CoMB nested L=18 conv chi=8"
"""
import argparse, csv, pathlib, re
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def kind_of(g):
    return re.sub(r" (E|K)=\d+", "", g.split(" chi=")[0].split(" L=")[0].replace(" nested", "").replace(" dense", "").replace(" warm", "").replace(" conv", ""))


def ktag_of(g):
    m = re.search(r" K=\d+", g)
    return m.group(0) if m else ""


def budget_of(g):
    if g.startswith("EnvMem"): return " conv"
    return " conv" if " conv" in g else (" warm" if " warm" in g else "")


def chi_of(g):
    return int(g.split(" chi=")[1])


def equiv(cm, curve):
    """The summary's rule: log-linear interpolation between the bracketing classical sizes; inf above, 0 below."""
    xs = sorted(curve)
    for x0, x1 in zip(xs[:-1], xs[1:]):
        if (curve[x0] - cm) * (curve[x1] - cm) <= 0:
            f = (np.log(curve[x0]) - np.log(cm)) / (np.log(curve[x0]) - np.log(curve[x1]))
            return float(np.exp(np.log(x0) + f * (np.log(x1) - np.log(x0))))
    return float("inf") if cm < min(curve.values()) else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--resamples", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--groups", nargs="*", default=None, help="coherent groups to analyze (default: all CoMB and EnvMem groups)")
    ap.add_argument("--csv", default=str(ROOT / "maxwell/seed_runs.csv"))
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    runs = {}
    for r in csv.DictReader(open(a.csv)):
        try: runs.setdefault(r["group"], []).append(float(r["test"]))
        except (ValueError, KeyError): pass
    runs = {g: np.array(v) for g, v in runs.items()}
    curves = {}
    for g, v in runs.items():
        if kind_of(g) == "HMM": curves.setdefault(budget_of(g) + ktag_of(g), {})[chi_of(g)] = v
    targets = a.groups or [g for g in runs if kind_of(g) in ("CoMB", "EnvMem")]
    print(f"equivalent classical memory with bootstrap intervals ({a.resamples} resamples of the seeds)\n")
    print(f"  {'group':34s} {'chi':>3s} {'n':>3s} {'estimate':>8s} {'median':>7s} {'68% interval':>15s} {'95% interval':>15s}  P(equiv > chi)")
    for g in targets:
        if g not in runs: print(f"  {g}: no runs"); continue
        curve = curves.get(budget_of(g) + ktag_of(g))
        if not curve or len(curve) < 2: print(f"  {g}: no classical curve with the same budget and levels"); continue
        chi, x = chi_of(g), runs[g]
        est = equiv(x.mean(), {k: v.mean() for k, v in curve.items()})
        boot = np.empty(a.resamples)
        for b in range(a.resamples):
            cm = rng.choice(x, x.size).mean()
            boot[b] = equiv(cm, {k: rng.choice(v, v.size).mean() for k, v in curve.items()})
        q = np.quantile(boot, [0.025, 0.16, 0.5, 0.84, 0.975])
        fmt = lambda v: "inf" if np.isinf(v) else f"{v:.1f}"
        print(f"  {g:34s} {chi:3d} {x.size:3d} {fmt(est):>8s} {fmt(q[2]):>7s} {'[' + fmt(q[1]) + ', ' + fmt(q[3]) + ']':>15s} {'[' + fmt(q[0]) + ', ' + fmt(q[4]) + ']':>15s}  {np.mean(boot > chi):.3f}")
    print("\nP(equiv > chi) is the fraction of resamples in which the coherent model is worth more classical states than its own size.")


if __name__ == "__main__":
    main()
