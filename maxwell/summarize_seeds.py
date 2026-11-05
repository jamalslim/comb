#!/usr/bin/env python3
"""Summarize the seed runs trained with the layer-resolved loss.

    python maxwell/summarize_seeds.py

For every run: the best validation MMD^2 (which also chose the run's best step), the MMD^2 on the 2000 held-out test
showers (used by nothing else: the number to quote), and the Table 1 metrics. Per model and memory size chi: mean +-
standard error over seeds. For every CoMB size: the comparison with the HMM and the dephased twin of the same size, and
the equivalent classical memory (the chi at which the HMMs' mean test MMD^2 crosses CoMB's, interpolated in log MMD^2
against log chi). Writes maxwell/seed_summary.json (read by maxwell/plot_memory_curve.py).
"""
import glob, json, pathlib, re, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import comb_ds1 as cd                                                             # noqa: E402
from comb.showers import zoo                                                              # noqa: E402
from comb.showers.mmd import Spectrum, median_sigma, cf_sum_and_sq                        # noqa: E402

D = cd.load_data()
m0, s0 = D["Y_tr"].mean(0), D["Y_tr"].std(0)
Zf, Zt = (D["Y_fit"] - m0) / s0, (D["Y_test"] - m0) / s0
sig = median_sigma(Zf)
ks = Spectrum(Zf.shape[1], [sig / 2, sig, 2 * sig])        # the kernel of the training loss
om = ks.sample(4096, np.random.default_rng(20_000))
ph_t, sq_t = cf_sum_and_sq(Zt, om)


def test_mmd(path):
    """Held-out MMD^2 of a trained run, computed exactly from its characteristic function at the fixed test
    frequencies.
    """
    q = zoo.load(path)[0].characteristic_function(om)
    return float(np.mean(np.abs(q) ** 2 - 2.0 * np.real(q * ph_t.conj()) + sq_t))


def group_of(name):
    """Group of a run name, e.g. 'CoMB chi=4' or 'HMM conv chi=8': the set of runs whose seeds are averaged
    together.
    """
    m = re.match(r"ds1_(envmem|envdeph)_chi(\d+)_E(\d+)_s\d+(?:_K(\d+))?(_warm|_conv)?_ly$", name)
    if m: return ("EnvMem dephased" if m.group(1) == "envdeph" else "EnvMem") + f" E={m.group(3)}" + (f" K={m.group(4)}" if m.group(4) else "") + {"_warm": " warm", "_conv": " conv"}.get(m.group(5), "") + f" chi={m.group(2)}"
    m = re.match(r"ds1_(comb|dephased)_chi(\d+)_s\d+(_dense)?(_nest)?(?:_L(\d+))?(?:_K(\d+))?(_warm|_conv)?_ly$", name)
    if m: return ("CoMB" if m.group(1) == "comb" else "dephased twin") + (" dense" if m.group(3) else "") + (" nested" if m.group(4) else "") + (f" L={m.group(5)}" if m.group(5) else "") + (f" K={m.group(6)}" if m.group(6) else "") + ({"_warm": " warm", "_conv": " conv"}.get(m.group(7), "")) + f" chi={m.group(2)}"
    m = re.match(r"ds1_hmm(\d+)_s\d+(?:_K(\d+))?(_warm|_conv)?_ly$", name)
    return (f"HMM{(' K=' + m.group(2)) if m.group(2) else ''}{ {'_warm': ' warm', '_conv': ' conv'}.get(m.group(3), '') } chi={m.group(1)}") if m else None


import os, time
def cached_test_mmd(name, npz):
    """The held-out MMD^2 of a run, computed once and kept in outputs/runs/<run>.test.json; recomputed if the run changed."""
    cache = ROOT / "outputs/runs" / f"{name}.test.json"
    stamp = os.path.getmtime(npz)
    if cache.exists():
        try:
            c = json.load(open(cache))
            if c.get("mtime") == stamp and c.get("n_freq") == len(om): return float(c["test"]), True
        except Exception:
            pass
    v = test_mmd(npz)
    json.dump(dict(test=v, mtime=stamp, n_freq=len(om)), open(cache, "w"))
    return v, False

todo = [j for j in sorted(glob.glob(str(ROOT / "outputs/runs/ds1_*_ly.json")))
        if group_of(pathlib.Path(j).stem) is not None and pathlib.Path(j[:-5] + ".npz").exists()]
print(f"scoring {len(todo)} runs on the held-out showers (cached runs are instant)", flush=True)
WORKERS = int(os.environ.get("SUMMARY_WORKERS", "1"))
if WORKERS > 1:                                   # score the runs not yet cached in parallel, one run per worker
    import multiprocessing as mp
    def _score(j):
        name = pathlib.Path(j).stem
        v, was = cached_test_mmd(name, j[:-5] + ".npz")
        return name, v, was
    t0p = time.time()
    done = 0
    with mp.get_context("fork").Pool(WORKERS) as pool:
        for name, v, was in pool.imap_unordered(_score, todo):
            done += 1
            if not was: print(f"  scored {done}/{len(todo)}  {name:36s} test {1e4 * v:.3f}  ({(time.time() - t0p) / 60:.1f} min so far)", flush=True)
    print(f"all {len(todo)} runs scored in {(time.time() - t0p) / 60:.1f} min with {WORKERS} workers", flush=True)
def ktag_of(g):
    """The ' K=<levels>' part of a group name, empty for the default 8 levels; runs are compared only with runs of
    the same number of levels.
    """
    m = re.search(r" K=\d+", g)
    return m.group(0) if m else ""          # 16-level runs are compared only with 16-level runs
def budget_of(g):
    """Training budget of a group: '' for 3,000 steps, ' warm' for 6,000 and ' conv' for convergence. Environment
    memories start from converged HMMs and count as converged.
    """
    if g.startswith("EnvMem"): return " conv"          # started from the converged HMM: compared with converged models
    return " conv" if " conv" in g else (" warm" if " warm" in g else "")
rows = {}
t0 = time.time()
for k, j in enumerate(todo, 1):
    name = pathlib.Path(j).stem
    g = group_of(name)
    npz = j[:-5] + ".npz"
    t1 = time.time()
    test, was_cached = cached_test_mmd(name, npz)
    print(f"  [{k}/{len(todo)}] {name:36s} test {1e4 * test:.3f}  {'cached' if was_cached else f'{time.time() - t1:.0f} s'}  ({(time.time() - t0) / 60:.1f} min so far)", flush=True)
    r = dict(name=name, val=float(json.load(open(j))["best_val"]), test=test)
    mf = ROOT / "outputs/samples" / f"{name}.metrics.json"
    if mf.exists():
        m = json.load(open(mf))["rows"]
        mo, nu, fl = m["model"], m["null"], m["floor"]
        r.update(kappa=(nu["spearman"] - mo["spearman"]) / (nu["spearman"] - fl["spearman"]), spearman=mo["spearman"],
                 W1_dense=mo["W1_dense"], W1_sparse=mo["W1_sparse"], zeros=mo["zeros"], E_spread=mo["E_spread"])
    rows.setdefault(g, []).append(r)
chi_of = lambda g: int(g.split(" chi=")[1])
kind_of = lambda g: re.sub(r" (E|K)=\d+", "", (lambda g: g.split(" chi=")[0].split(" L=")[0].replace(" nested", "").replace(" dense", "").replace(" warm", "").replace(" conv", ""))(g))
order = sorted(rows, key=lambda g: ({"CoMB": 0, "EnvMem": 1, "dephased twin": 2, "EnvMem dephased": 3, "HMM": 4}[kind_of(g)], chi_of(g)))

def ms(v):
    """Mean and standard error of the mean (NaN error for a single value)."""
    v = np.asarray(v, float)
    return float(v.mean()), (float(v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan"))

summary = {}
print("mean +- standard error over seeds (MMD^2 in units of 1e-4; test = 2000 held-out showers):")
for g in order:
    rs = rows[g]
    s = {"n": len(rs)}
    for key in ("val", "test", "kappa", "spearman", "W1_dense", "W1_sparse", "zeros", "E_spread"):
        vals = [r[key] for r in rs if key in r]
        if vals: s[key] = ms(vals)
    summary[g] = s
    line = f"  {g:20s} n={s['n']:3d}  val {1e4 * s['val'][0]:.3f} +- {1e4 * s['val'][1]:.3f}  test {1e4 * s['test'][0]:.3f} +- {1e4 * s['test'][1]:.3f}"
    if "kappa" in s:
        line += f"  kappa {100 * s['kappa'][0]:.1f}%  Spearman {s['spearman'][0]:.4f}  W1 {s['W1_dense'][0]:.3f}/{s['W1_sparse'][0]:.3f}  zeros {100 * s['zeros'][0]:.1f}%  E spread {s['E_spread'][0]:.2f}"
    print(line)

curves = {}                                   # classical curve per training budget: 3000 steps (""), 6000 (" warm"), converged (" conv")
for g in rows:
    if kind_of(g) == "HMM": curves.setdefault(budget_of(g) + ktag_of(g), {})[chi_of(g)] = float(np.mean([r["test"] for r in rows[g]]))
for b, c in sorted(curves.items()):
    label = (re.sub(r" K=\d+", "", b).strip() or "3000 steps") + ktag_of(b)
    print(f"\nclassical curve with budget '{label}': HMMs with {sorted(c)} states")
    pass
summary["equivalent classical memory"] = {}
for g in [g for g in order if kind_of(g) in ("CoMB", "EnvMem")]:
    chi = chi_of(g)
    c = np.array([r["test"] for r in rows[g]])
    print(f"\n{g} against the other models of the same size, on the held-out test MMD^2:")
    depth = g[len("CoMB"):g.index(" chi=")]
    others = (f"HMM{ktag_of(g)}{budget_of(g)} chi={chi}", f"dephased twin{depth} chi={chi}")       # same training budget and levels only
    if g.startswith("EnvMem"): others = (f"HMM{ktag_of(g)} conv chi={chi}", g.replace("EnvMem", "EnvMem dephased", 1))
    for other in others:
        if other not in rows: continue
        o = np.array([r["test"] for r in rows[other]])
        d = o.mean() - c.mean()
        se = np.sqrt((c.var(ddof=1) / len(c) if len(c) > 1 else 0) + (o.var(ddof=1) / len(o) if len(o) > 1 else 0))
        z = d / se if se > 0 else float("nan")
        print(f"  {other:20s} {other} minus CoMB: {1e4 * d:+.3f}e-4 ({100 * d / o.mean():+.0f}% of {other}; positive: CoMB better), {z:+.1f} standard errors;"
              f" {int(np.sum(c < o.min()))} of {len(c)} CoMB runs beat every {other} run")
        summary[f"{g} vs {other}"] = dict(difference=float(d), se=float(se), z=float(z))
    cm, eq = c.mean(), None
    hmean = curves.get(budget_of(g) + ktag_of(g), {})
    hchis = sorted(hmean)          # the classical curve with the same training budget and levels
    for x0, x1 in zip(hchis[:-1], hchis[1:]):
        if (hmean[x0] - cm) * (hmean[x1] - cm) <= 0:
            f = (np.log(hmean[x0]) - np.log(cm)) / (np.log(hmean[x0]) - np.log(hmean[x1]))
            eq = float(np.exp(np.log(x0) + f * (np.log(x1) - np.log(x0))))
            break
    if eq is not None: print(f"  equivalent classical memory: CoMB with chi = {chi} matches an HMM with about {eq:.1f} states")
    elif hchis and cm < min(hmean.values()): print(f"  equivalent classical memory: above {max(hchis)} states (CoMB with chi = {chi} beats every HMM size run)")
    elif hchis: print(f"  equivalent classical memory: below {min(hchis)} states")
    summary["equivalent classical memory"][g.replace("CoMB", "").strip()] = eq
json.dump(summary, open(ROOT / "maxwell/seed_summary.json", "w"), indent=1, default=float)
import csv                                                                     # every run, for maxwell/plot_results.py
cols = ["name", "group", "val", "test", "kappa", "spearman", "W1_dense", "W1_sparse", "zeros", "E_spread"]
with open(ROOT / "maxwell/seed_runs.csv", "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(cols)
    for g in order:
        for r in rows[g]:
            w.writerow([g if c == "group" else r.get(c, "") for c in cols])
print("written: maxwell/seed_runs.csv (every run)")
print("\nwritten: maxwell/seed_summary.json")
