#!/usr/bin/env python3
"""Re-place the readout bins of a trained Dataset 1 model at the data quantiles.

    python scripts/recalibrate_readout.py outputs/runs/ds1_comb_chi4_s0.npz
    -> writes outputs/runs/ds1_comb_chi4_s0_recal.npz

Why: each coordinate's marginal is set by the readout alone, and all dependence (the copula) by the memory.
Training changes the count probabilities, but the MMD kernel is as wide as the distance between whole showers
(about 27 standardized units), so it hardly sees structure inside one coordinate: the bin widths are not
re-matched. Narrow bins that now carry a lot of probability become spikes, and bins that lost their probability
become holes. In ds1_comb_chi4_s0 this happens to the first principal component, which sets how the energy is
split between layers 1 and 2, hence the double peaks in both layer energies.

What it does: the same quantile rule as at initialization (CoMB.init_readout), applied to the trained circuit's
exact count marginals, with the decoder (m, s, mu, P and the standardization of the scores) kept exactly as
trained. The circuit, the count probabilities and every correlation are unchanged; only the bin edges move.

Then: python scripts/comb_ds1.py sample --run outputs/runs/ds1_comb_chi4_s0_recal.npz   (and the plot command as usual)
"""
import sys, pathlib, argparse
import numpy as np


def find_shower_code(run, root=None):
    """The folder with the shower code: scripts/comb_ds1.py and comb/showers/zoo.py.
    Tried in order: --root, the project the model file lives in (<root>/outputs/runs/<model>.npz), the current
    folder, and the folder above this script."""
    cands = [pathlib.Path(root).expanduser()] if root else []
    rp = pathlib.Path(run).expanduser().resolve()
    cands += [rp.parents[k] for k in (2, 1, 0) if len(rp.parents) > k]
    cands += [pathlib.Path.cwd(), pathlib.Path(__file__).resolve().parents[1]]
    ok = lambda c: (c / "comb" / "showers" / "zoo.py").exists() and (c / "scripts" / "comb_ds1.py").exists()
    for c in cands:
        if ok(c):
            return c.resolve()
    # not on the usual paths: look in the neighbouring project folders (all comb folders usually sit side by side)
    found, seen = [], set()
    for base in {pathlib.Path.cwd().parent, rp.parents[2].parent if len(rp.parents) > 2 else rp.parent, pathlib.Path(__file__).resolve().parents[1].parent}:
        if not base.is_dir():
            continue
        for d in sorted(base.iterdir()):
            for c in ([d] + ([e for e in sorted(d.iterdir()) if e.is_dir()] if d.is_dir() else [])):
                try:
                    if c.is_dir() and c.resolve() not in seen and ok(c):
                        seen.add(c.resolve())
                        found.append(c.resolve())
                except PermissionError:
                    pass
    if len(found) == 1:
        print(f"found the shower code next to this folder: {found[0]}")
        return found[0]
    if len(found) > 1:
        # choose the folder this model was trained in: an identical copy of the model file, and the Dataset 1 data
        import filecmp, os
        def info(c):
            m = c / "outputs" / "runs" / rp.name
            same = m.exists() and filecmp.cmp(m, rp, shallow=False)
            data = (c / "data" / "cal_ds1_meta.json").exists()
            age = max(os.path.getmtime(c / "scripts" / "comb_ds1.py"), os.path.getmtime(c / "comb" / "showers" / "model.py") if (c / "comb" / "showers" / "model.py").exists() else 0)
            return same, data, age
        rows = [(c, *info(c)) for c in found]
        for want in (lambda r: r[1] and r[2], lambda r: r[1], lambda r: r[2]):
            hits = [r for r in rows if want(r)]
            if hits:
                best = max(hits, key=lambda r: r[3])
                why = ("has an identical copy of this model" if best[1] else "") + (" and " if best[1] and best[2] else "") + ("has the Dataset 1 data" if best[2] else "")
                note = f" (the most recently changed of {len(hits)})" if len(hits) > 1 else ""
                print(f"chose {best[0]}: it {why}{note}; pass --root to use another folder")
                return best[0]
        import time
        sys.exit("Several folders contain the shower code, and none has this model file or the Dataset 1 data:\n"
                 + "\n".join(f"    {str(r[0])}   (code changed {time.strftime('%Y-%m-%d', time.localtime(r[3]))})" for r in rows)
                 + "\nchoose the one that trained the model with --root \"<folder>\"")
    sys.exit("Could not find the shower code (a folder containing scripts/comb_ds1.py and comb/showers/zoo.py), neither on the usual\n"
             "paths nor in the folders next to this one. Locate it with\n"
             "    find ~ -path '*/scripts/comb_ds1.py' -not -path '*/.*' 2>/dev/null\n"
             "and pass the folder above scripts/:\n"
             "    python scripts/recalibrate_readout.py <model.npz> --root \"/path/to/the/shower/folder\"")


def recalibrate(model, Y_fit):
    """Quantile-match every coordinate's readout to Y_fit, given the model's current exact count marginals."""
    b = model.base
    params_before = np.array(model.get(), copy=True)
    X = (Y_fit - model.m) / model.s                      # standardized showers, as in training
    U = (X - model.mu) @ model.P                         # principal-component scores of the fitting showers
    old = [b.readout(j) for j in range(b.d)]
    b.init_readout(U, m=b.m, s=b.s)                      # same rule as at initialization; standardization kept
    model._update_affine()
    new = [b.readout(j) for j in range(b.d)]
    pc = b.count_marginals()
    return params_before, old, new, pc


def density_spread(p, delta):
    dens = p / np.maximum(delta, 1e-12)
    inner = dens[(p > 1e-3)]
    return inner.max() / inner.min() if inner.size else float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run", help="trained model, e.g. outputs/runs/ds1_comb_chi4_s0.npz")
    ap.add_argument("--root", help="folder with the shower code (scripts/comb_ds1.py and comb/showers/zoo.py), if it cannot be found automatically")
    ap.add_argument("--check", action="store_true", help="only check that the code and the model load; touches no data and writes nothing")
    a = ap.parse_args()
    run = pathlib.Path(a.run).expanduser()
    out = run.with_name(run.stem + "_recal.npz")
    if not run.exists():
        sys.exit(f"no model file at {run}")
    root = find_shower_code(run, a.root)
    sys.path.insert(0, str(root))
    for k in [m for m in list(sys.modules) if m == "comb" or m.startswith("comb.")]:
        del sys.modules[k]                               # never pick up a different comb package
    global zoo
    from comb.showers import zoo
    model, spec, extra = zoo.load(str(run))
    print(f"shower code: {root}\nmodel: {run} ({type(model).__name__}, {model.base.d} coordinates, {model.base.K} levels)")
    if a.check:
        print("check passed: the code and the model load; run again without --check to recalibrate")
        return
    import comb_ds1 as ds1
    D = ds1.load_data()
    Y_fit = D["Y_fit"]
    params_before, old, new, pc = recalibrate(model, Y_fit)
    same = np.array_equal(params_before[: model.base.n_inst_params()], np.array(model.get())[: model.base.n_inst_params()])
    print(f"circuit parameters unchanged: {same}")
    for j in range(min(3, model.base.d)):
        p = pc[j]
        print(f"coordinate {j}: count probabilities {np.round(p, 3).tolist()}")
        print(f"   bin widths before {np.round(old[j][0], 3).tolist()}  (density spread {density_spread(p, old[j][0]):.0f}x)")
        print(f"   bin widths after  {np.round(new[j][0], 3).tolist()}  (density spread {density_spread(p, new[j][0]):.0f}x)")
    zoo.save(str(out), model, spec, dict(extra, recalibrated_from=run.name))
    print(f"wrote {out.resolve()}")
    print("next, sample it from the shower folder (comb_ds1.py lives there, not in the digits folder):")
    print(f'    cd "{root}"')
    print(f'    python scripts/comb_ds1.py sample --run "{out.resolve()}"')


if __name__ == "__main__":
    main()
