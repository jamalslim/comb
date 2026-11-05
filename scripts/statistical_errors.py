#!/usr/bin/env python3
"""Statistical errors of every held-out result: a delete-a-group jackknife over the test digits.

    python scripts/statistical_errors.py --data uci            # writes outputs/strokes/ERRORS_uci.json
    python scripts/statistical_errors.py --data mnist

Every held-out objective compares a model's exact statistics with statistics of a finite set of test digits, so
its statistical uncertainty is that of the test set. The test digits are split into G = 50 groups and the
objective is recomputed G times, each time without one group; the standard error is
sqrt((G-1)/G * sum_g (theta_g - mean)^2). Unlike a bootstrap, this duplicates no digit, which matters because
the MMD statistics include each digit's similarity with itself. The model side is computed once per run (and
once per noise setting for the deployable circuits); only the data side is recomputed. The same groups are used
for every run, so differences between runs (a circuit and its dephased twin, say) get paired errors.
plot/digits_plots.py reads ERRORS_<data>.json for its error bars.
"""
import argparse, json, pathlib, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import comb_strokes as cs                                                    # noqa: E402
from comb.digits import kpoint as kp, pencf, strokes as st                          # noqa: E402


class Boot:
    def __init__(self, z, d, G=50, seed=7):
        self.z, self.d = z, d
        self.T = z["tokens_test"].astype(np.int64)
        n = len(self.T)
        groups = np.array_split(np.random.default_rng(seed).permutation(n), G)
        self.idx = [np.setdiff1d(np.arange(n), g) for g in groups]           # each subset leaves one group out
        self.lags = [l for l in cs.LAGS if l < d]
        self.s_te = kp.data_statistics(self.T, st.K_TOKENS, kmax=3, lags=self.lags)
        self.s_b = [kp.data_statistics(self.T[i], st.K_TOKENS, kmax=3, lags=self.lags) for i in self.idx]
        self.pc = {}

    def _path(self, every):
        if every not in self.pc:
            pc = pencf.PathCF(self.z["tokens_train"], self.d, st.K_TOKENS, every=every, seed=0)
            om = pc.frequencies(2048, np.random.default_rng(13))
            X = pencf.positions(self.T, pc.taus)
            ph_b = np.stack([pc.empirical_cf(X[i], om) for i in self.idx])
            self.pc[every] = (pc, om, pc.weights(om), ph_b)
        return self.pc[every]

    def __call__(self, model, spec):
        G = kp.token_kernel(st.K_TOKENS, spec["kernel"], kappa=spec["kappa"], rho_pen=spec["rho_pen"])
        w = tuple(spec["weights"])
        s_m, _ = kp.model_statistics(model, self.s_te)
        loc = np.array([kp.mmd_terms(s_m, sb, G, w)[0] for sb in self.s_b])
        glo = np.zeros_like(loc)
        if spec.get("path", 0) > 0:
            pc, om, W, ph_b = self._path(spec["path_every"])
            phi = pc.model_cf(model, W)[0]
            glo = spec["path"] * np.mean(np.abs(ph_b - phi[None, :]) ** 2, axis=1) / spec["path_reference"]
        return dict(total=(loc + glo).tolist(), local=loc.tolist(), **{"global": glo.tolist()})


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="uci")
    ap.add_argument("--groups", type=int, default=50)
    ap.add_argument("--no-deploy", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    z, meta = cs.load_prepared(a.data)
    d = meta["d"]
    boot = Boot(z, d, a.groups)
    out_f = cs.OUT / f"ERRORS_{a.data}.json"
    out = json.load(open(out_f)) if out_f.exists() else {}
    if out.get("groups") != a.groups:
        out = {}
    out.update(method="delete-a-group jackknife over the test digits", groups=a.groups, n_test=int(len(boot.T)))
    out.setdefault("runs", {})
    out.setdefault("deploy", {})
    names = sorted(p.name[:-len(".metrics.json")] for p in cs.OUT.glob(f"strokes_{a.data}_*.metrics.json"))
    for name in names:
        model, spec = cs.load_run(name)
        if spec.get("digit", -1) >= 0 or name in out["runs"]:
            continue
        out["runs"][name] = boot(model, spec)
        json.dump(out, open(out_f, "w"))
        v = np.array(out["runs"][name]["total"])
        se = np.sqrt((len(v) - 1) / len(v) * ((v - v.mean()) ** 2).sum())
        print(f"  {name}: standard error of the held-out objective {se:.4f}", flush=True)
    if not a.no_deploy:
        import deploy_strokes as dep
        for name in names:
            model, spec = cs.load_run(name)
            if spec.get("digit", -1) >= 0 or dep.deployable(model, spec, dep.NoiseModel()) is None:
                continue
            res = out["deploy"].setdefault(name, {})
            settings = [("noiseless", None), (dep.ION.name, dep.ION)] + [(f"SC x{l:g}", dep.SC.scaled(l)) for l in dep.SCALES[1:]]
            for key, noise in settings:
                if key in res:
                    continue
                m = model if noise is None else dep.deployable(model, spec, noise)
                res[key] = boot(m, spec)["total"]
                json.dump(out, open(out_f, "w"))
            print(f"  {name}: deployment errors done ({len(settings)} settings)", flush=True)
    json.dump(out, open(out_f, "w"))
    print(f"wrote {out_f} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
