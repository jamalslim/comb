#!/usr/bin/env python3
"""Hardware-free deployment study for the stroke models (exact noisy evaluation, no shots).

    python scripts/deploy_strokes.py --data uci                    # every deployable evaluated run
    python scripts/deploy_strokes.py --data uci --runs strokes_uci_tree_chi4_L4_path2_s0 strokes_uci_tree_dephased_chi4_L4_path2_s0

For every run: the exact held-out objective (k-point + path terms, the same weights and frequencies as in
training, on the test digits) under

  * the trapped-ion (H2-like) and superconducting (Heron r3-like) noise models of scripts/noise_study.py,
    copied verbatim with their sources;
  * the superconducting model scaled by lambda (all error probabilities and the idle time times lambda).

Tree runs (tree, tree_dephased) use comb/noisytree.py: one measure-and-reset window per node, so log2 K
windows per token, each charged the same idle time as a full round of monolithic CoMB (conservative for the
tree, whose nodes are shallower). Monolithic CoMB runs use comb/noise.py: one window per token. Runs without a
circuit (classical HMM, general memories) are skipped.

Writes outputs/strokes/DEPLOY_<data>.json and .md and outputs/figs/strokes_<data>_deploy.pdf. Reports, for
every coherent tree run with a dephased twin of the same node depth and seed, the noise scale lambda* at which
the coherent run stops beating the twin (linear interpolation between scanned scales).
"""
import argparse, json, pathlib, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import comb_strokes as cs                                                    # noqa: E402
from comb.digits import kpoint as kp, pencf                                         # noqa: E402
from comb.digits.noise import NoiseModel, NoisyCoMB                                 # noqa: E402
from comb.digits.noisytree import NoisyTreeCoMB                                     # noqa: E402

# Device numbers as in scripts/noise_study.py: Quantinuum H2 performance validation (docs.quantinuum.com);
# IBM Heron r3 medians reported for ibm_pittsburgh. p1, the reset errors and the round duration of the
# superconducting model are assumptions (stated there).
ION = NoiseModel(p1=2e-5, p2=1e-3, p_ro=1e-3, p_reset=1e-3, t_round=1.0, T1=np.inf,
                 T2=1.0 / (2 * 1.4e-3), name="trapped-ion (H2-like)")
SC = NoiseModel(p1=2.5e-4, p2=1.5e-3, p_ro=1.3e-2, p_reset=1e-2, t_round=2.0, T1=200.0, T2=200.0,
                name="superconducting (Heron r3-like)")
SCALES = (0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0)


def deployable(model, spec, noise):
    if spec["model"] in ("tree", "tree_dephased"):
        return NoisyTreeCoMB.from_model(model, noise)
    if spec["model"] == "comb" and not spec.get("n_env"):
        return NoisyCoMB.from_model(model, noise)
    return None


class Objective:
    """The training objective evaluated on the test digits, exactly."""

    def __init__(self, z, d):
        self.z, self.d, self.cache = z, d, {}
        self.s_te = cs.stats_for(z, d, 3, "test")

    def __call__(self, model, spec):
        G = kp.token_kernel(16, spec["kernel"], kappa=spec["kappa"], rho_pen=spec["rho_pen"])
        Lk = kp.kpoint_loss(model, self.s_te, G, tuple(spec["weights"]), grad=False)[0]
        Lp = 0.0
        if spec.get("path", 0) > 0:
            key = spec["path_every"]
            if key not in self.cache:
                pc = pencf.PathCF(self.z["tokens_train"], self.d, 16, every=key, seed=0)
                self.cache[key] = (pc, pc.frequencies(2048, np.random.default_rng(13)), pencf.positions(self.z["tokens_test"], pc.taus))
            pc, om, X = self.cache[key]
            Lp = spec["path"] * pc.loss(model, X, om, grad=False)[0] / spec["path_reference"]
        return dict(total=Lk + Lp, kpoint=Lk, path=Lp)


def threshold(scales, coh, deph):
    """Smallest lambda at which coh >= deph (linear interpolation); None if never within the scan."""
    diff = np.array(coh) - np.array(deph)
    if diff[0] >= 0:
        return 0.0
    for i in range(1, len(scales)):
        if diff[i] >= 0:
            a, b = scales[i - 1], scales[i]
            return float(a + (b - a) * (-diff[i - 1]) / (diff[i] - diff[i - 1]))
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="uci")
    ap.add_argument("--runs", nargs="*")
    a = ap.parse_args()
    z, meta = cs.load_prepared(a.data)
    d = meta["d"]
    names = a.runs or sorted(p.name[:-len(".metrics.json")] for p in cs.OUT.glob(f"strokes_{a.data}_*.metrics.json"))
    obj = Objective(z, d)
    out = {}
    for name in names:
        model, spec = cs.load_run(name)
        if deployable(model, spec, NoiseModel()) is None:
            continue
        res = {"noiseless": obj(model, spec)}
        res[ION.name] = obj(deployable(model, spec, ION), spec)
        for lam in SCALES[1:]:
            nz = SC.scaled(lam)
            res[f"SC x{lam:g}"] = obj(deployable(model, spec, nz), spec)
        out[name] = dict(spec=dict(model=spec["model"], L_node=spec.get("L_node"), L_v=spec.get("L_v"), seed=spec["seed"], path=spec.get("path", 0)), results=res)
        print(f"  {name}: noiseless {res['noiseless']['total']:.3f} | ion {res[ION.name]['total']:.3f} | "
              + " ".join(f"SCx{l:g} {res[f'SC x{l:g}']['total']:.3f}" for l in SCALES[1:]), flush=True)
    # thresholds: coherent tree vs its dephased twin (same node depth and seed)
    thr = {}
    for name, r in out.items():
        sp = r["spec"]
        if sp["model"] != "tree":
            continue
        twin = next((n for n, q in out.items() if q["spec"]["model"] == "tree_dephased" and q["spec"]["L_node"] == sp["L_node"]
                     and q["spec"]["seed"] == sp["seed"] and q["spec"]["path"] == sp["path"]), None)
        if twin is None:
            continue
        curve = lambda n: [out[n]["results"]["noiseless"]["total"]] + [out[n]["results"][f"SC x{l:g}"]["total"] for l in SCALES[1:]]
        thr[name] = dict(twin=twin, lambda_star=threshold(SCALES, curve(name), curve(twin)),
                         ion_coherent=out[name]["results"][ION.name]["total"], ion_dephased=out[twin]["results"][ION.name]["total"])
    OUTD = cs.OUT
    OUTD.mkdir(parents=True, exist_ok=True)
    json.dump(dict(noise_models=dict(ion=ION.as_dict(), sc=SC.as_dict(), scales=list(SCALES)), runs=out, thresholds=thr),
              open(OUTD / f"DEPLOY_{a.data}.json", "w"), indent=1)
    cols = ["noiseless", ION.name] + [f"SC x{l:g}" for l in SCALES[1:]]
    md = [f"# Deployment study ({a.data}): exact held-out objective under noise", "",
          "Lower is better. SC x1 is the superconducting (Heron r3-like) model; SC x lambda scales all its error rates and idle time.", "",
          "| run | " + " | ".join(c.replace("trapped-ion (H2-like)", "ion (H2-like)") for c in cols) + " |", "|---" * (len(cols) + 1) + "|"]
    for name, r in out.items():
        md.append(f"| {name.replace(f'strokes_{a.data}_', '')} | " + " | ".join(f"{r['results'][c]['total']:.3f}" for c in cols) + " |")
    if thr:
        md += ["", "## Noise threshold of the coherence advantage", ""]
        for name, t in thr.items():
            ls = "beyond the scan (coherent stays better up to x8)" if t["lambda_star"] is None else f"lambda* = {t['lambda_star']:.2f} x Heron-r3-like"
            md.append(f"- {name.replace(f'strokes_{a.data}_', '')} vs {t['twin'].replace(f'strokes_{a.data}_', '')}: {ls}; "
                      f"under the ion model {t['ion_coherent']:.3f} vs {t['ion_dephased']:.3f}")
    (OUTD / f"DEPLOY_{a.data}.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(figsize=(5.6, 2.8))
    xs = np.arange(len(SCALES))
    for name, r in out.items():
        y = [r["results"]["noiseless"]["total"]] + [r["results"][f"SC x{l:g}"]["total"] for l in SCALES[1:]]
        ax.plot(xs, y, "o-", ms=3, label=name.replace(f"strokes_{a.data}_", ""))
    ax.set_xticks(xs)
    ax.set_xticklabels(["0"] + [f"{l:g}" for l in SCALES[1:]])
    ax.set_xlabel("noise scale lambda (x Heron r3-like)")
    ax.set_ylabel("held-out objective")
    ax.set_yscale("log")
    ax.legend(fontsize=6.5, frameon=False)
    figs = ROOT / "outputs" / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    fig.savefig(figs / f"strokes_{a.data}_deploy.pdf", bbox_inches="tight")


if __name__ == "__main__":
    main()
