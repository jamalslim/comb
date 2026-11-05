#!/usr/bin/env python3
"""Run every digits experiment (exactly trainable coherent-memory generators on digits drawn as strokes).

    python scripts/paper2.py --data uci --list                         # the jobs, and which are done
    python scripts/paper2.py --data uci --stage tests                  # exactness tests (run once, a few minutes)
    python scripts/paper2.py --data uci --stage decisive --workers 4   # 4 trainings at a time
    python scripts/paper2.py --data mnist --stage all --workers 8
    python scripts/paper2.py --data mnist --stage report

Stages (run in this order; "all" runs them in sequence):
  tests        exactness tests of every module
  decisive     chi = 4, three seeds: general coherent memory (16 environment levels) vs the same memory dephased
               (identical parameters) vs the general classical HMM; plus the coherent memory without environment
  circuits     CoMB circuit (depth 4) and its dephased twin; tree-CoMB (one work qubit, node depth 4 and 6) and
               the dephased tree
  memory       classical HMMs with chi = 6, 8, 12, 16 (equivalent classical memory of the coherent one)
  conditional  one tree-CoMB per digit (class-conditional generation) and the quality gate
  deploy       exact noisy evaluation of the circuit runs (trapped-ion and Heron-like models, noise sweep)
  report       results table, equivalent classical memory, figures

Every job is "train, then evaluate" for one run. A finished job (metrics file present) is skipped. An
interrupted training resumes from its checkpoint, and a resumed run ends exactly where an uninterrupted one
would. Each worker runs one job single-threaded, so --workers should not exceed the number of cores. Logs go
to outputs/strokes/<run>.log (training) and <run>.eval.log (evaluation).
"""
import argparse, os, pathlib, subprocess, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import comb_strokes as cs                                                    # noqa: E402

PY = sys.executable
ENV = dict(os.environ, OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
TESTS = ["test_strokes.py", "test_kpoint.py", "test_pencf.py", "test_hqmm.py", "test_treecomb.py", "test_noisytree.py"]


def jobs(data, seeds, steps):
    """Stage -> list of train-argument lists (run names come from comb_strokes.run_name)."""
    P = ["--path", "2"]
    J = {"decisive": [], "circuits": [], "memory": [], "conditional": []}
    for s in seeds:
        J["decisive"] += [["--model", "hqmm", "--chi", "4", "--n_env", "4", "--seed", str(s)] + P,
                          ["--model", "hqmm_dephased", "--chi", "4", "--n_env", "4", "--seed", str(s)] + P,
                          ["--model", "hmm", "--chi", "4", "--seed", str(s)] + P]
    J["decisive"].append(["--model", "hqmm", "--chi", "4", "--n_env", "0", "--seed", "0"] + P)
    J["circuits"] = [["--model", "comb", "--chi", "4", "--seed", "0"] + P, ["--model", "dephased", "--chi", "4", "--seed", "0"] + P,
                     ["--model", "tree", "--chi", "4", "--L_node", "4", "--seed", "0"] + P,
                     ["--model", "tree_dephased", "--chi", "4", "--L_node", "4", "--seed", "0"] + P,
                     ["--model", "tree", "--chi", "4", "--L_node", "6", "--seed", "0"] + P,
                     ["--model", "tree_dephased", "--chi", "4", "--L_node", "6", "--seed", "0"] + P]
    J["memory"] = [["--model", "hmm", "--chi", str(c), "--seed", "0"] + P for c in (6, 8, 12, 16)]
    J["conditional"] = [["--model", "tree", "--chi", "4", "--L_node", "4", "--digit", str(k), "--seed", "0"] + P for k in range(10)]
    out = {}
    for stage, lst in J.items():
        out[stage] = []
        for args in lst:
            full = ["--data", data] + args + (["--steps", str(steps)] if steps else [])
            out[stage].append((cs.run_name(cs.parser().parse_args(["train"] + full)), full))
    return out


def done(name, what):
    return (cs.OUT / f"{name}.{'metrics.json' if what == 'eval' else 'json'}").exists()


def run_job(name, args):
    t0 = time.time()
    cs.OUT.mkdir(parents=True, exist_ok=True)
    if not done(name, "train"):
        r = subprocess.run([PY, "scripts/comb_strokes.py", "train"] + args, cwd=ROOT, env=ENV, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if r.returncode:
            return name, False, f"training failed: {r.stderr.strip().splitlines()[-1] if r.stderr.strip() else r.returncode}"
    data = args[args.index("--data") + 1]
    with open(cs.OUT / f"{name}.eval.log", "w") as f:
        r = subprocess.run([PY, "scripts/comb_strokes.py", "evaluate", "--data", data, "--run", name], cwd=ROOT, env=ENV, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        return name, False, f"evaluation failed, see {name}.eval.log"
    return name, True, f"{(time.time() - t0) / 60:.1f} min"


def run_stage(stage, J, workers):
    todo = [(n, a) for n, a in J[stage] if not done(n, "eval")]
    print(f"[{stage}] {len(J[stage]) - len(todo)} of {len(J[stage])} jobs already done; running {len(todo)} with {workers} worker(s)", flush=True)
    ok = True
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(run_job, n, a) for n, a in todo]
        for f in as_completed(futs):
            name, good, msg = f.result()
            ok &= good
            print(f"  {'done ' if good else 'ERROR'} {name}: {msg}", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="uci", choices=["uci", "mnist"])
    ap.add_argument("--stage", default="all", choices=["all", "tests", "decisive", "circuits", "memory", "conditional", "deploy", "report"])
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0, 1, 2])
    ap.add_argument("--steps", type=int, default=0, help="override training steps (default: 1500, the protocol of the paper)")
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    if not (cs.DATA / f"strokes_{a.data}.npz").exists():
        print(f"preparing {a.data} ...", flush=True)
        subprocess.run([PY, "scripts/comb_strokes.py", "prepare", "--data", a.data], cwd=ROOT, env=ENV, check=True)
    J = jobs(a.data, a.seeds, a.steps)
    if a.list:
        for stage, lst in J.items():
            print(f"[{stage}]")
            for n, _ in lst:
                print(f"  {'done   ' if done(n, 'eval') else ('trained' if done(n, 'train') else 'todo   ')} {n}")
        return
    stages = ["tests", "decisive", "circuits", "memory", "conditional", "deploy", "report"] if a.stage == "all" else [a.stage]
    for stage in stages:
        if stage == "tests":
            for t in TESTS:
                print(f"[tests] {t}", flush=True)
                subprocess.run([PY, f"scripts/{t}"], cwd=ROOT, env=ENV, check=True)
        elif stage in J:
            if not run_stage(stage, J, a.workers):
                sys.exit(f"stage {stage} had errors; rerun the same command to retry (finished jobs are kept)")
            if stage == "conditional":
                subprocess.run([PY, "scripts/class_quality.py", "--data", a.data, "--tag", "tree_chi4_L4_path2", "--seed", "0"], cwd=ROOT, env=ENV)
        elif stage == "deploy":
            runs = [n for n, _ in J["circuits"] if done(n, "eval")]
            if runs:
                subprocess.run([PY, "scripts/deploy_strokes.py", "--data", a.data, "--runs"] + runs, cwd=ROOT, env=ENV, check=True)
            else:
                print("[deploy] no finished circuit runs yet")
        elif stage == "report":
            for cmd in (["scripts/strokes_table.py", "--data", a.data], ["scripts/memory_equiv.py", "--data", a.data], ["scripts/comb_strokes.py", "plot", "--data", a.data]):
                subprocess.run([PY] + cmd, cwd=ROOT, env=ENV)
            print(f"[report] outputs/strokes/RESULTS_{a.data}.md, MEMORY_{a.data}.md, DEPLOY_{a.data}.md, CLASSES_{a.data}_*.md; figures in outputs/figs/")


if __name__ == "__main__":
    main()
