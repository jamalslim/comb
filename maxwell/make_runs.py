#!/usr/bin/env python3
"""Write maxwell/runs.txt (one 'model seed chi' per line) and print the sbatch command with the right array range.

    python maxwell/make_runs.py --seeds 0-29 --per-node 100                    # CoMB, dephased twin, HMM(chi=4)
    python maxwell/make_runs.py --seeds 0-9 --models hmm --hmm-chis 6 8 12 16  # add the classical memory curve
    python maxwell/make_runs.py --seeds 0-29 --comb-chis 8 --hmm-chis 8          # CoMB, twin and HMM at chi = 8

CoMB and its dephased twin take chi = 4 or 8 (two or three memory qubits; --comb-chis), the HMM any chi (--hmm-chis). Runs whose model file already
exists are skipped, so the command can be repeated to add seeds or memory sizes."""
import argparse, math, pathlib
ROOT = pathlib.Path(__file__).resolve().parents[1]
def name(m, s, chi, Lv=5, wiring="interleaved", dense=False):
    d = ("_dense" if dense and m != "hmm" else "") + ("_nest" if wiring == "nested" and m != "hmm" else "") + (f"_L{Lv}" if Lv != 5 else "")
    return {"comb": f"ds1_comb_chi{chi}_s{s}{d}_ly", "dephased": f"ds1_dephased_chi{chi}_s{s}{d}_ly", "hmm": f"ds1_hmm{chi}_s{s}_ly"}[m]
ap = argparse.ArgumentParser()
ap.add_argument("--models", nargs="+", default=["comb", "dephased", "hmm"])
ap.add_argument("--seeds", default="0-9")
ap.add_argument("--hmm-chis", nargs="+", type=int, default=[4])
ap.add_argument("--comb-chis", nargs="+", type=int, default=[4])
ap.add_argument("--Lv", type=int, default=5)
ap.add_argument("--wiring", choices=["interleaved", "nested"], default="interleaved")
ap.add_argument("--dense", action="store_true")
ap.add_argument("--per-node", type=int, default=30)
ap.add_argument("--list", default="maxwell/runs.txt", help="run list file; use a different one for each batch you submit")
a = ap.parse_args()
lo, hi = (int(x) for x in a.seeds.split("-"))
todo = [(m, s, c) for s in range(lo, hi + 1) for m in a.models for c in (a.hmm_chis if m == "hmm" else a.comb_chis)]
lines = [f"{m} {s} {c} {a.Lv if m != 'hmm' else 5} {a.wiring if m != 'hmm' else 'interleaved'} {'dense' if a.dense and m != 'hmm' else 'circuit'}" for m, s, c in todo
         if not (ROOT / "outputs/runs" / (name(m, s, c, a.Lv if m != "hmm" else 5, a.wiring, a.dense) + ".npz")).exists()]
(ROOT / a.list).write_text("\n".join(lines) + "\n")
chunks = max(1, math.ceil(len(lines) / a.per_node))
print(f"{len(lines)} runs to do -> {a.list} ({chunks} node(s) with up to {a.per_node} runs each)")
print(f"submit with:  RUNS_FILE={a.list} RUNS_PER_NODE={a.per_node} sbatch --array=0-{chunks - 1} maxwell/seeds.sbatch")
