#!/usr/bin/env python3
"""Train one model chain until it stops improving, with the same rule for every model.

    python maxwell/converge.py --family comb4 --seed 0 [--block 3000 --tol 0.01 --max-blocks 4]

A chain starts from an existing trained run and continues it in blocks of BLOCK steps (comb_ds1.py train --init-from,
which uses 0.1 x the default learning rates for every model). After each block the best validation MMD^2 of the new
block is compared with that of the previous one; the chain stops when the relative improvement is below TOL, or after
MAX_BLOCKS extra blocks. The final model is copied to a name ending in _conv_ly, which maxwell/summarize_seeds.py
groups as the converged model. The driver is idempotent: finished blocks are skipped, an interrupted block resumes from
its checkpoint, and a finished chain is left alone. A record of every chain is written to outputs/runs/<conv name>.chain.json.

Families (memory size from the name, or --chi for hmm):
  comb4, twin4            circuit CoMB and twin, chi = 4, five layers
  comb8L18, twin8L18      circuit CoMB and twin, chi = 8, nested wiring, 18 layers
  dense4, dtwin4          general coherent memory and its twin, chi = 4 (start from the existing _dense_warm runs)
  dense8, dtwin8          the same at chi = 8
  hmm                     classical HMM with --chi states (starts from its _warm run where one exists)
"""
import argparse, json, pathlib, shutil, subprocess, sys
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
RUNS = ROOT / "outputs/runs"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import comb_ds1 as cd                                                             # noqa: E402

FAM = {   # family: (model, chi, Lv, wiring, dense, first runs of the chain as name templates)
    "comb4":    ("comb", 4, 5, "interleaved", False, ["ds1_comb_chi4_s{s}_ly"]),
    "twin4":    ("dephased", 4, 5, "interleaved", False, ["ds1_dephased_chi4_s{s}_ly"]),
    "comb8L18": ("comb", 8, 18, "nested", False, ["ds1_comb_chi8_s{s}_nest_L18_ly"]),
    "twin8L18": ("dephased", 8, 18, "nested", False, ["ds1_dephased_chi8_s{s}_nest_L18_ly"]),
    "dense4":   ("comb", 4, 5, "interleaved", True, ["ds1_comb_chi4_s{s}_ly", "ds1_comb_chi4_s{s}_dense_warm_ly"]),
    "dtwin4":   ("dephased", 4, 5, "interleaved", True, ["ds1_dephased_chi4_s{s}_ly", "ds1_dephased_chi4_s{s}_dense_warm_ly"]),
    "dense8":   ("comb", 8, 5, "interleaved", True, ["ds1_comb_chi8_s{s}_nest_L18_ly", "ds1_comb_chi8_s{s}_dense_warm_ly"]),
    "dtwin8":   ("dephased", 8, 5, "interleaved", True, ["ds1_dephased_chi8_s{s}_nest_L18_ly", "ds1_dephased_chi8_s{s}_dense_warm_ly"]),
    "comb4K16": ("comb", 4, 8, "interleaved", False, ["ds1_comb_chi4_s{s}_L8_K16_ly"]),          # 16 levels per value: 4 work qubits, 8 layers
    "twin4K16": ("dephased", 4, 8, "interleaved", False, ["ds1_dephased_chi4_s{s}_L8_K16_ly"]),
}

class A:
    pass

def spec_name(model, chi, seed, Lv, wiring, dense, K=8):
    a = A()
    a.model, a.chi, a.seed, a.Lv, a.Lw, a.wiring, a.dense = model, chi, seed, Lv, 2, wiring, dense
    a.K = K
    a.nw = int(round(np.log2(K)))
    return cd.name_of(cd.spec_of(a))

def best_val(name):
    return float(json.load(open(RUNS / f"{name}.json"))["best_val"])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True, choices=list(FAM) + ["hmm"])
    ap.add_argument("--chi", type=int, default=0)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--block", type=int, default=3000)
    ap.add_argument("--tol", type=float, default=0.01)
    ap.add_argument("--max-blocks", type=int, default=4)
    ap.add_argument("--print-name", action="store_true", help="print the name of the converged run and exit")
    ap.add_argument("--K", type=int, default=8, help="levels per value (hmm family; the comb families fix it by name)")
    a = ap.parse_args()
    s = a.seed
    if a.family == "hmm":
        model, chi, Lv, wiring, dense = "hmm", a.chi, 5, "interleaved", False
        stem = spec_name("hmm", chi, s, Lv, wiring, dense, a.K)
        chain = [f"{stem}_ly"] + ([f"{stem}_warm_ly"] if (RUNS / f"{stem}_warm_ly.npz").exists() else [])
        conv = f"{stem}_conv_ly"
    else:
        model, chi, Lv, wiring, dense, tmpl = FAM[a.family]
        chain = [t.format(s=s) for t in tmpl]
        if a.family.endswith("K16"): a.K = 16
        conv = (spec_name(model, chi, s, Lv, wiring, dense, a.K) if dense else spec_name(model, chi, s, Lv, wiring, False, a.K)) + "_conv_ly"
    if a.print_name:
        print(conv)
        return
    if (RUNS / f"{conv}.npz").exists():
        print(f"{conv}: converged before")
        return
    for n in chain:
        if not (RUNS / f"{n}.npz").exists():
            print(f"{conv}: start run {n} missing, skipped")
            return
    base = spec_name(model, chi, s, Lv, wiring, dense, a.K)             # name stem of the blocks trained here
    vals = [best_val(n) for n in chain]
    reason = "max blocks"
    k = len(chain) - 1                                                  # extra blocks already done (e.g. the warm block)
    while k < a.max_blocks:
        if len(vals) >= 2 and (vals[-2] - vals[-1]) / vals[-2] < a.tol:
            reason = f"improvement below {100 * a.tol:.0f}%"
            break
        k += 1
        tag = f"_c{k}"
        name = f"{base}_warm{tag}_ly"
        if not (RUNS / f"{name}.npz").exists():
            args = ["python", "scripts/comb_ds1.py", "train", "--model", model, "--chi", str(chi), "--seed", str(s), "--layer-share", "0.2",
                    "--init-from", chain[-1], "--steps", str(a.block), "--tag", tag]
            if model != "hmm": args += ["--Lv", str(Lv), "--wiring", wiring] + (["--dense"] if dense else [])
            args += (["--K", str(a.K)] if model == "hmm" else ["--nw", str(int(round(np.log2(a.K))))])
            print(f"{conv}: block {k} from {chain[-1]}", flush=True)
            r = subprocess.run(args, cwd=ROOT)
            if r.returncode != 0 or not (RUNS / f"{name}.npz").exists():
                print(f"{conv}: block {k} failed (exit {r.returncode})")
                return
        chain.append(name)
        vals.append(best_val(name))
        print(f"{conv}: block {k} best validation {vals[-1]:.4e} (previous {vals[-2]:.4e})", flush=True)
    else:
        if len(vals) >= 2 and (vals[-2] - vals[-1]) / vals[-2] < a.tol: reason = f"improvement below {100 * a.tol:.0f}%"
    final = chain[-1]
    for ext in (".npz", ".json"):
        shutil.copyfile(RUNS / f"{final}{ext}", RUNS / f"{conv}{ext}")
    json.dump(dict(chain=chain, best_val=vals, stopped=reason, block=a.block, tol=a.tol), open(RUNS / f"{conv}.chain.json", "w"), indent=1)
    print(f"{conv}: converged after {len(chain) - 1} extra blocks ({reason}); best validation {vals[-1]:.4e}", flush=True)

if __name__ == "__main__":
    main()
