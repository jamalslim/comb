#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Train and evaluate the models in the paper's main results table (d = 12).

    python scripts/run_d12.py prep                       # tokenizer + classical baselines
    python scripts/run_d12.py quantum M3C0-L6-E450-P60   # the main model
    python scripts/run_d12.py quantum M0C0               # the no-memory control
    python scripts/run_d12.py report                     # the table

What each row of the table comes from:

    CoMB blockwise (L=6)        quantum M3C0-L6-E450-P60
    CoMB two-stage (L=8)        DEPTH=8 python scripts/train_two_stage.py
    CoMB hardware (L=2)         the shipped device run, outputs/ibm_run_M3W3L2.npz
    HMM (chi=8)                 prep
    no-memory control           quantum M0C0
    independent blocks          prep
    emission / statistical floor   computed by report

Training is likelihood-free throughout: the blockwise conditional kernel
score with its exact adjoint gradient. No negative log-likelihood is
computed for any CoMB model.

Do not run `quantum M3C0`: it writes outputs/theta_M3C0.npy, which holds
the exact parameters the hardware job ran.
"""

import sys
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from comb import d12_config
from _pipeline import phase_prep, phase_quantum, phase_report


def main():
    cfg = d12_config()
    cmd = sys.argv[1] if len(sys.argv) > 1 else "report"
    if cmd == "prep":
        phase_prep(cfg)
    elif cmd == "quantum":
        phase_quantum(cfg, sys.argv[2:])
    elif cmd == "report":
        phase_report(cfg)
    else:
        raise SystemExit(f"unknown command {cmd}")


if __name__ == "__main__":
    main()
