#!/usr/bin/env python3
"""Replot every figure of the paper.

    cd comb
    python plot/make_all.py                                  # figures -> outputs/paper_figures/
    python plot/make_all.py --out ../comb_paper/figures      # straight into the paper
    python plot/make_all.py --style CMS

This folder (plot/) sits inside the comb folder, and every figure is drawn from the data there: the calorimeter
showers (outputs/samples, outputs/hardware, outputs/runs) and the digits runs (outputs/strokes). Figures whose
runs exist are drawn; the others are skipped with a message, so the command can be rerun as runs finish.
mplhep (pip install mplhep) gives the HEP style.
"""
import argparse, pathlib, subprocess, sys

HERE = pathlib.Path(__file__).resolve().parent
COMB = HERE.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(COMB / "outputs" / "paper_figures"))
    ap.add_argument("--style", default="ROOT")
    ap.add_argument("--comb2", default=None, help="folder with the digits runs, if not the comb folder")
    a = ap.parse_args()
    ok = subprocess.run([sys.executable, str(HERE / "ds1_plots.py"), "--out", a.out, "--style", a.style]).returncode == 0
    sys.path.insert(0, str(HERE))
    from digits_plots import digits_folder
    root = digits_folder(a.comb2)
    for data in ("mnist", "uci"):
        err = root / "outputs" / "strokes" / f"ERRORS_{data}.json"
        script = root / "scripts" / "statistical_errors.py"
        if (root / "data" / f"strokes_{data}.npz").exists() and not err.exists() and script.exists():
            print(f"== statistical errors for {data} (jackknife over the test digits; once)")
            subprocess.run([sys.executable, str(script), "--data", data], cwd=root)
        cmd = [sys.executable, str(HERE / "digits_plots.py"), "--data", data, "--out", a.out, "--style", a.style] + (["--comb2", a.comb2] if a.comb2 else [])
        ok &= subprocess.run(cmd).returncode == 0
    print(f"\nfigures in {pathlib.Path(a.out).resolve()}" if ok else "\nsome figures failed; see the messages above")


if __name__ == "__main__":
    main()
