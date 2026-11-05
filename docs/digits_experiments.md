# Running the digits experiments

Exactly trainable coherent-memory generators (CoMB) on handwritten digits drawn as pen strokes.
Everything is driven by one script, `scripts/paper2.py`. It is resumable (finished jobs are skipped,
interrupted trainings continue exactly from their checkpoints) and runs independent trainings in parallel.

## 1. Setup (once)

Python 3.10 or newer, with numpy, scipy, scikit-learn and matplotlib:

    pip install numpy scipy scikit-learn matplotlib

- **UCI digits** come with scikit-learn: nothing to download.
- **MNIST** is downloaded on first use (about 11 MB) and accepted only if it has the official class counts and
  pixel mean. To use a copy you already have, put `data/mnist.npz` with keys x_train, y_train, x_test, y_test
  (uint8) in the package folder.

## 2. Check the installation (once, a few minutes: 2 minutes measured on one core)

    python scripts/paper2.py --data uci --stage tests

This runs six test files. Each compares the code with an independent reference: brute-force enumeration of
every token sequence, finite-difference gradients, node-by-node density-matrix simulation, and sampling.
Every test prints its error, and the run stops at the first failure.

## 3. Run the experiments

See the job list and what is already done:

    python scripts/paper2.py --data uci --list

Run a stage. Use as many workers as you have cores; each worker is single-threaded:

    python scripts/paper2.py --data uci --stage decisive    --workers 4
    python scripts/paper2.py --data uci --stage circuits    --workers 4
    python scripts/paper2.py --data uci --stage memory      --workers 4
    python scripts/paper2.py --data uci --stage conditional --workers 4
    python scripts/paper2.py --data uci --stage deploy
    python scripts/paper2.py --data uci --stage report

`--stage all` runs everything in this order. For MNIST, replace `--data uci` with `--data mnist`. To stop at
any time, press Ctrl-C; running the same command again continues where it stopped.

| stage | runs | what it answers |
|---|---|---|
| decisive | 10 | does a coherent memory beat the same memory dephased (identical parameters) and the best classical memory of the same size? Three seeds. |
| circuits | 6 | does a real circuit keep the advantage? CoMB circuit and tree-CoMB (one work qubit, node depth 4 and 6), each with its dephased twin |
| memory | 4 | how many classical states the 4-level coherent memory is worth (classical HMMs with 6, 8, 12, 16 states) |
| conditional | 10 | does CoMB draw recognizable digits? One tree-CoMB per digit, then the quality gate |
| deploy | none | exact behaviour of the circuit runs under trapped-ion and Heron-like noise, and the noise level at which coherence stops helping |
| report | none | tables and figures |

## 4. How long it takes

These timings were measured on one core of a small machine with the UCI digits (sequences of 56 pen moves). Each
run is 1500 optimizer steps.

| model | time per run |
|---|---|
| classical HMM, 4 states | about 5 minutes |
| dephased general memory | about 6 minutes |
| tree-CoMB / CoMB circuit | about 15 to 25 minutes |
| general coherent memory | about 20 to 25 minutes |

UCI in total is about 9 core-hours, so about 2 to 3 hours on 4 cores. MNIST sequences are longer (the length is
set by `prepare` from the 90th percentile of the digits' walks, at most 128 moves) and the cost per step grows
with the length, so plan on roughly 2 to 3 times the UCI time. A desktop with 8 cores should finish MNIST
overnight.

## 5. What you get

- `outputs/strokes/RESULTS_<data>.md` and `.tex`: every run, with means and spreads over seeds.
- `outputs/strokes/MEMORY_<data>.md`: the equivalent classical memory of the coherent model.
- `outputs/strokes/DEPLOY_<data>.md`: the circuits under noise, and the threshold of the coherence advantage.
- `outputs/strokes/CLASSES_<data>_tree_chi4_L4_path2_s0.md`: per-digit verdicts, with a figure of real digits
  next to generated ones.
- `outputs/figs/`: sample grids, training curves, deployment curve, digit-by-digit figure.

To share the results, zip `outputs/strokes/*.md`, `outputs/strokes/*.metrics.json` and `outputs/figs/` and send
them. Everything in the paper's tables comes from these files.

## 6. If something fails

- A failed job prints `ERROR` with the reason. Its training log is `outputs/strokes/<run>.log` and its evaluation
  log is `<run>.eval.log`. Rerunning the same command retries only the unfinished jobs.
- `python scripts/paper2.py --data uci --stage decisive --steps 20` runs a very short version of a stage, to check
  that everything works. Use a copy of the folder for this, because it creates runs with the same names as the
  real ones.
