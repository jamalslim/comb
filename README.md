# CoMB: the coherent memory Born machine

Code for the paper *Quantum Generative Modeling with a Coherent Memory, Exact Training and a Memory-Dimension
Advantage* by Jamal Slim, Saverio Monaco, Ran Xue, Dirk Krücker and Kerstin Borras (DESY and RWTH Aachen, 2026).

CoMB generates a sample one component per round on a register whose size does not depend on the dimension of the
data. In each round a work register is entangled with a memory register and measured; the outcome gives one
component. The memory is never measured and carries the correlations between rounds. The characteristic function of
the generated distribution is a product of small matrices, one per round, so MMD losses and their gradients are
computed exactly, at a cost linear in the data dimension, without sampling or likelihood. See `ALGORITHM.md`.

## Layout

| Path | Contents |
| --- | --- |
| `comb/showers/`, `scripts/comb_ds1.py` | calorimeter showers (CaloChallenge Dataset 1): circuits, classical baselines, exact losses, training, evaluation |
| `comb/digits/`, `scripts/comb_strokes.py` | handwritten digits drawn as pen strokes: k-point and path losses, general coherent memory, measurement trees, exact noise |
| `maxwell/` | SLURM scripts of the shower seed studies: 30 seeds per model, all training budgets, convergence, summaries, bootstrap intervals |
| `scripts/` | every script: the drivers `comb_ds1.py` and `comb_strokes.py`, the digits experiments (`paper2.py`), the device runs, the hardware-compatible shower model, analysis, tests |
| `plot/` | every figure of the paper |
| `docs/` | the digits method and experiments, the cure of the double peak, the hardware-compatible model, provenance |

## Installation

Python 3.10 or newer:

    pip install -r requirements.txt

Device runs also need `qiskit`, `qiskit-aer` and `qiskit-ibm-runtime`, and an IBM Quantum account. Save your
token once on your machine with `QiskitRuntimeService.save_account(...)`; never put a token into a file of this
repository.

## Showers in four commands

    python scripts/comb_ds1.py prepare
    python scripts/comb_ds1.py train --model comb --chi 4 --layer-share 0.2 --steps 3000 --seed 0 --tag _quickstart
    python scripts/comb_ds1.py sample --run outputs/runs/ds1_comb_chi4_s0_quickstart_ly.npz
    python scripts/comb_ds1.py plot

`prepare` downloads CaloChallenge Dataset 1 (photons, Zenodo record 8099322, about 0.5 GB) into `data/`; it needs
internet access once. `--layer-share 0.2` adds the layer-resolved terms of the loss, as for every model of the paper,
and the run name then ends in `_ly`. `--tag _quickstart` gives the run its own name, so it never overwrites the trained
models in `outputs/runs/`. `sample` needs the trained file with `--run`.

On one core, training takes about 1.8 s per step (3,000 steps: about an hour and a half), sampling about a minute and
the figures about 30 seconds. The baselines are `--model hmm` (classical hidden Markov model) and `--model dephased`
(the same circuit, its memory dephased after every round).

## Digits

    python scripts/paper2.py --data uci --stage tests      # the exactness tests, about a minute
    python scripts/paper2.py --data uci --list             # the experiments and their status

and then the stages described in `docs/digits_experiments.md`. The UCI digits ship with scikit-learn, so nothing is
downloaded; the MNIST variant (not used in the paper) downloads MNIST, about 11 MB, on first use.

## Reading the code

Start with `ALGORITHM.md` for the method. The two drivers show how the pieces fit together: `scripts/comb_ds1.py` for
the showers (its commands `train` and `sample` are the core) and `scripts/comb_strokes.py` for the digits. The models
are built in `comb/showers/zoo.py` and listed module by module for the digits in `docs/digits_overview.md`.

## Reproducing the paper

| Part of the paper | Code |
| --- | --- |
| Shower seed studies: Tables I to III and V | `maxwell/README.md`: run lists (`maxwell/make_runs.py`), `maxwell/seeds.sbatch`, summary with `maxwell/summarize_seeds.py` |
| Training to convergence (Table II) | `maxwell/converge.sbatch`, which runs `maxwell/converge.py` |
| Equivalent classical memory with 95% intervals | `maxwell/bootstrap_equiv.py` |
| Memory-compression figure of the showers | `maxwell/plot_memory_curve.py` |
| Device run on `ibm_kingston` | `scripts/hardware_run.py` (verify, submit, fetch, decode, complete, plot) |
| The hardware-compatible shower model | `docs/hardware_compatible_model.md` (`scripts/finetune_readout.py`, `scripts/quantile_readout.py`) |
| Digits: every experiment | `docs/digits_experiments.md` (`scripts/paper2.py`) |
| Digits: equivalent memory, jackknife errors, class-conditional quality, device noise | `scripts/memory_equiv.py`, `statistical_errors.py`, `class_quality.py`, `deploy_strokes.py` |
| Every number of the paper | `scripts/make_paper_numbers.py` (writes into `outputs/paper/`) |
| Every figure of the paper | `python plot/make_all.py` |

## Results in this repository

* `outputs/runs/`: the trained shower models behind the figures (CoMB seed 0, classical HMM seed 1), the model executed
  on the device and the hardware-compatible model
* `outputs/hardware/`: the records of both device runs on `ibm_kingston` (jobs `db34tb7r11fs7397r770` and
  `db34m0m8v0ts73c3t580`): the measured counts and the showers completed from them
* `results_reference/`: the digits reference models, their metrics and the result tables

`outputs/README.md` describes every file. The parameters of every seed of the seed studies (30 per model) are not
included; the scripts in `maxwell/` produce them.

## Data

`data/` is filled by the scripts. `python scripts/comb_ds1.py prepare` downloads CaloChallenge Dataset 1 (photons,
Zenodo record 8099322); its raw file (174 MB) is larger than GitHub accepts, so it is not stored here.
`python scripts/comb_strokes.py prepare --data uci` builds the pen strokes from the copy of the UCI digits that ships
with scikit-learn.

## Tests

    python scripts/test_comb.py && python scripts/test_kraus.py && python scripts/test_noise.py && python scripts/test_train.py
    python scripts/paper2.py --data uci --stage tests

The tests compare the code with independent references: brute-force enumeration of every token sequence,
finite-difference gradients, density-matrix simulation and sampling.

## Earlier version

The code of the first version of the paper (arXiv:2609.23050v1), with the six-qubit register and the token
readout, is kept unchanged at the tag `arxiv-v1` of this repository.

## Citation

See `CITATION.cff`.

## Licence

MIT, see `LICENSE`.
