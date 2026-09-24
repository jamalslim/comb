# CoMB

Code for a coherent-memory Born machine: a sequential quantum generative
model that produces calorimeter shower images one block at a time. Six
qubits, of which three are measured and reset at each step and three are
never measured and carry the dependence between steps.

## Install

```bash
git clone https://github.com/jamalslim/comb && cd comb
python -m venv .venv && source .venv/bin/activate
pip install -e ".[hardware,plots]"
```

`[hardware]` installs Qiskit and is only needed to run on a quantum
device. `[plots]` installs mplhep for the figure style.

Every command below is run from the repository root.

## Train and evaluate

```bash
python scripts/run_d12.py prep          # ~3 min
python scripts/run_d12.py quantum M3C0-L6-E450-P60   # ~20 min, the main model
python scripts/run_d12.py quantum M0C0  # ~2 min
DEPTH=8 python scripts/train_two_stage.py   # ~45 min, the two-stage model
python scripts/run_d12.py report
```

`prep` fits the tokenizer and the classical baselines. The first
`quantum` line trains the main model: three memory qubits, depth six, 450
epochs, stopping early if validation stalls for 60. `quantum M0C0` trains
the control with no memory register. `train_two_stage.py` trains the
table's two-stage model: the blockwise score for 450 epochs, then a
150-epoch fine-tune of the joint MMD against the data. `report` prints the results
table.

Train a single configuration by name:

```bash
python scripts/run_d12.py quantum M3C0-L6
```

Names are `M{memory qubits}C{0|1}[-L{depth}][-E{epochs}][-P{patience}]`, so `M3C0-L6`
has three memory qubits at depth six and `M0C0` has none. Training is
seeded; a rerun reproduces the same parameters.

## Checks

```bash
python scripts/dla.py                   # ~4 min
python scripts/audit_estimators.py      # ~6 min
```

`dla.py` computes the dynamical Lie algebra of the circuit by exact
integer arithmetic, for the ring, line and star entanglers, and exits
non-zero unless all three generate the full algebra.

`audit_estimators.py` checks the two sample-based gradient estimators,
parameter shift and score function, against the exact gradient. Neither
is used for training; they are what replaces the exact gradient once the
number of possible symbol sequences is too large to enumerate. Averaged
over more and more draws, an unbiased estimator must converge to the
exact gradient, and the script exits non-zero if either one does not.

## Figures

```bash
python scripts/plot_paper.py
python scripts/make_three_series.py
```

Both write to `figures/`. `make_three_series.py` overlays the data, the
noiseless simulator and the hardware run, all from the same parameters.

## Run on IBM hardware

Save your credentials once:

```python
from qiskit_ibm_runtime import QiskitRuntimeService
QiskitRuntimeService.save_account(channel="ibm_quantum",
                                  token="YOUR_TOKEN", overwrite=True)
```

Then:

```bash
python scripts/run_ibm.py verify          # required
python scripts/run_ibm.py submit ibm_kingston
python scripts/run_ibm.py analyze
```

`verify` runs the circuit on the Aer simulator and compares the result
against the model's exact output distribution. If they disagree, the
circuit or the bit decoding is wrong, and `submit` will refuse to run.

To use a different checkpoint or depth:

```bash
DEPTH=6 THETA=theta_M3C0-L6-E450-P60.npy \
  python scripts/run_ibm.py submit ibm_kingston
```

## Files you should not overwrite

Two files in `outputs/` are kept in the repository because they cannot be
regenerated:

- `theta_M3C0.npy`: the parameters the hardware job ran
- `ibm_run_M3W3L2.npz`: the measured tokens, job id, device and readout
  calibration from that run

Everything else in `outputs/` and `figures/` is produced by the scripts
and ignored by git.

## Troubleshooting

**`verify` fails.** Do not submit. The circuit or the bit decoding is
wrong. Qiskit returns bitstrings with qubit 0 on the right and classical
registers in reverse order of declaration; `counts_to_tokens` and
`samplerv2_tokens` in `src/comb/qiskit_hw.py` must agree on that.

**The two-qubit gate count after routing is high.** `submit` prints a
warning when routing adds more than 30% to the logical count, but it does
not stop you. The line entangler routes on heavy-hex without swaps; a ring needs them.

**A plotting script cannot find a parameter file.** It names the missing
file. Regenerate it with the matching `run_d12.py quantum` command.

**Which decoder turns tokens into pixels.** `detok_mode="causal"`, set in
`src/comb/config.py`, is the emission described in the paper: each block
is drawn from its own token, the previous block's token, and the previous
block's generated values, one block at a time. It cannot connect blocks
that are not adjacent, so any correlation between them comes from the
quantum token chain. Other modes exist for comparison only.

## Data

`data/cal_shower_img_12q.npy`: 47682 simulated electromagnetic showers,
downsampled to 12 cells.
Zenodo, doi:10.5281/zenodo.16027525.

## License

MIT. See `LICENSE`. To cite, see `CITATION.cff`.
