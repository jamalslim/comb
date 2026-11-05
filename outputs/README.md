# outputs

Results kept in the repository. Everything else the scripts write here (samples, figures, new runs) is not tracked.

## runs/: trained shower models

| File | Model |
| --- | --- |
| `ds1_comb_chi4_s0_ly.npz` | CoMB with 4 memory levels, seed 0, 3,000 steps: the representative run of the shower figures |
| `ds1_hmm4_s1_ly.npz` | classical HMM with 4 states, seed 1, 3,000 steps: the classical run of the shower figures |
| `ds1_comb_chi4_s0.npz` | the model whose first 12 rounds ran on `ibm_kingston` |
| `ds1_comb_chi4_s0_hw12r.npz` | the hardware-compatible model: the same first 12 rounds, everything else retrained (`docs/hardware_compatible_model.md`) |
| `*_ax16*`, `*_recal*` | runs of the cure of the double peak (`docs/double_peak_fix.md`) |

Each `.npz` has a `.json` with its specification and training record where one was written.

## hardware/: device records

`ds1_comb_chi4_s0_ibm_kingston_<job>.npz` holds the measured counts of a device run, and `<...>.samples.npz` the
showers completed from them by the exact model. Jobs: `db34tb7r11fs7397r770` (the run in the paper) and
`db34m0m8v0ts73c3t580`.

Load a model with `comb.showers.zoo.load(path)`, or use it directly:

    python scripts/comb_ds1.py sample --run outputs/runs/ds1_comb_chi4_s0_ly.npz
