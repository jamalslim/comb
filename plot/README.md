# plot: every figure of the CoMB paper

This folder goes inside the comb folder. The figures are drawn from the data in the comb folder around it, so
there are no paths to give:

    cd comb
    pip install mplhep                                   # the ROOT style; without it the same settings are applied directly
    python plot/make_all.py                              # every figure -> outputs/paper_figures/
    python plot/make_all.py --out ../comb_paper/figures  # or straight into the paper's figures folder

Single parts:

    python plot/ds1_plots.py                             # calorimeter showers
    python plot/digits_plots.py --data mnist             # handwritten digits (also --data uci)

`--style` picks the mplhep style (ROOT by default; ATLAS, CMS, LHCb2, ALICE). Figures are sized for the paper:
one column is 3.4 in and two columns are 7.0 in. Every figure whose runs exist is drawn; the others are skipped
with a message that names the missing file and the command that makes it. Each script prints the folder it reads
from and the folder it writes to.

Every plot is checked when it is saved: overlapping texts, texts outside the plot, and legends or annotations on
top of data are reported with the plot's name (OVERLAP in ...).

## Error bars

Data points carry statistical errors and are never joined by lines. Held-out results (memory compression, circuits,
noise) take their errors from `outputs/strokes/ERRORS_<data>.json`, a jackknife over the test digits written by
`python scripts/statistical_errors.py --data <data>` (copy the script into `comb/scripts/`; `make_all.py` runs it once
when the file is missing). All models are scored on the same test digits, so comparisons between models are shown
as paired differences with paired errors. Recognition rates carry binomial errors, and AUCs Hanley-McNeil errors.

## What is read from the comb folder

| data | written by |
|---|---|
| `outputs/samples/_test.npz` (Geant4 test showers), `outputs/samples/ds1_*.npz` (model showers) | `python scripts/comb_ds1.py sample` |
| `outputs/samples/<run>_<backend>.npz` (device-started showers) | `python scripts/hardware_run.py complete` |
| `outputs/hardware/*.samples.npz` (device record) | `python scripts/hardware_run.py` |
| `outputs/runs/*.json` (training histories) | training |
| `data/strokes_<data>.npz`, `data/mnist.npz`, `outputs/strokes/*` (digits) | `comb_strokes.py` / `scripts/paper2.py` |

If the digits runs were made in a separate comb2 folder, a `comb2` folder inside or next to the comb folder is
found automatically, or give it with `--comb2`.

## The plots

Every plot is a single plot in its own file, without a title, drawn at the size it is printed (one column, 3.375 in,
or a third of the page, 2.25 in). The paper places them side by side with (a), (b), (c) underneath.

| files | content |
|---|---|
| `ds1_corr_{geant4,sim,device}` | voxel correlation matrices: Geant4, CoMB in simulation, device-started CoMB |
| `ds1_energy_{total,layer0,layer1,layer2,layer3,layer12}` | total and layer energies, each with its ratio panel |
| `ds1_marginal_01` ... `_12` | twelve voxel spectra across the detector (voxel named on the axis), with ratio panels |
| `ds1_radial_layer{0,1,2,3,12}` | mean energy per voxel against the radial bin, per layer, with ratio panels |
| `ds1_hardware_tv`, `ds1_hardware_counts_{first,last}` | the device run: TV per round; count distributions of the first and last rounds |
| `ds1_training` | validation MMD² of CoMB and the HMM (one column) |
| `<data>_samples_{test,strokes,comb}` | test digits, the same digits as pen strokes, CoMB samples (one row per digit) |
| `<data>_statistics_{mi,penmoves,height,turning}` | dependence between pen moves against distance; stroke length, height, turning |
| `<data>_pixcorr_{test,coherent,conditional,dephased,hmm,tree}` | pixel-pixel correlation matrices of rendered digits (14 x 14, scan order): test digits and every model |
| `<data>_memory_compression` | held-out objective against memory size: classical HMMs of 4 to 16 states, the coherent memory and its dephased twin at 4 |
| `<data>_circuits_{total,local}` | every circuit next to its dephased twin, with the coherent memory and the classical HMM as reference lines |
| `<data>_realism` | two-sample AUC of every model; the class-conditional CoMB against the unconditional models |
| `<data>_recognition`, `<data>_training` | recognition per digit; training curves |
