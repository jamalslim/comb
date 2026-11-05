# Removing the double peaks in the layer-1 and layer-2 energies (Dataset 1)

## Cause
Each generated coordinate is one of 8 readout levels. The training loss (characteristic-function MMD with a Gaussian
mixture kernel whose bandwidths are of the order of the median distance between showers in 369 dimensions) only sees
the mean and width of each coordinate. It did not stop the circuit and readout from leaving gaps and narrow, crowded
bins in the leading principal component, which controls the split between layers 1 and 2. Result: two peaks.

## Fix (no smoothing, no post-processing)
The same architecture, data and protocol (5 qubits, 8 levels, 3000 steps, best validation), with one change to the
loss: 20% of the random frequencies point along the 16 leading principal axes, at a scale that resolves one readout
bin. The loss stays an exact MMD (a sum of kernels is a kernel); gradients use the existing exact code.

    python scripts/comb_ds1.py train --model comb --chi 4 --seed 0 --axis-share 0.2 --axis-k 16
    python scripts/comb_ds1.py train --model hmm  --chi 4 --seed 1 --axis-share 0.2 --axis-k 16
    python scripts/comb_ds1.py spikes --run outputs/runs/ds1_comb_chi4_s0_ax16.npz      # readout diagnostic
    python scripts/comb_ds1.py sample --run outputs/runs/ds1_comb_chi4_s0_ax16.npz      # Table 1 metrics

Without --axis-share the script trains exactly as before (the paper's loss).

## Results (20 000 generated showers against all 8 000 Geant4 showers)
| model                        | dip between humps (layer 1 / 2) | W1 to Geant4 (layer 1 / 2) | val. MMD^2 |
|------------------------------|---------------------------------|----------------------------|------------|
| CoMB, paper (old loss)       | 63% / 62%                       | 0.160 / 0.147 sd           | 1.84e-4    |
| CoMB, axis-resolved loss     | 2% / 5%                         | 0.151 / 0.136 sd           | 2.25e-4    |
| HMM, axis-resolved loss      | 1% / 0%                         | 0.149 / 0.119 sd           | 2.35e-4    |

Table 1 metrics, CoMB axis-resolved (paper's CoMB in brackets): kappa 84% (86%), Spearman 0.052 (0.050),
W1 dense / sparse 0.118 / 0.217 (0.116 / 0.219), zeros 23.1% (22.7%), E spread 1.09 (1.04),
largest shape W1 0.36 (0.37), classifier AUC 0.999 (1.000). HMM axis-resolved: kappa 89%, Spearman 0.046,
W1 0.120 / 0.215, zeros 23.4%, E spread 1.13, AUC 0.998.

Remaining differences are in the tails (too few layer-1 energies below ~3000 MeV, too few layer-2 energies above
~60 000 MeV), not peaks. The retrained CoMB has a new circuit: the ibm_kingston run belongs to the paper's model.
The validation MMD^2 above is the paper's (old) kernel, which cannot see the peaks.
