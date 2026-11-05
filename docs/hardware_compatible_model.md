# A Dataset 1 model that keeps the ibm_kingston runs valid and removes most of the double peak

The device ran the first 12 rounds of the paper's circuit. This model keeps those 12 rounds' circuit exactly, so the
device counts are genuine runs of it (its exact token probabilities equal the records' to 5e-16, both jobs).
Everything else is retrained: rounds 13-369 and the readouts (the classical mapping from tokens to values).
No smoothing, no post-processing of the counts.

## Recipe
    python scripts/finetune_readout.py --run outputs/runs/ds1_comb_chi4_s0.npz --full --freeze-rounds 12 \
           --steps 1100 --lr 0.003 --lr-circuit 0.003 --axis-share 0.2 --out outputs/runs/ds1_comb_chi4_s0_hw12.npz
    python scripts/quantile_readout.py --run outputs/runs/ds1_comb_chi4_s0_hw12.npz --coords 0 \
           --out outputs/runs/ds1_comb_chi4_s0_hw12q.npz
    python scripts/finetune_readout.py --run outputs/runs/ds1_comb_chi4_s0_hw12q.npz --full --freeze-rounds 12 \
           --freeze-readout 0 --steps 600 --lr 0.003 --lr-circuit 0.003 --axis-share 0.2 --out outputs/runs/ds1_comb_chi4_s0_hw12r.npz
    python scripts/comb_ds1.py sample --run outputs/runs/ds1_comb_chi4_s0_hw12r.npz
    python scripts/hardware_run.py complete outputs/runs/ds1_comb_chi4_s0_hw12r.npz \
           outputs/hardware/ds1_comb_chi4_s0_ibm_kingston_db34tb7r11fs7397r770.samples.npz
    python plot/ds1_plots.py --sim ds1_comb_chi4_s0_hw12r --hmm ds1_hmm4_s1_ax16 \
           --device ds1_comb_chi4_s0_hw12r_ibm_kingston --out outputs/paper_figures
    python scripts/comb_ds1.py plot --runs ds1_comb_chi4_s0_hw12r ds1_comb_chi4_s0_hw12r_ibm_kingston      # polar maps
(The HMM ds1_hmm4_s1_ax16 is in comb3_axis_fix.zip. The archive already contains the final model, samples and figures.)

## Results
Layer energies, 20 000 generated showers against all 8 000 Geant4 showers (Geant4 itself: dip 0% / 1%):
| model                      | dip between peaks (layer 1 / 2) | W1 to Geant4 (layer 1 / 2) |
|----------------------------|---------------------------------|----------------------------|
| paper's model              | 63% / 62%                       | 0.160 / 0.147 sd           |
| counts-compatible (this)   | 19% / 1%                        | 0.103 / 0.104 sd           |
| retrained from scratch     | 2% / 5% (but no device data)    | 0.151 / 0.136 sd           |

Table 1, simulation (paper's model in brackets): kappa 88% (86%), Spearman 0.046 (0.050), W1 dense / sparse
0.126 / 0.218 (0.116 / 0.219), zeros 22.8% (22.7%), E spread 1.08 (1.04), largest shape W1 0.32 (0.37), AUC 0.998.
Device-started showers, same record and computation (paper's model in brackets): Spearman err 0.061 (0.070),
W1 0.185 (0.190), E spread 1.23 (1.27). The hardware count plots are unchanged (same circuit).
Limit: a residual bump in layer 1 near 4000 MeV. Round 1 is device-tested, so its 8-token distribution (two tokens
nearly empty) is fixed, and the leading coordinate stays a coarse staircase; only a new device run of a retrained
round 1 removes it.
