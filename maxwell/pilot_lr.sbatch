#!/bin/bash
#SBATCH --job-name=comb-pilot
#SBATCH --partition=allcpu
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=maxwell/logs/%x_%j.out
# Pilot: does a smaller circuit learning rate let the 18-layer CoMB (chi = 8) train like the 5-layer ones?
# Seeds 0-3 at learning rates 0.05 (= 0.1 * sqrt(5/18)) and 0.03; run names carry the tag _lr0.05 / _lr0.03, so they
# never collide with the main 18-layer batch (learning rate 0.1). Compare the validation curves at equal steps.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source maxwell/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
for lr in 0.05 0.03; do for s in 0 1 2 3; do
  python scripts/comb_ds1.py train --model comb --chi 8 --Lv 18 --seed $s --layer-share 0.2 --lr $lr --tag _lr$lr > maxwell/logs/train_ds1_comb_chi8_s${s}_L18_lr${lr}_ly.log 2>&1 &
done; done
wait
