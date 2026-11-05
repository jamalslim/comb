#!/bin/bash
#SBATCH --job-name=comb-k16
#SBATCH --partition=allcpu
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=maxwell/logs/%x_%A_%a.out
# Sixteen levels per value (four work qubits, K = 16). This lowers the readout floor that caps every memory at eight
# levels, so the classical curve keeps falling to larger sizes and the exchange rate has room to grow.
#   task 0: HMM K=16, chi=4, 30 seeds: 3000 steps, then trained to convergence (also the start of the environment memories)
#   task 1: HMM K=16, chi=6 and 8, 10 seeds each, 3000 steps + convergence
#   task 2: HMM K=16, chi=12 and 16          task 3: HMM K=16, chi=24 and 32
#   task 4: CoMB circuit chi=4, four work qubits, eight layers, 30 seeds: 3000 steps + convergence
#   task 5: its dephased twin
# Run names carry _K16 and the summary keeps 16-level runs apart from 8-level runs.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source maxwell/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK=${SLURM_ARRAY_TASK_ID:-0}
hmm_train() {   # chi "seeds"
  local chi=$1 seeds=$2 s n
  for s in $seeds; do n=ds1_hmm${chi}_s${s}_K16_ly; [ -f outputs/runs/$n.npz ] && continue
    python scripts/comb_ds1.py train --model hmm --chi $chi --K 16 --seed $s --layer-share 0.2 --steps 3000 > maxwell/logs/train_$n.log 2>&1 &
  done; wait
  for s in $seeds; do python maxwell/converge.py --family hmm --chi $chi --K 16 --seed $s > maxwell/logs/conv_hmm${chi}_K16_s$s.log 2>&1 & done; wait
}
circ_train() {   # kind family "seeds"
  local kind=$1 fam=$2 seeds=$3 s n
  for s in $seeds; do n=ds1_${kind}_chi4_s${s}_L8_K16_ly; [ -f outputs/runs/$n.npz ] && continue
    python scripts/comb_ds1.py train --model $kind --chi 4 --nw 4 --Lv 8 --seed $s --layer-share 0.2 --steps 3000 > maxwell/logs/train_$n.log 2>&1 &
  done; wait
  for s in $seeds; do python maxwell/converge.py --family $fam --seed $s > maxwell/logs/conv_${fam}_s$s.log 2>&1 & done; wait
}
case $TASK in
  0) hmm_train 4 "$(seq 0 29)";;
  1) hmm_train 6 "$(seq 0 9)"; hmm_train 8 "$(seq 0 9)";;
  2) hmm_train 12 "$(seq 0 9)"; hmm_train 16 "$(seq 0 9)";;
  3) hmm_train 24 "$(seq 0 9)"; hmm_train 32 "$(seq 0 9)";;
  4) circ_train comb comb4K16 "$(seq 0 29)";;
  5) circ_train dephased twin4K16 "$(seq 0 29)";;
esac
