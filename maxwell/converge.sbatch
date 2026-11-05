#!/bin/bash
#SBATCH --job-name=comb-conv
#SBATCH --partition=allcpu
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --array=0-16
#SBATCH --output=maxwell/logs/%x_%A_%a.out
# Trains every model to convergence with one rule (maxwell/converge.py): continue in blocks of 3000 steps until a block
# improves the best validation loss by less than 1%, at most 4 extra blocks. One array task per family, all its seeds in
# parallel on one node. Finished blocks are skipped on restart, so a requeued or resubmitted job continues where it was.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source maxwell/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TASKS=("comb4 0-29" "twin4 0-29" "comb8L18 0-29" "twin8L18 0-29" "dense4 0-29" "dtwin4 0-29" "dense8 0-29" "dtwin8 0-29"
       "hmm:4 0-29" "hmm:8 0-29" "hmm:6 0-9" "hmm:12 0-9" "hmm:16 0-9" "hmm:24 0-9" "hmm:32 0-9" "hmm:48 0-9" "hmm:64 0-9")
read -r fam seeds <<< "${TASKS[${SLURM_ARRAY_TASK_ID:-0}]}"; read lo hi <<< "$(echo $seeds | tr '-' ' ')"
chi=""; [[ $fam == hmm:* ]] && { chi="--chi ${fam#hmm:}"; fam=hmm; }
echo "task ${SLURM_ARRAY_TASK_ID:-0}: family $fam $chi, seeds $lo-$hi, on $(hostname), $(nproc --all) cores"
for s in $(seq $lo $hi); do
  python maxwell/converge.py --family $fam $chi --seed $s > maxwell/logs/conv_${fam}${chi// /}_s$s.log 2>&1 &
done
wait
for s in $(seq $lo $hi); do
  n=$(python maxwell/converge.py --family $fam $chi --seed $s --print-name)
  [ -f outputs/runs/$n.npz ] || continue; [ -f outputs/samples/$n.metrics.json ] && continue
  case $n in ds1_hmm*) m=hmm;; ds1_dephased*) m=dephased;; *) m=comb;; esac
  python scripts/comb_ds1.py sample --model $m --seed $s --run outputs/runs/$n.npz > maxwell/logs/sample_$n.log 2>&1 &
done
wait
grep -h "converged\|failed\|missing" maxwell/logs/conv_${fam}*_s*.log
