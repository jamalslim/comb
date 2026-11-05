#!/bin/bash
#SBATCH --job-name=comb-env
#SBATCH --partition=allcpu
#SBATCH --nodes=1
#SBATCH --time=3-00:00:00
#SBATCH --requeue
#SBATCH --array=0-5
#SBATCH --output=maxwell/logs/%x_%A_%a.out
# The theory-consistent test of coherence. The general coherent memory with an environment E = chi^2 contains every
# classical chi-state HMM (Proposition 2), and its dephased twin is exactly the general classical HMM class in the same
# parameterization. Both start from the same converged classical HMM by exact embedding and are trained further with
# identical settings, so any difference between them is due to coherence alone.
#   tasks 0-3: chi=4 coherent, chi=4 twin, chi=8 coherent, chi=8 twin; SEEDS (default 0-29), STEPS (default 3000), LR (optional)
#   K=16 selects the 16-level runs (sources ds1_hmm{chi}_s{seed}_K16_conv_ly from k16.sbatch); default K=8
#   tasks 4-5: learning-rate scan at chi=4 (coherent, twin), seeds 0-2, LR in 0.0015 0.005 0.015; read with
#              grep -h "done:" maxwell/logs/train_ds1_env*_lr*.log
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source maxwell/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
STEPS=${STEPS:-3000}; K=${K:-8}; kt=""; [ "$K" = 8 ] || kt="_K$K"; read lo hi <<< "$(echo ${SEEDS:-0-29} | tr '-' ' ')"; TASK=${SLURM_ARRAY_TASK_ID:-0}
lrflag=""; [ -n "${LR:-}" ] && lrflag="--lr $LR"
run() {   # model chi seed tag extra-args
  local m=$1 chi=$2 s=$3 tag=$4; shift 4; local E=$((chi*chi))
  local src=ds1_hmm${chi}_s${s}${kt}_conv_ly name=ds1_${m}_chi${chi}_E${E}_s${s}${kt}_warm${tag}_ly
  [ -f outputs/runs/$name.npz ] && return
  [ -f outputs/runs/$src.npz ] || { echo "  $name: source $src missing"; return; }
  python scripts/comb_ds1.py train --model $m --chi $chi --seed $s --layer-share 0.2 --K $K --init-from $src --steps $STEPS --tag "$tag" "$@" > maxwell/logs/train_$name.log 2>&1 &
}
case $TASK in
  0) for s in $(seq $lo $hi); do run envmem 4 $s "" $lrflag; done;;
  1) for s in $(seq $lo $hi); do run envdeph 4 $s "" $lrflag; done;;
  2) for s in $(seq $lo $hi); do run envmem 8 $s "" $lrflag; done;;
  3) for s in $(seq $lo $hi); do run envdeph 8 $s "" $lrflag; done;;
  4|5) m=$([ $TASK = 4 ] && echo envmem || echo envdeph)
       for lr in 0.0015 0.005 0.015; do for s in 0 1 2; do run $m 4 $s "_lr${lr/./}" --lr $lr; done; done;;
esac
wait
if [ $TASK -le 3 ]; then
  m=$([ $((TASK % 2)) = 0 ] && echo envmem || echo envdeph); chi=$([ $TASK -le 1 ] && echo 4 || echo 8); E=$((chi*chi))
  for s in $(seq $lo $hi); do
    name=ds1_${m}_chi${chi}_E${E}_s${s}${kt}_warm_ly
    [ -f outputs/runs/$name.npz ] || continue; [ -f outputs/samples/$name.metrics.json ] && continue
    python scripts/comb_ds1.py sample --model $m --chi $chi --seed $s --run outputs/runs/$name.npz > maxwell/logs/sample_$name.log 2>&1 &
  done
  wait
fi
