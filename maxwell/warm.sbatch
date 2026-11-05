#!/bin/bash
#SBATCH --job-name=comb-warm
#SBATCH --partition=allcpu
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --array=0-2
#SBATCH --output=maxwell/logs/%x_%A_%a.out
# Warm starts at memory size CHI (default 8), one array task per kind, all seeds of that kind in parallel on one node:
#   task 0: the most general coherent memory (dense) started from each trained circuit run
#   task 1: its dephased twin started from each trained circuit twin
#   task 2: the classical HMM continued from its own trained run (the same extra training, for fairness)
# Each continues for STEPS (default 3000) with 0.1 x the default learning rates; the best step includes step 0, the
# starting point, so a warm run can never report worse than the model it started from. Then the Table 1 metrics.
#   SRC: suffix of the circuit runs to start from (default _nest_L18 for chi = 8; use SRC= for chi = 4)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source maxwell/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
CHI=${CHI:-8}; STEPS=${STEPS:-3000}; SRC=${SRC-_nest_L18}; read lo hi <<< "$(echo ${SEEDS:-0-29} | tr '-' ' ')"; TASK=${SLURM_ARRAY_TASK_ID:-0}
pick() { case $TASK in
  0) src=ds1_comb_chi${CHI}_s$1${SRC}_ly;     name=ds1_comb_chi${CHI}_s$1_dense_warm_ly;     args="--model comb --dense";     m=comb;;
  1) src=ds1_dephased_chi${CHI}_s$1${SRC}_ly; name=ds1_dephased_chi${CHI}_s$1_dense_warm_ly; args="--model dephased --dense"; m=dephased;;
  2) src=ds1_hmm${CHI}_s$1_ly;                name=ds1_hmm${CHI}_s$1_warm_ly;                args="--model hmm";              m=hmm;;
esac; }
echo "task $TASK, chi $CHI, seeds $lo-$hi, $STEPS steps, on $(hostname), $(nproc --all) cores"
for s in $(seq $lo $hi); do
  pick $s; [ -f outputs/runs/$name.npz ] && { echo "  $name: done before"; continue; }
  [ -f outputs/runs/$src.npz ] || { echo "  $name: source $src missing, skipped"; continue; }
  python scripts/comb_ds1.py train $args --chi $CHI --seed $s --layer-share 0.2 --init-from $src --steps $STEPS > maxwell/logs/train_$name.log 2>&1 &
done
wait
for s in $(seq $lo $hi); do
  pick $s; [ -f outputs/runs/$name.npz ] || continue; [ -f outputs/samples/$name.metrics.json ] && continue
  ( python scripts/comb_ds1.py sample --model $m --chi $CHI --seed $s --run outputs/runs/$name.npz > maxwell/logs/sample_$name.log 2>&1 \
      && echo "  $name: done" || echo "  $name: metrics FAILED (see maxwell/logs/sample_$name.log)" ) &
done
wait
