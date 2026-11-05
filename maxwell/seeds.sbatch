#!/bin/bash
#SBATCH --job-name=comb-seeds
#SBATCH --partition=allcpu
#SBATCH --nodes=1
#SBATCH --time=24:00:00
#SBATCH --requeue
#SBATCH --output=maxwell/logs/%x_%A_%a.out
# Trains the runs listed in maxwell/runs.txt ('model seed chi') with the layer-resolved loss, RUNS_PER_NODE at a time
# on one node (one thread each), then computes their Table 1 metrics, SAMPLE_PARALLEL at a time. Set the partition
# above to one you can use (sinfo, or my-partitions). Runs checkpoint every 100 steps and resume after preemption.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"
source maxwell/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
K=${RUNS_PER_NODE:-30}; P=${SAMPLE_PARALLEL:-$K}; TASK=${SLURM_ARRAY_TASK_ID:-0}
name_of() { local d=""; [ "${6:-circuit}" = "dense" ] && d="_dense"; [ "${5:-interleaved}" = "nested" ] && d="${d}_nest"; [ "${4:-5}" != "5" ] && d="${d}_L$4"; case "$1" in hmm) echo "ds1_hmm$3_s$2_ly";; comb) echo "ds1_comb_chi$3_s$2${d}_ly";; dephased) echo "ds1_dephased_chi$3_s$2${d}_ly";; esac; }
RUNS_FILE=${RUNS_FILE:-maxwell/runs.txt}; mapfile -t LINES < "$RUNS_FILE"
start=$(( TASK * K )); end=$(( start + K < ${#LINES[@]} ? start + K : ${#LINES[@]} ))
echo "task $TASK: runs $start to $(( end - 1 )) on $(hostname), $(nproc) cores, $(free -g | awk '/Mem/ {print $2}') GB"
for ((i=start; i<end; i++)); do
  read -r model seed chi lv wiring kind <<< "${LINES[$i]}"; [ -z "${model:-}" ] && continue; chi=${chi:-4}; lv=${lv:-5}; wiring=${wiring:-interleaved}; kind=${kind:-circuit}; dflag=""; [ "$kind" = "dense" ] && dflag="--dense"; name=$(name_of "$model" "$seed" "$chi" "$lv" "$wiring" "$kind")
  [ -f "outputs/runs/$name.npz" ] && { echo "  $name: already trained"; continue; }
  python scripts/comb_ds1.py train --model "$model" --chi "$chi" --Lv "$lv" --wiring "$wiring" $dflag --seed "$seed" --layer-share 0.2 ${EXTRA_TRAIN_ARGS:-} > "maxwell/logs/train_$name.log" 2>&1 &
done
wait
for ((i=start; i<end; i++)); do
  read -r model seed chi lv wiring kind <<< "${LINES[$i]}"; [ -z "${model:-}" ] && continue; chi=${chi:-4}; lv=${lv:-5}; wiring=${wiring:-interleaved}; kind=${kind:-circuit}; dflag=""; [ "$kind" = "dense" ] && dflag="--dense"; name=$(name_of "$model" "$seed" "$chi" "$lv" "$wiring" "$kind")
  [ -f "outputs/samples/$name.metrics.json" ] && continue
  while [ "$(jobs -rp | wc -l)" -ge "$P" ]; do sleep 2; done
  ( python scripts/comb_ds1.py sample --model "$model" --chi "$chi" --Lv "$lv" --wiring "$wiring" $dflag --seed "$seed" --run "outputs/runs/$name.npz" > "maxwell/logs/sample_$name.log" 2>&1 \
      && echo "  $name: done" || echo "  $name: metrics FAILED (see maxwell/logs/sample_$name.log)" ) &
done
wait
