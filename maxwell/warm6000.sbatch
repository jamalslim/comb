#!/bin/bash
#SBATCH --job-name=comb-6000
#SBATCH --partition=allcpu
#SBATCH --nodes=1
#SBATCH --time=2-00:00:00
#SBATCH --requeue
#SBATCH --array=0-7
#SBATCH --output=maxwell/logs/%x_%A_%a.out
# Equal training budget for every model: continue each 3000-step run for 3000 more steps (comb_ds1.py --init-from,
# 0.1 x the default learning rates, as for the HMM and general-memory runs that already have 6000 steps). Outputs are
# named ..._warm_ly. One array task per family, all seeds in parallel; runs already continued are skipped.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-.}"; source maxwell/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
TASK=${SLURM_ARRAY_TASK_ID:-0}
run() {   # model-args source-template output-template chi seeds...
  local args="$1" src="$2" out="$3" chi="$4"; shift 4
  for s in "$@"; do
    local S=${src//\{s\}/$s} O=${out//\{s\}/$s}
    [ -f outputs/runs/$O.npz ] && { echo "  $O: done before"; continue; }
    [ -f outputs/runs/$S.npz ] || { echo "  $O: source $S missing"; continue; }
    python scripts/comb_ds1.py train $args --chi $chi --seed $s --layer-share 0.2 --init-from $S --steps 3000 > maxwell/logs/train_$O.log 2>&1 &
  done
}
case $TASK in
  0) run "--model comb"     ds1_comb_chi4_s{s}_ly     ds1_comb_chi4_s{s}_warm_ly     4 $(seq 0 29);;
  1) run "--model dephased" ds1_dephased_chi4_s{s}_ly ds1_dephased_chi4_s{s}_warm_ly 4 $(seq 0 29);;
  2) run "--model comb --Lv 18 --wiring nested"     ds1_comb_chi8_s{s}_nest_L18_ly     ds1_comb_chi8_s{s}_nest_L18_warm_ly     8 $(seq 0 29);;
  3) run "--model dephased --Lv 18 --wiring nested" ds1_dephased_chi8_s{s}_nest_L18_ly ds1_dephased_chi8_s{s}_nest_L18_warm_ly 8 $(seq 0 29);;
  4) for c in 6 12 16; do run "--model hmm" ds1_hmm${c}_s{s}_ly ds1_hmm${c}_s{s}_warm_ly $c $(seq 0 9); done;;
  5) for c in 24 32; do run "--model hmm" ds1_hmm${c}_s{s}_ly ds1_hmm${c}_s{s}_warm_ly $c $(seq 0 9); done;;
  6) run "--model hmm" ds1_hmm48_s{s}_ly ds1_hmm48_s{s}_warm_ly 48 $(seq 0 9);;
  7) run "--model hmm" ds1_hmm64_s{s}_ly ds1_hmm64_s{s}_warm_ly 64 $(seq 0 9);;
esac
wait
for f in $(ls -t outputs/runs/*_warm_ly.npz 2>/dev/null); do
  n=$(basename $f .npz); [ -f outputs/samples/$n.metrics.json ] && continue
  find maxwell/logs -name "train_$n.log" -newer maxwell/warm6000.sbatch | grep -q . || continue
  case $n in ds1_hmm*) m=hmm;; ds1_dephased*) m=dephased;; *) m=comb;; esac
  s=$(echo $n | sed 's/.*_s\([0-9]*\)_.*/\1/')
  python scripts/comb_ds1.py sample --model $m --seed $s --run $f > maxwell/logs/sample_$n.log 2>&1 &
  while [ "$(jobs -rp | wc -l)" -ge "$(nproc --all)" ]; do sleep 2; done
done
wait
