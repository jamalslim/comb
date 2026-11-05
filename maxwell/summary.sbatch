#!/bin/bash
#SBATCH --job-name=comb-summary
#SBATCH --partition=allcpu
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=maxwell/summary_%j.txt
# Runs maxwell/summarize_seeds.py on a compute node, scoring the runs in parallel on all its cores. Runs scored before
# are read from their cache files, so an interrupted summary loses nothing. The output goes to maxwell/summary_<jobid>.txt.
cd "${SLURM_SUBMIT_DIR:-.}"; source maxwell/env.sh
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
SUMMARY_WORKERS=$(nproc --all) python maxwell/summarize_seeds.py
