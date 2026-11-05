# Many seeds on Maxwell (layer-resolved loss)

Two batches, each with its own run list (already prepared in this folder):
- maxwell/runs_main.txt       88 runs: CoMB, its dephased twin and the HMM (chi = 4), seeds 0-29 (2 already done)
- maxwell/runs_hmm_curve.txt  40 runs: HMMs with chi = 6, 8, 12, 16, seeds 0-9 (the classical memory curve)

Every run checkpoints every 100 steps and resumes, so preempted or requeued jobs continue where they stopped. Each
run uses one core and well under 1 GB of memory; RUNS_PER_NODE runs train at once on a node, then their metrics are
computed SAMPLE_PARALLEL at a time (default: the same number).

1. One-time Python environment (edit maxwell/env.sh if you keep it elsewhere):
       python3 -m venv ~/venvs/comb && source ~/venvs/comb/bin/activate
       pip install numpy scipy scikit-learn matplotlib mplhep
2. Set the partition in maxwell/seeds.sbatch to one you can use (sinfo, or my-partitions). Time limit: 24 h; the
   larger HMMs (chi = 12, 16) are the slowest runs.
3. Submit both batches (use RUNS_PER_NODE up to the number of cores of a node):
       RUNS_FILE=maxwell/runs_main.txt      RUNS_PER_NODE=96 sbatch --array=0-0 maxwell/seeds.sbatch
       RUNS_FILE=maxwell/runs_hmm_curve.txt RUNS_PER_NODE=40 sbatch --array=0-0 maxwell/seeds.sbatch
   With fewer cores per node, regenerate a list with a smaller --per-node and use the array range it prints, e.g.
       python maxwell/make_runs.py --seeds 0-29 --per-node 48 --list maxwell/runs_main.txt
   Watch with squeue -u $USER; logs are in maxwell/logs/.
4. Summarize: python maxwell/summarize_seeds.py
   Per model and HMM size: validation and held-out test MMD^2 and the Table 1 metrics (mean +- standard error);
   CoMB against the HMM (chi = 4) and its dephased twin in standard errors; the equivalent classical memory.
Quote the held-out test MMD^2: the validation MMD^2 also chose each run's best step.

## Larger memory: CoMB with chi = 8 (three memory qubits)

Generate the lists on Maxwell (runs that already exist are skipped), then submit with the array range printed:
    python maxwell/make_runs.py --seeds 0-29 --comb-chis 8 --hmm-chis 8 --per-node 30 --list maxwell/runs_chi8.txt
    python maxwell/make_runs.py --seeds 0-9 --models hmm --hmm-chis 24 32 48 64 --per-node 40 --list maxwell/runs_hmm_large.txt
    RUNS_FILE=maxwell/runs_chi8.txt      RUNS_PER_NODE=30 sbatch --array=0-2 maxwell/seeds.sbatch
    RUNS_FILE=maxwell/runs_hmm_large.txt RUNS_PER_NODE=40 sbatch --array=0-0 maxwell/seeds.sbatch
Timing (one free core): CoMB chi = 8 about 7.9 s per step (6.6 h for 3000 steps; expect 8-10 h on a shared node),
HMM chi = 64 about 5.8 s per step (4.8 h). The summary then reports, for every CoMB size, the comparison with the HMM
and the twin of the same size and the equivalent classical memory; plot_memory_curve.py draws all sizes.

## Circuit depth scaled with the memory (chi = 8, 18 layers)

With 5 layers, the round circuit reaches about 24% of the most general memory update at chi = 4 (58 of 240 angles) but
only about 7.5% at chi = 8 (72 of 960). With --Lv 18 it reaches about 24% again at chi = 8 (228 angles). Runs with a
depth other than 5 carry the suffix _L<depth> (ds1_comb_chi8_s0_L18_ly); default runs keep their names.

Once the chi = 8 runs of the previous section exist in outputs/runs/, this shows how much of its memory each model uses:

    python maxwell/memory_usage.py ds1_comb_chi4_s0_ly ds1_comb_chi8_s0_ly ds1_comb_chi8_s1_ly ds1_dephased_chi8_s0_ly

Then train the 18-layer circuits:

    python maxwell/make_runs.py --seeds 0-29 --models comb dephased --comb-chis 8 --Lv 18 --per-node 30 --list maxwell/runs_chi8_L18.txt
    RUNS_FILE=maxwell/runs_chi8_L18.txt RUNS_PER_NODE=30 sbatch --array=0-1 maxwell/seeds.sbatch
memory_usage.py reports how many memory levels and independent memory states a trained model actually uses.

## Nested wiring (chi = 8 that contains chi = 4)

With the default interleaved wiring, a third memory qubit is inserted into the line between w1 and w2
(m0-w0-m1-w1-m2-w2): the chi = 8 circuit then does not contain the chi = 4 circuit, and w2 is coupled only through m2.
With --wiring nested, extra memory qubits are appended after the work register (m0-w0-m1-w1-w2-m2): a trained chi = 4
CoMB placed in it with m2 idle reproduces its exact characteristic function to 3e-17. For one or two memory qubits both
wirings are identical. Runs with nested wiring carry the suffix _nest (ds1_comb_chi8_s0_nest_ly).
    python maxwell/make_runs.py --seeds 0-29 --models comb dephased --comb-chis 8 --wiring nested --per-node 30 --list maxwell/runs_chi8_nest.txt
    RUNS_FILE=maxwell/runs_chi8_nest.txt RUNS_PER_NODE=30 sbatch --array=0-1 maxwell/seeds.sbatch

## The most general coherent memory (--dense)

With --dense, the round circuit is replaced by a dense unitary exp(-iH) with H an arbitrary Hermitian matrix on the
whole register (D^2 real parameters per round, 4096 for chi = 8), so the model is the most general coherent memory of
dimension chi; the dephased twin of it is the most general classical-like memory of the same construction. Default
learning rate 0.1/sqrt(D) and initial spread scaled by 1/sqrt(D). Runs carry the suffix _dense (ds1_comb_chi8_s0_dense_ly).
    python maxwell/make_runs.py --seeds 0-29 --models comb dephased --comb-chis 4 8 --dense --per-node 30 --list maxwell/runs_dense.txt
    RUNS_FILE=maxwell/runs_dense.txt RUNS_PER_NODE=30 sbatch --array=0-3 maxwell/seeds.sbatch
Cost: about the 18-layer circuit's per step (chi = 8 about 6-9 s per step on a shared node); chi = 4 much less.

## Warm starts (is the circuit the bottleneck?)

The dense model trained from scratch was still improving at step 3000 (undertrained). A warm start instead begins
exactly at a trained circuit's solution (each round's unitary written as exp(-iH) with H = i log U; the characteristic
functions agree to 1e-17 before the first step) and continues for 3000 steps with 0.1 x the default learning rates.
The best step includes step 0, so a warm run never reports worse than its start. The HMM is continued for the same
number of steps from its own trained runs, for fairness. Runs carry the suffix _warm.
    sbatch maxwell/warm.sbatch                         # chi = 8: from the nested 18-layer runs (3 tasks, 30 seeds each)
    CHI=4 SRC= sbatch maxwell/warm.sbatch              # chi = 4: from the 5-layer runs

## Training every model to convergence

Fixed budgets of 3000 steps leave models unconverged (continuing the HMMs improved them by about 30%). converge.py
continues a trained run in blocks of 3000 steps, with 0.1 x the default learning rates, until a block improves the best
validation loss by less than 1% (at most 4 extra blocks), and saves the final model as <run>_conv_ly. The summary groups
these as "conv" and, when converged HMMs of at least two sizes exist, builds the classical curve from them.
    sbatch maxwell/converge.sbatch              # 17 tasks: circuits, twins, general memories and HMMs of every size
    grep -h "converged\|failed\|missing" maxwell/logs/conv_*.log | sort | uniq -c | head
    sbatch maxwell/summary.sbatch               # when the chains are done
Interrupted or requeued chains continue where they stopped; resubmitting the job is always safe.

## Equal budget of 6000 steps (simpler than convergence)

Every model gets the same training, 3000 steps plus a continuation of 3000 (0.1 x the default learning rates). The HMMs
at 4 and 8 states and the general memory already have it; warm6000.sbatch adds the circuit CoMB and its twin (chi = 4,
and chi = 8 nested with 18 layers) and the HMMs of every other size. The summary compares models only with models of the
same budget, and reads each equivalent memory from the classical curve of the same budget.
    sbatch maxwell/warm6000.sbatch       # 8 tasks; about 6 hours, 11 for the 64-state HMMs
    sbatch maxwell/summary.sbatch        # afterwards

## The theory-consistent test of coherence: memory with an environment (env.sbatch)

Proposition 2: a coherent memory with an environment of dimension E >= chi^2 contains every classical chi-state HMM.
Circuits and the environment-free dense memory do not (they are Born machines, incomparable with HMMs at equal chi).
comb/envmem.py implements the general memory with an environment (polar-parameterized isometry, exact gradient), its
dephased twin (which for E >= chi^2 is exactly the general classical HMM class, same parameterization and optimizer), and
exact embeddings of a trained HMM or circuit. env.sbatch starts both models from the SAME converged HMM
(ds1_hmm{chi}_s{seed}_conv_ly) by exact embedding (characteristic functions agree to 1e-17) and trains them further with
identical settings: any difference is due to coherence alone, and neither can end below its classical start.
    sbatch maxwell/env.sbatch                        # tasks 0-3: chi=4 and chi=8, coherent and twin, 30 seeds; 4-5: lr scan
    grep -h "done:" maxwell/logs/train_ds1_env*_lr*.log   # the scan: pick the learning rate with the best validation
    LR=<best> SEEDS=0-29 sbatch --array=0-3 maxwell/env.sbatch   # if the scan prefers another rate (rerun after deleting runs)
    sbatch maxwell/summary.sbatch                    # groups "EnvMem E=16 warm chi=4" and "EnvMem dephased E=16 warm chi=4"

## Sixteen levels per value (k16.sbatch, then env.sbatch with K=16)

Eight levels per value set a readout floor (about 1.08e-4 at convergence) that no memory can pass; it caps the
advantage at eight memory levels and limits it at four. k16.sbatch trains the 16-level versions (four work qubits):
the classical curve (HMMs with 4 to 32 states, trained to convergence), the chi=4 circuit with eight layers and its
twin (3000 steps + convergence). env.sbatch with K=16 then starts the environment memories from the converged 16-level
HMM. Run names carry _K16 and the summary never mixes the two readouts.
    j=$(sbatch --parsable maxwell/k16.sbatch)                                   # tasks 0-5
    K=16 sbatch --dependency=afterok:${j}_0 --array=0-1,4-5 maxwell/env.sbatch  # chi=4 environment memories + lr scan, after task 0
    sbatch maxwell/summary.sbatch                                               # groups "... K=16 ..." with their own classical curve
