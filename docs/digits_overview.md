# Digits: exactly trainable coherent-memory generators (CoMB) on handwritten digits drawn as pen strokes

CoMB generates sequences with a register of fixed size. A memory register is never measured; a work register
is measured and reset once per emitted token. The memory carries the history as a quantum state, and one run
of the circuit produces one complete sample.

The central identity is that, for any statistic whose weight factorises over rounds,

    E[ prod_t w_t(c_t) ] = Tr[ A_d ... A_1 (rho_0) ],    A_t = sum_c w_t(c) Lambda_{t,c},

where Lambda_{t,c} is the quantum instrument of round t. This gives losses that are computed exactly, without
sampling the model, together with exact gradients, at a cost linear in the sequence length. The application
in this package is handwritten digits drawn as pen strokes: 16 tokens per move, 8 directions times pen up or
down, emitted by 4 work qubits.

## Layout

```
(root of the repository)
  scripts/comb_strokes.py   driver: prepare | train | evaluate | plot
  comb/digits/
    circuit.py      layered circuits (RY, RZ, CZ) with exact adjoint; BatchedCircuit (all rounds in one pass)
    model.py        CountChain base class; CoMB circuit model (pure or environment work qubits, dephased twin);
                    classical HMM; batched instruments and their adjoint
    train.py        Adam and the learning-rate schedule
    mmd.py          kernel spectra and characteristic-function utilities (used by train.py)
    noise.py        exact noise model (gate depolarizing, reset, readout, idle T1/T2); NoisyCoMB
    strokes.py      MNIST loader with integrity checks; skeleton -> covering pen walk -> tokens; rendering;
                    geometry descriptors
    kpoint.py       exact k-point correlation MMD (windows of 1-3 moves and pairs at chosen lags), its exact
                    gradient, and training with checkpoint resume
    pencf.py        exact path MMD (characteristic function of the pen's positions) and its gradient
    hqmm.py         general coherent memory: an unrestricted isometry per round (= CoMB with an arbitrarily
                    deep circuit), with environment levels and a dephased twin
    treecomb.py     measurement-tree CoMB: one work qubit, log2(K) binary sub-rounds with feed-forward;
                    compile_tree (Theorem 5) and compose_tree
    noisytree.py    exact noisy tree-CoMB (readout errors propagate through feed-forward; idling per node)
  scripts/
    paper2.py           orchestrator: every experiment of the paper, resumable, parallel (--workers)
    strokes_table.py    results table (markdown and LaTeX), mean +- sd over seeds
    memory_equiv.py     equivalent classical memory of the 4-level coherent memory
    deploy_strokes.py   exact noisy evaluation (trapped-ion and Heron-like models, noise sweep, threshold)
    class_quality.py    class-conditional quality gate (does each digit's model draw that digit?)
    test_strokes.py     test_kpoint.py  test_pencf.py  test_hqmm.py  test_treecomb.py  test_noisytree.py
  results_reference/        in the release asset comb_trained_parameters_v1.0.0.tar.gz
    RESULTS_uci.md/.tex, DEPLOY_uci.md   results obtained so far on the UCI digits
    metrics/   the metrics and training records of every run so far
    models/    the trained parameters of those runs
    figures/   samples of the decisive comparison and its training curves
  docs/
    digits_overview.md    this file
    digits_experiments.md how to run every experiment (setup, stages, timings, outputs)
    digits_method.md      the method: definitions, Propositions 1-4, Theorem 5, proofs, checks
```

## Quick start

    pip install -r requirements.txt
    python scripts/paper2.py --data uci --stage tests            # all exactness tests, a few minutes
    python scripts/paper2.py --data uci --list                   # the experiments and their status
    python scripts/paper2.py --data uci --stage all --workers 4  # the whole paper on the UCI digits
    python scripts/paper2.py --data mnist --stage all --workers 8

See RUN_PAPER2.md for each stage, the expected run times and the output files.

## Using the reference models without retraining

Copy `results_reference/models/*.npz` and `results_reference/metrics/*.json` into `outputs/strokes/` (create the
folder). Then run `python scripts/comb_strokes.py prepare --data uci` once, after which `evaluate`, `scripts/deploy_strokes.py`
and the report stage work on them directly. The UCI data are rebuilt identically: same split, same seed, same
sequence length 56.

## What is verified (the tests print every error)

- Window, pair and path laws of every model equal brute-force enumeration of all token sequences, to 1e-14 to
  1e-16.
- Every gradient agrees with central finite differences to 1e-8 to 1e-10, including under noise.
- Batched circuits equal single-circuit simulation exactly.
- The noisy tree equals an independent node-by-node density-matrix simulation to 3e-16; every token map is
  completely positive and trace preserving.
- Theorem 5 (compilation into one work qubit) reproduces every Kraus operator to 1e-13 to 1e-15, including on a
  trained model.
- Proposition 4 (environment levels contain every classical HMM) holds numerically to 6e-17.
- Samplers reproduce the exact laws within sampling noise.

## Limits, stated plainly

- With memory dimension chi = 4 everything is classically simulable; that is what makes exact training possible.
  No computational quantum advantage is claimed.
- The coherence advantage measured so far (STROKES_METHOD.md, section 14) is for the general coherent memory
  (arbitrary circuit depth), on the 1,797 UCI digits, in correlation statistics. Whether a shallow circuit keeps
  it, and whether it holds on MNIST, are what the `circuits` stage and the MNIST runs decide.
- No model has yet drawn convincingly realistic digits unconditionally; the `conditional` stage tests one model
  per digit.
