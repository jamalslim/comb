# Stroke-CoMB: a coherent memory register that draws, trained by an exact k-point correlation MMD

> Working notes, written during the study: they record the method and the results as they came in. Where these notes
> and the paper differ, the paper is final.

## 1. Why the pixel approach failed, and what changes

The earlier MNIST attempts asked CoMB to emit 784 PCA scores or pixel codes, one per round, and trained
the characteristic-function MMD (CF-MMD) with random frequencies. Their samples were recognisable as
smudges but not as digits: a two-sample classifier separated them from real digits with AUC 0.96 to 1.00.
Two things were wrong at once:

- **Representation.** A digit is not a sequence of 784 weakly dependent numbers. It is a curve, and the
  information that makes it a digit (where the stroke turns, whether it closes) sits in the dependence
  between successive pen moves.
- **Loss.** A Gaussian-kernel CF-MMD in hundreds of dimensions is dominated by low-order structure. It
  barely weighs the multi-point correlations that make a stroke a digit, and its random frequencies add
  variance.

The new method changes both: the register now draws the digit one pen move per round, and the loss
compares the k-point correlations of the pen-move process exactly.

## 2. Representation: a digit as a pen-move sequence

Image -> binarise -> Zhang-Suen skeleton -> covering walk -> tokens.

- Token c in {0,...,15}: c mod 8 is one of 8 unit moves on the pixel grid, and c < 8 means pen down.
  The 16 tokens are the 16 outcomes of a 4-qubit work register.
- The walk starts at the top-left endpoint of each connected skeleton component and covers every
  skeleton pixel, retracing branches when it backtracks. The next pixel has the fewest unvisited onward
  neighbours (Warnsdorff's rule), ties broken by the smallest turn, so loops close. Pen-up moves connect
  components and pad the sequence to length d.
- The inverse map draws the pen-down moves, thickens, blurs and centres by centre of mass.

The round trip keeps digit identity. A classifier trained on round-trip renders reaches 92.3% on
held-out round-trip renders of the UCI digits. That is the ceiling any generator in this
representation can reach on that metric.

## 3. Model

This is the CoMB of the Dataset 1 paper with n_w = 4, a binary count and one round per pen move:

    P(c_1..c_d) = tr[ Lambda_{d,c_d} ... Lambda_{1,c_1}(rho_0) ],   Lambda_{t,c}(rho) = M_{t,c} W_t rho W_t^dag M_{t,c}^dag.

The memory (2 qubits, chi = 4) is never measured, and each round has its own angles. The dephased twin
(same circuit, memory dephased every round) and a general classical HMM with chi states and the same
16-token emissions are trained with the same loss.

## 4. Loss: the exact k-point correlation MMD

Let G be a K x K token kernel. The stroke kernel is G = G_pen (x) G_dir, where
G_dir[a, b] = exp(kappa (cos(theta_a - theta_b) - 1)) is a von Mises kernel on directions and
G_pen = [[1, rho], [rho, 1]]. With kappa = 2 and rho = 0.3, its eigenvalues lie in [0.077, 3.21]:
it is strictly positive definite, and neighbouring directions are similar (0.557).

Define the sequence kernel

    K(c, c') = sum_{k=1..3} lam_k / (d-k+1) * sum_t prod_{j<k} G[c_{t+j}, c'_{t+j}]
             + lam_p / |P| * sum_{(t,s) in P} G[c_t, c'_t] G[c_s, c'_s],

where P holds all pairs (t, s) with s - t in a lag set (3 up to 96).

**Proposition 1.** K is positive semidefinite, and

    MMD_K^2(P, Q) = sum_k lam_k mean_t <Delta W_k(t), G^{(x)k} Delta W_k(t)>
                  + lam_p mean_{(t,s)} <Delta P_ts, (G (x) G) Delta P_ts>,

where W_k(t) is the law of the window (c_t, ..., c_{t+k-1}), P_ts is the law of (c_t, c_s), and Delta is
the model law minus the data law. If G is strictly positive definite, MMD_K(P, Q) = 0 if and only if
every included window and pair law agrees.

*Proof.* Each summand is a product of PSD kernels on sub-tuples, hence PSD, and a sum of PSD kernels is
PSD. Its feature map is the tensor product of G's feature map over the sub-tuple, so the mean embedding
of a law is that tuple's marginal law mapped by G^{(x)k}. The MMD is the norm of the difference of the
embeddings, a sum of quadratic forms in the marginal differences. A strictly positive definite G^{(x)k}
makes each quadratic form vanish only at Delta = 0.

**Proposition 2 (exact evaluation).** For any CountChain model (CoMB, its twin, a classical HMM), with
transfer maps T_t = sum_c Lambda_{t,c} and states r_t = T_t ... T_1 rho_0,

    W_k(t)(a_1..a_k) = tr[ Lambda_{t+k-1,a_k} ... Lambda_{t,a_1}(r_{t-1}) ],
    P_ts(a, b)       = tr[ Lambda_{s,b} T_{s-1} ... T_{t+1} Lambda_{t,a}(r_{t-1}) ].

The loss and its gradient are therefore exact. Reverse mode runs through the same products and then
through the circuit. Cost per step is O(d K^3 D^2 + |P| L_max K D^2 / d) with D = chi^2 for the coherent
memory. There is no sampling, no frequency set and no teacher forcing: the statistics are those of the
free-running generator.

**Term normalisation.** Each term is divided by its value for a reference model: uniform tokens for the
one-point term, and independent tokens with the data's one-point laws for windows k >= 2 and for pairs.
The loss then reads "sum over terms of model discrepancy / no-dependence discrepancy". On held-out data,
kappa_term = (null - model) / (null - floor) is the share of each order of correlation captured, as in the
Dataset 1 paper.

## 5. What this loss cannot see

Matching windows up to length 3 and pairs at chosen lags does not fix correlations of order 4 and higher
(cf. the low-body argument of Rudolph et al., arXiv:2305.02881). Exact loop closure, for instance, is a
global constraint. Whatever global structure the samples show beyond the loss must come from the memory.
We therefore test global geometry that is *not* in the loss: closure distance, winding number, width,
height, total turning, pen-down length and pen-up count, as W1 distances between model samples and
held-out digits, relative to the train-versus-test floor.

## 6. Evaluation protocol

- Held-out k-point loss and kappa per term (exact for the model, empirical for the test set).
- Geometry W1/sd on 2000 samples against held-out digits (out-of-loss).
- Rendered samples: two-sample classifier AUC against round-trip renders of held-out digits (0.5 means
  indistinguishable; the floor is train vs test renders); digit-classifier confidence; entropy of the
  predicted class frequencies (3.32 bits = all ten digits equally often).
- The same protocol for CoMB, its dephased twin, a classical HMM of equal chi, and a classical HMM with
  chi = 16 as the larger classical reference.

## 7. Novelty, stated carefully

To our knowledge, no prior quantum sequential generator draws digits as pen strokes. The literature
search found HMMs for classical handwriting recognition and synthesis, and HQMMs for stochastic processes
(Monras et al. 2010; quantum dimension reduction of HMMs, arXiv:2601.16126), but none of this. ANOVA
kernels that isolate k-order correlations have been used to analyse Born machines on bitstrings
(arXiv:2511.01845). Here the window-and-lag product kernel is evaluated exactly, with exact gradients,
through the instrument chain of a coherent-memory generator, and it is used as the training loss.

## 8. Limits

- chi = 4 is classically simulable, so no computational advantage is claimed.
- The representation drops stroke thickness; renders use a fixed pen.
- Per-round parameters: the model is non-stationary in the pen-move index.
- The results in this folder are on the UCI digits bundled with scikit-learn (1797 images, 8 x 8,
  upsampled), because the machine that produced them had no network access. MNIST runs use the same
  commands with --data mnist; their numbers are not reported here until they have been run.

## 9. Global shape: the exact path MMD

Windows of up to three moves and pairs at chosen lags fix the local texture of a stroke, not the plan of a
digit. The pen position after tau moves, X_tau = sum_{t < tau} v(c_t), is a sum over all earlier moves, so
the joint law of the positions at times tau_1 < ... < tau_m (every 4 moves here) carries correlations of every
order between the moves.

With a Gaussian-mixture kernel on the stacked positions X in R^{2m}, Bochner's theorem gives
MMD^2 = E_omega |Phi_P(omega) - Phi_Q(omega)|^2.

**Proposition 3 (exact path characteristic function).** For any CountChain model,

    Phi_model(omega) = tr[ A_{d-1} ... A_0 (rho_0) ],   A_t = sum_c exp(i Omega_t . v(c)) Lambda_{t,c},
    Omega_t = sum_{j : tau_j > t} omega_j.

*Proof.* sum_j omega_j . X_tau_j = sum_t Omega_t . v(c_t), because move t enters every position observed
after it. Given the tokens, the exponential factorises over rounds. Summing over tokens round by round is
the instrument product, exactly as in the Dataset 1 readout, with a lattice weight in place of the sinc.

For independent tokens with one-point laws p_t, Phi = prod_t sum_c p_t(c) exp(i Omega_t . v(c)). The term is
normalised by this no-dependence value, like the k-point terms. Each training step uses 256 fresh frequencies
(seeded by the step); validation and evaluation use fixed frequency sets. The cost is
O(R d K D^2) per step, forward and backward.

Checks (scripts/test_pencf.py): the characteristic function equals brute-force enumeration over all 16^4
sequences to about 1e-14, for CoMB, its dephased twin, CoMB with an environment qubit and the classical HMM.
Gradients agree with central finite differences to about 1e-10.

## 10. Expressivity: environment qubits

In the circuit used so far, each token has exactly one Kraus operator (a pure instrument). A classical HMM
step "from state i emit token c and move to state j" is the measure-and-prepare map
rho -> sum_{i,j} T_ji E_cj <i|rho|i> |j><j|, which needs up to chi^2 Kraus operators per token.

**Proposition 4.** Let the last n_e work qubits be measured and reset but not emitted. If 2^n_e >= chi^2,
the set of instruments reachable with some unitary contains every classical chi-state HMM with the same
K tokens.

*Proof.* Take the isometry V|i>|0> = sum_{c,j} sqrt(T_ji E_cj) |j>|c>|(i,j)>, where (i,j) labels an
environment basis state. Different i give orthogonal environment states, and the columns of T and E sum to
one, so V^dag V = 1. Tracing out the environment gives exactly the measure-and-prepare instrument above. Any
isometry extends to a unitary on memory and work.

The option is current = "binary_env:n_e" in comb/digits/model.py (comb_strokes.py --n_env). Reaching the general
class still needs a deep enough circuit; the depth law of the Dataset 1 paper applies.

## 11. Findings so far (UCI digits, chi = 4, 1500 steps, seed 0)

See outputs/strokes/RESULTS_uci.md for the full table. In short:

- **Coherence helps.** CoMB beats its dephased twin on every measure (held-out k-point loss 0.935 vs 1.082).
- **The classical HMM still wins.** A general classical HMM with the same memory size beats both (0.589).
  The circuit's pure instruments and shallow depth are a smaller class than a general classical memory
  (section 10).
- **Global shape is not learned, by any model.** Even a chi = 16 HMM, which captures 99% of the 2- and 3-move
  dependence, misses digit height by 1.60 standard deviations, against a train-vs-test floor of 0.12. A
  two-sample classifier separates its drawings from real digits with AUC 0.81. The exact path MMD
  (section 9) targets this.

## 12. Running it

- Tests: scripts/test_kpoint.py, scripts/test_pencf.py, scripts/test_strokes.py.
- Training resumes from its checkpoint when rerun with the same arguments. A run interrupted and resumed
  ends with bit-identical parameters.
- MNIST (downloaded on first use, about 11 MB; the paper uses the UCI digits):

      python scripts/comb_strokes.py prepare --data mnist
      python scripts/comb_strokes.py train --data mnist --model comb --chi 4 --path 2
      python scripts/comb_strokes.py train --data mnist --model dephased --chi 4 --path 2
      python scripts/comb_strokes.py train --data mnist --model hmm --chi 4 --path 2
      python scripts/comb_strokes.py evaluate --data mnist --run strokes_mnist_comb_chi4_path2_s0
      python scripts/strokes_table.py --data mnist
      python scripts/comb_strokes.py plot --data mnist

## 13. The general coherent memory (comb/digits/hqmm.py)

Simulating environment qubits through the circuit is expensive (about 6 s per step with 2 environment
qubits). The memory-class question is separated from circuit depth by an unrestricted isometry per round,
V = A (A^dag A)^{-1/2} of a free complex matrix A. Every unitary on memory + work acting on |work = 0> gives
such an isometry and every isometry arises this way, so this is CoMB with an arbitrarily deep V circuit:
the largest class this register can reach. The gradient through the polar factor is exact (eigenvalues of
the chi x chi Gram matrix and divided differences).

Checks (scripts/test_hqmm.py):

- Laws agree with brute-force enumeration to 3e-16.
- Gradients of the k-point and path losses agree with finite differences to 1e-8.
- Sampled one-point laws agree with the exact ones.
- Proposition 4, numerically: a classical HMM (chi = 3, K = 16), written as the isometry of the proof with
  9 environment levels, reproduces every window law, pair law and path characteristic function of that HMM,
  to 6e-17 and 6e-16 respectively.

The comparison "general coherent memory with chi^2 environment levels vs general classical HMM, same chi" is
therefore between nested classes. The quantum class contains the classical one, so it can only fit at least
as well at the optimum. The open questions are whether training finds that optimum, and whether the extra
coherent freedom generalises to held-out digits.

## 14. The decisive experiment: coherence at fixed parameters

Setup: UCI digits, chi = 4, k-point + path loss (path weight 2), 1500 steps, model selection on validation,
three seeds per model. Three models:

- the general coherent memory with 16 environment levels (458,752 parameters);
- the same model with the memory dephased after every round: identical parameters and parameter count,
  coherence removed;
- the general classical HMM (4,480 parameters).

By Proposition 4 the coherent class contains the classical one.

| model | validation loss | held-out k-point loss (per seed) | kappa 2 / 3 / pairs / path | turning | height | AUC |
|---|---|---|---|---|---|---|
| general coherent memory | 0.720 +- 0.005 | 0.577 (0.577, 0.575, 0.579) | 97 / 96 / 75 / 90% | 0.68 | 1.69 | 0.893 |
| same memory, dephased | 0.840 +- 0.008 | 0.684 (0.705, 0.682, 0.666) | 93 / 92 / 71 / 90% | 1.04 | 1.63 | 0.890 |
| classical HMM | 0.842 +- 0.038 | 0.710 (0.681, 0.688, 0.759) | 91 / 90 / 72 / 92% | 1.11 | 1.54 | 0.881 |

On held-out digits, the train-vs-test floor of the k-point loss is 0.350 and the no-dependence value is 2.792.
Turning and height are W1 errors in units of the test standard deviation; their floor is at most 0.13. The
AUC floor is 0.48.

**What is established, on this data:**

- Every coherent run beats every dephased and every classical run, on validation and on held-out
  correlation statistics. The worst coherent held-out value is 0.579; the best of the others is 0.666.
- Removing coherence with everything else fixed costs 16% of the held-out k-point loss and 14% of the
  validation loss, and lands the model at the classical HMM's level. The gain is coherence, not parameter
  count.
- The gain is in local correlation structure (2- and 3-move windows, pairs, turning).

**What is not established:**

- Global shape and realism do not improve. Height is slightly worse than the HMM, and a two-sample
  classifier separates all three models' drawings from real digits about equally well (AUC 0.88 to 0.89).
  None draws convincing whole digits.
- This is the idealised memory, i.e. CoMB with an arbitrarily deep circuit. The shallow circuit used so far
  lost to the classical HMM (section 11). How deep a circuit must be to keep the gain is open.
- 1,797 small digits (8 x 8, upsampled); MNIST is run with run_mnist.sh.
- chi = 4 is classically simulable, so no computational advantage is claimed.

## 15. Measurement-tree CoMB: one work qubit, binary sub-rounds, feed-forward (comb/digits/treecomb.py)

**Theorem 5 (measurement-tree compilation).** Let {M_o}, o in {0,1}^n, be the Kraus operators of one round of
a coherent-memory generator (pure instrument on C^chi, sum_o M_o^dag M_o = 1, K = 2^n tokens). For every
prefix p of length l <= n let E_p = sum over tokens o starting with p of M_o^dag M_o, so E_() = 1. Define the
node operators

    N_{p,b} = V_{pb} E_{pb}^{1/2} E_p^{-1/2},    b = 0, 1,

with V = 1 at inner nodes and V_o the unitary of the polar decomposition M_o = V_o |M_o| at the leaves
(pseudo-inverse on the kernel of E_p). Then:

- (i) N_{p_{n-1},b_n} ... N_{(),b_1} = M_o for every token o;
- (ii) for every prefix p, N_{p,0}^dag N_{p,0} + N_{p,1}^dag N_{p,1} is the projector onto the support of E_p.

So every node is a two-outcome instrument, i.e. an isometry C^chi -> C^chi (x) C^2 implemented by a circuit on
the memory plus one work qubit, chosen by classical feed-forward on the bits already emitted.

*Proof.* (i) Along a path the factors E_q^{-1/2} E_q^{1/2} cancel pairwise, leaving V_o E_o^{1/2} = V_o |M_o| = M_o.
(ii) sum_b E_p^{-1/2} E_{pb} E_p^{-1/2} = E_p^{-1/2} E_p E_p^{-1/2}, because E_{p0} + E_{p1} = E_p, and the leaf
unitaries drop out of N^dag N.

The construction follows the binary search trees of Andersson and Oi for POVMs (PRA 77, 052104, 2008), which
were run on IBM dynamic circuits in PRX Quantum 5, 030315 (2024). It is extended here to the full instrument
(the post-measurement memory state), which a generator needs, and to exact training in tree form.

**Resources per token.** One monolithic round needs a general isometry C^chi -> C^{chi K} on n_m + n qubits
(2 chi^2 K - chi^2 real parameters; 496 for chi = 4, K = 16). The tree executes n = log2 K node isometries
C^chi -> C^{2 chi} on n_m + 1 qubits (3 chi^2 = 48 parameters each). The executed circuit per token therefore
scales as chi^2 log K instead of chi^2 K. The total number of stored parameters does not shrink: K - 1 node
circuits are kept, and feed-forward selects log2 K of them per token. The price is n mid-circuit
measurements, resets and feed-forward decisions per token instead of one.

**Exactness is unchanged.** The composed instrument of a round is a linear map, so the tilted-instrument
identity and every loss of this note apply as they are. instruments_all / instruments_all_vjp are batched
over rounds and nodes.

**Dephased twin.** The memory is dephased after W_t and after every node (a classical chain with
transition-emission matrices |N|^2 and |W|^2), with the same parameters.

**Checks (scripts/test_treecomb.py):**

- laws equal brute-force enumeration to 2e-16 (coherent) and 4e-16 (dephased);
- the dephased twin equals explicit density-matrix evolution dephased after W and every node to 1e-16;
- batched node circuits equal single-circuit simulation exactly, and nodes are isometries to 1e-15;
- gradients of the k-point + path loss agree with finite differences to 3e-9 (coherent) and 6e-9 (dephased);
- the feed-forward sampler reproduces the exact one-point and pair laws within sampling noise (60,000 samples);
- Theorem 5 reproduces every Kraus operator of random general memories (chi = 4, K = 16) to 1e-15, and of the
  trained general coherent memory (all 56 rounds) to 1e-13. The compiled tree's held-out k-point loss equals
  the original's to 8e-14.

**The decisive experiment (queued).** Train tree-CoMB with shallow node circuits (3 qubits, node depth 4 and 6)
and its dephased twin with the combined loss. If tree-CoMB approaches the general coherent memory (validation
0.720) and its twin does not, the coherence advantage is carried by a circuit of NISQ depth with one work
qubit. Run it with:

    python scripts/comb_strokes.py train --data uci --model tree --chi 4 --L_node 4 --path 2
    python scripts/comb_strokes.py train --data uci --model tree_dephased --chi 4 --L_node 4 --path 2

## 16. Deployment without hardware: exact noisy evaluation (comb/digits/noisytree.py, scripts/deploy_strokes.py)

The tilted-instrument identity holds for any completely positive instruments, not only for unitary circuits.
The loss of a deployed, noisy model is therefore computed exactly, without shots, under a device noise model,
and noise-aware training uses the same exact gradients.

**Noise model** (as in comb/digits/noise.py):

- depolarizing gate noise p1 after every rotation and p2 after every CZ;
- imperfect reset of the work qubit (starts in |1> with probability p_reset);
- readout flips with probability p_ro;
- amplitude damping and pure dephasing of every memory qubit during each measure-and-reset window.

Two device models are used:

- trapped-ion (H2-like): p2 = 1e-3, readout and reset 1e-3, memory dephasing 1.4e-3 per window;
- superconducting (Heron r3-like): p2 = 1.5e-3, p1 = 2.5e-4, readout 1.3e-2, reset 1e-2, T1 = T2 = 200 us,
  2 us per window.

The superconducting model's p1, its reset error and its window duration are assumptions.

**Specific to the tree:**

- A readout error changes both the emitted bit and the next node chosen by feed-forward. The noisy
  instruments are indexed by the recorded bit, which models this exactly.
- The tree has log2 K windows per token where monolithic CoMB has one. Each tree window is charged the same
  idle time as a full monolithic round, which is conservative for the tree.

**Checks (scripts/test_noisytree.py):**

- zero noise reproduces TreeCoMB to 1e-15;
- the batched noisy instruments equal an independent node-by-node density-matrix simulation (built on the
  tested comb.noise.noisy_circuit, with reset, readout confusion, feed-forward on the recorded bit and idling
  done by hand) to 3e-16 (coherent) and 6e-16 (dephased);
- every token map is completely positive (smallest Choi eigenvalue 6.6e-4) and each round preserves the
  trace to 1e-15;
- gradients under noise agree with finite differences to 2e-9 (coherent) and 4e-10 (dephased);
- exact sampling of recorded tokens reproduces the exact laws within sampling noise.

A related fix: NoisyCoMB no longer inherits the noiseless batched path of CoMB (zero noise reproduces CoMB to
1e-15, and noise is now applied).

**First result: the monolithic CoMB circuit** (depth 4, k-point + path, seed 0). Exact held-out objective:

| noise | objective |
|---|---|
| noiseless | 1.265 |
| ion-like | 1.664 |
| Heron-r3-like x0.25 | 1.668 |
| Heron-r3-like x1 | 4.32 |
| Heron-r3-like x8 | 25.2 |

**Pending:** the same study for tree-CoMB and its dephased twin. It reports lambda*, the noise scale at which
the coherent tree stops beating its twin: a device requirement anyone can test. It needs the trained trees
(`python scripts/paper2.py --data uci --stage circuits`); then run:

    python scripts/deploy_strokes.py --data uci --runs strokes_uci_tree_chi4_L4_path2_s0 strokes_uci_tree_dephased_chi4_L4_path2_s0

## 17. Making sure CoMB generates the dataset: class-conditional CoMB and the quality gate

Unconditional CoMB with a 4-level memory draws digit-like stroke fragments, not recognizable whole digits
(section 14). One memory has to hold both which digit it is drawing and where it is in the stroke. The standard
way to show that a generator produces recognizable digits is class-conditional generation: one CoMB per digit
(comb_strokes.py --digit k), each trained with the exact k-point + path MMD on the digits of its class only.

**The unconditional generator is the exact mixture.** Draw a label with the data's class frequencies, then
sample that digit's model. This is itself a CountChain: a classical label register next to the coherent memory,
i.e. a block-diagonal memory of 10 blocks. Every exact loss applies to it, because
Phi_mixture = sum_k pi_k Phi_k.

**The quality gate (scripts/class_quality.py)** reports, from samples of the trained models:

- for every digit k, the fraction of samples that a classifier trained on real digits (round-trip renders)
  reads as k, next to the same fraction for real held-out digits of class k (the ceiling the stroke
  representation allows), and the mean confidence;
- for the mixture, the two-sample AUC against real held-out renders (0.5 = indistinguishable; the floor is
  train vs test), the class entropy, and the mean intended-digit rate;
- a figure with one row per digit: real held-out digits next to generated ones.

The verdict criterion is stated and adjustable: a digit passes when its intended-digit rate is at least 0.8
times the ceiling (--target).

**Run it on MNIST** (resumable; digits can run in parallel on several cores):

    sh run_mnist_conditional.sh
    DIGITS="0 1 2" sh run_mnist_conditional.sh        # one terminal per group of digits

The report goes to outputs/strokes/CLASSES_mnist_tree_chi4_L4_path2_s0.md and the figure to
outputs/figs/strokes_mnist_tree_chi4_L4_path2_s0_classes.pdf.

The pipeline is checked end to end on two UCI digits (train, evaluate, gate). The full ten-digit run on UCI is
queued. Its numbers, and the MNIST numbers, are reported only once they have been run.
