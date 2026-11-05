# CoMB — the algorithm (one model, one training signal)

Register: n_m memory qubits (chi = 2^n_m), never measured; n_w work qubits, measured and reset at
every step. Register size is fixed by (n_m, n_w); the data dimension d sets the number of steps.

GENERATION of one sample y in R^d  (default: one round per coordinate)
  rho <- |0><0|^{n_m}
  for beta = 1 .. d:
      rho <- W_beta rho W_beta^dagger                        # memory rotation (L_w layers)
      reset work qubits to |0>; apply V_beta on memory+work  # joint unitary (L_v layers, CZ line)
      measure work qubits -> x in {0,1}^{n_w}; rho <- M_x rho M_x^dagger / p(x), M_x = (1 (x) <x|) V_beta (1 (x) |0>)
      c_beta <- sum_q 2^(n_w-q) x_q                          # the count: the work register read as an integer, K = 2^{n_w} levels
      y_beta <- g_beta(c_beta + u), u ~ U[0,1)               # learned monotone readout: K bin widths per coordinate (softmax), fixed support
  y <- mu + P diag(sigma) y                                   # fixed PCA decoder (change of coordinates, not trained)

TRAINING (exact, free-running, no samples of the model, no likelihood, no teacher forcing)
  Lambda_{beta,c} = count-resolved instrument of step beta (c = 0..K-1)            [comb.model]
  Phi_theta(omega) = Tr[ A_d(omega_d) ... A_1(omega_1)(rho_0) ],  A_beta(omega) = sum_c w_beta(c,omega) Lambda_{beta,c},
      w_beta(c,omega) = e^{i omega lbar_{beta,c}} sinc(omega delta_{beta,c}/2)     # exact characteristic function of the generated law
  loss = (1/R) sum_r | Phi_hat_data(omega_r) - Phi_theta(omega_r) |^2,  omega_r ~ Gaussian-mixture spectrum  (= unbiased kernel MMD^2)
  gradient: one reverse sweep through the contraction, instruments and circuits    [comb.train, Appendix A of the draft]
  cost: linear in d, polynomial in chi; Adam, 3000 steps, warm-up + cosine, validation-selected.
  readout initialised by the quantile rule (bins placed where the model's own count marginals put the data's quantiles).

PROPERTIES (proved in the draft, verified in scripts/test_comb.py)
  copula of y is set by the memory alone, marginals by the readout;  PSD rank of every cut <= chi;
  dephasing the memory after each step gives the classical chi-state chain with the same parameters (the "dephased twin");
  n_sub > 1 rounds per coordinate is supported (window = counting statistics; converges to photon counting) -- not used by default.

ABLATIONS (same register, same code)
  memory ablation  : n_m = 0, and the dephased twin (--model dephased), which removes the memory's coherence
  hardware         : comb/showers/qiskit_comb.py + scripts/hardware_run.py build, verify (numpy + Aer) and run the round
                     circuit above on a device and decode its bits into showers; the records of the ibm_kingston runs
                     come with the release asset (outputs/hardware/)
  noise            : comb/showers/noise.py (exact noisy model)

FILES   comb/showers/model.py (register, instruments, CF, sampling)  comb/showers/train.py (exact training)
        comb/showers/mmd.py (spectrum, MMD)  comb/showers/zoo.py (specs, save/load)  comb/showers/data.py (Dataset 1)
        scripts/comb_ds1.py (prepare / train / sample / plot for Dataset 1)  scripts/test_comb.py (verification suite)
        comb/showers/qiskit_comb.py, scripts/hardware_run.py (device path)
        comb/digits/ and scripts/comb_strokes.py: the handwritten digits (docs/digits_overview.md lists every module)
VERIFIED: gate list -> unitary 8e-16; count convention vs instrument 2e-16; test suite passes.
TO VERIFY ON YOUR MACHINE (qiskit): python scripts/hardware_run.py verify outputs/runs/<run>.npz   (TV at floor)
