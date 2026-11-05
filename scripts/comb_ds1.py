#!/usr/bin/env python3
"""Dataset 1 with the coherent memory register, four commands, everything in the foreground.

    python scripts/comb_ds1.py prepare                      # download (if needed) + prepare -> data/
    python scripts/comb_ds1.py train                        # train the model (default: CoMB chi=4, one round) -> outputs/runs/
    python scripts/comb_ds1.py sample                       # load the trained model, generate showers, print the metrics -> outputs/samples/
    python scripts/comb_ds1.py plot                         # figures from the generated showers -> outputs/figs/

Options:  train   --model comb|dephased|hmm  --chi 4  --steps 3000  --seed 0
          sample  --model/--chi/--seed as in train (or --run <file.npz>)  --n 1200
          plot    --runs <name1> <name2> ...  (default: every file in outputs/samples/)
"""
import sys, os, json, time, pathlib, argparse, urllib.request
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("COMB_DATA", str(ROOT / "data"))
DATA, RUNS, SAMPLES, FIGS = ROOT / "data", ROOT / "outputs/runs", ROOT / "outputs/samples", ROOT / "outputs/figs"


def need_shower_project(run=None):
    """comb_ds1.py is the shower (Dataset 1) script: it needs the shower project around it, i.e. comb/showers/zoo.py and data/
    next to this file. Stop with a clear message if it is run anywhere else (for example in the digits folder)."""
    if not (ROOT / "comb" / "showers" / "zoo.py").exists():
        sys.exit(f"comb_ds1.py is the shower (Dataset 1) script, but {ROOT} is not the shower project (no comb/showers/zoo.py here).\n"
                 f"Run it in the folder that contains comb/showers/zoo.py, data/ and outputs/runs/ds1_comb_chi4_s0.npz.\n"
                 f"To find that folder:  find ~ -path '*comb/showers/zoo.py'")
    if not DATA.exists():
        sys.exit(f"no data/ folder in {ROOT}: run  python scripts/comb_ds1.py prepare  first, or run in the shower project.")
    if run is not None and not pathlib.Path(run).exists():
        sys.exit(f"run file not found: {run}\nThe trained shower model is usually outputs/runs/ds1_comb_chi4_s0.npz in the shower project.")
for p in (DATA, RUNS, SAMPLES, FIGS): p.mkdir(parents=True, exist_ok=True)
URL = "https://zenodo.org/records/8099322/files/dataset_1_photons_1.hdf5?download=1"       # CaloChallenge Dataset 1, photons, training file

# ------------------------------------------------------------------------------------------------ prepare
def prepare(a):
    """Download Dataset 1 (photons, Zenodo record 8099322) into data/raw/ if it is not there, and prepare it in
    data/: the showers as model coordinates (total energy and energy fractions), with their splits.
    """
    raw = DATA / "raw" / "dataset_1_photons_1.hdf5"
    raw.parent.mkdir(exist_ok=True)
    if not raw.exists():
        print("downloading dataset_1_photons_1.hdf5 (about 0.5 GB) from Zenodo record 8099322 ...", flush=True)
        urllib.request.urlretrieve(URL, raw, reporthook=lambda b, bs, t: print(f"\r  {b * bs / 1e6:7.0f} MB", end="", flush=True))
        print()
    import h5py
    from sklearn.model_selection import train_test_split
    with h5py.File(raw, "r") as f:
        Einc = np.asarray(f["incident_energies"]).reshape(-1)
        S = np.asarray(f["showers"], dtype=np.float64)
    S = S[np.isclose(Einc, 65536.0, rtol=1e-3)]
    print(f"showers at 65.536 GeV: {len(S)}, voxels: {S.shape[1]}")
    E = S.sum(1)
    frac = S / np.where(E > 0, E, 1.0)[:, None]
    used = np.random.default_rng(123).choice(len(S), size=min(6000, len(S)), replace=False)          # the subset data.load() will use
    tr, _ = train_test_split(np.arange(len(used)), test_size=0.2, random_state=42)
    med = float(np.median(E[used][tr]))
    mad = float(1.4826 * np.median(np.abs(E[used][tr] - med)))
    t0, w = 0.3, 0.1
    X = np.concatenate([(t0 + w * (E - med) / mad)[:, None], frac], axis=1)                            # QFAN Eq. (10)
    np.save(DATA / "cal_ds1_photons_65GeV.npy", X)
    cov = S.std(0) / np.where(S.mean(0) > 0, S.mean(0), 1.0)
    json.dump(dict(n=len(S), d=X.shape[1], median_E=med, mad_E=mad, t0=t0, w=w, layers=[8, 160, 190, 5, 5],
                   sparse_voxel=(((S == 0).mean(0) > 0.05) | (cov > 1.0)).tolist()), open(DATA / "cal_ds1_meta.json", "w"), indent=1)
    print(f"prepared: data/cal_ds1_photons_65GeV.npy {X.shape}; training split: 4800 (4000 fit + 800 val) / 1200 test; "
          f"E median {med:.0f} MeV, MAD {mad:.0f} MeV; sparse voxels {sum(json.load(open(DATA / 'cal_ds1_meta.json'))['sparse_voxel'])}/368")

# ------------------------------------------------------------------------------------------------ shared
def ensure_tag():
    """Register the Dataset 1 file with the data loader comb.showers.data and return the loader."""
    from comb.showers import data as qd
    if "ds1" not in qd.FILES: qd.FILES["ds1"] = "cal_ds1_photons_65GeV.npy"
    return qd
def spec_of(a):
    """Model specification from the command-line options, including the nested wiring and the general coherent
    memory (--dense) when they are asked for.
    """
    spec = _spec_of(a)
    if a.model in ("envmem", "envdeph"):
        return spec
    if a.model != "hmm" and getattr(a, "wiring", "interleaved") == "nested":
        spec = dict(spec, wiring="nested")
    if a.model != "hmm" and getattr(a, "dense", False):
        spec = dict(spec, dense=True)                             # the most general coherent memory (DenseUnitary)
    return spec


def _spec_of(a):
    K = int(getattr(a, "K", 8) or 8)
    if a.model == "hmm": return dict(model="hmm", basis="pca", chi=a.chi, K=K, seed=a.seed)
    if a.model in ("envmem", "envdeph"):
        return dict(model="envmem", basis="pca", chi=a.chi, K=K, E=int(getattr(a, "env", 0) or a.chi ** 2), dephase=int(a.model == "envdeph"), seed=a.seed)
    return dict(model="comb", basis="pca", n_m=int(round(np.log2(a.chi))), n_w=int(getattr(a, "nw", 3) or 3), n_sub=1, L_v=getattr(a, "Lv", 5), L_w=getattr(a, "Lw", 2), current="binary",
                dephase=int(a.model == "dephased"), seed=a.seed)
def name_of(spec):
    """Run name of a model specification, e.g. ds1_comb_chi4_s0 or ds1_hmm8_s3. Tags for the general memory, the
    nested wiring, the circuit depth and the number of levels are added only when they differ from the defaults.
    """
    depth = ("_dense" if spec.get("dense") else "") + ("_nest" if spec.get("wiring") == "nested" else "") + ("" if (spec.get("L_v", 5), spec.get("L_w", 2)) == (5, 2) else f"_L{spec['L_v']}" + (f"w{spec['L_w']}" if spec.get("L_w", 2) != 2 else ""))
    K = 2 ** spec["n_w"] if spec["model"] == "comb" else spec.get("K", 8)
    ktag = "" if K == 8 else f"_K{K}"                              # 8 levels per value is the default and carries no tag
    if spec["model"] == "envmem":
        return f"ds1_{'envdeph' if spec['dephase'] else 'envmem'}_chi{spec['chi']}_E{spec['E']}_s{spec['seed']}{ktag}"
    return (f"ds1_hmm{spec['chi']}_s{spec['seed']}{ktag}" if spec["model"] == "hmm"
            else f"ds1_{'dephased' if spec['dephase'] else 'comb'}_chi{2 ** spec['n_m']}_s{spec['seed']}{depth}{ktag}")
def load_data():
    """The prepared Dataset 1 splits, loaded through the data loader."""
    qd = ensure_tag()
    D = qd.load("ds1", v8_n=2000)
    return D
def invert(X, meta):
    """Showers from model coordinates: the first coordinate gives the total energy, the others the energy fractions
    of the 368 cells (clipped at zero and renormalised).
    """
    E = meta["median_E"] + meta["mad_E"] * (X[:, 0] - meta["t0"]) / meta["w"]
    F = np.clip(X[:, 1:], 0, None)
    F = F / np.where(F.sum(1) > 0, F.sum(1), 1.0)[:, None]
    return F * E[:, None]


ANG, RAD = [1, 10, 10, 1, 1], [8, 16, 19, 5, 5]                  # Dataset 1 photon geometry: angular x radial bins per layer



def ring_major_perm(S, layers):
    """Permutation that orders every layer ring by ring (radial bin outer, angular bin inner), as in the QFAN paper.
    The stored order of the CaloChallenge file is detected from the data: within a ring the mean energies of the
    angular bins are alike, so the ordering whose consecutive groups of n_alpha voxels have the smaller spread of
    means is the ring-major one."""
    edges = np.concatenate([[0], np.cumsum(layers)])
    means = S.mean(0)
    perm = []
    note = []
    for li in range(len(layers)):
        idx = np.arange(edges[li], edges[li + 1])
        na, nr = ANG[li], RAD[li]
        if na == 1: perm += list(idx); continue
        m = means[idx]
        spread_ring = m.reshape(nr, na).std(1).mean()                     # already ring-major?
        spread_alpha = m.reshape(na, nr).T.std(1).mean()                  # stored angle-major: ring r = entries r, r+nr, r+2nr, ...
        if spread_alpha < spread_ring:
            perm += list(idx.reshape(na, nr).T.ravel())
            note.append(f"layer {li}: angle-major in file -> reordered")
        else:
            perm += list(idx)
            note.append(f"layer {li}: ring-major in file")
    return np.array(perm), note

# ------------------------------------------------------------------------------------------------ train
def train_cmd(a):
    """Train the model chosen by --model (CoMB, its dephased twin, the classical HMM or a general coherent memory)
    with the exact MMD loss, and save it as outputs/runs/<name>.npz with its training record (<name>.json).
    """
    need_shower_project()
    from comb.showers import zoo
    from comb.showers.train import train, frequency_pool
    from comb.showers.mmd import Spectrum, median_sigma
    D = load_data()
    spec = spec_of(a)
    name = name_of(spec) + ("_warm" if getattr(a, "init_from", "") else "") + getattr(a, "tag", "") + (f"_ax{a.axis_k}" if a.axis_share > 0 else "") + ("_ly" if getattr(a, "layer_share", 0) > 0 else "")
    d = D["Y_tr"].shape[1]
    m0, s0 = D["Y_tr"].mean(0), D["Y_tr"].std(0)
    sig = median_sigma((D["Y_tr"] - m0) / s0)
    kspec = Spectrum(d, [sig / 2, sig, 2 * sig])
    print(f"model {zoo.describe(spec)}  ->  outputs/runs/{name}.npz   ({a.steps} steps; progress every 25 steps)", flush=True)
    pool = frequency_pool(kspec, (D["Y_fit"] - m0) / s0, cache=str(DATA / "ds1_pool.npz"))
    if getattr(a, "layer_share", 0) > 0:                         # resolve the layer energies (see add_layer_frequencies)
        meta_ = json.load(open(DATA / "cal_ds1_meta.json"))
        use = tuple(int(x) for x in a.layers.split(","))
        rel = tuple(float(x) for x in a.layer_scales.split(","))
        pool = add_layer_frequencies(pool, (D["Y_fit"] - m0) / s0, s0, meta_["layers"], share=a.layer_share, rel=rel, use=use)
        print(f"layer-resolved MMD: layers {use} (index into the layer list) at scales {rel}, weight {a.layer_share}, pool {len(pool[0])} frequencies", flush=True)
    if a.axis_share > 0:                                         # resolve the leading principal axes (see add_axis_frequencies)
        pool = add_axis_frequencies(pool, (D["Y_fit"] - m0) / s0, k=a.axis_k, share=a.axis_share)
        print(f"axis-resolved MMD: {a.axis_k} leading axes, weight {a.axis_share}, pool {len(pool[0])} frequencies", flush=True)
    model = zoo.build(spec, d)
    lr = getattr(a, "lr", None) or (0.2 if spec["model"] == "hmm" else 0.05 if spec["model"] == "envmem" else (0.1 / np.sqrt(2 ** (spec["n_m"] + spec["n_w"])) if spec.get("dense") else 0.1))
    t = time.time()
    warm = getattr(a, "init_from", "")
    lr_ro = 0.03
    if warm:                                                     # start from a trained run; 0.1 x the default learning rates
        src_path = warm if warm.endswith(".npz") else str(RUNS / f"{warm}.npz")
        warm_start(model, src_path)
        if not getattr(a, "lr", None): lr *= 0.1
        lr_ro *= 0.1
        print(f"warm start from {src_path}: learning rates {lr:.4g} (parameters) and {lr_ro:.4g} (readout)", flush=True)
    res = train(model, D["Y_fit"], D["Y_v8"], kspec, steps=a.steps, R=256, lr=lr, lr_readout=lr_ro, warmup=50, eval_every=25, init_readout=not warm,
                R_val=4096, seed=a.seed, ckpt=str(RUNS / f"{name}.ckpt.pkl"), ckpt_every=100, log=lambda s: print(s, flush=True), m=m0, s=s0, pool=pool)
    model.set(res["params"])
    extra = dict(best_val=res["best_val"], best_step=res["best_step"], steps=a.steps, train_time=time.time() - t, params=int(model.n_params), K=int(model.K))
    zoo.save(str(RUNS / f"{name}.npz"), model, spec, extra)
    json.dump(dict(name=name, spec=spec, hist=res["hist"], **extra), open(RUNS / f"{name}.json", "w"), indent=1)
    print(f"done: best validation MMD^2 {res['best_val']:.3e} at step {res['best_step']}  ({(time.time() - t) / 60:.0f} min)  -> outputs/runs/{name}.npz")

def warm_start(model, src_path):
    """Start `model` at a trained run. Same model class: every parameter, readout and decoder array is copied, and training
    continues. Dense model from a circuit model: each round's circuit unitaries are written exactly in the dense form,
    U = exp(-iH) with H = i log U (principal branch, from a complex Schur form), so both models have identical
    characteristic functions before the first step; readout and decoder are copied."""
    import copy, scipy.linalg
    from comb.showers import zoo
    from comb.showers.envmem import EnvMemory, embed_hmm, embed_circuit
    from comb.showers.model import ClassicalHMM, CoMB
    src = zoo.load(src_path)[0]
    b, sb = model.base, src.base
    if isinstance(b, EnvMemory) and not isinstance(sb, EnvMemory):    # exact embedding of a trained HMM or circuit
        for k, v in vars(src).items():
            if k != "base" and isinstance(v, (np.ndarray, float, int)):
                setattr(model, k, copy.deepcopy(v))
        if isinstance(sb, ClassicalHMM): embed_hmm(b, sb)
        elif isinstance(sb, CoMB): embed_circuit(b, sb)
        else: sys.exit(f"--init-from: cannot embed {type(sb).__name__} into the environment memory")
        for k in ("h", "lo", "hi", "m", "s"):
            if hasattr(sb, k): setattr(b, k, np.array(getattr(sb, k), copy=True))
        return src
    if type(b) is not type(sb) or getattr(b, "chi", None) != getattr(sb, "chi", None) or getattr(b, "dephase", None) != getattr(sb, "dephase", None):
        sys.exit(f"--init-from: {src_path} is not the same kind of model (class, memory size and dephasing must match)")
    for k, v in vars(src).items():
        if k != "base" and isinstance(v, (np.ndarray, float, int)):
            setattr(model, k, copy.deepcopy(v))
    if getattr(b, "dense", False) and not getattr(sb, "dense", False):
        def as_dense(U):
            """Hermitian generator H with U = exp(-iH), packed as the parameters of the general coherent memory
            (real plus imaginary part), so that a trained circuit can be continued as a general memory.
            """
            T, Z = scipy.linalg.schur(np.asarray(U, dtype=np.complex128), output="complex")
            H = (Z * (-np.angle(np.diag(T)))[None, :]) @ Z.conj().T
            H = 0.5 * (H + H.conj().T)
            return (H.real + H.imag).reshape(-1)
        b.theta_v = np.array([as_dense(sb.cv.unitary(t)) for t in sb.theta_v])
        b.theta_w = np.array([as_dense(sb.cw.unitary(t)) for t in sb.theta_w])
        for k in ("h", "lo", "hi", "m", "s"):
            if hasattr(sb, k): setattr(b, k, np.array(getattr(sb, k), copy=True))
    else:
        if getattr(b, "dense", False) != getattr(sb, "dense", False):
            sys.exit("--init-from: a circuit model cannot start from a dense one")
        model.set(src.get())
        for k in ("lo", "hi", "m", "s"):
            if hasattr(sb, k): setattr(b, k, np.array(getattr(sb, k), copy=True))
    return src

# ------------------------------------------------------------------------------------------------ axis-resolved MMD (fix for spiky readouts)
def add_axis_frequencies(pool, Z_fit, k=16, share=0.2, rel=(2.0, 4.0, 8.0, 16.0), seed=7):
    """Append axis-aligned frequencies to the characteristic-function MMD pool.

    The Gaussian-mixture kernel of the training loss has bandwidths of the order of the median distance between
    showers in 369 dimensions, far wider than the spread of the data along any single principal axis. It therefore
    sees only the mean and the width of each coordinate, and it does not penalize a readout that piles probability
    into a few narrow bins (the double peaks of the layer-1 and layer-2 energies). Frequencies omega = t * p_j along
    the k leading principal axes p_j of the standardized fit data, with |t| ~ rel / sigma_j, add a kernel that
    resolves each of these axes down to sigma_j / 16. Since the decoder is x = mu + P u, omega . x = t u_j + const:
    each such frequency sees exactly one generated coordinate. The loss stays an exact MMD (a sum of characteristic
    kernels is a kernel), and its gradients go through the same exact code. `share` is the weight of the new kernel.
    """
    from comb.showers.mmd import empirical_cf
    om0, phi0 = pool
    X = np.asarray(Z_fit, float)
    ev, P = np.linalg.eigh(np.cov(X.T))
    ev, P = ev[::-1][:k], P[:, ::-1][:, :k]                   # the k axes of largest variance, as in the PCA decoder
    sig = np.sqrt(np.maximum(ev, 1e-12))
    n_add = int(round(len(om0) * share / (1.0 - share)))
    rng = np.random.default_rng(seed)
    j = rng.integers(0, k, n_add)
    t = rng.standard_normal(n_add) * rng.choice(np.asarray(rel), n_add) / sig[j]
    om = t[:, None] * P[:, j].T
    return np.concatenate([om0, om]), np.concatenate([phi0, empirical_cf(X, om)])

def add_layer_frequencies(pool, Z_fit, s0, layers, share=0.2, rel=(2.0, 4.0), use=(1, 2), energy=False, seed=8):
    """Append frequencies along the physics directions of a shower: the total-energy feature and the energy fraction of
    each calorimeter layer (sums of voxel fractions). Each is a linear function of the prepared vector, so the model's
    characteristic function along it is exact (the same identity as training). With |t| ~ rel / sigma (sigma the spread
    of the data along that direction), the loss resolves the distribution of every layer energy finely: a double peak
    in a layer energy is no longer invisible to it. A sum of kernels is a kernel, so the loss stays an exact MMD.
    Defaults: the dense layers 1 and 2 at scales 2 and 4. On Dataset 1 these see the double peak (2-3 times larger for
    the double-peaked model), while the thin layers 0, 3 and 12 are dominated by a mismatch shared by every model."""
    from comb.showers.mmd import empirical_cf
    om0, phi0 = pool
    X = np.asarray(Z_fit, float)
    d = X.shape[1]
    edges = np.concatenate([[0], np.cumsum(layers)]) + 1                         # fractions start at column 1
    dirs = [np.eye(d)[0] * s0[0]] if energy else []                             # the total-energy feature (optional)
    for k, (a, b) in enumerate(zip(edges[:-1], edges[1:])):
        if k in use: w = np.zeros(d); w[a:b] = s0[a:b]; dirs.append(w)           # sum of the layer's fractions
    dirs = np.array(dirs)
    sig = (X @ dirs.T).std(0)
    n_add = int(round(len(om0) * share / (1.0 - share)))
    rng = np.random.default_rng(seed)
    j = rng.integers(0, len(dirs), n_add)
    t = rng.standard_normal(n_add) * rng.choice(np.asarray(rel), n_add) / sig[j]
    om = t[:, None] * dirs[j]
    return np.concatenate([om0, om]), np.concatenate([phi0, empirical_cf(X, om)])


# ------------------------------------------------------------------------------------------------ spikes (readout diagnostic)
def spikes_cmd(a):
    """How peaked is each readout, and is the layer split bimodal? Peak ratio = the highest density (probability / width)
    of the 8 readout bins, relative to the average density over the occupied range (bins with p > 0.5%). A flat
    distribution gives 1, a smooth bell through quantile bins about 2 to 3; above 4, a few narrow bins carry much
    probability, which shows up as sharp peaks in the generated data."""
    need_shower_project(a.run)
    from comb.showers import zoo
    model, spec, extra = zoo.load(a.run)
    b = model.base
    rng = np.random.default_rng(0)
    C = np.asarray(b.sample_counts(a.n, rng))
    K = b.K
    delta = (b.hi - b.lo)[:, None] * np.exp(b.h) / np.exp(b.h).sum(1, keepdims=True)
    print(f"{a.run}: peak ratio of the {a.k} leading coordinates (highest density / average density over the occupied range)")
    for j in range(min(a.k, C.shape[1])):
        pj = np.bincount(C[:, j], minlength=K) / len(C)
        occ = pj > 0.005
        r = (pj / delta[j]).max() * delta[j][occ].sum() / pj[occ].sum()
        print(f"   coordinate {j:2d}: {r:5.1f}   ({occ.sum()} of {K} bins used)" + ("   <-- spiky" if r > 4 else ""))
    X = np.asarray(model.sample(a.n, np.random.default_rng(1)))
    F = X[:, 1:].clip(0)
    F = F / F.sum(1, keepdims=True)
    h, e = np.histogram(F[:, 8:168].sum(1), bins=40)
    peaks = [i for i in range(1, 39) if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] > 0.4 * h.max()]
    print(f"layer-1 energy fraction: {len(peaks)} major peak(s) at " + ", ".join(f"{0.5 * (e[i] + e[i + 1]):.3f}" for i in peaks))

# ------------------------------------------------------------------------------------------------ sample (deploy + metrics)
def sample_cmd(a):
    """Generate showers from a trained model and score them against the Geant4 test showers. Saves the showers as
    outputs/samples/<name>.npz and the metrics as <name>.metrics.json: correlation errors, per-cell Wasserstein
    distances, zeros, energy spread, shower shapes and a classifier AUC.
    """
    from comb.showers import zoo
    from scipy.stats import wasserstein_distance, spearmanr, rankdata
    def safe_corr(M):                     # Pearson matrix; a constant column gets correlation 0 with everything
        """Pearson correlation matrix of the columns of M; a constant column gets correlation 0 instead of NaN."""
        Z = np.asarray(M, dtype=float)
        Z = Z - Z.mean(0)
        sd = Z.std(0)
        ok = sd > 1e-12
        Z[:, ok] /= sd[ok]
        Z[:, ~ok] = 0.0
        return (Z.T @ Z) / len(Z)
    D = load_data()
    meta = json.load(open(DATA / "cal_ds1_meta.json"))
    sparse = np.array(meta["sparse_voxel"])
    path = a.run or str(RUNS / f"{name_of(spec_of(a))}.npz")
    name = pathlib.Path(path).stem
    if not pathlib.Path(path).exists(): sys.exit(f"no trained model at {path}; run:  python scripts/comb_ds1.py train" + ("" if a.model == "comb" else f" --model {a.model}"))
    model, spec, extra = zoo.load(path)
    Z = np.asarray(model.sample(a.n, np.random.default_rng(a.seed)), dtype=float)
    S_gen, S_te, S_tr = invert(Z, meta), invert(D["Y_te"], meta), invert(D["Y_tr"][:a.n], meta)
    perm, note = ring_major_perm(S_te, meta["layers"])
    S_gen, S_te, S_tr = S_gen[:, perm], S_te[:, perm], S_tr[:, perm]
    sparse = sparse[perm]
    print("  voxel order (ring by ring, as in the QFAN paper): " + "; ".join(note))
    rng = np.random.default_rng(1)
    Xn = D["Y_te"].copy()
    for j in range(Xn.shape[1]): Xn[:, j] = rng.permutation(Xn[:, j])
    S_null = invert(Xn, meta)[:, perm]
    np.savez(SAMPLES / f"{name}.npz", showers=S_gen, spec=json.dumps(spec))
    test_file = SAMPLES / "_test.npz"                             # the Geant4 test showers, shared by every run:
    if not test_file.exists() or not np.array_equal(np.load(test_file)["showers"], S_te):
        tmp = SAMPLES / f"_test.{os.getpid()}.npz"
        np.savez(tmp, showers=S_te)
        os.replace(tmp, test_file)   # written atomically, only if needed
    def cerr(A, meth):
        """Mean absolute difference between the off-diagonal correlations (Pearson or Spearman) of A and of the test
        showers.
        """
        C = lambda M: safe_corr(M) if meth == "pearson" else safe_corr(rankdata(M, axis=0))   # constant voxels -> 0, never a scalar NaN
        off = ~np.eye(A.shape[1], dtype=bool)
        return float(np.abs(C(A) - C(S_te))[off].mean())
    def w1(A, mask):
        """Mean Wasserstein-1 distance between A and the test showers over the cells in mask, in units of each
        cell's spread.
        """
        sd = S_te.std(0)
        v = [wasserstein_distance(A[:, j], S_te[:, j]) / sd[j] for j in np.where(mask)[0] if sd[j] > 0]
        return float(np.mean(v)) if v else float("nan")
    def row(S):
        """Every summary metric of one set of showers, as a dictionary."""
        madE = lambda M: np.median(np.abs(M.sum(1) - np.median(M.sum(1))))
        return dict(W1_dense=w1(S, ~sparse), W1_sparse=w1(S, sparse), zeros=float((S[:, sparse] == 0).mean()), pearson=cerr(S, "pearson"),
                    spearman=cerr(S, "spearman"), E_spread=float(madE(S) / madE(S_te)))
    rows = {k: row(S) for k, S in (("model", S_gen), ("null", S_null), ("floor", S_tr))}
    # shower shapes: radial centre and width of the energy in layers 1 and 2 (voxels ordered ring by ring, 10 angular bins per ring)
    edges = np.concatenate([[0], np.cumsum(meta["layers"])])
    def shapes(S):
        """Radial centre and radial width of the energy in layers 1 and 2, for each shower."""
        out = {}
        for li, nang in ((1, 10), (2, 10)):
            L = S[:, edges[li]:edges[li + 1]]
            r = np.arange(L.shape[1]) // nang
            tot = np.where(L.sum(1) > 0, L.sum(1), 1.0)
            c = (L * r).sum(1) / tot
            out[f"layer{li}_centre"] = c
            out[f"layer{li}_width"] = np.sqrt(np.maximum((L * r ** 2).sum(1) / tot - c ** 2, 0))
        return out
    sh_gen, sh_te, sh_tr = shapes(S_gen), shapes(S_te), shapes(S_tr)
    shape_w1 = {k: (float(wasserstein_distance(sh_gen[k], sh_te[k]) / sh_te[k].std()), float(wasserstein_distance(sh_tr[k], sh_te[k]) / sh_te[k].std())) for k in sh_te}
    # classifier test (CaloChallenge-style): can a classifier tell generated showers from Geant4? AUC 0.5 = indistinguishable
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    def feats(S):
        """Classifier features of each shower: log(1 + E) of every cell and of every layer, and the total energy."""
        return np.concatenate([np.log1p(S), np.log1p(np.stack([S[:, edges[i]:edges[i + 1]].sum(1) for i in range(len(meta["layers"]))], 1)), S.sum(1, keepdims=True)], 1)
    def auc(A, B):
        """Cross-validated ROC AUC of a gradient-boosted classifier that separates showers A from showers B (0.5:
        the two sets cannot be told apart).
        """
        n = min(len(A), len(B))
        Xc = np.concatenate([feats(A[:n]), feats(B[:n])])
        yc = np.r_[np.ones(n), np.zeros(n)]
        pr = cross_val_predict(HistGradientBoostingClassifier(max_iter=200, random_state=0), Xc, yc, cv=5, method="predict_proba")[:, 1]
        return float(roc_auc_score(yc, pr))
    auc_model, auc_floor = auc(S_gen, S_te), auc(S_tr, S_te)
    kappa = (rows["null"]["spearman"] - rows["model"]["spearman"]) / (rows["null"]["spearman"] - rows["floor"]["spearman"])
    json.dump(dict(run=path, spec=spec, n=a.n, rows=rows, kappa=float(kappa), shape_w1=shape_w1, auc=auc_model, auc_floor=auc_floor), open(SAMPLES / f"{name}.metrics.json", "w"), indent=1)
    r = rows["model"]
    print(f"{zoo.describe(spec)}:  kappa {100 * kappa:.0f}%   Spearman err {r['spearman']:.3f} (floor {rows['floor']['spearman']:.3f}, no-corr {rows['null']['spearman']:.3f})"
          f"   W1 dense {r['W1_dense']:.3f} sparse {r['W1_sparse']:.3f} (floor {rows['floor']['W1_dense']:.3f})   zeros {100 * r['zeros']:.1f}% (data {100 * rows['floor']['zeros']:.1f}%)   E spread {r['E_spread']:.2f}")
    print("  shower shapes, W1 in sd units (model / floor): " + "  ".join(f"{k} {v[0]:.2f}/{v[1]:.2f}" for k, v in shape_w1.items()))
    print(f"  classifier test (generated vs Geant4, 5-fold AUC; 0.5 = indistinguishable): model {auc_model:.3f}   floor (training vs test showers) {auc_floor:.3f}")
    print(f"  QFAN paper, 4 qubits, same data: kappa 48%, Spearman 0.105, W1 dense 0.276, sparse 0.381, zeros 3.8%, E spread 0.62")
    print(f"  showers saved: outputs/samples/{name}.npz ({a.n} x 368, MeV)")

# ------------------------------------------------------------------------------------------------ plot
def plot_cmd(a):
    """One figure per file (no subplots): outputs/figs/<kind>_<name>.pdf"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, Normalize
    try: import mplhep as hep; plt.style.use(getattr(hep.style, a.style)); HEP = True
    except Exception: HEP = False; print("  (mplhep not installed: pip install mplhep for the HEP style)")
    plt.rcParams.update({"font.size": 14, "legend.fontsize": 14, "legend.loc": "best", "figure.dpi": 350})
    meta = json.load(open(DATA / "cal_ds1_meta.json"))
    layers = meta["layers"]
    edges = np.concatenate([[0], np.cumsum(layers)])
    lnames = ["layer 0", "layer 1", "layer 2", "layer 3", "layer 12"]
    if not (SAMPLES / "_test.npz").exists(): sys.exit("no samples yet; run:  python scripts/comb_ds1.py sample")
    S_te = np.load(SAMPLES / "_test.npz")["showers"]
    files = [SAMPLES / f"{n}.npz" for n in a.runs] if a.runs else sorted(p for p in SAMPLES.glob("ds1_*.npz"))
    runs = []
    for i, f in enumerate(files):
        z = np.load(f)
        spec = json.loads(str(z["spec"]))
        lab = spec.get("label") or (f"HMM($\\chi={spec['chi']}$)" if spec["model"] == "hmm" else f"{'dephased twin' if spec.get('dephase') else 'CoMB'} $\\chi={2 ** spec['n_m']}$")
        col = "tab:green" if spec["model"] == "hmm" else ("tab:red" if "device" in f.stem or "ibm" in f.stem else ["tab:blue", "tab:purple", "tab:orange"][i % 3])
        runs.append(dict(name=f.stem, label=lab, style=dict(color=col, ls="--" if spec.get("dephase") else "-"), S=z["showers"]))
    if not runs: sys.exit("no generated showers in outputs/samples/; run:  python scripts/comb_ds1.py sample")
    def save(fig, name):
        """Save a figure as outputs/figs/<name>.pdf and close it."""
        fig.savefig(FIGS / f"{name}.pdf", bbox_inches="tight")
        plt.close(fig)
    def hist_ratio(xd, xms, xlabel, name, bins=40):
        """data points vs model histograms with a ratio panel -- one file. Every file has the same size and the same
        axes box (fixed margins, no tight cropping), the y axis is in scientific notation (labels of equal width), and
        the y range leaves headroom so the legend, two columns across the top, never covers the data."""
        fig = plt.figure(figsize=(5.2, 4.6))
        gs = fig.add_gridspec(2, 1, height_ratios=[3, 1], hspace=0.08, left=0.17, right=0.97, top=0.91, bottom=0.14)
        ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
        lo, hi = np.percentile(xd, [0.5, 99.5])
        pad = 0.05 * (hi - lo)
        lo, hi = max(lo - pad, xd.min()), hi + pad      # range from Geant4
        b = np.linspace(lo, hi, bins + 1)
        c = 0.5 * (b[1:] + b[:-1])
        hd, _ = np.histogram(xd, b, density=True)
        nd, _ = np.histogram(xd, b)
        ax.errorbar(c, hd, yerr=hd / np.sqrt(np.maximum(nd, 1)), fmt="k.", ms=5, label="Geant4")
        for xm, r in xms:
            hm, _ = np.histogram(xm, b, density=True)
            if HEP: hep.histplot(hm, b, ax=ax, label=r["label"], color=r["style"]["color"], linestyle=r["style"]["ls"], histtype="step")
            else: ax.step(c, hm, where="mid", label=r["label"], **r["style"])
            with np.errstate(divide="ignore", invalid="ignore"): axr.step(c, hm / hd, where="mid", **r["style"])

        top = max([np.max(hd + hd / np.sqrt(np.maximum(nd, 1)))] + [np.max(np.histogram(xm, b, density=True)[0]) for xm, _ in xms])
        ax.set_ylim(0, 1.6 * top)
        ax.set_xlim(b[0], b[-1])
        axr.set_xlim(b[0], b[-1])
        ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0), useMathText=True)
        axr.axhline(1, color="k", lw=0.6)
        axr.set_ylim(0, 2.5)
        axr.set_xlabel(xlabel)
        ax.set_ylabel("density")
        axr.set_ylabel("ratio")
        ax.set_xticklabels([])
        ax.legend(loc="upper center", ncol=2, frameon=False, fontsize=12, columnspacing=1.2, handlelength=1.5, borderaxespad=0.3)
        fig.align_ylabels([ax, axr])
        fig.savefig(FIGS / f"{name}.pdf")
        plt.close(fig)          # no tight cropping: identical size for every file




    # 1. correlation matrices: one file each (Geant4 and every run)
    for title, S, name in [("Geant4 test data", S_te, "geant4")] + [(r["label"], r["S"], r["name"]) for r in runs]:
        fig, ax = plt.subplots(figsize=(5.2, 4.6))
        im = ax.imshow(np.nan_to_num(np.corrcoef(S.T)), vmin=-1, vmax=1, cmap="RdBu_r")
        for e in edges[1:-1]: ax.axhline(e - 0.5, color="k", lw=0.4); ax.axvline(e - 0.5, color="k", lw=0.4)
        ax.set_xlabel("voxel")
        ax.set_ylabel("voxel")
        fig.colorbar(im, ax=ax, fraction=0.046, label="correlation")
        save(fig, f"corr_{name}")
    # 2. total and layer energies: one file each
    hist_ratio(S_te.sum(1), [(r["S"].sum(1), r) for r in runs], "total energy [MeV]", "energy_total")
    for li, nm in enumerate(lnames):
        f = lambda S, i=li: S[:, edges[i]:edges[i + 1]].sum(1)
        hist_ratio(f(S_te), [(f(r["S"]), r) for r in runs], f"{nm} energy [MeV]", f"energy_{nm.replace(' ', '')}")
    # 3. voxel marginals: one file each (12 voxels spread over the detector, or all 368 with --all-voxels)
    means = S_te.mean(0)
    if a.all_voxels: pick = list(range(S_te.shape[1]))
    else:
        pick = []
        for i in range(len(layers)):
            idx = np.arange(edges[i], edges[i + 1])
            order = idx[np.argsort(means[idx])]
            pick += [order[int(q * (len(order) - 1))] for q in ((0.3, 0.6, 0.9) if len(order) >= 3 else (0.5,))]
        pick = pick[:12]
    for v in pick:
        li = int(np.searchsorted(edges, v, side="right") - 1)
        hist_ratio(S_te[:, v], [(r["S"][:, v], r) for r in runs], f"{lnames[li]}, voxel {v} [MeV]", f"marginal_voxel{v:03d}", bins=30)


    # 4. mean shower per voxel, with ratio
    fig = plt.figure(figsize=(11, 4.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[3, 1.3], hspace=0.05)
    ax, axr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    x = np.arange(S_te.shape[1])
    ax.semilogy(x, np.maximum(means, 1e-9), "k.", ms=3, label="Geant4")
    for r in runs:
        m = r["S"].mean(0)
        ax.semilogy(x, np.maximum(m, 1e-9), ".", ms=3, color=r["style"]["color"], label=r["label"])
        with np.errstate(divide="ignore", invalid="ignore"): axr.plot(x, m / means, ".", ms=3, color=r["style"]["color"])
    for e in edges[1:-1]: ax.axvline(e - 0.5, color="k", lw=0.4); axr.axvline(e - 0.5, color="k", lw=0.4)
    axr.axhline(1, color="k", lw=0.6)
    axr.set_ylim(0, 2)
    axr.set_xlabel("voxel (layer by layer, ring by ring)")
    ax.set_ylabel("mean energy per voxel [MeV]")
    axr.set_ylabel("ratio")
    ax.legend(loc='best')
    ax.set_xticklabels([])
    save(fig, "mean_shower")
    # 5. radial profiles: one file per layer (layers 1 and 2)
    for li in (1, 2):
        L = S_te[:, edges[li]:edges[li + 1]]
        nr_ = L.shape[1] // 10
        prof = lambda M: M.reshape(len(M), nr_, 10).sum(2).mean(0)
        fig, ax = plt.subplots(figsize=(5.2, 4.2))
        ax.semilogy(np.arange(nr_), prof(L), "k.", ms=6, label="Geant4")
        for r in runs: ax.semilogy(np.arange(nr_), prof(r["S"][:, edges[li]:edges[li + 1]]), label=r["label"], **r["style"])
        ax.set_xlabel(f"{lnames[li]}: ring (radial bin)")
        ax.set_ylabel("mean energy per ring [MeV]")
        ax.legend(loc='best')
        save(fig, f"radial_{lnames[li].replace(' ', '')}")
   
   
   
    # 6. polar maps of the average shower: one file per (quantity, layer): Geant4, each run, each ratio
    def layer_map(S, li):
        """Mean energy per cell of layer li, arranged by (radial, angular) bin."""
        return S[:, edges[li]:edges[li + 1]].mean(0).reshape(RAD[li], ANG[li])
    
        
# ---- polar maps of the average shower per layer (QFAN Fig. 10): data / model / ratio, one figure per run
    from matplotlib.colors import LogNorm, Normalize
    ANG, RAD = [1, 10, 10, 1, 1], [8, 16, 19, 5, 5]                       # Dataset 1 photon geometry: angular x radial bins per layer
    def layer_map(S, li):                                                   # mean energy per voxel as (radial, angular), ring-major order
        """Mean energy per cell of layer li, arranged by (radial, angular) bin."""
        M = S[:, edges[li]:edges[li + 1]].mean(0)
        return M.reshape(RAD[li], ANG[li])
    for r in runs:
        fig, axes = plt.subplots(3, len(layers), figsize=(3.0 * len(layers), 8.6), subplot_kw=dict(projection="polar"))
        for li in range(len(layers)):
            Zd, Zm = layer_map(S_te, li), layer_map(r["S"], li)
            na = ANG[li]
            if na == 1: Zd, Zm, na = np.repeat(Zd, 36, 1), np.repeat(Zm, 36, 1), 36      # a full-circle cell does not draw; repeat it
            th = np.linspace(0, 2 * np.pi, na + 1)
            rr = np.arange(RAD[li] + 1)
            TH, RR = np.meshgrid(th, rr)
            lo = max(min(Zd[Zd > 0].min(), Zm[Zm > 0].min()), 1e-6 * Zd.max())
            hi = max(Zd.max(), Zm.max())
            norm = LogNorm(lo, hi)
            for row, Z, title in ((0, Zd, "Geant4"), (1, Zm, r["label"])):
                ax = axes[row, li]
                pc = ax.pcolormesh(TH, RR, np.maximum(Z, lo), norm=norm, cmap="viridis", shading="flat", edgecolors="face")
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_title(f"{lnames[li]}" if row == 0 else "", fontsize=14)
                ax.set_ylabel(title if li == 0 else "", labelpad=25, fontsize=14)
                cb = fig.colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, fraction=0.1, label="" if li == 2 else None)
                cb.ax.tick_params(labelsize=14)
            ax = axes[2, li]
            with np.errstate(divide="ignore", invalid="ignore"): ratio = np.where(Zd > 0, Zm / Zd, np.nan)
            pc = ax.pcolormesh(TH, RR, ratio, norm=Normalize(0.6, 1.4), cmap="coolwarm", shading="flat", edgecolors="face")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_ylabel("ratio" if li == 0 else "", labelpad=25, fontsize=14)
            fig.colorbar(pc, ax=ax, orientation="horizontal", pad=0.05, fraction=0.05, label="" if li == 2 else None)
        fig.savefig(FIGS / f"ds1_polar_{r['name']}.pdf", bbox_inches="tight")
        plt.close(fig)
    logs = [(r, json.load(open(RUNS / f"{r['name']}.json"))) for r in runs if (RUNS / f"{r['name']}.json").exists()]
    if not logs:                                  # fine-tuned or device-started runs without a training log: no empty figure
        print("  training curves: no training log (outputs/runs/<run>.json) for these runs; ds1_training.pdf not written")
    else:
        fig, ax = plt.subplots(figsize=(6, 4))
        for r, j in logs:
            h = j["hist"]
            ax.semilogy(h["step"][1:], h["val"][1:], label=r["label"], **r["style"])
            for st in j.get("stages", []):        # stage boundaries of a multi-stage run (e.g. fine-tuning after training)
                ax.axvline(st["step"], color="0.6", ls=":", lw=1)
                ax.text(st["step"], 0.97, " " + st["label"], transform=ax.get_xaxis_transform(), rotation=90, va="top", ha="left", fontsize=8, color="0.35")
        ax.set_xlabel("optimizer step")
        ax.set_ylabel("validation MMD$^2$ (exact)")
        ax.legend(fontsize=14, loc="best")
        fig.savefig(FIGS / "ds1_training.pdf", bbox_inches="tight")
        plt.close(fig)
    print("figures: outputs/figs/ds1_corr.pdf  ds1_energies.pdf  ds1_marginals.pdf  ds1_mean_shower.pdf  ds1_radial.pdf  ds1_polar_<run>.pdf  ds1_training.pdf   (runs: " + ", ".join(r["name"] for r in runs) + ")")




if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("prepare")
    for c in ("train", "sample"):
        p = sub.add_parser(c)
        p.add_argument("--model", choices=["comb", "dephased", "hmm", "envmem", "envdeph"], default="comb")
        p.add_argument("--env", type=int, default=0, help="environment dimension E of envmem/envdeph (default chi^2)")
        p.add_argument("--nw", type=int, default=3, help="work qubits of comb/dephased (K = 2^nw levels per value)")
        p.add_argument("--K", type=int, default=8, help="levels per value of hmm/envmem/envdeph")
        p.add_argument("--chi", type=int, default=4)
        p.add_argument("--Lv", type=int, default=5, help="layers of the round circuit V (CoMB and twin); scale with the memory, e.g. 18 for chi = 8")
        p.add_argument("--Lw", type=int, default=2, help="layers of the memory circuit W")
        p.add_argument("--lr", type=float, default=None, help="peak learning rate of the circuit angles (default 0.1; HMM 0.2)")
        p.add_argument("--tag", default="", help="label added to the run name, e.g. _lr0.05")
        p.add_argument("--init-from", default="", help="start from a trained run (name in outputs/runs or a path); a dense model starts from a circuit model's exact unitaries")
        p.add_argument("--dense", action="store_true", help="most general coherent memory: a dense unitary exp(-iH) in place of the round circuit (CoMB and twin)")
        p.add_argument("--wiring", choices=["interleaved", "nested"], default="interleaved", help="coupling line of the round circuit: interleaved (default) or nested (extra memory qubits after the work register)")
        p.add_argument("--seed", type=int, default=0)
        if c == "train":
            p.add_argument("--steps", type=int, default=3000)
            p.add_argument("--axis-share", type=float, default=0.0, help="weight of the axis-resolved MMD kernel (0: the paper's original loss; 0.2 fixes spiky readouts)")
            p.add_argument("--axis-k", type=int, default=16, help="number of leading principal axes resolved by the axis kernel")
            p.add_argument("--layer-share", type=float, default=0.0, help="weight of the layer-resolved MMD kernel (0: off)")
            p.add_argument("--layers", default="1,2", help="layers resolved (index into the layer list; default: the dense layers 1 and 2)")
            p.add_argument("--layer-scales", default="2,4", help="scales r of the layer frequencies, |t| ~ r / sigma")
        else: p.add_argument("--n", type=int, default=1200); p.add_argument("--run", default=None)
    ps = sub.add_parser("spikes")
    ps.add_argument("--run", required=True)
    ps.add_argument("--n", type=int, default=100000)
    ps.add_argument("--k", type=int, default=12)
    pp = sub.add_parser("plot")
    pp.add_argument("--runs", nargs="*", default=None)
    pp.add_argument("--style", default="ATLAS", help="mplhep style: ATLAS, CMS, LHCb2, ALICE")
    pp.add_argument("--all-voxels", action="store_true", help="one marginal figure for every one of the 368 voxels")
    a = ap.parse_args()
    dict(prepare=prepare, train=train_cmd, sample=sample_cmd, plot=plot_cmd, spikes=spikes_cmd)[a.cmd](a)
