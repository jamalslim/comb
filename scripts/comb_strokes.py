#!/usr/bin/env python3
"""Stroke-CoMB: a coherent memory register that draws digits one pen move per round,
trained with the exact k-point correlation MMD (comb/kpoint.py).

    python scripts/comb_strokes.py prepare  --data mnist            # or --data uci (sklearn's bundled 8x8 digits)
    python scripts/comb_strokes.py train    --data mnist --model comb --chi 4 --seed 0
    python scripts/comb_strokes.py train    --data mnist --model dephased --chi 4 --seed 0
    python scripts/comb_strokes.py train    --data mnist --model hmm --chi 4 --seed 0
    python scripts/comb_strokes.py evaluate --data mnist --run strokes_mnist_comb_chi4_s0
    python scripts/comb_strokes.py plot     --data mnist

One round per pen move: the work register (4 qubits) emits a token in 0..15 = 8 directions x pen up/down,
the memory (2 or 3 qubits) is never measured.  Every sample is one digit, drawn by one run of the circuit.
"""
import argparse, json, pathlib, pickle, sys, time
import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comb.digits.model import CoMB, ClassicalHMM
from comb.digits.hqmm import GeneralHQMM
from comb.digits.treecomb import TreeCoMB
from comb.digits import kpoint as kp
from comb.digits import strokes as st
from comb.digits import pencf

OUT = ROOT / "outputs" / "strokes"
DATA = ROOT / "data"
MNIST_COUNTS = {"train": [5923, 6742, 5958, 6131, 5842, 5421, 5918, 6265, 5851, 5949],
                "test": [980, 1135, 1032, 1010, 982, 892, 958, 1028, 974, 1009]}
LAGS = [3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32, 40, 48, 64, 80, 96]


def log(msg, path=None):
    """Print a message and append it to the log file, if one is given."""
    print(msg, flush=True)
    if path is not None:
        with open(path, "a") as f:
            f.write(msg + "\n")


# ============================================================================ data
def load_images(name):
    """Load the digit images with their labels, split into training, validation and test sets, together with the ink
    threshold: MNIST, or the UCI digits upsampled to 28 x 28.
    """
    if name == "mnist":
        m = st.load_mnist(DATA)
        for split, key in (("train", "y_train"), ("test", "y_test")):
            counts = np.bincount(m[key], minlength=10).tolist()
            if counts != MNIST_COUNTS[split]:
                raise SystemExit(f"this is not the official MNIST {split} set (class counts {counts}); delete {DATA / 'mnist.npz'} and retry")
        mean = m["x_train"].mean()
        if not 33.0 < mean < 33.6:
            raise SystemExit(f"MNIST train pixel mean {mean:.2f} is not 33.32; the cached file is not official MNIST")
        rng = np.random.default_rng(2026)
        perm = rng.permutation(60000)
        tr, va = perm[:55000], perm[55000:]
        return dict(x_train=m["x_train"][tr], y_train=m["y_train"][tr], x_val=m["x_train"][va], y_val=m["y_train"][va],
                    x_test=m["x_test"], y_test=m["y_test"], threshold=128, source="MNIST (official 60k/10k; 55k/5k train/validation split, seed 2026)")
    if name == "uci":
        from scipy import ndimage
        from sklearn.datasets import load_digits
        D = load_digits()
        def up(img8):
            """Upsample an 8 x 8 UCI digit (grey levels 0 to 16) to 28 x 28: cubic-spline zoom by 3, clipped to [0,
            1], centred with a 2-pixel border, as 8-bit grey levels.
            """
            z = ndimage.zoom(img8 / 16.0, 3.0, order=3).clip(0, 1)
            out = np.zeros((28, 28))
            out[2:26, 2:26] = z
            return (out * 255).round().astype(np.uint8)
        X = np.stack([up(a) for a in D.images])
        y = D.target.astype(np.uint8)
        rng = np.random.default_rng(2026)
        perm = rng.permutation(len(X))
        tr, va, te = perm[:1200], perm[1200:1497], perm[1497:]
        return dict(x_train=X[tr], y_train=y[tr], x_val=X[va], y_val=y[va], x_test=X[te], y_test=y[te], threshold=110,
                    source="UCI optical digits (scikit-learn load_digits, 1797 images, 8x8 upsampled to 28x28); 1200/297/300 split, seed 2026")
    raise SystemExit(f"unknown data {name}")


def prepare(a):
    """Turn the digit images into pen-stroke token sequences (skeleton, then a covering walk) and save them as
    data/strokes_<data>.npz, with a digit classifier (data/strokes_<data>_classifier.pkl) for judging generated
    samples.
    """
    t0 = time.time()
    D = load_images(a.data)
    thr = D["threshold"]
    enc = {}
    for split in ("train", "val", "test"):          # one pass at length 256; lengths are returned untruncated
        T, S, L, P = st.encode(D[f"x_{split}"], d=256, threshold=thr, log=print)
        enc[split] = (T, L)
    L_all = enc["train"][1]
    d = int(a.d) if a.d else int(min(128, 8 * np.ceil(np.percentile(L_all, 90) / 8)))
    out = dict(source=D["source"], d=d, K=st.K_TOKENS, threshold=thr)
    for split in ("train", "val", "test"):          # slicing equals encoding at length d (same walk, same padding)
        T, L = enc[split]
        out[f"tokens_{split}"] = np.ascontiguousarray(T[:, :d])
        out[f"labels_{split}"] = D[f"y_{split}"]
        out[f"lengths_{split}"] = L
    trunc = float((L_all > d).mean())
    path = DATA / f"strokes_{a.data}.npz"
    DATA.mkdir(exist_ok=True)
    np.savez_compressed(path, **{k: v for k, v in out.items() if isinstance(v, np.ndarray)}, meta=json.dumps(dict(source=out["source"], d=d, K=st.K_TOKENS, threshold=thr)))
    print(f"{D['source']}\n  d = {d} pen moves per digit (90th percentile of walk lengths; {100 * trunc:.1f}% of training digits truncated)")
    print(f"  walk length median {np.median(L_all):.0f}, max {L_all.max()}; saved {path}  ({time.time() - t0:.0f}s)")
    # digit classifier on round-trip renders: measures what the representation keeps, then scores model samples
    from sklearn.neural_network import MLPClassifier
    R_tr = st.render_many(out["tokens_train"][:20000]).reshape(-1, 784)
    R_te = st.render_many(out["tokens_test"]).reshape(-1, 784)
    clf = MLPClassifier(hidden_layer_sizes=(256,), max_iter=200, random_state=0, early_stopping=True).fit(R_tr, out["labels_train"][:20000])
    acc = float((clf.predict(R_te) == out["labels_test"]).mean())
    pickle.dump(clf, open(DATA / f"strokes_{a.data}_classifier.pkl", "wb"))
    print(f"  classifier on round-trip renders: test accuracy {100 * acc:.1f}% (the ceiling the representation allows)")


def load_prepared(name, digit=-1):
    """Prepared sequences; with digit >= 0 every split keeps only that digit (class-conditional CoMB)."""
    f = np.load(DATA / f"strokes_{name}.npz", allow_pickle=True)
    meta = json.loads(str(f["meta"]))
    z = {k: f[k] for k in f.files}
    if digit is not None and digit >= 0:
        for split in ("train", "val", "test"):
            keep = z[f"labels_{split}"] == digit
            for key in ("tokens", "labels", "lengths"):
                z[f"{key}_{split}"] = z[f"{key}_{split}"][keep]
    return z, meta


# ============================================================================ models
def build(model, d, chi, seed, L_v, L_w, n_env=0, L_node=4):
    """Untrained model of the given kind: the CoMB circuit or its dephased twin, the classical HMM, the measurement-
    tree CoMB (one work qubit) or the general coherent memory (hqmm), with or without dephasing.
    """
    n_m = int(round(np.log2(chi)))
    if model in ("comb", "dephased"):
        cur = f"binary_env:{n_env}" if n_env else "binary"
        return CoMB(d, n_m=n_m, n_w=4 + n_env, n_sub=1, L_v=L_v, L_w=L_w, seed=seed, current=cur, dephase=(model == "dephased"))
    if model == "hmm":
        return ClassicalHMM(d, chi=chi, K=st.K_TOKENS, seed=seed)
    if model in ("tree", "tree_dephased"):     # one work qubit, 4 binary sub-rounds per pen move, feed-forward
        return TreeCoMB(d, n_m=n_m, n_bits=4, L_node=L_node, L_w=L_w, seed=seed, dephase=(model == "tree_dephased"))
    if model in ("hqmm", "hqmm_dephased"):   # any circuit on memory + work: unrestricted isometry, 2^n_env environment levels
        return GeneralHQMM(d, chi=chi, K=st.K_TOKENS, E=2 ** n_env, seed=seed, dephase=(model == "hqmm_dephased"))
    raise SystemExit(f"unknown model {model}")


def run_name(a):
    """Run name from the options, e.g. strokes_uci_comb_chi4_path2_s0: data, model, memory size, environment or
    depth, path-loss weight, digit (class-conditional runs) and seed.
    """
    if a.model.startswith("tree"):
        env = f"_L{a.L_node}"
    elif a.model.startswith("hqmm"):
        env = f"_e{a.n_env}"
    elif a.model in ("comb", "dephased") and (a.n_env or a.L_v != 4):
        env = f"_e{a.n_env}L{a.L_v}"
    else:
        env = ""
    path = f"_path{a.path:g}" if a.path > 0 else ""
    path += f"_digit{a.digit}" if a.digit >= 0 else ""
    return f"strokes_{a.data}_{a.model}_chi{a.chi}{env}{path}_s{a.seed}"


def stats_for(z, d, kmax, split, n=None):
    """Window and pair statistics of the token sequences of one split: windows of up to kmax moves, pairs at the
    configured lags.
    """
    T = z[f"tokens_{split}"][:n] if n else z[f"tokens_{split}"]
    return kp.data_statistics(T.astype(np.int64), st.K_TOKENS, kmax=kmax, lags=[l for l in LAGS if l < d])


def term_weights(stats, G):
    """Normalise each term by its reference value on the training statistics: uniform tokens for the
    one-point term, independent tokens with the data's one-point laws for windows k >= 2 and pairs.
    The loss then reads 'sum over terms of (model discrepancy / reference discrepancy)'."""
    K, d = stats["K"], stats["d"]
    unif = dict(stats, W=[np.full_like(w, 1.0 / w[0].size) for w in stats["W"]], pairs=np.full_like(stats["pairs"], 1.0 / K ** 2))
    null = kp.independent_statistics(stats)
    _, vu, _, _ = kp.mmd_terms(unif, stats, G, (1, 1, 1, 1))
    _, vn, _, _ = kp.mmd_terms(null, stats, G, (1, 1, 1, 1))
    ref = [vu["window1"]] + [vn[f"window{k}"] for k in range(2, stats["kmax"] + 1)] + [vn["pairs"]]
    return tuple(1.0 / max(r, 1e-12) for r in ref), ref


def train(a):
    """Train one digits model with the exact k-point loss, plus the path loss when --path > 0, resuming from its
    checkpoint. Saves outputs/strokes/<name>.npz, its training record (<name>.json) and its log.
    """
    z, meta = load_prepared(a.data, a.digit)
    d = meta["d"]
    G = kp.token_kernel(st.K_TOKENS, a.kernel, kappa=a.kappa, rho_pen=a.rho_pen)
    s_tr = stats_for(z, d, 3, "train")
    s_va = stats_for(z, d, 3, "val")
    w, ref = term_weights(s_tr, G)
    model = build(a.model, d, a.chi, a.seed, a.L_v, a.L_w, a.n_env, a.L_node)
    name = run_name(a)
    OUT.mkdir(parents=True, exist_ok=True)
    logf = OUT / f"{name}.log"
    logf.write_text("")
    lr = a.lr if a.lr else {"hmm": 0.2, "hqmm": 0.05, "hqmm_dephased": 0.05}.get(a.model, 0.1)
    spec = dict(data=a.data, model=a.model, chi=a.chi, seed=a.seed, d=d, K=st.K_TOKENS, L_v=a.L_v, L_w=a.L_w, n_env=a.n_env, L_node=a.L_node, digit=a.digit, kernel=a.kernel, kappa=a.kappa,
                rho_pen=a.rho_pen, lr=lr, steps=a.steps, lags=s_tr["lags"], weights=list(w), reference=list(ref), source=meta["source"],
                n_params=int(model.n_inst_params()))
    log(f"{name}: {model.n_inst_params()} trainable parameters, d = {d}, K = 16, lr {lr}, {len(s_tr['pair_t'])} pair statistics", logf)
    extra = extra_val = None
    if a.path > 0:
        pc = pencf.PathCF(z["tokens_train"], d, st.K_TOKENS, every=a.path_every, seed=0)
        X_tr, X_va = pencf.positions(z["tokens_train"], pc.taus), pencf.positions(z["tokens_val"], pc.taus)
        ref = pc.reference(X_tr, s_tr["W"][0], pc.frequencies(4096, np.random.default_rng(7)))
        om_val = pc.frequencies(2048, np.random.default_rng(11))
        def extra(m, step):
            """Path term of the training loss and its gradient, with fresh random frequencies at every step, scaled
            like the k-point terms.
            """
            L, g = pc.loss(m, X_tr, pc.frequencies(a.R_path, np.random.default_rng(10_000 + step)), grad=True)
            return a.path * L / ref, (a.path / ref) * g
        def extra_val(m):
            """Path term of the validation loss, at fixed frequencies."""
            return a.path * pc.loss(m, X_va, om_val, grad=False)[0] / ref
        spec.update(path=a.path, path_every=a.path_every, R_path=a.R_path, path_reference=ref, path_sigma=pc.sigma)
        log(f"  path term: positions every {a.path_every} moves ({len(pc.taus)} times), bandwidth {pc.sigma:.1f}, reference {ref:.3e}", logf)
    res = kp.train_kpoint(model, s_tr, s_va, G, weights=w, steps=a.steps, lr=lr, warmup=50, val_every=25,
                          log=lambda m: log(m, logf), ckpt=OUT / f"{name}.ckpt.npz", extra=extra, extra_val=extra_val)
    np.savez(OUT / f"{name}.npz", params=model.get(), spec=json.dumps(spec))
    json.dump(dict(name=name, spec=spec, **res), open(OUT / f"{name}.json", "w"), indent=1)
    log(f"done: best validation loss {res['best_val']:.4f} at step {res['best_step']} ({res['train_time'] / 60:.1f} min)", logf)


def load_run(name):
    """Load a trained digits model and its specification from outputs/strokes/<name>.npz."""
    z = np.load(OUT / f"{name}.npz", allow_pickle=True)
    spec = json.loads(str(z["spec"]))
    m = build(spec["model"], spec["d"], spec["chi"], spec["seed"], spec["L_v"], spec["L_w"], spec.get("n_env", 0), spec.get("L_node", 4))
    m.set(z["params"])
    return m, spec


# ============================================================================ evaluation
def two_sample_auc(A, B, seed=0):
    """Cross-validated ROC AUC of a logistic regression that separates feature sets A and B (0.5:
    indistinguishable).
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_predict
    from sklearn.metrics import roc_auc_score
    n = min(len(A), len(B))
    X = np.concatenate([A[:n], B[:n]])
    y = np.r_[np.zeros(n), np.ones(n)]
    p = cross_val_predict(LogisticRegression(C=0.1, max_iter=2000), X, y, cv=5, method="predict_proba")[:, 1]
    return float(roc_auc_score(y, p))


def evaluate(a):
    """Held-out evaluation of a trained run on the test digits: exact k-point and path losses, generated samples and
    classifier checks. Writes outputs/strokes/<run>.metrics.json and <run>.samples.npz.
    """
    model, spec = load_run(a.run)
    z, meta = load_prepared(a.data, spec.get("digit", -1))
    d = meta["d"]
    G = kp.token_kernel(st.K_TOKENS, spec["kernel"], kappa=spec["kappa"], rho_pen=spec["rho_pen"])
    w = tuple(spec["weights"])
    s_tr, s_te = stats_for(z, d, 3, "train"), stats_for(z, d, 3, "test")
    s_m, _ = kp.model_statistics(model, s_te)
    null = kp.independent_statistics(s_te)
    rows = {}
    for label, s in (("model", s_m), ("null", null), ("floor", s_tr)):
        loss, vals, _, _ = kp.mmd_terms(s, s_te, G, w)
        rows[label] = dict(loss=loss, **{k: v for k, v in vals.items() if k != "pairs_by_lag"})
        rows[label]["pairs_by_lag"] = vals["pairs_by_lag"]
    kappa = {t: (rows["null"][t] - rows["model"][t]) / max(rows["null"][t] - rows["floor"][t], 1e-15) for t in ("window2", "window3", "pairs")}
    # global shape: exact path MMD on held-out digits (positions every 4 moves), for every run
    pc = pencf.PathCF(z["tokens_train"], d, st.K_TOKENS, every=4, seed=0)
    om = pc.frequencies(2048, np.random.default_rng(13))
    W = pc.weights(om)
    X_te = pencf.positions(z["tokens_test"], pc.taus)
    ph_te = pc.empirical_cf(X_te, om)
    path = dict(model=float(np.mean(np.abs(ph_te - pc.model_cf(model, W)[0]) ** 2)),
                null=float(np.mean(np.abs(ph_te - pc.null_cf(s_te["W"][0], W)) ** 2)),
                floor=float(np.mean(np.abs(ph_te - pc.empirical_cf(pencf.positions(z["tokens_train"], pc.taus), om)) ** 2)))
    kappa["path"] = (path["null"] - path["model"]) / max(path["null"] - path["floor"], 1e-15)
    rng = np.random.default_rng(1)
    n = a.n
    T_m = model.sample_counts(n, rng).astype(np.int8)
    T_te = z["tokens_test"][:n]
    g_m, g_te, g_tr = st.geometry_many(T_m), st.geometry_many(T_te), st.geometry_many(z["tokens_train"][:n])
    from scipy.stats import wasserstein_distance as w1
    sd = g_te.std(0) + 1e-9
    geo = {name: dict(model=float(w1(g_m[:, j], g_te[:, j]) / sd[j]), floor=float(w1(g_tr[:, j], g_te[:, j]) / sd[j])) for j, name in enumerate(st.GEOMETRY_NAMES)}
    R_m = st.render_many(T_m).reshape(-1, 784)
    R_te = st.render_many(T_te).reshape(-1, 784)
    R_tr = st.render_many(z["tokens_train"][:n]).reshape(-1, 784)
    clf = pickle.load(open(DATA / f"strokes_{a.data}_classifier.pkl", "rb"))
    def digits(R):
        """Digit classes the classifier assigns to rendered samples R: mean confidence, entropy of the class
        frequencies (in bits) and the class frequencies.
        """
        P = clf.predict_proba(R)
        lab = P.argmax(1)
        freq = np.bincount(lab, minlength=10) / len(lab)
        return dict(confidence=float(P.max(1).mean()), class_entropy_bits=float(-(freq[freq > 0] * np.log2(freq[freq > 0])).sum()), class_freq=freq.tolist())
    out = dict(run=a.run, spec=spec, n=n, kmmd=rows, path_mmd=path, kappa=kappa, geometry=geo,
               auc=two_sample_auc(R_m, R_te), auc_floor=two_sample_auc(R_tr, R_te),
               digits=dict(model=digits(R_m), test=digits(R_te)))
    if spec.get("digit", -1) >= 0:                  # class-conditional: does the model draw the intended digit?
        k = spec["digit"]
        out["intended"] = dict(digit=k, model=float((clf.predict(R_m) == k).mean()), test=float((clf.predict(R_te) == k).mean()))
        print(f"  intended digit {k}: classifier reads {100 * out['intended']['model']:.1f}% of samples as {k} "
              f"(real digits of class {k}: {100 * out['intended']['test']:.1f}%)")
    json.dump(out, open(OUT / f"{a.run}.metrics.json", "w"), indent=1)
    np.savez_compressed(OUT / f"{a.run}.samples.npz", tokens=T_m)
    print(f"{a.run}\n  held-out k-point loss: model {rows['model']['loss']:.3f}  null {rows['null']['loss']:.3f}  floor {rows['floor']['loss']:.3f}")
    print("  kappa (share of correlation captured): " + ", ".join(f"{k} {100 * v:.0f}%" for k, v in kappa.items()))
    print("  geometry W1/sd (model | floor): " + "; ".join(f"{k} {v['model']:.2f}|{v['floor']:.2f}" for k, v in geo.items()))
    print(f"  two-sample AUC on renders: model {out['auc']:.3f} (floor {out['auc_floor']:.3f});  classifier confidence {out['digits']['model']['confidence']:.2f} "
          f"(test {out['digits']['test']['confidence']:.2f}), class entropy {out['digits']['model']['class_entropy_bits']:.2f} bits (test {out['digits']['test']['class_entropy_bits']:.2f})")


# ============================================================================ plots
def plot(a):
    """Figures of all evaluated runs of one dataset, in outputs/figs/: generated strokes next to test digits, pair
    statistics by lag, and training curves.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 8, "axes.spines.top": False, "axes.spines.right": False})
    z, meta = load_prepared(a.data)
    runs = sorted(p.name[:-len(".metrics.json")] for p in OUT.glob(f"strokes_{a.data}_*.metrics.json"))
    if not runs:
        raise SystemExit("no evaluated runs")
    figs = ROOT / "outputs" / "figs"
    figs.mkdir(parents=True, exist_ok=True)
    cols = [("test digits", z["tokens_test"][:48])] + [(r.replace(f"strokes_{a.data}_", ""), np.load(OUT / f"{r}.samples.npz")["tokens"][:48]) for r in runs]
    fig, ax = plt.subplots(1, len(cols), figsize=(2.2 * len(cols), 2.6))
    ax = np.atleast_1d(ax)
    for axi, (title, T) in zip(ax, cols):
        R = st.render_many(T)
        grid = np.ones((6 * 29, 8 * 29))
        for i, img in enumerate(R[:48]):
            r, c = divmod(i, 8)
            grid[r * 29:r * 29 + 28, c * 29:c * 29 + 28] = 1 - img
        axi.imshow(grid, cmap="gray", vmin=0, vmax=1)
        axi.set_title(title, fontsize=8)
        axi.axis("off")
    fig.savefig(figs / f"strokes_{a.data}_samples.pdf", bbox_inches="tight", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    for r in runs:
        h = json.load(open(OUT / f"{r}.json"))["hist"]
        ax.semilogy(h["step"], h["val"], label=r.replace(f"strokes_{a.data}_", ""))
    ax.set_xlabel("step")
    ax.set_ylabel("validation k-point loss")
    ax.legend(fontsize=7, frameon=False)
    fig.savefig(figs / f"strokes_{a.data}_training.pdf", bbox_inches="tight")
    plt.close(fig)
    lags = None
    fig, ax = plt.subplots(figsize=(5.5, 2.6))
    for i, r in enumerate(runs):
        m = json.load(open(OUT / f"{r}.metrics.json"))
        pl = m["kmmd"]["model"]["pairs_by_lag"]
        lags = sorted(int(k) for k in pl)
        ax.semilogy(lags, [pl[str(l)] for l in lags], "o-", ms=3, label=r.replace(f"strokes_{a.data}_", ""))
        if i == 0:
            nl = m["kmmd"]["null"]["pairs_by_lag"]
            fl = m["kmmd"]["floor"]["pairs_by_lag"]
            ax.semilogy(lags, [nl[str(l)] for l in lags], "k--", lw=1, label="independent tokens")
            ax.semilogy(lags, [fl[str(l)] for l in lags], "k:", lw=1, label="floor (train vs test)")
    ax.set_xlabel("lag between pen moves")
    ax.set_ylabel("pair MMD$^2$ (held out)")
    ax.legend(fontsize=6.5, frameon=False)
    fig.savefig(figs / f"strokes_{a.data}_pairs_by_lag.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"figures in {figs}: strokes_{a.data}_samples.pdf, _training.pdf, _pairs_by_lag.pdf")


def parser():
    """The command-line options of comb_strokes.py."""
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "train", "evaluate", "plot"])
    ap.add_argument("--data", default="mnist", choices=["mnist", "uci"])
    ap.add_argument("--d", type=int, default=0)
    ap.add_argument("--model", default="comb", choices=["comb", "dephased", "hmm", "hqmm", "hqmm_dephased", "tree", "tree_dephased"])
    ap.add_argument("--chi", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=1500)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--L_v", type=int, default=4)
    ap.add_argument("--L_w", type=int, default=2)
    ap.add_argument("--n_env", type=int, default=0, help="environment work qubits: measured and reset, not emitted")
    ap.add_argument("--L_node", type=int, default=4, help="layers of each node circuit of the measurement-tree CoMB")
    ap.add_argument("--digit", type=int, default=-1, help="train and evaluate on one digit only (class-conditional CoMB)")
    ap.add_argument("--path", type=float, default=0.0, help="weight of the exact path MMD (global shape); 0 = off")
    ap.add_argument("--path_every", type=int, default=4)
    ap.add_argument("--R_path", type=int, default=256)
    ap.add_argument("--kernel", default="stroke", choices=["stroke", "delta"])
    ap.add_argument("--kappa", type=float, default=2.0)
    ap.add_argument("--rho_pen", type=float, default=0.3)
    ap.add_argument("--run", default="")
    ap.add_argument("--n", type=int, default=2000)
    return ap


def main():
    """Run the command given on the command line: prepare, train, evaluate or plot."""
    a = parser().parse_args()
    dict(prepare=prepare, train=train, evaluate=evaluate, plot=plot)[a.cmd](a)


if __name__ == "__main__":
    main()
