#!/usr/bin/env python3
"""Every figure of the handwritten-digits application, from the outputs of comb2.

    cd comb
    python plot/digits_plots.py --data mnist       # figures -> outputs/paper_figures/
    python plot/digits_plots.py --data uci

This folder (plot/) sits inside the comb folder. The digits runs are read from the comb folder (comb_strokes.py,
data/strokes_<data>.npz, outputs/strokes/); if they were made in a separate comb2 folder, the comb2 folder next
to the comb folder is used, or give it with --comb2.

Every plot is a single plot in its own file, without a title, drawn at the size it is printed. Data points carry
statistical errors and are never joined by lines; held-out results take their errors from ERRORS_<data>.json,
written by  python scripts/statistical_errors.py --data <data>  (a jackknife over the test digits):
  <data>_samples_{test,strokes,comb}.pdf                      a third of the page each
  <data>_statistics_{mi,penmoves,height,turning}.pdf          one column each
  <data>_pixcorr_{test,coherent,dephased,hmm,tree,conditional}.pdf   pixel-pixel correlation matrices (a third each)
  <data>_memory_compression.pdf                               held-out objective against memory size (one column)
  <data>_circuits_{total,local}.pdf                           every circuit next to its dephased twin (one column each)
  <data>_realism.pdf                                          two-sample AUC of every model (one column)
  <data>_noise_{objective,difference}.pdf                     one column each
  <data>_recognition.pdf, <data>_training.pdf                 one column each
A figure whose runs do not exist yet is skipped with a message.

* samples      real test digits | the same digits as pen strokes | CoMB samples, one measurement-tree CoMB
               per class (two memory qubits, one work qubit); one row per digit
* statistics   what the losses measure, for the test digits and the chi = 4 models: (a) dependence between
               pen moves against their distance (mutual information), (b) pen-down length, (c) height,
               (d) total turning; data as points with statistical errors, models as lines
* recognition  per digit: share of samples a classifier trained on real digits reads as intended, next to
               the share for real digits (the ceiling of the representation) and the 0.8 x ceiling criterion
* comparison   held-out k-point loss of every run, seeds as points, mean and sd as bars: (a) the memories at
               chi = 4, (b) circuits of fixed depth and their dephased twins
* memory       classical HMM loss against its memory size, and where the 4-level coherent memory falls
* noise        held-out objective of the circuits against the strength of Heron-like noise; ion-like marker
* training     validation loss against training step, every seed
"""
import argparse, json, pathlib, sys
import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
COMB = HERE.parent
sys.path.insert(0, str(HERE))


def digits_folder(given=None):
    """the folder with the digits runs: --comb2 if given, else the comb folder, else a comb2 folder next to it"""
    if given:
        return pathlib.Path(given).resolve()
    for cand in (COMB, COMB / "comb2", COMB.parent / "comb2"):
        if (cand / "scripts" / "comb_strokes.py").exists() and (cand / "outputs" / "strokes").exists():
            return cand
    return COMB
import hepstyle as hs                                                         # noqa: E402
plt = hs.plt
COL, WIDE = hs.COL, hs.WIDE
C = dict(coh=hs.COLORS["comb"], coh0=hs.COLORS["coh0"], deph=hs.COLORS["deph"], hmm=hs.COLORS["hmm"], tree=hs.COLORS["tree"],
         tree_d=hs.COLORS["tree_d"], comb="#4d4d4d", comb_d=hs.COLORS["comb_d"])
cs = st = None                                                                # comb2 modules, imported in main()


def load(data, tag):
    m, r = cs.OUT / f"strokes_{data}_{tag}.metrics.json", cs.OUT / f"strokes_{data}_{tag}.json"
    return (json.load(open(m)), json.load(open(r))) if m.exists() and r.exists() else None


def seeds(data, tag):
    return [x for x in (load(data, f"{tag}_s{s}") for s in range(10)) if x]


def tile(imgs, rows, cols, pad=2):
    """rows x cols images (28 x 28, values in [0, 1]) on a black canvas, MNIST style."""
    H = rows * 28 + (rows + 1) * pad
    W = cols * 28 + (cols + 1) * pad
    canvas = np.zeros((H, W))
    for i in range(rows):
        for j in range(cols):
            if imgs[i][j] is not None:
                y, x = pad + i * (28 + pad), pad + j * (28 + pad)
                canvas[y:y + 28, x:x + 28] = imgs[i][j]
    return canvas


def render(tok):
    return st.render(tok, thickness=1, blur=0.8)



# ----------------------------------------------------------------------------------------------- statistical errors
def _errors(data):
    """jackknife values of every held-out objective (scripts/statistical_errors.py), or None"""
    f = cs.OUT / f"ERRORS_{data}.json"
    return json.load(open(f)) if f.exists() else None


def _se(v):
    v = np.asarray(v, float)
    return float(np.sqrt((len(v) - 1) / len(v) * ((v - v.mean()) ** 2).sum()))


def _jk(E, data, tags, part="total"):
    """jackknife values of the mean over the given runs (one value per left-out group), or None"""
    if E is None:
        return None
    rows = [E["runs"].get(f"strokes_{data}_{t}", {}).get(part) for t in tags]
    return np.mean(rows, axis=0) if rows and all(r is not None for r in rows) else None


def _auc_se(A, n1, n2):
    """standard error of an AUC (Hanley and McNeil, 1982)"""
    q1, q2 = A / (2 - A), 2 * A * A / (1 + A)
    return float(np.sqrt(max(A * (1 - A) + (n1 - 1) * (q1 - A * A) + (n2 - 1) * (q2 - A * A), 0) / (n1 * n2)))


# ----------------------------------------------------------------------------------------------- samples
def fig_samples(data, out, n=8):
    """Three separate plots without titles, one row per digit: the test digits, the same digits as pen strokes,
    and CoMB samples (one measurement-tree CoMB per digit). The first two need only the data."""
    z, _ = cs.load_prepared(data)
    idx = {k: np.nonzero(z["labels_test"] == k)[0][:n] for k in range(10)}
    blocks = []
    try:
        D = cs.load_images(data)
        X = D["x_test"]
        blocks.append(("test", [[X[i] / 255.0 for i in idx[k]] for k in range(10)]))
    except SystemExit:
        print(f"  [{data}] samples: original images not found; test-digit plot skipped")
    blocks.append(("strokes", [[render(z["tokens_test"][i]) for i in idx[k]] for k in range(10)]))
    rng = np.random.default_rng(3)
    gen, have = [], 0
    for k in range(10):
        name = f"tree_chi4_L4_path2_digit{k}_s0"
        if (cs.OUT / f"strokes_{data}_{name}.npz").exists():
            model, _ = cs.load_run(f"strokes_{data}_{name}")
            gen.append([render(t) for t in model.sample_counts(n, rng)])
            have += 1
        else:
            gen.append([None] * n)
    if have:
        blocks.append(("comb", gen))
    else:
        print(f"  [{data}] samples: no per-digit models yet (stage conditional); CoMB plot skipped")
    w = (WIDE - 0.25) / 3
    h = w * (10 * 30 + 2) / (n * 30 + 2) * 0.93
    for tag, b in blocks:
        fig, ax = plt.subplots(figsize=(w, h))
        ax.imshow(tile(b, 10, n), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([2 + 14 + i * 30 for i in range(10)])
        ax.set_yticklabels([str(k) for k in range(10)])
        ax.tick_params(length=0, which="both")
        ax.minorticks_off()
        for sp in ax.spines.values():
            sp.set_visible(False)
        hs.save(fig, out / f"{data}_samples_{tag}.pdf")
    print(f"  [{data}] samples: " + ", ".join(f"{data}_samples_{t}.pdf" for t, _ in blocks) + (f" ({have} of 10 digit models)" if have else ""))


# ----------------------------------------------------------------------------------------------- statistics
def _mutual_information(T, lags, K=16):
    """mean over positions t of I(c_t; c_{t+l}) in bits, plug-in estimate, for every lag l"""
    T = np.asarray(T, dtype=np.int64)
    out = []
    for l in lags:
        a, b = T[:, :-l], T[:, l:]
        mi = []
        for t in range(a.shape[1]):
            J = np.bincount(a[:, t] * K + b[:, t], minlength=K * K).reshape(K, K) / len(T)
            pa, pb = J.sum(1, keepdims=True), J.sum(0, keepdims=True)
            m = J > 0
            mi.append((J[m] * np.log2(J[m] / (pa @ pb)[m])).sum())
        out.append(np.mean(mi))
    return np.array(out)


def fig_statistics(data, out, nmax=5000):
    """Four plots: dependence between pen moves against their distance (with the legend), and the distributions
    of pen-down moves, height and total turning; test digits as points, the chi = 4 models as lines."""
    z, meta = cs.load_prepared(data)
    T_te = z["tokens_test"][:nmax].astype(np.int64)
    n = len(T_te)
    groups = [("hqmm_chi4_e4_path2_s0", "coherent memory", C["coh"], "-"), ("hqmm_dephased_chi4_e4_path2_s0", "same, dephased", C["deph"], "--"),
              ("hmm_chi4_path2_s0", "classical HMM", C["hmm"], ":"), ("tree_chi4_L4_path2_s0", "tree-CoMB, $L=4$", C["tree"], "-.")]
    rng = np.random.default_rng(5)
    models = []
    for tag, lab, col, ls in groups:
        if (cs.OUT / f"strokes_{data}_{tag}.npz").exists():
            m, _ = cs.load_run(f"strokes_{data}_{tag}")
            models.append((lab, m.sample_counts(n, rng).astype(np.int64), col, ls))
    if not models:
        print(f"  [{data}] statistics: no chi = 4 runs yet; plots skipped")
        return
    lags = np.arange(1, min(41, T_te.shape[1] - 1))
    nb = 25 if n >= 2000 else 14
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    mi_d = _mutual_information(T_te, lags)
    groups = np.array_split(np.random.default_rng(7).permutation(n), 50)
    mi_g = np.array([_mutual_information(T_te[np.setdiff1d(np.arange(n), g)], lags) for g in groups])
    ax.errorbar(lags, mi_d, yerr=[_se(mi_g[:, j]) for j in range(len(lags))], fmt="o", ms=3, color="k", elinewidth=0.8, capsize=0, label="test digits", zorder=5)
    for lab, T, col, ls in models:
        ax.plot(lags, _mutual_information(T, lags), ls, color=col, lw=1.4, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("distance between moves")
    ax.set_ylabel("mutual information [bits]")
    lo_, hi_ = ax.get_ylim()
    ticks = [t for t in (0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1, 2) if lo_ <= t <= hi_]
    ax.set_yticks(ticks)
    ax.set_yticklabels([f"{t:g}" for t in ticks])
    ax.yaxis.set_minor_formatter(plt.NullFormatter())
    ax.legend(loc="upper right", fontsize=8)
    hs.save(fig, out / f"{data}_statistics_mi.pdf")
    G_te = st.geometry_many(T_te)
    Gs = [(lab, st.geometry_many(T), col, ls) for lab, T, col, ls in models]
    for ci, xl, tag in ((0, "pen-down moves", "penmoves"), (3, "height [pixels]", "height"), (5, "total turning [rad]", "turning")):
        fig, ax = plt.subplots(figsize=(COL, 2.5))
        xd = G_te[:, ci]
        lo, hi = np.percentile(np.concatenate([xd] + [g[:, ci] for _, g, _, _ in Gs]), [0.5, 99.5])
        bins = np.linspace(lo, hi, nb + 1)
        c = 0.5 * (bins[1:] + bins[:-1])
        w = np.diff(bins)
        nd, _ = np.histogram(xd, bins)
        ax.errorbar(c, nd / (n * w), yerr=np.sqrt(nd) / (n * w), fmt="o", ms=3, color="k", elinewidth=0.8, zorder=5)
        for lab, g, colr, ls in Gs:
            nm, _ = np.histogram(g[:, ci], bins)
            ax.stairs(nm / (len(g) * w), bins, color=colr, ls=ls, lw=1.3)
        ax.set_xlabel(xl)
        ax.set_ylabel("density")
        ax.set_ylim(0, None)
        hs.save(fig, out / f"{data}_statistics_{tag}.pdf")
    print(f"  [{data}] statistics: 4 plots, {len(models)} models, {n} digits each")


# ----------------------------------------------------------------------------------------------- pixel correlations
def _pixels(T):
    """render pen-stroke sequences and average to 14 x 14 pixels, flattened in scan order"""
    R = st.render_many(np.asarray(T, dtype=np.int64)).reshape(len(T), 14, 2, 14, 2).mean(axis=(2, 4))
    return R.reshape(len(T), 196)


def fig_pixel_correlation(data, out, nmax=2000):
    """Pixel-pixel correlation matrices (one plot per source): test digits rendered from their pen strokes,
    the chi = 4 models, and the class-conditional CoMB (the mixture of one tree circuit per digit)."""
    z, _ = cs.load_prepared(data)
    T_te = z["tokens_test"][:nmax].astype(np.int64)
    n = len(T_te)
    X_te = _pixels(T_te)
    keep = X_te.std(0) > 1e-3                         # pixels that vary across test digits
    rng = np.random.default_rng(11)
    sources = [("test", X_te)]
    for tag, key in (("hqmm_chi4_e4_path2_s0", "coherent"), ("hqmm_dephased_chi4_e4_path2_s0", "dephased"),
                     ("hmm_chi4_path2_s0", "hmm"), ("tree_chi4_L4_path2_s0", "tree")):
        if (cs.OUT / f"strokes_{data}_{tag}.npz").exists():
            m, _ = cs.load_run(f"strokes_{data}_{tag}")
            sources.append((key, _pixels(m.sample_counts(n, rng))))
    freq = np.bincount(z["labels_test"], minlength=10) / len(z["labels_test"])
    parts = []
    for k in range(10):
        name = f"strokes_{data}_tree_chi4_L4_path2_digit{k}_s0"
        if (cs.OUT / f"{name}.npz").exists() and round(freq[k] * n) > 0:
            m, _ = cs.load_run(name)
            parts.append(m.sample_counts(int(round(freq[k] * n)), rng))
    if len(parts) == 10:
        sources.append(("conditional", _pixels(np.concatenate(parts))))
    for tag, X in sources:
        C = np.nan_to_num(np.corrcoef(X[:, keep].T))
        fig, ax = plt.subplots(figsize=(hs.THIRD, hs.THIRD * 0.9))
        im = ax.imshow(C, vmin=-1, vmax=1, cmap="RdBu_r", interpolation="nearest")
        ax.set_xlabel("pixel (scan order)", fontsize=8)
        ax.set_ylabel("pixel (scan order)", fontsize=8)
        ax.minorticks_off()
        ax.tick_params(labelsize=7)
        ax.set_xticks([0, 50, 100, 150][: 1 + keep.sum() // 50])
        ax.set_yticks(ax.get_xticks())
        cb = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.03)
        cb.ax.tick_params(labelsize=7)
        cb.set_label("correlation", fontsize=8, loc="center")
        fig.get_layout_engine().set(w_pad=0.08)
        hs.save(fig, out / f"{data}_pixcorr_{tag}.pdf")
    print(f"  [{data}] pixel correlations: {len(sources)} plots ({', '.join(t for t, _ in sources)}), {int(keep.sum())} pixels")


# ----------------------------------------------------------------------------------------------- recognition
def fig_recognition(data, out):
    f = cs.OUT / f"CLASSES_{data}_tree_chi4_L4_path2_s0.json"
    if not f.exists():
        print(f"  [{data}] recognition: no quality-gate results yet; figure skipped")
        return
    j = json.load(open(f))
    ks = sorted(int(k) for k in j["digits"])
    r = {int(k): v for k, v in j["digits"].items()}
    gen = np.array([r[k]["intended"] for k in ks])
    real = np.array([r[k]["ceiling"] for k in ks])
    tgt = j.get("target", 0.8)
    fig, ax = plt.subplots(figsize=(COL, 2.4))
    x = np.arange(len(ks))
    w = 0.38
    n_g = np.array([r[k].get("n", 500) for k in ks])
    n_r = np.array([max(r[k].get("n_real", 30), 1) for k in ks])
    ax.bar(x - w / 2, 100 * real, w, color="0.85", edgecolor="0.45", lw=0.6, label="real digits",
           yerr=100 * np.sqrt(real * (1 - real) / n_r), error_kw=dict(elinewidth=0.8, capsize=1.5, ecolor="0.3"))
    ax.bar(x + w / 2, 100 * gen, w, color=C["tree"], label="CoMB samples",
           yerr=100 * np.sqrt(gen * (1 - gen) / n_g), error_kw=dict(elinewidth=0.8, capsize=1.5, ecolor="k"))
    ax.scatter(x + w / 2, 100 * tgt * real, marker="_", s=70, color="k", lw=1.2, zorder=3, label=f"criterion ({tgt:g}$\\times$ real)")
    ax.set_xticks(x)
    ax.set_xticklabels([str(k) for k in ks])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlabel("digit")
    ax.set_ylabel("read as intended [%]")
    ax.set_ylim(0, 104)
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=3, fontsize=7.5, handlelength=1.3, columnspacing=1.0)
    hs.save(fig, out / f"{data}_recognition.pdf")
    print(f"  [{data}] recognition: {sum(v['passed'] for v in r.values())} of {len(ks)} digits pass")


# ----------------------------------------------------------------------------------------------- comparison
def _parts(m):
    """held-out objective of one run: (total, local k-point part, global path part)"""
    k = m["kmmd"]["model"]["loss"]
    sp = m["spec"]
    p = sp.get("path", 0) * m["path_mmd"]["model"] / sp["path_reference"] if sp.get("path", 0) > 0 and m.get("path_mmd") else 0.0
    return k + p, k, p


def _floor(m):
    sp = m["spec"]
    k = m["kmmd"]["floor"]["loss"]
    p = sp.get("path", 0) * m["path_mmd"]["floor"] / sp["path_reference"] if sp.get("path", 0) > 0 and m.get("path_mmd") else 0.0
    return k + p, k, p


ROWS = [("hqmm_chi4_e4_path2", "coherent memory", "coh"), ("hqmm_chi4_e0_path2", "coherent memory, no env.", "coh0"),
        ("hqmm_dephased_chi4_e4_path2", "same memory, dephased", "deph"), ("hmm_chi4_path2", "classical HMM", "hmm"), None,
        ("comb_chi4_path2", "CoMB circuit, 4 layers", "comb"), ("dephased_chi4_path2", "dephased twin", "comb_d"),
        ("tree_chi4_L4_path2", "tree-CoMB, $L=4$", "tree"), ("tree_dephased_chi4_L4_path2", "dephased twin", "tree_d"),
        ("tree_chi4_L6_path2", "tree-CoMB, $L=6$", "tree"), ("tree_dephased_chi4_L6_path2", "dephased twin", "tree_d")]


# ----------------------------------------------------------------------------------------------- memory
def _xequiv(chis, curve, target):
    for i in range(1, len(chis)):
        if (curve[i - 1] - target) * (curve[i] - target) <= 0 and curve[i] != curve[i - 1]:
            return float(np.exp(np.log(chis[i - 1]) + (target - curve[i - 1]) / (curve[i] - curve[i - 1]) * (np.log(chis[i]) - np.log(chis[i - 1]))))
    return None


def fig_memory_compression(data, out):
    """Held-out objective relative to the 4-level coherent memory, against memory size: classical HMMs of growing size
    and the dephased twin at chi = 4. All models are scored on the same test digits, so the error bars are paired
    statistical errors of each difference (jackknife over the test digits); points are not joined. The classical
    memory size where the HMMs cross zero is the equivalent classical memory."""
    E = _errors(data)
    chis, vals, tags = [], [], []
    for c in (4, 6, 8, 12, 16):
        runs = seeds(data, f"hmm_chi{c}_path2")
        if runs:
            chis.append(c)
            vals.append([_parts(m)[0] for m, _ in runs])
            tags.append([f"hmm_chi{c}_path2_s{r['spec']['seed']}" for _, r in runs])
    coh = seeds(data, "hqmm_chi4_e4_path2")
    deph = seeds(data, "hqmm_dephased_chi4_e4_path2")
    if len(chis) < 2 or not coh:
        print(f"  [{data}] memory compression: needs classical HMMs at two or more sizes and the coherent memory; plot skipped")
        return
    tag_c = [f"hqmm_chi4_e4_path2_s{r['spec']['seed']}" for _, r in coh]
    cm = np.mean([_parts(m)[0] for m, _ in coh])
    jc = _jk(E, data, tag_c)
    def diff(vs, tg):
        jk = _jk(E, data, tg)
        return np.mean(vs) - cm, (_se(jk - jc) if jk is not None and jc is not None else 0.0)
    hm = [diff(v, t) for v, t in zip(vals, tags)]
    fig, ax = plt.subplots(figsize=(COL, 2.75))
    ax.axhline(0, color=C["coh"], lw=1.1, label="coherent memory, $\\chi=4$ (reference)")
    for c, v in zip(chis, vals):
        if len(v) > 1:
            ax.scatter([c] * len(v), np.array(v) - cm, s=10, color=C["hmm"], alpha=0.35, lw=0)
    ax.errorbar(chis, [h[0] for h in hm], yerr=[h[1] for h in hm], fmt="o", color=C["hmm"], ms=4.5, capsize=2, elinewidth=1.0, label="classical HMM")
    if deph:
        tag_d = [f"hqmm_dephased_chi4_e4_path2_s{r['spec']['seed']}" for _, r in deph]
        dm, de = diff([_parts(m)[0] for m, _ in deph], tag_d)
        ax.errorbar([4 * 1.08], [dm], yerr=[de], fmt="D", color=C["deph"], ms=4.5, mfc="white", mew=1.2, capsize=2, label="coherent memory, dephased")
    eq = _xequiv(chis, np.array([h[0] for h in hm]), 0.0)
    if eq:
        ax.annotate(f"$\\chi_{{\\rm equiv}}\\approx{eq:.1f}$", xy=(eq, 0), xytext=(6, -14), textcoords="offset points", fontsize=8.5, color=C["coh"])
    ax.set_xscale("log", base=2)
    ax.set_xticks(chis)
    ax.set_xticklabels([str(c) for c in chis])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.set_xlim(3.3, chis[-1] * 1.15)
    ax.set_xlabel("memory size $\\chi$ [states]")
    ax.set_ylabel("held-out objective $-$ coherent memory")
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo, hi + 0.35 * (hi - lo))
    ax.legend(loc="upper right", fontsize=7.5)
    hs.save(fig, out / f"{data}_memory_compression.pdf")
    print(f"  [{data}] memory compression: chi_equiv = {eq:.2f}" if eq else f"  [{data}] memory compression: outside the measured range")

CIRCUITS = [("comb_chi4_path2", "dephased_chi4_path2", "CoMB circuit\n4 work qubits"),
            ("tree_chi4_L4_path2", "tree_dephased_chi4_L4_path2", "tree-CoMB\n$L=4$"),
            ("tree_chi4_L6_path2", "tree_dephased_chi4_L6_path2", "tree-CoMB\n$L=6$")]


def fig_circuits(data, out):
    """Two plots, full held-out objective and its local part, relative to the general coherent memory (any circuit depth):
    every circuit (filled) next to its dephased twin (open), with paired statistical errors (jackknife over the test
    digits), and the classical HMM of the same size as a reference line."""
    E = _errors(data)
    rows = [(seeds(data, c), seeds(data, t), lab, c, t) for c, t, lab in CIRCUITS]
    rows = [r for r in rows if r[0] or r[1]]
    ideal = seeds(data, "hqmm_chi4_e4_path2")
    if not rows or not ideal:
        print(f"  [{data}] circuits: needs circuit runs and the coherent memory; plots skipped")
        return
    tag_i = [f"hqmm_chi4_e4_path2_s{r['spec']['seed']}" for _, r in ideal]
    hmm = seeds(data, "hmm_chi4_path2")
    for j, (tag, part, yl) in enumerate((("total", "total", "held-out objective $-$ coherent memory"), ("local", "local", "local part $-$ coherent memory"))):
        ref = np.mean([_parts(m)[j] for m, _ in ideal])
        jref = _jk(E, data, tag_i, part)
        fig, ax = plt.subplots(figsize=(COL, 2.6))
        x = np.arange(len(rows))
        ax.axhline(0, color=C["coh"], lw=1.1, label="coherent memory, any depth")
        if hmm:
            ax.axhline(np.mean([_parts(m)[j] for m, _ in hmm]) - ref, color=C["hmm"], ls=":", lw=1.2, label="classical HMM, $\\chi=4$")
        for i, (coh, twin, lab, ct, tt) in enumerate(rows):
            col = C["tree"] if "tree" in lab else C["comb"]
            for runs, base, off, mfc, lg in ((coh, ct, -0.13, col, "coherent"), (twin, tt, 0.13, "white", "dephased twin")):
                if not runs:
                    continue
                v = np.mean([_parts(m)[j] for m, _ in runs]) - ref
                jk = _jk(E, data, [f"{base}_s{r['spec']['seed']}" for _, r in runs], part)
                ax.errorbar([i + off], [v], yerr=[_se(jk - jref)] if jk is not None and jref is not None else None, fmt="o", color=col, ms=6,
                            mfc=mfc, mew=1.3, capsize=2, elinewidth=1.0, zorder=3, label=lg if i == len(rows) - 1 else None)
        ax.set_xticks(x)
        ax.set_xticklabels([r[2] for r in rows])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(-0.6, len(rows) - 0.4)
        ax.set_ylabel(yl)
        ax.legend(loc="center right", fontsize=7.5)
        hs.save(fig, out / f"{data}_circuits_{tag}.pdf")
    print(f"  [{data}] circuits: 2 plots, {len(rows)} circuits (relative to the coherent memory, paired errors)")

def fig_realism(data, out):
    """How well a classifier tells generated digits from real ones (two-sample AUC; 0.5 = indistinguishable), with
    statistical errors (Hanley-McNeil): the unconditional models, and the class-conditional CoMB."""
    rows = [("hqmm_chi4_e4_path2", "coherent memory", C["coh"]), ("hqmm_dephased_chi4_e4_path2", "same, dephased", C["deph"]),
            ("hmm_chi4_path2", "classical HMM", C["hmm"]), ("tree_chi4_L4_path2", "tree-CoMB, $L=4$", C["tree"]),
            ("comb_chi4_path2", "CoMB circuit", C["comb"])]
    z, _ = cs.load_prepared(data)
    n_te = len(z["tokens_test"])
    have = [(lab, col, [(m["auc"], min(m.get("n", n_te), n_te)) for m, _ in seeds(data, tag)]) for tag, lab, col in rows]
    have = [h for h in have if h[2]]
    f = cs.OUT / f"CLASSES_{data}_tree_chi4_L4_path2_s0.json"
    mix = json.load(open(f)).get("mixture") if f.exists() else None
    if not have and not mix:
        print(f"  [{data}] realism: no runs yet; plot skipped")
        return
    floor = next((m["auc_floor"] for tag, *_ in rows for m, _ in seeds(data, tag)), mix["auc_floor"] if mix else None)
    fig, ax = plt.subplots(figsize=(COL, 0.27 * (len(have) + (2 if mix else 0)) + 0.85))
    labels, ypos = [], []
    y = 0
    if mix:
        ax.errorbar([mix["auc"]], [y], xerr=[_auc_se(mix["auc"], n_te, n_te)], fmt="*", color=C["tree"], ms=11, capsize=2, zorder=4)
        labels.append("CoMB, one tree circuit per digit")
        ypos.append(y)
        y -= 1
        ax.axhline(y + 0.5, color="0.75", lw=0.6)
    for lab, col, v in have:
        y -= 1 if ypos else 0
        a = np.mean([u for u, _ in v])
        se = np.sqrt(np.mean([_auc_se(u, nn, nn) ** 2 for u, nn in v]) / len(v))
        ax.errorbar([a], [y], xerr=[se], fmt="o", color=col, ms=5, capsize=2, zorder=4)
        labels.append(lab + " (unconditional)" if lab == "coherent memory" else lab)
        ypos.append(y)
    ax.axvline(0.5, color="0.35", lw=0.9)
    if floor is not None:
        ax.axvline(floor, color="0.35", ls=":", lw=0.9)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=7.5)
    ax.yaxis.set_minor_locator(plt.NullLocator())
    ax.tick_params(axis="y", length=0)
    ax.set_ylim(min(ypos) - 0.7, 0.8)
    ax.set_xlim(0.40, 1.0)
    ax.set_xlabel("two-sample AUC (0.5: indistinguishable)")
    if floor is not None:
        ax.text(max(0.5, floor) + 0.005, min(ypos) - 0.55, "real vs real", fontsize=7, color="0.35", va="bottom", ha="left")
    hs.save(fig, out / f"{data}_realism.pdf")
    print(f"  [{data}] realism: {len(have)} unconditional models" + (f", class-conditional AUC {mix['auc']:.3f}" if mix else ""))

# ----------------------------------------------------------------------------------------------- noise
PAIRS = [("comb_chi4_path2_s0", "dephased_chi4_path2_s0", "CoMB circuit", "comb", "o"),
         ("tree_chi4_L4_path2_s0", "tree_dephased_chi4_L4_path2_s0", "tree-CoMB, $L=4$", "tree", "s"),
         ("tree_chi4_L6_path2_s0", "tree_dephased_chi4_L6_path2_s0", "tree-CoMB, $L=6$", "tree", "^")]


def fig_noise(data, out):
    """Two plots: the held-out objective of every circuit and its twin against the noise, and the difference between
    each coherent circuit and its twin (below zero, coherence helps); points with statistical errors (jackknife over
    the test digits, paired for the differences), no connecting lines."""
    f = cs.OUT / f"DEPLOY_{data}.json"
    if not f.exists():
        print(f"  [{data}] noise: no deployment results yet; plots skipped")
        return
    j = json.load(open(f))
    scales = j["noise_models"]["scales"]
    R = {k.replace(f"strokes_{data}_", ""): v["results"] for k, v in j["runs"].items()}
    E = _errors(data)
    JD = {k.replace(f"strokes_{data}_", ""): v for k, v in (E or {}).get("deploy", {}).items()}
    if not any(c in R for c, *_ in PAIRS):
        print(f"  [{data}] noise: no circuit runs in the deployment results; plots skipped")
        return
    keys = ["noiseless"] + [f"SC x{s_:g}" for s_ in scales[1:]]
    xs = np.arange(len(scales))
    xion = -1.3
    ionkey = lambda res: next(k for k in res if k.startswith("trapped"))
    def vals(tag, k):
        return R[tag][k]["total"], (np.array(JD[tag][k]) if tag in JD and k in JD[tag] else None)
    fig1, a1 = plt.subplots(figsize=(COL, 2.6))
    fig2, a2 = plt.subplots(figsize=(COL, 2.6))
    for n_, (coh, twin, lab, ck, mk) in enumerate(PAIRS):
        col = C["tree"] if ck == "tree" else C["comb"]
        off = 0.16 * (n_ - 1)
        for tag, mfc, dx, lg in ((coh, col, -0.04, lab), (twin, "white", 0.04, "dephased twin" if n_ == 1 else None)):
            if tag not in R:
                continue
            allk = [ionkey(R[tag])] + keys
            px = [xion] + list(xs)
            v = [vals(tag, k) for k in allk]
            a1.errorbar(np.array(px) + off + dx, [u for u, _ in v], yerr=[_se(jk) if jk is not None else 0 for _, jk in v],
                        fmt=mk, color=col, ms=4, mfc=mfc, mew=1.1, capsize=1.5, elinewidth=0.8, label=lg)
        if coh in R and twin in R:
            allk = [ionkey(R[coh])] + keys
            px = [xion] + list(xs)
            d = [vals(coh, k)[0] - vals(twin, k)[0] for k in allk]
            e = [(_se(vals(coh, k)[1] - vals(twin, k)[1]) if vals(coh, k)[1] is not None and vals(twin, k)[1] is not None else 0) for k in allk]
            t = j.get("thresholds", {}).get(f"strokes_{data}_{coh}", {})
            ls_ = t.get("lambda_star")
            a2.errorbar(np.array(px) + off, d, yerr=e, fmt=mk, color=col, ms=4, capsize=1.5, elinewidth=0.8,
                        label=lab + (f" ($\\lambda^\\star={ls_:.2f}$)" if ls_ is not None else ""))
    for ax in (a1, a2):
        ax.axvline(-0.65, color="0.7", lw=0.6)
        ax.set_xticks([xion] + list(xs))
        ax.set_xticklabels(["ion"] + ["0"] + [f"{s_:g}" for s_ in scales[1:]])
        ax.xaxis.set_minor_locator(plt.NullLocator())
        ax.set_xlim(xion - 0.5, xs[-1] + 0.45)
        ax.set_xlabel("noise: ion model, or Heron-like $\\times\\,\\lambda$")
    a1.set_yscale("log")
    a1.set_ylabel("held-out objective")
    a1.legend(loc="upper left", fontsize=7.5)
    a2.axhline(0, color="k", lw=0.7)
    a2.set_ylabel("coherent $-$ dephased twin")
    lim = max(abs(np.array(a2.get_ylim())))
    a2.set_ylim(-lim, lim)
    a2.legend(loc="upper right", fontsize=7.5)
    hs.save(fig1, out / f"{data}_noise_objective.pdf")
    hs.save(fig2, out / f"{data}_noise_difference.pdf")
    print(f"  [{data}] noise: 2 plots, {sum(c in R for c, *_ in PAIRS)} circuits" + (" (with statistical errors)" if JD else " (no ERRORS file: run scripts/statistical_errors.py for error bars)"))

# ----------------------------------------------------------------------------------------------- training
def fig_training(data, out):
    groups = [("hqmm_chi4_e4_path2", "coherent memory", C["coh"]), ("hqmm_dephased_chi4_e4_path2", "same, dephased", C["deph"]),
              ("hmm_chi4_path2", "classical HMM", C["hmm"]), ("tree_chi4_L4_path2", "tree-CoMB, $L=4$", C["tree"])]
    fig, ax = plt.subplots(figsize=(COL, 2.5))
    n = 0
    finals = []
    for tag, lab, col in groups:
        for i, (_, r) in enumerate(seeds(data, tag)):
            h = r["hist"]
            ax.plot(h["step"][1:], h["val"][1:], color=col, lw=1.0, alpha=0.9, label=lab if i == 0 else None)
            n += 1
            finals.append(min(h["val"][1:]))
    if n == 0:
        plt.close(fig)
        print(f"  [{data}] training: no runs yet; figure skipped")
        return
    lo = min(finals)
    ax.set_ylim(0.95 * lo, 2.3 * lo)
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss")
    fig.legend(*ax.get_legend_handles_labels(), loc="outside upper center", ncol=2, fontsize=7.5, handlelength=1.6, columnspacing=1.0)
    hs.save(fig, out / f"{data}_training.pdf")
    print(f"  [{data}] training: {n} runs")


def main():
    global cs, st
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--comb2", default=None, help="folder with the digits runs (default: the comb folder, or comb2 next to it)")
    ap.add_argument("--data", default="mnist", choices=["mnist", "uci"])
    ap.add_argument("--out", default=str(COMB / "outputs" / "paper_figures"))
    ap.add_argument("--style", default="ROOT")
    a = ap.parse_args()
    root = digits_folder(a.comb2)
    if not (root / "scripts" / "comb_strokes.py").exists():
        print(f"[{a.data}] no digits code in {root} (comb_strokes.py); give the folder with the digits runs with --comb2")
        return
    sys.path.insert(0, str(root))
    sys.path.insert(0, str(root / "scripts"))
    print(f"[{a.data}] digits runs from {root}")
    import comb_strokes as _cs
    from comb.digits import strokes as _st
    cs, st = _cs, _st
    if not (cs.DATA / f"strokes_{a.data}.npz").exists():
        print(f"[{a.data}] not prepared in {root}; run  python scripts/comb_strokes.py prepare --data {a.data}")
        return
    print(f"style: {'mplhep ' + a.style if hs.use(a.style) else 'matplotlib fallback (pip install mplhep for the HEP style)'}")
    out = pathlib.Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in (fig_samples, fig_statistics, fig_pixel_correlation, fig_memory_compression, fig_circuits, fig_realism, fig_recognition, fig_noise, fig_training):
        f(a.data, out)


if __name__ == "__main__":
    main()
