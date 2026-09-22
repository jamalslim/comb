"""
Shared end-to-end pipeline for CoMB (used by run_d12.py): the data
loading, splitting and baseline fitting the scripts share.

Phases:
    prep     — load data, tokenize, fit classical baselines
    quantum  — train one or more SequentialBornModel configs
    report   — assemble the honest final tables (token NLL + pixel
               metrics with floors and NN memorization checks)

Every per-model artifact is written to outputs/models/<name>.npz so the
report phase can be re-run at any time without retraining. Honesty
rules (enforced here, inherited from the v3 audit): no copula or any
post-hoc correlation-installing stage; the train-vs-test floor and the
tokenizer quantization floor accompany every pixel table; every sample
set carries a nearest-neighbor memorization check.
"""

import os
import re
import sys
import time
import pathlib

import numpy as np

# Shared loading and splitting. Everything downstream depends on this
# being deterministic: the hardware run is tied to one specific theta,
# and that theta is only reproducible if the split, the tokenizer fit and
# the RNG seeds are identical every time.
from sklearn.model_selection import train_test_split

PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from comb import (CoMBConfig, CoMBQSpec, SequentialBornModel,
                   BlockTokenizer, IndependentBlocks, EmpiricalJoint,
                   InhomogeneousHMM, train_blockscore, block_grams,
                   corr_nan_safe, pixel_report, nn_dist)
from comb.mmd import score_of_conditionals

OUT = PROJECT_ROOT / "outputs"
MODELS = OUT / "models"


def build_blocks(d, b):
    out, s = [], 0
    while s < d:
        out.append((s, min(b, d - s)))
        s += out[-1][1]
    return out


def load_all(cfg: CoMBConfig):
    X = np.load(PROJECT_ROOT / "data" / cfg.data.data_path
                ).astype(np.float64)
    idx = np.random.default_rng(cfg.data.subset_seed).choice(
        len(X), size=min(cfg.data.subset_n, len(X)), replace=False)
    X = X[idx]
    tr, te = train_test_split(np.arange(len(X)),
                              test_size=cfg.data.test_size,
                              random_state=cfg.data.split_seed)
    Y_tr, Y_te = X[tr], X[te]
    blocks = build_blocks(Y_tr.shape[1], cfg.tokenizer.block_size)
    tok = BlockTokenizer(blocks, K=2 ** cfg.tokenizer.n_work,
                         seed=cfg.data.seed,
                         detok_mode=cfg.tokenizer.detok_mode,
                         shrink=cfg.tokenizer.shrink,
                         clip_nonneg=cfg.tokenizer.clip_nonnegative
                         ).fit(Y_tr)
    T_tr, T_te = tok.tokenize(Y_tr), tok.tokenize(Y_te)
    vi = np.random.default_rng(cfg.data.seed + 1).choice(
        len(T_tr), size=cfg.data.val_n, replace=False)
    vmask = np.zeros(len(T_tr), dtype=bool)
    vmask[vi] = True
    return (Y_tr, Y_te, blocks, tok, T_tr, T_te,
            T_tr[~vmask], T_tr[vmask])


def _save(name, score, params, tokens, extra=None):
    MODELS.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(MODELS / f"{name}.npz", score=score,
                        params=params, tokens=tokens,
                        extra=np.array([extra or {}], dtype=object))
    print(f"  [SAVE] {name}: test score={score:.4f}  params={params}")


def phase_prep(cfg: CoMBConfig):
    (Y_tr, Y_te, blocks, tok, T_tr, T_te, T_fit, T_val) = load_all(cfg)
    B, K = len(blocks), 2 ** cfg.tokenizer.n_work
    rng = np.random.default_rng(cfg.data.seed)
    print(f"[DATA] n_train={len(Y_tr)} n_test={len(Y_te)} "
          f"d={Y_tr.shape[1]} B={B} K={K}")
    Kb = block_grams(tok)
    ind = IndependentBlocks(K).fit(T_fit)
    _save("Independent",
          score_of_conditionals(ind.conditionals(T_te), T_te, Kb),
          B * (K - 1), ind.sample(len(Y_te), rng))
    for chi in cfg.baselines.hmm_chis:
        t0 = time.time()
        hmm = InhomogeneousHMM(chi, K, B, seed=cfg.data.seed)
        hmm.fit(T_fit, T_val, n_restarts=cfg.baselines.hmm_restarts,
                n_iter=cfg.baselines.hmm_iters)
        _save(f"HMM(chi={chi})",
              score_of_conditionals(hmm.conditionals(T_te), T_te, Kb),
              hmm.n_params(), hmm.sample(len(Y_te), rng),
              extra=dict(seconds=time.time() - t0))
    ej = EmpiricalJoint(K, B, alpha=cfg.baselines.empirical_alpha
                        ).fit(T_fit)
    _save("EmpiricalJoint",
          score_of_conditionals(ej.conditionals(T_te), T_te, Kb),
          K ** B - 1, ej.sample(len(Y_te), rng))


def phase_quantum(cfg: CoMBConfig, names):
    """Config names: M{n_mem}C{0|1}[-L{depth}][-S][-E{epochs}][-P{patience}][-W...]

    -P sets early stopping: training stops once the validation score has
    not improved for that many epochs. The paper's main model is
    M3C0-L6-E450-P60; the controls use no early stopping.
    """
    (Y_tr, Y_te, blocks, tok, T_tr, T_te, T_fit, T_val) = load_all(cfg)
    B = len(blocks)
    rng = np.random.default_rng(cfg.data.seed)
    for name in names:
        mm = re.fullmatch(
            r"M(\d+)C([01])(?:-L(\d+))?(-S)?(?:-E(\d+))?(?:-P(\d+))?(?:-W\d*)?",
            name)
        if not mm:
            raise ValueError(f"bad config name {name}")
        m = SequentialBornModel(CoMBQSpec(
            n_mem=int(mm.group(1)), n_work=cfg.tokenizer.n_work,
            depth=int(mm.group(3) or 2), n_blocks=B,
            use_prefix=mm.group(2) == "1",
            share_theta=mm.group(4) is not None, seed=cfg.data.seed))
        epochs = int(mm.group(5) or cfg.train.epochs)
        patience = int(mm.group(6)) if mm.group(6) else None
        print(f"\n[CoMB {name}] p_theta={m.p_theta} epochs={epochs}")
        warm = cfg.train.warm_start or os.environ.get("CoMBQ_WARM", "")
        if warm:
            m.theta = np.load(warm)
            assert m.theta.size == m.p_theta
            print(f"  warm-started from {warm}")
        Kb = block_grams(tok)
        hist = train_blockscore(m, Kb, T_fit, T_val, epochs=epochs,
                                batch=cfg.train.batch, lr=cfg.train.lr,
                                warmup=cfg.train.lr_warmup,
                                clip=cfg.train.grad_clip,
                                seed=cfg.train.seed, patience=patience)
        _save(f"CoMB {name}", m.blockwise_score(T_te, Kb), m.p_theta,
              m.sample(len(Y_te), rng),
              extra=dict(best_val=hist["best_val_score"],
                         val_curve=hist["val_score"]))
        np.save(OUT / f"theta_{name}.npy", m.theta)


def phase_report(cfg: CoMBConfig):
    (Y_tr, Y_te, blocks, tok, T_tr, T_te, T_fit, T_val) = load_all(cfg)
    C_te = corr_nan_safe(Y_te)
    nn_ref = float(np.median(nn_dist(Y_te, Y_tr)))
    entries = {}
    for f in sorted(MODELS.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        key = "score" if "score" in z.files else "nll"
        entries[f.stem] = dict(score=float(z[key]),
                               params=int(z["params"]),
                               tokens=z["tokens"])
    print("\n[PIXEL METRICS] shared Gaussian de-tokenizer; NO copula")
    rows = {}
    rows["floor: train set"] = pixel_report(
        "floor: train set", Y_te, Y_tr, C_te, blocks, Y_tr, nn_ref)
    rows["floor: oracle tokens"] = pixel_report(
        "floor: oracle tokens", Y_te,
        tok.detokenize(T_te, np.random.default_rng(0)),
        C_te, blocks, Y_tr, nn_ref)
    for name, e in entries.items():
        rows[name] = pixel_report(
            name, Y_te,
            tok.detokenize(e["tokens"], np.random.default_rng(1)),
            C_te, blocks, Y_tr, nn_ref)
    # The paper's main table: the same five columns, in the same order.
    cols = ("offdiag", "cross", "W1", "W1(E)", "seam")
    def line(label, pr):
        return (f"{label:28s} {pr['corr_off']:8.3f} {pr['corr_cross']:8.3f} "
                f"{pr['w1']:8.4f} {pr['w1_E']:8.3f} {pr['seam']:+8.2f}")
    print("\n" + "=" * 76)
    print(f"{'model':28s} " + " ".join(f"{c:>8s}" for c in cols))
    print("-" * 76)
    for name in sorted(entries, key=lambda k: rows[k]["corr_cross"]):
        print(line(name, rows[name]))
    print("-" * 76)
    print(line("emission floor (oracle tok.)", rows["floor: oracle tokens"]))
    print(line("statistical floor (train/test)", rows["floor: train set"]))
    print("=" * 76)
    np.savez_compressed(OUT / f"comb_{cfg.run_tag}_final.npz",
                        entries=np.array([entries], dtype=object),
                        pixel_rows=np.array([rows], dtype=object))
    print(f"[SAVE] {OUT / f'comb_{cfg.run_tag}_final.npz'}")
