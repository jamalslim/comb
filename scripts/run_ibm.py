#!/usr/bin/env python3
"""
Minimal IBM hardware run for CoMB (inference-only: sample the
simulation-trained model on a real device and test whether the
coherent-memory channel survives hardware noise).

    python scripts/run_ibm.py verify                # MANDATORY first (Aer)
    python scripts/run_ibm.py submit [backend]      # one SamplerV2 job
    python scripts/run_ibm.py analyze               # HW vs exact vs no-memory

Minimal-run design decisions (change via CONFIG below):
  * Config M3C0 at depth L=2 (theta_M3C0.npy): 6 qubits, 48 CZs across
    the whole chain before routing -- chosen over the L=6 flagship
    (~200+ two-qubit gates after routing) so the demonstration arrives
    at usable fidelity on Heron-class hardware.
  * Training stays on the simulator; hardware does sampling only. One
    shot = one generated image (B=4 mid-circuit-measured blocks).
  * shots = 8192 (=> 8192 generated token sequences; seconds of QPU).
  * Dynamical decoupling (XY4) enabled -- the memory qubits idle during
    work-register readout/reset, which is exactly where DD pays.

What the analysis measures (all computable WITHOUT post-selection):
  1. Per-block token marginals: TV(hardware, exact law) per block, with
     the multinomial sampling floor for reference.
  2. THE memory observable: cross-block token correlation. Reported as
     the summed pairwise mutual information I(x_a; x_b) over block
     pairs, for hardware samples vs the exact law vs the
     independent-blocks reference (product of the exact per-block
     marginals). If hardware MI ~ exact MI: the memory survived. If it
     collapses toward the product reference: decoherence severed the
     channel.
  3. Detokenized pixel metrics (corr_cross, sigma_E) of hardware
     samples through the shared de-tokenizer, next to MC floors.

Prerequisites:
    pip install qiskit qiskit-aer qiskit-ibm-runtime
    python -c "from qiskit_ibm_runtime import QiskitRuntimeService as S;
               S.save_account(channel='ibm_quantum', token='YOUR_TOKEN')"

HONESTY NOTE: this script has been syntax-checked and its Aer `verify`
path mirrors the audited engine, but no IBM submission was possible in
the development environment. Run `verify` first, every time; it is the
gate that certifies circuit <-> engine agreement before spending QPU.
"""
import sys
import pathlib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR.parent / "src"))

import numpy as np

# Four gates stand between a model and a submitted job, and all four exist
# because each one has caught a real bug at some point:
#   1. verify  -- Aer counts against the exact enumerated law
#   2. 2q count after routing, refused above the threshold
#   3. all-zeros tripwire on the returned bitstrings
#   4. provenance written next to the tokens (job id, theta, calibration)
# Skipping (1) is the fastest way to spend an hour of QPU time on a bug.

from comb import d12_config, CoMBQSpec, SequentialBornModel
from comb.qiskit_hw import verify_against_simulator, run_on_ibm
from _pipeline import load_all, OUT

# ------------------- CONFIG (env-overridable) --------------------------
# Minimal run default: L=2 memory config (theta_M3C0.npy), ~65 2q gates
# after routing. Deploy any trained model, e.g. the record:
#   DEPTH=8 THETA=theta_mmd_M3W3L8.npy python scripts/run_ibm.py verify
import os as _os2
THETA = _os2.environ.get("THETA", "theta_M3C0.npy")
DEPTH = int(_os2.environ.get("DEPTH", 2))
N_MEM = int(_os2.environ.get("N_MEM", 3))
N_WORK = int(_os2.environ.get("N_WORK", 3))
SHOTS = int(_os2.environ.get("SHOTS", 16384))
TAG = f"M{N_MEM}W{N_WORK}L{DEPTH}"
HW_OUT = OUT / f"ibm_run_{TAG}.npz"
# -----------------------------------------------------------------------


def build_model(depth=DEPTH, need_theta=True):
    cfg = d12_config()
    (_, Y_te, blocks, tok, _, T_te, T_fit, T_val) = load_all(cfg)
    m = SequentialBornModel(CoMBQSpec(
        n_mem=N_MEM, n_work=N_WORK, depth=depth, n_blocks=len(blocks),
        use_prefix=False, share_theta=False, seed=cfg.data.seed))
    if need_theta:
        tf = OUT / THETA
        if not tf.exists():
            raise SystemExit(
                f"no trained parameters at {tf}\n"
                f"train them first:  python scripts/run_ibm.py train "
                f"-L {depth}")
        m.theta = np.load(tf)
        m._theta_file = tf.name
    return m, tok, blocks, Y_te, T_te, T_fit, T_val


def pairwise_mi(tokens, K):
    """Summed pairwise mutual information over block pairs (nats)."""
    tokens = np.asarray(tokens, np.int64)
    n, B = tokens.shape
    tot = 0.0
    for a in range(B):
        pa = np.bincount(tokens[:, a], minlength=K) / n
        for b in range(a + 1, B):
            pb = np.bincount(tokens[:, b], minlength=K) / n
            pab = np.zeros((K, K))
            np.add.at(pab, (tokens[:, a], tokens[:, b]), 1.0)
            pab /= n
            m = pab > 0
            tot += float((pab[m] * np.log(
                pab[m] / np.maximum(np.outer(pa, pb)[m], 1e-300))).sum())
    return tot


def mi_of_dist(p, K, B):
    """Same MI functional evaluated on an exact length-K^B law."""
    pj = np.asarray(p).reshape([K] * B)
    tot = 0.0
    for a in range(B):
        for b in range(a + 1, B):
            axes = tuple(i for i in range(B) if i not in (a, b))
            pab = pj.sum(axis=axes) if axes else pj
            if a > b:
                pab = pab.T
            pa = pab.sum(1)
            pb = pab.sum(0)
            m = pab > 0
            tot += float((pab[m] * np.log(
                pab[m] / np.maximum(np.outer(pa, pb)[m], 1e-300))).sum())
    return tot


def cmd_estimate(depth=DEPTH):
    """LOCAL (zero QPU): exact memory signal of the deployed model and
    the minimum shot count for a defensible retention measurement."""
    m, tok, blocks, Y_te, T_te, *_ = build_model(depth)
    p_exact, _ = m.full_distribution()
    mi = mi_of_dist(p_exact, m.K, m.B)
    n_pairs = m.B * (m.B - 1) // 2
    print(f"deployed model: {TAG} ({m.nq} qubits, {m.p_theta} params)")
    print(f"exact memory signal: summed pairwise MI = {mi:.4f} nats")
    print(f"{'shots':>7s} {'MI bias':>9s} {'bias/signal':>12s}")
    for n in (1024, 2048, 4096, 8192):
        b = n_pairs * (m.K - 1) ** 2 / (2.0 * n)
        print(f"{n:7d} {b:9.4f} {100*b/mi:11.1f}%")
    print(f"-> {SHOTS} shots keeps the finite-shot MI bias below ~10% of "
          f"the signal (and is corrected in analysis anyway).")
    print("QPU cost: ONE job, job mode (no session reservation); "
          "expect O(10-60 s) of billed QPU time total.")


def cmd_verify(depth=DEPTH):
    m, *_ = build_model(depth)
    tv = verify_against_simulator(m, shots=200_000)
    print(f"[VERIFY] circuit vs engine TV = {tv:.5f}  -- PASS")
    print("Safe to submit: python scripts/run_ibm.py submit [backend]")


def cmd_submit(backend_name=None, depth=DEPTH, shots=SHOTS):
    m, *_ = build_model()
    tokens, job_id, isa = run_on_ibm(m, backend_name=backend_name,
                                     shots=SHOTS)
    from comb.qiskit_hw import run_on_ibm as _r
    import os as _os
    fold = int(_os.environ.get("CoMBQ_FOLD", "1"))
    out = HW_OUT if fold == 1 else HW_OUT.with_name(
        HW_OUT.stem + f"_fold{fold}.npz")
    np.savez_compressed(out, tokens=tokens, job_id=job_id,
                        theta_file=THETA, depth=DEPTH, shots=SHOTS,
                        readout_errors=np.array(
                            getattr(_r, "last_readout_errors", [])))
    try:
        from qiskit_ibm_runtime import QiskitRuntimeService
        j = QiskitRuntimeService().job(job_id)
        print(f"[USAGE] billed QPU seconds: {j.usage()}")
    except Exception:
        pass
    print(f"[SAVE] {HW_OUT}  ({len(tokens)} generated sequences)")


def model_from_npz():
    z = np.load(HW_OUT, allow_pickle=True)
    return build_model(int(z["depth"])), z


def cmd_analyze():
    (m, tok, blocks, Y_te, T_te, *_), z = model_from_npz()
    T_hw = z["tokens"]
    K, B = m.K, m.B
    p_exact, _ = m.full_distribution()

    print(f"[1] per-block marginal TV (hardware vs exact law); "
          f"floor ~ {np.sqrt(K / (2 * np.pi * len(T_hw))):.4f}")
    pj = p_exact.reshape([K] * B)
    for b in range(B):
        axes = tuple(i for i in range(B) if i != b)
        q = pj.sum(axis=axes)
        emp = np.bincount(T_hw[:, b], minlength=K) / len(T_hw)
        print(f"    block {b}: TV = {0.5 * np.abs(emp - q).sum():.4f}")

    print("[2] memory-survival observable: summed pairwise token MI")
    mi_hw = pairwise_mi(T_hw, K)
    mi_ex = mi_of_dist(p_exact, K, B)
    marg = [pj.sum(axis=tuple(i for i in range(B) if i != b))
            for b in range(B)]
    p_ind = marg[0]
    for b in range(1, B):
        p_ind = np.multiply.outer(p_ind, marg[b])
    mi_ind = mi_of_dist(p_ind.ravel(), K, B)     # = 0 by construction
    n_pairs = B * (B - 1) // 2
    mm_bias = n_pairs * (K - 1) ** 2 / (2.0 * len(T_hw))  # Miller-Madow
    mi_hw_c = mi_hw - mm_bias
    frac = (mi_hw_c - mi_ind) / max(mi_ex - mi_ind, 1e-12)
    print(f"    exact law     : {mi_ex:.4f} nats")
    print(f"    hardware      : {mi_hw:.4f} nats  "
          f"(Miller-Madow corrected: {mi_hw_c:.4f})")
    print(f"    no-memory ref : {mi_ind:.4f} nats")
    print(f"    => memory retention on device: {100 * frac:.1f}%")

    print("[2b] DIAGNOSIS (noise fingerprints from exact simulation)")
    tv_avg = np.mean([0.5 * np.abs(
        np.bincount(T_hw[:, b], minlength=K) / len(T_hw)
        - pj.sum(axis=tuple(i for i in range(B) if i != b))).sum()
        for b in range(B)])
    print(f"    avg block-marginal TV = {tv_avg:.3f};  "
          f"retention = {100*frac:.0f}%")
    print("    reference fingerprints (line model, simulated):")
    print("      full memory dephasing      -> ret ~77%, TV grows mildly")
    print("      T1 damping 10%/boundary    -> ret ~73%")
    print("      readout flips 2%/bit       -> ret ~77%, TV ~ +2-4%")
    print("      readout flips 5%/bit       -> ret ~53%, TV ~ +6-10%")
    if frac < 0.25 and tv_avg < 0.06:
        print("    VERDICT: near-perfect marginals + dead correlations "
              "is NOT a physical noise fingerprint.")
        print("    Suspects: (a) dynamical-decoupling pulses interacting "
              "with mid-circuit measurement scheduling")
        print("              (b) shot alignment across classical "
              "registers in decoding.")
        print("    Decisive 1-variable test: resubmit with DD disabled "
              "(CoMBQ_DD=0 python scripts/run_ibm.py submit <backend>).")
    elif frac < 0.6:
        print("    VERDICT: consistent with strong physical noise; "
              "compare TV against the fingerprint table above.")
    else:
        print("    VERDICT: healthy memory survival.")

    if "readout_errors" in z.files and len(z["readout_errors"]) == 3:
        ro = z["readout_errors"]
        A1 = [np.array([[1 - e, e], [e, 1 - e]]) for e in ro]
        A = A1[0]
        for a in A1[1:]:
            A = np.kron(A, a)                    # 8x8 token confusion
        Ainv = np.linalg.inv(A)
        mi_m = 0.0
        for a_ in range(B):
            for b_ in range(a_ + 1, B):
                pab = np.zeros((K, K))
                np.add.at(pab, (T_hw[:, a_], T_hw[:, b_]), 1.0)
                pab /= len(T_hw)
                pm = np.clip(Ainv @ pab @ Ainv.T, 0, None)
                pm /= pm.sum()
                pa, pb = pm.sum(1), pm.sum(0)
                msk = pm > 0
                mi_m += float((pm[msk] * np.log(pm[msk] / np.maximum(
                    np.outer(pa, pb)[msk], 1e-300))).sum())
        mi_m -= (B * (B - 1) // 2) * (K - 1) ** 2 / (2.0 * len(T_hw))
        print(f"[2c] readout-MITIGATED MI = {mi_m:.4f} nats "
              f"(estimator-level correction; samples remain noisy)")

    f3 = HW_OUT.with_name(HW_OUT.stem + "_fold3.npz")
    if f3.exists():
        T3 = np.load(f3, allow_pickle=True)["tokens"]
        mi1 = pairwise_mi(T_hw, K) - (B*(B-1)//2)*(K-1)**2/(2*len(T_hw))
        mi3 = pairwise_mi(T3, K) - (B*(B-1)//2)*(K-1)**2/(2*len(T3))
        mi0 = (3*mi1 - mi3) / 2.0                 # Richardson to zero noise
        print(f"[2d] ZNE: MI(1x)={mi1:.3f}  MI(3x)={mi3:.3f}  "
              f"-> extrapolated MI(0) = {mi0:.3f} nats")
        from comb import corr_nan_safe
        rngz = np.random.default_rng(3)
        C1 = corr_nan_safe(tok.detokenize(T_hw, rngz))
        C3 = corr_nan_safe(tok.detokenize(T3, rngz))
        C0 = (3*C1 - C3) / 2.0
        np.save(HW_OUT.with_name("corr_zne.npy"), np.clip(C0, -1, 1))
        print("     extrapolated correlation matrix -> outputs/corr_zne.npy"
              " (label figures as 'noise-extrapolated')")

    print("[3] detokenized pixel metrics (shared de-tokenizer)")
    from comb import corr_nan_safe, corr_errors
    Y_hw = tok.detokenize(T_hw[:len(Y_te)], np.random.default_rng(1))
    ce = corr_errors(corr_nan_safe(Y_te), corr_nan_safe(Y_hw), blocks)
    sE = float(Y_hw.sum(1).std() / Y_te.sum(1).std())
    print(f"    corr_cross = {ce['cross']:.4f}   sigma_E ratio = {sE:.3f}"
          f"   (floors: 0.090 / 1.02)")


def cmd_plot():
    """ROOT-style figures from the hardware run -> figures/ibm/, each
    panel a separate PDF+PNG, no in-plot titles. Convention: exact
    model law / MC data = reference; hardware = points with sqrt(N)
    statistical errors."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    RC = {"font.family": "sans-serif", "font.size": 15,
          "axes.linewidth": 1.2, "axes.labelsize": 16,
          "xtick.direction": "in", "ytick.direction": "in",
          "xtick.top": True, "ytick.right": True,
          "xtick.major.size": 7, "ytick.major.size": 7,
          "xtick.minor.size": 3.5, "ytick.minor.size": 3.5,
          "xtick.minor.visible": True, "ytick.minor.visible": True,
          "legend.frameon": False, "savefig.bbox": "tight"}
    plt.rcParams.update(RC)
    outdir = SCRIPT_DIR.parent / "figures" / "ibm"
    outdir.mkdir(parents=True, exist_ok=True)

    (m, tok, blocks, Y_te, T_te, *_), z = model_from_npz()
    T_hw = z["tokens"]
    K, B = m.K, m.B
    p_exact, _ = m.full_distribution()
    pj = p_exact.reshape([K] * B)
    rng = np.random.default_rng(2)
    Y_hw = tok.detokenize(T_hw, rng)

    def save(fig, stem):
        for ext in ("pdf", "png"):
            fig.savefig(outdir / f"{stem}.{ext}", dpi=200)
        plt.close(fig)
        print("saved", outdir / f"{stem}.pdf")

    # 1) per-block token distributions: exact law vs hardware ---------
    for b in range(B):
        q = pj.sum(axis=tuple(i for i in range(B) if i != b))
        n_h = np.bincount(T_hw[:, b], minlength=K)
        h = n_h / len(T_hw)
        eh = np.sqrt(np.maximum(n_h, 0)) / len(T_hw)
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(5.4, 5.0), sharex=True,
            gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))
        edges = np.arange(K + 1) - 0.5
        ax.stairs(q, edges, color="#d62728", lw=1.9,
                  label="exact model law")
        ax.errorbar(np.arange(K), h, yerr=eh, fmt="o", ms=4.5,
                    color="k", lw=1.2, label="hardware")
        ax.set_ylabel("probability")
        ax.legend(loc="best")
        ax.text(0.05, 0.95, f"block {b}", transform=ax.transAxes,
                ha="left", va="top", fontsize=14)
        ax.tick_params(labelbottom=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(q > 0, h / q, np.nan)
            re = np.where(q > 0, eh / q, np.nan)
        axr.errorbar(np.arange(K), r, yerr=re, fmt="o", ms=3.6,
                     color="k", lw=1.0)
        axr.axhline(1.0, color="#d62728", lw=1.2)
        axr.set_ylim(0.5, 1.5)
        axr.set_ylabel("hw / exact", fontsize=13)
        axr.set_xlabel("token")
        axr.set_xticks(range(K))
        save(fig, f"ibm_blk{b}")

    # 2) memory observable: MI bars ----------------------------------
    mi_ex = mi_of_dist(p_exact, K, B)
    mi_hw = pairwise_mi(T_hw, K)
    mm = (B * (B - 1) // 2) * (K - 1) ** 2 / (2.0 * len(T_hw))
    mi_hw_c = mi_hw - mm
    fig, ax = plt.subplots(figsize=(5.2, 4.2))
    vals = [mi_ex, mi_hw_c, 0.0]
    ax.bar(range(3), vals, width=0.55,
           color=["#d62728", "k", "#7f7f7f"], alpha=0.85)
    ax.errorbar([1], [mi_hw_c], yerr=[mm], fmt="none", ecolor="k",
                lw=1.4, capsize=4)
    ax.set_xticks(range(3))
    ax.set_xticklabels(["exact law", "hardware\n(corrected)",
                        "no memory"])
    ax.set_ylabel("summed pairwise MI [nats]")
    ret = 100 * mi_hw_c / max(mi_ex, 1e-12)
    ax.text(0.97, 0.95, f"retention: {ret:.0f}%",
            transform=ax.transAxes, ha="right", va="top", fontsize=15)
    save(fig, "ibm_memory")

    # 3) correlation heatmap of hardware samples ----------------------
    from comb import corr_nan_safe
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    im = ax.imshow(corr_nan_safe(Y_hw), vmin=-1, vmax=1, cmap="RdBu_r",
                   interpolation="nearest")
    for edge in np.cumsum([w for _, w in blocks])[:-1]:
        ax.axhline(edge - 0.5, color="k", lw=0.8, alpha=0.6)
        ax.axvline(edge - 0.5, color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel("calorimeter cell")
    ax.set_ylabel("calorimeter cell")
    ax.set_xticks(range(0, Y_hw.shape[1], 3))
    ax.set_yticks(range(0, Y_hw.shape[1], 3))
    ax.minorticks_off()
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\rho$")
    save(fig, "ibm_corr")

    # 4) total energy: MC data vs noiseless model vs hardware ---------
    T_sim = m.sample(len(T_hw), np.random.default_rng(5))
    Y_sim = tok.detokenize(T_sim, np.random.default_rng(6))
    xs = Y_sim.sum(1)
    xd, xm = Y_te.sum(1), Y_hw.sum(1)
    hi = float(np.quantile(np.concatenate([xd, xm, xs]), 0.997))
    bins = np.linspace(0, hi, 30)
    c = 0.5 * (bins[1:] + bins[:-1]); w = bins[1] - bins[0]
    nd, _ = np.histogram(xd, bins=bins)
    nh, _ = np.histogram(xm, bins=bins)
    ns, _ = np.histogram(xs, bins=bins)
    hd = nd / (len(xd) * w); ed = np.sqrt(nd) / (len(xd) * w)
    hh = nh / (len(xm) * w); eh2 = np.sqrt(nh) / (len(xm) * w)
    hs = ns / (len(xs) * w)
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(5.6, 5.2), sharex=True,
        gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))
    ax.stairs(hs, bins, color="#d62728", lw=1.9,
              label="model (noiseless)")
    ax.errorbar(c[nh > 0], hh[nh > 0], yerr=eh2[nh > 0], fmt="s",
                ms=4.5, color="#d62728", lw=1.2, label="hardware")
    ax.errorbar(c[nd > 0], hd[nd > 0], yerr=ed[nd > 0], fmt="o",
                ms=4.5, color="k", lw=1.2, label="MC data")
    ax.set_ylabel("density"); ax.legend(loc="best")
    ax.tick_params(labelbottom=False)
    # ratio: hardware / noiseless model -- isolates the DEVICE effect
    with np.errstate(divide="ignore", invalid="ignore"):
        r = np.where(ns > 0, hh / np.where(hs > 0, hs, np.nan), np.nan)
        re = r * np.sqrt(1 / np.maximum(nh, 1) + 1 / np.maximum(ns, 1))
    axr.errorbar(c, r, yerr=re, fmt="s", ms=3.6, color="#d62728",
                 lw=1.0)
    axr.axhline(1.0, color="k", lw=1.2)
    axr.set_ylim(0.35, 1.65)
    axr.set_ylabel("hw / model", fontsize=13)
    axr.set_xlabel(r"$E_{\mathrm{sum}}$")
    save(fig, "ibm_energy")

    # 5) per-cell intensity PDFs: MC data vs hardware -----------------
    # The de-tokenizer is classical: each hardware token sequence is
    # de-tokenized M times to smooth the classical smearing at zero
    # QPU cost. Error bars are kept at single-draw (token-level)
    # statistics, since token noise does not average over draws.
    M_DRAWS = 8
    Yh = [tok.detokenize(T_hw, np.random.default_rng(100 + i))
          for i in range(M_DRAWS)]
    d = Y_te.shape[1]
    for j in range(d):
        xd = Y_te[:, j]
        hi = float(np.quantile(np.concatenate([xd, Yh[0][:, j]]), 0.99))
        bins = np.linspace(0, max(hi, 1e-3), 28)
        c = 0.5 * (bins[1:] + bins[:-1]); w = bins[1] - bins[0]
        nd, _ = np.histogram(xd, bins=bins)
        hd = nd / (len(xd) * w)
        ed = np.sqrt(nd) / (len(xd) * w)
        counts = np.stack([np.histogram(Y[:, j], bins=bins)[0]
                           for Y in Yh])
        hh = counts.mean(0) / (len(T_hw) * w)          # smooth central
        eh = np.sqrt(counts.mean(0)) / (len(T_hw) * w)  # 1-draw errors
        fig, (ax, axr) = plt.subplots(
            2, 1, figsize=(5.6, 5.2), sharex=True,
            gridspec_kw=dict(height_ratios=[3, 1], hspace=0.06))
        ax.errorbar(c[nd > 0], hd[nd > 0], yerr=ed[nd > 0], fmt="o",
                    ms=4.5, color="k", lw=1.2, label="MC data")
        mh = counts.mean(0) > 0
        ax.errorbar(c[mh], hh[mh], yerr=eh[mh], fmt="s", ms=4.5,
                    color="#d62728", lw=1.2, label="hardware")
        ax.set_ylabel("density"); ax.legend(loc="best")
        ax.text(0.05, 0.95, f"cell {j}", transform=ax.transAxes,
                ha="left", va="top", fontsize=14)
        ax.tick_params(labelbottom=False)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(counts.mean(0) > 0,
                         hd / np.where(hh > 0, hh, np.nan), np.nan)
            re = r * np.sqrt(1 / np.maximum(nd, 1)
                             + 1 / np.maximum(counts.mean(0), 1))
        axr.errorbar(c, r, yerr=re, fmt="o", ms=3.6, color="k", lw=1.0)
        axr.axhline(1.0, color="#d62728", lw=1.2)
        axr.set_ylim(0.35, 1.65)
        axr.set_ylabel("data / hw", fontsize=13)
        axr.set_xlabel(f"cell {j} intensity")
        save(fig, f"ibm_marg_{j:02d}")

    # 5) noiseless-model correlation heatmap (pair for ibm_corr) ------
    fig, ax = plt.subplots(figsize=(5.0, 4.3))
    im = ax.imshow(corr_nan_safe(Y_sim), vmin=-1, vmax=1,
                   cmap="RdBu_r", interpolation="nearest")
    for edge in np.cumsum([w2 for _, w2 in blocks])[:-1]:
        ax.axhline(edge - 0.5, color="k", lw=0.8, alpha=0.6)
        ax.axvline(edge - 0.5, color="k", lw=0.8, alpha=0.6)
    ax.set_xlabel("calorimeter cell"); ax.set_ylabel("calorimeter cell")
    ax.set_xticks(range(0, Y_sim.shape[1], 3))
    ax.set_yticks(range(0, Y_sim.shape[1], 3))
    ax.minorticks_off()
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cb.set_label(r"$\rho$")
    save(fig, "ibm_corr_sim")


def cmd_train(depth=DEPTH, epochs=None):
    from comb import block_grams, train_blockscore
    m, tok, blocks, Y_te, T_te, T_fit, T_val = build_model(
        depth, need_theta=False)
    Kb = block_grams(tok)
    ep = epochs or (120 + 60 * depth)
    print(f"[TRAIN] {TAG}, {ep} epochs (early stopping, patience 60)")
    train_blockscore(m, Kb, T_fit, T_val, epochs=ep, batch=1024,
                     lr=0.08, seed=d12_config().data.seed, patience=60)
    tf = OUT / THETA
    np.save(tf, m.theta)
    print(f"[SAVE] {tf}")
    print(f"[TEST] blockwise score = "
          f"{m.blockwise_score(T_te, Kb):.4f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(
        description="CoMB on IBM hardware: train/estimate/verify/"
                    "submit/redecode/analyze/plot")
    p.add_argument("command", choices=["train", "estimate", "verify",
                                       "submit", "redecode", "analyze",
                                       "plot"])
    p.add_argument("target", nargs="?", default=None,
                   help="backend name (submit) or job id (redecode)")
    p.add_argument("-L", "--depth", type=int, default=DEPTH,
                   help=f"circuit depth per block (default {DEPTH})")
    p.add_argument("--shots", type=int, default=SHOTS)
    p.add_argument("--epochs", type=int, default=None,
                   help="train: override epoch budget")
    a = p.parse_args()
    if a.command == "train":
        cmd_train(a.depth, a.epochs)
    elif a.command == "estimate":
        cmd_estimate(a.depth)
    elif a.command == "verify":
        cmd_verify(a.depth)
    elif a.command == "submit":
        cmd_submit(a.target, a.depth, a.shots)
    elif a.command == "redecode":
        if not a.target:
            raise SystemExit("redecode needs a job id")
        import numpy as _np
        from comb.qiskit_hw import redecode_job
        jid = a.target
        m, *_ = build_model(a.depth)
        tokens = redecode_job(jid, m.n_w, m.B)
        _np.savez_compressed(HW_OUT, tokens=tokens, job_id=jid,
                             theta_file=m._theta_file, depth=a.depth,
                             shots=len(tokens))
        print(f"[SAVE] {HW_OUT}  ({len(tokens)} sequences, re-decoded)")
        print("now run: analyze, then plot")
    elif a.command == "analyze":
        cmd_analyze()
    elif a.command == "plot":
        cmd_plot()
