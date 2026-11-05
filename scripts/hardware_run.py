#!/usr/bin/env python3
"""Device run of the one-round CoMB circuit (learned count readout).

    python scripts/hardware_run.py verify outputs/runs/<run>.npz              # Aer check of the circuit conventions (no account needed)
    python scripts/hardware_run.py submit outputs/runs/<run>.npz --backend ibm_kingston --shots 4000 [--steps d] [--batch m]
    python scripts/hardware_run.py duration outputs/runs/<run>.npz --backend ibm_kingston --steps 12   # time budget vs T1, T2
                                           # --batch m: m coordinates per measurement layer; --batch d: no mid-circuit measurement
    python scripts/hardware_run.py fetch  outputs/runs/<run>.npz --job <job id> [--steps d]   # retrieve a finished job's bits from IBM
    python scripts/hardware_run.py decode outputs/runs/<run>.npz outputs/hardware/<job>.npz   # bits -> counts -> showers
    python scripts/hardware_run.py plot   outputs/runs/<run>.npz outputs/hardware/<job>.samples.npz   # count-level figures -> outputs/figs/
    python scripts/hardware_run.py complete outputs/runs/<run>.npz outputs/hardware/<job>.samples.npz # device-started showers -> outputs/samples/
                                           # (device counts for the first k coordinates, exact model for the rest), then:  python scripts/comb_ds1.py plot

verify must report a total-variation distance at its multinomial floor before anything is submitted.
Each shot of the dynamic circuit is one shower (d measure-and-reset rounds on n_m + n_w qubits).
"""
import sys, json, pathlib, argparse, time
import numpy as np
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from comb.showers import zoo
from comb.showers.qiskit_comb import build_circuit, counts_from_bitstrings, samples_from_counts, verify_numpy, verify_aer, complete_from_device
from comb.showers.qiskit_comb import build_circuit_deferred, exact_joint_counts, deferred_distribution_numpy
ap = argparse.ArgumentParser()
ap.add_argument("cmd", choices=["verify", "submit", "fetch", "decode", "plot", "complete", "duration"])
ap.add_argument("run")
ap.add_argument("record", nargs="?")
ap.add_argument("--backend", default="ibm_kingston")
ap.add_argument("--shots", type=int, default=4000)
ap.add_argument("--steps", type=int, default=None)
ap.add_argument("--job", default=None)
ap.add_argument("--batch", type=int, default=None, help="coordinates per measurement layer; --batch = --steps removes every mid-circuit measurement")
a = ap.parse_args()
(ROOT / "outputs/hardware").mkdir(parents=True, exist_ok=True)
model, spec, extra = zoo.load(a.run)
name = pathlib.Path(a.run).stem
if a.cmd == "verify":
    print(f"gate list vs model unitaries: max |diff| {verify_numpy(model):.1e} (must be ~1e-15)")
    tv, floor = verify_aer(model, shots=40000)
    print(f"Aer, first coordinate: TV(empirical, exact) = {tv:.4f}, multinomial floor ~ {floor:.4f}  ->  {'OK' if tv < 3 * floor + 0.005 else 'FAILED: do not submit'}")
    ex, de = exact_joint_counts(model, 3), deferred_distribution_numpy(model, 3)
    err = max(abs(ex.get(k, 0) - de.get(k, 0)) for k in set(ex) | set(de))
    print(f"circuit without mid-circuit measurements, 3 coordinates: max |P - P_exact| = {err:.1e} (must be ~1e-16)")
elif a.cmd == "submit":
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    from qiskit import transpile
    qc = build_circuit_deferred(model, steps=a.steps, batch=a.batch) if a.batch else build_circuit(model, steps=a.steps)
    service = QiskitRuntimeService()
    backend = service.backend(a.backend)
    isa = transpile(qc, backend, optimization_level=1)
    n2q = sum(1 for inst in isa.data if inst.operation.num_qubits == 2)
    print(f"{a.backend}: {qc.num_qubits} qubits, {qc.num_clbits} measurements, two-qubit gates after routing {n2q} (logical {sum(1 for i in qc.data if i.operation.num_qubits == 2)})")
    job = SamplerV2(mode=backend).run([isa], shots=a.shots)
    print("job id:", job.job_id())
    res = job.result()[0]
    bits = next(iter(res.data.values())).get_bitstrings()
    out = ROOT / "outputs/hardware" / f"{name}_{a.backend}_{job.job_id()}.npz"
    np.savez(out, bitstrings=np.array(bits), job_id=job.job_id(), backend=a.backend, run=a.run, shots=a.shots, steps=a.steps or -1, batch=a.batch or 1, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    print("saved", out)
elif a.cmd == "duration":
    # Time budget of the dynamic circuit on the device: transpiled exactly as for 'submit', then scheduled with the
    # backend's own gate, measurement and reset durations; compared with T1 and T2 of the physical qubits it uses.
    from qiskit_ibm_runtime import QiskitRuntimeService
    from qiskit import transpile
    service = QiskitRuntimeService()
    backend = service.backend(a.backend)
    steps = a.steps or 12
    def timed(k):
        """Transpile the circuit of the first k rounds for the device, and return it with the device's estimate of
        its duration in seconds.
        """
        isa = transpile(build_circuit(model, steps=k), backend, optimization_level=1, seed_transpiler=0)
        return isa, isa.estimate_duration(backend.target, unit="s")
    isa1, t1 = timed(1)
    isa, tk = timed(steps)
    per_round = (tk - t1) / max(steps - 1, 1)
    used = sorted({isa.find_bit(q).index for inst in isa.data for q in inst.qubits})
    props = [backend.qubit_properties(q) for q in used]
    T1 = np.array([p.t1 for p in props if p is not None and p.t1])
    T2 = np.array([p.t2 for p in props if p is not None and p.t2])
    n2q = sum(1 for inst in isa.data if inst.operation.num_qubits == 2)
    full = model.base.d if hasattr(model, "base") else model.d
    gates_only = isa.copy_empty_like()                                       # same circuit without measurement and reset
    for inst in isa.data:
        if inst.operation.name not in ("measure", "reset"): gates_only.append(inst)
    share = 1.0 - gates_only.estimate_duration(backend.target, unit="s") / tk
    print(f"{a.backend}, physical qubits {used}: {n2q} two-qubit gates after routing for {steps} rounds")
    print(f"  duration: {steps} rounds {tk * 1e6:.1f} us, one round {per_round * 1e6:.2f} us  ->  {full} rounds (full shower) {(t1 + (full - 1) * per_round) * 1e3:.2f} ms")
    print(f"  measurement and reset take {100 * share:.0f}% of the circuit time; gates the rest")
    if len(T1) and len(T2):
        print(f"  coherence of these qubits: median T1 {np.median(T1) * 1e6:.0f} us, median T2 {np.median(T2) * 1e6:.0f} us"
              f"  ->  rounds per T2: {np.median(T2) / per_round:.0f}; the {steps}-round run lasts {tk / np.median(T2):.2f} T2")
elif a.cmd == "fetch":
    from qiskit_ibm_runtime import QiskitRuntimeService
    job = QiskitRuntimeService().job(a.job)
    res = job.result()[0]
    bits = next(iter(res.data.values())).get_bitstrings()
    backend = getattr(job, "backend", lambda: None)()
    bname = getattr(backend, "name", a.backend) if backend is not None else a.backend
    out = ROOT / "outputs/hardware" / f"{name}_{bname}_{a.job}.npz"
    np.savez(out, bitstrings=np.array(bits), job_id=a.job, backend=bname, run=a.run, shots=len(bits), steps=a.steps or -1, time=time.strftime("%Y-%m-%d %H:%M:%S"))
    print(f"fetched {len(bits)} shots from job {a.job} on {bname} -> {out}")
elif a.cmd == "plot":
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.stats import spearmanr
    rec = np.load(a.record, allow_pickle=True)
    C = rec["counts"]
    tv = rec["tv"]
    q = rec["exact_marginals"]
    k, K = C.shape[1], q.shape[1]
    b = model.base if hasattr(model, "base") else model
    figs = ROOT / "outputs/figs"
    figs.mkdir(parents=True, exist_ok=True)
    Cn = np.asarray(b.sample_counts(len(C), np.random.default_rng(0)))[:, :k]           # noiseless counts from the exact model
    floor = np.sqrt(K / (2 * np.pi * len(C)))
    tag = pathlib.Path(a.record).stem.replace(".samples", "")
    # 1. memory survival along the chain: TV(device, exact) per coordinate, with the noiseless sample's own TV as the floor
    tv_n = 0.5 * np.abs(np.stack([np.bincount(Cn[:, j], minlength=K) / len(Cn) for j in range(k)]) - q).sum(1)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.plot(range(1, k + 1), tv, "o-", color="tab:red", label=f"{rec.get('backend', 'device')} vs exact")
    ax.plot(range(1, k + 1), tv_n, "s--", color="tab:blue", label="noiseless sample vs exact (shot noise)")
    ax.axhline(floor, color="k", lw=0.6, ls=":", label="multinomial floor")
    ax.set_xlabel("coordinate (measure-and-reset round)")
    ax.set_ylabel("total-variation distance of the count law")
    ax.set_ylim(0, None)
    ax.legend(fontsize=8)
    fig.savefig(figs / f"hw_tv_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)
    # 2. count histograms per coordinate
    nr = (k + 3) // 4
    fig, axes = plt.subplots(nr, 4, figsize=(13, 2.6 * nr), squeeze=False)
    for j, ax in enumerate(axes.ravel()):
        if j >= k: ax.axis("off"); continue
        ax.bar(np.arange(K) - 0.2, np.bincount(C[:, j], minlength=K) / len(C), width=0.4, color="tab:red", label="device")
        ax.bar(np.arange(K) + 0.2, q[j], width=0.4, color="tab:blue", label="exact")
        ax.set_title(f"coordinate {j + 1}  (TV {tv[j]:.2f})", fontsize=9)
        ax.set_xticks(range(K))
        if j == 0: ax.legend(fontsize=8)
    fig.supxlabel("count c (work register read as a binary number)")
    fig.supylabel("probability")
    fig.savefig(figs / f"hw_counts_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)
    # 3. does the memory carry dependence on the device? Spearman correlations between the coordinates' counts
    S_dev, S_noi = np.nan_to_num(spearmanr(C).correlation), np.nan_to_num(spearmanr(Cn).correlation)
    off = ~np.eye(k, dtype=bool)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8))
    for ax, M, t in zip(axes, (S_noi, S_dev, S_dev - S_noi), ("exact model (noiseless samples)", str(rec.get("backend", "device")), "device - exact")):
        im = ax.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("coordinate")
        ax.set_ylabel("coordinate")
        fig.colorbar(im, ax=ax, fraction=0.046)
    fig.savefig(figs / f"hw_corr_{tag}.pdf", bbox_inches="tight")
    plt.close(fig)
    print(f"mean |Spearman| between coordinates: exact model {np.abs(S_noi[off]).mean():.3f}, device {np.abs(S_dev[off]).mean():.3f}; mean |device - exact| {np.abs(S_dev - S_noi)[off].mean():.3f}")
    print(f"TV per coordinate: device mean {tv.mean():.3f}, noiseless-sample mean {tv_n.mean():.3f}, floor {floor:.3f}")
    print(f"figures: outputs/figs/hw_tv_{tag}.pdf  hw_counts_{tag}.pdf  hw_corr_{tag}.pdf")
elif a.cmd == "complete":
    rec = np.load(a.record, allow_pickle=True)
    C_dev = rec["counts"]
    k = C_dev.shape[1]
    C = complete_from_device(model, C_dev, np.random.default_rng(0))
    Y = samples_from_counts(model, C, np.random.default_rng(1))
    meta = json.load(open(ROOT / "data/cal_ds1_meta.json"))
    med, mad, t0, w = meta["median_E"], meta["mad_E"], meta["t0"], meta["w"]
    E = med + mad * (Y[:, 0] - t0) / w
    F = np.clip(Y[:, 1:], 0, None)
    F = F / np.where(F.sum(1) > 0, F.sum(1), 1.0)[:, None]
    S = F * E[:, None]
    # same ring-by-ring order as comb_ds1.py sample (detected from the test showers it saved)
    S_te = np.load(ROOT / "outputs/samples/_test.npz")["showers"]
    sys.path.insert(0, str(ROOT))
    import importlib.util
    spec_ = importlib.util.spec_from_file_location("cd", ROOT / "scripts" / "comb_ds1.py")
    cd = importlib.util.module_from_spec(spec_)
    try: spec_.loader.exec_module(cd)
    except SystemExit: pass
    perm, _ = cd.ring_major_perm(S_te if False else S, meta["layers"])
    S = S[:, perm]   # S_te is already ring-major; reorder the new showers the same way
    backend = str(rec.get("backend", "device")) if "backend" in rec.files else "device"
    lab = dict(spec)
    lab["label"] = f"CoMB, {backend}"
    out = ROOT / "outputs/samples" / f"{name}_{backend}.npz"
    np.savez(out, showers=S, spec=json.dumps(lab), counts=C, k_device=k)
    from scipy.stats import spearmanr, wasserstein_distance
    off = ~np.eye(S.shape[1], dtype=bool)
    sp = float(np.abs(np.nan_to_num(spearmanr(S).correlation) - np.nan_to_num(spearmanr(S_te).correlation))[off].mean())
    sd = S_te.std(0)
    w1 = float(np.mean([wasserstein_distance(S[:, j], S_te[:, j]) / sd[j] for j in range(S.shape[1]) if sd[j] > 0]))
    madE = lambda M: np.median(np.abs(M.sum(1) - np.median(M.sum(1))))
    print(f"{len(S)} device-started showers -> {out}")
    print(f"  vs Geant4 test showers: Spearman err {sp:.3f}, W1 (all voxels, sd units) {w1:.3f}, zeros {100 * (S == 0).mean():.1f}% (data {100 * (S_te == 0).mean():.1f}%), E spread {madE(S) / madE(S_te):.2f}")
    print("  now:  python scripts/comb_ds1.py plot     (overlays Geant4, the simulated model and the device-started showers)")
else:
    rec = np.load(a.record, allow_pickle=True)
    steps = int(rec["steps"]) if int(rec["steps"]) > 0 else None
    C = counts_from_bitstrings(model, [str(s) for s in rec["bitstrings"]], steps=steps)
    Y = samples_from_counts(model, C, np.random.default_rng(0))
    # device count statistics against the exact (noiseless) ones, coordinate by coordinate
    b = model.base if hasattr(model, "base") else model
    q = np.asarray(b.count_marginals())[: C.shape[1]]
    emp = np.stack([np.bincount(C[:, k], minlength=q.shape[1]) / len(C) for k in range(C.shape[1])])
    tv = 0.5 * np.abs(emp - q).sum(1)
    floor = np.sqrt(q.shape[1] / (2 * np.pi * len(C)))
    print(f"device vs exact count laws: TV per coordinate mean {tv.mean():.3f}, max {tv.max():.3f} at coordinate {int(tv.argmax())}; shot-noise floor ~{floor:.3f}")
    print("  per coordinate: " + " ".join(f"{t:.2f}" for t in tv))
    out = pathlib.Path(a.record).with_suffix(".samples.npz")
    np.savez(out, counts=C, showers=Y, tv=tv, exact_marginals=q, spec=json.dumps(spec), backend=str(rec["backend"]) if "backend" in rec.files else "device")
    kind = "device showers" if C.shape[1] == b.d else f"partial run: standardized readout values of the first {C.shape[1]} coordinates (no full showers)"
    print(f"{len(Y)} {kind} -> {out}  (mean count per coordinate {C.mean():.2f})")
