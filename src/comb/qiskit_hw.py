"""
Qiskit adapter: hardware sampling circuits for CoMB.

Scope and honesty
-----------------
* Supports the MEMORY-ONLY conditioning configs (use_prefix=False,
  e.g. M3C0*): these need only mid-circuit measurement + reset of the
  work register, which is native on IBM Heron (ibm_kingston) and on
  trapped-ion systems. The classical-prefix configs (C1) require
  mid-circuit ANGLE feed-forward and are intentionally not emitted here.
* This module could not be executed in the build environment (no qiskit
  installed there); it mirrors SequentialBornModel._block_ops
  gate-for-gate and ships with `verify_against_simulator`, which MUST be
  run once on qiskit-aer before any hardware submission: it compares
  Aer sampling frequencies of the emitted circuit against
  model.full_distribution() and raises if the TV distance exceeds the
  multinomial floor. Do not skip this step.

One shot of the emitted circuit = one generated token sequence: the
work-register classical bits of block beta ARE the token x_beta. There
is no expectation estimation anywhere in generation.
"""
from __future__ import annotations

import os


from math import pi

import numpy as np


def build_sampling_circuit(model, fold: int = 1):
    """Return a qiskit QuantumCircuit sampling p_theta(x_1..x_B) for a
    positional-only SequentialBornModel."""
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    if model.spec.use_prefix:
        raise NotImplementedError(
            "hardware path supports memory-only conditioning (C0); the "
            "prefix channel requires dynamic-circuit angle feed-forward")
    n_m, n_w, nq, B, L = (model.n_m, model.n_w, model.nq, model.B,
                          model.L)
    qr = QuantumRegister(nq, "q")
    crs = [ClassicalRegister(n_w, f"blk{b}") for b in range(B)]
    qc = QuantumCircuit(qr, *crs)
    for beta in range(B):
        # positional encoding angles (deterministic, theta-free)
        a = 1.0 / (1.0 + np.exp(-model.pos[beta]))       # (n_w,)
        th = model._theta_block(beta)
        t = 0
        for _layer in range(L):
            for k in range(n_w):
                qc.ry(pi * float(a[k]), qr[n_m + k])
            for q in range(nq):
                qc.rz(float(th[t]), qr[q]); t += 1
                qc.ry(float(th[t]), qr[q]); t += 1
            for q in range(nq - 1):
                for _ in range(fold):          # CZ^fold = CZ for odd fold
                    qc.cz(qr[q], qr[q + 1])
            if nq > 2 and model.spec.entangler == "ring":
                for _ in range(fold):
                    qc.cz(qr[nq - 1], qr[0])
        qc.barrier()
        for k in range(n_w):
            qc.measure(qr[n_m + k], crs[beta][k])
        if beta < B - 1:
            for k in range(n_w):
                qc.reset(qr[n_m + k])
            qc.barrier()
    return qc


# Bit order. Qiskit prints bitstrings little-endian (qubit 0 rightmost)
# and separates classical registers by spaces, LAST-declared first. So the
# registers must be reversed before decoding, and within a register bit k
# lives at string position -1-k. Get either wrong and you get a plausible
# but permuted token distribution, which is exactly the kind of bug that
# survives to the device. samplerv2_tokens() below must use the identical
# convention; run_ibm.py verify exists to check they agree.
def counts_to_tokens(counts: dict, n_w: int, B: int):
    """Decode qiskit counts into (n_sequences, B) token arrays.

    Qiskit orders classical registers little-endian and separates them
    with spaces, LAST-declared register first; bit k of block beta was
    written from work qubit k, so within a register the bitstring reads
    (msb..lsb) = (k = n_w-1 .. 0), matching the engine's convention
    where work qubit n_m (k=0) is the most significant token bit —
    hence the reversal below.
    """
    seqs, weights = [], []
    for bitstr, c in counts.items():
        regs = bitstr.split()          # [blk{B-1}, ..., blk0]
        toks = []
        for beta in range(B):
            r = regs[B - 1 - beta]     # register for block beta
            x = 0
            for k in range(n_w):       # bit k = char at position -1-k
                x |= int(r[-1 - k]) << (n_w - 1 - k)
            toks.append(x)
        seqs.append(toks)
        weights.append(int(c))
    return np.array(seqs, dtype=np.int64), np.array(weights, dtype=np.int64)


def _verify_counts_path(model, shots: int = 200_000,
                             tv_margin: float = 4.0) -> float:
    """MANDATORY pre-hardware check. Samples the emitted circuit on
    qiskit-aer and compares against the exact enumerated distribution.
    Returns the TV distance; raises AssertionError on failure."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile

    qc = build_sampling_circuit(model)
    backend = AerSimulator()
    res = backend.run(transpile(qc, backend), shots=shots).result()
    seqs, w = counts_to_tokens(res.get_counts(), model.n_w, model.B)
    flat = np.zeros(len(seqs), dtype=np.int64)
    for b in range(model.B):
        flat = flat * model.K + seqs[:, b]
    emp = np.zeros(model.K ** model.B)
    np.add.at(emp, flat, w)
    emp /= emp.sum()
    p, _ = model.full_distribution()
    tv = 0.5 * float(np.abs(emp - p).sum())
    floor = float(np.sqrt(len(p) / (2 * np.pi * shots)))
    assert tv < tv_margin * floor, \
        f"circuit/engine mismatch: TV={tv:.5f} floor={floor:.5f}"
    return tv


# ----------------------------------------------------------------------
# IBM Runtime (SamplerV2) path — modern primitives API
# ----------------------------------------------------------------------

def transpile_for_backend(qc, backend, optimization_level: int = 3,
                          initial_layout=None):
    """Preset-pass-manager transpilation. On heavy-hex devices the CZ
    ring's wrap-around edge is routed with SWAPs; passing an
    initial_layout over a connected line of good qubits (check the
    backend's error map) typically reduces the inserted SWAP count."""
    from qiskit.transpiler.preset_passmanagers import \
        generate_preset_pass_manager
    pm = generate_preset_pass_manager(
        optimization_level=optimization_level, backend=backend,
        initial_layout=initial_layout)
    return pm.run(qc)


def samplerv2_tokens(result, n_w: int, B: int) -> np.ndarray:
    """
    Decode a SamplerV2 PrimitiveResult into per-shot token sequences,
    shape (shots, B), mirroring counts_to_tokens exactly.

    BUGFIX (post-mortem of an all-zeros hardware run): BitArray packs
    each shot's bits into the LOW end of its bytes; unpacking with
    numpy and taking the FIRST `num_bits` bits reads the zero padding.
    We therefore decode via BitArray.get_bitstrings(), whose strings
    follow the standard qiskit clbit order (leftmost char = highest
    clbit), and apply the same per-register mapping as
    counts_to_tokens: token bit for work qubit k = char at position
    -1-k, weighted 2^(n_w-1-k).
    """
    data = result[0].data
    reg0 = getattr(data, "blk0")
    shots = reg0.num_shots
    toks = np.zeros((shots, B), dtype=np.int64)
    for beta in range(B):
        arr = getattr(data, f"blk{beta}")
        strs = arr.get_bitstrings()
        x = np.zeros(shots, dtype=np.int64)
        for k in range(n_w):
            bit_k = np.frombuffer(
                "".join(sr[-1 - k] for sr in strs).encode(), dtype="S1"
            ) == b"1"
            x |= bit_k.astype(np.int64) << (n_w - 1 - k)
        toks[:, beta] = x
    return toks


def redecode_job(job_id: str, n_w: int, B: int,
                 channel_kwargs: dict | None = None):
    """Re-fetch a COMPLETED job from the IBM cloud and re-decode it
    with the fixed decoder. Costs zero QPU time."""
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = QiskitRuntimeService(**(channel_kwargs or {}))
    job = service.job(job_id)
    result = job.result()
    return samplerv2_tokens(result, n_w, B)


def pick_line_layout(backend, nq: int, n_work: int):
    """Choose the physical qubit path of length nq minimizing summed
    two-qubit gate error (+ readout error on the work positions, which
    are measured and reset every block). Returns a list of physical
    qubits for initial_layout, or None if the search fails."""
    try:
        target = backend.target
        edges, err = [], {}
        for gate in ("cz", "ecr", "cx"):
            if gate in target.operation_names:
                for q, props in target[gate].items():
                    if len(q) == 2 and props is not None:
                        e = getattr(props, "error", None)
                        if e is not None:
                            edges.append(tuple(q))
                            err[tuple(q)] = err[(q[1], q[0])] = float(e)
                break
        ro = {}
        if "measure" in target.operation_names:
            for q, props in target["measure"].items():
                if props is not None and getattr(props, "error", None) \
                        is not None:
                    ro[q[0]] = float(props.error)
        adj = {}
        for a, b in edges:
            adj.setdefault(a, set()).add(b)
            adj.setdefault(b, set()).add(a)

        best = [None, np.inf]

        def cost(path):
            c = sum(err[(path[i], path[i + 1])]
                    for i in range(len(path) - 1))
            c += 0.5 * sum(ro.get(q, 0.01) for q in path[-n_work:])
            return c

        def dfs(path, c):
            if c >= best[1]:
                return
            if len(path) == nq:
                best[0], best[1] = list(path), c
                return
            for nxt in adj.get(path[-1], ()):
                if nxt not in path:
                    dfs(path + [nxt],
                        c + err[(path[-1], nxt)])
        for start in adj:
            dfs([start], 0.0)
        if best[0] is not None:
            print(f"[LAYOUT] best physical path {best[0]}  "
                  f"(sum 2q err = {best[1]:.4f})")
        return best[0]
    except Exception as e:
        print(f"[LAYOUT] auto-selection failed ({e}); "
              "falling back to transpiler layout")
        return None


def run_on_ibm(model, backend_name: str = None, shots: int = 8192,
               use_dd: bool = True, initial_layout=None,
               channel_kwargs: dict | None = None):
    """
    Submit ONE sampling job: `shots` hardware shots = `shots` generated
    token sequences (one physical B-block rollout each). Returns
    (tokens, job_id, isa_circuit).

    Requires: pip install qiskit qiskit-ibm-runtime, and a saved account
    (QiskitRuntimeService.save_account(channel="ibm_quantum",
     token="...")). verify_against_simulator(model) MUST pass on
    qiskit-aer before calling this.
    """
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2

    service = QiskitRuntimeService(**(channel_kwargs or {}))
    backend = (service.backend(backend_name) if backend_name
               else service.least_busy(operational=True,
                                       simulator=False,
                                       min_num_qubits=model.nq))
    fold = int(os.environ.get("CoMBQ_FOLD", "1"))
    if fold not in (1, 3, 5):
        raise ValueError("CoMBQ_FOLD must be an odd int (1, 3, 5)")
    if fold > 1:
        print(f"[IBM] ZNE gate folding x{fold} "
              "(identical unitary, amplified 2q noise)")
    qc = build_sampling_circuit(model, fold=fold)
    n2q_logical = sum(1 for inst in qc.data
                      if inst.operation.num_qubits == 2)
    if initial_layout is None and model.spec.entangler == "line":
        initial_layout = pick_line_layout(backend, model.nq, model.n_w)
    isa = transpile_for_backend(qc, backend,
                                initial_layout=initial_layout)
    n2q = sum(1 for inst in isa.data
              if inst.operation.num_qubits == 2)
    print(f"[IBM] backend={backend.name}  entangler="
          f"{model.spec.entangler}  logical 2q={n2q_logical}  "
          f"after routing={n2q}  depth={isa.depth()}")
    if n2q > 1.3 * n2q_logical:
        print(f"[WARN] routing overhead {n2q}/{n2q_logical} = "
              f"{n2q / n2q_logical:.1f}x -- if entangler is 'ring' you "
              "are running STALE code; expected ~= logical count for "
              "the line model.")
    sampler = SamplerV2(mode=backend)
    if os.environ.get("CoMBQ_DD", "1") == "0":
        use_dd = False
        print("[IBM] dynamical decoupling DISABLED (CoMBQ_DD=0)")
    if use_dd:
        sampler.options.dynamical_decoupling.enable = True
        sampler.options.dynamical_decoupling.sequence_type = "XY4"
    if os.environ.get("CoMBQ_TWIRL", "0") == "1":
        try:
            sampler.options.twirling.enable_gates = True
            print("[IBM] Pauli gate-twirling ENABLED")
        except Exception as e:
            print(f"[IBM] twirling unavailable ({e})")
    # record work-qubit readout errors for post-hoc mitigation
    ro_err = []
    try:
        layout = isa.layout.final_index_layout()
        work_phys = [layout[q] for q in range(model.n_m, model.nq)]
        for pq in work_phys:
            ro_err.append(float(backend.target["measure"][(pq,)].error))
        print(f"[IBM] work-qubit readout errors: "
              f"{[round(e, 4) for e in ro_err]}")
    except Exception:
        ro_err = []
    run_on_ibm.last_readout_errors = ro_err
    job = sampler.run([isa], shots=shots)
    print(f"[IBM] job id: {job.job_id()}")
    result = job.result()
    tokens = samplerv2_tokens(result, model.n_w, model.B)
    return tokens, job.job_id(), isa


def verify_against_simulator(model, shots: int = 200_000):
    """Two-path verification, BOTH mandatory:
    (a) legacy counts path (backend.run/get_counts decoding);
    (b) the SamplerV2 path -- the IDENTICAL decode function used on
        hardware (samplerv2_tokens), exercised via qiskit-aer's
        SamplerV2. A decode bug in the hardware path now fails HERE,
        locally, before any QPU is spent.
    Returns the SamplerV2-path total variation distance.
    """
    tv_counts = _verify_counts_path(model, shots=shots)
    print(f"[VERIFY a] counts-path TV = {tv_counts:.5f}")
    from qiskit_aer.primitives import SamplerV2 as AerSamplerV2
    qc = build_sampling_circuit(model)
    res = AerSamplerV2().run([qc], shots=shots).result()
    toks = samplerv2_tokens(res, model.n_w, model.B)
    p_exact, seqs = model.full_distribution()
    idx = np.zeros(len(toks), dtype=np.int64)
    for b in range(model.B):
        idx = idx * model.K + toks[:, b]
    emp = np.bincount(idx, minlength=model.K ** model.B) / len(toks)
    tv = 0.5 * float(np.abs(emp - p_exact).sum())
    floor = np.sqrt(model.K ** model.B / (2 * np.pi * shots))
    print(f"[VERIFY b] SamplerV2-path TV = {tv:.5f} "
          f"(multinomial floor ~ {floor:.5f})")
    if tv > 5 * floor + 0.01:
        raise RuntimeError(
            f"SamplerV2 decode path FAILED verification: TV={tv:.4f}. "
            "Do NOT submit to hardware.")
    return tv
