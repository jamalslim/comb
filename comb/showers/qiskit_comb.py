"""Hardware path for the one-round CoMB circuit (learned count readout).

One sample = one execution of a dynamic circuit on n_m + n_w qubits:
    for every coordinate beta:  W_beta on the memory qubits; reset the work qubits (beta > 0);
                                V_beta on memory + work; measure the work qubits.
The measured bits give the counts c_beta (work register read as a binary number, first work
qubit = most significant bit, exactly as comb.model with current="binary"), the counts go through
the learned readout g_beta and the fixed decoder, and the result is a shower.

Conventions (checked by verify_numpy, and by verify_aer when qiskit is installed):
  * the model is big-endian: model qubit 0 is the most significant bit of a basis index;
    memory qubits are model indices 0..n_m-1, work qubits n_m..n_m+n_w-1;
  * comb.circuit.ry / rz are Qiskit's RY / RZ, gates are applied in the model's op order;
  * Qiskit prints bitstrings little-endian; we read every classical bit by its index instead.
"""
import numpy as np
from .circuit import ry, rz

def _base(model):
    return model.base if hasattr(model, "base") else model

def op_list(circuit, theta):
    """The circuit's gates as (kind, qubits, angle) in application order."""
    out = []
    for kind, qs, p in circuit.ops:
        out.append((kind, qs, None if p is None else float(theta[p])))
    return out

def build_circuit(model, steps=None):
    """QuantumCircuit for the first `steps` coordinates (default all d). Classical bit beta*n_w + q
    holds the measurement of work qubit q at coordinate beta."""
    from qiskit import QuantumCircuit
    b = _base(model)
    n_m, n_w, d = b.n_m, b.n_w, b.d if steps is None else int(steps)
    n = n_m + n_w
    qc = QuantumCircuit(n, d * n_w)
    def apply(ops, offset=0):
        for kind, qs, ang in ops:
            qs = [q + offset for q in qs]
            if kind == "ry": qc.ry(ang, qs[0])
            elif kind == "rz": qc.rz(ang, qs[0])
            elif kind == "cz": qc.cz(qs[0], qs[1])
            elif kind == "rzz": qc.rzz(ang, qs[0], qs[1])
            else: raise ValueError(kind)
    for beta in range(d):
        apply(op_list(b.cw, b.theta_w[beta]))                 # memory rotation (acts on model qubits 0..n_m-1)
        if beta > 0:
            for q in range(n_w): qc.reset(n_m + q)
        apply(op_list(b.cv, b.theta_v[beta]))                 # joint unitary on all qubits
        for q in range(n_w): qc.measure(n_m + q, beta * n_w + q)
    return qc

def counts_from_bitstrings(model, bitstrings, steps=None):
    """(shots, d) integer counts from Qiskit bitstrings (one string per shot, as returned by
    get_memory / BitArray.get_bitstrings). Bit index i of a string is character [-1 - i]."""
    b = _base(model)
    n_w, d = b.n_w, b.d if steps is None else int(steps)
    C = np.zeros((len(bitstrings), d), dtype=np.int64)
    for s_i, s in enumerate(bitstrings):
        for beta in range(d):
            c = 0
            for q in range(n_w):
                bit = int(s[-1 - (beta * n_w + q)])
                c = (c << 1) | bit                               # first work qubit is the MSB of the count
            C[s_i, beta] = c
    return C

def samples_from_counts(model, counts, rng):
    """Device counts -> showers, through the learned readout and the decoder (same code path as model.sample)."""
    b = _base(model)
    u = rng.random(counts.shape)
    y = np.empty(counts.shape, dtype=float)
    for beta in range(counts.shape[1]):
        y[:, beta] = b.apply_readout(beta, counts[:, beta], u[:, beta])
    k = counts.shape[1]
    if hasattr(model, "base") and k == b.d:                  # LinearDecoded, full run: y = m + s (Z A^T + b), Z standardized
        Z = (y - b.m) / b.s
        return model.m + model.s * (Z @ model.A.T + model.b)
    if hasattr(model, "base"):                               # partial run: the decoder needs all d coordinates; return the
        return (y - b.m[:k]) / b.s[:k]                       # standardized readout values (PC scores) of the generated ones
    return y

# ------------------------------------------------------------------------------------------- verification
def _unitary_from_ops(ops, n):
    U = np.eye(2 ** n, dtype=complex)
    for kind, qs, ang in ops:
        if kind in ("ry", "rz"):
            g = ry(ang) if kind == "ry" else rz(ang)
            q = qs[0]
            G = np.kron(np.kron(np.eye(2 ** q), g), np.eye(2 ** (n - q - 1)))
        elif kind == "cz":
            a, c = qs
            idx = np.arange(2 ** n)
            ba = (idx >> (n - 1 - a)) & 1
            bc = (idx >> (n - 1 - c)) & 1
            G = np.diag(np.where(ba & bc, -1.0, 1.0)).astype(complex)
        else:
            raise ValueError(kind)
        U = G @ U
    return U

def verify_numpy(model):
    """The gate list rebuilt as a unitary must equal the model's own unitary for every coordinate."""
    b = _base(model)
    err = 0.0
    for beta in range(b.d):
        err = max(err, np.abs(_unitary_from_ops(op_list(b.cv, b.theta_v[beta]), b.n_m + b.n_w) - b.cv.unitary(b.theta_v[beta])).max())
        err = max(err, np.abs(_unitary_from_ops(op_list(b.cw, b.theta_w[beta]), b.n_m) - b.cw.unitary(b.theta_w[beta])).max())
    return err

def exact_first_step_counts(model):
    """p(c_1) from the unitaries directly, for the Aer check."""
    b = _base(model)
    n_m, n_w = b.n_m, b.n_w
    n = n_m + n_w
    W = np.kron(b.cw.unitary(b.theta_w[0]), np.eye(2 ** n_w))
    V = b.cv.unitary(b.theta_v[0])
    psi = (V @ W)[:, 0]                                       # |0...0> in
    p = np.abs(psi) ** 2
    idx = np.arange(2 ** n)
    c = idx & ((1 << n_w) - 1)   # work bits are the low bits (big-endian, work last)
    return np.bincount(c, weights=p, minlength=2 ** n_w)

def verify_aer(model, shots=20000):
    """Run the first coordinate on the Aer simulator and compare the count histogram with the exact one."""
    from qiskit_aer import AerSimulator
    from qiskit import transpile
    qc = build_circuit(model, steps=1)
    sim = AerSimulator()
    res = sim.run(transpile(qc, sim), shots=shots, memory=True).result()
    C = counts_from_bitstrings(model, res.get_memory(), steps=1)[:, 0]
    emp = np.bincount(C, minlength=2 ** _base(model).n_w) / shots
    p = exact_first_step_counts(model)
    tv = 0.5 * np.abs(emp - p).sum()
    return tv, np.sqrt(len(p) / (2 * np.pi * shots))        # TV and its multinomial floor

# ------------------------------------------------------------------- completing a partial device run with the exact model
def complete_from_device(model, C_dev, rng):
    """Device counts for the first k coordinates -> full count records (n, d).
    The device's outcomes select the memory state exactly (M_{c_k} W_k ... M_{c_1} W_1 |0>, renormalized); the remaining
    coordinates are sampled from that state by the exact model. Coherent register, one round per coordinate, binary current."""
    b = _base(model)
    assert not b.dephase and b.n_sub == 1 and b.current == "binary", "completion implemented for the deployed configuration"
    n, k = C_dev.shape
    chi, A = b.chi, b.A
    psi = np.zeros((n, chi), dtype=complex)
    psi[:, 0] = 1.0
    C = np.zeros((n, b.d), dtype=np.int64)
    C[:, :k] = C_dev
    for beta in range(b.d):
        Vb, _ = b.cv.forward(b.theta_v[beta], b._E)
        Ms = np.stack([Vb[x::A, :] for x in range(A)])      # (A, chi, chi) Kraus operators
        psi = psi @ b.cw.unitary(b.theta_w[beta]).T                                                        # memory rotation
        amps = np.einsum("xij,nj->nxi", Ms, psi)                                                           # (n, A, chi): M_x psi for every outcome
        if beta < k: x = C_dev[:, beta]
        else:
            p = (np.abs(amps) ** 2).sum(2)
            p /= p.sum(1, keepdims=True)
            x = (p.cumsum(1) > rng.random((n, 1))).argmax(1)
            C[:, beta] = x
        psi = amps[np.arange(n), x]
        psi /= np.linalg.norm(psi, axis=1, keepdims=True)
    return C

# ------------------------------------------------------------- deferred measurement: fewer or no mid-circuit measurements
def build_circuit_deferred(model, steps=None, batch=None):
    """Same output distribution as build_circuit, with the work-register measurements deferred.

    CoMB never feeds a count forward, so by the deferred-measurement principle the work qubits of a coordinate can be
    measured at any later time if they are not reused. `batch` coordinates share one measurement layer: each coordinate
    in a batch gets its own fresh work register, and the whole batch is measured (and reset) once at its end.
      batch = 1      -> one measurement layer per coordinate (the original circuit), n_m + n_w qubits
      batch = steps  -> no mid-circuit measurement at all, everything measured at the end, n_m + steps*n_w qubits
    Classical bit beta*n_w + q holds work qubit q of coordinate beta, as in build_circuit, so decoding is unchanged."""
    from qiskit import QuantumCircuit
    b = _base(model)
    n_m, n_w = b.n_m, b.n_w
    d = b.d if steps is None else int(steps)
    batch = d if batch is None else max(1, min(int(batch), d))
    qc = QuantumCircuit(n_m + batch * n_w, d * n_w)
    def apply(ops, qmap):
        for kind, qs, ang in ops:
            qs = [qmap[q] for q in qs]
            if kind == "ry": qc.ry(ang, qs[0])
            elif kind == "rz": qc.rz(ang, qs[0])
            elif kind == "cz": qc.cz(qs[0], qs[1])
            elif kind == "rzz": qc.rzz(ang, qs[0], qs[1])
            else: raise ValueError(kind)
    for beta in range(d):
        slot = beta % batch
        apply(op_list(b.cw, b.theta_w[beta]), {q: q for q in range(n_m)})
        qmap = {q: q for q in range(n_m)}
        qmap.update({n_m + q: n_m + slot * n_w + q for q in range(n_w)})
        apply(op_list(b.cv, b.theta_v[beta]), qmap)
        if slot == batch - 1 or beta == d - 1:                       # end of a batch: measure its work registers
            first = beta - slot
            for g in range(first, beta + 1):
                for q in range(n_w): qc.measure(n_m + (g - first) * n_w + q, g * n_w + q)
            if beta < d - 1:
                for q in range(batch * n_w): qc.reset(n_m + q)
    return qc

def exact_joint_counts(model, steps):
    """P(c_1, ..., c_steps) of the model, from its Kraus operators (reference for the deferred circuit)."""
    b = _base(model)
    n_m, n_w = b.n_m, b.n_w
    chi, K = 2 ** n_m, 2 ** n_w
    rho0 = np.zeros((chi, chi), dtype=complex)
    rho0[0, 0] = 1.0
    states = {(): rho0}
    for beta in range(steps):
        W = b.cw.unitary(b.theta_w[beta])
        V = b.cv.unitary(b.theta_v[beta]).reshape(chi, K, chi, K)
        new = {}
        for hist, rho in states.items():
            rho = W @ rho @ W.conj().T
            for c in range(K):
                M = V[:, c, :, 0]                                    # (I (x) <c|) V (I (x) |0>), memory first, work last
                new[hist + (c,)] = M @ rho @ M.conj().T
        states = new
    return {h: float(np.real(np.trace(r))) for h, r in states.items()}

def deferred_distribution_numpy(model, steps):
    """Joint count distribution of the fully deferred circuit (no mid-circuit measurement), by state-vector simulation
    in numpy on n_m + steps*n_w qubits with the model's own gates; compared with exact_joint_counts by `verify`."""
    b = _base(model)
    n_m, n_w = b.n_m, b.n_w
    n = n_m + steps * n_w
    psi = np.zeros(2 ** n, dtype=complex)
    psi[0] = 1.0
    def apply_unitary(U, qubits):
        nonlocal psi
        k = len(qubits)
        rest = [q for q in range(n) if q not in qubits]
        t = np.transpose(psi.reshape([2] * n), qubits + rest).reshape(2 ** k, -1)
        t = (U @ t).reshape([2] * n)
        psi = np.transpose(t, np.argsort(qubits + rest)).reshape(-1)
    for beta in range(steps):
        apply_unitary(b.cw.unitary(b.theta_w[beta]), list(range(n_m)))
        apply_unitary(b.cv.unitary(b.theta_v[beta]), list(range(n_m)) + [n_m + beta * n_w + q for q in range(n_w)])
    p = (np.abs(psi) ** 2).reshape([2] * n).sum(axis=tuple(range(n_m))).reshape(-1)      # trace out the memory
    out = {}
    for idx, pr in enumerate(p):
        if pr < 1e-15: continue
        bits = format(idx, f"0{steps * n_w}b")                                           # coordinate 1's register first
        cs = tuple(int(bits[beta * n_w:(beta + 1) * n_w], 2) for beta in range(steps))
        out[cs] = out.get(cs, 0.0) + pr
    return out
