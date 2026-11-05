"""Measurement-tree CoMB: one work qubit, log2(K) binary sub-rounds per token, classical feed-forward.

Each round t first rotates the memory with W_t (a circuit on the n_m memory qubits) and then emits the
n = log2(K) bits of its token one at a time. The node that emits bit l is chosen by the bits already emitted
in this round (classical feed-forward). It is a circuit U_{t,l,p} on the memory and one work qubit prepared
in |0>, followed by a measurement of that qubit and a reset. On the memory the node acts with the Kraus pair

    N_{t,l,p,b} = (1 (x) <b|) U_{t,l,p} (1 (x) |0>),      b = 0, 1,

so token o = (b_1 ... b_n) has the Kraus operator

    M_{t,o} = N_{t,n-1,p_{n-1},b_n} ... N_{t,0,(),b_1} W_t,      Lambda_{t,o}(rho) = M rho M^dag.

With unrestricted nodes this is every coherent-memory generator with K = 2^n tokens per round: compile_tree
builds the nodes of any such generator exactly (STROKES_METHOD.md, Theorem 5). With shallow node circuits it
is the NISQ version: per token, n circuits on n_m + 1 qubits are executed, instead of one circuit on
n_m + n qubits.

Dephased twin (dephase=True): the memory is dephased after W_t and after every node, which gives a classical
chi-state chain with transition-emission matrices |N|^2 and |W|^2 and the same parameters.

Training stays exact: the instrument of a round is a linear map, so comb/kpoint.py and comb/pencf.py apply
unchanged through instruments_all / instruments_all_vjp, which are batched over rounds and nodes.
Package conventions: vec(rho) row-major, Lambda as kron(M, conj M), complex gradients G = dL/dRe + i dL/dIm.
"""
import numpy as np

from .circuit import Circuit, BatchedCircuit, line_edges
from .model import CountChain, _sample_columns


def node_edges(n_m):
    """CZ chain with the work qubit (index n_m) between the first two memory qubits: m0 - w - m1 - m2 ..."""
    w = n_m
    edges = [(0, w)]
    if n_m >= 2:
        edges.append((w, 1))
        edges += [(i, i + 1) for i in range(1, n_m - 1)]
    return edges


# ------------------------------------------------------------------------------------------ Theorem 5
def _psd_pow(E, p, tol=1e-12):
    w, U = np.linalg.eigh((E + E.conj().T) / 2)
    return (U * np.where(w > tol, np.maximum(w, tol) ** p, 0.0)) @ U.conj().T


def compile_tree(M):
    """Binary-tree nodes of a pure instrument.  M: (K, chi, chi) Kraus operators with sum_o M_o^dag M_o = 1,
    K = 2^n.  Returns N of shape (K - 1, 2, chi, chi) in heap order (node (l, p) at index 2^l - 1 + p) with
    compose_tree(N) == M.  Construction: E_p = sum over tokens starting with p of M_o^dag M_o;
    N_{p,b} = V E_{pb}^{1/2} E_p^{-1/2}, with V = 1 at inner nodes and V the polar unitary of M_o at the leaves;
    the product along a path telescopes to V_o |M_o| = M_o, and every node pair stacks to an isometry."""
    K, c, _ = M.shape
    n = int(round(np.log2(K)))
    assert 2 ** n == K, "K must be a power of two"
    gram = np.einsum("oji,ojk->oik", M.conj(), M)                                   # M_o^dag M_o
    E = [gram.reshape(2 ** l, K // 2 ** l, c, c).sum(1) for l in range(n + 1)]     # E[l][p]
    V = np.empty_like(M)
    for o in range(K):
        Uu, _, Wh = np.linalg.svd(M[o])
        V[o] = Uu @ Wh
    N = np.zeros((K - 1, 2, c, c), dtype=np.complex128)
    for l in range(n):
        for p in range(2 ** l):
            inv = _psd_pow(E[l][p], -0.5)
            for b in (0, 1):
                half = _psd_pow(E[l + 1][2 * p + b], 0.5)
                N[2 ** l - 1 + p, b] = (V[2 * p + b] if l == n - 1 else np.eye(c)) @ half @ inv
    return N


def compose_tree(N, W=None):
    """Kraus operators M (K, chi, chi) of the tree with nodes N (K - 1, 2, chi, chi) after the rotation W."""
    nn, _, c, _ = N.shape
    K = nn + 1
    n = int(round(np.log2(K)))
    P = (np.eye(c, dtype=np.complex128) if W is None else W)[None]
    for l in range(n):
        lo = 2 ** l - 1
        P = np.matmul(N[lo:lo + 2 ** l], P[:, None]).reshape(2 ** (l + 1), c, c)
    return P


# ------------------------------------------------------------------------------------------ model
class TreeCoMB(CountChain):
    inst_names = ("theta_n", "theta_w")

    def __init__(self, d, n_m=2, n_bits=4, L_node=4, L_w=2, seed=0, init_scale=0.6, dephase=False, entangler="cz"):
        self.d, self.n_m, self.n_bits = int(d), int(n_m), int(n_bits)
        self.chi, self.K = 2 ** self.n_m, 2 ** self.n_bits
        self.n_nodes = self.K - 1
        self.dephase = bool(dephase)
        self.cn = Circuit(self.n_m + 1, L_node, node_edges(self.n_m), entangler)
        self.cw = Circuit(self.n_m, L_w, line_edges(self.n_m), entangler) if L_w > 0 else None
        rng = np.random.default_rng(seed)
        self.theta_n = rng.normal(0, init_scale, (self.d, self.n_nodes, self.cn.n_params))
        self.theta_w = rng.normal(0, init_scale, (self.d, self.cw.n_params if self.cw else 0))
        self._init_readout_params()
        c = self.chi
        self._E = np.zeros((2 * c, c), dtype=np.complex128)
        self._E[np.arange(c) * 2, np.arange(c)] = 1.0   # |m>|0_w>
        self._bn = BatchedCircuit(self.cn)
        self._bw = BatchedCircuit(self.cw) if self.cw else None
        if self.dephase:
            self._v0 = np.zeros(c, dtype=np.complex128)
            self._v0[0] = 1.0
            self._tr_idx = np.arange(c)
        else:
            self._v0 = np.zeros(c * c, dtype=np.complex128)
            self._v0[0] = 1.0
            self._tr_idx = np.arange(c) * (c + 1)

    # ------------------------------------------------------------------ circuits
    def nodes(self):
        """Node Kraus pairs N (d, K - 1, 2, chi, chi) and the batched-circuit tape."""
        d, c, nn = self.d, self.chi, self.n_nodes
        S, tape = self._bn.forward(self.theta_n.reshape(d * nn, -1), self._E)       # (d*nn, 2c, c), rows m*2 + b
        return S.reshape(d, nn, c, 2, c).transpose(0, 1, 3, 2, 4), tape

    def rotations(self):
        if self._bw is None:
            return np.repeat(np.eye(self.chi, dtype=np.complex128)[None], self.d, axis=0), None
        return self._bw.forward(self.theta_w, np.eye(self.chi))

    def kraus(self):
        """Kraus operators of every round and token, M (d, K, chi, chi)."""
        N, _ = self.nodes()
        W, _ = self.rotations()
        return self._prefix(W, N)[-1]

    def _prefix(self, W, N):
        d, c = self.d, self.chi
        P = [W[:, None]]
        for l in range(self.n_bits):
            lo = 2 ** l - 1
            P.append(np.matmul(N[:, lo:lo + 2 ** l], P[-1][:, :, None]).reshape(d, 2 ** (l + 1), c, c))
        return P

    # ------------------------------------------------------------------ instruments (batched) and adjoint
    def instruments_all(self, keep_tape=False):
        d, c = self.d, self.chi
        N, tn = self.nodes()
        W, tw = self.rotations()
        if self.dephase:
            T = np.abs(N) ** 2
            P = [(np.abs(W) ** 2)[:, None]]
            for l in range(self.n_bits):
                lo = 2 ** l - 1
                P.append(np.matmul(T[:, lo:lo + 2 ** l], P[-1][:, :, None]).reshape(d, 2 ** (l + 1), c, c))
            lam = P[-1].astype(np.complex128)
            tape = dict(N=N, W=W, T=T, P=P, tn=tn, tw=tw)
        else:
            P = self._prefix(W, N)
            M = P[-1]
            lam = (M[:, :, :, None, :, None] * M.conj()[:, :, None, :, None, :]).reshape(d, self.K, c * c, c * c)
            tape = dict(N=N, W=W, P=P, tn=tn, tw=tw)
        return lam, (tape if keep_tape else None)

    def instruments_all_vjp(self, tape, G_lam):
        d, c, n, K = self.d, self.chi, self.n_bits, self.K
        N, W, P = tape["N"], tape["W"], tape["P"]
        if self.dephase:
            T = tape["T"]
            G = np.real(G_lam)
            G_T = np.zeros_like(T)
            for l in range(n - 1, -1, -1):
                lo = 2 ** l - 1
                Gc = G.reshape(d, 2 ** l, 2, c, c)
                G_T[:, lo:lo + 2 ** l] = np.matmul(Gc, np.swapaxes(P[l], -1, -2)[:, :, None])
                G = np.matmul(np.swapaxes(T[:, lo:lo + 2 ** l], -1, -2), Gc).sum(axis=2)
            G_N = 2.0 * G_T * N
            G_W = 2.0 * G[:, 0] * W
        else:
            M = P[-1]
            G6 = G_lam.reshape(d * K, c, c, c, c)
            Mf = M.reshape(d * K, c * c, 1)              # kron(M, conj M) adjoint
            t1 = np.matmul(G6.transpose(0, 1, 3, 2, 4).reshape(d * K, c * c, c * c), Mf)
            t2 = np.matmul(G6.conj().transpose(0, 2, 4, 1, 3).reshape(d * K, c * c, c * c), Mf)
            G = (t1 + t2).reshape(d, K, c, c)
            G_N = np.zeros_like(N)
            for l in range(n - 1, -1, -1):
                lo = 2 ** l - 1
                Gc = G.reshape(d, 2 ** l, 2, c, c)
                G_N[:, lo:lo + 2 ** l] = np.matmul(Gc, np.conj(np.swapaxes(P[l], -1, -2))[:, :, None])
                G = np.matmul(np.conj(np.swapaxes(N[:, lo:lo + 2 ** l], -1, -2)), Gc).sum(axis=2)
            G_W = G[:, 0]
        G_S = G_N.transpose(0, 1, 3, 2, 4).reshape(d * self.n_nodes, 2 * c, c)
        g_n = self._bn.vjp(tape["tn"], G_S).reshape(self.theta_n.shape)
        g_w = self._bw.vjp(tape["tw"], G_W) if self._bw is not None else np.zeros_like(self.theta_w)
        return [g_n, g_w]

    def pixel_instrument(self, beta, keep_tape=False):
        lam, _ = self.instruments_all(False)
        return lam[beta], None

    def pixel_instrument_vjp(self, beta, tape, G_lam):
        raise NotImplementedError("TreeCoMB is differentiated through instruments_all_vjp")

    # ------------------------------------------------------------------ sampling: the dynamic circuit with feed-forward
    def sample_counts(self, n, rng):
        d, c = self.d, self.chi
        N, _ = self.nodes()
        W, _ = self.rotations()
        counts = np.zeros((n, d), dtype=np.int64)
        rows = np.arange(n)
        if self.dephase:
            T = np.abs(N) ** 2
            state = np.zeros(n, dtype=np.int64)
            for t in range(d):
                state = _sample_columns(np.abs(W[t]) ** 2, state, rng)
                p = np.zeros(n, dtype=np.int64)
                for l in range(self.n_bits):
                    pr = T[t, 2 ** l - 1 + p][rows, :, :, state].reshape(n, 2 * c)               # P(b, j | i)
                    k = np.minimum((rng.random(n)[:, None] > np.cumsum(pr / pr.sum(1, keepdims=True), axis=1)).sum(1), 2 * c - 1)
                    state = k % c
                    p = 2 * p + k // c
                counts[:, t] = p
            return counts
        psi = np.zeros((n, c), dtype=np.complex128)
        psi[:, 0] = 1.0
        for t in range(d):
            psi = psi @ W[t].T
            p = np.zeros(n, dtype=np.int64)
            for l in range(self.n_bits):
                amp = np.einsum("nbij,nj->nbi", N[t, 2 ** l - 1 + p], psi)                       # (n, 2, chi)
                pb = np.real((amp * amp.conj()).sum(-1))
                pb /= pb.sum(1, keepdims=True)
                b = (rng.random(n) > pb[:, 0]).astype(np.int64)
                psi = amp[rows, b]
                psi /= np.linalg.norm(psi, axis=1, keepdims=True)
                p = 2 * p + b
            counts[:, t] = p
        return counts
