"""
Parameterized qubit circuits as dense matrices, with an exact reverse-mode
gradient (vector-Jacobian product).

The circuits used by the coherent memory register act on at most a handful of
qubits (memory + work qubits), so every gate is a dense 2^n x 2^n matrix and a
circuit is applied to a block of input columns.  Qubit 0 is the most
significant bit of the basis index.

A circuit of L layers is: per layer RY(theta) and RZ(theta) on every qubit,
followed by CZ on a fixed list of edges (and, optionally, trainable RZZ on the
same edges instead of CZ).  Every trainable gate is exp(-i theta P / 2) with a
Pauli generator P, so d/dtheta = (-i/2) P G.

Reverse mode.  For V = G_P ... G_1 E (E: the input columns) and a real loss
with complex gradient G_V = dL/dRe V + i dL/dIm V, the parameter gradient is

    dL/dtheta_j = (1/2) Im sum( conj(Y_j) * (P_j B_j) ),

where B_j = G_j ... G_1 E (the columns after gate j) and Y_j = Q_j^dag G_V with
Q_j = G_P ... G_{j+1}; Y_j is built by applying the inverse gates from the end.
Checked against finite differences in tests.
"""
from __future__ import annotations

import numpy as np

_I2 = np.eye(2, dtype=np.complex128)
_Y = np.array([[0, -1j], [1j, 0]], dtype=np.complex128)
_Z = np.array([[1, 0], [0, -1]], dtype=np.complex128)


def _single(n, q, m):
    """Embed a 2x2 matrix m on qubit q of n."""
    return np.kron(np.kron(np.eye(2 ** q), m), np.eye(2 ** (n - q - 1)))


def _zdiag(n, q):
    """Diagonal of Z on qubit q (as a vector of +-1)."""
    idx = np.arange(2 ** n)
    return 1.0 - 2.0 * ((idx >> (n - 1 - q)) & 1)


def ry(theta):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.array([[c, -s], [s, c]], dtype=np.complex128)


def rz(theta):
    return np.array([[np.exp(-0.5j * theta), 0], [0, np.exp(0.5j * theta)]], dtype=np.complex128)


class Circuit:
    """L layers of (RY, RZ on every qubit) + entangler on `edges`.

    entangler: "cz" (fixed CZ gates) or "rzz" (trainable RZZ(theta) gates).
    """

    def __init__(self, n: int, L: int, edges, entangler: str = "cz"):
        self.n, self.L, self.edges, self.ent = int(n), int(L), list(edges), entangler
        self.D = 2 ** self.n
        ops = []                       # (kind, qubits, param index or None)
        p = 0
        for _ in range(self.L):
            for q in range(self.n):
                ops.append(("ry", (q,), p))
                p += 1
                ops.append(("rz", (q,), p))
                p += 1
            for (a, b) in self.edges:
                if entangler == "cz":
                    ops.append(("cz", (a, b), None))
                else:
                    ops.append(("rzz", (a, b), p))
                    p += 1
        self.ops, self.n_params = ops, p
        # fixed generator data
        self._zd = [_zdiag(self.n, q) for q in range(self.n)]
        self._Yfull = [_single(self.n, q, _Y) for q in range(self.n)]

    # ------------------------------------------------------------------
    def _gate(self, op, theta):
        kind, qs, pidx = op
        if kind == "ry":
            return ("1q", (qs[0], ry(theta[pidx])))
        if kind == "rz":
            zd = self._zd[qs[0]]
            return ("diag", np.exp(-0.5j * theta[pidx] * zd))
        if kind == "cz":
            a, b = qs
            za, zb = self._zd[a], self._zd[b]
            return ("diag", np.where((za < 0) & (zb < 0), -1.0, 1.0).astype(np.complex128))
        if kind == "rzz":
            a, b = qs
            return ("diag", np.exp(-0.5j * theta[pidx] * self._zd[a] * self._zd[b]))
        raise ValueError(kind)

    def _apply(self, g, B, adjoint=False):
        typ, M = g
        if typ == "diag":
            return (M.conj() if adjoint else M)[:, None] * B
        if typ == "1q":
            q, G = M
            return self._apply_1q(G.conj().T if adjoint else G, q, B)
        return (M.conj().T if adjoint else M) @ B

    def _apply_1q(self, G, q, B):
        """(1 (x) G_q (x) 1) B without forming the D x D matrix."""
        k = B.shape[1]
        L, R = 2 ** q, 2 ** (self.n - q - 1)
        Bt = B.reshape(L, 2, R, k)
        return np.einsum("ab,lbrk->lark", G, Bt).reshape(self.D, k)

    def _gen_apply(self, op, B):
        """P_j B for the generator of a trainable gate."""
        kind, qs, _ = op
        if kind == "ry":
            return self._apply_1q(_Y, qs[0], B)
        if kind == "rz":
            return self._zd[qs[0]][:, None] * B
        if kind == "rzz":
            a, b = qs
            return (self._zd[a] * self._zd[b])[:, None] * B
        raise ValueError(kind)

    # ------------------------------------------------------------------
    def forward(self, theta, E):
        """V = U(theta) E for an input block E (D x k).  Returns V and a tape."""
        B = np.asarray(E, dtype=np.complex128)
        tape = []
        for op in self.ops:
            g = self._gate(op, theta)
            B = self._apply(g, B)
            tape.append((op, g, B))
        return B, tape

    def vjp(self, tape, G_V):
        """dL/dtheta from G_V = dL/dRe V + i dL/dIm V."""
        grad = np.zeros(self.n_params)
        Y = np.asarray(G_V, dtype=np.complex128)
        for op, g, B in reversed(tape):
            if op[2] is not None:
                grad[op[2]] = 0.5 * np.imag(np.sum(Y.conj() * self._gen_apply(op, B)))
            Y = self._apply(g, Y, adjoint=True)
        return grad

    def unitary(self, theta):
        V, _ = self.forward(theta, np.eye(self.D))
        return V


def interleaved_edges(n_m: int, n_w: int):
    """Line over [m0, a0, m1, a1, ...] (memory qubits 0..n_m-1, work qubits n_m..):
    every edge couples memory to work qubit where possible; heavy-hex native."""
    order = []
    for k in range(max(n_m, n_w)):
        if k < n_m:
            order.append(k)
        if k < n_w:
            order.append(n_m + k)
    return [(order[i], order[i + 1]) for i in range(len(order) - 1)]


def line_edges(n: int):
    return [(i, i + 1) for i in range(n - 1)]


class BatchedCircuit:
    """The gates of a Circuit applied to d parameter sets at once (one per round): forward(thetas (d, P), E)
    returns V (d, D, k); vjp(tape, G_V (d, D, k)) returns dL/dthetas (d, P).  Same conventions as Circuit."""

    def __init__(self, circuit):
        self.c, self.n, self.D = circuit, circuit.n, circuit.D

    def _apply_1q(self, G, q, B):
        d, D, k = B.shape
        L, R = 2 ** q, 2 ** (self.n - q - 1)
        return np.einsum("nab,nlbrk->nlark", G, B.reshape(d, L, 2, R, k)).reshape(d, D, k)

    def forward(self, thetas, E):
        thetas = np.asarray(thetas, dtype=np.float64)
        d = thetas.shape[0]
        B = np.repeat(np.asarray(E, dtype=np.complex128)[None], d, axis=0)
        tape = []
        for op in self.c.ops:
            kind, qs, p = op
            if kind == "ry":
                h = thetas[:, p] / 2
                G = np.zeros((d, 2, 2), dtype=np.complex128)
                G[:, 0, 0] = np.cos(h)
                G[:, 0, 1] = -np.sin(h)
                G[:, 1, 0] = np.sin(h)
                G[:, 1, 1] = np.cos(h)
                B = self._apply_1q(G, qs[0], B)
                tape.append((op, ("1q", qs[0], G), B))
            else:
                if kind == "rz":
                    diag = np.exp(-0.5j * thetas[:, p][:, None] * self.c._zd[qs[0]][None, :])
                elif kind == "cz":
                    za, zb = self.c._zd[qs[0]], self.c._zd[qs[1]]
                    diag = np.where((za < 0) & (zb < 0), -1.0, 1.0).astype(np.complex128)[None, :]
                elif kind == "rzz":
                    diag = np.exp(-0.5j * thetas[:, p][:, None] * (self.c._zd[qs[0]] * self.c._zd[qs[1]])[None, :])
                else:
                    raise ValueError(kind)
                B = diag[:, :, None] * B
                tape.append((op, ("diag", diag), B))
        return B, tape

    def vjp(self, tape, G_V):
        Y = np.asarray(G_V, dtype=np.complex128)
        d = Y.shape[0]
        grad = np.zeros((d, self.c.n_params))
        Ymat = np.repeat(_Y[None], d, axis=0)
        for op, g, B in reversed(tape):
            kind, qs, p = op
            if p is not None:
                if kind == "ry":
                    gen = self._apply_1q(Ymat, qs[0], B)
                elif kind == "rz":
                    gen = self.c._zd[qs[0]][None, :, None] * B
                else:
                    gen = (self.c._zd[qs[0]] * self.c._zd[qs[1]])[None, :, None] * B
                grad[:, p] = 0.5 * np.imag(np.sum(Y.conj() * gen, axis=(1, 2)))
            if g[0] == "1q":
                Y = self._apply_1q(np.conj(np.swapaxes(g[2], 1, 2)), g[1], Y)
            else:
                Y = np.conj(g[1])[:, :, None] * Y
        return grad
