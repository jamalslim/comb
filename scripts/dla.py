"""Dynamical Lie algebra of a circuit ansatz, by commutator closure.

The generators of a parameterized circuit span a Lie algebra g.  All
reachable unitaries lie in exp(g), so:

  dim g = poly(n)  ->  expectation values propagate analytically in the
                       adjoint representation (g-sim): no state vector,
                       cost O(dim g^3).  Classically simulable, hence NO
                       quantum advantage.
  dim g = exp(n)   ->  no such analytic propagation exists.

Pauli strings are encoded as (x, z) bit masks.  [P,Q] is nonzero iff they
anticommute, and is then proportional to the product string, so closure
is exact integer arithmetic with no floating point anywhere.
"""
import sys
import numpy as np
from itertools import combinations


def pstr(s):
    """'XIZY' -> (x_mask, z_mask)"""
    x = z = 0
    for i, c in enumerate(reversed(s)):
        if c in "XY": x |= 1 << i
        if c in "ZY": z |= 1 << i
    return (x, z)


def anticommute(p, q):
    """number of qubits where they anticommute, mod 2"""
    x1, z1 = p; x2, z2 = q
    return bin((x1 & z2) ^ (z1 & x2)).count("1") & 1


def prod(p, q):
    return (p[0] ^ q[0], p[1] ^ q[1])


def close(gens, max_dim=200000):
    """Lie closure: repeatedly add [P,Q] for anticommuting pairs."""
    basis = set(gens)
    frontier = list(basis)
    while frontier:
        new = set()
        cur = list(basis)
        for p in frontier:
            for q in cur:
                if anticommute(p, q):
                    r = prod(p, q)
                    if r != (0, 0) and r not in basis:
                        new.add(r)
        if not new:
            break
        basis |= new
        if len(basis) > max_dim:
            return basis, True
        frontier = list(new)
    return basis, False


def comb_generators(n_mem, n_work, entangler="ring"):
    """CoMB block: RZ, RY on every qubit plus a CZ entangler.

    entangler is "ring", "line" or "star". The star couples each work
    qubit to every memory qubit. The paper states the algebra is full for
    all three, so all three are checked below.
    """
    n = n_mem + n_work
    g = []
    for q in range(n):
        for P in "ZY":
            s = ["I"] * n; s[n - 1 - q] = P
            g.append(pstr("".join(s)))
    # CZ(a,b) = exp(i pi/4 (I-Z_a)(I-Z_b)), so its generator is Z_a Z_b.
    if entangler == "star":
        edges = [(n_mem + k, q) for k in range(n_work) for q in range(n_mem)]
    else:
        edges = [(q, q + 1) for q in range(n - 1)]
        if entangler == "ring" and n > 2:
            edges.append((n - 1, 0))
    for a, b in edges:
        s = ["I"] * n; s[n - 1 - a] = "Z"; s[n - 1 - b] = "Z"
        g.append(pstr("".join(s)))
    return g, n


if __name__ == "__main__":
    # Reproduces Eq. (dla) of the paper: the block ansatz generates the
    # full algebra su(2^n_q) for every entangler. Exact integer
    # arithmetic, no floating point. The n_q = 7 rows dominate the runtime.
    print("dynamical Lie algebra of the CoMB block ansatz")
    print(f"  {'entangler':>9} {'n_mem':>6} {'n_work':>7} {'n_q':>4} {'dim g':>8} "
          f"{'4^n_q - 1':>11}   verdict")
    all_full = True
    for ent in ("ring", "line", "star"):
        for n_mem in (1, 2, 3, 4):
            for n_work in (2, 3):
                g, n = comb_generators(n_mem, n_work, entangler=ent)
                b, _ = close(g, max_dim=200000)
                full = 4 ** n - 1
                ok = len(b) == full          # exact: the closure excludes the identity
                all_full &= ok
                v = "FULL su(2^n_q)" if ok else "sub-algebra"
                print(f"  {ent:>9} {n_mem:>6} {n_work:>7} {n:>4} {len(b):>8} {full:>11}   {v}")
        print()
    print("  polynomial dim g -> analytic propagation (g-sim), no state vector,")
    print("                      so no COMPUTATIONAL advantage is possible.")
    print("  exponential      -> no analytic shortcut; the state vector is needed.")
    sys.exit(0 if all_full else 1)
