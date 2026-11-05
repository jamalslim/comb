"""Handwritten digits drawn as pen strokes: exactly trainable coherent-memory generators (CoMB).

A coherent memory register that is never measured, plus a work register measured and reset once per emitted
token, generates sequences of any length with a register of fixed size. For any statistic whose weight factorises
over rounds, E[prod_t w_t(c_t)] = Tr[A_d ... A_1(rho_0)] with tilted maps A_t = sum_c w_t(c) Lambda_{t,c}. This
gives exact, sample-free losses and exact gradients: the k-point correlation MMD (kpoint.py) and the path MMD of
cumulative positions (pencf.py). Models: the CoMB circuit and its dephased twin, the classical HMM (model.py),
the general coherent memory (hqmm.py), the measurement-tree CoMB with one work qubit and feed-forward
(treecomb.py), and exact noisy versions for deployment studies (noise.py, noisytree.py).
See README.md, STROKES_METHOD.md (method, propositions, proofs) and RUN_PAPER2.md (how to run).
"""
from .circuit import Circuit, BatchedCircuit, interleaved_edges, line_edges
from .model import CoMB, ClassicalHMM, CountChain, kron_grad
from .hqmm import GeneralHQMM
from .treecomb import TreeCoMB, compile_tree, compose_tree

__version__ = "2.0.0"
