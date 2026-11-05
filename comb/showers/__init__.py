"""Calorimeter showers (CaloChallenge Dataset 1): the CoMB circuit with a learned count readout, its dephased twin,
the classical hidden Markov model and the general coherent memory, with exact characteristic-function MMD training."""
from .circuit import Circuit, interleaved_edges, line_edges
from .model import CoMB, ClassicalHMM, CountChain, kron_grad

