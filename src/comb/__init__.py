"""

# Public API. Everything a script needs should be importable straight
# from `comb`; if you find yourself reaching into a submodule, that is a
# sign the name belongs here instead.
CoMB: sequential Born machine with coherent quantum memory.

Successor architecture to CoMB [Slim, Monaco, Rehm, Kruecker, Borras]
in which the quantum register is used as MEMORY (n_mem unmeasured
qubits carrying inter-block conditioning coherently) and generation
proceeds by Born SAMPLING of a work register (mid-circuit measurement +
reset per block), rather than by decoding Pauli expectation values with
a classical regressor. See the accompanying paper for the full
formulation, the exact-gradient theory, the registered d=12 experiment,
and the detailed comparison with the original CoMB.

"""

from .config import (CoMBConfig, DataConfig, TokenizerConfig,
                     ModelConfig, TrainConfig, BaselineConfig,
                     d12_config)
from .engine import (init_state, apply_ry, apply_rz, apply_cz,
                     work_probs, project_and_reset, sample_outcomes)
from .model import CoMBQSpec, SequentialBornModel
from .tokenizer import BlockTokenizer
from .baselines import (IndependentBlocks, EmpiricalJoint,
                        InhomogeneousHMM)
from .train import train_blockscore, AdamOpt
from .mmd import (TokenKernel, ExactMMD, empirical_hist, sampled_mmd_grad, reinforce_mmd_grad,
                  enumerate_sequences,
                  block_grams)
from .evalx import (corr_nan_safe, corr_errors, w1_mean, nn_dist,
                    pixel_report)

__all__ = [
    "CoMBConfig", "DataConfig", "TokenizerConfig", "ModelConfig",
    "TrainConfig", "BaselineConfig", "d12_config",
    "init_state", "apply_ry", "apply_rz", "apply_cz", "work_probs",
    "project_and_reset", "sample_outcomes",
    "CoMBQSpec", "SequentialBornModel", "BlockTokenizer",
    "IndependentBlocks", "EmpiricalJoint", "InhomogeneousHMM",
    "train_blockscore", "AdamOpt",
    "TokenKernel", "ExactMMD", "empirical_hist", "sampled_mmd_grad", "reinforce_mmd_grad",
    "enumerate_sequences", "block_grams",
    "corr_nan_safe", "corr_errors", "w1_mean", "nn_dist",
    "pixel_report",
]

