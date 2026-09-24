"""
Typed configuration objects for CoMB: small per-subsystem
dataclasses plus named presets.

The d12 preset reproduces the registered experiment reported in the
manuscript: d=12 CLIC showers, b=3 -> B=4
blocks, K=2^{n_work}=8 tokens per block, per-block theta (share_theta
False; the shared-theta ablation is retained as a config switch and
documented as an expressivity failure in the manuscript).
"""

import dataclasses
from dataclasses import dataclass, field
from typing import Optional, Tuple


@dataclass
class DataConfig:
    data_path: str = "cal_shower_img_12q.npy"
    subset_n: int = 6000
    subset_seed: int = 123
    test_size: float = 0.2
    split_seed: int = 42
    val_n: int = 800          # token-level validation split (model select)
    seed: int = 7


@dataclass
class TokenizerConfig:
    block_size: int = 3        # b; B = ceil(d / b)
    n_work: int = 3            # K = 2^{n_work} tokens per block
    detok_mode: str = "causal"       # the paper's emission, Eq. (emission)
    #   neighbor: conditional pixel means from own + adjacent-block tokens,
    #   residuals drawn from their joint covariance. Declared, data-fit,
    #   identical for all models; recovers cross-block residual covariance
    #   that per-block independent sampling discards (oracle corr floor
    #   0.090 -> 0.018). No correlation loss is used.
    shrink: float = 0.10
    clip_nonnegative: bool = True


@dataclass
class ModelConfig:
    n_mem: int = 3             # coherent memory qubits; chi = 2^{n_mem}
    depth: int = 6             # layers per block (L)
    use_prefix: bool = False   # classical token-histogram channel (C1)
    share_theta: bool = False  # True = original-CoMB-style sharing
                               # (documented expressivity failure)


@dataclass
class TrainConfig:
    epochs: int = 120
    batch: int = 1024
    lr: float = 0.08
    lr_warmup: int = 5
    grad_clip: float = 10.0
    seed: int = 7
    warm_start: Optional[str] = None   # path to a theta .npy to resume


@dataclass
class BaselineConfig:
    hmm_chis: Tuple[int, ...] = (2, 4, 8)
    hmm_restarts: int = 8
    hmm_iters: int = 200
    empirical_alpha: float = 0.5


@dataclass
class CoMBConfig:
    data: DataConfig = field(default_factory=DataConfig)
    tokenizer: TokenizerConfig = field(default_factory=TokenizerConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    baselines: BaselineConfig = field(default_factory=BaselineConfig)
    run_tag: str = "d12"


def d12_config() -> CoMBConfig:
    """Registered experiment: d=12, b=3 -> B=4, n_work=3 (K=8),
    n_mem=3 (chi=8), depth 6, per-block theta."""
    return CoMBConfig()
