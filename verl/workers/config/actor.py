# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
from typing import Any, Optional

from omegaconf import MISSING

from verl.base_config import BaseConfig
from verl.trainer.config import CheckpointConfig
from verl.utils.profiler.config import ProfilerConfig

from .engine import FSDPEngineConfig, McoreEngineConfig
from .model import HFModelConfig
from .optimizer import OptimizerConfig

__all__ = ["PolicyLossConfig", "ActorConfig", "FSDPActorConfig", "McoreActorConfig"]


@dataclass
class PolicyLossConfig(BaseConfig):
    """Configuration for policy loss computation.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        loss_mode (str): Loss function mode. Options: 'vanilla', 'clip-cov', 'kl-cov', 'gpg'.
        clip_cov_ratio (float): Ratio of tokens to be clipped for clip-cov loss.
        clip_cov_lb (float): Lower bound for clip-cov loss.
        clip_cov_ub (float): Upper bound for clip-cov loss.
        kl_cov_ratio (float): Ratio of tokens to be applied KL penalty for kl-cov loss.
        ppo_kl_coef (float): KL divergence penalty coefficient.
        opd_loss_max_clamp (Optional[float]): For OPD loss only. If set, clamps per-token advantage
            (teacher_logp - student_logp) to [-v, v] before the PPO step, analogous to verl's
            official `DistillationLossConfig.loss_max_clamp`. None disables the clamp.
        opd_credit_rule (str): For OPD loss only. Chunk-level credit-assignment rule.
            'semantic' (default) = paper Eq.7-9, target_i = (L_T/L_S) * log p_i.
            'uniform' = target_i = L_T / n_tokens_in_chunk (P0-1 ablation).
            Both preserve chunk-level log-prob conservation.
        opd_adv_mode (str): For OPD loss only. Phase-1/2 magnitude ablation on the
            per-token advantage A_i = sign(Δ_i) * |Δ_i| after chunk-credit assignment
            and optional clamp. Options:
              * 'raw' (default) — identity; standard OPD.
              * 'perm' — shuffle |A_i| within each per-sign group; per-sign Σ|A|
                preserved and |A| histogram preserved, but token↔magnitude pairing
                broken. If Raw beats Perm, per-token magnitude carries real signal.
              * 'group_const' — replace |A_i| with per-sign-group mean |Δ|; per-sign
                Σ|A| preserved, within-group variance flattened. Equivalent to
                (M=M_raw, q=q_uniform).
              * 'sign' — |A_i| = 1; drops magnitude entirely. Equivalent to the
                removed ``opd_advantage_sign_only`` flag. NB: not scale-matched.
              * 'power' — coupled shaping A_i = sign(Δ_i) * |Δ_i|^α with α =
                ``opd_adv_power_alpha``. α=0 ⇔ 'sign', α=1 ⇔ 'raw'. Both group
                strength AND within-group allocation drift with α.
              * 'sign_mass_raw_alloc' — Phase-2 (2×2 factorial cell): M_g = c·n_g
                (count-based, globally scale-matched), q_i = q_i^raw. Isolates the
                effect of replacing raw group strength with count-based strength
                while keeping raw allocation.
              * 'alloc_power' — Phase-2 allocation-only sweep: M_g = S_g^raw
                (exact), q_i^(α) = |Δ_i|^α / Σ_{sign_group} |Δ_j|^α. Uses
                ``opd_adv_alloc_alpha``. α=1 ⇔ 'raw', α=0 ⇔ 'group_const'.
              * 'strength_interp' — Phase-2 strength-only sweep: q = q_uniform,
                M_+/M_- interpolates between raw pos-mass fraction (β=1) and
                count-based (β=0) at fixed total strength. Uses
                ``opd_adv_strength_beta``.
              * 'dctv' — Adaptive Distance-Calibrated TVOPD. A_i = c_t · sign(Δ_i)
                where c_t = min(c_cap, D̄_{t-1} / (M̄_{t-1} + ε)) is a single
                deterministic global scalar computed from the previous step's
                EMAs of (a) the sequence-level TV distance estimator
                D̂_t = mean_i [1 - exp(Δ_i)]_+ and (b) the predicted one-step
                TV motion M̂_t = 0.5 · η_t · (G_t^TV)^2 where G_t^TV is the
                pre-clip OPD gradient norm rescaled by 1/c_t. Because c_t is
                conditioned on the past, E[ĝ_t^adaptive | F_{t-1}] = c_t · g_t^TV
                — TV descent direction is preserved in expectation. Uses
                ``opd_dctv_beta_d``, ``opd_dctv_beta_m``, ``opd_dctv_eps``,
                ``opd_dctv_c_cap``.
            Sentinel tokens have Δ=0 → sign(0)=0 → no gradient contribution in any
            mode. See :func:`verl.trainer.ppo.core_algos._apply_adv_transform`.
        opd_adv_power_alpha (float): For OPD loss only, when opd_adv_mode='power'.
            Exponent α ≥ 0 applied to |Δ_i|. Ignored for other modes. Default 1.0
            (equivalent to raw when combined with mode='power').
        opd_adv_alloc_alpha (float): For OPD loss only, when opd_adv_mode=
            'alloc_power'. Exponent α ≥ 0 controlling within-sign-group allocation
            concentration: q_i^(α) = |Δ_i|^α / Σ_{sign_group} |Δ_j|^α. α=1 recovers
            raw allocation (identity endpoint); α=0 recovers uniform allocation
            (≡ 'group_const'). Ignored for other modes. Default 1.0.
        opd_adv_strength_beta (float): For OPD loss only, when opd_adv_mode=
            'strength_interp'. Interpolation weight β on the pos-mass fraction:
            r_β = β·r_raw + (1-β)·r_sign. β=1 ⇔ scale-matched GroupConst,
            β=0 ⇔ scale-matched Sign, β=0.5 is the mid-point. Ignored for other
            modes. Default 1.0.
        opd_dctv_beta_d (float): DCTV EMA smoothing on D̂_t. Default 0.95.
            Positioned as an estimator smoothing constant, not a method hparam.
            Ignored unless ``opd_adv_mode='dctv'``.
        opd_dctv_beta_m (float): DCTV EMA smoothing on M̂_t. Default 0.95.
            Estimator smoothing constant. Ignored unless ``opd_adv_mode='dctv'``.
        opd_dctv_eps (float): DCTV numerical floor in c_t = min(c_cap,
            D̄/(M̄+ε)) and in the G_TV = G_scaled / max(c, ε) recovery. Default
            1e-8. Ignored unless ``opd_adv_mode='dctv'``.
        opd_dctv_c_cap (float): DCTV upper bound on c_t. Default 1.0 (the natural
            "don't overshoot pure-TV" cap). Values > 1 allow the controller to
            *amplify* pure-TV when the estimator predicts under-shoot; > 1 is
            escape-hatch only and not the default.
        alm_binarization_temp (float): For ALM loss only. Divide chunk log-p by this temperature
            before the Bernoulli BCE. Default 1.0.
        alm_diff_fn (str): For ALM loss only. Elementwise divergence between chunk Bernoullis.
            Phase 1 supports only 'binary_ce'.
        alm_numerator (str): For ALM loss only. Per-chunk weight in aggregation.
            'chunk_count' (default), 'token_count' or 'log1p_token_count'.
        alm_denominator (str): For ALM loss only. Phase 1 supports only 'chunk_count'.
        alm_chunk_clamp (Optional[float]): For ALM loss only. Clamp chunk log-p to [-C, 0]
            before BCE. None disables.
    """

    loss_mode: str = "vanilla"
    clip_cov_ratio: float = 0.0002
    clip_cov_lb: float = 1.0
    clip_cov_ub: float = 5.0
    kl_cov_ratio: float = 0.0002
    ppo_kl_coef: float = 0.1
    opd_loss_max_clamp: Optional[float] = None
    opd_credit_rule: str = "semantic"
    opd_adv_mode: str = "raw"
    opd_adv_power_alpha: float = 1.0
    opd_adv_alloc_alpha: float = 1.0
    opd_adv_strength_beta: float = 1.0
    opd_dctv_beta_d: float = 0.95
    opd_dctv_beta_m: float = 0.95
    opd_dctv_eps: float = 1e-8
    opd_dctv_c_cap: float = 1.0
    alm_binarization_temp: float = 1.0
    alm_diff_fn: str = "binary_ce"
    alm_numerator: str = "chunk_count"
    alm_denominator: str = "chunk_count"
    alm_chunk_clamp: Optional[float] = None


@dataclass
class ActorConfig(BaseConfig):
    """Configuration for actor model training.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy. Must be specified.
        ppo_mini_batch_size (int): Mini-batch size for PPO training.
        ppo_micro_batch_size (Optional[int]): Micro-batch size for PPO training.
            If None, uses ppo_micro_batch_size_per_gpu.
        ppo_micro_batch_size_per_gpu (Optional[int]): Micro-batch size per GPU for PPO training.
        use_dynamic_bsz (bool): Whether to use dynamic batch sizing.
        ppo_max_token_len_per_gpu (int): Maximum token length per GPU for PPO training.
        clip_ratio (float): PPO clipping ratio for policy loss.
        clip_ratio_low (float): Lower bound for PPO clipping ratio.
        clip_ratio_high (float): Upper bound for PPO clipping ratio.
        policy_loss (PolicyLossConfig): Configuration for policy loss computation.
        clip_ratio_c (float): Clipping ratio for critic loss.
        loss_agg_mode (str): Loss aggregation mode. Options: 'token-mean', 'sample-mean'.
        loss_scale_factor (Optional[int]): Scale factor for 'seq-mean-token-sum-norm' loss aggregation mode.
            If None, uses response_length. Set to a constant to ensure consistent normalization.
        entropy_coeff (float): Entropy coefficient for regularization.
        use_kl_loss (bool): Whether to use KL divergence loss.
        use_torch_compile (bool): Whether to use torch.compile for optimization.
        kl_loss_coef (float): KL divergence loss coefficient.
        kl_loss_type (str): Type of KL loss to use.
        ppo_epochs (int): Number of PPO epochs per training step.
        shuffle (bool): Whether to shuffle data during training.
        checkpoint (CheckpointConfig): Configuration for checkpointing.
        optim (OptimizerConfig): Configuration for optimizer.
        use_fused_kernels (bool): Whether to use custom fused kernels (e.g., FlashAttention, fused MLP).
        data_loader_seed (int): Seed for data loader. If None, uses global seed.
    """

    _mutable_fields = BaseConfig._mutable_fields | {
        "ppo_mini_batch_size",
        "ppo_micro_batch_size",
        "ppo_micro_batch_size_per_gpu",
        "ppo_infer_micro_batch_size_per_gpu",
        "engine",
        "model_config",
    }

    strategy: str = MISSING
    ppo_mini_batch_size: int = 256
    ppo_micro_batch_size: Optional[int] = None  # deprecate
    ppo_micro_batch_size_per_gpu: Optional[int] = None
    ppo_infer_micro_batch_size_per_gpu: Optional[int] = None
    use_dynamic_bsz: bool = False
    ppo_max_token_len_per_gpu: int = 16384
    ppo_infer_max_token_len_per_gpu: int = 16384
    clip_ratio: float = 0.2
    clip_ratio_low: float = 0.2
    clip_ratio_high: float = 0.2
    freeze_vision_tower: bool = False
    policy_loss: PolicyLossConfig = field(default_factory=PolicyLossConfig)
    clip_ratio_c: float = 3.0
    loss_agg_mode: str = "token-mean"
    loss_scale_factor: Optional[int] = None
    entropy_coeff: float = 0
    calculate_entropy: bool = False
    use_kl_loss: bool = False
    use_torch_compile: bool = True
    kl_loss_coef: float = 0.001
    kl_loss_type: str = "low_var_kl"
    ppo_epochs: int = 1
    shuffle: bool = False
    data_loader_seed: int = 1
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    use_fused_kernels: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    engine: BaseConfig = field(default_factory=BaseConfig)
    rollout_n: int = MISSING  # must be override by sampling config
    model_config: HFModelConfig = field(default_factory=BaseConfig)

    # Store global batch info for loss aggregation:
    # dp_size: data parallel size
    # batch_num_tokens: number of valid tokens in global batch
    # global_batch_size: global batch size
    global_batch_info: dict = field(default_factory=dict)

    def __post_init__(self):
        """Validate actor configuration parameters."""
        assert self.strategy != MISSING
        assert self.rollout_n != MISSING
        if not self.use_dynamic_bsz:
            if self.ppo_micro_batch_size is not None and self.ppo_micro_batch_size_per_gpu is not None:
                raise ValueError(
                    "[actor] You have set both 'actor.ppo_micro_batch_size' AND 'actor.ppo_micro_batch_size_per_gpu'. "
                    "Please remove 'actor.ppo_micro_batch_size' because only '*_ppo_micro_batch_size_per_gpu' is "
                    "supported (the former is deprecated)."
                )
            else:
                assert not (self.ppo_micro_batch_size is None and self.ppo_micro_batch_size_per_gpu is None), (
                    "[actor] Please set at least one of 'actor.ppo_micro_batch_size' or "
                    "'actor.ppo_micro_batch_size_per_gpu' if use_dynamic_bsz is not enabled."
                )

        valid_loss_agg_modes = [
            "token-mean",
            "seq-mean-token-sum",
            "seq-mean-token-mean",
            "seq-mean-token-sum-norm",
        ]
        if self.loss_agg_mode not in valid_loss_agg_modes:
            raise ValueError(f"Invalid loss_agg_mode: {self.loss_agg_mode}")

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate actor configuration with runtime parameters."""
        if not self.use_dynamic_bsz:
            if train_batch_size < self.ppo_mini_batch_size:
                raise ValueError(
                    f"train_batch_size ({train_batch_size}) must be >= "
                    f"actor.ppo_mini_batch_size ({self.ppo_mini_batch_size})"
                )

            sp_size = getattr(self, "ulysses_sequence_parallel_size", 1)
            if self.ppo_micro_batch_size is not None:
                if self.ppo_mini_batch_size % self.ppo_micro_batch_size != 0:
                    raise ValueError(
                        f"ppo_mini_batch_size ({self.ppo_mini_batch_size}) must be divisible by "
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size})"
                    )
                if self.ppo_micro_batch_size * sp_size < n_gpus:
                    raise ValueError(
                        f"ppo_micro_batch_size ({self.ppo_micro_batch_size}) * "
                        f"ulysses_sequence_parallel_size ({sp_size}) must be >= n_gpus ({n_gpus})"
                    )

    @staticmethod
    def _check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
        """Validate mutually exclusive micro batch size configuration options."""
        param = "ppo_micro_batch_size"
        param_per_gpu = f"{param}_per_gpu"

        if mbs is None and mbs_per_gpu is None:
            raise ValueError(f"[{name}] Please set at least one of '{name}.{param}' or '{name}.{param_per_gpu}'.")

        if mbs is not None and mbs_per_gpu is not None:
            raise ValueError(
                f"[{name}] You have set both '{name}.{param}' AND '{name}.{param_per_gpu}'. Please remove "
                f"'{name}.{param}' because only '*_{param_per_gpu}' is supported (the former is deprecated)."
            )


@dataclass
class McoreActorConfig(ActorConfig):
    """Configuration for Megatron actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'megatron' for Megatron parallelism.
        load_weight (bool): Whether to load model weights from checkpoint.
        megatron (dict[str, Any]): Configuration for Megatron parallelism settings.
        profile (dict[str, Any]): Configuration for profiling settings.
    """

    strategy: str = "megatron"
    load_weight: bool = True
    megatron: McoreEngineConfig = field(default_factory=McoreEngineConfig)
    profile: dict[str, Any] = field(default_factory=dict)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.megatron


@dataclass
class FSDPActorConfig(ActorConfig):
    """Configuration for FSDP actor models.

    The inheritance from BaseConfig provides omegaconf.DictConfig-like interface for a dataclass config.

    Args:
        strategy (str): Training strategy set to 'fsdp' for Fully Sharded Data Parallel.
        grad_clip (float): Gradient clipping threshold.
        ulysses_sequence_parallel_size (int): [DEPRECATED] Ulysses sequence parallel size for long sequences.
        entropy_from_logits_with_chunking (bool): Whether to compute entropy from logits
            with chunking for memory efficiency.
        entropy_checkpointing (bool): Whether to use gradient checkpointing for entropy computation.
        fsdp_config (dict[str, Any]): Configuration for FSDP settings.
        use_remove_padding (bool): Whether to remove padding tokens in inputs during training
    """

    strategy: str = "fsdp"
    grad_clip: float = 1.0
    ulysses_sequence_parallel_size: int = 1
    entropy_from_logits_with_chunking: bool = False
    entropy_checkpointing: bool = False
    fsdp_config: FSDPEngineConfig = field(default_factory=FSDPEngineConfig)
    use_remove_padding: bool = False
    profiler: ProfilerConfig = field(default_factory=ProfilerConfig)
    use_rollout_log_probs: bool = False

    def __post_init__(self):
        """Validate FSDP actor configuration parameters."""
        super().__post_init__()
        self.engine = self.fsdp_config

        # backward compatibility
        if self.ulysses_sequence_parallel_size > 1:
            self.fsdp_config.ulysses_sequence_parallel_size = self.ulysses_sequence_parallel_size

    def validate(self, n_gpus: int, train_batch_size: int, model_config: dict = None):
        """Validate FSDP actor configuration with runtime parameters."""
        super().validate(n_gpus, train_batch_size, model_config)

        if self.strategy in {"fsdp", "fsdp2"} and self.ulysses_sequence_parallel_size > 1:
            if model_config and not model_config.get("use_remove_padding", False):
                raise ValueError(
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."
                )
