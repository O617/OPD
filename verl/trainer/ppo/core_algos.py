# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2022 The HuggingFace Team. All rights reserved.
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
"""
Core functions to implement PPO algorithms.
The function implemented in this file should be used by trainer with different distributed strategies to
implement PPO-like algorithms.
"""

__all__ = ["register_adv_est", "get_adv_estimator_fn", "AdvantageEstimator"]

import math
from collections import defaultdict
from enum import Enum
from typing import Any, Callable, Optional

import numpy as np
import torch
from omegaconf import DictConfig

import verl.utils.torch_functional as verl_F
from verl.trainer.config import AlgoConfig
from verl.utils import as_torch_index, group_mean_std
from verl.utils.import_utils import deprecated
from verl.workers.config import ActorConfig

PolicyLossFn = Callable[
    [
        torch.Tensor,  # old_log_prob
        torch.Tensor,  # log_prob
        torch.Tensor,  # advantages
        torch.Tensor,  # response_mask
        str,  # loss_agg_mode
        Optional[DictConfig | ActorConfig],  # config
        torch.Tensor | None,  # rollout_log_probs
    ],
    tuple[torch.Tensor, dict[str, Any]],
]

POLICY_LOSS_REGISTRY: dict[str, PolicyLossFn] = {}


def register_policy_loss(name: str) -> Callable[[PolicyLossFn], PolicyLossFn]:
    """Register a policy loss function with the given name.

    Args:
        name (str): The name to register the policy loss function under.

    Returns:
        function: Decorator function that registers the policy loss function.
    """

    def decorator(func: PolicyLossFn) -> PolicyLossFn:
        POLICY_LOSS_REGISTRY[name] = func
        return func

    return decorator


def get_policy_loss_fn(name):
    """Get the policy loss with a given name.

    Args:
        name: `(str)`
            The name of the policy loss.

    Returns:
        `(callable)`: The policy loss function.
    """
    loss_name = name
    if loss_name not in POLICY_LOSS_REGISTRY:
        raise ValueError(
            f"Unsupported loss mode: {loss_name}. Supported modes are: {list(POLICY_LOSS_REGISTRY.keys())}"
        )
    return POLICY_LOSS_REGISTRY[loss_name]


class AdvantageEstimator(str, Enum):
    """Using an enumeration class to avoid spelling errors in adv_estimator.

    Note(haibin.lin): this enum class is immutable after creation. Extending this
    enum for new estimators may not be necessary since users can always just call
    `verl.trainer.ppo.core_algos.register` with string name for a custom advantage
    estimator instead.
    """

    GAE = "gae"
    GRPO = "grpo"
    ON_POLICY_DITILL = "opd"
    REINFORCE_PLUS_PLUS = "reinforce_plus_plus"
    REINFORCE_PLUS_PLUS_BASELINE = "reinforce_plus_plus_baseline"
    REMAX = "remax"
    RLOO = "rloo"
    OPO = "opo"
    GRPO_PASSK = "grpo_passk"
    GPG = "gpg"
    RLOO_VECTORIZED = "rloo_vectorized"
    GRPO_VECTORIZED = "grpo_vectorized"


ADV_ESTIMATOR_REGISTRY: dict[str, Any] = {}


def register_adv_est(name_or_enum: str | AdvantageEstimator) -> Any:
    """Decorator to register a advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    """

    def decorator(fn):
        name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
        if name in ADV_ESTIMATOR_REGISTRY and ADV_ESTIMATOR_REGISTRY[name] != fn:
            raise ValueError(
                f"Adv estimator {name} has already been registered: {ADV_ESTIMATOR_REGISTRY[name]} vs {fn}"
            )
        ADV_ESTIMATOR_REGISTRY[name] = fn
        return fn

    return decorator


def get_adv_estimator_fn(name_or_enum):
    """Get the advantage estimator function with a given name.

    Args:
        name_or_enum: `(str)` or `(AdvantageEstimator)`
            The name or enum of the advantage estimator.

    Returns:
        `(callable)`: The advantage estimator function.
    """
    name = name_or_enum.value if isinstance(name_or_enum, Enum) else name_or_enum
    if name not in ADV_ESTIMATOR_REGISTRY:
        raise ValueError(f"Unknown advantage estimator simply: {name}")
    return ADV_ESTIMATOR_REGISTRY[name]


class AdaptiveKLController:
    """
    Adaptive KL controller described in the paper:
    https://arxiv.org/pdf/1909.08593.pdf
    """

    def __init__(self, init_kl_coef, target_kl, horizon):
        self.value = init_kl_coef
        self.target = target_kl
        self.horizon = horizon

    def update(self, current_kl, n_steps):
        """Update the KL coefficient based on current KL divergence.

        Args:
            current_kl (float): Current KL divergence value.
            n_steps (int): Number of steps taken.
        """
        target = self.target
        proportional_error = np.clip(current_kl / target - 1, -0.2, 0.2)
        mult = 1 + proportional_error * n_steps / self.horizon
        self.value *= mult


class FixedKLController:
    """Fixed KL controller."""

    def __init__(self, kl_coef):
        self.value = kl_coef

    def update(self, current_kl, n_steps):
        """Update method for fixed KL controller (no-op).

        Args:
            current_kl (float): Current KL divergence value (unused).
            n_steps (int): Number of steps taken (unused).
        """
        pass


def get_kl_controller(kl_ctrl):
    """Factory function to create appropriate KL controller based on configuration.

    Args:
        kl_ctrl: Configuration object containing KL controller settings.

    Returns:
        KL controller instance (FixedKLController or AdaptiveKLController).

    Raises:
        NotImplementedError: If controller type is not supported.
        AssertionError: If adaptive controller horizon is not positive.
    """
    if kl_ctrl.type == "fixed":
        return FixedKLController(kl_coef=kl_ctrl.kl_coef)
    elif kl_ctrl.type == "adaptive":
        assert kl_ctrl.horizon > 0, f"horizon must be larger than 0. Got {kl_ctrl.horizon}"
        return AdaptiveKLController(init_kl_coef=kl_ctrl.kl_coef, target_kl=kl_ctrl.target_kl, horizon=kl_ctrl.horizon)
    else:
        raise NotImplementedError


@register_adv_est(AdvantageEstimator.GAE)  # or simply: @register_adv_est("gae")
def compute_gae_advantage_return(
    token_level_rewards: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    gamma: torch.Tensor,
    lam: torch.Tensor,
):
    """Adapted from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        nextvalues = 0
        lastgaelam = 0
        advantages_reversed = []
        gen_len = token_level_rewards.shape[-1]

        for t in reversed(range(gen_len)):
            delta = token_level_rewards[:, t] + gamma * nextvalues - values[:, t]
            lastgaelam_ = delta + gamma * lam * lastgaelam

            # skip values and TD-error on observation tokens
            nextvalues = values[:, t] * response_mask[:, t] + (1 - response_mask[:, t]) * nextvalues
            lastgaelam = lastgaelam_ * response_mask[:, t] + (1 - response_mask[:, t]) * lastgaelam

            advantages_reversed.append(lastgaelam)
        advantages = torch.stack(advantages_reversed[::-1], dim=1)

        returns = advantages + values
        advantages = verl_F.masked_whiten(advantages, response_mask)
    return advantages, returns


@register_adv_est(AdvantageEstimator.ON_POLICY_DITILL)  # or simply: @register_adv_est("opd")
def compute_on_policy_advantage_return(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
):
    """
    
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        values: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length). [EOS] mask. The token after [EOS] have mask zero.
        gamma is `(float)`
            discounted factor used in RL
        lam: `(float)`
            lambda value when computing Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)

    """
    with torch.no_grad():
        advantages = token_level_rewards
        # advantages = verl_F.masked_whiten(advantages, response_mask)

    return advantages, advantages


# NOTE(sgm): this implementation only consider outcome supervision, where the reward is a scalar.
@register_adv_est(AdvantageEstimator.GRPO)  # or simply: @register_adv_est("grpo")
def compute_grpo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for GRPO, operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape is (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape is (bs, response_length)
        index: `(np.ndarray)`
            index array for grouping
        epsilon: `(float)`
            small value to avoid division by zero
        norm_adv_by_std_in_grpo: `(bool)`
            whether to scale the GRPO advantage
        config: `(Optional[AlgoConfig])`
            algorithm configuration object

    Note:
        If norm_adv_by_std_in_grpo is True, the advantage is scaled by the std, as in the original GRPO.
        If False, the advantage is not scaled, as in Dr.GRPO (https://arxiv.org/abs/2503.20783).

    Returns:
        advantages: `(torch.Tensor)`
            shape is (bs, response_length)
        Returns: `(torch.Tensor)`
            shape is (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            if norm_adv_by_std_in_grpo:
                scores[i] = (scores[i] - id2mean[index[i]]) / (id2std[index[i]] + epsilon)
            else:
                scores[i] = scores[i] - id2mean[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.GRPO_VECTORIZED)
def compute_grpo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Vectorized GRPO（outcome-only）:
      For each group g:
      a_i = \\frac{r_i - \\mu_g}{\\sigma_g} (or without dividing by \\sigma_g),
      then broadcast the scalar across the token dimension (multiplied by response_mask).。
    """
    with torch.no_grad():
        scores = token_level_rewards.sum(dim=-1)
        g = as_torch_index(index, device=scores.device)
        mean_g, std_g, _ = group_mean_std(scores, g, eps=epsilon)
        if norm_adv_by_std_in_grpo:
            scalars = (scores - mean_g[g]) / (std_g[g] + epsilon)
        else:
            scalars = scores - mean_g[g]
        advantages = scalars.unsqueeze(-1) * response_mask
        return advantages, advantages


@register_adv_est(AdvantageEstimator.GRPO_PASSK)  # or simply: @register_adv_est("grpo_passk")
def compute_grpo_passk_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    norm_adv_by_std_in_grpo: bool = True,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for Pass@k using a GRPO-style outcome reward formulation.
    Only the best response per group gets a non-zero advantage: r_max - r_second_max.

    Implemented as described in https://arxiv.org/abs/2503.19595.

    Args:
        token_level_rewards: (bs, response_length)
        response_mask: (bs, response_length)
        index: (bs,) → group ID per sample
        epsilon: float for numerical stability
        config: (AlgoConfig) algorithm settings, which contains "norm_adv_by_std_in_grpo"

    Returns:
        advantages: (bs, response_length)
        returns: (bs, response_length)
    """
    assert config is not None
    # if True, normalize advantage by std within group
    norm_adv_by_std_in_grpo = config.get("norm_adv_by_std_in_grpo", True)
    scores = token_level_rewards.sum(dim=-1)  # (bs,)
    advantages = torch.zeros_like(scores)

    id2scores = defaultdict(list)
    id2indices = defaultdict(list)

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            idx = index[i]
            id2scores[idx].append(scores[i])
            id2indices[idx].append(i)

        for idx in id2scores:
            rewards = torch.stack(id2scores[idx])  # (k,)
            if rewards.numel() < 2:
                raise ValueError(
                    f"Pass@k requires at least 2 samples per group. Got {rewards.numel()} for group {idx}."
                )
            topk, topk_idx = torch.topk(rewards, 2)
            r_max, r_second_max = topk[0], topk[1]
            i_max = id2indices[idx][topk_idx[0].item()]
            advantage = r_max - r_second_max
            if norm_adv_by_std_in_grpo:
                std = torch.std(rewards)
                advantage = advantage / (std + epsilon)
            advantages[i_max] = advantage

    advantages = advantages.unsqueeze(-1) * response_mask
    return advantages, advantages


@register_adv_est(
    AdvantageEstimator.REINFORCE_PLUS_PLUS_BASELINE
)  # or simply: @register_adv_est("reinforce_plus_plus_baseline")
def compute_reinforce_plus_plus_baseline_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: torch.Tensor,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RF++-baseline (https://arxiv.org/abs/2501.03262), operating only on Outcome reward
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = token_level_rewards.shape[-1]
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2mean[index[i]]

        scores = scores.unsqueeze(-1).tile([1, response_length]) * response_mask
        scores = verl_F.masked_whiten(scores, response_mask) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO)  # or simply: @register_adv_est("rloo")
def compute_rloo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                id2mean[idx] = torch.mean(torch.stack(id2score[idx]))
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            response_num = len(id2score[index[i]])
            if response_num > 1:
                scores[i] = scores[i] * response_num / (response_num - 1) - id2mean[index[i]] * response_num / (
                    response_num - 1
                )
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.OPO)  # or simply: @register_adv_est("opo")
def compute_opo_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for OPO based on https://arxiv.org/pdf/2505.23585

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    response_length = response_mask.sum(dim=-1)
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2len = defaultdict(list)
    id2bsl = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        for i in range(bsz):
            id2score[index[i]].append(scores[i])
            id2len[index[i]].append(response_length[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2bsl[idx] = torch.tensor(0.0)
            elif len(id2score[idx]) > 1:
                score_tensor = torch.stack(id2score[idx])
                len_tensor = torch.stack(id2len[idx])
                id2bsl[idx] = (len_tensor * score_tensor).sum() / len_tensor.sum()
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = scores[i] - id2bsl[index[i]]
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.REINFORCE_PLUS_PLUS)  # or simply: @register_adv_est("reinforce_plus_plus")
def compute_reinforce_plus_plus_outcome_advantage(
    token_level_rewards: torch.Tensor, response_mask: torch.Tensor, config: Optional[AlgoConfig] = None, **kwargs
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for REINFORCE++.
    This implementation is based on the paper: https://arxiv.org/abs/2501.03262

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    assert config is not None
    gamma = config.gamma
    with torch.no_grad():
        returns = torch.zeros_like(token_level_rewards)
        running_return = 0

        for t in reversed(range(token_level_rewards.shape[1])):
            running_return = token_level_rewards[:, t] + gamma * running_return
            returns[:, t] = running_return
            # Reset after EOS
            running_return = running_return * response_mask[:, t]

        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.REMAX)  # or simply: @register_adv_est("remax")
def compute_remax_outcome_advantage(
    token_level_rewards: torch.Tensor,
    reward_baselines: torch.Tensor,
    response_mask: torch.Tensor,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for ReMax, operating only on Outcome reward
    This implementation is based on the paper: https://arxiv.org/abs/2310.10505
    (with only one scalar reward for each response).

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        reward_baselines: `(torch.Tensor)`
            shape: (bs,)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """

    with torch.no_grad():
        returns = (token_level_rewards * response_mask).flip(dims=[-1]).cumsum(dim=-1).flip(dims=[-1])
        advantages = returns - reward_baselines.unsqueeze(-1) * response_mask

    return advantages, returns


@register_adv_est(AdvantageEstimator.GPG)  # or simply: @register_adv_est("gpg")
def compute_gpg_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    f_norm: float = 1.0,
    alpha: float = 1.0,
    config=None,
    **kwargs,
):
    """
    Compute advantage for GPG, operating only on Outcome reward
    (with only one scalar reward for each response).
    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        index: `(np.ndarray)`
            shape: (bs,)
        epsilon: (float)
        f_norm: (float)
        alpha: (float)
        config: (dict) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    id2score = defaultdict(list)
    id2mean = {}
    id2std = {}

    with torch.no_grad():
        bsz = scores.shape[0]
        m = torch.count_nonzero(scores)
        alpha = bsz / m.clamp(min=1)

        for i in range(bsz):
            id2score[index[i]].append(scores[i])

        for idx in id2score:
            if len(id2score[idx]) == 1:
                id2mean[idx] = torch.tensor(0.0)
                id2std[idx] = torch.tensor(1.0)
            elif len(id2score[idx]) > 1:
                scores_tensor = torch.stack(id2score[idx])
                id2mean[idx] = torch.mean(scores_tensor)
                id2std[idx] = torch.std(scores_tensor)
            else:
                raise ValueError(f"no score in prompt index: {idx}")
        for i in range(bsz):
            scores[i] = alpha * (scores[i] - id2mean[index[i]]) / (f_norm)
        scores = scores.unsqueeze(-1) * response_mask

    return scores, scores


@register_adv_est(AdvantageEstimator.RLOO_VECTORIZED)  # or simply: @register_adv_est("rloo_vectorized")
def compute_rloo_vectorized_outcome_advantage(
    token_level_rewards: torch.Tensor,
    response_mask: torch.Tensor,
    index: np.ndarray,
    epsilon: float = 1e-6,
    config: Optional[AlgoConfig] = None,
    **kwargs,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute advantage for RLOO based on https://arxiv.org/abs/2402.14740

    Args:
        token_level_rewards: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
        config: (AlgoConfig) algorithm config

    Returns:
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        Returns: `(torch.Tensor)`
            shape: (bs, response_length)
    """
    scores = token_level_rewards.sum(dim=-1)

    with torch.no_grad():
        inv = torch.from_numpy(np.unique(index, return_inverse=True)[1]).to(scores.device)

        c = torch.bincount(inv)[inv].to(scores.dtype)
        adv = ((c * scores - torch.bincount(inv, weights=scores)[inv]) / (c - 1).clamp_min(1)) * (c > 1)

        adv = adv.unsqueeze(-1) * response_mask

    return adv, adv


def compute_rewards(token_level_scores, old_log_prob, ref_log_prob, kl_ratio):
    """Compute token-level rewards with KL penalty.

    Args:
        token_level_scores (torch.Tensor): Token-level reward scores.
        old_log_prob (torch.Tensor): Log probabilities from current policy.
        ref_log_prob (torch.Tensor): Log probabilities from reference policy.
        kl_ratio (float): KL penalty coefficient.

    Returns:
        torch.Tensor: Token-level rewards with KL penalty applied.
    """
    kl = old_log_prob - ref_log_prob
    return token_level_scores - kl * kl_ratio


def agg_loss(
    loss_mat: torch.Tensor,
    loss_mask: torch.Tensor,
    loss_agg_mode: str,
    dp_size: int = 1,
    batch_num_tokens: Optional[int] = None,
    global_batch_size: Optional[int] = None,
    loss_scale_factor: Optional[int] = None,
):
    """
    Aggregate the loss across global batch to ensure the loss is invariant to fsdp/megatron parallelism.

    NOTE: The returned loss has different behaviors for different backend:
    - FSDP: the loss is directly used for backward.
    - Megatron: the loss should be scaled by `num_microbatches` and `cp_size` for pp schedule.

    Args:
        loss_mat: micro batch loss matrix, (bs, response_length)
        loss_mask: micro batch loss mask, (bs, response_length)
        loss_agg_mode: method to aggregate the loss matrix into a scalar
        dp_size: data parallel size
        batch_num_tokens: number of valid tokens in global batch
        global_batch_size: global batch size
        loss_scale_factor: scale factor for "seq-mean-token-sum-norm" mode. If None, uses loss_mask.shape[-1].
            Set this to a constant value to ensure consistent normalization throughout training.

    Returns:
        loss: `a scalar torch.Tensor`
            aggregated loss
    """
    if loss_agg_mode == "token-mean":
        if batch_num_tokens is None:
            batch_num_tokens = loss_mask.sum()
        loss = verl_F.masked_sum(loss_mat, loss_mask) / batch_num_tokens * dp_size
    elif loss_agg_mode == "seq-mean-token-sum":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)  # token-sum
        seq_mask = (torch.sum(loss_mask, dim=-1) > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-mean":
        seq_mask = torch.sum(loss_mask, dim=-1)  # per-sequence token count
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1) / (seq_mask + 1e-8)  # token-mean
        seq_mask = (seq_mask > 0).float()  # exclude fully masked sequences
        if global_batch_size is None:
            global_batch_size = seq_mask.sum()
        loss = verl_F.masked_sum(seq_losses, seq_mask) / global_batch_size * dp_size  # seq-mean
    elif loss_agg_mode == "seq-mean-token-sum-norm":
        seq_losses = torch.sum(loss_mat * loss_mask, dim=-1)
        if loss_scale_factor is None:
            loss_scale_factor = loss_mask.shape[-1]
        loss = torch.sum(seq_losses) / loss_scale_factor
    else:
        raise ValueError(f"Invalid loss_agg_mode: {loss_agg_mode}")

    return loss


@deprecated("verl.trainer.ppo.core_algos.compute_policy_loss_vanilla")
def compute_policy_loss(
    old_log_prob,
    log_prob,
    advantages,
    response_mask,
    cliprange=None,
    cliprange_low=None,
    cliprange_high=None,
    clip_ratio_c=3.0,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        clip_ratio_c (float, optional):
            Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
            Defaults to 3.0.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
    """
    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)
    pg_loss = agg_loss(loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

    return pg_loss, pg_clipfrac, ppo_kl, pg_clipfrac_lower


@register_policy_loss("vanilla")  # type: ignore[arg-type]
def compute_policy_loss_vanilla(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for PPO.

    Adapted from
    https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1122

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def _unpack_opd_signal(
    advantages: torch.Tensor, response_length: int
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Unpack teacher log-probs and (optional) chunk ids from the reward-manager payload.

    The OPD reward manager concatenates ``[teacher_logps_padded, chunk_ids_padded]`` along
    ``dim=-1``. This helper handles both packings (with or without chunk ids) and returns:

    - ``teacher_log_probs``: ``[..., T]`` teacher per-token gold log-p, with ``inf`` sentinels
      for special/unaligned tokens.
    - ``opd_chunk_ids``: ``[..., T]`` long tensor with chunk ids per token, ``-1`` for
      sentinels; ``None`` if the payload did not carry chunk ids.
    """
    teacher_log_probs = advantages[..., -response_length:]
    opd_chunk_ids: Optional[torch.Tensor] = None
    if advantages.shape[-1] % 2 == 0:
        split = advantages.shape[-1] // 2
        possible_chunk_ids = advantages[..., split:]
        finite_chunk_ids = possible_chunk_ids[torch.isfinite(possible_chunk_ids)]
        has_chunk_id_half = (
            split >= response_length
            and finite_chunk_ids.numel() > 0
            and bool(((finite_chunk_ids == -1) | (finite_chunk_ids >= 0)).all().item())
            and bool(torch.allclose(finite_chunk_ids, finite_chunk_ids.round()))
        )
        if has_chunk_id_half:
            teacher_log_probs = advantages[..., :split][..., -response_length:]
            opd_chunk_ids = possible_chunk_ids[..., -response_length:].round().long()
    return teacher_log_probs, opd_chunk_ids


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable ``log(1 - exp(x))`` for ``x <= 0``.

    Ported from tokenkit ``training/losses.py``. Uses ``log1p(-exp(x))`` for very negative
    ``x`` and ``log(-expm1(x))`` for ``x`` near zero, following Mächler (2012).
    """
    log_half = math.log(0.5)
    return torch.where(
        x < log_half,
        torch.log1p(-torch.exp(x)),
        torch.log(-torch.expm1(x)),
    )


def _apply_adv_transform(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    mode: str,
    power_alpha: float = 1.0,
    alloc_alpha: float = 1.0,
    strength_beta: float = 1.0,
    dctv_c: float = 1.0,
) -> torch.Tensor:
    """Phase-1/2 magnitude-ablation transform on the per-token OPD advantage.

    ``advantages``/``response_mask`` are ``(B, T)``; padding positions
    (``response_mask == 0``) are always set to 0 in the output regardless of
    mode, so downstream masked reductions stay well-defined.

    All shaping is per-sign-group and computed **within a single microbatch**.
    That is: "the group" refers to the union of positive-|Δ| tokens across
    the whole (B×T) tensor (or the negative-|Δ| tokens), NOT per-response.

    ----------------------------------------------------------------------
    Phase-1 modes (already used in the magnitude-ablation matrix)
    ----------------------------------------------------------------------
    * ``raw`` — identity (zeroed at pads).
    * ``perm`` — |A| shuffled within each sign group; per-sign Σ|A| exactly
      preserved, distribution of |A| preserved.
    * ``group_const`` — |A| replaced with the mean |Δ| of the sign group;
      per-sign Σ|A| exactly preserved, within-group variance flattened.
      Equivalent to (M=M_raw, q=q_uniform).
    * ``sign`` — |A| replaced with 1 (yields A ∈ {-1, 0, +1}). NB: not scale-
      matched — Σ|A| = n_+ + n_- which typically ≠ Σ|Δ|.
    * ``power`` — coupled shaping A_t = sign(Δ_t) * |Δ_t|^α with
      ``power_alpha = α``. Both M_± AND q_± drift with α (this is why the
      Phase-2 plan splits them into ``alloc_power`` and ``strength_interp``).

    ----------------------------------------------------------------------
    Phase-2 modes (orthogonal (M_±, q_±) decomposition; w_± = M_± · q_±)
    ----------------------------------------------------------------------
    Notation for a sign group g ∈ {+, -} inside one microbatch:
        n_g       := #tokens in group g
        S_g^raw   := Σ_{i∈g} |Δ_i|  (== M_g^raw, the raw per-group strength)
        q_i^raw   := |Δ_i| / S_g^raw for i ∈ g  (raw within-group allocation)
        q_i^unif  := 1 / n_g        for i ∈ g   (uniform allocation)

    * ``sign_mass_raw_alloc`` — (2×2 factorial cell: M=sign, q=raw)
        M_g^new := c · n_g   with c := (S_+^raw + S_-^raw) / (n_+ + n_-)
                              (global scale-match: preserves Σ|A^new| = Σ|Δ|)
        q_i^new := q_i^raw
        A_i     := sign(Δ_i) · M_g^new · q_i^raw
                 = sign(Δ_i) · c · n_g · |Δ_i| / S_g^raw
      Isolates the effect of replacing raw group-strength (S_g^raw) with
      count-based group-strength (c · n_g) while keeping the within-group
      allocation frozen.

    * ``alloc_power`` — allocation-only sweep (M=raw, q=q^(α))
        q_i^(α) := |Δ_i|^α / Σ_{j∈g} |Δ_j|^α
        M_g^new := S_g^raw   (raw per-group strength preserved exactly)
        A_i     := sign(Δ_i) · S_g^raw · q_i^(α)
      Endpoints: α=1 ≡ ``raw``; α=0 ≡ ``group_const`` (uniform allocation
      with raw group strength). Intermediate α ∈ (0,1) smoothly compresses
      within-group allocation without touching M_±.
      Uses ``alloc_alpha``.

    * ``strength_interp`` — strength-only sweep (M=M(β), q=q_uniform)
        r_raw  := S_+^raw / (S_+^raw + S_-^raw)     (raw pos-mass fraction)
        r_sign := n_+ / (n_+ + n_-)                 (count pos-mass fraction)
        r_β    := β · r_raw + (1-β) · r_sign
        S      := S_+^raw + S_-^raw                 (total strength, fixed)
        M_+^β  := r_β · S,   M_-^β := (1-r_β) · S
        A_i    := sign(Δ_i) · M_g^β / n_g            (q_i^unif per group)
      Endpoints: β=1 ≡ scale-matched GroupConst (M_g = S_g^raw ⇒ token-level
      identical to ``group_const``); β=0 ≡ scale-matched Sign (Σ|A|=Σ|Δ|,
      not Σ|A|=n_++n_-). β=0.5 is the mid-point.
      Uses ``strength_beta``.

    ----------------------------------------------------------------------
    Adaptive Distance-Calibrated TVOPD
    ----------------------------------------------------------------------
    * ``dctv`` — A_i := ``dctv_c`` · sign(Δ_i) on valid tokens (pads/zero-Δ
      remain 0). ``dctv_c`` is a scalar computed by the caller from the
      *previous* step's EMAs of the sequence-level TV distance estimator
      (D̄_{t-1}) and the predicted one-step TV motion (M̄_{t-1}), so from the
      transform's perspective it is just a global multiplier on top of the
      pure-sign field. See :class:`PolicyLossConfig`'s ``opd_dctv_*`` knobs
      and the DCTV controller in :meth:`DataParallelPPOActor.update_policy`.

    Zero-Δ positions carry ``sign(0) = 0`` and contribute no signal in any
    mode, which matches the sentinel-token invariant used elsewhere in the
    OPD loss (sentinel Δ = 0 → no gradient contribution). Degenerate groups
    (n_g == 0 or S_g^raw == 0) contribute nothing.
    """
    if mode == "raw":
        return advantages * response_mask.to(advantages.dtype)

    mask_bool = response_mask.bool()
    signs = torch.sign(advantages)
    mags = advantages.abs()

    pos_mask = mask_bool & (signs > 0)
    neg_mask = mask_bool & (signs < 0)

    new_mags = torch.zeros_like(mags)

    if mode == "perm":
        # Random per-sign-group permutation of the valid magnitudes. Uses
        # torch.randperm on device indices; verl doesn't seed torch.manual_seed
        # per-microbatch so different microbatches get independent perms —
        # acceptable for a randomization ablation.
        for group_mask in (pos_mask, neg_mask):
            n = int(group_mask.sum().item())
            if n == 0:
                continue
            grp_mags = mags[group_mask]
            perm = torch.randperm(n, device=grp_mags.device)
            new_mags[group_mask] = grp_mags[perm]
        return signs * new_mags

    if mode == "group_const":
        for group_mask in (pos_mask, neg_mask):
            n = int(group_mask.sum().item())
            if n == 0:
                continue
            mean_mag = mags[group_mask].mean()
            new_mags[group_mask] = mean_mag
        return signs * new_mags

    if mode == "sign":
        # |A| == 1 on all valid non-zero-Δ tokens; sign(0)=0 keeps degenerate
        # tokens contributing zero. Multiplied by response_mask so pads stay 0.
        return signs * response_mask.to(advantages.dtype)

    if mode == "power":
        # A_t = sign(Δ_t) * |Δ_t|^α, masked at pads. α outside [0,1] is
        # allowed (users may want α>1 to over-emphasize magnitude), but
        # α<0 is rejected because it blows up on small |Δ|.
        if power_alpha < 0:
            raise ValueError(f"opd_adv_power_alpha must be >= 0, got {power_alpha}")
        valid = mask_bool & (mags > 0)  # sign(0)=0 → keep Δ=0 tokens zero
        new_mags[valid] = mags[valid].pow(power_alpha)
        return signs * new_mags

    if mode == "sign_mass_raw_alloc":
        # Phase-2 (2×2 missing cell): M = c·n_g (count-based, globally scale-
        # matched so Σ|A^new| = Σ|Δ|); q = q_raw (raw within-group alloc).
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())
        n_total = n_pos + n_neg
        if n_total == 0:
            return new_mags  # nothing to do
        s_pos = mags[pos_mask].sum() if n_pos > 0 else advantages.new_zeros(())
        s_neg = mags[neg_mask].sum() if n_neg > 0 else advantages.new_zeros(())
        s_total = s_pos + s_neg
        # Guard degenerate microbatch where every valid |Δ| is zero.
        if not torch.isfinite(s_total) or s_total.item() <= 0:
            return signs * new_mags
        c = s_total / float(n_total)  # global scale-match constant
        for group_mask, n_g, s_g in (
            (pos_mask, n_pos, s_pos),
            (neg_mask, n_neg, s_neg),
        ):
            if n_g == 0 or s_g.item() <= 0:
                continue
            m_g = c * float(n_g)  # M_g^new = c · n_g
            # q_i^raw = |Δ_i| / s_g  ⇒  |A_i^new| = m_g · q_i^raw
            new_mags[group_mask] = m_g * mags[group_mask] / s_g
        return signs * new_mags

    if mode == "alloc_power":
        # Phase-2 allocation-only sweep: M = M_raw exactly; q_i^(α) = |Δ_i|^α /
        # Σ_g |Δ_j|^α. α=1 ≡ raw, α=0 ≡ group_const.
        if alloc_alpha < 0:
            raise ValueError(f"opd_adv_alloc_alpha must be >= 0, got {alloc_alpha}")
        for group_mask in (pos_mask, neg_mask):
            n_g = int(group_mask.sum().item())
            if n_g == 0:
                continue
            grp_mags = mags[group_mask]
            s_g = grp_mags.sum()
            if s_g.item() <= 0:
                continue
            # q_i^(α): |Δ_i|^α normalised over the sign group. α=0 uses
            # {|Δ|>0 → 1, else 0} rather than 0^0 = 1 so zero-Δ tokens stay
            # inert (matches sentinel semantics).
            if alloc_alpha == 0:
                pow_mags = (grp_mags > 0).to(grp_mags.dtype)
            else:
                pow_mags = grp_mags.pow(alloc_alpha)
            denom = pow_mags.sum()
            if denom.item() <= 0:
                continue
            # |A_i^new| = M_g^raw · q_i^(α) = s_g · pow_mags / denom.
            new_mags[group_mask] = s_g * pow_mags / denom
        return signs * new_mags

    if mode == "strength_interp":
        # Phase-2 strength-only sweep: q = q_uniform; M_+/M_- interpolates
        # between raw pos-mass fraction (β=1) and count-based (β=0) at fixed
        # total strength S = S_+^raw + S_-^raw.
        n_pos = int(pos_mask.sum().item())
        n_neg = int(neg_mask.sum().item())
        n_total = n_pos + n_neg
        if n_total == 0:
            return new_mags
        s_pos = mags[pos_mask].sum() if n_pos > 0 else advantages.new_zeros(())
        s_neg = mags[neg_mask].sum() if n_neg > 0 else advantages.new_zeros(())
        s_total = s_pos + s_neg
        if not torch.isfinite(s_total) or s_total.item() <= 0:
            return signs * new_mags
        # r_raw = S+ / (S+ + S-); r_sign = n+ / (n+ + n-).
        # Degenerate one-sided cases: if a group is empty, that endpoint of
        # r_* is 1.0 (all mass on the other sign), which the below handles
        # naturally because n_g == 0 ⇒ that group's assignment is skipped.
        r_raw = (s_pos / s_total).item()
        r_sign = float(n_pos) / float(n_total)
        r_beta = strength_beta * r_raw + (1.0 - strength_beta) * r_sign
        # Clamp to [0,1] to be safe against numerical drift when β is
        # outside [0,1]; users can still pass β<0 or β>1 to extrapolate.
        r_beta = max(0.0, min(1.0, r_beta))
        m_pos = r_beta * s_total.item()
        m_neg = (1.0 - r_beta) * s_total.item()
        if n_pos > 0 and m_pos > 0:
            new_mags[pos_mask] = m_pos / float(n_pos)
        if n_neg > 0 and m_neg > 0:
            new_mags[neg_mask] = m_neg / float(n_neg)
        return signs * new_mags

    if mode == "dctv":
        # Adaptive Distance-Calibrated TVOPD: |A| = dctv_c on all valid non-zero-Δ
        # tokens; sign(0)=0 keeps degenerate tokens at 0. dctv_c is a
        # deterministic (past-conditioned) global scalar computed by the caller
        # from D̄_{t-1} / (M̄_{t-1} + ε). This branch treats it as an opaque
        # multiplier — controller state lives in DataParallelPPOActor.
        return signs * response_mask.to(advantages.dtype) * float(dctv_c)

    raise ValueError(f"Unknown opd_adv_mode={mode!r}")


@register_policy_loss("opd")  # type: ignore[arg-type]
def compute_policy_loss_opd(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
    dctv_c_current: float = 1.0,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for OPD.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Log-probabilities of actions teacher gives.
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        config: `(verl.trainer.config.ActorConfig)`:
            config for the actor.
        rollout_log_probs: `(torch.Tensor)`:
            log probabilities of actions under the rollout policy, shape (batch_size, response_length).
        dctv_c_current (float, optional): DCTV controller's current scalar c_t.
            Only consumed when ``opd_adv_mode='dctv'``; ignored otherwise. Must be
            the *past-conditioned* value (min(c_cap, D̄_{t-1}/(M̄_{t-1}+ε))) so
            the transform is deterministic w.r.t. the current stochastic Δ.
            Defaults to 1.0 (equivalent to pure sign when combined with mode='dctv').
    """
    
    prior_log_probs = old_log_prob.detach()
    with torch.no_grad():
        response_length = prior_log_probs.shape[-1]
        teacher_log_probs, opd_chunk_ids = _unpack_opd_signal(advantages, response_length)

        # Special token positions are marked with inf sentinel (real logprobs ∈ (-inf, 0]).
        # Replace them with student logprobs so advantage = 0 → no loss contribution.
        sentinel_mask = torch.isinf(teacher_log_probs)
        if opd_chunk_ids is not None:
            sentinel_mask = sentinel_mask | (opd_chunk_ids < 0)
        # --- OPD metrics: count inf tokens in this micro-batch ---
        resp_mask_bool = response_mask.bool()
        opd_inf_tokens = int((sentinel_mask & resp_mask_bool).sum().item())
        opd_valid_tokens = int(response_mask.sum().item())
        opd_inf_ratio = opd_inf_tokens / max(opd_valid_tokens, 1)
        # ---
        if sentinel_mask.any():
            teacher_log_probs = torch.where(sentinel_mask, prior_log_probs, teacher_log_probs)

        # --- Credit assignment inside each synchronized chunk ---
        # opd_credit_rule selects how the teacher chunk log-likelihood is
        # distributed to student tokens *inside the chunk*. All rules preserve
        # chunk-level log-prob conservation: sum_i target_i == L_T^(c).
        #
        #   semantic (default, paper Eq.7-9):
        #     target_i = (L_T^(c) / L_S^(c)) * log p_i
        #   uniform (P0-1 ablation):
        #     target_i = L_T^(c) / n_tokens_in_chunk
        #
        # On 1:1 chunks (single-token) all rules reduce to target_i = L_T^(c),
        # i.e. the standard per-token teacher logprob.
        credit_rule = "semantic"
        _pl_cfg = config.get("policy_loss", None) if hasattr(config, "get") else None
        if _pl_cfg is not None:
            _rule = _pl_cfg.get("opd_credit_rule", None)
            if _rule is not None:
                credit_rule = str(_rule).lower()
        if credit_rule not in ("semantic", "uniform"):
            raise ValueError(
                f"Unknown opd_credit_rule={credit_rule!r}; expected 'semantic' or 'uniform'."
            )

        if opd_chunk_ids is None:
            # Fallback: no chunk info available, use simple difference
            advantages = teacher_log_probs - prior_log_probs
        else:
            target_log_probs = prior_log_probs.clone()
            for seq_id in range(teacher_log_probs.shape[0]):
                seq_valid = resp_mask_bool[seq_id] & (~sentinel_mask[seq_id])
                if not seq_valid.any():
                    continue
                for chunk_id in torch.unique(opd_chunk_ids[seq_id][seq_valid]):
                    chunk_mask = seq_valid & (opd_chunk_ids[seq_id] == chunk_id)
                    teacher_chunk_logp = teacher_log_probs[seq_id][chunk_mask].sum()
                    n_in_chunk = chunk_mask.sum().clamp_min(1)
                    if credit_rule == "uniform":
                        target_log_probs[seq_id][chunk_mask] = teacher_chunk_logp / n_in_chunk
                    else:
                        prior_chunk_logp = prior_log_probs[seq_id][chunk_mask].sum()
                        if torch.abs(prior_chunk_logp) < 1e-8:
                            # Degenerate case: student assigns ~0 logprob to chunk;
                            # fall back to uniform distribution of teacher budget
                            target_log_probs[seq_id][chunk_mask] = teacher_chunk_logp / n_in_chunk
                        else:
                            target_log_probs[seq_id][chunk_mask] = (
                                teacher_chunk_logp / prior_chunk_logp
                            ) * prior_log_probs[seq_id][chunk_mask]
            advantages = target_log_probs - prior_log_probs

        # Optional per-token advantage clamp (analogous to verl's `loss_max_clamp`).
        # Prevents extreme teacher/student log-prob gaps (e.g. -30) from producing
        # huge advantages that dominate the gradient for a single token.
        # Set via Hydra override:
        #   actor_rollout_ref.actor.policy_loss.opd_loss_max_clamp=10.0
        # null (default) = no clamp.
        opd_loss_max_clamp = None
        policy_loss_cfg = config.get("policy_loss", None) if hasattr(config, "get") else None
        if policy_loss_cfg is not None:
            opd_loss_max_clamp = policy_loss_cfg.get("opd_loss_max_clamp", None)
        if opd_loss_max_clamp is not None:
            advantages = torch.clamp(advantages, min=-opd_loss_max_clamp, max=opd_loss_max_clamp)

        # --- OPD Δ diagnostics (computed on raw Δ, BEFORE optional sign ablation) ---
        # Δ_t := teacher_log_probs - prior_log_probs on sampled tokens.
        # eff_mask excludes sentinel positions (where teacher was substituted with
        # student → Δ=0), so distribution-shape stats aren't biased by them.
        _eff_mask = resp_mask_bool & (~sentinel_mask)
        _n_eff = int(_eff_mask.sum().item())
        if _n_eff > 0:
            _adv_flat = advantages[_eff_mask]
            _abs_flat = _adv_flat.abs()
            _pos_mask = _adv_flat > 0
            _neg_mask = _adv_flat < 0
            _n_pos = int(_pos_mask.sum().item())
            _n_neg = int(_neg_mask.sum().item())
            opd_adv_mean = _adv_flat.mean().item()
            opd_adv_abs_mean = _abs_flat.mean().item()
            opd_adv_pos_mean = _adv_flat[_pos_mask].mean().item() if _n_pos > 0 else 0.0
            opd_adv_neg_abs_mean = (-_adv_flat[_neg_mask]).mean().item() if _n_neg > 0 else 0.0
            _abs_f32 = _abs_flat.float()
            opd_adv_abs_p50 = torch.quantile(_abs_f32, 0.5).item()
            opd_adv_abs_p90 = torch.quantile(_abs_f32, 0.9).item()
            opd_adv_near_zero_ratio = (_abs_flat < 0.1).float().mean().item()
            _tch_flat = teacher_log_probs[_eff_mask]
            _stu_flat = prior_log_probs[_eff_mask]
            opd_teacher_sample_logp_mean = _tch_flat.mean().item()
            opd_student_sample_logp_mean = _stu_flat.mean().item()
            opd_teacher_sample_logp_pos_mean = _tch_flat[_pos_mask].mean().item() if _n_pos > 0 else 0.0
            opd_student_sample_logp_pos_mean = _stu_flat[_pos_mask].mean().item() if _n_pos > 0 else 0.0
            opd_teacher_sample_logp_neg_mean = _tch_flat[_neg_mask].mean().item() if _n_neg > 0 else 0.0
            opd_student_sample_logp_neg_mean = _stu_flat[_neg_mask].mean().item() if _n_neg > 0 else 0.0
        else:
            opd_adv_mean = 0.0
            opd_adv_abs_mean = 0.0
            opd_adv_pos_mean = 0.0
            opd_adv_neg_abs_mean = 0.0
            opd_adv_abs_p50 = 0.0
            opd_adv_abs_p90 = 0.0
            opd_adv_near_zero_ratio = 0.0
            opd_teacher_sample_logp_mean = 0.0
            opd_student_sample_logp_mean = 0.0
            opd_teacher_sample_logp_pos_mean = 0.0
            opd_student_sample_logp_pos_mean = 0.0
            opd_teacher_sample_logp_neg_mean = 0.0
            opd_student_sample_logp_neg_mean = 0.0

        # --- DCTV sequence-level TV distance estimator ---
        # Per-token d_i := [1 - exp(Δ_i)]_+ ∈ [0,1]; under student sampling
        # E[d_i | s_i] = TV(π_T(·|s_i), π_S(·|s_i)). Reduce to the per-microbatch
        # token-level mean on eff_mask; the DP-reduced mini-batch mean is then
        # used by the DCTV controller in the actor loop (see
        # ``DataParallelPPOActor.update_policy``). Emit both the sum and the
        # count so downstream can do a tokens-weighted global average.
        # NOTE: computed on ``advantages`` at this point in the function, i.e.
        # AFTER chunk credit assignment and AFTER optional ``opd_loss_max_clamp``,
        # BEFORE any magnitude ablation. This keeps D̂ tied to the actual
        # per-token Δ that the loss consumes.
        if _n_eff > 0:
            _dctv_d = torch.clamp(1.0 - torch.exp(advantages), min=0.0)
            _dctv_d = _dctv_d * _eff_mask.to(_dctv_d.dtype)
            opd_dctv_D_sum_microbatch = _dctv_d.sum().item()
            opd_dctv_D_count_microbatch = int(_n_eff)
            opd_dctv_D_hat_microbatch = opd_dctv_D_sum_microbatch / max(
                opd_dctv_D_count_microbatch, 1
            )
        else:
            opd_dctv_D_sum_microbatch = 0.0
            opd_dctv_D_count_microbatch = 0
            opd_dctv_D_hat_microbatch = 0.0

        # Optional Phase-1/2 magnitude ablation on the final per-token advantage.
        # Modes are documented on ``PolicyLossConfig.opd_adv_mode`` and
        # implemented in :func:`_apply_adv_transform`:
        #   * ``raw`` (default) — identity
        #   * ``perm`` — shuffle |A| within each sign group (Σ|A|+, Σ|A|- preserved)
        #   * ``group_const`` — replace |A| with per-sign-group mean |Δ|
        #     (== M=M_raw, q=q_uniform)
        #   * ``sign`` — replace |A| with 1 (equivalent to the removed
        #     ``opd_advantage_sign_only`` flag; not scale-matched)
        #   * ``power`` — coupled shaping A_t = sign(Δ_t) * |Δ_t|^α with α =
        #     ``opd_adv_power_alpha`` (default 1.0; α=0 ⇔ sign, α=1 ⇔ raw).
        #     Both M_± and q_± drift with α.
        #   * ``sign_mass_raw_alloc`` — Phase-2 (2×2 factorial cell): M=c·n_g
        #     (scale-matched), q=q_raw. Isolates the M swap.
        #   * ``alloc_power`` — Phase-2 allocation-only sweep: M=M_raw,
        #     q_i^(α)=|Δ_i|^α / Σ_g |Δ_j|^α. Uses ``opd_adv_alloc_alpha``.
        #     α=1 ⇔ raw, α=0 ⇔ group_const.
        #   * ``strength_interp`` — Phase-2 strength-only sweep: q=q_uniform,
        #     r_β=β·r_raw+(1-β)·r_sign. Uses ``opd_adv_strength_beta``.
        #     β=1 ⇔ scale-matched group_const, β=0 ⇔ scale-matched sign.
        #   * ``dctv`` — Adaptive Distance-Calibrated TVOPD. |A|=c_t on valid
        #     non-zero-Δ tokens, where c_t is provided by the caller via the
        #     ``dctv_c_current`` kwarg (past-conditioned; see
        #     :meth:`DataParallelPPOActor.update_policy`).
        # Sentinel tokens have Δ=0 → sign(0)=0 → no gradient contribution in
        # any mode, matching the sentinel semantics used above.
        # Enabled via:
        #   actor_rollout_ref.actor.policy_loss.opd_adv_mode=<mode>
        #   actor_rollout_ref.actor.policy_loss.opd_adv_power_alpha=<float>
        #   actor_rollout_ref.actor.policy_loss.opd_adv_alloc_alpha=<float>
        #   actor_rollout_ref.actor.policy_loss.opd_adv_strength_beta=<float>
        opd_adv_mode = "raw"
        opd_adv_power_alpha = 1.0
        opd_adv_alloc_alpha = 1.0
        opd_adv_strength_beta = 1.0
        opd_dctv_beta_d_cfg = 0.95
        opd_dctv_beta_m_cfg = 0.95
        opd_dctv_eps_cfg = 1e-8
        opd_dctv_c_cap_cfg = 1.0
        if policy_loss_cfg is not None:
            opd_adv_mode = str(policy_loss_cfg.get("opd_adv_mode", "raw"))
            opd_adv_power_alpha = float(policy_loss_cfg.get("opd_adv_power_alpha", 1.0))
            opd_adv_alloc_alpha = float(policy_loss_cfg.get("opd_adv_alloc_alpha", 1.0))
            opd_adv_strength_beta = float(policy_loss_cfg.get("opd_adv_strength_beta", 1.0))
            opd_dctv_beta_d_cfg = float(policy_loss_cfg.get("opd_dctv_beta_d", 0.95))
            opd_dctv_beta_m_cfg = float(policy_loss_cfg.get("opd_dctv_beta_m", 0.95))
            opd_dctv_eps_cfg = float(policy_loss_cfg.get("opd_dctv_eps", 1e-8))
            opd_dctv_c_cap_cfg = float(policy_loss_cfg.get("opd_dctv_c_cap", 1.0))
        if opd_adv_mode != "raw":
            advantages = _apply_adv_transform(
                advantages,
                response_mask,
                opd_adv_mode,
                power_alpha=opd_adv_power_alpha,
                alloc_alpha=opd_adv_alloc_alpha,
                strength_beta=opd_adv_strength_beta,
                dctv_c=float(dctv_c_current),
            )

        # Post-transform diagnostics: |A| mean and N_eff ratio on the tensor
        # actually consumed by PPO. Combined with the pre-transform
        # ``actor/opd_adv_abs_mean`` above (raw Δ distribution) these anchor
        # the shaping story: for ``perm`` they should match raw's abs_mean and
        # n_eff_ratio; for ``group_const`` abs_mean matches raw but n_eff
        # jumps toward 1; for ``sign`` both saturate near 1.
        _adv_transformed_flat = advantages[_eff_mask] if _n_eff > 0 else None
        if _adv_transformed_flat is not None and _adv_transformed_flat.numel() > 0:
            _abs_adv_t = _adv_transformed_flat.abs()
            opd_adv_transformed_abs_mean = _abs_adv_t.mean().item()
            _sum_abs = _abs_adv_t.sum()
            _sum_sq = (_adv_transformed_flat * _adv_transformed_flat).sum()
            if _sum_sq.item() > 0:
                opd_adv_n_eff_ratio = float(
                    ((_sum_abs * _sum_abs) / _sum_sq / float(_abs_adv_t.numel())).item()
                )
            else:
                opd_adv_n_eff_ratio = 0.0
        else:
            opd_adv_transformed_abs_mean = 0.0
            opd_adv_n_eff_ratio = 0.0

        # --- Phase-2 (M_±, q_±) decomposition diagnostics ---
        # Computed on the POST-transform advantages restricted to eff_mask, then
        # split by sign into groups g ∈ {+, -}. All metrics here are per-
        # microbatch and averaged across microbatches by the metric aggregator.
        #   M_g          := Σ_{i∈g} |A_i^new|   (per-sign total strength)
        #   n_g          := #{i∈g}
        #   q_i          := |A_i^new| / M_g     (per-sign allocation dist.)
        #   H(q_g)       := -Σ_i q_i log q_i     (in nats)
        #   H_norm(q_g)  := H(q_g) / log(n_g)   (∈ [0,1]; 1 == uniform)
        # Also record Σ|A^new| / Σ|Δ| (scale ratio vs raw) so scale-matching
        # can be verified — for {raw, perm, group_const, sign_mass_raw_alloc,
        # alloc_power, strength_interp} this should be ~1.0.
        opd_M_pos = 0.0
        opd_M_neg = 0.0
        opd_M_pos_over_neg = 0.0  # M_+ / M_- (∞-safe; 0 if M_- == 0)
        opd_Hq_pos = 0.0
        opd_Hq_neg = 0.0
        opd_Hq_pos_norm = 0.0
        opd_Hq_neg_norm = 0.0
        opd_n_pos_transformed = 0
        opd_n_neg_transformed = 0
        opd_adv_scale_ratio = 0.0  # Σ|A^new| / Σ|Δ|
        if _adv_transformed_flat is not None and _adv_transformed_flat.numel() > 0:
            _sign_t = torch.sign(_adv_transformed_flat)
            _pos_t = _sign_t > 0
            _neg_t = _sign_t < 0
            _abs_t = _adv_transformed_flat.abs().float()  # float32 for stable log
            opd_n_pos_transformed = int(_pos_t.sum().item())
            opd_n_neg_transformed = int(_neg_t.sum().item())
            for is_pos, mask_g in ((True, _pos_t), (False, _neg_t)):
                n_g = int(mask_g.sum().item())
                if n_g <= 0:
                    continue
                m_g = _abs_t[mask_g].sum()
                if m_g.item() <= 0:
                    continue
                q = _abs_t[mask_g] / m_g
                # Guard log(0): mask out zero-q entries. Since q sums to 1 and
                # is nonnegative, log2 ≥ log; keep in nats (ln) for standard H.
                q_nz = q[q > 0]
                h = -(q_nz * torch.log(q_nz)).sum().item()
                # log(n_g) is the max entropy (uniform). Guard n_g == 1.
                h_norm = h / math.log(n_g) if n_g > 1 else 0.0
                if is_pos:
                    opd_M_pos = m_g.item()
                    opd_Hq_pos = h
                    opd_Hq_pos_norm = h_norm
                else:
                    opd_M_neg = m_g.item()
                    opd_Hq_neg = h
                    opd_Hq_neg_norm = h_norm
            if opd_M_neg > 0:
                opd_M_pos_over_neg = opd_M_pos / opd_M_neg
            # Σ|A^new| / Σ|Δ|. Reuse pre-transform stats: opd_adv_abs_mean is
            # (Σ|Δ|)/n_eff on the same eff_mask, so Σ|Δ| = opd_adv_abs_mean * n_eff.
            _sum_abs_delta = opd_adv_abs_mean * float(_n_eff)
            if _sum_abs_delta > 0:
                opd_adv_scale_ratio = float(_sum_abs.item() / _sum_abs_delta)

        # # ---- DEBUG: write seq 0 per-token student logprobs and advantages to file ----
        # # Actor worker stdout is captured by Ray, so write to a file instead.
        # # NOTE: no chunk merging — chunks would require metadata from reward worker;
        # # here we dump per-token info so you can identify the seq via input_ids and
        # # cross-reference with the reward-side alignment log.
        # _debug_file = "/tmp/opd_loss_debug.txt"
        # try:
        #     tch_seq0 = teacher_log_probs[0]
        #     stu_seq0 = student_log_probs[0]
        #     adv_seq0 = advantages[0]
        #     resp_mask_seq0 = response_mask[0].bool()
        #     valid_len = int(resp_mask_seq0.sum().item())
        #
        #     # Pull input_ids/responses that dp_actor stashed on this module
        #     _input_ids = globals().get("_DEBUG_INPUT_IDS", None)
        #     _responses = globals().get("_DEBUG_RESPONSES", None)
        #     _resp_ids0 = None
        #     if _responses is not None:
        #         _resp_ids0 = _responses[0].tolist()
        #     _input_ids0 = None
        #     if _input_ids is not None:
        #         _input_ids0 = _input_ids[0].tolist()
        #
        #     with open(_debug_file, "w") as _f:
        #         _f.write(f"teacher_log_probs.shape={teacher_log_probs.shape}, "
        #                  f"student_log_probs.shape={student_log_probs.shape}, "
        #                  f"advantages.shape={advantages.shape}, "
        #                  f"response_mask.shape={response_mask.shape}\n")
        #         _f.write(f"valid_len={valid_len}\n")
        #         if _input_ids0 is not None:
        #             _f.write(f"input_ids[0] (first 50): {_input_ids0[:50]}\n")
        #             _f.write(f"input_ids[0] (last 20):  {_input_ids0[-20:]}\n")
        #         if _resp_ids0 is not None:
        #             _f.write(f"responses[0] (first 50): {_resp_ids0[:50]}\n")
        #         _f.write("\n")
        #
        #         if valid_len > 0:
        #             tch_valid = tch_seq0[:valid_len]
        #             stu_valid = stu_seq0[:valid_len]
        #             adv_valid = adv_seq0[:valid_len]
        #
        #             _f.write("=" * 80 + "\n")
        #             _f.write(f"[OPD LOSS DEBUG] seq 0: per-token teacher/student/advantage\n")
        #             _f.write(f"(response_length={response_length}, valid_len={valid_len})\n")
        #             _f.write("=" * 80 + "\n")
        #             _f.write(f"{'pos':>4s}  {'tok_id':>7s}  {'teacher':>10s}  {'student':>10s}  {'advantage':>10s}\n")
        #             _f.write("-" * 80 + "\n")
        #
        #             for idx in range(valid_len):
        #                 tok_id = _resp_ids0[idx] if (_resp_ids0 is not None and idx < len(_resp_ids0)) else -1
        #                 _f.write(f"{idx:>4d}  {tok_id:>7d}  "
        #                          f"{tch_valid[idx].item():>10.4f}  "
        #                          f"{stu_valid[idx].item():>10.4f}  "
        #                          f"{adv_valid[idx].item():>10.4f}\n")
        #
        #             _f.write("\n" + "=" * 80 + "\n")
        #             _f.write(f"[SUMMARY seq 0] adv mean={adv_valid.mean().item():.4f}, "
        #                      f"std={adv_valid.std().item():.4f}, "
        #                      f"min={adv_valid.min().item():.4f}, "
        #                      f"max={adv_valid.max().item():.4f}\n")
        #             _f.write(f"[SUMMARY seq 0] tch mean={tch_valid.mean().item():.4f}, "
        #                      f"stu mean={stu_valid.mean().item():.4f}\n")
        #             _f.write("=" * 80 + "\n")
        #         else:
        #             _f.write("valid_len=0, no debug output\n")
        # except Exception:
        #     import traceback
        #     with open(_debug_file, "a") as _f:
        #         traceback.print_exc(file=_f)
        # # ---- END DEBUG ----

    # breakpoint()

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio
    clip_ratio_c = config.get(  # Lower bound of the ratio for dual-clip PPO. See https://arxiv.org/pdf/1912.09729.
        "clip_ratio_c", 3.0
    )

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high

    assert clip_ratio_c > 1.0, (
        "The lower bound of the clip_ratio_c for dual-clip PPO should be greater than 1.0,"
        + f" but get the value: {clip_ratio_c}."
    )

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability
    negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange
    pg_losses2 = -advantages * torch.clamp(
        ratio, 1 - cliprange_low, 1 + cliprange_high
    )  # - clip(ratio, 1-cliprange, 1+cliprange) * A
    clip_pg_losses1 = torch.maximum(
        pg_losses1, pg_losses2
    )  # max(-ratio * A, -clip(ratio, 1-cliprange, 1+cliprange) * A)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)

    pg_losses3 = -advantages * clip_ratio_c
    clip_pg_losses2 = torch.min(pg_losses3, clip_pg_losses1)
    pg_clipfrac_lower = verl_F.masked_mean(
        torch.gt(clip_pg_losses1, pg_losses3) * (advantages < 0).float(), response_mask
    )

    pg_losses = torch.where(advantages < 0, clip_pg_losses2, clip_pg_losses1)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )

    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
        "actor/opd_inf_tokens": opd_inf_tokens,
        "actor/opd_inf_ratio": opd_inf_ratio,
        "actor/opd_adv_pos_ratio": verl_F.masked_mean(
            (advantages > 0).float(), response_mask
        ).detach().item(),
        "actor/opd_adv_neg_ratio": verl_F.masked_mean(
            (advantages < 0).float(), response_mask
        ).detach().item(),
        # Δ magnitude / shape diagnostics (computed pre-sign, on eff_mask = response_mask & ~sentinel)
        "actor/opd_adv_mean": opd_adv_mean,
        "actor/opd_adv_abs_mean": opd_adv_abs_mean,
        "actor/opd_adv_pos_mean": opd_adv_pos_mean,
        "actor/opd_adv_neg_abs_mean": opd_adv_neg_abs_mean,
        "actor/opd_adv_abs_p50": opd_adv_abs_p50,
        "actor/opd_adv_abs_p90": opd_adv_abs_p90,
        "actor/opd_adv_near_zero_ratio": opd_adv_near_zero_ratio,
        # Sampled-token teacher/student logp distributions (cheap; no extra inference)
        "actor/opd_teacher_sample_logp_mean": opd_teacher_sample_logp_mean,
        "actor/opd_student_sample_logp_mean": opd_student_sample_logp_mean,
        "actor/opd_teacher_sample_logp_pos_mean": opd_teacher_sample_logp_pos_mean,
        "actor/opd_student_sample_logp_pos_mean": opd_student_sample_logp_pos_mean,
        "actor/opd_teacher_sample_logp_neg_mean": opd_teacher_sample_logp_neg_mean,
        "actor/opd_student_sample_logp_neg_mean": opd_student_sample_logp_neg_mean,
        # Post-transform magnitude ablation diagnostics (see opd_adv_mode)
        "actor/opd_adv_transformed_abs_mean": opd_adv_transformed_abs_mean,
        "actor/opd_adv_n_eff_ratio": opd_adv_n_eff_ratio,
        "actor/opd_adv_power_alpha": opd_adv_power_alpha,
        "actor/opd_adv_alloc_alpha": opd_adv_alloc_alpha,
        "actor/opd_adv_strength_beta": opd_adv_strength_beta,
        # Phase-2 (M_±, q_±) decomposition diagnostics on post-transform |A|.
        # M_g = Σ|A_i^new| for i in sign group g; q_i = |A_i^new| / M_g;
        # H(q_g) in nats; H_norm(q_g) = H(q_g) / log(n_g) ∈ [0,1] with 1 == uniform.
        # opd_adv_scale_ratio = Σ|A^new| / Σ|Δ| — should be ~1 for scale-matched
        # modes (raw / perm / group_const / sign_mass_raw_alloc / alloc_power /
        # strength_interp); ≠1 diagnoses drift.
        "actor/opd_M_pos": opd_M_pos,
        "actor/opd_M_neg": opd_M_neg,
        "actor/opd_M_pos_over_neg": opd_M_pos_over_neg,
        "actor/opd_Hq_pos": opd_Hq_pos,
        "actor/opd_Hq_neg": opd_Hq_neg,
        "actor/opd_Hq_pos_norm": opd_Hq_pos_norm,
        "actor/opd_Hq_neg_norm": opd_Hq_neg_norm,
        "actor/opd_n_pos_transformed": opd_n_pos_transformed,
        "actor/opd_n_neg_transformed": opd_n_neg_transformed,
        "actor/opd_adv_scale_ratio": opd_adv_scale_ratio,
        # DCTV per-microbatch stats. ``opd_dctv_D_sum`` / ``opd_dctv_D_count``
        # are the raw sum + eff-mask count of d_i on this microbatch (used by
        # the actor loop to build a DP-reduced mini-batch mean before updating
        # the D_ema state). ``opd_dctv_D_hat`` is the local per-microbatch
        # mean, kept for logging convenience. ``opd_dctv_c_current`` echoes
        # the caller-provided c_t so the log ties back to which controller
        # value shaped this microbatch's |A|.
        "actor/opd_dctv_D_sum": opd_dctv_D_sum_microbatch,
        "actor/opd_dctv_D_count": opd_dctv_D_count_microbatch,
        "actor/opd_dctv_D_hat_microbatch": opd_dctv_D_hat_microbatch,
        "actor/opd_dctv_c_current": float(dctv_c_current),
        # DCTV hyperparam echoes (mirror the opd_adv_power_alpha / alloc_alpha /
        # strength_beta echoes above). Constant per-run in practice but logged so
        # per-run wandb pages carry the estimator smoothing constants directly.
        "actor/opd_dctv_beta_d": opd_dctv_beta_d_cfg,
        "actor/opd_dctv_beta_m": opd_dctv_beta_m_cfg,
        "actor/opd_dctv_eps": opd_dctv_eps_cfg,
        "actor/opd_dctv_c_cap": opd_dctv_c_cap_cfg,
    }
    return pg_loss, pg_metrics


@register_policy_loss("alm")  # type: ignore[arg-type]
def compute_alm_loss(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Chunk-level supervised distillation loss in the ALM style (Minixhofer et al., 2025).

    Consumes the same ``[teacher_logps | chunk_ids]`` payload the OPD reward manager writes
    into ``advantages`` (see :func:`_unpack_opd_signal`). Unlike :func:`compute_policy_loss_opd`
    the teacher signal here is *not* a reward: we treat each aligned chunk as a Bernoulli
    ``{p, 1-p}`` on both sides (``p = exp(sum log p_token)``) and minimize the binary CE
    between the teacher and student Bernoullis. Gradient flows directly through the
    student chunk-sum log-p — no PPO ratio, no rollout, no clip.

    Signature is kept identical to the OPD PPO loss so the actor loop needs no changes;
    ``old_log_prob`` and ``rollout_is_weights`` are accepted but unused in Phase 1.

    Configurable via ``actor_rollout_ref.actor.policy_loss.*``:

    - ``alm_binarization_temp`` (float, default 1.0): divide chunk log-p by this temperature.
    - ``alm_diff_fn`` (str, default ``binary_ce``): only ``binary_ce`` implemented in Phase 1.
    - ``alm_numerator`` (str, default ``chunk_count``): per-chunk weight —
      ``chunk_count``, ``token_count`` or ``log1p_token_count``.
    - ``alm_denominator`` (str, default ``chunk_count``): only ``chunk_count`` implemented.
    - ``alm_chunk_clamp`` (Optional[float], default None): clamp chunk log-p to ``[-C, 0]``.
    """
    response_length = log_prob.shape[-1]
    teacher_log_probs, opd_chunk_ids = _unpack_opd_signal(advantages, response_length)

    # Resolve ALM config knobs with defaults so existing configs keep working.
    T_temp = 1.0
    diff_fn = "binary_ce"
    numerator_mode = "chunk_count"
    denominator_mode = "chunk_count"
    chunk_clamp: Optional[float] = None
    pl_cfg = config.get("policy_loss", None) if (config is not None and hasattr(config, "get")) else None
    if pl_cfg is not None:
        T_temp = float(pl_cfg.get("alm_binarization_temp", 1.0))
        diff_fn = str(pl_cfg.get("alm_diff_fn", "binary_ce")).lower()
        numerator_mode = str(pl_cfg.get("alm_numerator", "chunk_count")).lower()
        denominator_mode = str(pl_cfg.get("alm_denominator", "chunk_count")).lower()
        chunk_clamp = pl_cfg.get("alm_chunk_clamp", None)
    if diff_fn != "binary_ce":
        raise NotImplementedError(
            f"alm_diff_fn={diff_fn!r} not implemented; only 'binary_ce' available in Phase 1."
        )
    if denominator_mode != "chunk_count":
        raise NotImplementedError(
            f"alm_denominator={denominator_mode!r} not implemented; only 'chunk_count' available in Phase 1."
        )
    if numerator_mode not in ("chunk_count", "token_count", "log1p_token_count"):
        raise ValueError(
            f"Unknown alm_numerator={numerator_mode!r}; "
            "expected 'chunk_count', 'token_count' or 'log1p_token_count'."
        )

    # Sentinel: inf teacher log-p (special tokens, unaligned) or -1 chunk id.
    sentinel_mask = torch.isinf(teacher_log_probs)
    if opd_chunk_ids is not None:
        sentinel_mask = sentinel_mask | (opd_chunk_ids < 0)
    resp_mask_bool = response_mask.bool()
    valid = resp_mask_bool & ~sentinel_mask  # [B, T]

    alm_inf_tokens = int((sentinel_mask & resp_mask_bool).sum().item())
    alm_valid_tokens = int(resp_mask_bool.sum().item())
    alm_inf_ratio = alm_inf_tokens / max(alm_valid_tokens, 1)

    # Zero out sentinel positions before the chunk sum. ``teacher_log_probs`` carries
    # inf sentinels; ``log_prob`` should already be finite but we zero it too for symmetry.
    safe_teacher = torch.where(
        valid, teacher_log_probs, torch.zeros_like(teacher_log_probs)
    ).detach()
    safe_student = torch.where(valid, log_prob, torch.zeros_like(log_prob))

    if opd_chunk_ids is None:
        # Fallback: treat every valid token as its own singleton chunk.
        # `A` degenerates to a diagonal, so chunk-sum == token-level values.
        teacher_chunk_logp = safe_teacher
        student_chunk_logp = safe_student
        chunk_valid = valid
        chunk_count = valid.to(safe_student.dtype)
    else:
        # ``chunk_ids`` may collide across sequences (chunk 0 of seq A vs chunk 0 of seq B)
        # but that is fine: einsum sums per-sample along the K dim so cross-seq contributions
        # never mix.
        safe_ids = opd_chunk_ids.clamp_min(0)  # -1 → 0, but its row of `A` is zeroed via `valid`.
        max_id = int(safe_ids.max().item()) if safe_ids.numel() > 0 else 0
        K = max_id + 1
        # one_hot returns int64; cast to the student log-prob dtype so grad can flow.
        A = torch.nn.functional.one_hot(safe_ids, num_classes=K).to(safe_student.dtype)  # [B, T, K]
        A = A * valid.unsqueeze(-1).to(A.dtype)  # zero sentinel / padded rows

        teacher_chunk_logp = torch.einsum("btk,bt->bk", A, safe_teacher)  # no grad
        student_chunk_logp = torch.einsum("btk,bt->bk", A, safe_student)  # GRAD flows
        chunk_count = A.sum(dim=1)  # [B, K], tokens per chunk
        chunk_valid = chunk_count > 0  # [B, K]

    # Chunk log-p should never be positive; tiny numerical drift is possible.
    teacher_chunk_logp = teacher_chunk_logp.clamp(max=0.0)
    student_chunk_logp = student_chunk_logp.clamp(max=0.0)
    if chunk_clamp is not None:
        c = float(chunk_clamp)
        teacher_chunk_logp = teacher_chunk_logp.clamp(min=-c)
        student_chunk_logp = student_chunk_logp.clamp(min=-c)

    # Binary CE between the two chunk Bernoullis.
    # Cast to fp32 inside the numerically sensitive block; keep grad on student side.
    eps = 1e-6
    log_p_t = (teacher_chunk_logp.to(torch.float32) / T_temp) - eps
    log_p_s = (student_chunk_logp.to(torch.float32) / T_temp) - eps
    # Guard against log_p_s hitting 0 exactly (log1mexp(0) is -inf).
    log_p_s = log_p_s.clamp(max=-eps)
    log_p_t = log_p_t.clamp(max=-eps)

    p_t = torch.exp(log_p_t)
    one_minus_p_t = -torch.expm1(log_p_t)  # 1 - exp(log_p_t), stable near 0
    elem_loss = -(p_t * log_p_s + one_minus_p_t * log1mexp(log_p_s))
    elem_loss = elem_loss * chunk_valid.to(elem_loss.dtype)

    # Aggregation.
    if numerator_mode == "chunk_count":
        numer = chunk_valid.to(elem_loss.dtype)
    elif numerator_mode == "token_count":
        numer = chunk_count.to(elem_loss.dtype)
    else:  # log1p_token_count
        numer = torch.log1p(chunk_count.to(elem_loss.dtype))
    denom = chunk_valid.to(elem_loss.dtype).sum().clamp_min(1.0)
    alm_loss = (elem_loss * numer).sum() / denom
    # Cast back to student dtype so downstream ops (entropy/KL sum) stay in one dtype.
    alm_loss = alm_loss.to(log_prob.dtype)

    with torch.no_grad():
        if chunk_valid.any():
            teacher_mean_p = teacher_chunk_logp[chunk_valid].mean().item()
            student_mean_p = student_chunk_logp[chunk_valid].mean().item()
            teacher_min_p = teacher_chunk_logp[chunk_valid].min().item()
            student_min_p = student_chunk_logp[chunk_valid].min().item()
            chunk_count_max = int(chunk_count[chunk_valid].max().item())
            chunk_count_mean = float(chunk_count[chunk_valid].float().mean().item())
        else:
            teacher_mean_p = 0.0
            student_mean_p = 0.0
            teacher_min_p = 0.0
            student_min_p = 0.0
            chunk_count_max = 0
            chunk_count_mean = 0.0

    metrics = {
        "actor/alm_loss": alm_loss.detach().item(),
        "actor/alm_valid_chunks": int(chunk_valid.sum().item()),
        "actor/alm_sentinel_ratio": alm_inf_ratio,
        "actor/alm_teacher_mean_logp": teacher_mean_p,
        "actor/alm_student_mean_logp": student_mean_p,
        "actor/alm_teacher_min_logp": teacher_min_p,
        "actor/alm_student_min_logp": student_min_p,
        "actor/alm_chunk_size_max": chunk_count_max,
        "actor/alm_chunk_size_mean": chunk_count_mean,
    }
    return alm_loss, metrics


@register_policy_loss("gspo")
def compute_policy_loss_gspo(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "seq-mean-token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GSPO.

    See https://arxiv.org/pdf/2507.18071 for more details.

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. For GSPO, it is recommended to use "seq-mean-token-mean".
    """

    assert config is not None
    assert isinstance(config, ActorConfig)
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else config.clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else config.clip_ratio

    negative_approx_kl = log_prob - old_log_prob

    # compute sequence-level importance ratio:
    # si(θ) = (π_θ(yi|x)/π_θold(yi|x))^(1/|yi|) =
    # exp [(1/|y_i|) * Σ_t log(π_θ(y_i,t|x,y_i,<t)/π_θold(y_i,t|x,y_i,<t))]
    seq_lengths = torch.sum(response_mask, dim=-1).clamp(min=1)
    negative_approx_kl_seq = torch.sum(negative_approx_kl * response_mask, dim=-1) / seq_lengths

    # Combined ratio at token level:
    # s_i,t(θ) = sg[s_i(θ)] · π_θ(y_i,t|x, y_i,<t) / sg[π_θ(y_i,t|x, y_i,<t)]
    # In log space: log(s_i,t(θ)) = sg[log(s_i(θ))] + log_prob - sg[log_prob]
    log_seq_importance_ratio = log_prob - log_prob.detach() + negative_approx_kl_seq.detach().unsqueeze(-1)
    log_seq_importance_ratio = torch.clamp(log_seq_importance_ratio, max=10.0)  # clamp for numerical stability

    # finaly exp() to remove log
    seq_importance_ratio = torch.exp(log_seq_importance_ratio)

    pg_losses1 = -advantages * seq_importance_ratio
    pg_losses2 = -advantages * torch.clamp(seq_importance_ratio, 1 - clip_ratio_low, 1 + clip_ratio_high)
    pg_losses = torch.maximum(pg_losses1, pg_losses2)

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    # for GSPO, we need to aggregate the loss at the sequence level (seq-mean-token-mean)
    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode="seq-mean-token-mean", **config.global_batch_info
    )

    # For compatibility, return zero for pg_clipfrac_lower (not used in standard GSPO)
    pg_clipfrac = verl_F.masked_mean(torch.gt(pg_losses2, pg_losses1).float(), response_mask)
    pg_clipfrac_lower = torch.tensor(0.0, device=pg_loss.device)

    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("gpg")
def compute_policy_loss_gpg(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Adapted from
    https://github.com/AMAP-ML/GPG/blob/main/VisualThinker-R1-Zero/src/open-r1-multimodal/src/open_r1/trainer/grpo_trainer.py#L495
    Args:
        log_prob: `(torch.Tensor)`
            shape: (bs, response_length)
        advantages: `(torch.Tensor)`
            shape: (bs, response_length)
        response_mask: `(torch.Tensor)`
            shape: (bs, response_length)
    return:
        pg_loss: `a scalar torch.Tensor`
            policy gradient loss computed via GPG
    """
    assert config is not None
    pg_losses = -log_prob * advantages

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    return pg_loss, {}


@register_policy_loss("clip_cov")
def compute_policy_loss_clip_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        cliprange (float, optional):
            Clipping parameter ε for standard PPO. See https://arxiv.org/abs/1707.06347.
            Defaults to None (must be provided).
        cliprange_low (float, optional):
            Lower clip range for dual-clip PPO. Defaults to same as `cliprange`.
        cliprange_high (float, optional):
            Upper clip range for dual-clip PPO. Defaults to same as `cliprange`.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        clip_cvo_ratio (float, optional):
            Ratio for clipping the covariance. Defaults to 0.0002.
        clip_cov_lb (float, optional):
            Lower bound for clipping covariance. Defaults to 1.0.
        clip_cov_ub (float, optional):
            Upper bound for clipping covariance. Defaults to 5.0.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    clip_cov_ratio = config.policy_loss.clip_cov_ratio if config.policy_loss.clip_cov_ratio is not None else 0.0002
    cliprange = config.clip_ratio
    cliprange_low = config.clip_ratio_low if config.clip_ratio_low is not None else cliprange
    cliprange_high = config.clip_ratio_high if config.clip_ratio_high is not None else cliprange
    clip_cov_ub = config.policy_loss.clip_cov_ub if config.policy_loss.clip_cov_ub is not None else 5.0
    clip_cov_lb = config.policy_loss.clip_cov_lb if config.policy_loss.clip_cov_lb is not None else 1.0

    assert clip_cov_ratio > 0, "clip_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    ratio = torch.exp(negative_approx_kl)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    pg_losses1 = -advantages * ratio

    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    corr = torch.ones_like(advantages)
    pg_losses2 = -advantages * torch.clamp(ratio, 1 - cliprange_low, 1 + cliprange_high)
    clip_by_origin = (pg_losses2 > pg_losses1) & (response_mask > 0)

    cov_all = (advantages - verl_F.masked_mean(advantages, response_mask)) * (
        log_prob - verl_F.masked_mean(log_prob.detach(), response_mask)
    )
    cov_all[response_mask == 0] = -torch.inf
    cov_all[clip_by_origin] = -torch.inf

    clip_num = max(int(clip_cov_ratio * response_mask.sum().item()), 1)
    top_k_idx = (cov_all < clip_cov_ub) & (cov_all > clip_cov_lb) & (response_mask > 0)
    top_k_idx = torch.nonzero(top_k_idx)

    if len(top_k_idx) > 0:
        perm = torch.randperm(len(top_k_idx))
        top_k_idx = top_k_idx[perm[: min(clip_num, len(top_k_idx))]]
    else:
        top_k_idx = torch.empty((0, 2), device=cov_all.device, dtype=torch.long)

    corr[top_k_idx[:, 0], top_k_idx[:, 1]] = 0

    pg_clipfrac = verl_F.masked_mean((corr == 0).float(), response_mask)

    pg_losses = torch.maximum(pg_losses1, pg_losses2) * corr

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("kl_cov")
def compute_policy_loss_kl_cov(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for Clip-Cov.

    Adapted from
    https://github.com/PRIME-RL/Entropy-Mechanism-of-RL/blob/main/verl/trainer/ppo/core_algos.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".
        kl_cov_ratio (float, optional):
            Ratio for selecting the top-k covariance values. Defaults to 0.0002.
        ppo_kl_coef (float, optional):
            Coefficient for the KL penalty term in the loss. Defaults to 1.
    """
    assert config is not None
    assert not isinstance(config, AlgoConfig), "passing AlgoConfig not supported yet"
    assert config.policy_loss is not None

    kl_cov_ratio = config.policy_loss.kl_cov_ratio if config.policy_loss.kl_cov_ratio is not None else 0.0002
    ppo_kl_coef = config.policy_loss.ppo_kl_coef if config.policy_loss.ppo_kl_coef is not None else 1.0

    assert kl_cov_ratio > 0, "kl_cov_ratio should be larger than 0."

    negative_approx_kl = log_prob - old_log_prob
    abs_kl = negative_approx_kl.abs()
    ratio = torch.exp(negative_approx_kl)
    ppo_kl_abs = verl_F.masked_mean(negative_approx_kl.abs(), response_mask)
    pg_losses1 = -advantages * ratio
    pg_losses_kl = -advantages * ratio + ppo_kl_coef * abs_kl
    pg_losses = pg_losses1

    all_valid = response_mask > 0
    all_valid_idx = torch.nonzero(all_valid.reshape(-1), as_tuple=True)[0]
    all_valid_adv = advantages[all_valid].detach().reshape(-1).cpu()
    all_valid_logp = log_prob[all_valid].detach().reshape(-1).cpu()

    k = min(kl_cov_ratio, len(all_valid_adv))

    if k != 0:
        cov_lst_all = (all_valid_adv - all_valid_adv.mean()) * (all_valid_logp - all_valid_logp.mean())
        k_percent_nums = max(1, int(len(cov_lst_all) * kl_cov_ratio))
        large_cov_idxs = torch.topk(cov_lst_all, k_percent_nums, largest=True).indices

        if len(large_cov_idxs) != 0:
            large_cov_idxs = all_valid_idx[large_cov_idxs]
            pg_losses[large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]] = pg_losses_kl[
                large_cov_idxs // advantages.shape[1], large_cov_idxs % advantages.shape[1]
            ]

    # Apply rollout correction weights if provided
    if rollout_is_weights is not None:
        pg_losses = pg_losses * rollout_is_weights

    pg_loss = agg_loss(
        loss_mat=pg_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
    )
    pg_metrics = {
        "actor/ppo_kl": ppo_kl_abs.detach().item(),
    }
    return pg_loss, pg_metrics


@register_policy_loss("geo_mean")
def compute_policy_loss_geo_mean(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """
    Compute the clipped policy objective and related metrics for GMPO.

    Adapted from paper https://arxiv.org/abs/2507.20673
    https://github.com/callsys/GMPO/blob/main/train_zero_math_gmpo.py

    Args:
        old_log_prob (torch.Tensor):
            Log-probabilities of actions under the old policy, shape (batch_size, response_length).
        log_prob (torch.Tensor):
            Log-probabilities of actions under the current policy, shape (batch_size, response_length).
        advantages (torch.Tensor):
            Advantage estimates for each action, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the loss, shape (batch_size, response_length).
        loss_agg_mode (str, optional):
            not used
    """

    assert config is not None
    assert not isinstance(config, AlgoConfig)
    clip_ratio = config.clip_ratio  # Clipping parameter. See https://arxiv.org/abs/1707.06347.
    clip_ratio_low = config.clip_ratio_low if config.clip_ratio_low is not None else clip_ratio
    clip_ratio_high = config.clip_ratio_high if config.clip_ratio_high is not None else clip_ratio

    cliprange = clip_ratio
    cliprange_low = clip_ratio_low
    cliprange_high = clip_ratio_high
    if cliprange_low is None:
        cliprange_low = cliprange
    if cliprange_high is None:
        cliprange_high = cliprange

    negative_approx_kl = log_prob - old_log_prob
    # Clamp negative_approx_kl for stability (uncomment it if you like)
    # negative_approx_kl = torch.clamp(negative_approx_kl, min=-20.0, max=20.0)
    ppo_kl = verl_F.masked_mean(-negative_approx_kl, response_mask)

    # Clipping at token-level & Clipping wider
    sgn_advantage = torch.sign(advantages)
    negative_approx_kl_clamp = torch.clamp(negative_approx_kl, -cliprange_low, cliprange_high)
    negative_approx_kl_min = torch.min(sgn_advantage * negative_approx_kl, sgn_advantage * negative_approx_kl_clamp)
    negative_approx_kl_min = sgn_advantage * negative_approx_kl_min

    # Geometric-Mean Policy Optimization
    response_mask_sum = response_mask.sum(dim=-1)
    ratio = torch.exp((negative_approx_kl_min * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8))
    # we only support sequence level advantage for now,
    # otherwise, below would be not consistent with the paper
    advantage = (advantages * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
    pg_losses = -advantage * ratio

    # Apply rollout correction weights if provided
    # For geo_mean, IS weights are 2D (batch_size, seq_length) and need to be aggregated to sequence level
    if rollout_is_weights is not None:
        # Aggregate token-level weights to sequence level using geometric mean for consistency
        # Note: rollout_is_weights is always 2D regardless of aggregation mode
        seq_is_weights = torch.exp(
            (torch.log(rollout_is_weights + 1e-10) * response_mask).sum(dim=-1) / (response_mask_sum + 1e-8)
        )
        pg_losses = pg_losses * seq_is_weights

    pg_loss = torch.mean(pg_losses)

    # higher: ratio is too large that need clamp to clip_high (when adv > 0)
    clipped = torch.ne(negative_approx_kl, negative_approx_kl_clamp)
    pg_clipfrac = verl_F.masked_mean((clipped * (advantages > 0)).float(), response_mask)
    pg_clipfrac_lower = verl_F.masked_mean((clipped * (advantages < 0)).float(), response_mask)
    pg_metrics = {
        "actor/pg_clipfrac": pg_clipfrac.detach().item(),
        "actor/ppo_kl": ppo_kl.detach().item(),
        "actor/pg_clipfrac_lower": pg_clipfrac_lower.detach().item(),
    }
    return pg_loss, pg_metrics


def compute_entropy_loss(logits, response_mask, loss_agg_mode: str = "token-mean"):
    """Compute categorical entropy loss (For backward compatibility)

    Args:
        logits (torch.Tensor): shape is (bs, response_length, vocab_size)
        response_mask (torch.Tensor): shape is (bs, response_length)

    Returns:
        entropy: a scalar torch.Tensor

    """
    # compute entropy
    token_entropy = verl_F.entropy_from_logits(logits)  # (bs, response_len)
    entropy_loss = agg_loss(loss_mat=token_entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    return entropy_loss


def compute_value_loss(
    vpreds: torch.Tensor,
    returns: torch.Tensor,
    values: torch.Tensor,
    response_mask: torch.Tensor,
    cliprange_value: float,
    loss_agg_mode: str = "token-mean",
):
    """
    Compute the clipped value-function loss for PPO.

    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1151

    Args:
        vpreds (torch.FloatTensor):
            Predicted values from the value head, shape (batch_size, response_length).
        values (torch.FloatTensor):
            Old (baseline) values from the value head, shape (batch_size, response_length).
        returns (torch.FloatTensor):
            Ground-truth returns, shape (batch_size, response_length).
        response_mask (torch.Tensor):
            Mask indicating which tokens to include in the value loss calculation.
        cliprange_value (float):
            Clip range for value prediction updates.
        loss_agg_mode (str, optional):
            Aggregation mode for `agg_loss`. Defaults to "token-mean".

    Returns:
        vf_loss (torch.FloatTensor):
            A scalar tensor containing the aggregated value-function loss.
        vf_clipfrac (float):
            Fraction of elements where the clipped loss was used.
    """
    vpredclipped = verl_F.clip_by_value(vpreds, values - cliprange_value, values + cliprange_value)
    vf_losses1 = (vpreds - returns) ** 2
    vf_losses2 = (vpredclipped - returns) ** 2
    clipped_vf_losses = torch.max(vf_losses1, vf_losses2)
    vf_loss = 0.5 * agg_loss(loss_mat=clipped_vf_losses, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
    vf_clipfrac = verl_F.masked_mean(torch.gt(vf_losses2, vf_losses1).float(), response_mask)
    return vf_loss, vf_clipfrac


def kl_penalty(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob. Optionally using straight through to bind k2 on other
    kl penalty compute method for unbiased KL gradient estimation.
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    forward_score = kl_penalty_forward(logprob, ref_logprob, kl_penalty)
    if not kl_penalty.endswith("+") or kl_penalty in ("mse", "k2"):
        return forward_score

    """
    The expectation of k1 and k3 estimator is the expectaed value of KL, but the expected gradient of k1 and k3
    estimator is not the expectaed gradient of KL. On the other hand k2 estimator gives right gradient estimator, 
    so we use a straight through trick here if the kl_penalty method ends with '+', .e.g., k3+. 
    """
    backward_score = 0.5 * (logprob - ref_logprob).square()

    return backward_score - backward_score.detach() + forward_score.detach()


def kl_penalty_forward(logprob: torch.FloatTensor, ref_logprob: torch.FloatTensor, kl_penalty) -> torch.FloatTensor:
    """Compute KL divergence given logprob and ref_logprob.
    Copied from https://github.com/huggingface/trl/blob/main/trl/trainer/ppo_trainer.py#L1104
    See more description in http://joschu.net/blog/kl-approx.html

    Args:
        logprob:
        ref_logprob:

    Returns:
        kl_estimate
    """
    if kl_penalty in ("kl", "k1"):
        return logprob - ref_logprob

    if kl_penalty == "abs":
        return (logprob - ref_logprob).abs()

    if kl_penalty in ("mse", "k2"):
        return 0.5 * (logprob - ref_logprob).square()

    # J. Schulman. Approximating kl divergence, 2020.
    # # URL http://joschu.net/blog/kl-approx.html.
    if kl_penalty in ("low_var_kl", "k3"):
        kl = ref_logprob - logprob
        # For numerical stability
        kl = torch.clamp(kl, min=-20, max=20)
        ratio = torch.exp(kl)
        kld = (ratio - kl - 1).contiguous()
        return torch.clamp(kld, min=-10, max=10)

    if kl_penalty == "full":
        # so, here logprob and ref_logprob should contain the logits for every token in vocabulary
        raise NotImplementedError

    raise NotImplementedError


def compute_pf_ppo_reweight_data(
    data,
    reweight_method: str = "pow",
    weight_pow: float = 2.0,
):
    """Reweight the data based on the token_level_scores.

    Args:
        data: DataProto object, containing batch, non_tensor_batch and meta_info
        reweight_method: str, choices: "pow", "max_min", "max_random"
        weight_pow: float, the power of the weight

    Returns:

    """

    @torch.no_grad()
    def compute_weights(scores: torch.Tensor, reweight_method: str, weight_pow: float) -> torch.Tensor:
        """Compute importance weights for resampling based on scores.

        Args:
            scores (torch.Tensor): Tensor of scores to compute weights from.
            reweight_method (str): Method for computing weights ('pow', 'max_min', 'max_random').
            weight_pow (float): Power exponent for 'pow' method.

        Returns:
            torch.Tensor: Computed importance weights.

        Raises:
            ValueError: If reweight_method is not supported.
        """
        if reweight_method == "pow":
            weights = torch.pow(torch.abs(scores), weight_pow)
        elif reweight_method == "max_min":
            max_score = torch.max(scores)
            min_score = torch.min(scores)
            weights = torch.where((scores == max_score) | (scores == min_score), 1.0, 0.0)
        elif reweight_method == "max_random":
            max_score = torch.max(scores)
            weights = torch.where(scores == max_score, 0.4, 0.1)
        else:
            raise ValueError(f"Unsupported reweight_method: {reweight_method}")
        return weights

    scores = data.batch["token_level_scores"].sum(dim=-1)
    weights = compute_weights(scores, reweight_method, weight_pow)
    weights = torch.clamp(weights + 1e-8, min=1e-8)

    batch_size = scores.shape[0]
    sample_indices = torch.multinomial(weights, batch_size, replacement=True)

    resampled_batch = {key: tensor[sample_indices] for key, tensor in data.batch.items()}

    sample_indices_np = sample_indices.numpy()
    resampled_non_tensor_batch = {}
    for key, array in data.non_tensor_batch.items():
        if isinstance(array, np.ndarray):
            resampled_non_tensor_batch[key] = array[sample_indices_np]
        else:
            resampled_non_tensor_batch[key] = [array[i] for i in sample_indices_np]

    resampled_meta_info = {}
    for key, value in data.meta_info.items():
        if isinstance(value, list) and len(value) == batch_size:
            resampled_meta_info[key] = [value[i] for i in sample_indices_np]
        else:
            resampled_meta_info[key] = value

    from copy import deepcopy

    resampled_data = deepcopy(data)
    resampled_data.batch = type(data.batch)(resampled_batch)
    resampled_data.batch.batch_size = data.batch.batch_size
    resampled_data.non_tensor_batch = resampled_non_tensor_batch
    resampled_data.meta_info = resampled_meta_info

    return resampled_data


def compute_policy_loss_with_rollout_correction(
    rollout_log_prob,
    log_prob,
    advantages,
    eos_mask,
    loss_agg_mode="seq-mean-token-sum",
    config: Optional[ActorConfig] = None,
    loss_scale_factor=1.0,
    rollout_is: Optional[str] = None,
    rollout_is_threshold: float = 2.0,
    rollout_rs: Optional[str] = None,
    rollout_rs_threshold: Optional[float] = None,
    rollout_rs_threshold_lower: Optional[float] = None,
    rollout_token_veto_threshold: Optional[float] = None,
    rollout_is_batch_normalize: bool = False,
):
    """Compute policy loss with pure rollout correction (no PPO clipping).

    This function implements policy gradient with importance sampling correction
    for rollout-training policy mismatch, without PPO's clipping mechanism.

    Mathematical formulation:
        Without IS (rollout_is=None):
            L = -E[log π(a|s) * A(s,a)]
            Gradient: ∇_θ L = -E[∇log π(a|s) * A] (standard REINFORCE)

        With IS (rollout_is enabled):
            L = -E_π_rollout[w * log π(a|s) * A(s,a)]
            where w = π_current / π_rollout (truncated IS weight)
            Gradient: ∇_θ L = -E[w * ∇log π(a|s) * A] (IS-corrected policy gradient)

    Args:
        rollout_log_prob: Log probabilities from rollout policy (e.g., vLLM BF16).
            Shape: (batch_size, seq_length)
        log_prob: Log probabilities from current training policy.
            Shape: (batch_size, seq_length)
        advantages: Advantage estimates for each token.
            Shape: (batch_size, seq_length)
        eos_mask: Mask indicating valid tokens (1 for valid, 0 for padding).
            Shape: (batch_size, seq_length)
        loss_agg_mode: Loss aggregation strategy (see agg_loss for details).
        loss_scale_factor: Multiplicative scaling factor applied to final loss.
        rollout_is: IS aggregation level ("token", "sequence", or None).
        rollout_is_threshold: Upper threshold for truncating IS weights.
        rollout_rs: Rejection sampling aggregation level (or None to disable).
        rollout_rs_threshold: Upper threshold for rejection sampling.
        rollout_rs_threshold_lower: Lower threshold for rejection sampling.
        rollout_token_veto_threshold: Per-token veto threshold for catastrophic outliers.
        rollout_is_batch_normalize: Whether to normalize IS weights to have mean=1.0 per batch.

    Note:
        Unlike compute_policy_loss (PPO), this function:
        - Does NOT use PPO clipping (no old_log_prob needed)
        - Directly applies IS correction computed from current vs rollout
        - Computes IS/RS on-the-fly during training

    Usage:
        This function is called by the actor when:
        - bypass_mode=True (trainer uses rollout_log_prob as old_log_prob)
        - use_policy_gradient=True (actor uses this function instead of compute_policy_loss)

    Example config:
        algorithm:
          rollout_correction:
            bypass_mode: true
            use_policy_gradient: true
            rollout_is: "token"
            rollout_is_threshold: 2.0
            rollout_rs: "token"
            rollout_rs_threshold: 2.0
            rollout_rs_threshold_lower: 0.5

    """
    # Import rollout correction helper
    from verl.trainer.ppo.rollout_corr_helper import compute_rollout_correction_and_rejection_mask

    assert config is not None, "ActorConfig must be provided for rollout correction"

    # Compute IS weights and rejection mask on-the-fly
    # Use no_grad since weights are detached inside and metrics don't need gradients
    with torch.no_grad():
        rollout_is_weights_proto, modified_response_mask, rollout_metrics = (
            compute_rollout_correction_and_rejection_mask(
                old_log_prob=log_prob,  # Current policy
                rollout_log_prob=rollout_log_prob,  # Rollout policy
                response_mask=eos_mask,
                rollout_is=rollout_is,
                rollout_is_threshold=rollout_is_threshold,
                rollout_rs=rollout_rs,
                rollout_rs_threshold=rollout_rs_threshold,
                rollout_rs_threshold_lower=rollout_rs_threshold_lower,
                rollout_token_veto_threshold=rollout_token_veto_threshold,
                rollout_is_batch_normalize=rollout_is_batch_normalize,
            )
        )

    # Extract weights tensor from DataProto (or None if disabled)
    rollout_is_weights = rollout_is_weights_proto.batch["rollout_is_weights"] if rollout_is_weights_proto else None

    # Apply rejection mask (if RS is enabled)
    effective_mask = modified_response_mask if rollout_rs is not None else eos_mask

    # Compute pure policy gradient loss with IS correction
    # Standard REINFORCE: L = -E[log π(a|s) * A]
    # With IS: L = -E[w * log π(a|s) * A] where w = π_current / π_rollout
    #
    # Note: rollout_is_weights already contains w = π_current / π_rollout
    # So we apply it to the standard log-prob trick formula

    if rollout_is_weights is not None:
        # IS-corrected policy gradient: L = -E[stopgrad(w) · log π · A]
        pg_losses = -advantages * log_prob * rollout_is_weights
    else:
        # Standard REINFORCE: L = -E[log π · A]
        pg_losses = -advantages * log_prob

    # Aggregate loss (apply scale factor manually)
    pg_loss = (
        agg_loss(
            loss_mat=pg_losses,
            loss_mask=effective_mask,
            loss_agg_mode=loss_agg_mode,
            **config.global_batch_info,
        )
        * loss_scale_factor
    )

    # Compute KL divergence between current and rollout policy
    negative_approx_kl = log_prob - rollout_log_prob
    kl_divergence = verl_F.masked_mean(-negative_approx_kl, effective_mask)

    pg_metrics = rollout_metrics
    pg_metrics.update(
        {
            "actor/ppo_kl": kl_divergence.detach().item(),
        }
    )

    return pg_loss, pg_metrics


@register_policy_loss("rollout_correction")
def compute_policy_loss_rollout_correction_wrapper(
    old_log_prob: torch.Tensor,
    log_prob: torch.Tensor,
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    loss_agg_mode: str = "token-mean",
    config: Optional[ActorConfig] = None,
    rollout_is_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Wrapper for compute_policy_loss_with_rollout_correction to match PolicyLossFn interface.

    This function is used when algorithm.rollout_correction.use_policy_gradient=True.
    In this mode, the trainer has already set old_log_prob=rollout_log_prob (bypass mode).

    Args:
        old_log_prob: In bypass mode, this is actually rollout_log_prob
        log_prob: Current policy log probabilities
        advantages: Advantage estimates
        response_mask: Valid token mask
        loss_agg_mode: Loss aggregation mode
        config: Actor config containing rollout_correction settings
        rollout_is_weights: Pre-computed IS weights (ignored, computed internally)
    """
    assert config is not None, "config is required for rollout_correction loss mode"

    # Extract rollout_correction config
    # In ray_trainer, when use_policy_gradient=True, the rollout_correction config
    # is embedded in actor config's policy_loss field
    rollout_corr_config = config.policy_loss.get("rollout_correction", None) if hasattr(config, "policy_loss") else None

    if rollout_corr_config is None:
        raise ValueError(
            "rollout_correction config not found in policy_loss. "
            "When using loss_mode='rollout_correction', ensure rollout_correction config is passed."
        )

    # Extract parameters
    rollout_is = rollout_corr_config.get("rollout_is", None)
    rollout_is_threshold = rollout_corr_config.get("rollout_is_threshold", 2.0)
    rollout_rs = rollout_corr_config.get("rollout_rs", None)
    rollout_rs_threshold = rollout_corr_config.get("rollout_rs_threshold", None)
    rollout_rs_threshold_lower = rollout_corr_config.get("rollout_rs_threshold_lower", None)
    rollout_token_veto_threshold = rollout_corr_config.get("rollout_token_veto_threshold", None)
    rollout_is_batch_normalize = rollout_corr_config.get("rollout_is_batch_normalize", False)

    # Call the actual implementation
    # In bypass mode, old_log_prob IS rollout_log_prob
    return compute_policy_loss_with_rollout_correction(
        rollout_log_prob=old_log_prob,  # This is rollout_log_prob in bypass mode
        log_prob=log_prob,
        advantages=advantages,
        eos_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        loss_scale_factor=1.0,
        rollout_is=rollout_is,
        rollout_is_threshold=rollout_is_threshold,
        rollout_rs=rollout_rs,
        rollout_rs_threshold=rollout_rs_threshold,
        rollout_rs_threshold_lower=rollout_rs_threshold_lower,
        rollout_token_veto_threshold=rollout_token_veto_threshold,
        rollout_is_batch_normalize=rollout_is_batch_normalize,
    )
