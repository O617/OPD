# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # DCTV (opd_adv_mode='dctv') controller state. Persists across
        # ``update_policy`` calls within one run; not serialized to checkpoints
        # (first few steps after resume behave like pure sign until EMAs warm up,
        # which is acceptable).
        #   c            -- c_t currently applied by this and subsequent
        #                    microbatches; deterministic on all ranks.
        #   D_ema, M_ema -- EMAs updated after each optimizer step. ``None``
        #                    before the first update; first sample copies in.
        #   step         -- Count of optimizer steps taken under 'dctv' mode
        #                    (used for warmup and debug).
        self._dctv_state = {
            "c": 1.0,
            "D_ema": None,
            "M_ema": None,
            "step": 0,
        }

        # TV-guided LR scheduler state (independent of DCTV; heuristic
        # multiplicative annealing of the optimizer LR by
        #   c_t = clip((D̄_{t-1} + ε) / (D_ref + ε))^α, c_min, 1.0)
        # where D̄_t is a β-EMA of D̂_t and D_ref is captured at the first valid
        # step so c_0 = 1). Reuses the same per-microbatch D̂ estimator emitted
        # by ``compute_policy_loss_opd`` (see ``opd_dctv_D_sum/D_count``); the
        # scheduler does NOT require opd_adv_mode=dctv and in fact rejects it
        # (mutually exclusive — both scale globally).
        #   c        -- multiplier applied to optimizer LR at this and
        #                subsequent optimizer steps; deterministic on all ranks.
        #   D_ema    -- β-EMA of D̂_t. ``None`` until first valid step.
        #   D_ref    -- Baseline D̄ captured at first valid step; frozen after.
        #   step     -- Count of scheduler optimizer steps (for debug).
        self._tv_sched_state = {
            "c": 1.0,
            "D_ema": None,
            "D_ref": None,
            "step": 0,
        }

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    logits_rmpad.div_(temperature)

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                        else:
                            entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                self.compute_entropy_from_logits, logits_rmpad
                            )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            return entropy, log_probs

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # ---- DCTV controller sanity check ----
        # ``dctv`` mode assumes grad_norm captures the OPD gradient only (up to
        # scaling by c). Extra loss terms (KL to ref, entropy bonus) would leak
        # into grad_norm and bias M̂. Reject them explicitly rather than silently
        # producing a mis-calibrated c_t.
        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
        pl_cfg = self.config.policy_loss
        opd_adv_mode = str(pl_cfg.get("opd_adv_mode", "raw"))
        dctv_active = loss_mode == "opd" and opd_adv_mode == "dctv"
        if dctv_active:
            if self.config.use_kl_loss:
                raise ValueError(
                    "opd_adv_mode='dctv' is incompatible with actor.use_kl_loss=True: "
                    "the KL-to-ref term would contaminate grad_norm, biasing the DCTV "
                    "motion estimator M̂. Disable use_kl_loss or pick another mode."
                )
            if self.config.entropy_coeff != 0:
                raise ValueError(
                    f"opd_adv_mode='dctv' is incompatible with actor.entropy_coeff="
                    f"{self.config.entropy_coeff} != 0: the entropy bonus would "
                    "contaminate grad_norm. Set entropy_coeff=0 or pick another mode."
                )

        # ---- TV-guided LR scheduler sanity check ----
        # Reuses the D̂ estimator emitted by compute_policy_loss_opd (loss_mode
        # == "opd") for all opd_adv_mode values including 'sign'. Rejects DCTV
        # concurrently since both would multiplicatively scale the update.
        tv_sched_enabled = bool(pl_cfg.get("opd_tv_sched_enabled", False))
        tv_sched_active = tv_sched_enabled and loss_mode == "opd"
        if tv_sched_enabled and loss_mode != "opd":
            raise ValueError(
                f"opd_tv_sched_enabled=True requires policy_loss.loss_mode='opd' "
                f"(got '{loss_mode}'). The scheduler consumes D̂ from the OPD loss."
            )
        if tv_sched_active and dctv_active:
            raise ValueError(
                "opd_tv_sched_enabled=True is mutually exclusive with "
                "opd_adv_mode='dctv': both introduce a multiplicative global "
                "step scaling and stacking them is not defined."
            )

        # TV-scheduler target: 'lr' scales optimizer LR by c_t (c enters after
        # Adam's m/v normalization → directly multiplies Δθ). 'adv' scales the
        # policy-loss coefficient by c_t (c enters before Adam → g_t, m, v all
        # carry c; Adam's √v largely cancels c). Kept as a config knob so we
        # can sweep both semantics and disentangle the trust-ratio effect from
        # Adam-normalization effects. Anything else is a config error.
        _tv_sched_target = str(pl_cfg.get("opd_tv_sched_target", "lr"))
        if tv_sched_active and _tv_sched_target not in ("lr", "adv"):
            raise ValueError(
                f"opd_tv_sched_target must be 'lr' or 'adv' (got {_tv_sched_target!r})"
            )

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {}
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                # DCTV per-mini-batch D̂ accumulator (local to this rank).
                # Reset per mini-batch so each _optimizer_step gets exactly the
                # microbatches that contributed to *its* gradient. DP reduce
                # happens after the loop, before EMA update.
                _dctv_D_sum_local = 0.0
                _dctv_D_count_local = 0

                # TV-scheduler per-mini-batch D̂ accumulator. Same estimator
                # source as DCTV (emitted unconditionally by
                # compute_policy_loss_opd whenever _n_eff > 0); we keep a
                # separate pair so a single active mode is unambiguous.
                _tv_sched_D_sum_local = 0.0
                _tv_sched_D_count_local = 0

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    entropy, log_prob = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # # DEBUG: stash input_ids/responses so compute_policy_loss_opd can print them
                    # from verl.trainer.ppo import core_algos as _ca
                    # _ca._DEBUG_INPUT_IDS = model_inputs.get("input_ids", None)
                    # _ca._DEBUG_RESPONSES = model_inputs.get("responses", None)

                    # DCTV: pass the past-conditioned c_t only for the OPD loss.
                    # Other loss fns don't accept this kwarg.
                    extra_loss_kwargs = {}
                    if loss_mode == "opd":
                        extra_loss_kwargs["dctv_c_current"] = float(self._dctv_state["c"])

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                        **extra_loss_kwargs,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # TV-scheduler (target='adv'): multiply the policy-loss
                    # coefficient by c_t on EVERY micro-batch of this optimizer
                    # step. Uses self._tv_sched_state["c"] (past-conditioned:
                    # frozen before the step, updated after). This is the
                    # advantage-scale variant — the c factor propagates into
                    # g_t, m_t, v_t so Adam's √v̂ largely cancels it. The
                    # target='lr' branch below leaves pg_loss untouched and
                    # scales optimizer.param_groups[i]['lr'] instead.
                    if tv_sched_active and _tv_sched_target == "adv":
                        pg_loss = pg_loss * float(self._tv_sched_state["c"])

                    # Accumulate DCTV D̂ (sum + count on eff_mask, this rank) so
                    # we can DP-reduce a mini-batch-mean D̂ after the optimizer
                    # step. pg_metrics carries per-microbatch scalars already.
                    if dctv_active:
                        _dctv_D_sum_local += float(pg_metrics.get("actor/opd_dctv_D_sum", 0.0))
                        _dctv_D_count_local += int(pg_metrics.get("actor/opd_dctv_D_count", 0))

                    if tv_sched_active:
                        _tv_sched_D_sum_local += float(pg_metrics.get("actor/opd_dctv_D_sum", 0.0))
                        _tv_sched_D_count_local += int(pg_metrics.get("actor/opd_dctv_D_count", 0))


                    # Skip if using pure rollout correction mode (metrics already in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "rollout_correction" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        micro_batch_metrics["actor/kl_loss"] = kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    micro_batch_metrics["actor/pg_loss"] = pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                # TV-scheduler (target='lr'): scale LR down by c_used for
                # this optimizer step, then restore. Scaling the LR (not the
                # advantage) puts c AFTER Adam's per-parameter normalization,
                # i.e. it multiplies m̂ / (√v̂ + ε) without changing the
                # running moment estimates. target='adv' takes the other
                # branch: c is already multiplied into pg_loss above, so LR
                # here is left untouched.
                _tv_sched_base_lrs: list[float] = []
                _tv_sched_c_used = float(self._tv_sched_state["c"])
                if tv_sched_active and _tv_sched_target == "lr":
                    _tv_sched_base_lrs = [
                        float(group["lr"]) for group in self.actor_optimizer.param_groups
                    ]
                    for group, lr in zip(self.actor_optimizer.param_groups, _tv_sched_base_lrs):
                        group["lr"] = lr * _tv_sched_c_used

                grad_norm = self._optimizer_step()

                if tv_sched_active and _tv_sched_target == "lr":
                    for group, lr in zip(self.actor_optimizer.param_groups, _tv_sched_base_lrs):
                        group["lr"] = lr

                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}

                # ---- DCTV controller update (once per optimizer step) ----
                if dctv_active:
                    # (1) DP-reduce D̂: sum + count across all DP ranks so the
                    # controller state is bit-identical on every rank (a
                    # requirement for keeping c_t deterministic).
                    _reduce_buf = torch.tensor(
                        [_dctv_D_sum_local, float(_dctv_D_count_local)],
                        dtype=torch.float64,
                        device=get_device_id(),
                    )
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(
                            _reduce_buf, op=torch.distributed.ReduceOp.SUM
                        )
                    _D_sum = _reduce_buf[0].item()
                    _D_count = _reduce_buf[1].item()
                    _D_hat = _D_sum / max(_D_count, 1.0)

                    # Cached state prior to any update this step (for logging /
                    # G_tv reconstruction).
                    _c_used = float(self._dctv_state["c"])
                    _eps = float(pl_cfg.get("opd_dctv_eps", 1e-8))
                    _lr = float(self.actor_optimizer.param_groups[0]["lr"])
                    _G_scaled_finite = bool(torch.isfinite(grad_norm).item())
                    _G_scaled = float(grad_norm.detach().item()) if _G_scaled_finite else float("nan")

                    # (2) Recover pure-TV grad norm and (3) predicted one-step TV
                    # motion. Both are ONLY meaningful when grad_norm is finite;
                    # in the non-finite branch we log NaN to avoid contaminating
                    # the wandb time series with junk.
                    if _G_scaled_finite:
                        _G_tv = _G_scaled / max(_c_used, _eps)
                        _M_hat = 0.5 * _lr * (_G_tv ** 2)
                    else:
                        _G_tv = float("nan")
                        _M_hat = float("nan")

                    # (4) EMA / c update — skipped on non-finite grad_norm
                    # (mirrors the "skip optimizer step" branch in
                    # _optimizer_step). _c_next defaults to the current c so the
                    # controller freezes rather than drifting.
                    _c_next = _c_used
                    _R = float("nan")
                    _nonfinite_flag = 0
                    if _G_scaled_finite:
                        _beta_d = float(pl_cfg.get("opd_dctv_beta_d", 0.95))
                        _beta_m = float(pl_cfg.get("opd_dctv_beta_m", 0.95))
                        _c_cap = float(pl_cfg.get("opd_dctv_c_cap", 1.0))

                        st = self._dctv_state
                        st["D_ema"] = (
                            _D_hat
                            if st["D_ema"] is None
                            else _beta_d * st["D_ema"] + (1.0 - _beta_d) * _D_hat
                        )
                        st["M_ema"] = (
                            _M_hat
                            if st["M_ema"] is None
                            else _beta_m * st["M_ema"] + (1.0 - _beta_m) * _M_hat
                        )
                        st["step"] += 1

                        _R = st["D_ema"] / (st["M_ema"] + _eps)
                        _c_next = min(_c_cap, _R)
                        st["c"] = _c_next
                    else:
                        _nonfinite_flag = 1

                    mini_batch_metrics.update({
                        # D̂ / M̂ raw samples this step
                        "actor/opd_dctv_D_hat": _D_hat,
                        "actor/opd_dctv_D_count": int(_D_count),
                        "actor/opd_dctv_G_scaled": _G_scaled,
                        "actor/opd_dctv_G_tv": _G_tv,
                        "actor/opd_dctv_M_hat": _M_hat,
                        # EMA-smoothed state after this step's update
                        "actor/opd_dctv_D_ema": (
                            float(self._dctv_state["D_ema"])
                            if self._dctv_state["D_ema"] is not None
                            else 0.0
                        ),
                        "actor/opd_dctv_M_ema": (
                            float(self._dctv_state["M_ema"])
                            if self._dctv_state["M_ema"] is not None
                            else 0.0
                        ),
                        # Controller output. R is the primary diagnostic
                        # (method §7): early/mid should be >>1, tail should
                        # approach 1 concurrent with instability onset.
                        "actor/opd_dctv_R": float(_R),
                        "actor/opd_dctv_c_used": _c_used,
                        "actor/opd_dctv_c_next": float(_c_next),
                        # Runtime state
                        "actor/opd_dctv_lr": _lr,
                        "actor/opd_dctv_step": int(self._dctv_state["step"]),
                        "actor/opd_dctv_nonfinite_grad_norm": _nonfinite_flag,
                    })

                # ---- TV-scheduler controller update (once per optimizer step) ----
                # Past-conditioned by design: this step consumed c=self._tv_sched_state["c"];
                # after this step we compute D̂_t → D̄_t → c_{t+1} for the next
                # optimizer step. On non-finite grad_norm we freeze the state
                # (skip both D_ref bootstrap and EMA update) to mirror the
                # optimizer's skip-update path in _optimizer_step().
                if tv_sched_active:
                    _sched_reduce_buf = torch.tensor(
                        [_tv_sched_D_sum_local, float(_tv_sched_D_count_local)],
                        dtype=torch.float64,
                        device=get_device_id(),
                    )
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(
                            _sched_reduce_buf, op=torch.distributed.ReduceOp.SUM
                        )
                    _sched_D_sum = _sched_reduce_buf[0].item()
                    _sched_D_count = _sched_reduce_buf[1].item()
                    _sched_D_hat = _sched_D_sum / max(_sched_D_count, 1.0)

                    _sched_grad_finite = bool(torch.isfinite(grad_norm).item())
                    # lr_base / lr_effective semantics:
                    #   target='lr': _tv_sched_base_lrs was populated pre-step;
                    #     lr_effective = lr_base · c_used (what actually ran).
                    #   target='adv': we didn't touch the optimizer LR, so
                    #     lr_base is the untouched lr and lr_effective ≡ lr_base.
                    #     The c factor lived in pg_loss, not lr; wandb consumers
                    #     should read opd_tv_sched_c_used + target when reading
                    #     these two columns for the adv variant.
                    _sched_lr_base = (
                        float(_tv_sched_base_lrs[0])
                        if _tv_sched_base_lrs
                        else float(self.actor_optimizer.param_groups[0]["lr"])
                    )
                    if _tv_sched_target == "lr":
                        _sched_lr_effective = _sched_lr_base * _tv_sched_c_used
                    else:
                        _sched_lr_effective = _sched_lr_base

                    _sched_beta = float(pl_cfg.get("opd_tv_sched_beta", 0.95))
                    _sched_alpha = float(pl_cfg.get("opd_tv_sched_alpha", 1.0))
                    _sched_c_min = float(pl_cfg.get("opd_tv_sched_c_min", 0.1))
                    _sched_eps = 1e-8

                    _sched_c_next = float(self._tv_sched_state["c"])
                    _sched_ratio = float("nan")
                    _sched_nonfinite_flag = 0
                    if _sched_grad_finite:
                        st = self._tv_sched_state
                        st["D_ema"] = (
                            _sched_D_hat
                            if st["D_ema"] is None
                            else _sched_beta * st["D_ema"] + (1.0 - _sched_beta) * _sched_D_hat
                        )
                        if st["D_ref"] is None:
                            # Anchor at the first valid step so c_0 = 1 exactly.
                            st["D_ref"] = float(st["D_ema"])
                        st["step"] += 1

                        _sched_ratio = (
                            (st["D_ema"] + _sched_eps) / (st["D_ref"] + _sched_eps)
                        )
                        _sched_c_next = _sched_ratio ** _sched_alpha
                        _sched_c_next = min(1.0, max(_sched_c_min, _sched_c_next))
                        st["c"] = _sched_c_next
                    else:
                        _sched_nonfinite_flag = 1

                    mini_batch_metrics.update({
                        "actor/opd_tv_sched_D_hat": _sched_D_hat,
                        "actor/opd_tv_sched_D_count": int(_sched_D_count),
                        "actor/opd_tv_sched_D_ema": (
                            float(self._tv_sched_state["D_ema"])
                            if self._tv_sched_state["D_ema"] is not None
                            else 0.0
                        ),
                        "actor/opd_tv_sched_D_ref": (
                            float(self._tv_sched_state["D_ref"])
                            if self._tv_sched_state["D_ref"] is not None
                            else 0.0
                        ),
                        "actor/opd_tv_sched_D_ratio": float(_sched_ratio),
                        "actor/opd_tv_sched_c_used": float(_tv_sched_c_used),
                        "actor/opd_tv_sched_c_next": float(_sched_c_next),
                        "actor/opd_tv_sched_lr_base": _sched_lr_base,
                        "actor/opd_tv_sched_lr_effective": _sched_lr_effective,
                        "actor/opd_tv_sched_step": int(self._tv_sched_state["step"]),
                        "actor/opd_tv_sched_nonfinite_grad_norm": _sched_nonfinite_flag,
                        # Hparam echoes (constant per run) for wandb sanity.
                        "actor/opd_tv_sched_alpha": _sched_alpha,
                        "actor/opd_tv_sched_beta": _sched_beta,
                        "actor/opd_tv_sched_c_min": _sched_c_min,
                        # target='lr' → 0, target='adv' → 1. Simple int so wandb
                        # can group and split lines by scheduler semantics.
                        "actor/opd_tv_sched_target_is_adv": (
                            1 if _tv_sched_target == "adv" else 0
                        ),
                    })

                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
