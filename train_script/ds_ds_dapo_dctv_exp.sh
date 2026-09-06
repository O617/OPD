#!/usr/bin/env bash
# ============================================================================
# DS-Qwen -> DS-Distill-Qwen  |  DAPO-17k math  |  DCTV-OPD (Adaptive
# Distance-Calibrated TVOPD)
# ----------------------------------------------------------------------------
# Teacher : JustRL-DeepSeek-1.5B (Qwen2ForCausalLM, GRPO-trained), served at
#           TEACHER_SERVER_IP:TEACHER_SERVER_PORT (ZMQ, --n-logprobs 1, tp=2, 4 workers)
# Student : DeepSeek-R1-Distill-Qwen-1.5B (Qwen2ForCausalLM)
# Train   : DAPO-17k (dapo-math-17k.parquet)
# Eval    : AIME24 + AIME25 (n=4 each)
#
# Same tokenizer (vocab 151936, Qwen2 chat template) — OPD reward manager takes
# the fast `_is_same_tokenizer()` branch (retokenize returns batch as-is).
#
# Method (from DCTV_OPD_method.md):
#     A_i,t = c_t * sign(Δ_i,t)
#     c_t   = min(c_cap, D̄_{t-1} / (M̄_{t-1} + ε))
#     D̂_t   = mean_i [1 - exp(Δ_i)]_+      (sequence-level TV distance est.)
#     M̂_t   = 0.5 * η_t * (G_t^TV)^2       (predicted one-step TV motion)
#     G_t^TV = G_t^scaled / max(c_t, ε)     (undo global c scaling in grad_norm)
#
# c_0 = 1 (bootstrap: first step ≡ pure sign). EMAs adopt first sample as
# initial value (no explicit warmup hparam). Estimator smoothing constants
# β_D = β_M = 0.95 fixed. c_cap = 1.0 = "don't first-order overshoot pure TV".
# ε = 1e-8 numerical floor.
#
# Compatibility guards enforced in DataParallelPPOActor.update_policy for
# ADV_MODE=dctv:
#   * use_kl_loss=False (KL-to-ref grad would contaminate grad_norm → biased M̂)
#   * entropy_coeff=0    (same reason)
# Set explicitly below so the run never silently disables the check.
# ============================================================================
set -xeuo pipefail

# ---- Activate venv (editable-linked to this repo) ----
REPO_ROOT=${REPO_ROOT:-"$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel 2>/dev/null)"}
VENV_DIR=${VENV_DIR:-"${REPO_ROOT}/.venv"}
# Direct PATH export instead of `source activate` — the activate script hard-codes
# VIRTUAL_ENV and breaks if the venv has been moved between clusters.
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"
hash -r

# ---- ulimit for RDMA & Ray raylet ----
ulimit -l unlimited
ulimit -n 65536

# ---- NCCL InfiniBand / RoCE configuration ----
# HCA / ifname defaults assume a cluster exposing aggregated `mlx5_bond_*`
# devices behind `bond1`. Override NCCL_IB_HCA / NCCL_SOCKET_IFNAME (and
# related NVSHMEM_* vars) for clusters exposing per-GPU HCAs (e.g. ibpXXsY)
# or a different ethernet iface.
export NCCL_IB_TC=160
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_HCA=${NCCL_IB_HCA:-mlx5_bond_1,mlx5_bond_2,mlx5_bond_3,mlx5_bond_4,mlx5_bond_5,mlx5_bond_6,mlx5_bond_7,mlx5_bond_8}
export NCCL_IB_TIMEOUT=22
export NCCL_IB_SL=3
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-bond1}
export NCCL_DMABUF_ENABLE=0
export NCCL_NET_GDR_LEVEL=LOC
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-bond1}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-bond1}
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-bond1}
export NVSHMEM_HCA_LIST=${NVSHMEM_HCA_LIST:-mlx5_bond_1:1,mlx5_bond_2:1,mlx5_bond_3:1,mlx5_bond_4:1,mlx5_bond_5:1,mlx5_bond_6:1,mlx5_bond_7:1,mlx5_bond_8:1}
export NVSHMEM_IB_TRAFFIC_CLASS=160
export NVSHMEM_IB_TIMEOUT=22
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

# vllm+flashinfer version mismatch bypass
export FLASHINFER_DISABLE_VERSION_CHECK=1

# Optional HTTP proxy (set HTTP_PROXY_URL to enable). Teacher IP + any
# intra-cluster hosts must land in NO_PROXY, else ZMQ REQ goes through the
# proxy and hangs.
if [[ -n "${HTTP_PROXY_URL:-}" ]]; then
    export HTTPS_PROXY="${HTTP_PROXY_URL}"
    export HTTP_PROXY="${HTTP_PROXY_URL}"
    export https_proxy="${HTTP_PROXY_URL}"
    export http_proxy="${HTTP_PROXY_URL}"
fi
_default_no_proxy="localhost,127.0.0.1"
if [[ -n "${TEACHER_SERVER_IP:-}" ]]; then
    _default_no_proxy+=",${TEACHER_SERVER_IP}"
fi
if [[ -n "${NO_PROXY_EXTRA:-}" ]]; then
    _default_no_proxy+=",${NO_PROXY_EXTRA}"
fi
export NO_PROXY="${NO_PROXY:-${_default_no_proxy}}"
export no_proxy="${NO_PROXY}"

export WANDB_MODE=online
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}

project_name='ON_POLICY_DISTILL'

# ---- DCTV knobs (defaults match DCTV_OPD_method.md §7) ----
# β_D / β_M are estimator smoothing constants, not method hparams.
BETA_D=${BETA_D:-0.95}
BETA_M=${BETA_M:-0.95}
DCTV_EPS=${DCTV_EPS:-1e-8}
# c_cap = 1.0 is the natural "don't overshoot pure TV" cap; keep it default.
C_CAP=${C_CAP:-1.0}

SEED=${SEED:-63}
exp_name="OPD_DS_DS_DAPO_adv_dctv_seed${SEED}"

adv_estimator=opd

use_kl_in_reward=False
kl_coef=0.0
# HARD constraint for DCTV: use_kl_loss=False (KL-to-ref grad would leak into
# grad_norm → biased M̂). update_policy() will raise if this is True.
use_kl_loss=False
kl_loss_coef=0.0

clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 1))
max_response_length=$((1024 * 16))

loss_agg_mode="token-mean"

train_prompt_bsz=64
n_resp_per_prompt=1
train_prompt_mini_bsz=64

# Ray — each run manages a local single-node Ray head.
RAY_ADDRESS=${RAY_ADDRESS:-"127.0.0.1:6379"}
RAY_ADDRESS="${RAY_ADDRESS#http://}"
RAY_ADDRESS="${RAY_ADDRESS#https://}"
RAY_ADDRESS="${RAY_ADDRESS#ray://}"
export RAY_ADDRESS
NNODES=1
NGPUS_PER_NODE=8

if ! ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1; then
    echo "Ray head not found at ${RAY_ADDRESS}; starting a new head." >&2
    ray_port="${RAY_ADDRESS##*:}"
    ray start --head --port="${ray_port}" \
        --node-ip-address=127.0.0.1 \
        --num-gpus="${NGPUS_PER_NODE}" \
        --disable-usage-stats
    sleep 5
    ray status --address="${RAY_ADDRESS}" >/dev/null 2>&1 || {
        echo "ERROR: Ray head failed to start at ${RAY_ADDRESS}" >&2
        exit 1
    }
fi

# Paths (all required — no cluster-specific defaults)
: "${REPO_ROOT:?REPO_ROOT must be set (repo root, used for ckpts)}"
: "${DATA_ROOT:?DATA_ROOT must be set (contains DAPO-17k parquet + val_all/)}"
: "${TEACHER_SERVER_IP:?TEACHER_SERVER_IP must be set}"
: "${TEACHER_CKPT_PATH:?TEACHER_CKPT_PATH must be set (student tokenizer path for retokenize check)}"

RAY_DATA_HOME="${REPO_ROOT}"

# Student: DS-Distill-Qwen-1.5B (Qwen2ForCausalLM). If prewarmed to /root/,
# use the local copy; otherwise expect MODEL_PATH to be set explicitly.
if [[ -z "${MODEL_PATH:-}" ]]; then
    if [[ -d "/root/DS-Distill-Qwen-1.5B" ]]; then
        MODEL_PATH="/root/DS-Distill-Qwen-1.5B"
    else
        echo "ERROR: MODEL_PATH not set and /root/DS-Distill-Qwen-1.5B missing" >&2
        exit 1
    fi
fi

CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TRAIN_FILE=${TRAIN_FILE:-"${DATA_ROOT}/DAPO-17k/data/dapo-math-17k.parquet"}
TEST_FILE=${TEST_FILE:-"[${DATA_ROOT}/val_all/aime24.parquet,${DATA_ROOT}/val_all/aime25.parquet]"}

# Teacher: served at TEACHER_SERVER_IP:TEACHER_SERVER_PORT.
# TEACHER_CKPT_PATH is loaded by OPD reward manager to build teacher tokenizer;
# when teacher == student tokenizer, `_is_same_tokenizer()` returns True and the
# retokenize step is a no-op.
export TEACHER_SERVER_IP
export TEACHER_SERVER_PORT="${TEACHER_SERVER_PORT:-15555}"
export TEACHER_N_WORKERS="${TEACHER_N_WORKERS:-1}"
export TEACHER_CKPT_PATH
export TEACHER_MAX_SEQ_LEN="${TEACHER_MAX_SEQ_LEN:-30720}"
export HYDRA_FULL_ERROR=1
export RAY_DEBUG=legacy

# Algorithm
temperature=1.0
top_p=1.0
top_k=-1

val_top_p=0.95
val_top_k=20
val_temperature=0.6

# Performance Related Parameter
sp_size=2
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=True
gen_tp=2
fsdp_size=8

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.shuffle=True \
    data.seed=${SEED} \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.policy_loss.opd_adv_mode=dctv \
    actor_rollout_ref.actor.policy_loss.opd_dctv_beta_d=${BETA_D} \
    actor_rollout_ref.actor.policy_loss.opd_dctv_beta_m=${BETA_M} \
    actor_rollout_ref.actor.policy_loss.opd_dctv_eps=${DCTV_EPS} \
    actor_rollout_ref.actor.policy_loss.opd_dctv_c_cap=${C_CAP} \
    actor_rollout_ref.model.use_remove_padding=True \
    +actor_rollout_ref.model.override_config.max_position_embeddings=32768 \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.actor.policy_loss.loss_mode="opd" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.01 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.nccl_timeout=72000 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.90 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${val_temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${val_top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=4 \
    actor_rollout_ref.rollout.val_kwargs.max_tokens=31744 \
    actor_rollout_ref.ref.fsdp_config.param_offload=${offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    actor_rollout_ref.ref.fsdp_config.fsdp_size=${fsdp_size} \
    reward_model.reward_manager=opd \
    reward_model.enable=False \
    trainer.logger='["console","wandb"]' \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=25 \
    trainer.save_freq=500 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=2000 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10
