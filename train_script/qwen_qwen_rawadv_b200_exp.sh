#!/usr/bin/env bash
# ============================================================================
# Qwen -> Qwen  |  OPD raw-adv baseline (NO sign, NO TV scheduler)
# ----------------------------------------------------------------------------
# Single-node student (8 GPUs), remote shared teacher server.
#
# Required env:
#   WANDB_API_KEY       — wandb login token
#   REPO_ROOT           — absolute path to this repo (defaults to script's git root)
#   VENV_DIR            — absolute path to the python venv  (defaults to $REPO_ROOT/.venv)
#   DATA_ROOT           — dir containing {deepmath,dapo_17k}_messages.parquet
#                          and val_all/{aime24,aime25}.parquet
#   MODEL_PATH          — student init ckpt path
#   TEACHER_SERVER_IP   — remote teacher server IP
#   TEACHER_SERVER_PORT — remote teacher server port (default 15555)
#
# Optional env:
#   DATASET             = deepmath | dapo         (default deepmath)
#   SEED                = 42 | 63 | ...           (default 42)
#   TEACHER_N_WORKERS   = 8                        (teacher server slots)
#   HTTP_PROXY_URL      — set to enable outbound HTTP proxy (e.g. wandb egress)
#   NO_PROXY_EXTRA      — extra CIDRs to append to NO_PROXY (comma-separated)
#
# Perf knobs (see README §Perf-tune sweep):
#   SP_SIZE FSDP_SIZE GEN_TP OFFLOAD GPU_MEM_UTIL VAL_BEFORE_TRAIN
# ============================================================================
set -xeuo pipefail

REPO_ROOT=${REPO_ROOT:-"$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"}
VENV_DIR=${VENV_DIR:-"${REPO_ROOT}/.venv"}

# NOTE: cannot `source .venv/bin/activate` when the venv was created under a
# different absolute path — the activate script hard-codes VIRTUAL_ENV and PATH
# ends up pointing at a non-existent dir. Export PATH+VIRTUAL_ENV directly.
export VIRTUAL_ENV="${VENV_DIR}"
export PATH="${VENV_DIR}/bin:${PATH}"
hash -r

ulimit -l unlimited
ulimit -n 65536

# NCCL / IB / socket ifname. We let NCCL auto-detect HCA (omit NCCL_IB_HCA) so
# this works both on clusters exposing aggregated `mlx5_bond_*` devices and on
# clusters exposing per-GPU `ibpXXXsY` HCAs. Override NCCL_SOCKET_IFNAME if
# your ethernet iface is not eth0.
export NCCL_IB_TC=160
export NCCL_IB_DISABLE=0
export NCCL_IB_GID_INDEX=3
export NCCL_IB_TIMEOUT=22
export NCCL_IB_SL=3
export NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-eth0}
export NCCL_DMABUF_ENABLE=0
export NCCL_NET_GDR_LEVEL=LOC
export UCX_NET_DEVICES=${UCX_NET_DEVICES:-eth0}
export GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-eth0}
export NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME=${NVSHMEM_BOOTSTRAP_UID_SOCK_IFNAME:-eth0}
export NVSHMEM_IB_TRAFFIC_CLASS=160
export NVSHMEM_IB_TIMEOUT=22
export NCCL_DEBUG=${NCCL_DEBUG:-INFO}

export FLASHINFER_DISABLE_VERSION_CHECK=1

# Optional HTTP proxy (set HTTP_PROXY_URL to enable). The teacher IP and any
# other intra-cluster hosts must land in NO_PROXY, otherwise the ZMQ REQ
# socket goes through the proxy and hangs.
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

export WANDB_MODE=${WANDB_MODE:-online}
export WANDB_INIT_TIMEOUT=${WANDB_INIT_TIMEOUT:-300}

project_name='ON_POLICY_DISTILL'

DATASET=${DATASET:-deepmath}
SEED=${SEED:-42}

: "${DATA_ROOT:?DATA_ROOT must be set (dir containing *_messages.parquet + val_all/)}"
case "${DATASET}" in
    deepmath)
        TRAIN_FILE="${DATA_ROOT}/deepmath_messages.parquet"
        DATASET_TAG="DeepMath"
        ;;
    dapo)
        TRAIN_FILE="${DATA_ROOT}/dapo_17k_messages.parquet"
        DATASET_TAG="DapoMath"
        ;;
    *)
        echo "ERROR: unknown DATASET=${DATASET} (expected: deepmath | dapo)" >&2
        exit 1
        ;;
esac

exp_name="${EXP_NAME:-OPD_Qwen_Qwen_${DATASET_TAG}_rawadv_seed${SEED}}"

adv_estimator=opd

use_kl_in_reward=False
kl_coef=0.0
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

: "${MODEL_PATH:?MODEL_PATH must be set (student init ckpt)}"

RAY_DATA_HOME="${REPO_ROOT}"
CKPTS_DIR=${CKPTS_DIR:-"${RAY_DATA_HOME}/ckpts/${project_name}/${exp_name}"}
TEST_FILE="[${DATA_ROOT}/val_all/aime24.parquet,${DATA_ROOT}/val_all/aime25.parquet]"

: "${TEACHER_SERVER_IP:?TEACHER_SERVER_IP must be set (remote teacher server)}"
export TEACHER_SERVER_IP
export TEACHER_SERVER_PORT="${TEACHER_SERVER_PORT:-15555}"
export TEACHER_N_WORKERS="${TEACHER_N_WORKERS:-8}"
export TEACHER_MAX_SEQ_LEN="${TEACHER_MAX_SEQ_LEN:-30720}"
export HYDRA_FULL_ERROR=1
export RAY_DEBUG=legacy

temperature=1.0
top_p=1.0
top_k=-1

val_top_p=0.95
val_top_k=20
val_temperature=0.6

sp_size=${SP_SIZE:-2}
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
offload=${OFFLOAD:-True}
gen_tp=${GEN_TP:-2}
fsdp_size=${FSDP_SIZE:-8}
val_before_train=${VAL_BEFORE_TRAIN:-True}



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
    actor_rollout_ref.actor.policy_loss.opd_adv_mode=raw \
    actor_rollout_ref.actor.policy_loss.opd_tv_sched_enabled=False \
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
    actor_rollout_ref.rollout.gpu_memory_utilization=${GPU_MEM_UTIL:-0.90} \
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
    trainer.val_before_train=${val_before_train} \
    trainer.test_freq=25 \
    trainer.save_freq=500 \
    trainer.total_epochs=10 \
    trainer.total_training_steps=2000 \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10
