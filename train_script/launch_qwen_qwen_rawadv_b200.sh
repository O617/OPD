#!/usr/bin/env bash
# ============================================================================
# One-click launcher for the "raw adv" experiment set.
# ----------------------------------------------------------------------------
# Fans out N single-node student runs over SSH, all sharing one remote teacher
# server. Nodes are read from a fleet file (see FLEET_FILE below); each line
# is: "<host_or_ip> <label> <dataset> <seed>". Example:
#
#     10.0.0.1  launcher deepmath 42
#     10.0.0.2  worker-0 deepmath 63
#     10.0.0.3  worker-1 dapo     42
#     10.0.0.4  worker-2 dapo     63
#
# Required env:
#   WANDB_API_KEY       — wandb login token (propagated to remotes)
#   TEACHER_SERVER_IP   — remote teacher server IP
#   DATA_ROOT           — dir on each student containing *_messages.parquet
#   MODEL_PATH          — student init ckpt path on each student
#
# Optional env:
#   REPO_ROOT           — absolute repo path (default: this repo)
#   FLEET_FILE          — path to fleet layout file
#                          (default: <repo>/train_script/fleet.txt)
#   TEACHER_SERVER_PORT — default 15555
#   TEACHER_N_WORKERS   — default 8
#   STAGGER             — seconds between successive student launches
#                          (default 120; avoids all students hammering
#                           val_before_train simultaneously)
#   HTTP_PROXY_URL      — optional outbound proxy for wandb egress
#   NO_PROXY_EXTRA      — extra CIDRs for NO_PROXY
#   Perf knobs (see README): SP_SIZE FSDP_SIZE GEN_TP OFFLOAD GPU_MEM_UTIL
#                            VAL_BEFORE_TRAIN
#
# Usage:
#   export WANDB_API_KEY=...
#   export TEACHER_SERVER_IP=...
#   export DATA_ROOT=... MODEL_PATH=...
#   bash train_script/launch_qwen_qwen_rawadv_b200.sh          # launch all
#   bash train_script/launch_qwen_qwen_rawadv_b200.sh check    # snapshot
#   bash train_script/launch_qwen_qwen_rawadv_b200.sh kill     # stop all
# ============================================================================
set -euo pipefail

: "${WANDB_API_KEY:?WANDB_API_KEY must be exported before launching}"
: "${TEACHER_SERVER_IP:?TEACHER_SERVER_IP must be exported before launching}"
: "${DATA_ROOT:?DATA_ROOT must be exported before launching}"
: "${MODEL_PATH:?MODEL_PATH must be exported before launching}"

REPO_ROOT=${REPO_ROOT:-"$(git -C "$(dirname "${BASH_SOURCE[0]}")" rev-parse --show-toplevel)"}
SCRIPT=${REPO_ROOT}/train_script/qwen_qwen_rawadv_b200_exp.sh
FLEET_FILE=${FLEET_FILE:-${REPO_ROOT}/train_script/fleet.txt}

TEACHER_IP=${TEACHER_SERVER_IP}
TEACHER_PORT=${TEACHER_SERVER_PORT:-15555}
TEACHER_WORKERS=${TEACHER_N_WORKERS:-8}
STAGGER=${STAGGER:-120}

# Perf-tuned defaults for the B200 tune sweep (see README):
#   SP=1 FSDP=2 GEN_TP=4 OFFLOAD=False GPU_MEM_UTIL=0.65 VAL=True
#   -> step ~157s, val_before_train ~10 min, no OOM (peak ~168/179 GiB)
SP_SIZE=${SP_SIZE:-1}
FSDP_SIZE=${FSDP_SIZE:-2}
GEN_TP=${GEN_TP:-4}
OFFLOAD=${OFFLOAD:-False}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.65}
VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN:-True}

if [[ ! -f "${FLEET_FILE}" ]]; then
    cat >&2 <<EOF
ERROR: fleet file not found: ${FLEET_FILE}

Create one with lines: "<host> <label> <dataset> <seed>". See
${REPO_ROOT}/train_script/fleet.example.txt for a template.
EOF
    exit 1
fi

# Read fleet (skip blank + comment lines)
RUNS=()
while IFS= read -r line; do
    line="${line%%#*}"
    line="$(echo "$line" | xargs)"
    [[ -z "$line" ]] && continue
    RUNS+=("$line")
done < "${FLEET_FILE}"

if [[ ${#RUNS[@]} -eq 0 ]]; then
    echo "ERROR: no runs parsed from ${FLEET_FILE}" >&2
    exit 1
fi

# ----------------------------------------------------------------------------
launch_students() {
    echo ">>> Teacher check ${TEACHER_IP}:${TEACHER_PORT}"
    if ! timeout 5 bash -c "exec 3<>/dev/tcp/${TEACHER_IP}/${TEACHER_PORT}" 2>/dev/null; then
        echo "ERROR: teacher ${TEACHER_IP}:${TEACHER_PORT} not reachable" >&2
        exit 1
    fi
    echo "    teacher reachable."

    local first=1
    for run in "${RUNS[@]}"; do
        read -r host label dataset seed <<<"$run"
        LOG=/tmp/qwen_qwen_${dataset}_rawadv_seed${seed}.log
        if [[ $first -eq 1 ]]; then
            first=0
        else
            echo ">>> Sleeping ${STAGGER}s before next stu (stagger)..."
            sleep "${STAGGER}"
        fi
        echo ">>> Launching ${label} (${host})  dataset=${dataset}  seed=${seed}"
        ssh -f -n -o ConnectTimeout=8 -o StrictHostKeyChecking=no "${host}" "cd ${REPO_ROOT} && nohup bash -c '
            export WANDB_API_KEY=${WANDB_API_KEY}
            export REPO_ROOT=${REPO_ROOT}
            export DATA_ROOT=${DATA_ROOT}
            export MODEL_PATH=${MODEL_PATH}
            export DATASET=${dataset}
            export SEED=${seed}
            export TEACHER_SERVER_IP=${TEACHER_IP}
            export TEACHER_SERVER_PORT=${TEACHER_PORT}
            export TEACHER_N_WORKERS=${TEACHER_WORKERS}
            export SP_SIZE=${SP_SIZE}
            export FSDP_SIZE=${FSDP_SIZE}
            export GEN_TP=${GEN_TP}
            export OFFLOAD=${OFFLOAD}
            export GPU_MEM_UTIL=${GPU_MEM_UTIL}
            export VAL_BEFORE_TRAIN=${VAL_BEFORE_TRAIN}
            export HTTP_PROXY_URL=${HTTP_PROXY_URL:-}
            export NO_PROXY_EXTRA=${NO_PROXY_EXTRA:-}
            bash ${SCRIPT}
        ' > ${LOG} 2>&1 &"
        echo "    log -> ${host}:${LOG}"
    done
    echo ">>> All ${#RUNS[@]} stu dispatched. First val_before_train ~10 min; step:1 shortly after."
}

check_status() {
    echo ">>> Teacher ${TEACHER_IP}:${TEACHER_PORT}: $(
        timeout 3 bash -c "exec 3<>/dev/tcp/${TEACHER_IP}/${TEACHER_PORT}" 2>/dev/null \
            && echo REACHABLE || echo UNREACHABLE
    )"
    printf "%-20s %-8s %-9s %-4s | %-30s | %s\n" HOST node dataset seed step gpu
    for run in "${RUNS[@]}"; do
        read -r host label dataset seed <<<"$run"
        LOG=/tmp/qwen_qwen_${dataset}_rawadv_seed${seed}.log
        out=$(timeout 8 ssh -o ConnectTimeout=4 -o StrictHostKeyChecking=no "${host}" "
            step=\$(grep -oE 'step:[0-9]+' ${LOG} 2>/dev/null | tail -1)
            aime=\$(grep -oE 'aime25[^ ]* mean@[0-9]+=[0-9.]+' ${LOG} 2>/dev/null | tail -1)
            gpu=\$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | awk '{s+=\$1}END{print s}')
            proc=\$(ps -eo pid,cmd --no-headers 2>/dev/null | grep -v grep | grep -c 'python3 -m verl.trainer.main_ppo')
            printf '%-30s | gpu=%sMiB proc=%s %s\n' \"\${step:-step:?}\" \"\${gpu:-?}\" \"\${proc}\" \"\${aime}\"
        " 2>&1)
        printf "%-20s %-8s %-9s %-4s | %s\n" "$host" "$label" "$dataset" "$seed" "$out"
    done
}

kill_all() {
    echo ">>> Stopping stu processes on all nodes"
    for run in "${RUNS[@]}"; do
        read -r host label _ _ <<<"$run"
        (
            ssh -o ConnectTimeout=6 -o StrictHostKeyChecking=no "${host}" "
                pkill -f verl.trainer.main_ppo 2>/dev/null || true
                pkill -f 'ray::' 2>/dev/null || true
                ray stop --force 2>/dev/null || true
                pkill -9 -f raylet 2>/dev/null || true
                pkill -9 -f gcs_server 2>/dev/null || true
                pkill -9 -f plasma 2>/dev/null || true
            " 2>&1 | sed "s#^#[${label}] #"
        ) &
    done
    wait
    echo ">>> All done."
}

case "${1:-students}" in
    students|all) launch_students ;;
    check)        check_status ;;
    kill|stop)    kill_all ;;
    *)
        echo "Usage: $0 {students|check|kill}" >&2
        exit 1
        ;;
esac
