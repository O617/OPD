#!/usr/bin/env bash
# ============================================================================
# DS-Qwen -> DS-Distill-Qwen  |  DAPO-17k  |  DCTV-OPD 4-seed main experiment
# ----------------------------------------------------------------------------
# Fires 4 seeds (42, 63, 7, 21) — matches the existing sign/raw seed sets so
# cross-method (method × seed) heatmaps line up 1-to-1.
#
# Layout (choose 4 worker nodes; edit RUNS below to match the current claim):
#     <node-a> : dctv seed=42
#     <node-b> : dctv seed=63
#     <node-c> : dctv seed=7
#     <node-d> : dctv seed=21
#
# All 4 runs use the DCTV_OPD_method.md defaults: β_D = β_M = 0.95, c_cap = 1.0,
# ε = 1e-8. Nothing is swept.
#
# Usage:
#   bash train_script/launch_ds_ds_dapo_dctv.sh          # dry-run (default)
#   bash train_script/launch_ds_ds_dapo_dctv.sh --go     # actually dispatch
# ============================================================================
set -euo pipefail

CLUSTER_PREFIX="$CLUSTER_PREFIX"
REPO_ROOT="$DATA_ROOT/develop/OPD"
SCRIPT="${REPO_ROOT}/train_script/ds_ds_dapo_dctv_exp.sh"
LOG_DIR="${REPO_ROOT}/train_script/outputs/ds_ds_dapo_dctv_$(date +%Y%m%d_%H%M)"

# (node-short, seed) tuples. Adjust node-shorts to whichever slots are free
# when you launch. Seeds fixed at the canonical 4 (42, 63, 7, 21).
RUNS=(
    "worker-0:42"
    "worker-1:63"
    "worker-2:7"
    "worker-3:21"
)

MODE="dry"
case "${1:-}" in
    --go) MODE="go" ;;
    "")   ;;
    *)    echo "Unknown flag: $1" >&2; echo "Usage: $0 [--go]" >&2; exit 1 ;;
esac

mkdir -p "${LOG_DIR}"

WANDB_API_KEY=""
if [[ -r <netrc-path> ]]; then
    WANDB_API_KEY=$(awk '/machine api.wandb.ai/{f=1;next} f&&/password/{print $2;exit}' <netrc-path>)
fi
if [[ -z "${WANDB_API_KEY}" ]]; then
    echo "WARN: no wandb API key found in <netrc-path>" >&2
fi

launch_one() {
    local node_short="$1" seed="$2"
    local node="${CLUSTER_PREFIX}-${node_short}"
    local exp_tag="dctv_seed${seed}"
    local remote_log="/tmp/ds_ds_dapo_${exp_tag}.log"
    local local_log="${LOG_DIR}/${node_short}__${exp_tag}.launcher.log"

    # No DCTV knob overrides — defaults come from the trainer script and match
    # DCTV_OPD_method.md §7 (β_D=β_M=0.95, c_cap=1.0, ε=1e-8).
    local wrap_cmd="cd ${REPO_ROOT} && setsid nohup env WANDB_API_KEY=${WANDB_API_KEY} SEED=${seed} bash ${SCRIPT} > ${remote_log} 2>&1 </dev/null &"

    if [[ "${MODE}" == "dry" ]]; then
        echo "[DRY] ${node_short}: dctv seed=${seed} -> ${remote_log}"
    else
        echo "[GO ] ${node_short}: dctv seed=${seed} -> ${remote_log}"
        ssh -f -n \
            -o StrictHostKeyChecking=no \
            -o BatchMode=yes \
            -o ConnectTimeout=10 \
            "${node}" "${wrap_cmd}" > "${local_log}" 2>&1
    fi
}

for entry in "${RUNS[@]}"; do
    IFS=':' read -r node_short seed <<< "${entry}"
    launch_one "${node_short}" "${seed}"
done

if [[ "${MODE}" == "go" ]]; then
    echo ""
    echo "=== All 4 DCTV runs dispatched. Log dir: ${LOG_DIR} ==="
    echo ""
    echo "Check status (per-node process count + latest step):"
    echo "  declare -A TAGS=(\\"
    for entry in "${RUNS[@]}"; do
        IFS=':' read -r node_short seed <<< "${entry}"
        echo "    [${node_short}]=dctv_seed${seed} \\"
    done
    echo "  )"
    echo "  for n in \${!TAGS[@]}; do"
    echo "    tag=\${TAGS[\$n]}"
    echo "    ssh -o ConnectTimeout=4 ${CLUSTER_PREFIX}-\$n \\"
    echo "      \"grep -oE 'Training Progress: *[0-9]+%[^\\\"]*[0-9]+/2000' /tmp/ds_ds_dapo_\${tag}.log 2>/dev/null | tail -1\" &"
    echo "  done; wait"
fi
