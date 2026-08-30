#!/usr/bin/env bash
# ============================================================================
# Prewarm DS-Distill-Qwen-1.5B student to /root/ on every node in the H20 cluster.
# ----------------------------------------------------------------------------
# Student is 3.4GB; even from cephfs the wall time is short, but node-local
# avoids the 8× per-node reload cost and shared-FS contention when 16 runs
# start simultaneously.
#
# Pod restart wipes /root/ (overlay fs), so re-run before batch launches.
#
# Usage:
#   bash train_script/prewarm_ds_distill_qwen15b.sh          # ensure on all 16 nodes
#   bash train_script/prewarm_ds_distill_qwen15b.sh --force  # re-copy even if present
#   bash train_script/prewarm_ds_distill_qwen15b.sh --check  # check-only, no copy
# ============================================================================
set -euo pipefail

CLUSTER_PREFIX="$CLUSTER_PREFIX"
SRC="$DATA_ROOT/DS-Distill-Qwen-1.5B"
DST="/root/DS-Distill-Qwen-1.5B"

ALL_NODES=(
    "${CLUSTER_PREFIX}-launcher"
    "${CLUSTER_PREFIX}-worker-0"
    "${CLUSTER_PREFIX}-worker-1"
    "${CLUSTER_PREFIX}-worker-2"
    "${CLUSTER_PREFIX}-worker-3"
    "${CLUSTER_PREFIX}-worker-4"
    "${CLUSTER_PREFIX}-worker-5"
    "${CLUSTER_PREFIX}-worker-6"
)

MODE="ensure"
case "${1:-}" in
    --force) MODE="force" ;;
    --check) MODE="check" ;;
    -h|--help) grep -E '^# ' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
esac

if [[ ! -d "${SRC}" ]]; then
    echo "ERROR: source ${SRC} does not exist" >&2
    exit 1
fi
src_size=$(du -sh "${SRC}" | awk '{print $1}')
echo "Source: ${SRC} (${src_size})"
echo "Dest:   ${DST}"
echo "Mode:   ${MODE}"
echo ""

do_node() {
    local node="$1"
    local remote_check='
        if [ -d '"${DST}"' ] && [ -f '"${DST}"'/config.json ]; then
            echo "OK $(du -sh '"${DST}"' | awk "{print \$1}")"
        else
            echo "MISSING"
        fi
    '
    local status
    status=$(ssh -o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=5 "${node}" "${remote_check}" 2>&1 || echo "SSH_FAIL")
    printf "%-52s %s\n" "${node}" "${status}"

    if [[ "${MODE}" == "check" ]]; then
        return
    fi

    if [[ "${MODE}" == "ensure" && "${status}" == OK* ]]; then
        return
    fi

    if ! ssh -o StrictHostKeyChecking=no -o BatchMode=yes "${node}" \
        "SRC='${SRC}' DST='${DST}' "'bash -s' <<'REMOTE'
set -e
rm -rf "${DST}"
mkdir -p /root
cp -a "${SRC}" "${DST}"
echo "COPIED $(du -sh "${DST}" | awk '{print $1}')"
REMOTE
    then
        echo "COPY_FAILED on ${node}" >&2
        return 1
    fi
}

for node in "${ALL_NODES[@]}"; do
    do_node "${node}" &
done
wait

echo ""
echo "Done."
