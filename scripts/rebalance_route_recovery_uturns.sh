#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${PROJECT_ROOT}"

export PYTHONPATH="./:${PYTHONPATH:-}"
export MAGNUM_LOG="quiet"
export GLOG_minloglevel="3"
export HABITAT_LAB_LOG="50"
export PYTHONWARNINGS="ignore"

PYTHON_BIN="/opt/conda/envs/vln/bin/python"
INPUT_ROOT="/workspace/z_PanoVLN"
DATASET_NAME="r2r_rxr_recovery"
GPU_IDS="0,1,2,3,4,5,6,7"
PROCESSES_PER_GPU="2"
SEED="42"
IMAGE_WIDTH="1280"
IMAGE_HEIGHT="640"
RESUME="true"
PROGRESS_REFRESH_SECONDS="2"

IFS=',' read -r -a GPU_ARRAY <<< "${GPU_IDS}"
if [[ "${#GPU_ARRAY[@]}" -eq 0 ]]; then
    echo "GPU_IDS must contain at least one GPU id" >&2
    exit 1
fi
if [[ "${PROCESSES_PER_GPU}" -le 0 ]]; then
    echo "PROCESSES_PER_GPU must be positive" >&2
    exit 1
fi
NUM_WORKERS="$(( ${#GPU_ARRAY[@]} * PROCESSES_PER_GPU ))"

mkdir -p /workspace/tmp
LOCK_PATH="/workspace/tmp/${DATASET_NAME}.uturn_rebalance.lock"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    echo "Another U-turn rebalance launcher is already running: ${LOCK_PATH}" >&2
    exit 1
fi

PIDS=()
cleanup_workers() {
    local PID
    for PID in "${PIDS[@]}"; do
        if kill -0 "${PID}" 2>/dev/null; then
            kill -TERM "${PID}" 2>/dev/null || true
        fi
    done
    for PID in "${PIDS[@]}"; do
        wait "${PID}" 2>/dev/null || true
    done
}
trap cleanup_workers EXIT
trap 'exit 130' INT TERM

COMMON_ARGS=(
    --input_root "${INPUT_ROOT}"
    --dataset_name "${DATASET_NAME}"
    --image_width "${IMAGE_WIDTH}"
    --image_height "${IMAGE_HEIGHT}"
    --seed "${SEED}"
    --num_workers "${NUM_WORKERS}"
)
if [[ "${RESUME}" == "true" ]]; then
    COMMON_ARGS+=(--resume)
fi

echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "INPUT_ROOT: ${INPUT_ROOT}"
echo "DATASET_NAME: ${DATASET_NAME}"
echo "GPU_IDS: ${GPU_ARRAY[*]}"
echo "PROCESSES_PER_GPU: ${PROCESSES_PER_GPU}"
echo "TOTAL_WORKERS: ${NUM_WORKERS}"
echo "SEED: ${SEED}"

SUMMARY_PATH="${INPUT_ROOT}/sub_dataset/${DATASET_NAME}.summary.json"
if "${PYTHON_BIN}" - <<PY
import json
from pathlib import Path
p = Path("${SUMMARY_PATH}")
x = json.loads(p.read_text()) if p.is_file() else {}
raise SystemExit(0 if x.get("uturn_rebalance", {}).get("complete") is True else 1)
PY
then
    "${PYTHON_BIN}" src/data/rebalance_route_recovery_uturns.py \
        "${COMMON_ARGS[@]}" --verify_only
    PROGRESS_ROOT="${INPUT_ROOT}/sub_dataset/.${DATASET_NAME}.uturn_rebalance.inprogress"
    if [[ -d "${PROGRESS_ROOT}" ]]; then
        rm -rf "${PROGRESS_ROOT}"
    fi
    exit 0
fi

"${PYTHON_BIN}" src/data/rebalance_route_recovery_uturns.py \
    "${COMMON_ARGS[@]}" --initialize_only

LOG_ROOT="/workspace/tmp/${DATASET_NAME}_uturn_rebalance_logs"
mkdir -p "${LOG_ROOT}"
WORKER_INDEX="0"
for GPU_ID in "${GPU_ARRAY[@]}"; do
    for ((LOCAL_PROCESS=0; LOCAL_PROCESS<PROCESSES_PER_GPU; LOCAL_PROCESS++)); do
        LOG_PATH="${LOG_ROOT}/worker_${WORKER_INDEX}.log"
        "${PYTHON_BIN}" src/data/rebalance_route_recovery_uturns.py \
            "${COMMON_ARGS[@]}" \
            --worker_index "${WORKER_INDEX}" \
            --gpu_id "${GPU_ID}" \
            >"${LOG_PATH}" 2>&1 &
        PIDS+=("$!")
        WORKER_INDEX="$((WORKER_INDEX + 1))"
    done
done

PROGRESS_ROOT="${INPUT_ROOT}/sub_dataset/.${DATASET_NAME}.uturn_rebalance.inprogress"
TOTAL=$("${PYTHON_BIN}" - <<PY
import json
print(json.load(open("${PROGRESS_ROOT}/configuration.json"))["selected_trajectories"])
PY
)
while true; do
    RUNNING="0"
    for PID in "${PIDS[@]}"; do
        if kill -0 "${PID}" 2>/dev/null; then
            RUNNING="$((RUNNING + 1))"
        fi
    done
    COMPLETE=$(find "${PROGRESS_ROOT}/completed" -maxdepth 1 -name '*.json' 2>/dev/null | wc -l)
    printf '\rHabitat right U-turn rendering: %s/%s, workers running: %s' \
        "${COMPLETE}" "${TOTAL}" "${RUNNING}"
    if [[ "${RUNNING}" -eq 0 ]]; then
        break
    fi
    sleep "${PROGRESS_REFRESH_SECONDS}"
done
printf '\n'

FAILED="0"
for PID in "${PIDS[@]}"; do
    if ! wait "${PID}"; then
        FAILED="1"
    fi
done
PIDS=()
if [[ "${FAILED}" -ne 0 ]]; then
    echo "At least one U-turn worker failed. Logs: ${LOG_ROOT}" >&2
    exit 1
fi

"${PYTHON_BIN}" src/data/rebalance_route_recovery_uturns.py \
    "${COMMON_ARGS[@]}" --finalize_only

rm -rf "${LOG_ROOT}"
echo "Exact U-turn rebalance completed and verified."
