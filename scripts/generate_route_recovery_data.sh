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
OUTPUT_ROOT="/workspace/z_PanoVLN"
OUTPUT_DATASET_NAME="r2r_rxr_recovery"
DATASET_NAMES=(r2r rxr)
GPU_IDS="0,1,2,3,4,5,6,7"
DEPTH_CHOICES=(2.0 2.0 2.5 2.5 3.0 3.5)
PROCESSES_PER_GPU="2"
SEED="42"
IMAGE_WIDTH="1280"
IMAGE_HEIGHT="640"
MAX_MAPPING_ERROR="0.5"
MIN_FREE_DISK_GB="200"
ROUTE_ORDER="scene"
MAX_ROUTES_TO_TRY=""       # 留空表示处理该 worker 的全部物理路线。
MAX_SAMPLES_PER_DATASET="" # 留空表示保存全部成功轨迹；pilot 可设为小整数。
RESUME="true"
SHOW_PROGRESS="true"
PROGRESS_REFRESH_SECONDS="2"

mkdir -p /workspace/tmp
LOCK_PATH="/workspace/tmp/${OUTPUT_DATASET_NAME}.collection.lock"
exec 9>"${LOCK_PATH}"
if ! flock -n 9; then
    echo "Another recovery collection launcher is already running: ${LOCK_PATH}" >&2
    exit 1
fi

PIDS=()
MONITOR_PID=""
declare -A PID_TO_WORKER
declare -A PID_TO_LOG
stop_progress_monitor() {
    if [[ -n "${MONITOR_PID}" ]] && kill -0 "${MONITOR_PID}" 2>/dev/null; then
        kill -TERM "${MONITOR_PID}" 2>/dev/null || true
        wait "${MONITOR_PID}" 2>/dev/null || true
    fi
    MONITOR_PID=""
}
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
    stop_progress_monitor
}
handle_signal() {
    exit 130
}
trap cleanup_workers EXIT
trap handle_signal INT TERM

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python not found in PATH: ${PYTHON_BIN}" >&2
    exit 1
fi

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

echo "PROJECT_ROOT: ${PROJECT_ROOT}"
echo "INPUT_ROOT: ${INPUT_ROOT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "OUTPUT_DATASET_NAME: ${OUTPUT_DATASET_NAME}"
echo "DATASET_NAMES: ${DATASET_NAMES[*]}"
echo "GPU_IDS: ${GPU_ARRAY[*]}"
echo "PROCESSES_PER_GPU: ${PROCESSES_PER_GPU}"
echo "TOTAL_WORKERS: ${NUM_WORKERS}"
echo "DEPTH_CHOICES: ${DEPTH_CHOICES[*]}"
echo "RESUME: ${RESUME}"
echo "SHOW_PROGRESS: ${SHOW_PROGRESS}"

FINAL_MANIFEST="${OUTPUT_ROOT}/sub_dataset/${OUTPUT_DATASET_NAME}.jsonl"
FINAL_SUMMARY="${OUTPUT_ROOT}/sub_dataset/${OUTPUT_DATASET_NAME}.summary.json"
FINAL_ROUTE_OUTCOMES="${OUTPUT_ROOT}/sub_dataset/${OUTPUT_DATASET_NAME}.route_outcomes.jsonl"
PROGRESS_ROOT="${FINAL_MANIFEST}.inprogress"

COMMON_ARGS=(
    --input_root "${INPUT_ROOT}"
    --output_root "${OUTPUT_ROOT}"
    --output_dataset_name "${OUTPUT_DATASET_NAME}"
    --dataset_names "${DATASET_NAMES[@]}"
    --depth_choices "${DEPTH_CHOICES[@]}"
    --image_width "${IMAGE_WIDTH}"
    --image_height "${IMAGE_HEIGHT}"
    --seed "${SEED}"
    --max_mapping_error "${MAX_MAPPING_ERROR}"
    --min_free_disk_gb "${MIN_FREE_DISK_GB}"
    --route_order "${ROUTE_ORDER}"
    --num_workers "${NUM_WORKERS}"
)
if [[ -n "${MAX_ROUTES_TO_TRY}" ]]; then
    COMMON_ARGS+=(--max_routes_to_try "${MAX_ROUTES_TO_TRY}")
fi
if [[ -n "${MAX_SAMPLES_PER_DATASET}" ]]; then
    COMMON_ARGS+=(--max_samples_per_dataset "${MAX_SAMPLES_PER_DATASET}")
fi

RUN_IS_RESUME="false"
if [[ -d "${PROGRESS_ROOT}" ]]; then
    if [[ "${RESUME}" != "true" ]]; then
        echo "Unfinished collection exists but RESUME is not true: ${PROGRESS_ROOT}" >&2
        exit 1
    fi
    RUN_IS_RESUME="true"
elif [[ -e "${FINAL_MANIFEST}" || -e "${FINAL_SUMMARY}" || -e "${FINAL_ROUTE_OUTCOMES}" ]]; then
    VERIFY_CMD=(
        "${PYTHON_BIN}" src/data/generate_route_recovery_data.py
        "${COMMON_ARGS[@]}"
        --verify_only
    )
    "${VERIFY_CMD[@]}"
    echo "Recovery collection is already complete and verified: ${FINAL_MANIFEST}"
    exit 0
fi

INITIALIZE_CMD=(
    "${PYTHON_BIN}" src/data/generate_route_recovery_data.py
    "${COMMON_ARGS[@]}"
    --initialize_only
)
if [[ "${RUN_IS_RESUME}" == "true" ]]; then
    INITIALIZE_CMD+=(--resume)
fi
"${INITIALIZE_CMD[@]}"

LOG_ROOT="${PROGRESS_ROOT}/logs"
mkdir -p "${LOG_ROOT}"

for (( WORKER_INDEX=0; WORKER_INDEX<NUM_WORKERS; WORKER_INDEX++ )); do
    GPU_INDEX="$((WORKER_INDEX % ${#GPU_ARRAY[@]}))"
    GPU_ID="${GPU_ARRAY[GPU_INDEX]}"
    WORKER_NAME="$(printf 'worker_%02d' "${WORKER_INDEX}")"
    LOG_PATH="${LOG_ROOT}/${WORKER_NAME}.log"

    CMD=(
        "${PYTHON_BIN}" src/data/generate_route_recovery_data.py
        "${COMMON_ARGS[@]}"
        --gpu_id "${GPU_ID}"
        --worker_index "${WORKER_INDEX}"
    )
    if [[ "${RUN_IS_RESUME}" == "true" ]]; then
        CMD+=(--resume)
    fi

    printf 'Launching %s on GPU %s:' "${WORKER_NAME}" "${GPU_ID}"
    printf ' %q' "${CMD[@]}"
    printf '\nLog: %s\n' "${LOG_PATH}"
    "${CMD[@]}" >"${LOG_PATH}" 2>&1 &
    PID="$!"
    PIDS+=("${PID}")
    PID_TO_WORKER["${PID}"]="${WORKER_INDEX}"
    PID_TO_LOG["${PID}"]="${LOG_PATH}"
done

if [[ "${SHOW_PROGRESS}" == "true" ]]; then
    "${PYTHON_BIN}" src/data/monitor_route_recovery_progress.py \
        --progress_root "${PROGRESS_ROOT}" \
        --image_root "${OUTPUT_ROOT}/images/${OUTPUT_DATASET_NAME}" \
        --refresh_seconds "${PROGRESS_REFRESH_SECONDS}" &
    MONITOR_PID="$!"
fi

ACTIVE_PIDS=("${PIDS[@]}")
while [[ "${#ACTIVE_PIDS[@]}" -gt 0 ]]; do
    FINISHED_PID=""
    if wait -n -p FINISHED_PID "${ACTIVE_PIDS[@]}"; then
        echo "Completed worker ${PID_TO_WORKER[${FINISHED_PID}]}: ${PID_TO_LOG[${FINISHED_PID}]}"
    else
        STATUS="$?"
        echo "Failed worker ${PID_TO_WORKER[${FINISHED_PID}]} with status ${STATUS}: ${PID_TO_LOG[${FINISHED_PID}]}" >&2
        tail -n 80 "${PID_TO_LOG[${FINISHED_PID}]}" >&2 || true
        for PID in "${ACTIVE_PIDS[@]}"; do
            if [[ "${PID}" != "${FINISHED_PID}" ]]; then
                kill -TERM "${PID}" 2>/dev/null || true
            fi
        done
        PIDS=()
        for PID in "${ACTIVE_PIDS[@]}"; do
            if [[ "${PID}" != "${FINISHED_PID}" ]]; then
                PIDS+=("${PID}")
            fi
        done
        exit 1
    fi
    NEXT_ACTIVE_PIDS=()
    for PID in "${ACTIVE_PIDS[@]}"; do
        if [[ "${PID}" != "${FINISHED_PID}" ]]; then
            NEXT_ACTIVE_PIDS+=("${PID}")
        fi
    done
    ACTIVE_PIDS=("${NEXT_ACTIVE_PIDS[@]}")
    PIDS=("${ACTIVE_PIDS[@]}")
done
stop_progress_monitor

FINALIZE_CMD=(
    "${PYTHON_BIN}" src/data/generate_route_recovery_data.py
    "${COMMON_ARGS[@]}"
    --finalize_only
)
"${FINALIZE_CMD[@]}"

VERIFY_CMD=(
    "${PYTHON_BIN}" src/data/generate_route_recovery_data.py
    "${COMMON_ARGS[@]}"
    --verify_only
)
"${VERIFY_CMD[@]}"

echo "Recovery annotation: ${OUTPUT_ROOT}/sub_dataset/${OUTPUT_DATASET_NAME}.jsonl"
echo "Recovery summary: ${OUTPUT_ROOT}/sub_dataset/${OUTPUT_DATASET_NAME}.summary.json"
echo "Recovery route outcomes: ${OUTPUT_ROOT}/sub_dataset/${OUTPUT_DATASET_NAME}.route_outcomes.jsonl"
echo "Recovery images: ${OUTPUT_ROOT}/images/${OUTPUT_DATASET_NAME}"
