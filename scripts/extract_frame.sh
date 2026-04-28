#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"
echo "Switched to project root: $PROJECT_ROOT"

export PYTHONPATH="./:${PYTHONPATH:-}"
export MAGNUM_LOG="quiet"
export GLOG_minloglevel="3"
export HABITAT_LAB_LOG="50"
export PYTHONWARNINGS="ignore"
PYTHON_BIN="python"
OUTPUT_ROOT="/workspace/data/wz_data/dataset/NAVIDA_pano"
DATASET_NAMES=(envdrop)
GPU_IDS="1,2,3"
PROCESSES_PER_GPU="3"
SAVE_IMAGE="true"
SKIP_EXISTING_EPISODES="true"
MAX_EPISODES=""
EPISODE_IDS=""

mkdir -p "${OUTPUT_ROOT}"

echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "Habitat dataset paths: config/*.yaml"
echo "DATASET_NAMES: ${DATASET_NAMES[*]}"
echo "GPU_IDS: ${GPU_IDS}"
echo "PROCESSES_PER_GPU: ${PROCESSES_PER_GPU}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python not found in PATH" >&2
    exit 1
fi

read -r -a GPU_ID_ARRAY <<< "${GPU_IDS//,/ }"
GPU_COUNT="${#GPU_ID_ARRAY[@]}"
NUM_THREAD="$((GPU_COUNT * PROCESSES_PER_GPU))"

CMD=(
    "${PYTHON_BIN}" src/data/extract_frame.py
    --output_root "${OUTPUT_ROOT}"
    --dataset_name "${DATASET_NAMES[@]}"
    --gpu_ids "${GPU_IDS}"
    --num_thread "${NUM_THREAD}"
    --num_processes_per_gpu "${PROCESSES_PER_GPU}"
    --save_image "${SAVE_IMAGE}"
    --skip_existing_episodes "${SKIP_EXISTING_EPISODES}"
)

if [[ -n "${MAX_EPISODES}" ]]; then
    CMD+=(--max_episodes "${MAX_EPISODES}")
fi

if [[ -n "${EPISODE_IDS}" ]]; then
    read -r -a EPISODE_ID_ARRAY <<< "${EPISODE_IDS}"
    CMD+=(--episode_ids "${EPISODE_ID_ARRAY[@]}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${CMD[@]}"
