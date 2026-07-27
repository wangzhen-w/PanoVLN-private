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
OUTPUT_ROOT="/workspace/data2/dataset/PanoVLN"
DATASET_NAMES=(panovln)
# panovln directly imports train_gt.json.gz, so it does not need Habitat/GPU replay.
GOAL_RADIUS="0.3"  # Only used by datasets that still require Habitat replay.
GPU_IDS=""         # Set e.g. "0,1,2,3" when replaying another dataset.
PROCESSES_PER_GPU="1"
SKIP_EXISTING_EPISODES="true"
MAX_EPISODES=""
EPISODE_IDS=""
TEMP_ROOT=""

mkdir -p "${OUTPUT_ROOT}"
if [[ -n "${TEMP_ROOT}" ]]; then
    mkdir -p "${TEMP_ROOT}"
fi

echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "Habitat dataset paths: config/*.yaml"
echo "DATASET_NAMES: ${DATASET_NAMES[*]}"
echo "GPU_IDS: ${GPU_IDS}"
echo "PROCESSES_PER_GPU: ${PROCESSES_PER_GPU}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python not found in PATH" >&2
    exit 1
fi

CMD=(
    "${PYTHON_BIN}" src/data/preprocess.py
    --output_root "${OUTPUT_ROOT}"
    --goal_radius "${GOAL_RADIUS}"
    --skip_existing_episodes "${SKIP_EXISTING_EPISODES}"
    --dataset_name "${DATASET_NAMES[@]}"
)

if [[ -n "${GPU_IDS}" ]]; then
    CMD+=(
        --gpu_ids "${GPU_IDS}"
        --num_processes_per_gpu "${PROCESSES_PER_GPU}"
    )
fi

if [[ -n "${TEMP_ROOT}" ]]; then
    CMD+=(--temp_root "${TEMP_ROOT}")
fi

if [[ -n "${MAX_EPISODES}" ]]; then
    CMD+=(--max_episodes "${MAX_EPISODES}")
fi

if [[ -n "${EPISODE_IDS}" ]]; then
    read -r -a EPISODE_ID_ARRAY <<< "${EPISODE_IDS}"
    CMD+=(--episode_ids "${EPISODE_ID_ARRAY[@]}")
fi

if [[ -n "${GPU_IDS}" ]]; then
    CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${CMD[@]}"
else
    "${CMD[@]}"
fi
