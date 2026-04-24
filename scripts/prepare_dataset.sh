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
INPUT_ROOT="/workspace/code_dir/a_property/dataset/NAVIDA_pano"
OUTPUT_PATH="/workspace/code_dir/a_property/dataset/NAVIDA_pano/train_r2r_atomic.jsonl"
DATASET_NAMES=(r2r rxr)
TASK_TYPES=(vln)
MAX_MEMORY_IMAGES="10"
MEMORY_POOL_WINDOW_FRAMES="200"
MAX_EPISODES_PER_SUBSET=""

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "INPUT_ROOT: ${INPUT_ROOT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "DATASET_NAMES: ${DATASET_NAMES[*]}"
echo "TASK_TYPES: ${TASK_TYPES[*]}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python not found in PATH" >&2
    exit 1
fi

PREPARE_CMD=(
    "${PYTHON_BIN}" src/data/prepare_training_data.py
    --input_root "${INPUT_ROOT}"
    --dataset_name "${DATASET_NAMES[@]}"
    --task_type "${TASK_TYPES[@]}"
    --output_path "${OUTPUT_PATH}"
    --max_memory_images "${MAX_MEMORY_IMAGES}"
    --memory_pool_window_frames "${MEMORY_POOL_WINDOW_FRAMES}"
)

if [[ -n "${MAX_EPISODES_PER_SUBSET}" ]]; then
    PREPARE_CMD+=(--max_episodes_per_subset "${MAX_EPISODES_PER_SUBSET}")
fi

"${PREPARE_CMD[@]}"
