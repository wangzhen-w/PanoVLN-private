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
OUTPUT_PATH="/workspace/code_dir/a_property/dataset/NAVIDA_pano/train_r2r_rxr_4action_pano_full_history_word_stop_pad_probskip.jsonl"
DATASET_NAMES=(r2r rxr)
MAX_EPISODES_PER_SUBSET=""
PAD_STOP_TO_HORIZON="true"
SEED="42"
TURN_CHUNK_KEEP_PROB="0.50"
FORWARD_CHUNK_KEEP_PROB="0.05"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "INPUT_ROOT: ${INPUT_ROOT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "DATASET_NAMES: ${DATASET_NAMES[*]}"
echo "PAD_STOP_TO_HORIZON: ${PAD_STOP_TO_HORIZON}"
echo "SEED: ${SEED}"
echo "TURN_CHUNK_KEEP_PROB: ${TURN_CHUNK_KEEP_PROB}"
echo "FORWARD_CHUNK_KEEP_PROB: ${FORWARD_CHUNK_KEEP_PROB}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python not found in PATH" >&2
    exit 1
fi

PREPARE_CMD=(
    "${PYTHON_BIN}" src/data/prepare_training_data.py
    --input_root "${INPUT_ROOT}"
    --dataset_name "${DATASET_NAMES[@]}"
    --output_path "${OUTPUT_PATH}"
    --seed "${SEED}"
    --turn_chunk_keep_prob "${TURN_CHUNK_KEEP_PROB}"
    --forward_chunk_keep_prob "${FORWARD_CHUNK_KEEP_PROB}"
)

if [[ -n "${MAX_EPISODES_PER_SUBSET}" ]]; then
    PREPARE_CMD+=(--max_episodes_per_subset "${MAX_EPISODES_PER_SUBSET}")
fi

if [[ "${PAD_STOP_TO_HORIZON}" == "true" ]]; then
    PREPARE_CMD+=(--pad_stop_to_horizon)
fi

"${PREPARE_CMD[@]}"
