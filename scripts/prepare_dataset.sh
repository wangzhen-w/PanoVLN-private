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
INPUT_ROOT="/workspace/code/a_property/dataset/PanoVLN"
OUTPUT_PATH="/workspace/code/a_property/dataset/PanoVLN/train_r2r_rxr_ebs_event050_bg0p005_tail0p100.jsonl"
DATASET_NAMES=(r2r rxr)
MAX_EPISODES_PER_SUBSET=""
PAD_STOP_TO_HORIZON="true"
SEED="42"

EVENT_KEEP_PROB="0.50"
BACKGROUND_KEEP_PROB="0.05"
BODY_KEEP_ADVANCE="4"
TAIL_DENSE_KEEP_PROB="1.0"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "INPUT_ROOT: ${INPUT_ROOT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "DATASET_NAMES: ${DATASET_NAMES[*]}"
echo "PAD_STOP_TO_HORIZON: ${PAD_STOP_TO_HORIZON}"
echo "SEED: ${SEED}"
echo "EVENT_KEEP_PROB: ${EVENT_KEEP_PROB}"
echo "BACKGROUND_KEEP_PROB: ${BACKGROUND_KEEP_PROB}"
echo "BODY_KEEP_ADVANCE: ${BODY_KEEP_ADVANCE}"
echo "TAIL_DENSE_KEEP_PROB: ${TAIL_DENSE_KEEP_PROB}"

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
    --event_keep_prob "${EVENT_KEEP_PROB}"
    --background_keep_prob "${BACKGROUND_KEEP_PROB}"
    --body_keep_advance "${BODY_KEEP_ADVANCE}"
    --tail_dense_keep_prob "${TAIL_DENSE_KEEP_PROB}"
)

if [[ -n "${MAX_EPISODES_PER_SUBSET}" ]]; then
    PREPARE_CMD+=(--max_episodes_per_subset "${MAX_EPISODES_PER_SUBSET}")
fi

if [[ "${PAD_STOP_TO_HORIZON}" == "true" ]]; then
    PREPARE_CMD+=(--pad_stop_to_horizon)
fi

printf 'Running:'
printf ' %q' "${PREPARE_CMD[@]}"
printf '\n'
"${PREPARE_CMD[@]}"
