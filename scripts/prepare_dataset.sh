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

PYTHON_BIN="/opt/conda/bin/python"
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Python executable not found: ${PYTHON_BIN}" >&2
    exit 1
fi

INPUT_ROOT="/workspace/data2/dataset/PanoVLN"
OUTPUT_PATH="/workspace/data2/dataset/ablation/18-action/train_r2r_rxr_h18_stride4_onset_fwd18_stop_1-6_stride1_7-18_stride2_seed42.jsonl"
DATASET_NAMES=(r2r rxr)

PREPARE_CMD=(
    "${PYTHON_BIN}" src/data/prepare_training_data.py
    --input_root "${INPUT_ROOT}"
    --dataset_name "${DATASET_NAMES[@]}"
    --output_path "${OUTPUT_PATH}"
    --seed 42
)
PREPARE_CMD+=("$@")

echo "INPUT_ROOT: ${INPUT_ROOT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "DATASET_NAMES: ${DATASET_NAMES[*]}"
echo "ACTION_HORIZON: 18"
echo "EXECUTION_HORIZON: 6"
echo "BODY_STRIDE: 4"
echo "RULE: original-strategy-with-stride4"
printf 'Running:'
printf ' %q' "${PREPARE_CMD[@]}"
printf '\n'
"${PREPARE_CMD[@]}"
