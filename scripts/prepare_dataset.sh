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
OUTPUT_DIR="/workspace/data2/dataset/ablation/12-action"
OUTPUT_PATH="${OUTPUT_DIR}/train_r2r_rxr_maneuver_h12_seed42.jsonl"

PREPARE_CMD=(
    "${PYTHON_BIN}" src/data/prepare_training_data.py
    --input_root "${INPUT_ROOT}"
    --output_path "${OUTPUT_PATH}"
    --seed 42
)
PREPARE_CMD+=("$@")

echo "INPUT_ROOT: ${INPUT_ROOT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "DATASETS: r2r rxr"
echo "ACTION_HORIZON: 12"
echo "RULE_VERSION: maneuver_h12"
printf 'Running:'
printf ' %q' "${PREPARE_CMD[@]}"
printf '\n'
"${PREPARE_CMD[@]}"
