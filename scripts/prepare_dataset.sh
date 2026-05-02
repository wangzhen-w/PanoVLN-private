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

PYTHON_BIN="${PYTHON_BIN:-python}"
INPUT_ROOT="${INPUT_ROOT:-/workspace/code_dir/a_property/dataset/NAVIDA_pano}"
OUTPUT_PATH="${OUTPUT_PATH:-/workspace/code_dir/a_property/dataset/NAVIDA_pano/train_r2r_rxr_4action_stop_pad_sbs_tau1p35_beta0p40_taildense4.jsonl}"
DATASET_NAMES_STRING="${DATASET_NAMES:-r2r rxr}"
MAX_EPISODES_PER_SUBSET="${MAX_EPISODES_PER_SUBSET:-}"
PAD_STOP_TO_HORIZON="${PAD_STOP_TO_HORIZON:-true}"
SEED="${SEED:-42}"

SBS_TAU="${SBS_TAU:-1.35}"
SBS_BETA="${SBS_BETA:-0.40}"

read -r -a DATASET_NAMES <<< "${DATASET_NAMES_STRING}"
mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "INPUT_ROOT: ${INPUT_ROOT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "DATASET_NAMES: ${DATASET_NAMES[*]}"
echo "PAD_STOP_TO_HORIZON: ${PAD_STOP_TO_HORIZON}"
echo "SEED: ${SEED}"
echo "SBS_TAU: ${SBS_TAU}"
echo "SBS_BETA: ${SBS_BETA}"

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
    --sbs_tau "${SBS_TAU}"
    --sbs_beta "${SBS_BETA}"
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
