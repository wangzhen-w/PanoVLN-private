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
INPUT_JSONL="/workspace/code_dir/a_property/dataset/NAVIDA_pano/train_r2r_rxr_4action_pano_full_history_word_stop_pad.jsonl"
OUTPUT_JSONL="/workspace/code_dir/a_property/dataset/NAVIDA_pano/train_r2r_rxr_4action_pano_full_history_word_stop_pad_balanced_subset_v2.jsonl"
STATS_JSON="/workspace/code_dir/a_property/dataset/NAVIDA_pano/train_r2r_rxr_4action_pano_full_history_word_stop_pad_balanced_subset_v2.stats.json"
SEED="42"
TARGET_SAMPLES=""
WITH_REPLACEMENT="false"
WEIGHTS="first_stop=0.08,second_stop=0.08,third_stop=0.03,fourth_stop=0.03,nonstop_first_forward=0.33,nonstop_first_left=0.225,nonstop_first_right=0.225"

mkdir -p "$(dirname "${OUTPUT_JSONL}")"

echo "INPUT_JSONL: ${INPUT_JSONL}"
echo "OUTPUT_JSONL: ${OUTPUT_JSONL}"
echo "STATS_JSON: ${STATS_JSON}"
echo "SEED: ${SEED}"
echo "TARGET_SAMPLES: ${TARGET_SAMPLES:-auto}"
echo "WITH_REPLACEMENT: ${WITH_REPLACEMENT}"
echo "WEIGHTS: ${WEIGHTS}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python not found in PATH" >&2
    exit 1
fi

SAMPLE_CMD=(
    "${PYTHON_BIN}" src/data/sample_action_chunks.py
    --input_jsonl "${INPUT_JSONL}"
    --output_jsonl "${OUTPUT_JSONL}"
    --stats_json "${STATS_JSON}"
    --seed "${SEED}"
    --weights "${WEIGHTS}"
)

if [[ -n "${TARGET_SAMPLES}" ]]; then
    SAMPLE_CMD+=(--target_samples "${TARGET_SAMPLES}")
fi

if [[ "${WITH_REPLACEMENT}" == "true" ]]; then
    SAMPLE_CMD+=(--with_replacement)
fi

"${SAMPLE_CMD[@]}"
