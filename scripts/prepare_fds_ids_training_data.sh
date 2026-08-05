#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${PROJECT_ROOT}"

export PYTHONPATH="./:${PYTHONPATH:-}"

PYTHON_BIN="/opt/conda/envs/vln/bin/python"
EBS_INPUT_PATH="/workspace/data2/dataset/ablation/fds_ids/train_r2r_rxr_ebs_event050_bg005_tail080.jsonl"
IDS_OUTPUT_PATH="/workspace/data2/dataset/ablation/fds_ids/train_r2r_rxr_ids_event050_bg005_tail080.jsonl"
MIXED_OUTPUT_PATH="/workspace/data2/dataset/ablation/fds_ids/train_r2r_rxr_ebs_event050_bg005_tail080_with_ids.jsonl"
MAX_ROWS=""        # 留空生成全量；仅用于本地小样本检查。
OVERWRITE="false" # 只有明确设为 true 才覆盖已有产物。

if (( $# != 0 )); then
    echo "This controlled data-preparation script does not accept arguments: $*" >&2
    exit 1
fi
if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "python is not executable: ${PYTHON_BIN}" >&2
    exit 1
fi
if [[ ! -f "${EBS_INPUT_PATH}" ]]; then
    echo "EBS input JSONL not found: ${EBS_INPUT_PATH}" >&2
    exit 1
fi
if [[ "${OVERWRITE}" != "true" ]]; then
    for OUTPUT_PATH in "${IDS_OUTPUT_PATH}" "${MIXED_OUTPUT_PATH}"; do
        if [[ -e "${OUTPUT_PATH}" ]]; then
            echo "Output already exists; set OVERWRITE=true to replace it: ${OUTPUT_PATH}" >&2
            exit 1
        fi
    done
fi

mkdir -p "$(dirname "${IDS_OUTPUT_PATH}")"

CMD=(
    "${PYTHON_BIN}" src/data/prepare_inverse_dynamics_training_data.py
    --input_path "${EBS_INPUT_PATH}"
    --ids_output_path "${IDS_OUTPUT_PATH}"
    --mixed_output_path "${MIXED_OUTPUT_PATH}"
)
if [[ -n "${MAX_ROWS}" ]]; then
    CMD+=(--max_rows "${MAX_ROWS}")
fi
if [[ "${OVERWRITE}" == "true" ]]; then
    CMD+=(--overwrite)
fi

echo "EBS_INPUT_PATH: ${EBS_INPUT_PATH}"
echo "IDS_OUTPUT_PATH: ${IDS_OUTPUT_PATH}"
echo "MIXED_OUTPUT_PATH: ${MIXED_OUTPUT_PATH}"
echo "MAX_ROWS: ${MAX_ROWS:-all}"
echo "OVERWRITE: ${OVERWRITE}"
printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

echo "IDS training data: ${IDS_OUTPUT_PATH}"
echo "EBS + IDS training data: ${MIXED_OUTPUT_PATH}"
