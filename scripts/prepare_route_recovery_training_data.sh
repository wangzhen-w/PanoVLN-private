#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${PROJECT_ROOT}"

export PYTHONPATH="./:${PYTHONPATH:-}"

PYTHON_BIN="/opt/conda/envs/vln/bin/python"
INPUT_ROOT="/workspace/z_PanoVLN"
DATASET_NAME="r2r_rxr_recovery"
OUTPUT_PATH="/workspace/data2/dataset/ablation/route_recovery/train_r2r_rxr_recovery.jsonl"
EBS_TRAIN_JSONL="/workspace/data2/dataset/ablation/ebs/train_r2r_rxr_ebs_event050_bg005_tail080.jsonl"
MIXED_OUTPUT_PATH="/workspace/data2/dataset/ablation/route_recovery/train_r2r_rxr_ebs_event050_bg005_tail080_with_recovery.jsonl"
HISTORY_WINDOW_FRAMES="100"
SEED="42"
INCLUDE_CLEAN_ANCHOR="false" # 第一版只监督 recovery，正常导航由 EBS 单独提供。
VALIDATE_IMAGE_FILES="false" # 原始采集已完成全量校验；这里避免再次扫描 86 万张 JPEG。
MAX_TRAJECTORIES=""          # 留空表示使用全部成功采集的物理轨迹。
OVERWRITE="false"            # 只有明确设为 true 才覆盖已有训练数据。

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python not found in PATH: ${PYTHON_BIN}" >&2
    exit 1
fi
INPUT_JSONL="${INPUT_ROOT}/sub_dataset/${DATASET_NAME}.jsonl"
INPUT_SUMMARY="${INPUT_ROOT}/sub_dataset/${DATASET_NAME}.summary.json"
INPUT_PROGRESS="${INPUT_JSONL}.inprogress"
if [[ ! -f "${INPUT_JSONL}" ]]; then
    echo "Recovery annotation not found: ${INPUT_JSONL}" >&2
    exit 1
fi
if [[ ! -f "${INPUT_SUMMARY}" ]]; then
    echo "Recovery completion summary not found: ${INPUT_SUMMARY}" >&2
    exit 1
fi
if [[ -e "${INPUT_PROGRESS}" ]]; then
    echo "Recovery collection is still in progress: ${INPUT_PROGRESS}" >&2
    exit 1
fi
if [[ ! -f "${EBS_TRAIN_JSONL}" ]]; then
    echo "EBS training JSONL not found: ${EBS_TRAIN_JSONL}" >&2
    exit 1
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"

CMD=(
    "${PYTHON_BIN}" src/data/prepare_route_recovery_training_data.py
    --input_root "${INPUT_ROOT}"
    --dataset_name "${DATASET_NAME}"
    --output_path "${OUTPUT_PATH}"
    --history_window_frames "${HISTORY_WINDOW_FRAMES}"
    --seed "${SEED}"
)
if [[ "${INCLUDE_CLEAN_ANCHOR}" == "true" ]]; then
    CMD+=(--include_clean_anchor)
fi
if [[ "${VALIDATE_IMAGE_FILES}" == "true" ]]; then
    CMD+=(--validate_image_files)
fi
if [[ -n "${MAX_TRAJECTORIES}" ]]; then
    CMD+=(--max_trajectories "${MAX_TRAJECTORIES}")
fi
if [[ "${OVERWRITE}" == "true" ]]; then
    CMD+=(--overwrite)
fi

echo "INPUT_ROOT: ${INPUT_ROOT}"
echo "DATASET_NAME: ${DATASET_NAME}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "EBS_TRAIN_JSONL: ${EBS_TRAIN_JSONL}"
echo "MIXED_OUTPUT_PATH: ${MIXED_OUTPUT_PATH}"
echo "SEED: ${SEED}"
echo "INCLUDE_CLEAN_ANCHOR: ${INCLUDE_CLEAN_ANCHOR}"
echo "OVERWRITE: ${OVERWRITE}"
printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
"${CMD[@]}"

MERGE_CMD=(
    "${PYTHON_BIN}" src/data/merge_training_jsonl.py
    --input_paths "${EBS_TRAIN_JSONL}" "${OUTPUT_PATH}"
    --output_path "${MIXED_OUTPUT_PATH}"
)
if [[ "${OVERWRITE}" == "true" ]]; then
    MERGE_CMD+=(--overwrite)
fi

printf 'Running:'
printf ' %q' "${MERGE_CMD[@]}"
printf '\n'
"${MERGE_CMD[@]}"

echo "Recovery training data: ${OUTPUT_PATH}"
echo "Mixed training data: ${MIXED_OUTPUT_PATH}"
