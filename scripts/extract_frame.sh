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
OUTPUT_ROOT="/workspace/code/a_property/dataset/PanoVLN"
DATASET_NAMES=(dagger)  # r2r/rxr and other original datasets are also supported.
GPU_IDS="0,1,2,3,4,5,6,7"
PROCESSES_PER_GPU="6"
SAVE_IMAGE="true"  # Complete episodes in the selected format are skipped.
SKIP_EXISTING_EPISODES="true"
MAX_EPISODES=""
EPISODE_IDS=""

# png/jpeg are supported. PNG is always pixel-lossless; level 0 disables its
# lossless DEFLATE compression. JPEG settings are ignored when IMAGE_FORMAT=png.
IMAGE_FORMAT="jpeg"
PNG_COMPRESS_LEVEL="6"
JPEG_QUALITY="95"
JPEG_SUBSAMPLING="1"

mkdir -p "${OUTPUT_ROOT}"

echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "Habitat dataset paths: config/*.yaml"
echo "DATASET_NAMES: ${DATASET_NAMES[*]}"
echo "GPU_IDS: ${GPU_IDS}"
echo "PROCESSES_PER_GPU: ${PROCESSES_PER_GPU}"
echo "IMAGE_FORMAT: ${IMAGE_FORMAT}"
if [[ "${IMAGE_FORMAT,,}" == "png" ]]; then
    echo "PNG_COMPRESS_LEVEL: ${PNG_COMPRESS_LEVEL}"
else
    echo "JPEG_QUALITY: ${JPEG_QUALITY}"
    echo "JPEG_SUBSAMPLING: ${JPEG_SUBSAMPLING}"
fi

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python not found in PATH" >&2
    exit 1
fi

read -r -a GPU_ID_ARRAY <<< "${GPU_IDS//,/ }"
GPU_COUNT="${#GPU_ID_ARRAY[@]}"
NUM_THREAD="$((GPU_COUNT * PROCESSES_PER_GPU))"

CMD=(
    "${PYTHON_BIN}" src/data/extract_frame.py
    --output_root "${OUTPUT_ROOT}"
    --dataset_name "${DATASET_NAMES[@]}"
    --gpu_ids "${GPU_IDS}"
    --num_thread "${NUM_THREAD}"
    --num_processes_per_gpu "${PROCESSES_PER_GPU}"
    --save_image "${SAVE_IMAGE}"
    --skip_existing_episodes "${SKIP_EXISTING_EPISODES}"
    --image_format "${IMAGE_FORMAT}"
    --png_compress_level "${PNG_COMPRESS_LEVEL}"
    --jpeg_quality "${JPEG_QUALITY}"
    --jpeg_subsampling "${JPEG_SUBSAMPLING}"
)

if [[ -n "${MAX_EPISODES}" ]]; then
    CMD+=(--max_episodes "${MAX_EPISODES}")
fi

if [[ -n "${EPISODE_IDS}" ]]; then
    read -r -a EPISODE_ID_ARRAY <<< "${EPISODE_IDS}"
    CMD+=(--episode_ids "${EPISODE_ID_ARRAY[@]}")
fi

CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${CMD[@]}"
