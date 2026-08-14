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
MODEL_PATH="/workspace/data2/model/ablation_new/pbo_fd/panovggt_0.20_lr2e-5_singlepoint_8card_66.4pbo"
OUTPUT_ROOT="/workspace/code/a_property/dataset/PanoVLN"
REFERENCE_INPUT_ROOT="/workspace/code/a_property/dataset/PanoVLN"
SOURCE_DATASET_NAMES=(r2r rxr)
GPU_IDS="0,1,2,3,4,5,6,7"
PROCESSES_PER_GPU="3"
CPU_THREADS_PER_WORKER="2"
MAX_EPISODES=""
EPISODE_IDS=""
MAX_STEPS_PER_EPISODE="500"
# Intermediate reference-waypoint tolerance, matching StreamVLN/JanusVLN.
MIDGOAL_RADIUS="1.8"
# Single final-goal tolerance for both oracle STOP and saved-episode success.
GOAL_RADIUS="0.3"
ALPHA="0.5"
SEED="42"
ATTN_IMPLEMENTATION="sdpa"
SKIP_EXISTING_EPISODES="true"
SKIP_FAILED_EPISODES="true"
IMAGE_FORMAT="jpeg"  # 当前保存 JPEG；如需无损图像可改为 png。
PNG_COMPRESS_LEVEL="6"
JPEG_QUALITY="95"
JPEG_SUBSAMPLING="1"

mkdir -p "${OUTPUT_ROOT}"

echo "MODEL_PATH: ${MODEL_PATH}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "REFERENCE_INPUT_ROOT: ${REFERENCE_INPUT_ROOT}"
echo "SOURCE_DATASET_NAMES: ${SOURCE_DATASET_NAMES[*]}"
echo "GPU_IDS: ${GPU_IDS}"
echo "PROCESSES_PER_GPU: ${PROCESSES_PER_GPU}"
echo "CPU_THREADS_PER_WORKER: ${CPU_THREADS_PER_WORKER}"
echo "MIDGOAL_RADIUS: ${MIDGOAL_RADIUS}"
echo "GOAL_RADIUS: ${GOAL_RADIUS}"
echo "ALPHA: ${ALPHA}"
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
    "${PYTHON_BIN}" src/data/generate_dagger_data.py
    --source_dataset_name "${SOURCE_DATASET_NAMES[@]}"
    --model_path "${MODEL_PATH}"
    --output_root "${OUTPUT_ROOT}"
    --reference_input_root "${REFERENCE_INPUT_ROOT}"
    --gpu_ids "${GPU_IDS}"
    --num_thread "${NUM_THREAD}"
    --num_processes_per_gpu "${PROCESSES_PER_GPU}"
    --cpu_threads_per_worker "${CPU_THREADS_PER_WORKER}"
    --max_steps_per_episode "${MAX_STEPS_PER_EPISODE}"
    --midgoal_radius "${MIDGOAL_RADIUS}"
    --goal_radius "${GOAL_RADIUS}"
    --alpha "${ALPHA}"
    --seed "${SEED}"
    --attn_implementation "${ATTN_IMPLEMENTATION}"
    --skip_existing_episodes "${SKIP_EXISTING_EPISODES}"
    --skip_failed_episodes "${SKIP_FAILED_EPISODES}"
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

printf 'Running:'
printf ' %q' "${CMD[@]}"
printf '\n'
CUDA_VISIBLE_DEVICES="${GPU_IDS}" "${CMD[@]}"
