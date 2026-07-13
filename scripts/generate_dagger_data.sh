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
MODEL_PATH="/workspace/data1/model/ablation_new/spatial_encoder/Unik3D"
OUTPUT_ROOT="/workspace/data1/dataset/PanoVLN"
REFERENCE_INPUT_ROOT="/workspace/data1/dataset/PanoVLN"
DAGGER_DATASET_NAME="dagger"
SOURCE_DATASET_NAMES=(r2r rxr)
GPU_IDS="0,1,2,3,4,5,6,7"
PROCESSES_PER_GPU="4"
MAX_EPISODES=""
EPISODE_IDS=""
MAX_STEPS_PER_EPISODE="500"
# Expert/ShortestPathFollower waypoint tolerance for oracle action labels.
GOAL_RADIUS="0.25"
# Final distance-to-goal threshold for considering a mixed-policy rollout successful.
SUCCESS_RADIUS="0.5"
ALPHA="0.5"
SEED="42"
ATTN_IMPLEMENTATION="sdpa"
SKIP_EXISTING_EPISODES="true"
SKIP_FAILED_EPISODES="false"

mkdir -p "${OUTPUT_ROOT}"

echo "MODEL_PATH: ${MODEL_PATH}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "REFERENCE_INPUT_ROOT: ${REFERENCE_INPUT_ROOT}"
echo "DAGGER_DATASET_NAME: ${DAGGER_DATASET_NAME}"
echo "SOURCE_DATASET_NAMES: ${SOURCE_DATASET_NAMES[*]}"
echo "GPU_IDS: ${GPU_IDS}"
echo "PROCESSES_PER_GPU: ${PROCESSES_PER_GPU}"
echo "GOAL_RADIUS: ${GOAL_RADIUS}"
echo "SUCCESS_RADIUS: ${SUCCESS_RADIUS}"
echo "ALPHA: ${ALPHA}"

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
    --dagger_dataset_name "${DAGGER_DATASET_NAME}"
    --model_path "${MODEL_PATH}"
    --output_root "${OUTPUT_ROOT}"
    --reference_input_root "${REFERENCE_INPUT_ROOT}"
    --gpu_ids "${GPU_IDS}"
    --num_thread "${NUM_THREAD}"
    --num_processes_per_gpu "${PROCESSES_PER_GPU}"
    --max_steps_per_episode "${MAX_STEPS_PER_EPISODE}"
    --goal_radius "${GOAL_RADIUS}"
    --success_radius "${SUCCESS_RADIUS}"
    --alpha "${ALPHA}"
    --seed "${SEED}"
    --attn_implementation "${ATTN_IMPLEMENTATION}"
    --skip_existing_episodes "${SKIP_EXISTING_EPISODES}"
    --skip_failed_episodes "${SKIP_FAILED_EPISODES}"
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
