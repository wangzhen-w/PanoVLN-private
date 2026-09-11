#!/usr/bin/env bash
set -euo pipefail

# All available HM3D scenes contribute to one training trajectory dataset.
PYTHON_BIN="/opt/conda/envs/vln/bin/python"
MODE="collect"                          # collect | inspect | validate
HM3D_ROOT="/workspace/data2/dataset/general_VLN_data/HM3D"
PANO_VLN_ROOT="/workspace/data2/dataset/general_VLN_data/PanoVLN"
OUTPUT_ROOT="${PANO_VLN_ROOT}/trajectory"
DATASET_NAME="trajectories"
WORK_DIR="${PANO_VLN_ROOT}/.work/trajectory"
SCENE_IDS=()                            # Empty includes every available scene.
GPU_IDS=(0 1 2 3 4 5 6 7)
PROCESSES_PER_GPU=1
CPU_THREADS_PER_PROCESS=1
VALIDATION_GPU=0
OVERWRITE=false
VISUALIZE=false
VALIDATE_AFTER_COLLECTION=true
CLEAN_WORK_DIR_AFTER_VALIDATION=true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/trajectory/config/default.json"
DATASET_PATH="${OUTPUT_ROOT}/${DATASET_NAME}.json.gz"
STATS_PATH="${OUTPUT_ROOT}/${DATASET_NAME}_stats.json"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quiet"
export GLOG_minloglevel="3"
export PYTHONDONTWRITEBYTECODE="1"

if [[ $# -gt 0 && "$1" != --* ]]; then
    MODE="$1"
    shift
fi
if [[ "${MODE}" == "collect" || "${MODE}" == "inspect" ]]; then
    CMD=(
        "${PYTHON_BIN}" -m dataset_create.trajectory.pipeline --config "${CONFIG_PATH}" collect
        --scene-root "${HM3D_ROOT}" --output-root "${OUTPUT_ROOT}"
        --dataset-name "${DATASET_NAME}" --work-dir "${WORK_DIR}"
        --gpu-device-ids "${GPU_IDS[*]}" --processes-per-gpu "${PROCESSES_PER_GPU}"
    )
    if [[ ${#SCENE_IDS[@]} -gt 0 ]]; then
        CMD+=(--scene-ids "${SCENE_IDS[*]}")
    fi
    if [[ "${OVERWRITE}" == "true" ]]; then
        CMD+=(--overwrite)
    fi
    if [[ "${VISUALIZE}" == "true" ]]; then
        CMD+=(--visualize)
    fi
    if [[ "${MODE}" == "inspect" ]]; then
        exec "${CMD[@]}" --inspect "$@"
    fi
    "${CMD[@]}" "$@"
elif [[ "${MODE}" != "validate" ]]; then
    echo "usage: $0 {collect|inspect|validate}" >&2
    exit 2
fi

if [[ "${MODE}" == "validate" || "${VALIDATE_AFTER_COLLECTION}" == "true" ]]; then
    CMD=(
        "${PYTHON_BIN}" -m dataset_create.trajectory.pipeline --config "${CONFIG_PATH}" validate
        --scene-root "${HM3D_ROOT}" --dataset "${DATASET_PATH}" --stats "${STATS_PATH}"
        --gpu-device-id "${VALIDATION_GPU}"
    )
    if [[ "${CLEAN_WORK_DIR_AFTER_VALIDATION}" == "true" ]]; then
        CMD+=(--cleanup-work-dir "${WORK_DIR}")
    fi
    if [[ "${MODE}" == "validate" ]]; then
        CMD+=("$@")
    fi
    exec "${CMD[@]}"
fi
