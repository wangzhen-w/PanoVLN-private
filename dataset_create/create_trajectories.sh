#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "${SCRIPT_DIR}/.." && pwd )"
cd "${PROJECT_ROOT}"
echo "Switched to project root: ${PROJECT_ROOT}"

export PYTHONPATH="./:${PYTHONPATH:-}"
export MAGNUM_LOG="quiet"
export GLOG_minloglevel="3"
export HABITAT_LAB_LOG="50"
export PYTHONWARNINGS="ignore"
export PYTHONDONTWRITEBYTECODE="1"

# Python command from the currently activated Conda environment. With `(vln)`
# active this resolves to /opt/conda/envs/vln/bin/python.
PYTHON_BIN="python"

# HM3D root. It must contain train/ and val/ scene directories.
HM3D_ROOT="/workspace/data2/dataset/general_VLN_data/HM3D"

# Navigation/decision/route algorithm parameters. Normally this file is not edited.
CONFIG_PATH="${SCRIPT_DIR}/trajectory/config/default.json"

# HM3D split to collect. The current default is val (100 scenes). To operate on
# train instead, run `SPLIT=train ./dataset_create/create_trajectories.sh ...`.
SPLIT="${SPLIT:-train}"

# Parent directory for final datasets. SPLIT=val writes val/val.json.gz and
# val/val_stats.json; SPLIT=train writes train/train.json.gz and train/train_stats.json.
PANO_VLN_ROOT="/workspace/data2/dataset/general_VLN_data/PanoVLN"
OUTPUT_ROOT="${PANO_VLN_ROOT}/${SPLIT}"
DATASET_NAME="${SPLIT}"

# Optional explicit scene keys from SPLIT. Empty means every scene in SPLIT.
# Example: SCENE_IDS=(00000-kfPV7w3FaU5 00001-UVdNNRcVyV1)
SCENE_IDS=()

# GPUs allowed for collection. One persistent Habitat worker is bound to each ID.
GPU_IDS=(0 1 2 3 4 5 6 7)

# Number of Habitat processes per GPU. The validated and enforced value is 1.
PROCESSES_PER_GPU="1"

# GPU used by independent replay validation, which itself runs one process.
VALIDATION_GPU="0"

# true recollects completed scenes; false resumes compatible scene shards.
OVERWRITE="false"

# true saves audit images. Keep false for full production to avoid extra files/time.
VISUALIZE="false"

# true reloads the final dataset and replays every serialized trajectory.
VALIDATE_AFTER_COLLECTION="true"

# true removes hidden scene shards only after independent replay reaches 100%.
CLEAN_WORK_DIR_AFTER_VALIDATION="true"

# First argument: collect runs collection; validate checks an existing dataset.
RUN_MODE="${1:-collect}"

# Only production train and val splits are supported by this launcher.
if [[ "${SPLIT}" != "train" && "${SPLIT}" != "val" ]]; then
    echo "SPLIT must be train or val, got: ${SPLIT}" >&2
    exit 2
fi

# Derived paths; do not edit these independently.
DATASET_PATH="${OUTPUT_ROOT}/${DATASET_NAME}.json.gz"
STATS_PATH="${OUTPUT_ROOT}/${DATASET_NAME}_stats.json"
WORK_DIR="${OUTPUT_ROOT}/.work/${DATASET_NAME}"

if [[ "${PROCESSES_PER_GPU}" != "1" ]]; then
    echo "PROCESSES_PER_GPU must be 1 for the validated Habitat resource policy" >&2
    exit 2
fi
if [[ "${#GPU_IDS[@]}" -eq 0 ]]; then
    echo "GPU_IDS must not be empty" >&2
    exit 2
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "Python command was not found: ${PYTHON_BIN}" >&2
    exit 2
fi
PYTHON_BIN="$(command -v "${PYTHON_BIN}")"
if ! "${PYTHON_BIN}" -c 'import importlib.util, sys; names=("numpy", "scipy", "networkx", "tqdm", "habitat_sim", "habitat"); missing=[name for name in names if importlib.util.find_spec(name) is None]; print("Missing modules: " + ", ".join(missing), file=sys.stderr) if missing else None; sys.exit(bool(missing))'; then
    echo "The selected Python is missing a required trajectory dependency: ${PYTHON_BIN}" >&2
    echo "Activate the vln Conda environment before running this script." >&2
    exit 2
fi
if [[ "${VISUALIZE}" == "true" ]] && ! "${PYTHON_BIN}" -c 'import importlib.util, sys; sys.exit(importlib.util.find_spec("matplotlib") is None)'; then
    echo "VISUALIZE=true requires matplotlib in ${PYTHON_BIN}" >&2
    exit 2
fi

echo "PYTHON_BIN: ${PYTHON_BIN}"
echo "RUN_MODE: ${RUN_MODE}"
echo "HM3D_ROOT: ${HM3D_ROOT}"
echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "SPLIT: ${SPLIT}"
echo "DATASET_NAME: ${DATASET_NAME}"
echo "SCENE_IDS: ${SCENE_IDS[*]:-all}"
echo "GPU_IDS: ${GPU_IDS[*]}"
echo "PROCESSES_PER_GPU: ${PROCESSES_PER_GPU}"
echo "OVERWRITE: ${OVERWRITE}"
echo "VISUALIZE: ${VISUALIZE}"
echo "VALIDATE_AFTER_COLLECTION: ${VALIDATE_AFTER_COLLECTION}"
echo "CLEAN_WORK_DIR_AFTER_VALIDATION: ${CLEAN_WORK_DIR_AFTER_VALIDATION}"

if [[ "${RUN_MODE}" == "collect" ]]; then
    COLLECT_CMD=(
        "${PYTHON_BIN}" -m dataset_create.trajectory.pipeline
        --config "${CONFIG_PATH}"
        collect
        --scene-root "${HM3D_ROOT}"
        --output-root "${OUTPUT_ROOT}"
        --split "${SPLIT}"
        --dataset-name "${DATASET_NAME}"
        --gpu-device-ids "${GPU_IDS[*]}"
        --processes-per-gpu "${PROCESSES_PER_GPU}"
    )
    if [[ "${#SCENE_IDS[@]}" -gt 0 ]]; then
        COLLECT_CMD+=(--scene-ids "${SCENE_IDS[*]}")
    fi
    if [[ "${OVERWRITE}" == "true" ]]; then
        COLLECT_CMD+=(--overwrite)
    fi
    if [[ "${VISUALIZE}" == "true" ]]; then
        COLLECT_CMD+=(--visualize)
    fi

    printf 'Running:'
    printf ' %q' "${COLLECT_CMD[@]}"
    printf '\n'
    "${COLLECT_CMD[@]}"
elif [[ "${RUN_MODE}" != "validate" ]]; then
    echo "usage: $0 {collect|validate}" >&2
    exit 2
fi

if [[ "${RUN_MODE}" == "validate" || "${VALIDATE_AFTER_COLLECTION}" == "true" ]]; then
    VALIDATE_CMD=(
        "${PYTHON_BIN}" -m dataset_create.trajectory.pipeline
        --config "${CONFIG_PATH}"
        validate
        --scene-root "${HM3D_ROOT}"
        --dataset "${DATASET_PATH}"
        --stats "${STATS_PATH}"
        --gpu-device-id "${VALIDATION_GPU}"
    )
    if [[ "${CLEAN_WORK_DIR_AFTER_VALIDATION}" == "true" ]]; then
        VALIDATE_CMD+=(--cleanup-work-dir "${WORK_DIR}")
    fi

    printf 'Running:'
    printf ' %q' "${VALIDATE_CMD[@]}"
    printf '\n'
    "${VALIDATE_CMD[@]}"
fi
