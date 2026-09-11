#!/usr/bin/env bash
set -euo pipefail

# Edit these assignments for the current run. Appended CLI options override them.
PYTHON_BIN="/opt/conda/envs/vln/bin/python"
MODE="generate"                         # inspect | render | generate | export | validate
TRAJECTORIES="/workspace/data2/dataset/general_VLN_data/PanoVLN/trajectory/trajectories.json.gz"
SCENE_ROOT="/workspace/data2/dataset/general_VLN_data/HM3D"
OUTPUT_ROOT="/workspace/data2/dataset/general_VLN_data/PanoVLN"
NAME="train"
ERP_ROOT="${OUTPUT_ROOT}/image"
WORK_DIR="${OUTPUT_ROOT}/.work/instruction"

NUM_PROCESSES=48
GPU_DEVICE_IDS=(0 1 2 3 4 5 6 7)         # Six Habitat processes per GPU; Qwen is a separate service.
CPU_THREADS_PER_PROCESS=1               # Keep numerical libraries from oversubscribing CPU cores.
LIMIT=0                                 # 0 processes all input trajectories; a positive value caps the count.
SELECTION="first"                       # first | diverse; selection strategy matters only when LIMIT > 0.
ERP_WIDTH=1600
ERP_HEIGHT=800
JPEG_QUALITY=95
KEEP_WORK=false                         # Debug only. Final output is ERP + R2R JSON/gzip.

BASE_URL="http://127.0.0.1:10420/v1"
MODEL_NAME="Qwen3.8-27B"
API_KEY="test"
MEDIA_MODE="video"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONFIG_PATH="${SCRIPT_DIR}/instruction/config/default.json"
cd "${PROJECT_ROOT}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export OMP_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export OPENBLAS_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export MKL_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export NUMEXPR_NUM_THREADS="${CPU_THREADS_PER_PROCESS}"
export QWEN_API_KEY="${API_KEY}"
export MAGNUM_LOG="quiet"
export HABITAT_SIM_LOG="quiet"
export GLOG_minloglevel="3"
export PYTHONDONTWRITEBYTECODE="1"

if [[ $# -gt 0 && "$1" != --* ]]; then
    MODE="$1"
    shift
fi
if [[ "${MODE}" == "validate" ]]; then
    exec "${PYTHON_BIN}" -m dataset_create.instruction.pipeline validate \
        --dataset "${OUTPUT_ROOT}/${NAME}.json.gz" --scene-root "${SCENE_ROOT}" "$@"
fi

CMD=(
    "${PYTHON_BIN}" -m dataset_create.instruction.pipeline "${MODE}"
    --trajectories "${TRAJECTORIES}" --scene-root "${SCENE_ROOT}"
    --output-root "${OUTPUT_ROOT}" --name "${NAME}" --erp-root "${ERP_ROOT}"
    --work-dir "${WORK_DIR}" --config "${CONFIG_PATH}"
    --limit "${LIMIT}" --selection "${SELECTION}"
    --processes "${NUM_PROCESSES}" --gpu-device-ids "${GPU_DEVICE_IDS[@]}"
    --erp-width "${ERP_WIDTH}" --erp-height "${ERP_HEIGHT}" --jpeg-quality "${JPEG_QUALITY}"
    --base-url "${BASE_URL}" --model "${MODEL_NAME}" --media-mode "${MEDIA_MODE}"
)
if [[ "${KEEP_WORK}" == "true" ]]; then
    CMD+=(--keep-work)
fi
exec "${CMD[@]}" "$@"
