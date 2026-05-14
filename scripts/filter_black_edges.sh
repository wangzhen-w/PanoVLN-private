#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"
echo "Switched to project root: $PROJECT_ROOT"

export PYTHONPATH="./:${PYTHONPATH:-}"
export PYTHONWARNINGS="ignore"
PYTHON_BIN="python"
OUTPUT_ROOT="/workspace/code_dir/a_property/dataset/PanoVLN"
DATASET_NAMES=(r2r)
NUM_WORKERS="60"

cleanup() {
    trap - INT TERM HUP
    if [[ -n "${PYTHON_PID:-}" ]] && kill -0 "${PYTHON_PID}" 2>/dev/null; then
        echo "Stopping filter_black_edges process group: ${PYTHON_PID}" >&2
        kill -TERM "-${PYTHON_PID}" 2>/dev/null || kill -TERM "${PYTHON_PID}" 2>/dev/null || true
        sleep 1
        kill -KILL "-${PYTHON_PID}" 2>/dev/null || kill -KILL "${PYTHON_PID}" 2>/dev/null || true
    fi
    exit 130
}
trap cleanup INT TERM HUP

echo "OUTPUT_ROOT: ${OUTPUT_ROOT}"
echo "DATASET_NAMES: ${DATASET_NAMES[*]}"
echo "NUM_WORKERS: ${NUM_WORKERS}"

setsid "${PYTHON_BIN}" src/data/filter_black_edges.py \
    --output_root "${OUTPUT_ROOT}" \
    --dataset_name "${DATASET_NAMES[@]}" \
    --num_workers "${NUM_WORKERS}" &
PYTHON_PID=$!
wait "${PYTHON_PID}"
