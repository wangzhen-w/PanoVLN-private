#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

CONFIG_PATH="config/config.yaml"
OUTPUT_DIR="/workspace/data1/model/ablation_new/panovggt_new/panovggt_0.05_panoworld_8card"
GPU_DEVICES="0,1,2,3,4,5,6,7"
GPU_NUM="$(awk -F',' '{print NF}' <<< "$GPU_DEVICES")"
MASTER_ADDR="127.0.0.1"
MASTER_PORT="29520"

mkdir -p "$OUTPUT_DIR"

if ! command -v torchrun >/dev/null 2>&1; then
    echo "torchrun not found in PATH" >&2
    exit 1
fi

export NCCL_NVLS_ENABLE=0
export TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

CUDA_VISIBLE_DEVICES="$GPU_DEVICES" torchrun \
    --nproc_per_node="$GPU_NUM" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    train.py \
    --config "$CONFIG_PATH" \
    "$@" \
    2>&1 | tee "$OUTPUT_DIR/train.log"
