#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
echo "Switched to directory: $PROJECT_ROOT"

HOST="0.0.0.0"
PORT="8000"
MODEL_PATH="/workspace/data1/model/ablation_new/spatial_encoder/DAP"
GPU_IDS="4"
DAP_SOURCE_PATH="bundled"
DAP_MODEL_PATH=""
ATTN_IMPLEMENTATION="flash_attention_2"
MAX_MEMORY_IMAGES="10"
MEMORY_POOL_WINDOW_FRAMES="100"

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model path does not exist: $MODEL_PATH" >&2
    exit 1
fi

echo "HOST=$HOST"
echo "PORT=$PORT"
echo "MODEL_PATH=$MODEL_PATH"
echo "GPU_IDS=$GPU_IDS"
echo "DAP_SOURCE_PATH=$DAP_SOURCE_PATH"
echo "DAP_MODEL_PATH=${DAP_MODEL_PATH:-<VLN checkpoint>}"
echo "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
echo "MAX_MEMORY_IMAGES=$MAX_MEMORY_IMAGES"
echo "MEMORY_POOL_WINDOW_FRAMES=$MEMORY_POOL_WINDOW_FRAMES"

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PROJECT_ROOT}/src:${PYTHONPATH:-}"

CMD=(
    python3 -m realworld.server
    --host "$HOST"
    --port "$PORT"
    --model-path "$MODEL_PATH"
    --dap-source-path "$DAP_SOURCE_PATH"
    --attn-implementation "$ATTN_IMPLEMENTATION"
    --max-memory-images "$MAX_MEMORY_IMAGES"
    --memory-pool-window-frames "$MEMORY_POOL_WINDOW_FRAMES"
)

if [[ -n "$DAP_MODEL_PATH" ]]; then
    CMD+=(--dap-model-path "$DAP_MODEL_PATH")
fi

echo "Starting server; model will load before the HTTP interface is available..."
echo "CUDA_VISIBLE_DEVICES=$GPU_IDS ${CMD[*]}"
CUDA_VISIBLE_DEVICES="$GPU_IDS" "${CMD[@]}"
