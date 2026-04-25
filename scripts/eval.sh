#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"
echo "Switched to directory: $PROJECT_ROOT"

export MAGNUM_LOG=quiet
export HABITAT_SIM_LOG=quiet
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

MODEL_PATH="/workspace/data/wz_data/model/Trained/erp_linear_mem_100_10"
CONFIG_PATH="config/vln_r2r.yaml"
SAVE_PATH="/workspace/data/wz_data/vln_result/erp_linear_mem_100_10"
ATTN_IMPLEMENTATION="flash_attention_2"
MAX_MEMORY_IMAGES=10
MEMORY_POOL_WINDOW_FRAMES=100
TOTAL_MAX_EPISODES=0
EARLY_STOP_MAX_STEPS=0

GPU_IDS="0,1,2,3,6,7"
PROCS_PER_GPU=2
MAX_EPISODES=0
SAVE_TOPDOWN=false
SEED=42

mkdir -p "$SAVE_PATH"

IFS=',' read -r -a gpu_ids <<< "$GPU_IDS"
GPU_NUM="${#gpu_ids[@]}"
CHUNKS=$((GPU_NUM * PROCS_PER_GPU))

if [ "$CHUNKS" -le 0 ]; then
    echo "No eval workers configured."
    exit 1
fi

if [ ! -d "$MODEL_PATH" ]; then
    echo "Model path does not exist: $MODEL_PATH"
    exit 1
fi

echo "MODEL_PATH=$MODEL_PATH"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "SAVE_PATH=$SAVE_PATH"
echo "GPU_IDS=$GPU_IDS"
echo "PROCS_PER_GPU=$PROCS_PER_GPU"
echo "MAX_EPISODES=$MAX_EPISODES"
echo "TOTAL_MAX_EPISODES=$TOTAL_MAX_EPISODES"
echo "SAVE_TOPDOWN=$SAVE_TOPDOWN"
echo "SEED=$SEED"
echo "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
echo "MAX_MEMORY_IMAGES=$MAX_MEMORY_IMAGES"
echo "MEMORY_POOL_WINDOW_FRAMES=$MEMORY_POOL_WINDOW_FRAMES"
echo "EARLY_STOP_MAX_STEPS=$EARLY_STOP_MAX_STEPS"
echo "Total processes: $CHUNKS"

pids=()

cleanup_children() {
    if [ "${#pids[@]}" -eq 0 ]; then
        return
    fi

    echo
    echo "Stopping background eval processes..."
    for pid in "${pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait "${pids[@]}" 2>/dev/null || true
}

handle_interrupt() {
    trap - INT TERM
    cleanup_children
    exit 130
}

trap handle_interrupt INT TERM

split_id=0
for gpu_id in "${gpu_ids[@]}"; do
    for ((proc_idx=0; proc_idx<PROCS_PER_GPU; proc_idx++)); do
        CUDA_VISIBLE_DEVICES="$gpu_id" python src/eval/eval.py \
            --exp-config "$CONFIG_PATH" \
            --split-num "$CHUNKS" \
            --split-id "$split_id" \
            --forward-distance 25 \
            --turn-angle 15 \
            --max-memory-images "$MAX_MEMORY_IMAGES" \
            --memory-pool-window-frames "$MEMORY_POOL_WINDOW_FRAMES" \
            --model-path "$MODEL_PATH" \
            --max-episodes "$MAX_EPISODES" \
            --total-max-episodes "$TOTAL_MAX_EPISODES" \
            --save-topdown "$SAVE_TOPDOWN" \
            --seed "$SEED" \
            --attn-implementation "$ATTN_IMPLEMENTATION" \
            --early-stop-max-steps "$EARLY_STOP_MAX_STEPS" \
            --result-path "$SAVE_PATH" &
        pids+=($!)
        split_id=$((split_id + 1))
    done
done

status=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        status=1
    fi
done

trap - INT TERM

if [ "$status" -ne 0 ]; then
    echo "One or more eval workers failed."
    exit "$status"
fi

python src/eval/analyze_results.py --path "$SAVE_PATH"
