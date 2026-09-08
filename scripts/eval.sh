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

MODEL_PATH="/workspace/data2/model/ablation_new/panovggt_pre_merger/panovggt_0.30_lr2e-5_singlepoint_8card"
CONFIG_PATH="config/vln_r2r.yaml"
SAVE_PATH="/workspace/code/vln_result/ablation_new/panovggt_pre_merger/panovggt_0.30_lr2e-5_singlepoint_8card_uncertainty1.2_k4-8_stop12"
ATTN_IMPLEMENTATION="flash_attention_2"
MAX_MEMORY_IMAGES=10
MEMORY_POOL_WINDOW_FRAMES=100
# Positive integer: fixed execution length.
# "uncertainty": choose K from this prediction's logits, within REPLAN_ACTION_RANGE.
# Use a separate SAVE_PATH for each mode/range/budget/stop-window/seed; existing episodes are skipped.
ACTIONS_PER_REPLAN="uncertainty"
# Inclusive (minimum maximum) for uncertainty; fixed integer K is unaffected.
# Bash array syntax uses a space, not a comma
REPLAN_ACTION_RANGE=(4 8)
# Execute through STOP if it appears within the first N actions; 0 disables.
STOP_COMMIT_MAX_ACTIONS=12
# Budget for sum(-log p(action)) in uncertainty mode.
# This is a fixed input parameter, not recomputed from online episode history.
UNCERTAINTY_BUDGET=1.2
TOTAL_MAX_EPISODES=0
EARLY_STOP_MAX_STEPS=0

GPU_IDS="0,1,2,3"
PROCS_PER_GPU=3
CPU_THREADS_PER_WORKER=4
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

actions_per_replan_args=()
if [[ "$ACTIONS_PER_REPLAN" != "uncertainty" &&
      ! "$ACTIONS_PER_REPLAN" =~ ^[1-9][0-9]*$ ]]; then
    echo "ACTIONS_PER_REPLAN must be a positive integer or uncertainty: $ACTIONS_PER_REPLAN" >&2
    exit 1
fi
if [[ ! "$STOP_COMMIT_MAX_ACTIONS" =~ ^(0|[1-9][0-9]*)$ ]]; then
    echo "STOP_COMMIT_MAX_ACTIONS must be a nonnegative integer (0 disables): $STOP_COMMIT_MAX_ACTIONS" >&2
    exit 1
fi
actions_per_replan_args=(--actions-per-replan "$ACTIONS_PER_REPLAN"
                         --stop-commit-max-actions "$STOP_COMMIT_MAX_ACTIONS")
if [[ "$ACTIONS_PER_REPLAN" == "uncertainty" ]]; then
    actions_per_replan_args+=(--uncertainty-budget "$UNCERTAINTY_BUDGET")
    if [[ "${#REPLAN_ACTION_RANGE[@]}" -ne 2 ]] ||
       [[ ! "${REPLAN_ACTION_RANGE[0]}" =~ ^[1-9][0-9]*$ ||
          ! "${REPLAN_ACTION_RANGE[1]}" =~ ^[1-9][0-9]*$ ]]; then
        echo "REPLAN_ACTION_RANGE must contain two positive integers: (MIN MAX)" >&2
        exit 1
    fi
    if (( REPLAN_ACTION_RANGE[0] > REPLAN_ACTION_RANGE[1] )); then
        echo "REPLAN_ACTION_RANGE requires MIN <= MAX" >&2
        exit 1
    fi
    actions_per_replan_args+=(--replan-action-range "${REPLAN_ACTION_RANGE[@]}")
fi

echo "MODEL_PATH=$MODEL_PATH"
echo "CONFIG_PATH=$CONFIG_PATH"
echo "SAVE_PATH=$SAVE_PATH"
echo "GPU_IDS=$GPU_IDS"
echo "PROCS_PER_GPU=$PROCS_PER_GPU"
echo "CPU_THREADS_PER_WORKER=$CPU_THREADS_PER_WORKER"
echo "MAX_EPISODES=$MAX_EPISODES"
echo "TOTAL_MAX_EPISODES=$TOTAL_MAX_EPISODES"
echo "SAVE_TOPDOWN=$SAVE_TOPDOWN"
echo "SEED=$SEED"
echo "ATTN_IMPLEMENTATION=$ATTN_IMPLEMENTATION"
echo "MAX_MEMORY_IMAGES=$MAX_MEMORY_IMAGES"
echo "MEMORY_POOL_WINDOW_FRAMES=$MEMORY_POOL_WINDOW_FRAMES"
echo "ACTIONS_PER_REPLAN=$ACTIONS_PER_REPLAN"
echo "REPLAN_ACTION_RANGE=${REPLAN_ACTION_RANGE[*]}"
echo "UNCERTAINTY_BUDGET=$UNCERTAINTY_BUDGET"
echo "STOP_COMMIT_MAX_ACTIONS=$STOP_COMMIT_MAX_ACTIONS"
echo "EARLY_STOP_MAX_STEPS=$EARLY_STOP_MAX_STEPS"
echo "Total processes: $CHUNKS"

export VLN_EVAL_CPU_THREADS_PER_WORKER="$CPU_THREADS_PER_WORKER"

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
            "${actions_per_replan_args[@]}" \
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
