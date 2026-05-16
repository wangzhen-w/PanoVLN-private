#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

# source /opt/conda/bin/activate vllm

PYTHON_BIN="python"
PY_SCRIPT="${SCRIPT_DIR}/rewrite_scalevln_instructions.py"

INPUT_JSONL="/workspace/code_dir/a_property/dataset/PanoVLN/sub_dataset/scalevln.jsonl"
IMAGE_ROOT="/workspace/code_dir/a_property/dataset/PanoVLN/images/scalevln"
OUTPUT_JSONL="/workspace/code_dir/a_property/dataset/PanoVLN/sub_dataset/scalevln_qwen35_27b_r2rstyle.jsonl"

BASE_URL="http://172.17.0.4:10823/v1"
MODEL="Qwen"
API_KEY="EMPTY"

MAX_EPISODES=0
SAMPLE_MODE="first"
SEED=42

NUM_WORKERS=16
MAX_WAYPOINTS=12
START_WINDOW_FRAMES=6
ENDPOINT_WINDOW_FRAMES=6
TILE_WIDTH=256
TILE_HEIGHT=192
JPEG_QUALITY=85

TEMPERATURE=0.50
MAX_TOKENS=340
FACT_MAX_TOKENS=300
PLANNER_MAX_TOKENS=900
REVIEW_MAX_TOKENS=800
REQUEST_TIMEOUT=180

RETRIES=4
STAGE_RETRIES=2
SELF_CHECK=true
ROUTE_AUDIT=true
SPATIAL_AUDIT=true

pids=()

cleanup_children() {
  if [ "${#pids[@]}" -eq 0 ]; then
    return
  fi

  echo
  echo "Stopping instruction rewrite processes..."
  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -TERM -- "-$pid" 2>/dev/null || kill "$pid" 2>/dev/null || true
    fi
  done

  sleep 2

  for pid in "${pids[@]}"; do
    if kill -0 "$pid" 2>/dev/null; then
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
    fi
  done
  wait "${pids[@]}" 2>/dev/null || true
}

handle_interrupt() {
  trap - INT TERM
  cleanup_children
  exit 130
}

trap handle_interrupt INT TERM

setsid "${PYTHON_BIN}" "${PY_SCRIPT}" \
  --input-jsonl "${INPUT_JSONL}" \
  --image-root "${IMAGE_ROOT}" \
  --output-jsonl "${OUTPUT_JSONL}" \
  --provider qwen \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --api-key "${API_KEY}" \
  --sample-mode "${SAMPLE_MODE}" \
  --seed "${SEED}" \
  --max-episodes "${MAX_EPISODES}" \
  --num-workers "${NUM_WORKERS}" \
  --max-waypoints "${MAX_WAYPOINTS}" \
  --start-window-frames "${START_WINDOW_FRAMES}" \
  --endpoint-window-frames "${ENDPOINT_WINDOW_FRAMES}" \
  --tile-width "${TILE_WIDTH}" \
  --tile-height "${TILE_HEIGHT}" \
  --jpeg-quality "${JPEG_QUALITY}" \
  --temperature "${TEMPERATURE}" \
  --max-tokens "${MAX_TOKENS}" \
  --fact-max-tokens "${FACT_MAX_TOKENS}" \
  --planner-max-tokens "${PLANNER_MAX_TOKENS}" \
  --review-max-tokens "${REVIEW_MAX_TOKENS}" \
  --request-timeout "${REQUEST_TIMEOUT}" \
  --retries "${RETRIES}" \
  --stage-retries "${STAGE_RETRIES}" \
  --start-fact-pass true \
  --endpoint-fact-pass true \
  --route-plan-pass true \
  --require-start-facts true \
  --require-endpoint-facts true \
  --require-route-plan true \
  --self-check "${SELF_CHECK}" \
  --route-audit "${ROUTE_AUDIT}" \
  --spatial-audit "${SPATIAL_AUDIT}" \
  --drop-failed true \
  --resume true &
pids+=($!)

set +e
wait "${pids[0]}"
status=$?
set -e

trap - INT TERM

if [ "$status" -ne 0 ]; then
  exit "$status"
fi

echo "Clean output: ${OUTPUT_JSONL}"
echo "Progress directory: ${OUTPUT_JSONL%.jsonl}_progress"
echo "Merged metadata: ${OUTPUT_JSONL%.jsonl}_progress/candidates.jsonl"
echo "Failed metadata: ${OUTPUT_JSONL%.jsonl}_progress/failed.jsonl"
echo "Summary: ${OUTPUT_JSONL%.jsonl}_progress/summary.json"
