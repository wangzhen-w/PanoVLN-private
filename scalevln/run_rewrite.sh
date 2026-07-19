#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="python"

# 只需要修改这三个数据路径。
SOURCE_JSONL="/workspace/data1/dataset/PanoVLN/sub_dataset/scalevln.jsonl"
IMAGE_ROOT="/workspace/data1/dataset/PanoVLN/images/scalevln"
REWRITE_OUTPUT="/workspace/data1/dataset/PanoVLN/sub_dataset/scalevln_qwen36_27b_panovln.jsonl"

BASE_URL="http://127.0.0.1:10420/v1"
MODEL="Qwen3.6-27B"
API_KEY="test"

NUM_WORKERS=40
MAX_WAYPOINTS=12
ROUTE_EVIDENCE_MODE="auto"  # 长路线或多平移、多转向路线自动拆成局部视觉段。
SEGMENTED_MIN_ACTIONS=80  # 达到该长度必分段；复杂短路线也会由 auto 模式识别。
SEGMENT_MAX_WAYPOINTS=0  # 0 表示复用 MAX_WAYPOINTS。
SEGMENT_ROWS=5
SEGMENT_OVERLAP=1
START_WINDOW_FRAMES=6
ENDPOINT_WINDOW_FRAMES=6
CANDIDATE_COUNT=2  # 每条路线生成多个候选后由独立视觉 judge 选择，降低单次采样回归。
CANDIDATE_TEMPERATURE=0.4
TILE_WIDTH=320
TILE_HEIGHT=240
SHEET_JPEG_QUALITY=90
TEMPERATURE=0.40
PLANNER_TEMPERATURE=0.0
REVIEW_TEMPERATURE=0.0
MAX_TOKENS=420
FACT_MAX_TOKENS=320
SEGMENT_FACT_MAX_TOKENS=650
PLANNER_MAX_TOKENS=1600
REVIEW_MAX_TOKENS=1200
REQUEST_TIMEOUT=300
RETRIES=4
STAGE_RETRIES=2

WORK_DIR="${REWRITE_OUTPUT%.jsonl}_progress"

# 避免系统 HTTP 代理截获本机 Qwen API 请求。
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

"${PYTHON_BIN}" -m data_create.instruction.pipeline \
  --input-jsonl "${SOURCE_JSONL}" \
  --image-root "${IMAGE_ROOT}" \
  --output-jsonl "${REWRITE_OUTPUT}" \
  --work-dir "${WORK_DIR}" \
  --mode generate \
  --instruction-profile concise \
  --provider qwen \
  --base-url "${BASE_URL}" \
  --model "${MODEL}" \
  --api-key "${API_KEY}" \
  --num-workers "${NUM_WORKERS}" \
  --max-waypoints "${MAX_WAYPOINTS}" \
  --route-evidence-mode "${ROUTE_EVIDENCE_MODE}" \
  --segmented-min-actions "${SEGMENTED_MIN_ACTIONS}" \
  --segment-max-waypoints "${SEGMENT_MAX_WAYPOINTS}" \
  --segment-rows "${SEGMENT_ROWS}" \
  --segment-overlap "${SEGMENT_OVERLAP}" \
  --start-window-frames "${START_WINDOW_FRAMES}" \
  --endpoint-window-frames "${ENDPOINT_WINDOW_FRAMES}" \
  --tile-width "${TILE_WIDTH}" \
  --tile-height "${TILE_HEIGHT}" \
  --jpeg-quality "${SHEET_JPEG_QUALITY}" \
  --temperature "${TEMPERATURE}" \
  --planner-temperature "${PLANNER_TEMPERATURE}" \
  --review-temperature "${REVIEW_TEMPERATURE}" \
  --candidate-count "${CANDIDATE_COUNT}" \
  --candidate-temperature "${CANDIDATE_TEMPERATURE}" \
  --max-tokens "${MAX_TOKENS}" \
  --fact-max-tokens "${FACT_MAX_TOKENS}" \
  --segment-fact-max-tokens "${SEGMENT_FACT_MAX_TOKENS}" \
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
  --self-check true \
  --blind-grounding-audit true \
  --route-audit true \
  --spatial-audit true \
  --drop-failed false \
  --allow-incomplete false \
  --resume true

rm -rf "${WORK_DIR}"
echo "Rewrite: ${REWRITE_OUTPUT}"
