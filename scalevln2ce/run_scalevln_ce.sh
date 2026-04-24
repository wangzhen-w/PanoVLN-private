#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"
echo "Switched to directory: $SCRIPT_DIR"

PYTHON_BIN="python"
PY_SCRIPT="generate_scalevln_ce.py"

# 运行模式:
#   smoke 只做小样本自检，然后删除 smoke 临时目录
#   build 只构建 10 个 VLN-CE subset episode 文件
#   gt    只为已有 subset 生成 gt 文件
#   full  依次执行 smoke -> build -> gt
MODE="${1:-full}"

# 主输出配置
OUTPUT_ROOT="/workspace/data_dir/dataset/general_VLN_data/ScaleVLN_CE"
SUBSET_PREFIX="subset"
NUM_SUBSETS=10
SUBSET_SIZE=150000
GOAL_RADIUS=0.3
RAW_TOTAL=2891134
RAW_LOG_EVERY=100000
GT_LOG_EVERY=200
GPU_DEVICE_ID=0

# smoke test 配置
SMOKE_SUBSET_SIZE=8
SMOKE_MAX_EPISODES=8
SMOKE_RAW_LOG_EVERY=10

# 运行开关
MINIMAL_OBSERVATIONS=1
SORT_BY_SCENE=1
RESUME_GT=1
OVERWRITE_BUILD=0

SMOKE_ROOT=""

cleanup_temp_artifacts() {
  local root="$1"
  [[ -d "${root}" ]] || return 0
  find "${root}" -type d -name '.scene_batches' -prune -exec rm -rf {} + 2>/dev/null || true
  find "${root}" -type f -name '*.tmp' -delete 2>/dev/null || true
}

usage() {
  cat <<'EOF'
Usage:
  bash run_scalevln_ce.sh
  bash run_scalevln_ce.sh smoke
  bash run_scalevln_ce.sh build
  bash run_scalevln_ce.sh gt

Default mode is `full`, which runs:
  1. smoke test
  2. build 10 x 150k subsets
  3. generate GT for all subsets

Three core modes:
  smoke
    - make a temporary output root under ScaleVLN_CE
    - build 1 tiny subset
    - generate tiny GT
    - validate file structure and fields
    - keep smoke outputs on disk for manual inspection

  build
    - read ScaleVLN_total
    - remove trajectories already present in ScaleVLN_150k
    - convert discrete paths to VLN-CE episodes
    - distribute each scene across all subsets in a round-robin way
    - write subset_00 ... subset_09
    - does not generate GT

  gt
    - read existing subset_XX/scalevln_subset_150k.json.gz
    - group episodes by scene
    - run ShortestPathFollower(goal_radius=0.3)
    - write subset_XX/scalevln_subset_150k_gt.json.gz
    - resumes from intermediate jsonl when RESUME_GT=1

How to change parameters:
  edit the variable block at the top of this shell script directly
EOF
}

append_optional_gt_flags() {
  local -n args_ref=$1
  if [[ "${MINIMAL_OBSERVATIONS}" == "1" ]]; then
    args_ref+=(--minimal-observations)
  fi
  if [[ "${SORT_BY_SCENE}" == "1" ]]; then
    args_ref+=(--sort-by-scene)
  fi
  if [[ "${RESUME_GT}" == "1" ]]; then
    args_ref+=(--resume)
  fi
}

all_subset_datasets_exist() {
  local idx dataset_path
  for ((idx = 0; idx < NUM_SUBSETS; idx++)); do
    printf -v dataset_path '%s/%s_%02d/scalevln_subset_150k.json.gz' "${OUTPUT_ROOT}" "${SUBSET_PREFIX}" "${idx}"
    [[ -f "${dataset_path}" ]] || return 1
  done
  return 0
}

run_smoke_test() {
  mkdir -p "${OUTPUT_ROOT}"
  SMOKE_ROOT="$(mktemp -d "${OUTPUT_ROOT}/.smoke_test.XXXXXX")"

  echo "[smoke] output root: ${SMOKE_ROOT}"
  "${PYTHON_BIN}" "${PY_SCRIPT}" build-subsets \
    --output-root "${SMOKE_ROOT}" \
    --subset-prefix "${SUBSET_PREFIX}" \
    --num-subsets 1 \
    --subset-size "${SMOKE_SUBSET_SIZE}" \
    --goal-radius "${GOAL_RADIUS}" \
    --raw-total "${RAW_TOTAL}" \
    --log-every "${SMOKE_RAW_LOG_EVERY}" \
    --overwrite

  local gt_args=(
    generate-gt
    --output-root "${SMOKE_ROOT}"
    --subset-prefix "${SUBSET_PREFIX}"
    --subset-indices 0
    --goal-radius "${GOAL_RADIUS}"
    --gpu-device-id "${GPU_DEVICE_ID}"
    --gt-log-every 1
    --max-episodes "${SMOKE_MAX_EPISODES}"
  )
  append_optional_gt_flags gt_args
  "${PYTHON_BIN}" "${PY_SCRIPT}" "${gt_args[@]}"

  SMOKE_ROOT="${SMOKE_ROOT}" "${PYTHON_BIN}" - <<'PY'
import gzip
import json
import os

root = os.environ["SMOKE_ROOT"]
subset_dir = os.path.join(root, "subset_00")
dataset_path = os.path.join(subset_dir, "scalevln_subset_150k.json.gz")
gt_path = os.path.join(subset_dir, "scalevln_subset_150k_gt.json.gz")

assert os.path.exists(dataset_path), dataset_path
assert os.path.exists(gt_path), gt_path

with gzip.open(dataset_path, "rt", encoding="utf-8") as f:
    dataset = json.load(f)
episodes = dataset["episodes"]
assert len(episodes) > 0

with gzip.open(gt_path, "rt", encoding="utf-8") as f:
    gt = json.load(f)
assert len(gt) > 0

episode_ids = {str(ep["episode_id"]) for ep in episodes}
gt_ids = set(gt.keys())
assert gt_ids.issubset(episode_ids)

for episode_id, row in gt.items():
    assert "locations" in row and row["locations"], episode_id
    assert "actions" in row, episode_id
    assert row["forward_steps"] == len(row["locations"]) - 1, episode_id

print(f"[smoke] validated {len(gt)} gt entries")
PY

  cleanup_temp_artifacts "${SMOKE_ROOT}"
  echo "[smoke] passed"
  echo "[smoke] outputs kept at: ${SMOKE_ROOT}"
}

run_build() {
  mkdir -p "${OUTPUT_ROOT}"
  if [[ "${OVERWRITE_BUILD}" != "1" ]] && all_subset_datasets_exist; then
    echo "[build] subset datasets already exist, skip rebuild"
    return
  fi

  local build_args=(
    build-subsets
    --output-root "${OUTPUT_ROOT}"
    --subset-prefix "${SUBSET_PREFIX}"
    --num-subsets "${NUM_SUBSETS}"
    --subset-size "${SUBSET_SIZE}"
    --goal-radius "${GOAL_RADIUS}"
    --raw-total "${RAW_TOTAL}"
    --log-every "${RAW_LOG_EVERY}"
  )
  if [[ "${OVERWRITE_BUILD}" == "1" ]]; then
    build_args+=(--overwrite)
  fi

  "${PYTHON_BIN}" "${PY_SCRIPT}" "${build_args[@]}"
  cleanup_temp_artifacts "${OUTPUT_ROOT}"
}

run_gt() {
  mkdir -p "${OUTPUT_ROOT}"
  local idx
  for ((idx = 0; idx < NUM_SUBSETS; idx++)); do
    echo "[gt] subset ${idx}"
    local gt_args=(
      generate-gt
      --output-root "${OUTPUT_ROOT}"
      --subset-prefix "${SUBSET_PREFIX}"
      --subset-indices "${idx}"
      --goal-radius "${GOAL_RADIUS}"
      --gpu-device-id "${GPU_DEVICE_ID}"
      --gt-log-every "${GT_LOG_EVERY}"
    )
    append_optional_gt_flags gt_args
    "${PYTHON_BIN}" "${PY_SCRIPT}" "${gt_args[@]}"
    cleanup_temp_artifacts "${OUTPUT_ROOT}/${SUBSET_PREFIX}_$(printf '%02d' "${idx}")"
  done
}

case "${MODE}" in
  smoke)
    run_smoke_test
    ;;
  build)
    run_build
    ;;
  gt)
    run_gt
    ;;
  full)
    run_smoke_test
    run_build
    run_gt
    ;;
  -h|--help|help)
    usage
    ;;
  *)
    usage
    exit 1
    ;;
esac
