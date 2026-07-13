#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="python"
PY_SCRIPT="${SCRIPT_DIR}/generate_scalevln_ce.py"

# 正式 dataset 只需 build；gt/full 仅在制作期需要立即生成 action evidence 时使用。
MODE="build"

# 主输出配置
RAW_ANNOTATIONS="/workspace/data1/dataset/general_VLN_data/ScaleVLN_total/annotations/R2R_scalevln_ft_aug_enc.json"  # 原始离散 ScaleVLN annotation。
EXISTING_SUBSET="/workspace/data1/dataset/general_VLN_data/ScaleVLN_150k/scalevln_subset_150k.json.gz"  # 去重用的官方 150k 子集。
CONNECTIVITY_DIR="/workspace/data1/dataset/general_VLN_data/ScaleVLN_total/connectivity"  # HM3D connectivity。
CONNECTIVITY_MP3D_DIR="/workspace/data1/dataset/general_VLN_data/ScaleVLN_total/connectivity_mp3d"  # MP3D connectivity。
SCENES_DIR="/workspace/data1/dataset/janusvln_data/scene_datasets"  # Habitat 场景统一根目录。
CONFIG_PATH="${REPO_ROOT}/config/vln_scalevln.yaml"  # GT 回放使用的 Habitat 配置。
OUTPUT_ROOT="/workspace/data1/dataset/general_VLN_data/ScaleVLN_CE"
SUBSET_PREFIX="subset"
NUM_SUBSETS=10
SUBSET_SIZE=150000
GOAL_RADIUS=0.3
RAW_TOTAL=2891134
RAW_LOG_EVERY=100000
GT_LOG_EVERY=200
GPU_DEVICE_ID=0

# 运行开关
MINIMAL_OBSERVATIONS=1
SORT_BY_SCENE=1
RESUME_GT=1
OVERWRITE_BUILD=0

cleanup_temp_artifacts() {
  local root="$1"
  [[ -d "${root}" ]] || return 0
  find "${root}" -type d -name '.scene_batches' -prune -exec rm -rf {} + 2>/dev/null || true
  find "${root}" -type f -name '*.tmp' -delete 2>/dev/null || true
}

cleanup_non_dataset_artifacts() {
  local root="$1"
  [[ -d "${root}" ]] || return 0
  find "${root}" -mindepth 2 -maxdepth 2 -type f \
    \( -name 'manifest.json' \
       -o -name 'scalevln_subset_150k_gt.json.gz' \
       -o -name 'scalevln_subset_150k_gt.jsonl' \) -delete
}

usage() {
  echo "Unknown MODE=${MODE}; edit MODE at the top of run_scalevln_ce.sh." >&2
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

run_build() {
  mkdir -p "${OUTPUT_ROOT}"
  if [[ "${OVERWRITE_BUILD}" != "1" ]] && all_subset_datasets_exist; then
    cleanup_temp_artifacts "${OUTPUT_ROOT}"
    cleanup_non_dataset_artifacts "${OUTPUT_ROOT}"
    echo "[build] subset datasets already exist, skip rebuild"
    return
  fi

  local build_args=(
    build-subsets
    --raw-annotations "${RAW_ANNOTATIONS}"
    --existing-subset "${EXISTING_SUBSET}"
    --connectivity-dir "${CONNECTIVITY_DIR}"
    --connectivity-mp3d-dir "${CONNECTIVITY_MP3D_DIR}"
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
  cleanup_non_dataset_artifacts "${OUTPUT_ROOT}"
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
      --scenes-dir "${SCENES_DIR}"
      --config-path "${CONFIG_PATH}"
      --repo-root "${REPO_ROOT}"
      --gpu-device-id "${GPU_DEVICE_ID}"
      --gt-log-every "${GT_LOG_EVERY}"
    )
    append_optional_gt_flags gt_args
    "${PYTHON_BIN}" "${PY_SCRIPT}" "${gt_args[@]}"
    cleanup_temp_artifacts "${OUTPUT_ROOT}/${SUBSET_PREFIX}_$(printf '%02d' "${idx}")"
  done
}

case "${MODE}" in
  build)
    run_build
    ;;
  gt)
    run_gt
    ;;
  full)
    run_build
    run_gt
    ;;
  *)
    usage
    exit 1
    ;;
esac
