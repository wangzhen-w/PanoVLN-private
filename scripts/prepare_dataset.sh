#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
cd "$PROJECT_ROOT"
echo "Switched to project root: $PROJECT_ROOT"

export PYTHONPATH="./:${PYTHONPATH:-}"
export MAGNUM_LOG="quiet"
export GLOG_minloglevel="3"
export HABITAT_LAB_LOG="50"
export PYTHONWARNINGS="ignore"

PYTHON_BIN="/opt/conda/bin/python"  # 当前训练环境；需迁移时改这一处。
INPUT_ROOT="/workspace/data2/dataset/PanoVLN"
OUTPUT_PATH="/workspace/data2/dataset/ablation/instruction_compare/panovln_event085_bg005_tail100_30k_ep.jsonl"
# 新数据集使用 panovln；旧 ScaleVLN 指令消融分别使用 scalevln/scalevln_rewrite。
DATASET_NAMES=(panovln)
MAX_EPISODES_PER_SUBSET="30000"   # 随机选取的源episode数量；留空表示使用全部episode。
SUBSET_SEED="42"                # 固定episode随机排列；不同规模取同一排列的前缀。
PAD_STOP_TO_HORIZON="true"
SEED="42"                       # EBS action chunk采样seed，与episode抽样相互独立。

EVENT_KEEP_PROB="0.85"       # 保留含转向事件的 body action chunk 的概率。
BACKGROUND_KEEP_PROB="0.05"  # 保留纯前进 body action chunk 的概率。
BODY_KEEP_ADVANCE="4"        # body chunk 被保留后向前跳过的起始步数。
TAIL_DENSE_KEEP_PROB="1.00"   # 完整保留轨迹末尾4个dense起点。

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "INPUT_ROOT: ${INPUT_ROOT}"
echo "OUTPUT_PATH: ${OUTPUT_PATH}"
echo "DATASET_NAMES: ${DATASET_NAMES[*]}"
echo "MAX_EPISODES_PER_SUBSET: ${MAX_EPISODES_PER_SUBSET:-all}"
echo "SUBSET_SEED: ${SUBSET_SEED}"
echo "PAD_STOP_TO_HORIZON: ${PAD_STOP_TO_HORIZON}"
echo "SEED: ${SEED}"
echo "EVENT_KEEP_PROB: ${EVENT_KEEP_PROB}"
echo "BACKGROUND_KEEP_PROB: ${BACKGROUND_KEEP_PROB}"
echo "BODY_KEEP_ADVANCE: ${BODY_KEEP_ADVANCE}"
echo "TAIL_DENSE_KEEP_PROB: ${TAIL_DENSE_KEEP_PROB}"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "python not found in PATH" >&2
    exit 1
fi

PREPARE_CMD=(
    "${PYTHON_BIN}" src/data/prepare_training_data.py
    --input_root "${INPUT_ROOT}"
    --dataset_name "${DATASET_NAMES[@]}"
    --output_path "${OUTPUT_PATH}"
    --subset_seed "${SUBSET_SEED}"
    --seed "${SEED}"
    --event_keep_prob "${EVENT_KEEP_PROB}"
    --background_keep_prob "${BACKGROUND_KEEP_PROB}"
    --body_keep_advance "${BODY_KEEP_ADVANCE}"
    --tail_dense_keep_prob "${TAIL_DENSE_KEEP_PROB}"
)

if [[ -n "${MAX_EPISODES_PER_SUBSET}" ]]; then
    PREPARE_CMD+=(--max_episodes_per_subset "${MAX_EPISODES_PER_SUBSET}")
fi

if [[ "${PAD_STOP_TO_HORIZON}" == "true" ]]; then
    PREPARE_CMD+=(--pad_stop_to_horizon)
fi

printf 'Running:'
printf ' %q' "${PREPARE_CMD[@]}"
printf '\n'
"${PREPARE_CMD[@]}"
