#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# 直接修改这里的值，不从环境变量覆盖。
SAVE_ROOT="/workspace/data1/dataset/general_VLN_data/PanoVLN"  # 最终数据集输出目录。
SCENE_ROOT="/workspace/data1/dataset/general_VLN_data/HM3D"  # 含 train/val 的 HM3D 根目录。
SCENE_SPLIT="train"  # 使用 HM3D 的哪个 split。
TRAJECTORY_PROFILE_CONFIG="${REPO_ROOT}/data_create/config/trajectory_profiles.json"  # 仓库内匿名长度直方图。
NUM_TRAJECTORIES=210000  # 期望采集量；场景容量不足时发布实际通过质量门的最大规模。
COLLECT_RECOVERY_ROUNDS=1  # 首轮结束后最多补采几轮；不会为凑数量放宽质量门。
R2R_RATIO=0.35  # 全数据集中 R2R-like 短轨迹的比例。
RXR_RATIO=0.65  # 全数据集中 RxR-like 长/复杂轨迹的比例；两比例之和须为 1。
SEED=42  # 控制场景顺序和轨迹采样，保持结果可复现。
GOAL_RADIUS=0.3  # GT action 到终点 0.3 m 内即可 STOP。
GPU_DEVICE_IDS="0,1,2,3,4,5,6,7"  # 采轨迹、GT 和渲染共同使用的 Habitat GPU 列表。
COLLECT_PROCESSES_PER_GPU=2  # 每张 GPU 同时运行 2 个轨迹采样进程，共 16 个。
GT_PROCESSES_PER_GPU=2  # 每张 GPU 同时运行 2 个 expert GT 进程，共 16 个。
RENDER_PROCESSES_PER_GPU=5  # 每张 GPU 同时运行 5 个正式全景渲染进程，共 40 个。

MAX_ANCHOR_TURN_DEGREES=90  # synthetic detour 中间锚点允许的最大转角。
VISUAL_CHECK_WIDTH=512  # 采轨迹时低成本视觉预检的全景宽度。
VISUAL_CHECK_HEIGHT=256  # 采轨迹时低成本视觉预检的全景高度。
SENSOR_HEIGHT=1.25  # Habitat RGB 相机离地高度，单位为米。
MAX_VISUAL_CHECKPOINTS=24  # 每条候选轨迹最多抽查多少个视觉关键点。
MAX_BLACK_RATIO=0.10  # 单帧允许的黑色/无效全景区域最大比例。

PANORAMA_WIDTH=1600  # 正式发布图片的全景宽度。
PANORAMA_HEIGHT=800  # 正式发布图片的全景高度。
PANORAMA_JPEG_QUALITY=92  # 正式全景图片的 JPEG 质量。
TILE_WIDTH=384  # 从全景投影、交给 VLM 的透视图宽度。
TILE_HEIGHT=288  # 从全景投影、交给 VLM 的透视图高度。
SHEET_JPEG_QUALITY=90  # 发送给 VLM 的多视角拼图 JPEG 质量。
ROUTE_EVIDENCE_MODE="auto"  # 长路线自动拆成局部段理解，再合并成完整 instruction。
SEGMENTED_MIN_ACTIONS=80  # 达到该长度必分段；更短但平移多、转向复杂的路线也会自动分段。
SEGMENT_MAX_WAYPOINTS=0  # 0 表示分段复用当前 style 的 max-waypoints。
SEGMENT_ROWS=5  # 每个局部分段 sheet 最多放多少个 waypoint 行。
SEGMENT_OVERLAP=1  # 相邻分段共享的 waypoint 行数，用于保持上下文连续。
SEGMENT_FACT_MAX_TOKENS=650  # 单个局部分段 route facts 的最大输出长度。
CANDIDATE_COUNT=2  # 每条路线保留 grounded draft 与独立 language realization，再做视觉验真。
CANDIDATE_TEMPERATURE=0.4  # language-realization 温度；grounding 与 review 仍保持 0 以稳定事实。

BASE_URL="http://127.0.0.1:10430/v1"  # 多个 Qwen API 的聚合入口；运行前先启动 /workspace/code/occupy/vllm_api_aggregator.sh。
MODEL="Qwen3.6-35B-A3B"  # 服务中实际加载的模型名称。
API_KEY="test"  # 本地兼容接口的占位 key；按服务要求修改。
NUM_WORKERS=144  # instruction pipeline 的总并发；聚合器会按各 API 的实际容量自动分流。
EVIDENCE_WORKERS=16  # 独立 CPU 进程数，用于解码全景图、透视投影和制作 evidence sheet。

# 避免系统 HTTP 代理截获本机 Qwen API 请求。
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

# 以下路径由 SAVE_ROOT 自动派生，一般不需要修改。
WORK_ROOT="${SAVE_ROOT}/.work"  # 中间文件根目录；成功 export 后整体删除。
TRAJECTORY_DATASET="${WORK_ROOT}/trajectories.json.gz"  # 中间轨迹和 reference path。
TRAJECTORY_GT="${WORK_ROOT}/trajectories_gt.json.gz"  # 中间 Habitat expert actions。
GT_JOURNAL="${WORK_ROOT}/trajectories_gt.journal.jsonl"  # 兼容单进程 GT 进度；并行状态也保存在 .work。
IMAGE_ROOT="${SAVE_ROOT}/images"  # 正式产物：逐 action 的全景图片。
AGENT_INPUT_JSONL="${WORK_ROOT}/agent_input.jsonl"  # 中间 agent 输入，不含原 instruction。
INSTRUCTION_ROOT="${WORK_ROOT}/instructions"  # 中间 instruction 结果和 agent 进度。
INSTRUCTION_JSONL="${INSTRUCTION_ROOT}/dense.jsonl"  # 中间 Dense instruction 结果。
INSTRUCTION_WORK_DIR="${INSTRUCTION_ROOT}/dense_progress"  # instruction 断点续跑目录。
INSTRUCTION_RECORDS_JSONL_GZ="${SAVE_ROOT}/instruction_records.jsonl.gz"  # 正式产物：每条轨迹最新一次完整 agent 制作记录。
INSTRUCTION_SUMMARY_JSON="${SAVE_ROOT}/instruction_summary.json"  # 正式产物：instruction 生成规模和完成状态摘要。
FINAL_JSON="${SAVE_ROOT}/train.json"  # 正式产物：未压缩 VLN-CE dataset。
FINAL_JSON_GZ="${SAVE_ROOT}/train.json.gz"  # 正式产物：同内容压缩 dataset。
FINAL_GT_GZ="${SAVE_ROOT}/train_gt.json.gz"  # 正式产物：图片对应的 expert actions。

STAGE="full"  # 执行阶段：collect / gt / images / instruction / export / full。

prepare_workspace() {
  mkdir -p "${SAVE_ROOT}" "${WORK_ROOT}" "${INSTRUCTION_ROOT}"
}

final_dataset_is_published() {
  [[ -f "${FINAL_JSON}" && -f "${FINAL_JSON_GZ}" && -f "${FINAL_GT_GZ}" ]]
}

collect() {
  python -m data_create.trajectory.collect_hm3d collect \
    --scene-root "${SCENE_ROOT}" \
    --split "${SCENE_SPLIT}" \
    --trajectory-profile-config "${TRAJECTORY_PROFILE_CONFIG}" \
    --num-trajectories "${NUM_TRAJECTORIES}" \
    --max-recovery-rounds "${COLLECT_RECOVERY_ROUNDS}" \
    --r2r-ratio "${R2R_RATIO}" \
    --rxr-ratio "${RXR_RATIO}" \
    --max-anchor-turn-degrees "${MAX_ANCHOR_TURN_DEGREES}" \
    --visual-check-width "${VISUAL_CHECK_WIDTH}" \
    --visual-check-height "${VISUAL_CHECK_HEIGHT}" \
    --visual-check-sensor-height "${SENSOR_HEIGHT}" \
    --max-visual-checkpoints "${MAX_VISUAL_CHECKPOINTS}" \
    --max-black-ratio "${MAX_BLACK_RATIO}" \
    --goal-radius "${GOAL_RADIUS}" \
    --gpu-device-ids "${GPU_DEVICE_IDS}" \
    --processes-per-gpu "${COLLECT_PROCESSES_PER_GPU}" \
    --seed "${SEED}" \
    --output "${TRAJECTORY_DATASET}" \
    --resume
}

collect_if_needed() {
  if [[ -f "${TRAJECTORY_DATASET}" ]]; then
    echo "[full] trajectory dataset already published; skipping ${TRAJECTORY_DATASET}"
    return
  fi
  collect
}

generate_gt() {
  python -m data_create.trajectory.generate_gt \
    --dataset "${TRAJECTORY_DATASET}" \
    --output "${TRAJECTORY_GT}" \
    --journal "${GT_JOURNAL}" \
    --scene-root "${SCENE_ROOT}" \
    --config-path "${REPO_ROOT}/data_create/config/hm3d_vln.yaml" \
    --repo-root "${REPO_ROOT}" --goal-radius "${GOAL_RADIUS}" \
    --gpu-device-ids "${GPU_DEVICE_IDS}" \
    --processes-per-gpu "${GT_PROCESSES_PER_GPU}" \
    --sort-by-scene --minimal-observations --resume

  python -m data_create.trajectory.collect_hm3d validate-gt \
    --dataset "${TRAJECTORY_DATASET}" \
    --gt "${TRAJECTORY_GT}" \
    --goal-radius "${GOAL_RADIUS}" \
    --generation-jsonl "${AGENT_INPUT_JSONL}" \
    --drop-invalid
}

generate_gt_if_needed() {
  if [[ -f "${TRAJECTORY_GT}" && -f "${AGENT_INPUT_JSONL}" ]]; then
    echo "[full] GT and agent input already published; skipping ${TRAJECTORY_GT}"
    return
  fi
  generate_gt
}

render() {
  python -m data_create.trajectory.collect_hm3d render-panoramas \
    --dataset "${TRAJECTORY_DATASET}" \
    --gt "${TRAJECTORY_GT}" \
    --scene-root "${SCENE_ROOT}" \
    --output-root "${IMAGE_ROOT}" \
    --width "${PANORAMA_WIDTH}" \
    --height "${PANORAMA_HEIGHT}" \
    --sensor-height "${SENSOR_HEIGHT}" \
    --max-black-ratio "${MAX_BLACK_RATIO}" \
    --gpu-device-ids "${GPU_DEVICE_IDS}" \
    --processes-per-gpu "${RENDER_PROCESSES_PER_GPU}" \
    --jpeg-quality "${PANORAMA_JPEG_QUALITY}" \
    --generation-jsonl "${AGENT_INPUT_JSONL}" \
    --drop-invalid \
    --resume
}

render_if_needed() {
  # images/ 只会在整个渲染阶段成功后原子发布；目录存在即可视为本阶段完成。
  if [[ -d "${IMAGE_ROOT}" ]]; then
    echo "[full] panorama rendering already complete; skipping ${IMAGE_ROOT}"
    return
  fi
  render
}

generate_instruction() {
  python -m data_create.instruction.pipeline \
    --input-jsonl "${AGENT_INPUT_JSONL}" \
    --image-root "${IMAGE_ROOT}" \
    --output-jsonl "${INSTRUCTION_JSONL}" \
    --work-dir "${INSTRUCTION_WORK_DIR}" \
    --records-output-jsonl "${INSTRUCTION_RECORDS_JSONL_GZ}" \
    --summary-output-json "${INSTRUCTION_SUMMARY_JSON}" \
    --mode generate --instruction-profile dense \
    --provider qwen --base-url "${BASE_URL}" --model "${MODEL}" --api-key "${API_KEY}" \
    --num-workers "${NUM_WORKERS}" --evidence-workers "${EVIDENCE_WORKERS}" --max-waypoints 18 \
    --route-evidence-mode "${ROUTE_EVIDENCE_MODE}" \
    --segmented-min-actions "${SEGMENTED_MIN_ACTIONS}" \
    --segment-max-waypoints "${SEGMENT_MAX_WAYPOINTS}" \
    --segment-rows "${SEGMENT_ROWS}" \
    --segment-overlap "${SEGMENT_OVERLAP}" \
    --start-window-frames 6 --endpoint-window-frames 6 \
    --tile-width "${TILE_WIDTH}" --tile-height "${TILE_HEIGHT}" \
    --jpeg-quality "${SHEET_JPEG_QUALITY}" \
    --temperature 0.4 --max-tokens 420 \
    --planner-temperature 0.0 --review-temperature 0.0 \
    --candidate-count "${CANDIDATE_COUNT}" --candidate-temperature "${CANDIDATE_TEMPERATURE}" \
    --fact-max-tokens 320 --segment-fact-max-tokens "${SEGMENT_FACT_MAX_TOKENS}" \
    --planner-max-tokens 1600 --review-max-tokens 1200 \
    --request-timeout 300 --retries 3 --stage-retries 2 \
    --blind-grounding-audit true --drop-failed true --allow-incomplete false
}

generate_instruction_if_needed() {
  # 三个文件均原子发布；缺少审计产物时会从已有 journal 快速重建，不重复调用 Qwen。
  if [[ -f "${INSTRUCTION_JSONL}" && -f "${INSTRUCTION_RECORDS_JSONL_GZ}" && -f "${INSTRUCTION_SUMMARY_JSON}" ]]; then
    echo "[full] instruction outputs already published; skipping generation"
    return
  fi
  generate_instruction
}

export_dataset() {
  if [[ ! -f "${INSTRUCTION_JSONL}" || ! -f "${INSTRUCTION_RECORDS_JSONL_GZ}" || ! -f "${INSTRUCTION_SUMMARY_JSON}" ]]; then
    echo "Instruction output or persistent audit artifacts are missing; run STAGE=instruction before export." >&2
    return 1
  fi
  python -m data_create.export_vlnce \
    --dataset "${TRAJECTORY_DATASET}" \
    --source-gt "${TRAJECTORY_GT}" \
    --image-root "${IMAGE_ROOT}" \
    --variant "dense=${INSTRUCTION_JSONL}" \
    --output "${FINAL_JSON}" \
    --gzip-output "${FINAL_JSON_GZ}" \
    --gt-output "${FINAL_GT_GZ}" \
    --goal-radius "${GOAL_RADIUS}" \
    --allow-subset \
    --prune-unselected-images
  # GT 正式发布为压缩文件；清掉旧版本的未压缩副本和失败渲染目录。
  rm -f "${SAVE_ROOT}/train_gt.json"
  rm -rf "${IMAGE_ROOT}.failed"
  rm -rf "${WORK_ROOT}"
}

export_if_needed() {
  if final_dataset_is_published; then
    echo "[full] final dataset already published; nothing to resume under ${SAVE_ROOT}"
    return
  fi
  export_dataset
}

case "${STAGE}" in
  collect)
    prepare_workspace
    collect
    ;;
  gt)
    prepare_workspace
    generate_gt
    ;;
  images)
    prepare_workspace
    render
    ;;
  instruction)
    prepare_workspace
    generate_instruction
    ;;
  export)
    prepare_workspace
    export_dataset
    ;;
  full)
    if final_dataset_is_published; then
      echo "[full] final dataset already published; nothing to resume under ${SAVE_ROOT}"
      exit 0
    fi
    prepare_workspace
    collect_if_needed
    generate_gt_if_needed
    render_if_needed
    generate_instruction_if_needed
    export_if_needed
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; edit STAGE at the top of this script." >&2
    exit 2
    ;;
esac
