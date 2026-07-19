#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

# 直接修改这里的值，不从环境变量覆盖。
SAVE_ROOT="/workspace/data1/dataset/PanoVLN/generated/PanoVLN-HM3D"  # 最终数据集输出目录。
SCENE_ROOT="/workspace/data1/dataset/general_VLN_data/HM3D"  # 含 train/val 的 HM3D 根目录。
SCENE_SPLIT="train"  # 使用 HM3D 的哪个 split。
TRAJECTORY_PROFILE_CONFIG="${REPO_ROOT}/data_create/config/trajectory_profiles.json"  # 仓库内匿名长度直方图。
NUM_TRAJECTORIES=110000  # 要采集的物理轨迹总数；每条会生成两种 instruction。
R2R_RATIO=0.40  # 全数据集中 R2R-like 短轨迹的比例。
RXR_RATIO=0.60  # 全数据集中 RxR-like 长/复杂轨迹的比例；两比例之和须为 1。
SEED=42  # 控制场景顺序和轨迹采样，保持结果可复现。
GOAL_RADIUS=0.3  # GT action 到终点 0.3 m 内即可 STOP。
GPU_DEVICE_ID=0  # collect / GT / render 使用的逻辑 GPU 编号。

MAX_ANCHOR_TURN_DEGREES=90  # synthetic detour 中间锚点允许的最大转角。
VISUAL_CHECK_WIDTH=512  # 采轨迹时低成本视觉预检的全景宽度。
VISUAL_CHECK_HEIGHT=256  # 采轨迹时低成本视觉预检的全景高度。
SENSOR_HEIGHT=1.25  # Habitat RGB 相机离地高度，单位为米。
MAX_VISUAL_CHECKPOINTS=24  # 每条候选轨迹最多抽查多少个视觉关键点。
MAX_BLACK_RATIO=0.10  # 单帧允许的黑色/无效全景区域最大比例。

PANORAMA_WIDTH=2048  # 正式发布图片的全景宽度。
PANORAMA_HEIGHT=1024  # 正式发布图片的全景高度。
PANORAMA_JPEG_QUALITY=92  # 正式全景图片的 JPEG 质量。
TILE_WIDTH=384  # 从全景投影、交给 VLM 的透视图宽度。
TILE_HEIGHT=288  # 从全景投影、交给 VLM 的透视图高度。
SHEET_JPEG_QUALITY=90  # 发送给 VLM 的多视角拼图 JPEG 质量。
ROUTE_EVIDENCE_MODE="auto"  # 长路线自动拆成局部段理解，再合并成完整 instruction。
SEGMENTED_MIN_ACTIONS=80  # 达到该 action 数时启用分段视觉证据。
SEGMENT_MAX_WAYPOINTS=0  # 0 表示分段复用当前 style 的 max-waypoints。
SEGMENT_ROWS=5  # 每个局部分段 sheet 最多放多少个 waypoint 行。
SEGMENT_OVERLAP=1  # 相邻分段共享的 waypoint 行数，用于保持上下文连续。
SEGMENT_FACT_MAX_TOKENS=650  # 单个局部分段 route facts 的最大输出长度。
CANDIDATE_COUNT=2  # 每条路线生成多个候选后由独立视觉 judge 选择，降低单次采样回归。
CANDIDATE_TEMPERATURE=0.4  # 非首个候选的采样温度，提供不同自然表述供 judge 比较。

BASE_URL="http://127.0.0.1:10420/v1"  # OpenAI-compatible Qwen API pool 地址。
MODEL="Qwen3.6-27B"  # 服务中实际加载的模型名称。
API_KEY="test"  # 本地兼容接口的占位 key；按服务要求修改。
NUM_WORKERS=40  # instruction 并发轨迹数；不要超过 Qwen API pool 的 MAX_CONCURRENCY。

# 避免系统 HTTP 代理截获本机 Qwen API 请求。
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}127.0.0.1,localhost"
export no_proxy="${no_proxy:+${no_proxy},}127.0.0.1,localhost"

# 以下路径由 SAVE_ROOT 自动派生，一般不需要修改。
WORK_ROOT="${SAVE_ROOT}/.work"  # 中间文件根目录；成功 export 后整体删除。
TRAJECTORY_DATASET="${WORK_ROOT}/trajectories.json.gz"  # 中间轨迹和 reference path。
TRAJECTORY_GT="${WORK_ROOT}/trajectories_gt.json.gz"  # 中间 Habitat expert actions。
GT_JOURNAL="${WORK_ROOT}/trajectories_gt.journal.jsonl"  # GT 断点续跑日志；完成后自动删除。
IMAGE_ROOT="${SAVE_ROOT}/images"  # 正式产物：逐 action 的全景图片。
AGENT_INPUT_JSONL="${WORK_ROOT}/agent_input.jsonl"  # 中间 agent 输入，不含原 instruction。
INSTRUCTION_ROOT="${WORK_ROOT}/instructions"  # 中间 instruction 结果和 agent 进度。
FINAL_JSON="${SAVE_ROOT}/train.json"  # 正式产物：未压缩 VLN-CE dataset。
FINAL_JSON_GZ="${SAVE_ROOT}/train.json.gz"  # 正式产物：同内容压缩 dataset。
FINAL_GT_GZ="${SAVE_ROOT}/train_gt.json.gz"  # 正式产物：图片对应的 expert actions。

mkdir -p "${SAVE_ROOT}" "${WORK_ROOT}" "${INSTRUCTION_ROOT}"

STAGE="full"  # 执行阶段：collect / gt / images / instruction / export / full。

collect() {
  python -m data_create.trajectory.collect_hm3d collect \
    --scene-root "${SCENE_ROOT}" \
    --split "${SCENE_SPLIT}" \
    --trajectory-profile-config "${TRAJECTORY_PROFILE_CONFIG}" \
    --num-trajectories "${NUM_TRAJECTORIES}" \
    --r2r-ratio "${R2R_RATIO}" \
    --rxr-ratio "${RXR_RATIO}" \
    --max-anchor-turn-degrees "${MAX_ANCHOR_TURN_DEGREES}" \
    --visual-check-width "${VISUAL_CHECK_WIDTH}" \
    --visual-check-height "${VISUAL_CHECK_HEIGHT}" \
    --visual-check-sensor-height "${SENSOR_HEIGHT}" \
    --max-visual-checkpoints "${MAX_VISUAL_CHECKPOINTS}" \
    --max-black-ratio "${MAX_BLACK_RATIO}" \
    --goal-radius "${GOAL_RADIUS}" \
    --gpu-device-id "${GPU_DEVICE_ID}" \
    --seed "${SEED}" \
    --output "${TRAJECTORY_DATASET}"
}

generate_gt() {
  python -m data_create.trajectory.generate_gt \
    --dataset "${TRAJECTORY_DATASET}" \
    --output "${TRAJECTORY_GT}" \
    --journal "${GT_JOURNAL}" \
    --scene-root "${SCENE_ROOT}" \
    --config-path "${REPO_ROOT}/data_create/config/hm3d_vln.yaml" \
    --repo-root "${REPO_ROOT}" --goal-radius "${GOAL_RADIUS}" \
    --gpu-device-id "${GPU_DEVICE_ID}" \
    --sort-by-scene --minimal-observations --resume

  python -m data_create.trajectory.collect_hm3d validate-gt \
    --dataset "${TRAJECTORY_DATASET}" \
    --gt "${TRAJECTORY_GT}" \
    --goal-radius "${GOAL_RADIUS}" \
    --generation-jsonl "${AGENT_INPUT_JSONL}"
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
    --gpu-device-id "${GPU_DEVICE_ID}" \
    --jpeg-quality "${PANORAMA_JPEG_QUALITY}"
}

generate_style() {
  local style="$1"
  python -m data_create.instruction.pipeline \
    --input-jsonl "${AGENT_INPUT_JSONL}" \
    --image-root "${IMAGE_ROOT}" \
    --output-jsonl "${INSTRUCTION_ROOT}/${style}.jsonl" \
    --work-dir "${INSTRUCTION_ROOT}/${style}_progress" \
    --mode generate --instruction-profile "${style}" \
    --provider qwen --base-url "${BASE_URL}" --model "${MODEL}" --api-key "${API_KEY}" \
    --num-workers "${NUM_WORKERS}" --max-waypoints 18 \
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

export_dataset() {
  python -m data_create.export_vlnce \
    --dataset "${TRAJECTORY_DATASET}" \
    --source-gt "${TRAJECTORY_GT}" \
    --image-root "${IMAGE_ROOT}" \
    --variant "concise=${INSTRUCTION_ROOT}/concise.jsonl" \
    --variant "dense=${INSTRUCTION_ROOT}/dense.jsonl" \
    --output "${FINAL_JSON}" \
    --gzip-output "${FINAL_JSON_GZ}" \
    --gt-output "${FINAL_GT_GZ}" \
    --goal-radius "${GOAL_RADIUS}"
  # GT 正式发布为压缩文件；清掉旧版本的未压缩副本和失败渲染目录。
  rm -f "${SAVE_ROOT}/train_gt.json"
  rm -rf "${IMAGE_ROOT}.failed"
  rm -rf "${WORK_ROOT}"
}

case "${STAGE}" in
  collect) collect ;;
  gt) generate_gt ;;
  images) render ;;
  instruction)
    generate_style concise
    generate_style dense
    ;;
  export) export_dataset ;;
  full)
    collect
    generate_gt
    render
    generate_style concise
    generate_style dense
    export_dataset
    ;;
  *)
    echo "Unknown STAGE=${STAGE}; edit STAGE at the top of this script." >&2
    exit 2
    ;;
esac
