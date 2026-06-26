#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
echo "Switched to directory: $PROJECT_ROOT"

# Server
SERVER_BASE_URL="http://10.14.114.132:23264"

# Instruction
INSTRUCTION=""
INSTRUCTION_FILE=""

# Robot control
# dry-run: only print planned robot commands; sdk2: control Go2 through Unitree SDK2; ros2: control through Unitree ROS2 bridge.
CONTROL_BACKEND="dry-run"
# Safety latch for real robot motion. Keep empty for dry-run; set to "yes" only when CONTROL_BACKEND is sdk2 or ros2.
REAL_ROBOT_ACK=""
# Network interface used by Unitree SDK2 to communicate with the robot, for example eth0 or wlan0. Ignored unless CONTROL_BACKEND="sdk2".
NETWORK_INTERFACE="eth0"

# Camera
CAMERA="/dev/video0"
FRAME_WIDTH="2880"
FRAME_HEIGHT="1440"
CAMERA_FPS="30"
JPEG_QUALITY="90"
CAPTURE_FLUSH_FRAMES="2"

# Network upload. Use raw to disable robot-side resize.
UPLOAD_IMAGE_MODE="resize"
UPLOAD_WIDTH="1280"
UPLOAD_HEIGHT="640"

# Safety / replanning
MAX_REPLANS="60"
ACTIONS_PER_REPLAN="4"

# Motion primitives. Start conservatively on the real robot.
FORWARD_DISTANCE="0.25"
FORWARD_SPEED="0.20"
TURN_DEGREES="15.0"
YAW_SPEED="0.35"

if [[ -z "$INSTRUCTION" && -z "$INSTRUCTION_FILE" ]]; then
    echo "Please edit INSTRUCTION or INSTRUCTION_FILE in $0 before running." >&2
    exit 1
fi

if [[ "$CONTROL_BACKEND" != "dry-run" && "$REAL_ROBOT_ACK" != "yes" ]]; then
    echo "Set REAL_ROBOT_ACK=\"yes\" before using CONTROL_BACKEND=$CONTROL_BACKEND." >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

echo "SERVER_BASE_URL=$SERVER_BASE_URL"
echo "CONTROL_BACKEND=$CONTROL_BACKEND"
echo "REAL_ROBOT_ACK=$REAL_ROBOT_ACK"
echo "NETWORK_INTERFACE=$NETWORK_INTERFACE"
echo "CAMERA=$CAMERA"
echo "JPEG_QUALITY=$JPEG_QUALITY"
echo "CAPTURE_FLUSH_FRAMES=$CAPTURE_FLUSH_FRAMES"
echo "UPLOAD_IMAGE_MODE=$UPLOAD_IMAGE_MODE"
echo "UPLOAD_WIDTH=$UPLOAD_WIDTH"
echo "UPLOAD_HEIGHT=$UPLOAD_HEIGHT"
echo "MAX_REPLANS=$MAX_REPLANS"
echo "ACTIONS_PER_REPLAN=$ACTIONS_PER_REPLAN"
echo "FORWARD_DISTANCE=$FORWARD_DISTANCE"
echo "FORWARD_SPEED=$FORWARD_SPEED"
echo "TURN_DEGREES=$TURN_DEGREES"
echo "YAW_SPEED=$YAW_SPEED"

CMD=(
    python3 -m realworld.go2_client
    --server-base-url "$SERVER_BASE_URL"
    --control-backend "$CONTROL_BACKEND"
    --real-robot-ack "$REAL_ROBOT_ACK"
    --camera "$CAMERA"
    --jpeg-quality "$JPEG_QUALITY"
    --capture-flush-frames "$CAPTURE_FLUSH_FRAMES"
    --upload-image-mode "$UPLOAD_IMAGE_MODE"
    --upload-width "$UPLOAD_WIDTH"
    --upload-height "$UPLOAD_HEIGHT"
    --max-replans "$MAX_REPLANS"
    --actions-per-replan "$ACTIONS_PER_REPLAN"
    --forward-distance "$FORWARD_DISTANCE"
    --forward-speed "$FORWARD_SPEED"
    --turn-degrees "$TURN_DEGREES"
    --yaw-speed "$YAW_SPEED"
)

if [[ "$CONTROL_BACKEND" == "sdk2" ]]; then
    CMD+=(--network-interface "$NETWORK_INTERFACE")
fi
if [[ -n "$INSTRUCTION" ]]; then
    CMD+=(--instruction "$INSTRUCTION")
fi
if [[ -n "$INSTRUCTION_FILE" ]]; then
    CMD+=(--instruction-file "$INSTRUCTION_FILE")
fi
if [[ -n "$FRAME_WIDTH" ]]; then
    CMD+=(--frame-width "$FRAME_WIDTH")
fi
if [[ -n "$FRAME_HEIGHT" ]]; then
    CMD+=(--frame-height "$FRAME_HEIGHT")
fi
if [[ -n "$CAMERA_FPS" ]]; then
    CMD+=(--camera-fps "$CAMERA_FPS")
fi

"${CMD[@]}"
