#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"
echo "Switched to directory: $PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-/usr/bin/python3}"
ROS2_UNITREE_API_WS="${ROS2_UNITREE_API_WS:-$SCRIPT_DIR/ros2_unitree_api_ws}"

# Server
SERVER_BASE_URL="http://10.14.114.132:30110"

# Instruction
INSTRUCTION="Walk straight to the glass wall, then turn right and stop in front of the large TV set."
INSTRUCTION_FILE=""

# Robot control
# dry-run: only print planned robot commands; ros2: control Go2 through the Unitree ROS2 bridge.
CONTROL_BACKEND="ros2"
# Safety latch for real robot motion. Keep empty for dry-run; set to "yes" only when CONTROL_BACKEND is ros2.
REAL_ROBOT_ACK="yes"

# Camera
CAMERA="/dev/video0"
FRAME_WIDTH="2880"
FRAME_HEIGHT="1440"
CAMERA_FPS="30"
JPEG_QUALITY="90"
CAPTURE_FLUSH_FRAMES="2"
# Leave SAVE_CONTENTS as "none" to disable saving. Supported: images,video,json,all,none.
SAVE_OUTPUT_DIR="/workspace/tmp/vln_navigation_outputs"
SAVE_CONTENTS="all"
VIDEO_FPS="10"
VIDEO_WIDTH="1280"
VIDEO_HEIGHT="640"
VIDEO_WRITER="ffmpeg"
VIDEO_CODEC="libx264"

# Network upload. Use raw to disable robot-side resize.
UPLOAD_IMAGE_MODE="resize"
UPLOAD_WIDTH="1280"
UPLOAD_HEIGHT="640"

# Safety / replanning
MAX_REPLANS="100"
# 0 follows action_sequence_length reported by the model server.
ACTIONS_PER_REPLAN="0"
# Disable speculative replanning so every horizon is evaluated from fresh observations.
PREDICT_PREFETCH_AFTER_ACTIONS="0"

# Motion primitives. Start conservatively on the real robot.
FORWARD_DISTANCE="0.25"
FORWARD_SPEED="0.35"
TURN_DEGREES="15.0"
YAW_SPEED="0.80"
COMMAND_PERIOD="0.05"
SETTLE_TIME="0.20"  # units: seconds. Time to wait after each motion primitive before capturing the next frame.
POST_CAPTURE_SETTLE_TIME="0.2" # units: seconds. Time to wait after capturing a frame before executing the next motion primitive.

# ROS2 closed-loop motion control using /sportmodestate, based on the same idea as StreamVLN.
ODOM_CONTROL="yes"
ODOM_TIMEOUT="1.0"
FORWARD_TOLERANCE="0.04"
TURN_TOLERANCE_DEGREES="3.0"
FORWARD_KP="3.0"
FORWARD_KD="0.0"
TURN_KP="3.0"
TURN_KD="0.0"
MIN_FORWARD_SPEED="0.15"
MIN_YAW_SPEED="0.35"

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

if [[ "$CONTROL_BACKEND" == "ros2" ]]; then
    source /opt/ros/foxy/setup.bash
    if [[ -f "$ROS2_UNITREE_API_WS/install/setup.bash" ]]; then
        source "$ROS2_UNITREE_API_WS/install/setup.bash"
    else
        echo "ROS2 Unitree API workspace is not built: $ROS2_UNITREE_API_WS" >&2
        echo "Run: cd $ROS2_UNITREE_API_WS && source /opt/ros/foxy/setup.bash && colcon build --base-paths src --packages-select unitree_api unitree_go" >&2
        exit 1
    fi
fi

echo "SERVER_BASE_URL=$SERVER_BASE_URL"
echo "PYTHON_BIN=$PYTHON_BIN"
echo "ROS2_UNITREE_API_WS=$ROS2_UNITREE_API_WS"
echo "CONTROL_BACKEND=$CONTROL_BACKEND"
echo "REAL_ROBOT_ACK=$REAL_ROBOT_ACK"
echo "CAMERA=$CAMERA"
echo "JPEG_QUALITY=$JPEG_QUALITY"
echo "CAPTURE_FLUSH_FRAMES=$CAPTURE_FLUSH_FRAMES"
echo "SAVE_OUTPUT_DIR=$SAVE_OUTPUT_DIR"
echo "SAVE_CONTENTS=$SAVE_CONTENTS"
echo "VIDEO_FPS=$VIDEO_FPS"
echo "VIDEO_WIDTH=$VIDEO_WIDTH"
echo "VIDEO_HEIGHT=$VIDEO_HEIGHT"
echo "VIDEO_WRITER=$VIDEO_WRITER"
echo "VIDEO_CODEC=$VIDEO_CODEC"
echo "UPLOAD_IMAGE_MODE=$UPLOAD_IMAGE_MODE"
echo "UPLOAD_WIDTH=$UPLOAD_WIDTH"
echo "UPLOAD_HEIGHT=$UPLOAD_HEIGHT"
echo "MAX_REPLANS=$MAX_REPLANS"
echo "ACTIONS_PER_REPLAN=$ACTIONS_PER_REPLAN"
echo "PREDICT_PREFETCH_AFTER_ACTIONS=$PREDICT_PREFETCH_AFTER_ACTIONS"
echo "FORWARD_DISTANCE=$FORWARD_DISTANCE"
echo "FORWARD_SPEED=$FORWARD_SPEED"
echo "TURN_DEGREES=$TURN_DEGREES"
echo "YAW_SPEED=$YAW_SPEED"
echo "COMMAND_PERIOD=$COMMAND_PERIOD"
echo "SETTLE_TIME=$SETTLE_TIME"
echo "POST_CAPTURE_SETTLE_TIME=$POST_CAPTURE_SETTLE_TIME"
echo "ODOM_CONTROL=$ODOM_CONTROL"
echo "ODOM_TIMEOUT=$ODOM_TIMEOUT"
echo "FORWARD_TOLERANCE=$FORWARD_TOLERANCE"
echo "TURN_TOLERANCE_DEGREES=$TURN_TOLERANCE_DEGREES"
echo "FORWARD_KP=$FORWARD_KP"
echo "FORWARD_KD=$FORWARD_KD"
echo "TURN_KP=$TURN_KP"
echo "TURN_KD=$TURN_KD"
echo "MIN_FORWARD_SPEED=$MIN_FORWARD_SPEED"
echo "MIN_YAW_SPEED=$MIN_YAW_SPEED"

CMD=(
    "$PYTHON_BIN" -m realworld.go2_client
    --server-base-url "$SERVER_BASE_URL"
    --control-backend "$CONTROL_BACKEND"
    --real-robot-ack "$REAL_ROBOT_ACK"
    --camera "$CAMERA"
    --jpeg-quality "$JPEG_QUALITY"
    --capture-flush-frames "$CAPTURE_FLUSH_FRAMES"
    --save-output-dir "$SAVE_OUTPUT_DIR"
    --save-contents "$SAVE_CONTENTS"
    --video-fps "$VIDEO_FPS"
    --video-writer "$VIDEO_WRITER"
    --video-codec "$VIDEO_CODEC"
    --upload-image-mode "$UPLOAD_IMAGE_MODE"
    --upload-width "$UPLOAD_WIDTH"
    --upload-height "$UPLOAD_HEIGHT"
    --max-replans "$MAX_REPLANS"
    --actions-per-replan "$ACTIONS_PER_REPLAN"
    --prefetch-after-actions "$PREDICT_PREFETCH_AFTER_ACTIONS"
    --forward-distance "$FORWARD_DISTANCE"
    --forward-speed "$FORWARD_SPEED"
    --turn-degrees "$TURN_DEGREES"
    --yaw-speed "$YAW_SPEED"
    --command-period "$COMMAND_PERIOD"
    --settle-time "$SETTLE_TIME"
    --post-capture-settle-time "$POST_CAPTURE_SETTLE_TIME"
    --odom-timeout "$ODOM_TIMEOUT"
    --forward-tolerance "$FORWARD_TOLERANCE"
    --turn-tolerance-degrees "$TURN_TOLERANCE_DEGREES"
    --forward-kp "$FORWARD_KP"
    --forward-kd "$FORWARD_KD"
    --turn-kp "$TURN_KP"
    --turn-kd "$TURN_KD"
    --min-forward-speed "$MIN_FORWARD_SPEED"
    --min-yaw-speed "$MIN_YAW_SPEED"
)

if [[ "$ODOM_CONTROL" != "yes" ]]; then
    CMD+=(--disable-odom-control)
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
if [[ -n "$VIDEO_WIDTH" ]]; then
    CMD+=(--video-width "$VIDEO_WIDTH")
fi
if [[ -n "$VIDEO_HEIGHT" ]]; then
    CMD+=(--video-height "$VIDEO_HEIGHT")
fi

"${CMD[@]}"
