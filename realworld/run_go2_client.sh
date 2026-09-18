#!/usr/bin/env bash
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Runtime environment; navigation/camera/motion settings live in go2_client.yaml.
PYTHON_BIN="/usr/bin/python3"
ROS2_UNITREE_API_WS="$SCRIPT_DIR/ros2_unitree_api_ws"
CONFIG_PATH="$SCRIPT_DIR/go2_client.yaml"
export PYTHONUNBUFFERED=1
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
CLIENT_ARGS=(--config "$CONFIG_PATH" "$@")

# Inspection commands do not need ROS or access to the robot/camera.
for arg in "$@"; do
    if [[ "$arg" == "--help" || "$arg" == "-h" || "$arg" == "--print-config" ]]; then
        exec "$PYTHON_BIN" -m realworld.go2_client "${CLIENT_ARGS[@]}"
    fi
done

CONTROL_BACKEND="$("$PYTHON_BIN" -c 'from realworld.go2_client import parse_args; print(parse_args().control_backend)' "${CLIENT_ARGS[@]}")"
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

exec "$PYTHON_BIN" -m realworld.go2_client "${CLIENT_ARGS[@]}"
