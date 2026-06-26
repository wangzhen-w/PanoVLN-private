from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import requests


DEFAULT_SERVER_BASE_URL = "http://10.14.114.132:8000"
DEFAULT_PREDICT_PATH = "/predict"
DEFAULT_READY_PATH = "/ready"
ACTION_WORDS = {"stop", "forward", "left", "right"}
DEFAULT_REQUEST_TIMEOUT_S = 180.0
DEFAULT_SDK_TIMEOUT_S = 10.0
DEFAULT_JPEG_QUALITY = 90
DEFAULT_CAMERA_FOURCC = "MJPG"
DEFAULT_CAMERA_WARMUP_FRAMES = 10
DEFAULT_CAPTURE_FLUSH_FRAMES = 2
DEFAULT_MAX_MEMORY_IMAGES = 10
DEFAULT_MEMORY_POOL_WINDOW_FRAMES = 100
DEFAULT_HISTORY_LIMIT = DEFAULT_MEMORY_POOL_WINDOW_FRAMES + DEFAULT_MAX_MEMORY_IMAGES + 10
DEFAULT_UPLOAD_IMAGE_MODE = "resize"
DEFAULT_UPLOAD_IMAGE_SIZE = (1280, 640)
DEFAULT_ACTIONS_PER_REPLAN = 4
DEFAULT_COMMAND_PERIOD_S = 0.05
DEFAULT_SETTLE_TIME_S = 0.25
MAX_FORWARD_DISTANCE_M = 0.50
MAX_FORWARD_SPEED_MPS = 0.50
MAX_TURN_DEGREES = 45.0
MAX_YAW_SPEED_RADPS = 0.80
MAX_ACTION_DURATION_S = 5.0


@dataclass
class MotionConfig:
    forward_distance_m: float = 0.25
    forward_speed_mps: float = 0.20
    turn_degrees: float = 15.0
    yaw_speed_radps: float = 0.35
    command_period_s: float = 0.05
    settle_time_s: float = 0.25


def validate_motion_config(motion: MotionConfig) -> None:
    checks = (
        (motion.forward_distance_m, 0.0, MAX_FORWARD_DISTANCE_M, "forward_distance"),
        (motion.forward_speed_mps, 0.0, MAX_FORWARD_SPEED_MPS, "forward_speed"),
        (motion.turn_degrees, 0.0, MAX_TURN_DEGREES, "turn_degrees"),
        (motion.yaw_speed_radps, 0.0, MAX_YAW_SPEED_RADPS, "yaw_speed"),
    )
    for value, lower, upper, name in checks:
        if not (lower < float(value) <= upper):
            raise ValueError(f"{name} must be > {lower} and <= {upper}, got {value}")
    if motion.command_period_s <= 0:
        raise ValueError(f"command_period must be > 0, got {motion.command_period_s}")
    if motion.settle_time_s < 0:
        raise ValueError(f"settle_time must be >= 0, got {motion.settle_time_s}")


def build_vln_image_selection(
    current_step: int,
    last_frame_index: int,
    max_memory_images: int = 10,
    memory_pool_window_frames: int = 100,
) -> list[int]:
    max_memory_images = max(0, int(max_memory_images))
    memory_pool_window_frames = max(1, int(memory_pool_window_frames))
    current_frame_index = min(max(0, int(current_step)), int(last_frame_index))
    pool_start_frame = max(0, current_frame_index - memory_pool_window_frames + 1)
    candidate_frame_indices = list(range(pool_start_frame, current_frame_index + 1))

    total_selected_images = max_memory_images + 1
    if total_selected_images <= 0 or not candidate_frame_indices:
        return [current_frame_index]
    if total_selected_images == 1:
        return [current_frame_index]
    if len(candidate_frame_indices) <= total_selected_images:
        return candidate_frame_indices

    last_candidate_position = len(candidate_frame_indices) - 1
    selected_positions = [
        (slot * last_candidate_position) // (total_selected_images - 1)
        for slot in range(total_selected_images)
    ]
    return [candidate_frame_indices[position] for position in selected_positions]


def selected_history(
    history: Sequence[bytes],
    *,
    max_memory_images: int,
    memory_pool_window_frames: int,
) -> list[bytes]:
    if not history:
        return []
    indices = build_vln_image_selection(
        current_step=len(history) - 1,
        last_frame_index=len(history) - 1,
        max_memory_images=max_memory_images,
        memory_pool_window_frames=memory_pool_window_frames,
    )
    return [history[index] for index in indices]


def prepare_upload_images(
    images: Sequence[bytes],
    *,
    mode: str,
    jpeg_quality: int,
    target_size: tuple[int, int],
) -> list[bytes]:
    if mode == "raw":
        return list(images)
    if mode != "resize":
        raise ValueError(f"Unsupported upload image mode: {mode}")
    if not images:
        return []
    target_width, target_height = int(target_size[0]), int(target_size[1])
    if target_width <= 0 or target_height <= 0:
        raise ValueError(f"upload resize target must be positive, got {target_width}x{target_height}")

    import cv2
    import numpy as np

    prepared: list[bytes] = []
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, int(jpeg_quality)]
    for image_index, image_bytes in enumerate(images):
        encoded_array = np.frombuffer(image_bytes, dtype=np.uint8)
        frame = cv2.imdecode(encoded_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise RuntimeError(f"Could not decode camera frame {image_index} before upload")

        height, width = frame.shape[:2]
        if (width, height) != (target_width, target_height):
            interpolation = cv2.INTER_AREA if target_width <= width and target_height <= height else cv2.INTER_LINEAR
            frame = cv2.resize(frame, (target_width, target_height), interpolation=interpolation)

        ok, encoded = cv2.imencode(".jpg", frame, encode_params)
        if not ok:
            raise RuntimeError(f"JPEG upload encoding failed for frame {image_index}")
        prepared.append(encoded.tobytes())
    return prepared


class OpenCVCamera:
    def __init__(
        self,
        source: str | int,
        *,
        width: Optional[int],
        height: Optional[int],
        fps: Optional[int],
        jpeg_quality: int,
        fourcc: Optional[str],
        warmup_frames: int,
    ):
        import cv2

        self.cv2 = cv2
        self.jpeg_quality = int(jpeg_quality)
        self.cap, self.backend_name = self._open_capture(cv2, source)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera source: {source}")
        if fourcc:
            fourcc = str(fourcc).strip().upper()
            if len(fourcc) != 4:
                raise ValueError(f"camera fourcc must be four characters, got {fourcc!r}")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        if width:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        if height:
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))
        if fps:
            self.cap.set(cv2.CAP_PROP_FPS, int(fps))
        for _ in range(max(0, int(warmup_frames))):
            self.cap.read()
        actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        actual_fps = self.cap.get(cv2.CAP_PROP_FPS) or 0.0
        actual_fourcc = int(self.cap.get(cv2.CAP_PROP_FOURCC) or 0)
        actual_fourcc_text = actual_fourcc.to_bytes(4, "little", signed=False).decode(
            "latin1",
            errors="replace",
        )
        print(
            {
                "camera_backend": self.backend_name,
                "camera_width": actual_width,
                "camera_height": actual_height,
                "camera_fps": actual_fps,
                "camera_fourcc": actual_fourcc_text,
            },
            flush=True,
        )
        if actual_width > 0 and actual_height > 0:
            aspect = actual_width / max(actual_height, 1)
            if abs(aspect - 2.0) > 0.35:
                print(
                    f"warning: camera aspect ratio is {aspect:.2f}; "
                    "PanoVLN expects an equirectangular image close to 2:1.",
                    file=sys.stderr,
                    flush=True,
                )

    @staticmethod
    def _is_v4l2_source(source: str | int) -> bool:
        return isinstance(source, int) or str(source).startswith("/dev/video")

    @classmethod
    def _open_capture(cls, cv2, source: str | int):
        if cls._is_v4l2_source(source) and hasattr(cv2, "CAP_V4L2"):
            cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
            if cap.isOpened():
                return cap, "v4l2"
            cap.release()
        return cv2.VideoCapture(source), "default"

    def read_jpeg(self, flush_frames: int = 0) -> bytes:
        for _ in range(max(0, int(flush_frames))):
            self.cap.grab()
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera frame capture failed")
        ok, encoded = self.cv2.imencode(
            ".jpg",
            frame,
            [self.cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return encoded.tobytes()

    def close(self) -> None:
        self.cap.release()


class RobotBackend:
    def stand(self) -> None:
        pass

    def move(self, vx: float, vy: float, vyaw: float) -> None:
        raise NotImplementedError

    def stop(self) -> None:
        self.move(0.0, 0.0, 0.0)

    def close(self) -> None:
        self.stop()


class DryRunBackend(RobotBackend):
    def move(self, vx: float, vy: float, vyaw: float) -> None:
        print(f"[dry-run] move vx={vx:.3f} vy={vy:.3f} vyaw={vyaw:.3f}", flush=True)

    def stop(self) -> None:
        print("[dry-run] stop", flush=True)


class Sdk2SportBackend(RobotBackend):
    def __init__(self, network_interface: Optional[str], timeout_s: float = 10.0):
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize
        from unitree_sdk2py.go2.sport.sport_client import SportClient

        if network_interface:
            ChannelFactoryInitialize(0, network_interface)
        else:
            ChannelFactoryInitialize(0)
        self.client = SportClient()
        if hasattr(self.client, "SetTimeout"):
            self.client.SetTimeout(float(timeout_s))
        self.client.Init()

    def stand(self) -> None:
        for method_name in ("BalanceStand", "StandUp"):
            method = getattr(self.client, method_name, None)
            if method is not None:
                method()
                time.sleep(1.0)
                return

    def move(self, vx: float, vy: float, vyaw: float) -> None:
        self.client.Move(float(vx), float(vy), float(vyaw))

    def stop(self) -> None:
        method = getattr(self.client, "StopMove", None)
        if method is not None:
            try:
                result = method()
                if result == 0:
                    return
                print(
                    f"warning: StopMove returned {result}; falling back to Move(0,0,0)",
                    file=sys.stderr,
                    flush=True,
                )
            except Exception as exc:
                print(
                    f"warning: StopMove failed with {type(exc).__name__}: {exc}; "
                    "falling back to Move(0,0,0)",
                    file=sys.stderr,
                    flush=True,
                )
        self.move(0.0, 0.0, 0.0)


class Ros2SportBackend(RobotBackend):
    SPORT_API_ID_MOVE = 1008

    def __init__(self, node_name: str = "pano_vln_go2_client"):
        import rclpy
        from unitree_api.msg import Request
        from unitree_api.msg import RequestHeader

        self.rclpy = rclpy
        self.Request = Request
        self.RequestHeader = RequestHeader
        rclpy.init(args=None)
        self.node = rclpy.create_node(node_name)
        self.publisher = self.node.create_publisher(Request, "/api/sport/request", 5)

    def move(self, vx: float, vy: float, vyaw: float) -> None:
        parameter = json.dumps({"x": float(vx), "y": float(vy), "z": float(vyaw)})
        header = self.RequestHeader()
        header.identity._api_id = self.SPORT_API_ID_MOVE
        header.identity.id = time.monotonic_ns()
        self.publisher.publish(self.Request(parameter=parameter, header=header))
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def close(self) -> None:
        self.stop()
        self.node.destroy_node()
        self.rclpy.shutdown()


class VLNHttpClient:
    def __init__(self, server_base_url: str, timeout_s: float, predict_path: str = DEFAULT_PREDICT_PATH):
        self.server_base_url = server_base_url.rstrip("/")
        self.predict_url = self._join_url(server_base_url, predict_path)
        self.timeout_s = float(timeout_s)

    @staticmethod
    def _join_url(base_url: str, path: str) -> str:
        return f"{base_url.rstrip('/')}/{path.lstrip('/')}"

    def ready(self, timeout_s: float = 10.0) -> dict:
        ready_url = self._join_url(self.server_base_url, DEFAULT_READY_PATH)
        response = requests.get(ready_url, timeout=timeout_s)
        response.raise_for_status()
        return response.json()

    def predict(
        self,
        *,
        instruction: str,
        images: Sequence[bytes],
    ) -> dict:
        files = [
            ("images", (f"frame_{index:03d}.jpg", image, "image/jpeg"))
            for index, image in enumerate(images)
        ]
        data = {"instruction": instruction}
        response = requests.post(
            self.predict_url,
            data=data,
            files=files,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        return response.json()


def normalize_actions(actions: Iterable[object]) -> list[str]:
    normalized: list[str] = []
    for action in actions:
        value = str(action).strip().lower()
        if value not in ACTION_WORDS:
            continue
        normalized.append(value)
    if not normalized:
        normalized.append("stop")
    return normalized


def run_velocity(backend: RobotBackend, vx: float, vy: float, vyaw: float, duration_s: float, period_s: float) -> None:
    deadline = time.monotonic() + max(0.0, float(duration_s))
    while time.monotonic() < deadline:
        backend.move(vx, vy, vyaw)
        time.sleep(max(0.01, float(period_s)))
    backend.stop()


def execute_action(
    backend: RobotBackend,
    action: str,
    motion: MotionConfig,
) -> bool:
    if action == "stop":
        backend.stop()
        return True
    if action == "forward":
        duration = motion.forward_distance_m / max(motion.forward_speed_mps, 1e-6)
        duration = min(duration, MAX_ACTION_DURATION_S)
        run_velocity(
            backend,
            motion.forward_speed_mps,
            0.0,
            0.0,
            duration,
            motion.command_period_s,
        )
        return False
    if action in {"left", "right"}:
        yaw = abs(motion.yaw_speed_radps)
        yaw = yaw if action == "left" else -yaw
        duration = math.radians(motion.turn_degrees) / max(abs(motion.yaw_speed_radps), 1e-6)
        duration = min(duration, MAX_ACTION_DURATION_S)
        run_velocity(backend, 0.0, 0.0, yaw, duration, motion.command_period_s)
        return False
    return False


def read_instruction(args: argparse.Namespace) -> str:
    if args.instruction_file:
        return Path(args.instruction_file).read_text(encoding="utf-8").strip()
    if args.instruction:
        return args.instruction.strip()
    return input("Instruction: ").strip()


def build_backend(args: argparse.Namespace) -> RobotBackend:
    if args.control_backend == "dry-run":
        return DryRunBackend()
    if args.control_backend == "sdk2":
        return Sdk2SportBackend(args.network_interface, timeout_s=args.sdk_timeout)
    if args.control_backend == "ros2":
        return Ros2SportBackend()
    raise ValueError(f"Unsupported backend: {args.control_backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PanoVLN on a Unitree Go2.")
    parser.add_argument("--server-base-url", default=DEFAULT_SERVER_BASE_URL)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--instruction-file", default=None)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--control-backend", choices=("dry-run", "sdk2", "ros2"), default="dry-run")
    parser.add_argument("--real-robot-ack", default="")
    parser.add_argument("--network-interface", default=None, help="Unitree SDK2 NIC, for example eth0.")
    parser.add_argument("--sdk-timeout", type=float, default=DEFAULT_SDK_TIMEOUT_S)
    parser.add_argument("--camera", default="0", help="OpenCV camera index or device path.")
    parser.add_argument("--frame-width", type=int, default=None)
    parser.add_argument("--frame-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--camera-fourcc", default=DEFAULT_CAMERA_FOURCC)
    parser.add_argument("--camera-warmup-frames", type=int, default=DEFAULT_CAMERA_WARMUP_FRAMES)
    parser.add_argument("--capture-flush-frames", type=int, default=DEFAULT_CAPTURE_FLUSH_FRAMES)
    parser.add_argument("--history-limit", type=int, default=DEFAULT_HISTORY_LIMIT)
    parser.add_argument("--upload-image-mode", choices=("resize", "raw"), default=DEFAULT_UPLOAD_IMAGE_MODE)
    parser.add_argument("--upload-width", type=int, default=DEFAULT_UPLOAD_IMAGE_SIZE[0])
    parser.add_argument("--upload-height", type=int, default=DEFAULT_UPLOAD_IMAGE_SIZE[1])
    parser.add_argument("--upload-max-memory-images", type=int, default=DEFAULT_MAX_MEMORY_IMAGES)
    parser.add_argument("--upload-memory-pool-window-frames", type=int, default=DEFAULT_MEMORY_POOL_WINDOW_FRAMES)
    parser.add_argument("--max-replans", type=int, default=50, help="0 means unlimited.")
    parser.add_argument("--actions-per-replan", type=int, default=DEFAULT_ACTIONS_PER_REPLAN)
    parser.add_argument("--forward-distance", type=float, default=0.25)
    parser.add_argument("--forward-speed", type=float, default=0.20)
    parser.add_argument("--turn-degrees", type=float, default=15.0)
    parser.add_argument("--yaw-speed", type=float, default=0.35)
    parser.add_argument("--command-period", type=float, default=DEFAULT_COMMAND_PERIOD_S)
    parser.add_argument("--settle-time", type=float, default=DEFAULT_SETTLE_TIME_S)
    return parser.parse_args()


def parse_camera_source(raw: str) -> str | int:
    try:
        return int(raw)
    except ValueError:
        return raw


def main() -> None:
    args = parse_args()
    instruction = read_instruction(args)
    if not instruction:
        raise ValueError("Instruction is empty")
    if args.control_backend != "dry-run" and args.real_robot_ack != "yes":
        raise ValueError("Set REAL_ROBOT_ACK=\"yes\" in run_go2_client.sh before using sdk2 or ros2")
    if args.actions_per_replan <= 0:
        raise ValueError(f"actions_per_replan must be > 0, got {args.actions_per_replan}")

    motion = MotionConfig(
        forward_distance_m=args.forward_distance,
        forward_speed_mps=args.forward_speed,
        turn_degrees=args.turn_degrees,
        yaw_speed_radps=args.yaw_speed,
        command_period_s=args.command_period,
        settle_time_s=args.settle_time,
    )
    validate_motion_config(motion)
    camera = OpenCVCamera(
        parse_camera_source(args.camera),
        width=args.frame_width,
        height=args.frame_height,
        fps=args.camera_fps,
        jpeg_quality=args.jpeg_quality,
        fourcc=args.camera_fourcc,
        warmup_frames=args.camera_warmup_frames,
    )
    backend = build_backend(args)
    client = VLNHttpClient(args.server_base_url, timeout_s=args.request_timeout)
    history: deque[bytes] = deque(maxlen=max(1, args.history_limit))

    try:
        first_frame = camera.read_jpeg(flush_frames=args.capture_flush_frames)
        history.append(first_frame)
        ready = client.ready()
        print({"server_ready": ready}, flush=True)
        backend.stand()
        replans = 0
        while args.max_replans <= 0 or replans < args.max_replans:
            selected_images = selected_history(
                list(history),
                max_memory_images=args.upload_max_memory_images,
                memory_pool_window_frames=args.upload_memory_pool_window_frames,
            )
            request_images = prepare_upload_images(
                selected_images,
                mode=args.upload_image_mode,
                jpeg_quality=args.jpeg_quality,
                target_size=(args.upload_width, args.upload_height),
            )
            request_bytes = sum(len(image) for image in request_images)
            started = time.perf_counter()
            result = client.predict(
                instruction=instruction,
                images=request_images,
            )
            replans += 1

            actions = normalize_actions(
                result.get("executable_actions") or result.get("actions") or []
            )
            print(
                {
                    "replan": replans,
                    "actions": actions,
                    "raw_text": result.get("raw_text", ""),
                    "request_images": len(request_images),
                    "upload_image_mode": args.upload_image_mode,
                    "upload_mb": round(request_bytes / (1024 * 1024), 3),
                    "round_trip_s": time.perf_counter() - started,
                    "server_latency_s": result.get("latency_s"),
                },
                flush=True,
            )

            should_stop = False
            for action_index, action in enumerate(actions):
                if action_index >= args.actions_per_replan:
                    break
                should_stop = execute_action(backend, action, motion)
                if should_stop:
                    break
                time.sleep(max(0.0, motion.settle_time_s))
                history.append(camera.read_jpeg(flush_frames=args.capture_flush_frames))
            if should_stop:
                break
    except KeyboardInterrupt:
        print("Interrupted, stopping robot.", flush=True)
    finally:
        try:
            backend.close()
        finally:
            camera.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"fatal: {exc}", file=sys.stderr, flush=True)
        raise
