from __future__ import annotations

import argparse
import json
import math
import queue
import re
import shutil
import sys
import threading
import time
import subprocess
from collections import deque
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from realworld.client_config import config_arguments
from src.eval.action_policy import (
    ATOMIC_ACTION_NAMES,
    select_stop_commit_horizon,
    select_uncertainty_horizon,
    DEFAULT_REPLAN_ACTION_RANGE,
    DEFAULT_STOP_COMMIT_MAX_ACTIONS,
    DEFAULT_UNCERTAINTY_BUDGET,
    validate_replan_action_range,
    validate_stop_commit_max_actions,
    validate_uncertainty_budget,
)

DEFAULT_SERVER_BASE_URL = "http://10.14.114.132:8000"
DEFAULT_PREDICT_PATH = "/predict"
DEFAULT_READY_PATH = "/ready"
DEFAULT_MAX_MEMORY_IMAGES = 10
DEFAULT_MEMORY_POOL_WINDOW_FRAMES = 100
DEFAULT_HISTORY_LIMIT = DEFAULT_MEMORY_POOL_WINDOW_FRAMES + DEFAULT_MAX_MEMORY_IMAGES + 10
ACTION_WORDS = {"stop", "forward", "left", "right"}
SAVE_CONTENT_WORDS = {"images", "video", "json"}
DEFAULT_REQUEST_TIMEOUT_S = 180.0
DEFAULT_SDK_TIMEOUT_S = 10.0
DEFAULT_JPEG_QUALITY = 90
DEFAULT_CAMERA_FOURCC = "MJPG"
DEFAULT_CAMERA_WARMUP_FRAMES = 10
DEFAULT_CAPTURE_FLUSH_FRAMES = 2
DEFAULT_VIDEO_FPS = 10.0
DEFAULT_VIDEO_CODEC = "libx264"
DEFAULT_VIDEO_WRITER = "ffmpeg"
DEFAULT_SAVE_CONTENTS = "none"
DEFAULT_UPLOAD_IMAGE_MODE = "resize"
DEFAULT_UPLOAD_IMAGE_SIZE = (1280, 640)
# Zero means follow the action-sequence length advertised by the server.  This
# keeps real-world execution aligned with length-ablation checkpoints.
DEFAULT_ACTIONS_PER_REPLAN = 0
DEFAULT_PREFETCH_AFTER_ACTIONS = 0
DEFAULT_EXECUTION_MODE = "continuous"
DEFAULT_COMMAND_PERIOD_S = 0.05
DEFAULT_SETTLE_TIME_S = 0.1
DEFAULT_POST_CAPTURE_SETTLE_TIME_S = 0.1
MAX_FORWARD_DISTANCE_M = 0.50
MAX_FORWARD_SPEED_MPS = 0.50
MAX_TURN_DEGREES = 45.0
MAX_YAW_SPEED_RADPS = 1.20
MAX_ACTION_DURATION_S = 5.0


@dataclass
class SportOdometry:
    x: float
    y: float
    yaw: float
    forward_speed: float
    yaw_speed: float
    stamp_s: float


@dataclass
class MotionConfig:
    forward_distance_m: float = 0.25
    forward_speed_mps: float = 0.35
    turn_degrees: float = 15.0
    yaw_speed_radps: float = 0.80
    command_period_s: float = 0.05
    settle_time_s: float = DEFAULT_SETTLE_TIME_S
    post_capture_settle_time_s: float = DEFAULT_POST_CAPTURE_SETTLE_TIME_S
    odom_control: bool = True
    odom_timeout_s: float = 1.0
    forward_tolerance_m: float = 0.04
    turn_tolerance_degrees: float = 3.0
    forward_kp: float = 3.0
    forward_kd: float = 0.0
    turn_kp: float = 3.0
    turn_kd: float = 0.0
    min_forward_speed_mps: float = 0.15
    min_yaw_speed_radps: float = 0.35
    max_duration_s: float = MAX_ACTION_DURATION_S


@dataclass
class PredictionResult:
    actions: list[str]
    raw_text: str
    request_images: int
    upload_image_mode: str
    upload_mb: float
    round_trip_s: float
    server_latency_s: object
    uncertainty_actions: Optional[list[str]] = None
    action_uncertainties: Optional[list[float]] = None


def parse_actions_per_replan(value: str) -> int | str:
    if value == "uncertainty":
        return value
    try:
        count = int(value)
        if count >= 0:
            return count
    except (TypeError, ValueError):
        pass
    raise argparse.ArgumentTypeError("actions-per-replan must be a nonnegative integer or 'uncertainty'")


def clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), float(lower)), float(upper))


def wrap_angle_rad(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def signed_angle_error_rad(target: float, current: float) -> float:
    return wrap_angle_rad(float(target) - float(current))


def parse_save_contents(raw: str) -> set[str]:
    value = str(raw or "").strip().lower()
    if not value or value == "none":
        return set()
    tokens = {
        token.strip()
        for token in re.split(r"[, ]+", value)
        if token.strip()
    }
    if "all" in tokens:
        return set(SAVE_CONTENT_WORDS)
    unknown = tokens - SAVE_CONTENT_WORDS
    if unknown:
        allowed = ", ".join(sorted(SAVE_CONTENT_WORDS | {"all", "none"}))
        raise ValueError(f"Unsupported save content(s): {sorted(unknown)}. Allowed: {allowed}")
    return tokens


def write_navigation_json(path: Path, data: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_path.replace(path)


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
    if motion.post_capture_settle_time_s < 0:
        raise ValueError(
            f"post_capture_settle_time must be >= 0, got {motion.post_capture_settle_time_s}"
        )
    if motion.odom_timeout_s <= 0:
        raise ValueError(f"odom_timeout must be > 0, got {motion.odom_timeout_s}")
    if motion.forward_tolerance_m <= 0:
        raise ValueError(f"forward_tolerance must be > 0, got {motion.forward_tolerance_m}")
    if motion.turn_tolerance_degrees <= 0:
        raise ValueError(f"turn_tolerance_degrees must be > 0, got {motion.turn_tolerance_degrees}")
    if motion.min_forward_speed_mps < 0 or motion.min_forward_speed_mps > motion.forward_speed_mps:
        raise ValueError(
            "min_forward_speed must be >= 0 and <= forward_speed, "
            f"got {motion.min_forward_speed_mps}"
        )
    if motion.min_yaw_speed_radps < 0 or motion.min_yaw_speed_radps > motion.yaw_speed_radps:
        raise ValueError(
            "min_yaw_speed must be >= 0 and <= yaw_speed, "
            f"got {motion.min_yaw_speed_radps}"
        )


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
    if len(candidate_frame_indices) <= total_selected_images:
        return candidate_frame_indices
    if total_selected_images == 1:
        return [current_frame_index]

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
        self._lock = threading.Lock()
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

    def read_frame(self, flush_frames: int = 0):
        with self._lock:
            for _ in range(max(0, int(flush_frames))):
                self.cap.grab()
            ok, frame = self.cap.read()
            if not ok or frame is None:
                raise RuntimeError("Camera frame capture failed")
        return frame

    def read_jpeg(self, flush_frames: int = 0) -> bytes:
        frame = self.read_frame(flush_frames=flush_frames)
        ok, encoded = self.cv2.imencode(
            ".jpg",
            frame,
            [self.cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality],
        )
        if not ok:
            raise RuntimeError("JPEG encoding failed")
        return encoded.tobytes()

    def close(self) -> None:
        with self._lock:
            self.cap.release()


def resolve_video_path(path: Path) -> Path:
    raw_path = str(path)
    video_suffixes = {".mp4", ".avi", ".mov", ".mkv"}
    if raw_path.endswith("/") or path.suffix.lower() not in video_suffixes:
        path.mkdir(parents=True, exist_ok=True)
        return path / f"navigation_{time.strftime('%Y%m%d-%H%M%S')}.mp4"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


class NavigationVideoRecorder:
    def __init__(
        self,
        camera: OpenCVCamera,
        output_path: Path,
        *,
        fps: float,
        codec: str,
        writer_backend: str,
        width: Optional[int],
        height: Optional[int],
    ):
        self.camera = camera
        self.output_path = resolve_video_path(output_path)
        self.fps = float(fps)
        self.codec = str(codec).strip() or DEFAULT_VIDEO_CODEC
        self.writer_backend = str(writer_backend).strip().lower() or DEFAULT_VIDEO_WRITER
        self.width = int(width) if width else None
        self.height = int(height) if height else None
        self._stop_event = threading.Event()
        self._capture_thread: Optional[threading.Thread] = None
        self._writer_thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue[object] = queue.Queue(maxsize=96)
        self._writer = None
        self._captured_frame_count = 0
        self._video_frame_count = 0
        self._dropped_frame_count = 0
        self._started_s = 0.0
        self._error: Optional[BaseException] = None

    def start(self) -> None:
        if self.fps <= 0:
            raise ValueError(f"video_fps must be > 0, got {self.fps}")
        if (self.width is None) != (self.height is None):
            raise ValueError("video_width and video_height must either both be set or both be empty")
        if self.writer_backend not in {"ffmpeg", "opencv"}:
            raise ValueError(f"video_writer must be 'ffmpeg' or 'opencv', got {self.writer_backend!r}")
        if self.writer_backend == "ffmpeg" and shutil.which("ffmpeg") is None:
            raise RuntimeError("ffmpeg is not available on PATH")
        if self.writer_backend == "opencv" and len(self.codec) != 4:
            raise ValueError(f"video codec must be four characters, got {self.codec!r}")
        self._started_s = time.monotonic()
        self._capture_thread = threading.Thread(
            target=self._capture_loop,
            name="navigation-video-capture",
            daemon=True,
        )
        self._writer_thread = threading.Thread(
            target=self._writer_loop,
            name="navigation-video-writer",
            daemon=True,
        )
        self._writer_thread.start()
        self._capture_thread.start()
        print(
            {
                "video_recording": "started",
                "path": str(self.output_path),
                "writer": self.writer_backend,
                "fps": self.fps,
                "codec": self.codec,
                "width": self.width,
                "height": self.height,
            },
            flush=True,
        )

    def _open_writer(self, frame) -> None:
        if self.writer_backend == "ffmpeg":
            self._open_ffmpeg_writer(frame)
        else:
            self._open_opencv_writer(frame)

    def _open_opencv_writer(self, frame) -> None:
        height, width = frame.shape[:2]
        fourcc = self.camera.cv2.VideoWriter_fourcc(*self.codec)
        self._writer = self.camera.cv2.VideoWriter(
            str(self.output_path),
            fourcc,
            self.fps,
            (width, height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open video writer: {self.output_path}")

    def _ffmpeg_encoder_name(self) -> str:
        codec = self.codec.strip()
        codec_lower = codec.lower()
        if codec_lower == "mp4v":
            return "mpeg4"
        if codec_lower == "mjpg":
            return "mjpeg"
        return codec

    def _open_ffmpeg_writer(self, frame) -> None:
        height, width = frame.shape[:2]
        encoder = self._ffmpeg_encoder_name()
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            f"{self.fps:g}",
            "-i",
            "pipe:0",
            "-an",
            "-vcodec",
            encoder,
        ]
        if encoder == "libx264":
            cmd.extend(["-preset", "ultrafast", "-tune", "zerolatency", "-pix_fmt", "yuv420p"])
        elif encoder in {"mpeg4", "mjpeg"}:
            cmd.extend(["-q:v", "4"])
        cmd.append(str(self.output_path))
        self._writer = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )

    def _write_frame(self, frame) -> None:
        if self.writer_backend == "ffmpeg":
            if self._writer is None or self._writer.stdin is None:
                raise RuntimeError("ffmpeg writer is not open")
            self._writer.stdin.write(frame.tobytes())
            return
        self._writer.write(frame)

    def _close_writer(self) -> None:
        if self._writer is None:
            return
        if self.writer_backend == "ffmpeg":
            if self._writer.stdin is not None:
                self._writer.stdin.close()
                self._writer.stdin = None
            stderr = b""
            if self._writer.stderr is not None:
                stderr = self._writer.stderr.read()
            self._writer.wait(timeout=10.0)
            if self._writer.returncode not in (0, None):
                message = stderr.decode("utf-8", errors="replace").strip()
                raise RuntimeError(f"ffmpeg exited with code {self._writer.returncode}: {message}")
            return
        self._writer.release()

    def _prepare_frame(self, frame):
        if self.width is None or self.height is None:
            return frame
        if frame.shape[1] == self.width and frame.shape[0] == self.height:
            return frame
        interpolation = (
            self.camera.cv2.INTER_AREA
            if self.width <= frame.shape[1] and self.height <= frame.shape[0]
            else self.camera.cv2.INTER_LINEAR
        )
        return self.camera.cv2.resize(frame, (self.width, self.height), interpolation=interpolation)

    def _enqueue_frame(self, frame: object) -> None:
        try:
            self._frame_queue.put_nowait(frame)
            return
        except queue.Full:
            pass
        try:
            self._frame_queue.get_nowait()
            self._frame_queue.task_done()
            self._dropped_frame_count += 1
        except queue.Empty:
            pass
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            self._dropped_frame_count += 1

    def _capture_loop(self) -> None:
        period_s = 1.0 / self.fps
        next_capture_s = time.monotonic()
        try:
            while not self._stop_event.is_set():
                now = time.monotonic()
                if now < next_capture_s:
                    time.sleep(min(0.02, next_capture_s - now))
                    continue
                frame = self.camera.read_frame(flush_frames=0)
                frame = self._prepare_frame(frame)
                self._captured_frame_count += 1
                self._enqueue_frame(frame)
                next_capture_s += period_s
                if next_capture_s < time.monotonic() - period_s:
                    next_capture_s = time.monotonic()
        except BaseException as exc:
            self._error = exc
            print({"warning": "video_recording_failed", "error": repr(exc)}, file=sys.stderr, flush=True)

    def _writer_loop(self) -> None:
        try:
            while not self._stop_event.is_set() or not self._frame_queue.empty():
                try:
                    frame = self._frame_queue.get(timeout=0.1)
                except queue.Empty:
                    continue
                try:
                    if self._writer is None:
                        self._open_writer(frame)
                    self._write_frame(frame)
                    self._video_frame_count += 1
                finally:
                    self._frame_queue.task_done()
        except BaseException as exc:
            self._error = exc
            print({"warning": "video_writer_failed", "error": repr(exc)}, file=sys.stderr, flush=True)

    def close(self) -> dict[str, object]:
        self._stop_event.set()
        if self._capture_thread is not None:
            self._capture_thread.join(timeout=5.0)
        if self._writer_thread is not None:
            self._writer_thread.join(timeout=10.0)
        try:
            self._close_writer()
        except BaseException as exc:
            self._error = exc
            print({"warning": "video_writer_close_failed", "error": repr(exc)}, file=sys.stderr, flush=True)
        elapsed_s = max(time.monotonic() - self._started_s, 1e-6)
        summary = {
            "video_recording": "stopped",
            "path": str(self.output_path),
            "writer": self.writer_backend,
            "captured_frames": self._captured_frame_count,
            "video_frames": self._video_frame_count,
            "dropped_frames": self._dropped_frame_count,
            "elapsed_s": round(elapsed_s, 3),
            "target_video_fps": self.fps,
            "actual_capture_fps": round(self._captured_frame_count / elapsed_s, 3),
            "error": repr(self._error) if self._error is not None else None,
        }
        print(summary, flush=True)
        return summary


class RobotBackend:
    def stand(self) -> None:
        pass

    def move(self, vx: float, vy: float, vyaw: float) -> None:
        raise NotImplementedError

    def execute_motion_action(
        self, action: str, motion: MotionConfig, *, on_progress: Optional[Callable[[float], None]] = None,
    ) -> Optional[bool]:
        """Report forward metres or degrees turned in the requested direction."""
        return None

    def stop(self) -> None:
        self.move(0.0, 0.0, 0.0)

    def close(self) -> None:
        self.stop()


class DryRunBackend(RobotBackend):
    def move(self, vx: float, vy: float, vyaw: float) -> None:
        print(f"[dry-run] move vx={vx:.3f} vy={vy:.3f} vyaw={vyaw:.3f}", flush=True)

    def stop(self) -> None:
        print("[dry-run] stop", flush=True)


class Ros2SportBackend(RobotBackend):
    SPORT_API_ID_BALANCE_STAND = 1002
    SPORT_API_ID_MOVE = 1008
    SPORT_RESPONSE_TIMEOUT_S = 5.0

    def __init__(self, node_name: str = "pano_vln_go2_client"):
        import rclpy
        from unitree_api.msg import Request
        from unitree_api.msg import RequestHeader
        from unitree_api.msg import Response

        try:
            from unitree_go.msg import SportModeState
        except Exception as exc:
            SportModeState = None
            print(
                {
                    "warning": "unitree_go/SportModeState unavailable; falling back to open-loop motion",
                    "error": repr(exc),
                },
                flush=True,
            )

        self.rclpy = rclpy
        self.Request = Request
        self.RequestHeader = RequestHeader
        self.Response = Response
        self.SportModeState = SportModeState
        self._responses_by_request_id: dict[int, dict[str, int]] = {}
        self._odom: Optional[SportOdometry] = None
        rclpy.init(args=None)
        self.node = rclpy.create_node(node_name)
        self.publisher = self.node.create_publisher(Request, "/api/sport/request", 5)
        self.response_subscription = self.node.create_subscription(
            Response,
            "/api/sport/response",
            self._handle_response,
            10,
        )
        self.odom_subscription = None
        if SportModeState is not None:
            self.odom_subscription = self.node.create_subscription(
                SportModeState,
                "/sportmodestate",
                self._handle_sport_mode_state,
                10,
            )

    @staticmethod
    def _identity_api_id(identity: object) -> int:
        if hasattr(identity, "api_id"):
            return int(getattr(identity, "api_id"))
        return int(getattr(identity, "_api_id"))

    @staticmethod
    def _set_identity_api_id(identity: object, api_id: int) -> None:
        if hasattr(identity, "api_id"):
            setattr(identity, "api_id", int(api_id))
        else:
            setattr(identity, "_api_id", int(api_id))

    def _make_request(self, api_id: int, payload: dict[str, object], *, noreply: bool) -> tuple[object, int]:
        request_id = time.monotonic_ns()
        header = self.RequestHeader()
        self._set_identity_api_id(header.identity, api_id)
        header.identity.id = request_id
        if hasattr(header, "lease"):
            header.lease.id = 0
        if hasattr(header, "policy"):
            header.policy.priority = 0
            header.policy.noreply = bool(noreply)
        return self.Request(parameter=json.dumps(payload), header=header), request_id

    def _handle_response(self, response: object) -> None:
        header = getattr(response, "header", None)
        identity = getattr(header, "identity", None)
        status = getattr(header, "status", None)
        if identity is None or status is None:
            return
        request_id = int(getattr(identity, "id"))
        self._responses_by_request_id[request_id] = {
            "request_id": request_id,
            "api_id": self._identity_api_id(identity),
            "code": int(getattr(status, "code")),
        }

    def _handle_sport_mode_state(self, msg: object) -> None:
        try:
            position = getattr(msg, "position")
            velocity = getattr(msg, "velocity")
            imu_state = getattr(msg, "imu_state")
            rpy = getattr(imu_state, "rpy")
            self._odom = SportOdometry(
                x=float(position[0]),
                y=float(position[1]),
                yaw=float(rpy[2]),
                forward_speed=float(velocity[0]),
                yaw_speed=float(getattr(msg, "yaw_speed")),
                stamp_s=time.monotonic(),
            )
        except Exception as exc:
            print({"warning": "failed_to_parse_sportmodestate", "error": repr(exc)}, flush=True)

    def _wait_for_response(self, request_id: int, api_id: int, timeout_s: float) -> dict[str, int]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=0.05)
            response = self._responses_by_request_id.pop(request_id, None)
            if response is None:
                continue
            if response["api_id"] != int(api_id):
                raise RuntimeError(
                    f"Unexpected sport API response: expected api_id={api_id}, got {response}"
                )
            if response["code"] != 0:
                raise RuntimeError(f"Sport API request failed: {response}")
            return response
        raise TimeoutError(f"Timed out waiting for sport API response api_id={api_id} request_id={request_id}")

    def _latest_odom(self, timeout_s: float = 0.0, max_age_s: float = 0.5) -> Optional[SportOdometry]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            self.rclpy.spin_once(self.node, timeout_sec=0.0)
            if self._odom is not None and time.monotonic() - self._odom.stamp_s <= max_age_s:
                return self._odom
            if time.monotonic() >= deadline:
                return None
            self.rclpy.spin_once(self.node, timeout_sec=min(0.05, max(0.0, deadline - time.monotonic())))

    def _next_odom(self, *, after_stamp_s: Optional[float], timeout_s: float) -> Optional[SportOdometry]:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while time.monotonic() < deadline:
            self.rclpy.spin_once(self.node, timeout_sec=min(0.02, max(0.0, deadline - time.monotonic())))
            if self._odom is None:
                continue
            if after_stamp_s is None or self._odom.stamp_s > after_stamp_s:
                return self._odom
        return self._odom

    def _velocity_with_minimum(self, value: float, limit: float, minimum: float) -> float:
        value = clamp(value, -abs(limit), abs(limit))
        if value == 0.0 or abs(value) >= abs(minimum):
            return value
        return math.copysign(abs(minimum), value)

    def _run_closed_loop_turn(
        self, direction: float, motion: MotionConfig, *, on_progress: Optional[Callable[[float], None]] = None,
    ) -> bool:
        start = self._latest_odom(timeout_s=motion.odom_timeout_s)
        if start is None:
            print({"warning": "no_sportmodestate_for_closed_loop_turn; using open_loop"}, flush=True)
            return False

        # Accumulate yaw changes so merged turns can exceed 180 degrees (or a
        # full revolution) without taking the shorter path in the wrong direction.
        target_rotation = float(direction) * math.radians(motion.turn_degrees)
        rotation = 0.0
        previous_yaw = start.yaw
        tolerance = math.radians(motion.turn_tolerance_degrees)
        deadline = time.monotonic() + motion.max_duration_s
        last_error = target_rotation
        last_odom_stamp = start.stamp_s
        samples = 0

        while time.monotonic() < deadline:
            odom = self._next_odom(
                after_stamp_s=last_odom_stamp,
                timeout_s=max(0.05, motion.command_period_s),
            )
            if odom is None:
                break
            last_odom_stamp = odom.stamp_s
            rotation += signed_angle_error_rad(odom.yaw, previous_yaw)
            previous_yaw = odom.yaw
            error = target_rotation - rotation
            last_error = error
            samples += 1
            if on_progress is not None:
                on_progress(float(direction) * math.degrees(rotation))
            if abs(error) <= tolerance:
                break
            yaw_command = motion.turn_kp * error - motion.turn_kd * odom.yaw_speed
            yaw_command = self._velocity_with_minimum(
                yaw_command,
                motion.yaw_speed_radps,
                motion.min_yaw_speed_radps,
            )
            self.move(0.0, 0.0, yaw_command)
            time.sleep(max(0.01, motion.command_period_s))

        self.stop()
        print(
            {
                "closed_loop_turn": True,
                "target_degrees": round(float(direction) * motion.turn_degrees, 3),
                "remaining_error_degrees": round(math.degrees(last_error), 3),
                "samples": samples,
            },
            flush=True,
        )
        return True

    def _run_closed_loop_forward(
        self, motion: MotionConfig, *, on_progress: Optional[Callable[[float], None]] = None,
    ) -> bool:
        start = self._latest_odom(timeout_s=motion.odom_timeout_s)
        if start is None:
            print({"warning": "no_sportmodestate_for_closed_loop_forward; using open_loop"}, flush=True)
            return False

        target_distance = motion.forward_distance_m
        deadline = time.monotonic() + motion.max_duration_s
        last_error = target_distance
        last_odom_stamp = start.stamp_s
        samples = 0

        while time.monotonic() < deadline:
            odom = self._next_odom(
                after_stamp_s=last_odom_stamp,
                timeout_s=max(0.05, motion.command_period_s),
            )
            if odom is None:
                break
            last_odom_stamp = odom.stamp_s
            dx = odom.x - start.x
            dy = odom.y - start.y
            progress = dx * math.cos(start.yaw) + dy * math.sin(start.yaw)
            error = target_distance - progress
            last_error = error
            samples += 1
            if on_progress is not None:
                on_progress(progress)
            if error <= motion.forward_tolerance_m:
                break
            forward_command = motion.forward_kp * error - motion.forward_kd * odom.forward_speed
            forward_command = clamp(forward_command, 0.0, motion.forward_speed_mps)
            if forward_command > 0.0:
                forward_command = max(forward_command, motion.min_forward_speed_mps)
            yaw_error = signed_angle_error_rad(start.yaw, odom.yaw)
            yaw_command = motion.turn_kp * yaw_error - motion.turn_kd * odom.yaw_speed
            yaw_command = clamp(yaw_command, -motion.yaw_speed_radps, motion.yaw_speed_radps)
            self.move(forward_command, 0.0, yaw_command)
            time.sleep(max(0.01, motion.command_period_s))

        self.stop()
        print(
            {
                "closed_loop_forward": True,
                "target_m": round(target_distance, 3),
                "remaining_error_m": round(last_error, 3),
                "samples": samples,
            },
            flush=True,
        )
        return True

    def stand(self) -> None:
        request, request_id = self._make_request(
            self.SPORT_API_ID_BALANCE_STAND,
            {},
            noreply=False,
        )
        self.publisher.publish(request)
        response = self._wait_for_response(
            request_id,
            self.SPORT_API_ID_BALANCE_STAND,
            self.SPORT_RESPONSE_TIMEOUT_S,
        )
        print({"ros2_balance_stand_response": response}, flush=True)

    def move(self, vx: float, vy: float, vyaw: float) -> None:
        request, _ = self._make_request(
            self.SPORT_API_ID_MOVE,
            {"x": float(vx), "y": float(vy), "z": float(vyaw)},
            noreply=True,
        )
        self.publisher.publish(request)
        self.rclpy.spin_once(self.node, timeout_sec=0.0)

    def execute_motion_action(
        self, action: str, motion: MotionConfig, *, on_progress: Optional[Callable[[float], None]] = None,
    ) -> Optional[bool]:
        if not motion.odom_control or self.odom_subscription is None:
            return None
        if action == "forward":
            if self._run_closed_loop_forward(motion, on_progress=on_progress):
                return False
            return None
        if action == "left":
            if self._run_closed_loop_turn(1.0, motion, on_progress=on_progress):
                return False
            return None
        if action == "right":
            if self._run_closed_loop_turn(-1.0, motion, on_progress=on_progress):
                return False
            return None
        return None

    def close(self) -> None:
        self.stop()
        self.node.destroy_node()
        self.rclpy.shutdown()


class VLNHttpClient:
    def __init__(self, server_base_url: str, timeout_s: float, predict_path: str = DEFAULT_PREDICT_PATH, *,
                 uncertainty_max_actions: Optional[int] = None):
        import requests

        self.uncertainty_max_actions = uncertainty_max_actions
        self.server_base_url = server_base_url.rstrip("/")
        self.predict_url = self._join_url(server_base_url, predict_path)
        self.timeout_s = float(timeout_s)
        self.session = requests.Session()
        self.session.trust_env = False

    @staticmethod
    def _join_url(base_url: str, path: str) -> str:
        return f"{base_url.rstrip('/')}/{path.lstrip('/')}"

    def ready(self, timeout_s: float = 10.0) -> dict:
        ready_url = self._join_url(self.server_base_url, DEFAULT_READY_PATH)
        response = self.session.get(ready_url, timeout=timeout_s)
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
        if self.uncertainty_max_actions is not None:
            data.update(include_uncertainty="true", uncertainty_max_actions=str(self.uncertainty_max_actions))
        response = self.session.post(
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


def request_prediction(
    client: VLNHttpClient,
    *,
    instruction: str,
    history_snapshot: Sequence[bytes],
    max_memory_images: int,
    memory_pool_window_frames: int,
    upload_image_mode: str,
    jpeg_quality: int,
    upload_size: tuple[int, int],
) -> PredictionResult:
    selected_images = selected_history(
        history_snapshot,
        max_memory_images=max_memory_images,
        memory_pool_window_frames=memory_pool_window_frames,
    )
    request_images = prepare_upload_images(
        selected_images,
        mode=upload_image_mode,
        jpeg_quality=jpeg_quality,
        target_size=upload_size,
    )
    request_bytes = sum(len(image) for image in request_images)
    started = time.perf_counter()
    result = client.predict(
        instruction=instruction,
        images=request_images,
    )
    return PredictionResult(
        actions=normalize_actions(result.get("executable_actions") or result.get("actions") or []),
        raw_text=result.get("raw_text", ""),
        request_images=len(request_images),
        upload_image_mode=upload_image_mode,
        upload_mb=round(request_bytes / (1024 * 1024), 3),
        round_trip_s=time.perf_counter() - started,
        server_latency_s=result.get("latency_s"),
        uncertainty_actions=result.get("uncertainty_actions"),
        action_uncertainties=result.get("action_uncertainties"),
    )


class PendingPrediction:
    def __init__(self, *, target, kwargs: dict[str, object]):
        self._done = threading.Event()
        self._result: Optional[PredictionResult] = None
        self._error: Optional[BaseException] = None
        self._thread = threading.Thread(
            target=self._run,
            args=(target, kwargs),
            name="vln-prediction-prefetch",
            daemon=True,
        )
        self._thread.start()

    def _run(self, target, kwargs: dict[str, object]) -> None:
        try:
            self._result = target(**kwargs)
        except BaseException as exc:
            self._error = exc
        finally:
            self._done.set()

    def done(self) -> bool:
        return self._done.is_set()

    def result(self) -> PredictionResult:
        self._done.wait()
        if self._error is not None:
            raise self._error
        if self._result is None:
            raise RuntimeError("Prediction prefetch finished without a result")
        return self._result


def select_prediction_horizon(
    prediction: PredictionResult, *, uncertainty_budget: float,
    replan_action_range: tuple[int, int], stop_commit_max_actions: int,
) -> tuple[int, bool]:
    """Choose the atom count locally using the server's action uncertainties."""
    actions = prediction.actions
    ids = [ATOMIC_ACTION_NAMES.index(action) for action in actions]
    horizon = select_stop_commit_horizon(ids, stop_commit_max_actions)
    if horizon is not None:
        return horizon, True
    prefix = actions[:replan_action_range[1]]
    if "stop" in prefix:
        prefix = prefix[:prefix.index("stop") + 1]
    if prediction.uncertainty_actions != prefix:
        raise ValueError("Parsed actions do not match uncertainty logits")
    if prediction.action_uncertainties is None or len(prediction.action_uncertainties) != len(prefix):
        raise ValueError("Missing action uncertainties")
    return select_uncertainty_horizon(
        prediction.action_uncertainties, uncertainty_budget, replan_action_range,
    ), False


def run_velocity(
    backend: RobotBackend, vx: float, vy: float, vyaw: float, duration_s: float, period_s: float,
    *, on_elapsed: Optional[Callable[[float], None]] = None,
) -> None:
    duration_s = max(0.0, float(duration_s))
    started = time.monotonic()
    try:
        while True:
            elapsed = min(time.monotonic() - started, duration_s)
            if on_elapsed is not None:
                on_elapsed(elapsed)
            # Image capture takes time while the previous velocity command is
            # still active; do not add another command/sleep after the deadline.
            remaining = duration_s - (time.monotonic() - started)
            if remaining <= 0:
                break
            backend.move(vx, vy, vyaw)
            time.sleep(min(max(0.01, float(period_s)), remaining))
    finally:
        backend.stop()


def merge_consecutive_actions(actions: Sequence[str], max_actions: int) -> list[tuple[str, int]]:
    """Merge the executable prefix, counting the limit in original model actions."""
    merged: list[tuple[str, int]] = []
    for action in actions[:max_actions]:
        if merged and merged[-1][0] == action:
            merged[-1] = (action, merged[-1][1] + 1)
        else:
            merged.append((action, 1))
        if action == "stop":
            break
    return merged


def execute_action(
    backend: RobotBackend,
    action: str,
    motion: MotionConfig,
    repeat_count: int = 1,
    on_atom_complete: Optional[Callable[[int], None]] = None,
) -> bool:
    """Capture intermediate atom boundaries in motion; the caller captures the final stopped frame."""
    if repeat_count < 1:
        raise ValueError("repeat_count must be positive")
    if action == "stop":
        backend.stop()
        return True
    atom_size = motion.forward_distance_m if action == "forward" else motion.turn_degrees
    next_atom = 1

    def report_progress(progress: float) -> None:
        nonlocal next_atom
        if on_atom_complete is None:
            return
        while next_atom < repeat_count and progress + 1e-9 >= atom_size * next_atom:
            on_atom_complete(next_atom)
            next_atom += 1

    progress_callback = report_progress if on_atom_complete is not None and repeat_count > 1 else None

    def check_memory_boundaries() -> None:
        if progress_callback is not None and next_atom < repeat_count:
            # A timed-out motion must not silently shift the history/action alignment.
            raise RuntimeError(f"{action} ended before reaching all intermediate atom boundaries")

    # Validate primitive sizes at startup; merged targets are intentionally larger.
    # Retain the original per-action time budget for the whole continuous motion.
    motion = replace(
        motion,
        forward_distance_m=(
            motion.forward_distance_m * repeat_count if action == "forward" else motion.forward_distance_m
        ),
        turn_degrees=(
            motion.turn_degrees * repeat_count if action in {"left", "right"} else motion.turn_degrees
        ),
        max_duration_s=motion.max_duration_s * repeat_count,
    )
    try:
        backend_result = backend.execute_motion_action(action, motion, on_progress=progress_callback)
    except BaseException:
        backend.stop()
        raise
    if backend_result is not None:
        check_memory_boundaries()
        return backend_result
    if action == "forward":
        duration = motion.forward_distance_m / max(motion.forward_speed_mps, 1e-6)
        duration = min(duration, motion.max_duration_s)
        run_velocity(
            backend,
            motion.forward_speed_mps,
            0.0,
            0.0,
            duration,
            motion.command_period_s,
            on_elapsed=(lambda elapsed: report_progress(elapsed * motion.forward_speed_mps))
            if progress_callback is not None else None,
        )
        check_memory_boundaries()
        return False
    if action in {"left", "right"}:
        yaw = abs(motion.yaw_speed_radps)
        yaw = yaw if action == "left" else -yaw
        duration = math.radians(motion.turn_degrees) / max(abs(motion.yaw_speed_radps), 1e-6)
        duration = min(duration, motion.max_duration_s)
        run_velocity(
            backend, 0.0, 0.0, yaw, duration, motion.command_period_s,
            on_elapsed=(lambda elapsed: report_progress(math.degrees(elapsed * abs(yaw))))
            if progress_callback is not None else None,
        )
        check_memory_boundaries()
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
    if args.control_backend == "ros2":
        return Ros2SportBackend()
    raise ValueError(f"Unsupported backend: {args.control_backend}")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PanoVLN on a Unitree Go2.")
    parser.add_argument("--config", help="Grouped YAML configuration; explicit CLI options override it.")
    parser.add_argument("--print-config", action="store_true", help="Print resolved configuration and exit without opening hardware.")
    parser.add_argument("--server-base-url", default=DEFAULT_SERVER_BASE_URL)
    parser.add_argument("--instruction", default=None)
    parser.add_argument("--instruction-file", default=None)
    parser.add_argument("--request-timeout", type=float, default=DEFAULT_REQUEST_TIMEOUT_S)
    parser.add_argument("--control-backend", choices=("dry-run", "ros2"), default="dry-run")
    parser.add_argument("--real-robot-ack", default="")
    parser.add_argument("--camera", default="0", help="OpenCV camera index or device path.")
    parser.add_argument("--frame-width", type=int, default=None)
    parser.add_argument("--frame-height", type=int, default=None)
    parser.add_argument("--camera-fps", type=int, default=None)
    parser.add_argument("--jpeg-quality", type=int, default=DEFAULT_JPEG_QUALITY)
    parser.add_argument("--camera-fourcc", default=DEFAULT_CAMERA_FOURCC)
    parser.add_argument("--camera-warmup-frames", type=int, default=DEFAULT_CAMERA_WARMUP_FRAMES)
    parser.add_argument("--capture-flush-frames", type=int, default=DEFAULT_CAPTURE_FLUSH_FRAMES)
    parser.add_argument(
        "--save-output-dir",
        default="",
        help="Optional output directory for saved frames, video, and navigation JSON.",
    )
    parser.add_argument(
        "--save-contents",
        default=DEFAULT_SAVE_CONTENTS,
        help="Comma-separated outputs to save: images, video, json, all, or none.",
    )
    parser.add_argument("--video-fps", type=float, default=DEFAULT_VIDEO_FPS)
    parser.add_argument("--video-width", type=int, default=None)
    parser.add_argument("--video-height", type=int, default=None)
    parser.add_argument("--video-codec", default=DEFAULT_VIDEO_CODEC)
    parser.add_argument("--video-writer", choices=("ffmpeg", "opencv"), default=DEFAULT_VIDEO_WRITER)
    parser.add_argument("--upload-image-mode", choices=("resize", "raw"), default=DEFAULT_UPLOAD_IMAGE_MODE)
    parser.add_argument("--upload-width", type=int, default=DEFAULT_UPLOAD_IMAGE_SIZE[0])
    parser.add_argument("--upload-height", type=int, default=DEFAULT_UPLOAD_IMAGE_SIZE[1])
    parser.add_argument("--history-limit", type=int, default=DEFAULT_HISTORY_LIMIT)
    parser.add_argument("--upload-max-memory-images", type=int, default=DEFAULT_MAX_MEMORY_IMAGES)
    parser.add_argument("--upload-memory-pool-window-frames", type=int, default=DEFAULT_MEMORY_POOL_WINDOW_FRAMES)
    parser.add_argument("--max-replans", type=int, default=50, help="0 means unlimited.")
    parser.add_argument(
        "--execution-mode", choices=("atomic", "continuous"), default=DEFAULT_EXECUTION_MODE,
        help="atomic: stop and settle after every action; continuous: merge repeats and capture each atom boundary while moving.",
    )
    parser.add_argument(
        "--actions-per-replan",
        type=parse_actions_per_replan,
        default=DEFAULT_ACTIONS_PER_REPLAN,
        help="Original actions per plan: positive integer, 0 for model sequence length, or uncertainty for adaptive execution.",
    )
    parser.add_argument(
        "--replan-action-range", type=int, nargs=2, metavar=("MIN", "MAX"), default=DEFAULT_REPLAN_ACTION_RANGE,
        help="Inclusive atom-action count range for uncertainty mode (default: 4 8).",
    )
    parser.add_argument("--uncertainty-budget", type=float, default=DEFAULT_UNCERTAINTY_BUDGET)
    parser.add_argument(
        "--stop-commit-max-actions", type=int, default=DEFAULT_STOP_COMMIT_MAX_ACTIONS,
        help="In uncertainty mode, commit through STOP within this many actions, overriding budget/range; 0 disables.",
    )
    parser.add_argument(
        "--prefetch-after-actions",
        type=int,
        default=DEFAULT_PREFETCH_AFTER_ACTIONS,
        help="Start the next prediction after this many atoms, at a group boundary; 0 disables prefetch.",
    )
    parser.add_argument("--forward-distance", type=float, default=0.25)
    parser.add_argument("--forward-speed", type=float, default=0.35)
    parser.add_argument("--turn-degrees", type=float, default=15.0)
    parser.add_argument("--yaw-speed", type=float, default=0.80)
    parser.add_argument("--command-period", type=float, default=DEFAULT_COMMAND_PERIOD_S)
    parser.add_argument("--settle-time", type=float, default=DEFAULT_SETTLE_TIME_S)
    parser.add_argument(
        "--post-capture-settle-time",
        type=float,
        default=DEFAULT_POST_CAPTURE_SETTLE_TIME_S,
    )
    parser.add_argument(
        "--disable-odom-control",
        action="store_true",
        help="Disable /sportmodestate closed-loop control and use timed open-loop primitives.",
    )
    parser.add_argument(
        "--enable-odom-control", dest="disable_odom_control", action="store_false", default=argparse.SUPPRESS,
        help="Enable odometry control, overriding disable_odom_control in YAML.",
    )
    parser.add_argument("--odom-timeout", type=float, default=1.0)
    parser.add_argument("--forward-tolerance", type=float, default=0.04)
    parser.add_argument("--turn-tolerance-degrees", type=float, default=3.0)
    parser.add_argument("--forward-kp", type=float, default=3.0)
    parser.add_argument("--forward-kd", type=float, default=0.0)
    parser.add_argument("--turn-kp", type=float, default=3.0)
    parser.add_argument("--turn-kd", type=float, default=0.0)
    parser.add_argument("--min-forward-speed", type=float, default=0.15)
    parser.add_argument("--min-yaw-speed", type=float, default=0.35)
    argv = list(sys.argv[1:] if argv is None else argv)
    config_parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    config_parser.add_argument("--config")
    config_path = config_parser.parse_known_args(argv)[0].config
    # Help must remain available even if a configuration file is missing/broken.
    if config_path and not any(arg in {"-h", "--help"} for arg in argv):
        try:
            defaults = config_arguments(parser, config_path)
        except ValueError as exc:
            parser.error(str(exc))
        argv = defaults + argv
    return parser.parse_args(argv)


def parse_camera_source(raw: str) -> str | int:
    try:
        return int(raw)
    except ValueError:
        return raw


def save_navigation_frame(
    save_dir: Optional[Path],
    image: bytes,
    *,
    frame_index: int,
    label: str,
) -> Optional[Path]:
    if save_dir is None:
        return None
    save_dir.mkdir(parents=True, exist_ok=True)
    safe_label = re.sub(r"[^0-9A-Za-z_.-]+", "_", label).strip("_") or "frame"
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    path = save_dir / f"{frame_index:06d}_{safe_label}_{timestamp}.jpg"
    path.write_bytes(image)
    return path


def main() -> None:
    args = parse_args()
    if args.print_config:
        print(json.dumps(vars(args), ensure_ascii=False, indent=2))
        return
    instruction = read_instruction(args)
    if not instruction:
        raise ValueError("Instruction is empty")
    if args.control_backend != "dry-run" and args.real_robot_ack != "yes":
        raise ValueError('Set robot.real_robot_ack: "yes" in the YAML (or --real-robot-ack yes) before using ros2')
    args.uncertainty_budget = validate_uncertainty_budget(args.uncertainty_budget)
    args.replan_action_range = validate_replan_action_range(args.replan_action_range)
    args.stop_commit_max_actions = validate_stop_commit_max_actions(args.stop_commit_max_actions)
    if args.prefetch_after_actions < 0:
        raise ValueError("prefetch-after-actions must be nonnegative")
    if args.history_limit <= 0 or args.upload_memory_pool_window_frames <= 0 or args.upload_max_memory_images < 0:
        raise ValueError("History/window sizes must be positive and max memory images nonnegative")
    requested_save_contents = parse_save_contents(args.save_contents)
    save_output_dir = Path(args.save_output_dir).expanduser() if args.save_output_dir else None
    if requested_save_contents and save_output_dir is None:
        raise ValueError("Set --save-output-dir when --save-contents is not none")
    run_id = time.strftime("%Y%m%d-%H%M%S")
    if save_output_dir is not None:
        save_output_dir.mkdir(parents=True, exist_ok=True)
        save_image_dir = save_output_dir if "images" in requested_save_contents else None
        save_video_path = save_output_dir if "video" in requested_save_contents else None
        navigation_json_path = (
            save_output_dir / f"navigation_{run_id}.json"
            if "json" in requested_save_contents
            else None
        )
    else:
        save_image_dir = None
        save_video_path = None
        navigation_json_path = None

    motion = MotionConfig(
        forward_distance_m=args.forward_distance,
        forward_speed_mps=args.forward_speed,
        turn_degrees=args.turn_degrees,
        yaw_speed_radps=args.yaw_speed,
        command_period_s=args.command_period,
        settle_time_s=args.settle_time,
        post_capture_settle_time_s=args.post_capture_settle_time,
        odom_control=not args.disable_odom_control,
        odom_timeout_s=args.odom_timeout,
        forward_tolerance_m=args.forward_tolerance,
        turn_tolerance_degrees=args.turn_tolerance_degrees,
        forward_kp=args.forward_kp,
        forward_kd=args.forward_kd,
        turn_kp=args.turn_kp,
        turn_kd=args.turn_kd,
        min_forward_speed_mps=args.min_forward_speed,
        min_yaw_speed_radps=args.min_yaw_speed,
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
    client = VLNHttpClient(
        args.server_base_url, timeout_s=args.request_timeout,
        uncertainty_max_actions=args.replan_action_range[1] if args.actions_per_replan == "uncertainty" else None,
    )
    history: deque[bytes] = deque(maxlen=args.history_limit)
    video_recorder = (
        NavigationVideoRecorder(
            camera,
            save_video_path,
            fps=args.video_fps,
            codec=args.video_codec,
            writer_backend=args.video_writer,
            width=args.video_width,
            height=args.video_height,
        )
        if save_video_path is not None
        else None
    )
    run_log: dict[str, object] = {
        "run_id": run_id,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "finished_at": None,
        "status": "running",
        "instruction": instruction,
        "server_base_url": args.server_base_url,
        "control_backend": args.control_backend,
        "real_robot_ack": args.real_robot_ack,
        "camera": {
            "source": args.camera,
            "frame_width": args.frame_width,
            "frame_height": args.frame_height,
            "camera_fps": args.camera_fps,
            "jpeg_quality": args.jpeg_quality,
            "fourcc": args.camera_fourcc,
        },
        "output": {
            "output_dir": str(save_output_dir) if save_output_dir is not None else None,
            "save_contents": sorted(requested_save_contents) if save_output_dir is not None else [],
            "image_dir": str(save_image_dir) if save_image_dir is not None else None,
            "video_path": str(video_recorder.output_path) if video_recorder is not None else None,
            "json_path": str(navigation_json_path) if navigation_json_path is not None else None,
        },
        "motion": asdict(motion),
        "upload": {
            "mode": args.upload_image_mode,
            "width": args.upload_width,
            "height": args.upload_height,
            "max_memory_images": args.upload_max_memory_images,
            "memory_pool_window_frames": args.upload_memory_pool_window_frames,
        },
        "configured_actions_per_replan": args.actions_per_replan,
        "execution_mode": args.execution_mode,
        "uncertainty_budget": args.uncertainty_budget,
        "replan_action_range": args.replan_action_range,
        "stop_commit_max_actions": args.stop_commit_max_actions,
        "effective_actions_per_replan": None,
        "prefetch_after_actions": args.prefetch_after_actions,
        "server_ready": None,
        "history_limit": args.history_limit,
        "predictions": [],
        "executed_actions": [],
        "saved_frames": [],
        "video_summary": None,
    }
    frame_index = 0

    try:
        ready = client.ready()
        run_log["server_ready"] = ready
        print({"server_ready": ready}, flush=True)
        if args.actions_per_replan == "uncertainty":
            if ready.get("supports_action_uncertainty") is not True:
                raise ValueError("Model server lacks action uncertainty support; update and restart realworld.server")
            fixed_actions_per_replan = None
        elif args.actions_per_replan > 0:
            fixed_actions_per_replan = args.actions_per_replan
        else:
            fixed_actions_per_replan = ready.get("action_sequence_length")
            if (isinstance(fixed_actions_per_replan, bool)
                    or not isinstance(fixed_actions_per_replan, int) or fixed_actions_per_replan <= 0):
                raise ValueError("Server /ready must return a positive integer action_sequence_length when --actions-per-replan=0")
        if video_recorder is not None:
            video_recorder.start()
        first_frame = camera.read_jpeg(flush_frames=args.capture_flush_frames)
        history.append(first_frame)
        saved_path = save_navigation_frame(
            save_image_dir,
            first_frame,
            frame_index=frame_index,
            label="initial",
        )
        if saved_path is not None:
            print({"saved_frame": str(saved_path), "frame_index": frame_index}, flush=True)
            run_log["saved_frames"].append(
                {
                    "frame_index": frame_index,
                    "path": str(saved_path),
                    "label": "initial",
                }
            )
        frame_index += 1
        del first_frame
        backend.stand()
        replans = 0
        pending_prediction: Optional[PendingPrediction] = None
        while args.max_replans <= 0 or replans < args.max_replans:
            if pending_prediction is not None:
                prefetch_wait_started = time.perf_counter()
                prediction = pending_prediction.result()
                prefetch_wait_s: Optional[float] = time.perf_counter() - prefetch_wait_started
                prediction_source = "prefetch"
                pending_prediction = None
            else:
                prediction = request_prediction(
                    client,
                    instruction=instruction,
                    history_snapshot=list(history),
                    max_memory_images=args.upload_max_memory_images,
                    memory_pool_window_frames=args.upload_memory_pool_window_frames,
                    upload_image_mode=args.upload_image_mode,
                    jpeg_quality=args.jpeg_quality,
                    upload_size=(args.upload_width, args.upload_height),
                )
                prefetch_wait_s = None
                prediction_source = "sync"
            replans += 1

            actions = prediction.actions
            effective_actions_per_replan = fixed_actions_per_replan
            stop_committed = False
            if args.actions_per_replan == "uncertainty":
                effective_actions_per_replan, stop_committed = select_prediction_horizon(
                    prediction, uncertainty_budget=args.uncertainty_budget,
                    replan_action_range=args.replan_action_range,
                    stop_commit_max_actions=args.stop_commit_max_actions,
                )
            run_log["effective_actions_per_replan"] = effective_actions_per_replan
            plan_log = {
                **asdict(prediction), "replan": replans,
                "effective_actions_per_replan": effective_actions_per_replan,
                "stop_committed": stop_committed,
                "prediction_source": prediction_source, "prefetch_wait_s": prefetch_wait_s,
            }
            print(plan_log, flush=True)
            run_log["predictions"].append(plan_log)

            should_stop = False
            completed_actions = 0
            action_groups = (
                merge_consecutive_actions(actions, effective_actions_per_replan)
                if args.execution_mode == "continuous"
                else [(action, 1) for action in actions[:effective_actions_per_replan]]
            )
            for action, repeat_count in action_groups:
                action_index = completed_actions
                completed_actions += repeat_count

                def capture_atom_frame(atom_index: int) -> None:
                    nonlocal frame_index
                    # Called at each intermediate odometry boundary without stopping
                    # or settling, then once more after the group's final stop/settle.
                    captured_frame = camera.read_jpeg(flush_frames=args.capture_flush_frames)
                    history.append(captured_frame)
                    label = f"after_replan_{replans:03d}_action_{action_index + atom_index:02d}_{action}"
                    saved_path = save_navigation_frame(
                        save_image_dir, captured_frame, frame_index=frame_index, label=label,
                    )
                    if saved_path is not None:
                        print({"saved_frame": str(saved_path), "frame_index": frame_index}, flush=True)
                        run_log["saved_frames"].append(
                            {"frame_index": frame_index, "path": str(saved_path), "label": label}
                        )
                    frame_index += 1

                print(
                    {
                        "execute_action": action,
                        "execution_mode": args.execution_mode,
                        "replan": replans,
                        "action_index": action_index + 1,
                        "action_count": repeat_count,
                        "action_end_index": completed_actions,
                    },
                    flush=True,
                )
                action_started_s = time.perf_counter()
                should_stop = execute_action(
                    backend, action, motion, repeat_count=repeat_count, on_atom_complete=capture_atom_frame,
                )
                run_log["executed_actions"].append(
                    {
                        "replan": replans,
                        "action_index": action_index + 1,
                        "action_count": repeat_count,
                        "action_end_index": completed_actions,
                        "action": action,
                        "duration_s": time.perf_counter() - action_started_s,
                        "stopped": should_stop,
                    }
                )
                if should_stop:
                    break
                time.sleep(max(0.0, motion.settle_time_s))
                capture_atom_frame(repeat_count)
                can_prefetch_next_replan = args.max_replans <= 0 or replans < args.max_replans
                if (
                    args.prefetch_after_actions > 0
                    and completed_actions >= args.prefetch_after_actions
                    and pending_prediction is None
                    and can_prefetch_next_replan
                ):
                    pending_prediction = PendingPrediction(
                        target=request_prediction,
                        kwargs={
                            "client": client,
                            "instruction": instruction,
                            "history_snapshot": list(history),
                            "max_memory_images": args.upload_max_memory_images,
                            "memory_pool_window_frames": args.upload_memory_pool_window_frames,
                            "upload_image_mode": args.upload_image_mode,
                            "jpeg_quality": args.jpeg_quality,
                            "upload_size": (args.upload_width, args.upload_height),
                        },
                    )
                    print(
                        {
                            "prediction_prefetch": "submitted",
                            "after_replan": replans,
                            "after_action_index": completed_actions,
                            "history_frames": len(history),
                        },
                        flush=True,
                    )
                time.sleep(max(0.0, motion.post_capture_settle_time_s))
            if should_stop:
                break
    except KeyboardInterrupt:
        print("Interrupted, stopping robot.", flush=True)
        run_log["status"] = "interrupted"
    except Exception:
        run_log["status"] = "failed"
        raise
    finally:
        if run_log["status"] == "running":
            run_log["status"] = "finished"
        run_log["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
        try:
            backend.close()
        finally:
            try:
                if video_recorder is not None:
                    run_log["video_summary"] = video_recorder.close()
            finally:
                try:
                    if navigation_json_path is not None:
                        write_navigation_json(navigation_json_path, run_log)
                        print({"navigation_json": str(navigation_json_path)}, flush=True)
                finally:
                    camera.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"fatal: {exc}", file=sys.stderr, flush=True)
        raise
