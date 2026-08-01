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
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence

import requests


DEFAULT_SERVER_BASE_URL = "http://10.14.114.132:8000"
DEFAULT_PREDICT_PATH = "/predict"
DEFAULT_READY_PATH = "/ready"
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
DEFAULT_MAX_MEMORY_IMAGES = 10
DEFAULT_MEMORY_POOL_WINDOW_FRAMES = 100
DEFAULT_HISTORY_LIMIT = DEFAULT_MEMORY_POOL_WINDOW_FRAMES + DEFAULT_MAX_MEMORY_IMAGES + 10
DEFAULT_UPLOAD_IMAGE_MODE = "resize"
DEFAULT_UPLOAD_IMAGE_SIZE = (1280, 640)
DEFAULT_ACTIONS_PER_REPLAN = 4
DEFAULT_PREFETCH_AFTER_ACTIONS = 2
DEFAULT_COMMAND_PERIOD_S = 0.05
DEFAULT_SETTLE_TIME_S = 0.25
DEFAULT_POST_CAPTURE_SETTLE_TIME_S = 0.25
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
    settle_time_s: float = 0.25
    post_capture_settle_time_s: float = 0.25
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


@dataclass
class PredictionResult:
    actions: list[str]
    raw_text: str
    request_images: int
    upload_image_mode: str
    upload_mb: float
    round_trip_s: float
    server_latency_s: object


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


def selected_history_with_action_spans(
    history: Sequence[bytes],
    action_history: Sequence[str],
    *,
    max_memory_images: int,
    memory_pool_window_frames: int,
) -> tuple[list[bytes], list[list[str]]]:
    if len(action_history) != max(0, len(history) - 1):
        raise ValueError(
            "Action history is not aligned with captured frames: "
            f"actions={len(action_history)}, frames={len(history)}"
        )
    if not history:
        return [], []
    indices = build_vln_image_selection(
        current_step=len(history) - 1,
        last_frame_index=len(history) - 1,
        max_memory_images=max_memory_images,
        memory_pool_window_frames=memory_pool_window_frames,
    )
    selected_images = [history[index] for index in indices]
    action_spans = [
        list(action_history[start_index:end_index])
        for start_index, end_index in zip(indices, indices[1:])
    ]
    return selected_images, action_spans


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

    def execute_motion_action(self, action: str, motion: MotionConfig) -> Optional[bool]:
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

    def _run_closed_loop_turn(self, direction: float, motion: MotionConfig) -> bool:
        start = self._latest_odom(timeout_s=motion.odom_timeout_s)
        if start is None:
            print({"warning": "no_sportmodestate_for_closed_loop_turn; using open_loop"}, flush=True)
            return False

        target_yaw = wrap_angle_rad(start.yaw + float(direction) * math.radians(motion.turn_degrees))
        tolerance = math.radians(motion.turn_tolerance_degrees)
        deadline = time.monotonic() + MAX_ACTION_DURATION_S
        last_error = signed_angle_error_rad(target_yaw, start.yaw)
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
            error = signed_angle_error_rad(target_yaw, odom.yaw)
            last_error = error
            samples += 1
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

    def _run_closed_loop_forward(self, motion: MotionConfig) -> bool:
        start = self._latest_odom(timeout_s=motion.odom_timeout_s)
        if start is None:
            print({"warning": "no_sportmodestate_for_closed_loop_forward; using open_loop"}, flush=True)
            return False

        target_distance = motion.forward_distance_m
        deadline = time.monotonic() + MAX_ACTION_DURATION_S
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

    def execute_motion_action(self, action: str, motion: MotionConfig) -> Optional[bool]:
        if not motion.odom_control or self.odom_subscription is None:
            return None
        if action == "forward":
            if self._run_closed_loop_forward(motion):
                return False
            return None
        if action == "left":
            if self._run_closed_loop_turn(1.0, motion):
                return False
            return None
        if action == "right":
            if self._run_closed_loop_turn(-1.0, motion):
                return False
            return None
        return None

    def close(self) -> None:
        self.stop()
        self.node.destroy_node()
        self.rclpy.shutdown()


class VLNHttpClient:
    def __init__(self, server_base_url: str, timeout_s: float, predict_path: str = DEFAULT_PREDICT_PATH):
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
        interframe_actions: Sequence[Sequence[str]],
    ) -> dict:
        files = [
            ("images", (f"frame_{index:03d}.jpg", image, "image/jpeg"))
            for index, image in enumerate(images)
        ]
        data = {
            "instruction": instruction,
            "interframe_actions": json.dumps(list(interframe_actions)),
        }
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
    action_history_snapshot: Sequence[str],
    max_memory_images: int,
    memory_pool_window_frames: int,
    upload_image_mode: str,
    jpeg_quality: int,
    upload_size: tuple[int, int],
) -> PredictionResult:
    selected_images, interframe_actions = selected_history_with_action_spans(
        history=history_snapshot,
        action_history=action_history_snapshot,
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
        interframe_actions=interframe_actions,
    )
    return PredictionResult(
        actions=normalize_actions(result.get("executable_actions") or result.get("actions") or []),
        raw_text=result.get("raw_text", ""),
        request_images=len(request_images),
        upload_image_mode=upload_image_mode,
        upload_mb=round(request_bytes / (1024 * 1024), 3),
        round_trip_s=time.perf_counter() - started,
        server_latency_s=result.get("latency_s"),
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
    backend_result = backend.execute_motion_action(action, motion)
    if backend_result is not None:
        return backend_result
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
    if args.control_backend == "ros2":
        return Ros2SportBackend()
    raise ValueError(f"Unsupported backend: {args.control_backend}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PanoVLN on a Unitree Go2.")
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
    parser.add_argument("--history-limit", type=int, default=DEFAULT_HISTORY_LIMIT)
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
    parser.add_argument("--upload-max-memory-images", type=int, default=DEFAULT_MAX_MEMORY_IMAGES)
    parser.add_argument("--upload-memory-pool-window-frames", type=int, default=DEFAULT_MEMORY_POOL_WINDOW_FRAMES)
    parser.add_argument("--max-replans", type=int, default=50, help="0 means unlimited.")
    parser.add_argument("--actions-per-replan", type=int, default=DEFAULT_ACTIONS_PER_REPLAN)
    parser.add_argument(
        "--prefetch-after-actions",
        type=int,
        default=DEFAULT_PREFETCH_AFTER_ACTIONS,
        help="Start the next server prediction after this many non-stop actions; 0 disables prefetch.",
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
    parser.add_argument("--odom-timeout", type=float, default=1.0)
    parser.add_argument("--forward-tolerance", type=float, default=0.04)
    parser.add_argument("--turn-tolerance-degrees", type=float, default=3.0)
    parser.add_argument("--forward-kp", type=float, default=3.0)
    parser.add_argument("--forward-kd", type=float, default=0.0)
    parser.add_argument("--turn-kp", type=float, default=3.0)
    parser.add_argument("--turn-kd", type=float, default=0.0)
    parser.add_argument("--min-forward-speed", type=float, default=0.15)
    parser.add_argument("--min-yaw-speed", type=float, default=0.35)
    return parser.parse_args()


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
    instruction = read_instruction(args)
    if not instruction:
        raise ValueError("Instruction is empty")
    if args.control_backend != "dry-run" and args.real_robot_ack != "yes":
        raise ValueError("Set REAL_ROBOT_ACK=\"yes\" in run_go2_client.sh before using ros2")
    if args.actions_per_replan <= 0:
        raise ValueError(f"actions_per_replan must be > 0, got {args.actions_per_replan}")
    if args.prefetch_after_actions < 0:
        raise ValueError(f"prefetch_after_actions must be >= 0, got {args.prefetch_after_actions}")
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
    client = VLNHttpClient(args.server_base_url, timeout_s=args.request_timeout)
    history: deque[bytes] = deque(maxlen=max(1, args.history_limit))
    action_history: deque[str] = deque(maxlen=max(0, args.history_limit - 1))
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
        "actions_per_replan": args.actions_per_replan,
        "prefetch_after_actions": args.prefetch_after_actions,
        "server_ready": None,
        "predictions": [],
        "executed_actions": [],
        "saved_frames": [],
        "video_summary": None,
    }
    frame_index = 0

    try:
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
        ready = client.ready()
        print({"server_ready": ready}, flush=True)
        run_log["server_ready"] = ready
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
                    action_history_snapshot=list(action_history),
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
            print(
                {
                    "replan": replans,
                    "actions": actions,
                    "raw_text": prediction.raw_text,
                    "prediction_source": prediction_source,
                    "prefetch_wait_s": prefetch_wait_s,
                    "request_images": prediction.request_images,
                    "upload_image_mode": prediction.upload_image_mode,
                    "upload_mb": prediction.upload_mb,
                    "round_trip_s": prediction.round_trip_s,
                    "server_latency_s": prediction.server_latency_s,
                },
                flush=True,
            )
            run_log["predictions"].append(
                {
                    "replan": replans,
                    "actions": actions,
                    "raw_text": prediction.raw_text,
                    "prediction_source": prediction_source,
                    "prefetch_wait_s": prefetch_wait_s,
                    "request_images": prediction.request_images,
                    "upload_image_mode": prediction.upload_image_mode,
                    "upload_mb": prediction.upload_mb,
                    "round_trip_s": prediction.round_trip_s,
                    "server_latency_s": prediction.server_latency_s,
                }
            )

            should_stop = False
            for action_index, action in enumerate(actions):
                if action_index >= args.actions_per_replan:
                    break
                print(
                    {
                        "execute_action": action,
                        "replan": replans,
                        "action_index": action_index + 1,
                    },
                    flush=True,
                )
                action_started_s = time.perf_counter()
                should_stop = execute_action(backend, action, motion)
                run_log["executed_actions"].append(
                    {
                        "replan": replans,
                        "action_index": action_index + 1,
                        "action": action,
                        "duration_s": time.perf_counter() - action_started_s,
                        "stopped": should_stop,
                    }
                )
                if should_stop:
                    break
                time.sleep(max(0.0, motion.settle_time_s))
                captured_frame = camera.read_jpeg(flush_frames=args.capture_flush_frames)
                action_history.append(action)
                history.append(captured_frame)
                saved_path = save_navigation_frame(
                    save_image_dir,
                    captured_frame,
                    frame_index=frame_index,
                    label=f"after_replan_{replans:03d}_action_{action_index + 1:02d}_{action}",
                )
                if saved_path is not None:
                    print({"saved_frame": str(saved_path), "frame_index": frame_index}, flush=True)
                    run_log["saved_frames"].append(
                        {
                            "frame_index": frame_index,
                            "path": str(saved_path),
                            "label": f"after_replan_{replans:03d}_action_{action_index + 1:02d}_{action}",
                        }
                    )
                frame_index += 1
                completed_actions = action_index + 1
                can_prefetch_next_replan = args.max_replans <= 0 or replans < args.max_replans
                if (
                    args.prefetch_after_actions > 0
                    and completed_actions == args.prefetch_after_actions
                    and pending_prediction is None
                    and can_prefetch_next_replan
                ):
                    pending_prediction = PendingPrediction(
                        target=request_prediction,
                        kwargs={
                            "client": client,
                            "instruction": instruction,
                            "history_snapshot": list(history),
                            "action_history_snapshot": list(action_history),
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
