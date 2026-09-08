import argparse
import atexit
import copy
import ctypes
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import random
import re
import shutil
import signal
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.data.habitat_shortest_path import (
    ERP_IMAGE_SIZE,
    MOVE_FORWARD_ACTION,
    STOP_ACTION,
    TURN_LEFT_ACTION,
    TURN_RIGHT_ACTION,
    CONFIG as DATASET_CONFIG,
    action_to_int,
    build_locality_balanced_episode_splits,
    build_worker_assignments,
    euclidean_distance,
    extract_instruction,
    filter_episodes,
    frame_image_filename,
    get_goal_position,
    get_reference_positions,
    habitat,
    load_dataset,
    normalize_image_format,
    parse_episode_ids,
    parse_gpu_ids,
    positions_equal,
    remap_gpu_ids_to_visible_devices,
    reset_episode_output_dir,
    save_rgb_frame,
    silence_external_output,
    sort_episodes_for_scene_locality,
    to_position_list,
    validate_gpu_ids,
)
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

from src.eval.eval import (
    DEFAULT_MAX_MEMORY_IMAGES,
    DEFAULT_MEMORY_POOL_WINDOW_FRAMES,
    DEFAULT_REPLAN_ACTION_RANGE,
    PanoVLN_Agent,
    select_stop_commit_horizon,
    select_uncertainty_horizon,
)
from src.data.prepare_training_data import (
    parse_dagger_oracle_chunks,
    validate_dagger_execution_policy,
)
from src.data.dagger_replay import (
    capture_frame_state,
    capture_replay_config,
    portable_scene_id,
    validate_replay_metadata,
)


DEFAULT_ALPHA = 0.5
DEFAULT_DAGGER_MIDGOAL_RADIUS = 1.8
DEFAULT_DAGGER_GOAL_RADIUS = 0.3
DEFAULT_MODEL_PATH = "/workspace/data2/model/18-action/panovggt_base_18action"
DEFAULT_OUTPUT_ROOT = "/workspace/data2/dataset/PanoVLN"
DEFAULT_DAGGER_DATASET_NAME = "dagger"
DEFAULT_ACTION_HORIZON = 18
DEFAULT_UNCERTAINTY_BUDGET = 1.2
DEFAULT_STOP_ORACLE_MAX_ACTIONS = 12
DEFAULT_CPU_THREADS_PER_WORKER = 2
DAGGER_CPU_THREADS_ENV = "VLN_DAGGER_CPU_THREADS_PER_WORKER"
CPU_THREAD_ENV_VARS = (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
)
DEFAULT_SOURCE_DATASETS = ("r2r", "rxr")
QUEUE_POLL_TIMEOUT_SECONDS = 5
SUPPORTED_ACTION_IDS = {
    STOP_ACTION,
    MOVE_FORWARD_ACTION,
    TURN_LEFT_ACTION,
    TURN_RIGHT_ACTION,
}
class DAggerCollectionError(RuntimeError):
    pass


class RecoverableEpisodeNavigationError(DAggerCollectionError):
    """An episode-local follower failure that must not become an oracle label."""


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def configure_worker_thread_environment(cpu_threads_per_worker: int) -> None:
    cpu_threads_per_worker = int(cpu_threads_per_worker)
    if cpu_threads_per_worker <= 0:
        raise ValueError(
            "cpu_threads_per_worker must be positive, "
            f"got {cpu_threads_per_worker}"
        )
    thread_count = str(cpu_threads_per_worker)
    os.environ[DAGGER_CPU_THREADS_ENV] = thread_count
    for variable_name in CPU_THREAD_ENV_VARS:
        os.environ[variable_name] = thread_count
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def configure_worker_torch_threads() -> None:
    cpu_threads_per_worker = int(
        os.environ.get(
            DAGGER_CPU_THREADS_ENV,
            str(DEFAULT_CPU_THREADS_PER_WORKER),
        )
    )
    torch.set_num_threads(cpu_threads_per_worker)
    torch.set_num_interop_threads(1)


def arm_linux_parent_death_signal(expected_parent_pid: int) -> None:
    """Terminate a spawned collector worker if its owning process disappears."""

    if not sys.platform.startswith("linux"):
        return
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        errno_value = ctypes.get_errno()
        raise OSError(errno_value, os.strerror(errno_value))
    # The parent may have died between spawning this worker and arming prctl.
    if os.getppid() != int(expected_parent_pid):
        raise RuntimeError(
            f"DAgger collector parent changed before worker startup: "
            f"expected={expected_parent_pid}, actual={os.getppid()}"
        )


def str2bool(value):
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "t", "yes", "y"}:
        return True
    if normalized in {"0", "false", "f", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def dynamic_oracle_probability(step_index: int, reference_steps: int, alpha: float) -> float:
    reference_steps = max(1, int(reference_steps))
    step_index = max(0, int(step_index))
    alpha = float(alpha)
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    return min(1.0, max(0.0, 1.0 - alpha ** (step_index / reference_steps)))


def annotation_path(output_root: str, dagger_dataset_name: str) -> str:
    return os.path.join(output_root, "sub_dataset", f"{dagger_dataset_name}.jsonl")


def image_root(output_root: str, dagger_dataset_name: str) -> str:
    return os.path.join(output_root, "images", dagger_dataset_name)


def annotation_episode_id(annotation: Dict) -> str:
    episode_id = annotation.get("episode_id")
    if not isinstance(episode_id, str) or not episode_id:
        raise ValueError("DAgger annotation has no string episode_id")
    return episode_id


def annotation_image_id(annotation: Dict) -> str:
    episode_id = annotation_episode_id(annotation)
    trajectory_id = annotation.get("trajectory_id")
    if not isinstance(trajectory_id, str) or not trajectory_id:
        raise ValueError("DAgger annotation has no trajectory_id")
    if trajectory_id != episode_id:
        raise ValueError(
            "DAgger annotation episode_id and trajectory_id must match: "
            f"episode_id={episode_id!r}, trajectory_id={trajectory_id!r}"
        )
    return trajectory_id


def progress_dir_path(output_root: str, dagger_dataset_name: str) -> str:
    output_path = Path(annotation_path(output_root, dagger_dataset_name))
    return str(output_path.parent / f"{output_path.stem}_progress")


def rank_annotation_path(progress_dir: str, worker_index: int) -> str:
    return os.path.join(progress_dir, f"rank_{worker_index:03d}.jsonl")


def list_rank_annotation_paths(progress_dir: str) -> List[str]:
    if not os.path.isdir(progress_dir):
        return []
    return sorted(
        os.path.join(progress_dir, file_name)
        for file_name in os.listdir(progress_dir)
        if re.fullmatch(r"rank_\d+\.jsonl", file_name)
    )


def remove_if_exists(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def load_jsonl_index(path: str, tolerate_trailing_corrupt: bool = True) -> Dict[str, Dict]:
    index = {}
    if not os.path.exists(path):
        return index
    with open(path, "r", encoding="utf-8") as handle:
        lines = handle.readlines()
    non_empty_line_numbers = [
        line_number
        for line_number, line in enumerate(lines, start=1)
        if line.strip()
    ]
    last_non_empty_line = non_empty_line_numbers[-1] if non_empty_line_numbers else -1
    with open(path, "r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                if tolerate_trailing_corrupt and line_number == last_non_empty_line:
                    tqdm.write(
                        f"[dagger] ignoring trailing corrupt JSONL line "
                        f"{line_number} in {path}"
                    )
                    break
                raise
            episode_id = annotation_episode_id(item)
            if episode_id in index:
                raise ValueError(
                    f"Duplicate DAgger episode_id={episode_id!r} in {path}"
                )
            index[episode_id] = item
    return index


def write_jsonl_index(path: str, index: Dict[str, Dict]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as handle:
        for episode_id in sorted(index):
            handle.write(json.dumps(index[episode_id], ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp_path, path)


def write_jsonl_item(handle, item: Dict) -> None:
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def jsonl_ends_with_item(path: str, item: Dict) -> bool:
    """Return whether the last non-empty JSONL row is fully persisted as item."""

    try:
        with open(path, "rb") as handle:
            lines = handle.read().splitlines()
        last_line = next(line for line in reversed(lines) if line.strip())
        return json.loads(last_line) == item
    except (OSError, StopIteration, UnicodeDecodeError, json.JSONDecodeError):
        return False


def repair_jsonl_for_append(path: str) -> bool:
    """Make an interrupted JSONL journal safe to append to.

    A killed worker can leave a partial final JSON object.  Merely ignoring
    that object while reading is insufficient: the next append would be
    concatenated to the partial line.  Drop only a corrupt final non-empty
    line and ensure the last valid record is newline-terminated before the
    worker resumes writing.
    """

    if not os.path.exists(path):
        return False

    with open(path, "rb") as handle:
        raw_content = handle.read()
    if not raw_content:
        return False

    lines = raw_content.splitlines(keepends=True)
    non_empty_indices = [
        line_index
        for line_index, line in enumerate(lines)
        if line.strip()
    ]
    if not non_empty_indices:
        repaired_content = b""
    else:
        last_non_empty_index = non_empty_indices[-1]
        corrupt_final_line = False
        for line_index in non_empty_indices:
            try:
                json.loads(lines[line_index])
            except (json.JSONDecodeError, UnicodeDecodeError):
                if line_index != last_non_empty_index:
                    raise
                corrupt_final_line = True

        if corrupt_final_line:
            repaired_content = b"".join(lines[:last_non_empty_index])
        else:
            repaired_content = raw_content

        repaired_content = repaired_content.rstrip(b" \t\r\n")
        if repaired_content:
            repaired_content += b"\n"

    if repaired_content == raw_content:
        return False

    tmp_path = f"{path}.repair-{os.getpid()}.tmp"
    try:
        with open(tmp_path, "wb") as handle:
            handle.write(repaired_content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    return True


def count_saved_frames(episode_image_dir: str, image_format: str) -> int:
    if not os.path.isdir(episode_image_dir):
        return 0
    expected_suffix = Path(frame_image_filename(0, image_format)).suffix.lower()
    return sum(
        1
        for file_name in os.listdir(episode_image_dir)
        if re.fullmatch(r"frame_\d+\.[^.]+", file_name, flags=re.IGNORECASE)
        and Path(file_name).suffix.lower() == expected_suffix
    )


def expected_frame_path(
    episode_image_dir: str,
    frame_index: int,
    image_format: str,
) -> str:
    return os.path.join(
        episode_image_dir,
        frame_image_filename(frame_index, image_format),
    )


def is_raw_episode_complete(
    output_root: str,
    dagger_dataset_name: str,
    annotation: Dict,
    image_format: str,
    action_horizon: int,
    execution_policy: Dict,
) -> bool:
    if (
        isinstance(annotation.get("action_horizon"), bool)
        or not isinstance(annotation.get("action_horizon"), int)
        or annotation.get("action_horizon") != action_horizon
        or annotation.get("execution_policy") != execution_policy
    ):
        return False
    actions = annotation.get("actions")
    oracle_chunks = annotation.get("oracle_chunks")
    if (
        not isinstance(actions, list)
        or not actions
        or not isinstance(oracle_chunks, list)
        or not oracle_chunks
    ):
        return False
    try:
        actions = [int(action_id) for action_id in actions]
    except (TypeError, ValueError):
        return False
    if (
        actions[-1] != STOP_ACTION
        or STOP_ACTION in actions[:-1]
        or any(action_id not in SUPPORTED_ACTION_IDS for action_id in actions)
    ):
        return False

    try:
        parse_dagger_oracle_chunks(annotation, actions, "raw DAgger episode")
        validate_replay_metadata(annotation)
    except (TypeError, ValueError):
        return False

    try:
        episode_image_id = annotation_image_id(annotation)
    except ValueError:
        return False
    episode_image_dir = os.path.join(
        image_root(output_root, dagger_dataset_name), episode_image_id
    )
    action_count = len(actions)
    if action_count <= 0 or not os.path.isdir(episode_image_dir):
        return False
    expected_suffix = Path(frame_image_filename(0, image_format)).suffix.lower()
    all_frame_names = [
        file_name
        for file_name in os.listdir(episode_image_dir)
        if re.fullmatch(r"frame_\d+\.(?:jpg|jpeg|png)", file_name, flags=re.IGNORECASE)
    ]
    frame_names = [
        file_name
        for file_name in all_frame_names
        if Path(file_name).suffix.lower() == expected_suffix
    ]
    if len(frame_names) != len(all_frame_names):
        return False
    if len(frame_names) != action_count:
        return False
    for frame_index in range(action_count):
        frame_path = expected_frame_path(
            episode_image_dir,
            frame_index,
            image_format,
        )
        if not os.path.isfile(frame_path) or os.path.getsize(frame_path) <= 0:
            return False
    return True


def validate_safe_dataset_name(name: str, arg_name: str = "dagger_dataset_name") -> None:
    if name != DEFAULT_DAGGER_DATASET_NAME:
        raise ValueError(
            f"{arg_name} must be exactly {DEFAULT_DAGGER_DATASET_NAME!r}; "
            f"custom DAgger dataset names are not supported, got {name!r}"
        )


def assert_path_inside(path: str, parent: str) -> None:
    resolved_path = Path(path).resolve()
    resolved_parent = Path(parent).resolve()
    try:
        resolved_path.relative_to(resolved_parent)
    except ValueError as exc:
        raise ValueError(f"Refusing to operate on path outside {resolved_parent}: {resolved_path}") from exc


def assert_path_strictly_inside(path: str, parent: str) -> None:
    resolved_path = Path(path).resolve()
    resolved_parent = Path(parent).resolve()
    assert_path_inside(str(resolved_path), str(resolved_parent))
    if resolved_path == resolved_parent:
        raise ValueError(
            f"Refusing to operate on parent directory itself: {resolved_path}"
        )


def process_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def acquire_progress_lock(progress_dir: str) -> str:
    os.makedirs(progress_dir, exist_ok=True)
    lock_path = os.path.join(progress_dir, "dagger.lock")
    while True:
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(json.dumps({"pid": os.getpid(), "time": time.time()}) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            break
        except FileExistsError:
            try:
                with open(lock_path, "r", encoding="utf-8") as handle:
                    lock_info = json.loads(handle.readline() or "{}")
                lock_pid = int(lock_info.get("pid", -1))
            except Exception:
                lock_pid = -1
            if not process_is_alive(lock_pid):
                os.remove(lock_path)
                continue
            raise RuntimeError(
                f"Another DAgger collection appears to be running with lock {lock_path} "
                f"(pid={lock_pid}). Stop it or remove a stale lock before retrying."
            )

    def _release_lock() -> None:
        try:
            if os.path.exists(lock_path):
                os.remove(lock_path)
        except OSError:
            pass

    atexit.register(_release_lock)
    return lock_path


def assert_no_active_progress_lock(progress_dir: str) -> None:
    lock_path = os.path.join(progress_dir, "dagger.lock")
    if not os.path.exists(lock_path):
        return
    try:
        with open(lock_path, "r", encoding="utf-8") as handle:
            lock_info = json.loads(handle.readline() or "{}")
        lock_pid = int(lock_info.get("pid", -1))
    except Exception:
        lock_pid = -1
    if process_is_alive(lock_pid):
        raise RuntimeError(
            f"Refusing to clear progress directory with active DAgger lock "
            f"{lock_path} (pid={lock_pid})."
        )
    os.remove(lock_path)


def dagger_trajectory_id(source_dataset: str, source_episode_id) -> str:
    try:
        source_id = str(int(source_episode_id))
    except (TypeError, ValueError):
        source_id = hashlib.sha1(str(source_episode_id).encode("utf-8")).hexdigest()[:12]
    return f"{source_dataset}_{source_id}"


def configure_dagger_env(env_config, local_gpu_id: Optional[int], max_episode_steps: int) -> None:
    with habitat.config.read_write(env_config):
        if local_gpu_id is not None:
            env_config.habitat.simulator.habitat_sim_v0.gpu_device_id = int(local_gpu_id)
        if max_episode_steps > 0:
            env_config.habitat.environment.max_episode_steps = int(max_episode_steps)
        env_config.habitat.task.measurements = {}


def load_reference_action_lengths(
    reference_input_root: str,
    dataset_names: Sequence[str],
) -> Dict[Tuple[str, int], int]:
    lengths: Dict[Tuple[str, int], int] = {}
    for dataset_name in dataset_names:
        path = os.path.join(reference_input_root, "sub_dataset", f"{dataset_name}.jsonl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Missing reference annotation for Dynamic Ratio T: {path}"
            )
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                actions = [int(action) for action in item.get("actions", [])]
                if actions and actions[-1] == STOP_ACTION:
                    action_count = len(actions) - 1
                else:
                    action_count = len(actions)
                lengths[(dataset_name, int(item["episode_id"]))] = max(1, action_count)
    return lengths


def infer_reference_steps(episode, dataset_name: str, reference_lengths) -> int:
    key = (dataset_name, int(episode.episode_id))
    if key in reference_lengths:
        return reference_lengths[key]
    raise DAggerCollectionError(
        f"Missing reference action length for dataset={dataset_name} "
        f"episode_id={episode.episode_id}; Dynamic Ratio T must use "
        "ground-truth primitive action count excluding final stop."
    )


def validate_reference_coverage(
    source_dataset: str,
    episodes,
    reference_input_root: str,
) -> None:
    reference_lengths = load_reference_action_lengths(
        reference_input_root=reference_input_root,
        dataset_names=[source_dataset],
    )
    missing_episode_ids = [
        int(episode.episode_id)
        for episode in episodes
        if (source_dataset, int(episode.episode_id)) not in reference_lengths
    ]
    if missing_episode_ids:
        preview = missing_episode_ids[:20]
        raise ValueError(
            f"Reference annotation is missing {len(missing_episode_ids)} selected "
            f"{source_dataset} episodes for Dynamic Ratio T. First ids: {preview}"
        )


def reference_episode_ids(reference_lengths: Dict[Tuple[str, int], int], source_dataset: str):
    return {
        episode_id
        for dataset_name, episode_id in reference_lengths.keys()
        if dataset_name == source_dataset
    }


def validate_reference_ids_exist_in_habitat_dataset(
    source_dataset: str,
    dataset,
    allowed_episode_ids,
) -> None:
    habitat_episode_ids = {int(episode.episode_id) for episode in dataset.episodes}
    missing_episode_ids = sorted(set(allowed_episode_ids) - habitat_episode_ids)
    if missing_episode_ids:
        preview = missing_episode_ids[:20]
        raise ValueError(
            f"{source_dataset} reference jsonl contains {len(missing_episode_ids)} "
            f"episode ids that are missing from the Habitat dataset. First ids: {preview}"
        )


def initial_target_state(episode, current_position: Sequence[float]):
    target_positions = get_reference_positions(episode)
    if target_positions:
        next_target_index = 1 if positions_equal(target_positions[0], current_position) else 0
    else:
        target_positions = [get_goal_position(episode)]
        next_target_index = 0
    return target_positions, next_target_index


def target_reached_radius(
    target_positions,
    target_index: int,
    midgoal_radius: float,
    goal_radius: float,
) -> float:
    if target_index == len(target_positions) - 1:
        return float(goal_radius)
    return float(midgoal_radius)


def geodesic_distance_to_target(
    env,
    current_position: Sequence[float],
    target_position: Sequence[float],
) -> float:
    """Return Habitat's navigable distance between two positions.

    ``inf`` is a valid result for an unreachable target.  NaN and negative
    distances indicate a simulator failure and must not silently turn into an
    oracle STOP or a successful DAgger episode.
    """

    distance = float(
        env.sim.geodesic_distance(
            list(current_position),
            list(target_position),
        )
    )
    if math.isnan(distance) or distance < 0.0:
        raise DAggerCollectionError(
            "Habitat returned an invalid geodesic distance: "
            f"distance={distance}, start={list(current_position)}, "
            f"target={list(target_position)}"
        )
    return distance


def target_within_geodesic_radius(
    env,
    current_position: Sequence[float],
    target_position: Sequence[float],
    radius: float,
) -> bool:
    return geodesic_distance_to_target(
        env,
        current_position,
        target_position,
    ) <= float(radius)


def advance_target_index(
    env,
    target_positions,
    next_target_index: int,
    midgoal_radius: float,
    goal_radius: float,
) -> int:
    current_position = to_position_list(env.sim.get_agent_state().position)
    while next_target_index < len(target_positions):
        radius = target_reached_radius(
            target_positions,
            next_target_index,
            midgoal_radius,
            goal_radius,
        )
        if not target_within_geodesic_radius(
            env,
            current_position,
            target_positions[next_target_index],
            radius,
        ):
            break
        next_target_index += 1
    return next_target_index


def capture_agent_state(env):
    state = env.sim.get_agent_state()
    return (
        [float(value) for value in state.position],
        copy.copy(state.rotation),
    )


def restore_agent_state(env, state) -> None:
    position, rotation = state
    env.sim.set_agent_state(position=position, rotation=rotation, reset_sensors=True)


def next_oracle_action(env, follower, target_position, goal_radius: float) -> int:
    current_position = to_position_list(env.sim.get_agent_state().position)
    distance_to_target = geodesic_distance_to_target(
        env,
        current_position,
        target_position,
    )
    if distance_to_target <= float(goal_radius):
        return STOP_ACTION

    action = action_to_int(follower.get_next_action(target_position))
    if action == STOP_ACTION:
        current_position = to_position_list(env.sim.get_agent_state().position)
        distance_to_target = geodesic_distance_to_target(
            env,
            current_position,
            target_position,
        )
        if distance_to_target > float(goal_radius):
            raise RecoverableEpisodeNavigationError(
                "ShortestPathFollower returned stop before target: "
                f"geodesic_distance={distance_to_target:.3f}, "
                f"goal_radius={goal_radius}"
            )
    if action not in SUPPORTED_ACTION_IDS:
        raise DAggerCollectionError(f"Unsupported oracle action: {action}")
    return action


def oracle_action_for_current_state(
    env,
    midgoal_follower,
    goal_follower,
    target_positions,
    next_target_index: int,
    midgoal_radius: float,
    goal_radius: float,
) -> Tuple[int, int]:
    next_target_index = advance_target_index(
        env,
        target_positions,
        next_target_index,
        midgoal_radius,
        goal_radius,
    )
    if next_target_index >= len(target_positions):
        return STOP_ACTION, next_target_index
    radius = target_reached_radius(
        target_positions,
        next_target_index,
        midgoal_radius,
        goal_radius,
    )
    follower = (
        goal_follower
        if next_target_index == len(target_positions) - 1
        else midgoal_follower
    )
    return (
        next_oracle_action(
            env,
            follower,
            target_positions[next_target_index],
            radius,
        ),
        next_target_index,
    )


def preview_oracle_sequence(
    env,
    target_positions,
    next_target_index: int,
    midgoal_radius: float,
    goal_radius: float,
    action_horizon: int,
) -> List[int]:
    # GreedyGeodesicFollower records every queried action for anti-thrashing.
    # A preview executes counterfactual actions and then rewinds the simulator,
    # so its followers must be private to this preview and discarded with it.
    midgoal_follower = ShortestPathFollower(
        env.sim,
        goal_radius=midgoal_radius,
        return_one_hot=False,
        stop_on_error=True,
    )
    goal_follower = ShortestPathFollower(
        env.sim,
        goal_radius=goal_radius,
        return_one_hot=False,
        stop_on_error=True,
    )
    saved_state = capture_agent_state(env)
    preview_target_index = next_target_index
    actions: List[int] = []
    try:
        for _ in range(action_horizon):
            action, preview_target_index = oracle_action_for_current_state(
                env,
                midgoal_follower,
                goal_follower,
                target_positions,
                preview_target_index,
                midgoal_radius,
                goal_radius,
            )
            actions.append(action)
            if action == STOP_ACTION:
                break
            env.sim.step(action)
    finally:
        restore_agent_state(env, saved_state)

    if len(actions) < action_horizon:
        actions.extend([STOP_ACTION] * (action_horizon - len(actions)))
    return actions[:action_horizon]


def append_history_image(agent: PanoVLN_Agent, observation) -> None:
    image = Image.fromarray(observation["rgb"].astype("uint8")).convert("RGB")
    agent.rgb_history.append(image)


def save_episode_frame(
    episode_image_dir: str,
    frame_index: int,
    rgb,
    image_format: str,
    jpeg_quality: int,
    jpeg_subsampling: int,
    png_compress_level: int,
) -> None:
    save_rgb_frame(
        rgb,
        expected_frame_path(episode_image_dir, frame_index, image_format),
        jpeg_quality=jpeg_quality,
        jpeg_subsampling=jpeg_subsampling,
        png_compress_level=png_compress_level,
    )


def executable_prefix(action_ids: Sequence[int], execute_horizon: int) -> List[int]:
    prefix = list(action_ids[: max(1, int(execute_horizon))])
    if not prefix:
        return []
    if STOP_ACTION in prefix:
        return prefix[: prefix.index(STOP_ACTION) + 1]
    return prefix


def select_dagger_action_queue(
    model_actions: Sequence[int],
    oracle_actions: Sequence[int],
    horizon: int,
    stop_oracle_max_actions: int,
    beta: float,
    rng: random.Random,
) -> Tuple[str, List[int]]:
    """A nearby model STOP changes the actor, never the ordinary execution K."""
    if select_stop_commit_horizon(model_actions, stop_oracle_max_actions) is not None:
        source = "model_stop_oracle"
    elif len(model_actions) != len(oracle_actions):
        source = "model_fallback_oracle"
    else:
        source = "oracle" if rng.random() < beta else "model"
    actions = model_actions if source == "model" else oracle_actions
    selected = executable_prefix(actions, horizon)
    if not selected or (source == "model" and STOP_ACTION in selected):
        raise DAggerCollectionError("DAgger requires a nonempty queue with oracle-only STOP")
    return source, selected


def is_successful_dagger_termination(
    status: str,
    executed_actions: Sequence[int],
    final_geodesic_distance_to_goal: float,
    goal_radius: float,
) -> bool:
    """Accept only a real STOP within the final-goal geodesic radius."""

    return bool(
        status == "terminal"
        and executed_actions
        and int(executed_actions[-1]) == STOP_ACTION
        and float(final_geodesic_distance_to_goal) <= float(goal_radius)
    )


def model_predict_action_sequence(agent: PanoVLN_Agent, instruction: str):
    selected_indices = agent._select_image_indices()
    selected_images = agent._prepare_selected_images(selected_indices)
    return agent._predict_action_sequence_from_images(
        instruction=instruction,
        selected_images=selected_images,
    )


def execute_and_record_one_action(
    env,
    agent: PanoVLN_Agent,
    episode_image_dir: str,
    frame_index: int,
    action_id: int,
    image_format: str,
    jpeg_quality: int,
    jpeg_subsampling: int,
    png_compress_level: int,
    frame_states: List[Dict],
):
    observation = env.step(int(action_id))
    if int(action_id) == STOP_ACTION:
        return observation, frame_index
    frame_states.append(capture_frame_state(env.sim))
    save_episode_frame(
        episode_image_dir,
        frame_index,
        observation["rgb"],
        image_format,
        jpeg_quality,
        jpeg_subsampling,
        png_compress_level,
    )
    append_history_image(agent, observation)
    return observation, frame_index + 1


def collect_raw_dagger_episode(
    env,
    agent: PanoVLN_Agent,
    source_dataset: str,
    episode,
    dagger_id: str,
    output_root: str,
    dagger_dataset_name: str,
    reference_steps: int,
    rng: random.Random,
    midgoal_radius: float,
    goal_radius: float,
    alpha: float,
    action_horizon: int,
    execution_policy: Dict,
    max_steps_per_episode: int,
    image_format: str,
    jpeg_quality: int,
    jpeg_subsampling: int,
    png_compress_level: int,
) -> Tuple[Optional[Dict], Dict]:
    env.current_episode = episode
    observation = env.reset()
    agent.reset()

    instruction = extract_instruction(episode)
    if not instruction and "instruction" in observation:
        instruction = observation["instruction"].get("text", "")

    trajectory_id = dagger_trajectory_id(source_dataset, episode.episode_id)
    episode_image_dir = os.path.join(
        image_root(output_root, dagger_dataset_name),
        trajectory_id,
    )
    reset_episode_output_dir(episode_image_dir)
    save_episode_frame(
        episode_image_dir,
        0,
        observation["rgb"],
        image_format,
        jpeg_quality,
        jpeg_subsampling,
        png_compress_level,
    )
    append_history_image(agent, observation)

    frame_states = [capture_frame_state(env.sim)]
    replay_config = capture_replay_config(
        env, ERP_IMAGE_SIZE, image_format, jpeg_quality, jpeg_subsampling, png_compress_level,
    )
    scene_id = portable_scene_id(episode.scene_id, env._config.dataset.scenes_dir)
    start_position = to_position_list(env.sim.get_agent_state().position)
    current_position = list(start_position)
    target_positions, next_target_index = initial_target_state(episode, current_position)
    final_goal_position = get_goal_position(episode)
    if not positions_equal(target_positions[-1], final_goal_position):
        raise DAggerCollectionError(
            "Reference path and episode goal disagree: "
            f"episode_id={episode.episode_id}, "
            f"reference_end={target_positions[-1]}, goal={final_goal_position}"
        )

    executed_actions: List[int] = []
    oracle_chunks: List[Dict] = []
    policy_counts = {
        "oracle": 0, "model": 0, "model_stop_oracle": 0,
        "model_fallback_oracle": 0, "terminal_oracle": 0,
    }
    frame_index = 1
    env_steps = 0
    replan_index = 0
    status = "max_steps"
    executed_path_length = 0.0
    start_time = time.time()

    while not env.episode_over and env_steps < max_steps_per_episode:
        decision_step = frame_index - 1
        next_target_index = advance_target_index(
            env,
            target_positions,
            next_target_index,
            midgoal_radius,
            goal_radius,
        )
        beta = dynamic_oracle_probability(
            step_index=env_steps,
            reference_steps=reference_steps,
            alpha=alpha,
        )
        oracle_sequence = preview_oracle_sequence(
            env=env,
            target_positions=target_positions,
            next_target_index=next_target_index,
            midgoal_radius=midgoal_radius,
            goal_radius=goal_radius,
            action_horizon=action_horizon,
        )
        if oracle_sequence[0] == STOP_ACTION:
            executed_policy = "terminal_oracle"
            horizon = 1
            selected_actions = [STOP_ACTION]
        else:
            # Predict even on expert rounds: both actors share the ordinary K.
            _, model_action_ids = model_predict_action_sequence(agent, instruction)
            horizon = select_uncertainty_horizon(
                agent.prediction_action_uncertainties,
                execution_policy["uncertainty_budget"],
                execution_policy["replan_action_range"],
            )
            executed_policy, selected_actions = select_dagger_action_queue(
                model_action_ids, oracle_sequence, horizon,
                execution_policy["stop_oracle_max_actions"], beta, rng,
            )
        oracle_requests_stop = (
            executed_policy != "model" and STOP_ACTION in selected_actions
        )
        policy_counts[executed_policy] = policy_counts.get(executed_policy, 0) + 1
        active_chunk = {
            "step_index": int(decision_step),
            "oracle_actions": [int(action_id) for action_id in oracle_sequence],
            "execute_horizon": int(horizon),
            "executed_policy": executed_policy,
            "executed_count": 0,
        }
        oracle_chunks.append(active_chunk)

        for selected_action in selected_actions:
            if env_steps >= max_steps_per_episode:
                break

            # Only update reachability state here. Querying a persistent expert
            # follower while executing model actions would record expert actions
            # that were never executed and corrupt its anti-thrashing history.
            next_target_index = advance_target_index(
                env,
                target_positions,
                next_target_index,
                midgoal_radius,
                goal_radius,
            )
            oracle_requests_stop_now = next_target_index >= len(target_positions)

            action_to_execute = int(selected_action)
            if oracle_requests_stop_now and not oracle_requests_stop:
                terminal_decision_step = frame_index - 1
                if terminal_decision_step <= decision_step:
                    raise DAggerCollectionError(
                        "Oracle preview disagreed with the same real state: "
                        f"decision_step={decision_step}, "
                        f"terminal_step={terminal_decision_step}"
                    )
                active_chunk = {
                    "step_index": int(terminal_decision_step),
                    "oracle_actions": [STOP_ACTION] * action_horizon,
                    "execute_horizon": 1,
                    "executed_policy": "terminal_oracle",
                    "executed_count": 0,
                }
                oracle_chunks.append(active_chunk)
                policy_counts["terminal_oracle"] += 1
            if action_to_execute == STOP_ACTION and not oracle_requests_stop_now:
                raise DAggerCollectionError(
                    "Oracle preview requested STOP before the real trajectory "
                    "entered the final geodesic radius: "
                    f"decision_step={decision_step}, "
                    f"current_step={frame_index - 1}"
                )
            if oracle_requests_stop_now:
                action_to_execute = STOP_ACTION

            executed_actions.append(action_to_execute)
            previous_position = to_position_list(env.sim.get_agent_state().position)
            observation, frame_index = execute_and_record_one_action(
                env=env,
                agent=agent,
                episode_image_dir=episode_image_dir,
                frame_index=frame_index,
                action_id=action_to_execute,
                image_format=image_format,
                jpeg_quality=jpeg_quality,
                jpeg_subsampling=jpeg_subsampling,
                png_compress_level=png_compress_level,
                frame_states=frame_states,
            )
            current_position = to_position_list(env.sim.get_agent_state().position)
            executed_path_length += euclidean_distance(previous_position, current_position)
            env_steps += 1
            active_chunk["executed_count"] += 1

            if action_to_execute == STOP_ACTION:
                status = "terminal"
                break
            if env.episode_over:
                status = "env_episode_over"
                break

        replan_index += 1

    final_position = to_position_list(env.sim.get_agent_state().position)
    final_distance_to_goal = geodesic_distance_to_target(
        env,
        final_position,
        final_goal_position,
    )
    terminated_by_stop = bool(
        status == "terminal"
        and executed_actions
        and executed_actions[-1] == STOP_ACTION
    )
    episode_success = is_successful_dagger_termination(
        status=status,
        executed_actions=executed_actions,
        final_geodesic_distance_to_goal=final_distance_to_goal,
        goal_radius=goal_radius,
    )
    start_goal_distance = geodesic_distance_to_target(
        env,
        start_position,
        final_goal_position,
    )
    path_efficiency = start_goal_distance / max(start_goal_distance, executed_path_length, 1e-8)

    if not episode_success:
        if not terminated_by_stop:
            status = f"not_saved_without_stop:{status}"
        else:
            status = f"not_saved_outside_goal_radius:{status}"
        remove_if_exists(episode_image_dir)
        summary = {
            "episode_id": dagger_id,
            "source_dataset": source_dataset,
            "source_episode_id": str(episode.episode_id),
            "scene_id": getattr(episode, "scene_id", ""),
            "status": status,
            "success": bool(episode_success),
            "saved": False,
            "midgoal_radius": float(midgoal_radius),
            "goal_radius": float(goal_radius),
            "final_distance_to_goal": float(final_distance_to_goal),
            "start_goal_distance": float(start_goal_distance),
            "executed_path_length": float(executed_path_length),
            "path_efficiency": float(path_efficiency),
            "reference_steps": int(reference_steps),
            "oracle_chunks": len(oracle_chunks),
            "executed_steps": len(executed_actions),
            "saved_frames": 0,
            "image_format": image_format,
            "replans": replan_index,
            "dagger_alpha": float(alpha),
            "policy_counts": policy_counts,
            "execution_policy": dict(execution_policy),
            "seconds": time.time() - start_time,
        }
        return None, summary

    annotation = {
        "episode_id": dagger_id,
        "trajectory_id": trajectory_id,
        "instruction": instruction,
        "source_dataset": source_dataset,
        "source_episode_id": str(episode.episode_id),
        "scene_id": scene_id,
        "replay_config": replay_config,
        "frame_states": frame_states,
        "action_horizon": int(action_horizon),
        "execution_policy": dict(execution_policy),
        # Complete mixed-policy trajectory for image/history alignment.
        "actions": executed_actions,
        # Efficient-VLN expert labels at the real model decision states.
        "oracle_chunks": oracle_chunks,
    }
    validate_replay_metadata(annotation)
    summary = {
        "episode_id": dagger_id,
        "source_dataset": source_dataset,
        "source_episode_id": str(episode.episode_id),
        "scene_id": getattr(episode, "scene_id", ""),
        "status": status,
        "success": bool(episode_success),
        "saved": True,
        "midgoal_radius": float(midgoal_radius),
        "goal_radius": float(goal_radius),
        "final_distance_to_goal": float(final_distance_to_goal),
        "start_goal_distance": float(start_goal_distance),
        "executed_path_length": float(executed_path_length),
        "path_efficiency": float(path_efficiency),
        "reference_steps": int(reference_steps),
        "oracle_chunks": len(oracle_chunks),
        "executed_steps": len(executed_actions),
        "saved_frames": count_saved_frames(episode_image_dir, image_format),
        "image_format": image_format,
        "replans": replan_index,
        "dagger_alpha": float(alpha),
        "policy_counts": policy_counts,
        "execution_policy": dict(execution_policy),
        "seconds": time.time() - start_time,
    }
    return annotation, summary


def select_source_episodes(dataset, episode_ids, max_episodes, allowed_episode_ids=None):
    source_episodes = dataset.episodes
    if allowed_episode_ids is not None:
        allowed_episode_ids = {int(episode_id) for episode_id in allowed_episode_ids}
        source_episodes = [
            episode
            for episode in source_episodes
            if int(episode.episode_id) in allowed_episode_ids
        ]
    selected = filter_episodes(
        source_episodes,
        episode_ids=episode_ids,
        max_episodes=max_episodes if max_episodes is not None else None,
    )
    return sort_episodes_for_scene_locality(selected)


def dagger_worker(
    result_queue,
    expected_parent_pid,
    source_dataset,
    episode_ids,
    dagger_ids,
    model_path,
    output_root,
    dagger_dataset_name,
    reference_input_root,
    runtime_root,
    partial_annotation_path,
    worker_index,
    local_gpu_id,
    display_gpu_id,
    process_index_on_gpu,
    midgoal_radius,
    goal_radius,
    alpha,
    action_horizon,
    execution_policy,
    max_steps_per_episode,
    seed,
    attn_implementation,
    max_memory_images,
    memory_pool_window_frames,
    skip_failed_episodes,
    image_format,
    jpeg_quality,
    jpeg_subsampling,
    png_compress_level,
):
    env = None
    active_source_episode_id = None
    active_dagger_id = None
    try:
        arm_linux_parent_death_signal(expected_parent_pid)
        configure_worker_torch_threads()
        seed_all(seed + worker_index)
        if torch.cuda.is_available():
            torch.cuda.set_device(local_gpu_id)

        env_config, dataset = load_dataset(dataset_name=source_dataset)
        configure_dagger_env(
            env_config=env_config,
            local_gpu_id=local_gpu_id,
            max_episode_steps=max_steps_per_episode,
        )
        selected_episode_id_set = {int(episode_id) for episode_id in episode_ids}
        dataset.episodes = [
            episode
            for episode in dataset.episodes
            if int(episode.episode_id) in selected_episode_id_set
        ]
        dataset.episodes = sort_episodes_for_scene_locality(dataset.episodes)
        found_episode_ids = {int(episode.episode_id) for episode in dataset.episodes}
        missing_episode_ids = sorted(selected_episode_id_set - found_episode_ids)
        if missing_episode_ids:
            raise ValueError(
                f"[{source_dataset}] worker {worker_index} missing episode ids: "
                f"{missing_episode_ids}"
            )

        reference_lengths = load_reference_action_lengths(
            reference_input_root=reference_input_root,
            dataset_names=[source_dataset],
        )

        agent = PanoVLN_Agent(
            model_path=str(Path(model_path).expanduser()),
            lora_path=None,
            result_path=os.path.join(runtime_root, f"worker_{worker_index:03d}"),
            forward_distance=25,
            turn_angle=15,
            max_memory_images=max_memory_images,
            memory_pool_window_frames=memory_pool_window_frames,
            save_topdown=False,
            attn_implementation=attn_implementation,
            actions_per_replan="uncertainty",
            uncertainty_budget=execution_policy["uncertainty_budget"],
            replan_action_range=execution_policy["replan_action_range"],
            stop_commit_max_actions=0,
            collision_recovery_steps=0,
        )
        if agent.action_sequence_length != action_horizon:
            raise DAggerCollectionError(
                "DAgger action_horizon must match the loaded model's "
                "action_sequence_length: "
                f"requested={action_horizon}, "
                f"model={agent.action_sequence_length}"
            )

        with silence_external_output():
            env = habitat.Env(config=env_config.habitat, dataset=dataset)

        os.makedirs(os.path.dirname(partial_annotation_path), exist_ok=True)
        if repair_jsonl_for_append(partial_annotation_path):
            tqdm.write(
                f"[dagger:{source_dataset}] repaired interrupted JSONL journal "
                f"before resume: {partial_annotation_path}"
            )
        rng = random.Random(seed + worker_index)
        with open(partial_annotation_path, "a", encoding="utf-8") as annotation_handle:
            for episode in dataset.episodes:
                episode_start_time = time.time()
                dagger_id = str(dagger_ids[int(episode.episode_id)])
                active_source_episode_id = int(episode.episode_id)
                active_dagger_id = dagger_id
                annotation = None
                annotation_committed = False
                try:
                    reference_steps = infer_reference_steps(
                        episode,
                        dataset_name=source_dataset,
                        reference_lengths=reference_lengths,
                    )
                    annotation, summary = collect_raw_dagger_episode(
                        env=env,
                        agent=agent,
                        source_dataset=source_dataset,
                        episode=episode,
                        dagger_id=dagger_id,
                        output_root=output_root,
                        dagger_dataset_name=dagger_dataset_name,
                        reference_steps=reference_steps,
                        rng=rng,
                        midgoal_radius=midgoal_radius,
                        goal_radius=goal_radius,
                        alpha=alpha,
                        action_horizon=action_horizon,
                        execution_policy=execution_policy,
                        max_steps_per_episode=max_steps_per_episode,
                        image_format=image_format,
                        jpeg_quality=jpeg_quality,
                        jpeg_subsampling=jpeg_subsampling,
                        png_compress_level=png_compress_level,
                    )
                    if annotation is None:
                        result_status = "not_saved"
                        sample_count = 0
                    else:
                        write_jsonl_item(annotation_handle, annotation)
                        annotation_committed = True
                        result_status = "ok"
                        sample_count = len(annotation["actions"])
                    result_queue.put(
                        {
                            "status": result_status,
                            "worker_index": worker_index,
                            "display_gpu_id": display_gpu_id,
                            "process_index_on_gpu": process_index_on_gpu,
                            "source_dataset": source_dataset,
                            "source_episode_id": int(episode.episode_id),
                            "episode_id": dagger_id,
                            "samples": sample_count,
                            "episode_status": summary["status"],
                            "time_per_episode": time.time() - episode_start_time,
                        }
                    )
                except Exception as episode_error:
                    error = traceback.format_exc()
                    if annotation is not None and not annotation_committed:
                        # A flush/fsync failure can occur after the complete row
                        # reached the journal.  Close first so buffered bytes
                        # cannot appear after deciding whether images are safe
                        # to remove, then verify the exact persisted tail row.
                        try:
                            annotation_handle.close()
                        except OSError:
                            pass
                        annotation_committed = jsonl_ends_with_item(
                            partial_annotation_path,
                            annotation,
                        )
                    if not annotation_committed:
                        remove_if_exists(
                            os.path.join(
                                image_root(output_root, dagger_dataset_name),
                                dagger_trajectory_id(source_dataset, episode.episode_id),
                            )
                        )
                    if not (
                        skip_failed_episodes
                        and isinstance(
                            episode_error,
                            RecoverableEpisodeNavigationError,
                        )
                    ):
                        raise
                    result_queue.put(
                        {
                            "status": "skipped",
                            "worker_index": worker_index,
                            "display_gpu_id": display_gpu_id,
                            "process_index_on_gpu": process_index_on_gpu,
                            "source_dataset": source_dataset,
                            "source_episode_id": int(episode.episode_id),
                            "episode_id": dagger_id,
                            "time_per_episode": time.time() - episode_start_time,
                            "error": error,
                            "error_message": str(episode_error),
                        }
                    )
    except Exception:
        result_queue.put(
            {
                "status": "error",
                "worker_index": worker_index,
                "display_gpu_id": display_gpu_id,
                "process_index_on_gpu": process_index_on_gpu,
                "source_dataset": source_dataset,
                "source_episode_id": active_source_episode_id,
                "episode_id": active_dagger_id,
                "error": traceback.format_exc(),
            }
        )
    finally:
        if env is not None:
            with silence_external_output():
                env.close()


def build_worker_jobs(
    source_dataset,
    selected_episodes,
    requested_gpu_ids,
    visible_gpu_ids,
    num_thread,
    num_processes_per_gpu,
):
    worker_assignments = build_worker_assignments(
        requested_gpu_ids=requested_gpu_ids,
        visible_gpu_ids=visible_gpu_ids,
        num_processes_per_gpu=num_processes_per_gpu,
        max_workers=num_thread,
    )
    worker_assignments = worker_assignments[: min(len(worker_assignments), len(selected_episodes))]
    episode_splits = build_locality_balanced_episode_splits(
        selected_episodes,
        len(worker_assignments),
    )
    worker_jobs = []
    for worker_assignment, worker_episodes in zip(worker_assignments, episode_splits):
        if not worker_episodes:
            continue
        worker_jobs.append(
            {
                "assignment": worker_assignment,
                "episode_ids": [int(episode.episode_id) for episode in worker_episodes],
            }
        )
    return worker_jobs


def assert_matching_execution_policy(annotation: Dict, execution_policy: Dict) -> None:
    if annotation.get("execution_policy") != execution_policy:
        raise ValueError(
            "Existing DAgger data uses a different execution policy "
            f"(episode_id={annotation.get('episode_id')}). "
            "Use a separate output_root; fixed-6 data remains supported for training."
        )


def load_complete_existing_annotations(
    output_root,
    dagger_dataset_name,
    image_format,
    action_horizon,
    execution_policy,
    progress_dir=None,
):
    existing: Dict[str, Dict] = {}
    for annotation in load_jsonl_index(annotation_path(output_root, dagger_dataset_name)).values():
        assert_matching_execution_policy(annotation, execution_policy)
        if is_raw_episode_complete(
            output_root,
            dagger_dataset_name,
            annotation,
            image_format,
            action_horizon,
            execution_policy,
        ):
            existing[annotation_episode_id(annotation)] = annotation
    if progress_dir is not None:
        for path in list_rank_annotation_paths(progress_dir):
            for annotation in load_jsonl_index(path).values():
                assert_matching_execution_policy(annotation, execution_policy)
                episode_id = annotation_episode_id(annotation)
                current = existing.get(episode_id)
                if current == annotation:
                    continue
                if not is_raw_episode_complete(
                    output_root,
                    dagger_dataset_name,
                    annotation,
                    image_format,
                    action_horizon,
                    execution_policy,
                ):
                    continue
                if current is not None:
                    raise ValueError(
                        f"Conflicting complete annotations for episode_id={episode_id} "
                        f"while loading existing progress from {path}"
                    )
                existing[episode_id] = annotation
    return existing


def add_complete_annotation(
    merged_annotations: Dict[str, Dict],
    output_root: str,
    dagger_dataset_name: str,
    annotation: Dict,
    source_path: str,
    image_format: str,
    action_horizon: int,
    execution_policy: Dict,
) -> None:
    assert_matching_execution_policy(annotation, execution_policy)
    episode_id = annotation_episode_id(annotation)
    existing = merged_annotations.get(episode_id)
    if existing == annotation:
        return
    if not is_raw_episode_complete(
        output_root,
        dagger_dataset_name,
        annotation,
        image_format,
        action_horizon,
        execution_policy,
    ):
        tqdm.write(
            f"[dagger] skipping incomplete annotation episode_id="
            f"{annotation.get('episode_id')} from {source_path}"
        )
        return
    if existing is not None and existing != annotation:
        raise ValueError(
            f"Conflicting complete annotations for episode_id={episode_id} "
            f"while merging {source_path}"
        )
    merged_annotations[episode_id] = annotation


def merge_partial_outputs(
    output_root: str,
    dagger_dataset_name: str,
    progress_dir: str,
    existing_annotations: Dict[str, Dict],
    image_format: str,
    action_horizon: int,
    execution_policy: Dict,
):
    merged_annotations: Dict[str, Dict] = dict(existing_annotations)
    for path in list_rank_annotation_paths(progress_dir):
        for annotation in load_jsonl_index(path).values():
            add_complete_annotation(
                merged_annotations=merged_annotations,
                output_root=output_root,
                dagger_dataset_name=dagger_dataset_name,
                annotation=annotation,
                source_path=path,
                image_format=image_format,
                action_horizon=action_horizon,
                execution_policy=execution_policy,
            )
    write_jsonl_index(annotation_path(output_root, dagger_dataset_name), merged_annotations)


def process_source_dataset(
    source_dataset,
    output_root,
    dagger_dataset_name,
    model_path,
    reference_input_root,
    requested_gpu_ids,
    visible_gpu_ids,
    num_thread,
    num_processes_per_gpu,
    skip_existing_episodes,
    max_episodes,
    episode_ids,
    progress_dir,
    existing_annotations,
    midgoal_radius,
    goal_radius,
    alpha,
    action_horizon,
    execution_policy,
    max_steps_per_episode,
    seed,
    attn_implementation,
    max_memory_images,
    memory_pool_window_frames,
    skip_failed_episodes,
    image_format,
    jpeg_quality,
    jpeg_subsampling,
    png_compress_level,
):
    _, dataset = load_dataset(dataset_name=source_dataset)
    reference_lengths = load_reference_action_lengths(
        reference_input_root=reference_input_root,
        dataset_names=[source_dataset],
    )
    allowed_episode_ids = reference_episode_ids(reference_lengths, source_dataset)
    validate_reference_ids_exist_in_habitat_dataset(
        source_dataset=source_dataset,
        dataset=dataset,
        allowed_episode_ids=allowed_episode_ids,
    )
    selected_episodes = select_source_episodes(
        dataset=dataset,
        episode_ids=episode_ids,
        max_episodes=max_episodes,
        allowed_episode_ids=allowed_episode_ids,
    )
    if not selected_episodes:
        tqdm.write(f"[dagger:{source_dataset}] no selected episodes")
        return {}
    tqdm.write(
        f"[dagger:{source_dataset}] reference_jsonl_episodes="
        f"{len(allowed_episode_ids)}, selected_after_filter={len(selected_episodes)}"
    )

    pending_episodes = []
    dagger_ids = {}
    skipped = 0
    for episode in selected_episodes:
        dagger_id = dagger_trajectory_id(source_dataset, episode.episode_id)
        dagger_ids[int(episode.episode_id)] = dagger_id
        if skip_existing_episodes and dagger_id in existing_annotations:
            skipped += 1
            continue
        pending_episodes.append(episode)

    total_count = len(selected_episodes)
    if not pending_episodes:
        tqdm.write(
            f"[dagger:{source_dataset}] all {total_count} selected episodes already exist"
        )
        return existing_annotations

    worker_jobs = build_worker_jobs(
        source_dataset=source_dataset,
        selected_episodes=pending_episodes,
        requested_gpu_ids=requested_gpu_ids,
        visible_gpu_ids=visible_gpu_ids,
        num_thread=num_thread,
        num_processes_per_gpu=num_processes_per_gpu,
    )
    worker_labels = [
        f"gpu{job['assignment']['display_gpu_id']}:p{job['assignment']['process_index_on_gpu']}"
        for job in worker_jobs
    ]
    tqdm.write(
        f"[dagger:{source_dataset}] total={total_count}, skipped={skipped}, "
        f"pending={len(pending_episodes)}, workers={len(worker_jobs)}, "
        f"assignments={worker_labels}"
    )

    ctx = mp.get_context("spawn")
    parent_pid = os.getpid()
    result_queue = ctx.Queue()
    processes = []
    worker_bars = {}

    total_bar = tqdm(
        total=total_count,
        desc=f"dagger:{source_dataset} total",
        position=0,
        dynamic_ncols=True,
        file=sys.stdout,
    )
    if skipped:
        total_bar.update(skipped)
        total_bar.set_postfix_str(f"done={total_bar.n}/{total_count}, skipped={skipped}")

    for position, job in enumerate(worker_jobs, start=1):
        assignment = job["assignment"]
        partial_annotation_path = rank_annotation_path(progress_dir, assignment["worker_index"])
        worker_dagger_ids = {
            episode_id: dagger_ids[episode_id]
            for episode_id in job["episode_ids"]
        }
        process = ctx.Process(
            target=dagger_worker,
            args=(
                result_queue,
                parent_pid,
                source_dataset,
                job["episode_ids"],
                worker_dagger_ids,
                model_path,
                output_root,
                dagger_dataset_name,
                reference_input_root,
                os.path.join(progress_dir, "runtime"),
                partial_annotation_path,
                assignment["worker_index"],
                assignment["local_gpu_id"],
                assignment["display_gpu_id"],
                assignment["process_index_on_gpu"],
                midgoal_radius,
                goal_radius,
                alpha,
                action_horizon,
                execution_policy,
                max_steps_per_episode,
                seed,
                attn_implementation,
                max_memory_images,
                memory_pool_window_frames,
                skip_failed_episodes,
                image_format,
                jpeg_quality,
                jpeg_subsampling,
                png_compress_level,
            ),
        )
        process.start()
        processes.append(process)
        worker_bars[assignment["worker_index"]] = tqdm(
            total=len(job["episode_ids"]),
            desc=(
                f"{source_dataset} gpu{assignment['display_gpu_id']}"
                f" p{assignment['process_index_on_gpu']}"
            ),
            position=position,
            dynamic_ncols=True,
            file=sys.stdout,
            leave=True,
        )

    completed = skipped
    failed = 0
    failed_episode_ids = []
    not_saved = 0
    success = False
    try:
        while completed < total_count:
            try:
                result = result_queue.get(timeout=QUEUE_POLL_TIMEOUT_SECONDS)
            except queue.Empty:
                crashed_processes = [
                    process
                    for process in processes
                    if not process.is_alive() and process.exitcode not in (None, 0)
                ]
                if crashed_processes:
                    raise RuntimeError(
                        f"[dagger:{source_dataset}] worker crashed: "
                        f"{[(process.pid, process.exitcode) for process in crashed_processes]}"
                    )
                if processes and all(not process.is_alive() for process in processes):
                    raise RuntimeError(
                        f"[dagger:{source_dataset}] all workers exited before reporting "
                        f"all episodes: received={completed}/{total_count}, "
                        f"workers={[(process.pid, process.exitcode) for process in processes]}"
                    )
                continue

            if result["status"] == "error":
                episode_context = ""
                if result.get("episode_id") is not None:
                    episode_context = (
                        f" episode={result['episode_id']}"
                        f" source_episode={result.get('source_episode_id')}"
                    )
                raise RuntimeError(
                    f"[dagger:{source_dataset}] worker "
                    f"gpu{result['display_gpu_id']} "
                    f"p{result['process_index_on_gpu']}{episode_context} failed:\n"
                    f"{result['error']}"
                )

            if result["status"] == "skipped":
                failed += 1
                failed_episode_ids.append(result["episode_id"])
                tqdm.write(
                    f"[dagger:{source_dataset}] skipped recoverable episode "
                    f"{result['episode_id']}: {result.get('error_message', 'unknown error')}"
                )
            if result["status"] == "not_saved":
                not_saved += 1

            completed += 1
            total_bar.update(1)
            total_bar.set_postfix_str(
                f"done={completed}/{total_count}, skipped={skipped}, "
                f"not_saved={not_saved}, failed={failed}"
            )

            worker_bar = worker_bars[result["worker_index"]]
            worker_bar.update(1)
            worker_bar.set_postfix_str(
                f"src_ep={result['source_episode_id']}, "
                f"dagger_ep={result['episode_id']}, "
                f"status={result.get('episode_status', result['status'])}, "
                f"{result['time_per_episode']:.2f}s"
            )

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"[dagger:{source_dataset}] worker pid={process.pid} "
                    f"exited with code {process.exitcode}"
                )
        if failed_episode_ids:
            preview = failed_episode_ids[:20]
            tqdm.write(
                f"[dagger:{source_dataset}] recoverable failures={len(failed_episode_ids)}, "
                f"episode_ids={preview}"
            )
        success = True
    finally:
        if not success:
            for process in processes:
                if process.is_alive():
                    process.terminate()
            for process in processes:
                process.join()
        total_bar.close()
        for worker_bar in worker_bars.values():
            worker_bar.close()

    return existing_annotations


def validate_args(args) -> None:
    validate_safe_dataset_name(args.dagger_dataset_name)
    args.image_format = normalize_image_format(args.image_format)
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError(
            f"jpeg_quality must be in [1, 100], got {args.jpeg_quality}"
        )
    if args.jpeg_subsampling not in {0, 1, 2}:
        raise ValueError(
            "jpeg_subsampling must be one of 0, 1, 2, "
            f"got {args.jpeg_subsampling}"
        )
    if not 0 <= args.png_compress_level <= 9:
        raise ValueError(
            "png_compress_level must be in [0, 9], "
            f"got {args.png_compress_level}"
        )
    unsupported_sources = [
        dataset_name
        for dataset_name in args.source_dataset_name
        if dataset_name not in DATASET_CONFIG
    ]
    if unsupported_sources:
        raise ValueError(
            f"Unsupported source_dataset_name={unsupported_sources}; "
            f"supported={sorted(DATASET_CONFIG)}"
        )
    if not os.path.isdir(args.model_path):
        raise FileNotFoundError(f"model_path does not exist or is not a directory: {args.model_path}")
    if not os.path.isdir(args.reference_input_root):
        raise FileNotFoundError(
            f"reference_input_root does not exist or is not a directory: "
            f"{args.reference_input_root}"
        )
    if args.max_episodes is not None and args.max_episodes < 0:
        raise ValueError(f"max_episodes must be non-negative, got {args.max_episodes}")
    if args.num_thread <= 0:
        raise ValueError(f"num_thread must be positive, got {args.num_thread}")
    if args.num_processes_per_gpu is not None and args.num_processes_per_gpu <= 0:
        raise ValueError(
            f"num_processes_per_gpu must be positive, got {args.num_processes_per_gpu}"
        )
    if args.cpu_threads_per_worker <= 0:
        raise ValueError(
            "cpu_threads_per_worker must be positive, "
            f"got {args.cpu_threads_per_worker}"
        )
    if args.action_horizon != DEFAULT_ACTION_HORIZON:
        raise ValueError(
            f"action_horizon must be {DEFAULT_ACTION_HORIZON} for the current "
            f"DAgger pipeline, got {args.action_horizon}"
        )
    validate_dagger_execution_policy(execution_policy_from_args(args))
    if args.max_steps_per_episode <= 0:
        raise ValueError(
            f"max_steps_per_episode must be positive, got {args.max_steps_per_episode}"
        )
    if not math.isfinite(args.midgoal_radius) or args.midgoal_radius <= 0:
        raise ValueError(
            f"midgoal_radius must be positive, got {args.midgoal_radius}"
        )
    if not math.isfinite(args.goal_radius) or args.goal_radius <= 0:
        raise ValueError(f"goal_radius must be positive, got {args.goal_radius}")
    if not math.isfinite(args.alpha):
        raise ValueError(f"alpha must be finite, got {args.alpha}")
    dynamic_oracle_probability(0, 1, args.alpha)


def execution_policy_from_args(args) -> Dict:
    return {
        "actions_per_replan": "uncertainty",
        "uncertainty_budget": args.uncertainty_budget,
        "replan_action_range": list(args.replan_action_range),
        "stop_oracle_max_actions": args.stop_oracle_max_actions,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect raw PanoVLN DAgger data with Efficient-VLN dynamic ratio. "
            "Outputs sub_dataset/dagger.jsonl and images/dagger/."
        )
    )
    parser.set_defaults(dagger_dataset_name=DEFAULT_DAGGER_DATASET_NAME)
    parser.add_argument("--source_dataset_name", nargs="+", default=list(DEFAULT_SOURCE_DATASETS))
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference_input_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--num_thread", type=int, default=1)
    parser.add_argument("--num_processes_per_gpu", type=int, default=None)
    parser.add_argument(
        "--cpu_threads_per_worker",
        type=int,
        default=DEFAULT_CPU_THREADS_PER_WORKER,
        help=(
            "Maximum PyTorch/OpenMP CPU threads per spawned collector worker. "
            "Keep this small when running many GPU workers."
        ),
    )
    parser.add_argument("--episode_ids", nargs="*", default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)
    parser.add_argument(
        "--midgoal_radius",
        type=float,
        default=DEFAULT_DAGGER_MIDGOAL_RADIUS,
        help="Radius for considering an intermediate reference waypoint reached.",
    )
    parser.add_argument(
        "--goal_radius",
        type=float,
        default=DEFAULT_DAGGER_GOAL_RADIUS,
        help=(
            "Single final-goal radius used both for oracle STOP and for "
            "accepting a normally terminated DAgger episode."
        ),
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--action_horizon",
        type=int,
        default=DEFAULT_ACTION_HORIZON,
        help="Oracle-label and model-prediction length (fixed to 18).",
    )
    parser.add_argument(
        "--uncertainty_budget",
        type=float,
        default=DEFAULT_UNCERTAINTY_BUDGET,
        help="Uncertainty budget for the execution length shared by both actors.",
    )
    parser.add_argument(
        "--replan_action_range",
        type=int,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=list(DEFAULT_REPLAN_ACTION_RANGE),
        help="Inclusive bounds for the ordinary uncertainty execution length.",
    )
    parser.add_argument(
        "--stop_oracle_max_actions",
        type=int,
        default=DEFAULT_STOP_ORACLE_MAX_ACTIONS,
        help="A model STOP in the first N actions selects oracle; N must cover MAX.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="sdpa",
        choices=["sdpa", "flash_attention_2", "eager"],
    )
    parser.add_argument("--max_memory_images", type=int, default=DEFAULT_MAX_MEMORY_IMAGES)
    parser.add_argument(
        "--memory_pool_window_frames",
        type=int,
        default=DEFAULT_MEMORY_POOL_WINDOW_FRAMES,
    )
    parser.add_argument(
        "--skip_existing_episodes",
        nargs="?",
        const=True,
        default=True,
        type=str2bool,
    )
    parser.add_argument(
        "--skip_failed_episodes",
        nargs="?",
        const=True,
        default=False,
        type=str2bool,
        help=(
            "Skip only recognized episode-local navigation failures. "
            "Unexpected model, CUDA, I/O, and code errors always remain fatal."
        ),
    )
    parser.add_argument(
        "--image_format",
        type=str,
        default="png",
        choices=("jpg", "jpeg", "png"),
        help="Saved panorama format. DAgger defaults to lossless PNG.",
    )
    parser.add_argument("--jpeg_quality", type=int, default=75)
    parser.add_argument(
        "--jpeg_subsampling",
        type=int,
        default=2,
        choices=(0, 1, 2),
    )
    parser.add_argument(
        "--png_compress_level",
        type=int,
        default=6,
        choices=tuple(range(10)),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    execution_policy = execution_policy_from_args(args)
    configure_worker_thread_environment(args.cpu_threads_per_worker)
    configure_worker_torch_threads()
    seed_all(args.seed)

    requested_gpu_ids = parse_gpu_ids(args.gpu_ids)
    visible_gpu_ids = None
    if requested_gpu_ids is not None:
        visible_gpu_ids = remap_gpu_ids_to_visible_devices(requested_gpu_ids)
        validate_gpu_ids(visible_gpu_ids)
    if args.num_processes_per_gpu is not None and requested_gpu_ids is None:
        raise ValueError("--num_processes_per_gpu requires --gpu_ids")

    selected_episode_ids = parse_episode_ids(args.episode_ids)
    sub_dataset_root = os.path.join(args.output_root, "sub_dataset")
    images_root = os.path.join(args.output_root, "images")
    os.makedirs(sub_dataset_root, exist_ok=True)
    os.makedirs(images_root, exist_ok=True)
    assert_path_strictly_inside(
        annotation_path(args.output_root, args.dagger_dataset_name),
        sub_dataset_root,
    )
    assert_path_strictly_inside(
        image_root(args.output_root, args.dagger_dataset_name),
        images_root,
    )
    os.makedirs(image_root(args.output_root, args.dagger_dataset_name), exist_ok=True)

    progress_dir = progress_dir_path(
        output_root=args.output_root,
        dagger_dataset_name=args.dagger_dataset_name,
    )
    assert_path_strictly_inside(progress_dir, sub_dataset_root)
    if args.skip_existing_episodes:
        os.makedirs(progress_dir, exist_ok=True)
        lock_path = acquire_progress_lock(progress_dir)
        existing_annotations = load_complete_existing_annotations(
            output_root=args.output_root,
            dagger_dataset_name=args.dagger_dataset_name,
            image_format=args.image_format,
            action_horizon=args.action_horizon,
            execution_policy=execution_policy,
            progress_dir=progress_dir,
        )
    else:
        assert_no_active_progress_lock(progress_dir)
        remove_if_exists(progress_dir)
        os.makedirs(progress_dir, exist_ok=True)
        lock_path = acquire_progress_lock(progress_dir)
        existing_annotations = {}
        remove_if_exists(annotation_path(args.output_root, args.dagger_dataset_name))
        remove_if_exists(image_root(args.output_root, args.dagger_dataset_name))
        os.makedirs(image_root(args.output_root, args.dagger_dataset_name), exist_ok=True)

    print(
        "Dynamic Ratio DAgger raw collection: "
        f"beta_t=1-alpha^(t/T), alpha={args.alpha}, "
        f"T=ground-truth action steps excluding stop"
    )
    print(
        f"model={args.model_path}, output_root={args.output_root}, "
        f"dagger_dataset={args.dagger_dataset_name}, "
        f"sources={args.source_dataset_name}, image_format={args.image_format}"
    )
    print(
        f"gpu_ids={args.gpu_ids}, num_thread={args.num_thread}, "
        f"num_processes_per_gpu={args.num_processes_per_gpu}, "
        f"cpu_threads_per_worker={args.cpu_threads_per_worker}, "
        f"midgoal_radius={args.midgoal_radius}, goal_radius={args.goal_radius}, "
        f"action_horizon={args.action_horizon}, "
        f"execution_policy={execution_policy}, "
        f"progress_dir={progress_dir}, lock={lock_path}"
    )

    success = False
    try:
        for source_dataset in args.source_dataset_name:
            process_source_dataset(
                source_dataset=source_dataset,
                output_root=args.output_root,
                dagger_dataset_name=args.dagger_dataset_name,
                model_path=args.model_path,
                reference_input_root=args.reference_input_root,
                requested_gpu_ids=requested_gpu_ids,
                visible_gpu_ids=visible_gpu_ids,
                num_thread=args.num_thread,
                num_processes_per_gpu=args.num_processes_per_gpu,
                skip_existing_episodes=args.skip_existing_episodes,
                max_episodes=args.max_episodes,
                episode_ids=selected_episode_ids,
                progress_dir=progress_dir,
                existing_annotations=existing_annotations,
                midgoal_radius=args.midgoal_radius,
                goal_radius=args.goal_radius,
                alpha=args.alpha,
                action_horizon=args.action_horizon,
                execution_policy=execution_policy,
                max_steps_per_episode=args.max_steps_per_episode,
                seed=args.seed,
                attn_implementation=args.attn_implementation,
                max_memory_images=args.max_memory_images,
                memory_pool_window_frames=args.memory_pool_window_frames,
                skip_failed_episodes=args.skip_failed_episodes,
                image_format=args.image_format,
                jpeg_quality=args.jpeg_quality,
                jpeg_subsampling=args.jpeg_subsampling,
                png_compress_level=args.png_compress_level,
            )

        merge_partial_outputs(
            output_root=args.output_root,
            dagger_dataset_name=args.dagger_dataset_name,
            progress_dir=progress_dir,
            existing_annotations=existing_annotations,
            image_format=args.image_format,
            action_horizon=args.action_horizon,
            execution_policy=execution_policy,
        )
        success = True
    finally:
        if success:
            remove_if_exists(progress_dir)
        else:
            print(
                f"collection interrupted or failed; keeping progress directory for resume: "
                f"{progress_dir}"
            )

    final_annotations = load_jsonl_index(annotation_path(args.output_root, args.dagger_dataset_name))
    print(
        f"saved {len(final_annotations)} raw dagger episodes to "
        f"{annotation_path(args.output_root, args.dagger_dataset_name)}"
    )
    print(f"saved dagger images to {image_root(args.output_root, args.dagger_dataset_name)}")


if __name__ == "__main__":
    main()
