import argparse
import atexit
import copy
import hashlib
import json
import math
import multiprocessing as mp
import os
import queue
import random
import re
import shutil
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
    MOVE_FORWARD_ACTION,
    STOP_ACTION,
    TURN_LEFT_ACTION,
    TURN_RIGHT_ACTION,
    CONFIG as DATASET_CONFIG,
    ERP_IMAGE_SIZE,
    action_to_int,
    build_locality_balanced_episode_splits,
    build_worker_assignments,
    euclidean_distance,
    extract_instruction,
    filter_episodes,
    get_goal_position,
    get_reference_positions,
    habitat,
    load_dataset,
    parse_episode_ids,
    parse_gpu_ids,
    position_within_radius,
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
    ACTION_SEQUENCE_LENGTH,
    DEFAULT_MAX_MEMORY_IMAGES,
    DEFAULT_MEMORY_POOL_WINDOW_FRAMES,
    PanoVLN_Agent,
)


DEFAULT_ALPHA = 0.5
DEFAULT_DAGGER_GOAL_RADIUS = 0.25
DEFAULT_DAGGER_SUCCESS_RADIUS = 0.5
DEFAULT_MODEL_PATH = (
    "/workspace/data1/model/ablation_new/panovggt_new/panovggt_0.05_grouping_8card"
)
DEFAULT_OUTPUT_ROOT = "/workspace/data1/dataset/PanoVLN"
DEFAULT_DAGGER_DATASET_NAME = "dagger"
DEFAULT_SOURCE_DATASETS = ("r2r", "rxr")
QUEUE_POLL_TIMEOUT_SECONDS = 5
SAFE_DATASET_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")
SUPPORTED_ACTION_IDS = {
    STOP_ACTION,
    MOVE_FORWARD_ACTION,
    TURN_LEFT_ACTION,
    TURN_RIGHT_ACTION,
}
DEFAULT_DAGGER_ID_OFFSETS = {
    "r2r": 0,
    "rxr": 1_000_000,
    "envdrop": 2_000_000,
    "scalevln": 3_000_000,
    "scalevln_150k": 4_000_000,
}


class DAggerCollectionError(RuntimeError):
    pass


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


def summary_path(output_root: str, dagger_dataset_name: str) -> str:
    return os.path.join(output_root, "sub_dataset", f"{dagger_dataset_name}_summary.jsonl")


def image_root(output_root: str, dagger_dataset_name: str) -> str:
    return os.path.join(output_root, "images", dagger_dataset_name)


def progress_dir_path(output_root: str, dagger_dataset_name: str) -> str:
    output_path = Path(annotation_path(output_root, dagger_dataset_name))
    return str(output_path.parent / f"{output_path.stem}_progress")


def rank_annotation_path(progress_dir: str, worker_index: int) -> str:
    return os.path.join(progress_dir, f"rank_{worker_index:03d}.jsonl")


def rank_summary_path(progress_dir: str, worker_index: int) -> str:
    return os.path.join(progress_dir, f"rank_{worker_index:03d}.summary.jsonl")


def list_rank_paths(progress_dir: str, suffix: str) -> List[str]:
    if not os.path.isdir(progress_dir):
        return []
    return sorted(
        os.path.join(progress_dir, file_name)
        for file_name in os.listdir(progress_dir)
        if file_name.startswith("rank_") and file_name.endswith(suffix)
    )


def remove_if_exists(path: str) -> None:
    if os.path.isdir(path):
        shutil.rmtree(path)
    elif os.path.exists(path):
        os.remove(path)


def load_jsonl_index(path: str, tolerate_trailing_corrupt: bool = True) -> Dict[int, Dict]:
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
            index[int(item["episode_id"])] = item
    return index


def write_jsonl_index(path: str, index: Dict[int, Dict]) -> None:
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


def count_saved_frames(episode_image_dir: str) -> int:
    if not os.path.isdir(episode_image_dir):
        return 0
    return sum(
        1
        for file_name in os.listdir(episode_image_dir)
        if file_name.startswith("frame_") and file_name.endswith(".jpg")
    )


def expected_frame_path(episode_image_dir: str, frame_index: int) -> str:
    return os.path.join(episode_image_dir, f"frame_{frame_index}.jpg")


def image_file_is_valid(path: str) -> bool:
    if not os.path.exists(path) or os.path.getsize(path) <= 0:
        return False
    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            return image.size == ERP_IMAGE_SIZE and image.mode == "RGB"
    except Exception:
        return False


def is_raw_episode_complete(output_root: str, dagger_dataset_name: str, annotation: Dict) -> bool:
    episode_image_dir = os.path.join(
        image_root(output_root, dagger_dataset_name),
        str(annotation["episode_id"]),
    )
    action_count = len(annotation.get("actions", []))
    if action_count <= 0 or not os.path.isdir(episode_image_dir):
        return False
    frame_names = [
        file_name
        for file_name in os.listdir(episode_image_dir)
        if file_name.startswith("frame_") and file_name.endswith(".jpg")
    ]
    if len(frame_names) != action_count:
        return False
    for frame_index in range(action_count):
        if not image_file_is_valid(expected_frame_path(episode_image_dir, frame_index)):
            return False
    return True


def validate_safe_dataset_name(name: str, arg_name: str = "dagger_dataset_name") -> None:
    if not name or not SAFE_DATASET_NAME_PATTERN.fullmatch(name):
        raise ValueError(
            f"{arg_name} must be a non-empty basename containing only "
            f"letters, digits, '_', '-', or '.', got {name!r}"
        )


def assert_path_inside(path: str, parent: str) -> None:
    resolved_path = Path(path).resolve()
    resolved_parent = Path(parent).resolve()
    try:
        resolved_path.relative_to(resolved_parent)
    except ValueError as exc:
        raise ValueError(f"Refusing to operate on path outside {resolved_parent}: {resolved_path}") from exc


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


def dagger_episode_id(source_dataset: str, source_episode_id) -> int:
    try:
        source_id = int(source_episode_id)
    except (TypeError, ValueError):
        digest = hashlib.sha1(f"{source_dataset}:{source_episode_id}".encode("utf-8")).hexdigest()
        source_id = int(digest[:8], 16)
    offset = DEFAULT_DAGGER_ID_OFFSETS.get(source_dataset)
    if offset is None:
        dataset_index = int(hashlib.sha1(source_dataset.encode("utf-8")).hexdigest()[:4], 16)
        offset = 10_000_000 + dataset_index * 1_000_000
    return offset + source_id


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


def advance_target_index(env, target_positions, next_target_index: int, goal_radius: float) -> int:
    current_position = to_position_list(env.sim.get_agent_state().position)
    while next_target_index < len(target_positions) and position_within_radius(
        current_position,
        target_positions[next_target_index],
        goal_radius,
    ):
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
    if position_within_radius(current_position, target_position, goal_radius):
        return STOP_ACTION

    action = action_to_int(follower.get_next_action(target_position))
    if action == STOP_ACTION:
        current_position = to_position_list(env.sim.get_agent_state().position)
        if not position_within_radius(current_position, target_position, goal_radius):
            raise DAggerCollectionError(
                "ShortestPathFollower returned stop before target: "
                f"distance={euclidean_distance(current_position, target_position):.3f}, "
                f"goal_radius={goal_radius}"
            )
    if action not in SUPPORTED_ACTION_IDS:
        raise DAggerCollectionError(f"Unsupported oracle action: {action}")
    return action


def oracle_action_for_current_state(
    env,
    follower,
    target_positions,
    next_target_index: int,
    goal_radius: float,
) -> Tuple[int, int]:
    next_target_index = advance_target_index(
        env,
        target_positions,
        next_target_index,
        goal_radius,
    )
    if next_target_index >= len(target_positions):
        return STOP_ACTION, next_target_index
    return (
        next_oracle_action(
            env,
            follower,
            target_positions[next_target_index],
            goal_radius,
        ),
        next_target_index,
    )


def preview_oracle_sequence(
    env,
    follower,
    target_positions,
    next_target_index: int,
    goal_radius: float,
    action_horizon: int,
) -> List[int]:
    saved_state = capture_agent_state(env)
    preview_target_index = next_target_index
    actions: List[int] = []
    try:
        for _ in range(action_horizon):
            action, preview_target_index = oracle_action_for_current_state(
                env,
                follower,
                target_positions,
                preview_target_index,
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


def save_episode_frame(episode_image_dir: str, frame_index: int, rgb) -> None:
    save_rgb_frame(rgb, os.path.join(episode_image_dir, f"frame_{frame_index}.jpg"))


def executable_prefix(action_ids: Sequence[int], execute_horizon: int) -> List[int]:
    prefix = list(action_ids[: max(1, int(execute_horizon))])
    if not prefix:
        return []
    if STOP_ACTION in prefix:
        return prefix[: prefix.index(STOP_ACTION) + 1]
    return prefix


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
):
    observation = env.step(int(action_id))
    if int(action_id) == STOP_ACTION:
        return observation, frame_index
    save_episode_frame(episode_image_dir, frame_index, observation["rgb"])
    append_history_image(agent, observation)
    return observation, frame_index + 1


def collect_raw_dagger_episode(
    env,
    agent: PanoVLN_Agent,
    source_dataset: str,
    episode,
    dagger_id: int,
    output_root: str,
    dagger_dataset_name: str,
    reference_steps: int,
    rng: random.Random,
    goal_radius: float,
    success_radius: float,
    alpha: float,
    action_horizon: int,
    execute_horizon: int,
    max_steps_per_episode: int,
) -> Tuple[Optional[Dict], Dict]:
    env.current_episode = episode
    observation = env.reset()
    agent.reset()
    follower = ShortestPathFollower(
        env.sim,
        goal_radius=goal_radius,
        return_one_hot=False,
        stop_on_error=True,
    )

    instruction = extract_instruction(episode)
    if not instruction and "instruction" in observation:
        instruction = observation["instruction"].get("text", "")

    episode_image_dir = os.path.join(
        image_root(output_root, dagger_dataset_name),
        str(dagger_id),
    )
    reset_episode_output_dir(episode_image_dir)
    save_episode_frame(episode_image_dir, 0, observation["rgb"])
    append_history_image(agent, observation)

    start_position = to_position_list(env.sim.get_agent_state().position)
    current_position = list(start_position)
    target_positions, next_target_index = initial_target_state(episode, current_position)

    expert_actions: List[int] = []
    executed_actions: List[int] = []
    model_outputs: List[str] = []
    beta_values: List[float] = []
    policy_counts = {"oracle": 0, "model": 0, "model_fallback_oracle": 0, "terminal_oracle": 0}
    frame_index = 1
    env_steps = 0
    replan_index = 0
    status = "max_steps"
    executed_path_length = 0.0
    start_time = time.time()

    while not env.episode_over and env_steps < max_steps_per_episode:
        next_target_index = advance_target_index(
            env,
            target_positions,
            next_target_index,
            goal_radius,
        )
        beta = dynamic_oracle_probability(
            step_index=env_steps,
            reference_steps=reference_steps,
            alpha=alpha,
        )
        oracle_sequence = preview_oracle_sequence(
            env=env,
            follower=follower,
            target_positions=target_positions,
            next_target_index=next_target_index,
            goal_radius=goal_radius,
            action_horizon=action_horizon,
        )
        beta_values.append(beta)

        oracle_queue = executable_prefix(oracle_sequence, execute_horizon)
        if not oracle_queue:
            oracle_queue = [STOP_ACTION]

        if oracle_sequence[0] == STOP_ACTION:
            executed_policy = "terminal_oracle"
            selected_actions = [STOP_ACTION]
        elif rng.random() < beta:
            executed_policy = "oracle"
            selected_actions = oracle_queue
        else:
            executed_policy = "model"
            model_text, model_action_ids = model_predict_action_sequence(agent, instruction)
            model_outputs.append(model_text)
            model_queue = executable_prefix(model_action_ids, execute_horizon)
            selected_actions = model_queue
            model_stops_early = (
                bool(selected_actions)
                and selected_actions[0] == STOP_ACTION
                and oracle_sequence[0] != STOP_ACTION
            )
            if not selected_actions or model_stops_early:
                executed_policy = "model_fallback_oracle"
                selected_actions = oracle_queue
        policy_counts[executed_policy] = policy_counts.get(executed_policy, 0) + 1

        for selected_action in selected_actions:
            if env_steps >= max_steps_per_episode:
                break

            oracle_action, next_target_index = oracle_action_for_current_state(
                env,
                follower,
                target_positions,
                next_target_index,
                goal_radius,
            )
            expert_actions.append(int(oracle_action))

            action_to_execute = int(selected_action)
            if action_to_execute == STOP_ACTION and oracle_action != STOP_ACTION:
                action_to_execute = int(oracle_action)
            if oracle_action == STOP_ACTION:
                action_to_execute = STOP_ACTION

            executed_actions.append(action_to_execute)
            previous_position = to_position_list(env.sim.get_agent_state().position)
            observation, frame_index = execute_and_record_one_action(
                env=env,
                agent=agent,
                episode_image_dir=episode_image_dir,
                frame_index=frame_index,
                action_id=action_to_execute,
            )
            current_position = to_position_list(env.sim.get_agent_state().position)
            executed_path_length += euclidean_distance(previous_position, current_position)
            env_steps += 1

            if action_to_execute == STOP_ACTION:
                status = "terminal"
                break
            if env.episode_over:
                status = "env_episode_over"
                break

        replan_index += 1

    final_position = to_position_list(env.sim.get_agent_state().position)
    final_goal_position = get_goal_position(episode)
    final_distance_to_goal = euclidean_distance(final_position, final_goal_position)
    episode_success = final_distance_to_goal <= float(success_radius)
    terminal_ready = final_distance_to_goal <= float(goal_radius)
    start_goal_distance = euclidean_distance(start_position, final_goal_position)
    path_efficiency = start_goal_distance / max(start_goal_distance, executed_path_length, 1e-8)

    save_episode = bool(episode_success and terminal_ready)
    if not save_episode:
        if not episode_success:
            status = f"not_saved_outside_success_radius:{status}"
        else:
            status = f"not_saved_outside_goal_radius:{status}"
        remove_if_exists(episode_image_dir)
        summary = {
            "episode_id": int(dagger_id),
            "source_dataset": source_dataset,
            "source_episode_id": str(episode.episode_id),
            "scene_id": getattr(episode, "scene_id", ""),
            "status": status,
            "success": bool(episode_success),
            "saved": False,
            "terminal_ready": bool(terminal_ready),
            "goal_radius": float(goal_radius),
            "success_radius": float(success_radius),
            "final_distance_to_goal": float(final_distance_to_goal),
            "start_goal_distance": float(start_goal_distance),
            "executed_path_length": float(executed_path_length),
            "path_efficiency": float(path_efficiency),
            "reference_steps": int(reference_steps),
            "expert_steps": len(expert_actions),
            "executed_steps": len(executed_actions),
            "saved_frames": 0,
            "replans": replan_index,
            "dagger_alpha": float(alpha),
            "dagger_beta_values": beta_values,
            "policy_counts": policy_counts,
            "executed_actions": executed_actions,
            "model_outputs": model_outputs,
            "seconds": time.time() - start_time,
        }
        return None, summary

    if not expert_actions or expert_actions[-1] != STOP_ACTION:
        expert_actions.append(STOP_ACTION)
        executed_actions.append(STOP_ACTION)
        if not env.episode_over:
            env.step(STOP_ACTION)
        status = "max_steps_stop" if status == "max_steps" else status

    annotation = {
        "episode_id": int(dagger_id),
        "instruction": instruction,
        "actions": expert_actions,
    }
    summary = {
        "episode_id": int(dagger_id),
        "source_dataset": source_dataset,
        "source_episode_id": str(episode.episode_id),
        "scene_id": getattr(episode, "scene_id", ""),
        "status": status,
        "success": bool(episode_success),
        "saved": True,
        "terminal_ready": bool(terminal_ready),
        "goal_radius": float(goal_radius),
        "success_radius": float(success_radius),
        "final_distance_to_goal": float(final_distance_to_goal),
        "start_goal_distance": float(start_goal_distance),
        "executed_path_length": float(executed_path_length),
        "path_efficiency": float(path_efficiency),
        "reference_steps": int(reference_steps),
        "expert_steps": len(expert_actions),
        "executed_steps": len(executed_actions),
        "saved_frames": count_saved_frames(episode_image_dir),
        "replans": replan_index,
        "dagger_alpha": float(alpha),
        "dagger_beta_values": beta_values,
        "policy_counts": policy_counts,
        "executed_actions": executed_actions,
        "model_outputs": model_outputs,
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
    source_dataset,
    episode_ids,
    dagger_ids,
    model_path,
    output_root,
    dagger_dataset_name,
    reference_input_root,
    runtime_root,
    partial_annotation_path,
    partial_summary_path,
    worker_index,
    local_gpu_id,
    display_gpu_id,
    process_index_on_gpu,
    goal_radius,
    success_radius,
    alpha,
    action_horizon,
    execute_horizon,
    max_steps_per_episode,
    seed,
    attn_implementation,
    max_memory_images,
    memory_pool_window_frames,
    skip_failed_episodes,
):
    env = None
    try:
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
        )

        with silence_external_output():
            env = habitat.Env(config=env_config.habitat, dataset=dataset)

        os.makedirs(os.path.dirname(partial_annotation_path), exist_ok=True)
        rng = random.Random(seed + worker_index)
        with open(partial_annotation_path, "a", encoding="utf-8") as annotation_handle, open(
            partial_summary_path, "a", encoding="utf-8"
        ) as summary_handle:
            for episode in dataset.episodes:
                episode_start_time = time.time()
                dagger_id = int(dagger_ids[int(episode.episode_id)])
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
                        goal_radius=goal_radius,
                        success_radius=success_radius,
                        alpha=alpha,
                        action_horizon=action_horizon,
                        execute_horizon=execute_horizon,
                        max_steps_per_episode=max_steps_per_episode,
                    )
                    write_jsonl_item(summary_handle, summary)
                    if annotation is None:
                        result_status = "not_saved"
                        sample_count = 0
                    else:
                        write_jsonl_item(annotation_handle, annotation)
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
                            "summary_status": summary["status"],
                            "time_per_episode": time.time() - episode_start_time,
                        }
                    )
                except Exception:
                    error = traceback.format_exc()
                    summary = {
                        "episode_id": dagger_id,
                        "source_dataset": source_dataset,
                        "source_episode_id": str(episode.episode_id),
                        "status": "error",
                        "error": error,
                    }
                    write_jsonl_item(summary_handle, summary)
                    if not skip_failed_episodes:
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
                        }
                    )
    except Exception:
        result_queue.put(
            {
                "status": "error",
                "worker_index": worker_index,
                "display_gpu_id": display_gpu_id,
                "process_index_on_gpu": process_index_on_gpu,
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


def load_complete_existing_annotations(output_root, dagger_dataset_name, progress_dir=None):
    existing: Dict[int, Dict] = {}
    for annotation in load_jsonl_index(annotation_path(output_root, dagger_dataset_name)).values():
        if is_raw_episode_complete(output_root, dagger_dataset_name, annotation):
            existing[int(annotation["episode_id"])] = annotation
    if progress_dir is not None:
        for path in list_rank_paths(progress_dir, ".jsonl"):
            if path.endswith(".summary.jsonl"):
                continue
            for annotation in load_jsonl_index(path).values():
                if not is_raw_episode_complete(output_root, dagger_dataset_name, annotation):
                    continue
                episode_id = int(annotation["episode_id"])
                if episode_id in existing and existing[episode_id] != annotation:
                    raise ValueError(
                        f"Conflicting complete annotations for episode_id={episode_id} "
                        f"while loading existing progress from {path}"
                    )
                existing[episode_id] = annotation
    return existing


def add_complete_annotation(
    merged_annotations: Dict[int, Dict],
    output_root: str,
    dagger_dataset_name: str,
    annotation: Dict,
    source_path: str,
) -> None:
    if not is_raw_episode_complete(output_root, dagger_dataset_name, annotation):
        tqdm.write(
            f"[dagger] skipping incomplete annotation episode_id="
            f"{annotation.get('episode_id')} from {source_path}"
        )
        return
    episode_id = int(annotation["episode_id"])
    existing = merged_annotations.get(episode_id)
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
    existing_annotations: Dict[int, Dict],
):
    merged_annotations: Dict[int, Dict] = {}
    for annotation in existing_annotations.values():
        add_complete_annotation(
            merged_annotations=merged_annotations,
            output_root=output_root,
            dagger_dataset_name=dagger_dataset_name,
            annotation=annotation,
            source_path="existing_annotations",
        )
    for path in list_rank_paths(progress_dir, ".jsonl"):
        if path.endswith(".summary.jsonl"):
            continue
        for annotation in load_jsonl_index(path).values():
            add_complete_annotation(
                merged_annotations=merged_annotations,
                output_root=output_root,
                dagger_dataset_name=dagger_dataset_name,
                annotation=annotation,
                source_path=path,
            )
    write_jsonl_index(annotation_path(output_root, dagger_dataset_name), merged_annotations)

    merged_summaries = load_jsonl_index(summary_path(output_root, dagger_dataset_name))
    for path in list_rank_paths(progress_dir, ".summary.jsonl"):
        merged_summaries.update(load_jsonl_index(path))
    write_jsonl_index(summary_path(output_root, dagger_dataset_name), merged_summaries)


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
    goal_radius,
    success_radius,
    alpha,
    action_horizon,
    execute_horizon,
    max_steps_per_episode,
    seed,
    attn_implementation,
    max_memory_images,
    memory_pool_window_frames,
    skip_failed_episodes,
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

    existing_annotations = load_complete_existing_annotations(
        output_root,
        dagger_dataset_name,
        progress_dir=progress_dir,
    )
    pending_episodes = []
    dagger_ids = {}
    skipped = 0
    for episode in selected_episodes:
        dagger_id = dagger_episode_id(source_dataset, episode.episode_id)
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
        partial_summary_path = rank_summary_path(progress_dir, assignment["worker_index"])
        worker_dagger_ids = {
            episode_id: dagger_ids[episode_id]
            for episode_id in job["episode_ids"]
        }
        process = ctx.Process(
            target=dagger_worker,
            args=(
                result_queue,
                source_dataset,
                job["episode_ids"],
                worker_dagger_ids,
                model_path,
                output_root,
                dagger_dataset_name,
                reference_input_root,
                os.path.join(progress_dir, "runtime"),
                partial_annotation_path,
                partial_summary_path,
                assignment["worker_index"],
                assignment["local_gpu_id"],
                assignment["display_gpu_id"],
                assignment["process_index_on_gpu"],
                goal_radius,
                success_radius,
                alpha,
                action_horizon,
                execute_horizon,
                max_steps_per_episode,
                seed,
                attn_implementation,
                max_memory_images,
                memory_pool_window_frames,
                skip_failed_episodes,
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
                continue

            if result["status"] == "error":
                raise RuntimeError(
                    f"[dagger:{source_dataset}] worker "
                    f"gpu{result['display_gpu_id']} "
                    f"p{result['process_index_on_gpu']} failed:\n{result['error']}"
                )

            if result["status"] == "skipped":
                failed += 1
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
                f"status={result.get('summary_status', result['status'])}, "
                f"{result['time_per_episode']:.2f}s"
            )

        for process in processes:
            process.join()
            if process.exitcode != 0:
                raise RuntimeError(
                    f"[dagger:{source_dataset}] worker pid={process.pid} "
                    f"exited with code {process.exitcode}"
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
    if args.action_horizon != ACTION_SEQUENCE_LENGTH:
        raise ValueError(
            f"action_horizon must be {ACTION_SEQUENCE_LENGTH} to match "
            "the current training prompt."
        )
    if args.execute_horizon <= 0 or args.execute_horizon > args.action_horizon:
        raise ValueError(
            "execute_horizon must be positive and no larger than action_horizon, "
            f"got {args.execute_horizon}"
        )
    if args.max_steps_per_episode <= 0:
        raise ValueError(
            f"max_steps_per_episode must be positive, got {args.max_steps_per_episode}"
        )
    if not math.isfinite(args.goal_radius) or args.goal_radius <= 0:
        raise ValueError(f"goal_radius must be positive, got {args.goal_radius}")
    if not math.isfinite(args.success_radius) or args.success_radius <= 0:
        raise ValueError(f"success_radius must be positive, got {args.success_radius}")
    if not math.isfinite(args.alpha):
        raise ValueError(f"alpha must be finite, got {args.alpha}")
    dynamic_oracle_probability(0, 1, args.alpha)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Collect raw PanoVLN DAgger data with Efficient-VLN dynamic ratio. "
            "Outputs sub_dataset/<dagger>.jsonl and images/<dagger>/."
        )
    )
    parser.add_argument("--source_dataset_name", nargs="+", default=list(DEFAULT_SOURCE_DATASETS))
    parser.add_argument("--dagger_dataset_name", type=str, default=DEFAULT_DAGGER_DATASET_NAME)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--reference_input_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--gpu_ids", type=str, default=None)
    parser.add_argument("--num_thread", type=int, default=1)
    parser.add_argument("--num_processes_per_gpu", type=int, default=None)
    parser.add_argument("--episode_ids", nargs="*", default=None)
    parser.add_argument("--max_episodes", type=int, default=None)
    parser.add_argument("--max_steps_per_episode", type=int, default=500)
    parser.add_argument("--goal_radius", type=float, default=DEFAULT_DAGGER_GOAL_RADIUS)
    parser.add_argument(
        "--success_radius",
        type=float,
        default=DEFAULT_DAGGER_SUCCESS_RADIUS,
        help="Maximum final distance to goal for keeping a DAgger episode.",
    )
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument("--action_horizon", type=int, default=ACTION_SEQUENCE_LENGTH)
    parser.add_argument("--execute_horizon", type=int, default=ACTION_SEQUENCE_LENGTH)
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
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validate_args(args)
    seed_all(args.seed)

    requested_gpu_ids = parse_gpu_ids(args.gpu_ids)
    visible_gpu_ids = None
    if requested_gpu_ids is not None:
        visible_gpu_ids = remap_gpu_ids_to_visible_devices(requested_gpu_ids)
        validate_gpu_ids(visible_gpu_ids)
    if args.num_processes_per_gpu is not None and requested_gpu_ids is None:
        raise ValueError("--num_processes_per_gpu requires --gpu_ids")

    selected_episode_ids = parse_episode_ids(args.episode_ids)
    os.makedirs(os.path.join(args.output_root, "sub_dataset"), exist_ok=True)
    os.makedirs(image_root(args.output_root, args.dagger_dataset_name), exist_ok=True)
    assert_path_inside(annotation_path(args.output_root, args.dagger_dataset_name), args.output_root)
    assert_path_inside(summary_path(args.output_root, args.dagger_dataset_name), args.output_root)
    assert_path_inside(image_root(args.output_root, args.dagger_dataset_name), args.output_root)

    progress_dir = progress_dir_path(
        output_root=args.output_root,
        dagger_dataset_name=args.dagger_dataset_name,
    )
    assert_path_inside(progress_dir, args.output_root)
    if args.skip_existing_episodes:
        os.makedirs(progress_dir, exist_ok=True)
        lock_path = acquire_progress_lock(progress_dir)
        existing_annotations = load_complete_existing_annotations(
            output_root=args.output_root,
            dagger_dataset_name=args.dagger_dataset_name,
            progress_dir=progress_dir,
        )
    else:
        assert_no_active_progress_lock(progress_dir)
        remove_if_exists(progress_dir)
        os.makedirs(progress_dir, exist_ok=True)
        lock_path = acquire_progress_lock(progress_dir)
        existing_annotations = {}
        remove_if_exists(annotation_path(args.output_root, args.dagger_dataset_name))
        remove_if_exists(summary_path(args.output_root, args.dagger_dataset_name))
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
        f"sources={args.source_dataset_name}"
    )
    print(
        f"gpu_ids={args.gpu_ids}, num_thread={args.num_thread}, "
        f"num_processes_per_gpu={args.num_processes_per_gpu}, "
        f"goal_radius={args.goal_radius}, success_radius={args.success_radius}, "
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
                goal_radius=args.goal_radius,
                success_radius=args.success_radius,
                alpha=args.alpha,
                action_horizon=args.action_horizon,
                execute_horizon=args.execute_horizon,
                max_steps_per_episode=args.max_steps_per_episode,
                seed=args.seed,
                attn_implementation=args.attn_implementation,
                max_memory_images=args.max_memory_images,
                memory_pool_window_frames=args.memory_pool_window_frames,
                skip_failed_episodes=args.skip_failed_episodes,
            )

        merge_partial_outputs(
            output_root=args.output_root,
            dagger_dataset_name=args.dagger_dataset_name,
            progress_dir=progress_dir,
            existing_annotations=existing_annotations,
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
    print(f"saved collection summary to {summary_path(args.output_root, args.dagger_dataset_name)}")


if __name__ == "__main__":
    main()
