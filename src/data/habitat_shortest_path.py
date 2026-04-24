import math
import os
import sys
import warnings
from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple

import PIL.Image as Image
import torch

warnings.filterwarnings("ignore")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


DEFAULT_GOAL_RADIUS = 0.5
ERP_IMAGE_SIZE = (1280, 640)
LOCALITY_BLOCK_SIZE_MULTIPLIER = 2.0

CONFIG = {
    "r2r": {
        "config_path": "./config/vln_r2r_train.yaml",
        "image_dir": "r2r",
        "annotation_name": "r2r.jsonl",
        "default_split": "train",
    },
    "rxr": {
        "config_path": "./config/vln_rxr_train.yaml",
        "image_dir": "rxr",
        "annotation_name": "rxr.jsonl",
        "default_split": "train",
    },
    "envdrop": {
        "config_path": "./config/vln_envdrop.yaml",
        "image_dir": "envdrop",
        "annotation_name": "envdrop.jsonl",
        "default_split": "envdrop",
    },
    "scalevln": {
        "config_path": "./config/vln_scalevln.yaml",
        "image_dir": "scalevln",
        "annotation_name": "scalevln.jsonl",
    },
    "scalevln_150k": {
        "config_path": "./config/vln_scalevln.yaml",
        "image_dir": "scalevln_150k",
        "annotation_name": "scalevln_150k.jsonl",
    },
}


@contextmanager
def silence_external_output(enabled: bool = True):
    if not enabled:
        yield
        return

    stdout_fd = os.dup(1)
    stderr_fd = os.dup(2)

    with open(os.devnull, "w") as devnull:
        try:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)
            yield
        finally:
            sys.stdout.flush()
            sys.stderr.flush()
            os.dup2(stdout_fd, 1)
            os.dup2(stderr_fd, 2)
            os.close(stdout_fd)
            os.close(stderr_fd)


with silence_external_output():
    import habitat
    from habitat.config.default import get_config
    from habitat.sims.habitat_simulator.actions import HabitatSimActions
    from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

    from habitat_extensions import measures, task


STOP_ACTION = int(HabitatSimActions.stop)
MOVE_FORWARD_ACTION = int(HabitatSimActions.move_forward)
TURN_LEFT_ACTION = int(HabitatSimActions.turn_left)
TURN_RIGHT_ACTION = int(HabitatSimActions.turn_right)
SUPPORTED_ACTION_IDS = {
    STOP_ACTION,
    MOVE_FORWARD_ACTION,
    TURN_LEFT_ACTION,
    TURN_RIGHT_ACTION,
}


def resolve_existing_path(candidates: Iterable[str]) -> str:
    for path in candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "None of the candidate paths exist:\n" + "\n".join(candidates)
    )


def default_output_path(output_root: str, dataset_name: str) -> str:
    return os.path.join(output_root, "sub_dataset", f"{dataset_name}.jsonl")


def parse_gpu_ids(raw_gpu_ids) -> Optional[List[int]]:
    if raw_gpu_ids is None:
        return None

    if isinstance(raw_gpu_ids, str):
        gpu_tokens = raw_gpu_ids.replace(",", " ").split()
    else:
        gpu_tokens = []
        for item in raw_gpu_ids:
            gpu_tokens.extend(str(item).replace(",", " ").split())

    if not gpu_tokens:
        raise ValueError("gpu_ids cannot be empty")

    return [int(token) for token in gpu_tokens]


def remap_gpu_ids_to_visible_devices(gpu_ids: Optional[Sequence[int]]) -> Optional[List[int]]:
    if gpu_ids is None:
        return None

    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if not visible_devices:
        return list(gpu_ids)

    visible_tokens = [token.strip() for token in visible_devices.split(",") if token.strip()]
    if not visible_tokens:
        return list(gpu_ids)

    visible_index_by_token = {
        token: index for index, token in enumerate(visible_tokens)
    }

    if all(str(gpu_id) in visible_index_by_token for gpu_id in gpu_ids):
        return [visible_index_by_token[str(gpu_id)] for gpu_id in gpu_ids]

    if all(0 <= gpu_id < len(visible_tokens) for gpu_id in gpu_ids):
        return list(gpu_ids)

    raise ValueError(
        f"Requested gpu_ids={list(gpu_ids)} are incompatible with "
        f"CUDA_VISIBLE_DEVICES={visible_devices}"
    )


def validate_gpu_ids(gpu_ids: Optional[Sequence[int]]) -> None:
    if not gpu_ids:
        raise ValueError("At least one GPU id is required")

    if any(gpu_id < 0 for gpu_id in gpu_ids):
        raise ValueError(f"gpu_ids must be non-negative, got {list(gpu_ids)}")

    visible_gpu_count = torch.cuda.device_count()
    if visible_gpu_count == 0:
        return

    invalid_gpu_ids = [
        gpu_id for gpu_id in gpu_ids if gpu_id >= visible_gpu_count
    ]
    if invalid_gpu_ids:
        raise ValueError(
            f"Visible GPU count is {visible_gpu_count}, invalid gpu_ids={invalid_gpu_ids}"
        )


def expand_gpu_ids_for_workers(
    gpu_ids: Sequence[int],
    num_processes_per_gpu: Optional[int],
) -> List[int]:
    if num_processes_per_gpu is None:
        num_processes_per_gpu = 1

    if num_processes_per_gpu <= 0:
        raise ValueError(
            f"num_processes_per_gpu must be positive, got {num_processes_per_gpu}"
        )

    worker_gpu_ids = []
    for gpu_id in gpu_ids:
        worker_gpu_ids.extend([gpu_id] * num_processes_per_gpu)
    return worker_gpu_ids


def build_worker_assignments(
    requested_gpu_ids: Optional[Sequence[int]],
    visible_gpu_ids: Optional[Sequence[int]],
    num_processes_per_gpu: Optional[int],
    max_workers: Optional[int] = None,
):
    if max_workers is not None and max_workers <= 0:
        raise ValueError(f"max_workers must be positive, got {max_workers}")

    if visible_gpu_ids is None:
        worker_count = 1 if max_workers is None else max_workers
        return [
            {
                "worker_index": worker_index,
                "local_gpu_id": 0,
                "display_gpu_id": 0,
                "process_index_on_gpu": worker_index,
            }
            for worker_index in range(worker_count)
        ]

    worker_local_gpu_ids = expand_gpu_ids_for_workers(
        gpu_ids=visible_gpu_ids,
        num_processes_per_gpu=num_processes_per_gpu,
    )
    worker_display_gpu_ids = expand_gpu_ids_for_workers(
        gpu_ids=requested_gpu_ids,
        num_processes_per_gpu=num_processes_per_gpu,
    )

    if max_workers is not None:
        worker_local_gpu_ids = worker_local_gpu_ids[:max_workers]
        worker_display_gpu_ids = worker_display_gpu_ids[:max_workers]

    process_indices = {}
    worker_assignments = []
    for worker_index, (local_gpu_id, display_gpu_id) in enumerate(
        zip(worker_local_gpu_ids, worker_display_gpu_ids)
    ):
        process_index_on_gpu = process_indices.get(display_gpu_id, 0)
        process_indices[display_gpu_id] = process_index_on_gpu + 1
        worker_assignments.append(
            {
                "worker_index": worker_index,
                "local_gpu_id": local_gpu_id,
                "display_gpu_id": display_gpu_id,
                "process_index_on_gpu": process_index_on_gpu,
            }
        )
    return worker_assignments


def resolve_dataset_paths(
    dataset_name: str,
    input_root: str,
    output_root: Optional[str] = None,
    dataset_split: Optional[str] = None,
) -> Tuple[Optional[str], Optional[str], str, str, str]:
    dataset_config = CONFIG[dataset_name]
    annotation_path = None
    image_path = None
    if output_root is not None:
        annotation_path = os.path.join(
            output_root, "sub_dataset", dataset_config["annotation_name"]
        )
        image_path = os.path.join(output_root, "images", dataset_config["image_dir"])

    dataset_split = dataset_split or dataset_config.get("default_split")
    has_janus_layout = os.path.isdir(os.path.join(input_root, "datasets")) and os.path.isdir(
        os.path.join(input_root, "scene_datasets")
    )

    if has_janus_layout:
        scene_root = os.path.join(input_root, "scene_datasets")
        scene_dataset = None

        if dataset_name == "r2r":
            if dataset_split is None:
                raise ValueError("dataset_split is required for janus r2r")
            data_path = os.path.join(
                input_root, "datasets", "r2r", dataset_split, f"{dataset_split}.json.gz"
            )
        elif dataset_name == "rxr":
            if dataset_split is None:
                raise ValueError("dataset_split is required for janus rxr")
            guide_suffix = "_guide" if not dataset_split.endswith("_guide") else ""
            data_path = os.path.join(
                input_root, "datasets", "rxr", dataset_split, f"{dataset_split}{guide_suffix}.json.gz"
            )
        elif dataset_name == "envdrop":
            data_path = os.path.join(
                input_root, "datasets", "r2r", "envdrop", "envdrop.json.gz"
            )
        else:
            data_path = resolve_existing_path(
                [
                    os.path.join(input_root, "datasets", "scalevln", "scalevln_subset_150k.json.gz"),
                    os.path.join(input_root, "datasets", "scalevln", "scalevln_subset_150k.json"),
                    os.path.join(input_root, "datasets", "scalevln", "scalevln_150k", "scalevln_subset_150k.json.gz"),
                ]
            )
    else:
        if dataset_name in {"r2r", "rxr", "envdrop"}:
            scene_root = os.path.join(input_root, "Matterport3D", "mp3d_habitat")
            scene_dataset = os.path.join(scene_root, "mp3d_scene_dataset_config.json")
        else:
            scene_root = os.path.join(input_root, "HM3D")
            scene_dataset = os.path.join(
                scene_root, "hm3d_annotated_basis.scene_dataset_config.json"
            )

        if dataset_name == "r2r":
            if dataset_split is None:
                raise ValueError("dataset_split is required for r2r")
            data_path = os.path.join(
                input_root,
                "R2R_VLNCE_v1-3_preprocessed",
                dataset_split,
                f"{dataset_split}.json.gz",
            )
        elif dataset_name == "rxr":
            if dataset_split is None:
                raise ValueError("dataset_split is required for rxr")
            guide_suffix = "_guide" if not dataset_split.endswith("_guide") else ""
            data_path = os.path.join(
                input_root,
                "RxR_VLNCE_v0",
                dataset_split,
                f"{dataset_split}{guide_suffix}.json.gz",
            )
        elif dataset_name == "envdrop":
            data_path = os.path.join(
                input_root, "R2R_VLNCE_v1-3_preprocessed", "envdrop", "envdrop.json.gz"
            )
        else:
            data_path = resolve_existing_path(
                [
                    os.path.join(
                        input_root, "ScaleVLN_150k", "scalevln_subset_150k.json.gz"
                    ),
                    os.path.join(input_root, "StreamVLN", "scalevln_subset_150k.json.gz"),
                ]
            )

    return annotation_path, image_path, scene_root, scene_dataset, data_path


def build_env_config(dataset_name: str, input_root: str, dataset_split: Optional[str] = None):
    config_path = os.path.join(
        PROJECT_ROOT,
        CONFIG[dataset_name]["config_path"].lstrip("./"),
    )
    _, _, scene_root, scene_dataset, data_path = resolve_dataset_paths(
        dataset_name=dataset_name,
        input_root=input_root,
        dataset_split=dataset_split,
    )

    env_config = get_config(config_path)
    with habitat.config.read_write(env_config):
        env_config.habitat.dataset.scenes_dir = scene_root
        env_config.habitat.dataset.data_path = data_path
        if dataset_split is not None:
            env_config.habitat.dataset.split = dataset_split
        if scene_dataset and os.path.exists(scene_dataset):
            env_config.habitat.simulator.scene_dataset = scene_dataset
    return env_config


def load_dataset(dataset_name: str, input_root: str, dataset_split: Optional[str] = None):
    env_config = build_env_config(
        dataset_name=dataset_name,
        input_root=input_root,
        dataset_split=dataset_split,
    )
    dataset = habitat.datasets.make_dataset(
        id_dataset=env_config.habitat.dataset.type,
        config=env_config.habitat.dataset,
    )
    dataset.episodes = sorted(
        dataset.episodes, key=lambda episode: int(episode.episode_id)
    )
    return env_config, dataset


def parse_episode_ids(raw_episode_ids) -> Optional[List[int]]:
    if raw_episode_ids is None:
        return None

    if isinstance(raw_episode_ids, str):
        tokens = raw_episode_ids.replace(",", " ").split()
    else:
        tokens = []
        for item in raw_episode_ids:
            tokens.extend(str(item).replace(",", " ").split())

    if not tokens:
        return []

    return sorted({int(token) for token in tokens})


def filter_episodes(episodes, episode_ids=None, max_episodes=None):
    selected_episodes = list(episodes)

    if episode_ids is not None:
        episode_id_set = {int(episode_id) for episode_id in episode_ids}
        selected_episodes = [
            episode
            for episode in selected_episodes
            if int(episode.episode_id) in episode_id_set
        ]
        missing_episode_ids = sorted(
            episode_id_set
            - {int(episode.episode_id) for episode in selected_episodes}
        )
        if missing_episode_ids:
            raise ValueError(
                f"Missing episode ids for selection: {missing_episode_ids}"
            )

    if max_episodes is not None:
        selected_episodes = selected_episodes[:max_episodes]

    return selected_episodes


def group_episodes_by_scene(episodes):
    episodes_by_scene = {}
    for episode in episodes:
        episodes_by_scene.setdefault(episode.scene_id, []).append(episode)

    for scene_episodes in episodes_by_scene.values():
        scene_episodes.sort(key=lambda episode: int(episode.episode_id))

    return episodes_by_scene


def sort_episodes_for_scene_locality(episodes):
    episodes_by_scene = group_episodes_by_scene(episodes)
    ordered_episodes = []
    for scene_id in sorted(episodes_by_scene):
        ordered_episodes.extend(episodes_by_scene[scene_id])
    return ordered_episodes


def split_items_evenly(items, num_splits):
    if num_splits <= 0:
        raise ValueError(f"num_splits must be positive, got {num_splits}")

    base_chunk_size, num_larger_chunks = divmod(len(items), num_splits)
    chunks = []
    start_index = 0
    for split_index in range(num_splits):
        chunk_size = base_chunk_size + (1 if split_index < num_larger_chunks else 0)
        chunks.append(items[start_index : start_index + chunk_size])
        start_index += chunk_size
    return chunks


def build_locality_balanced_episode_splits(
    episodes,
    num_workers,
    block_size_multiplier: float = LOCALITY_BLOCK_SIZE_MULTIPLIER,
):
    if num_workers <= 0:
        raise ValueError(f"num_workers must be positive, got {num_workers}")

    ordered_episodes = sort_episodes_for_scene_locality(episodes)
    if not ordered_episodes:
        return []

    effective_workers = min(num_workers, len(ordered_episodes))
    target_worker_size = max(1, math.ceil(len(ordered_episodes) / effective_workers))
    max_block_size = max(1, math.ceil(target_worker_size * block_size_multiplier))

    blocks = []
    for scene_id, scene_episodes in sorted(group_episodes_by_scene(ordered_episodes).items()):
        if len(scene_episodes) <= max_block_size:
            blocks.append(scene_episodes)
            continue

        num_scene_chunks = max(2, math.ceil(len(scene_episodes) / max_block_size))
        blocks.extend(
            chunk
            for chunk in split_items_evenly(scene_episodes, num_scene_chunks)
            if chunk
        )

    effective_workers = min(effective_workers, len(blocks))
    remaining_episode_count = sum(len(block) for block in blocks)
    block_index = 0
    episode_splits = []

    for worker_index in range(effective_workers):
        workers_left = effective_workers - worker_index
        target_chunk_size = max(1, math.ceil(remaining_episode_count / workers_left))
        worker_blocks = []
        worker_episode_count = 0

        while block_index < len(blocks):
            blocks_left = len(blocks) - block_index
            if worker_blocks and blocks_left <= workers_left - 1:
                break

            next_block = blocks[block_index]
            next_block_size = len(next_block)
            if worker_blocks and worker_episode_count + next_block_size > target_chunk_size:
                break

            worker_blocks.append(next_block)
            worker_episode_count += next_block_size
            block_index += 1

        if not worker_blocks:
            worker_blocks.append(blocks[block_index])
            worker_episode_count += len(blocks[block_index])
            block_index += 1

        worker_episodes = []
        for block in worker_blocks:
            worker_episodes.extend(block)
        episode_splits.append(worker_episodes)
        remaining_episode_count -= worker_episode_count

    if block_index < len(blocks):
        for block in blocks[block_index:]:
            episode_splits[-1].extend(block)

    return episode_splits


def extract_instruction(episode) -> str:
    instruction = getattr(episode, "instruction", None)
    if instruction is not None:
        if hasattr(instruction, "instruction_text"):
            return instruction.instruction_text
        if isinstance(instruction, dict) and "instruction_text" in instruction:
            return instruction["instruction_text"]
        if isinstance(instruction, str):
            return instruction
    return ""


def to_position_list(point) -> List[float]:
    if hasattr(point, "position"):
        point = point.position
    return [float(value) for value in point]


def get_goal_position(episode) -> List[float]:
    goals = getattr(episode, "goals", None)
    if goals:
        return to_position_list(goals[0])
    goal = getattr(episode, "goal", None)
    if goal is not None:
        return to_position_list(goal)
    raise ValueError(f"Episode {episode.episode_id} has no goal position")


def get_reference_positions(episode) -> List[List[float]]:
    reference_path = getattr(episode, "reference_path", None)
    if not reference_path:
        return []
    return [to_position_list(point) for point in reference_path]


def euclidean_distance(a: Sequence[float], b: Sequence[float]) -> float:
    return math.sqrt(
        sum((float(value_a) - float(value_b)) ** 2 for value_a, value_b in zip(a, b))
    )


def positions_equal(
    a: Sequence[float],
    b: Sequence[float],
    tol: float = 1e-4,
) -> bool:
    return euclidean_distance(a, b) <= tol


def action_to_int(action) -> int:
    if isinstance(action, str):
        mapping = {
            "stop": STOP_ACTION,
            "move_forward": MOVE_FORWARD_ACTION,
            "turn_left": TURN_LEFT_ACTION,
            "turn_right": TURN_RIGHT_ACTION,
        }
        if action not in mapping:
            raise ValueError(f"Unsupported action string: {action}")
        return mapping[action]

    if hasattr(action, "name"):
        return action_to_int(action.name)

    action_id = int(action)
    if action_id not in SUPPORTED_ACTION_IDS:
        raise ValueError(f"Unsupported action id: {action_id}")
    return action_id


def count_saved_frames(episode_image_path: str) -> int:
    if not os.path.isdir(episode_image_path):
        return 0

    return sum(
        1
        for file_name in os.listdir(episode_image_path)
        if file_name.startswith("frame_") and file_name.endswith(".jpg")
    )


def reset_episode_output_dir(episode_image_path: str) -> None:
    os.makedirs(episode_image_path, exist_ok=True)
    for file_name in os.listdir(episode_image_path):
        if file_name.startswith("frame_") and file_name.endswith(".jpg"):
            os.remove(os.path.join(episode_image_path, file_name))


def save_rgb_frame(rgb, output_path: str) -> None:
    rgb_frame = Image.fromarray(rgb)
    if rgb_frame.mode != "RGB":
        rgb_frame = rgb_frame.convert("RGB")
    if rgb_frame.size != ERP_IMAGE_SIZE:
        rgb_frame = rgb_frame.resize(ERP_IMAGE_SIZE)
    rgb_frame.save(output_path)


def rollout_shortest_path_episode(
    env,
    episode,
    goal_radius: float = DEFAULT_GOAL_RADIUS,
    episode_image_path: Optional[str] = None,
):
    env.current_episode = episode
    observation = env.reset()
    follower = ShortestPathFollower(
        env.sim,
        goal_radius=goal_radius,
        return_one_hot=False,
        stop_on_error=True,
    )

    if episode_image_path is not None:
        reset_episode_output_dir(episode_image_path)
        save_rgb_frame(observation["rgb"], os.path.join(episode_image_path, "frame_0.jpg"))

    actions: List[int] = []
    current_position = to_position_list(env.sim.get_agent_state().position)
    target_positions = get_reference_positions(episode)
    if target_positions:
        next_target_index = 1 if positions_equal(target_positions[0], current_position) else 0
    else:
        target_positions = [get_goal_position(episode)]
        next_target_index = 0

    saved_frame_index = 0
    while next_target_index < len(target_positions):
        target_position = target_positions[next_target_index]
        next_action = action_to_int(follower.get_next_action(target_position))

        if next_action == STOP_ACTION:
            next_target_index += 1
            continue

        observation = env.step(next_action)
        actions.append(next_action)
        saved_frame_index += 1
        if episode_image_path is not None:
            save_rgb_frame(
                observation["rgb"],
                os.path.join(episode_image_path, f"frame_{saved_frame_index}.jpg"),
            )

        if env.episode_over:
            raise RuntimeError(
                f"Episode {episode.episode_id} terminated before final stop action"
            )

    actions.append(STOP_ACTION)
    env.step(STOP_ACTION)

    return {
        "episode_id": int(episode.episode_id),
        "instruction": extract_instruction(episode),
        "actions": actions,
    }
