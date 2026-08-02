import argparse
import json
import os
import random
import tempfile
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm


DEFAULT_ACTION_HORIZON = 4
DEFAULT_SEED = 42
DEFAULT_SUBSET_SEED = 42
EBS_SAMPLER = "ebs"
DEFAULT_EVENT_KEEP_PROB = 0.60
DEFAULT_BACKGROUND_KEEP_PROB = 0.11
DEFAULT_TAIL_DENSE_KEEP_PROB = 0.50
DEFAULT_BODY_KEEP_ADVANCE = DEFAULT_ACTION_HORIZON
STOP_ACTION_ID = 0
EVENT_ACTION_IDS = {2, 3}

DATASET_SPECS = {
    "r2r": {"image_dir": "r2r", "annotation_name": "r2r.jsonl"},
    "rxr": {"image_dir": "rxr", "annotation_name": "rxr.jsonl"},
    "envdrop": {"image_dir": "envdrop", "annotation_name": "envdrop.jsonl"},
    "scalevln": {
        "image_dir": "scalevln",
        "annotation_name": "scalevln.jsonl",
        "dataset_label": "scalevln",
    },
    # Instruction-only ablation: reuse the exact ScaleVLN trajectories/images.
    "scalevln_rewrite": {
        "image_dir": "scalevln",
        "annotation_name": "scalevln_rewrite.jsonl",
        "dataset_label": "scalevln",
    },
    "panovln": {
        "image_dir": "panovln",
        "annotation_name": "panovln.jsonl",
        "dataset_label": "panovln",
    },
    "scalevln_150k": {
        "image_dir": "scalevln_150k",
        "annotation_name": "scalevln_150k.jsonl",
    },
    "dagger": {"image_dir": "dagger", "annotation_name": "dagger.jsonl"},
}

INSTRUCTION_ABLATION_VARIANTS = frozenset(
    {"scalevln", "scalevln_rewrite"}
)


def action_id_to_str(action_id: int) -> str:
    # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
    if action_id == 0:
        return "stop"
    if action_id == 1:
        return "forward"
    if action_id == 2:
        return "left"
    if action_id == 3:
        return "right"
    raise ValueError(f"Invalid action ID: {action_id}")


def frame_index_from_filename(filename: str) -> str:
    return filename.split("_")[1].split(".")[0]


def annotation_image_id(episode_item: Dict[str, Any]) -> str:
    """Resolve shared trajectory images while preserving old episode-keyed data."""

    image_id = episode_item.get("trajectory_id")
    if image_id is None:
        image_id = episode_item.get("episode_id", episode_item.get("video_id"))
    if image_id is None or isinstance(image_id, bool) or not str(image_id).strip():
        raise ValueError("Annotation has no usable trajectory_id/episode_id/video_id")
    return str(image_id)


def to_relative_path(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def write_jsonl_item(handle, item: Dict) -> None:
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_dataset_config(input_root: str) -> Dict[str, Dict[str, str]]:
    return {
        dataset_name: {
            "image_path": os.path.join(input_root, "images", spec["image_dir"]),
            "annotation_path": os.path.join(
                input_root,
                "sub_dataset",
                spec["annotation_name"],
            ),
            "dataset_label": spec.get("dataset_label", dataset_name),
            "frame_index_fn": frame_index_from_filename,
        }
        for dataset_name, spec in DATASET_SPECS.items()
    }


def validate_selected_subsets(
    selected_subset_list: List[str],
    dataset_config: Dict[str, Dict[str, str]],
) -> None:
    if not selected_subset_list:
        raise ValueError("At least one dataset name is required")
    duplicates = sorted(
        name for name, count in Counter(selected_subset_list).items() if count > 1
    )
    if duplicates:
        raise ValueError(f"Duplicate dataset names are not allowed: {duplicates}")
    unsupported = sorted(set(selected_subset_list) - set(dataset_config))
    if unsupported:
        raise ValueError(
            f"Unsupported dataset names: {unsupported}; "
            f"supported={sorted(dataset_config)}"
        )
    if INSTRUCTION_ABLATION_VARIANTS.issubset(selected_subset_list):
        raise ValueError(
            "scalevln and scalevln_rewrite are paired instruction variants over the same "
            "trajectories. Generate them in separate runs with the same seed and "
            "sampling parameters; do not mix both into one training JSONL."
        )


def load_episode_images(
    episode_image_path: str,
    frame_index_fn,
    input_root: str,
    num_actions: int,
) -> List[str]:
    episode_image_list = os.listdir(episode_image_path)
    episode_image_list = sorted(
        episode_image_list,
        key=lambda file_name: int(frame_index_fn(file_name)),
    )
    episode_image_list = [
        to_relative_path(os.path.join(episode_image_path, image), input_root)
        for image in episode_image_list
    ]

    if len(episode_image_list) == num_actions:
        episode_image_list.append(episode_image_list[-1])
    elif len(episode_image_list) != num_actions + 1:
        raise ValueError(
            f"Unexpected number of frames in {episode_image_path}: "
            f"{len(episode_image_list)} vs actions={num_actions}"
        )

    return episode_image_list


def pad_terminal_action_ids(
    action_ids: List[int],
    action_horizon: int = DEFAULT_ACTION_HORIZON,
) -> List[int]:
    if len(action_ids) >= action_horizon:
        return action_ids[:action_horizon]
    if not action_ids or action_ids[-1] != STOP_ACTION_ID:
        raise ValueError(
            "Only terminal chunks ending in stop can be padded to "
            f"{action_horizon} actions, got {action_ids}"
        )
    return action_ids + [STOP_ACTION_ID] * (action_horizon - len(action_ids))


def load_subset_annotations(
    selected_subset_list: List[str],
    dataset_config: Dict[str, Dict[str, str]],
    max_episodes_per_subset: int = None,
    subset_seed: int = DEFAULT_SUBSET_SEED,
) -> Dict[str, List[Dict[str, Any]]]:
    annotations_by_subset = {}
    for subset in selected_subset_list:
        annotation_path = dataset_config[subset]["annotation_path"]
        annotation = []
        with open(annotation_path, "r", encoding="utf-8") as handle:
            for line in handle:
                annotation.append(json.loads(line))
        if max_episodes_per_subset is not None:
            if max_episodes_per_subset <= 0:
                raise ValueError(
                    "max_episodes_per_subset must be positive, got "
                    f"{max_episodes_per_subset}"
                )
            if max_episodes_per_subset > len(annotation):
                raise ValueError(
                    f"Requested {max_episodes_per_subset} episodes from {subset}, "
                    f"but only {len(annotation)} are available"
                )
            random.Random(subset_seed).shuffle(annotation)
            annotation = annotation[:max_episodes_per_subset]
            print(
                f"[{subset}] selected {len(annotation)} random episodes "
                f"with subset_seed={subset_seed}"
            )
        annotations_by_subset[subset] = annotation
    return annotations_by_subset


def validate_keep_probability(value: float, name: str) -> float:
    value = float(value)
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {value}")
    return value


def action_chunk_contains_event(action_ids: List[int]) -> bool:
    return any(action_id in EVENT_ACTION_IDS for action_id in action_ids)


def ebs_action_chunk_keep_probability(
    action_ids: List[int],
    event_keep_prob: float = DEFAULT_EVENT_KEEP_PROB,
    background_keep_prob: float = DEFAULT_BACKGROUND_KEEP_PROB,
) -> float:
    if action_chunk_contains_event(action_ids):
        return event_keep_prob
    return background_keep_prob


def compute_ebs_candidate_counts(
    annotations_by_subset: Dict[str, List[Dict[str, Any]]],
    action_horizon: int = DEFAULT_ACTION_HORIZON,
) -> Counter:
    candidate_counts = Counter()
    for annotation in annotations_by_subset.values():
        for episode_item in annotation:
            actions = [int(action_id) for action_id in episode_item["actions"]]
            num_actions = len(actions)
            dense_start = max(0, num_actions - action_horizon)
            body_stop = max(0, dense_start - action_horizon + 1)
            candidate_counts["terminal_dense"] += num_actions - dense_start
            for start_step in range(body_stop):
                action_ids = actions[start_step:start_step + action_horizon]
                if action_chunk_contains_event(action_ids):
                    candidate_counts["event_body"] += 1
                else:
                    candidate_counts["background_body"] += 1
    return candidate_counts


def print_ebs_sampling_summary(
    candidate_counts: Counter,
    event_keep_prob: float,
    background_keep_prob: float,
    tail_dense_keep_prob: float,
    body_keep_advance: int,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
) -> None:
    print(
        f"chunk_sampler={EBS_SAMPLER} "
        f"event_keep_prob={event_keep_prob} "
        f"background_keep_prob={background_keep_prob} "
        f"tail_dense_keep_prob={tail_dense_keep_prob} "
        f"tail_dense={action_horizon} "
        f"terminal_buffer={action_horizon - 1} "
        f"body_keep_advance={body_keep_advance}"
    )
    for class_name in ("event_body", "background_body", "terminal_dense"):
        print(f"candidate_class {class_name} candidates={candidate_counts[class_name]}")


def build_ebs_action_chunk_starts(
    actions: List[int],
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    event_keep_prob: float = DEFAULT_EVENT_KEEP_PROB,
    background_keep_prob: float = DEFAULT_BACKGROUND_KEEP_PROB,
    tail_dense_keep_prob: float = DEFAULT_TAIL_DENSE_KEEP_PROB,
    body_keep_advance: int = DEFAULT_BODY_KEEP_ADVANCE,
    rng: Optional[random.Random] = None,
) -> List[int]:
    action_horizon = max(1, int(action_horizon))
    body_keep_advance = max(1, int(body_keep_advance))
    event_keep_prob = validate_keep_probability(
        event_keep_prob,
        "event_keep_prob",
    )
    background_keep_prob = validate_keep_probability(
        background_keep_prob,
        "background_keep_prob",
    )
    tail_dense_keep_prob = validate_keep_probability(
        tail_dense_keep_prob,
        "tail_dense_keep_prob",
    )
    if rng is None:
        rng = random.Random(DEFAULT_SEED)

    num_actions = len(actions)
    if num_actions <= 0:
        raise ValueError(f"Episode action count must be positive, got {num_actions}")
    if actions[-1] != STOP_ACTION_ID:
        raise ValueError(f"Episode must end with stop, got last action {actions[-1]}")

    dense_start = max(0, num_actions - action_horizon)
    body_stop = max(0, dense_start - action_horizon + 1)
    start_steps = set()

    start_step = 0
    while start_step < body_stop:
        action_ids = actions[start_step:start_step + action_horizon]
        keep_prob = ebs_action_chunk_keep_probability(
            action_ids=action_ids,
            event_keep_prob=event_keep_prob,
            background_keep_prob=background_keep_prob,
        )
        if rng.random() < keep_prob:
            start_steps.add(start_step)
            start_step += body_keep_advance
        else:
            start_step += 1

    for start_step in range(dense_start, num_actions):
        if tail_dense_keep_prob >= 1.0 or rng.random() < tail_dense_keep_prob:
            start_steps.add(start_step)

    return sorted(start_steps)


def build_action_chunks(
    actions: List[int],
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    event_keep_prob: float = DEFAULT_EVENT_KEEP_PROB,
    background_keep_prob: float = DEFAULT_BACKGROUND_KEEP_PROB,
    tail_dense_keep_prob: float = DEFAULT_TAIL_DENSE_KEEP_PROB,
    body_keep_advance: int = DEFAULT_BODY_KEEP_ADVANCE,
    pad_stop_to_horizon: bool = False,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    start_steps = build_ebs_action_chunk_starts(
        actions=actions,
        action_horizon=action_horizon,
        event_keep_prob=event_keep_prob,
        background_keep_prob=background_keep_prob,
        tail_dense_keep_prob=tail_dense_keep_prob,
        body_keep_advance=body_keep_advance,
        rng=rng,
    )

    action_chunks = []
    for start_step in start_steps:
        action_ids = actions[start_step:start_step + action_horizon]
        real_action_count = len(action_ids)
        if pad_stop_to_horizon and len(action_ids) < action_horizon:
            action_ids = pad_terminal_action_ids(
                action_ids=action_ids,
                action_horizon=action_horizon,
            )
        action_chunks.append(
            {
                "action_ids": action_ids,
                "start_step": start_step,
                "end_step": start_step + real_action_count - 1,
                "real_action_count": real_action_count,
                "texts": [action_id_to_str(action_id) for action_id in action_ids],
            }
        )
    return action_chunks


def build_vln_images(
    episode_image_list: List[str],
    current_step: int,
) -> List[str]:
    current_frame_index = min(max(0, int(current_step)), len(episode_image_list) - 1)
    return episode_image_list[:current_frame_index + 1]


def process_dataset(
    selected_subset_list: List[str],
    dataset_config: Dict[str, Dict[str, str]],
    annotations_by_subset: Dict[str, List[Dict[str, Any]]],
    input_root: str,
    pad_stop_to_horizon: bool = False,
    seed: int = DEFAULT_SEED,
    event_keep_prob: float = DEFAULT_EVENT_KEEP_PROB,
    background_keep_prob: float = DEFAULT_BACKGROUND_KEEP_PROB,
    tail_dense_keep_prob: float = DEFAULT_TAIL_DENSE_KEEP_PROB,
    body_keep_advance: int = DEFAULT_BODY_KEEP_ADVANCE,
    output_handle=None,
):
    data2save = []
    total_samples = 0
    rng = random.Random(seed)
    seen_sample_keys: Set[Tuple[str, str, int]] = set()
    for subset in selected_subset_list:
        subset_config = dataset_config[subset]
        image_path = subset_config["image_path"]
        dataset_label = subset_config["dataset_label"]
        frame_index_fn = subset_config["frame_index_fn"]
        annotation = annotations_by_subset[subset]

        subset_sample_start = total_samples
        progress = tqdm(annotation, desc=subset, dynamic_ncols=True)

        for episode_item in progress:
            episode_id = episode_item["episode_id"]
            instruction = episode_item["instruction"]
            actions = [int(action_id) for action_id in episode_item["actions"]]
            assert actions[-1] == 0

            episode_image_dir = annotation_image_id(episode_item)
            episode_image_path = os.path.join(image_path, episode_image_dir)

            episode_image_list = load_episode_images(
                episode_image_path=episode_image_path,
                frame_index_fn=frame_index_fn,
                input_root=input_root,
                num_actions=len(actions),
            )

            action_chunks = build_action_chunks(
                actions,
                event_keep_prob=event_keep_prob,
                background_keep_prob=background_keep_prob,
                tail_dense_keep_prob=tail_dense_keep_prob,
                body_keep_advance=body_keep_advance,
                pad_stop_to_horizon=pad_stop_to_horizon,
                rng=rng,
            )

            for action_chunk in action_chunks:
                sample_key = (subset, str(episode_id), action_chunk["start_step"])
                if sample_key in seen_sample_keys:
                    raise ValueError(
                        "Duplicate training sample key generated: "
                        f"dataset={sample_key[0]} episode_id={sample_key[1]} "
                        f"step_index={sample_key[2]}"
                    )
                seen_sample_keys.add(sample_key)

                user_images = build_vln_images(
                    episode_image_list=episode_image_list,
                    current_step=action_chunk["start_step"],
                )

                sample = {
                    "instruction": instruction,
                    "action_sequence": list(action_chunk["texts"]),
                    "images": list(user_images),
                    "episode_id": str(episode_id),
                    # Keep the label identical for paired ScaleVLN rewrite
                    # ablation so the published samples differ only in instruction.
                    "dataset": dataset_label,
                    "step_index": action_chunk["start_step"],
                    "end_step": action_chunk["end_step"],
                    "real_action_count": action_chunk["real_action_count"],
                }
                if episode_item.get("trajectory_id") is not None:
                    sample["trajectory_id"] = str(episode_item["trajectory_id"])
                if output_handle is None:
                    data2save.append(sample)
                else:
                    write_jsonl_item(output_handle, sample)
                total_samples += 1

        subset_sample_count = total_samples - subset_sample_start
        print(
            f"[{subset}] episodes={len(annotation)} samples={subset_sample_count}"
        )

    if output_handle is None:
        return data2save
    return total_samples


def main(
    selected_subset_list: List[str],
    input_root: str,
    output_path: str,
    max_episodes_per_subset: int = None,
    subset_seed: int = DEFAULT_SUBSET_SEED,
    pad_stop_to_horizon: bool = False,
    seed: int = DEFAULT_SEED,
    event_keep_prob: float = DEFAULT_EVENT_KEEP_PROB,
    background_keep_prob: float = DEFAULT_BACKGROUND_KEEP_PROB,
    tail_dense_keep_prob: float = DEFAULT_TAIL_DENSE_KEEP_PROB,
    body_keep_advance: int = DEFAULT_BODY_KEEP_ADVANCE,
) -> None:
    dataset_config = build_dataset_config(input_root)
    validate_selected_subsets(selected_subset_list, dataset_config)
    annotations_by_subset = load_subset_annotations(
        selected_subset_list=selected_subset_list,
        dataset_config=dataset_config,
        max_episodes_per_subset=max_episodes_per_subset,
        subset_seed=subset_seed,
    )

    event_keep_prob = validate_keep_probability(
        event_keep_prob,
        "event_keep_prob",
    )
    background_keep_prob = validate_keep_probability(
        background_keep_prob,
        "background_keep_prob",
    )
    tail_dense_keep_prob = validate_keep_probability(
        tail_dense_keep_prob,
        "tail_dense_keep_prob",
    )
    candidate_counts = compute_ebs_candidate_counts(
        annotations_by_subset=annotations_by_subset,
    )
    print_ebs_sampling_summary(
        candidate_counts=candidate_counts,
        event_keep_prob=event_keep_prob,
        background_keep_prob=background_keep_prob,
        tail_dense_keep_prob=tail_dense_keep_prob,
        body_keep_advance=body_keep_advance,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    descriptor, temporary_path = tempfile.mkstemp(
        prefix=f".{os.path.basename(output_path)}.",
        suffix=".tmp",
        dir=output_dir or ".",
        text=True,
    )
    total_samples = 0
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output_handle:
            total_samples += process_dataset(
                selected_subset_list=selected_subset_list,
                dataset_config=dataset_config,
                annotations_by_subset=annotations_by_subset,
                input_root=input_root,
                pad_stop_to_horizon=pad_stop_to_horizon,
                seed=seed,
                event_keep_prob=event_keep_prob,
                background_keep_prob=background_keep_prob,
                tail_dense_keep_prob=tail_dense_keep_prob,
                body_keep_advance=body_keep_advance,
                output_handle=output_handle,
            )
            output_handle.flush()
            os.fsync(output_handle.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    print(f"total number of samples = {total_samples}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        nargs="+",
        default=["r2r"],
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default="/workspace/data2/dataset/PanoVLN",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="panovln_train_data.jsonl",
    )
    parser.add_argument(
        "--max_episodes_per_subset",
        type=int,
        default=None,
        help=(
            "If set, deterministically shuffle each source annotation with "
            "subset_seed and keep the requested prefix. Reusing subset_seed "
            "makes different subset sizes nested."
        ),
    )
    parser.add_argument(
        "--subset_seed",
        type=int,
        default=DEFAULT_SUBSET_SEED,
        help="Random seed for deterministic episode subset selection.",
    )
    parser.add_argument(
        "--pad_stop_to_horizon",
        action="store_true",
        help="Pad terminal chunks ending in stop to the action horizon with stop.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Random seed for EBS chunk sampling.",
    )
    parser.add_argument(
        "--event_keep_prob",
        type=float,
        default=DEFAULT_EVENT_KEEP_PROB,
        help=(
            "EBS keep probability for body chunks containing left/right events."
        ),
    )
    parser.add_argument(
        "--background_keep_prob",
        type=float,
        default=DEFAULT_BACKGROUND_KEEP_PROB,
        help=(
            "EBS keep probability for all-forward body chunks."
        ),
    )
    parser.add_argument(
        "--tail_dense_keep_prob",
        type=float,
        default=DEFAULT_TAIL_DENSE_KEEP_PROB,
        help=(
            "Keep probability for each terminal dense chunk, including the "
            "final stop-only chunk."
        ),
    )
    parser.add_argument(
        "--body_keep_advance",
        type=int,
        default=DEFAULT_BODY_KEEP_ADVANCE,
        help=(
            "Number of start steps to advance after keeping a body chunk. "
            "Defaults to the action horizon, preserving the original "
            "non-overlapping body sampling behavior."
        ),
    )
    args = parser.parse_args()

    main(
        selected_subset_list=args.dataset_name,
        input_root=args.input_root,
        output_path=args.output_path,
        max_episodes_per_subset=args.max_episodes_per_subset,
        subset_seed=args.subset_seed,
        pad_stop_to_horizon=args.pad_stop_to_horizon,
        seed=args.seed,
        event_keep_prob=args.event_keep_prob,
        background_keep_prob=args.background_keep_prob,
        tail_dense_keep_prob=args.tail_dense_keep_prob,
        body_keep_advance=args.body_keep_advance,
    )
