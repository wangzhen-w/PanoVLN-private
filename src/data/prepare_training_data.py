import argparse
import json
import math
import os
import random
from collections import Counter
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm import tqdm


DEFAULT_ACTION_HORIZON = 4
DEFAULT_SEED = 42
SBS_SAMPLER = "sbs"
DEFAULT_SBS_TAU = 1.35
DEFAULT_SBS_BETA = 0.40
STOP_ACTION_ID = 0
BALANCE_CLASSES = [
    "stop_pos_1",
    "stop_pos_2",
    "stop_pos_3",
    "stop_pos_4",
    "first_forward",
    "first_left",
    "first_right",
]


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


def to_relative_path(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def write_jsonl_item(handle, item: Dict) -> None:
    handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def build_dataset_config(input_root: str) -> Dict[str, Dict[str, str]]:
    dataset_names = ["r2r", "rxr", "envdrop", "scalevln", "scalevln_150k"]
    return {
        dataset_name: {
            "image_path": os.path.join(input_root, "images", dataset_name),
            "annotation_path": os.path.join(
                input_root,
                "sub_dataset",
                f"{dataset_name}.jsonl",
            ),
            "frame_index_fn": frame_index_from_filename,
        }
        for dataset_name in dataset_names
    }


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


def action_chunk_balance_class(action_ids: List[int]) -> str:
    for action_index, action_id in enumerate(action_ids):
        if action_id == STOP_ACTION_ID:
            return f"stop_pos_{action_index + 1}"
    first_action = action_id_to_str(action_ids[0])
    if first_action not in {"forward", "left", "right"}:
        raise ValueError(f"Unsupported first action for balance class: {first_action}")
    return f"first_{first_action}"


def load_subset_annotations(
    selected_subset_list: List[str],
    dataset_config: Dict[str, Dict[str, str]],
    max_episodes_per_subset: int = None,
) -> Dict[str, List[Dict[str, Any]]]:
    annotations_by_subset = {}
    for subset in selected_subset_list:
        annotation_path = dataset_config[subset]["annotation_path"]
        annotation = []
        with open(annotation_path, "r", encoding="utf-8") as handle:
            for line in handle:
                annotation.append(json.loads(line))
        if max_episodes_per_subset is not None:
            annotation = annotation[:max_episodes_per_subset]
        annotations_by_subset[subset] = annotation
    return annotations_by_subset


def compute_action_information(
    annotations_by_subset: Dict[str, List[Dict[str, Any]]],
) -> Tuple[Counter, Dict[int, float]]:
    action_counts = Counter()
    for annotation in annotations_by_subset.values():
        for episode_item in annotation:
            action_counts.update(int(action_id) for action_id in episode_item["actions"])

    total_actions = sum(action_counts.values())
    if total_actions <= 0:
        raise ValueError("No actions found in selected annotations")

    action_information = {}
    for action_id in range(4):
        count = action_counts[action_id]
        if count <= 0:
            raise ValueError(f"Action id {action_id} is absent from selected data")
        action_information[action_id] = -math.log(count / total_actions)
    return action_counts, action_information


def compute_candidate_class_counts(
    annotations_by_subset: Dict[str, List[Dict[str, Any]]],
    action_horizon: int = DEFAULT_ACTION_HORIZON,
) -> Counter:
    class_counts = Counter()
    for annotation in annotations_by_subset.values():
        for episode_item in annotation:
            actions = [int(action_id) for action_id in episode_item["actions"]]
            for start_step in range(len(actions)):
                action_ids = pad_terminal_action_ids(
                    actions[start_step:start_step + action_horizon],
                    action_horizon=action_horizon,
                )
                class_counts[action_chunk_balance_class(action_ids)] += 1

    for class_name in BALANCE_CLASSES:
        class_counts.setdefault(class_name, 0)
    return class_counts


def compute_balance_factors(
    class_counts: Counter,
    beta: float = DEFAULT_SBS_BETA,
) -> Dict[str, float]:
    beta = float(beta)
    if beta < 0:
        raise ValueError(f"sbs_beta must be non-negative, got {beta}")

    raw_factors = {}
    for class_name in BALANCE_CLASSES:
        count = class_counts[class_name]
        if count <= 0:
            raise ValueError(f"Balance class {class_name} has no candidates")
        raw_factors[class_name] = count ** (-beta)
    mean_factor = sum(raw_factors.values()) / len(raw_factors)
    return {
        class_name: raw_factors[class_name] / mean_factor
        for class_name in BALANCE_CLASSES
    }


def action_chunk_information_value(
    action_ids: List[int],
    action_information: Dict[int, float],
) -> float:
    return sum(action_information[action_id] for action_id in action_ids) / len(action_ids)


def sbs_action_chunk_keep_probability(
    action_ids: List[int],
    action_information: Dict[int, float],
    balance_factors: Dict[str, float],
    tau: float = DEFAULT_SBS_TAU,
) -> float:
    if tau <= 0:
        raise ValueError(f"sbs_tau must be positive, got {tau}")
    class_name = action_chunk_balance_class(action_ids)
    value = action_chunk_information_value(action_ids, action_information)
    balanced_value = value * balance_factors[class_name]
    return 1.0 - math.exp(-tau * balanced_value)


def print_sbs_sampling_summary(
    action_counts: Counter,
    action_information: Dict[int, float],
    class_counts: Counter,
    balance_factors: Dict[str, float],
    tau: float,
    beta: float,
) -> None:
    total_actions = sum(action_counts.values())
    print(
        f"chunk_sampler={SBS_SAMPLER} "
        f"tau={tau} beta={beta}"
    )
    for action_id in range(4):
        count = action_counts[action_id]
        freq = count / total_actions
        print(
            "action_info "
            f"{action_id_to_str(action_id)} count={count} "
            f"freq={freq:.10f} I={action_information[action_id]:.10f}"
        )
    for class_name in BALANCE_CLASSES:
        print(
            "balance_class "
            f"{class_name} candidates={class_counts[class_name]} "
            f"factor={balance_factors[class_name]:.10f}"
        )


def build_sbs_action_chunk_starts(
    actions: List[int],
    action_information: Dict[int, float],
    balance_factors: Dict[str, float],
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    tau: float = DEFAULT_SBS_TAU,
    rng: Optional[random.Random] = None,
) -> List[int]:
    action_horizon = max(1, int(action_horizon))
    if rng is None:
        rng = random.Random(DEFAULT_SEED)

    num_actions = len(actions)
    if num_actions <= 0:
        raise ValueError(f"Episode action count must be positive, got {num_actions}")
    if actions[-1] != STOP_ACTION_ID:
        raise ValueError(f"Episode must end with stop, got last action {actions[-1]}")

    dense_start = max(0, num_actions - action_horizon)
    start_steps = set(range(dense_start, num_actions))

    start_step = 0
    while start_step < dense_start:
        action_ids = pad_terminal_action_ids(
            actions[start_step:start_step + action_horizon],
            action_horizon=action_horizon,
        )
        keep_prob = sbs_action_chunk_keep_probability(
            action_ids=action_ids,
            action_information=action_information,
            balance_factors=balance_factors,
            tau=tau,
        )
        if rng.random() < keep_prob:
            start_steps.add(start_step)
            start_step += action_horizon
        else:
            start_step += 1

    return sorted(start_steps)


def build_action_chunks(
    actions: List[int],
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    sbs_tau: float = DEFAULT_SBS_TAU,
    action_information: Optional[Dict[int, float]] = None,
    balance_factors: Optional[Dict[str, float]] = None,
    pad_stop_to_horizon: bool = False,
    rng: Optional[random.Random] = None,
) -> List[Dict[str, Any]]:
    if action_information is None or balance_factors is None:
        raise ValueError(
            "SBS sampling requires action_information "
            "and balance_factors"
        )
    start_steps = build_sbs_action_chunk_starts(
        actions=actions,
        action_information=action_information,
        balance_factors=balance_factors,
        action_horizon=action_horizon,
        tau=sbs_tau,
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
    sbs_tau: float = DEFAULT_SBS_TAU,
    action_information: Optional[Dict[int, float]] = None,
    balance_factors: Optional[Dict[str, float]] = None,
    output_handle=None,
):
    data2save = []
    total_samples = 0
    rng = random.Random(seed)
    seen_sample_keys: Set[Tuple[str, str, int]] = set()
    for subset in selected_subset_list:
        subset_config = dataset_config[subset]
        image_path = subset_config["image_path"]
        frame_index_fn = subset_config["frame_index_fn"]
        annotation = annotations_by_subset[subset]

        subset_sample_start = total_samples
        progress = tqdm(annotation, desc=subset, dynamic_ncols=True)

        for episode_item in progress:
            episode_id = episode_item["episode_id"]
            instruction = episode_item["instruction"]
            actions = episode_item["actions"]
            assert actions[-1] == 0

            episode_image_dir = str(episode_item.get("episode_id", episode_item.get("video_id")))
            episode_image_path = os.path.join(image_path, episode_image_dir)

            episode_image_list = load_episode_images(
                episode_image_path=episode_image_path,
                frame_index_fn=frame_index_fn,
                input_root=input_root,
                num_actions=len(actions),
            )

            action_chunks = build_action_chunks(
                actions,
                sbs_tau=sbs_tau,
                action_information=action_information,
                balance_factors=balance_factors,
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
                    "dataset": subset,
                    "step_index": action_chunk["start_step"],
                    "end_step": action_chunk["end_step"],
                    "real_action_count": action_chunk["real_action_count"],
                }
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
    pad_stop_to_horizon: bool = False,
    seed: int = DEFAULT_SEED,
    sbs_tau: float = DEFAULT_SBS_TAU,
    sbs_beta: float = DEFAULT_SBS_BETA,
) -> None:
    dataset_config = build_dataset_config(input_root)
    annotations_by_subset = load_subset_annotations(
        selected_subset_list=selected_subset_list,
        dataset_config=dataset_config,
        max_episodes_per_subset=max_episodes_per_subset,
    )

    if sbs_tau <= 0:
        raise ValueError(f"sbs_tau must be positive, got {sbs_tau}")
    action_counts, action_information = compute_action_information(
        annotations_by_subset
    )
    class_counts = compute_candidate_class_counts(annotations_by_subset)
    balance_factors = compute_balance_factors(
        class_counts=class_counts,
        beta=sbs_beta,
    )
    print_sbs_sampling_summary(
        action_counts=action_counts,
        action_information=action_information,
        class_counts=class_counts,
        balance_factors=balance_factors,
        tau=sbs_tau,
        beta=sbs_beta,
    )

    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    total_samples = 0

    with open(output_path, "w", encoding="utf-8") as output_handle:
        total_samples += process_dataset(
            selected_subset_list=selected_subset_list,
            dataset_config=dataset_config,
            annotations_by_subset=annotations_by_subset,
            input_root=input_root,
            pad_stop_to_horizon=pad_stop_to_horizon,
            seed=seed,
            sbs_tau=sbs_tau,
            action_information=action_information,
            balance_factors=balance_factors,
            output_handle=output_handle,
        )

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
        default="/workspace/code_dir/a_property/dataset/NAVIDA_pano",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="navida_train_data.jsonl",
    )
    parser.add_argument(
        "--max_episodes_per_subset",
        type=int,
        default=None,
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
        help="Random seed for SBS chunk sampling.",
    )
    parser.add_argument(
        "--sbs_tau",
        type=float,
        default=DEFAULT_SBS_TAU,
        help=(
            "Scale parameter for SBS (Surprisal-Balanced Sampling): "
            "p_keep = 1 - exp(-tau * V(chunk) * B(class))."
        ),
    )
    parser.add_argument(
        "--sbs_beta",
        type=float,
        default=DEFAULT_SBS_BETA,
        help=(
            "Candidate-class balance strength for SBS. "
            "B(class) is proportional to candidate_count^(-beta)."
        ),
    )
    args = parser.parse_args()

    main(
        selected_subset_list=args.dataset_name,
        input_root=args.input_root,
        output_path=args.output_path,
        max_episodes_per_subset=args.max_episodes_per_subset,
        pad_stop_to_horizon=args.pad_stop_to_horizon,
        seed=args.seed,
        sbs_tau=args.sbs_tau,
        sbs_beta=args.sbs_beta,
    )
