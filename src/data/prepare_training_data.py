import argparse
import json
import os
from typing import Any, Dict, List

from tqdm import tqdm


DEFAULT_ACTION_HORIZON = 4
DEFAULT_ACTION_STRIDE = 4
STOP_ACTION_ID = 0


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


def build_action_chunk_starts(
    num_actions: int,
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_stride: int = DEFAULT_ACTION_STRIDE,
) -> List[int]:
    action_horizon = max(1, int(action_horizon))
    action_stride = max(1, int(action_stride))
    if num_actions <= 0:
        raise ValueError(f"Episode action count must be positive, got {num_actions}")

    if num_actions <= action_horizon:
        start_steps = [0]
    else:
        last_full_start = num_actions - action_horizon
        start_steps = list(range(0, last_full_start + 1, action_stride))
        if start_steps[-1] != last_full_start:
            start_steps.append(last_full_start)

    suffix_start = max(0, num_actions - action_horizon + 1)
    start_steps.extend(range(suffix_start, num_actions))
    return sorted(set(start_steps))


def build_action_chunks(
    actions: List[int],
    action_horizon: int = DEFAULT_ACTION_HORIZON,
    action_stride: int = DEFAULT_ACTION_STRIDE,
    pad_stop_to_horizon: bool = False,
) -> List[Dict[str, Any]]:
    action_chunks = []
    for start_step in build_action_chunk_starts(
        num_actions=len(actions),
        action_horizon=action_horizon,
        action_stride=action_stride,
    ):
        action_ids = actions[start_step:start_step + action_horizon]
        real_action_count = len(action_ids)
        if pad_stop_to_horizon and len(action_ids) < action_horizon:
            if not action_ids or action_ids[-1] != STOP_ACTION_ID:
                raise ValueError(
                    "Only terminal chunks ending in stop can be padded to "
                    f"{action_horizon} actions, got {action_ids}"
                )
            action_ids = action_ids + [STOP_ACTION_ID] * (action_horizon - len(action_ids))
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
    input_root: str,
    max_episodes_per_subset: int = None,
    pad_stop_to_horizon: bool = False,
    output_handle=None,
):
    data2save = []
    total_samples = 0
    for subset in selected_subset_list:
        subset_config = dataset_config[subset]
        image_path = subset_config["image_path"]
        annotation_path = subset_config["annotation_path"]
        frame_index_fn = subset_config["frame_index_fn"]

        annotation = []
        with open(annotation_path, "r", encoding="utf-8") as handle:
            for line in handle:
                annotation.append(json.loads(line))

        if max_episodes_per_subset is not None:
            annotation = annotation[:max_episodes_per_subset]

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
                pad_stop_to_horizon=pad_stop_to_horizon,
            )

            for action_chunk in action_chunks:
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
) -> None:
    dataset_config = build_dataset_config(input_root)
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    total_samples = 0

    with open(output_path, "w", encoding="utf-8") as output_handle:
        total_samples += process_dataset(
            selected_subset_list=selected_subset_list,
            dataset_config=dataset_config,
            input_root=input_root,
            max_episodes_per_subset=max_episodes_per_subset,
            pad_stop_to_horizon=pad_stop_to_horizon,
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
        default="/workspace/code_dir/a_property/NAVIDA_pano",
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
    args = parser.parse_args()

    main(
        selected_subset_list=args.dataset_name,
        input_root=args.input_root,
        output_path=args.output_path,
        max_episodes_per_subset=args.max_episodes_per_subset,
        pad_stop_to_horizon=args.pad_stop_to_horizon,
    )
