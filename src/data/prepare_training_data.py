import argparse
import json
import os
from typing import Dict, List, Tuple

from tqdm import tqdm


FORWARD_DISTANCE_CM = 25
TURN_ANGLE_DEGREE = 15
DEFAULT_MAX_MEMORY_IMAGES = 10
DEFAULT_MEMORY_POOL_WINDOW_FRAMES = 200
VLN_SYSTEM_PROMPT = (
    "You are a visual language navigation agent. "
    "Given a navigation instruction, your recent memory observations, and your current observation, "
    "predict the next action. "
    "Action space: move_forward (0.25 meters), turn_left (15 degrees), "
    "turn_right (15 degrees), stop. "
    "Use stop only when you think the goal has been reached. "
    "Reply with exactly one action."
)
IDM_SYSTEM_PROMPT = (
    "You are a visual language navigation agent. Given the current panoramic view and the goal panoramic view, "
    "output exactly one action that moves the robot from the current view toward the goal view: "
    "move_forward, turn_left, turn_right, or stop."
)


def action_id_to_str(action_id: int) -> str:
    # id: 0-stop, 1 move forward, 2 turn left, 3 turn right
    if action_id == 0:
        return "stop"
    if action_id == 1:
        return "move_forward"
    if action_id == 2:
        return "turn_left"
    if action_id == 3:
        return "turn_right"
    raise ValueError(f"Invalid action ID: {action_id}")


def frame_index_from_filename(filename: str) -> str:
    return filename.split("_")[1].split(".")[0]


def to_relative_path(path: str, root: str) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def write_jsonl(output_path: str, items: List[Dict]) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")


def text_content(text: str) -> Dict[str, str]:
    return {"type": "text", "text": text}


def image_content() -> Dict[str, str]:
    return {"type": "image"}


def build_system_prompt(task_type: str) -> str:
    if task_type == "vln":
        return VLN_SYSTEM_PROMPT
    if task_type == "idm":
        return IDM_SYSTEM_PROMPT
    raise NotImplementedError(f"Unsupported task type: {task_type}")


def build_vln_user_content(instruction: str, user_images: List[str]) -> List[Dict[str, str]]:
    if not user_images:
        raise ValueError("VLN samples require at least one image")

    num_memory_images = max(0, len(user_images) - 1)
    instruction = instruction.strip()
    content = [text_content(f"Instruction: {instruction}")]

    if num_memory_images > 0:
        content.append(
            text_content(
                "\nHistory memory observations are 90-degree perspective views ordered from older to newer:"
            )
        )
        content.extend(image_content() for _ in range(num_memory_images))

    content.extend(
        [
            text_content(
                "\nCurrent observation (360-degree panoramic view centered on the robot's current forward direction):"
            ),
            image_content(),
            text_content("\nPredict the next action."),
        ]
    )
    return content


def build_idm_user_content(user_images: List[str]) -> List[Dict[str, str]]:
    if len(user_images) != 2:
        raise ValueError(f"IDM samples require exactly 2 images, got {len(user_images)}")
    return [
        text_content(
            "You have been given an image of the current view "
        ),
        image_content(),
        text_content(
            " and an image of the goal view "
        ),
        image_content(),
        text_content(
            ". Analyze the two images to predict the navigation action that "
            "would move the robot from the current view to the goal view."
        ),
    ]


def build_training_messages(
    task_type: str,
    instruction: str,
    user_images: List[str],
    assistant_text: str,
) -> List[Dict]:
    if task_type == "vln":
        user_content = build_vln_user_content(
            instruction=instruction,
            user_images=user_images,
        )
    elif task_type == "idm":
        user_content = build_idm_user_content(user_images)
    else:
        raise NotImplementedError(f"Unsupported task type: {task_type}")

    return [
        {
            "role": "system",
            "content": [text_content(build_system_prompt(task_type))],
        },
        {
            "role": "user",
            "content": user_content,
        },
        {
            "role": "assistant",
            "content": [text_content(assistant_text)],
        },
    ]


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


def build_action_chunks(actions: List[int]) -> List[Dict[str, int]]:
    return [
        {
            "action_id": action_id,
            "start_step": step_index,
            "end_step": step_index,
            "text": action_id_to_str(action_id),
        }
        for step_index, action_id in enumerate(actions)
    ]


def build_vln_image_selection(
    current_step: int,
    last_frame_index: int,
    max_memory_images: int,
    memory_pool_window_frames: int,
) -> List[int]:
    max_memory_images = max(0, int(max_memory_images))
    memory_pool_window_frames = max(1, int(memory_pool_window_frames))
    current_frame_index = min(current_step, last_frame_index)
    pool_start_frame = max(0, current_frame_index - memory_pool_window_frames + 1)
    candidate_frame_indices = list(range(pool_start_frame, current_frame_index + 1))

    total_selected_images = max_memory_images + 1
    if total_selected_images <= 0 or not candidate_frame_indices:
        return [current_frame_index]

    if len(candidate_frame_indices) <= total_selected_images:
        return candidate_frame_indices

    last_candidate_position = len(candidate_frame_indices) - 1
    selected_positions = [
        (slot * last_candidate_position) // (total_selected_images - 1)
        for slot in range(total_selected_images)
    ]
    return [
        candidate_frame_indices[position]
        for position in selected_positions
    ]


def build_vln_images(
    episode_image_list: List[str],
    actions: List[int],
    current_step: int,
    max_memory_images: int,
    memory_pool_window_frames: int,
) -> List[str]:
    del actions
    selected_frame_indices = build_vln_image_selection(
        current_step=current_step,
        last_frame_index=len(episode_image_list) - 1,
        max_memory_images=max_memory_images,
        memory_pool_window_frames=memory_pool_window_frames,
    )
    return [episode_image_list[frame_index] for frame_index in selected_frame_indices]


def build_idm_images(episode_image_list: List[str], action_chunk: Dict[str, int]) -> List[str]:
    current_image = episode_image_list[action_chunk["start_step"]]
    goal_image = episode_image_list[min(action_chunk["end_step"] + 1, len(episode_image_list) - 1)]
    return [current_image, goal_image]


def process_single_type(
    selected_subset_list: List[str],
    dataset_config: Dict[str, Dict[str, str]],
    input_root: str,
    task_type: str,
    max_memory_images: int,
    memory_pool_window_frames: int,
    max_episodes_per_subset: int = None,
) -> List[Dict]:
    data2save = []
    for subset in selected_subset_list:
        subset_config = dataset_config[subset]
        image_path = subset_config["image_path"]
        annotation_path = subset_config["annotation_path"]
        frame_index_fn = subset_config["frame_index_fn"]
        sub_image_path = subset_config.get("sub_image_path")

        annotation = []
        with open(annotation_path, "r", encoding="utf-8") as handle:
            for line in handle:
                annotation.append(json.loads(line))

        if max_episodes_per_subset is not None:
            annotation = annotation[:max_episodes_per_subset]

        subset_sample_start = len(data2save)
        progress = tqdm(annotation, desc=f"{task_type}:{subset}", dynamic_ncols=True)

        for episode_item in progress:
            episode_id = episode_item["episode_id"]
            instruction = episode_item["instruction"]
            actions = episode_item["actions"]
            assert actions[-1] == 0

            episode_image_dir = str(episode_item.get("episode_id", episode_item.get("video_id")))
            episode_image_path = os.path.join(image_path, episode_image_dir)
            if sub_image_path is not None:
                episode_image_path = os.path.join(episode_image_path, sub_image_path)

            episode_image_list = load_episode_images(
                episode_image_path=episode_image_path,
                frame_index_fn=frame_index_fn,
                input_root=input_root,
                num_actions=len(actions),
            )

            action_chunks = build_action_chunks(actions)

            for action_chunk in action_chunks:
                if task_type == "idm" and action_chunk["action_id"] == 0:
                    continue

                if task_type == "vln":
                    user_images = build_vln_images(
                        episode_image_list=episode_image_list,
                        actions=actions,
                        current_step=action_chunk["start_step"],
                        max_memory_images=max_memory_images,
                        memory_pool_window_frames=memory_pool_window_frames,
                    )
                elif task_type == "idm":
                    user_images = build_idm_images(episode_image_list, action_chunk)
                else:
                    raise NotImplementedError(f"Unsupported task type: {task_type}")

                messages = build_training_messages(
                    task_type=task_type,
                    instruction=instruction,
                    user_images=user_images,
                    assistant_text=action_chunk["text"],
                )

                data2save.append(
                    {
                        "messages": messages,
                        "images": list(user_images),
                        "episode_id": str(episode_id),
                        "task type": task_type,
                    }
                )

        subset_sample_count = len(data2save) - subset_sample_start
        print(
            f"[{task_type}][{subset}] episodes={len(annotation)} "
            f"samples={subset_sample_count}"
        )

    return data2save


def main(
    selected_subset_list: List[str],
    task_type_list: List[str],
    input_root: str,
    output_path: str,
    max_memory_images: int,
    memory_pool_window_frames: int,
    max_episodes_per_subset: int = None,
) -> None:
    dataset_config = build_dataset_config(input_root)
    data2save = []

    for task_type in task_type_list:
        if task_type not in {"vln", "idm"}:
            raise NotImplementedError(f"Unsupported task type: {task_type}")

        data2save.extend(
            process_single_type(
                selected_subset_list=selected_subset_list,
                dataset_config=dataset_config,
                input_root=input_root,
                task_type=task_type,
                max_memory_images=max_memory_images,
                memory_pool_window_frames=memory_pool_window_frames,
                max_episodes_per_subset=max_episodes_per_subset,
            )
        )

    print(f"total number of samples = {len(data2save)}")
    write_jsonl(output_path, data2save)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        nargs="+",
        default=["r2r"],
    )
    parser.add_argument(
        "--task_type",
        nargs="+",
        default=["vln", "idm"],
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
        "--max_memory_images",
        type=int,
        default=DEFAULT_MAX_MEMORY_IMAGES,
    )
    parser.add_argument(
        "--memory_pool_window_frames",
        type=int,
        default=DEFAULT_MEMORY_POOL_WINDOW_FRAMES,
    )
    parser.add_argument(
        "--max_episodes_per_subset",
        type=int,
        default=None,
    )
    args = parser.parse_args()

    main(
        selected_subset_list=args.dataset_name,
        task_type_list=args.task_type,
        input_root=args.input_root,
        output_path=args.output_path,
        max_memory_images=args.max_memory_images,
        memory_pool_window_frames=args.memory_pool_window_frames,
        max_episodes_per_subset=args.max_episodes_per_subset,
    )
