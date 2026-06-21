import json
import math
import os
import random
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

try:
    from src.train.utils import build_prompt_and_target
except ModuleNotFoundError:
    from utils import build_prompt_and_target


DEFAULT_VLN_MEMORY_IMAGE_SIZE = (448, 224)
DEFAULT_VLN_CURRENT_OBSERVATION_IMAGE_SIZE = (960, 480)
DEFAULT_PANOVGGT_IMAGE_SIZE = (1036, 518)
DEFAULT_VLN_MAX_MEMORY_IMAGES = 10
DEFAULT_VLN_MEMORY_POOL_WINDOW_FRAMES = 100
DEFAULT_PANOVGGT_SPATIAL_MEMORY_FRAMES = 5
DEFAULT_ERP_TOP_CROP_DEGREES = 20
DEFAULT_ERP_BOTTOM_CROP_DEGREES = 20
PANOVGGT_SPATIAL_MEMORY_MIN_FORWARD_GAP = 2
VLN_ACTION_WORDS = {"forward", "left", "right", "stop"}
VLN_ACTION_ALIASES = {
    "move_forward": "forward",
    "move forward": "forward",
    "move-forward": "forward",
    "turn_left": "left",
    "turn left": "left",
    "turn-left": "left",
    "turn_right": "right",
    "turn right": "right",
    "turn-right": "right",
    "stop": "stop",
}
VLN_ACTION_SEQUENCE_LENGTH = 4
VLN_SYSTEM_PROMPT = (
    "You are an autonomous navigation assistant. "
    "Your task is to follow the navigation instruction. "
    "Given the instruction, your recent observations, and your current observation, "
    "devise an action sequence using the four actions: left or right by 15 degrees, "
    "forward by 25 centimeters, or stop once the task is complete. "
    "Return exactly four action words in execution order, separated by spaces. "
    "If the task is complete before four actions, fill the remaining positions with stop."
)


def crop_erp_latitude(
    image: Image.Image,
    top_crop_degrees: float = DEFAULT_ERP_TOP_CROP_DEGREES,
    bottom_crop_degrees: float = DEFAULT_ERP_BOTTOM_CROP_DEGREES,
) -> Image.Image:
    width, height = image.size
    if width <= 0 or height <= 1:
        return image

    top_crop_pixels = max(0, round(height * float(top_crop_degrees) / 180.0))
    bottom_crop_pixels = max(0, round(height * float(bottom_crop_degrees) / 180.0))

    top_crop_pixels = min(top_crop_pixels, height - 1)
    bottom_crop_pixels = min(bottom_crop_pixels, height - top_crop_pixels - 1)
    crop_bottom = height - bottom_crop_pixels

    if top_crop_pixels <= 0 and crop_bottom >= height:
        return image

    return image.crop((0, top_crop_pixels, width, crop_bottom))


def build_erp_image_geometry(
    top_crop_degrees: float = DEFAULT_ERP_TOP_CROP_DEGREES,
    bottom_crop_degrees: float = DEFAULT_ERP_BOTTOM_CROP_DEGREES,
) -> torch.Tensor:
    top_crop_degrees = max(0.0, float(top_crop_degrees))
    bottom_crop_degrees = max(0.0, float(bottom_crop_degrees))
    vertical_fov_degrees = 180.0 - top_crop_degrees - bottom_crop_degrees
    if vertical_fov_degrees <= 0.0:
        raise ValueError(
            "ERP crop removes the full vertical field of view: "
            f"top={top_crop_degrees}, bottom={bottom_crop_degrees}"
        )

    center_latitude_degrees = 0.5 * (top_crop_degrees - bottom_crop_degrees)
    return torch.tensor(
        [
            math.radians(vertical_fov_degrees),
            math.radians(center_latitude_degrees),
        ],
        dtype=torch.float32,
    )


def build_erp_image_geometry_batch(
    num_images: int,
    top_crop_degrees: float = DEFAULT_ERP_TOP_CROP_DEGREES,
    bottom_crop_degrees: float = DEFAULT_ERP_BOTTOM_CROP_DEGREES,
) -> torch.Tensor:
    num_images = max(0, int(num_images))
    if num_images == 0:
        return torch.zeros((0, 2), dtype=torch.float32)

    geometry = build_erp_image_geometry(
        top_crop_degrees=top_crop_degrees,
        bottom_crop_degrees=bottom_crop_degrees,
    )
    return geometry.unsqueeze(0).repeat(num_images, 1)


def resolve_current_image_index(num_images: int) -> int:
    num_images = int(num_images)
    if num_images <= 0:
        return -1
    return num_images - 1


def preprocess_vln_current_image(
    image: Image.Image,
    top_crop_degrees: float = DEFAULT_ERP_TOP_CROP_DEGREES,
    bottom_crop_degrees: float = DEFAULT_ERP_BOTTOM_CROP_DEGREES,
) -> Image.Image:
    processed_image = image.convert("RGB").resize(DEFAULT_VLN_CURRENT_OBSERVATION_IMAGE_SIZE)
    return crop_erp_latitude(
        processed_image,
        top_crop_degrees=top_crop_degrees,
        bottom_crop_degrees=bottom_crop_degrees,
    )


def preprocess_vln_memory_image(
    image: Image.Image,
    top_crop_degrees: float = DEFAULT_ERP_TOP_CROP_DEGREES,
    bottom_crop_degrees: float = DEFAULT_ERP_BOTTOM_CROP_DEGREES,
) -> Image.Image:
    processed_image = image.convert("RGB").resize(DEFAULT_VLN_MEMORY_IMAGE_SIZE)
    return crop_erp_latitude(
        processed_image,
        top_crop_degrees=top_crop_degrees,
        bottom_crop_degrees=bottom_crop_degrees,
    )


def preprocess_panovggt_current_image(image: Image.Image) -> torch.Tensor:
    processed_image = image.convert("RGB").resize(
        DEFAULT_PANOVGGT_IMAGE_SIZE,
        Image.Resampling.LANCZOS,
    )
    return TF.to_tensor(processed_image)


def _normalize_history_action(action: Any) -> Optional[str]:
    if isinstance(action, bool):
        return None
    if isinstance(action, int):
        if action == 0:
            return "stop"
        if action == 1:
            return "forward"
        if action == 2:
            return "left"
        if action == 3:
            return "right"
        return None
    return _normalize_vln_action(action)


def normalize_history_actions(actions: Any) -> List[str]:
    if not isinstance(actions, list):
        raise ValueError(
            "PanoVGGT spatial memory requires VLN examples to provide "
            "a list field 'history_actions'. Regenerate or patch the JSONL "
            "before enabling panovggt_spatial_memory."
        )
    normalized_actions = []
    for action_index, action in enumerate(actions):
        normalized_action = _normalize_history_action(action)
        if normalized_action is None:
            raise ValueError(
                "PanoVGGT spatial memory field 'history_actions' must contain "
                f"action ids or action words, got {action!r} at index {action_index}"
            )
        normalized_actions.append(normalized_action)
    return normalized_actions


def _frame_producing_action(
    frame_index: int,
    history_actions: Sequence[str],
) -> Optional[str]:
    if frame_index <= 0:
        return None
    action_index = frame_index - 1
    if action_index >= len(history_actions):
        return None
    return history_actions[action_index]


def build_forward_counts_by_frame(
    num_frames: int,
    history_actions: Sequence[str],
) -> List[int]:
    num_frames = max(0, int(num_frames))
    forward_counts = []
    count = 0
    for frame_index in range(num_frames):
        if frame_index > 0 and _frame_producing_action(frame_index, history_actions) == "forward":
            count += 1
        forward_counts.append(count)
    return forward_counts


def _select_with_forward_gap(
    *,
    current_frame_index: int,
    history_actions: Sequence[str],
    forward_counts: Sequence[int],
    target_count: int,
    min_forward_gap: int,
) -> List[int]:
    selected = [current_frame_index]
    last_selected_forward_count = int(forward_counts[current_frame_index])
    for frame_index in range(current_frame_index - 1, -1, -1):
        if _frame_producing_action(frame_index, history_actions) != "forward":
            continue
        frame_forward_count = int(forward_counts[frame_index])
        if min_forward_gap > 0 and (
            last_selected_forward_count - frame_forward_count < min_forward_gap
        ):
            continue
        selected.append(frame_index)
        last_selected_forward_count = frame_forward_count
        if len(selected) >= target_count:
            break
    return selected


def _fill_with_recent_real_frames(
    *,
    selected: List[int],
    current_frame_index: int,
    target_count: int,
) -> List[int]:
    selected_set = set(selected)
    for frame_index in range(current_frame_index - 1, -1, -1):
        if len(selected) >= target_count:
            break
        if frame_index in selected_set:
            continue
        selected.append(frame_index)
        selected_set.add(frame_index)
    return selected


def _pad_indices_at_front(indices: List[int], target_count: int) -> List[int]:
    if not indices:
        return []
    target_count = max(1, int(target_count))
    if len(indices) >= target_count:
        return indices[-target_count:]
    return [indices[0]] * (target_count - len(indices)) + indices


def build_panovggt_spatial_memory_selection(
    *,
    num_frames: int,
    history_actions: Any,
    total_frames: int = DEFAULT_PANOVGGT_SPATIAL_MEMORY_FRAMES,
) -> List[int]:
    """Select a fixed-size current-centric spatial window for PanoVGGT.

    The window always ends with the current frame. It first selects
    forward-produced spatial keyframes backwards with a two-forward-step gap.
    If that yields too few frames, it fills with the nearest real history
    frames, including turn-produced frames. Only trajectories shorter than the
    requested window are padded by repeating the earliest available frame.
    """
    num_frames = int(num_frames)
    total_frames = max(1, int(total_frames))
    if num_frames <= 0:
        return []

    current_frame_index = num_frames - 1
    normalized_actions = normalize_history_actions(history_actions)
    expected_action_count = current_frame_index
    if len(normalized_actions) != expected_action_count:
        raise ValueError(
            "PanoVGGT spatial memory requires len(history_actions) == num_frames - 1, "
            f"got len(history_actions)={len(normalized_actions)} and num_frames={num_frames}"
        )

    forward_counts = build_forward_counts_by_frame(
        num_frames,
        normalized_actions,
    )
    selected = _select_with_forward_gap(
        current_frame_index=current_frame_index,
        history_actions=normalized_actions,
        forward_counts=forward_counts,
        target_count=total_frames,
        min_forward_gap=PANOVGGT_SPATIAL_MEMORY_MIN_FORWARD_GAP,
    )
    if len(selected) < total_frames:
        selected = _fill_with_recent_real_frames(
            selected=selected,
            current_frame_index=current_frame_index,
            target_count=total_frames,
        )
    return _pad_indices_at_front(sorted(selected[-total_frames:]), total_frames)


def select_panovggt_spatial_memory_paths(
    image_paths: List[str],
    history_actions: Any,
    total_frames: int = DEFAULT_PANOVGGT_SPATIAL_MEMORY_FRAMES,
) -> Tuple[List[str], List[int]]:
    selected_indices = build_panovggt_spatial_memory_selection(
        num_frames=len(image_paths),
        history_actions=history_actions,
        total_frames=total_frames,
    )
    return [image_paths[index] for index in selected_indices], selected_indices


def build_vln_image_selection(
    current_step: int,
    last_frame_index: int,
    max_memory_images: int = DEFAULT_VLN_MAX_MEMORY_IMAGES,
    memory_pool_window_frames: int = DEFAULT_VLN_MEMORY_POOL_WINDOW_FRAMES,
) -> List[int]:
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

    last_candidate_position = len(candidate_frame_indices) - 1
    selected_positions = [
        (slot * last_candidate_position) // (total_selected_images - 1)
        for slot in range(total_selected_images)
    ]
    return [candidate_frame_indices[position] for position in selected_positions]


def select_vln_image_paths(
    image_paths: List[str],
    max_memory_images: int = DEFAULT_VLN_MAX_MEMORY_IMAGES,
    memory_pool_window_frames: int = DEFAULT_VLN_MEMORY_POOL_WINDOW_FRAMES,
) -> List[str]:
    if not image_paths:
        return []

    selected_indices = build_vln_image_selection(
        current_step=len(image_paths) - 1,
        last_frame_index=len(image_paths) - 1,
        max_memory_images=max_memory_images,
        memory_pool_window_frames=memory_pool_window_frames,
    )
    return [image_paths[index] for index in selected_indices]


def text_content(text: str) -> Dict[str, str]:
    return {"type": "text", "text": text}


def image_content() -> Dict[str, str]:
    return {"type": "image"}


def build_vln_user_content(instruction: str, num_images: int) -> List[Dict[str, str]]:
    if num_images <= 0:
        raise ValueError("VLN samples require at least one image")

    num_memory_images = max(0, num_images - 1)
    content = [text_content(f"Instruction: {instruction.strip()}")]

    if num_memory_images > 0:
        content.append(
            text_content(
                "\nHistory memory observations are panoramic views ordered from older to newer:"
            )
        )
        content.extend(image_content() for _ in range(num_memory_images))

    content.extend(
        [
            text_content("\nCurrent observation (panoramic view):"),
            image_content(),
            text_content("\nDevise the next action sequence."),
        ]
    )
    return content


def _resolve_image_path(path: str, image_root: Optional[str]) -> str:
    if os.path.isabs(path) or path.startswith(("http://", "https://", "file://")):
        return path
    if image_root is None:
        return path
    return os.path.join(image_root, path)


def _resolve_content_item(
    item: Dict[str, Any],
    image_root: Optional[str],
    example_images: List[str],
    image_index_ref: List[int],
    vision_paths: List[str],
) -> Dict[str, Any]:
    item_type = item.get("type")
    if item_type == "text":
        return {
            "type": "text",
            "text": item.get("text", ""),
        }

    if item_type == "image":
        image_path = item.get("image")
        if image_path is None:
            if image_index_ref[0] >= len(example_images):
                raise ValueError(
                    "Image placeholder count exceeds the number of entries in example['images']"
                )
            image_path = example_images[image_index_ref[0]]
            image_index_ref[0] += 1
        elif not isinstance(image_path, str) or not image_path:
            raise ValueError("Image content item must contain a non-empty 'image' path")

        resolved_path = _resolve_image_path(image_path, image_root)
        vision_paths.append(resolved_path)
        return {
            "type": "image",
        }

    raise NotImplementedError(f"Unsupported content item type: {item_type}")


def _require_non_empty_string(example: Dict[str, Any], field_name: str) -> str:
    value = example.get(field_name)
    if isinstance(value, str):
        value = value.strip()
    if isinstance(value, str) and value:
        return value
    raise ValueError(f"VLN example field '{field_name}' must be a non-empty string")


def _metadata_value_to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def build_episode_key(example: Dict[str, Any]) -> str:
    episode_id = _metadata_value_to_str(example.get("episode_id"))
    if episode_id is not None:
        dataset = _metadata_value_to_str(example.get("dataset")) or "unknown_dataset"
        return f"dataset={dataset}|episode_id={episode_id}"

    raise ValueError(
        "TRACE requires each training sample to contain episode_id"
    )


def _metadata_step_index(example: Dict[str, Any], fallback_index: int) -> int:
    step_index = example.get("step_index")
    if step_index is None:
        return int(fallback_index)
    return int(step_index)


def _metadata_action_sequence(example: Dict[str, Any]) -> List[str]:
    action_sequence = example.get("action_sequence")
    if not isinstance(action_sequence, list):
        return []
    return [str(action) for action in action_sequence]


def _metadata_end_step_index(
    example: Dict[str, Any],
    step_index: int,
    action_sequence: List[str],
) -> int:
    end_step = example.get("end_step")
    if end_step is not None:
        return int(end_step)
    if action_sequence:
        return int(step_index) + len(action_sequence) - 1
    return int(step_index)


def _extract_vln_instruction(example: Dict[str, Any]) -> str:
    instruction = _require_non_empty_string(example, "instruction")
    if isinstance(instruction, str) and instruction.strip():
        return instruction.strip()
    raise ValueError("VLN example is missing an instruction")


def _normalize_vln_action(action: Any) -> Optional[str]:
    if not isinstance(action, str):
        return None
    stripped_action = action.strip()
    if stripped_action in VLN_ACTION_WORDS:
        return stripped_action
    return VLN_ACTION_ALIASES.get(stripped_action.lower())


def _extract_vln_action_sequence(example: Dict[str, Any]) -> List[str]:
    action_sequence = example.get("action_sequence")
    if not isinstance(action_sequence, list):
        raise ValueError("VLN example field 'action_sequence' must be a list")
    if len(action_sequence) != VLN_ACTION_SEQUENCE_LENGTH:
        raise ValueError(
            "VLN example field 'action_sequence' must contain exactly "
            f"{VLN_ACTION_SEQUENCE_LENGTH} actions, got {len(action_sequence)}"
        )

    normalized_actions = []
    for action_index, action in enumerate(action_sequence):
        normalized_action = _normalize_vln_action(action)
        if normalized_action is None:
            raise ValueError(
                "VLN example field 'action_sequence' must contain only action words "
                f"{sorted(VLN_ACTION_WORDS)} or legacy action names, got "
                f"{action!r} at index {action_index}"
            )
        normalized_actions.append(normalized_action)
    if "stop" in normalized_actions and normalized_actions[-1] != "stop":
        raise ValueError("VLN action sequence must end immediately after stop")
    return normalized_actions


def apply_vln_memory_policy(example: Dict[str, Any]) -> Dict[str, Any]:
    raw_images = example.get("images", [])
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("VLN example field 'images' must contain the full image history")
    for image_path in raw_images:
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("VLN example field 'images' must contain non-empty string paths")

    selected_images = select_vln_image_paths(raw_images)
    instruction = _extract_vln_instruction(example)
    action_sequence = _extract_vln_action_sequence(example)

    normalized = dict(example)
    normalized["images"] = selected_images
    normalized["messages"] = [
        {
            "role": "system",
            "content": [text_content(VLN_SYSTEM_PROMPT)],
        },
        {
            "role": "user",
            "content": build_vln_user_content(
                instruction=instruction,
                num_images=len(selected_images),
            ),
        },
        {
            "role": "assistant",
            "content": [text_content(" ".join(action_sequence))],
        },
    ]
    return normalized


def resolve_messages_and_vision_paths(example: Dict[str, Any], image_root: Optional[str]):
    vision_paths = []
    raw_messages = example.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(
            "Internal VLN sample normalization failed to build messages from compact fields"
        )

    raw_images = example.get("images", [])
    if raw_images is None:
        raw_images = []
    if not isinstance(raw_images, list):
        raise ValueError("Training example field 'images' must be a list when present")
    image_index_ref = [0]

    messages = []
    for message_index, raw_message in enumerate(raw_messages):
        role = raw_message.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError(f"Message at index {message_index} is missing a valid 'role'")

        raw_content = raw_message.get("content", [])
        if isinstance(raw_content, str):
            raw_content = [{"type": "text", "text": raw_content}]
        if not isinstance(raw_content, list):
            raise ValueError(
                f"Message at index {message_index} must contain list-valued 'content'"
            )

        resolved_content = [
            _resolve_content_item(
                item=item,
                image_root=image_root,
                example_images=raw_images,
                image_index_ref=image_index_ref,
                vision_paths=vision_paths,
            )
            for item in raw_content
        ]

        message = dict(raw_message)
        message["content"] = resolved_content
        messages.append(message)

    if image_index_ref[0] != len(raw_images):
        raise ValueError(
            "The number of consumed image placeholders does not match the number of "
            f"entries in example['images'] ({image_index_ref[0]} consumed vs {len(raw_images)} provided)"
        )

    return messages, vision_paths


class SupervisedDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        processor,
        tokenizer,
        image_root: Optional[str],
        image_token: str,
        model_max_length: Optional[int],
        erp_top_crop_degrees: float = DEFAULT_ERP_TOP_CROP_DEGREES,
        erp_bottom_crop_degrees: float = DEFAULT_ERP_BOTTOM_CROP_DEGREES,
        panovggt_enabled: bool = False,
        panovggt_spatial_memory: bool = False,
        panovggt_spatial_memory_frames: int = DEFAULT_PANOVGGT_SPATIAL_MEMORY_FRAMES,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        prompt_format: str = "chat_template",
        collect_trace_metadata: bool = False,
    ):
        self.jsonl_path = jsonl_path
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_root = image_root
        processor_image_token = getattr(processor, "image_token", None)
        if processor_image_token is not None:
            self.image_token = processor_image_token
        else:
            self.image_token = image_token
        self.model_max_length = model_max_length
        self.erp_top_crop_degrees = float(erp_top_crop_degrees)
        self.erp_bottom_crop_degrees = float(erp_bottom_crop_degrees)
        self.panovggt_enabled = bool(panovggt_enabled)
        self.panovggt_spatial_memory = bool(panovggt_spatial_memory)
        self.panovggt_spatial_memory_frames = max(1, int(panovggt_spatial_memory_frames))
        self.prompt_format = prompt_format
        self._fp = None
        self.episode_keys = None
        self.step_indices = None
        self.end_step_indices = None
        self.action_sequences = None

        entries = []
        with open(self.jsonl_path, "rb") as handle:
            line_index = 0
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    entry = {"offset": offset}
                    if collect_trace_metadata:
                        example = json.loads(line)
                        action_sequence = _metadata_action_sequence(example)
                        step_index = _metadata_step_index(
                            example,
                            fallback_index=line_index,
                        )
                        entry["episode_key"] = build_episode_key(example)
                        entry["step_index"] = step_index
                        entry["end_step_index"] = _metadata_end_step_index(
                            example,
                            step_index=step_index,
                            action_sequence=action_sequence,
                        )
                        entry["action_sequence"] = action_sequence
                    entries.append(entry)
                    line_index += 1

        if shuffle:
            random.shuffle(entries)

        if max_samples is not None:
            entries = entries[:max_samples]

        self.offsets = [entry["offset"] for entry in entries]
        if collect_trace_metadata:
            self.episode_keys = [entry["episode_key"] for entry in entries]
            self.step_indices = [entry["step_index"] for entry in entries]
            self.end_step_indices = [entry["end_step_index"] for entry in entries]
            self.action_sequences = [entry["action_sequence"] for entry in entries]

    def __len__(self):
        return len(self.offsets)

    def __del__(self):
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def _get_fp(self):
        if self._fp is None:
            self._fp = open(self.jsonl_path, "r", encoding="utf-8")
        return self._fp

    def _load_example(self, index: int):
        handle = self._get_fp()
        handle.seek(self.offsets[index])
        return json.loads(handle.readline())

    def _load_images(self, image_paths: List[str]):
        images = []
        num_images = len(image_paths)
        for image_index, image_path in enumerate(image_paths):
            with Image.open(image_path) as image:
                is_current_observation = image_index == num_images - 1
                if is_current_observation:
                    processed_image = preprocess_vln_current_image(
                        image,
                        top_crop_degrees=self.erp_top_crop_degrees,
                        bottom_crop_degrees=self.erp_bottom_crop_degrees,
                    )
                else:
                    processed_image = preprocess_vln_memory_image(
                        image,
                        top_crop_degrees=self.erp_top_crop_degrees,
                        bottom_crop_degrees=self.erp_bottom_crop_degrees,
                    )
                images.append(processed_image)
        return images

    def _load_panovggt_images(self, image_paths: List[str]) -> torch.Tensor:
        images = []
        for image_path in image_paths:
            with Image.open(image_path) as image:
                images.append(preprocess_panovggt_current_image(image))
        return torch.stack(images, dim=0)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        raw_example = self._load_example(index)
        example = apply_vln_memory_policy(raw_example)
        messages, vision_paths = resolve_messages_and_vision_paths(
            example,
            image_root=self.image_root,
        )
        prompt_and_target = build_prompt_and_target(
            messages,
            self.prompt_format,
            processor=self.processor,
        )
        prompt_text = prompt_and_target["prompt"]
        target_text = prompt_and_target["target"]
        eos = self.tokenizer.eos_token or ""
        target_with_eos = target_text + eos
        full_text = prompt_text + target_with_eos

        if vision_paths:
            encoded = self.processor(
                text=full_text,
                images=self._load_images(
                    image_paths=vision_paths,
                ),
                return_tensors="pt",
                truncation=self.model_max_length is not None,
                max_length=self.model_max_length,
            )
        else:
            encoded = self.tokenizer(
                full_text,
                return_tensors="pt",
                truncation=self.model_max_length is not None,
                max_length=self.model_max_length,
            )

        target_ids = self.tokenizer(
            target_with_eos,
            add_special_tokens=False,
            return_tensors="pt",
            truncation=self.model_max_length is not None,
            max_length=self.model_max_length,
        )["input_ids"].squeeze(0)

        input_ids = encoded["input_ids"].squeeze(0)
        attention_mask = encoded.get("attention_mask")
        if attention_mask is not None:
            attention_mask = attention_mask.squeeze(0)
        else:
            attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        target_len = min(target_ids.size(0), input_ids.size(0))
        if target_len > 0:
            labels[:-target_len] = -100

        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        image_count = len(vision_paths)
        item["image_erp_geometry"] = build_erp_image_geometry_batch(
            image_count,
            top_crop_degrees=self.erp_top_crop_degrees,
            bottom_crop_degrees=self.erp_bottom_crop_degrees,
        )
        item["image_num_images"] = torch.tensor([image_count], dtype=torch.long)
        item["image_current_index"] = torch.tensor(
            [resolve_current_image_index(image_count)],
            dtype=torch.long,
        )

        if "mm_token_type_ids" in encoded:
            item["mm_token_type_ids"] = encoded["mm_token_type_ids"].squeeze(0)

        if self.panovggt_enabled:
            raw_images = raw_example.get("images", [])
            if raw_images:
                raw_image_paths = [
                    _resolve_image_path(image_path, self.image_root)
                    for image_path in raw_images
                ]
                if self.panovggt_spatial_memory:
                    panovggt_paths, _ = select_panovggt_spatial_memory_paths(
                        raw_image_paths,
                        history_actions=raw_example.get("history_actions"),
                        total_frames=self.panovggt_spatial_memory_frames,
                    )
                    item["panovggt_pixel_values"] = self._load_panovggt_images(
                        panovggt_paths,
                    ).unsqueeze(0)
                else:
                    with Image.open(raw_image_paths[-1]) as image:
                        item["panovggt_pixel_values"] = preprocess_panovggt_current_image(
                            image,
                        ).unsqueeze(0)

        for key in STACKABLE_KEYS:
            if key in encoded:
                item[key] = encoded[key]

        return item


STACKABLE_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "image_erp_geometry",
    "image_num_images",
    "image_current_index",
    "panovggt_pixel_values",
    "pixel_values_videos",
    "video_grid_thw",
)
