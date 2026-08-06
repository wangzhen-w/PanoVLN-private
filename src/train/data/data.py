import json
import math
import os
import random
import re
from itertools import combinations
from typing import Any, Dict, List, Optional

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
DEFAULT_ERP_TOP_CROP_DEGREES = 20
DEFAULT_ERP_BOTTOM_CROP_DEGREES = 20
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
VLN_ACTION_TO_ID = {
    "stop": 0,
    "forward": 1,
    "left": 2,
    "right": 3,
}
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


def build_vln_image_selection(
    current_step: int,
    last_frame_index: int,
    max_memory_images: int = DEFAULT_VLN_MAX_MEMORY_IMAGES,
    memory_pool_window_frames: int = DEFAULT_VLN_MEMORY_POOL_WINDOW_FRAMES,
    required_frame_indices: Optional[List[int]] = None,
) -> List[int]:
    max_memory_images = max(0, int(max_memory_images))
    memory_pool_window_frames = max(1, int(memory_pool_window_frames))
    current_frame_index = min(max(0, int(current_step)), int(last_frame_index))
    pool_start_frame = max(0, current_frame_index - memory_pool_window_frames + 1)
    candidate_frame_indices = list(range(pool_start_frame, current_frame_index + 1))

    required = sorted(
        {
            int(index)
            for index in (required_frame_indices or [])
            if pool_start_frame <= int(index) <= current_frame_index
        }
    )

    total_selected_images = max_memory_images + 1
    if total_selected_images <= 0 or not candidate_frame_indices:
        return [current_frame_index]

    if len(candidate_frame_indices) <= total_selected_images:
        selected_indices = candidate_frame_indices
    elif total_selected_images == 1:
        if any(index != current_frame_index for index in required):
            raise ValueError(
                "Cannot retain required VLN history frames without a memory slot: "
                f"required={required}, budget={total_selected_images}"
            )
        selected_indices = [current_frame_index]
    else:
        mandatory_indices = sorted(
            {pool_start_frame, current_frame_index, *required}
        )
        if len(mandatory_indices) > total_selected_images:
            raise ValueError(
                "Cannot retain required VLN history frames within the configured "
                f"memory budget: required={required}, budget={total_selected_images}"
            )

        # Start from the original uniform grid.  If a required anchor is
        # missing, jointly redistribute the grid so that adjacent temporal
        # gaps remain as even as possible under the anchor constraint.
        last_candidate_position = len(candidate_frame_indices) - 1
        selected_positions = [
            (slot * last_candidate_position) // (total_selected_images - 1)
            for slot in range(total_selected_images)
        ]
        selected_indices = [
            candidate_frame_indices[position] for position in selected_positions
        ]
        missing_required = [
            index for index in required if index not in selected_indices
        ]
        if not missing_required:
            return selected_indices

        internal_mandatory = [
            index
            for index in mandatory_indices
            if index not in {pool_start_frame, current_frame_index}
        ]
        best_selection = None
        best_score = None
        for internal_slots in combinations(
            range(1, total_selected_images - 1),
            len(internal_mandatory),
        ):
            anchor_indices = [
                pool_start_frame,
                *internal_mandatory,
                current_frame_index,
            ]
            anchor_slots = [0, *internal_slots, total_selected_images - 1]
            if any(
                right_index - left_index < right_slot - left_slot
                for left_index, right_index, left_slot, right_slot in zip(
                    anchor_indices,
                    anchor_indices[1:],
                    anchor_slots,
                    anchor_slots[1:],
                )
            ):
                continue

            selection = [pool_start_frame] * total_selected_images
            for left_index, right_index, left_slot, right_slot in zip(
                anchor_indices,
                anchor_indices[1:],
                anchor_slots,
                anchor_slots[1:],
            ):
                frame_span = right_index - left_index
                slot_span = right_slot - left_slot
                for offset in range(slot_span + 1):
                    selection[left_slot + offset] = left_index + (
                        offset * frame_span + slot_span // 2
                    ) // slot_span

            gaps = [
                right - left for left, right in zip(selection, selection[1:])
            ]
            gap_uniformity_cost = sum(gap * gap for gap in gaps)
            original_grid_distance = sum(
                (frame_index - original_index) ** 2
                for frame_index, original_index in zip(
                    selection,
                    selected_indices,
                )
            )
            candidate_score = (
                gap_uniformity_cost,
                original_grid_distance,
                selection,
            )
            if best_score is None or candidate_score < best_score:
                best_score = candidate_score
                best_selection = selection

        if best_selection is None:
            raise ValueError(
                "Cannot distribute required VLN history frames within the "
                f"configured memory budget: required={required}, "
                f"budget={total_selected_images}"
            )
        selected_indices = best_selection

    return selected_indices


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


def _extract_vln_history_actions(
    example: Dict[str, Any],
    current_step: int,
) -> Optional[List[str]]:
    history_actions = example.get("history_actions")
    if history_actions is None:
        return None
    if not isinstance(history_actions, list):
        raise ValueError("VLN example field 'history_actions' must be a list")
    normalized_actions = []
    for action_index, action in enumerate(history_actions):
        normalized_action = _normalize_vln_action(action)
        if normalized_action is None:
            raise ValueError(
                "VLN example field 'history_actions' contains an invalid action "
                f"at index {action_index}: {action!r}"
            )
        if normalized_action == "stop":
            raise ValueError("VLN history_actions cannot contain terminal stop")
        normalized_actions.append(normalized_action)
    if len(normalized_actions) != current_step:
        raise ValueError(
            "VLN history/image alignment mismatch: "
            f"step_index={current_step}, history_actions={len(normalized_actions)}"
        )
    return normalized_actions


def _extract_real_action_count(
    example: Dict[str, Any],
    action_sequence: List[str],
) -> int:
    try:
        first_stop_index = action_sequence.index("stop")
    except ValueError:
        expected_count = len(action_sequence)
    else:
        expected_count = first_stop_index + 1
        if any(action != "stop" for action in action_sequence[first_stop_index + 1:]):
            raise ValueError(
                "VLN action_sequence cannot contain executable actions after stop"
            )

    value = example.get("real_action_count")
    if value is None:
        return expected_count
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError("VLN example field 'real_action_count' must be an integer")
    if value < 1 or value > VLN_ACTION_SEQUENCE_LENGTH:
        raise ValueError(
            "VLN example field 'real_action_count' must be in [1, 4], "
            f"got {value}"
        )
    if value != expected_count:
        raise ValueError(
            "VLN real_action_count/action_sequence mismatch: "
            f"real_action_count={value}, expected={expected_count}, "
            f"action_sequence={action_sequence}"
        )
    return int(value)


def _forward_target_observation_path(
    current_path: str,
    action_sequence: List[str],
) -> str:
    directory, filename = os.path.split(current_path)
    match = re.fullmatch(r"frame_(\d+)(\.[^.]+)", filename)
    if match is None:
        raise ValueError(
            "Forward dynamics requires frame_<step> image names, got "
            f"{current_path!r}"
        )
    # Each non-STOP action produces one stored observation.  A real STOP in the
    # fourth slot is an identity transition, so its conceptual t+4 state reuses
    # the stored t+3 panorama.
    target_offset = sum(action != "stop" for action in action_sequence)
    target_filename = (
        f"frame_{int(match.group(1)) + target_offset}{match.group(2)}"
    )
    return os.path.join(directory, target_filename)


def apply_vln_memory_policy(
    example: Dict[str, Any],
    *,
    pbo_enabled: bool = False,
    forward_dynamics_enabled: bool = False,
) -> Dict[str, Any]:
    raw_images = example.get("images", [])
    if not isinstance(raw_images, list) or not raw_images:
        raise ValueError("VLN example field 'images' must contain the full image history")
    for image_path in raw_images:
        if not isinstance(image_path, str) or not image_path:
            raise ValueError("VLN example field 'images' must contain non-empty string paths")

    current_step = example.get("step_index", len(raw_images) - 1)
    if isinstance(current_step, bool) or not isinstance(current_step, int) or current_step < 0:
        raise ValueError("VLN example field 'step_index' must be a non-negative integer")
    if len(raw_images) != current_step + 1:
        raise ValueError(
            "VLN history/image alignment mismatch: "
            f"step_index={current_step}, images={len(raw_images)}"
        )

    instruction = _extract_vln_instruction(example)
    action_sequence = _extract_vln_action_sequence(example)
    history_actions = _extract_vln_history_actions(example, current_step)
    real_action_count = _extract_real_action_count(example, action_sequence)

    pbo_valid = bool(
        pbo_enabled
        and history_actions is not None
        and len(history_actions) >= VLN_ACTION_SEQUENCE_LENGTH
    )
    four_step_memory_anchor = (
        current_step - VLN_ACTION_SEQUENCE_LENGTH
        if current_step >= VLN_ACTION_SEQUENCE_LENGTH
        else -1
    )
    selected_indices = build_vln_image_selection(
        current_step=current_step,
        last_frame_index=len(raw_images) - 1,
        required_frame_indices=(
            [four_step_memory_anchor] if four_step_memory_anchor >= 0 else None
        ),
    )
    selected_images = [raw_images[index] for index in selected_indices]
    pbo_start_image_index = (
        selected_indices.index(four_step_memory_anchor) if pbo_valid else -1
    )

    if pbo_valid:
        pbo_action_labels = [
            VLN_ACTION_TO_ID[action]
            for action in history_actions[-VLN_ACTION_SEQUENCE_LENGTH:]
        ]
    else:
        pbo_action_labels = [-100] * VLN_ACTION_SEQUENCE_LENGTH

    user_content = build_vln_user_content(
        instruction=instruction,
        num_images=len(selected_images),
    )

    normalized = dict(example)
    normalized["images"] = selected_images
    normalized["_pbo_valid"] = pbo_valid
    normalized["_pbo_start_image_index"] = pbo_start_image_index
    normalized["_pbo_action_labels"] = pbo_action_labels
    forward_dynamics_valid = bool(
        forward_dynamics_enabled
        and real_action_count == VLN_ACTION_SEQUENCE_LENGTH
    )
    normalized["_forward_dynamics_valid"] = forward_dynamics_valid
    normalized["_forward_target_image"] = (
        _forward_target_observation_path(raw_images[-1], action_sequence)
        if forward_dynamics_valid
        else None
    )
    normalized["messages"] = [
        {
            "role": "system",
            "content": [text_content(VLN_SYSTEM_PROMPT)],
        },
        {
            "role": "user",
            "content": user_content,
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
        pbo_enabled: bool = False,
        forward_dynamics_enabled: bool = False,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        prompt_format: str = "chat_template",
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
        self.pbo_enabled = bool(pbo_enabled)
        self.forward_dynamics_enabled = bool(forward_dynamics_enabled)
        self.prompt_format = prompt_format
        self._fp = None

        entries = []
        with open(self.jsonl_path, "rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    entries.append(offset)

        if shuffle:
            random.shuffle(entries)

        if max_samples is not None:
            entries = entries[:max_samples]

        self.offsets = entries

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

    def _load_images(
        self,
        image_paths: List[str],
    ):
        processed_images = []
        raw_images = []
        num_images = len(image_paths)
        for image_index, image_path in enumerate(image_paths):
            with Image.open(image_path) as image:
                raw_image = image.convert("RGB")
                is_current_observation = image_index == num_images - 1
                if is_current_observation:
                    processed_image = preprocess_vln_current_image(
                        raw_image,
                        top_crop_degrees=self.erp_top_crop_degrees,
                        bottom_crop_degrees=self.erp_bottom_crop_degrees,
                    )
                else:
                    processed_image = preprocess_vln_memory_image(
                        raw_image,
                        top_crop_degrees=self.erp_top_crop_degrees,
                        bottom_crop_degrees=self.erp_bottom_crop_degrees,
                    )
                raw_images.append(raw_image)
                processed_images.append(processed_image)
        return processed_images, raw_images

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = apply_vln_memory_policy(
            self._load_example(index),
            pbo_enabled=self.pbo_enabled,
            forward_dynamics_enabled=self.forward_dynamics_enabled,
        )
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

        raw_images = []
        if vision_paths:
            processed_images, raw_images = self._load_images(
                image_paths=vision_paths,
            )
            encoded = self.processor(
                text=full_text,
                images=processed_images,
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

        image_count = len(vision_paths)
        image_erp_geometry = build_erp_image_geometry_batch(
            image_count,
            top_crop_degrees=self.erp_top_crop_degrees,
            bottom_crop_degrees=self.erp_bottom_crop_degrees,
        )
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        item["image_erp_geometry"] = image_erp_geometry
        item["image_num_images"] = torch.tensor([image_count], dtype=torch.long)
        item["image_current_index"] = torch.tensor(
            [resolve_current_image_index(image_count)],
            dtype=torch.long,
        )
        item["pbo_action_labels"] = torch.tensor(
            [example["_pbo_action_labels"]],
            dtype=torch.long,
        )
        item["pbo_valid_mask"] = torch.tensor(
            [example["_pbo_valid"]],
            dtype=torch.bool,
        )
        item["pbo_start_image_index"] = torch.tensor(
            [example["_pbo_start_image_index"]],
            dtype=torch.long,
        )
        item["forward_dynamics_valid_mask"] = torch.tensor(
            [example["_forward_dynamics_valid"]],
            dtype=torch.bool,
        )

        if "mm_token_type_ids" in encoded:
            item["mm_token_type_ids"] = encoded["mm_token_type_ids"].squeeze(0)

        if self.panovggt_enabled and vision_paths:
            item["panovggt_pixel_values"] = preprocess_panovggt_current_image(
                raw_images[-1]
            ).unsqueeze(0)

        if example["_forward_dynamics_valid"]:
            target_path = _resolve_image_path(
                example["_forward_target_image"],
                self.image_root,
            )
            current_path = vision_paths[-1]
            if os.path.abspath(target_path) == os.path.abspath(current_path):
                target_raw_image = raw_images[-1]
            else:
                with Image.open(target_path) as image:
                    target_raw_image = image.convert("RGB")
            item["forward_target_panovggt_pixel_values"] = (
                preprocess_panovggt_current_image(target_raw_image).unsqueeze(0)
            )

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
    "pbo_action_labels",
    "pbo_valid_mask",
    "pbo_start_image_index",
    "forward_dynamics_valid_mask",
    "forward_target_panovggt_pixel_values",
)
