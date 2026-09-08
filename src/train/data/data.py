import json
import math
import os
import random
from functools import lru_cache
from typing import Any, Dict, List, Optional

import torch
import torch.nn.functional as F
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
DEFAULT_VLN_ACTION_SEQUENCE_LENGTH = 4
# Backward-compatible name used by the existing DAgger collector.  Policy
# training/evaluation use the value stored in model.config instead.
VLN_ACTION_SEQUENCE_LENGTH = DEFAULT_VLN_ACTION_SEQUENCE_LENGTH
VLN_VIEW_MODES = {"panorama", "erp_180", "perspective"}
DEFAULT_VLN_VIEW_MODE = "panorama"
DEFAULT_PERSPECTIVE_XFOV_DEGREES = 90.0
DEFAULT_PERSPECTIVE_YFOV_DEGREES = 90.0
DEFAULT_PERSPECTIVE_IMAGE_WIDTH = 320
DEFAULT_PERSPECTIVE_IMAGE_HEIGHT = 320
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
ACTION_COUNT_WORDS = {
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
    10: "ten",
    11: "eleven",
    12: "twelve",
    13: "thirteen",
    14: "fourteen",
    15: "fifteen",
    16: "sixteen",
}


def validate_action_sequence_length(action_sequence_length: int) -> int:
    if isinstance(action_sequence_length, bool):
        raise ValueError("action_sequence_length must be a positive integer")
    action_sequence_length = int(action_sequence_length)
    if action_sequence_length <= 0:
        raise ValueError(
            "action_sequence_length must be a positive integer, "
            f"got {action_sequence_length}"
        )
    return action_sequence_length


def normalize_vln_view_mode(view_mode: str) -> str:
    normalized = str(view_mode).strip().lower()
    if normalized not in VLN_VIEW_MODES:
        raise ValueError(
            f"view_mode must be one of {sorted(VLN_VIEW_MODES)}, got {view_mode!r}"
        )
    return normalized


def build_vln_system_prompt(
    action_sequence_length: int = DEFAULT_VLN_ACTION_SEQUENCE_LENGTH,
) -> str:
    action_sequence_length = validate_action_sequence_length(action_sequence_length)
    count_text = ACTION_COUNT_WORDS.get(
        action_sequence_length,
        str(action_sequence_length),
    )
    action_word = "action word" if action_sequence_length == 1 else "action words"
    action_count_noun = "action" if action_sequence_length == 1 else "actions"
    return (
        "You are an autonomous navigation assistant. "
        "Your task is to follow the navigation instruction. "
        "Given the instruction, your recent observations, and your current observation, "
        "devise an action sequence using the four actions: left or right by 15 degrees, "
        "forward by 25 centimeters, or stop once the task is complete. "
        f"Return exactly {count_text} {action_word} in execution order, separated by spaces. "
        f"If the task is complete before {count_text} {action_count_noun}, "
        "fill the remaining positions "
        "with stop."
    )


VLN_SYSTEM_PROMPT = build_vln_system_prompt()


@lru_cache(maxsize=32)
def _perspective_sampling_grid(
    output_width: int,
    output_height: int,
    xfov_degrees: float,
    yfov_degrees: float,
) -> torch.Tensor:
    output_width = int(output_width)
    output_height = int(output_height)
    xfov_degrees = float(xfov_degrees)
    yfov_degrees = float(yfov_degrees)
    if output_width <= 0 or output_height <= 0:
        raise ValueError(
            "Perspective output dimensions must be positive, "
            f"got {output_width}x{output_height}"
        )
    if not 0.0 < xfov_degrees < 180.0 or not 0.0 < yfov_degrees < 180.0:
        raise ValueError(
            "Perspective FOV values must be in (0, 180) degrees, "
            f"got xfov={xfov_degrees}, yfov={yfov_degrees}"
        )

    x = (
        (torch.arange(output_width, dtype=torch.float32) + 0.5)
        / float(output_width)
        * 2.0
        - 1.0
    )
    y = 1.0 - (
        (torch.arange(output_height, dtype=torch.float32) + 0.5)
        / float(output_height)
        * 2.0
    )
    ray_x = x * math.tan(math.radians(xfov_degrees) * 0.5)
    ray_y = y * math.tan(math.radians(yfov_degrees) * 0.5)
    ray_x = ray_x.unsqueeze(0).expand(output_height, output_width)
    ray_y = ray_y.unsqueeze(1).expand(output_height, output_width)
    longitude = torch.atan2(ray_x, torch.ones_like(ray_x))
    latitude = torch.atan2(ray_y, torch.sqrt(1.0 + ray_x.square()))
    return torch.stack(
        [
            longitude / math.pi,
            -2.0 * latitude / math.pi,
        ],
        dim=-1,
    ).unsqueeze(0)


def project_equirectangular_to_perspective(
    image: Image.Image,
    *,
    xfov_degrees: float = DEFAULT_PERSPECTIVE_XFOV_DEGREES,
    yfov_degrees: float = DEFAULT_PERSPECTIVE_YFOV_DEGREES,
    output_width: int = DEFAULT_PERSPECTIVE_IMAGE_WIDTH,
    output_height: int = DEFAULT_PERSPECTIVE_IMAGE_HEIGHT,
) -> Image.Image:
    """Project the forward-facing center of an ERP panorama at runtime."""

    source = TF.to_tensor(image.convert("RGB")).unsqueeze(0)
    grid = _perspective_sampling_grid(
        int(output_width),
        int(output_height),
        float(xfov_degrees),
        float(yfov_degrees),
    ).to(device=source.device)
    projected = F.grid_sample(
        source,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=False,
    )[0].clamp_(0.0, 1.0)
    return TF.to_pil_image(projected)


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


def crop_erp_center_180(image: Image.Image) -> Image.Image:
    """Crop the forward-facing central 180 degrees from a full 360-degree ERP."""

    width, height = image.size
    if width <= 1 or height <= 0:
        return image

    crop_width = max(1, width // 2)
    crop_left = (width - crop_width) // 2
    return image.crop((crop_left, 0, crop_left + crop_width, height))


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
    view_mode: str = DEFAULT_VLN_VIEW_MODE,
    perspective_xfov_degrees: float = DEFAULT_PERSPECTIVE_XFOV_DEGREES,
    perspective_yfov_degrees: float = DEFAULT_PERSPECTIVE_YFOV_DEGREES,
    perspective_image_width: int = DEFAULT_PERSPECTIVE_IMAGE_WIDTH,
    perspective_image_height: int = DEFAULT_PERSPECTIVE_IMAGE_HEIGHT,
) -> Image.Image:
    view_mode = normalize_vln_view_mode(view_mode)
    if view_mode == "perspective":
        return project_equirectangular_to_perspective(
            image,
            xfov_degrees=perspective_xfov_degrees,
            yfov_degrees=perspective_yfov_degrees,
            output_width=perspective_image_width,
            output_height=perspective_image_height,
        )
    processed_image = image.convert("RGB").resize(DEFAULT_VLN_CURRENT_OBSERVATION_IMAGE_SIZE)
    if view_mode == "erp_180":
        return crop_erp_center_180(processed_image)
    return crop_erp_latitude(
        processed_image,
        top_crop_degrees=top_crop_degrees,
        bottom_crop_degrees=bottom_crop_degrees,
    )


def preprocess_vln_memory_image(
    image: Image.Image,
    top_crop_degrees: float = DEFAULT_ERP_TOP_CROP_DEGREES,
    bottom_crop_degrees: float = DEFAULT_ERP_BOTTOM_CROP_DEGREES,
    view_mode: str = DEFAULT_VLN_VIEW_MODE,
    perspective_xfov_degrees: float = DEFAULT_PERSPECTIVE_XFOV_DEGREES,
    perspective_yfov_degrees: float = DEFAULT_PERSPECTIVE_YFOV_DEGREES,
    perspective_image_width: int = DEFAULT_PERSPECTIVE_IMAGE_WIDTH,
    perspective_image_height: int = DEFAULT_PERSPECTIVE_IMAGE_HEIGHT,
) -> Image.Image:
    view_mode = normalize_vln_view_mode(view_mode)
    if view_mode == "perspective":
        return project_equirectangular_to_perspective(
            image,
            xfov_degrees=perspective_xfov_degrees,
            yfov_degrees=perspective_yfov_degrees,
            output_width=perspective_image_width,
            output_height=perspective_image_height,
        )
    processed_image = image.convert("RGB").resize(DEFAULT_VLN_MEMORY_IMAGE_SIZE)
    if view_mode == "erp_180":
        return crop_erp_center_180(processed_image)
    return crop_erp_latitude(
        processed_image,
        top_crop_degrees=top_crop_degrees,
        bottom_crop_degrees=bottom_crop_degrees,
    )


def preprocess_panovggt_current_image(
    image: Image.Image,
    view_mode: str = DEFAULT_VLN_VIEW_MODE,
    perspective_xfov_degrees: float = DEFAULT_PERSPECTIVE_XFOV_DEGREES,
    perspective_yfov_degrees: float = DEFAULT_PERSPECTIVE_YFOV_DEGREES,
    perspective_image_width: int = DEFAULT_PERSPECTIVE_IMAGE_WIDTH,
    perspective_image_height: int = DEFAULT_PERSPECTIVE_IMAGE_HEIGHT,
) -> torch.Tensor:
    processed_image = image.convert("RGB")
    view_mode = normalize_vln_view_mode(view_mode)
    if view_mode == "perspective":
        # Never expose the original 360-degree image to the auxiliary visual
        # encoder during a perspective-view ablation.  Keep the configured
        # 320x320 projection for every visual branch.
        processed_image = project_equirectangular_to_perspective(
            processed_image,
            xfov_degrees=perspective_xfov_degrees,
            yfov_degrees=perspective_yfov_degrees,
            output_width=perspective_image_width,
            output_height=perspective_image_height,
        )
    else:
        processed_image = processed_image.resize(
            DEFAULT_PANOVGGT_IMAGE_SIZE,
            Image.Resampling.LANCZOS,
        )
        if view_mode == "erp_180":
            processed_image = crop_erp_center_180(processed_image)
    return TF.to_tensor(processed_image)


def build_vln_image_geometry_batch(
    num_images: int,
    *,
    view_mode: str = DEFAULT_VLN_VIEW_MODE,
    top_crop_degrees: float = DEFAULT_ERP_TOP_CROP_DEGREES,
    bottom_crop_degrees: float = DEFAULT_ERP_BOTTOM_CROP_DEGREES,
) -> Optional[torch.Tensor]:
    view_mode = normalize_vln_view_mode(view_mode)
    if view_mode == "perspective":
        # The PanoVGGT/Qwen alignment uses the full normalized image plane for
        # perspective inputs; ERP latitude/longitude metadata does not apply.
        return None
    if view_mode == "erp_180":
        vertical_geometry = build_erp_image_geometry_batch(
            num_images,
            top_crop_degrees=0.0,
            bottom_crop_degrees=0.0,
        )
        horizontal_geometry = torch.tensor(
            [[math.pi, 0.0]],
            dtype=torch.float32,
        ).repeat(max(0, int(num_images)), 1)
        return torch.cat((vertical_geometry, horizontal_geometry), dim=1)
    return build_erp_image_geometry_batch(
        num_images,
        top_crop_degrees=top_crop_degrees,
        bottom_crop_degrees=bottom_crop_degrees,
    )


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

    if total_selected_images == 1:
        return [current_frame_index]

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


def build_vln_user_content(
    instruction: str,
    num_images: int,
    view_mode: str = DEFAULT_VLN_VIEW_MODE,
) -> List[Dict[str, str]]:
    if num_images <= 0:
        raise ValueError("VLN samples require at least one image")

    view_mode = normalize_vln_view_mode(view_mode)
    num_memory_images = max(0, num_images - 1)
    content = [text_content(f"Instruction: {instruction.strip()}")]

    if num_memory_images > 0:
        if view_mode == "panorama":
            memory_text = (
                "\nHistory memory observations are panoramic views "
                "ordered from older to newer:"
            )
        else:
            memory_text = (
                "\nHistory memory observations are ordered from older to newer:"
            )
        content.append(
            text_content(memory_text)
        )
        content.extend(image_content() for _ in range(num_memory_images))

    current_text = (
        "\nCurrent observation (panoramic view):"
        if view_mode == "panorama"
        else "\nCurrent observation:"
    )
    content.extend(
        [
            text_content(current_text),
            image_content(),
            text_content("\nDevise the next action sequence."),
        ]
    )
    return content


def _resolve_image_path(path: str, image_root: Optional[str]) -> str:
    if os.path.isabs(path) or path.startswith(("http://", "https://", "file://")):
        resolved_path = path
    elif image_root is None:
        resolved_path = path
    else:
        resolved_path = os.path.join(image_root, path)

    # Existing VLN JSONL files reference frame_*.jpg.  Allow the extraction
    # pipeline to switch to lossless PNG without rewriting those large files.
    if (
        not resolved_path.startswith(("http://", "https://", "file://"))
        and not os.path.exists(resolved_path)
    ):
        stem, extension = os.path.splitext(resolved_path)
        if extension.lower() in {".jpg", ".jpeg"}:
            png_path = f"{stem}.png"
            if os.path.exists(png_path):
                return png_path
    return resolved_path


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


def _extract_vln_action_sequence(
    example: Dict[str, Any],
    action_sequence_length: int,
) -> List[str]:
    action_sequence_length = validate_action_sequence_length(action_sequence_length)
    action_sequence = example.get("action_sequence")
    if not isinstance(action_sequence, list):
        raise ValueError("VLN example field 'action_sequence' must be a list")
    if len(action_sequence) != action_sequence_length:
        raise ValueError(
            "VLN example field 'action_sequence' must contain exactly "
            f"{action_sequence_length} actions, got {len(action_sequence)}"
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
    action_sequence_length: int,
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
    if value < 1 or value > action_sequence_length:
        raise ValueError(
            "VLN example field 'real_action_count' must be in "
            f"[1, {action_sequence_length}], "
            f"got {value}"
        )
    if value != expected_count:
        raise ValueError(
            "VLN real_action_count/action_sequence mismatch: "
            f"real_action_count={value}, expected={expected_count}, "
            f"action_sequence={action_sequence}"
        )
    return int(value)


def apply_vln_memory_policy(
    example: Dict[str, Any],
    *,
    action_sequence_length: int = DEFAULT_VLN_ACTION_SEQUENCE_LENGTH,
    view_mode: str = DEFAULT_VLN_VIEW_MODE,
) -> Dict[str, Any]:
    action_sequence_length = validate_action_sequence_length(action_sequence_length)
    view_mode = normalize_vln_view_mode(view_mode)
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
    action_sequence = _extract_vln_action_sequence(example, action_sequence_length)
    _extract_vln_history_actions(example, current_step)
    _extract_real_action_count(example, action_sequence, action_sequence_length)

    selected_indices = build_vln_image_selection(
        current_step=current_step,
        last_frame_index=len(raw_images) - 1,
    )
    selected_images = [raw_images[index] for index in selected_indices]

    user_content = build_vln_user_content(
        instruction=instruction,
        num_images=len(selected_images),
        view_mode=view_mode,
    )

    normalized = dict(example)
    normalized["images"] = selected_images
    normalized["messages"] = [
        {
            "role": "system",
            "content": [text_content(build_vln_system_prompt(action_sequence_length))],
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
        action_sequence_length: int = DEFAULT_VLN_ACTION_SEQUENCE_LENGTH,
        view_mode: str = DEFAULT_VLN_VIEW_MODE,
        perspective_xfov_degrees: float = DEFAULT_PERSPECTIVE_XFOV_DEGREES,
        perspective_yfov_degrees: float = DEFAULT_PERSPECTIVE_YFOV_DEGREES,
        perspective_image_width: int = DEFAULT_PERSPECTIVE_IMAGE_WIDTH,
        perspective_image_height: int = DEFAULT_PERSPECTIVE_IMAGE_HEIGHT,
        erp_top_crop_degrees: float = DEFAULT_ERP_TOP_CROP_DEGREES,
        erp_bottom_crop_degrees: float = DEFAULT_ERP_BOTTOM_CROP_DEGREES,
        panovggt_enabled: bool = False,
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
        self.action_sequence_length = validate_action_sequence_length(
            action_sequence_length
        )
        self.view_mode = normalize_vln_view_mode(view_mode)
        self.perspective_xfov_degrees = float(perspective_xfov_degrees)
        self.perspective_yfov_degrees = float(perspective_yfov_degrees)
        self.perspective_image_width = int(perspective_image_width)
        self.perspective_image_height = int(perspective_image_height)
        self.erp_top_crop_degrees = float(erp_top_crop_degrees)
        self.erp_bottom_crop_degrees = float(erp_bottom_crop_degrees)
        self.panovggt_enabled = bool(panovggt_enabled)
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
                        view_mode=self.view_mode,
                        perspective_xfov_degrees=self.perspective_xfov_degrees,
                        perspective_yfov_degrees=self.perspective_yfov_degrees,
                        perspective_image_width=self.perspective_image_width,
                        perspective_image_height=self.perspective_image_height,
                    )
                else:
                    processed_image = preprocess_vln_memory_image(
                        raw_image,
                        top_crop_degrees=self.erp_top_crop_degrees,
                        bottom_crop_degrees=self.erp_bottom_crop_degrees,
                        view_mode=self.view_mode,
                        perspective_xfov_degrees=self.perspective_xfov_degrees,
                        perspective_yfov_degrees=self.perspective_yfov_degrees,
                        perspective_image_width=self.perspective_image_width,
                        perspective_image_height=self.perspective_image_height,
                    )
                raw_images.append(raw_image)
                processed_images.append(processed_image)
        return processed_images, raw_images

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = apply_vln_memory_policy(
            self._load_example(index),
            action_sequence_length=self.action_sequence_length,
            view_mode=self.view_mode,
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
        image_erp_geometry = build_vln_image_geometry_batch(
            image_count,
            view_mode=self.view_mode,
            top_crop_degrees=self.erp_top_crop_degrees,
            bottom_crop_degrees=self.erp_bottom_crop_degrees,
        )
        item = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
        if image_erp_geometry is not None:
            item["image_erp_geometry"] = image_erp_geometry
        item["image_num_images"] = torch.tensor([image_count], dtype=torch.long)
        item["image_current_index"] = torch.tensor(
            [resolve_current_image_index(image_count)],
            dtype=torch.long,
        )

        if "mm_token_type_ids" in encoded:
            item["mm_token_type_ids"] = encoded["mm_token_type_ids"].squeeze(0)

        if self.panovggt_enabled and vision_paths:
            item["panovggt_pixel_values"] = preprocess_panovggt_current_image(
                raw_images[-1],
                view_mode=self.view_mode,
                perspective_xfov_degrees=self.perspective_xfov_degrees,
                perspective_yfov_degrees=self.perspective_yfov_degrees,
                perspective_image_width=self.perspective_image_width,
                perspective_image_height=self.perspective_image_height,
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
)
