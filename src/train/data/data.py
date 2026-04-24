import json
import math
import os
import random
from functools import lru_cache
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import Dataset

try:
    from src.train.utils import build_prompt_and_target
except ModuleNotFoundError:
    from utils import build_prompt_and_target


DEFAULT_IMAGE_SIZE = (640, 320)
DEFAULT_VLN_MEMORY_IMAGE_SIZE = (256, 256)
DEFAULT_VLN_MEMORY_HORIZONTAL_FOV_DEGREES = 90.0
DEFAULT_VLN_CURRENT_OBSERVATION_IMAGE_SIZE = (640, 320)
DEFAULT_IDM_IMAGE_SIZE = (640, 320)
DEFAULT_ERP_TOP_CROP_DEGREES = 20
DEFAULT_ERP_BOTTOM_CROP_DEGREES = 20
SUPPORTED_TASK_TYPES = {"vln", "idm"}
LONGITUDE_PROMPT_STEP_DEG = 15
LONGITUDE_PROMPT_LABEL_STEP_DEG = 15
LONGITUDE_PROMPT_LINE_WIDTH_PX = 1
LONGITUDE_PROMPT_COLOR = (0, 255, 80)
LONGITUDE_PROMPT_ALPHA = 72
LONGITUDE_PROMPT_LABEL_BOTTOM_MARGIN_PX = 4
LONGITUDE_PROMPT_LABEL_FONT_SIZE = 7
LONGITUDE_PROMPT_LABEL_PADDING_PX = 0
LONGITUDE_PROMPT_LABEL_SHADOW_ALPHA = 80
LONGITUDE_PROMPT_FONT_PATHS = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/liberation2/LiberationSans-Regular.ttf",
)


def resolve_runtime_image_size(image_size):
    if image_size is None:
        return DEFAULT_IMAGE_SIZE
    if len(image_size) != 2:
        raise ValueError(f"image_size must contain exactly 2 integers, got {image_size}")
    return (int(image_size[0]), int(image_size[1]))


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


@lru_cache(maxsize=8)
def _build_perspective_sampling_grid(
    source_width: int,
    source_height: int,
    output_width: int,
    output_height: int,
    horizontal_fov_degrees: float,
):
    horizontal_fov_radians = math.radians(float(horizontal_fov_degrees))
    vertical_fov_radians = 2.0 * math.atan(
        math.tan(horizontal_fov_radians / 2.0) * (float(output_height) / float(output_width))
    )

    x_coordinates = np.linspace(
        -math.tan(horizontal_fov_radians / 2.0),
        math.tan(horizontal_fov_radians / 2.0),
        output_width,
        dtype=np.float32,
    )
    y_coordinates = np.linspace(
        math.tan(vertical_fov_radians / 2.0),
        -math.tan(vertical_fov_radians / 2.0),
        output_height,
        dtype=np.float32,
    )
    grid_x, grid_y = np.meshgrid(x_coordinates, y_coordinates)
    grid_z = np.ones_like(grid_x, dtype=np.float32)

    norm = np.sqrt(grid_x ** 2 + grid_y ** 2 + grid_z ** 2)
    dir_x = grid_x / norm
    dir_y = grid_y / norm
    dir_z = grid_z / norm

    longitude = np.arctan2(dir_x, dir_z)
    latitude = np.arcsin(np.clip(dir_y, -1.0, 1.0))

    sample_x = (longitude / (2.0 * math.pi) + 0.5) * float(source_width)
    sample_y = (0.5 - latitude / math.pi) * float(source_height)
    return sample_x.astype(np.float32), sample_y.astype(np.float32)


def project_erp_to_perspective(
    image: Image.Image,
    output_size=DEFAULT_VLN_MEMORY_IMAGE_SIZE,
    horizontal_fov_degrees: float = DEFAULT_VLN_MEMORY_HORIZONTAL_FOV_DEGREES,
) -> Image.Image:
    source = np.asarray(image.convert("RGB"), dtype=np.float32)
    source_height, source_width = source.shape[:2]
    output_width, output_height = int(output_size[0]), int(output_size[1])

    sample_x, sample_y = _build_perspective_sampling_grid(
        source_width=source_width,
        source_height=source_height,
        output_width=output_width,
        output_height=output_height,
        horizontal_fov_degrees=horizontal_fov_degrees,
    )

    base_x = np.floor(sample_x).astype(np.int32)
    base_y = np.floor(sample_y).astype(np.int32)
    next_x = base_x + 1
    next_y = base_y + 1

    base_x_wrapped = np.mod(base_x, source_width)
    next_x_wrapped = np.mod(next_x, source_width)
    base_y_clipped = np.clip(base_y, 0, source_height - 1)
    next_y_clipped = np.clip(next_y, 0, source_height - 1)

    weight_x = (sample_x - np.floor(sample_x)).astype(np.float32)[..., None]
    weight_y = (sample_y - np.floor(sample_y)).astype(np.float32)[..., None]

    top_left = source[base_y_clipped, base_x_wrapped]
    top_right = source[base_y_clipped, next_x_wrapped]
    bottom_left = source[next_y_clipped, base_x_wrapped]
    bottom_right = source[next_y_clipped, next_x_wrapped]

    top = top_left * (1.0 - weight_x) + top_right * weight_x
    bottom = bottom_left * (1.0 - weight_x) + bottom_right * weight_x
    projected = top * (1.0 - weight_y) + bottom * weight_y
    projected = np.clip(projected, 0.0, 255.0).astype(np.uint8)
    return Image.fromarray(projected, mode="RGB")


def preprocess_vln_memory_image(image: Image.Image) -> Image.Image:
    return project_erp_to_perspective(
        image=image,
        output_size=DEFAULT_VLN_MEMORY_IMAGE_SIZE,
        horizontal_fov_degrees=DEFAULT_VLN_MEMORY_HORIZONTAL_FOV_DEGREES,
    )


def preprocess_vln_current_image(
    image: Image.Image,
    add_visual_prompt: bool = False,
) -> Image.Image:
    processed_image = image.convert("RGB").resize(DEFAULT_VLN_CURRENT_OBSERVATION_IMAGE_SIZE)
    if add_visual_prompt:
        processed_image = add_longitude_visual_prompt(processed_image)
    return processed_image


def resolve_longitude_prompt_font():
    for font_path in LONGITUDE_PROMPT_FONT_PATHS:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, LONGITUDE_PROMPT_LABEL_FONT_SIZE)
    return ImageFont.load_default()


def format_longitude_label(longitude_deg: int) -> str:
    if longitude_deg > 0:
        return f"+{longitude_deg}°"
    return f"{longitude_deg}°"


def add_longitude_visual_prompt(
    image: Image.Image,
    line_step_deg: int = LONGITUDE_PROMPT_STEP_DEG,
) -> Image.Image:
    width, height = image.size
    if width <= 1 or height <= 0:
        return image

    base = image.convert("RGBA")
    overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = resolve_longitude_prompt_font()

    longitude_deg = -180
    while longitude_deg <= 180:
        x_position = round((longitude_deg + 180) / 360 * (width - 1))
        draw.line(
            ((x_position, 0), (x_position, height - 1)),
            fill=(*LONGITUDE_PROMPT_COLOR, LONGITUDE_PROMPT_ALPHA),
            width=LONGITUDE_PROMPT_LINE_WIDTH_PX,
        )
        if longitude_deg % LONGITUDE_PROMPT_LABEL_STEP_DEG == 0 and abs(longitude_deg) != 180:
            label_text = format_longitude_label(longitude_deg)
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            box_width = text_width + 2 * LONGITUDE_PROMPT_LABEL_PADDING_PX
            box_height = text_height + 2 * LONGITUDE_PROMPT_LABEL_PADDING_PX
            box_left = x_position - box_width // 2
            box_left = min(max(0, box_left), max(0, width - box_width - 1))
            box_top = height - box_height - LONGITUDE_PROMPT_LABEL_BOTTOM_MARGIN_PX
            box_top = max(0, box_top)
            box_right = box_left + box_width
            box_bottom = box_top + box_height
            draw.rounded_rectangle(
                ((box_left, box_top), (box_right, box_bottom)),
                radius=2,
                fill=(0, 0, 0, LONGITUDE_PROMPT_LABEL_SHADOW_ALPHA),
            )
            text_x = box_left + (box_width - text_width) / 2 - text_bbox[0]
            text_y = box_top + (box_height - text_height) / 2 - text_bbox[1]
            draw.text(
                (text_x, text_y),
                label_text,
                font=font,
                fill=LONGITUDE_PROMPT_COLOR,
            )
        longitude_deg += line_step_deg

    return Image.alpha_composite(base, overlay).convert("RGB")


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


def resolve_messages_and_vision_paths(example: Dict[str, Any], image_root: Optional[str]):
    vision_paths = []
    task_type = example.get("task type")
    if task_type is not None and task_type not in SUPPORTED_TASK_TYPES:
        raise NotImplementedError(f"Unsupported task type: {task_type}")

    raw_messages = example.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError(
            "Training example is missing prebuilt 'messages'. "
            "Regenerate the dataset with src/data/prepare_training_data.py."
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
        image_size: Optional[List[int]] = None,
        max_samples: Optional[int] = None,
        shuffle: bool = True,
        add_visual_prompt: bool = True,
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
        self.image_size = resolve_runtime_image_size(image_size)
        self.add_visual_prompt = add_visual_prompt
        self.prompt_format = prompt_format
        self._fp = None
        self._task_types = None

        offsets = []
        with open(self.jsonl_path, "rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)

        if shuffle:
            random.shuffle(offsets)

        if max_samples is not None:
            offsets = offsets[:max_samples]

        self.offsets = offsets

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

    def _resolve_image_processing(self, task_type: Optional[str], image_index: int, num_images: int):
        if task_type == "vln":
            if image_index == num_images - 1:
                return DEFAULT_VLN_CURRENT_OBSERVATION_IMAGE_SIZE, self.add_visual_prompt
            return DEFAULT_VLN_MEMORY_IMAGE_SIZE, False

        if task_type == "idm":
            return DEFAULT_IDM_IMAGE_SIZE, False

        return self.image_size, self.add_visual_prompt

    def _load_images(self, image_paths: List[str], task_type: Optional[str]):
        images = []
        num_images = len(image_paths)
        for image_index, image_path in enumerate(image_paths):
            with Image.open(image_path) as image:
                if task_type == "vln":
                    is_current_observation = image_index == num_images - 1
                    if is_current_observation:
                        processed_image = preprocess_vln_current_image(
                            image=image,
                            add_visual_prompt=self.add_visual_prompt,
                        )
                    else:
                        processed_image = preprocess_vln_memory_image(image)
                else:
                    resize_size, add_visual_prompt = self._resolve_image_processing(
                        task_type=task_type,
                        image_index=image_index,
                        num_images=num_images,
                    )
                    processed_image = image.convert("RGB").resize(resize_size)
                    processed_image = crop_erp_latitude(processed_image)
                    if add_visual_prompt:
                        processed_image = add_longitude_visual_prompt(processed_image)
                images.append(processed_image)
        return images

    def get_task_types(self) -> List[Optional[str]]:
        if self._task_types is not None:
            return self._task_types

        task_types = []
        with open(self.jsonl_path, "r", encoding="utf-8") as handle:
            for offset in self.offsets:
                handle.seek(offset)
                example = json.loads(handle.readline())
                task_types.append(example.get("task type"))

        self._task_types = task_types
        return self._task_types

    def __getitem__(self, index: int) -> Dict[str, Any]:
        example = self._load_example(index)
        task_type = example.get("task type")
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
                    task_type=task_type,
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

        if "mm_token_type_ids" in encoded:
            item["mm_token_type_ids"] = encoded["mm_token_type_ids"].squeeze(0)

        for key in STACKABLE_KEYS:
            if key in encoded:
                item[key] = encoded[key]

        return item


STACKABLE_KEYS = (
    "pixel_values",
    "image_grid_thw",
    "pixel_values_videos",
    "video_grid_thw",
)
