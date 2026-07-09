import copy
import json
import os
from typing import Any, Dict, List, Optional

import torch
from PIL import Image
from torch.utils.data import Dataset

try:
    from src.train.data.data import (
        DEFAULT_VLN_CURRENT_OBSERVATION_IMAGE_SIZE,
        build_erp_image_geometry_batch,
        crop_erp_latitude,
        preprocess_panovggt_current_image,
        resolve_current_image_index,
    )
    from src.train.utils import build_prompt_and_target
except ModuleNotFoundError:
    from data.data import (
        DEFAULT_VLN_CURRENT_OBSERVATION_IMAGE_SIZE,
        build_erp_image_geometry_batch,
        crop_erp_latitude,
        preprocess_panovggt_current_image,
        resolve_current_image_index,
    )
    from utils import build_prompt_and_target


def text_content(text: str) -> Dict[str, str]:
    return {"type": "text", "text": text}


def image_content() -> Dict[str, str]:
    return {"type": "image"}


def _collect_image_refs(sample: Dict[str, Any]) -> List[str]:
    refs = sample.get("images", [])
    if refs is None:
        refs = []
    if not isinstance(refs, list):
        raise ValueError("PanoWorld sample field 'images' must be a list")
    return [str(ref) for ref in refs]


def _content_has_image(content: Any) -> int:
    if isinstance(content, str):
        return content.count("<image>")
    if isinstance(content, list):
        return sum(
            1
            for item in content
            if isinstance(item, dict) and item.get("type") == "image"
        )
    return 0


def _ensure_image_placeholders(content: Any, image_count: int) -> Any:
    if image_count <= 0:
        return content
    existing_images = _content_has_image(content)
    missing_images = max(0, image_count - existing_images)
    if missing_images == 0:
        return content

    if isinstance(content, str):
        prefix = "\n".join("<image>" for _ in range(missing_images))
        return f"{prefix}\n{content}" if content else prefix
    if isinstance(content, list):
        normalized = copy.deepcopy(content)
        normalized.extend(image_content() for _ in range(missing_images))
        return normalized
    return [image_content() for _ in range(missing_images)]


def _find_system_message_index(messages: List[Dict[str, Any]]) -> int:
    for index, message in enumerate(messages):
        if message.get("role") == "system":
            return index
    return -1


def _normalize_messages(
    sample: Dict[str, Any],
    *,
    system_prompt: Optional[str],
    auto_insert_media_placeholders: bool,
) -> List[Dict[str, Any]]:
    raw_messages = sample.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        raise ValueError("PanoWorld sample must contain non-empty 'messages'")

    messages = copy.deepcopy(raw_messages)
    system_index = _find_system_message_index(messages)
    if system_prompt:
        if system_index < 0:
            messages.insert(0, {"role": "system", "content": [text_content(system_prompt)]})
        elif not messages[system_index].get("content"):
            messages[system_index]["content"] = [text_content(system_prompt)]

    system_index = _find_system_message_index(messages)
    if system_index > 0:
        system_message = messages.pop(system_index)
        messages.insert(0, system_message)

    if auto_insert_media_placeholders:
        image_count = len(_collect_image_refs(sample))
        for message in messages:
            if message.get("role") == "user":
                message["content"] = _ensure_image_placeholders(
                    message.get("content", []),
                    image_count,
                )
                break

    return messages


def _resolve_image_path(path: str, image_root: Optional[str]) -> str:
    if os.path.isabs(path) or path.startswith(("http://", "https://", "file://")):
        return path
    if image_root:
        return os.path.abspath(os.path.join(image_root, path))
    return os.path.abspath(path)


def _load_image(path: str, image_root: Optional[str]) -> Image.Image:
    resolved_path = _resolve_image_path(path, image_root)
    with Image.open(resolved_path) as image:
        return image.convert("RGB")


class PanoWorldSupervisedDataset(Dataset):
    def __init__(
        self,
        jsonl_path: str,
        processor,
        tokenizer,
        image_root: Optional[str],
        image_token: str,
        model_max_length: Optional[int],
        erp_top_crop_degrees: float = 0.0,
        erp_bottom_crop_degrees: float = 0.0,
        panovggt_enabled: bool = False,
        max_samples: Optional[int] = None,
        prompt_format: str = "chat_template",
        system_prompt: Optional[str] = None,
        auto_insert_media_placeholders: bool = True,
    ):
        self.jsonl_path = os.path.abspath(jsonl_path)
        self.processor = processor
        self.tokenizer = tokenizer
        self.image_root = os.path.abspath(image_root) if image_root else None
        processor_image_token = getattr(processor, "image_token", None)
        self.image_token = processor_image_token or image_token
        self.model_max_length = model_max_length
        self.erp_top_crop_degrees = float(erp_top_crop_degrees)
        self.erp_bottom_crop_degrees = float(erp_bottom_crop_degrees)
        self.panovggt_enabled = bool(panovggt_enabled)
        self.prompt_format = prompt_format
        self.system_prompt = system_prompt
        self.auto_insert_media_placeholders = bool(auto_insert_media_placeholders)
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

        if max_samples is not None:
            entries = entries[:max_samples]
        self.offsets = entries

    def __len__(self) -> int:
        return len(self.offsets)

    def __del__(self):
        fp = getattr(self, "_fp", None)
        if fp is not None:
            fp.close()
            self._fp = None

    def _get_fp(self):
        if self._fp is None:
            self._fp = open(self.jsonl_path, "r", encoding="utf-8")
        return self._fp

    def _load_sample(self, index: int) -> Dict[str, Any]:
        handle = self._get_fp()
        handle.seek(self.offsets[index])
        return json.loads(handle.readline())

    def _load_images(self, image_refs: List[str]) -> List[Image.Image]:
        return [_load_image(ref, self.image_root) for ref in image_refs]

    def _preprocess_qwen_image(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB").resize(DEFAULT_VLN_CURRENT_OBSERVATION_IMAGE_SIZE)
        if self.erp_top_crop_degrees <= 0.0 and self.erp_bottom_crop_degrees <= 0.0:
            return image
        return crop_erp_latitude(
            image,
            top_crop_degrees=self.erp_top_crop_degrees,
            bottom_crop_degrees=self.erp_bottom_crop_degrees,
        )

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = self._load_sample(index)
        image_refs = _collect_image_refs(sample)
        messages = _normalize_messages(
            sample,
            system_prompt=self.system_prompt,
            auto_insert_media_placeholders=self.auto_insert_media_placeholders,
        )

        prompt_and_target = build_prompt_and_target(
            messages,
            self.prompt_format,
            processor=self.processor,
        )
        prompt_text = prompt_and_target["prompt"]
        target_text = prompt_and_target["target"]
        target_with_eos = target_text + (self.tokenizer.eos_token or "")
        full_text = prompt_text + target_with_eos

        raw_images = self._load_images(image_refs) if image_refs else []
        qwen_images = [self._preprocess_qwen_image(image) for image in raw_images]
        encode_kwargs = {
            "text": full_text,
            "return_tensors": "pt",
            "truncation": self.model_max_length is not None,
        }
        if self.model_max_length is not None:
            encode_kwargs["max_length"] = self.model_max_length
        if qwen_images:
            encoded = self.processor(images=qwen_images, **encode_kwargs)
        else:
            encoded = self.tokenizer(**encode_kwargs)

        target_kwargs = {
            "add_special_tokens": False,
            "return_tensors": "pt",
            "truncation": self.model_max_length is not None,
        }
        if self.model_max_length is not None:
            target_kwargs["max_length"] = self.model_max_length
        target_ids = self.tokenizer(target_with_eos, **target_kwargs)["input_ids"].squeeze(0)

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

        image_count = len(image_refs)
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

        if self.panovggt_enabled and raw_images:
            item["panovggt_pixel_values"] = preprocess_panovggt_current_image(
                raw_images[-1],
            ).unsqueeze(0)

        for key in ("pixel_values", "image_grid_thw", "pixel_values_videos", "video_grid_thw"):
            if key in encoded:
                item[key] = encoded[key]

        return item
