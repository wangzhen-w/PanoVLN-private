from __future__ import annotations

import io
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence

import torch
from PIL import Image
from transformers import AutoProcessor


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
for _path in (str(REPO_ROOT), str(SRC_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from src.qwen_vl import Qwen3_5Config, Qwen3_5ForConditionalGenerationForPanoVLN
from src.train.data.data import (
    DEFAULT_ERP_BOTTOM_CROP_DEGREES,
    DEFAULT_ERP_TOP_CROP_DEGREES,
    DEFAULT_VLN_MAX_MEMORY_IMAGES,
    DEFAULT_VLN_MEMORY_POOL_WINDOW_FRAMES,
    VLN_SYSTEM_PROMPT,
    build_erp_image_geometry_batch,
    build_vln_image_selection,
    build_vln_user_content,
    preprocess_panovggt_current_image,
    preprocess_vln_current_image,
    preprocess_vln_memory_image,
    resolve_current_image_index,
    text_content,
)
from src.train.utils import build_prompt_and_target, sync_model_special_tokens


DEFAULT_MODEL_PATH = (
    "/workspace/data1/model/ablation_new/panovggt_new/"
    "panovggt_0.05_grouping_8card"
)
ACTION_WORDS = ("stop", "forward", "left", "right")
ACTION_SEQUENCE_LENGTH = 4
REPLAN_ACTION_COUNT_WITHOUT_STOP = 4
DEFAULT_REALWORLD_GENERATION_KWARGS = {
    "max_new_tokens": 24,
    "temperature": 0,
    "top_p": None,
    "num_beams": 1,
}
ATOMIC_ACTION_VARIANTS = {
    "stop": ("stop",),
    "forward": ("forward", "move_forward", "move forward", "move-forward"),
    "left": ("left", "turn_left", "turn left", "turn-left"),
    "right": ("right", "turn_right", "turn right", "turn-right"),
}
ATOMIC_ACTION_PATTERNS = [
    (
        action_name,
        re.compile(
            r"(?:"
            + "|".join(
                r"(?<![0-9a-z_])" + re.escape(variant.lower()) + r"(?![0-9a-z_])"
                for variant in ATOMIC_ACTION_VARIANTS[action_name]
            )
            + r")"
        ),
    )
    for action_name in ACTION_WORDS
]


@dataclass
class InferenceConfig:
    model_path: str = DEFAULT_MODEL_PATH
    panovggt_checkpoint_path: Optional[str] = None
    attn_implementation: Optional[str] = "flash_attention_2"
    max_memory_images: int = DEFAULT_VLN_MAX_MEMORY_IMAGES
    memory_pool_window_frames: int = DEFAULT_VLN_MEMORY_POOL_WINDOW_FRAMES


@dataclass
class PredictionResult:
    actions: list[str]
    executable_actions: list[str]
    raw_text: str
    prompt_images: int
    latency_s: float


def _load_image(image: Image.Image | bytes | bytearray | str | os.PathLike[str]) -> Image.Image:
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, (bytes, bytearray)):
        return Image.open(io.BytesIO(image)).convert("RGB")
    return Image.open(image).convert("RGB")


def _select_images(
    images: Sequence[Image.Image],
    *,
    max_memory_images: int,
    memory_pool_window_frames: int,
) -> list[Image.Image]:
    if not images:
        raise ValueError("At least one image is required")
    if max_memory_images <= 0:
        return [images[-1]]
    selected_indices = build_vln_image_selection(
        current_step=len(images) - 1,
        last_frame_index=len(images) - 1,
        max_memory_images=max_memory_images,
        memory_pool_window_frames=memory_pool_window_frames,
    )
    return [images[index] for index in selected_indices]


def parse_action_sequence(text: str, max_actions: int = ACTION_SEQUENCE_LENGTH) -> list[str]:
    if "</think>" in text.lower():
        text = re.split(r"</think>", text, flags=re.IGNORECASE)[-1]

    action_text = " ".join(text.split()).lower().strip(" \t\r\n`'\".,;:!?()[]{}")
    if not action_text:
        return []

    matches = []
    for action_name, pattern in ATOMIC_ACTION_PATTERNS:
        for match in pattern.finditer(action_text):
            matches.append((match.start(), match.end(), action_name))

    matches.sort(key=lambda item: (item[0], -(item[1] - item[0])))
    actions = []
    last_end = -1
    for start, end, action_name in matches:
        if start < last_end:
            continue
        actions.append(action_name)
        last_end = end
        if len(actions) >= max_actions:
            break

    return actions


def build_executable_action_queue(actions: Iterable[str]) -> list[str]:
    action_list = list(actions)[:ACTION_SEQUENCE_LENGTH]
    if not action_list:
        return ["stop"]
    if "stop" in action_list:
        return action_list[: action_list.index("stop") + 1]
    return action_list[:REPLAN_ACTION_COUNT_WITHOUT_STOP]


class PanoVLNPredictor:
    def __init__(self, config: InferenceConfig):
        self.config = config
        self.processor = None
        self.tokenizer = None
        self.model = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.erp_top_crop_degrees = DEFAULT_ERP_TOP_CROP_DEGREES
        self.erp_bottom_crop_degrees = DEFAULT_ERP_BOTTOM_CROP_DEGREES
        self._load()

    @staticmethod
    def _checkpoint_has_weight_prefix(checkpoint_path: str, prefix: str) -> bool | None:
        checkpoint_dir = Path(str(checkpoint_path))
        if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
            return None

        for index_name in ("model.safetensors.index.json", "pytorch_model.bin.index.json"):
            index_path = checkpoint_dir / index_name
            if not index_path.exists():
                continue
            try:
                weight_map = json.loads(index_path.read_text()).get("weight_map", {})
            except Exception:
                return None
            return any(str(key).startswith(prefix) for key in weight_map)

        safetensors_path = checkpoint_dir / "model.safetensors"
        if safetensors_path.exists():
            try:
                from safetensors import safe_open

                with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
                    return any(str(key).startswith(prefix) for key in handle.keys())
            except Exception:
                return None

        torch_path = checkpoint_dir / "pytorch_model.bin"
        if torch_path.exists():
            try:
                state_dict = torch.load(str(torch_path), map_location="cpu")
                if (
                    isinstance(state_dict, dict)
                    and "state_dict" in state_dict
                    and isinstance(state_dict["state_dict"], dict)
                ):
                    state_dict = state_dict["state_dict"]
                if not isinstance(state_dict, dict):
                    return None
                return any(str(key).startswith(prefix) for key in state_dict)
            except Exception:
                return None

        return None

    def _resolve_panovggt_checkpoint(self, model_config: Any) -> None:
        if not bool(getattr(model_config, "panovggt_enabled", False)):
            return

        has_saved_panovggt_weights = self._checkpoint_has_weight_prefix(
            self.config.model_path,
            "panovggt.",
        )
        if has_saved_panovggt_weights is True:
            return

        selected = self.config.panovggt_checkpoint_path
        if selected:
            if Path(str(selected)).exists():
                setattr(model_config, "panovggt_checkpoint_path", str(selected))
                return
            raise FileNotFoundError(f"Explicit PanoVGGT checkpoint does not exist: {selected}")

        config_checkpoint = getattr(model_config, "panovggt_checkpoint_path", None)
        if config_checkpoint and Path(str(config_checkpoint)).exists():
            return

        if has_saved_panovggt_weights is None:
            return

        raise FileNotFoundError(
            "PanoVGGT is enabled, but the VLN checkpoint does not appear to contain "
            "saved PanoVGGT weights and config.json does not point to an existing "
            "PanoVGGT checkpoint. Pass --panovggt-checkpoint with the correct model.pt path."
        )

    def _load(self) -> None:
        model_path = self.config.model_path
        self.processor = AutoProcessor.from_pretrained(
            model_path,
            use_fast=True,
        )
        if hasattr(self.processor, "tokenizer"):
            self.tokenizer = self.processor.tokenizer
        else:
            raise ValueError("Real-world inference requires a processor tokenizer")
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        model_config = Qwen3_5Config.from_pretrained(model_path)
        self._resolve_panovggt_checkpoint(model_config)

        kwargs: dict[str, Any] = {
            "config": model_config,
            "torch_dtype": torch.bfloat16,
        }
        if self.config.attn_implementation:
            kwargs["attn_implementation"] = self.config.attn_implementation

        self.model = Qwen3_5ForConditionalGenerationForPanoVLN.from_pretrained(
            model_path,
            **kwargs,
        )
        self.model.to(self.device)
        self.model.eval()
        self.model.config.use_cache = True
        if hasattr(self.model.config, "text_config") and self.model.config.text_config is not None:
            self.model.config.text_config.use_cache = True
        sync_model_special_tokens(self.model, self.tokenizer)
        self.erp_top_crop_degrees = float(
            getattr(self.model.config, "erp_top_crop_degrees", DEFAULT_ERP_TOP_CROP_DEGREES)
        )
        self.erp_bottom_crop_degrees = float(
            getattr(self.model.config, "erp_bottom_crop_degrees", DEFAULT_ERP_BOTTOM_CROP_DEGREES)
        )

    def _input_device(self) -> torch.device:
        return next(self.model.parameters()).device

    def _prepare_images(self, raw_images: Sequence[Image.Image]) -> tuple[list[Image.Image], torch.Tensor | None]:
        num_images = len(raw_images)
        processed_images: list[Image.Image] = []
        for image_index, image in enumerate(raw_images):
            if image_index == num_images - 1:
                processed_images.append(
                    preprocess_vln_current_image(
                        image,
                        top_crop_degrees=self.erp_top_crop_degrees,
                        bottom_crop_degrees=self.erp_bottom_crop_degrees,
                    )
                )
            else:
                processed_images.append(
                    preprocess_vln_memory_image(
                        image,
                        top_crop_degrees=self.erp_top_crop_degrees,
                        bottom_crop_degrees=self.erp_bottom_crop_degrees,
                    )
                )

        panovggt_enabled = bool(getattr(self.model.config, "panovggt_enabled", False))
        panovggt_pixel_values = None
        if panovggt_enabled:
            panovggt_pixel_values = preprocess_panovggt_current_image(raw_images[-1]).unsqueeze(0)
        return processed_images, panovggt_pixel_values

    def _move_batch_to_device(self, batch: dict[str, Any]) -> dict[str, Any]:
        device = self._input_device()
        moved: dict[str, Any] = {}
        for key, value in batch.items():
            if torch.is_tensor(value):
                moved[key] = value.to(device=device)
            else:
                moved[key] = value
        return moved

    def predict(
        self,
        instruction: str,
        images: Sequence[Image.Image | bytes | bytearray | str | os.PathLike[str]],
    ) -> PredictionResult:
        instruction = instruction.strip()
        if not instruction:
            raise ValueError("instruction must be non-empty")
        start = time.perf_counter()

        loaded_images = [_load_image(image) for image in images]
        selected_images = _select_images(
            loaded_images,
            max_memory_images=self.config.max_memory_images,
            memory_pool_window_frames=self.config.memory_pool_window_frames,
        )
        processed_images, panovggt_pixel_values = self._prepare_images(selected_images)

        messages = [
            {
                "role": "system",
                "content": [text_content(VLN_SYSTEM_PROMPT)],
            },
            {
                "role": "user",
                "content": build_vln_user_content(
                    instruction=instruction,
                    num_images=len(processed_images),
                ),
            },
        ]
        prompt_text = build_prompt_and_target(
            messages,
            "chat_template",
            processor=self.processor,
            require_target=False,
        )["prompt"]

        encoded = self.processor(
            text=[prompt_text],
            images=processed_images,
            return_tensors="pt",
            padding=True,
        )
        image_count = len(processed_images)
        encoded["image_erp_geometry"] = build_erp_image_geometry_batch(
            image_count,
            top_crop_degrees=self.erp_top_crop_degrees,
            bottom_crop_degrees=self.erp_bottom_crop_degrees,
        )
        encoded["image_num_images"] = torch.tensor([image_count], dtype=torch.long)
        encoded["image_current_index"] = torch.tensor(
            [resolve_current_image_index(image_count)],
            dtype=torch.long,
        )
        if panovggt_pixel_values is not None:
            encoded["panovggt_pixel_values"] = panovggt_pixel_values

        batch = self._move_batch_to_device(dict(encoded))
        input_len = int(batch["input_ids"].shape[-1])
        generation_kwargs = dict(DEFAULT_REALWORLD_GENERATION_KWARGS)

        with torch.inference_mode():
            generated = self.model.generate(
                **batch,
                eos_token_id=self.tokenizer.eos_token_id,
                pad_token_id=self.tokenizer.pad_token_id,
                do_sample=True if generation_kwargs["temperature"] > 0 else False,
                temperature=generation_kwargs["temperature"],
                top_p=generation_kwargs["top_p"],
                num_beams=generation_kwargs["num_beams"],
                max_new_tokens=generation_kwargs["max_new_tokens"],
            )

        new_tokens = generated[:, input_len:]
        raw_text = self.processor.batch_decode(
            new_tokens,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )[0]
        if "</think>" in raw_text:
            raw_text = raw_text.split("</think>", 1)[-1]
        raw_text = raw_text.strip()

        actions = parse_action_sequence(raw_text)
        return PredictionResult(
            actions=actions,
            executable_actions=build_executable_action_queue(actions),
            raw_text=raw_text,
            prompt_images=image_count,
            latency_s=time.perf_counter() - start,
        )
