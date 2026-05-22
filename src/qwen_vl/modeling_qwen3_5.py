import json
import math
import sys
from pathlib import Path
from types import MethodType
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5CausalLMOutputWithPast,
    Qwen3_5ForConditionalGeneration,
    repeat_kv,
    rotate_half,
)


PANOVGGT_AGGREGATOR_LAYER = -1
PANOVGGT_CONTEXT_DIM = 2048
PANOVGGT_MLP_HIDDEN_SIZE = 4096
ACTION_CALIBRATOR_MAX_STEPS = 4
ACTION_CALIBRATOR_ACTIONS = ("stop", "forward", "left", "right")
ACTION_CALIBRATOR_MOVEMENT_ACTION_INDICES = (1, 2, 3)
ACTION_CALIBRATOR_FIRST_TOKEN_IDS = (9215, 13048, 2282, 1246)
ACTION_CALIBRATOR_NEXT_TOKEN_IDS = (2842, 4487, 2047, 1245)
ACTION_CALIBRATOR_TOKEN_TO_ACTION = {
    token_id: action_index
    for action_index, token_ids in enumerate(zip(ACTION_CALIBRATOR_FIRST_TOKEN_IDS, ACTION_CALIBRATOR_NEXT_TOKEN_IDS))
    for token_id in token_ids
}
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
VENDORED_PANOVGGT_DIR = SRC_ROOT / "panovggt"
VENDORED_PANOVGGT_CONFIG_PATH = VENDORED_PANOVGGT_DIR / "training" / "config" / "default.yaml"


def _inverse_sigmoid(value: float) -> float:
    value = min(max(float(value), 1e-6), 1.0 - 1e-6)
    return math.log(value / (1.0 - value))


def _bounded_raw_alpha(alpha_init: float, alpha_max: float) -> torch.Tensor:
    alpha_max = float(alpha_max)
    alpha_init = float(alpha_init)
    if alpha_max <= 0.0:
        return torch.tensor(0.0, dtype=torch.float32)
    alpha_init = min(max(alpha_init, 1e-6), alpha_max * (1.0 - 1e-6))
    return torch.tensor(_inverse_sigmoid(alpha_init / alpha_max), dtype=torch.float32)


def ensure_panovggt_config(config) -> None:
    vision_config = config.vision_config
    text_config = getattr(config, "text_config", None)
    text_hidden_size = getattr(text_config, "hidden_size", getattr(vision_config, "out_hidden_size", 3584))
    defaults = {
        "panovggt_enabled": False,
        "panovggt_checkpoint_path": "/workspace/code_dir/a_property/model/PanoVGGT/model.pt",
        "panovggt_alpha_init": 0.1,
        "panovggt_alpha_max": 0.2,
        "panovggt_force_fp32": False,
        "panovggt_output_dim": int(getattr(vision_config, "out_hidden_size", text_hidden_size)),
    }
    for field_name, default_value in defaults.items():
        if not hasattr(config, field_name):
            setattr(config, field_name, default_value)


def _ensure_vendored_panovggt_available() -> None:
    if not VENDORED_PANOVGGT_DIR.exists():
        raise FileNotFoundError(
            "PanoVGGT is enabled but the vendored source directory is missing: "
            f"{VENDORED_PANOVGGT_DIR}"
        )
    if not VENDORED_PANOVGGT_CONFIG_PATH.exists():
        raise FileNotFoundError(
            "PanoVGGT is enabled but the vendored config is missing: "
            f"{VENDORED_PANOVGGT_CONFIG_PATH}"
        )

    src_root = str(SRC_ROOT)
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


def build_panovggt_model_from_vendored_config():
    _ensure_vendored_panovggt_available()

    try:
        from omegaconf import OmegaConf
        from panovggt.models import aggregator as panovggt_aggregator
        from panovggt.models.panovggt_model import PanoVGGTModel
    except Exception as exc:
        raise ImportError(
            "PanoVGGT is enabled but could not be imported from "
            f"{VENDORED_PANOVGGT_DIR}"
        ) from exc

    cfg = OmegaConf.load(str(VENDORED_PANOVGGT_CONFIG_PATH))
    OmegaConf.resolve(cfg)
    mc = cfg.model
    original_dinov2_loader = getattr(panovggt_aggregator.Aggregator, "_try_load_dinov2", None)
    if original_dinov2_loader is not None:
        panovggt_aggregator.Aggregator._try_load_dinov2 = lambda *args, **kwargs: None
    try:
        model = PanoVGGTModel(
            img_size=cfg.img_size,
            patch_size=cfg.patch_size,
            embed_dim=cfg.embed_dim,
            enable_camera=False,
            enable_depth=False,
            enable_point=False,
            enable_global_points=False,
            aggregator=OmegaConf.to_container(mc.aggregator, resolve=True),
        )
    finally:
        if original_dinov2_loader is not None:
            panovggt_aggregator.Aggregator._try_load_dinov2 = original_dinov2_loader
    freeze_panovggt_model(model)
    return model


def freeze_panovggt_model(model: nn.Module | None) -> None:
    if model is None:
        return
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


def load_panovggt_checkpoint(model: nn.Module, checkpoint_path: str) -> None:
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    for key in ("model_state_dict", "model", "state_dict"):
        if isinstance(ckpt, dict) and key in ckpt:
            ckpt = ckpt[key]
            break
    if not isinstance(ckpt, dict):
        raise RuntimeError(f"Unexpected PanoVGGT checkpoint format: {type(ckpt)!r}")
    state_dict = {
        (key[7:] if key.startswith("module.") else key): value
        for key, value in ckpt.items()
    }
    load_result = model.load_state_dict(state_dict, strict=False)
    missing_aggregator_keys = [
        key for key in getattr(load_result, "missing_keys", []) if key.startswith("aggregator.")
    ]
    if missing_aggregator_keys:
        raise RuntimeError(
            "PanoVGGT checkpoint is missing aggregator weights, "
            f"first missing key: {missing_aggregator_keys[0]}"
        )
    freeze_panovggt_model(model)

class PanoVGGTGeometryMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        ensure_panovggt_config(config)

        self.enabled = bool(getattr(config, "panovggt_enabled", False))
        self.layer = PANOVGGT_AGGREGATOR_LAYER
        self.context_dim = PANOVGGT_CONTEXT_DIM
        self.output_dim = int(getattr(config, "panovggt_output_dim", config.text_config.hidden_size))
        self.hidden_dim = PANOVGGT_MLP_HIDDEN_SIZE
        self.alpha_init = float(getattr(config, "panovggt_alpha_init", 0.1))
        self.alpha_max = float(getattr(config, "panovggt_alpha_max", 0.2))
        self.spatial_merge_size = int(getattr(config.vision_config, "spatial_merge_size", 2))

        self.input_norm = nn.RMSNorm(self.context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.context_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.output_norm = nn.RMSNorm(self.output_dim, eps=1e-6)
        self.raw_alpha = nn.Parameter(_bounded_raw_alpha(self.alpha_init, self.alpha_max))
        self.reset_parameters()

    @property
    def alpha(self) -> torch.Tensor:
        return float(self.alpha_max) * torch.sigmoid(self.raw_alpha)

    def reset_parameters(self) -> None:
        self.input_norm.reset_parameters()
        self.output_norm.reset_parameters()
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        with torch.no_grad():
            self.raw_alpha.copy_(_bounded_raw_alpha(self.alpha_init, self.alpha_max))

    def _infer_patch_grid(
        self,
        *,
        patch_count: int,
        image_height: int,
        image_width: int,
        patch_size: int,
    ) -> tuple[int, int]:
        if patch_size > 0:
            patch_h = image_height // patch_size
            patch_w = image_width // patch_size
            if patch_h > 0 and patch_w > 0 and patch_h * patch_w == patch_count:
                return patch_h, patch_w

        aspect = max(float(image_width) / max(float(image_height), 1.0), 1e-6)
        best_pair = None
        best_error = float("inf")
        for patch_h in range(1, int(math.sqrt(patch_count)) + 2):
            if patch_count % patch_h != 0:
                continue
            patch_w = patch_count // patch_h
            error = abs((float(patch_w) / float(patch_h)) - aspect)
            if error < best_error:
                best_error = error
                best_pair = (patch_h, patch_w)
        if best_pair is None:
            raise AssertionError(f"Cannot infer PanoVGGT patch grid for {patch_count} tokens")
        return best_pair

    def _sample_geometry_grid(
        self,
        source_grid: torch.Tensor,
        target_h: int,
        target_w: int,
        geometry: torch.Tensor | None,
    ) -> torch.Tensor:
        if target_h <= 0 or target_w <= 0:
            return source_grid.new_zeros((0, source_grid.shape[1]))

        if geometry is None:
            vertical_fov = math.pi
            center_latitude = 0.0
        else:
            vertical_fov = float(geometry[0].detach().cpu())
            center_latitude = float(geometry[1].detach().cpu())

        ys = torch.arange(target_h, device=source_grid.device, dtype=torch.float32) + 0.5
        xs = torch.arange(target_w, device=source_grid.device, dtype=torch.float32) + 0.5
        lat = (ys[:, None] / float(target_h) - 0.5) * vertical_fov + center_latitude
        lon = (xs[None, :] / float(target_w) - 0.5) * (2.0 * math.pi)
        lat = lat.expand(target_h, target_w)
        lon = lon.expand(target_h, target_w)
        grid = torch.stack(
            [
                (lon / math.pi).clamp(-1.0 + 1e-6, 1.0 - 1e-6),
                (lat / (0.5 * math.pi)).clamp(-1.0 + 1e-6, 1.0 - 1e-6),
            ],
            dim=-1,
        ).unsqueeze(0)
        sampled = F.grid_sample(
            source_grid.float(),
            grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        return sampled[0].permute(1, 2, 0).reshape(target_h * target_w, source_grid.shape[1])

    def forward(
        self,
        panovggt_pixel_values: torch.Tensor,
        panovggt_model: nn.Module,
        target_grid_thw: torch.Tensor,
        target_lengths: list[int],
        image_erp_geometry: torch.Tensor | None,
        output_device: torch.device,
        output_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        if (
            not self.enabled
            or panovggt_model is None
            or panovggt_pixel_values is None
            or panovggt_pixel_values.numel() == 0
        ):
            return []

        if panovggt_pixel_values.ndim == 4:
            panovggt_pixel_values = panovggt_pixel_values.unsqueeze(1)
        if panovggt_pixel_values.ndim != 5:
            raise AssertionError(
                "Expected panovggt_pixel_values shape [B, S, 3, H, W] "
                f"or [B, 3, H, W], got {tuple(panovggt_pixel_values.shape)}"
            )

        encoder = panovggt_model
        encoder_param = next(encoder.parameters())
        param = next(self.mlp.parameters())
        images = panovggt_pixel_values.to(device=encoder_param.device, dtype=encoder_param.dtype)

        with torch.no_grad():
            aggregated = encoder.aggregator(images)
        if isinstance(aggregated, (list, tuple)):
            token_list = aggregated[0]
            patch_start_idx = int(aggregated[1])
            tokens = token_list[self.layer] if isinstance(token_list, list) else token_list
        else:
            tokens = aggregated
            patch_start_idx = 0

        if tokens.ndim != 4:
            raise AssertionError(f"Expected PanoVGGT tokens [B, S, P, C], got {tuple(tokens.shape)}")
        tokens = tokens[:, -1, patch_start_idx:, :]
        if tokens.shape[-1] != self.context_dim:
            raise AssertionError(
                f"PanoVGGT context dim mismatch: expected {self.context_dim}, got {tokens.shape[-1]}"
            )

        _, _, _, image_h, image_w = images.shape
        patch_size = int(getattr(encoder, "patch_size", 0))
        patch_h, patch_w = self._infer_patch_grid(
            patch_count=int(tokens.shape[1]),
            image_height=int(image_h),
            image_width=int(image_w),
            patch_size=patch_size,
        )
        source_grid = tokens.transpose(1, 2).reshape(tokens.shape[0], tokens.shape[-1], patch_h, patch_w)

        deltas = []
        geometry = image_erp_geometry
        if geometry is not None:
            geometry = geometry.to(device=param.device, dtype=torch.float32)
        for sample_index, (grid_thw, target_len) in enumerate(zip(target_grid_thw.tolist(), target_lengths)):
            num_frames, grid_h, grid_w = [int(value) for value in grid_thw]
            if num_frames != 1:
                raise AssertionError(
                    "PanoVGGT geometry fusion expects a single current panorama per Qwen image, "
                    f"got image_grid_thw={grid_thw}"
                )
            target_h = max(1, grid_h // self.spatial_merge_size)
            target_w = max(1, grid_w // self.spatial_merge_size)
            expected_len = target_h * target_w
            if expected_len != int(target_len):
                raise AssertionError(
                    "Cannot map PanoVGGT geometry to the Qwen 2D visual-token grid: "
                    f"image_grid_thw={grid_thw}, spatial_merge_size={self.spatial_merge_size}, "
                    f"expected_len={expected_len}, qwen_len={int(target_len)}"
                )

            geo = self._sample_geometry_grid(
                source_grid[sample_index:sample_index + 1],
                target_h=target_h,
                target_w=target_w,
                geometry=None if geometry is None else geometry[sample_index],
            )
            if geo.shape[0] != int(target_len):
                raise AssertionError(
                    "PanoVGGT sampled geometry length mismatch: "
                    f"sampled_len={geo.shape[0]}, qwen_len={int(target_len)}"
                )
            projected = self.output_norm(self.mlp(self.input_norm(geo.to(dtype=param.dtype))))
            projected = self.alpha.to(dtype=projected.dtype) * projected
            deltas.append(projected.to(device=output_device, dtype=output_dtype))

        return deltas


def ensure_action_calibrator_config(config) -> None:
    text_config = getattr(config, "text_config", None)
    hidden_size = getattr(text_config, "hidden_size", getattr(config.vision_config, "out_hidden_size", 3584))
    defaults = {
        "action_calibrator_enabled": False,
        "action_calibrator_hidden_size": 64,
        "action_calibrator_max_delta": 0.35,
        "action_calibrator_delta_scale": 1.0,
        "action_calibrator_l2_weight": 0.0,
        "action_calibrator_turn_angle_deg": 15.0,
        "action_calibrator_inference_enabled": True,
        "action_calibrator_attention_layer_indices": [19, 23, 27, 31],
        "action_calibrator_attention_layers": None,
        "action_calibrator_step_decay": [1.0, 0.75, 0.55, 0.40],
        "action_calibrator_model_hidden_size": int(hidden_size),
    }
    for field_name, default_value in defaults.items():
        if not hasattr(config, field_name):
            setattr(config, field_name, default_value)
    if getattr(config, "action_calibrator_attention_layer_indices", None) is None:
        legacy_layers = getattr(config, "action_calibrator_attention_layers", None)
        if isinstance(legacy_layers, (list, tuple)):
            setattr(config, "action_calibrator_attention_layer_indices", [int(index) for index in legacy_layers])
        elif legacy_layers is None:
            setattr(config, "action_calibrator_attention_layer_indices", [19, 23, 27, 31])


class ActionCalibrator(nn.Module):
    FEATURE_SIZE = 4

    def __init__(self, config):
        super().__init__()
        ensure_action_calibrator_config(config)
        self.enabled = bool(getattr(config, "action_calibrator_enabled", False))
        self.inner_size = int(getattr(config, "action_calibrator_hidden_size", 64))
        self.max_delta = float(getattr(config, "action_calibrator_max_delta", 0.35))
        self.delta_scale = float(getattr(config, "action_calibrator_delta_scale", 1.0))
        self.turn_angle_deg = float(getattr(config, "action_calibrator_turn_angle_deg", 15.0))
        self.attention_layer_indices = self._normalize_attention_layer_indices(
            getattr(config, "action_calibrator_attention_layer_indices", None)
        )
        self.attention_layers = getattr(config, "action_calibrator_attention_layers", None)

        step_decay = list(getattr(config, "action_calibrator_step_decay", [1.0, 0.75, 0.55, 0.40]))
        if not step_decay:
            step_decay = [1.0]
        while len(step_decay) < ACTION_CALIBRATOR_MAX_STEPS:
            step_decay.append(float(step_decay[-1]))
        self.step_decay = tuple(float(value) for value in step_decay[:ACTION_CALIBRATOR_MAX_STEPS])

        self.mlp = nn.Sequential(
            nn.LayerNorm(self.FEATURE_SIZE),
            nn.Linear(self.FEATURE_SIZE, self.inner_size),
            nn.SiLU(),
            nn.Linear(self.inner_size, len(ACTION_CALIBRATOR_MOVEMENT_ACTION_INDICES)),
        )
        self.reset_parameters()

    @staticmethod
    def _normalize_attention_layer_indices(value) -> tuple[int, ...]:
        if value is None:
            return ()
        if isinstance(value, int):
            return (int(value),)
        return tuple(int(index) for index in value)

    def reset_parameters(self) -> None:
        self.mlp[0].reset_parameters()
        self.mlp[1].reset_parameters()
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(
        self,
        attention_to_image: torch.Tensor,
        image_yaw: torch.Tensor,
        prefix_yaw_deg: torch.Tensor,
        decode_step: torch.Tensor,
    ) -> torch.Tensor:
        if (
            not self.enabled
            or attention_to_image.numel() == 0
            or image_yaw.numel() == 0
            or self.max_delta <= 0.0
            or self.delta_scale == 0.0
        ):
            return attention_to_image.new_zeros(
                (int(attention_to_image.shape[0]), len(ACTION_CALIBRATOR_MOVEMENT_ACTION_INDICES))
            )

        attn = attention_to_image.detach().to(torch.float32).clamp_min(0.0)
        eps = torch.finfo(attn.dtype).eps
        image_mass = attn.sum(dim=-1).clamp_min(eps)
        norm_attn = attn / image_mass.unsqueeze(-1)

        prefix = torch.deg2rad(prefix_yaw_deg.to(device=attn.device, dtype=torch.float32))
        yaw = image_yaw.to(device=attn.device, dtype=torch.float32)
        rel_yaw = torch.atan2(
            torch.sin(yaw.unsqueeze(0) - prefix.unsqueeze(-1)),
            torch.cos(yaw.unsqueeze(0) - prefix.unsqueeze(-1)),
        )

        sin_yaw = torch.sin(rel_yaw)
        cos_yaw = torch.cos(rel_yaw)
        front_kernel = cos_yaw.clamp_min(0.0)
        right_kernel = sin_yaw.clamp_min(0.0)
        left_kernel = (-sin_yaw).clamp_min(0.0)
        back_kernel = (-cos_yaw).clamp_min(0.0)

        e_front = (norm_attn * front_kernel).sum(dim=-1)
        e_left = (norm_attn * left_kernel).sum(dim=-1)
        e_right = (norm_attn * right_kernel).sum(dim=-1)
        e_back = (norm_attn * back_kernel).sum(dim=-1)
        mean_sin = (norm_attn * sin_yaw).sum(dim=-1)
        mean_cos = (norm_attn * cos_yaw).sum(dim=-1)
        concentration = torch.sqrt(mean_sin.square() + mean_cos.square()).clamp(max=1.0)
        confidence = image_mass.clamp(max=1.0).sqrt() * concentration
        step = decode_step.to(device=attn.device, dtype=torch.long).clamp(
            min=0,
            max=ACTION_CALIBRATOR_MAX_STEPS - 1,
        )

        features = torch.stack(
            [
                e_front,
                e_left,
                e_right,
                e_back,
            ],
            dim=-1,
        )
        param_dtype = next(self.mlp.parameters()).dtype
        delta = self.mlp(features.to(dtype=param_dtype)).to(torch.float32)
        delta = delta - delta.mean(dim=-1, keepdim=True)
        delta = self.max_delta * torch.tanh(self.delta_scale * delta)
        step_decay = attn.new_tensor(self.step_decay, dtype=torch.float32)[step].unsqueeze(-1)
        delta = delta * confidence.unsqueeze(-1) * step_decay
        return delta.to(dtype=attention_to_image.dtype)


class Qwen3_5ForConditionalGenerationForPanoVLN(Qwen3_5ForConditionalGeneration):
    _keys_to_ignore_on_load_unexpected = list(
        getattr(Qwen3_5ForConditionalGeneration, "_keys_to_ignore_on_load_unexpected", []) or []
    ) + [r"panovggt\..*"]

    PANOVGGT_STATE_KEYS = (
        "panovggt_mlp.raw_alpha",
        "panovggt_mlp.input_norm.weight",
        "panovggt_mlp.mlp.0.weight",
        "panovggt_mlp.mlp.0.bias",
        "panovggt_mlp.mlp.2.weight",
        "panovggt_mlp.mlp.2.bias",
        "panovggt_mlp.output_norm.weight",
    )
    ACTION_CALIBRATOR_STATE_KEYS = (
        "action_calibrator.mlp.0.weight",
        "action_calibrator.mlp.0.bias",
        "action_calibrator.mlp.1.weight",
        "action_calibrator.mlp.1.bias",
        "action_calibrator.mlp.3.weight",
        "action_calibrator.mlp.3.bias",
    )

    def __init__(self, config):
        panovggt_enabled = bool(getattr(config, "panovggt_enabled", False))
        action_calibrator_enabled = bool(getattr(config, "action_calibrator_enabled", False))
        if panovggt_enabled:
            ensure_panovggt_config(config)
        if action_calibrator_enabled:
            ensure_action_calibrator_config(config)
        super().__init__(config)

        if panovggt_enabled:
            ensure_panovggt_config(config)
        if action_calibrator_enabled:
            ensure_action_calibrator_config(config)

        self.panovggt_mlp = PanoVGGTGeometryMLP(config) if panovggt_enabled else None
        self.action_calibrator = ActionCalibrator(config) if action_calibrator_enabled else None
        self.panovggt = None
        self._panovggt_weights_ready = False
        self._panovggt_dtype = None
        self._panovggt_device = None
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_geometry = None
        self._pano_runtime_panovggt_pixel_values = None
        self._action_calibrator_attention_plan = None
        self._action_calibrator_attention_slices = None
        self._action_calibrator_capture_layer_indices = set()
        self._install_image_feature_hook()
        self._install_action_calibrator_attention_hook()
        self.register_load_state_dict_post_hook(self._load_missing_pano_parameters_post_hook)

    def _panovggt_enabled(self) -> bool:
        return self.panovggt_mlp is not None and bool(getattr(self.panovggt_mlp, "enabled", False))

    def _action_calibrator_enabled(self) -> bool:
        return self.action_calibrator is not None and bool(getattr(self.action_calibrator, "enabled", False))

    def _action_calibrator_current_image_context(
        self,
        input_ids: torch.Tensor,
        image_grid_thw: torch.Tensor,
        batch_index: int,
    ):
        input_device = input_ids.device
        image_token_id = int(self.config.image_token_id)
        spatial_merge = int(self.model.visual.spatial_merge_size)
        split_sizes = (image_grid_thw.prod(-1) // (spatial_merge**2)).to(
            device=input_device,
            dtype=torch.long,
        )
        num_images = int(image_grid_thw.shape[0])
        image_num_images = self._pano_runtime_image_num_images
        if image_num_images is None:
            if int(input_ids.shape[0]) == 1:
                image_num_images = torch.tensor([num_images], device=input_device, dtype=torch.long)
            elif num_images == int(input_ids.shape[0]):
                image_num_images = torch.ones(int(input_ids.shape[0]), device=input_device, dtype=torch.long)
            else:
                return None
        else:
            image_num_images = image_num_images.to(device=input_device, dtype=torch.long)
        if int(image_num_images.sum().item()) != num_images:
            return None
        image_current_index = self._pano_runtime_image_current_index
        if image_current_index is None:
            image_current_index = image_num_images - 1
        else:
            image_current_index = image_current_index.to(device=input_device, dtype=torch.long)

        image_count = int(image_num_images[batch_index].item())
        if image_count <= 0:
            return None
        current_index = int(image_current_index[batch_index].item())
        if current_index < 0 or current_index >= image_count:
            return None
        image_offset = int((image_num_images.cumsum(0) - image_num_images)[batch_index].item())
        sample_lengths = split_sizes[image_offset:image_offset + image_count]
        image_positions = torch.nonzero(input_ids[batch_index] == image_token_id, as_tuple=False).flatten()
        if int(image_positions.numel()) != int(sample_lengths.sum().item()):
            return None
        current_start = int(sample_lengths[:current_index].sum().item())
        current_len = int(sample_lengths[current_index].item())
        current_positions = image_positions[current_start:current_start + current_len]
        if int(current_positions.numel()) != current_len or current_len <= 0:
            return None

        flat_image_index = image_offset + current_index
        _, grid_h, grid_w = [
            int(v) for v in image_grid_thw[flat_image_index].detach().to(device="cpu", dtype=torch.long).tolist()
        ]
        target_h = max(1, grid_h // spatial_merge)
        target_w = max(1, grid_w // spatial_merge)
        if target_h * target_w != current_len:
            return None
        yaw = (
            (torch.arange(target_w, device=input_device, dtype=torch.float32) + 0.5)
            / float(target_w)
            * (2.0 * math.pi)
            - math.pi
        )
        yaw = yaw.unsqueeze(0).expand(target_h, target_w).reshape(-1)
        return current_positions, yaw

    @staticmethod
    def _select_rope_positions(rope: torch.Tensor, batch_index: int, positions: torch.Tensor) -> torch.Tensor:
        positions = positions.to(device=rope.device, dtype=torch.long)
        if rope.dim() == 3:
            return rope[batch_index : batch_index + 1].index_select(1, positions)
        if rope.dim() == 2:
            return rope.index_select(0, positions).unsqueeze(0)
        raise ValueError(f"Unsupported rotary embedding shape for PAAC capture: {tuple(rope.shape)}")

    @staticmethod
    def _apply_rope_to_selected(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        cos = cos.unsqueeze(1)
        sin = sin.unsqueeze(1)
        rotary_dim = int(cos.shape[-1])
        x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]
        x_embed = (x_rot * cos) + (rotate_half(x_rot) * sin)
        return torch.cat([x_embed, x_pass], dim=-1)

    def _selected_action_calibrator_capture_layers(self) -> set[int]:
        if not self._action_calibrator_enabled():
            return set()
        if self._action_calibrator_capture_layer_indices:
            return self._action_calibrator_capture_layer_indices
        layers = getattr(getattr(self.model, "language_model", None), "layers", [])
        full_attention_indices = [
            int(index)
            for index, layer in enumerate(layers)
            if getattr(layer, "layer_type", None) == "full_attention" and hasattr(layer, "self_attn")
        ]
        explicit_indices = tuple(getattr(self.action_calibrator, "attention_layer_indices", ()) or ())
        if explicit_indices:
            full_attention_set = set(full_attention_indices)
            missing_indices = [index for index in explicit_indices if index not in full_attention_set]
            if missing_indices:
                raise ValueError(
                    "ActionCalibrator capture layers must be full-attention layer indices; "
                    f"got invalid indices {missing_indices}, available full-attention layers are {full_attention_indices}"
                )
            self._action_calibrator_capture_layer_indices = set(explicit_indices)
            return self._action_calibrator_capture_layer_indices

        legacy_layers = getattr(self.action_calibrator, "attention_layers", None)
        num_layers = max(1, int(legacy_layers if legacy_layers is not None else 4))
        self._action_calibrator_capture_layer_indices = set(full_attention_indices[-num_layers:])
        return self._action_calibrator_capture_layer_indices

    def _prepare_action_calibrator_attention_capture(
        self,
        input_ids: torch.Tensor | None,
        labels: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        logits_to_keep: int | torch.Tensor,
    ) -> None:
        self._action_calibrator_attention_plan = None
        self._action_calibrator_attention_slices = None
        if (
            not self._action_calibrator_enabled()
            or input_ids is None
            or image_grid_thw is None
            or image_grid_thw.numel() == 0
            or isinstance(logits_to_keep, torch.Tensor)
        ):
            return

        plan = {}
        if labels is not None:
            shift_labels = labels[:, 1:]
            for batch_index in range(int(labels.shape[0])):
                query_positions = []
                for shift_pos in torch.nonzero(shift_labels[batch_index] != -100, as_tuple=False).flatten().tolist():
                    action_id = self._action_calibrator_label_action(int(shift_labels[batch_index, shift_pos].item()))
                    if action_id is not None:
                        query_positions.append(int(shift_pos))
                if not query_positions:
                    continue
                context = self._action_calibrator_current_image_context(input_ids, image_grid_thw, batch_index)
                if context is None:
                    continue
                image_positions, _ = context
                plan[batch_index] = {
                    "query_positions": torch.tensor(query_positions, device=input_ids.device, dtype=torch.long),
                    "image_positions": image_positions.to(device=input_ids.device, dtype=torch.long),
                }
        else:
            seq_len = int(input_ids.shape[1])
            for batch_index in range(int(input_ids.shape[0])):
                if attention_mask is None:
                    query_pos = seq_len - 1
                else:
                    query_pos = int(attention_mask[batch_index].sum().item()) - 1
                if query_pos < 0 or query_pos >= seq_len:
                    continue
                prefix_actions = self._action_calibrator_prefix_actions(input_ids, batch_index)
                if 0 in prefix_actions:
                    continue
                context = self._action_calibrator_current_image_context(input_ids, image_grid_thw, batch_index)
                if context is None:
                    continue
                image_positions, _ = context
                plan[batch_index] = {
                    "query_positions": torch.tensor([query_pos], device=input_ids.device, dtype=torch.long),
                    "image_positions": image_positions.to(device=input_ids.device, dtype=torch.long),
                }

        if plan:
            self._action_calibrator_attention_plan = plan
            self._action_calibrator_attention_slices = {}

    def _clear_action_calibrator_attention_capture(self) -> None:
        self._action_calibrator_attention_plan = None
        self._action_calibrator_attention_slices = None

    def _capture_action_calibrator_attention_slice(
        self,
        attention_module: nn.Module,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: torch.Tensor | None,
    ) -> None:
        plan = self._action_calibrator_attention_plan
        if not plan or int(getattr(attention_module, "layer_idx", -1)) not in self._selected_action_calibrator_capture_layers():
            return

        with torch.no_grad():
            batch_size, seq_len, _ = hidden_states.shape
            cos, sin = position_embeddings
            layer_store = self._action_calibrator_attention_slices
            if layer_store is None:
                return
            key_index = torch.arange(seq_len, device=hidden_states.device, dtype=torch.long)
            hidden_detached = hidden_states.detach()

            for batch_index, item in plan.items():
                if batch_index >= batch_size:
                    continue
                query_positions = item["query_positions"].to(device=hidden_states.device, dtype=torch.long)
                image_positions = item["image_positions"].to(device=hidden_states.device, dtype=torch.long)
                if int(query_positions.numel()) == 0 or int(image_positions.numel()) == 0:
                    continue
                if int(query_positions.max().item()) >= seq_len or int(image_positions.max().item()) >= seq_len:
                    continue

                q_hidden = hidden_detached[batch_index : batch_index + 1].index_select(1, query_positions)
                q_proj = attention_module.q_proj(q_hidden).view(
                    1,
                    int(query_positions.numel()),
                    -1,
                    attention_module.head_dim * 2,
                )
                query_states, _ = torch.chunk(q_proj, 2, dim=-1)
                query_states = attention_module.q_norm(query_states).transpose(1, 2)

                k_hidden = hidden_detached[batch_index : batch_index + 1]
                key_states = attention_module.k_proj(k_hidden).view(1, seq_len, -1, attention_module.head_dim)
                key_states = attention_module.k_norm(key_states).transpose(1, 2)

                cos_q = self._select_rope_positions(cos, batch_index, query_positions)
                sin_q = self._select_rope_positions(sin, batch_index, query_positions)
                cos_k = self._select_rope_positions(cos, batch_index, key_index)
                sin_k = self._select_rope_positions(sin, batch_index, key_index)
                query_states = self._apply_rope_to_selected(query_states, cos_q, sin_q)
                key_states = self._apply_rope_to_selected(key_states, cos_k, sin_k)
                key_states = repeat_kv(key_states, attention_module.num_key_value_groups)

                scores = torch.matmul(
                    query_states.to(torch.float32),
                    key_states.to(torch.float32).transpose(-1, -2),
                ) * float(attention_module.scaling)

                causal_mask = key_index.unsqueeze(0) > query_positions.unsqueeze(1)
                if attention_mask is not None and attention_mask.dim() == 4:
                    mask = attention_mask[batch_index : batch_index + 1].index_select(2, query_positions)
                    mask = mask[..., :seq_len]
                    scores = scores + mask.to(device=scores.device, dtype=scores.dtype)
                elif attention_mask is not None and attention_mask.dim() == 2:
                    padding_mask = attention_mask[batch_index, :seq_len].to(device=scores.device) == 0
                    combined_mask = causal_mask | padding_mask.unsqueeze(0)
                    scores = scores.masked_fill(combined_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)
                else:
                    scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), torch.finfo(scores.dtype).min)

                attn = torch.softmax(scores, dim=-1)
                image_attn = attn.index_select(-1, image_positions).mean(dim=1).squeeze(0)
                layer_store.setdefault(batch_index, []).append(image_attn.detach().to(device=hidden_states.device))

    def _action_calibrator_attention_to_image(
        self,
        attentions,
        batch_index: int,
        query_positions: torch.Tensor,
        image_positions: torch.Tensor,
        output_device: torch.device,
    ) -> torch.Tensor | None:
        del attentions, query_positions, image_positions
        layer_store = self._action_calibrator_attention_slices or {}
        chunks = layer_store.get(batch_index, [])

        if not chunks:
            return None
        return torch.stack([chunk.to(device=output_device, dtype=torch.float32) for chunk in chunks], dim=0).mean(dim=0)

    @staticmethod
    def _action_calibrator_label_action(token_id: int):
        return ACTION_CALIBRATOR_TOKEN_TO_ACTION.get(int(token_id))

    @staticmethod
    def _action_calibrator_prefix_actions(input_ids: torch.Tensor, batch_index: int) -> list[int]:
        prefix_actions = []
        for token_id in reversed(input_ids[batch_index].detach().to(device="cpu", dtype=torch.long).tolist()):
            action_id = ACTION_CALIBRATOR_TOKEN_TO_ACTION.get(int(token_id))
            if action_id is None:
                break
            prefix_actions.append(int(action_id))
            if len(prefix_actions) >= ACTION_CALIBRATOR_MAX_STEPS - 1:
                break
        prefix_actions.reverse()
        return prefix_actions

    @staticmethod
    def _action_calibrator_prefix_yaw(action_ids: Sequence[int], turn_angle: float) -> float:
        prefix_yaw = 0.0
        for action_id in action_ids:
            if action_id == 2:
                prefix_yaw -= turn_angle
            elif action_id == 3:
                prefix_yaw += turn_angle
            elif action_id == 0:
                break
        return prefix_yaw

    def _apply_action_calibrator(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor | None,
        labels: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        attentions,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if (
            not self._action_calibrator_enabled()
            or labels is None
            or input_ids is None
            or image_grid_thw is None
            or image_grid_thw.numel() == 0
        ):
            return logits, None
        calibrator = self.action_calibrator
        calibrated_logits = logits.clone()
        delta_chunks = []
        action_token_sets = (
            torch.tensor(ACTION_CALIBRATOR_FIRST_TOKEN_IDS, device=logits.device, dtype=torch.long),
            torch.tensor(ACTION_CALIBRATOR_NEXT_TOKEN_IDS, device=logits.device, dtype=torch.long),
        )
        movement_indices = torch.tensor(
            ACTION_CALIBRATOR_MOVEMENT_ACTION_INDICES,
            device=logits.device,
            dtype=torch.long,
        )

        shift_labels = labels[:, 1:]
        for batch_index in range(int(labels.shape[0])):
            action_label_positions = []
            action_ids = []
            for shift_pos in torch.nonzero(shift_labels[batch_index] != -100, as_tuple=False).flatten().tolist():
                action_id = self._action_calibrator_label_action(int(shift_labels[batch_index, shift_pos].item()))
                if action_id is None:
                    continue
                action_label_positions.append(int(shift_pos))
                action_ids.append(int(action_id))
            if not action_label_positions:
                continue

            context = self._action_calibrator_current_image_context(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                batch_index=batch_index,
            )
            if context is None:
                continue
            image_positions, image_yaw = context

            prefix_yaws = []
            prefix_yaw = 0.0
            active_mask = []
            stopped = False
            turn_angle = float(getattr(calibrator, "turn_angle_deg", 15.0))
            for action_id in action_ids:
                prefix_yaws.append(prefix_yaw)
                active_mask.append(0.0 if stopped else 1.0)
                if stopped:
                    continue
                if action_id == 0:
                    stopped = True
                elif action_id == 2:
                    prefix_yaw -= turn_angle
                elif action_id == 3:
                    prefix_yaw += turn_angle

            action_positions = torch.tensor(action_label_positions, device=logits.device, dtype=torch.long)
            attention_to_image = self._action_calibrator_attention_to_image(
                attentions=attentions,
                batch_index=batch_index,
                query_positions=action_positions,
                image_positions=image_positions,
                output_device=logits.device,
            )
            if attention_to_image is None:
                continue

            delta = calibrator(
                attention_to_image=attention_to_image,
                image_yaw=image_yaw,
                prefix_yaw_deg=torch.tensor(prefix_yaws, device=logits.device, dtype=torch.float32),
                decode_step=torch.arange(len(action_label_positions), device=logits.device, dtype=torch.float32),
            )
            delta = delta * torch.tensor(active_mask, device=logits.device, dtype=torch.float32).unsqueeze(-1)
            delta_chunks.append(delta.float())
            for action_order, shift_pos in enumerate(action_label_positions):
                token_ids = action_token_sets[0 if action_order == 0 else 1]
                movement_token_ids = token_ids.index_select(0, movement_indices)
                calibrated_logits[batch_index, shift_pos, movement_token_ids] = (
                    calibrated_logits[batch_index, shift_pos, movement_token_ids]
                    + delta[action_order].to(calibrated_logits.dtype)
                )

        if not delta_chunks:
            dummy = sum(param.sum() for param in calibrator.parameters()) * 0.0
            return calibrated_logits, dummy
        deltas = torch.cat(delta_chunks, dim=0)
        return calibrated_logits, deltas.pow(2).mean()

    def _apply_action_calibrator_inference(
        self,
        logits: torch.Tensor,
        input_ids: torch.Tensor | None,
        attention_mask: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
        logits_to_keep: int | torch.Tensor,
        attentions,
    ) -> torch.Tensor:
        if (
            not self._action_calibrator_enabled()
            or not bool(getattr(self.config, "action_calibrator_inference_enabled", True))
            or input_ids is None
            or image_grid_thw is None
            or image_grid_thw.numel() == 0
            or isinstance(logits_to_keep, torch.Tensor)
        ):
            return logits

        calibrator = self.action_calibrator
        calibrated_logits = logits.clone()
        action_token_sets = (
            torch.tensor(ACTION_CALIBRATOR_FIRST_TOKEN_IDS, device=logits.device, dtype=torch.long),
            torch.tensor(ACTION_CALIBRATOR_NEXT_TOKEN_IDS, device=logits.device, dtype=torch.long),
        )
        movement_indices = torch.tensor(
            ACTION_CALIBRATOR_MOVEMENT_ACTION_INDICES,
            device=logits.device,
            dtype=torch.long,
        )
        seq_len = int(input_ids.shape[1])
        kept = int(logits_to_keep) if isinstance(logits_to_keep, int) else 0

        for batch_index in range(int(input_ids.shape[0])):
            if attention_mask is None:
                query_pos = seq_len - 1
            else:
                query_pos = int(attention_mask[batch_index].sum().item()) - 1
            if query_pos < 0 or query_pos >= seq_len:
                continue

            if kept == 0:
                logit_pos = query_pos
            elif query_pos >= seq_len - kept:
                logit_pos = query_pos - (seq_len - kept)
            else:
                continue
            if logit_pos < 0 or logit_pos >= int(logits.shape[1]):
                continue

            prefix_actions = self._action_calibrator_prefix_actions(input_ids, batch_index)
            if 0 in prefix_actions:
                continue
            decode_step = min(len(prefix_actions), ACTION_CALIBRATOR_MAX_STEPS - 1)
            context = self._action_calibrator_current_image_context(
                input_ids=input_ids,
                image_grid_thw=image_grid_thw,
                batch_index=batch_index,
            )
            if context is None:
                continue
            image_positions, image_yaw = context
            action_position = torch.tensor([query_pos], device=logits.device, dtype=torch.long)
            attention_to_image = self._action_calibrator_attention_to_image(
                attentions=attentions,
                batch_index=batch_index,
                query_positions=action_position,
                image_positions=image_positions,
                output_device=logits.device,
            )
            if attention_to_image is None:
                continue

            turn_angle = float(getattr(calibrator, "turn_angle_deg", 15.0))
            prefix_yaw = self._action_calibrator_prefix_yaw(prefix_actions, turn_angle)
            token_ids = action_token_sets[0 if decode_step == 0 else 1]
            movement_token_ids = token_ids.index_select(0, movement_indices)
            delta = calibrator(
                attention_to_image=attention_to_image,
                image_yaw=image_yaw,
                prefix_yaw_deg=torch.tensor([prefix_yaw], device=logits.device, dtype=torch.float32),
                decode_step=torch.tensor([decode_step], device=logits.device, dtype=torch.float32),
            )[0]
            calibrated_logits[batch_index, logit_pos, movement_token_ids] = (
                calibrated_logits[batch_index, logit_pos, movement_token_ids] + delta.to(calibrated_logits.dtype)
            )

        return calibrated_logits

    def _install_action_calibrator_attention_hook(self) -> None:
        if not self._action_calibrator_enabled():
            return
        layers = getattr(getattr(self.model, "language_model", None), "layers", [])
        owner = self

        for layer in layers:
            attention_module = getattr(layer, "self_attn", None)
            if attention_module is None or hasattr(attention_module, "_paac_origin_forward"):
                continue
            attention_module._paac_origin_forward = attention_module.forward

            def forward_with_paac_capture(
                this,
                hidden_states,
                position_embeddings,
                attention_mask=None,
                past_key_values=None,
                **kwargs,
            ):
                owner._capture_action_calibrator_attention_slice(
                    attention_module=this,
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                )
                return this._paac_origin_forward(
                    hidden_states=hidden_states,
                    position_embeddings=position_embeddings,
                    attention_mask=attention_mask,
                    past_key_values=past_key_values,
                    **kwargs,
                )

            attention_module.forward = MethodType(forward_with_paac_capture, attention_module)

    def _install_image_feature_hook(self) -> None:
        if not self._panovggt_enabled():
            return
        if hasattr(self.model, "_pano_origin_get_image_features"):
            return

        self.model._pano_origin_get_image_features = self.model.get_image_features
        owner = self

        def get_image_features_with_pano_residuals(this, pixel_values, image_grid_thw=None, **kwargs):
            vision_output = this._pano_origin_get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs,
            )
            if image_grid_thw is None or vision_output.pooler_output is None:
                return vision_output

            current_indices = owner._current_image_flat_indices(image_grid_thw)
            if not current_indices:
                return vision_output

            image_embeds = list(vision_output.pooler_output)
            current_items = [
                (batch_index, image_index)
                for batch_index, image_index in current_indices
                if 0 <= image_index < len(image_embeds)
            ]
            if not current_items:
                return vision_output

            panovggt_pixel_values = owner._pano_runtime_panovggt_pixel_values
            if (
                owner._panovggt_enabled()
                and panovggt_pixel_values is not None
                and panovggt_pixel_values.numel() > 0
            ):
                panovggt_items = [
                    (batch_index, image_index)
                    for batch_index, image_index in current_items
                    if 0 <= batch_index < int(panovggt_pixel_values.shape[0])
                ]
                if panovggt_items:
                    batch_indices = torch.tensor(
                        [item[0] for item in panovggt_items],
                        device=panovggt_pixel_values.device,
                        dtype=torch.long,
                    )
                    target_indices = [item[1] for item in panovggt_items]
                    target_grid_thw = image_grid_thw[target_indices].detach().to(device="cpu", dtype=torch.long)
                    target_lengths = [int(image_embeds[index].shape[0]) for index in target_indices]
                    geometry = owner._pano_runtime_image_geometry
                    if geometry is not None:
                        geometry = geometry[target_indices].detach().to(device="cpu", dtype=torch.float32)

                    panovggt_model = owner._ensure_panovggt_model(
                        device=image_embeds[target_indices[0]].device,
                        dtype=image_embeds[target_indices[0]].dtype,
                    )
                    if panovggt_model is None or owner.panovggt_mlp is None:
                        return vision_output
                    deltas = owner.panovggt_mlp(
                        panovggt_pixel_values.index_select(0, batch_indices),
                        panovggt_model=panovggt_model,
                        target_grid_thw=target_grid_thw,
                        target_lengths=target_lengths,
                        image_erp_geometry=geometry,
                        output_device=image_embeds[target_indices[0]].device,
                        output_dtype=image_embeds[target_indices[0]].dtype,
                    )
                    for image_index, delta in zip(target_indices, deltas):
                        if delta.shape != image_embeds[image_index].shape:
                            raise AssertionError(
                                "PanoVGGT delta shape mismatch: "
                                f"delta={tuple(delta.shape)}, qwen={tuple(image_embeds[image_index].shape)}"
                            )
                        image_embeds[image_index] = image_embeds[image_index] + delta

            vision_output.pooler_output = tuple(image_embeds)
            return vision_output

        self.model.get_image_features = MethodType(get_image_features_with_pano_residuals, self.model)

    def _set_runtime_pano_context(
        self,
        *,
        image_grid_thw: torch.Tensor | None,
        image_num_images: torch.Tensor | None,
        image_current_index: torch.Tensor | None,
        image_erp_geometry: torch.Tensor | None,
        panovggt_pixel_values: torch.Tensor | None,
    ) -> None:
        self._pano_runtime_grid_thw = image_grid_thw
        self._pano_runtime_image_num_images = image_num_images
        self._pano_runtime_image_current_index = image_current_index
        self._pano_runtime_image_geometry = image_erp_geometry
        self._pano_runtime_panovggt_pixel_values = panovggt_pixel_values

    def _clear_runtime_pano_context(self) -> None:
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_geometry = None
        self._pano_runtime_panovggt_pixel_values = None

    def _load_external_panovggt_weights(self) -> None:
        if not self._panovggt_enabled():
            return
        if self.panovggt is None:
            self.panovggt = build_panovggt_model_from_vendored_config()
        load_panovggt_checkpoint(
            self.panovggt,
            str(getattr(self.config, "panovggt_checkpoint_path")),
        )
        self._panovggt_weights_ready = True
        self._panovggt_device = None
        self._panovggt_dtype = None

    @staticmethod
    def _load_prefixed_checkpoint_tensors(checkpoint_dir: Path, prefix: str) -> dict[str, torch.Tensor]:
        if not checkpoint_dir.exists() or not checkpoint_dir.is_dir():
            return {}

        state_dict: dict[str, torch.Tensor] = {}

        def add_tensor(key: str, tensor: torch.Tensor) -> None:
            if key.startswith(prefix):
                state_dict[key[len(prefix) :]] = tensor

        safetensors_index = checkpoint_dir / "model.safetensors.index.json"
        if safetensors_index.exists():
            from safetensors import safe_open

            weight_map = json.loads(safetensors_index.read_text()).get("weight_map", {})
            files_to_keys: dict[str, list[str]] = {}
            for key, filename in weight_map.items():
                if key.startswith(prefix):
                    files_to_keys.setdefault(filename, []).append(key)
            for filename, keys in files_to_keys.items():
                with safe_open(str(checkpoint_dir / filename), framework="pt", device="cpu") as handle:
                    for key in keys:
                        add_tensor(key, handle.get_tensor(key))
            return state_dict

        safetensors_path = checkpoint_dir / "model.safetensors"
        if safetensors_path.exists():
            from safetensors import safe_open

            with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
                for key in handle.keys():
                    add_tensor(key, handle.get_tensor(key))
            return state_dict

        def add_from_torch_file(path: Path, keys: set[str] | None = None) -> None:
            loaded = torch.load(str(path), map_location="cpu")
            if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                loaded = loaded["state_dict"]
            if not isinstance(loaded, dict):
                return
            for key, tensor in loaded.items():
                if keys is not None and key not in keys:
                    continue
                add_tensor(key, tensor)

        torch_index = checkpoint_dir / "pytorch_model.bin.index.json"
        if torch_index.exists():
            weight_map = json.loads(torch_index.read_text()).get("weight_map", {})
            files_to_keys: dict[str, set[str]] = {}
            for key, filename in weight_map.items():
                if key.startswith(prefix):
                    files_to_keys.setdefault(filename, set()).add(key)
            for filename, keys in files_to_keys.items():
                add_from_torch_file(checkpoint_dir / filename, keys)
            return state_dict

        torch_path = checkpoint_dir / "pytorch_model.bin"
        if torch_path.exists():
            add_from_torch_file(torch_path)

        return state_dict

    def _load_saved_panovggt_weights(self, pretrained_model_name_or_path) -> bool:
        if not self._panovggt_enabled():
            return False

        checkpoint_dir = Path(str(pretrained_model_name_or_path))
        state_dict = self._load_prefixed_checkpoint_tensors(checkpoint_dir, "panovggt.")
        if not state_dict:
            return False

        if self.panovggt is None:
            self.panovggt = build_panovggt_model_from_vendored_config()

        incompatible = self.panovggt.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            raise RuntimeError(
                "Saved PanoVGGT checkpoint keys do not match the vendored PanoVGGT model: "
                f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
            )
        freeze_panovggt_model(self.panovggt)
        self._panovggt_weights_ready = True
        self._panovggt_device = None
        self._panovggt_dtype = None
        return True

    def _mark_panovggt_weights_ready(self) -> None:
        if self.panovggt is None:
            return
        freeze_panovggt_model(self.panovggt)
        self._panovggt_weights_ready = True
        self._panovggt_device = None
        self._panovggt_dtype = None

    def _ensure_panovggt_model(self, device: torch.device, dtype: torch.dtype):
        if not self._panovggt_enabled():
            return None
        if self.panovggt is None:
            self.panovggt = build_panovggt_model_from_vendored_config()
        if not self._panovggt_weights_ready:
            self._load_external_panovggt_weights()
        if bool(getattr(self.config, "panovggt_force_fp32", False)):
            target_dtype = torch.float32
        else:
            # Keep the frozen encoder in Qwen's low-precision vision dtype when
            # possible. PanoVGGT's attention has a bf16 flash path; forcing fp32
            # makes its 36-layer panorama encoder much slower.
            target_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.float32
        if self._panovggt_device != device or self._panovggt_dtype != target_dtype:
            self.panovggt.to(device=device, dtype=target_dtype)
            self._panovggt_device = device
            self._panovggt_dtype = target_dtype
        freeze_panovggt_model(self.panovggt)
        return self.panovggt

    def _current_image_flat_indices(self, image_grid_thw: torch.Tensor | None) -> list[tuple[int, int]]:
        if image_grid_thw is None or image_grid_thw.numel() == 0:
            return []

        num_images = int(image_grid_thw.shape[0])
        device = image_grid_thw.device
        image_num_images = self._pano_runtime_image_num_images
        if image_num_images is None:
            image_num_images = torch.tensor([num_images], device=device, dtype=torch.long)
        else:
            image_num_images = image_num_images.to(device=device, dtype=torch.long)

        if int(image_num_images.sum().item()) != num_images:
            raise AssertionError(
                "image_num_images does not sum to image_grid_thw rows: "
                f"counts={image_num_images.tolist()}, grids={num_images}"
            )

        image_current_index = self._pano_runtime_image_current_index
        if image_current_index is None:
            image_current_index = image_num_images - 1
        else:
            image_current_index = image_current_index.to(device=device, dtype=torch.long)

        offsets = image_num_images.cumsum(0) - image_num_images
        valid = image_num_images > 0
        if torch.any(valid):
            invalid = (image_current_index < 0) | (image_current_index >= image_num_images)
            if torch.any(invalid & valid):
                raise AssertionError(
                    "image_current_index is out of range for at least one sample: "
                    f"indices={image_current_index.tolist()}, counts={image_num_images.tolist()}"
                )

        flat_indices = offsets + image_current_index
        return [
            (batch_index, int(flat_indices[batch_index].item()))
            for batch_index in range(int(image_num_images.shape[0]))
            if bool(valid[batch_index].item())
        ]

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        pixel_values: torch.Tensor | None = None,
        pixel_values_videos: torch.FloatTensor | None = None,
        image_grid_thw: torch.LongTensor | None = None,
        video_grid_thw: torch.LongTensor | None = None,
        mm_token_type_ids: torch.IntTensor | None = None,
        logits_to_keep: int | torch.Tensor = 0,
        image_erp_geometry: torch.FloatTensor | None = None,
        image_num_images: torch.LongTensor | None = None,
        image_current_index: torch.LongTensor | None = None,
        panovggt_pixel_values: torch.Tensor | None = None,
        **kwargs,
    ):
        if self._panovggt_enabled() or self._action_calibrator_enabled():
            self._set_runtime_pano_context(
                image_grid_thw=image_grid_thw,
                image_num_images=image_num_images,
                image_current_index=image_current_index,
                image_erp_geometry=image_erp_geometry,
                panovggt_pixel_values=panovggt_pixel_values,
            )
        try:
            requested_output_attentions = kwargs.pop("output_attentions", None)
            self._prepare_action_calibrator_attention_capture(
                input_ids=input_ids,
                labels=labels,
                attention_mask=attention_mask,
                image_grid_thw=image_grid_thw,
                logits_to_keep=logits_to_keep,
            )
            outputs = self.model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                mm_token_type_ids=mm_token_type_ids,
                output_attentions=requested_output_attentions,
                **kwargs,
            )

            hidden_states = outputs[0]
            slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
            logits = self.lm_head(hidden_states[:, slice_indices, :])

            calibration_l2 = None
            if labels is not None and (not isinstance(logits_to_keep, int) or logits_to_keep == 0):
                logits, calibration_l2 = self._apply_action_calibrator(
                    logits=logits,
                    input_ids=input_ids,
                    labels=labels,
                    image_grid_thw=image_grid_thw,
                    attentions=outputs.attentions,
                )
            elif labels is None:
                logits = self._apply_action_calibrator_inference(
                    logits=logits,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    image_grid_thw=image_grid_thw,
                    logits_to_keep=logits_to_keep,
                    attentions=outputs.attentions,
                )

            loss = None
            if labels is not None:
                loss = self.loss_function(
                    logits=logits,
                    labels=labels,
                    vocab_size=self.config.text_config.vocab_size,
                )
                if calibration_l2 is not None and self._action_calibrator_enabled():
                    l2_weight = float(getattr(self.config, "action_calibrator_l2_weight", 0.0))
                    if l2_weight > 0.0:
                        loss = loss + l2_weight * calibration_l2

            return Qwen3_5CausalLMOutputWithPast(
                loss=loss,
                logits=logits,
                past_key_values=outputs.past_key_values,
                hidden_states=outputs.hidden_states,
                attentions=outputs.attentions if bool(requested_output_attentions) else None,
                rope_deltas=outputs.rope_deltas,
            )
        finally:
            self._clear_action_calibrator_attention_capture()
            self._clear_runtime_pano_context()

    def _load_missing_pano_parameters_post_hook(self, module, incompatible_keys) -> None:
        del module
        missing_keys = set(incompatible_keys.missing_keys)

        panovggt_module = self.panovggt_mlp
        if (
            panovggt_module is not None
            and panovggt_module.enabled
            and any(key.startswith("panovggt_mlp.") for key in missing_keys)
        ):
            panovggt_module.reset_parameters()
        action_calibrator_module = self.action_calibrator
        if (
            action_calibrator_module is not None
            and action_calibrator_module.enabled
            and any(key.startswith("action_calibrator.") for key in missing_keys)
        ):
            action_calibrator_module.reset_parameters()
        if panovggt_module is not None and panovggt_module.enabled:
            if any(key.startswith("panovggt.") for key in missing_keys):
                self._load_external_panovggt_weights()
            else:
                self._mark_panovggt_weights_ready()

    def _reset_panovggt_parameters_after_pretrained_load(self) -> None:
        if self._panovggt_enabled() and self.panovggt_mlp is not None:
            self.panovggt_mlp.reset_parameters()

    def _reset_action_calibrator_parameters_after_pretrained_load(self) -> None:
        if self._action_calibrator_enabled() and self.action_calibrator is not None:
            self.action_calibrator.reset_parameters()

    @classmethod
    def _checkpoint_has_panovggt_weights(cls, pretrained_model_name_or_path) -> bool | None:
        return cls._checkpoint_has_any_weights(pretrained_model_name_or_path, cls.PANOVGGT_STATE_KEYS)

    @classmethod
    def _checkpoint_has_action_calibrator_weights(cls, pretrained_model_name_or_path) -> bool | None:
        return cls._checkpoint_has_any_weights(pretrained_model_name_or_path, cls.ACTION_CALIBRATOR_STATE_KEYS)

    @classmethod
    def _checkpoint_has_any_weights(cls, pretrained_model_name_or_path, state_keys) -> bool | None:
        checkpoint_dir = Path(str(pretrained_model_name_or_path))
        if not checkpoint_dir.exists():
            return None

        index_candidates = (
            checkpoint_dir / "model.safetensors.index.json",
            checkpoint_dir / "pytorch_model.bin.index.json",
        )
        for index_path in index_candidates:
            if not index_path.exists():
                continue
            try:
                weight_map = json.loads(index_path.read_text()).get("weight_map", {})
            except Exception:
                return None
            return any(key in weight_map for key in state_keys)

        safetensors_path = checkpoint_dir / "model.safetensors"
        if safetensors_path.exists():
            try:
                from safetensors import safe_open

                with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
                    keys = set(handle.keys())
                return any(key in keys for key in state_keys)
            except Exception:
                return None

        return None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        has_panovggt_weights = cls._checkpoint_has_panovggt_weights(pretrained_model_name_or_path)
        has_action_calibrator_weights = cls._checkpoint_has_action_calibrator_weights(pretrained_model_name_or_path)
        if has_panovggt_weights is False:
            model._reset_panovggt_parameters_after_pretrained_load()
        if has_action_calibrator_weights is False:
            model._reset_action_calibrator_parameters_after_pretrained_load()
        if model._panovggt_enabled():
            loaded_saved_panovggt = (
                model._load_saved_panovggt_weights(pretrained_model_name_or_path)
                if has_panovggt_weights is True
                else False
            )
            if not loaded_saved_panovggt and model.panovggt is None:
                model._load_external_panovggt_weights()
        return model

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        is_first_iteration=False,
        panovggt_pixel_values=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            position_ids=position_ids,
            use_cache=use_cache,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            is_first_iteration=is_first_iteration,
            panovggt_pixel_values=panovggt_pixel_values,
            **kwargs,
        )
        if not is_first_iteration and use_cache:
            model_inputs["panovggt_pixel_values"] = None
        return model_inputs
