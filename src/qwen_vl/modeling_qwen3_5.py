import json
import math
import sys
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    ALL_ATTENTION_FUNCTIONS,
    Qwen3_5ForConditionalGeneration,
    apply_rotary_pos_emb,
    eager_attention_forward,
)


PANOVGGT_AGGREGATOR_LAYER = -1
PANOVGGT_CONTEXT_DIM = 2048
PANOVGGT_MLP_HIDDEN_SIZE = 4096
ACTION_BEARING_HIDDEN_SIZE = 2048
ACTION_BEARING_TURN_ANGLE_DEG = 15.0
ACTION_BEARING_MAX_STEPS = 4
ACTION_BEARING_SIGMA_STEPS = 0.6
ACTION_BEARING_LEFT_TOKEN_ID = 2282
ACTION_BEARING_RIGHT_TOKEN_ID = 1246
ACTION_BEARING_FRONT_TOKEN_ID = 6735
ACTION_BEARING_ONE_TOKEN_ID = 799
ACTION_BEARING_TWO_TOKEN_ID = 1330
ACTION_BEARING_THREE_TOKEN_ID = 2250
ACTION_BEARING_FOUR_TOKEN_ID = 2943
ACTION_BEARING_BIN_TOKEN_IDS = (
    (ACTION_BEARING_LEFT_TOKEN_ID, ACTION_BEARING_FOUR_TOKEN_ID),
    (ACTION_BEARING_LEFT_TOKEN_ID, ACTION_BEARING_THREE_TOKEN_ID),
    (ACTION_BEARING_LEFT_TOKEN_ID, ACTION_BEARING_TWO_TOKEN_ID),
    (ACTION_BEARING_LEFT_TOKEN_ID, ACTION_BEARING_ONE_TOKEN_ID),
    (ACTION_BEARING_FRONT_TOKEN_ID,),
    (ACTION_BEARING_RIGHT_TOKEN_ID, ACTION_BEARING_ONE_TOKEN_ID),
    (ACTION_BEARING_RIGHT_TOKEN_ID, ACTION_BEARING_TWO_TOKEN_ID),
    (ACTION_BEARING_RIGHT_TOKEN_ID, ACTION_BEARING_THREE_TOKEN_ID),
    (ACTION_BEARING_RIGHT_TOKEN_ID, ACTION_BEARING_FOUR_TOKEN_ID),
)
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


def _parse_action_bearing_inject_layers(raw_layers) -> tuple[int | None, tuple[int, ...] | None]:
    if raw_layers is None:
        return None, None
    if isinstance(raw_layers, bool):
        raise TypeError(f"action_bearing_inject_layers must be an int or a list of ints, got {raw_layers!r}")
    if isinstance(raw_layers, int):
        if raw_layers < 0:
            raise ValueError(f"action_bearing_inject_layers must be >= 0, got {raw_layers}")
        return int(raw_layers), None
    if isinstance(raw_layers, str):
        raw_layers = raw_layers.strip()
        if raw_layers.lower() in {"", "all", "none", "null"}:
            return None, None
        raw_layers = raw_layers.strip("[]")
        raw_layers = [part.strip() for part in raw_layers.split(",") if part.strip()]
    if not isinstance(raw_layers, (list, tuple)):
        raise TypeError(
            "action_bearing_inject_layers must be an int legacy count or an explicit list of decoder layer indices, "
            f"got {type(raw_layers).__name__}"
        )

    indices = []
    for layer_idx in raw_layers:
        if isinstance(layer_idx, str):
            layer_idx = layer_idx.strip()
            if not layer_idx:
                continue
            layer_idx = int(layer_idx)
        if isinstance(layer_idx, bool) or not isinstance(layer_idx, int):
            raise TypeError(
                "action_bearing_inject_layers explicit mode expects integer decoder layer indices, "
                f"got {layer_idx!r}"
            )
        if layer_idx < 0:
            raise ValueError(f"action_bearing_inject_layers layer indices must be >= 0, got {layer_idx}")
        indices.append(int(layer_idx))
    return None, tuple(dict.fromkeys(indices))


def ensure_erp_vision_config(vision_config) -> None:
    defaults = {
        "erp_pos_enabled": False,
        "erp_pos_hidden_size": getattr(vision_config, "hidden_size", 1024),
        "erp_pos_alpha_value": 0.02,
        "erp_assume_centered": True,
        "erp_center_latitude_deg": 0.0,
        "erp_apply_to_current_only": True,
    }
    for field_name, default_value in defaults.items():
        if not hasattr(vision_config, field_name):
            setattr(vision_config, field_name, default_value)


class ERPPositionMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        ensure_erp_vision_config(config)

        self.enabled = bool(getattr(config, "erp_pos_enabled", False))
        self.hidden_size = int(getattr(config, "erp_pos_hidden_size", config.hidden_size))
        self.register_buffer(
            "alpha_value",
            torch.tensor(float(getattr(config, "erp_pos_alpha_value", 0.02)), dtype=torch.float32),
        )
        self.assume_centered = bool(getattr(config, "erp_assume_centered", True))
        self.center_latitude_deg = float(getattr(config, "erp_center_latitude_deg", 0.0))
        self.apply_to_current_only = bool(getattr(config, "erp_apply_to_current_only", True))
        self.output_hidden_size = int(config.hidden_size)
        self.initializer_range = float(getattr(config, "initializer_range", 0.02))

        self.mlp = nn.Sequential(
            nn.Linear(4, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.output_hidden_size),
        )
        self.output_norm = nn.RMSNorm(self.output_hidden_size, eps=1e-6)
        self.reset_parameters(self.initializer_range)

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_value

    @staticmethod
    def infer_vertical_fov_radians(height: int, width: int) -> float:
        if height <= 0 or width <= 0:
            return 0.0
        return min(math.pi, 2.0 * math.pi * float(height) / float(width))

    def reset_parameters(self, init_std: float | None = None) -> None:
        init_std = self.initializer_range if init_std is None else float(init_std)
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=init_std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        self.output_norm.reset_parameters()

    def _build_position_features(
        self,
        *,
        num_frames: int,
        height: int,
        width: int,
        device: torch.device,
        dtype: torch.dtype,
        vertical_fov: float,
        center_latitude: float,
    ) -> torch.Tensor:
        if not (0.0 < vertical_fov <= math.pi):
            raise AssertionError(f"Invalid ERP vertical_fov={vertical_fov}")
        if abs(center_latitude) > (0.5 * math.pi):
            raise AssertionError(f"Invalid ERP center_latitude={center_latitude}")

        ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5
        xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5

        lat = (ys[:, None] / float(height) - 0.5) * vertical_fov + center_latitude
        lon = (xs[None, :] / float(width) - 0.5) * (2.0 * math.pi)
        lat = lat.expand(height, width)
        lon = lon.expand(height, width)

        position_features = torch.stack(
            [torch.sin(lat), torch.cos(lat), torch.sin(lon), torch.cos(lon)],
            dim=-1,
        ).reshape(height * width, 4)
        if num_frames > 1:
            position_features = position_features.repeat(num_frames, 1)
        return position_features.to(dtype=dtype)

    def apply_to_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        grid_thw: torch.Tensor | None,
        image_apply_mask: torch.Tensor | None = None,
        image_erp_geometry: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled or grid_thw is None or grid_thw.numel() == 0:
            return patch_tokens

        mlp_param = next(self.mlp.parameters())
        param_device = mlp_param.device
        param_dtype = mlp_param.dtype
        default_center_latitude = 0.0
        if not self.assume_centered:
            default_center_latitude = math.radians(self.center_latitude_deg)

        if grid_thw.ndim != 2 or int(grid_thw.shape[-1]) != 3:
            raise AssertionError(f"Expected image_grid_thw shape [N, 3], got {tuple(grid_thw.shape)}")
        num_images = int(grid_thw.shape[0])
        if image_apply_mask is not None:
            if image_apply_mask.ndim != 1 or int(image_apply_mask.shape[0]) != num_images:
                raise AssertionError(
                    f"Expected image_apply_mask to have shape [{num_images}], "
                    f"got {tuple(image_apply_mask.shape)}"
                )
            apply_mask_list = image_apply_mask.detach().to(device="cpu", dtype=torch.bool).tolist()
        else:
            apply_mask_list = [True] * num_images

        geometry_list = None
        if image_erp_geometry is not None:
            if image_erp_geometry.ndim != 2 or tuple(image_erp_geometry.shape) != (num_images, 2):
                raise AssertionError(
                    "Expected image_erp_geometry to have shape "
                    f"[{num_images}, 2], got {tuple(image_erp_geometry.shape)}"
                )
            geometry_list = image_erp_geometry.detach().to(device="cpu", dtype=torch.float32).tolist()

        split_sizes = [
            int(size)
            for size in grid_thw.prod(-1).detach().to(device="cpu", dtype=torch.long).tolist()
        ]
        if sum(split_sizes) != int(patch_tokens.shape[0]):
            raise AssertionError(
                "Patch token count does not match image_grid_thw for ERP position MLP: "
                f"tokens={int(patch_tokens.shape[0])}, expected={sum(split_sizes)}"
            )
        token_splits = list(patch_tokens.split(split_sizes, dim=0))
        for image_index, (tokens, (num_frames, height, width)) in enumerate(
            zip(token_splits, grid_thw.tolist())
        ):
            if not apply_mask_list[image_index]:
                continue
            if int(height) <= 0 or int(width) <= 0:
                continue

            if geometry_list is not None:
                vertical_fov = float(geometry_list[image_index][0])
                center_latitude = float(geometry_list[image_index][1])
            else:
                vertical_fov = self.infer_vertical_fov_radians(int(height), int(width))
                center_latitude = default_center_latitude

            position_features = self._build_position_features(
                num_frames=int(num_frames),
                height=int(height),
                width=int(width),
                device=param_device,
                dtype=param_dtype,
                vertical_fov=vertical_fov,
                center_latitude=center_latitude,
            )
            position_delta = self.output_norm(self.mlp(position_features))
            position_delta = self.alpha.to(dtype=position_delta.dtype) * position_delta
            token_splits[image_index] = tokens + position_delta.to(
                device=tokens.device,
                dtype=tokens.dtype,
            )

        return torch.cat(token_splits, dim=0)


def ensure_panovggt_config(config) -> None:
    vision_config = config.vision_config
    text_config = getattr(config, "text_config", None)
    text_hidden_size = getattr(text_config, "hidden_size", getattr(vision_config, "out_hidden_size", 3584))
    defaults = {
        "panovggt_enabled": False,
        "panovggt_checkpoint_path": "/workspace/code/a_property/model/PanoVGGT/model.pt",
        "panovggt_alpha_value": 0.1,
        "panovggt_sampling_mode": "grouping",
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
        self.sampling_mode = str(getattr(config, "panovggt_sampling_mode", "grouping")).lower()
        if self.sampling_mode not in {"singlepoint", "grouping"}:
            raise ValueError(
                "panovggt_sampling_mode must be 'singlepoint' or 'grouping', "
                f"got {self.sampling_mode!r}"
            )
        self.register_buffer(
            "alpha_value",
            torch.tensor(float(getattr(config, "panovggt_alpha_value", 0.1)), dtype=torch.float32),
        )
        self.spatial_merge_size = int(getattr(config.vision_config, "spatial_merge_size", 2))
        if self.spatial_merge_size <= 0:
            raise ValueError(f"Qwen spatial_merge_size must be positive, got {self.spatial_merge_size}")
        self.spatial_merge_unit = self.spatial_merge_size**2
        if self.sampling_mode == "grouping":
            self.mlp_input_dim = self.context_dim * self.spatial_merge_unit
        else:
            self.mlp_input_dim = self.context_dim

        self.input_norm = nn.RMSNorm(self.context_dim, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(self.mlp_input_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.output_norm = nn.RMSNorm(self.output_dim, eps=1e-6)
        self.reset_parameters()

    @property
    def alpha(self) -> torch.Tensor:
        return self.alpha_value

    def reset_parameters(self) -> None:
        self.input_norm.reset_parameters()
        self.output_norm.reset_parameters()
        for module in self.mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

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

    def _sample_token_geometry_singlepoint(
        self,
        source_grid: torch.Tensor,
        target_h: int,
        target_w: int,
        geometry: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sampled = self._sample_geometry_grid(
            source_grid,
            target_h=target_h,
            target_w=target_w,
            geometry=geometry,
        )
        return self.input_norm(sampled.to(dtype=dtype))

    def _sample_token_geometry_grouping(
        self,
        source_grid: torch.Tensor,
        target_h: int,
        target_w: int,
        geometry: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        group_h = target_h * self.spatial_merge_size
        group_w = target_w * self.spatial_merge_size
        sampled = self._sample_geometry_grid(
            source_grid,
            target_h=group_h,
            target_w=group_w,
            geometry=geometry,
        )
        sampled = self.input_norm(sampled.to(dtype=dtype))
        sampled = sampled.reshape(
            target_h,
            self.spatial_merge_size,
            target_w,
            self.spatial_merge_size,
            self.context_dim,
        )
        sampled = sampled.permute(0, 2, 1, 3, 4).contiguous()
        return sampled.reshape(target_h * target_w, self.mlp_input_dim)

    def _sample_token_geometry(
        self,
        source_grid: torch.Tensor,
        target_h: int,
        target_w: int,
        geometry: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.sampling_mode == "grouping":
            return self._sample_token_geometry_grouping(
                source_grid,
                target_h=target_h,
                target_w=target_w,
                geometry=geometry,
                dtype=dtype,
            )
        return self._sample_token_geometry_singlepoint(
            source_grid,
            target_h=target_h,
            target_w=target_w,
            geometry=geometry,
            dtype=dtype,
        )

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
        if int(source_grid.shape[0]) != len(target_lengths) or len(target_lengths) != int(target_grid_thw.shape[0]):
            raise AssertionError(
                "PanoVGGT batch size mismatch: "
                f"source={int(source_grid.shape[0])}, target_lengths={len(target_lengths)}, "
                f"target_grid_thw={int(target_grid_thw.shape[0])}"
            )
        if geometry is not None and int(geometry.shape[0]) != len(target_lengths):
            raise AssertionError(
                "PanoVGGT geometry batch size mismatch: "
                f"geometry={int(geometry.shape[0])}, target_lengths={len(target_lengths)}"
            )
        for sample_index, (grid_thw, target_len) in enumerate(zip(target_grid_thw.tolist(), target_lengths)):
            num_frames, grid_h, grid_w = [int(value) for value in grid_thw]
            if num_frames != 1:
                raise AssertionError(
                    "PanoVGGT geometry fusion expects a single current panorama per Qwen image, "
                    f"got image_grid_thw={grid_thw}"
                )
            if grid_h % self.spatial_merge_size != 0 or grid_w % self.spatial_merge_size != 0:
                raise AssertionError(
                    "PanoVGGT target grid is not divisible by Qwen spatial_merge_size: "
                    f"image_grid_thw={grid_thw}, spatial_merge_size={self.spatial_merge_size}"
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

            sample_geometry = None if geometry is None else geometry[sample_index]
            geo = self._sample_token_geometry(
                source_grid[sample_index:sample_index + 1],
                target_h=target_h,
                target_w=target_w,
                geometry=sample_geometry,
                dtype=param.dtype,
            )
            if geo.shape[0] != int(target_len):
                raise AssertionError(
                    "PanoVGGT sampled geometry length mismatch: "
                    f"sampled_len={geo.shape[0]}, qwen_len={int(target_len)}"
                )
            projected = self.output_norm(self.mlp(geo))
            projected = self.alpha.to(dtype=projected.dtype) * projected
            deltas.append(projected.to(device=output_device, dtype=output_dtype))

        return deltas


def ensure_action_bearing_config(config) -> None:
    vision_config = config.vision_config
    text_config = getattr(config, "text_config", None)
    text_hidden_size = getattr(text_config, "hidden_size", getattr(vision_config, "out_hidden_size", 3584))
    output_dim = int(text_hidden_size)
    defaults = {
        "action_bearing_enabled": False,
        "action_bearing_key_alpha_value": 0.02,
        "action_bearing_value_alpha_value": 0.05,
        "action_bearing_inject_layers": None,
        "action_bearing_output_dim": output_dim,
    }
    for field_name, default_value in defaults.items():
        if not hasattr(config, field_name):
            setattr(config, field_name, default_value)


class ActionBearingKV(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        ensure_action_bearing_config(config)

        self.enabled = bool(getattr(config, "action_bearing_enabled", False))
        self.output_dim = int(getattr(config, "action_bearing_output_dim", config.text_config.hidden_size))
        self.hidden_dim = ACTION_BEARING_HIDDEN_SIZE
        self.register_buffer(
            "key_alpha_value",
            torch.tensor(float(getattr(config, "action_bearing_key_alpha_value", 0.02)), dtype=torch.float32),
        )
        self.register_buffer(
            "value_alpha_value",
            torch.tensor(float(getattr(config, "action_bearing_value_alpha_value", 0.05)), dtype=torch.float32),
        )
        self.inject_layers, self.inject_layer_indices = _parse_action_bearing_inject_layers(
            getattr(config, "action_bearing_inject_layers", 16)
        )
        self.inject_layer_index_set = frozenset(self.inject_layer_indices or ())
        self.turn_angle_deg = ACTION_BEARING_TURN_ANGLE_DEG
        self.max_steps = ACTION_BEARING_MAX_STEPS
        self.sigma_steps = ACTION_BEARING_SIGMA_STEPS
        self.spatial_merge_size = int(getattr(config.vision_config, "spatial_merge_size", 2))

        if self.turn_angle_deg <= 0.0:
            raise ValueError(f"action_bearing_turn_angle_deg must be positive, got {self.turn_angle_deg}")
        if self.max_steps < 1:
            raise ValueError(f"action_bearing_max_steps must be >= 1, got {self.max_steps}")
        if self.sigma_steps <= 0.0:
            raise ValueError(f"action_bearing_sigma_steps must be positive, got {self.sigma_steps}")
        self.num_bins = 2 * self.max_steps + 1
        if len(ACTION_BEARING_BIN_TOKEN_IDS) != self.num_bins:
            raise AssertionError(
                "ACTION_BEARING_BIN_TOKEN_IDS must match the action-bearing bin count: "
                f"token_id_rows={len(ACTION_BEARING_BIN_TOKEN_IDS)}, num_bins={self.num_bins}"
            )
        self.bin_embeddings = nn.Parameter(torch.empty(self.num_bins, self.output_dim))
        self.input_norm = nn.RMSNorm(self.output_dim, eps=1e-6)
        self.adapter = nn.Sequential(
            nn.Linear(self.output_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.output_norm = nn.RMSNorm(self.output_dim, eps=1e-6)
        self.reset_parameters(float(getattr(config.vision_config, "initializer_range", 0.02)))

    @property
    def key_alpha(self) -> torch.Tensor:
        return self.key_alpha_value

    @property
    def value_alpha(self) -> torch.Tensor:
        return self.value_alpha_value

    @property
    def turn_angle_radians(self) -> float:
        return math.radians(self.turn_angle_deg)

    def should_inject_layer(self, layer_idx: int | None) -> bool:
        if not self.enabled:
            return False
        if layer_idx is None:
            return False
        layer_idx = int(layer_idx)
        if self.inject_layer_indices is not None:
            return layer_idx in self.inject_layer_index_set
        if self.inject_layers is None:
            return True
        if self.inject_layers <= 0:
            return False
        return layer_idx < self.inject_layers

    def reset_parameters(self, init_std: float = 0.02) -> None:
        nn.init.normal_(self.bin_embeddings, mean=0.0, std=float(init_std))
        self.input_norm.reset_parameters()
        self.output_norm.reset_parameters()
        for module in self.adapter:
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=float(init_std))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def initialize_bin_embeddings_from_text(self, input_embeddings: nn.Embedding | None) -> bool:
        if input_embeddings is None or not hasattr(input_embeddings, "weight"):
            return False
        weight = input_embeddings.weight
        if int(weight.shape[1]) != self.output_dim:
            return False

        phrase_embeddings = []
        vocab_size = int(weight.shape[0])
        for phrase_token_ids in ACTION_BEARING_BIN_TOKEN_IDS:
            token_ids = [self._valid_token_id(token_id, vocab_size) for token_id in phrase_token_ids]
            if any(token_id is None for token_id in token_ids):
                return False
            ids = torch.tensor(token_ids, device=weight.device, dtype=torch.long)
            phrase_embeddings.append(weight.index_select(0, ids).mean(dim=0))

        with torch.no_grad():
            init_values = torch.stack(phrase_embeddings, dim=0).to(
                device=self.bin_embeddings.device,
                dtype=self.bin_embeddings.dtype,
            )
            self.bin_embeddings.copy_(init_values)
        return True

    @staticmethod
    def _valid_token_id(token_id, vocab_size: int) -> int | None:
        if token_id is None:
            return None
        token_id = int(token_id)
        if token_id < 0 or token_id >= vocab_size:
            return None
        return token_id

    def compute_soft_action_scores(self, yaw: torch.Tensor) -> torch.Tensor:
        steps = (yaw.to(dtype=torch.float32) / float(self.turn_angle_radians)).clamp(
            min=-float(self.max_steps),
            max=float(self.max_steps),
        )
        centers = torch.arange(
            -self.max_steps,
            self.max_steps + 1,
            device=steps.device,
            dtype=torch.float32,
        )
        distances = steps.unsqueeze(-1) - centers
        scores = torch.exp(-0.5 * (distances / float(self.sigma_steps)).pow(2))
        return scores / scores.sum(dim=-1, keepdim=True).clamp_min(1e-6)

    def _build_grid_yaw(self, target_h: int, target_w: int, device: torch.device) -> torch.Tensor:
        del target_h
        xs = torch.arange(target_w, device=device, dtype=torch.float32) + 0.5
        yaw = (xs / float(target_w) - 0.5) * (2.0 * math.pi)
        return yaw.unsqueeze(0).expand(1, target_w)

    def _target_grid_hw(self, grid_thw: list[int], target_len: int) -> tuple[int, int]:
        num_frames, grid_h, grid_w = [int(value) for value in grid_thw]
        if num_frames != 1:
            raise AssertionError(
                "Action-Bearing KV expects a single current panorama per Qwen image, "
                f"got image_grid_thw={grid_thw}"
            )
        if grid_h % self.spatial_merge_size != 0 or grid_w % self.spatial_merge_size != 0:
            raise AssertionError(
                "Action-Bearing KV target grid is not divisible by Qwen spatial_merge_size: "
                f"image_grid_thw={grid_thw}, spatial_merge_size={self.spatial_merge_size}"
            )
        target_h = max(1, grid_h // self.spatial_merge_size)
        target_w = max(1, grid_w // self.spatial_merge_size)
        expected_len = target_h * target_w
        if expected_len != int(target_len):
            raise AssertionError(
                "Cannot map Action-Bearing scores to the Qwen 2D visual-token grid: "
                f"image_grid_thw={grid_thw}, spatial_merge_size={self.spatial_merge_size}, "
                f"expected_len={expected_len}, qwen_len={int(target_len)}"
            )
        return target_h, target_w

    def build_image_hidden(
        self,
        grid_thw: torch.Tensor,
        target_len: int,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if not self.enabled or int(target_len) <= 0:
            return torch.empty((0, self.output_dim), device=device, dtype=dtype)
        target_h, target_w = self._target_grid_hw(
            grid_thw.detach().to(device="cpu", dtype=torch.long).tolist(),
            target_len=int(target_len),
        )
        yaw = self._build_grid_yaw(target_h, target_w, device=device).expand(target_h, target_w)
        yaw_flat = yaw.reshape(-1)
        scores = self.compute_soft_action_scores(yaw_flat).to(dtype=self.bin_embeddings.dtype)
        action_prior = scores @ self.bin_embeddings
        base = self.input_norm(action_prior)
        hidden = self.output_norm(base + self.adapter(base))
        return hidden.to(device=device, dtype=dtype)


class Qwen3_5ForConditionalGenerationForPanoVLN(Qwen3_5ForConditionalGeneration):
    _keys_to_ignore_on_load_unexpected = list(
        getattr(Qwen3_5ForConditionalGeneration, "_keys_to_ignore_on_load_unexpected", []) or []
    ) + [r"panovggt\..*"]

    ERP_STATE_KEYS = (
        "erp_position_mlp.alpha_value",
        "erp_position_mlp.mlp.0.weight",
        "erp_position_mlp.mlp.0.bias",
        "erp_position_mlp.mlp.2.weight",
        "erp_position_mlp.mlp.2.bias",
        "erp_position_mlp.output_norm.weight",
    )
    PANOVGGT_STATE_KEYS = (
        "panovggt_mlp.alpha_value",
        "panovggt_mlp.input_norm.weight",
        "panovggt_mlp.mlp.0.weight",
        "panovggt_mlp.mlp.0.bias",
        "panovggt_mlp.mlp.2.weight",
        "panovggt_mlp.mlp.2.bias",
        "panovggt_mlp.output_norm.weight",
    )
    ACTION_BEARING_STATE_KEYS = (
        "action_bearing_kv.key_alpha_value",
        "action_bearing_kv.value_alpha_value",
        "action_bearing_kv.bin_embeddings",
        "action_bearing_kv.input_norm.weight",
        "action_bearing_kv.adapter.0.weight",
        "action_bearing_kv.adapter.0.bias",
        "action_bearing_kv.adapter.2.weight",
        "action_bearing_kv.adapter.2.bias",
        "action_bearing_kv.output_norm.weight",
    )

    def __init__(self, config):
        vision_config = getattr(config, "vision_config", None)
        erp_pos_enabled = bool(
            getattr(vision_config, "erp_pos_enabled", getattr(config, "erp_pos_enabled", False))
        )
        panovggt_enabled = bool(getattr(config, "panovggt_enabled", False))
        action_bearing_enabled = bool(getattr(config, "action_bearing_enabled", False))
        if erp_pos_enabled:
            ensure_erp_vision_config(config.vision_config)
        if panovggt_enabled:
            ensure_panovggt_config(config)
        if action_bearing_enabled:
            ensure_action_bearing_config(config)
        super().__init__(config)

        if erp_pos_enabled:
            ensure_erp_vision_config(self.model.visual.config)
        if panovggt_enabled:
            ensure_panovggt_config(config)
        if action_bearing_enabled:
            ensure_action_bearing_config(config)

        self.erp_position_mlp = ERPPositionMLP(self.model.visual.config) if erp_pos_enabled else None
        self.panovggt_mlp = PanoVGGTGeometryMLP(config) if panovggt_enabled else None
        self.action_bearing_kv = ActionBearingKV(config) if action_bearing_enabled else None
        if self.action_bearing_kv is not None:
            self.action_bearing_kv.initialize_bin_embeddings_from_text(self.get_input_embeddings())
        self.panovggt = None
        self._panovggt_weights_ready = False
        self._panovggt_dtype = None
        self._panovggt_device = None
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_geometry = None
        self._pano_runtime_panovggt_pixel_values = None
        self._install_erp_position_hook()
        self._install_image_feature_hook()
        self._install_action_bearing_kv_hook()
        self.register_load_state_dict_post_hook(self._load_missing_pano_parameters_post_hook)

    def _erp_pos_enabled(self) -> bool:
        return self.erp_position_mlp is not None and bool(getattr(self.erp_position_mlp, "enabled", False))

    def _panovggt_enabled(self) -> bool:
        return self.panovggt_mlp is not None and bool(getattr(self.panovggt_mlp, "enabled", False))

    def _action_bearing_enabled(self) -> bool:
        return self.action_bearing_kv is not None and bool(getattr(self.action_bearing_kv, "enabled", False))

    def _install_erp_position_hook(self) -> None:
        if not self._erp_pos_enabled():
            return
        visual = self.model.visual
        if hasattr(visual.patch_embed, "_pano_origin_forward"):
            return

        visual.patch_embed._pano_origin_forward = visual.patch_embed.forward
        owner = self

        def patch_embed_with_erp(this, hidden_states):
            outputs = this._pano_origin_forward(hidden_states)
            if not owner._erp_pos_enabled():
                return outputs
            grid_thw = owner._pano_runtime_grid_thw
            if grid_thw is None or outputs.numel() == 0:
                return outputs

            image_apply_mask = None
            if owner.erp_position_mlp.apply_to_current_only:
                image_apply_mask = torch.zeros(
                    int(grid_thw.shape[0]),
                    device=grid_thw.device,
                    dtype=torch.bool,
                )
                for _, image_index in owner._current_image_flat_indices(grid_thw):
                    if 0 <= image_index < int(image_apply_mask.shape[0]):
                        image_apply_mask[image_index] = True

            return owner.erp_position_mlp.apply_to_patch_tokens(
                patch_tokens=outputs,
                grid_thw=grid_thw,
                image_apply_mask=image_apply_mask,
                image_erp_geometry=owner._pano_runtime_image_geometry,
            )

        visual.patch_embed.forward = MethodType(patch_embed_with_erp, visual.patch_embed)

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
            if owner._panovggt_enabled():
                if panovggt_pixel_values is None or panovggt_pixel_values.numel() == 0:
                    raise AssertionError("PanoVGGT is enabled but panovggt_pixel_values is missing")
                if int(panovggt_pixel_values.shape[0]) != len(current_items):
                    raise AssertionError(
                        "PanoVGGT current-image batch size mismatch: "
                        f"panovggt_batch={int(panovggt_pixel_values.shape[0])}, "
                        f"current_items={len(current_items)}"
                    )
                panovggt_items = list(current_items)
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

    def _install_action_bearing_kv_hook(self) -> None:
        if not self._action_bearing_enabled():
            return
        language_model = getattr(self.model, "language_model", None)
        layers = getattr(language_model, "layers", None)
        if layers is None:
            return

        action_bearing_kv = self.action_bearing_kv
        full_attention_layers = tuple(
            layer_index
            for layer_index, decoder_layer in enumerate(layers)
            if getattr(decoder_layer, "self_attn", None) is not None
        )
        self._pano_action_bearing_attention_layers = full_attention_layers
        if action_bearing_kv is not None and action_bearing_kv.inject_layer_indices is not None:
            full_attention_layer_set = set(full_attention_layers)
            missing_layers = [
                layer_index
                for layer_index in action_bearing_kv.inject_layer_indices
                if layer_index not in full_attention_layer_set
            ]
            if missing_layers:
                raise ValueError(
                    "Action-Bearing KV can only inject Qwen full-attention decoder layers. "
                    f"Requested layers {missing_layers} are not full-attention layers; "
                    f"available full-attention layers are {list(full_attention_layers)}."
                )
        self._pano_action_bearing_inject_layers = tuple(
            layer_index
            for layer_index in full_attention_layers
            if action_bearing_kv is not None and action_bearing_kv.should_inject_layer(layer_index)
        )

        owner = self
        for layer_index, decoder_layer in enumerate(layers):
            attention = getattr(decoder_layer, "self_attn", None)
            if attention is None or hasattr(attention, "_pano_origin_forward"):
                continue

            attention._pano_origin_forward = attention.forward
            attention._pano_action_bearing_layer_index = layer_index

            def forward_with_action_bearing_kv(
                this,
                hidden_states: torch.Tensor,
                position_embeddings: tuple[torch.Tensor, torch.Tensor],
                attention_mask: torch.Tensor | None,
                past_key_values=None,
                **kwargs,
            ):
                action_hidden = kwargs.pop("action_bearing_hidden", None)
                action_batch_indices = kwargs.pop("action_bearing_batch_indices", None)
                action_token_indices = kwargs.pop("action_bearing_token_indices", None)

                input_shape = hidden_states.shape[:-1]
                hidden_shape = (*input_shape, -1, this.head_dim)

                query_states, gate = torch.chunk(
                    this.q_proj(hidden_states).view(*input_shape, -1, this.head_dim * 2),
                    2,
                    dim=-1,
                )
                gate = gate.reshape(*input_shape, -1)

                query_states = this.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
                key_states = this.k_norm(this.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                value_states = this.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

                if (
                    owner._action_bearing_enabled()
                    and action_hidden is not None
                    and action_batch_indices is not None
                    and action_token_indices is not None
                    and action_hidden.numel() > 0
                ):
                    key_states, value_states = owner._apply_action_bearing_to_kv(
                        attention_module=this,
                        key_states=key_states,
                        value_states=value_states,
                        action_hidden=action_hidden,
                        action_batch_indices=action_batch_indices,
                        action_token_indices=action_token_indices,
                    )

                cos, sin = position_embeddings
                query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

                if past_key_values is not None:
                    key_states, value_states = past_key_values.update(key_states, value_states, this.layer_idx)

                attention_interface = ALL_ATTENTION_FUNCTIONS.get_interface(
                    this.config._attn_implementation,
                    eager_attention_forward,
                )

                attn_output, attn_weights = attention_interface(
                    this,
                    query_states,
                    key_states,
                    value_states,
                    attention_mask,
                    dropout=0.0 if not this.training else this.attention_dropout,
                    scaling=this.scaling,
                    **kwargs,
                )

                attn_output = attn_output.reshape(*input_shape, -1).contiguous()
                attn_output = attn_output * torch.sigmoid(gate)

                attn_output = this.o_proj(attn_output)
                return attn_output, attn_weights

            attention.forward = MethodType(forward_with_action_bearing_kv, attention)

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

    def _build_action_bearing_context(
        self,
        input_ids: torch.Tensor | None,
        image_grid_thw: torch.Tensor | None,
    ) -> dict[str, torch.Tensor]:
        action_bearing_kv = self.action_bearing_kv
        if (
            not self._action_bearing_enabled()
            or action_bearing_kv is None
            or input_ids is None
            or image_grid_thw is None
            or image_grid_thw.numel() == 0
        ):
            return {}
        if (
            not bool((action_bearing_kv.key_alpha_value > 0.0).detach().cpu().item())
            and not bool((action_bearing_kv.value_alpha_value > 0.0).detach().cpu().item())
        ):
            return {}

        if input_ids.ndim != 2:
            raise AssertionError(f"Expected input_ids shape [B, L], got {tuple(input_ids.shape)}")

        input_device = input_ids.device
        image_token_id = int(self.config.image_token_id)
        if not bool(torch.any(input_ids == image_token_id).item()):
            return {}

        image_grid_thw = image_grid_thw.to(device=input_device, dtype=torch.long)
        num_images = int(image_grid_thw.shape[0])
        spatial_merge_size = int(self.model.visual.spatial_merge_size)
        if bool(torch.any(torch.remainder(image_grid_thw[:, 1], spatial_merge_size) != 0).item()) or bool(
            torch.any(torch.remainder(image_grid_thw[:, 2], spatial_merge_size) != 0).item()
        ):
            raise AssertionError(
                "image_grid_thw is not divisible by Qwen spatial_merge_size for action-bearing KV: "
                f"image_grid_thw={image_grid_thw.tolist()}, spatial_merge_size={spatial_merge_size}"
            )
        split_sizes = (image_grid_thw.prod(-1) // spatial_merge_size**2).to(
            device=input_device,
            dtype=torch.long,
        )

        image_num_images = self._pano_runtime_image_num_images
        if image_num_images is None:
            batch_size = int(input_ids.shape[0])
            if batch_size == 1:
                image_num_images = torch.tensor([num_images], device=input_device, dtype=torch.long)
            elif num_images == batch_size:
                image_num_images = torch.ones(batch_size, device=input_device, dtype=torch.long)
            else:
                raise AssertionError(
                    "image_num_images is required for batched multi-image action-bearing KV inputs: "
                    f"input_batch={batch_size}, image_grid_rows={num_images}"
                )
        else:
            image_num_images = image_num_images.to(device=input_device, dtype=torch.long)
        if int(image_num_images.shape[0]) != int(input_ids.shape[0]):
            raise AssertionError(
                "image_num_images batch size does not match input_ids for action-bearing KV: "
                f"counts_batch={int(image_num_images.shape[0])}, input_batch={int(input_ids.shape[0])}"
            )
        if int(image_num_images.sum().item()) != num_images:
            raise AssertionError(
                "image_num_images does not sum to image_grid_thw rows: "
                f"counts={image_num_images.tolist()}, grids={num_images}"
            )

        image_current_index = self._pano_runtime_image_current_index
        if image_current_index is None:
            image_current_index = image_num_images - 1
        else:
            image_current_index = image_current_index.to(device=input_device, dtype=torch.long)

        offsets = image_num_images.cumsum(0) - image_num_images
        action_hidden_chunks = []
        batch_index_chunks = []
        token_index_chunks = []
        param = action_bearing_kv.bin_embeddings

        for batch_index in range(int(input_ids.shape[0])):
            image_count = int(image_num_images[batch_index].item())
            if image_count <= 0:
                continue
            current_index = int(image_current_index[batch_index].item())
            if current_index < 0 or current_index >= image_count:
                raise AssertionError(
                    "image_current_index is out of range for at least one sample: "
                    f"indices={image_current_index.tolist()}, counts={image_num_images.tolist()}"
                )

            image_offset = int(offsets[batch_index].item())
            sample_lengths = split_sizes[image_offset:image_offset + image_count]
            image_token_positions = torch.nonzero(
                input_ids[batch_index] == image_token_id,
                as_tuple=False,
            ).flatten()
            expected_tokens = int(sample_lengths.sum().item())
            if int(image_token_positions.numel()) != expected_tokens:
                raise AssertionError(
                    "Image placeholder count does not match image_grid_thw for action-bearing KV: "
                    f"batch={batch_index}, placeholders={int(image_token_positions.numel())}, "
                    f"expected={expected_tokens}"
                )

            current_token_start = int(sample_lengths[:current_index].sum().item())
            current_token_len = int(sample_lengths[current_index].item())
            current_positions = image_token_positions[current_token_start:current_token_start + current_token_len]
            if int(current_positions.numel()) != current_token_len:
                raise AssertionError(
                    "Current image token span is shorter than expected for action-bearing KV: "
                    f"batch={batch_index}, span={int(current_positions.numel())}, expected={current_token_len}"
                )
            if current_token_len > 1 and not bool(torch.all(current_positions[1:] == current_positions[:-1] + 1).item()):
                raise AssertionError(
                    "Current image token span is not contiguous for action-bearing KV: "
                    f"batch={batch_index}, token_positions={current_positions.tolist()}"
                )

            flat_image_index = image_offset + current_index
            action_hidden = action_bearing_kv.build_image_hidden(
                image_grid_thw[flat_image_index],
                target_len=current_token_len,
                device=param.device,
                dtype=param.dtype,
            )
            action_hidden_chunks.append(action_hidden)
            batch_index_chunks.append(
                torch.full(
                    (current_token_len,),
                    batch_index,
                    device=input_device,
                    dtype=torch.long,
                )
            )
            token_index_chunks.append(current_positions.to(device=input_device, dtype=torch.long))

        if not action_hidden_chunks:
            return {}

        return {
            "action_bearing_hidden": torch.cat(action_hidden_chunks, dim=0),
            "action_bearing_batch_indices": torch.cat(batch_index_chunks, dim=0),
            "action_bearing_token_indices": torch.cat(token_index_chunks, dim=0),
        }

    def _apply_action_bearing_to_kv(
        self,
        *,
        attention_module: nn.Module,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        action_hidden: torch.Tensor,
        action_batch_indices: torch.Tensor,
        action_token_indices: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        action_bearing_kv = self.action_bearing_kv
        if not self._action_bearing_enabled() or action_bearing_kv is None:
            return key_states, value_states

        seq_len = int(key_states.shape[2])
        action_batch_indices = action_batch_indices.to(device=key_states.device, dtype=torch.long)
        action_token_indices = action_token_indices.to(device=key_states.device, dtype=torch.long)
        valid = (
            (action_batch_indices >= 0)
            & (action_batch_indices < int(key_states.shape[0]))
            & (action_token_indices >= 0)
            & (action_token_indices < seq_len)
        )
        if not bool(torch.all(valid).item()):
            action_batch_indices = action_batch_indices[valid]
            action_token_indices = action_token_indices[valid]
            action_hidden = action_hidden[valid.to(device=action_hidden.device)]
        if int(action_token_indices.numel()) == 0:
            return key_states, value_states

        layer_idx = getattr(
            attention_module,
            "_pano_action_bearing_layer_index",
            getattr(attention_module, "layer_idx", None),
        )
        if not action_bearing_kv.should_inject_layer(layer_idx):
            return key_states, value_states

        use_key_delta = bool((action_bearing_kv.key_alpha_value > 0.0).detach().cpu().item())
        use_value_delta = bool((action_bearing_kv.value_alpha_value > 0.0).detach().cpu().item())
        if not use_key_delta and not use_value_delta:
            return key_states, value_states

        action_hidden = action_hidden.to(device=key_states.device, dtype=key_states.dtype)
        selected_count = int(action_hidden.shape[0])

        if use_key_delta:
            key_delta = attention_module.k_norm(
                attention_module.k_proj(action_hidden).view(selected_count, -1, attention_module.head_dim)
            )
            key_delta = key_delta.to(dtype=key_states.dtype)
            full_key_delta = torch.zeros_like(key_states)
            full_key_delta[action_batch_indices, :, action_token_indices, :] = key_delta
            key_alpha = action_bearing_kv.key_alpha.to(device=key_states.device, dtype=key_states.dtype)
            key_states = key_states + key_alpha * full_key_delta

        if use_value_delta:
            value_delta = attention_module.v_proj(action_hidden).view(selected_count, -1, attention_module.head_dim)
            value_delta = value_delta.to(dtype=value_states.dtype)
            full_value_delta = torch.zeros_like(value_states)
            full_value_delta[action_batch_indices, :, action_token_indices, :] = value_delta
            value_alpha = action_bearing_kv.value_alpha.to(device=value_states.device, dtype=value_states.dtype)
            value_states = value_states + value_alpha * full_value_delta

        return key_states, value_states

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
        if self._erp_pos_enabled() or self._panovggt_enabled() or self._action_bearing_enabled():
            self._set_runtime_pano_context(
                image_grid_thw=image_grid_thw,
                image_num_images=image_num_images,
                image_current_index=image_current_index,
                image_erp_geometry=image_erp_geometry,
                panovggt_pixel_values=panovggt_pixel_values,
            )
        action_bearing_context = self._build_action_bearing_context(input_ids, image_grid_thw)
        try:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                logits_to_keep=logits_to_keep,
                **action_bearing_context,
                **kwargs,
            )
        finally:
            self._clear_runtime_pano_context()

    def _load_missing_pano_parameters_post_hook(self, module, incompatible_keys) -> None:
        del module
        missing_keys = set(incompatible_keys.missing_keys)

        erp_module = self.erp_position_mlp
        if (
            erp_module is not None
            and erp_module.enabled
            and any(key.startswith("erp_position_mlp.") for key in missing_keys)
        ):
            erp_module.reset_parameters(erp_module.initializer_range)

        panovggt_module = self.panovggt_mlp
        if (
            panovggt_module is not None
            and panovggt_module.enabled
            and any(key.startswith("panovggt_mlp.") for key in missing_keys)
        ):
            panovggt_module.reset_parameters()
        if panovggt_module is not None and panovggt_module.enabled:
            if any(key.startswith("panovggt.") for key in missing_keys):
                self._load_external_panovggt_weights()
            else:
                self._mark_panovggt_weights_ready()

    def _reset_erp_parameters_after_pretrained_load(self) -> None:
        if self._erp_pos_enabled() and self.erp_position_mlp is not None:
            self.erp_position_mlp.reset_parameters(self.erp_position_mlp.initializer_range)

    def _reset_panovggt_parameters_after_pretrained_load(self) -> None:
        if self._panovggt_enabled() and self.panovggt_mlp is not None:
            self.panovggt_mlp.reset_parameters()

    @classmethod
    def _checkpoint_has_erp_weights(cls, pretrained_model_name_or_path) -> bool | None:
        return cls._checkpoint_has_any_weights(pretrained_model_name_or_path, cls.ERP_STATE_KEYS)

    @classmethod
    def _checkpoint_has_panovggt_weights(cls, pretrained_model_name_or_path) -> bool | None:
        return cls._checkpoint_has_any_weights(pretrained_model_name_or_path, cls.PANOVGGT_STATE_KEYS)

    @classmethod
    def _checkpoint_has_action_bearing_weights(cls, pretrained_model_name_or_path) -> bool | None:
        return cls._checkpoint_has_any_weights(pretrained_model_name_or_path, cls.ACTION_BEARING_STATE_KEYS)

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
            return all(key in weight_map for key in state_keys)

        safetensors_path = checkpoint_dir / "model.safetensors"
        if safetensors_path.exists():
            try:
                from safetensors import safe_open

                with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
                    keys = set(handle.keys())
                return all(key in keys for key in state_keys)
            except Exception:
                return None

        torch_path = checkpoint_dir / "pytorch_model.bin"
        if torch_path.exists():
            try:
                loaded = torch.load(str(torch_path), map_location="cpu")
                if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
                    loaded = loaded["state_dict"]
                if not isinstance(loaded, dict):
                    return None
                return all(key in loaded for key in state_keys)
            except Exception:
                return None

        return None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        has_erp_weights = cls._checkpoint_has_erp_weights(pretrained_model_name_or_path)
        if has_erp_weights is False:
            model._reset_erp_parameters_after_pretrained_load()
        has_panovggt_weights = cls._checkpoint_has_panovggt_weights(pretrained_model_name_or_path)
        if has_panovggt_weights is False:
            model._reset_panovggt_parameters_after_pretrained_load()
        if model._panovggt_enabled():
            loaded_saved_panovggt = (
                model._load_saved_panovggt_weights(pretrained_model_name_or_path)
                if has_panovggt_weights is True
                else False
            )
            if not loaded_saved_panovggt and model.panovggt is None:
                model._load_external_panovggt_weights()
        has_action_bearing_weights = cls._checkpoint_has_action_bearing_weights(pretrained_model_name_or_path)
        if (
            model._action_bearing_enabled()
            and model.action_bearing_kv is not None
            and has_action_bearing_weights is False
        ):
            model.action_bearing_kv.reset_parameters(
                float(getattr(model.config.vision_config, "initializer_range", 0.02))
            )
            model.action_bearing_kv.initialize_bin_embeddings_from_text(model.get_input_embeddings())
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
        image_erp_geometry=None,
        image_num_images=None,
        image_current_index=None,
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
            image_erp_geometry=image_erp_geometry,
            image_num_images=image_num_images,
            image_current_index=image_current_index,
            panovggt_pixel_values=panovggt_pixel_values,
            **kwargs,
        )
        if not is_first_iteration and use_cache:
            model_inputs["image_erp_geometry"] = None
            model_inputs["image_num_images"] = None
            model_inputs["image_current_index"] = None
            model_inputs["panovggt_pixel_values"] = None
        return model_inputs
