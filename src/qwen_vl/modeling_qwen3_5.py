import json
import math
import sys
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration


PANOVGGT_AGGREGATOR_LAYER = -1
PANOVGGT_CONTEXT_DIM = 2048
PANOVGGT_MLP_HIDDEN_SIZE = 4096
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
        raise ValueError(f"alpha_max must be positive, got {alpha_max}")
    alpha_init = min(max(alpha_init, 1e-6), alpha_max * (1.0 - 1e-6))
    return torch.tensor(_inverse_sigmoid(alpha_init / alpha_max), dtype=torch.float32)


def ensure_erp_vision_config(vision_config) -> None:
    defaults = {
        "erp_pos_enabled": True,
        "erp_pos_hidden_size": getattr(vision_config, "hidden_size", 1024),
        "erp_pos_init_scale": 0.0,
        "erp_pos_alpha_init": 0.02,
        "erp_pos_alpha_max": 0.1,
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

        self.enabled = bool(getattr(config, "erp_pos_enabled", True))
        self.hidden_size = int(getattr(config, "erp_pos_hidden_size", config.hidden_size))
        self.alpha_init = float(getattr(config, "erp_pos_alpha_init", 0.02))
        self.alpha_max = float(getattr(config, "erp_pos_alpha_max", 0.1))
        self.assume_centered = bool(getattr(config, "erp_assume_centered", True))
        self.center_latitude_deg = float(getattr(config, "erp_center_latitude_deg", 0.0))
        self.apply_to_current_only = bool(getattr(config, "erp_apply_to_current_only", False))
        self.output_hidden_size = int(config.hidden_size)
        self.initializer_range = float(config.initializer_range)

        self.mlp = nn.Sequential(
            nn.Linear(4, self.hidden_size),
            nn.GELU(),
            nn.Linear(self.hidden_size, self.output_hidden_size),
        )
        self.output_norm = nn.RMSNorm(self.output_hidden_size, eps=1e-6)
        self.raw_alpha = nn.Parameter(_bounded_raw_alpha(self.alpha_init, self.alpha_max))
        self.reset_parameters(self.initializer_range)
        self.register_load_state_dict_post_hook(self._load_state_dict_post_hook)

    @property
    def alpha(self) -> torch.Tensor:
        return float(self.alpha_max) * torch.sigmoid(self.raw_alpha)

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
        with torch.no_grad():
            self.raw_alpha.copy_(_bounded_raw_alpha(self.alpha_init, self.alpha_max))

    def _load_state_dict_post_hook(self, module, incompatible_keys) -> None:
        del module
        missing_keys = set(incompatible_keys.missing_keys)
        if any("erp_position_mlp.mlp" in key for key in missing_keys):
            self.reset_parameters(self.initializer_range)
        if any("erp_position_mlp.output_norm" in key for key in missing_keys):
            self.output_norm.reset_parameters()
        if any(key.endswith("erp_position_mlp.raw_alpha") or key == "raw_alpha" for key in missing_keys):
            with torch.no_grad():
                self.raw_alpha.copy_(_bounded_raw_alpha(self.alpha_init, self.alpha_max))

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

        num_images = int(grid_thw.shape[0])
        if image_apply_mask is not None:
            if image_apply_mask.ndim != 1 or image_apply_mask.shape[0] != num_images:
                raise AssertionError(
                    f"Expected image_apply_mask to have shape [{num_images}], "
                    f"got {tuple(image_apply_mask.shape)}"
                )
            apply_mask_list = image_apply_mask.detach().to(device="cpu", dtype=torch.bool).tolist()
        else:
            apply_mask_list = [True] * num_images

        geometry_list = None
        if image_erp_geometry is not None:
            if image_erp_geometry.ndim != 2 or image_erp_geometry.shape != (num_images, 2):
                raise AssertionError(
                    "Expected image_erp_geometry to have shape "
                    f"[{num_images}, 2], got {tuple(image_erp_geometry.shape)}"
                )
            geometry_list = image_erp_geometry.detach().to(device="cpu", dtype=torch.float32).tolist()

        split_sizes = [
            int(size)
            for size in grid_thw.prod(-1).detach().to(device="cpu", dtype=torch.long).tolist()
        ]
        token_splits = list(patch_tokens.split(split_sizes, dim=0))
        for image_index, (tokens, (num_frames, height, width)) in enumerate(
            zip(token_splits, grid_thw.tolist())
        ):
            if not apply_mask_list[image_index]:
                continue
            if height <= 0 or width <= 0:
                continue

            if geometry_list is not None:
                vertical_fov = float(geometry_list[image_index][0])
                center_latitude = float(geometry_list[image_index][1])
            else:
                vertical_fov = self.infer_vertical_fov_radians(height, width)
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
        "panovggt_checkpoint_path": "/workspace/code_dir/a_property/model/PanoVGGT/model.pt",
        "panovggt_alpha_init": 0.1,
        "panovggt_alpha_max": 0.2,
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


def build_current_image_mask(
    image_grid_thw: torch.Tensor | None,
    image_num_images: torch.Tensor | None = None,
    image_current_index: torch.Tensor | None = None,
) -> torch.Tensor | None:
    if image_grid_thw is None or image_grid_thw.numel() == 0:
        return None

    num_images = int(image_grid_thw.shape[0])
    device = image_grid_thw.device

    if image_num_images is None:
        image_num_images = torch.tensor([num_images], device=device, dtype=torch.long)
    else:
        if image_num_images.ndim != 1:
            raise AssertionError(
                f"Expected image_num_images to have shape [batch_size], "
                f"got {tuple(image_num_images.shape)}"
            )
        image_num_images = image_num_images.to(device=device, dtype=torch.long)

    if int(image_num_images.sum().item()) != num_images:
        raise AssertionError(
            "image_num_images does not sum to image_grid_thw rows: "
            f"counts={image_num_images.tolist()}, grids={num_images}"
        )

    if image_current_index is None:
        image_current_index = image_num_images - 1
    else:
        if image_current_index.ndim != 1 or image_current_index.shape[0] != image_num_images.shape[0]:
            raise AssertionError(
                "Expected image_current_index to have shape [batch_size], "
                f"got {tuple(image_current_index.shape)}"
            )
        image_current_index = image_current_index.to(device=device, dtype=torch.long)

    valid_samples = image_num_images > 0
    if torch.any(valid_samples):
        invalid_indices = (image_current_index < 0) | (image_current_index >= image_num_images)
        if torch.any(invalid_indices & valid_samples):
            raise AssertionError(
                "image_current_index is out of range for at least one sample: "
                f"indices={image_current_index.tolist()}, counts={image_num_images.tolist()}"
            )

    offsets = image_num_images.cumsum(0) - image_num_images
    current_flat_indices = offsets + image_current_index
    image_apply_mask = torch.zeros(num_images, device=device, dtype=torch.bool)
    if torch.any(valid_samples):
        image_apply_mask[current_flat_indices[valid_samples]] = True
    return image_apply_mask


class Qwen3_5ForConditionalGenerationForPanoVLN(Qwen3_5ForConditionalGeneration):
    _keys_to_ignore_on_load_unexpected = list(
        getattr(Qwen3_5ForConditionalGeneration, "_keys_to_ignore_on_load_unexpected", []) or []
    ) + [r"panovggt\..*"]

    ERP_STATE_KEYS = (
        "erp_position_mlp.raw_alpha",
        "erp_position_mlp.mlp.0.weight",
        "erp_position_mlp.mlp.0.bias",
        "erp_position_mlp.mlp.2.weight",
        "erp_position_mlp.mlp.2.bias",
        "erp_position_mlp.output_norm.weight",
    )
    PANOVGGT_STATE_KEYS = (
        "panovggt_mlp.raw_alpha",
        "panovggt_mlp.input_norm.weight",
        "panovggt_mlp.mlp.0.weight",
        "panovggt_mlp.mlp.0.bias",
        "panovggt_mlp.mlp.2.weight",
        "panovggt_mlp.mlp.2.bias",
        "panovggt_mlp.output_norm.weight",
    )

    def __init__(self, config):
        ensure_erp_vision_config(config.vision_config)
        ensure_panovggt_config(config)
        super().__init__(config)

        visual_config = self.model.visual.config
        ensure_erp_vision_config(visual_config)
        ensure_panovggt_config(config)
        self.erp_position_mlp = ERPPositionMLP(visual_config)
        self.panovggt_mlp = PanoVGGTGeometryMLP(config)
        self.panovggt = None
        self._panovggt_weights_ready = False
        self._panovggt_dtype = None
        self._panovggt_device = None
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_erp_geometry = None
        self._pano_runtime_panovggt_pixel_values = None
        self._install_patch_embed_hook()
        self._install_image_feature_hook()
        self.register_load_state_dict_post_hook(self._load_missing_pano_parameters_post_hook)

    def _install_patch_embed_hook(self) -> None:
        visual = self.model.visual
        if hasattr(visual.patch_embed, "_pano_origin_forward"):
            return

        visual.patch_embed._pano_origin_forward = visual.patch_embed.forward
        owner = self

        def patch_embed_with_erp(this, hidden_states):
            outputs = this._pano_origin_forward(hidden_states)
            grid_thw = owner._pano_runtime_grid_thw
            if (
                not owner.erp_position_mlp.enabled
                or grid_thw is None
                or outputs.numel() == 0
            ):
                return outputs

            image_apply_mask = None
            if owner.erp_position_mlp.apply_to_current_only:
                image_apply_mask = build_current_image_mask(
                    image_grid_thw=grid_thw,
                    image_num_images=owner._pano_runtime_image_num_images,
                    image_current_index=owner._pano_runtime_image_current_index,
                )
            return owner.erp_position_mlp.apply_to_patch_tokens(
                patch_tokens=outputs,
                grid_thw=grid_thw,
                image_apply_mask=image_apply_mask,
                image_erp_geometry=owner._pano_runtime_image_erp_geometry,
            )

        visual.patch_embed.forward = MethodType(patch_embed_with_erp, visual.patch_embed)

    def _install_image_feature_hook(self) -> None:
        if hasattr(self.model, "_pano_origin_get_image_features"):
            return

        self.model._pano_origin_get_image_features = self.model.get_image_features
        owner = self

        def get_image_features_with_panovggt(this, pixel_values, image_grid_thw=None, **kwargs):
            vision_output = this._pano_origin_get_image_features(
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                **kwargs,
            )
            if not owner.panovggt_mlp.enabled:
                return vision_output
            panovggt_pixel_values = owner._pano_runtime_panovggt_pixel_values
            if (
                panovggt_pixel_values is None
                or image_grid_thw is None
                or vision_output.pooler_output is None
            ):
                return vision_output

            current_indices = owner._current_image_flat_indices(image_grid_thw)
            if not current_indices:
                return vision_output

            image_embeds = list(vision_output.pooler_output)
            valid_items = []
            for batch_index, image_index in current_indices:
                if image_index < 0 or image_index >= len(image_embeds):
                    continue
                if batch_index < 0 or batch_index >= int(panovggt_pixel_values.shape[0]):
                    continue
                valid_items.append((batch_index, image_index))
            if not valid_items:
                return vision_output

            batch_indices = torch.tensor(
                [item[0] for item in valid_items],
                device=panovggt_pixel_values.device,
                dtype=torch.long,
            )
            target_indices = [item[1] for item in valid_items]
            target_grid_thw = image_grid_thw[target_indices].detach().to(device="cpu", dtype=torch.long)
            target_lengths = [int(image_embeds[index].shape[0]) for index in target_indices]
            geometry = owner._pano_runtime_image_erp_geometry
            if geometry is not None:
                geometry = geometry[target_indices].detach().to(device="cpu", dtype=torch.float32)

            panovggt_model = owner._ensure_panovggt_model(
                device=image_embeds[target_indices[0]].device,
                dtype=image_embeds[target_indices[0]].dtype,
            )
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

        self.model.get_image_features = MethodType(get_image_features_with_panovggt, self.model)

    def _set_runtime_erp_context(
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
        self._pano_runtime_image_erp_geometry = image_erp_geometry
        self._pano_runtime_panovggt_pixel_values = panovggt_pixel_values

    def _clear_runtime_erp_context(self) -> None:
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_erp_geometry = None
        self._pano_runtime_panovggt_pixel_values = None

    def _load_external_panovggt_weights(self) -> None:
        if not self.panovggt_mlp.enabled:
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
        if not self.panovggt_mlp.enabled:
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
        if not self.panovggt_mlp.enabled:
            return None
        if self.panovggt is None:
            self.panovggt = build_panovggt_model_from_vendored_config()
        if not self._panovggt_weights_ready:
            self._load_external_panovggt_weights()
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
        self._set_runtime_erp_context(
            image_grid_thw=image_grid_thw,
            image_num_images=image_num_images,
            image_current_index=image_current_index,
            image_erp_geometry=image_erp_geometry,
            panovggt_pixel_values=panovggt_pixel_values,
        )
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
                **kwargs,
            )
        finally:
            self._clear_runtime_erp_context()

    def _load_missing_pano_parameters_post_hook(self, module, incompatible_keys) -> None:
        del module
        erp_module = self.erp_position_mlp
        missing_keys = set(incompatible_keys.missing_keys)
        if erp_module.enabled:
            if any(key.startswith("erp_position_mlp.mlp") for key in missing_keys):
                erp_module.reset_parameters(erp_module.initializer_range)
            if any(key.startswith("erp_position_mlp.output_norm") for key in missing_keys):
                erp_module.output_norm.reset_parameters()
            if any(key == "erp_position_mlp.raw_alpha" or key == "raw_alpha" for key in missing_keys):
                with torch.no_grad():
                    erp_module.raw_alpha.copy_(_bounded_raw_alpha(erp_module.alpha_init, erp_module.alpha_max))

        panovggt_module = self.panovggt_mlp
        if panovggt_module.enabled and any(key.startswith("panovggt_mlp.") for key in missing_keys):
            panovggt_module.reset_parameters()
        if panovggt_module.enabled:
            if any(key.startswith("panovggt.") for key in missing_keys):
                self._load_external_panovggt_weights()
            else:
                self._mark_panovggt_weights_ready()

    def _reset_erp_parameters_after_pretrained_load(self) -> None:
        erp_module = self.erp_position_mlp
        if not erp_module.enabled:
            return
        erp_module.reset_parameters(erp_module.initializer_range)

    def _reset_panovggt_parameters_after_pretrained_load(self) -> None:
        if self.panovggt_mlp.enabled:
            self.panovggt_mlp.reset_parameters()

    @classmethod
    def _checkpoint_has_erp_weights(cls, pretrained_model_name_or_path) -> bool | None:
        return cls._checkpoint_has_any_weights(pretrained_model_name_or_path, cls.ERP_STATE_KEYS)

    @classmethod
    def _checkpoint_has_panovggt_weights(cls, pretrained_model_name_or_path) -> bool | None:
        return cls._checkpoint_has_any_weights(pretrained_model_name_or_path, cls.PANOVGGT_STATE_KEYS)

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
        has_erp_weights = cls._checkpoint_has_erp_weights(pretrained_model_name_or_path)
        if has_erp_weights is False:
            model._reset_erp_parameters_after_pretrained_load()
        has_panovggt_weights = cls._checkpoint_has_panovggt_weights(pretrained_model_name_or_path)
        if has_panovggt_weights is False:
            model._reset_panovggt_parameters_after_pretrained_load()
        if model.panovggt_mlp.enabled:
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
