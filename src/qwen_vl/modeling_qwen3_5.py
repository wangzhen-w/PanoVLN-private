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
PANOVGGT_AGGREGATOR_CONTEXT_DIM = 2048
PANOVGGT_POINT_HIDDEN_DIM = 1024
PANOVGGT_FEATURE_SOURCES = {"aggregator", "point_hidden"}
PANOVGGT_INJECTION_STAGES = {"post_merger", "pre_merger"}
PANOVGGT_MLP_HIDDEN_SIZE = 4096
PBO_ACTION_HORIZON = 4
PBO_NUM_ACTIONS = 4
# Six is the current endpoint-only PBO input. Legacy checkpoints may store
# pbo_input_vector_count=7 and prepend one LLM instruction-context vector.
PBO_INPUT_VECTOR_COUNT = 6
PBO_SUPPORTED_INPUT_VECTOR_COUNTS = {6, 7}
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
VENDORED_PANOVGGT_DIR = SRC_ROOT / "panovggt"
VENDORED_PANOVGGT_CONFIG_PATH = VENDORED_PANOVGGT_DIR / "training" / "config" / "default.yaml"

def ensure_panovggt_config(config) -> None:
    vision_config = config.vision_config
    text_config = getattr(config, "text_config", None)
    text_hidden_size = getattr(text_config, "hidden_size", getattr(vision_config, "out_hidden_size", 3584))
    defaults = {
        "panovggt_enabled": False,
        "panovggt_checkpoint_path": "/workspace/code/a_property/model/PanoVGGT/model.pt",
        "panovggt_alpha_value": 0.1,
        "panovggt_feature_source": "aggregator",
        "panovggt_injection_stage": "post_merger",
        "panovggt_sampling_mode": "grouping",
        "panovggt_force_fp32": False,
        "panovggt_output_dim": int(getattr(vision_config, "out_hidden_size", text_hidden_size)),
    }
    for field_name, default_value in defaults.items():
        if not hasattr(config, field_name):
            setattr(config, field_name, default_value)


def ensure_pbo_config(config) -> None:
    defaults = {
        "pbo_enabled": False,
        "pbo_loss_weight": 0.1,
        "pbo_head_hidden_size": 512,
        "pbo_input_vector_count": PBO_INPUT_VECTOR_COUNT,
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


def normalize_panovggt_feature_source(feature_source: str) -> str:
    feature_source = str(feature_source).lower()
    if feature_source not in PANOVGGT_FEATURE_SOURCES:
        raise ValueError(
            "panovggt_feature_source must be 'aggregator' or 'point_hidden', "
            f"got {feature_source!r}"
        )
    return feature_source


def normalize_panovggt_injection_stage(injection_stage: str) -> str:
    injection_stage = str(injection_stage).lower()
    if injection_stage not in PANOVGGT_INJECTION_STAGES:
        raise ValueError(
            "panovggt_injection_stage must be 'post_merger' or 'pre_merger', "
            f"got {injection_stage!r}"
        )
    return injection_stage


def build_panovggt_model_from_vendored_config(feature_source: str = "aggregator"):
    _ensure_vendored_panovggt_available()
    feature_source = normalize_panovggt_feature_source(feature_source)
    enable_point = feature_source == "point_hidden"

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
            enable_point=enable_point,
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
    required_prefixes = ["aggregator."]
    if bool(getattr(model, "enable_point", False)):
        required_prefixes.extend(("point_decoder.", "pos_adapters.point."))
    missing_required_keys = [
        key
        for key in getattr(load_result, "missing_keys", [])
        if any(key.startswith(prefix) for prefix in required_prefixes)
    ]
    if missing_required_keys:
        raise RuntimeError(
            "PanoVGGT checkpoint is missing required weights, "
            f"first missing key: {missing_required_keys[0]}"
        )
    freeze_panovggt_model(model)

class PanoVGGTGeometryMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        ensure_panovggt_config(config)

        self.enabled = bool(getattr(config, "panovggt_enabled", False))
        self.layer = PANOVGGT_AGGREGATOR_LAYER
        self.feature_source = normalize_panovggt_feature_source(
            getattr(config, "panovggt_feature_source", "aggregator")
        )
        self.injection_stage = normalize_panovggt_injection_stage(
            getattr(config, "panovggt_injection_stage", "post_merger")
        )
        self.context_dim = (
            PANOVGGT_POINT_HIDDEN_DIM
            if self.feature_source == "point_hidden"
            else PANOVGGT_AGGREGATOR_CONTEXT_DIM
        )
        if self.injection_stage == "pre_merger":
            self.output_dim = int(getattr(config.vision_config, "hidden_size"))
        else:
            self.output_dim = int(getattr(config, "panovggt_output_dim", config.text_config.hidden_size))
        self.hidden_dim = PANOVGGT_MLP_HIDDEN_SIZE
        self.sampling_mode = str(getattr(config, "panovggt_sampling_mode", "grouping")).lower()
        if self.sampling_mode not in {"singlepoint", "grouping"}:
            raise ValueError(
                "panovggt_sampling_mode must be 'singlepoint' or 'grouping', "
                f"got {self.sampling_mode!r}"
            )
        self.alpha_init = float(getattr(config, "panovggt_alpha_value", 0.1))
        self.register_buffer(
            "alpha_value",
            torch.tensor(self.alpha_init, dtype=torch.float32),
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
        self.alpha_value.fill_(self.alpha_init)

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

    def _to_qwen_pre_merger_order(
        self,
        tokens: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        if grid_h % self.spatial_merge_size != 0 or grid_w % self.spatial_merge_size != 0:
            raise AssertionError(
                "PanoVGGT pre-merger target grid is not divisible by Qwen spatial_merge_size: "
                f"grid=({grid_h}, {grid_w}), spatial_merge_size={self.spatial_merge_size}"
            )
        if int(tokens.shape[0]) != int(grid_h * grid_w):
            raise AssertionError(
                "PanoVGGT pre-merger token length mismatch before Qwen order conversion: "
                f"tokens={int(tokens.shape[0])}, grid=({grid_h}, {grid_w})"
            )
        tokens = tokens.reshape(
            grid_h // self.spatial_merge_size,
            self.spatial_merge_size,
            grid_w // self.spatial_merge_size,
            self.spatial_merge_size,
            tokens.shape[-1],
        )
        tokens = tokens.permute(0, 2, 1, 3, 4).contiguous()
        return tokens.reshape(grid_h * grid_w, -1)

    def _point_decoder_xpos(
        self,
        *,
        encoder: nn.Module,
        pos_2d: torch.Tensor | None,
        batch_frames: int,
        token_count: int,
        patch_start_idx: int,
        patch_h: int,
        patch_w: int,
        device: torch.device,
    ) -> torch.Tensor | None:
        if getattr(encoder.aggregator, "rope", None) is None:
            return None
        if pos_2d is not None:
            if int(pos_2d.shape[0]) != batch_frames or int(pos_2d.shape[1]) != token_count:
                raise AssertionError(
                    "PanoVGGT point decoder xpos shape mismatch: "
                    f"xpos={tuple(pos_2d.shape)}, expected=({batch_frames}, {token_count}, 2)"
                )
            return pos_2d
        if not hasattr(encoder.aggregator, "position_getter"):
            raise AssertionError("PanoVGGT point decoder requires aggregator.position_getter for RoPE xpos")
        pos_2d = encoder.aggregator.position_getter(batch_frames, patch_h, patch_w, device)
        pos_2d = pos_2d + 1
        pos_special = torch.zeros(
            batch_frames,
            patch_start_idx,
            2,
            device=device,
            dtype=pos_2d.dtype,
        )
        return torch.cat([pos_special, pos_2d], dim=1)

    def _decode_point_hidden_tokens(
        self,
        *,
        encoder: nn.Module,
        tokens: torch.Tensor,
        patch_start_idx: int,
        image_h: int,
        image_w: int,
        pos_2d: torch.Tensor | None,
    ) -> torch.Tensor:
        if not hasattr(encoder, "point_decoder") or not hasattr(encoder, "_get_branch_pos_embed"):
            raise AssertionError(
                "panovggt_feature_source='point_hidden' requires PanoVGGT to be built with enable_point=True"
            )
        batch_size, num_frames, token_count, token_dim = [int(value) for value in tokens.shape]
        if token_dim != PANOVGGT_AGGREGATOR_CONTEXT_DIM:
            raise AssertionError(
                "PanoVGGT point decoder expects aggregator tokens with "
                f"{PANOVGGT_AGGREGATOR_CONTEXT_DIM} channels, got {token_dim}"
            )
        patch_size = int(getattr(encoder, "patch_size", 0))
        patch_count = token_count - int(patch_start_idx)
        patch_h, patch_w = self._infer_patch_grid(
            patch_count=patch_count,
            image_height=int(image_h),
            image_width=int(image_w),
            patch_size=patch_size,
        )
        batch_frames = batch_size * num_frames
        decoder_tokens = tokens.reshape(batch_frames, token_count, token_dim)
        decoder_xpos = self._point_decoder_xpos(
            encoder=encoder,
            pos_2d=pos_2d,
            batch_frames=batch_frames,
            token_count=token_count,
            patch_start_idx=patch_start_idx,
            patch_h=patch_h,
            patch_w=patch_w,
            device=decoder_tokens.device,
        )
        pos_embed = encoder._get_branch_pos_embed(
            patch_h,
            patch_w,
            patch_start_idx,
            decoder_tokens.device,
            decoder_tokens.dtype,
            "point",
            batch_frames,
        )
        point_hidden = encoder.point_decoder(decoder_tokens, pos_embed=pos_embed, xpos=decoder_xpos)
        if point_hidden.ndim != 3:
            raise AssertionError(
                f"Expected PanoVGGT point hidden [B*S, P, C], got {tuple(point_hidden.shape)}"
            )
        if int(point_hidden.shape[0]) != batch_frames or int(point_hidden.shape[1]) != token_count:
            raise AssertionError(
                "PanoVGGT point hidden shape mismatch: "
                f"point_hidden={tuple(point_hidden.shape)}, expected=({batch_frames}, {token_count}, C)"
            )
        return point_hidden.reshape(batch_size, num_frames, token_count, int(point_hidden.shape[-1]))

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
            pos_2d = None
            if isinstance(aggregated, (list, tuple)):
                token_list = aggregated[0]
                patch_start_idx = int(aggregated[1])
                pos_2d = aggregated[2] if len(aggregated) > 2 else None
                tokens = token_list[self.layer] if isinstance(token_list, list) else token_list
            else:
                tokens = aggregated
                patch_start_idx = 0

            if tokens.ndim != 4:
                raise AssertionError(f"Expected PanoVGGT tokens [B, S, P, C], got {tuple(tokens.shape)}")
            _, _, _, image_h, image_w = images.shape
            if self.feature_source == "point_hidden":
                tokens = self._decode_point_hidden_tokens(
                    encoder=encoder,
                    tokens=tokens,
                    patch_start_idx=patch_start_idx,
                    image_h=int(image_h),
                    image_w=int(image_w),
                    pos_2d=pos_2d,
                )
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
            if self.injection_stage == "pre_merger":
                target_h = grid_h
                target_w = grid_w
            else:
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
            if self.injection_stage == "pre_merger":
                projected = self._to_qwen_pre_merger_order(projected, grid_h=grid_h, grid_w=grid_w)
            deltas.append(projected.to(device=output_device, dtype=output_dtype))

        return deltas


class Qwen3_5ForConditionalGenerationForPanoVLN(Qwen3_5ForConditionalGeneration):
    _keys_to_ignore_on_load_unexpected = list(
        getattr(Qwen3_5ForConditionalGeneration, "_keys_to_ignore_on_load_unexpected", []) or []
    ) + [
        r"panovggt\..*",
    ]

    PANOVGGT_STATE_KEYS = (
        "panovggt_mlp.alpha_value",
        "panovggt_mlp.input_norm.weight",
        "panovggt_mlp.mlp.0.weight",
        "panovggt_mlp.mlp.0.bias",
        "panovggt_mlp.mlp.2.weight",
        "panovggt_mlp.mlp.2.bias",
        "panovggt_mlp.output_norm.weight",
    )
    def __init__(self, config):
        ensure_pbo_config(config)
        panovggt_enabled = bool(getattr(config, "panovggt_enabled", False))
        if panovggt_enabled:
            ensure_panovggt_config(config)
        super().__init__(config)

        if panovggt_enabled:
            ensure_panovggt_config(config)

        self.panovggt_mlp = PanoVGGTGeometryMLP(config) if panovggt_enabled else None
        text_hidden_size = int(config.text_config.hidden_size)
        pbo_hidden_size = int(getattr(config, "pbo_head_hidden_size", 512))
        pbo_input_vector_count = int(
            getattr(config, "pbo_input_vector_count", PBO_INPUT_VECTOR_COUNT)
        )
        if pbo_input_vector_count not in PBO_SUPPORTED_INPUT_VECTOR_COUNTS:
            raise ValueError(
                "pbo_input_vector_count must be 6 or 7, "
                f"got {pbo_input_vector_count}"
            )
        self.pbo_head = None
        if bool(getattr(config, "pbo_enabled", False)):
            self.pbo_head = nn.Sequential(
                nn.LayerNorm(text_hidden_size * pbo_input_vector_count),
                nn.Linear(
                    text_hidden_size * pbo_input_vector_count,
                    pbo_hidden_size,
                ),
                nn.GELU(),
                nn.Linear(
                    pbo_hidden_size,
                    PBO_ACTION_HORIZON * PBO_NUM_ACTIONS,
                ),
            )
            self.pbo_head.apply(self._init_weights)

        self.panovggt = None
        self._panovggt_weights_ready = False
        self._panovggt_dtype = None
        self._panovggt_device = None
        self._panovggt_mlp_load_was_incompatible = False
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_geometry = None
        self._pano_runtime_panovggt_pixel_values = None
        self._pano_runtime_in_image_features = False
        self._pano_runtime_visual_grid_thw = None
        self._capture_auxiliary_hidden = False
        self._auxiliary_last_hidden = None
        self.model.language_model.norm.register_forward_hook(
            self._capture_auxiliary_hidden_hook
        )
        self._install_pano_merger_hook()
        self._install_image_feature_hook()
        self.register_load_state_dict_pre_hook(self._drop_incompatible_panovggt_mlp_pre_hook)
        self.register_load_state_dict_post_hook(self._load_missing_pano_parameters_post_hook)

    def _capture_auxiliary_hidden_hook(self, module, inputs, output) -> None:
        del module, inputs
        if self._capture_auxiliary_hidden:
            self._auxiliary_last_hidden = output

    def _pbo_enabled(self) -> bool:
        return self.pbo_head is not None and bool(getattr(self.config, "pbo_enabled", False))

    @staticmethod
    def _distributed_weighted_mean(
        local_sum: torch.Tensor,
        local_weight_sum: torch.Tensor,
    ) -> torch.Tensor:
        weight_sum = local_weight_sum.detach().to(
            device=local_sum.device,
            dtype=torch.float32,
        )
        world_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(
                weight_sum,
                op=torch.distributed.ReduceOp.SUM,
            )
            world_size = torch.distributed.get_world_size()
        if float(weight_sum.item()) <= 0.0:
            return local_sum * 0.0
        # DDP/DeepSpeed averages gradients across data-parallel ranks. Scaling
        # each local sum by world_size/global_weight therefore produces the true
        # global weighted mean after gradient reduction.
        return local_sum * (float(world_size) / weight_sum)

    @classmethod
    def _distributed_masked_mean(
        cls,
        local_sum: torch.Tensor,
        local_count: int,
    ) -> torch.Tensor:
        count = torch.tensor(
            float(local_count),
            device=local_sum.device,
            dtype=torch.float32,
        )
        return cls._distributed_weighted_mean(local_sum, count)

    @classmethod
    def _compute_weighted_causal_lm_loss(
        cls,
        logits: torch.Tensor,
        labels: torch.Tensor,
        loss_weights: torch.Tensor,
    ) -> torch.Tensor:
        if logits.ndim != 3 or labels.ndim != 2:
            raise ValueError(
                "Weighted causal LM loss expects [batch, sequence, vocab] logits "
                "and [batch, sequence] labels"
            )
        if logits.shape[:2] != labels.shape or loss_weights.shape != labels.shape:
            raise ValueError(
                "logits, labels, and loss_weights must share batch/sequence shapes: "
                f"logits={tuple(logits.shape)}, labels={tuple(labels.shape)}, "
                f"loss_weights={tuple(loss_weights.shape)}"
            )

        shift_logits = logits[:, :-1, :].contiguous().float()
        shift_labels = labels[:, 1:].contiguous().to(device=logits.device)
        shift_weights = loss_weights[:, 1:].contiguous().to(
            device=logits.device,
            dtype=torch.float32,
        )
        if not bool(torch.isfinite(shift_weights).all().item()):
            raise ValueError("loss_weights must contain only finite values")
        if bool((shift_weights < 0).any().item()):
            raise ValueError("loss_weights must be non-negative")

        valid_mask = shift_labels.ne(-100)
        effective_weights = shift_weights * valid_mask.to(dtype=torch.float32)
        token_losses = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).reshape_as(shift_labels)
        local_loss_sum = (token_losses * effective_weights).sum()
        return cls._distributed_weighted_mean(
            local_loss_sum,
            effective_weights.sum(),
        )

    def _panovggt_enabled(self) -> bool:
        return self.panovggt_mlp is not None and bool(getattr(self.panovggt_mlp, "enabled", False))

    def _panovggt_feature_source(self) -> str:
        if self.panovggt_mlp is not None:
            return self.panovggt_mlp.feature_source
        return normalize_panovggt_feature_source(getattr(self.config, "panovggt_feature_source", "aggregator"))

    def _panovggt_injection_stage(self) -> str:
        if self.panovggt_mlp is not None:
            return self.panovggt_mlp.injection_stage
        return normalize_panovggt_injection_stage(getattr(self.config, "panovggt_injection_stage", "post_merger"))

    def _runtime_grid_matches_image_grid(self, grid_thw: torch.Tensor | None) -> bool:
        runtime_grid = self._pano_runtime_grid_thw
        if grid_thw is None or runtime_grid is None:
            return False
        if grid_thw is runtime_grid:
            return True
        if grid_thw.device == runtime_grid.device and tuple(grid_thw.shape) == tuple(runtime_grid.shape):
            return grid_thw.data_ptr() == runtime_grid.data_ptr()
        return False

    def _select_current_items(
        self,
        image_grid_thw: torch.Tensor,
        num_image_outputs: int | None = None,
    ) -> list[tuple[int, int]]:
        current_indices = self._current_image_flat_indices(image_grid_thw)
        if not current_indices:
            return []
        return [
            (batch_index, image_index)
            for batch_index, image_index in current_indices
            if 0 <= image_index < int(image_grid_thw.shape[0])
            and (num_image_outputs is None or image_index < num_image_outputs)
        ]

    def _panovggt_current_batch_inputs(
        self,
        *,
        panovggt_pixel_values: torch.Tensor,
        current_items: list[tuple[int, int]],
    ) -> torch.Tensor:
        if panovggt_pixel_values is None or panovggt_pixel_values.numel() == 0:
            raise AssertionError("PanoVGGT is enabled but panovggt_pixel_values is missing")
        if int(panovggt_pixel_values.shape[0]) != len(current_items):
            raise AssertionError(
                "PanoVGGT current-image batch size mismatch: "
                f"panovggt_batch={int(panovggt_pixel_values.shape[0])}, "
                f"current_items={len(current_items)}"
            )
        batch_indices = torch.tensor(
            [item[0] for item in current_items],
            device=panovggt_pixel_values.device,
            dtype=torch.long,
        )
        return panovggt_pixel_values.index_select(0, batch_indices)

    def _target_geometry_for_indices(
        self,
        target_indices: list[int],
    ) -> torch.Tensor | None:
        geometry = self._pano_runtime_image_geometry
        if geometry is None:
            return None
        return geometry[target_indices].detach().to(device="cpu", dtype=torch.float32)

    def _apply_post_merger_panovggt_residual(self, vision_output, image_grid_thw: torch.Tensor):
        if image_grid_thw is None or vision_output.pooler_output is None:
            return vision_output

        image_embeds = list(vision_output.pooler_output)
        current_items = self._select_current_items(
            image_grid_thw,
            num_image_outputs=len(image_embeds),
        )
        if not current_items:
            return vision_output

        panovggt_pixel_values = self._pano_runtime_panovggt_pixel_values
        panovggt_inputs = self._panovggt_current_batch_inputs(
            panovggt_pixel_values=panovggt_pixel_values,
            current_items=current_items,
        )
        target_indices = [item[1] for item in current_items]
        target_grid_thw = image_grid_thw[target_indices].detach().to(device="cpu", dtype=torch.long)
        target_lengths = [int(image_embeds[index].shape[0]) for index in target_indices]
        geometry = self._target_geometry_for_indices(target_indices)

        panovggt_model = self._ensure_panovggt_model(
            device=image_embeds[target_indices[0]].device,
            dtype=image_embeds[target_indices[0]].dtype,
        )
        if panovggt_model is None or self.panovggt_mlp is None:
            return vision_output
        deltas = self.panovggt_mlp(
            panovggt_inputs,
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

    def _apply_pre_merger_panovggt_residual(
        self,
        hidden_states: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        if image_grid_thw is None or hidden_states.numel() == 0:
            return hidden_states

        current_items = self._select_current_items(image_grid_thw)
        if not current_items:
            return hidden_states

        panovggt_pixel_values = self._pano_runtime_panovggt_pixel_values
        panovggt_inputs = self._panovggt_current_batch_inputs(
            panovggt_pixel_values=panovggt_pixel_values,
            current_items=current_items,
        )
        target_indices = [item[1] for item in current_items]
        target_grid_thw = image_grid_thw[target_indices].detach().to(device="cpu", dtype=torch.long)
        target_lengths = [
            int(num_frames * grid_h * grid_w)
            for num_frames, grid_h, grid_w in target_grid_thw.tolist()
        ]
        geometry = self._target_geometry_for_indices(target_indices)

        panovggt_model = self._ensure_panovggt_model(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if panovggt_model is None or self.panovggt_mlp is None:
            return hidden_states
        deltas = self.panovggt_mlp(
            panovggt_inputs,
            panovggt_model=panovggt_model,
            target_grid_thw=target_grid_thw,
            target_lengths=target_lengths,
            image_erp_geometry=geometry,
            output_device=hidden_states.device,
            output_dtype=hidden_states.dtype,
        )

        split_lengths = [
            int(num_frames * grid_h * grid_w)
            for num_frames, grid_h, grid_w in image_grid_thw.detach().to(device="cpu", dtype=torch.long).tolist()
        ]
        offsets = []
        offset = 0
        for length in split_lengths:
            offsets.append(offset)
            offset += length
        if offset != int(hidden_states.shape[0]):
            raise AssertionError(
                "Qwen pre-merger hidden length does not match image_grid_thw: "
                f"hidden={int(hidden_states.shape[0])}, grid_total={offset}"
            )

        hidden_states = hidden_states.clone()
        for image_index, delta in zip(target_indices, deltas):
            start = offsets[image_index]
            length = split_lengths[image_index]
            if tuple(delta.shape) != (length, int(hidden_states.shape[-1])):
                raise AssertionError(
                    "PanoVGGT pre-merger delta shape mismatch: "
                    f"delta={tuple(delta.shape)}, qwen=({length}, {int(hidden_states.shape[-1])})"
                )
            hidden_states[start:start + length] = hidden_states[start:start + length] + delta
        return hidden_states

    def _install_pano_merger_hook(self) -> None:
        if not self._panovggt_enabled() or self._panovggt_injection_stage() != "pre_merger":
            return
        visual = self.model.visual
        if hasattr(visual.merger, "_pano_origin_forward"):
            return

        visual.merger._pano_origin_forward = visual.merger.forward
        owner = self

        def merger_with_pano_pre_merger_residual(this, hidden_states):
            if (
                owner._panovggt_enabled()
                and owner._panovggt_injection_stage() == "pre_merger"
                and owner._pano_runtime_in_image_features
            ):
                hidden_states = owner._apply_pre_merger_panovggt_residual(
                    hidden_states=hidden_states,
                    image_grid_thw=owner._pano_runtime_visual_grid_thw,
                )
            return this._pano_origin_forward(hidden_states)

        visual.merger.forward = MethodType(merger_with_pano_pre_merger_residual, visual.merger)

    def _install_image_feature_hook(self) -> None:
        if not self._panovggt_enabled():
            return
        if hasattr(self.model, "_pano_origin_get_image_features"):
            return

        self.model._pano_origin_get_image_features = self.model.get_image_features
        owner = self

        def get_image_features_with_pano_residuals(this, pixel_values, image_grid_thw=None, **kwargs):
            is_image_visual_call = owner._runtime_grid_matches_image_grid(image_grid_thw)
            previous_in_image_features = owner._pano_runtime_in_image_features
            previous_visual_grid = owner._pano_runtime_visual_grid_thw
            owner._pano_runtime_in_image_features = is_image_visual_call
            owner._pano_runtime_visual_grid_thw = image_grid_thw if is_image_visual_call else None
            try:
                vision_output = this._pano_origin_get_image_features(
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    **kwargs,
                )
            finally:
                owner._pano_runtime_in_image_features = previous_in_image_features
                owner._pano_runtime_visual_grid_thw = previous_visual_grid
            if not is_image_visual_call:
                return vision_output
            if (
                owner._panovggt_enabled()
                and owner._panovggt_injection_stage() == "post_merger"
            ):
                vision_output = owner._apply_post_merger_panovggt_residual(
                    vision_output,
                    image_grid_thw,
                )
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
        self._pano_runtime_in_image_features = False
        self._pano_runtime_visual_grid_thw = None

    def _drop_incompatible_panovggt_mlp_pre_hook(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        del module, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        if self.panovggt_mlp is None:
            return
        key_prefix = f"{prefix}panovggt_mlp."
        checkpoint_keys = [key for key in state_dict.keys() if key.startswith(key_prefix)]
        if not checkpoint_keys:
            return
        target_state = self.panovggt_mlp.state_dict()
        incompatible = False
        unknown_keys = []
        for key in checkpoint_keys:
            local_key = key[len(key_prefix) :]
            target_tensor = target_state.get(local_key)
            source_tensor = state_dict[key]
            if target_tensor is None:
                unknown_keys.append(key)
                continue
            if tuple(source_tensor.shape) != tuple(target_tensor.shape):
                incompatible = True
                break
        if not incompatible:
            for key in unknown_keys:
                del state_dict[key]
            return
        for key in checkpoint_keys:
            del state_dict[key]
        self._panovggt_mlp_load_was_incompatible = True

    def _load_external_panovggt_weights(self) -> None:
        if not self._panovggt_enabled():
            return
        if self.panovggt is None:
            self.panovggt = build_panovggt_model_from_vendored_config(self._panovggt_feature_source())
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
            self.panovggt = build_panovggt_model_from_vendored_config(self._panovggt_feature_source())

        incompatible = self.panovggt.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys or incompatible.unexpected_keys:
            self.panovggt = None
            self._panovggt_weights_ready = False
            return False
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
            self.panovggt = build_panovggt_model_from_vendored_config(self._panovggt_feature_source())
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

    def _split_llm_image_hidden_states(
        self,
        *,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
        mm_token_type_ids: torch.Tensor | None,
        image_grid_thw: torch.Tensor,
        image_num_images: torch.Tensor,
        image_erp_geometry: torch.Tensor | None,
    ) -> list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]]:
        batch_size = int(hidden_states.shape[0])
        image_num_images = image_num_images.to(device="cpu", dtype=torch.long)
        if int(image_num_images.numel()) != batch_size:
            raise AssertionError(
                "image_num_images batch mismatch: "
                f"counts={int(image_num_images.numel())}, batch={batch_size}"
            )
        image_grid_cpu = image_grid_thw.detach().to(device="cpu", dtype=torch.long)
        if int(image_num_images.sum().item()) != int(image_grid_cpu.shape[0]):
            raise AssertionError(
                "image_num_images does not sum to image_grid_thw rows: "
                f"counts={image_num_images.tolist()}, grids={int(image_grid_cpu.shape[0])}"
            )

        geometry_cpu = None
        if image_erp_geometry is not None:
            geometry_cpu = image_erp_geometry.detach().to(device="cpu", dtype=torch.float32)
            if int(geometry_cpu.shape[0]) != int(image_grid_cpu.shape[0]):
                raise AssertionError(
                    "image_erp_geometry/image_grid_thw row mismatch: "
                    f"geometry={int(geometry_cpu.shape[0])}, grids={int(image_grid_cpu.shape[0])}"
                )

        merge_size = int(self.config.vision_config.spatial_merge_size)
        grid_offset = 0
        grouped_hidden_states = []
        for batch_index, num_images_tensor in enumerate(image_num_images):
            num_images = int(num_images_tensor.item())
            sample_grids = image_grid_cpu[grid_offset:grid_offset + num_images]
            sample_geometry = (
                None
                if geometry_cpu is None
                else geometry_cpu[grid_offset:grid_offset + num_images]
            )
            grid_offset += num_images

            if mm_token_type_ids is not None:
                visual_positions = torch.nonzero(
                    mm_token_type_ids[batch_index] == 1,
                    as_tuple=False,
                ).flatten()
            else:
                visual_positions = torch.nonzero(
                    input_ids[batch_index] == int(self.config.image_token_id),
                    as_tuple=False,
                ).flatten()

            image_lengths = []
            for grid in sample_grids.tolist():
                grid_t, grid_h, grid_w = [int(value) for value in grid]
                if grid_h % merge_size != 0 or grid_w % merge_size != 0:
                    raise AssertionError(
                        "Image grid is not divisible by Qwen spatial merge size: "
                        f"grid={grid}, merge_size={merge_size}"
                    )
                image_lengths.append(
                    grid_t * (grid_h // merge_size) * (grid_w // merge_size)
                )
            if int(visual_positions.numel()) != sum(image_lengths):
                raise AssertionError(
                    "LLM visual-token count does not match image grids: "
                    f"sample={batch_index}, tokens={int(visual_positions.numel())}, "
                    f"expected={sum(image_lengths)}, lengths={image_lengths}"
                )

            sample_visual_hidden = hidden_states[batch_index].index_select(
                0,
                visual_positions.to(device=hidden_states.device),
            )
            sample_groups = []
            token_offset = 0
            for image_index, (grid, image_length) in enumerate(
                zip(sample_grids, image_lengths)
            ):
                geometry = (
                    None if sample_geometry is None else sample_geometry[image_index]
                )
                sample_groups.append(
                    (
                        sample_visual_hidden[token_offset:token_offset + image_length],
                        grid,
                        geometry,
                    )
                )
                token_offset += image_length
            grouped_hidden_states.append(sample_groups)

        return grouped_hidden_states

    @staticmethod
    def spherical_yaw_fourier_pool(
        image_hidden_states: torch.Tensor,
        image_grid_thw: torch.Tensor,
        image_erp_geometry: torch.Tensor | None,
        spatial_merge_size: int,
    ) -> torch.Tensor:
        grid_t, grid_h, grid_w = [
            int(value)
            for value in image_grid_thw.detach().to(device="cpu", dtype=torch.long).tolist()
        ]
        if grid_h % spatial_merge_size != 0 or grid_w % spatial_merge_size != 0:
            raise AssertionError(
                "Image grid is not divisible by the spatial merge size: "
                f"grid={(grid_t, grid_h, grid_w)}, merge={spatial_merge_size}"
            )
        pooled_h = grid_h // spatial_merge_size
        pooled_w = grid_w // spatial_merge_size
        expected_tokens = grid_t * pooled_h * pooled_w
        if int(image_hidden_states.shape[0]) != expected_tokens:
            raise AssertionError(
                "Fourier pooling token/grid mismatch: "
                f"tokens={int(image_hidden_states.shape[0])}, expected={expected_tokens}"
            )

        hidden = image_hidden_states.float().reshape(
            grid_t,
            pooled_h,
            pooled_w,
            int(image_hidden_states.shape[-1]),
        )
        device = hidden.device
        if image_erp_geometry is None:
            vertical_fov = math.pi
            center_latitude = 0.0
        else:
            geometry_values = image_erp_geometry.detach().to(
                device="cpu",
                dtype=torch.float32,
            ).tolist()
            vertical_fov = float(geometry_values[0])
            center_latitude = float(geometry_values[1])

        latitude = (
            center_latitude
            - 0.5 * vertical_fov
            + (torch.arange(pooled_h, device=device, dtype=torch.float32) + 0.5)
            * (vertical_fov / pooled_h)
        )
        longitude = (
            -math.pi
            + (torch.arange(pooled_w, device=device, dtype=torch.float32) + 0.5)
            * (2.0 * math.pi / pooled_w)
        )
        area_weight = latitude.cos().clamp_min(0.0).view(1, pooled_h, 1, 1)
        cosine_basis = longitude.cos().view(1, 1, pooled_w, 1)
        sine_basis = longitude.sin().view(1, 1, pooled_w, 1)
        denominator = area_weight.sum() * float(grid_t * pooled_w)
        denominator = denominator.clamp_min(torch.finfo(torch.float32).eps)

        zero_order = (hidden * area_weight).sum(dim=(0, 1, 2)) / denominator
        cosine_order = (
            2.0 * (hidden * area_weight * cosine_basis).sum(dim=(0, 1, 2))
            / denominator
        )
        sine_order = (
            2.0 * (hidden * area_weight * sine_basis).sum(dim=(0, 1, 2))
            / denominator
        )
        return torch.cat((zero_order, cosine_order, sine_order), dim=-1)

    @staticmethod
    def _first_supervised_token_index(sample_labels: torch.Tensor) -> int:
        supervised_positions = torch.nonzero(
            sample_labels != -100,
            as_tuple=False,
        ).flatten()
        if supervised_positions.numel() == 0:
            return -1
        return int(supervised_positions[0].item())

    def _compute_pbo_loss(
        self,
        *,
        hidden_states: torch.Tensor,
        image_groups: list[list[tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]]],
        labels: torch.Tensor,
        image_current_index: torch.Tensor,
        pbo_action_labels: torch.Tensor | None,
        pbo_valid_mask: torch.Tensor | None,
        pbo_start_image_index: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.pbo_head is None:
            return hidden_states.sum() * 0.0

        batch_size, _, hidden_size = hidden_states.shape
        pbo_input_vector_count = int(
            getattr(self.config, "pbo_input_vector_count", PBO_INPUT_VECTOR_COUNT)
        )
        if pbo_input_vector_count not in PBO_SUPPORTED_INPUT_VECTOR_COUNTS:
            raise AssertionError(
                "Unsupported PBO input vector count: "
                f"{pbo_input_vector_count}"
            )
        head_dtype = next(self.pbo_head.parameters()).dtype
        features = torch.zeros(
            (batch_size, hidden_size * pbo_input_vector_count),
            device=hidden_states.device,
            dtype=head_dtype,
        )
        if pbo_valid_mask is None:
            valid_mask = torch.zeros(batch_size, device=hidden_states.device, dtype=torch.bool)
        else:
            valid_mask = pbo_valid_mask.to(device=hidden_states.device, dtype=torch.bool)

        current_indices = image_current_index.to(device="cpu", dtype=torch.long)
        start_indices = (
            torch.full((batch_size,), -1, dtype=torch.long)
            if pbo_start_image_index is None
            else pbo_start_image_index.to(device="cpu", dtype=torch.long)
        )
        merge_size = int(self.config.vision_config.spatial_merge_size)
        for batch_index in torch.nonzero(valid_mask, as_tuple=False).flatten().tolist():
            start_index = int(start_indices[batch_index].item())
            current_index = int(current_indices[batch_index].item())
            sample_groups = image_groups[batch_index]
            if not (0 <= start_index < len(sample_groups)):
                raise AssertionError(
                    f"PBO start image index is invalid: sample={batch_index}, index={start_index}, "
                    f"num_images={len(sample_groups)}"
                )
            if not (0 <= current_index < len(sample_groups)):
                raise AssertionError(
                    f"PBO current image index is invalid: sample={batch_index}, index={current_index}, "
                    f"num_images={len(sample_groups)}"
                )

            previous_hidden, previous_grid, previous_geometry = sample_groups[start_index]
            current_hidden, current_grid, current_geometry = sample_groups[current_index]
            previous_fourier = self.spherical_yaw_fourier_pool(
                previous_hidden,
                previous_grid,
                previous_geometry,
                merge_size,
            )
            current_fourier = self.spherical_yaw_fourier_pool(
                current_hidden,
                current_grid,
                current_geometry,
                merge_size,
            )
            feature_vectors = [previous_fourier, current_fourier]
            if pbo_input_vector_count == 7:
                context_index = (
                    self._first_supervised_token_index(labels[batch_index]) - 1
                )
                if context_index < 0:
                    raise AssertionError(
                        f"PBO sample {batch_index} has no prompt token before "
                        "the assistant response"
                    )
                feature_vectors.insert(
                    0,
                    hidden_states[batch_index, context_index].float(),
                )
            pbo_feature = torch.cat(feature_vectors, dim=-1)
            features[batch_index] = pbo_feature.to(dtype=head_dtype)

        logits = self.pbo_head(features).reshape(
            batch_size,
            PBO_ACTION_HORIZON,
            PBO_NUM_ACTIONS,
        )
        local_loss_sum = logits.sum() * 0.0
        local_target_count = 0
        if bool(valid_mask.any().item()):
            if pbo_action_labels is None:
                raise AssertionError(
                    "PBO has valid samples but pbo_action_labels is missing"
                )
            targets = pbo_action_labels.to(device=logits.device, dtype=torch.long)
            valid_targets = targets[valid_mask].reshape(-1)
            local_target_count = int((valid_targets != -100).sum().item())
            local_loss_sum = F.cross_entropy(
                logits[valid_mask].reshape(-1, PBO_NUM_ACTIONS).float(),
                valid_targets,
                reduction="sum",
            )
        return self._distributed_masked_mean(
            local_loss_sum,
            local_target_count,
        )

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_values=None,
        inputs_embeds: torch.FloatTensor | None = None,
        labels: torch.LongTensor | None = None,
        loss_weights: torch.FloatTensor | None = None,
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
        pbo_action_labels: torch.LongTensor | None = None,
        pbo_valid_mask: torch.BoolTensor | None = None,
        pbo_start_image_index: torch.LongTensor | None = None,
        **kwargs,
    ):
        if self._panovggt_enabled():
            self._set_runtime_pano_context(
                image_grid_thw=image_grid_thw,
                image_num_images=image_num_images,
                image_current_index=image_current_index,
                image_erp_geometry=image_erp_geometry,
                panovggt_pixel_values=panovggt_pixel_values,
            )
        auxiliary_enabled = bool(
            labels is not None
            and self._pbo_enabled()
        )
        weighted_lm_enabled = labels is not None and loss_weights is not None
        self._capture_auxiliary_hidden = auxiliary_enabled
        self._auxiliary_last_hidden = None
        try:
            outputs = super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                labels=None if weighted_lm_enabled else labels,
                pixel_values=pixel_values,
                pixel_values_videos=pixel_values_videos,
                image_grid_thw=image_grid_thw,
                video_grid_thw=video_grid_thw,
                mm_token_type_ids=mm_token_type_ids,
                logits_to_keep=logits_to_keep,
                **kwargs,
            )
            if weighted_lm_enabled:
                outputs.loss = self._compute_weighted_causal_lm_loss(
                    outputs.logits,
                    labels,
                    loss_weights,
                )
            if not auxiliary_enabled:
                return outputs
            hidden_states = self._auxiliary_last_hidden
            if hidden_states is None:
                raise AssertionError("Failed to capture the final LLM hidden states")
            if input_ids is None or labels is None:
                raise AssertionError("PBO loss requires input_ids and labels")

            batch_size = int(hidden_states.shape[0])
            if image_num_images is None:
                image_num_images = torch.zeros(
                    batch_size,
                    device=hidden_states.device,
                    dtype=torch.long,
                )
            if image_current_index is None:
                image_current_index = image_num_images - 1
            if image_grid_thw is not None:
                image_groups = self._split_llm_image_hidden_states(
                    hidden_states=hidden_states,
                    input_ids=input_ids,
                    mm_token_type_ids=mm_token_type_ids,
                    image_grid_thw=image_grid_thw,
                    image_num_images=image_num_images,
                    image_erp_geometry=image_erp_geometry,
                )
            else:
                image_groups = [[] for _ in range(batch_size)]

            auxiliary_loss = hidden_states.sum() * 0.0
            if self._pbo_enabled():
                pbo_loss = self._compute_pbo_loss(
                    hidden_states=hidden_states,
                    image_groups=image_groups,
                    labels=labels,
                    image_current_index=image_current_index,
                    pbo_action_labels=pbo_action_labels,
                    pbo_valid_mask=pbo_valid_mask,
                    pbo_start_image_index=pbo_start_image_index,
                )
                auxiliary_loss = auxiliary_loss + float(
                    getattr(self.config, "pbo_loss_weight", 0.1)
                ) * pbo_loss

            outputs.loss = auxiliary_loss if outputs.loss is None else outputs.loss + auxiliary_loss
            return outputs
        finally:
            self._capture_auxiliary_hidden = False
            self._auxiliary_last_hidden = None
            self._clear_runtime_pano_context()

    def _load_missing_pano_parameters_post_hook(self, module, incompatible_keys) -> None:
        del module
        missing_keys = set(incompatible_keys.missing_keys)

        panovggt_module = self.panovggt_mlp
        if (
            panovggt_module is not None
            and panovggt_module.enabled
            and (
                self._panovggt_mlp_load_was_incompatible
                or any(key.startswith("panovggt_mlp.") for key in missing_keys)
            )
        ):
            panovggt_module.reset_parameters()
            self._panovggt_mlp_load_was_incompatible = False
        if panovggt_module is not None and panovggt_module.enabled:
            if any(key.startswith("panovggt.") for key in missing_keys):
                self._load_external_panovggt_weights()
            else:
                self._mark_panovggt_weights_ready()

    def _reset_panovggt_parameters_after_pretrained_load(self) -> None:
        if self._panovggt_enabled() and self.panovggt_mlp is not None:
            self.panovggt_mlp.reset_parameters()

    @classmethod
    def _checkpoint_has_panovggt_weights(cls, pretrained_model_name_or_path) -> bool | None:
        return cls._checkpoint_has_any_weights(pretrained_model_name_or_path, cls.PANOVGGT_STATE_KEYS)

    @classmethod
    def _local_pbo_head_input_width(cls, pretrained_model_name_or_path) -> int | None:
        checkpoint_dir = Path(str(pretrained_model_name_or_path))
        if not checkpoint_dir.is_dir():
            return None

        tensor_key = "pbo_head.0.weight"
        safetensors_path = checkpoint_dir / "model.safetensors"
        if not safetensors_path.exists():
            index_path = checkpoint_dir / "model.safetensors.index.json"
            if not index_path.exists():
                return None
            try:
                weight_map = json.loads(index_path.read_text())["weight_map"]
                shard_name = weight_map.get(tensor_key)
            except (KeyError, OSError, TypeError, ValueError):
                return None
            if not shard_name:
                return None
            safetensors_path = checkpoint_dir / shard_name

        try:
            from safetensors import safe_open

            with safe_open(
                str(safetensors_path),
                framework="pt",
                device="cpu",
            ) as handle:
                if tensor_key not in handle.keys():
                    return None
                shape = tuple(handle.get_slice(tensor_key).get_shape())
        except Exception:
            return None
        if len(shape) != 1:
            return None
        return int(shape[0])

    @classmethod
    def _configure_pbo_input_width_from_checkpoint(
        cls,
        pretrained_model_name_or_path,
        config,
    ) -> int | None:
        input_width = cls._local_pbo_head_input_width(
            pretrained_model_name_or_path
        )
        if input_width is None:
            return None
        hidden_size = int(config.text_config.hidden_size)
        if input_width % hidden_size != 0:
            raise ValueError(
                "PBO checkpoint input width is not divisible by the text hidden "
                f"size: width={input_width}, hidden_size={hidden_size}"
            )
        vector_count = input_width // hidden_size
        if vector_count not in PBO_SUPPORTED_INPUT_VECTOR_COUNTS:
            raise ValueError(
                "Unsupported PBO checkpoint input width: "
                f"{vector_count} vectors ({input_width} features)"
            )
        config.pbo_input_vector_count = vector_count
        return vector_count

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
        checkpoint_pbo_width = cls._local_pbo_head_input_width(
            pretrained_model_name_or_path
        )
        if checkpoint_pbo_width is not None:
            config = kwargs.get("config")
            if config is None:
                config_kwargs = {
                    key: kwargs[key]
                    for key in (
                        "cache_dir",
                        "force_download",
                        "local_files_only",
                        "revision",
                        "token",
                        "subfolder",
                    )
                    if key in kwargs
                }
                config = cls.config_class.from_pretrained(
                    pretrained_model_name_or_path,
                    **config_kwargs,
                )
                kwargs["config"] = config
            cls._configure_pbo_input_width_from_checkpoint(
                pretrained_model_name_or_path,
                config,
            )
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
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
