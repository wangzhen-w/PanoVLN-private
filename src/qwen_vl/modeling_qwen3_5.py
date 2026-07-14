import json
import math
import sys
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration


DA2_FEATURE_SPECS = {
    "decoder_stage0": ((0, 512),),
    "decoder_stage1": ((1, 256),),
    "decoder_stage2": ((2, 128),),
    "decoder_multiscale": ((0, 512), (1, 256), (2, 128)),
}
DA2_FEATURE_SOURCES = set(DA2_FEATURE_SPECS)
DA2_INJECTION_STAGES = {"post_merger", "pre_merger"}
DA2_MLP_HIDDEN_SIZE = 4096
BUNDLED_DA2_SOURCE = "bundled"
DEFAULT_DA2_SOURCE_PATH = Path(__file__).resolve().parents[1]
DEFAULT_DA2_MODEL_PATH = Path("/workspace/data1/model/DA-2")
ERP_ANGLE_EPS = 1e-6
ERP_FOURIER_NUM_HARMONICS = 16
ERP_FOURIER_ENCODING_DIM = ERP_FOURIER_NUM_HARMONICS * 4


def validate_erp_angles(vertical_fov: float, center_latitude: float) -> tuple[float, float]:
    if not math.isfinite(vertical_fov) or not (0.0 < vertical_fov <= math.pi + ERP_ANGLE_EPS):
        raise AssertionError(f"Invalid ERP vertical_fov={vertical_fov}")
    half_pi = 0.5 * math.pi
    if not math.isfinite(center_latitude) or abs(center_latitude) > half_pi + ERP_ANGLE_EPS:
        raise AssertionError(f"Invalid ERP center_latitude={center_latitude}")
    return min(vertical_fov, math.pi), max(-half_pi, min(half_pi, center_latitude))


def reorder_erp_features_to_qwen_patch_order(
    features: torch.Tensor,
    *,
    num_frames: int,
    height: int,
    width: int,
    spatial_merge_size: int,
) -> torch.Tensor:
    num_frames = int(num_frames)
    height = int(height)
    width = int(width)
    spatial_merge_size = int(spatial_merge_size)
    if num_frames <= 0:
        raise AssertionError(f"ERP Fourier num_frames must be positive, got {num_frames}")
    if spatial_merge_size <= 0:
        raise AssertionError(
            f"ERP Fourier spatial_merge_size must be positive, got {spatial_merge_size}"
        )
    if height % spatial_merge_size != 0 or width % spatial_merge_size != 0:
        raise AssertionError(
            "ERP Fourier grid is not divisible by Qwen spatial_merge_size: "
            f"grid=({height}, {width}), spatial_merge_size={spatial_merge_size}"
        )
    if features.ndim != 2 or int(features.shape[0]) != height * width:
        raise AssertionError(
            "ERP Fourier row-major feature shape mismatch before Qwen order conversion: "
            f"features={tuple(features.shape)}, grid=({height}, {width})"
        )

    ordered = features.reshape(
        height // spatial_merge_size,
        spatial_merge_size,
        width // spatial_merge_size,
        spatial_merge_size,
        features.shape[-1],
    )
    ordered = ordered.permute(0, 2, 1, 3, 4).contiguous().reshape(height * width, -1)
    if num_frames > 1:
        ordered = ordered.repeat(num_frames, 1)
    return ordered


def build_erp_fourier_features(
    *,
    num_frames: int,
    height: int,
    width: int,
    spatial_merge_size: int,
    device: torch.device,
    dtype: torch.dtype,
    vertical_fov: float,
    center_latitude: float,
) -> torch.Tensor:
    vertical_fov, center_latitude = validate_erp_angles(vertical_fov, center_latitude)
    ys = torch.arange(height, device=device, dtype=torch.float32) + 0.5
    xs = torch.arange(width, device=device, dtype=torch.float32) + 0.5

    # image_erp_geometry stores its vertical center in image-row coordinates
    # (positive down). Convert it here to the physical positive-up pitch used
    # by the ERP prompt while keeping image-row sampling coordinates unchanged.
    pitch = (0.5 - ys[:, None] / float(height)) * vertical_fov - center_latitude
    yaw = (xs[None, :] / float(width) - 0.5) * (2.0 * math.pi)
    pitch = pitch.expand(height, width)
    yaw = yaw.expand(height, width)

    scales = torch.arange(
        1,
        ERP_FOURIER_NUM_HARMONICS + 1,
        device=device,
        dtype=torch.float32,
    )
    yaw_phase = yaw.reshape(height * width, 1) * scales
    pitch_phase = pitch.reshape(height * width, 1) * scales
    yaw_features = torch.stack([yaw_phase.sin(), yaw_phase.cos()], dim=-1).reshape(height * width, -1)
    pitch_features = torch.stack([pitch_phase.sin(), pitch_phase.cos()], dim=-1).reshape(height * width, -1)
    embeddings = torch.cat([yaw_features, pitch_features], dim=-1)
    embeddings = reorder_erp_features_to_qwen_patch_order(
        embeddings,
        num_frames=num_frames,
        height=height,
        width=width,
        spatial_merge_size=spatial_merge_size,
    )
    return embeddings.to(dtype=dtype)


def infer_erp_vertical_fov_radians(height: int, width: int) -> float:
    if height <= 0 or width <= 0:
        return 0.0
    return min(math.pi, 2.0 * math.pi * float(height) / float(width))


def ensure_erp_fourier_linear_config(vision_config) -> None:
    defaults = {
        "erp_fourier_linear_enabled": False,
        "erp_fourier_linear_alpha_value": 0.01,
        "erp_fourier_linear_apply_to_current_only": True,
    }
    for field_name, default_value in defaults.items():
        if not hasattr(vision_config, field_name):
            setattr(vision_config, field_name, default_value)


class ERPFourierLinearAdapter(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        ensure_erp_fourier_linear_config(config)

        self.enabled = bool(getattr(config, "erp_fourier_linear_enabled", False))
        self.output_hidden_size = int(config.hidden_size)
        self.alpha_init = float(getattr(config, "erp_fourier_linear_alpha_value", 0.01))
        self.apply_to_current_only = bool(
            getattr(config, "erp_fourier_linear_apply_to_current_only", True)
        )
        self.spatial_merge_size = int(getattr(config, "spatial_merge_size", 2))
        if self.enabled and self.alpha_init <= 0.0:
            raise ValueError(
                "erp_fourier_linear_alpha_value must be positive when ERP Fourier linear is enabled; "
                "use erp_fourier_linear_enabled=false to disable the adapter."
            )

        self.proj = nn.Linear(ERP_FOURIER_ENCODING_DIM, self.output_hidden_size, bias=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init_std = self.alpha_init * math.sqrt(2.0 / float(ERP_FOURIER_ENCODING_DIM))
        nn.init.normal_(self.proj.weight, mean=0.0, std=init_std)

    def apply_to_patch_tokens(
        self,
        patch_tokens: torch.Tensor,
        grid_thw: torch.Tensor | None,
        image_apply_mask: torch.Tensor | None = None,
        image_erp_geometry: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.enabled or grid_thw is None or grid_thw.numel() == 0:
            return patch_tokens
        if grid_thw.ndim != 2 or int(grid_thw.shape[-1]) != 3:
            raise AssertionError(f"Expected image_grid_thw shape [N, 3], got {tuple(grid_thw.shape)}")

        param = self.proj.weight
        param_device = param.device
        param_dtype = param.dtype
        default_center_latitude = 0.0

        num_images = int(grid_thw.shape[0])
        if image_apply_mask is not None:
            if image_apply_mask.ndim != 1 or int(image_apply_mask.shape[0]) != num_images:
                raise AssertionError(
                    f"Expected image_apply_mask to have shape [{num_images}], got {tuple(image_apply_mask.shape)}"
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
                "Patch token count does not match image_grid_thw for ERP Fourier linear: "
                f"tokens={int(patch_tokens.shape[0])}, expected={sum(split_sizes)}"
            )

        token_splits = list(patch_tokens.split(split_sizes, dim=0))
        for image_index, (tokens, (num_frames, height, width)) in enumerate(zip(token_splits, grid_thw.tolist())):
            if not apply_mask_list[image_index] or int(height) <= 0 or int(width) <= 0:
                continue

            if geometry_list is not None:
                vertical_fov = float(geometry_list[image_index][0])
                center_latitude = float(geometry_list[image_index][1])
            else:
                vertical_fov = infer_erp_vertical_fov_radians(int(height), int(width))
                center_latitude = default_center_latitude

            fourier_features = build_erp_fourier_features(
                num_frames=int(num_frames),
                height=int(height),
                width=int(width),
                spatial_merge_size=self.spatial_merge_size,
                device=param_device,
                dtype=param_dtype,
                vertical_fov=vertical_fov,
                center_latitude=center_latitude,
            )
            if int(fourier_features.shape[0]) != int(tokens.shape[0]):
                raise AssertionError(
                    "ERP Fourier linear feature count mismatch: "
                    f"features={int(fourier_features.shape[0])}, tokens={int(tokens.shape[0])}"
                )
            delta = self.proj(fourier_features)
            token_splits[image_index] = tokens + delta.to(device=tokens.device, dtype=tokens.dtype)

        return torch.cat(token_splits, dim=0)


def ensure_da2_config(config) -> None:
    vision_config = config.vision_config
    text_config = getattr(config, "text_config", None)
    text_hidden_size = getattr(text_config, "hidden_size", getattr(vision_config, "out_hidden_size", 3584))
    defaults = {
        "da2_enabled": False,
        "da2_source_path": BUNDLED_DA2_SOURCE,
        "da2_model_path": str(DEFAULT_DA2_MODEL_PATH),
        "da2_alpha_value": 0.1,
        "da2_feature_source": "decoder_multiscale",
        "da2_injection_stage": "post_merger",
        "da2_sampling_mode": "grouping",
        "da2_force_fp32": False,
        "da2_output_dim": int(getattr(vision_config, "out_hidden_size", text_hidden_size)),
    }
    for field_name, default_value in defaults.items():
        if not hasattr(config, field_name):
            setattr(config, field_name, default_value)


def _is_da2_source_tree(source_dir: Path) -> bool:
    package_dir = source_dir / "da2"
    return (
        source_dir.is_dir()
        and (package_dir / "model" / "spherevit.py").is_file()
        and (package_dir / "configs" / "infer.json").is_file()
    )


def _resolve_da2_source_candidate(path: Path) -> Path | None:
    path = path.expanduser().resolve()
    for candidate in (path, path / "src"):
        if _is_da2_source_tree(candidate):
            return candidate
    return None


def resolve_da2_source_path(source_path: str | Path | None = None) -> Path:
    raw_source_path = "" if source_path is None else str(source_path).strip()
    use_bundled_source = raw_source_path.lower() in {
        "",
        "none",
        "null",
        BUNDLED_DA2_SOURCE,
    }
    requested_path = (
        DEFAULT_DA2_SOURCE_PATH
        if use_bundled_source
        else Path(raw_source_path).expanduser()
    )
    requested_source = _resolve_da2_source_candidate(requested_path)
    if requested_source is not None:
        return requested_source

    bundled_source = _resolve_da2_source_candidate(DEFAULT_DA2_SOURCE_PATH)
    if not use_bundled_source and bundled_source is not None:
        return bundled_source

    raise FileNotFoundError(
        "DA2 is enabled but no valid source tree was found. "
        f"requested={requested_path.resolve()}, bundled={DEFAULT_DA2_SOURCE_PATH.resolve()}"
    )


def _ensure_da2_source_available(source_path: str | Path | None) -> Path:
    source_dir = resolve_da2_source_path(source_path)

    source_str = str(source_dir)
    while source_str in sys.path:
        sys.path.remove(source_str)
    sys.path.insert(0, source_str)

    package_dir = (source_dir / "da2").resolve()
    loaded_package = sys.modules.get("da2")
    if loaded_package is not None:
        loaded_file = getattr(loaded_package, "__file__", None)
        loaded_paths = getattr(loaded_package, "__path__", ())
        loaded_from_expected_source = (
            loaded_file is not None
            and package_dir in Path(loaded_file).resolve().parents
        ) or any(Path(path).resolve() == package_dir for path in loaded_paths)
        if not loaded_from_expected_source:
            for module_name in tuple(sys.modules):
                if module_name == "da2" or module_name.startswith("da2."):
                    del sys.modules[module_name]
    return source_dir


def normalize_da2_feature_source(feature_source: str) -> str:
    feature_source = str(feature_source).lower()
    if feature_source not in DA2_FEATURE_SOURCES:
        raise ValueError(
            "da2_feature_source must be one of "
            f"{sorted(DA2_FEATURE_SOURCES)}, "
            f"got {feature_source!r}"
        )
    return feature_source


def normalize_da2_injection_stage(injection_stage: str) -> str:
    injection_stage = str(injection_stage).lower()
    if injection_stage not in DA2_INJECTION_STAGES:
        raise ValueError(
            "da2_injection_stage must be 'post_merger' or 'pre_merger', "
            f"got {injection_stage!r}"
        )
    return injection_stage


def build_da2_model(
    source_path: str,
    model_path: str | None,
    *,
    load_pretrained_weights: bool = True,
):
    source_dir = _ensure_da2_source_available(source_path)
    try:
        from da2.model import SphereViT
    except Exception as exc:
        raise ImportError(
            "DA2 is enabled but could not be imported from "
            f"{source_dir}: {type(exc).__name__}: {exc}"
        ) from exc

    architecture_path = source_dir / "da2" / "configs" / "infer.json"
    architecture_config = json.loads(architecture_path.read_text(encoding="utf-8"))
    model = SphereViT(architecture_config)

    if not load_pretrained_weights:
        freeze_da2_model(model)
        return model

    if not model_path:
        raise FileNotFoundError(
            "DA2 pretrained weights are required when the VLN checkpoint does not "
            "contain da2.* tensors"
        )
    model_dir = Path(model_path).expanduser().resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(
            "DA2 is enabled but its pretrained model directory is missing: "
            f"{model_dir}"
        )
    model_file = model_dir / "model.safetensors"
    if not model_file.is_file():
        raise FileNotFoundError(f"DA2 model.safetensors is missing from {model_dir}")

    from safetensors.torch import load_file

    state_dict = load_file(str(model_file), device="cpu")
    try:
        model.load_state_dict(state_dict, strict=True, assign=True)
    except TypeError:
        model.load_state_dict(state_dict, strict=True)
    freeze_da2_model(model)
    return model


def freeze_da2_model(model: nn.Module | None) -> None:
    if model is None:
        return
    model.eval()
    for param in model.parameters():
        param.requires_grad = False


class DA2GeometryMLP(nn.Module):
    def __init__(self, config) -> None:
        super().__init__()
        ensure_da2_config(config)

        self.enabled = bool(getattr(config, "da2_enabled", False))
        self.feature_source = normalize_da2_feature_source(
            getattr(config, "da2_feature_source", "decoder_multiscale")
        )
        self.injection_stage = normalize_da2_injection_stage(
            getattr(config, "da2_injection_stage", "post_merger")
        )
        self.feature_spec = DA2_FEATURE_SPECS[self.feature_source]
        self.context_dim = sum(channels for _, channels in self.feature_spec)
        if self.injection_stage == "pre_merger":
            self.output_dim = int(getattr(config.vision_config, "hidden_size"))
        else:
            self.output_dim = int(getattr(config, "da2_output_dim", config.text_config.hidden_size))
        self.hidden_dim = DA2_MLP_HIDDEN_SIZE
        self.sampling_mode = str(getattr(config, "da2_sampling_mode", "grouping")).lower()
        if self.sampling_mode not in {"singlepoint", "grouping"}:
            raise ValueError(
                "da2_sampling_mode must be 'singlepoint' or 'grouping', "
                f"got {self.sampling_mode!r}"
            )
        self.alpha_init = float(getattr(config, "da2_alpha_value", 0.1))
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

    def _sample_feature_pyramid(
        self,
        source_grids: list[torch.Tensor],
        target_h: int,
        target_w: int,
        geometry: torch.Tensor | None,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        sample_h = target_h
        sample_w = target_w
        if self.sampling_mode == "grouping":
            sample_h *= self.spatial_merge_size
            sample_w *= self.spatial_merge_size

        sampled_levels = [
            self._sample_geometry_grid(
                source_grid,
                target_h=sample_h,
                target_w=sample_w,
                geometry=geometry,
            )
            for source_grid in source_grids
        ]
        sampled = torch.cat(sampled_levels, dim=-1)
        sampled = self.input_norm(sampled.to(dtype=dtype))
        if self.sampling_mode == "singlepoint":
            return sampled

        sampled = sampled.reshape(
            target_h,
            self.spatial_merge_size,
            target_w,
            self.spatial_merge_size,
            self.context_dim,
        )
        sampled = sampled.permute(0, 2, 1, 3, 4).contiguous()
        return sampled.reshape(target_h * target_w, self.mlp_input_dim)

    def _to_qwen_pre_merger_order(
        self,
        tokens: torch.Tensor,
        grid_h: int,
        grid_w: int,
    ) -> torch.Tensor:
        if grid_h % self.spatial_merge_size != 0 or grid_w % self.spatial_merge_size != 0:
            raise AssertionError(
                "DA2 pre-merger target grid is not divisible by Qwen spatial_merge_size: "
                f"grid=({grid_h}, {grid_w}), spatial_merge_size={self.spatial_merge_size}"
            )
        if int(tokens.shape[0]) != int(grid_h * grid_w):
            raise AssertionError(
                "DA2 pre-merger token length mismatch before Qwen order conversion: "
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

    def forward(
        self,
        da2_pixel_values: torch.Tensor,
        da2_model: nn.Module,
        target_grid_thw: torch.Tensor,
        target_lengths: list[int],
        image_erp_geometry: torch.Tensor | None,
        output_device: torch.device,
        output_dtype: torch.dtype,
    ) -> list[torch.Tensor]:
        if (
            not self.enabled
            or da2_model is None
            or da2_pixel_values is None
            or da2_pixel_values.numel() == 0
        ):
            return []

        if da2_pixel_values.ndim != 4:
            raise AssertionError(
                "Expected da2_pixel_values shape [B, 3, H, W], "
                f"got {tuple(da2_pixel_values.shape)}"
            )

        encoder = da2_model
        encoder_param = next(encoder.parameters())
        param = next(self.mlp.parameters())
        images = da2_pixel_values.to(device=encoder_param.device, dtype=encoder_param.dtype)
        if int(images.shape[1]) != 3:
            raise AssertionError(f"DA2 expects RGB inputs, got {tuple(images.shape)}")

        batch_size, _, image_h, image_w = [int(value) for value in images.shape]
        if image_w != 2 * image_h:
            raise AssertionError(
                "DA2 expects a full 2:1 ERP panorama: "
                f"shape=({image_h}, {image_w})"
            )

        autocast_enabled = (
            images.device.type == "cuda"
            and encoder_param.dtype in (torch.bfloat16, torch.float16)
        )
        with torch.no_grad(), torch.autocast(
            device_type=images.device.type,
            dtype=encoder_param.dtype,
            enabled=autocast_enabled,
        ):
            feature_pyramid = encoder.extract_decoder_features(images)

        if not isinstance(feature_pyramid, (tuple, list)) or len(feature_pyramid) != 3:
            raise AssertionError(
                "DA2 must return its three-stage spherical decoder pyramid, "
                f"got {type(feature_pyramid)!r} with length "
                f"{len(feature_pyramid) if isinstance(feature_pyramid, (tuple, list)) else 'n/a'}"
            )
        source_grids = []
        for stage_index, expected_channels in self.feature_spec:
            source_grid = feature_pyramid[stage_index]
            if source_grid.ndim != 4:
                raise AssertionError(
                    f"Expected DA2 decoder stage {stage_index} as [B, C, H, W], "
                    f"got {tuple(source_grid.shape)}"
                )
            if int(source_grid.shape[0]) != batch_size:
                raise AssertionError(
                    "DA2 feature batch mismatch: "
                    f"expected {batch_size}, got {int(source_grid.shape[0])} at stage {stage_index}"
                )
            if int(source_grid.shape[1]) != expected_channels:
                raise AssertionError(
                    "DA2 decoder channel mismatch: "
                    f"stage={stage_index}, expected={expected_channels}, "
                    f"got={int(source_grid.shape[1])}"
                )
            source_grids.append(source_grid)

        deltas = []
        geometry = image_erp_geometry
        if geometry is not None:
            geometry = geometry.to(device=param.device, dtype=torch.float32)
        if batch_size != len(target_lengths) or len(target_lengths) != int(target_grid_thw.shape[0]):
            raise AssertionError(
                "DA2 batch size mismatch: "
                f"source={batch_size}, target_lengths={len(target_lengths)}, "
                f"target_grid_thw={int(target_grid_thw.shape[0])}"
            )
        if geometry is not None and int(geometry.shape[0]) != len(target_lengths):
            raise AssertionError(
                "DA2 geometry batch size mismatch: "
                f"geometry={int(geometry.shape[0])}, target_lengths={len(target_lengths)}"
            )
        for sample_index, (grid_thw, target_len) in enumerate(zip(target_grid_thw.tolist(), target_lengths)):
            num_frames, grid_h, grid_w = [int(value) for value in grid_thw]
            if num_frames != 1:
                raise AssertionError(
                    "DA2 geometry fusion expects a single current panorama per Qwen image, "
                    f"got image_grid_thw={grid_thw}"
                )
            if grid_h % self.spatial_merge_size != 0 or grid_w % self.spatial_merge_size != 0:
                raise AssertionError(
                    "DA2 target grid is not divisible by Qwen spatial_merge_size: "
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
                    "Cannot map DA2 geometry to the Qwen 2D visual-token grid: "
                    f"image_grid_thw={grid_thw}, spatial_merge_size={self.spatial_merge_size}, "
                    f"expected_len={expected_len}, qwen_len={int(target_len)}"
                )

            sample_geometry = None if geometry is None else geometry[sample_index]
            geo = self._sample_feature_pyramid(
                [
                    source_grid[sample_index:sample_index + 1]
                    for source_grid in source_grids
                ],
                target_h=target_h,
                target_w=target_w,
                geometry=sample_geometry,
                dtype=param.dtype,
            )
            if geo.shape[0] != int(target_len):
                raise AssertionError(
                    "DA2 sampled geometry length mismatch: "
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
    ) + [r"da2\..*"]

    DA2_STATE_KEYS = (
        "da2_mlp.alpha_value",
        "da2_mlp.input_norm.weight",
        "da2_mlp.mlp.0.weight",
        "da2_mlp.mlp.0.bias",
        "da2_mlp.mlp.2.weight",
        "da2_mlp.mlp.2.bias",
        "da2_mlp.output_norm.weight",
    )
    def __init__(self, config):
        vision_config = getattr(config, "vision_config", None)
        erp_fourier_linear_enabled = bool(
            getattr(
                vision_config,
                "erp_fourier_linear_enabled",
                getattr(config, "erp_fourier_linear_enabled", False),
            )
        )
        da2_enabled = bool(getattr(config, "da2_enabled", False))
        if erp_fourier_linear_enabled:
            ensure_erp_fourier_linear_config(config.vision_config)
        if da2_enabled:
            ensure_da2_config(config)
        super().__init__(config)

        if erp_fourier_linear_enabled:
            ensure_erp_fourier_linear_config(self.model.visual.config)
        if da2_enabled:
            ensure_da2_config(config)

        self.erp_fourier_linear_adapter = (
            ERPFourierLinearAdapter(self.model.visual.config)
            if erp_fourier_linear_enabled
            else None
        )
        self.da2_mlp = DA2GeometryMLP(config) if da2_enabled else None
        self.da2 = None
        self._da2_weights_ready = False
        self._da2_dtype = None
        self._da2_device = None
        self._da2_mlp_load_was_incompatible = False
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_geometry = None
        self._pano_runtime_da2_pixel_values = None
        self._pano_runtime_in_image_features = False
        self._pano_runtime_visual_grid_thw = None
        self._install_pano_patch_embed_hook()
        self._install_pano_merger_hook()
        self._install_image_feature_hook()
        self.register_load_state_dict_pre_hook(self._drop_incompatible_da2_mlp_pre_hook)
        self.register_load_state_dict_post_hook(self._load_missing_pano_parameters_post_hook)

    def _erp_fourier_linear_enabled(self) -> bool:
        return (
            self.erp_fourier_linear_adapter is not None
            and bool(getattr(self.erp_fourier_linear_adapter, "enabled", False))
        )

    def _da2_enabled(self) -> bool:
        return self.da2_mlp is not None and bool(getattr(self.da2_mlp, "enabled", False))

    def _da2_injection_stage(self) -> str:
        if self.da2_mlp is not None:
            return self.da2_mlp.injection_stage
        return normalize_da2_injection_stage(getattr(self.config, "da2_injection_stage", "post_merger"))

    def _runtime_grid_matches_image_grid(self, grid_thw: torch.Tensor | None) -> bool:
        runtime_grid = self._pano_runtime_grid_thw
        if grid_thw is None or runtime_grid is None:
            return False
        if grid_thw is runtime_grid:
            return True
        if grid_thw.device == runtime_grid.device and tuple(grid_thw.shape) == tuple(runtime_grid.shape):
            return grid_thw.data_ptr() == runtime_grid.data_ptr()
        return False

    def _current_image_apply_mask(self, grid_thw: torch.Tensor) -> torch.Tensor:
        image_apply_mask = torch.zeros(
            int(grid_thw.shape[0]),
            device=grid_thw.device,
            dtype=torch.bool,
        )
        for _, image_index in self._current_image_flat_indices(grid_thw):
            if 0 <= image_index < int(image_apply_mask.shape[0]):
                image_apply_mask[image_index] = True
        return image_apply_mask

    def _install_pano_patch_embed_hook(self) -> None:
        if not self._erp_fourier_linear_enabled():
            return
        visual = self.model.visual
        if hasattr(visual.patch_embed, "_pano_origin_forward"):
            return

        visual.patch_embed._pano_origin_forward = visual.patch_embed.forward
        owner = self

        def patch_embed_with_pano_adapters(this, hidden_states):
            outputs = this._pano_origin_forward(hidden_states)
            grid_thw = owner._pano_runtime_grid_thw
            if grid_thw is None or outputs.numel() == 0:
                return outputs

            if owner._erp_fourier_linear_enabled():
                image_apply_mask = None
                if owner.erp_fourier_linear_adapter.apply_to_current_only:
                    image_apply_mask = owner._current_image_apply_mask(grid_thw)
                outputs = owner.erp_fourier_linear_adapter.apply_to_patch_tokens(
                    patch_tokens=outputs,
                    grid_thw=grid_thw,
                    image_apply_mask=image_apply_mask,
                    image_erp_geometry=owner._pano_runtime_image_geometry,
                )

            return outputs

        visual.patch_embed.forward = MethodType(patch_embed_with_pano_adapters, visual.patch_embed)

    def _select_da2_current_items(
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

    def _da2_current_batch_inputs(
        self,
        *,
        da2_pixel_values: torch.Tensor,
        current_items: list[tuple[int, int]],
    ) -> torch.Tensor:
        if da2_pixel_values is None or da2_pixel_values.numel() == 0:
            raise AssertionError("DA2 is enabled but da2_pixel_values is missing")
        if int(da2_pixel_values.shape[0]) != len(current_items):
            raise AssertionError(
                "DA2 current-image batch size mismatch: "
                f"da2_batch={int(da2_pixel_values.shape[0])}, "
                f"current_items={len(current_items)}"
            )
        batch_indices = torch.tensor(
            [item[0] for item in current_items],
            device=da2_pixel_values.device,
            dtype=torch.long,
        )
        return da2_pixel_values.index_select(0, batch_indices)

    def _target_geometry_for_indices(
        self,
        target_indices: list[int],
    ) -> torch.Tensor | None:
        geometry = self._pano_runtime_image_geometry
        if geometry is None:
            return None
        return geometry[target_indices].detach().to(device="cpu", dtype=torch.float32)

    def _apply_post_merger_da2_residual(self, vision_output, image_grid_thw: torch.Tensor):
        if image_grid_thw is None or vision_output.pooler_output is None:
            return vision_output

        image_embeds = list(vision_output.pooler_output)
        current_items = self._select_da2_current_items(image_grid_thw, num_image_outputs=len(image_embeds))
        if not current_items:
            return vision_output

        da2_pixel_values = self._pano_runtime_da2_pixel_values
        da2_inputs = self._da2_current_batch_inputs(
            da2_pixel_values=da2_pixel_values,
            current_items=current_items,
        )
        target_indices = [item[1] for item in current_items]
        target_grid_thw = image_grid_thw[target_indices].detach().to(device="cpu", dtype=torch.long)
        target_lengths = [int(image_embeds[index].shape[0]) for index in target_indices]
        geometry = self._target_geometry_for_indices(target_indices)

        da2_model = self._ensure_da2_model(
            device=image_embeds[target_indices[0]].device,
            dtype=image_embeds[target_indices[0]].dtype,
        )
        if da2_model is None or self.da2_mlp is None:
            return vision_output
        deltas = self.da2_mlp(
            da2_inputs,
            da2_model=da2_model,
            target_grid_thw=target_grid_thw,
            target_lengths=target_lengths,
            image_erp_geometry=geometry,
            output_device=image_embeds[target_indices[0]].device,
            output_dtype=image_embeds[target_indices[0]].dtype,
        )
        for image_index, delta in zip(target_indices, deltas):
            if delta.shape != image_embeds[image_index].shape:
                raise AssertionError(
                    "DA2 delta shape mismatch: "
                    f"delta={tuple(delta.shape)}, qwen={tuple(image_embeds[image_index].shape)}"
                )
            image_embeds[image_index] = image_embeds[image_index] + delta

        vision_output.pooler_output = tuple(image_embeds)
        return vision_output

    def _apply_pre_merger_da2_residual(
        self,
        hidden_states: torch.Tensor,
        image_grid_thw: torch.Tensor | None,
    ) -> torch.Tensor:
        if image_grid_thw is None or hidden_states.numel() == 0:
            return hidden_states

        current_items = self._select_da2_current_items(image_grid_thw)
        if not current_items:
            return hidden_states

        da2_pixel_values = self._pano_runtime_da2_pixel_values
        da2_inputs = self._da2_current_batch_inputs(
            da2_pixel_values=da2_pixel_values,
            current_items=current_items,
        )
        target_indices = [item[1] for item in current_items]
        target_grid_thw = image_grid_thw[target_indices].detach().to(device="cpu", dtype=torch.long)
        target_lengths = [
            int(num_frames * grid_h * grid_w)
            for num_frames, grid_h, grid_w in target_grid_thw.tolist()
        ]
        geometry = self._target_geometry_for_indices(target_indices)

        da2_model = self._ensure_da2_model(
            device=hidden_states.device,
            dtype=hidden_states.dtype,
        )
        if da2_model is None or self.da2_mlp is None:
            return hidden_states
        deltas = self.da2_mlp(
            da2_inputs,
            da2_model=da2_model,
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
                    "DA2 pre-merger delta shape mismatch: "
                    f"delta={tuple(delta.shape)}, qwen=({length}, {int(hidden_states.shape[-1])})"
                )
            hidden_states[start:start + length] = hidden_states[start:start + length] + delta
        return hidden_states

    def _install_pano_merger_hook(self) -> None:
        if not self._da2_enabled() or self._da2_injection_stage() != "pre_merger":
            return
        visual = self.model.visual
        if hasattr(visual.merger, "_pano_origin_forward"):
            return

        visual.merger._pano_origin_forward = visual.merger.forward
        owner = self

        def merger_with_pano_pre_merger_residual(this, hidden_states):
            if (
                owner._da2_enabled()
                and owner._da2_injection_stage() == "pre_merger"
                and owner._pano_runtime_in_image_features
            ):
                hidden_states = owner._apply_pre_merger_da2_residual(
                    hidden_states=hidden_states,
                    image_grid_thw=owner._pano_runtime_visual_grid_thw,
                )
            return this._pano_origin_forward(hidden_states)

        visual.merger.forward = MethodType(merger_with_pano_pre_merger_residual, visual.merger)

    def _install_image_feature_hook(self) -> None:
        if not self._da2_enabled():
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
            if owner._da2_injection_stage() == "pre_merger":
                return vision_output
            vision_output = owner._apply_post_merger_da2_residual(vision_output, image_grid_thw)
            return vision_output

        self.model.get_image_features = MethodType(get_image_features_with_pano_residuals, self.model)

    def _set_runtime_pano_context(
        self,
        *,
        image_grid_thw: torch.Tensor | None,
        image_num_images: torch.Tensor | None,
        image_current_index: torch.Tensor | None,
        image_erp_geometry: torch.Tensor | None,
        da2_pixel_values: torch.Tensor | None,
    ) -> None:
        self._pano_runtime_grid_thw = image_grid_thw
        self._pano_runtime_image_num_images = image_num_images
        self._pano_runtime_image_current_index = image_current_index
        self._pano_runtime_image_geometry = image_erp_geometry
        self._pano_runtime_da2_pixel_values = da2_pixel_values

    def _clear_runtime_pano_context(self) -> None:
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_geometry = None
        self._pano_runtime_da2_pixel_values = None
        self._pano_runtime_in_image_features = False
        self._pano_runtime_visual_grid_thw = None

    def _drop_incompatible_da2_mlp_pre_hook(
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
        if self.da2_mlp is None:
            return
        key_prefix = f"{prefix}da2_mlp."
        checkpoint_keys = [key for key in state_dict.keys() if key.startswith(key_prefix)]
        if not checkpoint_keys:
            return
        target_state = self.da2_mlp.state_dict()
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
        self._da2_mlp_load_was_incompatible = True

    def _load_external_da2_weights(self) -> None:
        if not self._da2_enabled():
            return
        if self.da2 is None:
            self.da2 = build_da2_model(
                source_path=str(getattr(self.config, "da2_source_path")),
                model_path=str(getattr(self.config, "da2_model_path")),
            )
        freeze_da2_model(self.da2)
        self._da2_weights_ready = True
        self._da2_device = None
        self._da2_dtype = None

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

    def _load_saved_da2_weights(self, pretrained_model_name_or_path) -> bool:
        if not self._da2_enabled():
            return False

        checkpoint_dir = Path(str(pretrained_model_name_or_path))
        state_dict = self._load_prefixed_checkpoint_tensors(checkpoint_dir, "da2.")
        if not state_dict:
            return False

        if self.da2 is None:
            self.da2 = build_da2_model(
                source_path=str(getattr(self.config, "da2_source_path")),
                model_path=None,
                load_pretrained_weights=False,
            )

        try:
            try:
                self.da2.load_state_dict(state_dict, strict=True, assign=True)
            except TypeError:
                self.da2.load_state_dict(state_dict, strict=True)
        except RuntimeError as exc:
            self.da2 = None
            self._da2_weights_ready = False
            raise RuntimeError(
                "The VLN checkpoint contains da2.* tensors that do not match the "
                "bundled DA2 architecture"
            ) from exc
        freeze_da2_model(self.da2)
        self._da2_weights_ready = True
        self._da2_device = None
        self._da2_dtype = None
        return True

    def _mark_da2_weights_ready(self) -> None:
        if self.da2 is None:
            return
        freeze_da2_model(self.da2)
        self._da2_weights_ready = True
        self._da2_device = None
        self._da2_dtype = None

    def _ensure_da2_model(self, device: torch.device, dtype: torch.dtype):
        if not self._da2_enabled():
            return None
        if self.da2 is None:
            self.da2 = build_da2_model(
                source_path=str(getattr(self.config, "da2_source_path")),
                model_path=str(getattr(self.config, "da2_model_path")),
            )
        if not self._da2_weights_ready:
            self._load_external_da2_weights()
        if device.type != "cuda" or bool(getattr(self.config, "da2_force_fp32", False)):
            target_dtype = torch.float32
        else:
            # Keep the frozen encoder in Qwen's low-precision vision dtype when possible.
            target_dtype = dtype if dtype in (torch.bfloat16, torch.float16) else torch.float32
        if self._da2_device != device or self._da2_dtype != target_dtype:
            self.da2.to(device=device, dtype=target_dtype)
            self._da2_device = device
            self._da2_dtype = target_dtype
        freeze_da2_model(self.da2)
        return self.da2

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
        da2_pixel_values: torch.Tensor | None = None,
        **kwargs,
    ):
        if (
            self._erp_fourier_linear_enabled()
            or self._da2_enabled()
        ):
            self._set_runtime_pano_context(
                image_grid_thw=image_grid_thw,
                image_num_images=image_num_images,
                image_current_index=image_current_index,
                image_erp_geometry=image_erp_geometry,
                da2_pixel_values=da2_pixel_values,
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
            self._clear_runtime_pano_context()

    def _load_missing_pano_parameters_post_hook(self, module, incompatible_keys) -> None:
        del module
        missing_keys = set(incompatible_keys.missing_keys)

        erp_fourier_linear_module = self.erp_fourier_linear_adapter
        if (
            erp_fourier_linear_module is not None
            and erp_fourier_linear_module.enabled
            and any(key.startswith("erp_fourier_linear_adapter.") for key in missing_keys)
        ):
            erp_fourier_linear_module.reset_parameters()

        da2_module = self.da2_mlp
        if (
            da2_module is not None
            and da2_module.enabled
            and (
                self._da2_mlp_load_was_incompatible
                or any(key.startswith("da2_mlp.") for key in missing_keys)
            )
        ):
            da2_module.reset_parameters()
            self._da2_mlp_load_was_incompatible = False
        if da2_module is not None and da2_module.enabled:
            if any(key.startswith("da2.") for key in missing_keys):
                self._load_external_da2_weights()
            else:
                self._mark_da2_weights_ready()

    def _reset_erp_fourier_linear_parameters_after_pretrained_load(self) -> None:
        if self._erp_fourier_linear_enabled() and self.erp_fourier_linear_adapter is not None:
            self.erp_fourier_linear_adapter.reset_parameters()

    def _reset_da2_parameters_after_pretrained_load(self) -> None:
        if self._da2_enabled() and self.da2_mlp is not None:
            self.da2_mlp.reset_parameters()

    @classmethod
    def _checkpoint_has_da2_weights(cls, pretrained_model_name_or_path) -> bool | None:
        return cls._checkpoint_has_any_weights(pretrained_model_name_or_path, cls.DA2_STATE_KEYS)

    def _checkpoint_has_erp_fourier_linear_weights(self, pretrained_model_name_or_path) -> bool | None:
        if self.erp_fourier_linear_adapter is None:
            return None
        state_keys = tuple(
            f"erp_fourier_linear_adapter.{key}"
            for key in self.erp_fourier_linear_adapter.state_dict().keys()
        )
        return self._checkpoint_has_any_weights(pretrained_model_name_or_path, state_keys)

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
        has_erp_fourier_linear_weights = model._checkpoint_has_erp_fourier_linear_weights(
            pretrained_model_name_or_path
        )
        if has_erp_fourier_linear_weights is False:
            model._reset_erp_fourier_linear_parameters_after_pretrained_load()
        has_da2_adapter_weights = cls._checkpoint_has_da2_weights(pretrained_model_name_or_path)
        if has_da2_adapter_weights is False:
            model._reset_da2_parameters_after_pretrained_load()
        if model._da2_enabled():
            resolved_source = resolve_da2_source_path(
                getattr(model.config, "da2_source_path", None)
            )
            model.config.da2_source_path = (
                BUNDLED_DA2_SOURCE
                if resolved_source == DEFAULT_DA2_SOURCE_PATH.resolve()
                else str(resolved_source)
            )
            loaded_saved_da2 = model._load_saved_da2_weights(pretrained_model_name_or_path)
            if not loaded_saved_da2 and model.da2 is None:
                if has_da2_adapter_weights is True:
                    raise RuntimeError(
                        "The VLN checkpoint contains da2_mlp.* adapter weights but "
                        "does not contain the required da2.* encoder weights"
                    )
                model._load_external_da2_weights()
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
        da2_pixel_values=None,
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
            da2_pixel_values=da2_pixel_values,
            **kwargs,
        )
        if not is_first_iteration and use_cache:
            model_inputs["image_erp_geometry"] = None
            model_inputs["image_num_images"] = None
            model_inputs["image_current_index"] = None
            model_inputs["da2_pixel_values"] = None
        return model_inputs
