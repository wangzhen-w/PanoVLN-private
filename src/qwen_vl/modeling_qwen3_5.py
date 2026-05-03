import json
import math
from pathlib import Path
from types import MethodType

import torch
import torch.nn as nn
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5ForConditionalGeneration


def ensure_erp_vision_config(vision_config) -> None:
    defaults = {
        "erp_pos_enabled": True,
        "erp_pos_hidden_size": getattr(vision_config, "hidden_size", 1024),
        "erp_pos_init_scale": 0.0,
        "erp_assume_centered": True,
        "erp_center_latitude_deg": 0.0,
        "erp_apply_to_current_only": False,
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
        self.init_scale = float(getattr(config, "erp_pos_init_scale", 0.0))
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
        self.alpha = nn.Parameter(torch.tensor(self.init_scale, dtype=torch.float32))
        self.reset_parameters(self.initializer_range)
        self.register_load_state_dict_post_hook(self._load_state_dict_post_hook)

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
        with torch.no_grad():
            self.alpha.fill_(self.init_scale)

    def _load_state_dict_post_hook(self, module, incompatible_keys) -> None:
        del module
        missing_keys = set(incompatible_keys.missing_keys)
        if any("erp_position_mlp.mlp" in key for key in missing_keys):
            self.reset_parameters(self.initializer_range)
        if any(key.endswith("erp_position_mlp.alpha") or key == "alpha" for key in missing_keys):
            with torch.no_grad():
                self.alpha.fill_(self.init_scale)

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
            position_delta = self.mlp(position_features)
            position_delta = self.alpha.to(dtype=position_delta.dtype) * position_delta
            token_splits[image_index] = tokens + position_delta.to(
                device=tokens.device,
                dtype=tokens.dtype,
            )

        return torch.cat(token_splits, dim=0)


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
    ERP_STATE_KEYS = (
        "erp_position_mlp.alpha",
        "erp_position_mlp.mlp.0.weight",
        "erp_position_mlp.mlp.0.bias",
        "erp_position_mlp.mlp.2.weight",
        "erp_position_mlp.mlp.2.bias",
    )

    def __init__(self, config):
        ensure_erp_vision_config(config.vision_config)
        super().__init__(config)

        visual_config = self.model.visual.config
        ensure_erp_vision_config(visual_config)
        self.erp_position_mlp = ERPPositionMLP(visual_config)
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_erp_geometry = None
        self._install_patch_embed_hook()
        self.register_load_state_dict_post_hook(self._load_missing_erp_parameters_post_hook)

    def _install_patch_embed_hook(self) -> None:
        visual = self.model.visual
        if hasattr(visual.patch_embed, "_pano_origin_forward"):
            return

        visual.patch_embed._pano_origin_forward = visual.patch_embed.forward
        owner = self

        def patch_embed_with_erp(this, hidden_states):
            outputs = this._pano_origin_forward(hidden_states)
            grid_thw = owner._pano_runtime_grid_thw
            if grid_thw is None or outputs.numel() == 0:
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

    def _set_runtime_erp_context(
        self,
        *,
        image_grid_thw: torch.Tensor | None,
        image_num_images: torch.Tensor | None,
        image_current_index: torch.Tensor | None,
        image_erp_geometry: torch.Tensor | None,
    ) -> None:
        self._pano_runtime_grid_thw = image_grid_thw
        self._pano_runtime_image_num_images = image_num_images
        self._pano_runtime_image_current_index = image_current_index
        self._pano_runtime_image_erp_geometry = image_erp_geometry

    def _clear_runtime_erp_context(self) -> None:
        self._pano_runtime_grid_thw = None
        self._pano_runtime_image_num_images = None
        self._pano_runtime_image_current_index = None
        self._pano_runtime_image_erp_geometry = None

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
        **kwargs,
    ):
        self._set_runtime_erp_context(
            image_grid_thw=image_grid_thw,
            image_num_images=image_num_images,
            image_current_index=image_current_index,
            image_erp_geometry=image_erp_geometry,
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

    def _load_missing_erp_parameters_post_hook(self, module, incompatible_keys) -> None:
        del module
        erp_module = self.erp_position_mlp
        if not erp_module.enabled:
            return

        missing_keys = set(incompatible_keys.missing_keys)
        if any(key.startswith("erp_position_mlp.mlp") for key in missing_keys):
            erp_module.reset_parameters(erp_module.initializer_range)
        if any(key == "erp_position_mlp.alpha" or key == "alpha" for key in missing_keys):
            with torch.no_grad():
                erp_module.alpha.fill_(erp_module.init_scale)

    def _reset_erp_parameters_after_pretrained_load(self) -> None:
        erp_module = self.erp_position_mlp
        if not erp_module.enabled:
            return
        erp_module.reset_parameters(erp_module.initializer_range)

    @classmethod
    def _checkpoint_has_erp_weights(cls, pretrained_model_name_or_path) -> bool | None:
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
            return any(key in weight_map for key in cls.ERP_STATE_KEYS)

        safetensors_path = checkpoint_dir / "model.safetensors"
        if safetensors_path.exists():
            try:
                from safetensors import safe_open

                with safe_open(str(safetensors_path), framework="pt", device="cpu") as handle:
                    keys = set(handle.keys())
                return any(key in keys for key in cls.ERP_STATE_KEYS)
            except Exception:
                return None

        return None

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        model = super().from_pretrained(pretrained_model_name_or_path, *model_args, **kwargs)
        has_erp_weights = cls._checkpoint_has_erp_weights(pretrained_model_name_or_path)
        if has_erp_weights is False:
            model._reset_erp_parameters_after_pretrained_load()
        return model
