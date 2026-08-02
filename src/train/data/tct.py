"""Tangent-Canonical Tokenization (TCT) for equirectangular panoramas.

The Qwen processor is still used to choose the visual grid and to build all
multimodal placeholders.  This module only replaces its flattened patch
pixels: every original ERP patch center becomes the optical axis of a small,
gravity-referenced local perspective support sampled from the uncropped
panorama.  The original patch embedding and panoramic token lattice remain
unchanged.
"""

from __future__ import annotations

import math
from collections import defaultdict
from functools import lru_cache
from typing import Sequence

import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import functional as TF


@lru_cache(maxsize=32)
def _cached_tangent_sampling_grid(
    grid_h: int,
    grid_w: int,
    patch_size: int,
    vertical_fov: float,
    center_pitch: float,
) -> torch.Tensor:
    """Return an align_corners=False grid into the middle copy of a 3x ERP.

    Pitch follows the image convention used by the existing PanoVGGT mapping:
    negative values point upward and positive values point downward.
    """

    grid_h = int(grid_h)
    grid_w = int(grid_w)
    patch_size = int(patch_size)
    if grid_h <= 0 or grid_w <= 0 or patch_size <= 0:
        raise ValueError(
            "TCT grid and patch dimensions must be positive, got "
            f"grid=({grid_h}, {grid_w}), patch_size={patch_size}"
        )
    vertical_fov = float(vertical_fov)
    center_pitch = float(center_pitch)
    if not 0.0 < vertical_fov <= math.pi:
        raise ValueError(f"TCT vertical_fov must be in (0, pi], got {vertical_fov}")

    # These are exactly the centers of the original cropped-ERP patch lattice.
    rows = torch.arange(grid_h, dtype=torch.float64) + 0.5
    cols = torch.arange(grid_w, dtype=torch.float64) + 0.5
    center_pitch_grid = (rows / float(grid_h) - 0.5) * vertical_fov + center_pitch
    center_yaw_grid = (cols / float(grid_w) - 0.5) * (2.0 * math.pi)

    # One local view covers one equatorial ERP cell.  The footprint does not
    # shrink by cos(latitude). TCT instead uses the same local angular support
    # convention at every latitude. No learned or tuned FOV is introduced.
    horizontal_patch_fov = (2.0 * math.pi) / float(grid_w)
    vertical_patch_fov = vertical_fov / float(grid_h)
    patch_coordinates = (
        (torch.arange(patch_size, dtype=torch.float64) + 0.5) / float(patch_size)
    ) * 2.0 - 1.0
    tangent_x = patch_coordinates * math.tan(0.5 * horizontal_patch_fov)
    tangent_y = patch_coordinates * math.tan(0.5 * vertical_patch_fov)

    pitch = center_pitch_grid[:, None, None, None]
    yaw = center_yaw_grid[None, :, None, None]
    local_x = tangent_x[None, None, None, :]
    local_y = tangent_y[None, None, :, None]

    sin_pitch = torch.sin(pitch)
    cos_pitch = torch.cos(pitch)
    sin_yaw = torch.sin(yaw)
    cos_yaw = torch.cos(yaw)

    # World Y points up. Local +x is image-right and local +y is image-down.
    center_ray = torch.stack(
        (
            cos_pitch * sin_yaw,
            -sin_pitch.expand(-1, grid_w, -1, -1),
            cos_pitch * cos_yaw,
        ),
        dim=-1,
    )
    right_basis = torch.stack(
        (
            cos_yaw.expand(grid_h, -1, -1, -1),
            torch.zeros_like(sin_yaw).expand(grid_h, -1, -1, -1),
            -sin_yaw.expand(grid_h, -1, -1, -1),
        ),
        dim=-1,
    )
    down_basis = torch.stack(
        (
            -sin_pitch * sin_yaw,
            -cos_pitch.expand(-1, grid_w, -1, -1),
            -sin_pitch * cos_yaw,
        ),
        dim=-1,
    )

    ray = center_ray + local_x.unsqueeze(-1) * right_basis + local_y.unsqueeze(-1) * down_basis
    ray = F.normalize(ray, dim=-1)
    sample_yaw = torch.atan2(ray[..., 0], ray[..., 2])
    sample_pitch = torch.asin((-ray[..., 1]).clamp(-1.0, 1.0))

    # Tile the ERP three times horizontally. Sampling from the middle copy
    # makes bilinear interpolation continuous across the +/-pi seam.
    longitude_fraction = torch.remainder(sample_yaw / (2.0 * math.pi) + 0.5, 1.0)
    sample_x = 2.0 * (1.0 + longitude_fraction) / 3.0 - 1.0
    sample_y = 2.0 * sample_pitch / math.pi
    sampling_grid = torch.stack((sample_x, sample_y), dim=-1)

    # Convert [patch_row, patch_col, y, x] to one patch-aligned mosaic. No
    # resize or convolution ever crosses the artificial cell boundaries.
    sampling_grid = sampling_grid.permute(0, 2, 1, 3, 4).contiguous()
    return sampling_grid.reshape(grid_h * patch_size, grid_w * patch_size, 2).float()


def _pack_static_patches_in_qwen_order(
    patch_mosaic: torch.Tensor,
    *,
    grid_h: int,
    grid_w: int,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
) -> torch.Tensor:
    """Flatten a TCT mosaic in Qwen2-VL's exact block-major order."""

    channels, mosaic_h, mosaic_w = patch_mosaic.shape
    if (mosaic_h, mosaic_w) != (grid_h * patch_size, grid_w * patch_size):
        raise AssertionError(
            "TCT patch mosaic shape mismatch: "
            f"mosaic={tuple(patch_mosaic.shape)}, grid=({grid_h}, {grid_w}), "
            f"patch_size={patch_size}"
        )
    if grid_h % merge_size != 0 or grid_w % merge_size != 0:
        raise AssertionError(
            "TCT grid must be divisible by Qwen merge_size: "
            f"grid=({grid_h}, {grid_w}), merge_size={merge_size}"
        )

    patches = patch_mosaic.reshape(
        channels,
        grid_h,
        patch_size,
        grid_w,
        patch_size,
    ).permute(1, 3, 0, 2, 4)
    # A still image is duplicated along Qwen's temporal patch dimension.
    patches = patches.unsqueeze(3).expand(
        grid_h,
        grid_w,
        channels,
        temporal_patch_size,
        patch_size,
        patch_size,
    )
    patches = patches.reshape(
        grid_h // merge_size,
        merge_size,
        grid_w // merge_size,
        merge_size,
        channels,
        temporal_patch_size,
        patch_size,
        patch_size,
    ).permute(0, 2, 1, 3, 4, 5, 6, 7)
    return patches.contiguous().reshape(
        grid_h * grid_w,
        channels * temporal_patch_size * patch_size * patch_size,
    )


def build_tct_pixel_values(
    raw_erp_images: Sequence[Image.Image],
    image_grid_thw: torch.Tensor,
    image_erp_geometry: torch.Tensor,
    *,
    patch_size: int,
    temporal_patch_size: int,
    merge_size: int,
    do_rescale: bool,
    rescale_factor: float,
    do_normalize: bool,
    image_mean: Sequence[float],
    image_std: Sequence[float],
) -> torch.Tensor:
    """Build TCT pixels while preserving Qwen's token count and ordering."""

    grid_thw = torch.as_tensor(image_grid_thw, dtype=torch.long).cpu()
    geometry = torch.as_tensor(image_erp_geometry, dtype=torch.float32).cpu()
    num_images = len(raw_erp_images)
    if grid_thw.ndim != 2 or tuple(grid_thw.shape[1:]) != (3,):
        raise ValueError(f"Expected image_grid_thw [N, 3], got {tuple(grid_thw.shape)}")
    if geometry.shape != (num_images, 2):
        raise ValueError(
            "TCT image geometry/image count mismatch: "
            f"geometry={tuple(geometry.shape)}, images={num_images}"
        )
    if int(grid_thw.shape[0]) != num_images:
        raise ValueError(
            "TCT image_grid_thw/image count mismatch: "
            f"grid_rows={int(grid_thw.shape[0])}, images={num_images}"
        )

    patch_size = int(patch_size)
    temporal_patch_size = int(temporal_patch_size)
    merge_size = int(merge_size)
    if patch_size <= 0 or temporal_patch_size <= 0 or merge_size <= 0:
        raise ValueError("TCT processor patch dimensions must be positive")

    grouped_indices: dict[tuple[int, int, float, float], list[int]] = defaultdict(list)
    grid_rows = grid_thw.tolist()
    for image_index, (grid_t, grid_h, grid_w) in enumerate(grid_rows):
        if int(grid_t) != 1:
            raise ValueError(
                "TCT currently supports still ERP images only, got "
                f"image_grid_thw[{image_index}]={grid_rows[image_index]}"
            )
        vertical_fov, center_pitch = geometry[image_index].tolist()
        cache_key = (
            int(grid_h),
            int(grid_w),
            round(float(vertical_fov), 10),
            round(float(center_pitch), 10),
        )
        grouped_indices[cache_key].append(image_index)

    packed_by_index: list[torch.Tensor | None] = [None] * num_images
    mean = torch.as_tensor(image_mean, dtype=torch.float32).view(1, -1, 1, 1)
    std = torch.as_tensor(image_std, dtype=torch.float32).view(1, -1, 1, 1)
    if mean.shape[1] != 3 or std.shape[1] != 3:
        raise ValueError("TCT expects three-channel image_mean and image_std")
    if bool(do_normalize) and torch.any(std == 0):
        raise ValueError("TCT image_std must be nonzero")

    for (grid_h, grid_w, vertical_fov, center_pitch), image_indices in grouped_indices.items():
        # Normalize the uncropped ERP resolution with antialiased Lanczos. This
        # keeps the original full sphere and avoids aliasing for memory images.
        source_width = grid_w * patch_size
        source_height = max(2, source_width // 2)
        source_batch = []
        for image_index in image_indices:
            source_image = raw_erp_images[image_index].convert("RGB")
            # Resize longitude periodically as well as sampling it periodically.
            # Resizing one ERP directly would let Lanczos see image borders at
            # +/-pi before the later tiling can repair them.  The middle copy of
            # this resized 3x panorama has circular neighbors on both sides.
            tiled_image = Image.new(
                "RGB",
                (3 * source_image.width, source_image.height),
            )
            for tile_index in range(3):
                tiled_image.paste(source_image, (tile_index * source_image.width, 0))
            resized_tiled_image = tiled_image.resize(
                (3 * source_width, source_height),
                Image.Resampling.LANCZOS,
            )
            source_batch.append(TF.pil_to_tensor(resized_tiled_image).float())
            tiled_image.close()
            resized_tiled_image.close()
        source = torch.stack(source_batch, dim=0)
        if bool(do_rescale):
            source = source * float(rescale_factor)
        if bool(do_normalize):
            source = (source - mean) / std

        sampling_grid = _cached_tangent_sampling_grid(
            grid_h,
            grid_w,
            patch_size,
            vertical_fov,
            center_pitch,
        ).unsqueeze(0).expand(len(image_indices), -1, -1, -1)
        mosaics = F.grid_sample(
            source,
            sampling_grid,
            mode="bilinear",
            padding_mode="border",
            align_corners=False,
        )
        for batch_index, image_index in enumerate(image_indices):
            packed_by_index[image_index] = _pack_static_patches_in_qwen_order(
                mosaics[batch_index],
                grid_h=grid_h,
                grid_w=grid_w,
                patch_size=patch_size,
                temporal_patch_size=temporal_patch_size,
                merge_size=merge_size,
            )

    if any(value is None for value in packed_by_index):
        raise AssertionError("TCT failed to construct one or more image patch tensors")
    return torch.cat([value for value in packed_by_index if value is not None], dim=0)


def replace_qwen_pixel_values_with_tct(
    encoded,
    raw_erp_images: Sequence[Image.Image],
    image_erp_geometry: torch.Tensor,
    image_processor,
) -> None:
    """Replace only pixel_values in an existing Qwen processor result."""

    if "pixel_values" not in encoded or "image_grid_thw" not in encoded:
        raise ValueError("TCT requires processor pixel_values and image_grid_thw")
    original_pixel_values = encoded["pixel_values"]
    pixel_values = build_tct_pixel_values(
        raw_erp_images,
        encoded["image_grid_thw"],
        image_erp_geometry,
        patch_size=int(image_processor.patch_size),
        temporal_patch_size=int(image_processor.temporal_patch_size),
        merge_size=int(image_processor.merge_size),
        do_rescale=bool(getattr(image_processor, "do_rescale", True)),
        rescale_factor=float(getattr(image_processor, "rescale_factor", 1.0 / 255.0)),
        do_normalize=bool(getattr(image_processor, "do_normalize", True)),
        image_mean=getattr(image_processor, "image_mean", (0.5, 0.5, 0.5)),
        image_std=getattr(image_processor, "image_std", (0.5, 0.5, 0.5)),
    )
    if tuple(pixel_values.shape) != tuple(original_pixel_values.shape):
        raise AssertionError(
            "TCT must preserve Qwen pixel_values shape: "
            f"tct={tuple(pixel_values.shape)}, processor={tuple(original_pixel_values.shape)}"
        )
    encoded["pixel_values"] = pixel_values.to(dtype=original_pixel_values.dtype)
