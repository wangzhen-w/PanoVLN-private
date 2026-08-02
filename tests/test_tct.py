import math
import unittest

import torch
import numpy as np
from PIL import Image
from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor

from src.train.data.tct import (
    _cached_tangent_sampling_grid,
    _pack_static_patches_in_qwen_order,
    build_tct_pixel_values,
)
from src.train.data.data import build_erp_image_geometry


class TctTest(unittest.TestCase):
    def test_optical_axes_match_original_erp_patch_centers(self):
        grid_h, grid_w = 5, 14
        vertical_fov = math.radians(140.0)
        center_pitch = math.radians(10.0)
        grid = _cached_tangent_sampling_grid(
            grid_h,
            grid_w,
            1,
            vertical_fov,
            center_pitch,
        )

        sampled_fraction = (grid[..., 0].double() + 1.0) * 1.5 - 1.0
        sampled_yaw = (sampled_fraction - 0.5) * (2.0 * math.pi)
        sampled_pitch = grid[..., 1].double() * (0.5 * math.pi)
        expected_yaw = (
            (torch.arange(grid_w, dtype=torch.float64) + 0.5) / grid_w - 0.5
        ) * (2.0 * math.pi)
        expected_pitch = (
            (torch.arange(grid_h, dtype=torch.float64) + 0.5) / grid_h - 0.5
        ) * vertical_fov + center_pitch

        # Compare periodic longitude through sin/cos so the ERP seam is unambiguous.
        self.assertTrue(
            torch.allclose(torch.sin(sampled_yaw), torch.sin(expected_yaw)[None], atol=1e-6)
        )
        self.assertTrue(
            torch.allclose(torch.cos(sampled_yaw), torch.cos(expected_yaw)[None], atol=1e-6)
        )
        self.assertTrue(torch.allclose(sampled_pitch, expected_pitch[:, None], atol=1e-6))

    def test_asymmetric_crop_geometry_keeps_image_down_pitch_sign(self):
        # Cropping more from the bottom leaves a band centered above the
        # horizon, hence a negative image-down pitch.
        geometry = build_erp_image_geometry(
            top_crop_degrees=20.0,
            bottom_crop_degrees=40.0,
        )
        self.assertAlmostEqual(float(geometry[0]), math.radians(120.0), places=6)
        self.assertAlmostEqual(float(geometry[1]), math.radians(-10.0), places=6)

        grid_h, grid_w = 4, 8
        grid = _cached_tangent_sampling_grid(
            grid_h,
            grid_w,
            1,
            float(geometry[0]),
            float(geometry[1]),
        )
        sampled_pitch = grid[..., 1].double() * (0.5 * math.pi)
        expected_pitch = (
            (torch.arange(grid_h, dtype=torch.float64) + 0.5) / grid_h - 0.5
        ) * math.radians(120.0) - math.radians(10.0)
        self.assertTrue(torch.allclose(sampled_pitch, expected_pitch[:, None], atol=1e-6))

    def test_packing_matches_qwen_block_major_order_and_temporal_duplication(self):
        grid_h, grid_w = 4, 6
        patch_size = 2
        merge_size = 2
        patch_ids = torch.arange(grid_h * grid_w, dtype=torch.float32).reshape(grid_h, grid_w)
        mosaic = patch_ids.repeat_interleave(patch_size, dim=0).repeat_interleave(
            patch_size,
            dim=1,
        ).unsqueeze(0)

        packed = _pack_static_patches_in_qwen_order(
            mosaic,
            grid_h=grid_h,
            grid_w=grid_w,
            patch_size=patch_size,
            temporal_patch_size=2,
            merge_size=merge_size,
        )
        expected_ids = []
        for block_row in range(grid_h // merge_size):
            for block_col in range(grid_w // merge_size):
                for intra_row in range(merge_size):
                    for intra_col in range(merge_size):
                        expected_ids.append(
                            (block_row * merge_size + intra_row) * grid_w
                            + block_col * merge_size
                            + intra_col
                        )

        unpacked = packed.reshape(grid_h * grid_w, 1, 2, patch_size, patch_size)
        self.assertEqual(unpacked[:, 0, 0, 0, 0].tolist(), expected_ids)
        self.assertTrue(torch.equal(unpacked[:, :, 0], unpacked[:, :, 1]))

    def test_packing_is_identical_to_huggingface_image_processor(self):
        grid_h, grid_w, patch_size = 4, 6, 16
        values = torch.arange(3 * grid_h * patch_size * grid_w * patch_size)
        mosaic = values.remainder(251).to(torch.uint8).reshape(
            3,
            grid_h * patch_size,
            grid_w * patch_size,
        )
        image = Image.fromarray(mosaic.permute(1, 2, 0).numpy())
        image_processor = Qwen2VLImageProcessor(
            do_resize=False,
            do_rescale=False,
            do_normalize=False,
            patch_size=patch_size,
            temporal_patch_size=2,
            merge_size=2,
        )

        reference = image_processor(images=[image], return_tensors="pt")["pixel_values"]
        packed = _pack_static_patches_in_qwen_order(
            mosaic.float(),
            grid_h=grid_h,
            grid_w=grid_w,
            patch_size=patch_size,
            temporal_patch_size=2,
            merge_size=2,
        )
        self.assertTrue(torch.equal(packed, reference))

    def test_preserves_known_qwen_shapes_and_constant_panorama(self):
        memory_images = [Image.new("RGB", (64, 32), (128, 64, 255)) for _ in range(10)]
        current_image = Image.new("RGB", (64, 32), (128, 64, 255))
        images = memory_images + [current_image]
        image_grid_thw = torch.tensor([[1, 10, 28]] * 10 + [[1, 24, 60]])
        geometry = torch.tensor([[math.radians(140.0), 0.0]] * len(images))

        pixel_values = build_tct_pixel_values(
            images,
            image_grid_thw,
            geometry,
            patch_size=16,
            temporal_patch_size=2,
            merge_size=2,
            do_rescale=True,
            rescale_factor=1.0 / 255.0,
            do_normalize=True,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        )

        self.assertEqual(pixel_values.shape, (4240, 1536))
        unpacked = pixel_values.reshape(4240, 3, 2, 16, 16)
        expected = torch.tensor(
            [2.0 * 128.0 / 255.0 - 1.0, 2.0 * 64.0 / 255.0 - 1.0, 1.0]
        )
        self.assertTrue(
            torch.allclose(unpacked.mean(dim=(0, 2, 3, 4)), expected, atol=1e-5)
        )
        self.assertTrue(torch.equal(unpacked[:, :, 0], unpacked[:, :, 1]))

    def test_grid_wraps_longitude_seam_inside_periodic_tile(self):
        grid = _cached_tangent_sampling_grid(
            4,
            8,
            16,
            math.radians(140.0),
            0.0,
        )
        # The middle ERP occupies [-1/3, +1/3] of the normalized 3x tiled source.
        # Samples on both sides of the seam remain connected to its neighboring copy.
        self.assertGreaterEqual(float(grid[..., 0].min()), -1.0 / 3.0 - 1e-6)
        self.assertLessEqual(float(grid[..., 0].max()), 1.0 / 3.0 + 1e-6)
        first_patch = grid[:16, :16, 0]
        last_patch = grid[:16, -16:, 0]
        self.assertTrue(torch.any(first_patch > 0.30))
        self.assertTrue(torch.any(last_patch < -0.30))

    def test_periodic_panorama_is_continuous_through_sampled_seam(self):
        grid_h, grid_w, patch_size = 4, 8, 16
        target_width = grid_w * patch_size
        source_width = 3 * target_width
        source_height = source_width // 2
        source_yaw = (
            (np.arange(source_width, dtype=np.float64) + 0.5) / source_width - 0.5
        ) * (2.0 * math.pi)
        red = np.broadcast_to((np.sin(source_yaw) + 1.0) * 127.5, (source_height, source_width))
        green = np.broadcast_to((np.cos(source_yaw) + 1.0) * 127.5, (source_height, source_width))
        blue = np.full_like(red, 127.5)
        image = Image.fromarray(np.stack((red, green, blue), axis=-1).round().astype(np.uint8))
        vertical_fov = math.radians(140.0)

        actual = build_tct_pixel_values(
            [image],
            torch.tensor([[1, grid_h, grid_w]]),
            torch.tensor([[vertical_fov, 0.0]]),
            patch_size=patch_size,
            temporal_patch_size=2,
            merge_size=2,
            do_rescale=True,
            rescale_factor=1.0 / 255.0,
            do_normalize=False,
            image_mean=(0.5, 0.5, 0.5),
            image_std=(0.5, 0.5, 0.5),
        )
        grid = _cached_tangent_sampling_grid(
            grid_h,
            grid_w,
            patch_size,
            vertical_fov,
            0.0,
        )
        fraction = (grid[..., 0].double() + 1.0) * 1.5 - 1.0
        sampled_yaw = (fraction - 0.5) * (2.0 * math.pi)
        expected_mosaic = torch.stack(
            (
                0.5 * (torch.sin(sampled_yaw) + 1.0),
                0.5 * (torch.cos(sampled_yaw) + 1.0),
                torch.full_like(sampled_yaw, 0.5),
            ),
            dim=0,
        ).float()
        expected = _pack_static_patches_in_qwen_order(
            expected_mosaic,
            grid_h=grid_h,
            grid_w=grid_w,
            patch_size=patch_size,
            temporal_patch_size=2,
            merge_size=2,
        )
        self.assertLess(float((actual - expected).abs().max()), 0.02)


if __name__ == "__main__":
    unittest.main()
