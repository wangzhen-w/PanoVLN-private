import math
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch
import torch.nn as nn
from safetensors.torch import save_file
from transformers.models.qwen3_5.modeling_qwen3_5 import (
    Qwen3_5ForConditionalGeneration,
)

from src.qwen_vl.modeling_qwen3_5 import (
    PBO_ACTION_HORIZON,
    PBO_INPUT_VECTOR_COUNT,
    PBO_NUM_ACTIONS,
    Qwen3_5ForConditionalGenerationForPanoVLN,
)
from src.train.data.data import (
    VLN_ACTION_TO_ID,
    apply_vln_memory_policy,
    build_vln_image_selection,
)
from src.train import utils as train_utils
from src.train.config.config import ModelConfig


def _vln_example(current_step: int) -> dict:
    history_pattern = ["forward", "left", "forward", "right"]
    history_actions = [
        history_pattern[index % len(history_pattern)]
        for index in range(current_step)
    ]
    return {
        "instruction": "Walk to the end of the corridor.",
        "images": [f"frame_{index}.jpg" for index in range(current_step + 1)],
        "step_index": current_step,
        "history_actions": history_actions,
        "action_sequence": ["forward"] * PBO_ACTION_HORIZON,
        "real_action_count": PBO_ACTION_HORIZON,
    }


class PBODataPolicyTest(unittest.TestCase):
    def test_exact_t_minus_four_uniform_hit_enables_pbo(self):
        current_step = 20
        example = _vln_example(current_step)

        normalized = apply_vln_memory_policy(example, pbo_enabled=True)
        selected_indices = build_vln_image_selection(
            current_step=current_step,
            last_frame_index=current_step,
        )
        anchor = current_step - PBO_ACTION_HORIZON

        self.assertIn(anchor, selected_indices)
        self.assertTrue(normalized["_pbo_valid"])
        self.assertEqual(
            normalized["_pbo_start_image_index"],
            selected_indices.index(anchor),
        )
        self.assertEqual(
            normalized["_pbo_action_labels"],
            [VLN_ACTION_TO_ID[action] for action in example["history_actions"][-4:]],
        )

    def test_missing_t_minus_four_is_masked_without_changing_uniform_images(self):
        current_step = 21
        example = _vln_example(current_step)

        without_pbo = apply_vln_memory_policy(example, pbo_enabled=False)
        with_pbo = apply_vln_memory_policy(example, pbo_enabled=True)
        selected_indices = build_vln_image_selection(
            current_step=current_step,
            last_frame_index=current_step,
        )

        self.assertNotIn(current_step - PBO_ACTION_HORIZON, selected_indices)
        self.assertEqual(with_pbo["images"], without_pbo["images"])
        self.assertFalse(with_pbo["_pbo_valid"])
        self.assertEqual(with_pbo["_pbo_start_image_index"], -1)
        self.assertEqual(with_pbo["_pbo_action_labels"], [-100] * 4)


class _RecordingPBOHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))
        self.last_features = None

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        self.last_features = features.detach().clone()
        return features.new_zeros(
            (features.shape[0], PBO_ACTION_HORIZON * PBO_NUM_ACTIONS)
        ) + self.anchor * 0.0


class PBOModelTest(unittest.TestCase):
    def test_head_keeps_512_hidden_units_with_six_vector_input(self):
        hidden_size = 8
        config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden_size),
            vision_config=SimpleNamespace(spatial_merge_size=1),
            pbo_enabled=True,
            pbo_head_hidden_size=512,
            panovggt_enabled=False,
            initializer_range=0.02,
        )

        def fake_base_init(model, model_config):
            nn.Module.__init__(model)
            model.config = model_config
            model.model = nn.Module()
            model.model.language_model = nn.Module()
            model.model.language_model.norm = nn.Identity()

        with patch.object(
            Qwen3_5ForConditionalGeneration,
            "__init__",
            fake_base_init,
        ):
            model = Qwen3_5ForConditionalGenerationForPanoVLN(config)

        self.assertEqual(PBO_INPUT_VECTOR_COUNT, 6)
        self.assertEqual(model.pbo_head[0].normalized_shape, (6 * hidden_size,))
        self.assertEqual(model.pbo_head[1].in_features, 6 * hidden_size)
        self.assertEqual(model.pbo_head[1].out_features, 512)

    def test_legacy_head_uses_saved_seven_vector_width(self):
        hidden_size = 8
        config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=hidden_size),
            vision_config=SimpleNamespace(spatial_merge_size=1),
            pbo_enabled=True,
            pbo_head_hidden_size=512,
            pbo_input_vector_count=7,
            panovggt_enabled=False,
            initializer_range=0.02,
        )

        def fake_base_init(model, model_config):
            nn.Module.__init__(model)
            model.config = model_config
            model.model = nn.Module()
            model.model.language_model = nn.Module()
            model.model.language_model.norm = nn.Identity()

        with patch.object(
            Qwen3_5ForConditionalGeneration,
            "__init__",
            fake_base_init,
        ):
            model = Qwen3_5ForConditionalGenerationForPanoVLN(config)

        self.assertEqual(model.pbo_head[0].normalized_shape, (7 * hidden_size,))
        self.assertEqual(model.pbo_head[1].in_features, 7 * hidden_size)

    def test_pbo_feature_contains_only_the_two_endpoint_fourier_vectors(self):
        hidden_size = 2
        model = object.__new__(Qwen3_5ForConditionalGenerationForPanoVLN)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=1)
        )
        model.pbo_head = _RecordingPBOHead()

        grid = torch.tensor([1, 1, 2], dtype=torch.long)
        geometry = torch.tensor([math.pi, 0.0], dtype=torch.float32)
        previous_hidden = torch.tensor(
            [[1.0, 2.0], [3.0, 4.0]],
            requires_grad=True,
        )
        current_hidden = torch.tensor(
            [[5.0, 6.0], [7.0, 8.0]],
            requires_grad=True,
        )
        hidden_states = torch.randn(1, 5, hidden_size, requires_grad=True)

        loss = model._compute_pbo_loss(
            hidden_states=hidden_states,
            image_groups=[
                [
                    (previous_hidden, grid, geometry),
                    (current_hidden, grid, geometry),
                ]
            ],
            labels=torch.tensor([[-100, -100, 1, 2, 3]]),
            image_current_index=torch.tensor([1]),
            pbo_action_labels=torch.tensor([[1, 2, 1, 3]]),
            pbo_valid_mask=torch.tensor([True]),
            pbo_start_image_index=torch.tensor([0]),
        )

        expected = torch.cat(
            (
                model.spherical_yaw_fourier_pool(
                    previous_hidden,
                    grid,
                    geometry,
                    spatial_merge_size=1,
                ),
                model.spherical_yaw_fourier_pool(
                    current_hidden,
                    grid,
                    geometry,
                    spatial_merge_size=1,
                ),
            )
        ).unsqueeze(0)

        self.assertEqual(model.pbo_head.last_features.shape, (1, 6 * hidden_size))
        torch.testing.assert_close(model.pbo_head.last_features, expected)
        self.assertTrue(torch.isfinite(loss))

    def test_legacy_seven_vector_feature_prepends_instruction_context(self):
        hidden_size = 2
        model = object.__new__(Qwen3_5ForConditionalGenerationForPanoVLN)
        nn.Module.__init__(model)
        model.config = SimpleNamespace(
            vision_config=SimpleNamespace(spatial_merge_size=1),
            pbo_input_vector_count=7,
        )
        model.pbo_head = _RecordingPBOHead()

        grid = torch.tensor([1, 1, 2], dtype=torch.long)
        geometry = torch.tensor([math.pi, 0.0], dtype=torch.float32)
        previous_hidden = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        current_hidden = torch.tensor([[5.0, 6.0], [7.0, 8.0]])
        hidden_states = torch.tensor(
            [[[10.0, 11.0], [12.0, 13.0], [14.0, 15.0], [16.0, 17.0]]],
            requires_grad=True,
        )
        labels = torch.tensor([[-100, -100, -100, 1]])

        loss = model._compute_pbo_loss(
            hidden_states=hidden_states,
            image_groups=[
                [
                    (previous_hidden, grid, geometry),
                    (current_hidden, grid, geometry),
                ]
            ],
            labels=labels,
            image_current_index=torch.tensor([1]),
            pbo_action_labels=torch.tensor([[1, 2, 1, 3]]),
            pbo_valid_mask=torch.tensor([True]),
            pbo_start_image_index=torch.tensor([0]),
        )

        expected = torch.cat(
            (
                hidden_states[0, 2],
                model.spherical_yaw_fourier_pool(
                    previous_hidden,
                    grid,
                    geometry,
                    spatial_merge_size=1,
                ),
                model.spherical_yaw_fourier_pool(
                    current_hidden,
                    grid,
                    geometry,
                    spatial_merge_size=1,
                ),
            )
        ).unsqueeze(0)
        self.assertEqual(model.pbo_head.last_features.shape, (1, 7 * hidden_size))
        torch.testing.assert_close(model.pbo_head.last_features, expected)
        self.assertTrue(torch.isfinite(loss))


class PBOConfigCompatibilityTest(unittest.TestCase):
    def test_legacy_width_is_inferred_when_config_metadata_is_missing(self):
        config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=2),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint_dir = Path(temporary_directory)
            save_file(
                {"pbo_head.0.weight": torch.zeros(14)},
                checkpoint_dir / "model.safetensors",
            )
            vector_count = (
                Qwen3_5ForConditionalGenerationForPanoVLN
                ._configure_pbo_input_width_from_checkpoint(
                    checkpoint_dir,
                    config,
                )
            )

        self.assertEqual(vector_count, 7)
        self.assertEqual(config.pbo_input_vector_count, 7)

    def test_training_loader_preserves_legacy_checkpoint_input_width(self):
        checkpoint_config = SimpleNamespace(
            pbo_enabled=True,
            pbo_loss_weight=0.1,
            pbo_head_hidden_size=512,
            pbo_input_vector_count=7,
        )
        cfg = SimpleNamespace(
            model=ModelConfig(
                name_or_path="unused",
                pbo_enabled=True,
            )
        )

        with patch.object(
            train_utils.Qwen3_5Config,
            "from_pretrained",
            return_value=checkpoint_config,
        ):
            loaded_config = train_utils._load_model_config(cfg)

        self.assertEqual(loaded_config.pbo_input_vector_count, 7)


if __name__ == "__main__":
    unittest.main()
