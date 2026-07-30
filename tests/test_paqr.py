import math
import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn
from transformers import AutoTokenizer

from src.qwen_vl.modeling_qwen3_5 import (
    PAQR_DEFAULT_ACTION_TOKEN_IDS,
    PAQR_DEFAULT_STOP_TOKEN_ID,
    PanoramicActionQueryReadout,
    Qwen3_5ForConditionalGenerationForPanoVLN,
)
from src.train.utils import validate_paqr_tokenizer


MODEL_PATH = "/workspace/code/a_property/model/Qwen3.5-4B"


def paqr_config(**overrides):
    values = {
        "paqr_enabled": True,
        "paqr_action_token_ids": list(PAQR_DEFAULT_ACTION_TOKEN_IDS),
        "paqr_stop_token_id": PAQR_DEFAULT_STOP_TOKEN_ID,
        "paqr_temperature": 0.1,
        "paqr_prior_init": 0.02,
        "paqr_prior_max": 0.25,
        "paqr_logit_scale_max": 0.5,
        "paqr_first_action_only": True,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class PAQRModuleTest(unittest.TestCase):
    def test_has_only_four_trainable_scalars(self):
        module = PanoramicActionQueryReadout(paqr_config())
        parameters = dict(module.named_parameters())
        self.assertEqual(
            set(parameters),
            {
                "raw_gate",
                "raw_turn_cos",
                "raw_turn_sin",
                "raw_forward_cos",
            },
        )
        self.assertTrue(all(parameter.numel() == 1 for parameter in parameters.values()))
        self.assertTrue(all(parameter.requires_grad for parameter in parameters.values()))

    def test_prior_is_weak_periodic_and_directionally_initialized(self):
        module = PanoramicActionQueryReadout(paqr_config())
        yaw = torch.tensor(
            [-math.pi, -math.radians(15), 0.0, math.radians(15), math.pi]
        )
        prior = module.circular_prior(yaw)

        self.assertLessEqual(
            float(prior.detach().abs().max()),
            module.prior_max + 1e-7,
        )
        torch.testing.assert_close(prior[:, 0], prior[:, -1], atol=1e-7, rtol=0)
        detached_prior = prior.detach()
        self.assertGreater(float(detached_prior[0, 1]), float(detached_prior[2, 1]))
        self.assertGreater(float(detached_prior[2, 3]), float(detached_prior[0, 3]))
        self.assertGreater(float(detached_prior[1, 2]), float(detached_prior[0, 2]))
        self.assertGreater(float(detached_prior[1, 2]), float(detached_prior[2, 2]))

    def test_attention_is_normalized_and_lm_head_queries_are_detached(self):
        torch.manual_seed(7)
        module = PanoramicActionQueryReadout(paqr_config())
        hidden = torch.randn(12, 8, requires_grad=True)
        lm_head = torch.randn(14000, 8, requires_grad=True)
        yaw = torch.linspace(-math.pi, math.pi, 12)

        evidence, attention, content, prior = module.compute_evidence(
            hidden,
            yaw,
            lm_head,
        )
        self.assertEqual(tuple(evidence.shape), (3,))
        self.assertEqual(tuple(attention.shape), (3, 12))
        self.assertEqual(tuple(content.shape), (3, 12))
        self.assertEqual(tuple(prior.shape), (3, 12))
        torch.testing.assert_close(
            attention.sum(dim=-1),
            torch.ones(3),
            atol=1e-6,
            rtol=0,
        )
        self.assertLessEqual(float(content.detach().abs().max()), 1.0 + 1e-6)

        with torch.no_grad():
            module.raw_gate.fill_(0.2)
        delta = module(hidden, yaw, lm_head)
        delta.sum().backward()
        self.assertIsNone(lm_head.grad)
        self.assertIsNotNone(hidden.grad)

    def test_zero_gate_is_exact_identity_but_receives_gradient(self):
        torch.manual_seed(11)
        module = PanoramicActionQueryReadout(paqr_config())
        hidden = torch.randn(16, 10, requires_grad=True)
        lm_head = torch.randn(14000, 10, requires_grad=True)
        yaw = torch.linspace(-math.pi, math.pi, 16)

        delta = module(hidden, yaw, lm_head)
        torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0, rtol=0)
        weights = torch.tensor([1.0, -0.5, 0.25])
        (delta * weights).sum().backward()
        self.assertIsNotNone(module.raw_gate.grad)
        self.assertGreater(abs(float(module.raw_gate.grad)), 0.0)
        self.assertIsNone(lm_head.grad)

    def test_zero_gate_identity_and_gradient_hold_after_bf16_model_cast(self):
        torch.manual_seed(13)
        module = PanoramicActionQueryReadout(paqr_config()).to(
            dtype=torch.bfloat16
        )
        self.assertTrue(
            all(
                parameter.dtype == torch.bfloat16
                for parameter in module.parameters()
            )
        )
        hidden = torch.randn(16, 10, dtype=torch.bfloat16)
        lm_head = torch.randn(14000, 10, dtype=torch.bfloat16)
        yaw = torch.linspace(-math.pi, math.pi, 16)

        delta = module(hidden, yaw, lm_head)
        torch.testing.assert_close(delta, torch.zeros_like(delta), atol=0, rtol=0)
        (delta * torch.tensor([1.0, -0.5, 0.25])).sum().backward()
        self.assertIsNotNone(module.raw_gate.grad)
        self.assertGreater(abs(float(module.raw_gate.grad)), 0.0)

    def test_uses_live_lm_head_rows(self):
        module = PanoramicActionQueryReadout(paqr_config())
        hidden = torch.eye(4)
        yaw = torch.linspace(-math.pi, math.pi, 4)
        lm_head = torch.zeros(14000, 4)
        lm_head[PAQR_DEFAULT_ACTION_TOKEN_IDS[0], 0] = 1.0
        lm_head[PAQR_DEFAULT_ACTION_TOKEN_IDS[1], 1] = 1.0
        lm_head[PAQR_DEFAULT_ACTION_TOKEN_IDS[2], 2] = 1.0
        _, _, before, _ = module.compute_evidence(hidden, yaw, lm_head)

        lm_head[PAQR_DEFAULT_ACTION_TOKEN_IDS[0]].zero_()
        lm_head[PAQR_DEFAULT_ACTION_TOKEN_IDS[0], 3] = 1.0
        _, _, after, _ = module.compute_evidence(hidden, yaw, lm_head)
        self.assertFalse(torch.equal(before[0], after[0]))
        torch.testing.assert_close(before[1:], after[1:])


class PAQRRoutingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(19)
        self.hidden_size = 6
        self.vocab_size = 14000
        self.model = object.__new__(Qwen3_5ForConditionalGenerationForPanoVLN)
        nn.Module.__init__(self.model)
        self.model.config = SimpleNamespace(
            image_token_id=99,
            paqr_stop_token_id=PAQR_DEFAULT_STOP_TOKEN_ID,
            vision_config=SimpleNamespace(spatial_merge_size=2),
        )
        self.model.paqr = PanoramicActionQueryReadout(paqr_config())
        with torch.no_grad():
            self.model.paqr.raw_gate.fill_(0.2)
        self.model.lm_head = nn.Linear(
            self.hidden_size,
            self.vocab_size,
            bias=False,
        )
        self.model._pano_runtime_image_num_images = torch.tensor([2, 1])
        self.model._pano_runtime_image_current_index = torch.tensor([1, 0])

        self.input_ids = torch.zeros((2, 14), dtype=torch.long)
        self.input_ids[0, 1:9] = 99
        self.input_ids[1, 2:10] = 99
        self.image_grid_thw = torch.tensor(
            [
                [1, 4, 4],
                [1, 4, 4],
                [1, 4, 8],
            ],
            dtype=torch.long,
        )
        self.hidden = torch.randn(2, 14, self.hidden_size)

    def test_selects_only_each_samples_current_panorama(self):
        selected = self.model._current_panorama_hidden_tokens(
            hidden_states=self.hidden,
            input_ids=self.input_ids,
            image_grid_thw=self.image_grid_thw,
        )
        self.assertEqual(set(selected), {0, 1})
        torch.testing.assert_close(selected[0][0], self.hidden[0, 5:9])
        torch.testing.assert_close(selected[1][0], self.hidden[1, 2:10])
        self.assertEqual(selected[0][1].numel(), 4)
        self.assertEqual(selected[1][1].numel(), 8)

    def test_changes_only_three_s1_logits_and_leaves_stop_unchanged(self):
        labels = torch.full((2, 14), -100, dtype=torch.long)
        labels[0, 11] = PAQR_DEFAULT_ACTION_TOKEN_IDS[2]
        labels[1, 12] = PAQR_DEFAULT_STOP_TOKEN_ID
        logits = torch.randn(2, 14, self.vocab_size)
        original = logits.clone()

        output = self.model._apply_paqr_to_logits(
            logits=logits,
            hidden_states=self.hidden,
            input_ids=self.input_ids,
            labels=labels,
            attention_mask=torch.ones_like(self.input_ids),
            image_grid_thw=self.image_grid_thw,
            logits_to_keep=0,
        )
        difference = output - original
        changed = torch.nonzero(difference, as_tuple=False)
        expected_positions = {
            (0, 10, token_id) for token_id in PAQR_DEFAULT_ACTION_TOKEN_IDS
        } | {
            (1, 11, token_id) for token_id in PAQR_DEFAULT_ACTION_TOKEN_IDS
        }
        self.assertEqual({tuple(row.tolist()) for row in changed}, expected_positions)
        detached_difference = difference.detach()
        self.assertEqual(
            float(detached_difference[0, 10, PAQR_DEFAULT_STOP_TOKEN_ID]),
            0.0,
        )
        self.assertEqual(
            float(detached_difference[1, 11, PAQR_DEFAULT_STOP_TOKEN_ID]),
            0.0,
        )

    def test_zero_gate_preserves_routed_logits_and_loss_but_backpropagates(self):
        with torch.no_grad():
            self.model.paqr.raw_gate.zero_()
        labels = torch.full((2, 14), -100, dtype=torch.long)
        labels[0, 11] = PAQR_DEFAULT_ACTION_TOKEN_IDS[2]
        labels[1, 12] = PAQR_DEFAULT_ACTION_TOKEN_IDS[0]
        logits = self.model.lm_head(self.hidden)
        original = logits.detach().clone()

        output = self.model._apply_paqr_to_logits(
            logits=logits,
            hidden_states=self.hidden,
            input_ids=self.input_ids,
            labels=labels,
            attention_mask=torch.ones_like(self.input_ids),
            image_grid_thw=self.image_grid_thw,
            logits_to_keep=0,
        )
        self.assertTrue(torch.equal(output.detach(), original))
        targets = torch.tensor(
            [PAQR_DEFAULT_ACTION_TOKEN_IDS[2], PAQR_DEFAULT_ACTION_TOKEN_IDS[0]]
        )
        base_loss = nn.functional.cross_entropy(
            torch.stack((original[0, 10], original[1, 11])),
            targets,
        )
        routed_loss = nn.functional.cross_entropy(
            torch.stack((output[0, 10], output[1, 11])),
            targets,
        )
        self.assertEqual(float(routed_loss.detach()), float(base_loss))

        routed_loss.backward()
        self.assertIsNotNone(self.model.paqr.raw_gate.grad)
        self.assertGreater(abs(float(self.model.paqr.raw_gate.grad)), 0.0)

    def test_logit_slice_mapping_catches_missing_s1(self):
        positions = self.model._logit_source_positions(14, 1, torch.device("cpu"))
        torch.testing.assert_close(positions, torch.tensor([13]))
        labels = torch.full((2, 14), -100, dtype=torch.long)
        labels[:, 11] = PAQR_DEFAULT_ACTION_TOKEN_IDS[0]
        with self.assertRaisesRegex(AssertionError, "absent or duplicated"):
            self.model._apply_paqr_to_logits(
                logits=torch.randn(2, 1, self.vocab_size),
                hidden_states=self.hidden,
                input_ids=self.input_ids,
                labels=labels,
                attention_mask=torch.ones_like(self.input_ids),
                image_grid_thw=self.image_grid_thw,
                logits_to_keep=1,
            )


class PAQRTokenizerContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    def test_exact_first_action_ids(self):
        cfg = SimpleNamespace(model=paqr_config())
        validate_paqr_tokenizer(cfg, self.tokenizer)
        self.assertEqual(
            [self.tokenizer.encode(word, add_special_tokens=False)[0] for word in (
                "left",
                "forward",
                "right",
                "stop",
            )],
            [2282, 13048, 1246, 9215],
        )


if __name__ == "__main__":
    unittest.main()
