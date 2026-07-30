import math
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer

from src.qwen_vl.modeling_qwen3_5 import (
    ACTION_BEARING_TURN_ANGLE_DEG,
    PAQR_DEFAULT_ACTION_TOKEN_IDS,
    PAQR_DEFAULT_STOP_TOKEN_ID,
    PanoramicActionQueryReadout,
    Qwen3_5ForConditionalGenerationForPanoVLN,
)
from src.train.utils import validate_paqr_tokenizer


MODEL_PATH = "/workspace/code/a_property/model/Qwen3.5-4B"


def paqr_config(hidden_size=8, reader_dim=4, **overrides):
    values = {
        "paqr_enabled": True,
        "paqr_variant": "full",
        "paqr_action_token_ids": list(PAQR_DEFAULT_ACTION_TOKEN_IDS),
        "paqr_stop_token_id": PAQR_DEFAULT_STOP_TOKEN_ID,
        "paqr_reader_dim": reader_dim,
        "paqr_prior_init": 0.02,
        "paqr_prior_max": 0.25,
        "paqr_first_action_only": True,
        "text_config": SimpleNamespace(
            hidden_size=hidden_size,
            initializer_range=0.02,
        ),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def unit_rms(hidden):
    hidden = hidden.float()
    return hidden * torch.rsqrt(
        hidden.square().mean(dim=-1, keepdim=True) + 1e-6
    )


class FullPAQRModuleTest(unittest.TestCase):
    def test_formula_shapes_and_manual_oracle(self):
        torch.manual_seed(3)
        hidden_size = 5
        reader_dim = 3
        module = PanoramicActionQueryReadout(
            paqr_config(hidden_size=hidden_size, reader_dim=reader_dim)
        )
        with torch.no_grad():
            module.output_proj.weight.copy_(
                torch.tensor([[0.20, -0.10, 0.05]])
            )

        panorama = torch.randn(7, hidden_size)
        decision = torch.randn(hidden_size)
        yaw = torch.linspace(-math.pi, math.pi, 7)
        lm_head = torch.randn(14000, hidden_size)

        queries, evidence, attention, content, prior = module.compute_evidence(
            panorama,
            decision,
            yaw,
            lm_head,
        )
        self.assertEqual(tuple(queries.shape), (3, reader_dim))
        self.assertEqual(tuple(evidence.shape), (3, reader_dim))
        self.assertEqual(tuple(attention.shape), (3, 7))
        self.assertEqual(tuple(content.shape), (3, 7))
        self.assertEqual(tuple(prior.shape), (3, 7))

        action_ids = torch.tensor(PAQR_DEFAULT_ACTION_TOKEN_IDS)
        action_input = unit_rms(lm_head.index_select(0, action_ids))
        decision_input = unit_rms(decision)
        panorama_input = unit_rms(panorama)
        expected_queries = F.layer_norm(
            F.linear(decision_input, module.state_proj.weight).unsqueeze(0)
            + F.linear(action_input, module.action_proj.weight),
            (reader_dim,),
            module.query_norm.weight,
            module.query_norm.bias,
            module.query_norm.eps,
        )
        expected_keys = F.linear(panorama_input, module.key_proj.weight)
        expected_values = F.linear(panorama_input, module.value_proj.weight)
        expected_content = (
            expected_queries.float() @ expected_keys.float().transpose(0, 1)
        ) / math.sqrt(reader_dim)
        expected_prior = module.circular_prior(yaw)
        expected_attention = torch.softmax(
            expected_content + expected_prior,
            dim=-1,
        )
        expected_evidence = expected_attention @ expected_values
        expected_delta = F.linear(
            expected_queries * expected_evidence,
            module.output_proj.weight,
        ).squeeze(-1)

        torch.testing.assert_close(queries, expected_queries)
        torch.testing.assert_close(content, expected_content)
        torch.testing.assert_close(prior, expected_prior)
        torch.testing.assert_close(attention, expected_attention)
        torch.testing.assert_close(evidence, expected_evidence)
        torch.testing.assert_close(
            module(panorama, decision, yaw, lm_head),
            expected_delta,
        )
        torch.testing.assert_close(
            attention.sum(dim=-1),
            torch.ones(3),
            atol=1e-6,
            rtol=0,
        )

    def test_parameter_and_checkpoint_state_key_contract(self):
        hidden_size = 11
        reader_dim = 7
        module = PanoramicActionQueryReadout(
            paqr_config(hidden_size=hidden_size, reader_dim=reader_dim)
        )
        expected_names = {
            "state_proj.weight",
            "action_proj.weight",
            "key_proj.weight",
            "value_proj.weight",
            "query_norm.weight",
            "query_norm.bias",
            "output_proj.weight",
            "raw_prior_scale",
        }
        parameters = dict(module.named_parameters())
        self.assertEqual(set(parameters), expected_names)
        self.assertEqual(set(module.state_dict()), expected_names)
        self.assertEqual(
            set(Qwen3_5ForConditionalGenerationForPanoVLN.PAQR_STATE_KEYS),
            {f"paqr.{name}" for name in expected_names},
        )
        expected_numel = 4 * hidden_size * reader_dim + 3 * reader_dim + 3
        self.assertEqual(
            sum(parameter.numel() for parameter in parameters.values()),
            expected_numel,
        )
        self.assertTrue(all(parameter.requires_grad for parameter in parameters.values()))

    def test_zero_output_is_exact_identity_and_only_output_learns_first(self):
        torch.manual_seed(5)
        module = PanoramicActionQueryReadout(
            paqr_config(hidden_size=8, reader_dim=4)
        )
        panorama = torch.randn(9, 8, requires_grad=True)
        decision = torch.randn(8, requires_grad=True)
        yaw = torch.linspace(-math.pi, math.pi, 9)
        lm_head = torch.randn(14000, 8, requires_grad=True)

        self.assertTrue(torch.equal(
            module.output_proj.weight,
            torch.zeros_like(module.output_proj.weight),
        ))
        delta = module(panorama, decision, yaw, lm_head)
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        (delta * torch.tensor([1.0, -0.7, 0.3])).sum().backward()

        output_grad = module.output_proj.weight.grad
        self.assertIsNotNone(output_grad)
        self.assertTrue(torch.isfinite(output_grad).all())
        self.assertGreater(float(output_grad.abs().sum()), 0.0)

        for name, parameter in module.named_parameters():
            if name == "output_proj.weight":
                continue
            if parameter.grad is not None:
                self.assertTrue(
                    torch.equal(parameter.grad, torch.zeros_like(parameter.grad)),
                    msg=f"{name} must not learn before zero-initialized W_o moves",
                )
        for hidden in (panorama, decision):
            self.assertIsNotNone(hidden.grad)
            self.assertTrue(torch.equal(hidden.grad, torch.zeros_like(hidden.grad)))
        self.assertIsNone(lm_head.grad)

    def test_zero_output_identity_survives_bf16_cast(self):
        torch.manual_seed(7)
        module = PanoramicActionQueryReadout(
            paqr_config(hidden_size=8, reader_dim=4)
        ).to(dtype=torch.bfloat16)
        panorama = torch.randn(8, 8, dtype=torch.bfloat16)
        decision = torch.randn(8, dtype=torch.bfloat16)
        yaw = torch.linspace(-math.pi, math.pi, 8)
        lm_head = torch.randn(14000, 8, dtype=torch.bfloat16)

        delta = module(panorama, decision, yaw, lm_head)
        self.assertEqual(delta.dtype, torch.bfloat16)
        self.assertTrue(torch.equal(delta, torch.zeros_like(delta)))
        delta.sum().backward()
        self.assertIsNotNone(module.output_proj.weight.grad)
        self.assertTrue(torch.isfinite(module.output_proj.weight.grad).all())

    def test_nonzero_output_propagates_to_full_reader_but_not_lm_head_rows(self):
        torch.manual_seed(11)
        module = PanoramicActionQueryReadout(
            paqr_config(hidden_size=8, reader_dim=5)
        )
        with torch.no_grad():
            module.output_proj.weight.fill_(0.1)
        panorama = torch.randn(13, 8, requires_grad=True)
        decision = torch.randn(8, requires_grad=True)
        yaw = torch.linspace(-math.pi, math.pi, 13)
        lm_head = torch.randn(14000, 8, requires_grad=True)

        delta = module(panorama, decision, yaw, lm_head)
        (delta * torch.tensor([1.0, -0.6, 0.25])).sum().backward()

        for name, parameter in module.named_parameters():
            self.assertIsNotNone(parameter.grad, msg=f"{name} has no gradient")
            self.assertTrue(
                torch.isfinite(parameter.grad).all(),
                msg=f"{name} has a non-finite gradient",
            )
            self.assertGreater(
                float(parameter.grad.abs().sum()),
                0.0,
                msg=f"{name} has an all-zero gradient after W_o is nonzero",
            )
        for name, hidden in (("panorama", panorama), ("decision", decision)):
            self.assertIsNotNone(hidden.grad, msg=f"{name} has no gradient")
            self.assertTrue(torch.isfinite(hidden.grad).all())
            self.assertGreater(float(hidden.grad.abs().sum()), 0.0)
        self.assertIsNone(
            lm_head.grad,
            "The auxiliary action-semantics path must detach live LM-head rows",
        )

    def test_queries_are_conditioned_on_decision_state_and_action_semantics(self):
        torch.manual_seed(13)
        module = PanoramicActionQueryReadout(
            paqr_config(hidden_size=8, reader_dim=5)
        )
        with torch.no_grad():
            module.output_proj.weight.fill_(0.1)
        panorama = torch.randn(10, 8)
        decision = torch.randn(8)
        yaw = torch.linspace(-math.pi, math.pi, 10)
        lm_head = torch.randn(14000, 8)

        before_q, _, _, _, _ = module.compute_evidence(
            panorama,
            decision,
            yaw,
            lm_head,
        )
        before_delta = module(panorama, decision, yaw, lm_head)

        changed_decision = decision.roll(1) + torch.linspace(-0.5, 0.5, 8)
        state_q, _, _, _, _ = module.compute_evidence(
            panorama,
            changed_decision,
            yaw,
            lm_head,
        )
        state_delta = module(panorama, changed_decision, yaw, lm_head)
        self.assertTrue(torch.all((state_q - before_q).abs().sum(dim=-1) > 0))
        self.assertFalse(torch.equal(state_delta, before_delta))

        changed_lm_head = lm_head.clone()
        changed_lm_head[PAQR_DEFAULT_ACTION_TOKEN_IDS[0]] = torch.randn(8) * 4.0
        action_q, _, _, _, _ = module.compute_evidence(
            panorama,
            decision,
            yaw,
            changed_lm_head,
        )
        action_delta = module(panorama, decision, yaw, changed_lm_head)
        self.assertFalse(torch.equal(action_q[0], before_q[0]))
        torch.testing.assert_close(action_q[1:], before_q[1:], atol=0, rtol=0)
        self.assertNotEqual(
            float(action_delta[0].detach()),
            float(before_delta[0].detach()),
        )
        torch.testing.assert_close(
            action_delta[1:],
            before_delta[1:],
            atol=0,
            rtol=0,
        )

    def test_prior_is_weak_bounded_periodic_and_directional(self):
        module = PanoramicActionQueryReadout(
            paqr_config(hidden_size=8, reader_dim=4)
        )
        turn = math.radians(ACTION_BEARING_TURN_ANGLE_DEG)
        yaw = torch.tensor([-math.pi, -turn, 0.0, turn, math.pi])
        prior = module.circular_prior(yaw)

        torch.testing.assert_close(
            module.prior_scale,
            torch.full((3,), module.prior_init),
            atol=1e-7,
            rtol=0,
        )
        self.assertLessEqual(
            float(prior.detach().abs().max()),
            module.prior_max + 1e-7,
        )
        torch.testing.assert_close(prior[:, 0], prior[:, -1], atol=1e-7, rtol=0)
        torch.testing.assert_close(
            module.circular_prior(yaw + 2.0 * math.pi),
            prior,
            atol=1e-7,
            rtol=0,
        )
        self.assertEqual(int(prior[0].argmax()), 1)
        self.assertEqual(int(prior[1].argmax()), 2)
        self.assertEqual(int(prior[2].argmax()), 3)


class FullPAQRRoutingTest(unittest.TestCase):
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
        self.model.paqr = PanoramicActionQueryReadout(
            paqr_config(hidden_size=self.hidden_size, reader_dim=4)
        )
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

    def test_routes_correct_decision_hidden_and_changes_only_three_logits(self):
        labels = torch.full((2, 14), -100, dtype=torch.long)
        labels[0, 11] = PAQR_DEFAULT_ACTION_TOKEN_IDS[2]
        labels[1, 12] = PAQR_DEFAULT_STOP_TOKEN_ID
        logits = torch.randn(2, 14, self.vocab_size)
        original = logits.clone()
        calls = []

        def fake_reader(**kwargs):
            calls.append({
                key: value.detach().clone()
                for key, value in kwargs.items()
                if key != "lm_head_weight"
            })
            return kwargs["panorama_hidden"].new_tensor([0.1, -0.2, 0.3])

        with mock.patch.object(
            self.model.paqr,
            "forward",
            side_effect=fake_reader,
        ):
            output = self.model._apply_paqr_to_logits(
                logits=logits,
                hidden_states=self.hidden,
                input_ids=self.input_ids,
                labels=labels,
                attention_mask=torch.ones_like(self.input_ids),
                image_grid_thw=self.image_grid_thw,
                logits_to_keep=0,
            )

        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(calls[0]["decision_hidden"], self.hidden[0, 10])
        torch.testing.assert_close(calls[1]["decision_hidden"], self.hidden[1, 11])
        torch.testing.assert_close(calls[0]["panorama_hidden"], self.hidden[0, 5:9])
        torch.testing.assert_close(calls[1]["panorama_hidden"], self.hidden[1, 2:10])

        difference = output - original
        changed = torch.nonzero(difference, as_tuple=False)
        expected_positions = {
            (0, 10, token_id) for token_id in PAQR_DEFAULT_ACTION_TOKEN_IDS
        } | {
            (1, 11, token_id) for token_id in PAQR_DEFAULT_ACTION_TOKEN_IDS
        }
        self.assertEqual({tuple(row.tolist()) for row in changed}, expected_positions)
        self.assertEqual(
            float(difference[0, 10, PAQR_DEFAULT_STOP_TOKEN_ID]),
            0.0,
        )
        self.assertEqual(
            float(difference[1, 11, PAQR_DEFAULT_STOP_TOKEN_ID]),
            0.0,
        )

    def test_zero_output_preserves_routed_logits_and_loss_but_trains_output(self):
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
        base_loss = F.cross_entropy(
            torch.stack((original[0, 10], original[1, 11])),
            targets,
        )
        routed_loss = F.cross_entropy(
            torch.stack((output[0, 10], output[1, 11])),
            targets,
        )
        self.assertEqual(float(routed_loss.detach()), float(base_loss))

        routed_loss.backward()
        gradient = self.model.paqr.output_proj.weight.grad
        self.assertIsNotNone(gradient)
        self.assertTrue(torch.isfinite(gradient).all())
        self.assertGreater(float(gradient.abs().sum()), 0.0)

    def test_inference_uses_last_active_hidden_as_decision_state(self):
        attention_mask = torch.zeros_like(self.input_ids)
        attention_mask[0, :11] = 1
        attention_mask[1, :12] = 1
        calls = []

        def fake_reader(**kwargs):
            calls.append(kwargs["decision_hidden"].detach().clone())
            return kwargs["panorama_hidden"].new_zeros(3)

        with mock.patch.object(
            self.model.paqr,
            "forward",
            side_effect=fake_reader,
        ):
            self.model._apply_paqr_to_logits(
                logits=torch.randn(2, 14, self.vocab_size),
                hidden_states=self.hidden,
                input_ids=self.input_ids,
                labels=None,
                attention_mask=attention_mask,
                image_grid_thw=self.image_grid_thw,
                logits_to_keep=0,
            )

        self.assertEqual(len(calls), 2)
        torch.testing.assert_close(calls[0], self.hidden[0, 10])
        torch.testing.assert_close(calls[1], self.hidden[1, 11])

    def test_logit_slice_mapping_catches_missing_first_action_position(self):
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
            [
                self.tokenizer.encode(word, add_special_tokens=False)[0]
                for word in ("left", "forward", "right", "stop")
            ],
            [2282, 13048, 1246, 9215],
        )


if __name__ == "__main__":
    unittest.main()
