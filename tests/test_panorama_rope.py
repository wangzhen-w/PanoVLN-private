import copy
import json

import torch
from transformers import Qwen3_5Config
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5VisionConfig
from transformers.models.qwen3_5.modeling_qwen3_5 import Qwen3_5VisionModel

from src.qwen_vl.panorama_rope import (
    build_block_major_vision_coordinates,
    build_headwise_rolling_position_embeddings,
    ensure_panorama_rope_config,
    install_headwise_panorama_rope,
)


def _tiny_vision_model(attn_implementation: str = "eager") -> Qwen3_5VisionModel:
    config = Qwen3_5VisionConfig(
        depth=1,
        hidden_size=32,
        intermediate_size=64,
        num_heads=4,
        in_channels=3,
        patch_size=2,
        spatial_merge_size=2,
        temporal_patch_size=2,
        out_hidden_size=32,
        num_position_embeddings=64,
    )
    config._attn_implementation = attn_implementation
    return Qwen3_5VisionModel(config).eval()


def _stock_position_embeddings(model, grid_thw):
    rotary = model.rot_pos_emb(grid_thw)
    full_phase = torch.cat((rotary, rotary), dim=-1)
    return full_phase.cos(), full_phase.sin()


def test_block_major_coordinates_match_qwen_merger_order():
    grid_thw = torch.tensor([[1, 4, 6]], dtype=torch.long)
    rows, columns, widths = build_block_major_vision_coordinates(
        grid_thw,
        spatial_merge_size=2,
        device=torch.device("cpu"),
    )
    expected_coordinates = [
        (0, 0), (0, 1), (1, 0), (1, 1),
        (0, 2), (0, 3), (1, 2), (1, 3),
        (0, 4), (0, 5), (1, 4), (1, 5),
        (2, 0), (2, 1), (3, 0), (3, 1),
        (2, 2), (2, 3), (3, 2), (3, 3),
        (2, 4), (2, 5), (3, 4), (3, 5),
    ]
    assert list(zip(rows.tolist(), columns.tolist())) == expected_coordinates
    assert widths.tolist() == [6] * 24


def test_head_zero_is_exact_stock_anchor_and_vertical_phase_is_unchanged():
    model = _tiny_vision_model()
    grid_thw = torch.tensor([[1, 4, 6]], dtype=torch.long)
    stock_cos, stock_sin = _stock_position_embeddings(model, grid_thw)
    rolling_cos, rolling_sin = build_headwise_rolling_position_embeddings(model, grid_thw)

    torch.testing.assert_close(rolling_cos[:, 0], stock_cos, rtol=0, atol=0)
    torch.testing.assert_close(rolling_sin[:, 0], stock_sin, rtol=0, atol=0)

    axis_phase_width = model.rotary_pos_emb.inv_freq.numel()
    row_rotary_width = axis_phase_width
    torch.testing.assert_close(
        rolling_cos[:, :, :row_rotary_width],
        rolling_cos[:, :1, :row_rotary_width].expand_as(
            rolling_cos[:, :, :row_rotary_width]
        ),
        rtol=0,
        atol=0,
    )


def test_packed_grids_use_each_panorama_width_independently():
    model = _tiny_vision_model()
    packed_grid = torch.tensor([[1, 4, 6], [2, 2, 4]], dtype=torch.long)
    packed_cos, packed_sin = build_headwise_rolling_position_embeddings(model, packed_grid)

    individual = [
        build_headwise_rolling_position_embeddings(model, row.unsqueeze(0))
        for row in packed_grid
    ]
    expected_cos = torch.cat([item[0] for item in individual], dim=0)
    expected_sin = torch.cat([item[1] for item in individual], dim=0)
    torch.testing.assert_close(packed_cos, expected_cos, rtol=0, atol=0)
    torch.testing.assert_close(packed_sin, expected_sin, rtol=0, atol=0)


def test_rolling_moves_the_original_seam_to_local_distance_in_shifted_heads():
    model = _tiny_vision_model()
    grid_thw = torch.tensor([[1, 2, 8]], dtype=torch.long)
    num_heads = model.config.num_heads
    width = int(grid_thw[0, 2])

    head_ids = torch.arange(num_heads, dtype=torch.long)
    offsets = torch.div(width * head_ids, num_heads, rounding_mode="floor")
    left_edge = torch.remainder(torch.tensor(0.0) + offsets, width)
    right_edge = torch.remainder(torch.tensor(float(width - 1)) + offsets, width)
    signed_delta = left_edge - right_edge

    assert signed_delta[0].item() == -(width - 1)
    torch.testing.assert_close(signed_delta[1:], torch.ones_like(signed_delta[1:]))


def test_non_divisible_widths_receive_even_integer_head_origins():
    expected = {
        28: [0, 1, 3, 5, 7, 8, 10, 12, 14, 15, 17, 19, 21, 22, 24, 26],
        60: [0, 3, 7, 11, 15, 18, 22, 26, 30, 33, 37, 41, 45, 48, 52, 56],
    }
    head_ids = torch.arange(16, dtype=torch.long)
    for width, expected_offsets in expected.items():
        offsets = torch.div(width * head_ids, 16, rounding_mode="floor")
        assert offsets.tolist() == expected_offsets
        gaps = (torch.roll(offsets, shifts=-1) - offsets) % width
        assert int(gaps.max() - gaps.min()) <= 1


def test_panorama_rope_fields_are_checkpoint_serializable():
    config = Qwen3_5Config()
    config.panorama_rope_enabled = True
    ensure_panorama_rope_config(config)
    serialized = json.loads(config.to_json_string())
    assert serialized["panorama_rope_enabled"] is True
    assert serialized["panorama_rope_variant"] == "headwise_rolling"


def test_installation_adds_no_checkpoint_state_and_forward_backward_is_finite():
    torch.manual_seed(7)
    baseline = _tiny_vision_model()
    model = copy.deepcopy(baseline)
    state_before = {key: value.clone() for key, value in model.state_dict().items()}
    install_headwise_panorama_rope(model)
    install_headwise_panorama_rope(model)

    assert set(model.state_dict()) == set(state_before)
    for key, value in model.state_dict().items():
        torch.testing.assert_close(value, state_before[key], rtol=0, atol=0)

    grid_thw = torch.tensor([[1, 4, 6], [1, 2, 4]], dtype=torch.long)
    num_tokens = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())
    patch_volume = 3 * 2 * 2 * 2
    pixel_values = torch.randn(num_tokens, patch_volume, requires_grad=True)
    output = model(pixel_values, grid_thw=grid_thw, return_dict=True)

    assert output.last_hidden_state.shape == (num_tokens, 32)
    assert output.pooler_output.shape == (num_tokens // 4, 32)
    assert torch.isfinite(output.last_hidden_state).all()
    loss = output.pooler_output.float().square().mean()
    loss.backward()
    assert pixel_values.grad is not None
    assert torch.isfinite(pixel_values.grad).all()
    assert pixel_values.grad.abs().sum().item() > 0


def test_eager_and_sdpa_attention_backends_are_finite():
    grid_thw = torch.tensor([[1, 4, 6], [1, 2, 4]], dtype=torch.long)
    num_tokens = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())
    patch_volume = 3 * 2 * 2 * 2

    for attn_implementation in ("eager", "sdpa"):
        model = _tiny_vision_model(attn_implementation)
        install_headwise_panorama_rope(model)
        pixel_values = torch.randn(num_tokens, patch_volume, requires_grad=True)
        output = model(pixel_values, grid_thw=grid_thw, return_dict=True)
        loss = output.pooler_output.float().square().mean()
        loss.backward()

        assert torch.isfinite(output.last_hidden_state).all()
        assert pixel_values.grad is not None
        assert torch.isfinite(pixel_values.grad).all()


def test_patched_forward_preserves_transformers_output_contract():
    model = _tiny_vision_model()
    install_headwise_panorama_rope(model)
    grid_thw = torch.tensor([[1, 4, 6]], dtype=torch.long)
    pixel_values = torch.randn(24, 24)

    output = model(
        pixel_values,
        grid_thw=grid_thw,
        return_dict=True,
        output_hidden_states=True,
        output_attentions=True,
    )
    assert len(output.hidden_states) == 2
    assert len(output.attentions) == 1
    assert isinstance(model(pixel_values, grid_thw=grid_thw, return_dict=False), tuple)
