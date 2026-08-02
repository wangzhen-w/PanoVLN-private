"""Panorama-aware RoPE rolling for the Qwen3.5 vision tower.

The implementation follows the multi-origin idea from PanoSplatt3R: every
attention head keeps Qwen's pretrained rotary frequencies, but places the
horizontal branch cut at a different longitude.  The vertical rotary phases,
learned absolute position embedding, patch order, merger, and language-model
RoPE are left unchanged.

This module intentionally patches only model instances created by the local
PanoVLN wrapper.  It does not modify the installed Transformers package and it
does not add trainable parameters or checkpoint tensors.
"""

from types import MethodType

import torch
import torch.nn.functional as F
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen3_5


PANORAMA_ROPE_VARIANT = "headwise_rolling"


def ensure_panorama_rope_config(config) -> None:
    """Populate and validate the checkpoint-persistent panorama RoPE fields."""
    if not hasattr(config, "panorama_rope_enabled"):
        config.panorama_rope_enabled = False
    if not hasattr(config, "panorama_rope_variant"):
        config.panorama_rope_variant = PANORAMA_ROPE_VARIANT

    variant = str(config.panorama_rope_variant).lower()
    if variant != PANORAMA_ROPE_VARIANT:
        raise ValueError(
            "Only panorama_rope_variant='headwise_rolling' is supported, "
            f"got {config.panorama_rope_variant!r}"
        )
    config.panorama_rope_variant = variant


def _validated_grid_thw_list(
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
) -> list[tuple[int, int, int]]:
    if grid_thw is None or grid_thw.ndim != 2 or int(grid_thw.shape[1]) != 3:
        shape = None if grid_thw is None else tuple(grid_thw.shape)
        raise ValueError(f"grid_thw must have shape [num_items, 3], got {shape}")

    grid_thw_list = [tuple(int(value) for value in item) for item in grid_thw.tolist()]
    if not grid_thw_list:
        raise ValueError("grid_thw must contain at least one image")
    for num_frames, height, width in grid_thw_list:
        if num_frames <= 0 or height <= 0 or width <= 0:
            raise ValueError(
                "grid_thw entries must be positive, "
                f"got {(num_frames, height, width)}"
            )
        if height % spatial_merge_size != 0 or width % spatial_merge_size != 0:
            raise ValueError(
                "Vision grid height and width must be divisible by spatial_merge_size, "
                f"got grid={(num_frames, height, width)}, merge={spatial_merge_size}"
            )
    return grid_thw_list


def build_block_major_vision_coordinates(
    grid_thw: torch.Tensor,
    *,
    spatial_merge_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return row, column, and per-token width in Qwen's pre-merger order.

    Qwen groups tokens by merger block before the vision transformer.  A plain
    row-major meshgrid would therefore attach phases to the wrong patch tokens.
    """
    grid_thw_list = _validated_grid_thw_list(
        grid_thw,
        spatial_merge_size=spatial_merge_size,
    )

    row_chunks = []
    column_chunks = []
    width_chunks = []
    for num_frames, height, width in grid_thw_list:
        merged_height = height // spatial_merge_size
        merged_width = width // spatial_merge_size

        block_rows = torch.arange(merged_height, device=device, dtype=torch.long)
        block_columns = torch.arange(merged_width, device=device, dtype=torch.long)
        intra_rows = torch.arange(spatial_merge_size, device=device, dtype=torch.long)
        intra_columns = torch.arange(spatial_merge_size, device=device, dtype=torch.long)

        row_ids = block_rows[:, None, None, None] * spatial_merge_size
        row_ids = row_ids + intra_rows[None, None, :, None]
        row_ids = row_ids.expand(
            merged_height,
            merged_width,
            spatial_merge_size,
            spatial_merge_size,
        ).reshape(-1)

        column_ids = block_columns[None, :, None, None] * spatial_merge_size
        column_ids = column_ids + intra_columns[None, None, None, :]
        column_ids = column_ids.expand(
            merged_height,
            merged_width,
            spatial_merge_size,
            spatial_merge_size,
        ).reshape(-1)

        if num_frames > 1:
            row_ids = row_ids.repeat(num_frames)
            column_ids = column_ids.repeat(num_frames)

        row_chunks.append(row_ids)
        column_chunks.append(column_ids)
        width_chunks.append(torch.full_like(column_ids, width))

    row_ids = torch.cat(row_chunks, dim=0)
    column_ids = torch.cat(column_chunks, dim=0)
    token_widths = torch.cat(width_chunks, dim=0)

    expected_tokens = sum(t * h * w for t, h, w in grid_thw_list)
    if int(row_ids.numel()) != expected_tokens:
        raise AssertionError(
            "Panorama RoPE coordinate count does not match grid_thw: "
            f"coordinates={int(row_ids.numel())}, expected={expected_tokens}"
        )
    return row_ids, column_ids, token_widths


def build_headwise_rolling_position_embeddings(
    vision_model,
    grid_thw: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build head-specific cos/sin tensors with distributed horizontal seams.

    For head ``m`` and a panorama of token width ``W``, the horizontal
    coordinate is ``(column + floor(W * m / num_heads)) mod W``.  The original
    implementation rolls an integer coordinate grid and assumes a convenient
    width.  Using ``floor(W*m/M)`` is its evenly spaced generalization for our
    non-divisible widths (W=28 for history and W=60 for the current image), and
    keeps every rotary lookup on Qwen's pretrained integer position lattice.
    """
    spatial_merge_size = int(vision_model.spatial_merge_size)
    num_heads = int(vision_model.config.num_heads)
    inv_freq = vision_model.rotary_pos_emb.inv_freq
    if inv_freq.ndim != 1:
        raise AssertionError(f"Expected 1D vision RoPE frequencies, got {tuple(inv_freq.shape)}")

    row_ids, column_ids, token_widths = build_block_major_vision_coordinates(
        grid_thw,
        spatial_merge_size=spatial_merge_size,
        device=inv_freq.device,
    )
    head_ids = torch.arange(num_heads, device=inv_freq.device, dtype=torch.long)
    head_offsets = torch.div(
        token_widths[:, None] * head_ids[None, :],
        num_heads,
        rounding_mode="floor",
    )
    rolled_columns = torch.remainder(
        column_ids[:, None] + head_offsets,
        token_widths[:, None],
    )

    phase_dtype = inv_freq.dtype
    row_positions = row_ids.to(dtype=phase_dtype)
    rolled_columns = rolled_columns.to(dtype=phase_dtype)
    row_phases = row_positions[:, None, None] * inv_freq[None, None, :]
    row_phases = row_phases.expand(-1, num_heads, -1)
    column_phases = rolled_columns[:, :, None] * inv_freq[None, None, :]
    rotary_phases = torch.cat((row_phases, column_phases), dim=-1)
    full_phases = torch.cat((rotary_phases, rotary_phases), dim=-1)

    expected_head_dim = int(vision_model.config.hidden_size) // num_heads
    if tuple(full_phases.shape[1:]) != (num_heads, expected_head_dim):
        raise AssertionError(
            "Panorama RoPE phase shape does not match vision attention heads: "
            f"phases={tuple(full_phases.shape)}, "
            f"expected=[tokens, {num_heads}, {expected_head_dim}]"
        )
    return full_phases.cos(), full_phases.sin()


def apply_headwise_vision_rotary(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply already head-shaped vision rotary embeddings to Q and K."""
    if cos.ndim != 3 or sin.ndim != 3:
        raise ValueError(
            "Head-wise panorama RoPE expects cos/sin shaped [tokens, heads, head_dim], "
            f"got cos={tuple(cos.shape)}, sin={tuple(sin.shape)}"
        )
    if tuple(query_states.shape) != tuple(cos.shape) or tuple(key_states.shape) != tuple(cos.shape):
        raise ValueError(
            "Panorama RoPE tensor shape mismatch: "
            f"query={tuple(query_states.shape)}, key={tuple(key_states.shape)}, cos={tuple(cos.shape)}"
        )

    query_dtype = query_states.dtype
    key_dtype = key_states.dtype
    query_float = query_states.float()
    key_float = key_states.float()
    cos_float = cos.float()
    sin_float = sin.float()
    query_embed = query_float * cos_float + qwen3_5.rotate_half(query_float) * sin_float
    key_embed = key_float * cos_float + qwen3_5.rotate_half(key_float) * sin_float
    return query_embed.to(query_dtype), key_embed.to(key_dtype)


def _attention_forward_with_headwise_rope(
    self,
    hidden_states: torch.Tensor,
    cu_seqlens: torch.Tensor,
    rotary_pos_emb: torch.Tensor | None = None,
    position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
    **kwargs,
) -> torch.Tensor:
    del rotary_pos_emb
    if position_embeddings is None:
        raise ValueError("Vision attention requires precomputed position_embeddings")
    cos, sin = position_embeddings
    if cos.ndim != 3:
        return self._panorama_rope_original_forward(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    seq_length = hidden_states.shape[0]
    query_states, key_states, value_states = (
        self.qkv(hidden_states)
        .reshape(seq_length, 3, self.num_heads, -1)
        .permute(1, 0, 2, 3)
        .unbind(0)
    )
    query_states, key_states = apply_headwise_vision_rotary(
        query_states,
        key_states,
        cos,
        sin,
    )

    query_states = query_states.transpose(0, 1).unsqueeze(0)
    key_states = key_states.transpose(0, 1).unsqueeze(0)
    value_states = value_states.transpose(0, 1).unsqueeze(0)

    attention_interface = qwen3_5.ALL_ATTENTION_FUNCTIONS.get_interface(
        self.config._attn_implementation,
        qwen3_5.eager_attention_forward,
    )
    if qwen3_5.is_flash_attention_requested(self.config):
        max_seqlen = (cu_seqlens[1:] - cu_seqlens[:-1]).max()
        attn_output, _ = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            cu_seq_lens_q=cu_seqlens,
            cu_seq_lens_k=cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
            is_causal=False,
            **kwargs,
        )
    else:
        lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        splits = [
            torch.split(tensor, lengths.tolist(), dim=2)
            for tensor in (query_states, key_states, value_states)
        ]
        attn_outputs = [
            attention_interface(
                self,
                query,
                key,
                value,
                attention_mask=None,
                scaling=self.scaling,
                dropout=0.0 if not self.training else self.attention_dropout,
                is_causal=False,
                **kwargs,
            )[0]
            for query, key, value in zip(*splits)
        ]
        attn_output = torch.cat(attn_outputs, dim=1)

    attn_output = attn_output.reshape(seq_length, -1).contiguous()
    return self.proj(attn_output)


@qwen3_5.merge_with_config_defaults
@qwen3_5.capture_outputs
def _vision_forward_with_headwise_rope(
    self,
    hidden_states: torch.Tensor,
    grid_thw: torch.Tensor,
    **kwargs,
):
    hidden_states = self.patch_embed(hidden_states)
    hidden_states = hidden_states + self.fast_pos_embed_interpolate(grid_thw)
    position_embeddings = build_headwise_rolling_position_embeddings(self, grid_thw)

    seq_len, _ = hidden_states.size()
    hidden_states = hidden_states.reshape(seq_len, -1)
    if int(position_embeddings[0].shape[0]) != seq_len:
        raise AssertionError(
            "Panorama RoPE token count does not match patch embeddings: "
            f"rope={int(position_embeddings[0].shape[0])}, patches={seq_len}"
        )

    cu_seqlens = torch.repeat_interleave(
        grid_thw[:, 1] * grid_thw[:, 2],
        grid_thw[:, 0],
    ).cumsum(
        dim=0,
        dtype=grid_thw.dtype if torch.jit.is_tracing() else torch.int32,
    )
    cu_seqlens = F.pad(cu_seqlens, (1, 0), value=0)

    for block in self.blocks:
        hidden_states = block(
            hidden_states,
            cu_seqlens=cu_seqlens,
            position_embeddings=position_embeddings,
            **kwargs,
        )

    merged_hidden_states = self.merger(hidden_states)
    return qwen3_5.BaseModelOutputWithPooling(
        last_hidden_state=hidden_states,
        pooler_output=merged_hidden_states,
    )


def install_headwise_panorama_rope(vision_model) -> None:
    """Install the fixed rolling implementation on one Qwen vision instance."""
    if bool(getattr(vision_model, "_panorama_rope_installed", False)):
        return
    if not isinstance(vision_model, qwen3_5.Qwen3_5VisionModel):
        raise TypeError(
            "Panorama RoPE requires Qwen3_5VisionModel, "
            f"got {type(vision_model)!r}"
        )

    expected_head_dim = int(vision_model.config.hidden_size) // int(vision_model.config.num_heads)
    frequency_dim = int(vision_model.rotary_pos_emb.inv_freq.numel())
    if expected_head_dim != frequency_dim * 4:
        raise ValueError(
            "Unsupported Qwen vision RoPE layout: "
            f"head_dim={expected_head_dim}, inv_freq_dim={frequency_dim}"
        )

    vision_model._panorama_rope_original_forward = vision_model.forward
    for block in vision_model.blocks:
        attention = block.attn
        if not hasattr(attention, "_panorama_rope_original_forward"):
            attention._panorama_rope_original_forward = attention.forward
            attention.forward = MethodType(_attention_forward_with_headwise_rope, attention)

    vision_model.forward = MethodType(_vision_forward_with_headwise_rope, vision_model)
    vision_model._panorama_rope_installed = True
