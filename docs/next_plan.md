# Next Plan: Panorama-Aware Position Encoding

## Current Version To Train First

Use the current top-level ERP MLP implementation as the main version:

- Dataset: EBS, `train_r2r_rxr_4action_stop_pad_ebs_event0p50_bg0p05_taildense4.jsonl`
- Model change: PanoVGGT-style spherical absolute position embedding.
- Formula per image patch:

```text
lat = (patch_y / patch_h - 0.5) * vertical_fov + center_latitude
lon = (patch_x / patch_w - 0.5) * 2pi
erp_feat = [sin(lat), cos(lat), sin(lon), cos(lon)]
patch_token = patch_token + alpha * MLP(erp_feat)
```

- MLP: `Linear(4, hidden) -> GELU -> Linear(hidden, vision_hidden)`.
- `alpha` initializes to `0.0`, so training starts exactly from the original Qwen behavior.
- The MLP is registered at the top-level model as `model.erp_position_mlp`, not under `visual`.
- Default scope is all panoramic image patch tokens, not only the current observation.
- Because VLN images are cropped by 20 degrees at both top and bottom, use `vertical_fov = 140 deg` and `center_latitude = 0 deg`.

This version has already passed a real 4-GPU smoke train on GPUs 0-3. The ERP gate was updated from `0.0` to a non-zero value, confirming that the module participates in forward/backward/optimizer.

## Immediate Ablations

Run these before moving to more invasive attention changes:

1. `no_erp`: original Qwen-VL training path without ERP MLP.
2. `erp_all_frame`: current implementation, ERP MLP on all panorama image patch tokens.
3. `erp_current_only`: set `erp_apply_to_current_only=True`, only current observation receives ERP MLP.

The expected main comparison is `no_erp` vs `erp_all_frame`. `erp_current_only` answers whether history memory images benefit from the same spherical prior.

## Next Version Candidate 1: Spherical RoPE

Goal: make the vision attention itself aware of spherical coordinates, while avoiding an explicit `N x N` relative bias matrix.

Basic idea:

```text
lat = (patch_y / patch_h - 0.5) * vertical_fov + center_latitude
lon = (patch_x / patch_w - 0.5) * 2pi
```

Then use `lat/lon` to rotate `q/k` in the vision transformer. A safer version is additive rather than replacement:

- Keep Qwen's original visual position embedding and RoPE.
- Add a small spherical rotary branch on a limited slice of `q/k` dimensions.
- Initialize its scale/gate to `0.0`, similar to the ERP MLP gate.

Potential feature choices:

```text
2-axis spherical RoPE:
  rotary axis 1: lat
  rotary axis 2: lon

3D direction variant:
  x = cos(lat) * sin(lon)
  y = sin(lat)
  z = cos(lat) * cos(lon)
```

Recommended first implementation:

- Do not replace Qwen's existing RoPE.
- Add `spherical_rope_alpha` initialized to `0.0`.
- Apply only inside the visual transformer attention.
- Start with all-frame scope, matching current ERP MLP.
- Keep the ERP MLP disabled in this ablation, so the gain/loss is attributable to spherical RoPE.

Pros:

- More structural than absolute ERP MLP.
- Does not require a full pairwise attention bias matrix.
- Can preserve flash attention compatibility if implemented through `q/k` rotation before attention.

Risks:

- Qwen's vision tower already has its own visual position mechanism.
- Incorrect dimension split or frequency scaling can hurt pretrained visual features.
- More invasive than ERP MLP because it touches attention internals.

## Next Version Candidate 2: Spherical Relative Attention Bias

Goal: directly bias attention logits using spherical distances and periodic longitude differences.

For patch pair `(i, j)`:

```text
delta_lon = atan2(sin(lon_i - lon_j), cos(lon_i - lon_j))
cos_d = sin(lat_i) * sin(lat_j)
      + cos(lat_i) * cos(lat_j) * cos(delta_lon)
d_sphere = arccos(clamp(cos_d, -1, 1))
attention_logit(i, j) += bias(i, j)
```

Possible bias parameterizations:

```text
bucketed:
  bias = table_head[lat_bucket, lon_bucket, distance_bucket]

mlp:
  bias = MLP_head([
      sin(delta_lat), cos(delta_lat),
      sin(delta_lon), cos(delta_lon),
      d_sphere
  ])
```

Recommended first implementation:

- Try bucketed bias first. It is easier to debug and more stable than an MLP over all pairs.
- Gate the entire bias with `spherical_bias_alpha = 0.0`.
- Apply only to image patch tokens inside each panorama at first.
- Avoid cross-image pairwise bias in the first version.
- Disable ERP MLP for this ablation.

Pros:

- Most explicit way to tell attention which patches are close on the sphere.
- Naturally handles ERP left/right seam through periodic `delta_lon`.
- Could help navigation if local spherical adjacency matters.

Risks:

- Pairwise bias is expensive. Current observations can produce around `24 x 60 = 1440` patch tokens, so a full bias matrix is large.
- Arbitrary attention bias may break optimized flash attention paths.
- More memory-sensitive than spherical RoPE.

## Recommended Experiment Order

1. Finish training the current `erp_all_frame` version.
2. Compare online SR against EBS baseline without ERP.
3. If ERP helps or is neutral, keep it as the main panorama-aware baseline.
4. Implement `erp_current_only` as a cheap ablation.
5. Implement spherical RoPE next, because it is more efficient than pairwise relative bias.
6. Implement spherical relative attention bias only if spherical RoPE is promising or if attention locality appears to be the limiting factor.

## Paper Framing

Current ERP MLP can be written as:

> Inspired by PanoVGGT, we add a spherical-aware absolute position embedding to panoramic visual patch tokens. Each patch center is represented by a periodic ERP coordinate feature `[sin(lat), cos(lat), sin(lon), cos(lon)]`, projected through a lightweight MLP and added to the patch representation with a learnable zero-initialized gate.

Spherical RoPE / relative bias can be positioned as follow-up stronger variants:

- ERP MLP: absolute spherical position prior.
- Spherical RoPE: relative spherical phase prior in `q/k`.
- Spherical relative bias: explicit pairwise spherical distance prior in attention logits.
