# PanoVGGT Integration Notes

## Efficient-VLN

- Efficient-VLN uses StreamVGGT as a streaming 3D geometry encoder on top of Qwen2.5-VL.
- The useful signal is StreamVGGT's 3D geometry latent tokens, not a rendered depth map.
- The paper maps these geometry latent tokens through a 2-layer MLP so their channel dimension matches the Qwen visual representation.
- Fusion is residual and simple: the aligned geometry representation is added element-wise to the 2D visual representation before the LLM consumes it.
- This is the pattern used for the current implementation: geometry latent tokens -> MLP -> weighted residual addition.

Source: https://arxiv.org/pdf/2512.10310

## JanusVLN

- JanusVLN uses VGGT as a spatial geometry encoder alongside Qwen2.5-VL's visual semantic encoder.
- The relevant feature for this integration is the VGGT encoder / fusion latent representation, not explicit depth, point cloud, or camera-pose losses.
- JanusVLN's dual implicit memory stores spatial-geometry and visual-semantic KV caches with initial and sliding windows.
- That memory design is intentionally not copied in this v1 integration. It would require cross-step state, cache reset semantics, sequence-aware training, and more invasive generation/evaluation changes.
- The current VLN code already feeds Qwen historical panoramic observations. PanoVGGT is therefore used only as a current-frame geometry enhancer.

Sources:
- https://openreview.net/forum?id=RnuB0Nlbd5
- `/workspace/library/JanusVLN`

## PanoVGGT Local Findings

- Official repository: https://github.com/YijingGuo-June/PanoVGGT
- Vendored code for this project: `src/panovggt`
- Local checkpoint: `/workspace/code_dir/a_property/model/PanoVGGT/model.pt`
- Official preprocessing in `panovggt.utils.basic.load_images_as_tensor` resizes panoramic images to `518 x 1036`, preserving the 2:1 equirectangular aspect ratio.
- Default config uses `patch_size=14`, `embed_dim=1024`, `num_register_tokens=5`, and aggregator depth `36`.
- `PanoVGGTModel.aggregator(images)` returns `([output], patch_start_idx, pos_2d)`.
- For the official input size, the raw patch grid is `37 x 74`, and the returned token count is `5 + 37 * 74 = 2743`.
- The integration must not hardcode this grid. It should infer patch-grid shape from the input tensor and aggregator metadata, then verify it matches the returned token count.

## Integration Decision

- Use PanoVGGT current frame only.
- Freeze PanoVGGT and train only a small geometry MLP plus a bounded gate.
- Remove PanoVGGT register tokens, resample the raw PanoVGGT latent grid to the current Qwen merged visual grid, and add the projected residual only to the current image's Qwen visual tokens.
- Resolve PanoVGGT's roughly 28-pixel geometry scale versus Qwen3.5-VL's roughly 32-pixel merged visual scale by token-space resampling, not by hard pixel-grid alignment.
- Keep the existing Qwen history-image prompt path unchanged.
- Keep structural PanoVGGT integration constants in code, not YAML:
  - `PANOVGGT_AGGREGATOR_LAYER = -1`
  - `PANOVGGT_CONTEXT_DIM = 2048`
  - `PANOVGGT_MLP_HIDDEN_SIZE = 4096`
- Keep only clear runtime controls in config: `panovggt_enabled`, `panovggt_checkpoint_path`, `panovggt_alpha_init`, and `panovggt_alpha_max`.
- The PanoVGGT Python package and official default config are vendored under `src/panovggt`, so no external source path is required when opening this VLN repo.
- The frozen PanoVGGT encoder is registered as `model.panovggt`, so it is saved inside the final Hugging Face model directory. The trainable geometry adapter is registered separately as `model.panovggt_mlp`.

## Token Alignment

- PanoVGGT and Qwen do not produce the same token grid.
- PanoVGGT receives the current raw panorama resized to `[3, 518, 1036]`. With the local official config this gives a raw geometry grid of `37 x 74` after removing register tokens, but the implementation still infers this from the returned tensor shape.
- Qwen receives its normal selected history images plus the cropped/resized current image. Its visual output length is derived from `image_grid_thw` after Qwen's `spatial_merge_size`.
- At fusion time, the implementation builds the target Qwen merged grid from the actual current image's `image_grid_thw`, samples the PanoVGGT latent grid onto that target grid in ERP latitude/longitude space, projects channels from `2048` to Qwen visual hidden size, and adds the result element-wise.
- This means a PanoVGGT `37 x 74` grid can be fused into a Qwen target such as `12 x 30` without assuming either model's patch size.

## Gate And Normalization

- PanoVGGT latent tokens and Qwen visual tokens come from different hidden spaces, so the geometry MLP uses RMSNorm before and after the 2-layer MLP.
- The residual gate is bounded and positive:

```python
alpha = alpha_max * sigmoid(raw_alpha)
```

- The parameter is still `raw_alpha`; `alpha_init` controls the starting residual weight and `alpha_max` caps the maximum.
- `raw_alpha` is initialized with the inverse sigmoid of `alpha_init / alpha_max`. This is only to make the actual forward-time value equal the requested initial alpha. For example, with `alpha_init=0.1` and `alpha_max=0.3`, the desired sigmoid value is `1/3`, so `raw_alpha` starts at `logit(1/3)`. If `raw_alpha` were initialized directly to `0.1`, the actual alpha would be `0.3 * sigmoid(0.1)`, about `0.157`, not `0.1`.
- Current experiment settings:
  - PanoVGGT: `alpha_init=0.2`, `alpha_max=0.4`
  - ERP position MLP: `alpha_init=0.005`, `alpha_max=0.05`
- ERP uses a smaller gate because it is injected near the visual patch embedding and affects the full vision tower. PanoVGGT is injected after Qwen visual encoding, so its residual can be larger.
