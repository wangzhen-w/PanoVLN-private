# PanoVGGT 集成说明

## Efficient-VLN

- Efficient-VLN 在 Qwen2.5-VL 之上使用 StreamVGGT 作为流式 3D 几何编码器。
- 真正有用的信号是 StreamVGGT 的 3D geometry latent tokens，而不是渲染出来的 depth map。
- 论文把这些几何 latent tokens 通过 2 层 MLP 映射到与 Qwen visual representation 一致的通道维度。
- 融合方式很简单：对齐后的 geometry representation 在进入 LLM 前，按元素加到 2D visual representation 上。
- 当前实现沿用这个模式：`geometry latent tokens -> MLP -> weighted residual addition`。

来源：https://arxiv.org/pdf/2512.10310

## JanusVLN

- JanusVLN 使用 VGGT 作为空间几何编码器，与 Qwen2.5-VL 的视觉语义编码器并行。
- 对当前集成最相关的是 VGGT encoder / fusion latent representation，而不是显式 depth、point cloud 或 camera-pose loss。
- JanusVLN 的 dual implicit memory 会保存 spatial-geometry 和 visual-semantic 两类 KV cache，并使用 initial window 和 sliding window。
- 当前 v1 集成不复制这套 memory 设计。它需要跨 step 状态、cache reset 语义、序列化训练，以及更侵入式的 generation/evaluation 改动。
- 当前 VLN 代码已经把历史 panoramic observations 喂给 Qwen，所以 PanoVGGT 只作为 current-frame geometry enhancer 使用。

来源：

- https://openreview.net/forum?id=RnuB0Nlbd5
- `/workspace/library/JanusVLN`

## PanoVGGT 本地结论

- 官方仓库：https://github.com/YijingGuo-June/PanoVGGT
- 本项目内置代码：`src/panovggt`
- 本地 checkpoint：`/workspace/code_dir/a_property/model/PanoVGGT/model.pt`
- 官方预处理 `panovggt.utils.basic.load_images_as_tensor` 会把全景图 resize 到 `518 x 1036`，保持 2:1 equirectangular aspect ratio。
- 默认配置使用 `patch_size=14`、`embed_dim=1024`、`num_register_tokens=5`，aggregator depth 为 `36`。
- `PanoVGGTModel.aggregator(images)` 返回 `([output], patch_start_idx, pos_2d)`。
- 对官方输入尺寸，原始 patch grid 是 `37 x 74`，返回 token 数是 `5 + 37 * 74 = 2743`。
- 集成时不能硬编码这个 grid。实现应该根据输入 tensor 和 aggregator metadata 推断 patch-grid shape，并校验它与返回 token 数一致。

## 集成决策

- 只使用 PanoVGGT current frame。
- 冻结 PanoVGGT，只训练一个小的 geometry MLP 和一个 bounded gate。
- 去掉 PanoVGGT register tokens，把原始 PanoVGGT latent grid 重采样到当前 Qwen merged visual grid，只把投影后的 residual 加到当前图像的 Qwen visual tokens 上。
- PanoVGGT 的 geometry scale 约为 28 pixels，Qwen3.5-VL merged visual scale 约为 32 pixels。这里通过 token-space resampling 对齐，而不是硬做 pixel-grid alignment。
- 保持现有 Qwen history-image prompt path 不变。
- 结构性 PanoVGGT 集成常量保留在代码里，不放进 YAML：
  - `PANOVGGT_AGGREGATOR_LAYER = -1`
  - `PANOVGGT_CONTEXT_DIM = 2048`
  - `PANOVGGT_MLP_HIDDEN_SIZE = 4096`
- config 里只保留清晰的运行时控制项：`panovggt_enabled`、`panovggt_checkpoint_path`、`panovggt_alpha_init`、`panovggt_alpha_max`。
- PanoVGGT Python package 和官方默认配置已经内置在 `src/panovggt` 下，所以打开 VLN repo 时不需要额外外部源码路径。
- 冻结的 PanoVGGT encoder 注册为 `model.panovggt`，因此会保存在最终 Hugging Face model 目录中。可训练的 geometry adapter 单独注册为 `model.panovggt_mlp`。

## Token 对齐

- PanoVGGT 和 Qwen 产生的 token grid 不一致。
- PanoVGGT 接收当前 raw panorama，resize 到 `[3, 518, 1036]`。在本地官方配置下，去掉 register tokens 后得到 `37 x 74` 的原始 geometry grid，但实现仍然从返回 tensor shape 动态推断。
- Qwen 接收正常选择的 history images 加上裁剪和 resize 后的当前图像。它的 visual output length 由 `image_grid_thw` 和 Qwen 的 `spatial_merge_size` 决定。
- 融合时，实现会从当前图像真实的 `image_grid_thw` 构建目标 Qwen merged grid，在 ERP latitude/longitude 空间把 PanoVGGT latent grid 采样到目标 grid，再把通道从 `2048` 投影到 Qwen visual hidden size，并按元素相加。
- 因此，PanoVGGT 的 `37 x 74` grid 可以融合到 Qwen 的目标 grid，例如 `12 x 30`，不需要假设两个模型的 patch size 相同。

## Gate 与归一化

- PanoVGGT latent tokens 和 Qwen visual tokens 来自不同 hidden space，所以 geometry MLP 在 2 层 MLP 前后都使用 RMSNorm。
- residual gate 是有界且为正的：

```python
alpha = alpha_max * sigmoid(raw_alpha)
```

- 参数名仍然是 `raw_alpha`；`alpha_init` 控制初始 residual weight，`alpha_max` 控制最大值。
- `raw_alpha` 使用 `alpha_init / alpha_max` 的 inverse sigmoid 初始化。这样做只是为了让 forward 时的实际值等于请求的初始 alpha。例如 `alpha_init=0.1`、`alpha_max=0.3` 时，目标 sigmoid 值是 `1/3`，所以 `raw_alpha` 初始化为 `logit(1/3)`。如果直接把 `raw_alpha` 初始化为 `0.1`，实际 alpha 会变成 `0.3 * sigmoid(0.1)`，约为 `0.157`，并不是 `0.1`。
- 当前实验设置：
  - PanoVGGT：`alpha_init=0.2`，`alpha_max=0.4`
  - ERP position MLP：`alpha_init=0.005`，`alpha_max=0.05`
- ERP 使用更小的 gate，因为它注入在 visual patch embedding 附近，会影响完整 vision tower。PanoVGGT 注入在 Qwen visual encoding 之后，所以 residual 可以更大。
