# 全景感知空间定位（Panorama-Aware Spatial Grounding）

## 目标

目标是验证 **panorama-aware spatial grounding 是否能进一步提升 VLN action prediction**。

这个目标刻意比“验证原始 `sin/cos` ERP position embedding 是否有效”更宽。ERP-derived angular features 只是暴露 panorama geometry 的一种轻量实现方式。真正的问题是：模型是否能从与 VLN 动作空间对齐的 current-view spatial grounding 中获益。

## 动机

PanoVGGT 实验证明，geometry-aware current-view information 对这个任务有用。但是，把简单 ERP MLP 插在 patch embedding 后没有带来明确增益，而且 checkpoint 对比中它的可训练 gate 几乎没有移动。这说明失败更可能来自具体注入设计，而不是 VLN 中不存在空间信息价值。

对 VLN action prediction 来说，模型需要的不只是绝对 patch location。它更需要 action-relative information：

- 当前视野中哪些区域对应 front、left、right、back；
- 哪些区域对应 left 15 degrees、right 15 degrees 这类临近转向方向；
- 哪些 visual tokens 位于 horizon 附近，因为导航相关物体和可通行区域通常集中在这里；
- panorama layout 应该如何调制当前 RGB features，使 action decoder 更容易预测 `forward`、`left`、`right` 或 `stop`。

第一轮只把 spatial grounding 注入当前观测。history-frame spatial grounding 等 current-view grounding 被证明有效后再考虑。

## 与 PanoVGGT 的关系

PanoVGGT 和 ERP-derived spatial features 不应该在同一个 fusion point 都实现为 additive content delta。

推荐职责分离：

- PanoVGGT 继续作为 Qwen visual encoding 和 visual merger 之后的 content/geometry residual branch。
- Panorama-aware spatial grounding 应该作为 modulation 或 attention adaptation，而不是另一个 learned content residual。

这种分离让组合实验更容易解释：

- PanoVGGT 回答：额外的 visual geometry content 是否有帮助？
- Spatial grounding 回答：action-relative panorama structure 是否能帮助模型更好地使用当前 RGB features？

## 共享方向特征

两个准备尝试的实验都可以从同一组 current-view direction features 开始。对当前 panorama 的每个 visual token，根据 token grid 近似计算 ERP yaw 和 pitch：

```text
yaw   = (x / W - 0.5) * 2pi
pitch = (y / H - 0.5) * vertical_fov + center_latitude
```

然后构造 action-relative feature vector：

```text
[
  sin(yaw), cos(yaw),
  sin(pitch), cos(pitch),
  frontness,
  leftness,
  rightness,
  backness,
  left_15_score,
  right_15_score,
  left_30_score,
  right_30_score,
  horizon_score
]
```

推荐定义：

```text
frontness = relu(cos(yaw))
backness  = relu(-cos(yaw))
leftness  = relu(-sin(yaw))
rightness = relu(sin(yaw))

left_15_score  = exp(-wrap(yaw + 15deg)^2 / sigma_yaw^2)
right_15_score = exp(-wrap(yaw - 15deg)^2 / sigma_yaw^2)
left_30_score  = exp(-wrap(yaw + 30deg)^2 / sigma_yaw^2)
right_30_score = exp(-wrap(yaw - 30deg)^2 / sigma_yaw^2)

horizon_score = exp(-pitch^2 / sigma_pitch^2)
```

如果图像坐标系导致 left/right 符号约定相反，在做一个小的可视化 sanity check 后交换 `leftness` 和 `rightness`。

## 实验一：Current-Only Action-Relative GeoFiLM

### 假设

当 action-relative panorama grounding 用于调制当前 visual tokens，而不是添加新 content 时，它可以帮助 VLN action prediction。

### 注入位置

把模块应用到 Qwen visual encoder 之后、visual merger 之前的当前图像 visual tokens 上：

```text
Qwen patch embedding
-> Qwen visual transformer
-> Action-relative GeoFiLM on current-view tokens
-> Qwen visual merger
-> language model
```

这个位置比 patch embedding 更晚，所以信号不容易被完整 vision tower 洗掉。它也比 PanoVGGT residual fusion point 更早，所以不会和 PanoVGGT 在 post-merger 位置竞争 additive content delta。

### 模块

使用 scale-only FiLM：

```text
scale = tanh(MLP(direction_features))
x = x * (1 + alpha * scale)
```

第一版不要使用 additive `beta` branch。`beta` branch 会把模块重新变成 content delta，从而更难和 PanoVGGT 区分。

推荐第一版设置：

```yaml
pano_direction_film_enabled: true
pano_direction_film_current_only: true
pano_direction_film_hidden_size: 512
pano_direction_film_alpha_init: 0.05
pano_direction_film_alpha_max: 0.2
```

给 FiLM 模块使用单独 optimizer group：

```text
base_lr: 2e-5
pano_direction_film_lr: 1e-4 或 2e-4
```

### 诊断指标

训练时记录：

- `pano_direction_film/alpha`
- `pano_direction_film/scale_norm`
- `pano_direction_film/modulated_delta_norm / visual_token_norm`
- `pano_direction_film/grad_norm`

如果 `alpha` 和 `grad_norm` 不动，说明 action loss 没有用到这个分支。如果它们动了但 evaluation 没有提升，问题更可能在 representation 或 injection location。

### 主要对比

```text
A. baseline
B. old patch-level ERP MLP
C. current-only action-relative GeoFiLM
D. PanoVGGT only
E. PanoVGGT + current-only action-relative GeoFiLM
```

预期解释：

- `C > A`：panorama-aware spatial grounding 本身有效。
- `C ~= A` 且 `E > D`：spatial branch 单独较弱，但能与 PanoVGGT 互补。
- `C ~= A` 且 `E ~= D`：PanoVGGT 可能已经覆盖这类 spatial signal，或者 adapter 表达能力不足。
- `C < A`：modulation 太强、方向约定错误，或者注入位置有害。

## 实验二：Current-Only Spherical Q/K Adapter

### 假设

如果 spatial grounding 改变当前 panorama tokens 之间的 attention 方式，而不是只做 token channel scaling，可能会更有效。

### 注入位置

在 Qwen visual attention blocks 内部应用一个 current-only adapter。adapter 应该影响 `q` 和 `k`，不影响 `v`，这样它改变的是 spatial matching 和 attention structure，而不是直接注入新 content。

```text
visual hidden states
-> q/k projection
-> spherical q/k adapter on current panorama tokens
-> attention
```

### 更稳妥的第一版

在 `q` 和 `k` 的小 slice 或 low-rank projection 上使用 additive low-rank direction adapter：

```text
q = q + alpha * A_q(MLP(direction_features))
k = k + alpha * A_k(MLP(direction_features))
```

adapter 保持 gated，并初始化到接近 0。不要替换 Qwen 原有的 visual position mechanism。

推荐第一版设置：

```yaml
spherical_qk_adapter_enabled: true
spherical_qk_adapter_current_only: true
spherical_qk_adapter_hidden_size: 256
spherical_qk_adapter_rank: 64
spherical_qk_adapter_alpha_init: 0.01
spherical_qk_adapter_alpha_max: 0.1
spherical_qk_adapter_layers: "last_4"
```

先从最后几个 visual layers 开始。这能降低风险，也让 adapter 更接近任务级视觉推理。

### 可选的 Rotary 版本

如果 additive low-rank adapter 稳定，再测试对 q/k 小维度 slice 使用 spherical rotary branch：

```text
q_slice, k_slice = spherical_rope(q_slice, k_slice, yaw, pitch)
q = concat(q_base, gated(q_slice))
k = concat(k_base, gated(k_slice))
```

这个版本比 GeoFiLM 更结构化，但也更侵入。它应该作为第二步，而不是第一版实现。

### 诊断指标

记录：

- `spherical_qk_adapter/alpha`
- `spherical_qk_adapter/q_delta_norm / q_norm`
- `spherical_qk_adapter/k_delta_norm / k_norm`
- `spherical_qk_adapter/grad_norm`
- 如果代价低，可以记录 current panorama tokens 上的 attention entropy

### 主要对比

```text
A. baseline
B. current-only action-relative GeoFiLM
C. current-only Spherical Q/K Adapter
D. PanoVGGT only
E. PanoVGGT + best spatial grounding adapter
```

预期解释：

- 如果 GeoFiLM 有效，Spherical Q/K Adapter 用来测试 attention-level spatial grounding 是否更强。
- 如果 GeoFiLM 中性但 Q/K 提升，说明有用信号更可能是 relational，而不是 channel-wise。
- 如果两者都中性，而 PanoVGGT 有效，说明任务可能更需要 semantic geometry content，而不是显式 angular grounding。

## 推荐顺序

1. 实现并训练 current-only action-relative GeoFiLM。
2. 如果它能训练起来，并且结果中性或正向，再跑 PanoVGGT + GeoFiLM。
3. 只有在理解 GeoFiLM 诊断结果后，再实现 current-only Spherical Q/K Adapter。
4. 使用表现最好的 spatial grounding adapter 做最终 PanoVGGT 组合实验。

## 论文表述

推荐表述：

> 我们研究 panorama-aware spatial grounding 是否能进一步提升 VLN action prediction。我们不把 ERP 坐标当作独立的 absolute position embedding 来验证，而是把 panorama geometry 转换成 action-relative directional features，并通过轻量的 current-view spatial grounding modules 注入模型。

这些实验应该被描述为 spatial grounding ablations，而不是狭义的 raw sinusoidal ERP embeddings 验证。
