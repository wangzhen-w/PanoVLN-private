# ActionCalibrator

`ActionCalibrator` 是一个用于第二阶段训练的 movement logit calibration 模块。它不重新定义导航动作空间，也不替代 `lm_head` 的 action prediction，而是在 base model 已经给出 action logits 之后，利用当前 action token 对全景图像 token 的 attention 分布，给 `forward`、`left`、`right` 三个运动动作加一个很小的 residual bias。`stop` 不属于这个 residual 的输出空间，并且当 base model 在四个动作中已经认为 `stop` 是 top-1 时，整个 action position 的 logits 会完全 bypass，不做任何修改。

设当前 action position 的 base logits 为

```text
z = [z_stop, z_forward, z_left, z_right].
```

`ActionCalibrator` 的目标不是重新预测 `z`，而是只学习一个三维 residual

```text
delta = [delta_forward, delta_left, delta_right],
```

最后得到

```text
z'_stop    = z_stop
z'_forward = z_forward + delta_forward
z'_left    = z_left    + delta_left
z'_right   = z_right   + delta_right
```

但这个更新只在 base top-1 不是 `stop` 时生效。如果

```text
argmax([z_stop, z_forward, z_left, z_right]) = stop,
```

则直接令

```text
z' = z.
```

因此，这个模块不会把一个 base model 已经想停的样本强行改成运动动作。注意这里的 `argmax` 只在四个 action verbalizer 的 logits 内部计算，不是在整个 vocabulary 上计算。

模块的输入不是完整视觉特征，也不是 base logits。base logits 只用于上面的 stop bypass；真正进入 MLP 的输入来自 action-token attention 的全景环形几何统计。对于一个 action query，代码会在指定的 full-attention decoder layers 上重算并截取该 query 到当前 panorama image tokens 的 attention。默认层为

```yaml
action_calibrator_attention_layer_indices: [19, 23, 27, 31]
```

这些层的 attention 会被平均。设平均后的 attention 为

```text
a_i, i = 1 ... N,
```

其中 `i` 是当前全景图的 image token index。由于只关心当前 panorama 的空间分布，这里的 attention 只保留 image token 部分，不使用文本 token attention。代码中会对 attention 做非负裁剪和归一化：

```text
m       = sum_i max(a_i, 0)
alpha_i = max(a_i, 0) / max(m, eps).
```

其中 `m` 是 action query 落在当前图像 token 上的总 attention mass，`alpha_i` 是归一化后的图像内 attention 分布。

每个 image token 根据它在全景 token grid 中的水平位置被赋予一个 yaw：

```text
theta_i in [-pi, pi].
```

因为模型一次输出最多 4 个动作，后续 action token 的语义方向应当相对于前面已经生成或 teacher-forced 的动作进行短程修正。代码根据前缀动作维护一个 `prefix_yaw`：

```text
forward: prefix_yaw 不变
left:    prefix_yaw -= action_calibrator_turn_angle_deg
right:   prefix_yaw += action_calibrator_turn_angle_deg
stop:    后续位置不再校准
```

当前默认

```yaml
action_calibrator_turn_angle_deg: 15.0
```

对第 `k` 个 action position，先计算每个 image token 相对于当前前缀方向的环形角度：

```text
phi_i = wrap(theta_i - prefix_yaw_k),
```

其中

```text
wrap(x) = atan2(sin(x), cos(x)).
```

在计算方向特征前，`ActionCalibrator` 不再直接使用完整的 `alpha_i` 分布。离线可视化显示，action-token attention 往往由一个空间连续的主块加若干小热点组成；如果直接对全分布做一阶圆统计，这些小热点会把 yaw bias 拉偏。现在代码把全景 attention 先转成当前图像的二维 token grid，并在这个 grid 上做 dominant connected component pooling。

具体地，先对 attention map 做一个固定的 3x3 轻微平滑，然后用相对阈值取候选 foreground：

```text
S = Smooth(alpha)
M_i = 1[S_i >= 0.35 * max_j S_j].
```

连通域在全景拓扑上定义：垂直方向不 wrap，水平方向 circular wrap，因此最左列和最右列被视为相邻。代码在 `M` 上找连通域，并选择原始 attention mass 最大的主连通域：

```text
C* = argmax_C sum_{i in C} alpha_i.
```

然后只在主连通域内重新归一化 attention：

```text
beta_i = alpha_i / sum_{j in C*} alpha_j,  i in C*
beta_i = 0,                                i notin C*.
```

这里 `beta` 才是后续用于方向 pooling 的分布。这个设计不是额外传感器或 waypoint，而是利用全景图“水平闭合环 + 导航意图通常形成空间主模态”的结构先验：孤立小热点可以保留在 LLM 原始 logits 中，但不应该支配 movement bias。

然后用 `sin(phi_i)` 和 `cos(phi_i)` 构造四个方向 kernel：

```text
front_i = max(cos(phi_i), 0)
right_i = max(sin(phi_i), 0)
left_i  = max(-sin(phi_i), 0)
back_i  = max(-cos(phi_i), 0).
```

这一步体现了全景图的水平环结构：模块没有把 panorama 粗暴切成固定区域，而是用连续的环形三角函数把主连通域 attention mass 投影到前、左、右、后四个方向。四个 attention-derived features 为

```text
e_front = sum_i beta_i * front_i
e_left  = sum_i beta_i * left_i
e_right = sum_i beta_i * right_i
e_back  = sum_i beta_i * back_i.
```

最终进入 MLP 的特征就是

```text
x = [e_front, e_left, e_right, e_back].
```

`ActionCalibrator` 不是让 MLP 从零开始重新学习 yaw 到 action 的映射。模块先构造一个可解释的 attention-to-action prior：

```text
p_forward = e_front - max(e_left, e_right) - e_back
p_left    = e_left - e_right
p_right   = e_right - e_left
```

写成向量为

```text
p = [p_forward, p_left, p_right].
```

这个 prior 的含义是直接的：如果 attention 主要落在正前方，则提高 `forward`；如果 attention 明显偏左/偏右，则分别提高 `left`/`right`；如果 attention 落在背后，`forward` 会被压低，而左右方向只由轻微的左右不对称决定。它只使用全景环上的相对 yaw 统计，不使用 ground-truth waypoint，也不引入额外传感器。

MLP 只学习这个 prior 之上的 residual。当前 MLP 结构为

```text
h   = SiLU(Linear(LayerNorm(x)))
r   = Linear(h)
```

其中

```text
r in R^3
```

对应 `forward`、`left`、`right` 三个 movement actions。最后一层 `Linear` 的权重和 bias 在初始化时全置零，所以刚打开第二阶段训练时

```text
r = 0.
```

因此初始行为不是一个随机 MLP policy，而是上面的 hand-designed attention prior；MLP 只负责在二阶段中学习 residual correction。这一点比纯 zero-init 更适合从强 Stage1/PanoVGGT checkpoint 上做 Stage2，因为 PanoVGGT 的经验已经说明，最外层强度参数即使可学习也往往不会大幅移动，如果整个 calibration 从 0 强度开始，短程二阶段训练可能很难真正起效。

prior 和 residual 相加后，代码先做 movement 内部零均值：

```text
u = p + r
u_centered = u - mean(u).
```

这样做的目的是让 residual 更像是在 `forward/left/right` 三个运动动作内部重排偏好，而不是整体抬高所有 movement logits 去压低 `stop` 的相对优势。虽然 base top-1 为 stop 时会完全 bypass，但在 base top-1 为 movement 时，零均值仍然能减少模块学成“统一加强运动动作”的风险。

之后 residual 经过有界变换和最外层 alpha：

```text
b = alpha * tanh(action_calibrator_delta_scale * u_centered).
```

这里的 `b` 仍然是三维向量，对应 `forward`、`left`、`right`。`alpha` 是一个和 PanoVGGT MLP 类似的 bounded scalar：

```text
alpha = action_calibrator_alpha_max * sigmoid(raw_alpha).
```

默认配置为

```yaml
action_calibrator_alpha_init: 0.225
action_calibrator_alpha_max: 0.45
```

也就是说 `alpha` 同时承担两个角色：它既是全局 calibration strength，也是 logit bias 的硬上限。Stage2 第 0 step 的最大 movement 修正幅度为 `0.225`，训练中最多增长到 `0.45`。这样比同时保留 `alpha` 和 `max_delta` 更干净，因为论文里只需要解释一个“有界校准强度”。

这个设计保留了 PanoVGGT 的经验：`raw_alpha` 仍然可学习，但方法不依赖它大幅变化；如果它基本不动，初始化值也已经能提供合理的 calibration。`alpha` 太小会让模块难以修复 movement 错误，太大则可能覆盖原本正确的 `lm_head` 判断，因此默认把最大值限制在一个 residual bias 的范围内。

`action_calibrator_delta_scale` 出现在 `tanh` 内部。默认值为

```yaml
action_calibrator_delta_scale: 1.0
```

它控制 prior + MLP residual 进入 `tanh` 前的放大倍率：

```text
tanh(delta_scale * u_centered).
```

当 `delta_scale` 较小时，`tanh` 更接近线性区，residual 更温和，训练更不容易饱和；当 `delta_scale` 较大时，输出更容易接近 `alpha` 给出的上限，模块会更快、更强地改变 logits，但也更容易过校准。当前默认 `1.0` 表示不额外放大，让 attention prior、MLP residual 和训练过程自己控制修正强度。

除了 MLP 输出本身，代码还会根据 attention 分布计算一个 confidence。先计算主连通域 attention `beta` 在环形 yaw 上的一阶方向矩：

```text
s = sum_i beta_i * sin(phi_i)
c = sum_i beta_i * cos(phi_i)
r = sqrt(s^2 + c^2).
```

其中 `r` 越大，说明主连通域在某个 yaw 方向上越集中；`r` 越小，说明主连通域本身也比较分散。设主连通域占完整 image attention 的质量为

```text
rho = sum_{i in C*} alpha_i.
```

最终 confidence 为

```text
conf = sqrt(min(m, 1)) * r * sqrt(rho).
```

这里 `m` 是 action query 落在当前 image tokens 上的 attention mass。如果模型这一步几乎没有看当前图像、主连通域方向不集中，或者最大连通域只占很小一部分 attention，则 `conf` 会很小，ActionCalibrator 的 bias 会自动变弱。这样实现上是“只从主连通域读方向”，但不会让一个很小的孤立热点接管 logits。

由于模型一次输出 4 个 action token，代码还提供 action-position decay：

```text
d_k = action_calibrator_step_decay[k].
```

默认配置是

```yaml
action_calibrator_step_decay: [1.0, 0.75, 0.55, 0.40]
```

于是第 `k` 个 action token 的最终 residual 为

```text
delta_k = b_k * conf_k * d_k.
```

如果不希望后续 action token 的 bias 变弱，可以把这个配置改成

```yaml
action_calibrator_step_decay: [1.0, 1.0, 1.0, 1.0]
```

训练时，如果打开了 `action_calibrator_l2_weight`，代码还会对实际产生的 residual 加 L2 正则。当前 loss 形式可以写成

```text
L = CE(z', y) + lambda_l2 * mean(delta^2),
```

其中

```yaml
action_calibrator_l2_weight: 0.0
```

对应

```text
lambda_l2 = 0.
```

也就是说当前默认不启用 residual L2。这个参数的作用是进一步惩罚过大的 logit 修正，防止模块长期贴近 `alpha` 给出的上限。如果发现 Stage2 训练中 `ActionCalibrator` 很快学出较大的 delta，或者验证集上 base-correct 样本被明显破坏，可以把它设成一个很小的值，例如 `1e-4`。但默认保持 `0.0` 更简单，因为当前已经有 zero-init residual、bounded alpha、`tanh`、attention confidence、step decay、stop bypass 和 movement zero-mean 这些保守机制。

训练路径中，`lm_head` 先产生原始 logits，`ActionCalibrator` 再计算上面的 `delta` 并写回 movement action token logits，最后用校准后的 logits 进入原始 cross entropy loss。换句话说，如果第二阶段打开这个模块，CE 看到的是

```text
z'
```

而不是未校准的

```text
z.
```

这使得梯度会直接训练 `ActionCalibrator` 的 MLP。attention 作为输入被 detach，因此梯度不会通过 attention capture 反传到 LLM attention 本身；但如果第二阶段配置里同时给 LLM、vision、visual merger 或 PanoVGGT MLP 一个很小学习率，这些模块仍然会通过正常 CE 路径被轻微更新。

推荐的 Stage2 起始学习率是

```yaml
language_model_lr: 2.0e-6
visual_lr: 1.0e-6
visual_merger_lr: 2.0e-6
panovggt_mlp_lr: 2.0e-6
action_calibrator_lr: 5.0e-5
```

这种设置让 `ActionCalibrator` 承担主要学习压力，同时允许已有 VLN 主干小幅适配新的 movement bias。强 PanoVGGT checkpoint 上的 Stage2 不建议只训练 `action_calibrator`：只训 AC 可以作为 sanity check，但更正式的 Stage2 应该至少小学习率解冻 language model、visual merger 和 PanoVGGT MLP；vision encoder 可以用更小的 `visual_lr`。PanoVGGT encoder 本身仍然是冻结的，`panovggt_mlp_lr` 只作用于接入 Qwen token space 的几何 MLP。

推理路径使用同样的公式和边界。模型先给出 base logits，如果四个 action verbalizer 内部的 top-1 是 `stop`，则不调用 movement 写回；否则根据当前 action query 的 panorama attention 计算 `delta`，只加到 `forward`、`left`、`right` 对应 token 上。因为推理时 attention capture 需要看到完整前缀中的 image tokens，开启 `ActionCalibrator` 时会禁用 KV cache，避免只看到当前 token 而捕获不到图像 attention。关闭 `ActionCalibrator` 时不会安装 attention hook，不会改变 logits、loss 或 generation cache 行为。

总体来说，`ActionCalibrator` 的技术策略可以概括为

```text
action-token attention over current panorama
    -> ring yaw features [front, left, right, back]
    -> zero-initialized small MLP
    -> zero-mean bounded movement residual
    -> confidence and step scaling
    -> add only to forward/left/right logits
    -> full bypass if base action top-1 is stop.
```

它利用的是全景图的环形 yaw 结构和模型自身 action-token attention 中已有的空间意图，但只把这个信号作为小幅 logits bias，而不是直接把 attention readout 变成动作决策。
