# SBS: Surprisal-Balanced Sampling

## 目标

训练样本由一条 VLN episode 的离散动作序列切成固定长度 action chunk。当前数据构造默认且唯一使用 **SBS: Surprisal-Balanced Sampling**：每个 chunk 的保留概率由动作信息量和候选类别的自动平衡因子共同决定，不再手写目标 bucket 比例。

严格说，这里使用的是动作的 self-information / surprisal，即 `-log(freq(a))`；它来自动作分布的熵项，因此也可以直观理解为 entropy-aware sample selection。

核心约束如下：

- 总数据量通过 `tau` 连续控制，推荐配置保持在 50 万以内。
- 最后 4 个 terminal start 始终 dense 保留，保证 online 推理中包含 `stop` 时执行完整 action sequence 的训练覆盖。
- 不手动区分 stop 在第 1/2 位还是第 3/4 位；四个 stop position 使用同一套候选频率平衡公式。
- 比例不是简单降采样得到，而是由 chunk 价值和候选类别频率共同控制。

## 动作信息量

设原始 episode 动作序列中的动作集合为：

```text
stop, forward, left, right
```

先在选定的 source datasets 上统计动作频率：

```text
freq(a) = count(a) / sum_b count(b)
```

动作信息量定义为：

```text
I(a) = -log(freq(a))
```

因此，高频动作 `forward` 的信息量低，低频动作 `stop/left/right` 的信息量高。对当前 `r2r+rxr` source data，统计值为：

| action | count | freq | I(a) |
| --- | ---: | ---: | ---: |
| stop | 28408 | 0.011774 | 4.441874 |
| forward | 1546223 | 0.640841 | 0.444974 |
| left | 426949 | 0.176951 | 1.731880 |
| right | 411223 | 0.170434 | 1.769409 |

对一个 padded 后长度为 4 的 chunk：

```text
C = (a_1, a_2, a_3, a_4)
```

chunk 价值定义为：

```text
V(C) = mean(I(a_i))
```

例如：

- `FFFF` 的价值最低，因为全是高频 `forward`。
- `LLLL/RRRR` 的价值高于 `FFFF`，因为 turn 动作更稀有。
- terminal stop chunk 的价值更高，尤其是 `SSSS` 或 `FSSS`。

## 自动平衡类别

只用 `V(C)` 会提高 stop/turn 的保留概率，但仍可能让高频候选类别主导总量。因此引入候选类别平衡因子。每个候选 chunk 属于以下 7 类之一：

```text
stop_pos_1
stop_pos_2
stop_pos_3
stop_pos_4
first_forward
first_left
first_right
```

分类规则：

- 如果 chunk 内包含 `stop`，类别为第一个 `stop` 出现的位置。
- 如果不包含 `stop`，类别由第一个动作决定。

先在完整候选池上统计每个类别的候选数量 `N_c`。对当前 `r2r+rxr`：

| class | candidates |
| --- | ---: |
| stop_pos_1 | 28408 |
| stop_pos_2 | 28408 |
| stop_pos_3 | 28408 |
| stop_pos_4 | 28408 |
| first_forward | 1478091 |
| first_left | 418246 |
| first_right | 402834 |

平衡因子定义为：

```text
B(c) = N_c^(-beta)
```

然后把所有类别的 `B(c)` 归一化到均值为 1。推荐：

```text
beta = 0.40
```

这是一种温和的自动平衡：它不会强行把类别采成均匀分布，但会抑制过多的 `first_forward`，并补偿候选数较少的 stop 类别。当前推荐配置下的归一化平衡因子为：

| class | B(c) |
| --- | ---: |
| stop_pos_1 | 1.430597 |
| stop_pos_2 | 1.430597 |
| stop_pos_3 | 1.430597 |
| stop_pos_4 | 1.430597 |
| first_forward | 0.294451 |
| first_left | 0.487889 |
| first_right | 0.495271 |

## 保留概率

对 body 区域的候选 chunk，保留概率为：

```text
p_keep(C) = 1 - exp(-tau * V(C) * B(class(C)))
```

推荐：

```text
tau = 1.35
beta = 0.40
```

`tau` 控制总体数据量，`beta` 控制类别平衡强度。二者都有明确含义，适合做 ablation：

- 调 `tau`：主要改变总样本量。
- 调 `beta`：主要改变 `first_forward / first_left / first_right / stop` 的相对比例。
- `beta=0`：退化为只按 chunk 信息量采样，不做类别频率平衡。

## 顺序扫描与跳跃

设动作序列长度为 `T`，action horizon 为：

```text
H = 4
```

末尾 dense 区域起点为：

```text
d = max(0, T - H)
```

body 区域只扫描：

```text
s < d
```

扫描过程：

```text
s = 0
while s < d:
    C_s = A[s : s + H]
    p = 1 - exp(-tau * V(C_s) * B(class(C_s)))

    if Bernoulli(p) is accepted:
        keep s
        s = s + H
    else:
        s = s + 1
```

接受后跳过 `H` 个 step，减少高度重叠的 body chunk。拒绝后只移动一步，避免漏掉后续可能更有价值的 chunk。

## Terminal Dense Stop

最后 4 个起点强制保留：

```text
s in {d, d+1, ..., T-1}
```

这部分不再随机采样，也不受 `tau/beta` 影响。原因是 online 测试时如果模型输出 action sequence 包含 `stop`，会执行完整 sequence；因此训练数据必须密集覆盖 terminal stop 附近的不同剩余 horizon。

若最后 chunk 长度不足 4，并且以 `stop` 结束，则右侧 padding `stop`，使训练目标仍为 4 个动作：

```text
F S -> F S S S
S -> S S S S
```

`real_action_count` 保留 padding 前的真实动作数。

## 推荐配置的实际比例

用 `r2r+rxr`、`seed=42`、`tau=1.35`、`beta=0.40`、`tail_dense=4` 生成的数据量为：

```text
497012 samples
```

关键比例：

| dataset | rows | r2r % | stop % | FFFF % | turn % | first-F % | first-L % | first-R % | stop@1 % | stop@2 % | stop@3 % | stop@4 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| subset_v2 | 355100 | 28.26 | 22.00 | 12.56 | 65.44 | 33.00 | 22.50 | 22.50 | 8.00 | 8.00 | 3.00 | 3.00 |
| SBS | 497012 | 28.05 | 22.86 | 11.05 | 66.09 | 39.43 | 19.16 | 18.55 | 5.72 | 5.72 | 5.72 | 5.72 |

这个配置保留了 subset v2 中表现较好的几个宏观比例：`stop` 总量、`FFFF`、`turn chunk` 都接近；同时避免了手动写死 bucket 目标比例。

## 生成命令

默认脚本：

```bash
./scripts/prepare_dataset.sh
```

等价于：

```bash
python src/data/prepare_training_data.py \
  --input_root /workspace/code_dir/a_property/dataset/NAVIDA_pano \
  --dataset_name r2r rxr \
  --output_path /workspace/code_dir/a_property/dataset/NAVIDA_pano/train_r2r_rxr_4action_stop_pad_sbs_tau1p35_beta0p40_taildense4.jsonl \
  --pad_stop_to_horizon \
  --seed 42 \
  --sbs_tau 1.35 \
  --sbs_beta 0.40
```

降低总量但保持同一比例机制：

```bash
SBS_TAU=1.20 \
SBS_BETA=0.40 \
OUTPUT_PATH=/workspace/code_dir/a_property/dataset/NAVIDA_pano/train_r2r_rxr_4action_stop_pad_sbs_tau1p20_beta0p40_taildense4.jsonl \
./scripts/prepare_dataset.sh
```

去掉类别平衡的对照：

```bash
SBS_TAU=0.48 \
SBS_BETA=0.0 \
OUTPUT_PATH=/workspace/code_dir/a_property/dataset/NAVIDA_pano/train_r2r_rxr_4action_stop_pad_sbs_tau0p48_beta0_taildense4.jsonl \
./scripts/prepare_dataset.sh
```
