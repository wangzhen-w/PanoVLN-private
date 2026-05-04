# EBS: Event-Balanced Sampling

## 目标

当前训练数据构造默认且唯一使用 **EBS: Event-Balanced Sampling**。EBS 保留 probability-skip 实验中表现最好的采样行为，但把方法表述为基于 VLN 动作语义的 event/background 采样，而不是手写 7 个 bucket 的目标比例。

当前实验结论：

- 手动 bucket subset v2 的 habitat-online SR 约为 `48.3`。
- probability-skip 的 habitat-online SR 约为 `55%`，是当前最强结果。
- SBS 的训练日志中 stop 学习失败，主要问题是 `FFFF` 过多，并且采到了 terminal dense 前的 near-goal non-stop overlap 样本。

因此，EBS 的目标不是拟合 subset v2，而是复现 probability-skip 的成功结构。

## 动作语义

VLN 的 4 个动作分为三类：

```text
completion: stop
event: left, right
background: forward
```

直观上：

- `forward` 是高频背景动作，连续纯 forward chunk 信息密度低。
- `left/right` 是路径决策事件，应该比纯 forward chunk 更高概率保留。
- `stop` 是任务完成信号，通过 terminal coverage 单独保证。

## Terminal Coverage

每条 episode 的最后 4 个起点固定保留：

```text
s in {T - 4, T - 3, T - 2, T - 1}
```

其中 `T - 1` 是 `stop` 动作所在起点。若最后 chunk 不足 4 个动作，并且以 `stop` 结束，则右侧 padding `stop`：

```text
F S -> F S S S
S -> S S S S
```

`real_action_count` 保留 padding 前的真实动作数。保留最后 4 个 terminal start 的原因是 online 测试时如果模型输出 action sequence 包含 `stop`，会执行完整 sequence；dense terminal coverage 能覆盖不同剩余 horizon 下的完成动作。

## Terminal Buffer

body 区域不扫描紧邻 terminal dense 前的 3 个 overlap 起点：

```text
dense_start = max(0, T - H)
body_stop = max(0, dense_start - H + 1)
```

只扫描：

```text
s < body_stop
```

这样可以避免采到 `T-7, T-6, T-5, T-4` 这一类 near-goal non-stop chunk。它们离完成点很近，但标签仍然不是 `stop`，会削弱模型学习 stop 的信号。

## Body Sampling

设 action horizon：

```text
H = 4
```

对 body chunk：

```text
C = (a_1, a_2, a_3, a_4)
```

若 chunk 内包含 `left/right`，则视为 event chunk：

```text
event(C) = any(a_i in {left, right})
```

否则它是 background chunk，也就是纯 `FFFF`。

保留概率为：

```text
p_keep(C) = p_event       if event(C)
p_keep(C) = p_background  otherwise
```

推荐默认值：

```text
p_event = 0.50
p_background = 0.05
seed = 42
tail_dense = 4
```

接受后跳过 4 个 step，减少高度重叠的 body chunk；拒绝后只移动 1 个 step，避免漏掉后续事件 chunk：

```text
s = 0
while s < body_stop:
    C_s = A[s : s + H]
    p = p_event if C_s contains left/right else p_background

    if Bernoulli(p) is accepted:
        keep s
        s = s + H
    else:
        s = s + 1
```

## 推荐配置的实际比例

用 `r2r+rxr`、`seed=42`、`p_event=0.50`、`p_background=0.05`、`tail_dense=4` 生成的数据量为：

```text
495081 samples
```

关键比例：

| dataset | rows | r2r % | stop % | FFFF % | turn % | first-F % | first-L % | first-R % | stop@1 % | stop@2 % | stop@3 % | stop@4 % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| subset_v2 | 355100 | 28.26 | 22.00 | 12.56 | 65.44 | 33.00 | 22.50 | 22.50 | 8.00 | 8.00 | 3.00 | 3.00 |
| probskip | 495081 | 27.50 | 22.95 | 3.72 | 73.33 | 45.86 | 15.90 | 15.28 | 5.74 | 5.74 | 5.74 | 5.74 |
| SBS | 497012 | 28.05 | 22.86 | 11.05 | 66.09 | 39.43 | 19.16 | 18.55 | 5.72 | 5.72 | 5.72 | 5.72 |
| EBS | 495081 | 27.50 | 22.95 | 3.72 | 73.33 | 45.86 | 15.90 | 15.28 | 5.74 | 5.74 | 5.74 | 5.74 |

EBS 与 probskip 的比例一致，但方法描述从“概率跳过”整理为 completion / event / background 三类采样。相比 SBS，EBS 将 `FFFF` 从约 `11%` 降到约 `4%`，同时去掉 near-goal non-stop overlap 负例。

## 生成命令

默认脚本：

```bash
./scripts/prepare_dataset.sh
```

等价于：

```bash
python src/data/prepare_training_data.py \
  --input_root /workspace/code_dir/a_property/dataset/PanoVLN \
  --dataset_name r2r rxr \
  --output_path /workspace/code_dir/a_property/dataset/PanoVLN/train_r2r_rxr_4action_stop_pad_ebs_event0p50_bg0p05_taildense4.jsonl \
  --pad_stop_to_horizon \
  --seed 42 \
  --event_keep_prob 0.50 \
  --background_keep_prob 0.05
```

更强 event 采样的 ablation：

```bash
EVENT_KEEP_PROB=0.60 \
BACKGROUND_KEEP_PROB=0.05 \
OUTPUT_PATH=/workspace/code_dir/a_property/dataset/PanoVLN/train_r2r_rxr_4action_stop_pad_ebs_event0p60_bg0p05_taildense4.jsonl \
./scripts/prepare_dataset.sh
```

更少 background 的 ablation：

```bash
EVENT_KEEP_PROB=0.50 \
BACKGROUND_KEEP_PROB=0.03 \
OUTPUT_PATH=/workspace/code_dir/a_property/dataset/PanoVLN/train_r2r_rxr_4action_stop_pad_ebs_event0p50_bg0p03_taildense4.jsonl \
./scripts/prepare_dataset.sh
```
