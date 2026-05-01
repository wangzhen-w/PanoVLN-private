# Probability-Skip Action Chunk Sampling for VLN Training

## 摘要

本文描述一种用于视觉语言导航训练数据构造的概率跳跃式动作片段采样策略。给定一条 episode 的离散动作序列，方法以固定长度动作窗口构造训练 chunk，并根据 chunk 内部动作组成决定保留概率。与逐步密集枚举不同，该策略在主体区域中采用“接受即跳跃”的顺序扫描过程，从而抑制高度重叠的相邻样本；同时在 episode 末尾保留 dense sampling，以提高终止动作附近样本的覆盖率。

## 问题定义

设一条 episode 的动作序列为

```text
A = (a_0, a_1, ..., a_{T-1}),
```

其中每个动作属于四类离散动作：

```text
stop, forward, left, right.
```

采样目标是在动作序列上选择若干起点集合 `S`。每个起点 `s in S` 对应一个长度不超过 `H` 的动作 chunk：

```text
C_s = (a_s, a_{s+1}, ..., a_{min(s+H-1, T-1)}),
```

其中 `H` 为动作预测 horizon，当前实现默认 `H = 4`。若启用 stop padding，则末尾长度不足 `H` 且以 `stop` 结束的 chunk 会在右侧补齐 `stop`，使训练目标保持固定长度。

## Chunk 类型与保留概率

采样概率仅由 chunk 内容决定，而不以第一个动作作为主要依据。对任意候选 chunk `C_s`，定义三类互斥类别：

```text
stop chunk:
    C_s 中包含 stop

turn chunk:
    C_s 中不包含 stop，但包含 left 或 right

forward chunk:
    C_s 中只包含 forward
```

对应的默认保留概率为：

```text
P(keep | stop chunk)    = 1.00
P(keep | turn chunk)    = 0.50
P(keep | forward chunk) = 0.05
```

该设计将终止相关片段作为确定性样本，将包含转向的片段作为高信息密度样本进行较高概率保留，并将连续前进片段作为低信息密度背景样本进行低概率保留。

## 主体区域的概率跳跃扫描

主体采样采用从左到右的顺序过程。设末尾 dense 区域起点为：

```text
d = max(0, T - H).
```

为保证主体 chunk 不与末尾 dense 区域发生动作 step 重叠，主体扫描的右边界为：

```text
b = max(0, d - H + 1).
```

主体候选起点只在 `[0, b)` 中产生。扫描过程如下：

```text
s = 0
while s < b:
    C_s = A[s : s + H]
    p = P(keep | C_s)

    if Bernoulli(p) is accepted:
        keep s
        s = s + H
    else:
        s = s + 1
```

因此，在主体区域中，一旦某个起点被接受，下一个候选起点会移动到当前 chunk 末尾之后。该机制保证主体区域内任意两个已采样 chunk 的动作 step 不重叠。

## 末尾 Dense Stop 采样

为了提高 stop 附近样本的覆盖率，末尾区域使用确定性 dense sampling。具体地，所有满足下式的起点均被强制保留：

```text
s in {d, d+1, ..., T-1}.
```

当 `H = 4` 时，这对应最后 4 个起点。该区域内部允许 chunk 之间发生重叠，因为其目的不是构造非重叠主体轨迹片段，而是密集覆盖 terminal stop 附近的不同剩余 horizon。

由于主体扫描边界为 `b = max(0, d - H + 1)`，主体 chunk 不会伸入末尾 dense 区域。因此，除末尾 dense 区域内部外，采样得到的 chunk 在动作 step 维度上互不重叠。

## 去重与可复现性

最终样本以 `(dataset, episode_id, step_index)` 作为唯一键。采样过程中若产生重复键，则视为数据构造错误并终止生成。

随机性由全局伪随机数生成器控制，默认 seed 为：

```text
seed = 42
```

在输入 episode 顺序固定的条件下，采样结果可复现。

## 输出约束

采样策略只决定 chunk 起点集合，不改变训练样本的 JSONL 字段结构。每个保留起点仍输出与训练代码兼容的字段，包括：

```text
instruction
action_sequence
images
episode_id
dataset
step_index
end_step
real_action_count
```

若启用 stop padding，`action_sequence` 会被补齐到固定 horizon 长度；`real_action_count` 仍表示 padding 前的真实动作数量。
