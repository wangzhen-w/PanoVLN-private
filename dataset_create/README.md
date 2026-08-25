# HM3D 决策轨迹采集

这套代码为 PanoVLN 制作 HM3D 导航轨迹。它只负责确定 agent 从哪里出发、走哪条自然最短路、途中经过哪些真实决策位置，以及如何用基础动作完整回放。图片渲染和 instruction 生成不在这一步进行。

## 轨迹是怎样得到的

整个过程分为三步：

```text
HM3D NavMesh
    ↓
Navigation Region Partition
    ↓
Panorama-Verified Decision Relations
    ↓
Region-Structured Route Collection
```

第一步把连续的 NavMesh 划分成房间、走廊、楼梯连接区等 Navigation Regions。这里不使用房间语义，只关心哪些空间连在一起，以及它们通过什么位置连接。

第二步寻找真正需要选择方向的位置。一个位置只有在当前全景中能同时看到正确去路和至少一条其他可通行去路，而且这些去路通向不同区域、不会很快重新汇合时，才会被认为是有效决策。分支之间不要求固定的角度差。

第三步从不同区域选择起点和终点，直接使用 Habitat 的自然 shortest path，不人为添加 decision waypoint。路径必须至少经过一个有效决策。具有相同决策顺序和相近区域级结构的坐标变体只保留一条，避免某个建筑因为大量相似坐标而产生几千条重复轨迹。

## 输出内容

每个 split 只产生两个正式文件。train 和 val 分开放置：

```text
PanoVLN/
├── train/
│   ├── train.json.gz
│   └── train_stats.json
└── val/
    ├── val.json.gz
    └── val_stats.json
```

`train.json.gz` 保存轨迹。每条轨迹包含 HM3D scene ID、起点、目标、实际停止状态、自然最短路、region/connection sequence、基础动作，以及每个决策位置的 incoming、selected 和 alternative branches。

`train_stats.json` 保存整体数量、长度和决策数量分布，以及每个场景的 regions、decision relations、route families、最终轨迹数和异常原因。

采集过程中会使用隐藏的 `.work` scene shards 来支持断点续跑。最终数据独立回放达到 100% 后，这些中间文件会自动清除。

## 如何运行

先打开 `dataset_create/create_trajectories.sh`。脚本顶部的每个参数都有注释，通常只需要检查 HM3D 路径、输出路径和 GPU 列表。

脚本当前默认采集完整的 100 个 val 场景：

```bash
cd /workspace/code/VLN
./dataset_create/create_trajectories.sh collect
```

需要重新操作 800 个 train 场景时，显式指定 split：

```bash
SPLIT=train ./dataset_create/create_trajectories.sh collect
```

只验证已经生成的数据：

```bash
./dataset_create/create_trajectories.sh validate
```

`SCENE_IDS` 可以指定少量场景，为空时采集整个 split。每张 GPU 当前只允许一个 Habitat 进程。

## 后续回放和渲染

轨迹以 start pose 和 `action_ids` 为回放依据。`decision_events[*].action_index` 表示到达该决策状态前已经执行的动作数，因此执行 `action_ids[:action_index]` 后即可重新渲染对应 ERP。每个事件还保存 decision pose，以及 incoming、selected 和 alternatives 各分支的 region、connection 与 anchor position。这些信息既能渲染决策位置和终点附近的 RGB/depth，也能判断 agent 最终选择了哪个分支。

采集结束后脚本会重新加载最终数据并逐条执行动作，同时核对 decision pose、final pose、目标到达和碰撞。验证不通过时不会清理工作目录，方便继续检查或重新采集。
