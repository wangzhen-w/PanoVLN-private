# PanoVLN 数据制作

全部可用 HM3D 场景共同制作一份训练数据，不划分训练集与验证集。HM3D 原始资源目录中的 `train/`、`val/` 仍是场景文件路径的一部分，不决定 PanoVLN 的用途。

## 目录

```text
/workspace/data2/dataset/general_VLN_data/PanoVLN/
├── trajectory/
│   ├── trajectories.json.gz       # 原始动作、pose、决策事件
│   └── trajectories_stats.json
├── image/
│   └── trajectory_.../frame_0.jpg  # clean ERP，每个动作前一张
├── train.json
├── train.json.gz                   # 标准 R2R，全部用于训练
└── .work/                         # 制作期间的检查点
    ├── trajectory/
    └── instruction/
```

轨迹文件与最终 R2R 文件分开，避免覆盖动作和决策元数据。`.work/` 支持中断续跑，两个阶段各用一个子目录；对应阶段成功完成后自动清理其检查点。图片、视频和 compass 按当前轨迹即时制作；最终只保留训练 ERP 和 R2R 文件，原始轨迹继续作为动作对齐依据。

## 轨迹制作

[create_trajectories.sh](create_trajectories.sh) 顶部直接设置场景根目录、输出、GPU 和运行模式。默认扫描所有含 GLB 和 NavMesh 的 HM3D 场景，收集到同一个 `trajectory/trajectories.json.gz`。`SCENE_IDS=()` 表示使用全部场景，也可设置明确的场景列表。

```bash
cd /workspace/code/VLN
./dataset_create/create_trajectories.sh inspect
./dataset_create/create_trajectories.sh collect
./dataset_create/create_trajectories.sh validate
```

采集过程先划分 Navigation Regions，再通过全景深度检验有区分意义的分支，最后从不同区域采集自然最短路径。路线必须经过有效决策，相同决策顺序和相近区域结构的坐标变体仅保留一个代表。它不人为添加 decision waypoint，也不按场景设置固定轨迹配额。

每条轨迹保存起点、目标、实际 stop pose、region/connection sequence、基础动作与 decision events。事件的 `action_index` 是到达该状态前已经执行的动作数，执行 `action_ids[:action_index]` 即可复现对应 pose。

采集完成的场景有独立检查点。中断后重新运行 `collect` 复用这些场景；已存在完整最终轨迹时默认保留，`OVERWRITE=true` 才重新采集。`validate` 会独立回放全部动作，检查决策 pose、最终 pose、碰撞和目标到达，全部通过后清理轨迹检查点。

## Instruction 与图片制作

[create_instructions.sh](create_instructions.sh) 默认读取统一轨迹文件，GPU 0–7、16 个 Habitat 进程，每个进程并发处理 9 条轨迹，API 并发上限为 144；`LIMIT=0` 处理全部轨迹。只将未完成任务按场景分批，空闲进程从共享队列领取下一批。API 等待可以重叠，渲染仍在每个进程的主线程依次执行，每张卡只常驻两个模拟器。

```bash
./dataset_create/create_instructions.sh inspect
./dataset_create/create_instructions.sh generate
```

按自然片段、视觉材料、局部语言、轻量整理、局部验证这五阶段处理。通过验证后导出 clean ERP 和标准 R2R；无法清楚描述或验证的轨迹隔离。详情见 [instruction/README.md](instruction/README.md)。

中断后使用相同命令继续；不要在未完成时删除 `.work/instruction`。已完成轨迹跳过，未完成轨迹复用已保存的局部文本和模型响应。最后输出 `image/`、`train.json` 和 `train.json.gz`。

## 已有轨迹合并

现有的两部分轨迹已合并为 **105,307 条**统一训练输入。共检查 900 个场景，其中 872 个产生符合要求的轨迹。每条轨迹的 ID、动作、pose、决策事件与停止位置均保持原样，没有重新采集或重规划路线。

旧目录及历史测试输出已移到 `/workspace/data2/dataset/general_VLN_data/PanoVLN_previous_layout/` 备份；生产入口只读取新的统一轨迹文件，已有轨迹不必重新制作。

[trajectory/merge.py](trajectory/merge.py) 可合并参数一致、场景互不重叠的已有轨迹集，同时核对 ID、逐场景数量并合并统计。已有回放结论明确标为继承自内容未变的源轨迹，不当作一次新的全量回放。

```bash
python -m dataset_create.trajectory.merge \
  --datasets /path/to/first.json.gz /path/to/second.json.gz \
  --stats /path/to/first_stats.json /path/to/second_stats.json \
  --output-root /workspace/data2/dataset/general_VLN_data/PanoVLN/trajectory
```
