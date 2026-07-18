# PanoVLN-HM3D 数据集自动制作

`PanoVLN` 是论文、模型与数据制作方法的总名称；本目录产出的 HM3D 自采数据集
称为 `PanoVLN-HM3D`，与 ScaleVLN instruction rewrite ablation 明确区分。

`data_create/` 从 HM3D 建模文件直接制作新的 VLN 数据，不依赖 ScaleVLN 的
trajectory 或 instruction。最终产物是带 reference path 和自然语言 instruction 的
VLN-CE episode、逐帧全景图以及与图像严格对齐的 expert GT actions。

```text
HM3D basis.glb + basis.navmesh
  -> 高质量 R2R-short / RxR-long trajectory
  -> ShortestPathFollower expert actions
  -> 轨迹回放与 360°视觉证据采集
  -> 视觉证据抽取、结构化路线计划、写作、独立审核和修复
  -> R2R VLN-CE train.json + train.json.gz + train_gt.json.gz
```

## 目录

```text
data_create/
├── README.md
├── run_data_creation.sh
├── config/hm3d_vln.yaml
├── config/trajectory_profiles.json
├── trajectory/
│   ├── collect_hm3d.py
│   └── generate_gt.py
├── instruction/
│   ├── actions.py
│   ├── evidence.py
│   ├── prompts.py
│   ├── qa.py
│   ├── runner.py
│   └── pipeline.py              # 兼容入口，实际逻辑拆在上述模块
└── export_vlnce.py
```

代码目录不保存轨迹、图片、进度或验证报告。质量检查直接作为各阶段的发布门，
失败时命令返回非零并拒绝生成正式文件。

## 为什么保留全景图

全景图只是可复用的 360°视觉母图，不会原样输入 VLM。每个被选中的轨迹位置
都会从全景母图投影出普通透视图：

- route：left / forward / right / back；
- start 和 endpoint：额外包含 forward-down；
- VLM 收到的是按路线顺序排列的透视图 contact sheet。

这样既保留了全景 VLN 需要的完整空间信息，又避免让通用 VLM 直接理解畸变较大
的 equirectangular 图。路口侧面、身后 landmark 和最终目标不容易因为相机初始
朝向而遗漏；修改视角、FOV 或 agent 提示词时，也不必重新启动 Habitat 渲染。
最终 VLN-CE dataset 不引用这些图片，它们主要是 instruction 制作证据，也可供
后续全景训练数据预处理复用。

正式配置使用 `2048×1024` 全景、JPEG 质量 92，并投影为 `384×288` 透视 tile、
contact sheet JPEG 质量 90。只提高全景而保留原来的 `256×192` tile 无法明显改善
VLM 最终看到的 landmark；两级分辨率必须一起提高。`512×384` tile 会让长路线的
多图请求接近当前 Qwen3.6-27B 的上下文上限，因此没有作为默认值。
正式图片尺寸由 `run_data_creation.sh` 的 `PANORAMA_WIDTH/HEIGHT` 传给
`render-panoramas`；`hm3d_vln.yaml` 也保持相同的 2K 默认。GT 阶段使用
`--minimal-observations` 关闭 RGB sensor，所以不会为求 action 额外渲染高清图。
与原来的 `1280×640` 相比，2K 全景的渲染像素约为 2.56 倍，应按磁盘容量调整
`NUM_TRAJECTORIES` 分批生产。

## 配置与运行

所有用户配置都在 [run_data_creation.sh](run_data_creation.sh) 顶部直接赋值：

```bash
SAVE_ROOT="/workspace/data1/dataset/PanoVLN/generated/PanoVLN-HM3D"
SCENE_ROOT="/workspace/data1/dataset/general_VLN_data/HM3D"
SCENE_SPLIT="train"
TRAJECTORY_PROFILE_CONFIG="${REPO_ROOT}/data_create/config/trajectory_profiles.json"
NUM_TRAJECTORIES=110000
R2R_RATIO=0.40
RXR_RATIO=0.60
SEED=42
GOAL_RADIUS=0.3
GPU_DEVICE_ID=0

MAX_ANCHOR_TURN_DEGREES=90
VISUAL_CHECK_WIDTH=512
VISUAL_CHECK_HEIGHT=256
SENSOR_HEIGHT=1.25
MAX_VISUAL_CHECKPOINTS=24
MAX_BLACK_RATIO=0.10

PANORAMA_WIDTH=2048
PANORAMA_HEIGHT=1024
PANORAMA_JPEG_QUALITY=92
TILE_WIDTH=384
TILE_HEIGHT=288
SHEET_JPEG_QUALITY=90

BASE_URL="http://127.0.0.1:10420/v1"
MODEL="Qwen3.6-27B"
API_KEY="test"
NUM_WORKERS=40
STAGE="full"
```

`SCENE_ROOT` 是唯一的场景根目录，内部应具有下面的 split-aware 结构：

```text
SCENE_ROOT/
├── train/<scene_dir>/<scene_name>.basis.glb
├── train/<scene_dir>/<scene_name>.basis.navmesh
└── val/...
```

制作阶段的采样、GT 和图片渲染都使用这个根目录。公开 episode 按 Habitat 官方
目录约定写成 `hm3d/train/<scene_dir>/<scene_name>.basis.glb`，不写机器相关的绝对路径；
制作代码会自动去掉 `hm3d/` 命名空间再访问 `SCENE_ROOT`。开源用户加载正式数据时，
将 Habitat `SCENES_DIR` 指向包含 `hm3d/` 的统一 `data/scene_datasets` 根目录即可。

`GOAL_RADIUS=0.3` 由轨迹采集、ShortestPathFollower GT、严格 replay validation
和最终 train dataset 共同使用，避免生成停止标准与发布 episode 的 goal 定义不一致。
数据结构和词表保持 R2R VLN-CE 兼容，但 radius 不逐值复刻 R2R train 的 3.0 m；
它采用更接近本数据 GT 终点的 0.3 m 训练定义。

运行：

```bash
cd /workspace/code/VLN
bash data_create/run_data_creation.sh
```

`STAGE` 可取：

- `collect`：按 scene 批量采集 trajectory；
- `gt`：生成 expert actions，严格验证后生成 agent 输入；
- `images`：回放 expert actions 并采集全景视觉证据；
- `instruction`：为同一批 trajectory 生成 concise 和 dense instruction；
- `export`：发布 `train.json`、同内容的 `train.json.gz` 以及
  `train_gt.json.gz`，随后清理中间文件；
- `full`：依次完成全部阶段。

## 场景覆盖与 trajectory 分配

`R2R_RATIO` 和 `RXR_RATIO` 控制两类 trajectory 的数量比例，必须是非负有限数且
在浮点容差内相加等于 1。R2R 数量按 `NUM_TRAJECTORIES * R2R_RATIO` 取最近整数，RxR 使用
剩余数量，因此总数不会因取整变化。例如当前 110,000 条、40%/60% 会严格生成
44,000 条 R2R-like trajectory 和 66,000 条 RxR-like trajectory。family 列表再按
`SEED` 做可复现打乱；失败的候选只会在当前 family 内重采，不会悄悄改变最终比例。
该比例对完整数据集全局成立，不要求每个 scene 内部也恰好达到 40%/60%；采集摘要会
同时打印目标数量、实际数量和实际比例，目标与实际数量不一致时不会通过质量门。

R2R-like 的目标长度来自 R2R action 统计并限制在 5--20 m，使用起终点最短路径；
RxR-like 的目标长度来自 RxR action 统计并限制在 5--40 m，且可包含通过全部几何
质量门的合理 detour。`trajectory_profiles.json` 只保存 MOVE_FORWARD 数量换算得到的
匿名距离直方图，不含两套数据的 instruction、trajectory、scene ID 或 episode ID；
因此开源运行不依赖仓库外的 R2R/RxR JSONL。每条最终 trajectory 仍会分别生成
concise 与 dense 两个 instruction，语言版本数量不受这两个 trajectory 比例控制。
`R2R_RATIO/RXR_RATIO` 决定两类轨迹各采多少，profile 则决定每类内部的目标长度
按什么频率抽样；它是当前 collector 的必需输入，不是额外数据集产物。
直方图的来源和复现元数据写在配置文件内：R2R 使用 `R2RVLNCE-v1`
train，RxR 使用 `RxRVLNCE-v1` train 的 guide / en-US / en-IN；先用同仓库
`habitat_shortest_path.py` 生成 action JSONL，再按
`distance_m = count(action == MOVE_FORWARD) * 0.25` 计算并做闭区间筛选。
该长度先验应在论文和数据发布页归因
[R2R](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R)
与 [RxR](https://github.com/google-research-datasets/RxR)，并遵守各自的许可和数据使用条款。

正式脚本没有 `MAX_SCENES`。默认使用 `SCENE_SPLIT` 下所有同时具有
`basis.glb` 和 `basis.navmesh` 的场景，并按照 `SEED` 做可复现乱序，避免固定取
字典序靠前的少数场景。

当前分配规则会在剩余场景间动态均分剩余 trajectory：所有场景都能采满时，每个
场景的数量最多相差 1；某个场景无法产生当前要求的长轨迹或 detour 时，其缺口会
由后续场景承担，因此不承诺严格等量。阶段结束时会在终端打印实际场景数量范围。
若总数不足，collection 只写入隐藏工作区的 partial 文件并失败，不会把不完整数据
伪装成正式数据集。

需要人工指定场景集合时，可直接调用 collector 的 `--scene-ids`；正式默认入口不
提供按前 N 个场景截断的选项。

## Scene batching

三个 Habitat 阶段都不会逐 episode 切换场景：

- collect 外层按 scene 循环，在同一个 simulator 中采完该 scene 的 quota；
- GT 先按 `scene_id` 分组，同一个 Habitat Env 处理该 scene 的全部 episode；
- image render 按 `(scene_id, episode_id)` 排序，并持续复用当前 simulator；
- instruction 阶段只读取已经保存的图片，不启动 Habitat。

每个 scene 在 collect、GT、render 三个独立阶段各加载一次。GT 支持 journal 断点续跑；
collect 和 render 失败时可只重跑当前阶段，无需重新执行已经完成的前置阶段。

## 输出

成功完成 `export` 后，公开产物只有：

```text
SAVE_ROOT/
├── images/<trajectory_id>/frame_*.jpg
├── train.json
├── train.json.gz
└── train_gt.json.gz
```

原始 trajectory GT、agent input、instruction JSONL 和生成进度只存在于
`SAVE_ROOT/.work/`。全部检查通过并发布 dataset/GT 文件后，脚本自动删除
`.work/`；中途失败时保留它以便从对应阶段恢复。旧的 `subset_00` 是无实际分片的
遗留层级，已经删除。

复用旧 `SAVE_ROOT` 时，成功发布还会清理历史未压缩 `train_gt.json` 和
`images.failed/`，避免它们混入正式数据目录。

`train_gt.json.gz` 采用 R2R schema，key 与最终 `train.json` 的 episode ID 严格
一致；每条记录只含 `locations`、`actions`、`forward_steps`。actions 最后一项是
唯一 STOP=0，concise/dense 两个 episode 共享完全相同的 GT。原始无 STOP GT 仍
只在 `.work/` 中，exporter 核验后生成最终配对 GT。

图片与 action 的对应关系也是发布硬检查：`frame_0.jpg` 是起始状态；对最后的
STOP 之前每个 `actions[k]`，执行后得到 `frame_{k+1}.jpg`；最后一个 STOP 在最终
frame 执行。因此每条 trajectory 都满足
`图片数 == len(train_gt[episode_id].actions)`（这里 actions 已包含 STOP）。

读取时使用两个不同的 ID：

```python
episode_gt = train_gt[str(episode["episode_id"])]
frame_dir = image_root / str(episode["trajectory_id"])
```

不能默认用 `episode_id` 查图片。这样同一 trajectory 的 concise/dense episode 各有
独立 R2R-compatible GT key，却共享同一个图片目录。仓库的
`src/data/prepare_training_data.py` 已改为优先使用 annotation 的 `trajectory_id`，
字段不存在时回退 `episode_id`，所以旧 R2R/RxR/ScaleVLN 三字段 annotation 行为不变。
把本数据转换为训练 JSONL 时必须保留 `trajectory_id`。

`train.json` 与 `train.json.gz` 内容完全相同，使用脚本统一配置的 0.3 m goal radius
和整数 episode/trajectory ID。concise/dense 共享完全相同的场景、trajectory、制作期
actions 和图片，仅改变语言密度；actions 放在按最终 episode ID 索引的
`train_gt.json.gz`，不嵌入 episode。

`instruction_vocab` 只保留与 ScaleVLN 一致的空结构，各 episode 的
`instruction_tokens` 固定为 `null`。空根字段供 `R2RVLNCE-v1` Habitat dataset loader
完成 schema 解析；训练和评测只消费 `instruction_text`，由各研究者使用自己的
tokenizer，不在发布数据中绑定 R2R 的旧 vocabulary，也不做无实际意义的 OOV 检查。

最终 JSON 的根字段和 episode 字段与现有 `ScaleVLN_150k/scalevln_subset_150k.json.gz`
一致：根节点只有 `episodes`、`instruction_vocab`；episode 保留 `scene_id`、起点
位姿、goal、`reference_path`、`trajectory_id`、instruction 和 geodesic distance，
不包含 `actions` 或 `locations`；这些字段按 R2R 约定放在 GT 文件中。也可以直接
从 reference path 重新调用 Habitat 复算，使用本目录 `config/hm3d_vln.yaml` 中的
0.25 m forward step、15° turn 和同一个 0.3 m goal radius。
同一条 trajectory 的多个语言版本共享 `trajectory_id`，该 ID 也就是对应图片目录名。

## 质量门

Trajectory 至少检查：

- 起终点位于主 navmesh island，并具有 obstacle clearance；
- 距离、楼层数和 reference/shortest ratio 与 route family 匹配；
- 每条边连通，不出现尖角、非局部 revisit、反向共享边或原路折返；
- detour anchor 不形成不自然的 turn-around；
- 进入目标邻域后不再次明显离开；
- 接受前沿 route keypoint 稀疏渲染 `512×256` 全景，任一检查点近黑扫描
  缺失比例超过 `0.10` 就换一条路线；
- route hash 唯一，expert replay 无碰撞且最终到达目标。

`MAX_ANCHOR_TURN_DEGREES=90` 来自同 seed 的 RxR 100 条对照：相对 105°，两组
都 100/100 采满，时间从 139.5 s 变为 140.6 s，但 100°以上的 synthetic anchor
急转从 6 条降为 0。`MAX_BLACK_RATIO=0.10` 也做了 mixed 100 条对照：两组都采满，
0.35 会放入 11 条超过 0.10 的路线（其中 8 条人工确认存在大面积 scan void）；
0.10 的 accepted p95/max 为 0.042/0.094，耗时只从 133.5 s 增至 144.3 s。
高清 `2048×1024` render 会用同一个阈值再查完整 action replay，低清预检负责在
昂贵 GT、高清渲染和 VLM 调用之前淘汰明显脏路线。

Instruction 系统的正式顺序为：

```text
actions + selected panorama frames
  -> perspective evidence sheets: START / ROUTE / ENDPOINT
  -> Qwen structured route plan + draft instruction
  -> deterministic route/endpoint/format gate
  -> Qwen blind grounding audit
  -> repair when either gate fails
  -> atomic clean JSONL publication
```

自采数据使用 `generate` 模式，进入 agent 前会清除任何 source instruction。
`validate-gt` 还会把 simulator GT 的升降统计作为只读 trajectory metadata 交给
同一套系统：楼梯上/下方向以真实 elevation 为硬约束，VLM 只负责识别楼梯和地标。
对仅有 actions 的旧数据，系统用离散动作做保守的 dead-reckoning 复杂度提示，避免
长路线被压成直线路线。最终 clean JSONL 只保留 `episode_id`、`instruction`、`actions`、
`instruction_profile`、`input_fingerprint`、`pipeline_fingerprint` 和可选
`trajectory_id`；contact sheet、raw response、repair/audit 结果只在 work dir 中用于恢复
和人工审查。
任何 trajectory、图片或 instruction 未通过硬检查时，正式 dataset 都不会发布。
