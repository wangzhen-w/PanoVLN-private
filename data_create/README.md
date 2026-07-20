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
  -> 视觉证据抽取、长路线分段理解、结构化路线计划、写作、独立审核和修复
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
│   ├── generate_gt.py
│   └── parallel_gt.py
├── instruction/
│   ├── actions.py
│   ├── evidence.py
│   ├── prompts.py
│   ├── qa.py
│   ├── runner.py
│   ├── trajectory_metadata.py
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

保存下来的全景已经按 agent 当前朝向对齐，正式脚本刻意不传
`--use-action-heading`，避免根据离散 actions 再旋转一次导致 FINAL FORWARD
指向侧面物体。只有确认输入全景是全局固定朝向时，才应打开该选项。

instruction 系统不会要求 Qwen 一次读完超长路线的巨大拼图。简单路线直接使用
START / ROUTE / ENDPOINT 三张 evidence sheet；长路线或“平移多且包含多个大转向”
的复杂路线默认在 `auto` 模式下使用分段证据：
先把 route waypoints 切成带 1 行 overlap 的局部分段，让 Qwen 为每段抽取
grounded route facts，再由全局 writer 结合 START / ROUTE overview / ENDPOINT
合并成一条自然 instruction。每个分段还带真实 low-level action span 摘要，
明确哪些视觉变化只是原地转向或小范围对齐，避免把相机朝向变化误写成“穿过某个房间”。

这样既保留了全景 VLN 需要的完整空间信息，又避免让通用 VLM 直接理解畸变较大
的 equirectangular 图。路口侧面、身后 landmark 和最终目标不容易因为相机初始
朝向而遗漏；修改视角、FOV 或 agent 提示词时，也不必重新启动 Habitat 渲染。
最终 VLN-CE dataset 不引用这些图片，它们主要是 instruction 制作证据，也可供
后续全景训练数据预处理复用。

正式配置使用 `1600×800` 全景、JPEG 质量 92，并投影为 `384×288` 透视 tile、
contact sheet JPEG 质量 90。只提高全景而保留原来的 `256×192` tile 无法明显改善
VLM 最终看到的 landmark；两级分辨率必须一起提高。`512×384` tile 会让长路线的
多图请求接近当前 Qwen3.6-35B-A3B 的上下文上限，因此没有作为默认值。
正式图片尺寸由 `run_data_creation.sh` 的 `PANORAMA_WIDTH/HEIGHT` 传给
`render-panoramas`，因此该脚本中的值是发布图片分辨率的唯一配置。GT 阶段使用
`--minimal-observations` 关闭 `hm3d_vln.yaml` 的 RGB sensor，所以 YAML 中的 sensor
尺寸不会改变正式图片，也不会为求 action 额外渲染高清图。
与原来的 `1280×640` 相比，当前全景的渲染像素约为 1.56 倍，应按磁盘容量调整
`NUM_TRAJECTORIES` 分批生产。

## 配置与运行

所有用户配置都在 [run_data_creation.sh](run_data_creation.sh) 顶部直接赋值：

```bash
SAVE_ROOT="/workspace/data1/dataset/general_VLN_data/PanoVLN"
SCENE_ROOT="/workspace/data1/dataset/general_VLN_data/HM3D"
SCENE_SPLIT="train"
TRAJECTORY_PROFILE_CONFIG="${REPO_ROOT}/data_create/config/trajectory_profiles.json"
NUM_TRAJECTORIES=210000
COLLECT_RECOVERY_ROUNDS=1
R2R_RATIO=0.35
RXR_RATIO=0.65
SEED=42
GOAL_RADIUS=0.3
GPU_DEVICE_IDS="0,1,2,3,4,5,6,7"
COLLECT_PROCESSES_PER_GPU=2
GT_PROCESSES_PER_GPU=2
RENDER_PROCESSES_PER_GPU=2

MAX_ANCHOR_TURN_DEGREES=90
VISUAL_CHECK_WIDTH=512
VISUAL_CHECK_HEIGHT=256
SENSOR_HEIGHT=1.25
MAX_VISUAL_CHECKPOINTS=24
MAX_BLACK_RATIO=0.10

PANORAMA_WIDTH=1600
PANORAMA_HEIGHT=800
PANORAMA_JPEG_QUALITY=92
TILE_WIDTH=384
TILE_HEIGHT=288
SHEET_JPEG_QUALITY=90

ROUTE_EVIDENCE_MODE="auto"
SEGMENTED_MIN_ACTIONS=80
SEGMENT_MAX_WAYPOINTS=0
SEGMENT_ROWS=5
SEGMENT_OVERLAP=1
SEGMENT_FACT_MAX_TOKENS=650

BASE_URL="http://127.0.0.1:10420/v1"
MODEL="Qwen3.6-35B-A3B"
API_KEY="test"
NUM_WORKERS=96
EVIDENCE_WORKERS=8
STAGE="full"
```

`GPU_DEVICE_IDS` 是逗号分隔的 Habitat GPU 列表，例如 `"0,1,2,3"`。
collect、GT 与 render 的实际进程上限分别是 GPU 数乘以对应的
`*_PROCESSES_PER_GPU`。当前 8 卡配置分别启动最多 16、16、40 个进程；若需与其他任务共享
GPU，直接缩短列表或降低相应阶段的每卡进程数。
`NUM_WORKERS` 表示同时进入多 agent/Qwen 阶段的 episode 数；
`EVIDENCE_WORKERS` 是独立 CPU 进程数，负责全景图解码、透视投影和 evidence sheet 编码。
两者与 Habitat 进程数无关。evidence 阶段使用有界预取队列，不会把全部 episode
一次性读入内存。
GT 阶段关闭 RGB observations，因此通常主要受 Habitat 路径规划和 CPU 吞吐限制。

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
- `instruction`：为每条 trajectory 生成一条经过候选筛选和审核的 Dense instruction；
- `export`：发布 `train.json`、同内容的 `train.json.gz` 以及
  `train_gt.json.gz`，随后清理中间文件；
- `full`：依次完成全部阶段。

## 场景覆盖与 trajectory 分配

`R2R_RATIO` 和 `RXR_RATIO` 控制两类 trajectory 的数量比例，必须是非负有限数且
在浮点容差内相加等于 1。`NUM_TRAJECTORIES` 是期望规模而非发布下限：R2R 期望数量按
`NUM_TRAJECTORIES * R2R_RATIO` 取最近整数，RxR 使用剩余数量。family 列表再按
`SEED` 做可复现打乱；失败候选只在当前 family 内重采，不会悄悄改变比例。
该比例对完整数据集全局成立，不要求每个 scene 内部也达到 35%/65%。若 HM3D 在所有
硬质量门下只能提供约 190k 条，collector 会确定性发布不超过 190k 的最大 35%/65%
平衡子集，而不会降低几何或视觉质量标准来凑到 210k。终端摘要同时打印期望数量、
实际数量和 family counts。

R2R-like 的目标长度来自 R2R action 统计并限制在 5--20 m，使用起终点最短路径；
RxR-like 的目标长度来自 RxR action 统计并限制在 5--40 m，且可包含通过全部几何
质量门的合理 detour。`trajectory_profiles.json` 只保存 MOVE_FORWARD 数量换算得到的
匿名距离直方图，不含两套数据的 instruction、trajectory、scene ID 或 episode ID；
因此开源运行不依赖仓库外的 R2R/RxR JSONL。每条最终 trajectory 只生成一条
Dense instruction；这里的 Dense 表示信息足以清楚执行路线，并不要求逐 action 描述。
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
若总数不足，collection 仍只发布所有硬检查都通过的数据；不足目标的状态会在终端
标为 `quality_limited`，而不是把期望规模伪装成实际规模。

需要人工指定场景集合时，可直接调用 collector 的 `--scene-ids`；正式默认入口不
提供按前 N 个场景截断的选项。

## Scene batching

三个 Habitat 阶段都不会逐 episode 切换场景：

- collect 将互不重叠的 scene 分片交给独立进程，每个进程在同一个 simulator 中
  采完当前 scene 的 quota；同一个 scene 不会同时交给两个 GPU worker；
- GT 将完整 scene 按预计路径长度平衡给多个 GPU worker，同一个 Habitat Env 处理
  该 scene 的全部 episode；
- image render 以完整 scene 为调度单元，并按预计 frame 数平衡各进程负载；一个
  worker 加载 scene 后连续回放其中所有待处理 episode；
- instruction 阶段只读取已经保存的图片、actions 和 trajectory metadata，不启动
  Habitat；其中 `reference_path` 会自动派生 `vertical_motion`，作为上/下楼描述的
  硬约束传给 writer、audit 和 deterministic QA。

每个被处理的 scene 在 collect、GT、render 三个独立阶段内各自只由一个 worker 加载。
GT 为各 worker 保存独立 journal，主进程显示聚合进度条；它也会继承旧单进程
`GT_JOURNAL` 已完成的 episode。全部完成后按 dataset 顺序原子合并 GT 并删除 worker
状态，因此中断后使用相同 GPU/进程配置重跑即可续跑；
render 也会为每个已通过回放检查的 episode 原子保存图片目录并追加 journal。中断后
重新运行同一个 `images` 阶段，会校验 dataset、GT 和渲染参数 fingerprint，跳过已完成
episode，并继续显示一个聚合的 `panorama` 进度条；同一个 scene 的待处理 episode
仍会一起回放。GPU 列表和每卡进程数是运行调度参数，不进入图片 fingerprint，因此
恢复时可以按当时的空闲显存调整并行度。
在 `full` 阶段中，每个原子发布的阶段产物都会直接作为完成标记：已有 trajectory 时
跳过采样，同时已有 GT 与 agent input 时跳过 GT 生成和校验，已有正式 `images/` 时跳过
渲染，已有最终 instruction JSONL 时跳过 instruction。instruction 中断时只读取 agent
input 和 progress journal，不检查 trajectory、GT，也不为输入或图片计算 fingerprint。
每条 `status=success` 的 journal 记录直接按 `episode_id` 视为完成。因此在任一阶段中断后
可以直接重新运行正式脚本，从最近一个
未完成阶段继续。显式选择单独的 `gt`、`images`、`instruction` 等 `STAGE` 仍会执行该
阶段，便于需要时主动重建。instruction 调度器将 CPU evidence 进程池与 API 线程池分开，
只保留一个有界预取窗口，不会把全部 episode 预先塞入 executor；
中断时会取消尚未开始的任务，已经完成的结果仍保留在 journal 中。
恢复时进度条从 journal 中累计成功数开始显示。如果主动更换了 prompt、视觉证据或
agent 工作流并希望全部重写，应使用新的 `INSTRUCTION_WORK_DIR`，或先删除旧版本的
progress 目录；同一个 work dir 的语义就是继续同一批生成任务。
collect 本身需要
原子发布目标规模或 quality-limited 规模后才进入后续阶段；采集过程中会把每条已接受
trajectory 和 sampler 状态事务性地
写入 SQLite checkpoint。单进程时路径是
`trajectories.json.gz.collect.sqlite3`；多进程时每个 scene shard 在
`trajectories.json.gz.collect_workers/` 下维护独立 checkpoint，主进程只显示一个聚合的
`trajectory` 进度条。若在 collect 内部中断，重新运行同一个 `collect` 或 `full` 阶段
会校验各分片采样配置 fingerprint 并继续；恢复后的待采样 episode 仍按 scene 成批处理。
首轮某个 scene shard 即使耗尽自己的场景也不会拖垮其他 worker；主进程会等待所有
分片结束，再按缺少的 R2R/RxR 数量进行最多 `COLLECT_RECOVERY_ROUNDS` 轮全场景补采。
默认只补采一轮；结束后对 route hash 去重，并按目标比例保留最大可用规模。所有轨迹
重新编号并通过全局检查后才原子发布。checkpoint 和 worker 临时目录随后自动删除，
不会成为公开产物。

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
`SAVE_ROOT/.work/`。图片渲染中断时，已验证图片和 resume journal 暂存在
`SAVE_ROOT/images.rendering/`；成功后该目录会原子发布为 `images/`，其中的状态文件
会在发布前删除。全部检查通过并发布 dataset/GT 文件后，脚本自动删除
`.work/`；中途失败时保留它以便从对应阶段恢复。旧的 `subset_00` 是无实际分片的
遗留层级，已经删除。

复用旧 `SAVE_ROOT` 时，成功发布还会清理历史未压缩 `train_gt.json` 和
`images.failed/`，避免它们混入正式数据目录。

`train_gt.json.gz` 采用 R2R schema，key 与最终 `train.json` 的 episode ID 严格
一致；每条记录只含 `locations`、`actions`、`forward_steps`。actions 最后一项是
唯一 STOP=0。每条物理 trajectory 对应一个 episode 和一份 GT；原始无 STOP GT 仍
只在 `.work/` 中，exporter 核验后生成最终 GT。

图片与 action 的对应关系也是发布硬检查：`frame_0.jpg` 是起始状态；对最后的
STOP 之前每个 `actions[k]`，执行后得到 `frame_{k+1}.jpg`；最后一个 STOP 在最终
frame 执行。因此每条 trajectory 都满足
`图片数 == len(train_gt[episode_id].actions)`（这里 actions 已包含 STOP）。

读取时使用两个不同的 ID：

```python
episode_gt = train_gt[str(episode["episode_id"])]
frame_dir = image_root / str(episode["trajectory_id"])
```

不能默认用 `episode_id` 查图片。`trajectory_id` 明确记录图片所属的物理轨迹，
`episode_id` 则与 R2R-compatible GT key 对齐。仓库的
`src/data/prepare_training_data.py` 已改为优先使用 annotation 的 `trajectory_id`，
字段不存在时回退 `episode_id`，所以旧 R2R/RxR/ScaleVLN 三字段 annotation 行为不变。
把本数据转换为训练 JSONL 时必须保留 `trajectory_id`。

`train.json` 与 `train.json.gz` 内容完全相同，使用脚本统一配置的 0.3 m goal radius
和整数 episode/trajectory ID。最终数据保持“一条物理 trajectory、一条 Dense
instruction、一个 episode”；actions 放在按最终 episode ID 索引的
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
`trajectory_id` 是物理轨迹 ID，也就是对应图片目录名。

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
正式 `1600×800` render 会用同一个阈值再查完整 action replay，低清预检负责在
昂贵 GT、高清渲染和 VLM 调用之前淘汰明显脏路线。

Instruction 系统的正式顺序为：

```text
actions + selected panorama frames
  -> perspective evidence sheets: START / ROUTE / ENDPOINT + forward-first FINAL-only
  -> Qwen semantic stop-location extraction from ENDPOINT approach + labeled FINAL views
  -> Qwen endpoint fact audit; failed corrections require a fresh verification
  -> long or action-complex routes: segment sheets + action-span-grounded route facts
  -> grounded Qwen writer drafts from verified semantic route/endpoint facts
  -> independent language-realization writer reorganizes the grounded draft without new visual claims
  -> independent visual judge reads raw evidence and verifies both drafts; publish the approved realization or fall back to the grounded draft
  -> deterministic format and simulator-geometry gate
  -> Qwen blind grounding audit reads raw evidence without derived facts/plans
  -> blind correction and re-audit when grounding fails
  -> atomic clean JSONL publication
```

自采数据使用 `generate` 模式，进入 agent 前会清除任何 source instruction。
`validate-gt` 还会把 simulator GT 的升降统计作为只读 trajectory metadata 交给
同一套系统：楼梯上/下方向以真实 elevation 为硬约束，VLM 只负责识别楼梯和地标。
对仅有 actions 的旧数据，系统用离散动作做保守的 dead-reckoning 复杂度和分段
action-span 提示，避免长路线被压成直线路线，也避免把原地转向误写成穿过房间。
endpoint fact pass 会同时读取最后几个平移位置和带文本标签的 FORWARD / FORWARD-DOWN /
LEFT / RIGHT / BACK 单视角图，把“相机当前站立位置”与“正前方物体是否已到达”分开，专门抽取最终 stop 的位置类型、当前站位 anchor、
正前方/侧向/身后 anchor 和必须避免的终点/朝向说法；随后再用同一组 FINAL views 做一次
endpoint fact audit，检查 forward/side-view 是否串列，以及前方 anchor 是已到达还是在 FINAL 后仍位于前方。
endpoint agent 只输出结构化语义事实，不提前起草可发布的 stop phrase。audit 如果改写了 facts，改写结果必须再通过一次验证才能进入 writer；持续失败
时丢弃整份派生 endpoint facts，writer 回退到原始 ENDPOINT / FINAL evidence，而不是传播
未验证事实或让整批任务失败。grounded writer 只受路线顺序、关键决策、视觉 grounding、楼梯方向和
终点语义约束；language-realization writer 以该事实底稿为内容边界，重新组织自然表达，不重复做一次
容易产生不同路线解释的视觉规划。系统不规定开头、句数、句法、导航动词或必须出现 `stop/wait`；
起点背景、直接动作、朝向、空间过渡或 landmark 都可以自然组织在开头，`Start` 既不要求也不禁止，
但不能让任一形式成为整套数据的默认起手式。长度随路线复杂度
自适应，并把相邻语义事件组织成连贯指引，而不是逐 action 复述。endpoint facts 只帮助 grounded writer
生成底稿，不再被 deterministic QA 当作视觉真值。candidate judge 和最终 blind audit 只读取原始视觉、actions 和几何
约束，从而能够纠正多个候选共享的错误 endpoint 解释。judge/audit 必须先分别写出
原始证据中的空间序列、instruction 实际表达的空间序列以及前方 anchor 是否在 FINAL 相机处已到达，
只有路线与 endpoint 两项都匹配才允许发布。
为避免针对少数 episode 堆物体级硬规则，默认每条路线保留 grounded draft 和独立 language realization。
language-realization agent 会按全局 `SEED` 和 `episode_id` 稳定抽取一个高层 discourse intent：
action-led、context-led、transition-led、orientation-led、progress-led 或 free。它们只提示信息组织重心，
不提供固定首词或句式；不符合当前视觉事实时允许自然回退。context-led 明确允许 `Start/Begin`，因此该机制
控制的是大规模语料的表达分布，而不是把某个合法词设为禁词。
候选在展示给 visual judge 前会随机匿名排序并按展示顺序重新编号，避免候选编号与 agent 职责绑定造成
选择偏差。language realization 只有在独立 visual candidate judge 根据原始 EARLY ROUTE / ROUTE
segments / ENDPOINT / FINAL evidence 确认路线与终点都匹配时才发布，否则回退到 grounded draft；
deterministic QA 只负责格式、数据泄漏、
空输出、内部数据痕迹和 simulator/GT 提供的通用物理约束。词数、句数、固定词、颜色材质、
朝向短语和 landmark 词表不作为发布硬门。模型派生 facts 与 instruction 的
不一致只记为诊断 warning，最终语义发布门由独立视觉 audit 决定。
视觉 evidence 构建会在单个 episode 内复用已解码全景、透视投影和固定相机映射；
route、segment、endpoint 和 FINAL tiles 中的重叠视图不重复计算。这些是像素等价的性能优化，
不改变 VLM 实际解码到的图像、prompt 或 agent 调用顺序。
最终 clean JSONL 只保留 `episode_id`、`instruction`、`actions`、
`instruction_profile` 和可选 `trajectory_id`；contact sheet、raw response、
repair/audit 结果只在 work dir 中用于恢复
和人工审查。

Qwen 客户端默认通过请求字段
`chat_template_kwargs={"enable_thinking": false}` 显式关闭 thinking；prompt 中不再附加
`/no_think` 文本指令。需要实验 thinking 模式时可传 `--disable-thinking false`。
GT 和高清图片回放后的逐 episode 硬检查都会自动同时过滤 trajectory、GT、agent
输入和图片中的失败样本，因此最终规模可以略低于采样规模；质量优先于凑齐精确
数量。图片阶段只汇总已记录的帧数，不会为报告重新读取数 TB JPEG 计算全量哈希。
Instruction 断点续跑会复用已成功 episode，也会识别已穷尽生成、修复和复审的质量失败；
后者是终止状态，不会在下次启动时重试，并从最终 instruction、dataset、GT 和图片集同步剔除。
API 超时、服务不可用、文件读取异常等运行时失败不会被误判为低质量样本，仍会续跑。
正式 agent input 和图片树一经发布即视为
不可变。Qwen 启动前不会遍历图片或计算哈希。实际构建每条
视觉证据时仍会严格检查 action/frame 数量并读取对应图片。
只要还有运行时失败或其他未完成 episode，正式 dataset 就不会发布；终止质量失败不会阻塞其余高质量数据发布。

## Instruction 迭代验收

任何 prompt、视觉证据、分段、QA 或 repair 逻辑的优化，都必须用同一批 episode
做 previous/current 对比，不能只看当前输出。最小验收集应同时包含短路线、长路线、
低于长度阈值但含多个转向的复杂路线、楼梯/landing、跨房间/走廊转换、复杂办公/开放空间，以及至少一条
HM3D 自采长路线。对 ScaleVLN rewrite，还要同时展示原始 ScaleVLN instruction、
历史 `/sub_dataset/PanoVLN.jsonl` rewrite、上一版系统输出和当前输出。

每轮优化后的人工审查至少需要 5 个独立评审视角：

- 路线执行性：follower 是否能根据 instruction 到达正确终点；
- 视觉 grounding：写出的 landmark、房间名、终点 anchor 是否被图像支持；
- 长路线/楼梯/空间转换：上/下楼、landing、hallway/room transition 是否清楚；
- 语言自然度：是否把语义事件组织成连贯路线，而不是机械逐 action 复述；不要求模仿固定 R2R/RxR 表面风格；
- 回归与数据集质量：是否相比上一版整体提升，短路线是否未退化，失败率和成本是否可接受。

只有当多数评审认为当前版本在关键路线信息、grounding 和终点表达上明确优于上一版，
且没有新增系统性错误时，才把该改动作为默认流程。若发现退化样本，必须记录具体
episode、旧/新输出、对应 contact sheets 和错误类型，再继续迭代。
