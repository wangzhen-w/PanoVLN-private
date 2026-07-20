# ScaleVLN 转换与 instruction rewrite 消融

`scalevln/` 只负责两件事：

1. 将原始离散 ScaleVLN path 转为 VLN-CE episode；
2. 使用共享的 `data_create.instruction.pipeline` 重写 ScaleVLN instruction，
   与原 instruction 做严格同轨迹消融。

HM3D 自采数据不放在这里，见 `../data_create/README.md`。

## 目录

```text
scalevln/
├── README.md
├── generate_scalevln_ce.py       # ScaleVLN -> VLN-CE
├── run_scalevln_ce.sh             # 转换脚本；可选生成制作期 GT
├── run_rewrite.sh                 # instruction rewrite
└── audit_instruction_quality.py   # 可选文本/action 质量审计
```

运行产物、审计结果和图片不写入代码目录。

## 1. 转换为 VLN-CE

转换器使用 `ijson` 流式读取约 290 万条的原始 annotation，首次使用前安装：

```bash
pip install ijson
```

所有路径和开关都在 [run_scalevln_ce.sh](run_scalevln_ce.sh) 顶部直接赋值。
修改：

```bash
MODE="build"  # 正式数据默认只 build；gt / full 仅用于制作期回放
RAW_ANNOTATIONS="/path/to/ScaleVLN_total/annotations/R2R_scalevln_ft_aug_enc.json"
EXISTING_SUBSET="/path/to/ScaleVLN_150k/scalevln_subset_150k.json.gz"
CONNECTIVITY_DIR="/path/to/ScaleVLN_total/connectivity"
CONNECTIVITY_MP3D_DIR="/path/to/ScaleVLN_total/connectivity_mp3d"
SCENES_DIR="/path/to/scene_datasets"
CONFIG_PATH="${REPO_ROOT}/config/vln_scalevln.yaml"
OUTPUT_ROOT="/workspace/data1/dataset/general_VLN_data/ScaleVLN_CE"
NUM_SUBSETS=10
SUBSET_SIZE=150000
```

然后运行：

```bash
cd /workspace/code/VLN
bash scalevln/run_scalevln_ce.sh
```

- `build`：生成正式 VLN-CE subset，这是默认模式和需要保留的数据；
- `gt`：按 scene 生成制作期 ShortestPathFollower actions/locations，供立即采图或核验；
- `full`：依次执行 build、gt，只用于上述制作流程。

类似 R2R 的 `train_gt.json.gz` 不属于本项目的数据集发布契约。ScaleVLN 正式产物
只需要 `scalevln_subset_150k.json.gz`；需要 action sequence 时，再根据 episode 的
reference path 和统一的 `GOAL_RADIUS=0.3` 调用 Habitat 生成即可。
旧脚本曾写出的 `manifest.json` 和可选 gt/full 模式生成的 GT sidecar 都不是正式
训练输入；默认 build 不再生成 manifest，并会清理 OUTPUT_ROOT 下已有 subset 中
的这些非 dataset 文件。若主动选择 gt/full，制作期 GT 会保留供后续阶段使用；完成
后切回 build 运行一次即可恢复每个 subset 只含 dataset 的发布目录。

## 2. Rewrite instruction

[run_rewrite.sh](run_rewrite.sh) 顶部只需要配置三个数据路径：

```bash
SOURCE_JSONL="/workspace/data1/dataset/PanoVLN/sub_dataset/scalevln.jsonl"
IMAGE_ROOT="/workspace/data1/dataset/PanoVLN/images/scalevln"
REWRITE_OUTPUT="/workspace/data1/dataset/PanoVLN/sub_dataset/scalevln_qwen36_27b_panovln.jsonl"
```

然后运行：

```bash
cd /workspace/code/VLN
bash scalevln/run_rewrite.sh
```

这里明确使用 `--mode generate`。`SOURCE_JSONL` 文件物理上仍保留原 instruction，
但 pipeline 在加载后立即把 `instruction` 置空；传给 Qwen 的只有 episode ID、actions、
trajectory metadata 和 panorama-derived evidence sheets。因此它与 `data_create`
自采数据使用的是同一套 source-text-blind instruction 系统。正式脚本固定使用
`--instruction-profile dense`，每条轨迹只发布一条经过候选筛选和审核的 instruction；
不再额外生成 Concise 版本。

正式 rewrite 默认使用 `ROUTE_EVIDENCE_MODE="auto"`：短路线仍用 START / ROUTE /
ENDPOINT 三张 overview evidence sheet，同时额外生成 forward-first FINAL-only
endpoint evidence。endpoint agent 会读取最后几个平移位置，并把 FINAL observation 拆成带文本标签的 FORWARD /
FORWARD-DOWN / LEFT / RIGHT / BACK 单视角图，先抽取 stop 位置类型、当前站位 anchor、
正前方 anchor 是否已到达、侧向/身后 anchor 和必须避免的终点/朝向说法，再做一次 endpoint fact audit 检查
forward/side-view 是否串列、前方景物是在 FINAL 相机处还是仍位于前方；endpoint agent 只输出语义事实，
不生成需要 writer 复制的固定 stop phrase。audit 改写的 facts
必须再验证后才能进入 writer，持续不一致时丢弃派生 facts 并退回原始视觉证据。80 action 及以上的长路线，以及低于阈值但平移多、
包含多个大转向的复杂路线，会把 route waypoints 拆成多个局部分段，先生成 grounded route facts，再结合 endpoint facts
交给 writer 自由组织成一条自然 instruction。writer 不规定固定开头、句数、句法、动词或
`stop/wait` 字面词，只要求路线可执行、视觉事实正确和终点清楚。每个分段同时传入真实 action span 摘要，标出大转向、
小平移和真实 forward 运动，降低把视角旋转误写成空间移动的概率。ScaleVLN JSONL
中已有的 `trajectory_metadata.reference_path` 会自动派生 `vertical_motion`，
用于约束上楼/下楼方向；这仍然不使用原 instruction。

ScaleVLN 保存的全景已经以 agent 当前朝向为中心，正式脚本不传
`--use-action-heading`。不要在默认 rewrite 中根据 actions 再做 heading re-rotation，
否则 FINAL FORWARD 会被转到侧面视图，终点 anchor 会系统性漂移。
脚本中 `NUM_WORKERS` 控制 Qwen/agent 并发，`EVIDENCE_WORKERS` 控制独立 CPU
视觉预处理进程。同一 episode 内重叠的 route、segment、endpoint 和 FINAL 视图会复用
完全相同的解码/投影结果，不改变 VLM 看到的像素、prompt 或 agent 顺序。

正式脚本还会为每条路线生成两个候选 instruction，再由独立 visual candidate judge
直接读取原始视觉、actions 和几何约束，选择更忠实、可执行、终点更准的一版。judge
不会接收 writer 使用的 endpoint facts 或 route plan，避免共享错误事实导致集体误判。
judge 与最终 audit 都要显式比较原始证据空间序列和 instruction 表达的空间序列，并通过
五张独立 FINAL 透视图判断正前方 anchor 是否已在相机处到达。路线和 endpoint 都匹配、
且独立阶段不存在明确的 reached/ahead 冲突时才允许发布。
这个机制用于降低单次生成偶然回归，避免依赖针对少数 canary episode 的物体级硬规则。

ScaleVLN 旧数据使用不带 split 的 `hm3d/<scene_dir>/<scene_name>.basis.glb`
或 `mp3d/...` scene ID；`PanoVLN-HM3D` 新数据则使用
`hm3d/train/<scene_dir>/<scene_name>.basis.glb`。前者由统一 `SCENES_DIR` 解析，
后者在制作阶段由 HM3D 专用 `SCENE_ROOT` 解析，两个根目录不要混用。

注意：仓库外当前同名的 104,141 条 rewrite 文件是历史生成结果经过 158 条对齐
过滤后的数据；修改脚本本身不会追溯性地改变它。只有重新完整运行
`run_rewrite.sh` 后，才能把该路径上的结果当作新的 source-text-blind ablation。
小样本验证中 `mode=generate`、所有 `candidate.old_instruction` 均为空，说明信息隔离
实际生效，而不只是 prompt 里要求模型忽略原文。

处理顺序：

```text
ScaleVLN trajectory/actions + panoramas
  -> shared source-text-blind evidence / FINAL endpoint facts / segmented-facts generation
  -> endpoint-assisted multi-candidate writing / raw-evidence visual judge / blind audit
  -> 已穷尽修复与复审仍未通过的 episode 终止丢弃
  -> 原子发布 REWRITE_OUTPUT
```

`REWRITE_OUTPUT` 是唯一正式 rewrite，不生成 candidate、original-baseline 副本或
pair manifest。脚本使用 `--drop-failed true --allow-incomplete false`：质量门已穷尽的
episode 记为终止失败，续跑时不再调用模型，也不进入正式 rewrite；API 超时、
服务不可用或文件读取异常等运行时失败仍保留 progress 供续跑，并在完成前拒绝发布。

共享系统包含三类针对 ScaleVLN 脏轨迹/不稳定审核的保护：终点站位语义必须从
ENDPOINT evidence 保留到最终 instruction，但可自由改写；楼梯上/下方向优先使用 trajectory metadata；
actions 会生成 major-turn、dead-reckoning 复杂度和分段 action-span 提示，避免长路线
被压缩成直线路线，也避免把原地转向误写成穿过房间。这些门不会利用原 instruction。

每次修改 rewrite 的 prompt、视觉证据或 QA 逻辑后，必须在同一批 episode 上保留
previous/current 对比，而不是只看当前版本。最小人工验收需要覆盖短路线、长路线、
低于阈值但含多个转向的复杂路线、楼梯/landing、跨房间/走廊转换和复杂开放空间；每条都同时展示
原始 ScaleVLN、历史 `/workspace/data1/dataset/PanoVLN/sub_dataset/PanoVLN.jsonl`
rewrite、上一版系统输出和当前输出。评审至少包含路线执行性、视觉 grounding、
长路线/楼梯/空间转换、语言自然度、回归与数据集质量 5 个独立视角。只有多数评审确认当前
版本在关键路线信息、grounding 和终点表达上明确提升，且短路线未退化，才将改动
作为默认 rewrite 流程。

## 158 条失败样本

158 条历史 rewrite 失败样本已经同时从 source 和 rewrite 实验输入中原子删除：

```text
scalevln.jsonl:                         104,141
scalevln_qwen36_27b_panovln.jsonl:      104,141
ID/order/action mismatch:                     0
```

因此 `scalevln.jsonl` 本身就是 original-instruction baseline，不再需要额外 aligned
文件。两份 JSONL 的 episode ID、顺序和 actions 完全一致，只差 instruction。

最终训练前，需要分别从这两份 annotation、使用相同 seed 和相同
`prepare_training_data.py` 配置重新生成训练 JSONL。不要复用包含全部
104,299 episode 的旧派生训练文件，否则两组样本不一致。

## 可选审计

```bash
python scalevln/audit_instruction_quality.py \
  --candidate /workspace/data1/dataset/PanoVLN/sub_dataset/scalevln_qwen36_27b_panovln.jsonl \
  --source /workspace/data1/dataset/PanoVLN/sub_dataset/scalevln.jsonl
```

审计结果默认写入：

```text
/workspace/data1/dataset/PanoVLN/audits/<candidate_name>/
```
