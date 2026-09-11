# PanoVLN instruction 制作

接收 `dataset_create/trajectory` 的真实轨迹，按需渲染、分段生成导航语言、局部验证，导出 clean ERP 和标准 Habitat R2R 数据。目标是让 navigator 看懂路径，尤其能在决策区域选择正确入口，并在合理的目的地区域停下。轻微重复、无关的外观差异和细小转角不影响入库。

## 运行与配置

依赖 Habitat-Sim 0.3.3、NumPy、SciPy、NetworkX、Pillow、requests、imageio 和 imageio-ffmpeg。本机的 `/opt/conda/envs/vln/bin/python` 已具备这些依赖；Qwen 通过单独部署的 vLLM 服务调用。

打开 [create_instructions.sh](../create_instructions.sh)，直接修改顶部赋值：

| 参数 | 用途 |
| --- | --- |
| `PYTHON_BIN` | 含 Habitat-Sim 的 Python |
| `TRAJECTORIES` | 输入轨迹 `.json.gz` 或 `.json` |
| `SCENE_ROOT` | HM3D 场景根目录 |
| `OUTPUT_ROOT`、`NAME` | R2R 输出目录和文件名 |
| `ERP_ROOT` | 训练图片目录，默认 `${OUTPUT_ROOT}/image` |
| `WORK_DIR` | 中断续跑检查点，默认 `${OUTPUT_ROOT}/.work/instruction`；完成后自动清理 |
| `NUM_PROCESSES` | 工作进程数，默认 48；同时也是客户端 API 请求并发上限 |
| `GPU_DEVICE_IDS=(0 1 2 3 4 5 6 7)` | Habitat 使用的 GPU 列表，默认每张卡六个进程 |
| `CPU_THREADS_PER_PROCESS` | 每个进程的数值库线程数，默认 1，控制 OpenMP/BLAS 线程开销 |
| `LIMIT` | 处理条数上限；默认 `0` 处理所有筛选后的轨迹，`24` 表示最多处理 24 条 |
| `SELECTION` | `first` 按输入顺序；`diverse` 优先覆盖不同场景及较多决策 |
| `ERP_WIDTH`、`ERP_HEIGHT` | ERP 分辨率，默认 1600×800，比例必须为 2:1 |
| `JPEG_QUALITY` | 训练图片 JPEG 质量，默认 95 |
| `BASE_URL`、`MODEL_NAME`、`API_KEY` | 本地模型接口、模型名、密钥 |
| `MEDIA_MODE` | 默认 `video`；`frames` 使用逐帧兼容输入 |
| `KEEP_WORK` | 默认 `false`；调试时设为 `true` 保留中间材料 |

脚本默认使用 GPU 0–7、48 个进程处理统一训练轨迹，已设置 `LIMIT=0`、`NAME="train"`。不再按源场景目录划分数据集。当前输入为：

```text
/workspace/data2/dataset/general_VLN_data/PanoVLN/trajectory/trajectories.json.gz
```

```bash
cd /workspace/code/VLN

# 检查输入和配置，不渲染、不调用模型。
./dataset_create/create_instructions.sh inspect

# 按脚本顶部配置制作数据。
./dataset_create/create_instructions.sh generate

# 检查已导出的 R2R 格式与场景路径。
./dataset_create/create_instructions.sh validate

```

也可追加 CLI 选项，例如 `--trajectory-ids trajectory_...` 或 `--scene-ids 00800-TEEsavR23oF`。CLI 参数覆盖脚本赋值；通过 CLI 改输出位置时应同时指定 `--erp-root` 和 `--work-dir`。日常运行直接修改脚本顶部即可，不需要环境变量覆盖。

每个工作进程持有一个 Habitat 模拟器，按场景复用，依次完成渲染、生成、验证和 ERP 导出。同一进程每次最多发出一个 API 请求，因此 48 个进程的客户端请求并发上限是 48；正在渲染或导出的进程不占 API 请求并发，实际并发通常较低。各进程等待 API 时仍保留场景，每卡六个进程需要容纳六个模拟器的显存。

制作时由主进程显示一条总 `tqdm` 进度条，按已处理轨迹数更新，包含总数、百分比、耗时、速度、预计剩余时间，以及 `accepted`（通过）、`quarantined`（隔离）、`error`（待续跑）和 `resumed`（复用检查点）数量。复用的轨迹也计入总进度；进度达到 100% 表示本轮选中的轨迹均已处理，不代表全部入库，最终仍需汇总和导出。工作进程不逐条刷屏，程序错误通过进度条上方的日志显示，详细结果保存在检查点及最终汇总中。`render` 模式同样显示总进度，并增加 `prepared` 数量。

场景初始化和关闭时的 Habitat 原生日志临时捕获，操作失败会输出捕获的诊断并保留异常。当前只使用 RGB、深度、场景几何和 NavMesh，不需要语义传感器或物体类别标注；可选 semantic 描述缺失不影响这些输入。

Instruction 阶段直接使用 Habitat-Sim，没有运行 Habitat-Lab 的任务 measures，也不重复重建场景的导航区域分区。每段审核共享一次 clean 视频观察和一次完整文字比较，同段的多个决策、普通运动与停止不再分别调用模型看视频。文字修复复用同一份观察；发现首个失败片段后先修复，再检查后续片段。入库仍要求全部决策、运动和停止检查通过；实际吞吐还取决于片段数量、修复次数和模型服务负载。

`GPU_DEVICE_IDS` 仅指定 Habitat 的 GPU，不改变已有 vLLM 服务的 GPU 分配。Qwen 默认地址为 `http://127.0.0.1:10420/v1`，模型为 `Qwen3.8-27B`；模型服务的实际处理并发由其部署配置和吞吐决定。密钥不写入请求记录或运行清单。

## 最终产物

```text
/workspace/data2/dataset/general_VLN_data/PanoVLN/
├── train.json
├── train.json.gz
└── image/
    └── trajectory_.../
        ├── frame_0.jpg
        ├── frame_1.jpg
        └── ...
```

两个 R2R 文件解压后内容相同，只包含自动验证通过的轨迹。训练图片使用 Habitat 原生 equirectangular 相机，处于真实 agent pose，保持水平，采用轨迹元数据中的传感器高度。训练相机与制作 instruction 的俯视透视相机分开。

**`frame_i.jpg` 对应执行 `action_ids[i]` 前的观察帧。** 每个动作都保留图片，包括原地旋转及最终 STOP；不保存 STOP 执行后的重复帧。因此每条轨迹的图片数等于源轨迹的动作数。图片目录使用原始 `trajectory_id`，可与源轨迹动作直接关联。ERP 没有路线、候选标签或其他叠加。

图片只在轨迹通过验证后导出，全部完成后才发布该轨迹的目录。强制中断时留下的半成品目录会在该轨迹续跑时重建，不会当成完整训练图片。失败轨迹不生成正式训练图片。`ERP_ROOT` 可以单独指定，以匹配训练任务的图片目录配置；改变相机分辨率时使用新的图片目录。

## 临时材料与恢复

默认边处理边渲染，不预先提取全量图片。每条轨迹进入 `accepted` 或 `quarantined` 后删除视频、compass、透视图片、depth、路线数组和模型请求记录；紧凑的过程记录暂留到汇总。整批处理完成、R2R 导出成功后删除整个 `WORK_DIR`，最终目录不保留制作 instruction 的媒体或 sidecar。

进程中断、API/程序/IO 错误会保留工作目录。用相同配置重跑 `generate` 可恢复未完成阶段，已完成样本不重复推理。已保存的局部文本和响应会复用；中断时尚未落盘的当前调用或写入可能重做。制作未完成时不要手动删除检查点。运行清单记录代码、配置、源数据路径和轨迹元数据指纹；每条记录额外绑定轨迹内容与场景文件信息。输出有进程锁，请为同时进行的不同任务设置独立的 `NAME` 和 `WORK_DIR`。

完成后再次运行会检查已有 JSON/gzip 和 ERP，返回 `already_exported`。**完成的数据集不会因扩大 `LIMIT` 自动追加；新的选择或配置使用新的 `NAME` 和 `WORK_DIR`。** 如果调试期间仍保留工作目录，则可在配置不变的情况下扩大选择并复用已有记录。

调试时设置 `KEEP_WORK=true`，工作目录内保留：

```text
WORK_DIR/
├── manifest.json
├── summary.json
├── last_run.json
└── episodes/<hash>/
    ├── record.json                 # pose、segment、clause 来源、验证与修复历史
    ├── ground_route.npz
    ├── requests/<hash>.json
    └── media/s000/attempt_0/       # clean/route 视频与帧、compass、depth
```

`render` 只准备视觉材料，随后相同配置的 `generate` 可以复用。`export` 从保留的工作目录重新汇总，不调用模型；关闭 `KEEP_WORK` 后成功导出会清理工作目录。存在未完成样本时默认不导出；`export --allow-incomplete` 可显式导出当前已通过部分，工作目录仍保留。两份 R2R 的发布或最后清理若中断，也会在恢复时补齐；完成标记最后删除，避免从头生成。`--retry-quarantined` 可在保留的工作目录中开启新一轮局部修复。

## 五阶段

1. **自然分段**：围绕决策接近、进入入口后的过程、明显转向和终点接近划分。普通路段不按固定帧数或每个小转角硬切。相邻段可共享接近上下文；每个动作和决策各有唯一归属，最后一段包含真实到达和停止。
2. **制作视觉材料**：按真实动作和 pose 回放，普通段使用局部视频，决策段增加八方向 compass，终点展示实际接近过程。生成端显示贴地 GT 路线，并保留 clean 版本用于验证。
3. **局部语言生成**：Qwen 描述 navigator 应该怎么走。视觉细节服务于识别入口、路径或停止位置；选择依据在作出选择时可用。视频帧之间的真实朝向变化明确标为 LEFT/RIGHT，辅助定位转向方向；是否进入新入口仍以可见路径为准，不能把小幅朝向调整写成新选择。可参考上一段的称呼。输出保留局部文本与 segment 的关系。
4. **轻量整理**：改善衔接、合并重复，不增加路线事实。关键选择和停止 clause 的原文、顺序及来源由程序检查；最终 instruction 不依赖路线标记。
5. **局部验证与修复**：先在不提供 instruction 的情况下观察完整 clean 片段，再一次性核对该段完整文字中的入口选择、运动顺序与停止要求。各项结论分别保存，但共享视觉观察和文字比较请求。失败定位到相关 segment，优先修改局部语言；作者认为材料或切分不足时返回上游。默认每段最多两次文字修复、一次上游修复，仍无法明确表达或验证的轨迹隔离。

提示词集中在 [prompts.py](prompts.py)，参数集中在 [config/default.json](config/default.json)。任务定义面向通用导航，不包含按轨迹、场景、房间或物体名称分派的规则。

### 路线和 compass

程序先核对真实回放的决策 pose、最终 pose、碰撞和目标到达。RGB 与未归一化 depth 来自同一透视相机，默认 448×448、90° 水平视场、向下俯视 30°；投影使用相机实际世界外参。

GT polyline 通过向下 ray cast 落到真实地面。再用 depth 重建可见表面世界坐标，只对临近路线的朝上表面着色。线宽和箭头按世界空间尺寸定义，因此透视自然缩放；被遮挡的地面没有可见像素，不会穿墙画线。缺少几何支撑的路线点不绘制。默认线宽 6.5 cm、箭头间隔 75 cm。

Compass 的八张图来自同一位置和时刻，以 agent 实际朝向为正前方，每隔 45° 取一张透视图：

| 左前 | 前 | 右前 |
| --- | --- | --- |
| 左 | 当前朝向 ↑ | 右 |
| 左后 | 后 | 右后 |

中心不是第九个视角。决策 compass 在标注的选择时刻及分支 anchor 之前取景，必要时沿真实轨迹回退，展示入口及其周围环境。图中不再把抽象 NavMesh connection 中心标成 A/B/C 建筑入口；这些中心与真实门口可能并不对应。生成端通过贴地路线确认经过哪里，clean 版本保留相同场景信息。

视频按位移和累计转角选帧，保留片段边界、决策和真实 STOP。生成和审核均使用完整核心区间；较早的接近上下文通过选择前 compass 提供，不再另存每个决策的重复审核视频。clean 视频有意展示实际通行结果，用于路线一致性审核。选定帧通过 vLLM OpenCV 解码器全部传入，关闭模型处理器的再次选帧。

### 验证标准

- **决策**：观察任务只接收完整 clean 片段视频、选择前 compass 和真实朝向/高度变化，不接收 instruction、GT 分支答案或路线标记。它记录实际通行顺序、选择前可用的线索及其他入口。随后文字审核对每个决策分别判断 `path_matches` 与 `choice_is_clear`，两项均为真、状态通过且置信度足够才入库。含糊描述即使与实际视频相容，也不能直接通过。观察不随局部文案变化，可由请求缓存复用。
- **终点**：使用 clean 接近视频与真实 stop compass，检查文字是否指向同一局部目的地，是否会明显停早或走过头。不要求区分同一目的地区域内的细小站位差别。
- **普通运动**：核对完整局部句子的运动顺序，包括关键选择之后的“then”“again”，避免截取句子后把同一次转弯重复用作依据。真实朝向变化只说明方向，不证明有新入口。明显左右写反、上下楼错误或进入错误空间会失败；轻微调整、沿走廊/楼梯自然弯行、粗略地标名称和无害重复可接受，不要求逐帧复述动作。

该方法审核可导航性及路径一致性，不再把单个字母选择题作为入库条件，也不是一次独立的盲导航成功实验。模型观察和比较仍可能出错；自动通过不等于独立的人类正确率。抽样、对照实验与视觉复核结果见 [VALIDATION.md](VALIDATION.md)。

## R2R 格式

导出字段与 `ScaleVLN_150k/scalevln_subset_150k.json.gz` 一致，根字段为 `episodes` 和 `instruction_vocab`，每个 episode 为：

```json
{
  "episode_id": 0,
  "trajectory_id": "trajectory_...",
  "scene_id": "train/00000-kfPV7w3FaU5/kfPV7w3FaU5.basis.glb",
  "start_position": [0.0, 0.0, 0.0],
  "start_rotation": [0.0, 0.0, 0.0, 1.0],
  "info": {"geodesic_distance": 10.0},
  "goals": [{"position": [1.0, 0.0, 2.0], "radius": 0.25}],
  "instruction": {"instruction_text": "...", "instruction_tokens": null},
  "reference_path": [[0.0, 0.0, 0.0], [1.0, 0.0, 2.0]]
}
```

`episode_id` 保留该轨迹在源数据中的索引；`start_rotation` 为 xyzw。`reference_path` 是实际回放的移动点，去除纯旋转和 STOP 的重复位置。目标与 reference path 末点均为真实 stop position；最短路距离也计算到真实停止点。

场景路径相对 `SCENE_ROOT`，本机为 `train/...` 或 `val/...`，仅用于定位 HM3D 文件；这些场景全部进入同一份训练输入。Habitat 的 `data_path` 指向导出的 `.json.gz`，`scenes_dir` 保持 `/workspace/data2/dataset/general_VLN_data/HM3D`。原始轨迹文件继续作为动作与图片对齐依据，不向标准 R2R 添加制作过程字段。
