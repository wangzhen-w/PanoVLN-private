# ScaleVLN Instruction Rewrite Pipeline

本文档只说明当前 ScaleVLN instruction rewrite pipeline，以及
`run_rewrite_instructions_full.sh` 中每个参数的作用。

pipeline 的目标是把 ScaleVLN 原始弱 instruction 改写成更接近 R2R 人工标注风格的
英文导航指令。最终 clean JSONL 保持训练 schema 不变：

```json
{"episode_id": 0, "instruction": "...", "actions": [3, 3, 1, 0]}
```

也就是只替换 `instruction`，`episode_id` 和 `actions` 原样保留。多次 retry 仍失败的
episode 不会用原始 instruction 回填，而是从最终 clean output 中丢弃。

## Current Pipeline

每条 episode 的处理是一个多 agent 协同流程。所有 agent 使用脚本中配置的同一个
OpenAI-compatible vLLM 模型服务。整体顺序是：

```text
input JSONL row
  -> RouteEvidenceAgent
  -> StartFactAgent
  -> EndpointFactAgent
  -> RoutePlannerAgent
  -> retry loop:
       InstructionWriterAgent
       SelfCheckCriticAgent
       RouteCoherenceAuditorAgent
       SpatialBoundaryAuditorAgent
       FinalQualityGateAgent
  -> clean JSONL row or dropped failed episode
```

### 1. RouteEvidenceAgent

这个阶段把全景轨迹转换成模型容易理解的透视图证据。训练数据本身是全景图，但生成
instruction 时，模型更擅长看普通透视图，所以 pipeline 会从全景图中裁出多个方向视角，
再拼成 contact sheet。

每条 episode 会生成三类 evidence：

- `start sheet`：起点附近的若干帧，每帧包含 `left / forward / right / back`。
- `route sheet`：整条路线的关键帧，每帧包含 `left / forward / right`。
- `endpoint sheet`：终点附近的若干帧，每帧包含 `left / forward / right / forward-down`。

`start sheet` 主要用于判断：

- agent 一开始是否在房间内。
- 第一段是否必须写 `exit the room`、`leave the room`、`walk out` 之类的起点边界。
- 第一处明显转向是 left、right、turn around，还是直接 forward。
- 原始 instruction 的第一步方向是否被旧标注误导。

`route sheet` 主要用于判断：

- 中间经过了哪些空间，例如 hallway、living room、dining area、kitchen、bedroom。
- 重要转向和空间切换是否需要写进 instruction。
- 哪些 landmark 可靠，例如 sofa、staircase、table、window、bookshelf。
- 哪些说法不稳定，例如强行把开放空间写成 hallway。

`endpoint sheet` 主要用于判断：

- 最终停点到底在什么区域。
- 终点附近哪些 landmark 可靠。
- 是否应该避免 doorway、inside、landing、top/bottom of stairs、second step 等边界词。
- 原始 instruction 的终点描述是否被最终视觉证据支持。

### 2. StartFactAgent

StartFactAgent 只看起点证据，并结合 action-derived first-turn constraint，输出结构化起点事实。

它负责回答：

- 起点可见区域是什么，例如 bedroom、bathroom、kitchen、open room，或者只说 room。
- 是否存在明确的离开房间/穿过门口动作。
- 是否必须保留起点边界，例如 `must_preserve_start_boundary=true`。
- 推荐第一句应该怎么写，例如 `Turn left and walk out of the room.`。
- 哪些起点说法不该写，例如旧 instruction 说 left，但 action 和图像支持 right。

这个 agent 的输出会约束后续 writer。若它认为必须保留起点边界，最终 instruction
不能直接跳过第一段，从中间空间开始写。

### 3. EndpointFactAgent

EndpointFactAgent 只看终点证据，输出保守的终点事实。

它负责回答：

- 最终停点所在区域，例如 living room、bathroom、kitchen、stair area、hallway end。
- 终点附近确定可见的 landmark。
- 原始 destination hint 是否被最终画面支持。
- 推荐的安全 stop phrase，例如 `stop near the stairs`、`stop in front of the sink`。
- 哪些终点说法需要避免，例如 `doorway`、`landing`、`second step`、`counter`、`kitchen`。

终点 agent 的优先级很高。后续 route planner 和 writer 可以使用 route sheet 补充路径，
但最终 stop phrase 不能比 endpoint facts 更激进。例如 endpoint facts 只支持
`near the stairs`，writer 不应扩写成 `at the top of the stairs`。

### 4. RoutePlannerAgent

RoutePlannerAgent 读取 start facts、endpoint facts、route sheet 和 action summary，
生成一份人类级 route plan。

route plan 不是最终 instruction，而是给 writer 的中间结构。它通常包含：

- `route_overview`：整条路线的简短概括。
- `major_segments`：3 到 6 段主要路线段。
- `safe_landmarks`：可放心使用的 landmark。
- `avoid_claims`：后续 writer 应避免的旧标注错误或视觉不确定说法。
- `start.safe_start_phrase`：安全起始短语。
- `destination.safe_stop_phrase`：安全终点短语。
- `writing_guidance`：给 writer 的简短约束。

每个 major segment 会描述：

- 这段大概对应的 action/waypoint 范围。
- 这一段的自然语言移动描述。
- 可靠 landmark。
- 不确定说法。
- confidence。

RoutePlannerAgent 的作用是防止 writer 只看起点和终点，漏掉中间路线。长轨迹尤其依赖
这个阶段，因为长轨迹如果直接让 writer 看 contact sheet 写整句，容易跳过中间房间、
走廊、转向或关键 landmark。

### 5. InstructionWriterAgent

InstructionWriterAgent 根据 start facts、endpoint facts、route plan、route sheet 和
endpoint sheet 写出最终候选 instruction。

writer 的目标不是逐 token 翻译 action，而是写成 R2R 风格的人类导航指令：

- 一条自然英文 instruction。
- 覆盖主要路线段，但不机械列出每个 left/right/forward。
- 起点必须符合 StartFactAgent 的约束。
- 终点必须符合 EndpointFactAgent 的推荐 stop phrase 或保守等价说法。
- 中间路线必须按照 RoutePlannerAgent 的 major segments 顺序表达。
- left/right 优先用于真实路线选择和转向，不用于不稳定的家具相对位置。
- 不提 image、frame、row、panorama、contact sheet、action、dataset 等生成痕迹。

writer 的 `temperature` 当前设为 `0.50`。这是为了让语言更自然，不把所有样本写成同一种
模板。质量控制由后续 critic、auditor、final gate 负责。

### 6. SelfCheckCriticAgent

SelfCheckCriticAgent 模拟人工数据审查，检查候选 instruction 是否和视觉证据及 route plan
匹配。

它会评估：

- grounding：landmark 和空间描述是否有视觉依据。
- navigation：转向和空间顺序是否能导航。
- endpoint：最终停点是否正确。
- R2R style：是否像人工指令，而不是 action list。
- hallucination risk：是否有明显幻觉。

如果 self-check 判定 `pass`，候选继续进入下一阶段。若判定 `borderline` 或 `fail`，
它必须给出 `corrected_instruction`。pipeline 会用修正后的 instruction 再审一轮。

对于只涉及 minor style 的 borderline，例如轻微重复、措辞不够自然，但 grounding、
navigation、endpoint 都可靠，pipeline 可以接受或轻修后接受。对于路线、方向、终点错误，
则会继续 retry 或最终丢弃。

### 7. RouteCoherenceAuditorAgent

RouteCoherenceAuditorAgent 专门检查整条路线的连贯性，重点看中间过程，而不是只看起点和终点。

它会检查：

- 是否覆盖 route plan 的每个 major segment。
- 是否漏掉第一段 exit/leave/walk-out 起点边界。
- 是否跳过重要中间空间，例如 hallway、dining area、bathroom、bedroom。
- 是否把多个明确转向压缩成 `make several turns` 这类不可执行描述。
- 是否出现前后矛盾，例如先说进入某房间，后面又说从那个房间外进入。
- 是否过度使用 `walk forward`、数字角度、机械 action 风格。
- 是否使用不稳定的 object-side 说法，例如 `the sofa on your left`。

如果 auditor 修正 instruction，pipeline 会再审一次。若只剩 minor style borderline，
可以接受；若仍然存在 wrong endpoint、wrong room、wrong turn、missing segment 等严重问题，
则触发 retry。

### 8. SpatialBoundaryAuditorAgent

SpatialBoundaryAuditorAgent 专门检查空间边界词。这个 agent 的范围比 route auditor 更窄，
只关注容易出错的边界表达。

它重点检查：

- `doorway` vs `near the doorway`。
- `inside the room` vs `near the entrance`。
- `hallway` vs open living/dining/kitchen area。
- `through the kitchen` vs `past the kitchen area`。
- `top of the stairs`、`bottom of the stairs`、`landing`、`second step`、`on the stairs`。
- foyer/open area/hallway 这类房间标签是否过拟合。

如果边界不清楚，pipeline 偏向保守表达，例如：

- `near the stairs`
- `by the stairs`
- `near the doorway`
- `near the entrance`
- `in the area with ...`

若 auditor 指出的是旧 instruction 里的词，而当前 generated instruction 已经修掉，
pipeline 会把它识别为 stale audit 并接受当前结果。

### 9. Text Cleanup

writer 和 reviewer 的输出会经过轻量文本清洗。这个阶段不重新理解路线，只修明显的语言问题：

- `turn right about 30 degrees` 规范成 `turn right`。
- `make a 90 degree left turn` 规范成 `turn left`。
- 悬空的 `keeping the ...` 改成 `passing the ...`。
- 删除非导航物体的 `on your left/right` 或 `on the left/right`。
- 保留真实路线选择，例如 `doorway on the right`。

这个步骤的目的是减少简单文本瑕疵进入 critic/auditor，提升效率，同时不改变路线语义。

### 10. FinalQualityGateAgent

FinalQualityGateAgent 做确定性 blocker 检查。它不会因为小的风格问题丢样本，只拦截明确危险的问题。

主要 blocker 包括：

- 第一句方向和 action-derived first turn 明显相反。
- action 不支持 `turn around`，但 instruction 以 `turn around` 开头。
- StartFactAgent 要求保留起点边界，但最终 instruction 完全跳过 exit/leave/walk-out。
- EndpointFactAgent 明确标为不支持的终点危险词仍出现在最终 stop phrase 中。
- EndpointFactAgent 标为不确定的精确 stair/doorway/landing 等边界词仍被强行写入终点。

如果 FinalQualityGateAgent 发现 blocker，本轮 writer 结果会失败，并把错误作为 repair note
传回下一次 writer retry。达到 `RETRIES` 上限后仍失败，则该 episode 进入 failed metadata，
并从 clean output 中丢弃。

### 11. Retry, Resume, And Output

pipeline 有两类 retry：

- stage retry：用于 StartFactAgent、EndpointFactAgent、RoutePlannerAgent 这类结构化阶段。
- writer retry：用于 writer + critic/auditor + final gate 的整体循环。

所有结构化 JSON 输出都会尝试 JSON repair。repair 的 token 上限会根据原始响应长度和
planner/review token 设置自适应，减少因为模型 JSON 格式偶发错误导致的非质量失败。

断点文件自动放在 `OUTPUT_JSONL` 同目录下，目录名为 `<output_jsonl_stem>_progress`。运行中每个
worker/rank 会增量写：

- `candidates_rank*.jsonl`
- `failed_rank*.jsonl`

正常结束时会合并成：

- `candidates.jsonl`
- `failed.jsonl`
- `summary.json`

如果运行中断，下次 `--resume true` 会读取已有 candidate，只处理还没有成功的 episode。

## Script Parameters

`run_rewrite_instructions_full.sh` 中的参数都在脚本内直接赋值。

### Path And Runtime

- `SCRIPT_DIR`：当前脚本所在目录。脚本会 `cd` 到这里，保证相对路径稳定。
- `PYTHON_BIN`：运行 pipeline 的 Python 命令，当前是 `python`。
- `PY_SCRIPT`：主程序路径，当前是 `${SCRIPT_DIR}/rewrite_scalevln_instructions.py`。
- `INPUT_JSONL`：原始 ScaleVLN 子集 JSONL 路径。
- `IMAGE_ROOT`：episode 全景图根目录。每个 episode 的图像目录由主程序按 episode id 查找。
- `OUTPUT_JSONL`：最终 clean JSONL 输出路径。主程序会自动推导断点目录为
  `${OUTPUT_JSONL%.jsonl}_progress`。

### Model Service

- `BASE_URL`：OpenAI-compatible vLLM 服务地址，当前是 `http://127.0.0.1:10426/v1`。
- `MODEL`：vLLM 中注册的模型名，当前是 `Qwen3.5-27B`。
- `API_KEY`：OpenAI client 需要的 API key。当前本地 vLLM 使用 `test`。

### Dataset Selection

- `MAX_EPISODES`：最多处理多少条 episode。`0` 表示处理全量。
- `SAMPLE_MODE`：抽样方式。full run 当前用 `first`。主程序还支持 `random` 和 `stratified`。
- `SEED`：当 `SAMPLE_MODE=random` 或 `stratified` 时使用的随机种子。

### Parallelism

- `NUM_WORKERS`：并发 worker 数。每个 worker 独立处理 episode，并写自己的 rank shard。
  当前设置为 `24`。

### Visual Evidence Rendering

- `MAX_WAYPOINTS`：route sheet 最多抽取多少个中间关键帧。越大越不容易漏中间过程，
  但图像输入更重，速度更慢。
- `START_WINDOW_FRAMES`：start sheet 使用多少个起始帧。
- `ENDPOINT_WINDOW_FRAMES`：endpoint sheet 使用多少个终点帧。
- `TILE_WIDTH`：每个透视图 tile 的宽度。
- `TILE_HEIGHT`：每个透视图 tile 的高度。
- `JPEG_QUALITY`：contact sheet 保存和传给模型时的 JPEG 质量。

### Generation And Review

- `TEMPERATURE`：writer 生成温度。当前是 `0.50`，用于提升语言自然度。
- `MAX_TOKENS`：writer 输出 token 上限。
- `FACT_MAX_TOKENS`：StartFactAgent 和 EndpointFactAgent 输出 token 上限。
- `PLANNER_MAX_TOKENS`：RoutePlannerAgent 输出 token 上限。
- `REVIEW_MAX_TOKENS`：SelfCheckCriticAgent、RouteCoherenceAuditorAgent、
  SpatialBoundaryAuditorAgent 输出 token 上限。
- `REQUEST_TIMEOUT`：单次模型请求超时时间，单位秒。

### Retry And Agent Switches

- `RETRIES`：writer + critic/auditor/final gate 整体循环的 retry 次数。
- `STAGE_RETRIES`：start facts、endpoint facts、route plan 等阶段的 retry 次数。
- `SELF_CHECK`：是否启用 SelfCheckCriticAgent。
- `ROUTE_AUDIT`：是否启用 RouteCoherenceAuditorAgent。
- `SPATIAL_AUDIT`：是否启用 SpatialBoundaryAuditorAgent。

### Fixed CLI Switches In The Script

脚本还固定传入以下开关：

- `--provider qwen`：只使用 qwen provider。
- `--start-fact-pass true`：启用 StartFactAgent。
- `--endpoint-fact-pass true`：启用 EndpointFactAgent。
- `--route-plan-pass true`：启用 RoutePlannerAgent。
- `--require-start-facts true`：缺少 start facts 时不继续生成。
- `--require-endpoint-facts true`：缺少 endpoint facts 时不继续生成。
- `--require-route-plan true`：缺少 route plan 时不继续生成。
- `--drop-failed true`：失败 episode 不进入最终 clean output。
- `--resume true`：启用断点续跑。
