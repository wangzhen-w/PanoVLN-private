# ScaleVLN to VLN-CE

最简单的入口：

```bash
bash /workspace/code_dir/VLN/scalevln2ce/run_scalevln_ce.sh
```

脚本顶部变量已经改成了直接赋值形式，直接编辑
[run_scalevln_ce.sh](/workspace/code_dir/VLN/scalevln2ce/run_scalevln_ce.sh)
里的参数块即可，不需要再传环境变量。

也支持：

```bash
bash /workspace/code_dir/VLN/scalevln2ce/run_scalevln_ce.sh smoke
bash /workspace/code_dir/VLN/scalevln2ce/run_scalevln_ce.sh build
bash /workspace/code_dir/VLN/scalevln2ce/run_scalevln_ce.sh gt
```

三种核心模式说明：

1. `smoke`
   - 用一个临时目录做小样本自检
   - 跑一个很小的 `build-subsets`
   - 跑一个很小的 `generate-gt`
   - 校验输出字段
   - 保留 smoke 输出目录，方便手动检查
   - 只清理 `.scene_batches` 和 `.tmp` 这类中间临时文件

2. `build`
   - 只构建 `subset_00 ... subset_09` 的 `scalevln_subset_150k.json.gz`
   - 这一步把离散 `ScaleVLN_total` 转成 VLN-CE episode 格式
   - 同一个 scene 的轨迹会按轮转方式分散到不同 subset
   - 目标是让每个 subset 都覆盖更多 scene，而不是让前几个 subset 吃掉大块 scene
   - 不生成 gt

3. `gt`
   - 只读取已有 subset episode 文件
   - 调 `ShortestPathFollower(goal_radius=0.3)` 生成 `scalevln_subset_150k_gt.json.gz`
   - 默认按 scene 分组并开启断点续跑

`full` 不是新的基础模式，它只是顺序执行：
`smoke -> build -> gt`

`generate_scalevln_ce.py` covers two stages:

1. `build-subsets`
   - stream `ScaleVLN_total/annotations/R2R_scalevln_ft_aug_enc.json`
   - skip all trajectories already present in `ScaleVLN_150k/scalevln_subset_150k.json.gz`
   - write `10 x 150k` VLN-CE episode subsets under `ScaleVLN_CE/subset_XX/`

2. `generate-gt`
   - group each subset by scene
   - create per-scene temporary datasets
   - run `habitat.tasks.nav.shortest_path_follower.ShortestPathFollower`
   - write resumable `jsonl`, then finalize to `scalevln_subset_150k_gt.json.gz`

## Build 10 subsets

```bash
python /workspace/code_dir/VLN/scalevln2ce/generate_scalevln_ce.py \
  build-subsets \
  --output-root /workspace/code_dir/a_property/dataset/general_VLN_data/ScaleVLN_CE \
  --num-subsets 10 \
  --subset-size 150000 \
  --overwrite \
  --log-every 100000
```

## Generate GT for one subset

```bash
python /workspace/code_dir/VLN/scalevln2ce/generate_scalevln_ce.py \
  generate-gt \
  --output-root /workspace/code_dir/a_property/dataset/general_VLN_data/ScaleVLN_CE \
  --subset-indices 0 \
  --minimal-observations \
  --gt-log-every 200 \
  --resume
```

## Generate GT for all subsets sequentially

```bash
for idx in $(seq 0 9); do
  python /workspace/code_dir/VLN/scalevln2ce/generate_scalevln_ce.py \
    generate-gt \
    --output-root /workspace/code_dir/a_property/dataset/general_VLN_data/ScaleVLN_CE \
    --subset-indices "${idx}" \
    --minimal-observations \
    --gt-log-every 200 \
    --resume
done
```
