# Legacy PhysicIQ67 PCK50 项目交接文档

最后核对时间：2026-08-08 07:11 UTC

## 1. 项目目标

本项目围绕 Wan2.2 TI2V 5B 的 self-attention Head 运动对应能力进行 PCK@32 统计，最终目标是把以下五组 720 个物理 Head 的排名、性能矩阵、重叠度和相关性统一展示在 8092 可视化页面中：

1. Legacy S039
2. GT teacher-forced
3. LoRA
4. Wan2.2 Baseline
5. Three-model combined，即 GT、LoRA、Baseline 的等权综合

页面只保留两个统计视图：

- `S039`：直接使用当前系列在 S039 的 PCK@32。
- `all_steps_mean`：先对每个物理 `(Block, Head)` 的 S000-S039 PCK@32 做算术平均，再对 720 个 Head 重新排序。这里平均的是 PCK 数值，不是 rank。

每个视图都需要：

- 30 × 24 PCK@32 性能矩阵；
- 完整 720 Head 排名；
- Top10、Top30、Top50、Top100；
- 五个系列全部 10 个两两组合的 Top-K 交集、覆盖率和 Jaccard；
- 对齐相同 720 个物理 Head 后的 Pearson、Spearman、平均 PCK 差和平均绝对 PCK 差。

## 2. 当前项目由两部分组成

### 2.1 已完成并已接入页面的五系列导出

当前 `/wan22-ti2v-legacy-pck50?v=2` 页面使用：

- 旧 Legacy：6 cases × 50 seeds = 300 runs，micro PCK@32；
- GT、LoRA、Baseline：各 50 case、seed=42，per-case macro PCK@32；
- Three-model combined：GT、LoRA、Baseline 对同一个 step/block/head 的 macro PCK 等权平均。

稳定导出文件：

```text
/data/gaoya/agent-data/outputs/pck_head_rankings/pck_head_rankings.json
/data/gaoya/agent-data/outputs/pck_head_rankings/pck_head_rankings.md
```

生成脚本：

```text
/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/export_pck_head_rankings.py
```

旧 Legacy 汇总：

```text
/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50
```

三模型汇总：

```text
/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case/block_step_head_summary.csv
/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case/three_model_combined_summary.csv
```

注意：旧五系列导出已经完整，不能在新的 67-case Legacy 只完成一部分时覆盖它。

### 2.2 正在运行的 Legacy PhysicIQ67 扩展统计

新实验将 Legacy 扩展到：

```text
67 cases × 50 unique seeds = 3350 runs
```

新实验独立保存，未覆盖旧 6-case Legacy：

```text
/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50
```

新区域缓存：

```text
/data/gaoya/agent-data/cache/wan22_ti2v_legacy_firstlatent_physiciq67_regions_704x1280
```

新实验完成后，需要明确选择：

- 用 67-case Legacy 替换页面中的旧 6-case Legacy；或
- 保留旧系列，并新增一个独立的 `Legacy PhysicIQ67` 系列。

当前需求语境倾向于前者，但在最终导出前仍应确认。不要把两个 Legacy 的 `correct32` 和 `comparisons` 直接相加，因为 case 集合、自动区域和研究目的不同。

## 3. 当前运行状态

2026-08-08 07:11 UTC 快照：

| 项目 | 状态 |
|---|---:|
| 区域缓存 | 67 / 67 |
| 文件系统完成标记 | 48 / 3350 |
| PCK `error.txt` | 0 |
| 最近一次增量聚合 | 47 / 3350 |
| GPU | 物理 GPU 6、7 |
| GPU 显存 | 各约 25.6 GiB |
| GPU 利用率 | 约 94%-100% |
| 预计总耗时 | 两卡约 55-70 小时 |

文件系统完成数可能比 `aggregate/summary.json` 大 1-2，因为聚合每 300 秒刷新一次。

`task_summary.json` 是任务清单生成时的快照，不是实时进度。当前清单是在完整冒烟 run 完成后生成的，因此记录 `completed_runs=1`、`missing_runs=3349` 是正常的。

实时状态以以下三处为准：

```text
runs/**/complete.json
aggregate/summary.json
tmux legacy_physiciq67_pck50
```

## 4. 数据来源

67 个 case 的输入列表：

```text
/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt
```

Formal compare 目录：

```text
/data/gaoya/AAA_test_video/0623/test/v2v/train0705_formal_compare/physicIQ/basemodel/wan2p2_ti2v5B_aligned49_steps40_512x896_49f_defaultnegprompt
```

该目录中有 67 个 case 视频和对应 JSON。类别分布：

| 类别 | case 数 |
|---|---:|
| Solid_Mechanics | 39 |
| Fluid_Dynamics | 15 |
| Optics | 8 |
| Thermodynamics | 3 |
| Magnetism | 2 |

50 个唯一 seed：

```text
/data/gaoya/agent-data/outputs/attention_lora_seed_sweep_case001460/seeds.txt
```

Wan2.2 TI2V 5B 权重：

```text
/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
```

CoTracker checkpoint：

```text
/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth
```

## 5. 统计口径

### 5.1 生成设置

| 项目 | 值 |
|---|---|
| backend | Legacy DiffSynth |
| model | Wan2.2 TI2V 5B |
| resolution | 704 × 1280 |
| output frames | 49 |
| FPS | 30 |
| sampling steps | 40，S000-S039 |
| solver | UniPC |
| CFG | 5.0 |
| sample shift | 5.0 |
| seeds | 每个 case 50 个唯一 seed |

### 5.2 Query 与 Head 设置

| 项目 | 值 |
|---|---|
| pixel query frame | frame 0 |
| latent query frame | latent index 0 |
| latent anchor pixel frames | 0, 4, 8, ..., 48 |
| query scope | object regions only |
| points | 每个自动识别对象 8 点 |
| physical heads | 30 blocks × 24 heads = 720 |
| capture | self-attention post RMSNorm、post 3D RoPE、pre flash-attention |
| CFG branch | positive conditional first call only |
| matching | 每个目标 latent frame 独立 Q-to-K argmax |
| head averaging | 不做；每个 Head 单独统计 |
| metric | PCK@32，阈值 32 像素 |

背景区域也会缓存 8 个点，用于检查区域划分，但 `object_queries()` 会明确过滤 background，背景不进入 PCK。

### 5.3 每个 run 的 PCK 计算

1. 对生成视频中的 object query points 执行 CoTracker。
2. 将 CoTracker 轨迹采样到 latent anchor frames。
3. 只保留 query frame 和目标 frame 都可见的比较。
4. 对每个 S/B/H，计算 attention Q-to-K argmax 预测点与 CoTracker 点的欧氏像素误差。
5. `error <= 32` 计入 `correct32`。
6. 每个 run 写出 `(40, 30, 24)` 的 `correct32`、`comparisons` 和 `error_sum`。
7. 全局聚合采用 micro aggregation：先累加 count，再计算 `100 * correct32 / comparisons`。

完整冒烟 run 已验证：

```text
captured_combinations = 28800
expected_combinations = 40 × 30 × 24 = 28800
metrics shape = (40, 30, 24)
```

## 6. 自动对象区域与例外

67-case JSON 没有旧 6-case 那样的人工 object phrase，因此采用：

```text
caption physical noun phrases
  -> GroundingDINO first-frame boxes
  -> SAM2 video propagation
  -> query-frame exclusive object masks
  -> 每对象 farthest-point sampling 8 点
```

每个 case 的实际 caption phrase、检测框、轨迹质量、SAM2 mask、采样点和过滤记录都保存在该 case 的 `regions.json`。

有 3 个 Fluid Dynamics case 出现不同 phrase 对应几乎相同的 SAM2 mask。处理规则是按 provider 已有的轨迹质量顺序保留有效轨迹，丢弃无法提供 8 个独立 query 像素的后续轨迹：

| case | 被丢弃 phrase | 独立像素 |
|---|---|---:|
| `physicIQ_Fluid_Dynamics_0068_perspective-center_trimmed-glass-stays-same` | `bright red liquid` | 0 |
| `physicIQ_Fluid_Dynamics_0137_perspective-center_trimmed-paper-in-water` | `green band showing the water` | 6 |
| `physicIQ_Fluid_Dynamics_0065_perspective-center_trimmed-fill-glass-red-drink` | `bright red liquid` | 0 |

记录位置：

```text
regions.json -> grounding_debug.exclusive_query_mask_filter
```

其余 64 个 case 没有触发该过滤。不要删除这些 provenance 字段，它们是解释不同 case object 数量和 comparisons 数量的依据。

## 7. 缓存方案 A

每个 case 目录只保留：

```text
complete.json
regions.json
regions.npz
```

默认不保存 PNG。67 个 case 缓存总计约 45 MiB。

`regions.npz` 包含：

```text
query_points
masks_rhw
context_frame_rgb
```

`regions.json` 包含：

```text
caption 与输入来源
自动抽取 phrase
GroundingDINO / SAM2 debug
对象与背景 region 切片
mask area
query frame
采样和重叠过滤元数据
```

cache root 下还可能存在轻量 worker 完成标记；方案 A 的“三个文件”约束指每个 case 目录。

## 8. 代码结构

| 文件 | 作用 |
|---|---|
| `legacy_ti2v_firstlatent_physiciq67_common.py` | 67 case、seed、模型、输入、输出和任务路径的单一配置源 |
| `prepare_legacy_ti2v_firstlatent_physiciq67_tasks.py` | 写 case manifest、实时扫描完成标记、生成 missing task JSONL |
| `precompute_legacy_ti2v_firstlatent_physiciq67_regions.py` | caption -> GDINO -> SAM2 -> 方案 A cache |
| `run_legacy_ti2v_firstlatent_physiciq67_pck_worker.py` | Wan 推理、attention hook、CoTracker、PCK 计算和单 run 写盘 |
| `run_legacy_ti2v_firstlatent_physiciq67_pck_task_worker.py` | 读取显式 JSONL，并按 worker id 做稳定分片 |
| `aggregate_legacy_ti2v_firstlatent_physiciq67_pck50.py` | 增量 micro 聚合、S/B/H 排名和跨 step 的 B/H 排名 |
| `launch_legacy_ti2v_firstlatent_physiciq67_gpu_worker.sh` | 单个物理 GPU 的区域阶段、67/67 barrier 和 PCK 串联启动 |
| `sam2_region_query_utils.py` | 共享 region cache、mask 侵蚀、点采样、读写与可选可视化 |
| `export_pck_head_rankings.py` | 五系列 S039/平均时间步导出、Top-K 重叠与相关性 |
| `serve_latent_block_head_viewer_with_metrics.py` | 8092 总览、Legacy 页面、API、下载和五系列比较 UI |

核心实现复用了旧 6-case worker 的 `CompactFirstLatentCapture`、CoTracker 和 PCK 逻辑。新实验没有修改旧 6-case 输出。

## 9. 产物结构

### 9.1 单 run

```text
OUTPUT_ROOT/runs/<case_key>/seed_<seed>/
  generated.mp4
  metrics.npz
  manifest.json
  complete.json
```

`metrics.npz`：

```text
correct32                    int32   [40, 30, 24]
comparisons                  int32   [40, 30, 24]
error_sum                    float64 [40, 30, 24]
per_object_correct32         int32   [objects, 40, 30, 24]
per_object_comparisons       int32   [objects, 40, 30, 24]
```

只有同时存在 `complete.json` 和 `metrics.npz` 的 run 才会被聚合。

### 9.2 增量聚合

```text
OUTPUT_ROOT/aggregate/
  combined_counts.npz
  ranking.json
  summary.json
  final_top10.json        # 仅 3350/3350 后生成
```

`ranking.json`：

- `global_step_block_head`：40 × 30 × 24 个 S/B/H，按 PCK@32 排序；
- `block_head_across_all_steps`：对 40 step 累加 count 后得到的 720 个 B/H 排名。

注意：这里的 `block_head_across_all_steps` 是跨 step 的 micro PCK；五系列导出中的 `all_steps_mean` 是 40 个 per-step PCK 的算术平均。两者权重定义不同，最终页面继续采用 `export_pck_head_rankings.py` 已定义的算术平均 view。

## 10. tmux 与 GPU

会话：

```bash
tmux attach -t legacy_physiciq67_pck50
```

窗口：

| window | 作用 |
|---|---|
| `0:gpu6` | worker 0，物理 GPU 6 |
| `1:gpu7` | worker 1，物理 GPU 7 |
| `2:aggregate` | 300 秒周期增量聚合 |

物理 GPU 绑定：

```text
worker 0: CUDA_VISIBLE_DEVICES=6，进程内 --device cuda:0
worker 1: CUDA_VISIBLE_DEVICES=7，进程内 --device cuda:0
```

禁止使用物理 GPU 4。

区域阶段环境：

```text
/home/gaoya/miniconda3/envs/wan-cu128/bin/python
```

原因：该环境具备 `decord`、GroundingDINO 和 SAM2。

Legacy PCK 阶段环境：

```text
/data/gaoya/miniconda3/envs/wan/bin/python
```

两个 GPU pane 的前台命令：

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
bash AAA_my_test/launch_legacy_ti2v_firstlatent_physiciq67_gpu_worker.sh 0 6
```

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
bash AAA_my_test/launch_legacy_ti2v_firstlatent_physiciq67_gpu_worker.sh 1 7
```

聚合前台命令：

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
/data/gaoya/miniconda3/envs/wan/bin/python \
  AAA_my_test/aggregate_legacy_ti2v_firstlatent_physiciq67_pck50.py \
  --watch --interval 300
```

日志：

```text
OUTPUT_ROOT/logs/gpu6.log
OUTPUT_ROOT/logs/gpu7.log
OUTPUT_ROOT/logs/aggregate.log
```

GPU 日志采用 append-only。里面保留了区域阶段首次运行时的 3 个 overlap traceback，后续已经修复并重试成功。因此不能仅凭日志中出现历史 `Traceback` 判断当前失败；应同时检查当前进程、case `complete.json` 和现存 `error.txt`。

## 11. 日常监控命令

实时完成数与错误数：

```bash
OUT=/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50
find "$OUT/runs" -name complete.json -type f | wc -l
find "$OUT/runs" -name error.txt -type f | wc -l
```

增量聚合状态：

```bash
sed -n '1,120p' \
  /data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/aggregate/summary.json
```

worker 进程：

```bash
ps -eo pid,etimes,args | \
  rg 'run_legacy_ti2v_firstlatent_physiciq67_pck_task_worker|aggregate_legacy_ti2v_firstlatent_physiciq67_pck50' | \
  rg -v 'rg '
```

GPU：

```bash
nvidia-smi --id=6,7 \
  --query-gpu=index,memory.used,memory.total,utilization.gpu \
  --format=csv,noheader
```

最近完成的 run：

```bash
find "$OUT/runs" -name complete.json -type f \
  -printf '%TY-%Tm-%Td %TH:%TM:%TS %h\n' | sort | tail
```

tmux pane：

```bash
tmux capture-pane -p -t legacy_physiciq67_pck50:gpu6 -S -40
tmux capture-pane -p -t legacy_physiciq67_pck50:gpu7 -S -40
tmux capture-pane -p -t legacy_physiciq67_pck50:aggregate -S -20
```

## 12. 恢复策略

### 12.1 只有一个 GPU worker 停止

不要重写 `missing_tasks.jsonl`。活着的 worker 已经按旧 JSONL 的行号确定了自己的奇偶分片，重写清单会改变分片边界，可能导致两卡重复或遗漏任务。

在对应 tmux pane 直接重新执行原命令：

```bash
bash AAA_my_test/launch_legacy_ti2v_firstlatent_physiciq67_gpu_worker.sh 0 6
```

或：

```bash
bash AAA_my_test/launch_legacy_ti2v_firstlatent_physiciq67_gpu_worker.sh 1 7
```

launcher 会跳过已完成的 67 个 region cache；task worker 也会跳过已有 `complete.json` 的 run。

### 12.2 两个 GPU worker 都停止

可以重新生成当前 missing task 清单，再同时启动两卡：

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
/data/gaoya/miniconda3/envs/wan/bin/python \
  AAA_my_test/prepare_legacy_ti2v_firstlatent_physiciq67_tasks.py
```

然后分别执行 GPU 6、7 launcher。只有在两个旧 worker 都确认停止后，才采用这套做法。

### 12.3 单个 run 失败

worker 当前是 fail-fast：发生异常后会在该 run 目录写 `error.txt` 并退出。

处理顺序：

1. 阅读该 run 的 `error.txt` 和对应 GPU 日志。
2. 确认是 transient CUDA/IO 问题还是确定性的 case 问题。
3. 修复后重启同一个 worker，不需要手工删除 `error.txt`；只要没有 `complete.json`，worker 会重试并覆盖生成结果。
4. 成功后应检查是否仍残留旧 `error.txt`。PCK worker 当前不会自动删除历史错误文件，如有成功完成但残留错误，需要人工核对后移除或修正 worker。

不要通过伪造 `complete.json` 绕过失败。

### 12.4 aggregate 停止

aggregate 是只读 run 指标、重写 aggregate 结果的幂等过程。直接执行第 10 节的聚合命令即可，不需要停止 GPU worker。

## 13. 完成验收

必须同时满足：

1. `runs/**/complete.json` 数量为 3350。
2. `runs/**/metrics.npz` 数量为 3350。
3. 不存在未解释的 `runs/**/error.txt`。
4. `aggregate/summary.json` 中 `completed_runs=3350`。
5. `aggregate/summary.json` 中 `final=true`。
6. `aggregate/final_top10.json` 存在。
7. `per_case` 恰好有 67 个 key，每个值都是 50。
8. `combined_counts.npz` 的三组核心数组 shape 都是 `(40, 30, 24)`。
9. 每个 S/B/H 的 comparisons 大于 0。

建议验收脚本：

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
/data/gaoya/miniconda3/envs/wan/bin/python - <<'PY'
import json
from pathlib import Path

import numpy as np

root = Path('/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50')
complete = list((root / 'runs').glob('*/*/complete.json'))
metrics = list((root / 'runs').glob('*/*/metrics.npz'))
errors = list((root / 'runs').glob('*/*/error.txt'))
summary = json.loads((root / 'aggregate/summary.json').read_text())
with np.load(root / 'aggregate/combined_counts.npz') as arrays:
    shapes = {key: arrays[key].shape for key in ('correct32', 'comparisons', 'error_sum')}
    minimum_comparisons = int(arrays['comparisons'].min())

assert len(complete) == 3350, len(complete)
assert len(metrics) == 3350, len(metrics)
assert not errors, errors[:5]
assert summary['completed_runs'] == 3350
assert summary['final'] is True
assert len(summary['per_case']) == 67
assert set(summary['per_case'].values()) == {50}
assert set(shapes.values()) == {(40, 30, 24)}
assert minimum_comparisons > 0
assert (root / 'aggregate/final_top10.json').is_file()
print('PASS', shapes, 'min comparisons', minimum_comparisons)
PY
```

## 14. 完成后更新五系列排名

当前 `export_pck_head_rankings.py` 明确硬编码旧 Legacy：

```python
LEGACY_ROOT = Path('/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_pck50')
```

若确认用 67-case 结果替换旧 Legacy，至少需要：

1. 将 `LEGACY_ROOT` 改为新 PhysicIQ67 output root。
2. 更新 `PROVENANCE['legacy_s039']` 中的 runs、67-case 数据来源、自动区域策略和实际 comparisons。
3. 保留两个 view：`s039` 和 `all_steps_mean`。
4. 重新运行导出脚本。
5. 核对每个 view 五个 series 都有 720 个 ranked heads。
6. 核对每个 view 都有 10 个 pairwise comparisons。
7. 核对每个 pair 都有 Top10/30/50/100、Pearson 和 Spearman。

导出命令：

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
/data/gaoya/miniconda3/envs/wan/bin/python \
  AAA_my_test/export_pck_head_rankings.py
```

导出会覆盖：

```text
/data/gaoya/agent-data/outputs/pck_head_rankings/pck_head_rankings.json
/data/gaoya/agent-data/outputs/pck_head_rankings/pck_head_rankings.md
```

覆盖前建议保留旧 6-case 导出的快照，以便比较和回滚。

## 15. 8092 可视化接入

服务脚本：

```text
/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/serve_latent_block_head_viewer_with_metrics.py
```

当前服务进程：

```text
/data/gaoya/miniconda3/envs/wan/bin/python \
  AAA_my_test/serve_latent_block_head_viewer_with_metrics.py \
  --host 0.0.0.0 --port 8092
```

前台启动命令：

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
/data/gaoya/miniconda3/envs/wan/bin/python \
  AAA_my_test/serve_latent_block_head_viewer_with_metrics.py \
  --host 0.0.0.0 --port 8092
```

页面与 API：

```text
http://127.0.0.1:8092/
http://127.0.0.1:8092/wan22-ti2v-legacy-pck50?v=2
http://127.0.0.1:8092/api/wan22-ti2v-legacy-pck50/catalog
http://127.0.0.1:8092/api/wan22-ti2v-legacy-pck50/comparison
http://127.0.0.1:8092/api/pck-head-rankings
http://127.0.0.1:8092/downloads/pck-head-rankings.json
http://127.0.0.1:8092/downloads/pck-head-rankings.md
```

当前 comparison API 会在每次请求时重新读取 `pck_head_rankings.json`，因此只更新稳定导出 JSON 时通常不需要重启服务。

但是 Legacy catalog 和性能矩阵仍硬编码旧 6-case 的：

```text
WAN22_TI2V_LEGACY_PCK50_ROOT
WAN22_TI2V_LEGACY_PCK50_CACHE
WAN22_TI2V_LEGACY_PCK50_CASES
```

如果页面还要浏览新的 67-case 视频、region 和实时 PCK 进度，必须同步更新这三个 server 配置，并前台重启 8092 服务。不要只替换 ranking JSON，否则比较表是新数据，但视频、进度和性能矩阵可能仍来自旧实验。

旧参考页 `neighbor-diagonal-ranking?v=4` 是另一套 S039、5-case、三模型对角线统计。它用于视觉形式参考，不应被当作本项目 50-case/67-case 排名的数据源。

## 16. 统计解释注意事项

旧五系列页面中的 Legacy 与三模型参考系列并非完全相同的实验设计：

| 项目 | Legacy | 三模型参考 |
|---|---|---|
| case/seed | 旧版 6 × 50；新版目标 67 × 50 | 每模型 50 case × seed 42 |
| resolution | 704 × 1280 | 512 × 896 |
| query pixel frame | 0 | 4 |
| query latent index | 0 | 1 |
| aggregation | micro | per-case macro |

因此 Head 重叠和相关性说明的是：在各自既定 protocol 下，同一物理 Head 的排名是否一致。它不是严格同样本、同 seed、同 query frame 的受控模型优劣比较。

三模型综合只综合 GT、LoRA、Baseline，不包含 Legacy。页面上的 `Legacy × Three-model combined` 是 Legacy 与该三模型综合系列的两两比较。

## 17. 不要做的事情

- 不要使用 GPU 4。
- 不要把大缓存、视频或模型写到 `/home/gaoya`。
- 不要覆盖旧 6-case Legacy output root。
- 不要在一个 worker 仍运行时重写 `missing_tasks.jsonl`。
- 不要把部分完成的 67-case aggregate 导出为最终五系列排名。
- 不要把 `block_head_across_all_steps` 的 micro PCK 与 `all_steps_mean` 的算术平均混为一谈。
- 不要仅根据 append-only 日志中的历史 traceback 判断当前状态。
- 不要删除 `regions.json` 中的 provenance 和 overlap filter 记录。
- 不要在没有完成验收的情况下生成或伪造 `final_top10.json`。

## 18. 最短接手路径

接手者首先执行：

```bash
tmux attach -t legacy_physiciq67_pck50
```

然后检查：

```bash
OUT=/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50
find "$OUT/runs" -name complete.json -type f | wc -l
find "$OUT/runs" -name error.txt -type f | wc -l
nvidia-smi --id=6,7 --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

如果两卡都在运行且错误数为 0，继续等待并观察 `aggregate/summary.json`。达到 3350/3350 后执行第 13 节验收，再按第 14、15 节更新稳定排名导出与 8092 页面。
