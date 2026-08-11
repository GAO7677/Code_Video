# Legacy PhysicIQ67 PCK50 项目交接文档

最后核对时间：2026-08-11 04:54 UTC

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

注意：旧五系列导出已经完整。67-case Legacy 虽已完成，但尚未自动替换该稳定导出；切换前仍需按第 14 节确认页面口径与 provenance。

### 2.2 已完成的 Legacy PhysicIQ67 扩展统计

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

该统计已于 2026-08-10 17:09 UTC 完成全部 3350 runs。正式接入五系列页面前，仍需要明确选择：

- 用 67-case Legacy 替换页面中的旧 6-case Legacy；或
- 保留旧系列，并新增一个独立的 `Legacy PhysicIQ67` 系列。

当前需求语境倾向于前者，但在最终导出前仍应确认。不要把两个 Legacy 的 `correct32` 和 `comparisons` 直接相加，因为 case 集合、自动区域和研究目的不同。

## 3. 当前运行状态

2026-08-11 04:54 UTC 最终核对：

| 项目 | 状态 |
|---|---:|
| 区域缓存 | 67 / 67 |
| `complete.json` | 3350 / 3350 |
| `metrics.npz` | 3350 / 3350 |
| PCK `error.txt` | 0 |
| 最终聚合 | 3350 / 3350，`final=true` |
| 67 个 case | 每个均为 50 seeds |
| `final_top10.json` | 已生成 |
| PCK worker | GPU 6、7 均已完成并退出 |

最终聚合时间为 2026-08-10 17:09 UTC；当前文件系统完成数与 `aggregate/summary.json` 已一致，不再存在增量聚合延迟。

`task_summary.json` 是任务清单生成时的快照，不是实时进度。当前清单是在完整冒烟 run 完成后生成的，因此记录 `completed_runs=1`、`missing_runs=3349` 是正常的。

实时状态以以下三处为准：

```text
runs/**/complete.json
aggregate/summary.json
tmux legacy_physiciq67_pck50
```

### 3.1 最终 S039 Rank JSON 与版本差异

后续新实验推荐固定使用最终 3350-run 排名：

```text
/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/pck_head_scopes_s039_latest3350.json
/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/cases_other10_6seeds_latest3350.json
```

最终 Rank JSON 已验证：720 个唯一 `(Block, Head)`、3350 个唯一 source runs、Top100 与配套 manifest 完全一致，并与最终 aggregate 的 S039 排序及 PCK 数值一致。

下表均以最终 3350-run 版本为参照；Top-K 表示成员集合重合，PCK 变化单位为百分点：

| 旧版本 | Runs | Top10 | Top30 | Top50 | Top100 | Spearman ρ | 720-head 平均名次变化 | 平均绝对 PCK 变化 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `frozen134` | 134 | 7/10 | 18/30 | 23/50 | 50/100 | 0.96846 | 40.15 | 6.669 |
| `latest2735` | 2735 | 10/10 | 30/30 | 50/50 | 99/100 | 0.99978 | 2.78 | 0.316 |
| `latest3027` | 3027 | 10/10 | 30/30 | 50/50 | 100/100 | 0.99994 | 1.47 | 0.183 |

关键差异：

- `2735 → 3350` 的 Top100 只有边界替换：`L24H03` 移出，`L23H09` 进入最终第 100 名。
- `3027 → 3350` 的 Top10/30/50/100 成员均不变，Top10 顺序也不变；只有集合内部小幅重排。
- `frozen134` 是早期快照，Top100 只有一半与最终版重合，不应作为新实验的默认排名。

已经生成的旧消融视频必须继续保留其 manifest 中的 `frozen134` 或 `s039r2735` provenance，不能事后改称由 3350-run 排名生成。最终版只用于新批次，或在明确重新生成旧批次时使用。

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

## 8. 代码结构与调用关系

### 8.1 项目边界

这里的“PCK Head 统计核心”只包括：构造 object query、捕获 self-attention Q/K、用 CoTracker 形成参考轨迹、计算每个 `(Step, Block, Head)` 的 PCK@32、聚合和排名。后续 attention-zero、temporal tube、VBench 和轨迹/感知指标只消费已经选出的 Head，不会反向写入 PCK 统计。

核心数据流：

```text
PhysicIQ67 case JSON + 50 seeds
  -> legacy_ti2v_firstlatent_physiciq67_common.py
  -> precompute_legacy_ti2v_firstlatent_physiciq67_regions.py
       -> 67-case GroundingDINO/SAM2 region cache
  -> prepare_legacy_ti2v_firstlatent_physiciq67_tasks.py
       -> physiciq67_cases.json + missing_tasks.jsonl
  -> launch_legacy_ti2v_firstlatent_physiciq67_gpu_worker.sh
       -> run_legacy_ti2v_firstlatent_physiciq67_pck_task_worker.py
       -> run_legacy_ti2v_firstlatent_physiciq67_pck_worker.py
       -> runs/<case>/seed_<seed>/{generated.mp4,metrics.npz,manifest.json,complete.json}
  -> aggregate_legacy_ti2v_firstlatent_physiciq67_pck50.py
       -> aggregate/{combined_counts.npz,ranking.json,summary.json,final_top10.json}
  -> export_pck_head_rankings.py
       -> 五系列 S039 / all_steps_mean 排名、Top-K 重合和相关性
  -> serve_latent_block_head_viewer_with_metrics.py
       -> 8092 页面与 API
```

### 8.2 PhysicIQ67 核心统计脚本

| 文件 | 阶段 | 功能与主要产物 |
|---|---|---|
| `legacy_ti2v_firstlatent_physiciq67_common.py` | 配置 | 67 case、50 seeds、Wan 权重、输入、region cache、run 输出和任务路径的单一配置源；同时提供 `all_tasks()`、`run_dir()` 和 case manifest 写入。 |
| `sam2_region_query_utils.py` | 区域公共库 | GroundingDINO/SAM2 region cache 的读写、mask 侵蚀、互斥 object mask、farthest-point query 采样、region 元数据和可选可视化。旧 6-case 与新 67-case 共用。 |
| `precompute_legacy_ti2v_firstlatent_physiciq67_regions.py` | 区域预计算 | 从 caption 抽取物理名词短语，执行 GroundingDINO 首帧检测和 SAM2 视频传播，写每个 case 的 `regions.json`、`regions.npz`、`complete.json`。支持多 worker 分片。 |
| `prepare_legacy_ti2v_firstlatent_physiciq67_tasks.py` | 任务准备 | 扫描 `runs/**/complete.json`，写 `physiciq67_cases.json`、`task_summary.json` 和显式 `missing_tasks.jsonl`；后者是双 worker 稳定分片的依据。 |
| `launch_legacy_ti2v_firstlatent_physiciq67_gpu_worker.sh` | 启动编排 | 绑定物理 GPU 6/7；先跑 region 预计算，等待 67/67 barrier，再启动对应 PCK task worker。进程内 `cuda:0` 由 `CUDA_VISIBLE_DEVICES` 映射到物理卡。 |
| `run_legacy_ti2v_firstlatent_physiciq67_pck_task_worker.py` | 任务分片 | 读取固定 JSONL，按行号 `index % num_workers` 做稳定分片；跳过已有 `complete.json`，复用一次加载的 Wan 与 CoTracker 逐任务执行，失败时写 `error.txt` 并 fail-fast。 |
| `run_legacy_ti2v_firstlatent_physiciq67_pck_worker.py` | 单 run 核心 | 加载 region cache，生成 49 帧视频，安装 attention hook，运行 CoTracker，对 40×30×24 组合计算 `correct32`、`comparisons`、`error_sum` 及 per-object 数组，原子写 `metrics.npz`，最后写 `complete.json`。 |
| `run_legacy_ti2v_firstlatent_pck_worker.py` | 共享计算内核 | 原始 6-case worker，同时向 67-case worker提供 `CompactFirstLatentCapture`、Wan pipeline 构建、CoTracker、`object_queries()` 和 PCK 公共逻辑；attention 捕获位置为 post RMSNorm、post 3D RoPE、pre flash-attention。 |
| `aggregate_legacy_ti2v_firstlatent_physiciq67_pck50.py` | 增量聚合 | 只读取同时有 `complete.json` 与 `metrics.npz` 的 run；micro 累加 count，写 `combined_counts.npz`、全 28,800 个 S/B/H 的 `ranking.json` 和实时 `summary.json`；仅 3350/3350 时写 `final_top10.json`。支持 `--watch --interval 300`。 |

### 8.3 旧 6-case Legacy 基线脚本

旧 6-case 结果是当前稳定页面中的 Legacy 系列，也是新 67-case 的实现基线。两套输出 root 独立，不能直接把 count 相加。

| 文件 | 功能 |
|---|---|
| `legacy_ti2v_firstlatent_common.py` | 定义旧 6 cases、50 seeds、人工 object phrase、输入、cache 和输出路径。 |
| `precompute_legacy_ti2v_firstlatent_regions.py` | 为旧 6 cases 构建首帧 GroundingDINO + SAM2 query cache。 |
| `run_legacy_ti2v_firstlatent_pck_worker.py` | 旧 6-case 的生成、attention 捕获、CoTracker 和 PCK 单 run 实现；也是新 67-case 的共享内核。 |
| `run_legacy_ti2v_firstlatent_pck_task_worker.py` | 旧实验的显式 JSONL runner；用于按固定任务清单恢复或重新分片。 |
| `aggregate_legacy_ti2v_firstlatent_pck50.py` | 汇总旧 300 runs，产物结构与 67-case aggregate 对齐。 |
| `run_legacy_ti2v_firstlatent_top10_heatmaps_worker.py` | 等待旧 aggregate 的最终 Top10，确定性重跑各 seed 并生成 object-query attention 热力图；不参与 PCK 数值聚合。 |
| `launch_legacy_ti2v_firstlatent_pck50_gpu0123.sh` | 旧实验的 region、四卡 PCK、aggregate 和最终 Top10 heatmap tmux 编排入口。 |

### 8.4 三模型参考系列与五系列导出

| 文件 | 功能 |
|---|---|
| `aggregate_allblocks_allsteps_headwise_50case.py` | 验证并聚合 GT teacher-forced、LoRA、Wan2.2 Baseline 各 50 cases 的全 40 steps、30 blocks、24 heads Q@K 结果，输出 `block_step_head_summary.csv` 等文件。 |
| `aggregate_three_model_combined_rankings.py` | 对 GT、LoRA、Baseline 的同一 S/B/H 做等模型权重平均，生成 `three_model_combined_summary.csv`。Three-model combined 不包含 Legacy。 |
| `export_pck_head_rankings.py` | 读取 Legacy `combined_counts.npz`、三模型 summary 和 combined summary，生成两个 view（`s039`、`all_steps_mean`）的五系列720 Head排名、Top10/30/50/100交集、Jaccard、Pearson/Spearman及 Markdown/JSON。当前 `LEGACY_ROOT` 仍指向旧 6-case；67-case 已完成第13节验收，但切换前仍需按第14节确认页面口径与 provenance。 |

必须区分两个跨 step 定义：aggregate 的 `block_head_across_all_steps` 是先合并 count 的 micro PCK；页面的 `all_steps_mean` 是40个 per-step PCK的算术平均。五系列正式导出采用后者。

### 8.5 可视化脚本

| 文件 | 功能 |
|---|---|
| `build_legacy_ti2v_firstlatent_physiciq67_visual_samples.py` | 从已完成 runs 固定抽样，保存单 run 的 S039/all-steps-mean 30×24矩阵、region 图和当时的 provisional S039 Top100 快照到 `visual_samples/samples.json`。已有100个 entries 时默认保留旧快照，不会自动追随 aggregate。 |
| `run_legacy_ti2v_firstlatent_physiciq67_visual_sample_heatmaps.py` | 对 manifest 中的样本重跑并捕获 provisional S039 Top100 attention heatmap，仅用于定性检查。 |
| `serve_latent_block_head_viewer.py` | 最底层三模型 per-head Q@K 轨迹/矩阵查看器。 |
| `serve_latent_block_head_viewer_alltoken.py` | 在基础查看器上增加全 token、Top/Bottom Head、消融和 overlay 数据。 |
| `serve_latent_block_head_viewer_with_metrics.py` | 当前 8092 服务入口；加载前两层 viewer，并增加旧 6-case PCK 页面、五系列比较 API、PhysicIQ67 样例、消融视频和指标页面。它只读取结果，不计算或修改 PCK。 |

### 8.6 排名下游脚本，不属于 PCK 统计核心

下列脚本使用 PCK 排名选择 Top/Bottom Head，但输出是干预视频或视频质量/轨迹指标。它们不能作为 PCK 排名的数据源：

| 文件 | 功能 |
|---|---|
| `build_legacy_attention_zero_seed47326_manifest.py` | 为固定 seed=47326 样例建立消融 manifest，继承当时冻结的 S039 Top100。 |
| `build_legacy_object_ablation_other10_6seed_manifest.py` | 建立10-case×6-seed复现实验 manifest；默认继承旧快照，`--latest-ranking` 可从当前 67-case aggregate 生成新 S039 Top100 快照。 |
| `build_frozen_s039_head_scope_ranking.py` | 在冻结 Top100 的语义下重建完整720 Head顺序，供 Top100、Bottom100、All720 head-scope 对照使用。 |
| `run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations.py` | 对 Top30/50/100执行固定 F00 object-query attention matrix block 消融并审计实际修改事件。 |
| `run_legacy_ti2v_temporal_object_tube_ablations.py` | 将 object query扩展为冻结的全 latent 时间 tube，执行 M1/M2/M3及不同 head scope 干预。 |
| `object_query_ablation_metrics/*.py` | 计算消融视频相对 GT/Baseline 的轨迹、形状、RAFT、DINOv2、LPIPS、VBench 和像素指标；这些是干预效果评估，不是 Attention Q→K PCK。 |

消融评价中全部已实现指标的定义、精确公式、数值方向和代码路径统一记录在：
[`object_query_ablation_metrics/METRICS_IMPLEMENTATION_INDEX.md`](object_query_ablation_metrics/METRICS_IMPLEMENTATION_INDEX.md)。

因此，排查“某个 Head 为什么进入排名”时，应依次检查 PCK worker 的 `metrics.npz`、aggregate 的 `combined_counts.npz/ranking.json` 和 export 的 `pck_head_rankings.json`，不要从消融 manifest 或25项视频指标报告反推 PCK 排名。

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

2026-08-11 已按本节脚本实际验收通过：3350 个 `complete.json`、3350 个 `metrics.npz`、0 个 `error.txt`；`summary.final=true`；67 个 case 均为 50 runs；三个核心数组 shape 均为 `(40, 30, 24)`，最小 comparisons 为 `617255`；`aggregate/final_top10.json` 存在。

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

67-case PCK 统计与最终 S039 720-head JSON 已完成，但本节所述五系列稳定导出尚未自动切换，不能把“最终 Rank JSON 已生成”误写成“8092 五系列页面已替换”。

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

当前统计已经完成，不需要重启 GPU 6/7 worker。接手者首先复核最终聚合与版本化排名：

```bash
OUT=/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50
find "$OUT/runs" -name complete.json -type f | wc -l
find "$OUT/runs" -name error.txt -type f | wc -l
sed -n '1,40p' "$OUT/aggregate/summary.json"
ls -lh "$OUT/visual_samples/attention_zero_seed47326/pck_head_scopes_s039_latest3350.json"
```

预期分别为 `3350`、`0`、`completed_runs=3350` 且 `final=true`。之后按第 14、15 节决定是否用 67-case Legacy 替换旧五系列导出与 8092 页面；该选择目前仍未自动执行。

## 19. 001460 / seed 47326：108 项 S039 Top100 Mean Overlay

实验矩阵固定为：

```text
3 targets（Object A / Object B / all_objects）
× 3 intervention Head Scopes（Top100 / Bottom100 / All720）
× 3 operators（M1 / M2 / M3）
× 4 temporal scopes（All-time / Same / Future / Past）
= 108
```

完整编号清单：

```text
/data/gaoya/agent-data/outputs/object_query_attention_overlays/
  m123_head_scope_s039_top100_mean_v1/EXPERIMENT_LIST.md
/data/gaoya/agent-data/outputs/object_query_attention_overlays/
  m123_head_scope_s039_top100_mean_v1/experiment_list.json
```

观察协议必须与干预 Head Scope 分开解释：

- 干预 Head Scope 是各视频实际消融的 `Top100`、`Bottom100` 或 `All720`。
- 观察 Head 永远是冻结 S039 PCK ranking 的 Top100，用于跨实验统一比较。
- Query 永远是 F04 的固定 Object A/B SAM2 queries，每个对象 8 点，latent `tq=1`。
- 本 case 的实际生成分辨率是 `704×1280`，因此运行时 latent attention grid 为 `13×22×40=11440` tokens；F04 cache 的 `512×896` query 坐标按归一化位置映射到该 `22×40` 网格，不能沿用旧 `13×16×28` 常量。
- 只抓 denoising `S039`；两个 CFG 调用平均。
- `Before` 是该消融重放运行中进入当前算子前的 `softmax(QK^T/sqrt(d))`。
- `Effective After` 只把该 M1/M2/M3 算子实际删除的 entries 设为 0，不重新归一化。
- `Removed = max(Before - Effective After, 0)`。
- 每张图横向展示 `F00/F04/…/F48`；Query 行先对 8 点求和，再对 100 heads 和两个 CFG 调用取平均。

因此 Bottom100 实验的 Top100 局部 `Effective After` 与 `Before` 相同，因为两个 Head 集合不相交；但其 `Before` 仍可能包含早期层和早期去噪步 Bottom100 干预传来的上游变化。

相关代码：

```text
AAA_my_test/run_legacy_m123_head_scope_s039_top100_mean.py
AAA_my_test/render_m123_s039_top100_mean_overlay.py
AAA_my_test/run_legacy_m123_s039_top100_mean_gpu.sh
AAA_my_test/launch_legacy_m123_s039_top100_mean_tmux.sh
AAA_my_test/test_m123_s039_top100_mean_capture.py
```

当前队列（按用户优先级改为 GPU 2/3 双 worker）：

```bash
tmux attach -t m123_s039_top100_mean_capture
```

该 session 当前只有 GPU 2/3 两个 worker，每卡 54 项；worker 已进入实际推理，不使用 GPU 4，也不会覆盖已有 108 个 `generated.mp4`。为腾出 GPU 2/3，本轮仅停止了旧 session `m123_priority_001460_then_latest` 中对应的 `:2`、`:3` 两个窗口；GPU 0/1/5/6/7 上的任务未改动。

首轮已验收 4 项并继续增量生成；每个完成目录包含 2 个 NPZ、Object A/B 各 3 张单行图和 1 张三行 comparison、`overlay_manifest.json`、`manifest.json`、`complete.json`。审计要求严格为 100 个物理 observation heads、每 head 两个 CFG 调用，即每对象 200 head instances。

独立 Overlay 页面：

```text
http://localhost:8092/object-query-m123-s039-top100-mean-overlays?v=5
```

页面按 Target、Head Scope、M1/M2/M3 和 All-time/Same/Future/Past 筛选，每 20 秒检查一次新产物并增量刷新。三行 comparison 的含义固定为：

1. `Pre-mask / Before`：当前消融重放在 S039、进入当前算子前的实时 softmax，已包含上游干预效应，不是未消融 Baseline。
2. `Effective After`：只将当前算子精确删除的 coefficient entries 设为 0；不重新 softmax、不重归一化、不改 Q/K/V。
3. `Removed=max(Before-Effective After,0)`：当前算子在 observation Top100 中直接删除的 coefficient mass；不是相对 Baseline 的 attention 差，也不是视频运动差。

总入口 `http://localhost:8092/` 的 `M1/M2/M3 Head-Scope Overlay` 卡片直接链接该独立页面。

## 20. 001460 / seed 47326：108 项 Query-side Receiver Overlay

这一组图补充第 19 节的固定 Object Query、Key-side 图。它不再问“固定 Object Query 读取了哪些 Key”，而是把 Query 位置 `q` 遍历到整个时空 token 网格，回答“被选中 Object 的 K/V 信息由谁读取”。实验清单仍是同一组 108 项，Target、Head Scope、M1/M2/M3 和 All-time/Same/Future/Past 必须与对应消融视频一一匹配。

对每个实验的真实干预 Head Scope、真实 source 集合 `K_sel(q)`，在 denoising `S039` 计算两种 Query-side 量：

```text
Coefficient mass:
S(q) = sum_{k in K_sel(q)} A(q,k)

Value contribution norm:
E(q) = mean_h || sum_{k in K_sel(q)} A_h(q,k) V_h(k) ||_2
```

精确含义：

- `S(q)` 是该 Query 在当前算子下被删除的 attention coefficient mass；理论范围为 `[0,1]`。它只表示“读了多少”，不考虑 Value 的大小、方向和抵消。
- `E(q)` 是该 Query 因删除相应 K/V 信息而失去的实际 head 输出向量范数；非负。实现先在每个 head 内完成向量求和与 L2 norm，再跨 Head/CFG 平均，避免不同 head 的向量坐标系被错误相加。
- 两行都画回完整 Query 时空网格，横向列为 `Q00/F00, Q01/F04, ..., Q12/F48`。青色轮廓标出当前 Object tube 在每个 Query 时刻的位置。
- 每个实验的两行分别使用该实验自身跨时刻 `P99.5` 固定色标；同一行内部可以比较时间，但不同实验之间不能只凭颜色深浅比较绝对值，跨实验比较必须读取 manifest 中的数值统计。
- 这里只在 `S039` 做观测，但生成时的 M1/M2/M3 干预仍作用于原消融配置指定的全部 denoising steps；不要把 observation step 和 intervention step 混为一谈。

M1/M2/M3 的 Query-side receiver 集合分别是：

| Operator | 被保留用于求和的 source `k` | 输出 Query `q` 的诊断含义 |
|---|---|---|
| M1 | `k in R_target`，且 `q in R_target` | Object tube 内部哪些 Query 正在读取该 Object 自身的 K/V |
| M2 | `k in C`，且 `q in R_target` | Object Query 从背景/其他对象 K/V 接收了多少信息 |
| M3 | `k in R_target`，且 `q in C` | 背景/其他对象 Query 从目标 Object K/V 接收了多少信息，即目标信息的广播接收者 |

`Same/Future/Past` 继续对每个 Query 时刻逐点施加 `t_k=t_q`、`t_k<t_q`、`t_k>t_q` 条件，而不是先把某个固定帧整体切走。因此 M3 的图尤其适合区分 Object B、背景或其他时刻的 token 是否在读取目标 Object。

相关代码：

```text
AAA_my_test/run_legacy_m123_s039_query_receiver.py
AAA_my_test/render_m123_s039_query_receiver_overlay.py
AAA_my_test/run_legacy_m123_s039_query_receiver_gpu.sh
AAA_my_test/launch_legacy_m123_s039_query_receiver_tmux.sh
AAA_my_test/test_m123_s039_query_receiver.py
```

输出目录：

```text
/data/gaoya/agent-data/outputs/object_query_attention_overlays/
  m123_head_scope_s039_query_receiver_v1/
```

每个完成实验包含原始 `receiver.npz`、`S(q)` 单行图、`E(q)` 单行图、两行 comparison、`overlay_manifest.json`、`manifest.json` 和 `complete.json`。页面在第 19 节同一个实验卡片中显示 receiver 产物；未生成时只显示等待状态，不伪造空图。

当前 tmux 队列：

```bash
tmux attach -t m123_s039_query_receiver_capture
```

GPU2 先运行 `single_object / Object A / M3 All-time / Top100` pilot；pilot 写出 `complete.json` 后 GPU2/3 才自动展开余下实验。GPU 4 明确禁用。页面每 20 秒轮询 receiver 进度并自动刷新：

```text
http://localhost:8092/object-query-m123-s039-top100-mean-overlays?v=5
```
