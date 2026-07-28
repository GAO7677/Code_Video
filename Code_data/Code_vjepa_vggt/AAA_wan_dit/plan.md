# S/T/C Head 类别效应与消融剂量解耦实验计划

## 1. 研究目标

本实验要回答两个不同的问题，二者不得混为一个结论：

1. **整类总效应**：删除全部 S、T 或 C 类 head 时，模型输出发生多大变化。
2. **类别特异效应**：在消融 head 数量、所在 block 和实际扰动能量相同的条件下，S、T、C head 是否仍产生不同影响。

当前 all-category 实验只能回答第一个问题。由于公共稳定 head 数量为
`S=159、T=13、C=20`，all-S 的干预规模分别是 all-T 的 12.23 倍、all-C
的 7.95 倍，不能据此直接判断单个 S-head 更重要。

## 2. 当前分类与干预定义

### 2.1 分类数据

- 模型：Wan+LoRA、Wan+xSSC、PhysRVG。
- 当前公共快照：22 seeds × 20 source cases = 440 samples/model。
- DiT：30 blocks × 24 heads，共 720 个 head。
- 注意力采样去噪步：5、15、25、35。
- 时间 latent：13。
- 分类输入：全 token 时间注意力矩阵与 moving-object 轨迹富集特征。

### 2.2 类别分数

所有特征先在同一去噪步内转换为 head 间相对 rank，再计算。以下 rank 和 score
均为可重新生成的派生数据，不是分类数据的唯一存档：

```text
score_S = 0.55 rank(local_enrichment)
        + 0.45 rank(same_frame_mass)

score_T = 0.55 rank(trajectory_selectivity_log2)
        + 0.25 rank(trajectory_enrichment)
        + 0.20 rank(mean_time_distance)

score_C = 0.55 rank(object_context_enrichment)
        + 0.25 rank(full_context_enrichment)
        + 0.20 rank(history_bias)
```

P、G 类保留用于对照，但本实验的主问题为 S/T/C。

### 2.3 原始分类特征的强制保存

必须保存每个 head 的全部原始具体数值，不能只保存 rank、score 或最终 role。
最小保存粒度为：

```text
(model, source_case, seed, denoise_step, block, head)
```

每条记录至少包含：

```text
local_enrichment_raw
same_frame_mass_raw
trajectory_selectivity_log2_raw
trajectory_enrichment_raw
mean_time_distance_raw
object_context_enrichment_raw
full_context_enrichment_raw
history_bias_raw
```

所有数值必须保存计算得到的未格式化浮点值；CSV/页面中的小数截断只能用于展示，
不能覆盖原始文件。对于由比值、富集度或对数计算得到的特征，还必须保存其计算
组成，包括：

- 原始分子和分母。
- 有效 query、key、trajectory 和 object token 数。
- attention mass 总量及对应区域的 attention mass。
- epsilon、log 底数、归一化常数和空集合处理方式。
- trajectory/object/context 有效性标记及无效原因。
- 输入时间长度、空间 token 网格和 context 帧索引。

原始特征与派生结果分文件保存：

```text
raw_head_features.parquet       # 不可变原始统计量及其组成
derived_head_ranks.parquet      # rank及rank计算元数据
derived_head_scores.parquet     # score、margin、support和role
```

`raw_head_features.parquet` 是后续分析的 source of truth。重新调整 score 权重、
增加或删除特征、修改 rank 范围或 tie 规则时，只允许从该文件重新派生，不得从
已经舍入的表格、score 或 role 反推。

为了能够精确复现 rank，派生文件还要记录：

- 参与排序的 head 集合和有效 mask。
- rank 的作用域（同 model/source/seed/step 内）。
- 升序或降序方向。
- ties 的处理方法。
- NaN/Inf 和无效轨迹的处理方法。
- score 公式、权重及版本号。
- 原始特征文件的 SHA256。

如果现有实现只输出 rank/score，则 Phase 0 不通过，必须先补齐原始特征导出，
再开展新的分类或消融实验。

### 2.4 稳定分类规则

单个 model-case-seed 样本中：

- 对去噪步 5、15、25、35 的类别分数取平均。
- 最高分为候选类别。
- 第一、第二名 margin < 0.08，或四个去噪步获胜一致率 < 0.75 时标记为 M。

跨样本聚合中：

- 对当前完成的 22 seeds × 20 cases 平均类别分数。
- 聚合最高分为候选类别。
- 聚合 margin < 0.08 或样本支持率 < 0.50 时标记为 M。
- T/P 仅使用轨迹有效样本；S/C/G 使用全部样本。

最终“公共稳定 head”要求同一 `(block, head)` 在三个模型中具有相同的非 M
聚合类别。当前数量如下：

| 类别 | 公共 head 数 | 覆盖 block 数 |
|---|---:|---:|
| S | 159 | 30 |
| T | 13 | 10 |
| C | 20 | 11 |
| P | 82 | 26 |
| G | 75 | 26 |

### 2.5 消融语义

当前消融在 self-attention 输出投影 `self_attn.o` 之前，将选定 head 的完整
输出切片置零：

```text
per_head[..., selected_heads, :] = 0
output = self_attn.o(concat(per_head))
```

剩余 head 不进行重新归一化。因此，head 数量和各 head 输出能量都会改变实际
干预强度。

## 3. 预注册假设

### H1：纯数量假设

如果当前 S-all 的较差结果主要来自 head 数量更多，则在匹配 head 数量、block
分布和扰动能量后，S/T/C 的影响差异应显著缩小，类别主效应接近零。

### H2：类别特异假设

如果 S-head 具有类别特异的重要功能，则在相同数量和相近扰动能量下，
S 子集仍应比 T/C 子集引起更大的输出变化或更明显的质量下降。

### H3：能量假设

如果差异主要由 head 输出幅度造成，则加入实际扰动能量后，类别系数应明显
减小，而扰动能量应能解释主要方差。

### H4：非线性交互假设

如果大规模 S 消融触发类别内部交互或分布外行为，则小规模 S 子集的剂量曲线
不能外推到 all-S；all-S 会显著偏离小 k 曲线的置信区间。

## 4. 必须控制的混杂因素

| 混杂因素 | 风险 | 控制方式 |
|---|---|---|
| Head 数量 | S=159，T=13，C=20 | 固定 k 的随机子集 |
| Block 分布 | S遍布30层，T/C集中于部分层 | block/depth匹配与协变量控制 |
| 输出能量 | 不同head的输出幅度不同 | 匹配实际 projected perturbation energy |
| 分类置信度 | S内部可能高度异质 | support、margin分层与高置信度敏感性分析 |
| Case/seed | 视频生成随机性较大 | 同source、同seed、同model配对 |
| 去噪阶段 | Head作用随阶段变化 | 固定区间并显式估计 Role×Stage |
| 多head交互 | 总效应不等于单head效应之和 | 剂量曲线和all-class外推检验 |
| Case复用 | 当前分类和评测都使用test_5 | 确认性实验使用held-out cases |
| 指标歧义 | Physics指标排序并不完全一致 | 预注册主指标并单独报告各指标 |

## 5. 实验总体结构

实验按四个阶段执行。后一个阶段开始前必须完成前一阶段的数据完整性检查。

### Phase 0：冻结分类与干预审计

目标是保证后续比较只改变目标 head 集合。

1. 冻结当前公共分类文件、SHA256、22 seeds 快照和分类代码版本。
2. 导出逐 model/source/seed/step/block/head 的全部原始特征、计算组成和有效性
   标记，并检查主键唯一性。
3. 仅从原始特征文件重新生成 rank、score、margin、support 和 role；与当前分类
   逐项对比，差异必须为零或有明确记录的数值精度原因。
4. 输出每类 head 的 `(block, head, score_role, margin, support)`。
5. 统计 S/T/C 的精确 block 分布。
6. 对三个模型验证相同 `(block, head)` 指向相同 head 切片。
7. 运行 sham ablation：安装空干预路径但不置零任何 head，输出必须与 baseline
   完全一致或处于已定义的数值容差内。
8. 所有结果 JSON 必须记录 head 数量、head 列表、block 直方图、去噪区间和分类
   文件 hash。

通过条件：

- 原始特征表主键完整且唯一，预期记录无静默缺失。
- 原始特征能够独立重建当前全部 rank、score 和 role。
- 三模型目标列表完全一致。
- sham 与 baseline 的 latent/output 一致。
- 每个实际任务的 hook 调用次数符合预期。

### Phase 1：数量匹配随机子集实验

这是判断数量混杂的首要实验。

#### 主设计

- 类别：S、T、C。
- 固定数量：`k=8`。
- 每类随机子集：预注册 `R=20` 组。
- 每个随机子集在三个模型中使用相同 `(block, head)`。
- 每个 source、seed、model、stage 均共享同一个 baseline。
- 随机子集列表在生成视频前冻结，不得根据结果重新抽样。

选择 k=8 的原因：

- 小于最小类别 T 的 13，三个类别均有足够组合空间。
- 干预强度足以产生可测变化。
- 避免 T=13 时只能使用完整 T 集、无法估计子集方差。

#### Depth 匹配

对每个 T 子集，从 S 和 C 中选择 depth 分布最接近的子集。匹配变量为：

```text
early_count = blocks 0-9
middle_count = blocks 10-19
late_count = blocks 20-29
mean_block
std_block
```

使用预注册距离函数进行最小距离匹配。每个 matched triplet
`(S subset, T subset, C subset)` 作为一个随机化区组。

#### 精确 block 敏感性实验

S/T/C 共同包含 head 的 blocks 为 `B09、B15、B16、B17、B28`。增加一个
`k=5` 敏感性实验，每个类别在上述五个 block 中各选择一个 head。该实验用于
消除 block 位置差异，但不作为唯一主实验，因为 T 类可形成的独立组合有限。

#### 随机 head 对照

为每个 matched triplet 增加一个 `Random-k`：

- 从非目标公共 head 或稳定 M/其他类别中抽取。
- 与目标子集匹配 depth 分布和扰动能量。
- 用于判断观测下降是否只是“任意关闭 k 个 attention head”的通用后果。

### Phase 2：剂量反应实验

使用：

```text
k ∈ {1, 2, 4, 8}
```

每个 role-k 组合至少抽取 10 个冻结子集。额外保留：

- `k=13`：连接到 all-T。
- all-S=159、all-T=13、all-C=20：仅作为整类总效应端点。

优先使用 nested subsets：

```text
subset_1 ⊂ subset_2 ⊂ subset_4 ⊂ subset_8
```

这样同一条曲线增加 head 时只增加新 head，降低子集组成噪声。另保留独立随机
子集作为稳健性检查。

本阶段要回答：

- 影响是否随 k 单调增加。
- S/T/C 在小 k 区域的边际斜率是否不同。
- 是否存在饱和、抵消或突变。
- all-S 是否偏离小 k 曲线外推。

### Phase 3：扰动能量匹配实验

数量相同不代表实际干预强度相同。对 baseline 前向记录每个目标 head 经过输出
投影后的贡献，定义：

```text
delta_attn = self_attn.o(z) - self_attn.o(mask(z))

relative_energy =
sqrt(sum_steps ||delta_attn||_F^2)
/
sqrt(sum_steps ||self_attn.o(z)||_F^2)
```

能量按 model、source、seed、stage 分别计算。禁止仅用 `num_heads` 代替能量。

执行两种分析：

1. 在固定 k 子集中，将 `relative_energy` 作为协变量。
2. 从 S/T/C 候选子集中选择能量差在预注册 caliper 内的 matched triplet。

建议初始 caliper 为 matched triplet 平均能量的 10%；若无法形成足够匹配，
放宽规则必须在查看质量指标前完成并记录。

可选验证使用 soft ablation：

```text
z_head <- (1 - alpha) * z_head
```

通过选择 alpha 使三类被移除的总能量相近。soft ablation 仅作为敏感性实验，
不能替代 zero-ablation 主实验。

## 6. 数据划分

### 6.1 Pilot cases

现有 test_5 的20个case可用于：

- 检查实现正确性。
- 估计方差和计算预算。
- 选择合理 k、R 和能量 caliper。

这些case已经参与head分类，不得作为确认性结论的唯一证据。

### 6.2 Confirmatory held-out cases

确认性实验必须使用未参与head分类的source cases。冻结前按物理类型分层：

- 重力/落体。
- 刚体碰撞。
- 摩擦/滑动。
- 多物体交互。
- 软体/形变。
- 液体或其他连续介质。

每类物理现象应包含多个case，且三模型使用完全相同的case列表。case列表必须在
生成前写入配置文件并计算SHA256。

### 6.3 Seeds

Pilot 使用至少2个公共seed。确认性实验使用至少6个公共seed，并保持三个模型、
全部role和全部subset完全配对。优先使用现有：

```text
851, 3278, 11395, 20379, 28221, 32098
```

## 7. 去噪区间

主分析区间：

```text
[0,5), [5,10), [10,20), [20,30)
```

`[0,10)`作为组合区间敏感性分析。暂不将`[30,40)`纳入主结论。

所有区间使用左闭右开语义，并对CFG的conditional/unconditional调用应用相同
mask。Role×Stage 交互必须显式报告，禁止跨阶段简单平均后得出唯一结论。

## 8. 评价指标

### 8.1 Primary endpoint A：输出因果影响

使用既有轨迹/光流分析中的 `Motion Impact`，比较消融视频与同
model-source-seed baseline 的轨迹差异。该指标回答“干预改变输出多少”，不直接
等价于“质量变差多少”。

### 8.2 Primary endpoint B：物理合理性

Physics-IQ with context 与 WMReward surprise 分别作为共同主指标：

- Physics-IQ with context：越高越好。
- WMReward surprise：越低越好。

仅当二者方向一致时，才声明“物理质量一致改善/下降”。二者冲突时标记为
metric disagreement，不强行合成单一结论。

### 8.3 Secondary endpoints

- GT-relative trajectory plausibility/gain。
- PMF with/without context。
- VideoPhy2 PC generated-only。
- VideoPhy2 PC raw/full-video。
- Cosmos-Reason1。
- VBench consistency、smoothness、dynamic、quality。
- 追踪失败率和生成失败率。

VideoPhy2 与 Cosmos 分数离散且存在较多排序冲突，仅作为诊断指标。

## 9. 统计分析

### 9.1 基本观测量

所有指标先转换为相对同一 baseline 的配对变化：

```text
DeltaY = metric(ablated) - metric(baseline)
```

WMReward 等“越低越好”指标在 improvement 分析时反向，但保留原始 DeltaY。

### 9.2 主模型

对每个主指标拟合层级模型：

```text
DeltaY ~ Role
       + f(log2(k))
       + Role:f(log2(k))
       + relative_energy
       + depth_histogram
       + Stage
       + Role:Stage
       + Model
       + (1 | SourceCase)
       + (1 | Seed)
       + (1 | MatchedSubsetTriplet)
```

如果数据不足以稳定拟合全部随机效应，则使用case与seed双层cluster bootstrap，
不得把同一case/seed下的视频当作独立样本。

### 9.3 预注册对比

每个 stage 的主要对比为：

```text
S(k=8) - T(k=8)
S(k=8) - C(k=8)
T(k=8) - C(k=8)
```

同时报告：

- 点估计。
- 95%置信区间。
- 标准化效应量。
- case级和seed级方差。
- matched subset间方差。

共同主指标和三个role对比使用Holm校正。Secondary指标不用于确认性成功判定，
但必须完整展示，不筛选显著结果。

### 9.4 禁止的分析方式

- 禁止直接用 `DeltaMetric / num_heads` 作为单head效应。
- 禁止只比较 all-S、all-T、all-C 后声明类别重要性。
- 禁止根据结果选择“最好看的”随机子集。
- 禁止将同一source或seed下的多个视频视为独立样本。
- 禁止因某个指标不支持假设而在结果阶段更换主指标。

## 10. 判定规则

### 数量效应主导

满足以下现象时，支持“当前S-all差异主要来自剂量”：

- 固定 k 后 S-T、S-C 的效应明显缩小且置信区间包含零。
- relative_energy 或 k 能解释主要变化。
- S/T/C 的小 k 剂量曲线斜率接近。

### 类别效应成立

仅在以下条件同时满足时支持“S具有类别特异影响”：

- 固定 k、depth和energy后，S与T/C仍存在预注册方向的实质差异。
- 结果在held-out cases上成立。
- 至少两个模型方向一致，第三个模型不存在明确反向的大效应。
- Physics-IQ与WMReward共同主指标不存在系统性反向。

### 非线性交互成立

满足以下任一条件时，不使用“平均单head贡献”解释all-S：

- all-S落在小 k 剂量曲线95%预测区间之外。
- Role×k交互显著且效应量具有实际意义。
- S子集影响随k出现饱和、抵消或突变。

## 11. Pilot与确认性停止规则

Pilot只用于估计：

- baseline方差。
- subset方差。
- 能量分布。
- 追踪失败率。
- 单任务显存、时间和存储成本。

Pilot结束后，在查看held-out结果前完成基于cluster bootstrap或仿真的power分析，
冻结：

- held-out case数量。
- seed数量。
- 每类subset重复数R。
- 最小可检测效应MDE。
- 最终统计模型和排除规则。

确认性实验不得因中间曲线符合预期而提前停止。只允许因实现错误、数据损坏或
预注册资源上限停止。

## 12. 推荐执行顺序

1. Phase 0：分类和hook审计。
2. Pilot-A：test_5，k=8，R=5，2 seeds，选择两个去噪区间。
3. Pilot-B：增加k={1,2,4,8}，估计剂量曲线与subset方差。
4. Pilot-C：记录并匹配projected perturbation energy。
5. 冻结确认性case、sample size和全部subset。
6. 在held-out cases上运行确认性实验。
7. 统一计算轨迹、Physics-IQ、WMReward和secondary指标。
8. 执行层级分析与cluster bootstrap。
9. 单独报告“整类总效应”和“固定剂量类别效应”。

## 13. 输出与复现要求

代码与轻量配置放在：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/
```

大规模结果放在：

```text
/data/gaoya/agent-data/outputs/wan_dit_head_role_dose_control/
```

必须输出：

```text
configs/
  frozen_head_report.sha256
  heldout_cases.txt
  matched_subsets.json
  experiment_config.json

generation/
metrics/
motion_features/
head_classification/
  raw_head_features.parquet
  raw_feature_schema.json
  derived_head_ranks.parquet
  derived_head_scores.parquet
  classification_manifest.json
analysis/
  per_video.csv
  matched_contrasts.csv
  dose_response.csv
  mixed_effects_summary.md
  bootstrap_intervals.csv
  decision_report.md
```

每个结果JSON至少记录：

- Git commit或代码hash。
- 原始分类特征、派生 rank、派生 score 文件的路径与 SHA256。
- score 公式版本、具体权重和 rank 规则。
- source、seed、model、stage。
- role、k、subset id、head列表。
- block直方图。
- 分类score/margin/support。
- relative perturbation energy。
- baseline路径。
- 完整推理配置。

执行GPU使用 `0,1,2,3,5,6,7`，不使用GPU4。

## 14. 最终报告必须回答

1. S-all影响大是否能由159个head的剂量解释。
2. 固定k后，S/T/C哪一类的单位干预影响更大。
3. 控制projected energy后，类别差异是否仍存在。
4. 类别效应是否随model和去噪阶段改变。
5. all-category结果是否属于小规模效应的线性累积。
6. 哪些结论能在held-out cases复现。
7. 哪些Physics指标给出一致判断，哪些存在歧义。

只有完成数量匹配、能量匹配和held-out确认性实验后，才允许将结果解释为
“head类别的功能差异”；否则只能解释为“删除该类别全部head后的系统总效应”。
