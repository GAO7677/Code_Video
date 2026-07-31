# Wan S-Head 跨模型消融分析

## 1. 分析范围

本文比较 Wan+LoRA、Wan+xSSC 与 Wan+OpenVid LoRA(step-10000)。公平对照部分固定为 seed 851、相同 20 个 source cases、相同 baseline 配对、相同 head 集合和相同去噪区间。视频为 49 帧。

OpenVid 使用此前冻结的公共 S-head 列表，没有针对 OpenVid 重新分类。因此本文能比较相同干预位置的响应，不能判断这些位置是否也是 OpenVid 自身最稳定的 S heads。

### 证据标签

| 标签 | 含义 |
|---|---|
| G3-D | 三个受测模型均有直接配对指标支持；若涉及区间，三个模型的 95% CI 均满足所述方向。只表示当前三模型内复现，不代表外部模型普适性。 |
| G3-R | 三个模型出现同方向均值或排序，但模型间差值本身未完成显著性检验，或至少一个模型的区间跨 0。 |
| 模型内 | 当前模型/配置上的描述性结果，不声称跨模型成立。 |
| I | 从现象推导的机制解释或后续假设，不是指标直接证明的事实。 |

## 2. 核心结论

| 证据 | 结论 | 支撑与边界 |
|---|---|---|
| G3-D | 全程消融 Local-dominant all 后，三模型 PMF 均下降 | Wan+LoRA -1.610 [-1.924, -1.291]；Wan+xSSC -1.701 [-2.100, -1.310]；Wan+OpenVid LoRA -1.547 [-1.861, -1.226]。三组按 case bootstrap 95% CI 均低于 0。该结论只覆盖当前三模型、seed 851 和 20 个 case，不能外推到其他模型或数据分布。 |
| G3-R | 固定为 32 heads 时，0–10 阶段的 Same-frame 平均 Impact 均高于 Local | Wan+LoRA Same=0.524, Local=0.360；Wan+xSSC Same=0.424, Local=0.381；Wan+OpenVid LoRA Same=0.624, Local=0.495。这是三模型同方向均值，不是 Same−Local 差值的显著性检验。 |
| G3-R | Local-dominant Late × 0–10 在三模型中均出现负向 GT gain 和 PMF 均值 | Wan+LoRA GT gain=-0.068, PMF Δ=-0.386；Wan+xSSC GT gain=-0.043, PMF Δ=-0.131；Wan+OpenVid LoRA GT gain=-0.050, PMF Δ=-0.363。其中 xSSC 的 PMF 和部分 GT gain 区间跨 0，因此只能称为重复方向，不能称为三模型均已显著。 |
| 模型内 | 固定子类别实验的平均运动敏感度存在模型差异 | Wan+LoRA=0.402；Wan+xSSC=0.381；Wan+OpenVid LoRA=0.509。OpenVid 均值最高、xSSC 最低是当前配置上的描述性排序；训练权重、条件分支和 head 分类来源均有差异，不能据此归因于某个模块。 |
| 模型内 | Local+Same union 的结果不呈统一交互方向 | Union−max(single)：Wan+LoRA: [00,10) +0.013, [10,20) +0.015, [00,40) +0.009；Wan+xSSC: [00,10) +0.058, [10,20) -0.014, [00,40) +0.058；Wan+OpenVid LoRA: [00,10) -0.042, [10,20) -0.019, [00,40) -0.012。该量同时改变 head 数，且网络响应非线性，只能描述联合消融结果，不能解释为严格的协同或抵消因果效应。 |
| I | “Same-frame 更敏感、Local 更支撑物理连续性”是机制假设，不是已证实功能标签 | 这一解释来自 Motion Impact 与 PMF 的组合模式。要验证功能分工，仍需等 head 数、等输出能量、更多 seeds、held-out cases，以及单 head 或小 k 干预。 |

## 3. 等量 32-head 公平对照

`Motion Impact` 只表示相对同 case baseline 的运动变化大小；`GT gain > 0` 表示轨迹指标更接近 GT，不等同于整体物理质量提高。

| 模型 | 阶段 | Local Impact | Same Impact | Union Impact | Local GT gain | Same GT gain | Union GT gain |
|---|---:|---:|---:|---:|---:|---:|---:|
| Wan+LoRA | 0-10 | 0.360 | 0.524 | 0.537 | -0.001 | +0.038 | +0.049 |
| Wan+LoRA | 10-20 | 0.270 | 0.269 | 0.285 | +0.016 | +0.024 | +0.048 |
| Wan+LoRA | 0-40 | 0.398 | 0.482 | 0.491 | +0.011 | +0.040 | +0.057 |
| Wan+xSSC | 0-10 | 0.381 | 0.424 | 0.482 | +0.004 | +0.023 | +0.008 |
| Wan+xSSC | 10-20 | 0.241 | 0.261 | 0.247 | -0.013 | +0.010 | +0.009 |
| Wan+xSSC | 0-40 | 0.432 | 0.452 | 0.510 | +0.000 | +0.036 | -0.002 |
| Wan+OpenVid LoRA | 0-10 | 0.495 | 0.624 | 0.582 | +0.020 | +0.043 | +0.050 |
| Wan+OpenVid LoRA | 10-20 | 0.382 | 0.468 | 0.449 | +0.003 | +0.007 | +0.032 |
| Wan+OpenVid LoRA | 0-40 | 0.531 | 0.531 | 0.520 | -0.005 | +0.097 | +0.120 |

## 4. 主导类别全程消融

Local-dominant all 为 100 heads，Same-frame-dominant all 为 59 heads。`Impact/head` 只是总 Impact 除以 head 数的近似归一化，不是可加的单-head 因果效应。

| 模型 | Impact/head Local / Same | GT gain Local / Same | PMF Δ Local [95% CI] | PMF Δ Same [95% CI] | Physics-IQ Δ Local / Same |
|---|---:|---:|---:|---:|---:|
| Wan+LoRA | 0.00505 / 0.00760 | +0.033 / +0.094 | -1.610 [-1.924, -1.291] | -0.147 [-0.318, +0.027] | -3.119 / +9.869 |
| Wan+xSSC | 0.00471 / 0.00794 | +0.038 / +0.073 | -1.701 [-2.100, -1.310] | -0.182 [-0.417, +0.056] | -9.173 / +0.275 |
| Wan+OpenVid LoRA | 0.00558 / 0.00844 | +0.099 / +0.142 | -1.547 [-1.861, -1.226] | -0.063 [-0.199, +0.062] | +5.543 / +19.965 |

Physics-IQ 与 PMF 在多组消融中方向不同。这里保留原始变化，不把二者合成为“物理正确性”总分，也不把 Physics-IQ 上升单独解释为质量提高。

## 5. 深度相关现象

| 模型 | 0-10 单位 head Impact 最大子集 | Impact/head | Local-Late 0-10 GT gain [95% CI] | PMF Δ [95% CI] |
|---|---|---:|---:|---:|
| Wan+LoRA | Local dominant / B10-19 | 0.02770 | -0.068 [-0.126, -0.008] | -0.386 [-0.700, -0.089] |
| Wan+xSSC | Same-frame dominant / B10-19 | 0.02698 | -0.043 [-0.101, +0.008] | -0.131 [-0.347, +0.074] |
| Wan+OpenVid LoRA | Same-frame dominant / B10-19 | 0.03581 | -0.050 [-0.111, +0.005] | -0.363 [-0.527, -0.210] |

## 6. 已有分析的整合与修正

- 保留：Motion Impact 与质量方向必须分开；Physics-IQ 与 PMF 存在方向冲突；不同 head 数必须同时报告总 Impact 和近似 Impact/head。
- 收紧：旧页面中“Middle 最重要”“Local-Late 负责正确运动”等表述，改为模型内最高均值或机制假设。现有指标只能说明干预响应，不能直接确定功能。
- 收紧：OpenVid 的平均 Impact 较高只作为当前配置排序；由于 OpenVid 没有独立重做 head 分类，不能据此断言其更依赖 S-head。
- 不合并：k=5/k=8 剂量实验、全部 S 多 seed 实验和本报告的 seed851 32/59/100-head 实验回答的问题不同，页面继续保留原表，但不混成一个效应量。

## 7. 局限与下一步验证

1. 当前三模型公平比较只有一个 seed 和 20 个 case；case 类型也不是独立总体抽样。
2. OpenVid 缺少本批 VideoPhy2/Cosmos 完整分数，跨模型结论主要依赖 Motion、GT、PMF 与 Physics-IQ。
3. Union 同时增加 head 数，不能作为严格的交互项；应补等数量、等 block、等输出能量设计。
4. Impact/head 是近似剂量归一化。网络存在非线性，不能由 all-head 结果反推单 head 效应。
5. 若要声称机制或外部模型泛化，需要更多 seeds、held-out cases、OpenVid 独立分类和小 k 干预。

## 8. 数据来源

- Motion 汇总：`/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/multiseed/motion-n-analysis/partial/aggregate_metrics.csv`
- 联合消融诊断：`/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/multiseed/motion-n-analysis/partial/interaction_diagnostics.csv`
- 配对 benchmark 原始记录：`/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/head-role-dose-control-pilot/manifest.json`
- 全部 S 多 seed benchmark：`/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/multiseed/benchmark-metrics/paired_vs_baseline_summary.csv`
- 数量控制结果：`/data/gaoya/agent-data/outputs/wan_dit_fulltoken_moving_pilot/gallery/head-role-dose-control-pilot/metrics/partial_aggregate.csv`
