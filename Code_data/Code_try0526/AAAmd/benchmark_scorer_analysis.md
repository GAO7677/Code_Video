# Benchmark 与 Scorer 简析

本文只基于当前仓库代码和已经生成的结果文件。能形成稳定结论的主体是 `PDI-Bench`、`Dataset_physV(ABC)`、`Physics-IQ` 和 `PhyGenBench`。核心结论很直接：当前没有任何单一指标可以同时覆盖几何一致性、时序一致性、物理合理性和外观鲁棒性。

## Benchmark

| Benchmark | 当前角色 | 关键特征 |
|---|---|---|
| A / PDI-Bench | 真实视频与生成视频对比集 | 15 个 case，4 个方法，含遮挡、跟踪、镜头变化和自然运动，适合看整体一致性，但不等于纯物理 benchmark。 |
| B1 / Ball-Block Physics | 受控刚体碰撞 | 8 个 case，固定外观与相机，只改恢复系数 `e`、摩擦 `mu` 和球质量 `m`。 |
| B2 / JEPA Sensitivity | 受控运动扫描 | 20 个 case，固定外观，只改速度、质量、重力、木块质量，以及 `nomiss` / 反向碰撞这类极端设置。 |
| B3 / Appearance Sensitivity | 外观扰动检查 | 24 个 case，物理轨迹沿用 B1，只改渲染版本、颜色、背景和光照，用来测指标是否把“视觉风格”误当成“物理差异”。 |
| C / Shuffle sanity | 时序打乱检查 | 20 个配对样本，内容不变，只打乱帧顺序，用来检查指标是否真的依赖时间结构。 |
| D / Physics-IQ | 更广的物理现象集 | 66 个 case，覆盖刚体、流体、光学、热学、磁学等。当前 ctx08 只有 23/66 完整覆盖，其他方法是 66/66。 |
| E / PhyGenBench | 开放式 T2V 物理生成集 | 160 个 case。当前官方汇总里只保留 `wan22-5B-TI2V` 和 `VACE_1p3B_TI2V`，`FLUX_1_Kontext` 只是首帧，不参与方法汇总。 |

`B1/B2/B3/C` 都属于自建 `Dataset_physV`。这四组的价值不是“谁分数高谁就更物理”，而是分别把物理参数、运动参数、外观参数和时间结构拆开，逼着指标暴露偏好。

## Scorer

| Scorer | 代码口径 | 方向 | 特性 |
|---|---|---|---|
| Official PDI | `metric_results.official_pdi` | 越低越好 | 几何审计分数，主要依赖分割、跟踪、深度和投影关系。当前结果里 `traj_component` 贡献很小，`scale / rigidity / vp` 更像主驱动。它不是纯物理分数，更像“几何前端稳定性 + 局部结构一致性”。 |
| WMReward Surprise | `metric_results.wmreward_jepa.surprise` | 越低越好 | 对齐官方 `compute_wmreward.py` 默认口径，本质是 V-JEPA 的滑窗未来预测误差。它能看出时间预测难度，但区分度普遍偏弱。 |
| V-JEPA Proxy | `RelRaw / DeltaRel / DeltaProf` | 越低越好 | 项目内自定义诊断，不是官方 benchmark。当前更应该拆开看三项误差，而不是看任何一个合成总分。 |
| Cosmos Reason1 | `metric_results.cosmos_reason1.score` | 越高越好 | Cosmos cookbook 里的 LLM judge，按固定 prompt 输出 1 到 5 分。可解释性强，但离散、粗粒度，也会受视觉呈现影响。 |
| VideoPhy-2 | `SA / PC / Joint` | 越高越好 | `SA` 是 caption match，`PC` 是物理 commonsense，`Joint = 1[SA>=4 且 PC>=4]`。它很适合看明显违和，但不适合细粒度排序。 |

从实现上看，VideoPhy-2 的 `SA` 和 `PC` 都是 1 到 5 的离散 judge；Cosmos Reason1 也是 1 到 5；WMReward 和 V-JEPA Proxy 才更接近连续预测误差。也就是说，前两者更像“粗粒度裁判”，后两者更像“预测一致性诊断”。

## Analysis

### A / PDI-Bench

A 组里 GT 在 `PDI`、`Cosmos Reason1` 和 `VideoPhy-2` 上都最强，说明真实视频在这些指标看来更接近“自然物理+自然外观”的分布。`WMReward Surprise` 在这组里几乎打平，方法间差异很小，因此它不适合拿来做 A 组主排序。

更具体地说，A 组的 `PDI` 是清楚拉开方法的：GT `0.144`，VACE `0.466`，ctx08 `0.569`，wan `0.878`。但 `WMReward` 的方法差异很小，说明它更像短窗可预测性，而不是强物理排序器。

### B1 / B2 / B3

B1 和 B2 是最像“物理实验”的两组，因为它们把单一变量拆得很干净。这里的结果说明三件事。

第一，`Official PDI` 有区分度。B1 里 GT `0.384`，wan `3.690`，VACE TI2V `0.162`，ctx08 `2.547`；B2 里 GT `1.307`，wan `2.582`，VACE TI2V `1.025`，ctx08 `3.285`。也就是说，它能抓住一部分物理/几何差异。

第二，`Cosmos Reason1` 和 `VideoPhy-2` 更偏向“明显正确的物理常识”。B1 里 GT 的 `Cosmos=4.25`、`Joint=1.0`，而三个生成方法的 `Joint` 都是 0。B2 里 GT 的 `Joint=0.95`，其余方法也基本是 0。它们对“明显像真的”很敏感，但分数离散，细分能力有限。

第三，`WMReward Surprise` 在 B1/B2 仍然比较压缩。它能给出方向，但很难单独支撑方法排序。换句话说，B1/B2 里它更像辅助量，不像主裁判。

B3 是最关键的 sanity check。它只改渲染，不改物理，所以一个理想的物理指标不应该把颜色、亮度、背景当成主要信号。当前结果说明，所有 scorer 都不同程度地受外观影响，因此 B3 不能缺席，缺了就很容易把“风格差异”误判成“物理差异”。

### C / Shuffle sanity

C 组是时间结构检查。按项目里的 20 个配对样本重算后，`Official PDI / WMReward / Cosmos / VideoPhy-2 / V-JEPA` 的平均变化如下：

- `Official PDI` `0.6735 -> 0.8369`，`+0.1634`
- `WMReward Surprise` `0.4160 -> 0.4328`，`+0.0168`
- `Cosmos Reason1` `3.8000 -> 3.6000`，`-0.2000`
- `V-JEPA RelRaw` `0.0374 -> 0.0403`，`+0.0029`
- `V-JEPA DeltaRel` `0.3999 -> 0.4384`，`+0.0385`
- `V-JEPA DeltaProf` `0.2853 -> 0.3105`，`+0.0252`
- `VideoPhy-2 SA` `3.8667 -> 3.8667`，`0`
- `VideoPhy-2 PC` `4.0667 -> 3.8000`，`-0.2667`
- `VideoPhy-2 Joint` `0.4667 -> 0.4667`，`0`

这里最重要的不是某个单项数值，而是结论一致性。`Official PDI`、`WMReward` 和 `Cosmos` 都只表现出小幅变化，说明它们对“打乱时间顺序”有反应，但反应并不强。`VideoPhy-2` 里只有 `PC` 明显下降，`SA` 和 `Joint` 基本不动，说明它更像在看物理常识违和，而不是严格的时序结构。`V-JEPA Proxy` 的三个子项方向也不完全一致，说明它更适合作为诊断量，而不是稳定的时序打乱检测器。

### 指标敏感性

| 指标 | 对 B 组扰动的表现 | 对 C 组打乱的表现 | 结论 |
|---|---|---|---|
| Official PDI | B1/B2 能拉开方法，但 B3 也会明显波动，说明它不只看物理。比如 B3 里 GT `0.3639`，VACE TI2V `1.0315`，wan `2.9830`。 | `0.6735 -> 0.8369`，有变化但不剧烈。 | 对几何/外观都敏感，不是纯物理分数。 |
| WMReward Surprise | GT 一般低于生成方法，但 B1/B2/B3 的差距都不大。 | `+0.0168`，变化最小之一。 | 更像短窗可预测性，时序打乱敏感性偏弱。 |
| Cosmos Reason1 | B1/B2/B3 都能把 GT 和生成方法拉开，但分数是离散的，且会受渲染影响。 | `-0.2000`，只轻微下降。 | 适合看“明显像不像”，不适合细粒度排序。 |
| V-JEPA RelRaw | 能区分部分方法，但 B3 也会受外观影响。 | `+0.0029`，变化很小。 | 有时序信号，但强度不够。 |
| V-JEPA DeltaRel / DeltaProf | 在 B 组和 C 组都在变，但不同子项不总是同向。 | `+0.0385 / +0.0252`。 | 更适合做诊断，不适合直接当主分数。 |
| VideoPhy-2 SA / PC / Joint | B 组里 GT 通常最好，B3 也会受外观影响。 | 只有 `PC` 明显下降，`SA` 和 `Joint` 基本不变。 | `PC` 比 `SA` 更敏感于时序破坏，`Joint` 太粗。 |

### D / Physics-IQ

D 组是目前最能暴露 scorer 冲突的 benchmark。它覆盖面更广，但也正因为更广，单一 scorer 更难同时兼顾所有现象。

在当前完整覆盖的 66 个样本上，GT、wan 和 VACE TI2V 都有完整结果，ctx08 只有 23 个样本。方法排序在不同 scorer 下明显不一致：`PDI` 偏向 ctx08 `0.356`，`Cosmos` 偏向 VACE TI2V `3.182`，`VideoPhy-2` 对 GT 更友好，`WMReward` 则几乎把几种方法压得很近。这个组最能说明：不同 scorer 看的是不同方面，方法排名不是唯一的。

### E / PhyGenBench

E 组更开放，prompt 更杂，生成链路也更像真实 T2V 使用场景。这里 `PDI` 反而偏向 wan：wan `0.465`，VACE TI2V `0.603`。但 `WMReward`、`Cosmos` 和 `VideoPhy-2` 更偏向 VACE TI2V：`WMReward` `0.418 < 0.426`，`Cosmos` `3.375 > 3.156`，`VideoPhy-2 Joint` `0.450 > 0.369`。

这说明 E 组里没有统一赢家。你想看几何一致性，wan 更好；你想看 judge-style 的物理合理性和 caption 对齐，VACE TI2V 更好。这个分歧本身就是结论，不是噪声。

## 结论

- `Official PDI` 适合看几何一致性，但不能单独等于物理正确性。
- `WMReward Surprise` 适合看短窗预测难度，区分度偏弱。
- `V-JEPA Proxy` 适合做诊断，不适合当主总分。
- `Cosmos Reason1` 和 `VideoPhy-2` 适合抓明显违和，但粒度粗，容易受风格影响。
- `B3` 和 `C` 是必需的 sanity check。没有它们，就无法判断一个 scorer 到底是在看物理，还是在看外观/顺序。
- 如果要做方法排名，至少要把 `PDI`、`WMReward/V-JEPA`、`Cosmos/VideoPhy` 三类信号一起读，不能只报一个分数。
