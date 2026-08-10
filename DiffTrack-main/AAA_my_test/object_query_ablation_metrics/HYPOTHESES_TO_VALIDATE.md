# Object Query 消融：待验证结论

本文只登记**尚未被充分验证的假设**。观察到的单 case 现象不能直接写成模型机制结论；每条假设都必须给出精确定义、当前证据、反例、混杂因素和后续验收标准。

## HYP-001：Past ≈ All-time

**状态：待验证。** 当前只有 `0613pybullet_sample_001460_w002 / seed=47326` 的系统比较；结果支持“固定 F04 的渲染 overlay 看起来相似”，但尚不支持“底层 attention 或生成运动普遍最相似”。

### 1. 精确定义

在 case、seed、Object target、M1/M2/M3、head scope 和其他推理配置完全相同的条件下，以 All-time 为 reference，分别比较 Past、Same 和 Future：

| 时间范围 | 实现后缀 | 删除的 Q→K 项 | 信息方向 |
|---|---|---|---|
| All-time | `*_only` | 不限制 `tq/tk` | 删除该 M 算子的全部时间组合 |
| Past | `*_past` | `tk > tq` | 未来 K/V → 更早 Query；反时间方向控制 |
| Same | `*_same` | `tk = tq` | 同一 latent 时刻 |
| Future | `*_future` | `tk < tq` | 历史 K/V → 更晚 Query |

待验证命题为：

> 对相同 Object target，`distance(All-time, Past)` 是否系统性小于 `distance(All-time, Same)` 和 `distance(All-time, Future)`？

该命题必须拆成三个层次，不能互相替代：

1. **HYP-001A · 生成运动：** Past 视频的目标对象轨迹和速度最接近 All-time。
2. **HYP-001B · Raw attention：** Past 的 `before / effective_after / removed` 原始 attention 数组最接近 All-time。
3. **HYP-001C · 渲染 overlay：** Past 的三行 JPEG overlay 在视觉上最接近 All-time。

### 2. 当前单样本证据

当前比较单位为同一 `Object target × M1/M2/M3 × head scope`，候选为 Past、Same、Future。运动主指标使用 All-time 与候选视频之间的 CoTracker target Center-ADE；全帧 SSIM 只用于外观审计，不代替运动指标。

#### Top100：Object A/B × M1/M2/M3，共 6 组

| 比较量 | Past 最接近的组数 | Same | Future | 当前解读 |
|---|---:|---:|---:|---|
| Target Center-ADE | **4/6** | 0/6 | 2/6 | 有局部 Past 趋势，但 Object A 部分组存在严重 track loss |
| Target Velocity Error | **3/6** | 2/6 | 1/6 | Past 没有取得多数优势 |
| Target Point-ADE | 2/6 | 0/6 | **4/6** | 不支持 Past 普遍最接近 |
| Full-frame SSIM | **3/6** | 1/6 | 2/6 | 只能说明像素/外观接近，不能说明运动接近 |

#### Top100 + Bottom100 + All720：共 18 组

| 比较量 | Past 最接近的组数 | Same | Future | 当前解读 |
|---|---:|---:|---:|---|
| Target Center-ADE | 6/18 | 6/18 | 6/18 | 三者持平，不支持普遍 Past ≈ All-time |
| Target Velocity Error | 7/18 | 7/18 | 4/18 | Past 与 Same 持平 |
| Target Point-ADE | 5/18 | 7/18 | 6/18 | Same 略多，但差异不足以形成结论 |
| Full-frame SSIM | 4/18 | 8/18 | 6/18 | Same 最多；静态背景会稀释对象差异 |

#### Top100 attention overlay：6 组，每组观察直接受算子作用的 Query 区域

| Raw attention 比较量 | Past 最接近 | Same | Future | 当前解读 |
|---|---:|---:|---:|---|
| `before` cosine | 2/6 | 0/6 | **4/6** | 上游响应并非普遍 Past 最接近 |
| `effective_after` cosine | 1/6 | 2/6 | **3/6** | 不支持 Past 普遍最接近 |
| `removed` cosine | 2/6 | **3/6** | 1/6 | M1 中 Same-frame 集中响应可能占主导 |
| `removed` relative-L1 | **4/6** | 2/6 | 0/6 | 从绝对质量差看存在 Past 趋势，但与 cosine 结论不一致 |

最终渲染的 JPEG 三行 overlay 中，Past 对 All-time 的 SSIM 在 `6 组 × 3 行 = 18/18` 比较里最高。这个结果只支持 **HYP-001C 的视觉现象**，不能证明 HYP-001A/B。

### 3. 为什么固定 F04 overlay 容易出现 Past ≈ All-time

当前 overlay 固定 Query 为 `F04 / latent tq=1`，共有 13 个 latent key 时刻：

- Past：`tk>1`，覆盖 `K02–K12`，即 11 个 key blocks；
- Same：只覆盖 `K01`；
- Future：只覆盖 `K00`；
- All-time：覆盖 `K00–K12`。

因此 Past 与 All-time 在掩码支持集合上共享 `11/13` 个 key blocks，渲染图天然容易相似。但这不是完整生成过程的一般证明：消融实际作用于所有 Query 时刻，在全部 `tq` 上汇总后 Past/Future 的 pair 数量对称；同时 Same-frame attention 可能高度集中，attention mass 也不与 block 数量成正比。

JPEG overlay 还包含三个额外混杂因素：

1. 大量相同或近似的视频底图像素；
2. 每个实验独立按帧做 P99.5 色标归一化，绝对响应强度差异会被压缩；
3. 相同标题布局、Query 面板和边框。

因此后续验证必须以 raw arrays 为主，JPEG SSIM 只能作为“为什么肉眼看起来像”的辅助量。

### 4. 后续验证协议

#### 实验单位与配对

- 基本单位：`case × seed × Object target × M ID × head scope`。
- 每个单位必须同时存在 All-time、Past、Same、Future，且使用相同 seed、checkpoint、prompt/source、推理步数和 CFG 配置。
- Object A、Object B 分层报告；M1、M2、M3 分层报告；Top100、Bottom100、All720 分层报告，不能只给总体平均。

#### 主指标

| 层次 | 主指标 | 最接近的定义 | 必要门控 |
|---|---|---|---|
| 生成运动 | Pairwise target Center-ADE | 越小越接近 All-time | `common_center_coverage ≥0.8`；失败组不参与 ADE 胜负 |
| 生成运动 | Pairwise Velocity Vector Error | 越小越接近 All-time | 报告有效速度帧数 |
| 最终状态 | Pairwise Center-FDE | 越小越接近 All-time | 报告最后共同有效帧，防止提前失踪造成假 FDE |
| Attention pattern | Raw-map cosine | 越大越接近 All-time | 分别报告三行，不做 JPEG 归一化 |
| Attention mass | Raw-map relative-L1 | 越小越接近 All-time | 分别报告三行及逐 key-time 质量 |
| 视觉辅助 | Rendered-overlay SSIM | 越大越相似 | 仅作渲染视觉诊断，不作为机制证据 |

#### Query-time 去偏

- 不能只使用 `tq=1` 的固定 F04 Query 得出时间方向结论。
- 至少对全部 13 个 latent Query 时刻分别提取 raw maps，再对 `tq` 做宏平均。
- 同时单独报告中间时刻（例如 `tq=6`），避免 F04 位于序列前端导致 Past 支持集合远大于 Future。

#### 统计判据

- 每组分别记录 Past、Same、Future 中谁最接近 All-time；在预设容差内记为 tie，不强行排序。
- 以 `case × seed` 为 cluster 做 bootstrap 置信区间，避免把同视频下多个 M/head scope 当作独立样本。
- 三候选的无偏机会基线为 `1/3`。只有 Past 胜率的 cluster-bootstrap 95% CI 下界高于 `1/3`，才能写成“存在系统性 Past 趋势”。
- 若要写成“Past 通常最接近 All-time”，还要求 Past 非 tie 胜率超过 `50%`，并在 Object A/B、M1/M2/M3 的主要分层中方向一致。

### 5. 当前结论边界

目前允许的表述：

> 在 `0613pybullet_sample_001460_w002 / seed=47326` 的固定 F04 Top100 可视化中，Past 与 All-time 的**渲染 overlay**最相似；Top100 目标中心轨迹也出现部分 Past 趋势。

目前不允许的表述：

> Past attention 或 Past 消融视频在一般情况下必然最接近 All-time。

跨 case、跨 seed、全 Query-time 的 raw attention 与质量门控轨迹结果完成前，HYP-001 保持“待验证”。
