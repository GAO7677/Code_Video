# Training-Free M1 Control：严格实验计划

## 0. 文档状态与结论边界

- 状态：**执行前计划**；本文件不代表实验已经完成。
- 研究对象：Wan2.2-TI2V-5B Legacy self-attention 中，`latest3350` PCK 排名选出的 heads。
- 主目标：判断 Top100 heads 的 M1（`R K/V → R Query`）contribution 是否能成为稳定、可控的 inference-time signal，并区分其主要影响对象生存/身份、轨迹/速度，还是两者。
- `R`：由同 case、同 seed、无干预 Baseline 上冻结的 CoTracker 点轨迹映射得到的 13-frame sparse latent object-token tube。
- `C`：同一 latent-video `T×H×W` 网格中不属于 `R` 的 token。
- 当前干预只支持“对 frozen sparse tube 的控制”这一结论，不能直接外推为完整 object mask 的全部信息流。
- 所有输出变化首先是 **vs 同 seed Baseline 的干预效应**；只有使用合格 simulator GT 的指标才能讨论物理正确性改善或恶化。

相关依据：

- Stage 3 结果：`/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/stage3_final_analysis/STAGE3_FINAL_REPORT.md`
- 总实验计划：`../plan.md`
- 信息流实现：`../../run_legacy_ti2v_temporal_object_tube_ablations.py`
- 当前 contrast runner：`../run_top100_m1_perturbed_attention_guidance.py`
- 指标索引：`../METRICS_IMPLEMENTATION_INDEX.md`

---

## 1. 为什么做，以及当前证据还缺什么

Stage 3 的主要发现是：Top100-M1 knockout 比 Bottom100-M1 更容易破坏对象生存和身份，并且 Top100 下 M1 比 M2 更容易改变中心轨迹。然而 Top100-M1 的 removed AV norm 同时远高于 Bottom100/Random100，因此现有证据不能区分：

1. Top100 只是承载了更大的 `R→R` contribution；
2. 相同 contribution 强度下，Top100 的方向本身更有效；
3. 增强该 contribution 是否能改善生成，而不只是 knockout 会破坏生成。

本计划分别用 Soft Scaling、Raw Contrast Guidance 和 Pairwise Norm-matched Guidance 回答这三个问题。

### 1.1 预注册假设

| ID | If | Then | Because | 反证/证据不足条件 |
|---|---|---|---|---|
| H-TF1 | M1 contribution 对对象状态是必要且具有连续剂量响应 | 在 `α∈{-1,-0.5,0}` 上，削弱越强，Disappearance、身份漂移和轨迹偏差总体越强 | 同一 `A[R,R]V_R` 被连续缩放，而非更换信息流 | 变化不随削弱程度排序，或只由少数 case/seed 驱动 |
| H-TF2 | 正向增强可用作控制 | `α>0` 或 `λ>0` 相比 Baseline 至少在一个预注册 endpoint 上产生稳定方向，并且 guardrail 不恶化超过阈值 | 模型预测被推离 M1 knockout 方向 | 只产生全局画质崩坏、对象 tube 脱离，或 case 方向不一致 |
| H-TF3 | Top100 具有 head-direction specificity | pairwise norm-matched 后，Top100 方向相对 Bottom100 和多数 layer-matched Random100 仍更有效 | 作用不再仅由 raw norm 更大解释 | 优势在 norm-matched 后消失，或仅相对一个随机 draw 成立 |
| H-TF4 | M1 同时承载身份维持和运动连续性 | 生存/身份与轨迹 endpoint 都超过预设 MDE | R→R 同时影响对象表示和时序传播 | 只有一类 endpoint 超过 MDE；此时结论必须降级为相应单一功能 |
| H-TF5 | 时间方向存在功能分化 | held-out 中 M1-Future 更影响轨迹、M1-Same 更影响身份，且不是 dose 差异造成 | 历史→未来和同帧边承担不同作用 | pilot 中选择后无法在 held-out 重现，或 dose 支持不重叠 |

### 1.2 预设最小有意义效应（MDE）

以下阈值用于判断“作用是否具有实际意义”，不替代置信区间：

| Endpoint | 初始 MDE | 依据 |
|---|---:|---|
| Disappearance / Identity Failure | 5 percentage points | 小于 Stage 3 Top−Bottom 约 12 pp 的一半，要求可见且不只统计显著 |
| Center-ADE | `0.05 D0` | 约为首帧对象 bbox 对角线的 5% |
| Center-FDE | `0.10 D0` | 终态允许高于逐帧平均阈值 |
| Velocity Vector Error | `0.01 D0/frame` | 用于筛选可感知运动变化；正式确认前用 pilot 方差复核 |
| DINO cosine change | `0.02` | 连续身份/语义外观变化的初始工程阈值；需在 TF-1 后冻结 |

若 TF-1 显示指标噪声或天然波动大于上述 MDE，必须在读取 confirmatory 数据前修订并版本化，不能根据显著性反向选择阈值。

---

## 2. 冻结的数学定义

对某 layer/head：

\[
A_h=\operatorname{softmax}(Q_hK_h^\top/\sqrt d),\qquad
Y_h=A_hV_h.
\]

对象 Query 行可分解为：

\[
Y_{R,h}=M_{RR,h}+M_{RC,h},
\]

\[
M_{RR,h}=A_h[R,R]V_h[R],\qquad
M_{RC,h}=A_h[R,C]V_h[C].
\]

所有 M1 操作均在 softmax 后、self-attention output projection 前修改 `A·V` contribution：

- 不修改 Q/K/V projection；
- 不重算 softmax；
- 不重新归一化剩余 attention；
- 未选 heads、FFN、cross-attention 和 `C→C` 保持不变。

### 2.1 实验 A：M1 Soft Scaling

\[
Y_{R,h}^{\alpha}=Y_{R,h}+\alpha M_{RR,h}
=(1+\alpha)M_{RR,h}+M_{RC,h}.
\]

| `α` | 精确含义 |
|---:|---|
| `-1.0` | 完整删除 M1 contribution；必须与 Stage-3 M1 knockout 数值等价 |
| `-0.5` | M1 contribution 保留 50% |
| `0` | no-op Baseline；必须确定性等价 |
| `+0.5` | M1 contribution 变为 150% |
| `+1.0` | M1 contribution 变为 200% |

主版本采用 `cfg_branches=both`，即 conditional 和 unconditional 都使用相同 `α`，因为它最忠实复现 Stage 3 knockout。`conditional_only` 仅用于 TF-0 分支审计和 inference-control sensitivity，不与主结果合并。

### 2.2 实验 B：M1 Conditional Contrast Guidance

每个 denoising step：

\[
\epsilon_c=\operatorname{DiT}(x_t,c),\quad
\epsilon_{c,M1}=\operatorname{DiT}_{M1\ knockout}(x_t,c),\quad
\epsilon_u=\operatorname{DiT}(x_t,\varnothing),
\]

\[
g_{M1}=\epsilon_c-\epsilon_{c,M1},
\]

\[
\epsilon_{guided}
=\epsilon_u+s(\epsilon_c-\epsilon_u)+\lambda g_{M1},
\qquad s=5.
\]

设置：

\[
\lambda\in\{-1,-0.5,0,0.5,1\}.
\]

解释边界：

- `λ>0`：推离 conditional-M1 knockout；
- `λ<0`：向 conditional-M1 knockout 靠近；
- `λ=0`：复用同 seed Baseline，不额外生成；
- 当 CFG=`5` 时，`λ=-1` 只移动了 conditional-knockout CFG 距离的 `1/5`，**不等于完整 knockout**；
- 当前定义是 conditional-only contrast，不等同于 Stage 3 同时消融 conditional/unconditional 的完整 CFG 差分。

### 2.3 Unconditional 差分审计

在 TF-0/TF-1 记录：

\[
g_c=\epsilon_c-\epsilon_{c,M1},\qquad
g_u=\epsilon_u-\epsilon_{u,M1},
\]

\[
g_{full}=5g_c-4g_u.
\]

逐 step 报告：`RMS(g_c)`、`RMS(g_u)`、`cos(g_c,g_u)`、`cos(g_c,g_full)` 和 `RMS(g_u)/RMS(g_c)`。若 case-balanced 的 `RMS(g_u)/RMS(g_c)` 中位数超过 `0.25`，文档必须把当前实验明确称为 **conditional-M1 guidance**，并新增 full-CFG contrast 作为独立后续，不能把当前结果表述为完整 Stage-3 M1 control。

### 2.4 实验 C：Head-group controls

| Head group | 定义 | 用途 |
|---|---|---|
| Top100 | latest3350 rank 1–100 | 主信号 |
| Bottom100 | latest3350 最低 100 | 排名负对照 |
| Random100-draw0/1/2 | 与 Top100 per-layer histogram 完全一致，且与 Top/Bottom 不重叠 | 排除层分布与单次随机抽样 |

Raw 结果回答“真实可用控制信号谁更强”；pairwise norm-matched 结果回答“相同 prediction-space 强度下，Top100 的方向是否更有效”。两者不得混写。

### 2.5 Pairwise norm matching

只做以下冻结配对：

1. Top100 vs Bottom100；
2. Top100 vs Random100-draw0；
3. Top100 vs Random100-draw1；
4. Top100 vs Random100-draw2。

先在同 `case×seed×target` 的无 guidance baseline latent path 上校准。对每个 step 和一对 head groups `a,b`：

\[
r_{a,t}=\operatorname{RMS}(g_{a,t}),\quad
r_{b,t}=\operatorname{RMS}(g_{b,t}),\quad
r_t^*=\min(r_{a,t},r_{b,t}),
\]

\[
s_{a,t}=r_t^*/(r_{a,t}+\varepsilon),\qquad
s_{b,t}=r_t^*/(r_{b,t}+\varepsilon).
\]

生成时使用：

\[
\tilde g_{a,t}=s_{a,t}g_{a,t},\qquad
\tilde g_{b,t}=s_{b,t}g_{b,t}.
\]

该方法只缩小较强信号，不放大较弱信号。若任一组 `r/RMS(εc)<10^{-4}` 的 step 超过 10%，该 pair 标记为无共同有效 dose，不做 direction-specificity 宣称。必须同时报告 baseline-path 目标 norm 和实际 guided path 达成的 norm。

---

## 3. 实验矩阵与执行阶段

### 3.0 全阶段统一控制变量

除实验矩阵明确指定的因素外，每个配对单元必须固定：

| 项目 | 冻结值/规则 |
|---|---|
| Model | 同一个 Wan2.2-TI2V-5B Legacy checkpoint 与权重 hash |
| First frame / prompt | 同 case 内内容和 hash 完全一致 |
| Noise | 同 seed、同初始 latent；不得用不同 seed 作直接成对比较 |
| Sampling | 40 steps、CFG=5、UniPC、shift=5 |
| Output | 49 frames、1280×704、30 fps |
| Object tube | 同 case、同 seed 的无干预 Baseline 上冻结；13 latent anchors |
| Head ranking | `latest3350` 同一文件、SHA256 和 layer-head 列表 |
| Time scope | 默认 All-time；TF-5 才允许 Same/Future/Past |
| Intervention location | post-softmax `A·V`、self-attention output projection 之前 |
| Metrics | 同一版本代码、模型权重、mask/track/reference 和阈值 |

任何一项不一致都不得进入 paired table。

### TF-0：实现与代数硬门槛

使用一个 smoke 单元：

- case：`0613pybullet_sample_001460_w002`；
- seed：`47326`；
- target：`object_A`；
- heads：Top100；
- 40 denoising steps、CFG=5、49 frames、1280×704、UniPC、shift=5；
- frozen same-seed Baseline tube；
- 不使用 GPU4。

必须通过：

1. `α=0` 与同一代码、同一 GPU/runtime 现场生成的无 hook Baseline：attention output 必须 `torch.equal`，解码后的 RGB frames 必须 `np.array_equal`；历史归档 Baseline 仅报告环境漂移，不作失败条件；
2. `α=-1, branches=both` 与同一代码、同一 GPU/runtime 现场生成的 Stage-3 `self_only` reference：最终 decoded RGB MAE 不得超过 `1/255`；同时报告与历史 Stage-3 视频的漂移，但不把软件/runtime 漂移误判成公式失败；
3. `M_RR + M_RC = Y_R` 在 CPU FP32 reference 中以 `rtol=1e-5, atol=1e-6` 作为硬门控；生产 BF16 fused-attention 的三次独立 kernel launch 只记录逐元素 mismatch、全局 relative-L2、最大逐调用 relative-L2 和非有限值，逐元素 `rtol=1e-3, atol=1e-3` 不作为硬门控；
4. 只修改 manifest 中的 100 个 physical heads；
5. CPU FP32 contribution decomposition 通过 `rtol=1e-5, atol=1e-6`；实际生成覆盖恰好 `100 heads × 40 steps × 2 CFG branches = 8000` 个 soft-scaling head-events，BF16 residual 不得出现 NaN/Inf；
6. contrast guidance 每步恰好 clean conditional、M1 conditional、clean unconditional 三次 DiT forward；
7. runner 接受负 `λ`，但 `λ=0` 直接复用 Baseline；
8. DiffSynth wrapper 返回 `εc+(λ/CFG)g` 后，pipeline 最终结果严格等于目标公式；
9. frozen tube 包含 13 个 latent anchors，映射在全程不漂移；
10. manifest 记录代码 hash、ranking SHA256、head list、branch、α/λ、seed、scheduler、模型和 dose。

任一失败：停止后续 GPU 队列，只输出诊断。

### TF-1：范围与安全性 pilot

冻结范围：

- 3 cases：`001460`、`000331`、ball-and-block；
- seeds：`47326, 42`；
- target：每个 case 的 `object_A`；
- heads：Top100；
- Soft Scaling：5 个 `α`；
- Raw Contrast：5 个 `λ`。

唯一生成量：

- 6 个同 seed Baseline，可在 Soft Scaling 与 Contrast Guidance 间复用；
- Soft Scaling 非零 `α`：`3×2×4=24`；
- Contrast 非零 `λ`：`3×2×4=24`；
- 合计最多 54 个视频，其中当前已完成且 manifest 完全一致的正向 contrast 可经 inventory 复用。

TF-1 只用于：

- 检查负值是否数值稳定；
- 检查 `+1` 是否过冲；
- 冻结 tube-departure 阈值；
- 估计 case-level 方差、正式 MDE 和所需 case 数；
- 冻结 norm calibration 的最小有效 RMS。

3 cases × 2 seeds 不能用于总体显著性结论；seed 不是独立 case。

### TF-2：Soft Scaling + Head controls

在 TF-1 的相同 6 个 `case×seed×object_A` 单元上运行：

- head groups：Top100、Bottom100、Random100-draw0/1/2，共 5 组；
- `α∈{-1,-0.5,0,0.5,1}`；
- `α=0` 共用 Baseline；
- 非零新增上限：`6×5×4=120` 个视频。

Primary analysis：

1. weakening arm `α=-1,-0.5,0` 是否按预期排序；
2. positive arm `α=0,0.5,1` 是否改善、饱和或过冲；
3. 每个 head group 的 raw dose-response；
4. Top vs Bottom、Top vs 三个 Random 的 case-level paired contrast；
5. 所有 head comparison 同时报告 removed AV norm，不做简单 outcome/norm 除法。

### TF-3：Raw + Pairwise Norm-matched Contrast Guidance

Raw：

- 与 TF-2 相同的 6 个单元和 5 个 head groups；
- `λ∈{-1,-0.5,0,0.5,1}`；
- 非零新增上限：`6×5×4=120`。

Norm-matched：

- 只运行 `λ∈{-1,+1}`，避免无必要扩大矩阵；
- 四个冻结 pair；
- 每 pair 两个 matched head directions；
- 新增上限：`6×4 pairs×2 groups×2 λ=96`，若缩放系数恰为 1 且配置相同可复用 raw 结果。

判定 Top100 direction-specificity 至少要求：

1. Top−Bottom 的主 endpoint effect 超过 MDE，case-bootstrap CI 不跨 0；
2. Top 相对 3 个 Random draws 中至少 2 个方向一致且超过 MDE；
3. 合并 Random draw 的 hierarchical/cluster analysis 方向一致；
4. achieved guidance RMS 的共同支持充分；
5. 不是由 tube departure、track loss 或全局画质崩坏解释。

否则只能说 Top100 raw guidance 更强，不能说每单位方向更有效。

### TF-4：Held-out 扩展与统计确认

TF-1/2/3 都是 development pilot。正式确认必须：

1. 使用未参与 latest3350 排名构建、Stage 3 页面挑选和 TF pilot 的 cases；
2. seeds 与 pilot 分离；
3. case 是最高独立单位；一个 case 内先平均 seed/target，再对 cases 等权；
4. 使用 TF pilot 的 case-level paired variance，在不查看 confirmation outcome 的情况下计算 power；
5. power≥0.8；少于 8 个独立 cases 一律保持 exploratory；
6. 固定样本数与停止规则，不根据中间结果追加到显著。

确认集优先覆盖：单对象/多对象、交互/非交互、不同对象尺寸和不同基础运动幅度。所有可用 single-object targets 均纳入，不能只选择视觉效果明显的 `object_A`。

### TF-5：Stage 4 后的 temporal M1 control

只有 Stage 4 在 case-level dose audit 后发现可重复的方向差异才进入本阶段。候选为：

- M1-Same：`t_k=t_q`；
- M1-Future：`t_k<t_q`；
- M1-Past：`t_k>t_q`。

Stage 4 pilot 用来选择最多两个 temporal candidates；选择规则、head group、α/λ 和 endpoint 随后冻结。验证必须换 held-out cases/seeds，避免用同一批数据既选方向又证明方向。

预注册问题：

- Future：Target Center-ADE、signed trajectory response、Velocity Vector Error；
- Same：DINO identity、center-aligned LPIPS、Disappearance；
- Past：作为反时间方向对照，不解释成物理因果。

---

## 4. 指标、分类规则与质量门控

### 4.1 Primary endpoints

| 功能 | Primary endpoint | 计算 | 方向解释 |
|---|---|---|---|
| Survival | Disappearance rate | selected object 非存活帧比例，综合 mask、身份和面积失败 | 越大表示破坏越强；不是质量改善 |
| Identity | Continuous DINO distance | 中心对齐、mask pooling 后 `1−cosine` | 越大表示身份/语义外观变化更强 |
| Trajectory | Center-ADE / D0 | 与同 seed Baseline 中心逐帧距离均值除以 F00 bbox 对角线 | 越大表示轨迹干预更强 |
| Signed trajectory | Knockout-axis coefficient `β` | 候选位移在同单位 M1 knockout 位移方向上的投影 | `β≈-α` 表示 signed soft-scaling 响应符合线性预期 |

其中：

\[
\Delta p_t^\alpha=p_t^\alpha-p_t^{base},\qquad
\Delta p_t^{KO}=p_t^{\alpha=-1}-p_t^{base},
\]

\[
\beta_\alpha=
\frac{\sum_t \Delta p_t^\alpha\cdot\Delta p_t^{KO}}
{\sum_t\|\Delta p_t^{KO}\|^2+\varepsilon}.
\]

ADE 是无符号距离，不能单独验证正负 `α/λ` 的单调方向；必须与 `β` 联读。

### 4.2 Secondary endpoints

- Center-FDE / D0；
- Velocity speed/direction/vector error；
- PCK@5/10/20%D0、Point-ADE；
- center-aligned LPIPS、Shape IoU、area/aspect/circularity；
- Track Loss、Mask Absence、Identity Failure、Area Failure、terminal missing；
- Other-object Center-ADE、Outside-object LPIPS；
- 有资格 GT：GT-ADE change、GT velocity change、contact-time、post-contact velocity；
- VBench/global quality 仅作 sanity guardrail。

### 4.3 功能归类规则

对 held-out case-level effect 和 95% CI：

| 结果 | 允许结论 |
|---|---|
| Survival/Identity 超过 MDE，Trajectory 未超过 | 更偏对象身份/生存 |
| Trajectory 超过 MDE，Survival/Identity 未超过 | 更偏轨迹/速度 |
| 两类均超过 MDE | 两者同时影响 |
| CI 宽、case 方向不一致或门控失败高 | 证据不足，不强行分类 |

不同单位的指标数值不能直接比较大小后宣称“更偏哪类”。

### 4.4 Guardrails 与硬停止条件

每个结果同时报告：

1. 全部样本的 survival/failure；
2. 通过门控样本的轨迹；
3. 未通过样本及路径；
4. frozen tube coverage 和 departure onset；
5. outside-object、other-object 和全局质量。

停止/降级条件：

- 任一 no-op 或 knockout equivalence 测试失败；
- selected head、step、CFG branch 覆盖不完整；
- Baseline、candidate 的 seed/prompt/first frame/scheduler 不一致；
- `Q/K/V token count != T×H×W`；
- candidate 相对 frozen tube 的覆盖率低于 TF-1 冻结阈值，却仍继续解释后段轨迹；
- trajectory gate failure 相比 Baseline 增加超过 20 pp；此时轨迹均值降级，优先报告 failure；
- full-frame/global quality 出现系统性崩坏；
- norm matching 的有效 dose 无共同支持；
- 缺失值被填 0、失败样本被静默删除；
- 根据中期结果提前停止或追加样本到显著。

---

## 5. 统计分析

### 5.1 独立单位

- 最高独立单位：case；
- seed、object、α/λ、head group 均为 case 内配对重复测量；
- 先在 case 内平均 seed/target，再对 case 等权；
- 不能把视频数当样本量，增加 seed 不能替代增加 case。

### 5.2 Soft Scaling dose-response

分别分析：

1. weakening arm `[-1,-0.5,0]`；
2. enhancement arm `[0,0.5,1]`。

每个 endpoint 报告：

- case-level 曲线；
- 相邻剂量 paired difference；
- Spearman monotonic coefficient；
- 线性项和二次项的 case-cluster bootstrap CI；
- 正向过冲：`metric(+1)−metric(+0.5)`；
- signed trajectory 的 `β` 对理论 `−α` 的偏差。

不要求从 `-1` 到 `+1` 全区间线性。正向增强允许改善、饱和或过冲三种结论。

### 5.3 Contrast Guidance

Raw 与 norm-matched 分表报告。主 contrasts：

- `λ=+1 vs 0`；
- `λ=-1 vs 0`；
- signed dose-response slope；
- Top−Bottom；
- Top−Random draw0/1/2；
- norm-matched Top−control。

Primary endpoint family 使用 BH-FDR；secondary 只报告 effect、CI 和校正结果，不从单个未校正 `p<0.05` 下结论。倍数仅在分母远离 0 时报告，同时给绝对差。

### 5.4 Power 与停止规则

TF-1 结束后，用 case-level paired effect 的标准差计算：

\[
n\approx\left(\frac{z_{1-\alpha/2}+z_{1-\beta}}{\text{MDE}/\sigma_d}\right)^2.
\]

- `power≥0.8`；
- alpha 按 confirmatory primary family 校正后冻结；
- 输出 `case count × MDE × power` 曲线；
- 若可用 held-out cases 不足，明确停止为 exploratory。

---

## 6. 关键伪代码

### 6.1 Object tube 和 M1 contribution

```python
def frozen_object_tube(baseline_video, query_points):
    tracks = cotracker(baseline_video, query_points)  # [49, P, 2]
    anchors = [0, 4, 8, ..., 48]                     # 13 latent anchors
    return map_pixels_to_latent_tokens(tracks[anchors])


def m1_contribution(q, k, v, r_rows, selected_heads):
    # original_attn 内部计算 A=softmax(QK^T/sqrt(d))，不修改 logits。
    v_r = zeros_like(v)
    v_r[:, r_rows, selected_heads, :] = v[:, r_rows, selected_heads, :]
    contribution = original_attn(q, k, v_r)          # A @ V_R
    return contribution[:, r_rows, selected_heads, :] # A[R,R]V_R
```

### 6.2 Soft Scaling hook

```python
def soft_scale_attention(q, k, v, block, alpha, branch):
    y = original_attn(q, k, v)
    if block not in selected_blocks:
        return y
    if branch_policy == "conditional_only" and branch != "conditional":
        return y

    r_rows = frozen_tube_rows(current_latent_grid)
    m_rr = m1_contribution(q, k, v, r_rows, heads_by_block[block])

    # y_R^alpha = y_R + alpha * A[R,R]V_R
    y[:, r_rows, heads_by_block[block], :] += alpha * m_rr

    dose_logger.write(
        step=current_step,
        cfg_branch=branch,
        block=block,
        heads=heads_by_block[block],
        base_m1_rms=rms(m_rr),
        applied_delta_rms=abs(alpha) * rms(m_rr),
        signed_alpha=alpha,
    )
    return y
```

代数断言：

```python
assert_close(run(alpha=0), baseline)
assert_close(run(alpha=-1, branches="both"), stage3_m1_knockout)
assert modified_head_events == 100 * 40 * 2
```

### 6.3 Raw Contrast Guidance

```python
for step, timestep in enumerate(scheduler.timesteps):
    eps_c = dit(x_t, prompt, perturbation=None)
    eps_m1 = dit(x_t, prompt, perturbation="M1_knockout")
    eps_u = dit(x_t, negative_prompt, perturbation=None)

    g = eps_c - eps_m1
    eps = eps_u + cfg * (eps_c - eps_u) + lambda_ * g
    x_t = scheduler.step(eps, timestep, x_t)
```

若复用 pipeline 内部 CFG：

```python
# pipeline 随后计算 eps_u + cfg * (returned_cond - eps_u)
returned_cond = eps_c + (lambda_ / cfg) * (eps_c - eps_m1)
```

必须允许 `lambda_ < 0`；`lambda_ == 0` 不运行额外 perturbed forward，直接复用 Baseline。

### 6.4 Unconditional/full-CFG audit

```python
eps_u_clean = dit(x_t, negative_prompt, perturbation=None)
eps_u_m1 = dit(x_t, negative_prompt, perturbation="M1_knockout")

g_c = eps_c_clean - eps_c_m1
g_u = eps_u_clean - eps_u_m1
g_full = cfg * g_c - (cfg - 1) * g_u

audit.write(
    conditional_rms=rms(g_c),
    unconditional_rms=rms(g_u),
    ratio=rms(g_u) / max(rms(g_c), eps),
    conditional_unconditional_cos=cosine(g_c, g_u),
    conditional_full_cos=cosine(g_c, g_full),
)
```

### 6.5 Pairwise norm calibration

```python
def calibrate_on_baseline_path(unit, head_pair):
    # 不读取 outcome，只在同一 baseline x_t 上计算 prediction-space dose。
    scales = []
    x_t = initialize_same_seed_latent(unit.seed)

    for step in range(40):
        eps_c = clean_conditional(x_t)
        g = {}
        for group in head_pair:
            eps_ko = conditional_with_m1_knockout(x_t, heads=group)
            g[group] = eps_c - eps_ko

        r = {group: rms(value) for group, value in g.items()}
        target = min(r.values())  # 只缩小强信号，不放大弱信号
        scales.append({group: target / max(r[group], EPS) for group in head_pair})

        eps_base = clean_cfg_prediction(x_t)
        x_t = scheduler.step(eps_base, step, x_t)  # 始终沿 baseline path

    save_frozen_scales(unit, head_pair, scales)
```

### 6.6 Norm-matched generation

```python
scales = load_frozen_scales(unit, head_pair)

for step in range(40):
    eps_c = clean_conditional(x_t)
    eps_ko = conditional_with_m1_knockout(x_t, heads=current_group)
    g_raw = eps_c - eps_ko
    g_matched = scales[step][current_group] * g_raw

    eps = eps_u + cfg * (eps_c - eps_u) + lambda_ * g_matched
    achieved_rms_logger.write(step, rms(g_raw), rms(g_matched))
    x_t = scheduler.step(eps, step, x_t)
```

### 6.7 Signed trajectory response

```python
delta_ko = centers(alpha=-1) - centers(baseline)

for alpha in [-1, -0.5, 0, 0.5, 1]:
    delta = centers(alpha=alpha) - centers(baseline)
    beta = sum(dot(delta[t], delta_ko[t]) for t in valid_frames) / (
        sum(squared_norm(delta_ko[t]) for t in valid_frames) + EPS
    )
    # 理想局部线性响应：beta ≈ -alpha
```

### 6.8 执行器与幂等性

```python
for task in frozen_manifest.tasks:
    key = sha256(canonical_json(task.identity_fields))
    out = OUTPUT_ROOT / task.case / f"seed_{task.seed:05d}" / key

    if validate_complete(out, task):
        continue

    assert task.device != "cuda:4"
    write_running_manifest_atomically(out, task)
    try:
        video, audit = run_task(task)
        validate_audit(task, audit)
        write_video_atomically(video, out / "generated.mp4")
        write_complete_atomically(out, audit)
    except Exception:
        write_error(out, traceback.format_exc())
        raise
```

---

## 7. 代码与产物结构

计划新增代码：

```text
training_free_m1_control/
├── plan.md
├── experiment_spec_v1.json
├── build_manifest.py
├── run_soft_scaling.py
├── run_contrast_guidance.py
├── calibrate_pairwise_norms.py
├── analyze_training_free_controls.py
├── dashboard.py
├── bench.sh
└── tests/
    ├── test_soft_scaling_algebra.py
    ├── test_knockout_equivalence.py
    ├── test_guidance_equation.py
    ├── test_negative_lambda.py
    ├── test_norm_matching.py
    └── test_manifest_identity.py
```

大体量产物统一写入：

```text
/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/
└── latest3350_v1/training_free_m1_control_v1/
    ├── frozen_spec/
    ├── baseline_tracks/
    ├── soft_scaling/
    ├── contrast_raw/
    ├── contrast_norm_matched/
    ├── temporal/
    ├── metrics/
    ├── analysis/
    └── logs/
```

每个 task 必须保存：

- case、seed、target、input JSON、first frame、prompt hash；
- model/checkpoint、scheduler、CFG、steps、resolution、fps；
- ranking 文件路径/SHA256、具体 layer-head list；
- `α/λ`、branch policy、raw/norm-matched、calibration hash；
- frozen tracks 路径/SHA256、13 anchor token indices；
- 每 step/head/branch 的 M1 dose、prediction delta 和 achieved norm；
- generated video、metrics、overlay、complete/error marker；
- git commit、dirty diff hash、GPU、开始/结束时间。

唯一任务键至少包含：

```text
(case, seed, target, head_scope_hash, method, alpha_or_lambda,
 branch_policy, time_scope, norm_calibration_hash,
 model_hash, scheduler_hash, code_hash)
```

---

## 8. 当前正向 guidance pilot 的处理

独立实时可视化入口：

```text
http://localhost:8092/training-free-m1-control?v=1
```

页面按选中的 `case×seed` 分成 Soft Scaling 与 Conditional Contrast Guidance
两行，列固定为 `-1/-0.5/0/+0.5/+1`。只渲染当前选择的十个槽位，视频接近
视口时才加载；顶部同步显示 TF-0 硬门控和正式 intervention 完成进度。该页面
只展示已落盘事实，不将 Pending 结果或尚未计算的指标填成数值。

现有页面中的：

```text
3 cases × seeds {47326,42} × Top100 × M1/M2/M3 × λ {0.5,1}
```

只作为 range-finding 与工程复用候选。纳入 TF-1 前必须逐项审计：

- M1、Top100、all-time、object_A；
- same-seed Baseline 和 frozen tracks；
- CFG=5、40 steps、相同模型/scheduler；
- conditional-only perturbed branch；
- guidance equation 和 `modified_head_events=4000`；
- 完整 manifest/dose/video。

它缺少负 `λ`、Bottom/Random、Soft Scaling 和 held-out cases，因此不能单独支持 training-free control 的机制或泛化结论。M2/M3 正向结果保留为额外 flow 对照，不混入本计划的 M1 primary family。

---

## 9. Gate 与执行清单

### 9.0 当前 GPU7 自动 Phase-B/D 验证（已确认配置）

本轮不执行 Same/Future/Past，只验证 `all_time` M1 正向直接增强：

- cases：`test_5.txt` 的 20 个唯一 case；
- seeds：固定 `90094, 68613, 35075, 32466, 13248`；
- target：每个 case 自动对象顺序中的 `object_A`；
- heads：`latest3350` Top100；
- Phase B：`alpha={0.1,0.25}`，作用于 denoising step `0..39`；
- Phase B 选参：视频生成完成后，分别计算 simulator GT full-frame MSE 与
  GT-relative CoTracker Center-ADE/D0。seed 先在 case 内平均，两个候选在每项
  可用指标上按“越小越好”排名，再对 case 等权平均；总排名完全相同时选择
  较小的 `alpha=0.1`；这些指标不进入 guidance；
- Phase D：只用 Phase B 选定的单一 alpha，对比 `0..9`、`0..19`、`0..39`；
  其中 `0..39` 与 Phase B 的同 alpha 计算完全相同，直接软链接复用，不重复生成；
- Region：只读取既有自动 GroundingDINO/SAM2 source tube 的 F00 点和 F00 mask；
  source video 的未来轨迹/mask 不进入生成；
- 生成 tube：由同 case、同 seed 的 clean Baseline 通过 CoTracker 冻结；
- 执行：GPU7 单队列、tmux、阶段失败即停止、所有阶段断点续跑。

自动阶段：manifest/F00 cache → 缺失 Baseline → frozen Baseline tracks →
Phase B → 生成后选 alpha → Phase D。预计新增干预视频 `200+200=400`，另有
100 个 Phase-D full40 条目复用 Phase-B 文件；clean Baseline 只补缺失项。

### Gate A：允许实现

- [ ] 用户确认本计划中的 branch policy、MDE、pilot case/seed 和三份 Random100。
- [ ] 冻结 `experiment_spec_v1.json`，写入所有 hash。

### Gate B：允许 TF-1 GPU pilot

- [ ] TF-0 十项硬门槛全部通过。
- [ ] 负 `λ`、`α=-1/0` 单元测试通过。
- [ ] 一个真实 GPU smoke 的视频、dose、metrics、overlay 可完整读取。

### Gate C：允许 TF-2/TF-3

- [ ] TF-1 没有数值爆炸或系统性全局崩坏。
- [ ] tube coverage 阈值、MDE、norm RMS 门槛已冻结。
- [ ] 任务 inventory 排除重复生成。

### Gate D：允许 held-out confirmation

- [ ] case-level variance、MDE 和 power 曲线已冻结。
- [ ] held-out cases/seeds 未参与 ranking、pilot 或页面挑选。
- [ ] 固定样本数、primary contrasts 和停止规则。

### Gate E：允许 temporal guidance

- [ ] Stage 4 directional dose 与 outcome 已完成。
- [ ] temporal candidate 选择不超过两个，且选择规则有审计记录。
- [ ] 使用新的 held-out cases/seeds 验证。

---

## 10. 最终允许的结论模板

只有相应证据齐备时才使用：

1. **Raw controllability：**“在固定 inference 设置下，Top100-M1 的 raw prediction contrast 比对照产生更强/更弱的输出变化。”
2. **Direction specificity：**“在 pairwise norm-matched 后，Top100 相比 Bottom/Random 的方向对预注册 endpoint 更有效。”
3. **Functional tendency：**“影响超过 MDE 的 endpoint 主要位于身份/生存、轨迹/速度或两类同时。”
4. **Physical improvement：**仅当 simulator GT endpoint 改善且生成质量 guardrail 未恶化时，允许写“物理质量改善”。
5. **Mechanism boundary：**即便以上成立，也只能说明 M1 contribution 是有效控制信号；仅凭干预结果不能证明 message 的完整语义内容。
