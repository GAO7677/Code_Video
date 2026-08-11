# Object Query Self-Attention 信息流消融：严格执行计划

## 0. 文档状态

- 状态：**Gate 0 已确认；Stage 0 已完成；Stage 1 执行中**。
- Gate 0 后先执行只读清单审计和 query-time ranking validation；尚未启动新的大规模消融视频矩阵。
- 目标：在新的 `latest3350` PCK head 排名下，区分 `R→R`、`C→R`、`R→C` 三类 self-attention 信息流更主要地影响对象轨迹、对象外观，还是对象外区域，并比较 Top100、Bottom100、随机匹配 100 heads 与 All720。
- 结论边界：`R` 是由追踪点构成的**稀疏 object-token tube**，不是完整对象 mask；因此结论首先针对该 tube 表示，不能直接外推成“完整对象区域的全部信息流”。

相关入口：

- 消融定义：`07_OBJECT_QUERY_ATTENTION_ABLATION_MATRIX.md`
- 当前实现：`run_legacy_ti2v_temporal_object_tube_ablations.py`
- 指标实现：`METRICS_IMPLEMENTATION_INDEX.md`
- 待验证假设：`HYPOTHESES_TO_VALIDATE.md`
- 一键补指标：`bench_missing.sh`

---

## 1. 首先冻结的数学定义

令某层、某 head 的 self-attention 输出为

\[
Y_{q,h}=\sum_k A_{qk,h}V_{k,h}.
\]

当前干预采用 **post-softmax contribution subtraction，不重新归一化**：

\[
Y'_{q,h}=Y_{q,h}-\sum_{k\in B(q)}A_{qk,h}V_{k,h}.
\]

它删除的是选定边对输出的实际 value contribution；不改变 Q/K/V projection，不重算 softmax，其他 heads、residual、FFN 和 cross-attention 均保留。

### 1.1 三个主信息流

| ID | 被删除的边 | 精确计算项 | 首要诊断问题 |
|---|---|---|---|
| M1 | `R K/V → R Query` | `Σ A[R_q,R_k]V[R_k]` | object tube 内部是否传递对象自身的时序/外观状态 |
| M2 | `C K/V → R Query` | `Σ A[R_q,C_k]V[C_k]` | 环境、其他对象是否向目标对象输入约束/交互信息 |
| M3 | `R K/V → C Query` | `Σ A[C_q,R_k]V[R_k]` | 目标对象是否向其他对象和背景广播状态 |

`C→C` 始终保留。M1/M2/M3 是**方向性边消融**，不能被写成“删除对象 token 本身”。

执行前还要冻结 token universe：`C` 必须定义为同一 self-attention 序列中 `R` 以外的明确 token 集；若序列中存在 special/reference/text token，必须逐类列出是否属于 `C`，不能静默混入。

### 1.2 latent-video 时间分解

| 展示名 | 条件 | 含义 |
|---|---|---|
| All-time | 所有 `t_q,t_k` | 删除该信息流的全部 latent-video 时序边；旧命名 `Only` 统一改为 `All-time` |
| Same | `t_k=t_q` | 同时刻边 |
| Future | `t_k<t_q` | 历史 K/V 向未来 Query 的边 |
| Past | `t_k>t_q` | 未来 K/V 向过去 Query 的反向控制边 |

必须满足：`All-time = Same ∪ Future ∪ Past`，三者两两不相交。这里的 latent-video 时间与 diffusion denoising step 是两个不同维度。

### 1.3 head group

| group | 定义 | 可支持的结论 |
|---|---|---|
| Top100-latest3350 | 新排名最高的 100 个 layer-head | 相对高 PCK heads 的必要性 |
| Bottom100-latest3350 | 新排名最低的 100 个 layer-head | 排名负对照 |
| Random100-layer-matched | 每层数量分布与 Top100 完全一致的冻结随机 100 heads | 排除 layer 分布和“删除 100 heads”本身的影响 |
| All720 | 全部 30×24 layer-head | 最大干预/上界 sanity check，不作为 head 特异性证据 |

不允许把 All720 效应除以 7.2 后与 100 heads 直接比较；生成网络非线性，该归一化没有因果意义。

---

## 2. 预注册假设与反证条件

### H1：Top100 具有排名特异性

- If：latest3350 PCK 排名选中了与对象时序定位更相关的 heads。
- Then：在相同 M、时间范围和 object 下，Top100 的轨迹效应应稳定大于 Bottom100，并大于 layer-matched Random100。
- Because：差异不能仅由删除 head 数量或层分布解释。
- 反证：Top100 与 Random100 的 case-cluster bootstrap CI 大量重叠，或优势只由极少数 case/对象驱动。

### H2：M1 更直接影响目标对象自身状态

- If：`R→R` 主要承载 object tube 内的状态延续。
- Then：M1 应优先改变目标对象轨迹/速度，或中心对齐后的对象外观；对 other-object/background 的影响应弱于 M3。
- 反证：M1 的主要效应只出现在 outside-object 指标，目标对象轨迹和外观均无稳定变化。

### H3：M2 更直接影响环境对对象的约束

- If：`C→R` 承载接触、场景约束或其他对象对目标对象的输入。
- Then：M2 在交互 case 中应更明显改变目标对象轨迹、接触时刻或碰撞后速度，且强于非交互 case。
- 反证：效应只表现为对象外观/全局画质崩坏，且与是否存在交互无关。

### H4：M3 更直接影响对外传播

- If：`R→C` 承载对象状态向环境/其他对象的广播。
- Then：M3 的 other-object ADE 或 outside-object 变化应相对 M1/M2 更强，同时目标对象自身效应可较小。
- 反证：M3 只改变目标对象自身，未产生可重复的跨对象/对象外效应。

### H5：时序方向具有选择性

- If：历史到未来边承担前向状态传播。
- Then：Future 应比 Past 更稳定地影响后续轨迹；Same 更偏同帧结构/局部交互。
- 反证：Future/Past/Same 在多 case、多 seed 下没有稳定差异，或差异完全由删除贡献量解释。

以上均是待验证假设，不在数据分析前写成结论。

---

## 3. 指标与决策层级

所有干预效应的主 reference 为**同 case、同 seed 的未消融 Baseline**。vs GT 仅用于回答物理正确性是否改善/恶化，不能替代 vs Baseline 的因果干预效应。

### 3.1 Primary endpoints

| 信息类型 | 主指标 | 计算与方向 |
|---|---|---|
| 目标对象轨迹 | `Center-ADE / d0` | 候选与 Baseline 中心逐帧距离均值，除以 F00 bbox 对角线；越大表示轨迹干预越强 |
| 目标对象外观 | `Center-aligned Object LPIPS` | 先中心对齐、固定 crop、mask 外置灰再算 LPIPS；越大表示非位置造成的局部外观变化更强 |
| 对外传播 | `Other-object Center-ADE / d0` | 消融 A 后，未选中对象相对 Baseline 的中心轨迹变化；越大表示跨对象传播更强 |
| 对象生存 | `Track/Mask Retention` 与 `Disappearance Onset` | 有效追踪/分割帧比例及首次持续消失帧；这是实验结果和 guardrail，不能把失败样本静默删除 |

### 3.2 Secondary endpoints

- 轨迹：Center-FDE、Point-ADE、PCK@5/10/20%、速度大小/方向/向量误差、RAFT ROI flow disagreement。
- 外观/形状：DINOv2 cosine、center-aligned Shape IoU、area/aspect/circularity error。
- 对象外：Outside-object LPIPS、other-object RAFT/轨迹指标。
- 物理交互子集：GT contact-time error change、post-contact velocity error change。
- 生成质量 sanity：VBench subject consistency、motion smoothness、dynamic degree、quality suite。

### 3.3 明确降级为 diagnostic 的指标

- Full-frame MAE/SSIM/PSNR、Temporal Δ-MAE：容易被静态背景或外观变化主导，不作为运动轨迹主结论。
- Raw-mask IoU：混合位置和形状，只用于解释。
- All720：只表示强干预上界，不能证明某类 heads 的专门功能。

### 3.4 质量门控与缺失值

同一结果同时报告：

1. 全部生成样本上的 survival/failure rate；
2. 通过轨迹质量门控样本上的轨迹指标；
3. 未通过样本的明确路径和失败原因。

禁止把缺失轨迹/缺失 mask 填 0；禁止只保留“好追踪”的视频而不报告淘汰率。

---

## 4. 必须记录的消融剂量，避免语义混杂

对每个 layer/head、denoising step、CFG branch、`t_q/t_k` block 记录：

\[
m_A=\sum_{k\in B(q)}A_{qk},\qquad
m_V=\left\|\sum_{k\in B(q)}A_{qk}V_k\right\|_2,
\]

并记录 `removed contribution norm / original attention-output norm`。

解释规则：

- 若 Top100 输出变化更大但 `m_V` 也显著更大，只能说其被删除的贡献更强，不能直接说其编码了更特异的语义。
- 只有在贡献剂量控制、layer-matched Random100 对照后仍存在差异，才能支持 head-group specificity。
- 除原始效应排序外，补充按 `m_V` 分层/回归后的敏感性分析；不以简单除法作为唯一校正。

---

## 5. 分阶段执行方案

### Gate 0 — 冻结选择（当前阶段，必须人工确认）

- [x] 确认 discovery/confirmation seed 划分。
- [x] 确认 Random100 重复预算。
- [x] 确认 held-out confirmatory cases 来源与数量策略。
- [x] 确认 probe/rescue 是否纳入本轮范围。
- [x] 生成冻结实验规格 `experiment_spec_latest3350.json`；运行时 manifest 继续补充 head/code/model/scheduler hash。

**停止条件：以上未确认，不进入 Stage 1；更不会提交大规模 GPU 生成。**

### Stage 0 — 现有产物盘点与可复用性审计

目标：避免重复计算，也避免混用新旧排名或不同实现生成的结果。

步骤：

1. 扫描现有 `complete.json`、`manifest.json`、视频和 metrics report。
2. 以 `(case, seed, target, head-scope hash, M, time mode, denoise window, code hash)` 建唯一键。
3. 将结果分类为：完全可复用、缺指标可补、配置不一致必须重跑、损坏/不完整。
4. All720 只有在实现、scheduler、seed 和其他配置完全一致时才复用；它虽不依赖排名，仍依赖运行配置。
5. 输出 machine-readable inventory 和人类可读进度表；本阶段不删除旧产物。

通过条件：每个复用视频都可追溯到完整 manifest；重复键无冲突。失败则暂停并报告冲突路径。

执行结果（2026-08-11）：见
`/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/stage0_inventory/`。
共审计 2,690 个 manifest，2,624 个完整；latest3350 的 72 个结果全部完整且有指标，但仅覆盖
`0613pybullet_sample_001460_w002 / seed 47326 / Top100+Bottom100`。未发现重复实验键或 JSON 损坏。
旧结果普遍缺精确 code provenance，因此仅作 exploratory reuse；confirmatory run 必须使用新冻结 manifest。

### Stage 1 — 验证 latest3350 ranking 是否适用于所有 `Q_t`（不生成视频）

问题：当前 latest3350 排名来自特定 query/step 统计；它不自动证明同一 Top100 对 tube 内所有帧 query 均有效。

步骤：

1. 对多个 query-time anchor（至少覆盖前/中/后）计算每个 layer-head 的 PCK、mean error。
2. 计算相邻/跨时刻 head ranking 的 Spearman、Top100 Jaccard、Top-vs-Bottom 稳定性。
3. 可视化固定 `Q_t` 对所有 `K_t'` 的高响应位置与 GT tube 的对应关系。
4. ranking 使用与 confirmatory cases 分离的数据；若无法确认数据隔离，该结果只标 exploratory。

预设分支：

- 若 query-time 稳定性足够，则继续使用 `Top100-latest3350/Bottom100-latest3350`。
- 若明显不稳定，则新增按多个 `Q_t` 聚合的 `TubeTop100/TubeBottom100`，并将 S039 Top/Bottom 保留为对照，不偷偷替换定义。

“足够稳定”的数值阈值需在执行前根据 anchor 数量冻结，不能看完结果后再定。

### Stage 2 — 实现审计、单元测试和 smoke test

必须通过的测试：

1. Same/Future/Past 两两不相交，且并集严格等于 All-time。
2. 对同一 q/head，三段被删 contribution 的和与 All-time 被删 contribution 数值一致（容差提前冻结）。
3. Top/Bottom 各恰好 100、互不重叠；Random100 从 Top/Bottom 之外采样，并与 Top100 的 per-layer histogram 完全一致。
4. hook 覆盖预期全部 self-attention blocks、40 个 denoising steps 和两个 CFG branches；实际 scheduler timestep/sigma 同步记录。
5. no-op hook 与 Baseline 在确定性容差内一致。
6. 未选 heads、cross-attention、FFN 不被改写。
7. 一个 case × 一个 seed × 一个 target 完成 smoke，且视频、manifest、剂量日志和指标均可被 dashboard 读取。

任何一项失败，停止大规模运行。

### Stage 3 — Primary：All-time 信息类型筛查

固定因素：

- flow：M1、M2、M3；
- head group：Top100、Bottom100、Random100-layer-matched、All720；
- time：All-time；
- target：每个 case 的各 object，外加有定义时的 all_objects；
- seed：discovery seeds；
- reference：同 seed Baseline。

现有 10-case 方案曾按每 seed 约 33 targets 估算；Stage 0 必须从最终 manifest 重算，不把该数写成既定事实。

若最终为 33 targets × 3 discovery seeds：

- 单个 Random100 的主矩阵：`33 × 3 × 4 head groups × 3 flows = 1188` 个候选视频；
- 若三个 Random100 都全量运行：head variants 为 6，总计 `1782` 个候选视频；
- Baseline 可在同 case/seed 下复用，不按每个消融单独生成。

筛查输出：每个 primary metric 的 case-cluster effect、95% CI、case/object/seed 分层图、Top/Bottom/Random/All720 倍数对比和原始效应，不只给全局平均。

### Stage 4 — latent-video 时间方向分解

在有单对象和多对象交互的代表集上运行 M1/M2/M3 × Same/Future/Past；All-time 从 Stage 3 复用。

候选代表集（Gate 0 后冻结）：

- `0613pybullet_sample_001460_w002`；
- `0613pybullet_sample_000331_w001`；
- PhysicIQ ball-and-block motion-to-end case。

按先前对象清单估算为 11 targets/seed。若 3 seeds、Top/Bottom/All720、3 flows、4 time modes：共 `1188` cells，其中与 Stage 3 重合的 All-time cells复用，预计新增 `891`；最终以 Stage 0 inventory 为准。

分析重点：Future vs Past 的方向选择性、Same 的局部结构效应，以及这些差异能否在控制 `m_V` 后保留。

### Stage 5 — M2/M3 双向边界交互（2×2）

同一 head group/time setting 下比较：

| 条件 | C→R | R→C |
|---|---:|---:|
| Baseline | 保留 | 保留 |
| M2 | 删除 | 保留 |
| M3 | 保留 | 删除 |
| M6 | 删除 | 删除 |

目的：判断输入和输出边的效应是否可加，或是否存在双向耦合。报告 M6 相对 `M2 + M3` 的 interaction contrast，不用肉眼把 M6 解释成两者简单相加。

### Stage 6 — diffusion denoising 阶段定位

这是与 Same/Future/Past 独立的轴。先对 Stage 3/4 中有选择性效应的少量配置运行：

- high-noise window：暂定 steps 0–12；
- mid：暂定 13–26；
- low-noise：暂定 27–39；
- all：0–39。

执行前必须按实际 scheduler 的 timestep/sigma 校验并冻结窗口；禁止预先宣称“早期=轨迹、晚期=外观”。若窗口边界不对应近似等 sigma 区间，则改为按 sigma 分位数切分。

### Stage 7 — 基线 message probe；rescue 为可选后续

仅靠 knockout 最多证明必要性，不能充分证明被删信息“具体是什么”。因此提取 Baseline message：

- `M_RR = Σ A[R_q,R_k]V[R_k]`；
- `M_R←C = Σ A[R_q,C_k]V[C_k]`；
- `M_C←R = Σ A[C_q,R_k]V[R_k]`。

在 case-held-out split 上训练简单线性 probe，分别预测：位移/速度/contact、对象 DINO/shape/identity、other-object future displacement。加入 permutation label 和 Random100 controls。

若用户将 rescue 纳入范围，再对最强且可重复的少数 contrast 做 baseline-activation rescue；不在全矩阵上盲目扩张。

### Stage 8 — Confirmatory run

仅在 Stage 3–7 的代码、hypotheses、primary metrics 和 contrasts 冻结后执行：

- 使用未参与 exploratory inspection 的 seeds；
- 使用未参与 latest3350 ranking 构建/选择的 held-out cases；
- 固定样本量和停止规则，不根据中间结果提前停或追加到显著；
- 只验证预注册主对比，探索性发现另表报告。

确认集样本量不预先拍脑袋固定。Stage 0/3 pilot 后只使用 case-level paired effect 的方差估计，生成 `case count × MDE × power` 曲线；在校正后的主检验显著性水平下，以 power≥0.8 和用户认可的最小有意义效应（MDE）冻结 case 数。若可获得 case 数达不到要求，则明确降级为 exploratory/pilot，不做总体因果宣称。

### Stage 9 — 汇总、可视化与文档

1. 每个 case 页面并排展示 Baseline、Top100、Bottom100、Random100、All720；按 M1/M2/M3 分板块。
2. 每个视频下方折叠栏显示轨迹、外观、传播、生存和质量指标，以及精确计算含义。
3. 总览给出样本结构：case 数、每 case seed 数、对象/target 数、有效/失败数；不只报“视频总数”。
4. 给出 case-level paired effect、95% CI、head-group 倍数差、贡献剂量和代表性/反例视频路径。
5. 同步更新 definitions、hypotheses、metrics index 和结论 md；页面只读取统一 report，避免页面与 md 数值漂移。

---

## 6. 统计分析计划

- 最高独立聚类单位是 **case**；seed、object、variant 均嵌套于 case，不把视频数当独立 `n`。
- 主分析采用 case-cluster bootstrap 95% CI；同时展示每个 case 的 paired differences。
- primary contrasts 预注册为：
  1. Top100 − Bottom100；
  2. Top100 − layer-matched Random100；
  3. M1 vs M2 vs M3 的信息类型差；
  4. Future − Past；
  5. M3 对 other-object ADE 的方向性对比。
- 多重比较对预注册 primary family 使用 BH-FDR；secondary/exploratory 只报告 effect size、CI 与校正后的 q-value，不以单个 p<0.05 下结论。
- 倍数只在分母远离 0 且方向一致时报告；同时必须展示绝对差和 CI。分母接近 0 时标为不稳定，不制造巨大倍数。
- 若 confirmatory held-out case 数太少，结果明确标记 pilot/exploratory，不包装成总体规律。
- 样本量依据是 case-level paired effects，而不是生成视频总数；多个 seeds 主要降低 case 内 Monte Carlo 噪声，不能替代增加独立 cases。

---

## 7. 复现、资源与停止规则

每个任务必须保存：

- 输入 JSON、case/target/seed；
- head scope 文件绝对路径、SHA256 和具体 layer-head 列表；
- git commit 与工作区 diff hash；
- checkpoint/model hash；
- scheduler/timestep/sigma；
- M、time mode、denoise window、CFG branch 覆盖；
- 输出视频、日志、complete marker、metrics report；
- GPU id、开始/结束时间、失败原因。

资源规则：不使用 GPU4；大体量结果写入 `/data/gaoya/agent-data/outputs`。GPU 分配只在具体 stage 启动前检查，不在计划中假定空闲设备。

硬停止条件：

1. no-op 不确定、时间分区代数测试失败或 hook 覆盖不完整；
2. head 文件 hash/定义与 manifest 不一致；
3. Baseline 与候选 seed/config 不一致；
4. 重复唯一键产生冲突输出；
5. survival failure 激增但未能区分模型崩坏与目标效应；
6. 指标缺失被静默填值；
7. confirmatory 数据与 ranking/探索集发生泄漏。

遇到硬停止条件，只完成诊断报告，不继续消耗大规模 GPU。

---

## 8. Gate 0 待用户确认的问题

1. **Random100 预算**：建议“1 个 layer-matched Random100 跑完整 discovery，另外 2 个随机 draw 只跑代表子集做稳健性检查”；是否接受？若 3 个都全量，按 33 targets × 3 seeds 的暂估会从 1188 增至 1782 个候选视频。
2. **Seed 划分**：建议 discovery=`13248, 47326, 90094`，confirmation=`32466, 35075, 68613`。其中 `47326` 已被多次观察，只作为 exploratory；是否接受？
3. **Held-out cases**：latest3350 构建数据可能与现有 case 重叠。建议先从 PhysicIQ67/PyBullet 建立完全未参与 ranking/页面挑选的候选池，覆盖单对象、多对象和碰撞，再用 pilot 方差与目标 MDE 决定最终 case 数；少于约 8 个独立 cases 默认只作 pilot。候选 cases 由我筛选，还是由你指定？
4. **Probe/rescue 范围**：建议 message probe 纳入本轮、放在因果筛查之后；activation rescue 只对最强少数结果作为第二阶段。是否接受？
5. **ranking 稳定阈值**：建议 Stage 1 在看结果前冻结 Top100 Jaccard/Spearman 的通过阈值；我可以先根据 anchor 数和 bootstrap 噪声做 null calibration，再提交阈值供你确认。是否接受这种定阈值方式？

收到以上确认后，先只执行 **Stage 0 inventory + Stage 1 ranking validation** 并回报，再决定是否进入 GPU 密集的 Stage 2/3。
