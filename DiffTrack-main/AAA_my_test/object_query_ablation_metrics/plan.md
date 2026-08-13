# Object Query Self-Attention 信息流消融：严格执行计划

## 0. 文档状态

- 状态：**Gate 0、Stage 0–3 已完成；Stage 3 最终 discovery 报告已冻结；Stage 4 尚未启动**。
- 下一步不是直接提交时间消融矩阵，而是先通过 Stage 4.0 的 directional-dose、token-universe、指标 smoke 和 inventory 门槛。
- 目标：在新的 `latest3350` PCK head 排名下，区分 `R→R`、`C→R`、`R→C` 三类 self-attention 信息流更主要地影响对象轨迹、对象外观，还是对象外区域，并比较 Top100、Bottom100、随机匹配 100 heads 与 All720。
- 结论边界：`R` 是由追踪点构成的**稀疏 object-token tube**，不是完整对象 mask；因此结论首先针对该 tube 表示，不能直接外推成“完整对象区域的全部信息流”。

相关入口：

- 消融定义：`07_OBJECT_QUERY_ATTENTION_ABLATION_MATRIX.md`
- 当前实现：`run_legacy_ti2v_temporal_object_tube_ablations.py`
- 指标实现：`METRICS_IMPLEMENTATION_INDEX.md`
- 待验证假设：`HYPOTHESES_TO_VALIDATE.md`
- Training-Free M1 control：`training_free_m1_control/plan.md`
- Training-Free M1 可视化：`http://localhost:8092/training-free-m1-control?v=1`
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

Stage 4 的 token universe 已明确冻结：`C` 是同一个 latent-video `T×H×W` 网格中 `R` 之外的 token。运行时必须满足
`Q/K/V token count = T×H×W`；若存在 special/reference/text token，当前实现直接停止，不能静默混入 `C`。

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

执行结果（2026-08-11）：5 个外部 PyBullet cases × 3 discovery seeds，共 15/15 runs 完成。
冻结阈值下判定为 **PASS**：query-time Top100 两两 median Jaccard=`0.7391`，
median Spearman=`0.9817`；fixed latest3350 Top100 在 13/13 query anchors 上优于 Bottom100；
case-level Top−Bottom PCK@32 均值=`51.128 pp`，case-cluster bootstrap 95% lower bound=`29.921 pp`。
因此主矩阵继续使用 fixed latest3350 Top/Bottom；`TubeTop/TubeBottom` 保留为敏感性分析，不替换主定义。
完整报告和固定 Q overlay 位于
`/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/stage1_query_time_validation/analysis/`。

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

执行结果（2026-08-11）：**PASS**。9 项 CPU 代数/回归测试全部通过；3 个
layer-matched Random100 draw 均为 100 heads、与 fixed Top100/Bottom100 不重叠，且逐层
数量直方图与 Top100 完全一致。真实 GPU smoke 使用
`0613pybullet_sample_000301_w000 / seed 47326 / object_A`，分别验证
Random100-M2 和 All720-M3：视频均可完整解码为 49 帧（704×1280），dose 张量均为
`40 × 2 × 30 × 24`，有效事件数分别精确等于 `100×40×2=8000` 和
`720×40×2=57600`，无缺失或 NaN。Stage 2 smoke 输出位于
`/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/stage2_smoke_videos/`。

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

实际完成结果（2026-08-12）：10 cases、每个 seed 共 33 targets、3 discovery seeds，
`33 × 3 seeds × 4 head groups × 3 flows = 1188` 个任务全部生成。29/30 个 case-seed
具有同 seed Baseline，Fast/Trajectory/Survival 各完成 1152 条；唯一缺失 reference 的
`crop_top60px / seed 47326` 共 36 条不进入 outcome 统计。最终报告：
`/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/stage3_final_analysis/STAGE3_FINAL_REPORT.md`。

### Stage 4 — latent-video 时间方向分解

#### 4.0 目的与结论边界

Stage 4 要回答的不是“Top100 是否重要”——Stage 3 已经完成该筛查——而是把每类信息流沿 latent-video
时间轴拆开，判断其作用主要来自：

| 时间模式 | 被删除的边 | 主要诊断问题 |
|---|---|---|
| Same | `t_k=t_q` | 同帧结构、身份或局部交互是否重要 |
| Future | `t_k<t_q` | 历史 K/V 是否向未来 Query 传播状态 |
| Past | `t_k>t_q` | 生成是否依赖未来 latent 对过去 Query 的双向上下文 |

分别对 M1/M2/M3 解释：M1 检查对象自身状态何时维持，M2 检查环境/其他对象约束何时进入对象，
M3 检查对象状态何时传播到对象外。由于 Wan latent-video self-attention 可以双向访问整段序列，
`Past` 不是“物理未来真的导致过去”，而是**反时间方向的模型上下文对照**。

Stage 4 仍然是 discovery/pilot。它最多说明某类已删除 contribution 在某一时间方向上对输出是必要的；
仅凭 knockout 不能直接识别 message 的完整语义，也不能把 vs Baseline 的变化写成相对 GT 的物理退化。

#### 4.1 Stage 4.0 — 执行前硬门槛

在提交任何 GPU 大矩阵前，必须全部通过：

1. Same/Future/Past 两两不相交，且 contribution 的**向量和**等于 All-time；三段 norm 不要求相加等于 All-time norm。
2. 9 个 `M1/M2/M3 × Same/Future/Past` 模式均记录有限的 `attention_mass`、`removed_value_norm`、
   `original_output_norm`、ratio、affected-query count，以及 query-summed dose。
3. 每个选中 physical head 必须恰好记录 `40 denoising steps × 2 CFG branches` 个 dose 事件；不完整直接失败。
4. `Q/K/V token count = T×H×W`；发现未分类 special/reference/text token 直接失败。
5. 一个 case × 一个 seed × 一个 single-object target 完成真实 GPU smoke：视频 49 帧可解码，manifest、dose、
   Fast/Trajectory/Survival 以及 center-aligned LPIPS/shape 全链路可读取。
6. inventory 按 case、seed、target、head hash、flow、time mode、protocol、temporal runner +
   base ablator 联合代码指纹和 dose 完整性重新判断复用；
   旧 directional 视频若没有有效 dose，只能作视觉历史对照，不能进入 Stage 4 主分析。

任何一项失败，停止 Stage 4A，不允许用缺失 dose 的输出先做机制结论。

当前预检状态（2026-08-12）：CPU 代数、9 个 directional exact-dose、dose coverage hard-fail 和
token-universe hard-fail 测试已经通过；实现协议升级为
`attention_matrix_ablation_temporal_direction_v2_dose`。真实 GPU smoke 与 LPIPS/shape 全链路尚未执行，
因此 Stage 4.0 **尚未 PASS**。

#### 4.2 Stage 4A — 三 case 机制 pilot

冻结代表集：

| Case | objects | targets/seed | 角色 |
|---|---:|---:|---|
| `0613pybullet_sample_001460_w002` | 2 | 3 | 已有历史时间消融观察；用于复核和实现连续性 |
| `0613pybullet_sample_000331_w001` | 2 | 3 | 外部 PyBullet 多对象对照 |
| `physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end` | 4 | 5 | 多对象/交互复杂度对照 |

每 seed 合计 11 targets，seeds 固定为 discovery `13248, 47326, 90094`。single-object 和 all-objects
必须分层报告：single-object 的 `C` 包含其他对象和背景，all-objects 的 `C` 主要是对象外区域，二者语义不同。

Head scope：

- Top100、Bottom100、Random100-layer-matched-draw0 全量运行；Random100 是排除 layer 分布和“任意删除
  100 heads”效应的必要对照。
- All720 只在 `0613pybullet_sample_001460_w002 / seed 47326` 的 3 targets 上作强干预 sentinel，
  不进入 head-specific 主对比。Stage 3 的 All720 轨迹门控失败率已达 M1 51.0%、M2 47.9%、M3 36.5%。

最大新增预算（inventory 无可复用 directional dose 时）：

| 部分 | cells |
|---|---:|
| 11 targets/seed（3 cases 合计）× 3 seeds × 3 head scopes × 3 flows × 3 directions | 891 |
| `001460` 缺失的 3 targets × 3 seeds × 3 head scopes × 3 flows All-time | 81 |
| All720 sentinel：3 targets × 1 seed × 3 flows × 3 directions | 27 |
| 最大新增总数 | 999 |

`000331` 和 ball-and-block 的 All-time Top/Bottom/Random 可从 Stage 3 候选中审计复用；`001460` 不在
Stage 3 的 10-case 矩阵中，不能默认已有可复用 All-time。最终任务数只采用 Stage 4 inventory 结果。

Stage 4 inventory（2026-08-12）已核实：1215 个所需 cells 中可复用 All-time 216 个，必须新生成/重跑
999 个，恰好为 directional 891、`001460` All-time 81、All720 sentinel 27。旧目录中 261 个 cells
只有可视视频但不满足主分析复用条件；其中 directional 均缺 v2 dose/provenance。报告位于
`/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/latest3350_v1/stage4_preflight_inventory/`。

3 个 case 只用于机制 pilot，不进行“总体显著”宣称。case 是最高独立单位；即使三个 case 方向完全一致，
双侧 exact sign-flip 的最小 p 值也只有 0.25，增加 seed 不能替代增加独立 case。

#### 4.3 冻结主对比与指标

主 reference 是同 case、同 seed Baseline。对每个 directional mode 先计算其自身相对 Baseline 的效应；
“哪个方向最像 All-time”降为 secondary，因为生成网络非线性，通常不满足
`Effect(All)=Effect(Same)+Effect(Future)+Effect(Past)`。

| ID | 冻结主问题 | 主 outcome | Guardrail / 条件 |
|---|---|---|---|
| T1 | Top100-M1 的 Future 是否强于 Past | Target Center-ADE / D0 的 `Future−Past` | Track Loss、Disappearance；只在通过门控轨迹上算 ADE |
| T2 | 交互 case 中 Top100-M2 的 Future 是否强于 Past | GT contact-time / post-contact velocity change；无合格 GT 时降级为 Baseline-relative velocity | 必须先审计 simulator GT 资格，禁止拿缺失 GT 的 case 填 0 |
| T3 | single-object Top100-M3 的 Future 是否强于 Past | Other-object Center-ADE / D0 的 `Future−Past` | other-object track coverage；all-objects 不混入该主对比 |

Same 相对 `0.5×(Future+Past)`、Center-FDE/PCK/velocity、DINO/shape/LPIPS、Outside-object LPIPS
和 All-time 相似度均为 secondary/diagnostic。center-aligned LPIPS 只有在 Stage 4.0 smoke 全链路通过后才可报告；
在此之前不能继续用 frozen-ROI MAE 冒充“纯外观”。

#### 4.4 剂量控制与允许的解释

每个 directional variant 同时报告：

- 每 affected Query 的平均 removed attention mass / removed AV norm；
- affected-query count；
- query-summed attention mass / removed-norm；
- removed/output norm ratio；
- step、CFG、layer/head 分层分布。

先报告原始 `Future−Past` outcome，再检查对应 dose 差异和支持区间。只有 Future/Past 的 dose 有共同支持时，
才进行 case-cluster 分层/回归敏感性分析；若支持区间不重叠，只能说“某方向删除的 contribution 更多且输出变化更大”，
不能说“每单位该方向信息更关键”。禁止用 outcome 或 norm 简单相除作为唯一剂量校正。

#### 4.5 Stage 4B — 扩展 case 后的统计验证

Stage 4A 完成后，用 case-level paired `Future−Past` 差异估计方差，并在**不读取 confirmation 结果**的前提下冻结：

1. 每个主对比的最小有意义效应 MDE；
2. 经主检验 family 校正后的 alpha；
3. power≥0.8 所需独立 case 数；
4. interaction/non-interaction 与 single/multi-object 分层比例。

少于 8 个独立 cases 一律保持 pilot；8 只是下限，不是自动充分样本量。Stage 4B 才使用未参与页面挑选的
held-out cases/seeds。若可用 case 数达不到 power 需求，明确停止在 exploratory，不追加样本直到显著。

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

当前实时入口（2026-08-11）：

- 独立页面：`http://localhost:8092/object-query-information-flow-validation?v=1`；
- 8092 总入口 `/` 已增加 `42 / LATEST3350 VALIDATION` 卡片；
- 页面已接入 Stage 1 的 13-anchor 数值与 F00/F24/F48 overlays、Stage 2 的 9 项硬门槛与两个真实 smoke，以及 Stage 3 的 1188-cell 实时进度、Baseline、已生成 M1/M2/M3 × 四种 head scope 视频和按视频懒加载的 attention-dose；
- 未生成视频不创建空卡片；轨迹、外观、背景和对象存活指标在统一 `report.json` 生成前明确标为未完成，不用 attention-dose 冒充结果指标；
- 页面模块：`AAA_my_test/object_query_ablation_metrics/information_flow_validation_dashboard.py`，主服务路由：`AAA_my_test/serve_latent_block_head_viewer_with_metrics.py`。

---

## 6. 统计分析计划

- 最高独立聚类单位是 **case**；seed、object、variant 均嵌套于 case，不把视频数当独立 `n`。
- 主分析采用 case-cluster bootstrap 95% CI；同时展示每个 case 的 paired differences。
- primary contrasts 预注册为：
  1. Top100 − Bottom100；
  2. Top100 − layer-matched Random100；
  3. M1 vs M2 vs M3 的信息类型差；
  4. Stage 4 T1：Top100-M1 `Future − Past` 的 Target Center-ADE / D0；
  5. Stage 4 T2：交互 case 中 Top100-M2 `Future − Past` 的 GT contact/post-contact velocity；
  6. Stage 4 T3：single-object Top100-M3 `Future − Past` 的 Other-object Center-ADE / D0。
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
8. directional dose 任一 selected head 未覆盖完整 `40×2` 事件，或出现 NaN/Inf；
9. `Q/K/V token count != T×H×W`，说明存在尚未分类的非视频 token；
10. Stage 4 center-aligned LPIPS/shape smoke 未通过却仍被列为外观主证据；
11. 仅 3-case Stage 4A pilot 被用于显著性或总体机制宣称。

遇到硬停止条件，只完成诊断报告，不继续消耗大规模 GPU。

---

## 8. 已确认选择与下一道人工 Gate

Gate 0 的 Random100、seed split、held-out 策略、probe/rescue 范围和 ranking 稳定阈值已经确认并执行，
不再作为“待确认问题”重复列出。原始冻结选择保存在 `experiment_spec_latest3350.json`，不得覆盖。

Stage 4 的增补冻结在 `experiment_spec_stage4_temporal_v1.json`。下一次人工确认发生在：

1. Stage 4.0 CPU、真实 GPU smoke、LPIPS/shape 和 inventory 全部通过之后、启动 Stage 4A 之前；
2. Stage 4A 完成并给出方差/MDE/power 曲线之后、选择 Stage 4B case 数之前。

截至本文当前版本，不自动启动 Stage 4A GPU 矩阵。

---

## 9. Direct-attention multicase pilot（2026-08-13）

本轮用于复核单个 `001460 / seed 47326` 结论，不改变上面 M1/M2/M3 Stage 4 的冻结设计。

- 固定 5 个跨域 eligible cases：3 个 PyBullet、1 个 Kubric wall-collision、1 个 Physics-IQ；每个 case 固定一个预先筛选的 object target。
- 固定 seeds：`13248 / 47326 / 90094`。
- 每个 case-seed 生成 `Baseline + 3 head groups × 3 directions = 10` 个视频：Top100、Bottom100、layer-matched Random100 分别执行 Context Query→Future Key、Future Query→Context Key、Bidirectional direct-attention control。
- 总规模：5 cases × 3 seeds × 10 variants = 150 videos；已有焦点单元 10 个结果复用，预计新增 140 个。
- GPU2 负责 seeds `13248 / 47326`，空闲的 GPU1 负责 seed `90094`；两者要求显存连续三个 30 秒采样周期低于 12 GB 才启动，不停止已有任务。原 GPU3 等待队列在尚未生成任何结果时撤销。GPU4 禁用。
- 生成结束后在 GPU2 顺序执行官方 VBench 七项：Subject、Background、Temporal Flickering、Motion Smoothness、Dynamic Degree、Aesthetic、Imaging，共计划 1050 个 video-metric scores。
- 统计以 case 为最高独立单位：seed 先在 case 内平均，再对 case 等权。主指标是与同 case-seed Baseline 配对的 `ΔGT Center-ADE/D0`。
- 最优配置只有在 trajectory gate pass rate 不低于 Baseline−5pp，且 VBench Subject、Background、Imaging 均不低于 Baseline−0.02 时才有资格参与；合格配置中 paired ΔADE/D0 最低者胜出。FDE、PCK、Track Loss 和其余 VBench 作为解释/guardrail，不混成任意加权总分。
- 实时页面：`http://localhost:8092/gt-stc-direct-attention-multicase?v=1`；未完成生成、轨迹或 VBench 项明确显示 Pending。
- 计算入口：
  - `AAA_my_test/wan_context_point_guidance/launch_direct_attention_multicase_gpu2.sh`
  - `AAA_my_test/wan_context_point_guidance/launch_direct_attention_multicase_gpu1.sh`
  - `AAA_my_test/wan_context_point_guidance/launch_direct_attention_multicase_vbench_gpu2.sh`
