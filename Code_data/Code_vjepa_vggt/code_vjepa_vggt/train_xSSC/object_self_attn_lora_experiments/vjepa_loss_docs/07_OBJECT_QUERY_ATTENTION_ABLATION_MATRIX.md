# Object Query Attention 消融矩阵：固定 Q00 与全时序 Tube

## 1. 本文回答什么

本文统一说明两组使用同一批 Top PCK heads 的因果干预：

1. **固定 Q00 消融**：只把 F00 上的稀疏 object-query points 映射为 latent `t=0` token 集合。
2. **全时序 Object Query Tube 消融**：在同 seed、无干预 baseline 视频上冻结 CoTracker 轨迹，把同一批 object points 在 13 个 latent 时刻的位置合成一个时空 token 集合。

两组实验使用相同的模型、seed、Top100 heads、40 个去噪步、两个 CFG 分支以及 M1–M7/C1 算子。左右对照中改变的是被干预集合 `R`，不是消融公式。

> 重要限制：`R_tube` 通常远大于 `R_fixed`，因此左右差异同时包含“时间覆盖范围扩大”和“被消融 token 数增加”两部分。更强变化不能直接解释为 PCK head 对所有 query 帧都同样有效。

## 2. 先区分两条时间轴

| 时间轴 | 索引 | 本实验中的设置 | 含义 |
|---|---|---|---|
| 视频/latent 时间 | `t=0...12` | `Q00...Q12` 对应视频 `F00,F04,...,F48` | 决定哪些时空 token 属于 `R` |
| 扩散去噪时间 | `s=0...39` | `S000...S039` 全部执行 | 决定在生成过程的哪些 denoising steps 应用消融 |

“全时序 Tube”指覆盖全部 13 个 **latent 视频时刻**；“应用全部时间步”指干预覆盖全部 40 个 **去噪步骤**。二者不能混为一谈。

Wan 当前 latent token 网格为：

\[
T\times H\times W=13\times22\times40.
\]

49 帧生成视频并不存在 49 个独立 query 时间行；每 4 帧对应一个 latent anchor。

## 3. 两种 `R` 的精确定义

### 3.1 固定 Q00：`R_fixed`

设对象在 F00 的第 `p` 个稀疏点为 `(x_{0,p},y_{0,p})`，则：

\[
r_{0,p}
=
\left\lfloor\frac{y_{0,p}H}{704}\right\rfloor W
+
\left\lfloor\frac{x_{0,p}W}{1280}\right\rfloor,
\qquad
R_{\text{fixed}}=\operatorname{unique}_p(r_{0,p}).
\]

- 只包含 latent `t=0`。
- 是 F00 稀疏点的 token 代理，不是完整 object mask。
- 不把同一空间位置复制到其他 latent 帧。

### 3.2 全时序 Tube：`R_tube`

先在 seed 匹配的无干预 baseline 视频上，从 F00 的相同稀疏点启动 CoTracker。对每个 latent anchor `t`，取轨迹位置 `(x_{t,p},y_{t,p})`：

\[
r_{t,p}
=
tHW
+
\left\lfloor\frac{y_{t,p}H}{704}\right\rfloor W
+
\left\lfloor\frac{x_{t,p}W}{1280}\right\rfloor,
\]

\[
R_{\text{tube}}
=
\operatorname{unique}_{t,p}(r_{t,p}),
\qquad t=0,...,12.
\]

- 轨迹在干预前冻结；不会在消融生成结果上重新追踪。
- “轨迹 GT”实际是 baseline 生成视频上的 **CoTracker pseudo-GT**，不是数据集真实 GT。
- 当前 pilot 对所有 13 个 anchor 使用有限的 CoTracker 预测坐标；visibility 只写入审计，不用于删除 token。
- `R_tube` 是一个联合集合。因此 `A[R_tube,R_tube]` 同时含有帧内读取和跨帧读取，不等于 13 个彼此独立的逐帧消融。

### 3.3 `C` 随 `R` 一起变化

设完整 self-attention token 集合为 `N`：

\[
C_{\text{fixed}}=N\setminus R_{\text{fixed}},
\qquad
C_{\text{tube}}=N\setminus R_{\text{tube}}.
\]

所以左右两侧虽然执行相同的 M1–M7/C1，实际矩阵分区大小不同。

## 4. 共同的 Attention 分块

对一个物理 self-attention head：

\[
A=\operatorname{softmax}(QK^\top/\sqrt d),
\qquad
Y=AV.
\]

给定任一 `R∈{R_fixed,R_tube}`，定义：

| 分块 | 矩阵区域 | 信息流方向 |
|---|---|---|
| `S` | `A[R,R]` | `R K/V → R Query` |
| `I` | `A[R,C]` | `C K/V → R Query` |
| `O` | `A[C,R]` | `R K/V → C Query` |
| `B` | `A[C,C]` | `C K/V → C Query`；M1–M7 均保留 |

矩阵行代表接收信息的 Query，矩阵列代表被读取的 K/V。所谓“Query 行消融”是把该行的 `A@V` 更新置零，不是从序列中删除 token。

## 5. M1–M7：相同公式，不同 `R` 实例

未干预时两个输出分块为：

\[
Y_R=SV_R+IV_C,
\qquad
Y_C=OV_R+BV_C.
\]

下表中的撇号表示当前选中 head、当前去噪步和当前 CFG 分支经过干预后的 `A@V` 输出。它不表示完整 transformer block 的最终 token，因为残差、其他 heads 和 FFN 仍然存在。

| ID | 实现名 | 置零块与被切断流向 | 干预后的精确计算 | 理论后果 / 诊断问题 |
|---|---|---|---|---|
| M1 | `self_only` | `S=0`；`R K/V ──X──> R Query` | `Y'_R=IV_C`；`Y'_C=OV_R+BV_C` | 只删除 `SV_R`。检验 R 内部 Value 是否支持 R 接收端；Tube 中 `S` 同时包含帧内与跨帧 tube 内连接。 |
| M2 | `incoming_only` | `I=0`；`C K/V ──X──> R Query` | `Y'_R=SV_R`；`Y'_C=OV_R+BV_C` | 只删除 `IV_C`。R 仍可内部读取并继续向 C 输出；检验背景、其他对象和 tube 外 token 输入 R 的作用。 |
| M3 | `outgoing_only` | `O=0`；`R K/V ──X──> C Query` | `Y'_R=SV_R+IV_C`；`Y'_C=BV_C` | 只删除 `OV_R`。R 自身的读取不变；检验 R Value 向其余 token 广播的作用。若接近 baseline，只能说最终视频中的 O 边际效应较弱。 |
| M4 | `query_row` | `S=I=0`；`全部 K/V ──X──> R Query` | `Y'_R=0`；`Y'_C=OV_R+BV_C` | 删除该 head 对 R 的全部接收端更新。R Value 仍可被 C 读取；R token 本身不会被清零，因为残差和其他模块仍保留。 |
| M5 | `key_value_column` | `S=O=0`；`R Value ──X──> 全部 Query` | `Y'_R=IV_C`；`Y'_C=BV_C` | 保持原 softmax `A`，删除所有 R Value 贡献且不重归一化；严格等价于 K 不变、只令 `V_R=0`，不等价于 C1。 |
| M6 | `cross_boundary` | `I=O=0`；`C→R` 与 `R→C` 双向切断 | `Y'_R=SV_R`；`Y'_C=BV_C` | 隔离 R 与 C，但保留 R 内部和 C 内部通信；检验跨边界双向耦合的联合效应。 |
| M7 | `row_and_column` | `S=I=O=0`；所有涉及 R 的流向切断 | `Y'_R=0`；`Y'_C=BV_C` | 删除该 head 中 R 的全部接收与发送通信，只保留 C→C；仍不删除残差、其他 heads、FFN 或 cross-attention。 |

M1–M7 是 post-softmax `A@V` 分块置零且不重新归一化。它们只在固定二分集合 `{R,C}` 下构成完整的七种“涉及 R 的非空矩阵块组合”。

### 5.1 M1–M3 的时间分解实验

时间方向实验只对 `R_tube={R_0,...,R_12}` 定义，不生成 Fixed 版本。令矩阵行的 Query 时间为 `t_q`，列的 K/V 时间为 `t_k`：

- `future` 只删除 `t_q>t_k`，即过去 K/V 向未来 Query 的贡献；
- `past` 是反向控制，只删除 `t_q<t_k`，即未来 K/V 向过去 Query 的贡献；
- `same` 只删除 `t_q=t_k`，即同一 latent 时刻内的对应 S/I/O 连接，并保留全部跨时刻连接。

这里的 `t_q=t_k` 是“Query 与 K/V 属于同一 latent 帧”，不是只删除矩阵主对角线 `q=k`。例如 M1-same 会删除同一时刻内所有 `R_t Query × R_t K/V` 配对。

| ID | 实现名 | 精确删除项 | 诊断含义 |
|---|---|---|---|
| M1-same | `self_same` | `Σ_{t_k=t_q} A[R_tq,R_tk]V_Rtk` | 对象 tube 内同一时刻的自交互；跨帧 R→R 全部保留 |
| M2-same | `incoming_same` | `Σ_{t_k=t_q} A[R_tq,C_tk]V_Ctk` | 同一时刻环境/其他 token 向对象 Query 的输入 |
| M3-same | `outgoing_same` | `Σ_{t_k=t_q} A[C_tq,R_tk]V_Rtk` | 同一时刻对象 Value 向环境/其他 token 的广播 |
| M1-future | `self_future` | `Σ_{t_k<t_q} A[R_tq,R_tk]V_Rtk` | 对象历史状态向未来对象状态的传播 |
| M2-future | `incoming_future` | `Σ_{t_k<t_q} A[R_tq,C_tk]V_Ctk` | 历史背景/其他对象向未来对象状态的输入 |
| M3-future | `outgoing_future` | `Σ_{t_k<t_q} A[C_tq,R_tk]V_Rtk` | 历史对象状态向未来背景/其他对象的广播 |
| M1-past | `self_past` | `Σ_{t_k>t_q} A[R_tq,R_tk]V_Rtk` | M1-future 的未来→过去反向控制 |
| M2-past | `incoming_past` | `Σ_{t_k>t_q} A[R_tq,C_tk]V_Ctk` | M2-future 的未来→过去反向控制 |
| M3-past | `outgoing_past` | `Σ_{t_k>t_q} A[C_tq,R_tk]V_Rtk` | M3-future 的未来→过去反向控制 |

九项都先用原始 `Q/K` 和完整 K 序列保持 baseline softmax 分母，再从对应 Query 输出中减去指定时间块的 `A@V` contribution；不修改 Q/K 投影、不重新归一化。它们同样作用于 Top100 heads、全部 40 个去噪步和两个 CFG 分支。对每个基础块都有 `base = same ∪ future ∪ past` 的互斥完备时间分解；因此 Same、Future、Past 必须联合比较，避免把被删除连接数量或同帧局部交互误判成特定的过去→未来因果方向。

## 6. C1–C3：不要与矩阵分块混用

| ID | 实现名 | 精确计算逻辑 | 理论后果 / 与 M1–M7 的区别 |
|---|---|---|---|
| C1 | `literal_kv_zero` | `K'_R=V'_R=0`，`A'=softmax(QK'^T/√d)`，`Y'=A'V'` | R Value 为零，但 R 列 logits 变成 0 后仍进入 softmax 并占概率质量。与 M5 的差异测的是 K 改动带来的重新路由效应。 |
| C2 | `qk_logits_zero` | 对选中 head 的全部 token 令 `Q_h=0`；于是 `A_h=softmax(0)=1/N`、`Y_h=mean(V_h)` | 把该 head 变成全局均匀 Value 平均，不是零输出，也不依赖 R。 |
| C3 | `full_head_output` | 直接令选中 head 的整个 `Y_h=A_hV_h=0` | 删除整个 head 的输出，不改变其他 heads；不依赖 R，且与 C2 的均匀输出不同。 |
| Baseline | 无 | 保持原始 Q/K/V、softmax 和所有 head 输出 | 同 seed 的无干预参照。 |

C2、C3 不依赖 `R`，因此固定 Q00 与 Tube 对照不重复生成；页面共用已有控制视频。

## 7. 当前实验矩阵

### 7.1 固定 Q00 主矩阵

| 维度 | 水平 |
|---|---|
| Target scope | 每个 `single_object`；`all_objects` 并集 |
| Object-dependent operators | M1–M7、C1 |
| Head count | Top30、Top50、Top100；新增 9 case 当前优先 Top100 |
| Head selection | 冻结的 provisional S039 PCK ranking |
| Denoising | S000–S039 全 40 步 |
| CFG | conditional 与 unconditional |
| Seed | 新增 9 case 统一为 `47326` |

对具有 `n` 个对象且完整生成 Top30/50/100 的 case，视频数为：

\[
3\times\left(8\times(n+1)+2\right).
\]

### 7.2 `0613pybullet_sample_001460_w002` Tube pilot

| 项目 | 设置 |
|---|---|
| Case / seed | `0613pybullet_sample_001460_w002` / `47326` |
| 对象 | `object_A=sphere`、`object_B=box` |
| Target sets | `object_A`、`object_B`、`all_objects` |
| Heads | 同一冻结 Top100 |
| Operators | 每个 target 执行 M1–M7、C1 |
| 新视频数 | `3 targets × 8 operators = 24` |
| C2/C3 | 与 R 无关，复用固定实验，不重新生成 |
| CoTracker anchor visibility | object_A、object_B 均为 `99.04%`；仅审计，不作为 token 删除条件 |

已完成样例审计显示：

| Target | `|R_fixed|` | `|R_tube|` |
|---|---:|---:|
| object_A | 6 | 79 |
| object_B | 8 | 104 |
| all_objects | 14 | 179 |

object_A 的 13 个 latent 时刻分别为 `[6,7,6,8,6,7,6,5,6,6,5,6,5]`。这说明左右干预剂量并不相等；`all_objects` 的 179 小于 79+104，是因为两个对象轨迹映射后存在 token 重合并被联合去重。

### 7.3 五个新 seed 的重复实验

为区分算子效应与单次扩散随机性的偶然结果，`0613pybullet_sample_001460_w002` 追加五个独立 seed：`90094`、`68613`、`35075`、`32466`、`13248`。

| 维度 | 设置 |
|---|---|
| 每个 seed 的 baseline | 使用该 seed 已生成、未干预的视频；不跨 seed 共用 baseline |
| 固定协议 | `3 targets × 8 object-dependent operators = 24` 个 Top100 视频 |
| Tube 协议 | `3 targets × 8 object-dependent operators = 24` 个 Top100 视频 |
| Tube 时间分解 | `3 targets × 3 base blocks (M1/M2/M3) × 3 time blocks (Same/Future/Past) = 27` 个 Top100 视频 |
| 原 Fixed/Tube 规模 | `5 seeds × (24 Fixed + 24 Tube) = 240` 个视频 |
| 时间分解新增规模 | `5 seeds × 27 = 135` 个视频；扩展后五 seed 合计 `375` 个消融视频 |
| Head 排名 | 六个 seed 均使用与 seed 47326 相同的冻结 provisional Top100 排名，以固定被干预 head 这一变量 |
| Tube 轨迹 | 每个 seed 从自己的 baseline 独立运行 CoTracker、冻结轨迹并映射到 latent token；不复用 seed 47326 的 token 集合 |
| 采样控制 | 同一 seed 内 baseline、Fixed 与 Tube 保持相同扩散 seed；跨 seed 只改变采样随机性 |
| 时间分解计算资源 | 4-way task shard 使用 GPU 1/2/3/5；全程不使用 GPU 4 |

跨 seed 结论应比较同一 `protocol × target × operator` 相对各自 baseline 的效应分布。若多个 seed 中方向一致，可提高结果对采样随机性的稳健性；若差异显著，则说明当前最终视频后果具有 seed 依赖。它仍不能单独证明 Top100 对所有时刻都具有 tracking 特异性。

## 8. 左右对比应该如何解释

| 观察 | 可以支持的描述 | 不能直接声称 |
|---|---|---|
| Tube 比 fixed 变化更大 | 扩大到整条时空 tube 后，联合干预效应更强 | 每一帧的 PCK head 都同样准确；差异可能只是 token 数更多 |
| Tube 与 fixed 接近 | 额外时刻未显著增加最终视频差异，或模型存在冗余/饱和 | 只有 Q00 有效 |
| 只有 M1/M6 在 Tube 明显变化 | tube 内部或 tube–外部边界通信值得进一步检查 | 已证明某个确定的物理因果通路 |
| 不同对象差异不同 | 两个稀疏对象代理对这些 heads 的敏感度不同 | 球或盒子的完整语义表征已被定位 |

这个 Tube 消融是**因果干预实验**，但不直接回答“给定 Q10，Top PCK heads 在所有 K 帧的高响应是否落在轨迹上”。后者需要单独计算 `query time × key time` 的 13×13 响应/PCK 矩阵，并把峰值与 CoTracker pseudo-GT 对齐。

## 9. 视频相似度实测与“看起来一致”的原因

### 9.1 计算口径

单 seed 管线对 49 个视频统一计算：1 个 baseline、24 个固定 Q00 Top100 消融、24 个 Tube Top100 消融，共 624 组两两视频比较。当前已对同一个 case 的 6 个 seed 全部计算这套指标，即 `6 个 case-seed 样本×49 个视频=294 个视频`，其中 288 个为消融视频。

9.2–9.5 中已经写出的具体数值是早期 `seed=47326` 单 seed 诊断，用于解释指标和典型现象，**不是六 seed 均值**。当前可视化页面的四张指标表已改为 9.6 所定义的六 seed 严格共同 cohort 聚合。

| 指标 | 计算与作用 |
|---|---|
| Decoded equality | 对原始尺寸全部 49 个解码 BGR 帧及 shape 做 SHA-256；判断是否真正逐像素相同 |
| MAE | 全部帧缩放到 320×176 后的通道绝对误差均值，再除以 255；越低越相似 |
| PSNR | 由 320×176 像素 MSE 计算；越高越相似 |
| SSIM | OpenCV 解码 BGR 上计算逐帧三通道 SSIM，再取 49 帧均值；共同置换为 RGB 时数值等价；越高越相似 |
| `Δt-MAE` | 两视频相邻帧差分之间的 MAE；单独检查运动变化 |

结果中 **decoded equality 重复组为 0**：没有两个视频逐像素完全相同。因此页面中看起来一致的视频是“高相似但仍有差异”，不是同一 MP4 被重复引用。

### 9.2 每个算子的平均相似度

下表均在 object_A、object_B、all_objects 三个 target 上取平均：

| ID | Fixed vs baseline SSIM | Tube vs baseline SSIM | Fixed vs Tube SSIM | 直接读法 |
|---|---:|---:|---:|---|
| M1 | 0.9591 | 0.9661 | 0.9638 | 删除 S 后仍整体相似，但 Fixed object_A/all_objects 的变化相对更大 |
| M2 | 0.9668 | 0.9681 | 0.9697 | 只删 I 的最终视频变化较小 |
| M3 | **0.9693** | **0.9701** | **0.9770** | 三类平均中最接近参照；本 case 中只删 O 的可见边际效应最弱 |
| M4 | **0.9553** | 0.9646 | **0.9533** | Fixed 平均最偏离 baseline，Fixed–Tube 同算子差异也最大 |
| M5 | 0.9642 | 0.9683 | 0.9661 | 删除 R 的全部 Value 贡献有可见效应，但仍受其他路径补偿 |
| M6 | 0.9673 | 0.9664 | 0.9626 | 双向隔离并未简单等于 M2 或 M3，联合非线性传播仍存在 |
| M7 | 0.9635 | 0.9633 | 0.9587 | 所有涉及 R 的通信被删，Fixed–Tube 差异较明显 |
| C1 | 0.9670 | 0.9632 | 0.9706 | 重算 softmax 后仍较相似，但不能据此把 C1 当成 M5 |

单项极值也与这一趋势一致：

- 最接近 baseline：Tube object_B M3，SSIM 0.973924、MAE 0.005303。
- 最偏离 baseline：Fixed all_objects M4，SSIM 0.943443、MAE 0.009870。
- 同 target 的最相似算子对：Fixed object_B M1↔M5，SSIM 0.983535、MAE 0.003441；两者只相差额外删除 `O`。
- 同算子 Fixed↔Tube 最相似：M3 的 object_A，SSIM 0.979184。

### 9.3 用嵌套算子定位“弱边际信息流”

M1–M7 具有严格的集合嵌套关系。选择只相差一个矩阵块的算子对，可以把高相似度解释为该块在对应干预上下文中的**最终视频边际效应较弱**：

| 只相差的块 | 配对 |
|---|---|
| `S` | Baseline↔M1、M2↔M4、M3↔M5、M6↔M7 |
| `I` | Baseline↔M2、M1↔M4、M3↔M6、M5↔M7 |
| `O` | Baseline↔M3、M1↔M5、M2↔M6、M4↔M7 |

四组配对的 SSIM 均值如下；数值越高表示“再多切这个块”后最终视频越相似：

| 协议 | Target | 仅差 S | 仅差 I | 仅差 O |
|---|---|---:|---:|---:|
| Fixed | object_A | 0.9642 | 0.9725 | 0.9712 |
| Fixed | object_B | 0.9749 | 0.9767 | **0.9794** |
| Fixed | all_objects | 0.9631 | **0.9706** | 0.9667 |
| Tube | object_A | 0.9671 | 0.9683 | **0.9686** |
| Tube | object_B | 0.9689 | 0.9675 | **0.9710** |
| Tube | all_objects | 0.9662 | **0.9686** | 0.9686 |

这说明不同 target/协议没有一个全局恒定的“无用块”，但 object_B 的 O 边际效应在 Fixed 和 Tube 中都相对较弱。必须注意：这里诊断的是经过残差网络和完整扩散轨迹传播后的最终像素效应，**不能反推该 attention block 的内部响应为零**。

### 9.4 为什么不同消融仍可能看起来一致

1. `R_fixed` 只有 6–14 个 token，`R_tube` 也只有 79–179 个 token，而完整序列有 `13×22×40=11440` 个 token；变化可能集中在肉眼很难发现的小区域。
2. 仅干预 Top100/720 个物理 self-attention heads；其余 heads、残差、FFN、cross-attention 和后续扩散步可提供替代路径。
3. 嵌套算子可能只多删除一个边际贡献较弱的块。例如 Fixed object_B 的 M1 与 M5 只相差 O，实测 SSIM 0.9835。
4. C1 与 M5 在视觉上可接近，但计算不等价。若 R 列原概率质量较小，或 K 置零引起的 softmax 重路由较弱，C1 的额外影响可能不明显。
5. SSIM/PSNR 是全图指标，局部对象、接触时刻或短暂轨迹差异会被大面积相同背景稀释；应同时逐帧播放、看局部 crop，并结合 `Δt-MAE`。
6. 所有实验使用同 seed 和相同采样条件，本来就共享大部分生成结构；高相似是预期现象之一，不等于 hook 未执行。是否执行必须结合 manifest 中的 40 步×2 CFG×Top100 审计判断。

### 9.5 RAFT 运动相似度：能否识别“画面相似但运动不同”

可以。本次进一步对全部 49 个视频提取 RAFT 光流：每个视频计算 `F00→F01, ..., F47→F48` 共 48 个 forward flow field。模型固定为 torchvision RAFT Large `C_T_SKHT_V2`，输入统一缩放到 640×352，12 次迭代、FP32、确定性 CUDA 算法。光流是模型估计，不是真实 GT。

为了避免全图静态背景淹没小物体运动，指标在 object_A、object_B、all_objects 等空间范围内计算。ROI 必须相对当前 reference 冻结，而不是从每个待测消融视频重新跟踪：

- `vs Baseline`：使用 Baseline CoTracker 轨迹逐帧求凸包，并在 640×352 分辨率膨胀 6 px。
- `vs GT`：使用 source render CoTracker 轨迹按相同规则构造 source-reference ROI。

同一个 reference 下的 48 个消融结果共享完全相同的 ROI，因此不会把候选视频自身的跟踪失败混入 ROI 定义。不能让 `vs GT` 继续沿用 Baseline ROI：当 Baseline 与 source 轨迹已分离时，那样会在 source 视频中采到背景而不是 GT 对象运动。

| 指标 | 精确含义 | 读法 |
|---|---|---|
| Flow EPE | `mean ||u_reference - u_candidate||₂` | 两个估计光流场的空间、方向和幅值联合差异；越低越相似。即使 reference 是 source render，这仍是 RAFT 跨视频 disagreement，不是光流真值 EPE |
| EPE/ref | `Flow EPE / mean ||u_reference||₂` | 对当前 reference 运动量归一化；reference 近静止时不稳定 |
| Flow vector cosine | 把 ROI 内全部光流向量展平后的 cosine | 越接近 1，整体方向场越一致；近零光流时方向不稳定 |
| Motion/ref | `mean ||u_candidate||₂ / mean ||u_reference||₂` | 约 1 表示平均运动幅值接近 reference，约 0 表示候选运动几乎消失；不能单独判断方向和位置是否正确 |
| Motion-profile correlation | 两视频 48 个逐帧平均光流幅值序列的 Pearson `r` | 判断运动发生时机是否一致；只看强弱时序，不保证空间位置一致 |

旧的专项 `raft_motion_top100_v1` 中，240 组 RAFT 比较由 48 个 `vs Baseline`、24 个同算子 `Fixed vs Tube`、168 个同协议同 target 的算子两两比较组成。新的完整指标 `report.json` 另为每个消融计算 `2 references × 3 scopes`，即 `48 × 2 × 3 = 288` 组 reference-relative RAFT 汇总；其中 GT 侧采用 source-reference ROI。两套统计回答的问题不同，不能混成同一张结果行。

Baseline 自身的估计运动量证明两个对象的动力学角色明显不同：

| 范围 | Mean magnitude (px/frame) | P95 magnitude (px/frame) | 解释 |
|---|---:|---:|---|
| Global | 0.146970 | 0.083698 | 大部分背景接近静止；少量高速像素使 mean 高于 P95 |
| object_A ROI | **2.670425** | **7.237125** | object_A 是主要运动物体 |
| object_B ROI | 0.131149 | 0.248365 | object_B 接近静止；倍率和方向指标需谨慎 |
| all_objects ROI | 0.583361 | 4.006410 | 被较大的近静止 object_B ROI 稀释 |

object_A 是更可靠的运动诊断对象。下表给出每个实验相对 baseline 的平均运动幅值比例与归一化 EPE：

| ID | Fixed Motion/base | Fixed EPE/ref | Tube Motion/base | Tube EPE/ref | 主要观察 |
|---|---:|---:|---:|---:|---|
| M1 | **0.014** | 0.996 | 0.562 | 0.868 | Fixed 删除 `S` 后 object_A 运动几乎消失；Tube 仍保留约 56% 幅值 |
| M2 | 0.583 | 0.926 | 0.619 | 0.858 | 两侧均保留约六成幅值，但光流场仍明显偏离 baseline |
| M3 | 0.714 | 0.845 | **0.799** | 0.923 | 幅值保留较多，不代表空间/方向场正确；Tube 的 EPE/ref 仍高 |
| M4 | **0.016** | 0.993 | 0.354 | 0.882 | Fixed 删除 R Query 行后运动同样几乎消失；Tube 保留约 35% |
| M5 | 0.389 | 0.987 | 0.449 | 0.978 | 保持原 softmax、删除 R Value 后，两侧只剩约四成运动 |
| M6 | 0.471 | 0.968 | 0.505 | 0.944 | 双向跨边界隔离后约保留一半幅值 |
| M7 | 0.473 | 0.997 | 0.381 | 0.966 | 删除所有涉及 R 的连接后方向/空间场仍高度偏离 |
| C1 | **0.774** | **0.740** | 0.660 | 0.984 | Fixed C1 比 Fixed M5 更保留运动，实证表明 softmax 重路由不能与 post-softmax 列删除混同 |

最关键的反例是 Fixed object_A M1/M4：其全图像素 SSIM 仍分别为 0.9547/0.9519，肉眼可能觉得主体画面相近，但 object_A 的 RAFT Motion/base 只有 0.014/0.016。也就是说，两项干预几乎消除了球的运动，而全图 SSIM 被静态背景和共同外观稀释。Tube M1/M4 则分别保留 0.562/0.354，说明 `R_fixed` 与 `R_tube` 的干预范围确实造成了不同运动后果。

RAFT 也使“不同消融为何像同一个结果”更容易定位：Fixed object_A 的 M1↔M4 在 ROI 内 Flow EPE 仅 0.0416 px，两者都把球压到近静止状态；此时 flow cosine 只有 0.190，不是运动差异很大，而是两个近零向量的方向本来就不稳定。Fixed object_B 的 M1↔M5 则同时具有 Flow EPE 0.0506 px、flow cosine 0.979、motion-profile `r=0.982`，与其像素 SSIM 0.9835 一致，支持“在这个近静止对象和该干预上下文中，额外删除 O 的最终边际效应较弱”。相反，同算子 Fixed↔Tube 不能仅凭画面判定相同：object_A M1/M4 的 Tube/Fixed 运动倍率分别达到 40.7×/21.5×，主要原因是 Fixed 分母已经接近零。M3 是 object_A 中 Fixed↔Tube 运动场最接近的一项，flow cosine 0.862、motion-profile `r=0.885`，但 Flow EPE 仍为 0.963 px。

48 个 `vs baseline` 实验中，像素 SSIM 与 target ROI 运动指标的相关性为：

| 配对 | Pearson | Spearman |
|---|---:|---:|
| SSIM vs Flow EPE | -0.287 | -0.611 |
| SSIM vs Flow vector cosine | 0.424 | 0.268 |
| SSIM vs Motion-profile correlation | 0.519 | 0.569 |

相关方向总体合理，但绝对值远小于 1，说明外观相似度与运动相似度相关而不等价。RAFT 因而能补充判断生成运动是否保持。`vs Baseline` 仍不能证明运动“物理正确”；新增 `vs GT` 可以检查与 source render 运动的差异，但 RAFT 本身仍是估计器，不应称为真实 flow GT。

页面中的 HSV 光流视频使用 `max(P99.5_baseline, P99.5_source)` 作为统一颜色上限；当前结果为 6.616 px。色相表示方向、亮度表示幅值。统一尺度允许 Fixed、Tube、Baseline 与 source render 直接目测对照；不要把黑暗区域自动解释为失败，它也可能表示真实的近零运动。overlay 上的橙/青轮廓就是该行实际进入 RAFT 指标计算的 reference-frozen Object A/B ROI。

### 9.6 指标曲线与数值审计的展示规范（修正版）

当前页面的统计样本严格固定为：

| 项目 | 当前口径 |
|---|---|
| 真实 case 数 | `1`：`0613pybullet_sample_001460_w002` |
| Seed / case-seed 样本数 | `6`：`13248, 32466, 35075, 47326, 68613, 90094` |
| 每个样本的视频 | `1 Baseline + 24 Fixed + 24 Tube = 49` |
| 总视频数 | `6×49=294`，其中消融视频 `6×48=288` |
| 表格实验行 | `48`个唯一 `protocol × target × operator`；每张表按 protocol 分为 24 行 |
| 聚合单位 | case-seed 宏平均；每个 seed 先与自己的同 seed Baseline 比较 |
| 共同 cohort 约束 | 所有展示的标量都必须有同一批 `N=6` 的有限值才计算均值；任一 seed 缺失或不可定义就显示 `—/N/A`，禁止按列缩小分母 |
| 视频与 overlay | 只用 `seed=47326` 作为可视化代表；它们不是六 seed 平均图像，不改变表格的 `N=6` 口径 |

严格聚合可写成

\[
\bar m_e=\frac{1}{6}\sum_{s\in\{13248,32466,35075,47326,68613,90094\}}m_{e,s},
\]

其中 \(e\) 是一个固定的 `protocol × target × operator × reference × metric-object`。只有 6 个 \(m_{e,s}\) 全部为有限值时才显示 \(\bar m_e\)；否则显示 `N/A`。因此页面上不会出现某一列是 6 个 seed 平均、另一列却只是 3–5 个 seed 平均的情况。需要注意，这个口径衡量的是**同一 case 上跨采样 seed 的稳健性**，不是跨 6 个不同场景的泛化均值。

页面顶部设置“六 seed 实验结论”区。其中的直接效应、跨对象传播、关键算子对比、GT 变化和全帧 SSIM 数值均在页面加载时从当前聚合报告动态计算，不把数值手写死在页面中。该区必须同时显示“单一 case、六 seed”的结论边界。

主展示改为“每个指标一张曲线图”，`vs Baseline` 和 `vs GT` 分成两个独立图组。每张图使用相同的视觉编码：

| 视觉元素 | 精确含义 |
|---|---|
| 横轴 1–8 | 固定对应 `M1, M2, M3, M4, M5, M6, M7, C1` 八个消融算子 |
| 红色 | 被消融 target 是 `object_A` |
| 青色 | 被消融 target 是 `object_B` |
| 紫色 | 被消融 target 是 `all_objects` |
| 实线 | Tube `R_tube` |
| 虚线 | Fixed `R_fixed` |
| 线上单点 | 对应 `protocol × target × operator` 在严格共同 cohort 上的 `N=6` 均值 |
| 断线 | 该点有任一 seed 缺失或不可定义，因此不使用更小分母聚合 |

页面顶部的 `Metric object` 选择器只决定“评估 object_A 还是 object_B”；曲线颜色表示的始终是**被消融 target**，两者不能混淆。对 #8/#9/#11/#13/#20–#24 这些含多个子量的复合指标，同一指标卡右上角提供分量切换，不把不同量纲画在同一纵轴。

原四张宽表作为折叠的“数值审计表”保留，便于查看每个精确数值：

| 审计表 | 行数 | 每一行的唯一键 |
|---|---:|---|
| Fixed `R_fixed` vs Baseline | 24 | `fixed × target × operator × Baseline` |
| Fixed `R_fixed` vs GT | 24 | `fixed × target × operator × GT` |
| Tube `R_tube` vs Baseline | 24 | `tube × target × operator × Baseline` |
| Tube `R_tube` vs GT | 24 | `tube × target × operator × GT` |

因此一组消融在逻辑上仍有 Baseline 与 GT 两个 reference 结果，但它们位于不同曲线图组与不同审计表，不进行交错或平均。数值审计例如：

| Experiment ID | Reference | #1 GT Center-ADE Change | #2 Baseline Center-ADE | #7 Center-FDE | #14 RAFT ROI EPE | #16 DINO Similarity | #17 Object LPIPS | #24 SSIM/PSNR/MAE | ... |
|---|---|---:|---:|---:|---:|---:|---:|---|---|
| `fixed:single_object:object_A:self_only` | Baseline | — | 数值 | Baseline-relative | Baseline-ROI | Baseline crop | Baseline crop | Baseline frame | ... |
| `fixed:single_object:object_A:self_only` | GT | GT error change | — | simulator GT | source-ROI | source crop | source crop | source frame | ... |

这两行不能合并或取平均，因为它们回答不同问题：

- `vs Baseline`：同 seed 未消融结果是反事实参照，衡量“干预造成了多大变化”。
- `vs GT`：中心、速度、接触使用 `states.npz` simulator GT；没有 simulator 对应定义的点轨迹、mask、外观、像素和光流使用 source render 前 49 帧，衡量“结果离物理/source reference 多远”。

对象级指标由页面的 `Metric object` 选择 `object_A` 或 `object_B`，但行键始终保留被消融的 target。这样可以明确区分“消融 A 后评估 A”和“消融 A 后评估 B（对外传播）”，不能把两个对象的分数无定义地平均。

不适用于某个 reference 的指标显示 `—`：例如 #1 只在 GT 行有意义，#2 与 #6 只在 Baseline 行有意义。若 CoTracker 有效点不足或未检测到持续接触，也显示 `—/N/A`，不得用 0 填补。指标定义与数值结果分成两张表；定义表只解释公式、方向与对应 overlay，不再承载实验结果。

#### #1–#25 指标含义与读法

箭头表示数值方向，但“更大/更小”不总等于“生成质量更好”。尤其 #2、#6 衡量干预效应强度，#15 的目标是接近 1，#20–#23 只作生成质量 sanity check。

| # | 指标 | 表示什么 | 数值如何解释 |
|---:|---|---|---|
| 1 | GT Center-ADE Change | 消融相对 simulator GT 的中心轨迹误差，减去 Baseline 的同一误差 | **越小越好**；`0` 表示未改变 Baseline 的 GT 误差，正值表示轨迹变差，负值表示改善 |
| 2 | Baseline-relative Center-ADE | 消融中心轨迹与同 seed Baseline 中心轨迹的平均距离 | **越大表示干预可见效应越强**，但不表示物理上更差 |
| 3 | GT Velocity Error Change | 消融相对 GT 的四帧差分速度向量误差，减去 Baseline 的同一误差 | **越小越好**；正值表示速度物理误差增加，负值表示改善 |
| 4 | Contact-time Error Change | 候选 mask 接触时刻相对 simulator 接触时刻的误差，减去 Baseline 的同一误差 | **越小越好**；单位 frame；正值更差，未形成持续接触记 N/A |
| 5 | Post-contact Velocity Error Change | GT 接触后 8 帧窗口内，候选与 Baseline 相对 GT 的速度误差差 | **越小越好**；正值表示碰撞后运动更差 |
| 6 | Other-object Center-ADE | 单对象消融后，未被选中对象相对 Baseline 的轨迹变化 | **越大表示跨对象传播/spillover 越强**，不是质量分数 |
| 7 | Center-FDE | 最后共同有效帧的对象中心到所选 reference 的距离 | **越小越接近 reference** |
| 8 | Object-normalized PCK@5/10/20% | 点误差小于 F00 对象 bbox 对角线 `5%/10%/20%` 的比例 | **越大越接近 reference** |
| 9 | Native PCK@16/32/64 | 在 1280×704 输出坐标中，点误差小于 `16/32/64 px` 的比例 | **越大越接近 reference**；不是 Attention Q→K PCK |
| 10 | Point-ADE | 所有共同可见 CoTracker 表面点的平均距离 | **越小越接近 reference**；球体滚动会改变表面点对应关系 |
| 11 | Velocity Speed / Direction / Vector Error | 四帧差分速度的大小误差、方向角误差和完整向量误差 | 三项均 **越小越接近 reference**；方向只统计双方非静止帧 |
| 12 | Center-aligned Shape IoU | 只平移质心、不缩放后，两对象 SAM2 mask 的 IoU | **越大表示形状越接近**；`1` 为完全重合 |
| 13 | Area / Aspect / Circularity Error | mask 面积对数比、bbox 长宽比对数比及圆度差 | 三项均 **越小越接近 reference**；`0` 最好 |
| 14 | RAFT ROI Flow EPE | 在 reference-frozen ROI 内，候选与 reference RAFT 光流的端点差 | **越小越接近 reference**；是两个估计光流的 disagreement，不是真实 flow GT |
| 15 | RAFT Motion Magnitude Ratio | 候选 ROI 平均运动幅值除以 reference ROI 平均运动幅值 | **越接近 1 越相似**；`<1` 表示运动量减少，`>1` 表示增加；reference 近静止时不稳定 |
| 16 | Object DINOv2 Similarity | 质心对齐、固定 crop、mask pooling 后的 DINOv2 cosine | **越大表示对象身份/语义外观越接近**；理论上 1 最相似 |
| 17 | Object LPIPS | 质心对齐、mask 外置灰后的对象 crop LPIPS-Alex | **越小表示局部纹理和形状越接近**；0 最相似 |
| 18 | Outside-object LPIPS | 排除膨胀对象 mask 并集后，剩余区域的 LPIPS | **越小表示背景/非对象区域 spillover 越弱** |
| 19 | Raw-mask IoU | 不做对齐的候选/reference SAM2 mask IoU | **越大越接近**；同时混合对象位置与形状变化 |
| 20 | VBench Subject Consistency | 官方 VBench 主体跨帧一致性 | **越大越一致**；冻结视频也可能得高分，只作 sanity check |
| 21 | VBench Motion Smoothness | 官方 VBench/AMT 运动平滑度 | **越大越平滑**；不表示方向或物理结果正确 |
| 22 | VBench Dynamic Degree | 官方 VBench 对视频是否具有足够运动的分数 | **越大表示更动态**，只用于识别冻结；不是物理正确性分数 |
| 23 | VBench Quality Suite | Background、Flicker、Imaging、Aesthetic 四项官方分数 | 各项通常 **越大越好**，仅检查生成崩坏和视觉质量 |
| 24 | Full-frame SSIM / PSNR / MAE | 候选与 reference 的全帧像素相似度 | **SSIM/PSNR 越大越相似，MAE 越小越相似**；静态背景会稀释对象差异 |
| 25 | Temporal Δ-MAE | 两视频相邻帧差分之间的平均绝对误差 | **越小表示逐帧变化模式越接近** |

## 10. Head 排名与因果解释限制

1. Top heads 只由 S039 positive-conditional 的 `Q00 → K01...K12` PCK 排名选出。
2. Tube 实验把这些 heads 扩展到 Q00–Q12，但这只是测试其干预效应，不等于重新验证每个 query 时刻的 tracking accuracy。
3. 干预覆盖全部 40 个 denoising steps 和两个 CFG 分支；不能假设这些 head 在早期步骤或 unconditional 分支仍是 Top tracking heads。
4. 排名是 aggregate 未完成时冻结的 provisional snapshot；特异性结论仍需 random、bottom 或 layer-matched controls。
5. 视频结果经过残差、其他 heads、FFN、cross-attention 和扩散动力学传播。身份漂移、碰撞改变等只能是待检验解释，不能写成理论必然结果。

## 11. 实现、输出与页面

- 固定 Q00 脚本：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations.py`
- 全时序 Tube 脚本：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_legacy_ti2v_temporal_object_tube_ablations.py`
- 视频相似度脚本：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/analyze_legacy_ti2v_object_ablation_video_similarity.py`
- 五 seed 页面 manifest 构建脚本：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/build_legacy_object_ablation_001460_5seed_manifest.py`
- 五 seed GPU worker：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_legacy_object_ablation_001460_5seed_gpu.sh`
- 五 seed 相似度等待器：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/wait_legacy_object_ablation_001460_5seed_similarity.sh`
- RAFT 运动相似度脚本：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/analyze_legacy_ti2v_object_ablation_raft_motion.py`
- 单 seed 49 视频完整指标代码：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/object_query_ablation_metrics/`
- 全指标一键入口：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/object_query_ablation_metrics/bench.sh`；输入单个 `seed_<seed>` 结果目录，或包含多个直接 `seed_*` 子目录的 case 目录。
- 六 seed 严格共同 cohort 聚合脚本：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/object_query_ablation_metrics/aggregate_reports.py`
- 六 seed 聚合报告：`/data/gaoya/agent-data/outputs/object_query_ablation_metrics/0613pybullet_sample_001460_w002/aggregate/report.json`
- 标量完整性审计：`/data/gaoya/agent-data/outputs/object_query_ablation_metrics/0613pybullet_sample_001460_w002/aggregate/scalar_completeness.csv`；只有 `finite_sample_count=expected_sample_count=6` 的标量才可显示数值。
- 单 seed 报告：同一输出根目录下的 `seed_<seed>/report.json` 和 `seed_<seed>/summary.csv`。
- 固定 Q00 输出：`/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_v2`
- Tube 输出：`/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1`
- Tube 轨迹审计：Tube 输出目录下的 `frozen_baseline_tracks/tracks.npz` 与 `manifest.json`
- 相似度明细：Tube case/seed 输出目录下的 `video_similarity_top100.json` 和 `video_similarity_top100.csv`
- RAFT 缓存与明细：Tube case/seed 输出目录下的 `raft_motion_top100_v1/flows`、`flow_videos`、`raft_motion_similarity_top100.json` 和 `raft_motion_similarity_top100.csv`
- 新 seed 页面入口：保持同一 case，把 URL 的 `seed` 改为 `90094`、`68613`、`35075`、`32466` 或 `13248`；页面会只展示当前已经生成的卡片。
- 对比页面：`http://localhost:8092/wan22-ti2v-legacy-physiciq67-samples?v=20&case=0613pybullet_sample_001460_w002&seed=47326`。页面对 `object_A`、`object_B`、`all_objects` 分别建立独立 section，每个 section 内将 `Fixed R_fixed` 与 `Tube R_tube` 拆成两条横向视频行，并为每条行设置可见、可拖动的水平滑动条。C2/C3 显示在独立的 `Global all-token controls` 行；未生成项不占位，页面顶部会分别标出 Fixed/Tube 的实时完成数。视频卡片下方的 VBench、像素相似度、RAFT 指标和光流详情暂时隐藏，但指标数据及页面上方汇总分析仍保留。seed 47326 完整口径为 48 个 R-dependent 视频加 2 个全局控制视频。
- 指标曲线页面：`http://localhost:8092/object-query-ablation-metrics?v=7`。页面顶部先从当前聚合报告动态生成六 seed 结论摘要；主视图将 `vs Baseline` 与 `vs GT` 分开，每个指标一张曲线图；横轴 1–8 为 M1–M7/C1，红/青/紫分别为被消融 target A/B/all，Tube 为实线，Fixed 为虚线。复合指标可在卡片内切换分量；四张原始宽表折叠保留作数值审计。所有可显示标量是同一批 6 个 case-seed 样本的宏平均，不完整点断线并显示 `N/A`。页面中的视频与 trajectory/mask/RAFT/pixel/perceptual overlay 仅用 `seed=47326` 作为代表样本。

每个 manifest 必须记录：`target_scope`、`mask_mode`、冻结 Top100 entries、实际 token indices、逐 latent token 数、40 步双 CFG 调用审计、轨迹来源以及 softmax 是否重算。

## 12. 重跑命令

### 12.1 一键计算全部 #1–#25 指标与 overlay

输入目录必须已有 `video_similarity_top100.json`，且其 inventory 严格包含 `1` 个同 seed Baseline、`24` 个 Fixed Top100 和 `24` 个 Tube Top100 视频。默认流水线依次补齐官方 VBench、CoTracker、SAM2、候选/source-render RAFT、DINOv2/LPIPS、非神经指标及全部 overlay，随后执行完整性校验并重建共同 seed 汇总；各阶段都有内容哈希/文件缓存，可以断点续跑。

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
GPU=5 bash AAA_my_test/object_query_ablation_metrics/bench.sh \
  /data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1/0613pybullet_sample_001460_w002/seed_47326
```

若输入上一级 case 目录，则顺序处理其所有直接 `seed_*` 子目录：

```bash
GPU=5 bash AAA_my_test/object_query_ablation_metrics/bench.sh \
  /data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1/0613pybullet_sample_001460_w002
```

正式运行前可用 `--dry-run` 完成输入、Baseline、source render、`states.npz`、region cache 与 frozen tracks 校验，并打印将执行的全部命令。GPU 参数是物理编号，禁止使用 GPU 4。默认报告写入 `/data/gaoya/agent-data/outputs/object_query_ablation_metrics/<case>/seed_<seed>/`，汇总写入同一 case 下的 `aggregate/`。

### 12.2 生成与单阶段重跑

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
CUDA_VISIBLE_DEVICES=3 /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u AAA_my_test/run_legacy_ti2v_temporal_object_tube_ablations.py --case 0613pybullet_sample_001460_w002 --seed 47326 --device cuda
```

脚本按 `complete.json + manifest.json + generated.mp4` 自动跳过已经完成的任务，可安全恢复。

相似度重算不使用 GPU：

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
/data/gaoya/miniconda3/envs/wan/bin/python -u AAA_my_test/analyze_legacy_ti2v_object_ablation_video_similarity.py --workers 6
```

五个新 seed 的完整批次由一个 tmux session 管理；GPU 0/1/2/3 分担 Tube 生成，随后各自继续 Fixed 分片。相似度 window 会等待每个 seed 的 24 个 Fixed 和 24 个 Tube 视频全部完成，再为每个 seed 自动写入 `video_similarity_top100.json` 与 `.csv`：

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
bash AAA_my_test/start_legacy_object_ablation_001460_5seed_tmux.sh
tmux attach -t legacy_object_ablation_001460_5seed
```

五个新 seed 的 M1/M2/M3 × Same/Future/Past 使用同一 Tube runner 和每个 seed 自己冻结的 CoTracker 轨迹。四个 worker 在每个 seed 的 27 个任务上做稳定切片，已有 `complete.json` 会自动跳过：

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
bash AAA_my_test/run_legacy_temporal_sfp_multiseed_001460_gpu.sh 1 0 4
```

实际批次同时在 GPU 1/2/3/5 上运行 worker 0/1/2/3；页面 catalog 对六个 seed 共审计 `6 × 27 = 162` 个时间分解结果。

RAFT 首次提取需要 GPU，后续会复用 `/data` 下的 float16 flow cache。不得使用 GPU 4；下面示例使用 GPU 3：

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3 /data/gaoya/miniconda3/envs/wan/bin/python -u AAA_my_test/analyze_legacy_ti2v_object_ablation_raft_motion.py --device cuda --batch-size 4
```
