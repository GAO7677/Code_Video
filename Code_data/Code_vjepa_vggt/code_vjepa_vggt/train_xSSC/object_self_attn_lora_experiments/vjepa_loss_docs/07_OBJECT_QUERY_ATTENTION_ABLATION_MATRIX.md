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
| 总规模 | `5 seeds × (24 Fixed + 24 Tube) = 240` 个新视频 |
| Head 排名 | 六个 seed 均使用与 seed 47326 相同的冻结 provisional Top100 排名，以固定被干预 head 这一变量 |
| Tube 轨迹 | 每个 seed 从自己的 baseline 独立运行 CoTracker、冻结轨迹并映射到 latent token；不复用 seed 47326 的 token 集合 |
| 采样控制 | 同一 seed 内 baseline、Fixed 与 Tube 保持相同扩散 seed；跨 seed 只改变采样随机性 |
| 计算资源 | GPU 0/1/2/3 并行；明确不使用 GPU 4 |

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

本次对同一个 case/seed 的 49 个视频统一计算：1 个 baseline、24 个固定 Q00 Top100 消融、24 个 Tube Top100 消融，共 624 组比较。

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

为了避免全图静态背景淹没小物体运动，指标同时在四个空间范围计算：全图、object_A、object_B、all_objects。对象 ROI 不是从每个待测视频重新跟踪，而是固定使用同一 baseline CoTracker 轨迹：对每个源帧的轨迹点求凸包，并在 640×352 分辨率膨胀 6 px。这样不同消融视频使用完全相同的动态 ROI，不会把待测结果的跟踪失败混入 ROI 定义。

| 指标 | 精确含义 | 读法 |
|---|---|---|
| Flow EPE | `mean ||u_baseline - u_candidate||₂` | 两个估计光流场的空间、方向和幅值联合差异；越低越相似。这里是跨视频 disagreement，不是相对 GT 的 EPE |
| EPE/ref | `Flow EPE / mean ||u_baseline||₂` | 对 baseline 运动量归一化，便于比较不同 ROI；baseline 近静止时不稳定 |
| Flow vector cosine | 把 ROI 内全部光流向量展平后的 cosine | 越接近 1，整体方向场越一致；近零光流时方向不稳定 |
| Motion/base | `mean ||u_candidate||₂ / mean ||u_baseline||₂` | 约 1 表示平均运动幅值保持，约 0 表示运动几乎消失；不能单独判断方向和位置是否正确 |
| Motion-profile correlation | 两视频 48 个逐帧平均光流幅值序列的 Pearson `r` | 判断运动发生时机是否一致；只看强弱时序，不保证空间位置一致 |

240 组 RAFT 比较由 48 个 `vs baseline`、24 个同算子 `Fixed vs Tube`、168 个同协议同 target 的算子两两比较组成。与像素分析的 624 个全 pair 不同，RAFT 只保留能直接回答实验问题的配对。

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

相关方向总体合理，但绝对值远小于 1，说明外观相似度与运动相似度相关而不等价。RAFT 因而能补充判断生成运动是否保持。不过它仍不能证明运动“物理正确”：当前只比较消融结果是否接近未干预 baseline；若 baseline 本身不物理，RAFT 也不会将其纠正为真实运动。

页面中的 HSV 光流视频统一使用 baseline 全图光流幅值 99.5% 分位 6.616 px 作为颜色上限：色相表示方向、亮度表示幅值。统一尺度允许 Fixed、Tube 与 baseline 直接目测对照；不要把黑暗区域自动解释为失败，它也可能表示真实的近零运动。

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
- 固定 Q00 输出：`/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_v2`
- Tube 输出：`/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1`
- Tube 轨迹审计：Tube 输出目录下的 `frozen_baseline_tracks/tracks.npz` 与 `manifest.json`
- 相似度明细：Tube case/seed 输出目录下的 `video_similarity_top100.json` 和 `video_similarity_top100.csv`
- RAFT 缓存与明细：Tube case/seed 输出目录下的 `raft_motion_top100_v1/flows`、`flow_videos`、`raft_motion_similarity_top100.json` 和 `raft_motion_similarity_top100.csv`
- 新 seed 页面入口：保持同一 case，把 URL 的 `seed` 改为 `90094`、`68613`、`35075`、`32466` 或 `13248`；页面会只展示当前已经生成的卡片。
- 对比页面：`http://localhost:8092/wan22-ti2v-legacy-physiciq67-samples?v=19&case=0613pybullet_sample_001460_w002&seed=47326`。页面对 `object_A`、`object_B`、`all_objects` 分别建立独立 section，每个 section 内将 `Fixed R_fixed` 与 `Tube R_tube` 拆成两条横向视频行，并为每条行设置可见、可拖动的水平滑动条。C2/C3 显示在独立的 `Global all-token controls` 行；未生成项不占位，页面顶部会分别标出 Fixed/Tube 的实时完成数。seed 47326 完整口径为 48 个 R-dependent 视频加 2 个全局控制视频。

每个 manifest 必须记录：`target_scope`、`mask_mode`、冻结 Top100 entries、实际 token indices、逐 latent token 数、40 步双 CFG 调用审计、轨迹来源以及 softmax 是否重算。

## 12. 重跑命令

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

RAFT 首次提取需要 GPU，后续会复用 `/data` 下的 float16 flow cache。不得使用 GPU 4；下面示例使用 GPU 3：

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
CUBLAS_WORKSPACE_CONFIG=:4096:8 CUDA_VISIBLE_DEVICES=3 /data/gaoya/miniconda3/envs/wan/bin/python -u AAA_my_test/analyze_legacy_ti2v_object_ablation_raft_motion.py --device cuda --batch-size 4
```
