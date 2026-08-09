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

| ID | 实现名 | 置零块 | 被切断的信息流 | 精确计算 | 固定 Q00 在问什么 | 全时序 Tube 在问什么 | 可能观察，非必然结果 |
|---|---|---|---|---|---|---|---|
| M1 | `self_only` | `S` | `R K/V ──X──> R Query` | `A[R,R]=0`；保留 `A[R,C]` | F00 稀疏 token 是否靠彼此维持局部表征 | tube 内帧内和跨帧对象连接是否提供自支持 | 身份、形状或时序一致性可能减弱 |
| M2 | `incoming_only` | `I` | `C K/V ──X──> R Query` | `A[R,C]=0`；R Query 只读 R | 外部 token 是否向 F00 Query 输入上下文 | tube 各时刻是否依赖 tube 外的背景、其他对象和非轨迹 token | 环境或交互响应可能减弱 |
| M3 | `outgoing_only` | `O` | `R K/V ──X──> C Query` | `A[C,R]=0`；R Query 的读取保持 | F00 对象 Value 是否向其他位置广播 | 整条 tube 是否作为跨时空信息源影响其余 token | 其他区域受对象影响可能减弱 |
| M4 | `query_row` | `S+I` | `全部 K/V ──X──> R Query` | `A[R,:]=0`，故该 head 的 `Y[R]=0` | 删除 F00 接收端的完整 head 更新 | 删除 tube 全部 13 个时刻接收端的完整 head 更新 | R 位置的该 head 更新完全消失 |
| M5 | `key_value_column` | `S+O` | `R Value ──X──> 全部 Query` | 保持 A 不变，令 `A[:,R]=0`，不重归一化；严格等价于只令 `V_R=0` | 删除 F00 token 的 Value 输出 | 删除整条 tube 在所有时刻的 Value 输出 | 全局不再接收 R 的 Value 信息 |
| M6 | `cross_boundary` | `I+O` | `C→R` 与 `R→C` 双向切断 | `A[R,C]=A[C,R]=0`；保留 `R→R`、`C→C` | 隔离 F00 稀疏集合与其余 token | 隔离整条时空 tube，同时保留 tube 内帧内/跨帧读取 | tube 内部可能保持，但外部交互可能减弱 |
| M7 | `row_and_column` | `S+I+O` | 全部输入到 R；R 输出到 C | `A[R,:]=0` 且 `A[C,R]=0`；保留 `C→C` | 同时删除 F00 的接收端和发送端 | 同时删除整条 tube 的接收端和向外发送端 | 通常比单行或单列干预更强 |

M1–M7 是 post-softmax `A@V` 分块置零且不重新归一化。它们只在固定二分集合 `{R,C}` 下构成完整的七种“涉及 R 的非空矩阵块组合”。

## 6. C1–C3：不要与矩阵分块混用

| ID | 实现名 | 实际操作 | 是否依赖 `R` | 精确含义 |
|---|---|---|---|---|
| C1 | `literal_kv_zero` | 在选中 head 上令 `K_R=V_R=0`，重新计算 attention | 是 | R 列 logits 变成 0，但列仍参与 softmax 并占用概率质量；不等价于 M5 |
| C2 | `qk_logits_zero` | 在选中 head 的全部 tokens 上令 `q_h=0`，重算 softmax | 否 | `QK^T=0`，所以 `A_h=1/N`、`Y_h=mean(V_h)`，不是零输出 |
| C3 | `full_head_output` | 令选中 head 的整个 `Y_h=A_hV_h=0` | 否 | 直接删除整个 head 输出 |
| Baseline | 无 | 不干预 | 否 | 原始 Q/K/V、softmax 和 head 输出全部保留 |

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

已完成样例审计显示，`object_A` 的 `R_fixed` 有 6 个唯一 token，`R_tube` 有 79 个唯一 token；13 个 latent 时刻分别为 `[6,7,6,8,6,7,6,5,6,6,5,6,5]`。这说明左右干预剂量并不相等。

## 8. 左右对比应该如何解释

| 观察 | 可以支持的描述 | 不能直接声称 |
|---|---|---|
| Tube 比 fixed 变化更大 | 扩大到整条时空 tube 后，联合干预效应更强 | 每一帧的 PCK head 都同样准确；差异可能只是 token 数更多 |
| Tube 与 fixed 接近 | 额外时刻未显著增加最终视频差异，或模型存在冗余/饱和 | 只有 Q00 有效 |
| 只有 M1/M6 在 Tube 明显变化 | tube 内部或 tube–外部边界通信值得进一步检查 | 已证明某个确定的物理因果通路 |
| 不同对象差异不同 | 两个稀疏对象代理对这些 heads 的敏感度不同 | 球或盒子的完整语义表征已被定位 |

这个 Tube 消融是**因果干预实验**，但不直接回答“给定 Q10，Top PCK heads 在所有 K 帧的高响应是否落在轨迹上”。后者需要单独计算 `query time × key time` 的 13×13 响应/PCK 矩阵，并把峰值与 CoTracker pseudo-GT 对齐。

## 9. Head 排名与因果解释限制

1. Top heads 只由 S039 positive-conditional 的 `Q00 → K01...K12` PCK 排名选出。
2. Tube 实验把这些 heads 扩展到 Q00–Q12，但这只是测试其干预效应，不等于重新验证每个 query 时刻的 tracking accuracy。
3. 干预覆盖全部 40 个 denoising steps 和两个 CFG 分支；不能假设这些 head 在早期步骤或 unconditional 分支仍是 Top tracking heads。
4. 排名是 aggregate 未完成时冻结的 provisional snapshot；特异性结论仍需 random、bottom 或 layer-matched controls。
5. 视频结果经过残差、其他 heads、FFN、cross-attention 和扩散动力学传播。身份漂移、碰撞改变等只能是待检验解释，不能写成理论必然结果。

## 10. 实现、输出与页面

- 固定 Q00 脚本：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations.py`
- 全时序 Tube 脚本：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_legacy_ti2v_temporal_object_tube_ablations.py`
- 固定 Q00 输出：`/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_v2`
- Tube 输出：`/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1`
- Tube 轨迹审计：Tube 输出目录下的 `frozen_baseline_tracks/tracks.npz` 与 `manifest.json`
- 左右对比页面：`http://localhost:8092/wan22-ti2v-legacy-physiciq67-samples?v=10&case=0613pybullet_sample_001460_w002&seed=47326`

每个 manifest 必须记录：`target_scope`、`mask_mode`、冻结 Top100 entries、实际 token indices、逐 latent token 数、40 步双 CFG 调用审计、轨迹来源以及 softmax 是否重算。

## 11. Tube pilot 重跑命令

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
CUDA_VISIBLE_DEVICES=3 /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u AAA_my_test/run_legacy_ti2v_temporal_object_tube_ablations.py --case 0613pybullet_sample_001460_w002 --seed 47326 --device cuda
```

脚本按 `complete.json + manifest.json + generated.mp4` 自动跳过已经完成的任务，可安全恢复。
