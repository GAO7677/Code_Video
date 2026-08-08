# Object Query Attention 消融实验矩阵

## 1. 实验对象和符号

对每个 self-attention head 定义：

\[
A=\operatorname{softmax}(QK^\top/\sqrt d),\qquad Y=AV.
\]

- `R`：F00 上某一对象的稀疏 query points 映射并去重后的 latent spatial tokens；`all_objects` 时为所有对象点的并集。
- `C`：当前 attention 序列中除 `R` 外的全部 tokens。
- `S=A[R,R]`：selected token 内部连接。
- `I=A[R,C]`：unselected K/V 到 selected Query 的连接。
- `O=A[C,R]`：selected K/V 到 unselected Query 的连接。

`R` 不是完整 object mask，也不是跨时间 object tube。它只代表 F00 的稀疏采样 token；因此下文的“对象”均指该稀疏代理集合。

## 2. 七种完整矩阵区域消融

| ID | 实现名 | 置零矩阵块 | 精确计算含义 | 理论诊断目标 | 可能观察，非结果保证 |
|---|---|---|---|---|---|
| M1 | `self_only` | `S` | `R` Query 不读取 `R` K/V | 稀疏对象 token 的内部自支持 | 局部身份或形状维持变弱 |
| M2 | `incoming_only` | `I` | `R` Query 不读取 `C` K/V | 外部场景向选中 Query 的输入 | 对环境、其他对象或运动背景响应变弱 |
| M3 | `outgoing_only` | `O` | `C` Query 不读取 `R` K/V | 选中 token 向其他 token 的输出 | 其他区域受该对象影响减弱 |
| M4 | `query_row` | `S+I` | `A[R,:]=0`，所以 `Y[R]=0` | 删除选中 Query 在该 head 的全部更新 | 选中位置的 head 信息通路消失 |
| M5 | `key_value_column` | `S+O` | `A[:,R]=0`，不重新归一化 | 删除选中 token 的全部 Value 贡献 | 全局不再接收该稀疏对象信息 |
| M6 | `cross_boundary` | `I+O` | 双向跨边界连接置零，保留 `A[R,R]` | 隔离 `R` 与 `C`，同时保留内部连接 | 对象内部可能保持但交互减弱 |
| M7 | `row_and_column` | `S+I+O` | `R` 不读取任何 token，`C` 也不读取 `R` | 删除所有涉及 `R` 的连接 | 比单行或单列更强的联合效应 |

这七种组合只在“固定二分集合 `{R,C}`、对涉及 `R` 的 attention entries 做二值置零”的定义下完备，并不覆盖缩放、噪声、替换、动态轨迹 mask 或分时段干预。

## 3. 必须同时展示的算子对照

| ID | 实现名 | 实际操作 | 与 M5 是否等价 | 含义 |
|---|---|---|---|---|
| C1 | `literal_kv_zero` | 在选中 head 上令 `K_R=V_R=0` 后重新计算 attention | 否 | 对应列仍进入 softmax；其 logits 变为 0，并占用概率质量 |
| C2 | `qk_logits_zero` | 在选中 head 的全部 tokens 上令 `q_h=0`，故完整 `QK^T=0`，重新计算 softmax | 否 | `softmax(0)=1/N`，所以每行输出同一个 `mean(V_h)`，并非零输出 |
| C3 | `full_head_output` | 令选中 head 的整个 `Y_h=A_hV_h=0` | 否 | 删除整个 head 输出；与令 `QK^T=0` 不同 |
| Baseline | 无 | 不干预 | — | 同 seed 基线视频 |

M5 使用 `V_R=0, K不变` 的精确等价实现：softmax 权重 `A` 不变，因此输出严格等于 post-softmax `A[:,R]=0` 且不重新归一化。C1 同时修改 K，必须与 M5 分开解释。

旧实现曾把整 head 输出置零命名为 `full_qk`，这个名称不成立。现在 C2 和 C3 分开生成、分开标注，不能互相替代。

## 4. 目标集合、Top-N 和时间矩阵

| 维度 | 水平 |
|---|---|
| Target scope | 每个 `single_object`；所有对象稀疏 token 的 `all_objects` 并集 |
| Matrix/operator | 每个目标集合执行 M1–M7、C1 |
| Head count | Top30、Top50、Top100 |
| Head selection | 冻结的 provisional S039 PCK ranking |
| Denoising step | S000–S039 全部 40 步 |
| CFG branch | conditional 和 unconditional |
| All-token controls | 每个 case 的 Top30/50/100 分别执行 C2（QK logits zero）和 C3（full-head output zero） |
| Seed | 新增 9 case 统一为 `47326`；原 6 个冻结样例保留各自既有 seed |

每个具有 `n` 个对象的 case，需要：

\[
3\times\left(8\times(n+1)+2\right)
\]

个消融视频。其中 `8=M1...M7+C1`，`n+1` 包含每个单对象和所有对象并集，最后的 `2` 是不依赖对象集合的 C2、C3。

当前 15 个 case 的完整任务量：

| 样例组 | Cases | Target sets | M1–M7+C1 | C2+C3 | 总视频数 |
|---|---:|---:|---:|---:|---:|
| 原冻结可视化样例 | 6 | 24 | 576 | 36 | 612 |
| seed=47326 新增样例 | 9 | 30 | 720 | 54 | 774 |
| 合计 | 15 | 54 | 1296 | 90 | 1386 |

## 5. 排名和因果解释限制

1. Top heads 由 S039 positive-conditional 的 F00 Query 到未来 K 的 PCK 排名选出，但干预扩展到全部步骤和两个 CFG 分支。
2. 因此不能假设这些 head 在早期步骤或 unconditional 分支同样是 Top tracking heads。
3. 排名快照是在 aggregate 未完成时冻结的 provisional ranking，必须保留随机 head、bottom head 或 layer-matched control 才能支持特异性结论。
4. 视频变化是经过残差、其他 heads、FFN、cross-attention 和扩散动力学传播后的结果；身份漂移、碰撞改变等只能作为待检验假设，不能写成理论必然结果。

## 6. 实现与输出

- 实现脚本：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations.py`
- CPU 数学测试：`/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/test_legacy_attention_matrix_ablations.py`
- 原样例输出：`/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_matrix_ablations_v2`
- seed=47326 输出：`/data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_v2`
- 可视化页面：`http://localhost:8092/wan22-ti2v-legacy-physiciq67-samples?v=2`

每个输出 manifest 必须记录 `target_scope`、`mask_mode`、`zeroed_matrix_blocks`、实际 token indices、40 步双 CFG 调用审计和冻结 Top-N entries。页面必须把 `qk_logits_zero`（均匀注意力）与 `full_head_output`（零输出）显示为两行，不得再混用 `full_qk` 标签。

## 7. 重跑命令

原冻结样例的第 `i` 个 worker：

```bash
CUDA_VISIBLE_DEVICES=<非4的物理GPU> /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations.py --worker-id <i> --num-workers 5
```

seed=47326 新增样例的第 `i` 个 worker：

```bash
CUDA_VISIBLE_DEVICES=<非4的物理GPU> /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations.py --worker-id <i> --num-workers 5 --manifest-path /data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/cases.json --output-root /data/gaoya/agent-data/outputs/wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/attention_zero_seed47326/attention_matrix_ablations_v2
```
