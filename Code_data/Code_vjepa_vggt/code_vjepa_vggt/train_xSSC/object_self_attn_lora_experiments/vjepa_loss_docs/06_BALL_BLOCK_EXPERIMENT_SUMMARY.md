# Ball-block 表征与 WMReward 实验总结

## 1. 实验范围

- 数据：30 个 `ball_block` 视频，每个 150 帧、60 FPS。
- 组成：8 个原始参数、10 个中等运动控制、12 个极端控制。
- V-JEPA 表征比较：使用前 49 帧。
- 修正版 WMReward：使用完整 150 帧、16 帧窗口、stride 8。

## 2. WMReward 修正版协议

旧版 `ctx=1/5/8/10` 结果不再用于结论，原因如下：

1. ViT-G 的 `tubelet_size=2`，旧实现使用 `context_frames // 2`。因此旧 `ctx=1` 实际没有 context token，旧 `ctx=5` 实际只有 4 帧 context。
2. odd context 的 raw-frame shuffle 边界与 causal token mask 边界不一致。
3. 模型输出形状是 `[B,N_pred,D]`；旧版 `cosine_similarity(..., dim=1)` 在 token 轴上计算 cosine，不是每个 token 在特征维上的 cosine。

修正版固定为：

| 项目 | 设置 |
|---|---|
| Context frames | `2 / 4 / 8 / 10`，全部与 tubelet 对齐 |
| Effective context | 与标注 context 完全一致 |
| Window / stride | `16 / 8` |
| Predicted future frames | `14 / 12 / 8 / 6`；随 context 改变 |
| Cosine axis | `dim=-1`，每个预测 token 的 feature cosine |
| Aggregation | 先对预测 token 求均值，再对 17 个重叠窗口求均值 |
| Future shuffle | 每个窗口保持 context 有序，只独立打乱 future frames |
| Shuffle seed | `20260808` |

因此 context 实验仍然是“context 长度与预测 horizon 联合改变”的比较，不能把差异单独归因于 context 信息量。

## 3. 完整实验矩阵

| 实验 | 数据单位 | 条件 | 每条件样本 | 总分数 | 主要输出 |
|---|---|---|---:|---:|---|
| V-JEPA 两两比较 | 原始视频对 | 原始 8 个视频的 28 对 | 1 个确定性 pair | 28 | overlay、热力图、时间曲线 |
| 控制变量比较 | 单个视频 | restitution、friction、mass、speed、yaw、distance | 每参数点 1 个视频 | 30 videos | 基准相似度、趋势图、组内矩阵 |
| WMReward context v2 | 单个视频 | ctx `2/4/8/10` | 30 | 120 | Surprise、Similarity、分布与逐视频曲线 |
| Future shuffle v2 | 同视频配对 | ctx `2/4/8/10` × original/shuffle | 30 | 120 pairs | paired delta、ratio、方向计数 |

重叠窗口不是独立样本。推断统计应先聚合到视频级，或使用以视频为 cluster 的 bootstrap/permutation；不得把 17 个窗口当作 17 个独立观测。

## 4. 当前可以保留的描述性结论

以下结论只适用于当前数据、模型和实现：

| 结论 | 证据边界 |
|---|---|
| V-JEPA 不只响应碰撞 | 无碰撞或前 49 帧未碰撞的视频仍存在特征差异 |
| 当前测试范围内极端 speed 条件差异最大 | 只是在已测试参数点中观察到；不同变量的扰动尺度不可直接等量比较 |
| 极端 distance 和 mass 也产生明显表征差异 | 每个参数点只有一条确定性视频，不能估计生成方差 |
| restitution 和 friction 在当前取值范围内差异相对较小 | 不能外推到其他场景或参数范围 |

旧文档中的以下说法已撤回：

- “`ctx=8` 最优/最稳定”：最低均值不等于最优，稳定性需要方差或重复实验；旧数据中 ctx8 的跨视频标准差反而最大。
- “Future shuffle 不会显著提高 Surprise”：旧实验只支持平均变化很小，不能在未指定检验方法时写“显著”。
- “context 效应大于物理参数效应”：旧比较把条件均值差与条件内标准差直接比较，二者不是同一种效应量。

## 5. 修正版结果

### 5.1 Context/horizon 联合条件

统计单位是每个视频在 17 个重叠窗口上的聚合分数；SD 和正态近似 95% CI 均跨 30 条确定性视频计算。

| Context frames | Predicted future frames | Mean Surprise | Sample SD | 95% CI |
|---:|---:|---:|---:|---:|
| 2 | 14 | 0.450206 | 0.007511 | [0.447518, 0.452893] |
| 4 | 12 | 0.433449 | 0.004526 | [0.431829, 0.435068] |
| 8 | 8 | 0.418425 | 0.005097 | [0.416601, 0.420249] |
| 10 | 6 | 0.424839 | 0.005015 | [0.423045, 0.426634] |

描述性结果为 Surprise 从 ctx2 到 ctx8 下降，在 ctx10 回升。由于 context 增加时预测 future horizon 同时从 14 帧缩短到 6 帧，这个模式不能解释为“ctx8 的 context 信息最优”，也不能单独归因于 context 长度。

### 5.2 Future shuffle

original 与 shuffle 按同一 `(video, context)` 配对；每个 context 有 30 个唯一视频对。Delta 定义为 `shuffled surprise - original surprise`，正值表示打乱 future 后 Surprise 增加。

| Context | Original mean | Shuffled mean | Mean paired delta | 95% CI of delta | Mean ratio | `+ / - / 0` |
|---:|---:|---:|---:|---:|---:|---:|
| 2 | 0.450206 | 0.453846 | +0.003640 | [0.002960, 0.004320] | 1.00808× | 30 / 0 / 0 |
| 4 | 0.433449 | 0.436859 | +0.003411 | [0.002763, 0.004058] | 1.00785× | 30 / 0 / 0 |
| 8 | 0.418425 | 0.420273 | +0.001848 | [0.001428, 0.002269] | 1.00442× | 30 / 0 / 0 |
| 10 | 0.424839 | 0.425692 | +0.000852 | [0.000395, 0.001309] | 1.00202× | 24 / 6 / 0 |

在这 30 条确定性视频与固定 shuffle seed 下，future-frame 置乱的平均效应为小幅提高 Surprise，且效应随 context 增大而减小；ctx2/4/8 的每条视频 delta 都为正，ctx10 有 6/30 为负。区间是跨视频对的正态近似描述区间；这些受控参数视频不是独立的随机重复，单一 shuffle seed 也不足以概括所有乱序方式。

## 6. 仍需验证的假设

| 假设 | 所需对照 |
|---|---|
| V-JEPA 差异在碰撞后增大 | 按真实接触帧逐案例对齐 |
| 参数偏离基准越远，相似度越低 | 每参数增加更多点和重复生成 seed |
| mean 聚合稀释局部碰撞异常 | 对比 mean、max、碰撞窗口局部分数 |
| WMReward 对时间顺序不敏感 | 多个 shuffle seed、倒序、跨视频 future 替换、噪声帧 |
| WMReward 可作为 DiT 物理 loss | 固定训练预算的 loss/no-loss 对照及盲评 |

## 7. 不能直接得出的结论

- V-JEPA 相似度越高，物理越正确。
- WMReward Surprise 越低，碰撞越真实。
- 所有特征差异都由碰撞造成。
- 当前变量影响排序可以推广到其他场景。
- 当前实验已经证明辅助 loss 能提升 DiT。

## 8. 结果位置

- V-JEPA 分组结果：`/data/gaoya/agent-data/outputs/vjepa_ball_block_pairwise/ball_block49_native_rect_vitl_with_raw_20260808/controlled_groups.html`
- 修正版 WMReward context：`/data/gaoya/agent-data/outputs/wmreward_ball_block_context_full150_corrected_v2/index.html`
- 修正版 Future shuffle：`/data/gaoya/agent-data/outputs/wmreward_ball_block_context_full150_corrected_v2/future_shuffle.html`
- 旧版结果只作实现审计：`/data/gaoya/agent-data/outputs/wmreward_ball_block_context_full150`

Object Query 的固定 Q00 / 全时序 Tube 定义、M1–M7 矩阵区域消融、C1–C3 算子控制、像素相似度、RAFT 动态 ROI 运动相似度及“不同消融为何看起来一致”的诊断，统一见 `07_OBJECT_QUERY_ATTENTION_ABLATION_MATRIX.md`。
