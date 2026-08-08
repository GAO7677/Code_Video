# Ball-block 表征与 WMReward 实验总结

## 1. 实验范围

- 数据：30 个 `ball_block` 视频，每个 150 帧、60 FPS。
- 组成：8 个原始参数、10 个中等运动控制、12 个极端控制。
- V-JEPA：使用前 49 帧，比较 normalized patch token cosine。
- WMReward：使用完整 150 帧，16 帧窗口，stride 8。

## 2. 已完成实验

| 实验 | 设计 | 输出 |
|---|---|---|
| V-JEPA 两两比较 | 原始 8 个视频，共 28 对 | overlay、热力图、时间曲线 |
| 控制变量比较 | 分别改变 restitution、friction、mass、speed、yaw、distance | 基准相似度、趋势图、组内矩阵 |
| 极端参数比较 | 新增 12 个极端单变量视频 | 极端点趋势与碰撞诊断 |
| WMReward context | 30 个视频分别使用 1、5、8、10 帧 context | 120 个 Surprise 分数 |
| Future shuffle | 每个窗口固定 context，仅打乱 future 帧 | 120 组原始/shuffle 配对 |

## 3. 有较充分论据的结论

结论仅适用于当前数据、模型和实现。

| 结论 | 证据 |
|---|---|
| WMReward 对 context 长度非常敏感 | Surprise 均值：`ctx1=0.9750`、`ctx5=0.7000`、`ctx8=0.6826`、`ctx10=0.6930` |
| 当前设置下 `ctx=8` 最优 | 其平均 Surprise 最低 |
| Future shuffle 不会显著提高 Surprise | `ctx5/8` 仅提高约 `0.14%/0.16%`；`ctx1/10` 平均略降 |
| V-JEPA 不只响应碰撞 | 无碰撞或前 49 帧未碰撞的视频仍有明显特征差异 |
| 极端速度影响最大 | `speed=2x` 相对基准 cosine 为 `0.84859` |
| 极端距离和质量也有明显影响 | `distance=2x: 0.87571`；`mass=10x: 0.88893` |
| restitution 和 friction 的影响相对较小 | 极端设置多数仍在 `0.907–0.925` |
| context 效应大于视频间物理参数效应 | context 改变约 `0.29`；同一 context 下标准差约 `0.002–0.006` |

## 4. 有一定证据，但仍需验证

| 结论 | 限制 |
|---|---|
| V-JEPA 差异常在碰撞后增大 | 需要逐案例按真实接触帧对齐 |
| 参数偏离基准越远，相似度通常越低 | 每个参数点只有一条确定性视频 |
| 大质量影响可能趋于饱和 | `5x` 与 `10x` 接近，但缺少更多质量点 |
| WMReward 更接近整段可预测性，而非碰撞判别 | 需要显式物理正确/错误视频对照 |
| mean 聚合会稀释局部碰撞异常 | 尚未比较 mean、max 和碰撞窗口局部分数 |
| WMReward 对帧顺序不敏感 | 目前只有一个 shuffle seed 和一种干预方式 |

## 5. 当前主要是推断

| 推断 | 尚缺证据 |
|---|---|
| `ctx=8` 最好是因为更接近训练分布 | 尚未核对训练 context 分布 |
| yaw 正负不对称来自相机、遮挡或背景 | 缺少镜像相机和去背景对照 |
| 早期热力图差异来自全局时序注意力 | 缺少 causal encoder 或截断 clip 对照 |
| Shuffle 不敏感是因为相邻帧过于相似 | 缺少倒序、跨视频替换和噪声帧实验 |
| 官方 cosine loss 的 `dim=1` 可能不合理 | 需要打印实际 tensor shape 确认维度语义 |
| V-JEPA/WMReward loss 能改善 DiT 物理一致性 | 需要训练对照和生成质量评测 |

## 6. 不能直接得出的结论

- V-JEPA 相似度越高，物理越正确。
- WMReward Surprise 越低，碰撞越真实。
- 所有特征差异都由碰撞造成。
- 当前变量影响排序可以推广到其他场景。
- 当前实验已经证明辅助 loss 能提升 DiT。

## 7. 结果位置

- V-JEPA 分组结果：`/data/gaoya/agent-data/outputs/vjepa_ball_block_pairwise/ball_block49_native_rect_vitl_with_raw_20260808/controlled_groups.html`
- WMReward context 结果：`/data/gaoya/agent-data/outputs/wmreward_ball_block_context_full150/index.html`
- Future shuffle 结果：`/data/gaoya/agent-data/outputs/wmreward_ball_block_context_full150/future_shuffle.html`
- WMReward CSV：`/data/gaoya/agent-data/outputs/wmreward_ball_block_context_full150/wmreward_scores.csv`
- Future shuffle CSV：`/data/gaoya/agent-data/outputs/wmreward_ball_block_context_full150/future_shuffle/future_shuffle_scores.csv`

## 8. 总结

1. V-JEPA 会响应轨迹、初始几何和碰撞后状态，但不是纯碰撞指标。
2. WMReward 强烈依赖 context 设置，当前 `ctx=8` 最稳定。
3. 当前 WMReward 对窗口内 future shuffle 基本不敏感。
4. 是否适合作为 DiT 训练 loss，仍需训练对照实验验证。
