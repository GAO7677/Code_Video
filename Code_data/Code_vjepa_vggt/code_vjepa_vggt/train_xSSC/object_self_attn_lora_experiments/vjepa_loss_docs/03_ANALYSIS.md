# 分析

## 1. 设计是否合理

作为辅助感知损失，该设计基本合理：它复用同一次 DiT forward，在可微 tensor 路径中还原 x0，并用冻结视频编码器约束 future 表征。

它不是直接的物理 loss。V-JEPA 更偏向语义、运动和时空一致性，对碰撞、速度守恒或轨迹合理性没有显式约束，因此只能作为正则项，不能替代物理指标。

## 2. 为什么 loss 看起来小

- 主 loss 在 latent/velocity 空间平均，数值小不等于梯度无效。
- 辅助项权重只有 `0.01`。
- 辅助项受 sigma gate、每 2 个 micro-forward cadence 和 future token mask 限制。
- token 与 batch 维度均做平均。

判断标准应是分项曲线、梯度比例、验证指标和同配置对照，不应只看总 loss 的绝对值。

## 3. 梯度风险

中断前快照中 `grad_norm=0.00453`，远低于 clip 阈值 `1.0`，未见爆炸。`grad_v_vjepa_to_main_ratio=0.84535` 表明辅助梯度并非可以忽略；cosine 接近 0 表明两项目标在该 batch 上近似正交。

单点快照不能排除偶发尖峰。继续训练时应监控：

- `grad_norm`、non-finite 次数和 clip 前梯度。
- 主 loss 与辅助 loss 的输出梯度比例、cosine。
- sigma、local/global 采样模式与梯度尖峰的关联。
- Tiny VAE raw RGB 越界比例和 range penalty。

## 4. `pred_x0` 接近 GT 的原因

`pred_x0_raw = x_t - sigma * pred_v` 后，代码会将条件帧 latent 替换为 GT。前 8 帧接近或等于 GT 是设计结果，不是模型已准确预测全部视频。

可视化应区分：

- `pred_x0_raw`：恢复 context 前。
- `pred_x0_context_restored`：训练 V-JEPA loss 实际使用。
- future-only 误差：排除已知 context 后再统计。

## 5. 配置漂移

历史 run 的 `resolved_experiment_config.json` 为：

```text
vjepa.num_frames = 16
vjepa.frame_sampling = mixed
```

当前 `formal_full_sa_no_object_gpu27_vjepa_loss.json` 为：

```text
vjepa.num_frames = 49
vjepa.frame_sampling = full
```

因此当前 JSON 不能精确复现 step 3463 run。任何续训或对照都应先固定新实验名，并明确选择“16/mixed”或“49/full”。

## 6. 仍缺少的关键对照

- 相同初始化、数据顺序和训练步数的 `V-JEPA weight=0` 对照。
- `0.001 / 0.003 / 0.01` 权重消融。
- `16/mixed` 与 `49/full` 的显存、速度和质量对照。
- raw/restored x0 及 future-only 指标对照。
- 统一 checkpoint 上的 FVD、LPIPS、V-JEPA distance 和物理指标。

在这些对照完成前，当前证据只支持“实现可运行且未见明显梯度异常”，不支持“V-JEPA loss 改善生成质量”。

