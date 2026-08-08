# 方法

## 1. 目标

在 Wan2.2 的 flow-matching DiT loss 上加入冻结 V-JEPA2.1 ViT-L 的视频特征约束，使预测视频在时空表征上接近 GT。

## 2. x0 还原

Wan 的训练路径满足：

```text
x_t = (1 - sigma) * x0 + sigma * noise
v_target = noise - x0
pred_x0_raw = x_t - sigma * pred_v
```

随后将 context latent 替换为干净 GT latent：

```text
pred_x0 = restore_condition_latents(pred_x0_raw)
```

因此辅助 loss 比较的是 context-restored prediction，而不是完全自由预测。

## 3. 辅助 loss

```text
L_feat = mean_future ||normalize(f_pred) - normalize(f_gt)||_2^2
L_range = mean(relu(-rgb_raw)^2) + mean(relu(rgb_raw - 1)^2)
L_total = L_DiT + I * 0.01 * w_t * (L_feat + 0.1 * L_range)
```

其中：

- `I`：仅在 `sigma in [0.2, 0.8]` 且每 2 个 micro-forward 时启用。
- `w_t`：Wan 原生 timestep weight，并在 sigma gate 内归一化。
- `L_feat`：只监督完全位于 future 的 temporal tubelets。
- 特征先做 L2 normalize，再对通道平方和、对 token 求平均。

## 4. 视频与特征路径

```text
pred_v -> pred_x0 -> restore context -> Tiny VAE -> RGB -> V-JEPA
GT latent ---------------------------> Tiny VAE -> RGB -> V-JEPA
```

- 两个分支使用同一个冻结 `taew2_2` Tiny VAE。
- Tiny VAE 顺序解码，降低峰值显存。
- RGB clamp 使用 straight-through gradient，并对越界 raw RGB 加惩罚。
- V-JEPA 使用 FP32；GT 分支无梯度，预测分支保留输入梯度。
- Tiny VAE 和 V-JEPA 不更新、不保存到训练 checkpoint，训练过程不经过 MP4。

## 5. 实际训练采样

历史 run 的 resolved config：

- 49 帧训练视频，8 帧 context。
- V-JEPA 每次取 16 帧。
- 50% global uniform，50% context-boundary local。
- local 模式保留 4 帧 context，并取跨越 context 边界的连续片段。

