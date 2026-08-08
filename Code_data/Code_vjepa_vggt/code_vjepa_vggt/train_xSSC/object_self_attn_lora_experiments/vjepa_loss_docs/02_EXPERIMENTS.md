# 实验记录

## 1. 高噪声 x0 可视化

入口：`run_train_from_config_with_viz.py`

- 从最高 sigma 候选区间内均匀抽取 timestep，仅增加一次可视化 forward，不改变训练 loss。
- 用 Tiny VAE 解码 `GT`、`x_t` 和 `pred_x0`。
- 视频 FPS 读取原始 GT 视频；读取失败才使用回退值。
- 页面采用手动刷新和全部重新播放，不自动刷新。

注意：页面中的 `pred_x0` 已恢复 context latent，因此 context 段天然接近 GT。

## 2. 离线 V-JEPA2 特征 MSE 原型

入口：`/home/gaoya/Code_Video/vjepa2_tinyvae_mse/compute_vjepa2_feature_mse.py`

| 项目 | 值 |
|---|---|
| 模型 | V-JEPA2.1 ViT-L 384，FP32 |
| 输入 | `gt_x0.mp4` 与 `pred_x0.mp4`，64 帧 |
| case 数 | 3 |
| sigma | 0.4508、0.6200、0.9122 |
| feature MSE | 1.0314、1.1624、1.3649 |
| mean ± std | 1.1862 ± 0.1372 |

该实验验证了特征提取与 MSE 计算链路。它经过 MP4 I/O，不可直接用于训练反传。

## 3. V-JEPA loss Smoke

- 使用正式 GPU01 配置的 step-000500 checkpoint。
- 1 个样本推理成功，失败数为 0。
- 该结果只证明训练、保存和推理链路可运行，不代表方法有效。

## 4. GPU01 正式训练

| 项目 | 值 |
|---|---|
| 模型 | Wan2.2 TI2V-5B，Full-SA LoRA，object branch 关闭 |
| GPU | 0、1 |
| 数据 | PyBullet 30% + Kubric 30% + OpenVidHD 40% |
| 单卡 batch / 累积 | 1 / 4，effective batch 8 |
| 学习率 | `1e-4` |
| 梯度裁剪 | `1.0` |
| V-JEPA 权重 | `0.01` |
| W&B | `2eemg19t` |
| 状态 | 约 step 3463 收到 SIGINT，中断 checkpoint 已保存 |

中断前一次记录：

| 指标 | 值 |
|---|---:|
| `train/loss` | 0.04347 |
| `grad_norm` | 0.00453 |
| `grad_v_main_norm` | 0.00079 |
| `grad_v_vjepa_norm` | 0.00067 |
| `grad_v_vjepa_to_main_ratio` | 0.84535 |
| `grad_v_main_vjepa_cosine` | 0.00871 |

该快照未显示梯度爆炸。V-JEPA 输出梯度与主 loss 同量级、方向近似正交，需要继续看时间序列和无辅助 loss 对照。

## 5. Flow/V-JEPA 局部热图

已比较：

- `step03463_lora` 与 `no_step03463_lora`。
- PyBullet 多物体 case 与 11 个 v2v JSON case。
- 默认裁剪、native rect `384x672`、全视频输入。
- 原始 V-JEPA map 与 Flow 高损失区域加权 map。
- 单帧页、全帧页和汇总比较页。

当前只确认局部差异可被稳定计算和展示，尚未建立热图变化与生成质量提升的因果关系。

