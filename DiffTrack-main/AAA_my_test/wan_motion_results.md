# Wan2.2-TI2V-5B Motion Q/K 实验结果

## 实验设置

- 数据：ToyDataset 的 50 个 base case；48 个 case 在 frame 0 有合法对象 query，case 006/014 的对象首帧不可见，仅保留背景结果。
- Query：每个可见对象和背景各 32 点，renderer instance mask 内采样；CoTracker 提供轨迹与 visibility。
- Wan 协议：49 帧整体经过原生时间 VAE 得到 13 个 latent frames，使用 pixel frames `[0,4,...,48]` 作为比较锚点，不做 13→49 插值。
- 扫描：layers `[0, 5, 11, 17, 23, 29]`，scheduler step indices `[0, 12, 24, 36, 49]`，一次 forward 同时捕获 6 层 post-RMSNorm/post-3D-RoPE Q/K。
- 匹配：`1/sqrt(128)` 缩放，逐目标帧空间 softmax，多头平均，并对 `Q_query·K_target` 和 `K_query·Q_target` 取均值。
- 主指标：每个 case 内先合并 moving-object，再跨 case macro-average；moving 定义为整段最大可见位移中位数大于 16 px。

![layer-step PCK32 heatmap](/data/gaoya/agent-data/outputs/wan22_motion_qk/aggregate_base/layer_step_pck32_heatmap.png)

## 结果

| 配置 | Moving PCK@32 | 95% CI | Mean error | Direction cosine | F1 / F2 / F3 PCK@32 |
|---|---:|---:|---:|---:|---:|
| Layer 17, step 49, t=92, σ=0.0925 | 90.13% | [86.26, 93.33] | 20.06px | 0.877 | 80.53 / 93.11 / 94.46 |
| Layer 17, step 36, t=660, σ=0.6601 | 88.13% | [84.36, 91.50] | 23.81px | 0.869 | 76.88 / 91.12 / 93.90 |

- Layer 17 / step 49 在 48 个 motion case 上达到 `90.13% PCK@32`，而 query 不动的静态 baseline 为 `26.69%`；mean error 为 `20.06px`，静态 baseline 为 `179.18px`。预测与 GT 平均位移分别为 `174.11px` 和 `179.18px`，方向余弦为 `0.877`，说明高准确率不是 query 保持不动造成的。
- Layer 17 / step 36 为稳定的中噪声候选，PCK@32 为 `88.13%`、mean error 为 `23.81px`。与 step 49 配对比较，step 49 的 PCK 平均高 `1.99` 点，但 95% CI 接近包含 0；mean error 平均低 `3.75px`，95% CI 为 `[-5.65,-1.87]px`，42/48 case 更低。
- Layer 23 / step 49 的 moving PCK@32 为 `85.05%`，但 mean error 为 `41.51px`，明显弱于 layer 17；layer 5/11 在部分噪声点可用，但跨 family 稳定性不足。layer 0/29 和最高噪声 step 的结果接近失败，不能用于 motion 对应。
- Layer 17 / step 49 的背景 PCK@32 为 `96.06%`，mean error 为 `17.63px`；背景 GT 平均位移仅 `0.98px`。该误差主要来自 32px DiT token 网格的中心量化，不能解读为背景真实运动。低运动对象 PCK@32 为 `98.18%`。

## 结论

Wan2.2-TI2V-5B 中用于对象 motion 对应的首选位置是 **Transformer layer 17 的低噪声 step 49 (`t=92`, `sigma=0.0925`)**；如果研究目标要求在较强噪声下仍可提取 motion，使用 **layer 17 / step 36 (`t=660`, `sigma=0.6601`)**。真正有判别力的是 layer 17 Q/K 所恢复的对象位移、方向和对象内部几何，而不是背景准确率或单一 attention 最大值。

这个结果证明 layer 17 Q/K 包含可靠的跨 latent 时间块 correspondence，但它本身还不是“物理合理性分数”。下一步比较 GT 与物理不合理视频时，应固定 layer 17/step 49 为主配置、layer 17/step 36 为噪声鲁棒性对照，统计 trajectory jerk、方向突变、刚性误差和遮挡恢复；不能在每个视频上重新挑 layer/step，否则会产生选择偏差。

## 产物

- 计划入口：`AAA_my_test/plan.md`
- 完整 Wan 计划：`AAA_my_test/wan_motion_plan.md`
- 单 case gate：`/data/gaoya/agent-data/outputs/wan22_motion_qk/single_case/case_019_wheel_hits_block_base/gate_report.json`
- 50-case 原始指标：`/data/gaoya/agent-data/outputs/wan22_motion_qk/batch_base/`
- 聚合排名：`/data/gaoya/agent-data/outputs/wan22_motion_qk/aggregate_base/summary.csv`
- 最佳配置：`/data/gaoya/agent-data/outputs/wan22_motion_qk/aggregate_base/best_configs.json`

## 限制

- 原生 VAE 的 13 个 latent frames 表示时间块，当前结果不是逐像素帧光流。
- CoTracker 是伪 GT；小物体、快速旋转和遮挡仍可能引入误差。
- 本批只完成 50 个 base case；150 个颜色/形状/背景 variants 尚未运行，需在固定 layer 17 配置下作为鲁棒性验证，而不是重新搜索超参数。
- 本地 Wan `model.py` 包含 checkpoint 中不存在的 object-cross-attention 参数，但 object gate 初始化为 0 且本实验不传 object context；本次分析路径是官方 self-attention Q/K。
