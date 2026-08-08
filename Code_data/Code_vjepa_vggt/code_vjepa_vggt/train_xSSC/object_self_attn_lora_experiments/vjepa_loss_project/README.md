# V-JEPA Loss 实验归档

更新时间：2026-08-08

本目录汇总 Wan2.2 DiT 上的 V-JEPA 辅助 loss、Tiny VAE 解码、特征 MSE 和局部热图实验。原始代码、配置和大体积结果保持在原位置。

## 文档索引

| 文档 | 内容 |
|---|---|
| [01_METHOD.md](docs/01_METHOD.md) | loss 定义与梯度链路 |
| [02_EXPERIMENTS.md](docs/02_EXPERIMENTS.md) | 已完成实验与运行状态 |
| [03_ANALYSIS.md](docs/03_ANALYSIS.md) | 结论、风险与待验证项 |
| [04_REPRODUCE.md](docs/04_REPRODUCE.md) | 代码、权重、结果和复现入口 |
| [05_LEGACY_INDEX.md](docs/05_LEGACY_INDEX.md) | 早期 V-JEPA 结果目录索引 |

## 当前结论

- V-JEPA loss 已接入原 Full-SA/no-object 训练，只更新 DiT LoRA。
- 正式 run 使用 GPU 0、1，运行至约 step 3463 后被 SIGINT 中断。
- 单次梯度快照未显示爆炸，但尚无同初始化、同数据的无 V-JEPA 对照实验，不能判断生成质量是否改善。
- 可视化中的 `pred_x0` 已恢复 GT context latent，应理解为 `pred_x0_context_restored`。
- 历史 run 的真实配置是 16 帧、mixed 采样；当前同名 JSON 已变为 49 帧、full 采样。复现历史 run 必须使用 run 内的 resolved config。

## 原始文档

- [VJEPA_LOSS_TRAINING_REQUIREMENTS.md](docs/VJEPA_LOSS_TRAINING_REQUIREMENTS.md)
- [HANDOFF.md](docs/HANDOFF.md)
- [项目实验记录](../EXPERIMENT_RECORD.md)
- [V-JEPA2 MSE 原型说明](docs/VJEPA2_FEATURE_MSE.md)

## Ball-block 视频对实验

`visualize_ball_block_pairwise.py` 对目录内全部视频取前 49 帧，以 `384x672`
native-rectangle 输入提取特征。页面先展示 8 个无 overlay 的原视频和 49 帧
拼接图，再展示全部 pair 的 overlay 视频、49 帧热力图拼接图和差异排序。

`visualize_ball_block_temporal_similarity.py` 复用缓存特征，计算 28 个视频对在
49 帧时间轴上的平均 patch cosine similarity，并生成总览、分 pair 曲线和统计表。
