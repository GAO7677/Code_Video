# Wan2.2-TI2V-5B Motion Q/K 实验计划

## 1. 目标

在 Wan2.2-TI2V-5B 的 self-attention 中定位能够稳定表示对象跨帧运动对应关系的 Transformer layer 和 diffusion step。实验不以背景匹配准确率作为主要证据，而是分别统计运动物体、低运动物体和背景，要求候选 layer/step 能跟随对象位移，并保持对象内部几何一致性。

模型使用官方权重 `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`。代码只写入 `AAA_my_test`，结果写入 `/data/gaoya/agent-data/outputs/wan22_motion_qk`。

## 2. 核心协议

### 2.1 主协议：Wan 原生时间 VAE

- 输入 49 帧视频，保持原始宽高比并中心裁剪为 `1280x704`。
- 49 帧整体经过 Wan2.2 VAE，得到 13 个 latent frames；使用像素帧 `[0, 4, 8, ..., 48]` 的 CoTracker 轨迹作为 latent 时间锚点。
- 采用 TI2V 原生条件结构：latent frame 0 保持干净并使用 timestep 0，其余 latent 使用相同随机噪声和指定 scheduler timestep。
- prompt 使用每个样本 metadata 中的 caption；所有 layer/step、GT/variant 使用同一 seed。
- 在 self-attention 中提取经过 Q/K RMSNorm 和 3D RoPE 之后、进入 FlashAttention 之前的 Q/K。
- 用 frame 0 query 与每个目标 latent frame 的空间 key 匹配，同时加入反向 `Q_target -> K_query` 分数；每个 head 先对目标空间 softmax，再平均 heads，最后取 argmax。
- attention matching 使用 `1/sqrt(head_dim)`，Wan 5B 的 head dimension 为 128。

该协议忠于 Wan 的 VAE 时间压缩和 TI2V 首帧条件结构。13 个 latent frames 表示时间块，不把结果插值成 49 帧。

### 2.2 辅助协议：逐帧 VAE 探针

- 每帧独立 VAE 编码，以 `frame 0 + 12 target frames` 组成 chunk，建立一像素帧对应一 latent frame 的精确映射。
- 仅用于单 case 和少量代表 case 验证时间压缩是否改变结论。
- 逐帧 latent 不属于 Wan 原生训练分布，因此不得作为 Wan 原生 motion 能力的唯一证据，也不对全数据集默认运行。

## 3. 数据与伪 GT

### 3.1 单 case gate

- 样本：`case_019_wheel_hits_block/base`。
- 区域：`driver_0`、`target_0`、background。
- 每个区域在 frame 0 mask 内最远点采样 32 点；对象 mask 腐蚀后采样，背景排除对象边缘和画面边界。
- 使用 renderer 的无损 instance-id mask，不使用 SAM2。
- CoTracker 生成 49 帧轨迹和 visibility，指标只使用可见点。

### 3.2 批量统计

- 第一批：ToyDataset 全部 50 个 base case，覆盖单物体 F1、双物体碰撞 F2 和三物体链式碰撞 F3。
- 第二批：仅在第一批完成并选出稳定候选后，按需运行 150 个 appearance/shape variants，测试 layer/step 对背景、颜色和形状变化的鲁棒性。
- 每个样本按实际对象数独立采样对象区域，再加 background；不假定所有 case 都有两个物体。
- 单 case 选 layer/step 只用于可视化，不允许在批量统计中按样本或按对象分别挑最优配置。

## 4. 扫描空间

- Layers：`[0, 5, 11, 17, 23, 29]`。
- Scheduler：UniPC、50 steps、shift 5.0。
- Step indices：`[0, 12, 24, 36, 49]`；结果同时保存真实 scheduler timestep 和 `sigma=timestep/1000`。
- 一次 DiT forward 同时处理 6 个 layer，因此每个样本只需要 5 次主 forward。
- 如果最佳配置落在 layer 或 step 扫描边界，第二轮在邻域加密；否则不扩大扫描。

## 5. 指标

### 5.1 点对应

- PCK@8、PCK@16、PCK@32、PCK@64。
- mean/median endpoint error。
- 以 DiT token stride 32 归一化的误差。
- GT token rank、GT 邻域 attention mass、top-1 margin、attention entropy。

PCK@8 低于 Wan 的空间 token stride，只作为辅助指标。主要点对应指标为 PCK@32、median error 和相对静态 baseline 的提升。

### 5.2 Motion 与几何

- motion endpoint error：预测位移与 CoTracker 位移之差。
- motion magnitude ratio：预测位移长度 / GT 位移长度。
- direction cosine：预测和 GT 位移方向一致性。
- object rigidity error：对象内部点对距离变化与 CoTracker 的差异。
- static baseline：所有帧均预测 query 初始位置。

汇总时分别报告：

- `moving_object`：GT 总位移大于 16 px 的对象点。
- `low_motion_object`：GT 总位移不超过 16 px 的对象点。
- `background`。

layer/step 的主要排序分数只使用 `moving_object`，背景仅用于检查位置编码和相机稳定性。

## 6. 单 case 通过门槛

只有以下条件全部满足才启动批量：

1. 49 pixel frames 必须编码为 13 latent frames，DiT grid、Q/K shape、坐标映射全部通过断言。
2. 6 layers x 5 steps 的 30 个组合全部产生有限指标，无 OOM、NaN 或缺失结果。
3. 至少一个中层配置（layer 5-23）在 moving-object 上满足以下任一条件：PCK@32 比静态 baseline 高至少 5 个百分点，或 mean error 至少降低 10%。
4. 该配置的预测轨迹不是静态退化：预测平均位移至少达到 GT 平均位移的 10%。
5. background 指标单独记录，不参与上述 gate。

如果 gate 未通过，停止自动批量，保留失败报告并先检查：坐标映射、时间锚点、Q/K 捕获位置、噪声构造和 token 分辨率。不得通过降低 gate 或挑单个点来强行通过。

## 7. 批量选择规则

- 对 50 个 base case 汇总每个固定 layer/step 的 moving-object 指标。
- 主排序：跨 case macro-average PCK@32；次排序：median error、direction cosine、rigidity error。
- 同时报告 bootstrap 95% confidence interval、每个 F1/F2/F3 family 的结果和成功 case 比例。
- 候选配置必须在多个 case/family 上稳定，不能只依赖 case 019。
- 最终保留最多 3 个候选 layer/step：最佳准确率、最佳中高噪声配置、最佳低噪声配置；如果它们实际为同一配置则合并。

## 8. 存储与清理

- 不保存完整 Q/K、hidden feature、VAE latent、逐 layer attention matrix 或去噪中间视频。
- 每层 hook 内即时把 Q/K 转换为 query 轨迹和紧凑统计，随后删除 Q/K。
- 每个样本只保存压缩 NPZ/JSON：query、CoTracker anchor tracks、visibility、每个 layer/step 的预测坐标和指标。
- 默认只为单 case 和全局最佳候选生成轨迹视频/heatmap；其他配置可由 seed、manifest 和指标重新生成。
- `.tmp`、失败 forward 缓存和可再生可视化由清理脚本删除；CoTracker 轨迹体积小且生成成本较高，默认保留。
- 每个 worker 在样本完成后调用 `torch.cuda.empty_cache()`，写入 `complete.json` 后才视为成功，支持断点续跑。
- 启动前检查 `/data` 可用空间；低于 20 GB 时停止创建可视化，低于 10 GB 时停止新样本并安全退出。

## 9. 执行阶段

1. `prepare`：为 case 019 创建区域 query、CoTracker 轨迹和 visibility。
2. `single-case scan`：运行原生协议完整 6x5 扫描。
3. `gate`：生成 `gate_report.json` 和 layer-step 表；通过后才继续。
4. `batch prepare`：流式准备 50 个 base case 的紧凑 CoTracker 文件。
5. `batch scan`：按 case 分片到 tmux worker；每个 worker 固定 GPU，模型只加载一次。
6. `aggregate`：输出 layer-step 排名、family 分层统计、置信区间和候选配置。
7. `visualize`：仅重建最佳候选的代表 case 可视化并整理 Markdown。

## 10. 产物

- 单 case：`single_case/case_019_wheel_hits_block_base/`。
- 批量轨迹：`tracks_base/`。
- 批量指标：`batch_base/worker_*/`。
- 聚合：`aggregate_base/summary.csv`、`aggregate_base/best_configs.json`。
- 日志：`logs/`。
- 运行 manifest 记录模型路径、git commit、数据路径、seed、scheduler、layer、step、真实 timestep、输入尺寸和协议版本。

## 11. 解释边界

- 原生协议证明的是 Wan latent 时间块中的跨时间对应，不是逐像素帧光流。
- Q/K 可匹配不等于模型真实全时空 attention 全部集中在 GT；matching softmax 和全局 attention 指标分开报告。
- CoTracker 是伪 GT，遮挡、快速旋转和小物体存在误差；visibility 和 renderer mask 用于过滤，但不能消除所有跟踪偏差。
- 仅当固定 layer/step 在多 case 的 moving-object 指标上稳定优于静态 baseline，才能称其“可用于判断 motion”。
