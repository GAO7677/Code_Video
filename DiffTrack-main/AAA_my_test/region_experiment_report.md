# CogVideoX 区域级 Q/K 跨帧对应实验

## 1. 实验目的

检验 CogVideoX-2B 的 self-attention Q/K 是否真正包含**物体级跨帧点对应**，并判断此前全图网格得到的高 PCK 是否主要由静态背景贡献。同时对比两种真实视频编码方式，确认时序 VAE 压缩和轨迹插值会不会扭曲运动物体的匹配结果。

## 2. 公共设置

- 视频：`case_019_wheel_hits_block/base`，前 49 帧，固定相机。
- 区域：物体 A（高速 wheel）、物体 B（crate box）、背景。
- Mask：数据集自带的无损 renderer instance IDs；物体 mask 腐蚀后采样，背景排除物体及其边缘。
- Query：第 0 帧，每个区域独立采样 32 点，共 96 点。
- 伪真值：CoTracker3 Offline 轨迹与 visibility。
- Q/K：CogVideoX-2B，`layer=17`，step 49；前后向 Q/K correlation 对称化后取稠密对应。
- 指标：仅在 CoTracker 可见位置统计位置误差与 PCK。

三个区域的采样位置：

| 物体 A | 物体 B | 背景 |
|---|---|---|
| ![物体 A 采样](../../../../../data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks_framewise/case_019_wheel_hits_block_base/layer17_step49/object_a/mask_points.png) | ![物体 B 采样](../../../../../data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks_framewise/case_019_wheel_hits_block_base/layer17_step49/object_b/mask_points.png) | ![背景采样](../../../../../data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks_framewise/case_019_wheel_hits_block_base/layer17_step49/background/mask_points.png) |

> 若 Markdown 阅读器不允许跨工作区加载图片，可直接查看[本地 dashboard](http://127.0.0.1:8765/)或下文视频。

## 3. 实验一：整段时序 VAE 编码

### 方案

将完整 49 帧一次性输入 CogVideoX 时序 VAE：

```text
49 个像素帧
  -> 时序 VAE 编码
13 个 latent frames
  -> 计算 13 个时间位置的 Q/K correspondence
  -> 线性插值
49 帧预测轨迹
```

然后将插值得到的 49 帧轨迹与逐帧 CoTracker 轨迹比较。

### 结果

| 区域 | CoTracker 可见率 | 平均误差 | PCK@4 | PCK@8 | PCK@16 |
|---|---:|---:|---:|---:|---:|
| 物体 A：高速 wheel | 56.95% | 30.86 px | 2.79% | 11.38% | 32.29% |
| 物体 B：crate box | 100.00% | 6.31 px | 52.80% | 75.26% | 91.28% |
| 背景 | 98.72% | 4.02 px | 50.99% | 96.83% | 100.00% |

可视化：

- 物体 A：[CoTracker 与 Q/K 叠加视频](/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks/case_019_wheel_hits_block_base/layer17_step49/object_a/overlay_comparison.mp4)
- 物体 B：[CoTracker 与 Q/K 叠加视频](/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks/case_019_wheel_hits_block_base/layer17_step49/object_b/overlay_comparison.mp4)
- 背景：[CoTracker 与 Q/K 叠加视频](/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks/case_019_wheel_hits_block_base/layer17_step49/background/overlay_comparison.mp4)

视频中圆点/粗线是 CoTracker，方框/细线是 Q/K。物体 A 的 Q/K 轨迹在高速运动开始后表现出明显滞后，而背景基本重合。

### 为什么这个方案不适合逐帧点匹配

1. **时间轴不再一一对应。** 49 个像素帧被压缩为 13 个 latent frames，一个 latent slice 汇聚邻近多帧信息，不能严格对应某一像素帧。
2. **引入 VAE 时序混合。** latent descriptor 已包含相邻帧内容，无法将后续匹配完全归因于 DiT 的 Q/K correspondence。
3. **36 帧不是模型预测。** 13→49 插值产生的大部分轨迹点并非 Q/K 直接输出，却被作为逐帧预测参与 PCK。
4. **对运动速度有系统性偏置。** 背景近似静止，插值几乎不造成误差；高速运动、碰撞和方向突变不满足线性假设，因此误差被显著放大。

因此，实验一只能作为 `temporal VAE + interpolation` 基线，不能据此判断 Q/K 缺乏运动物体 correspondence。

## 4. 实验二：逐帧独立 VAE 编码

### 方案

按照 DiffTrack 论文 point-tracking 主评测协议，将目标帧分为 4 个 chunk：

```text
chunk 0: frame 0 + frames 1-12
chunk 1: frame 0 + frames 13-24
chunk 2: frame 0 + frames 25-36
chunk 3: frame 0 + frames 37-48
```

每个 chunk 的 13 帧分别独立 VAE 编码。实际运行中四个 chunk 均满足：

```text
13 pixel frames -> 13 latent frames
```

第 0 帧 Q 与每个真实目标帧 K 直接匹配，再按原帧索引拼成 49 帧轨迹；**不进行时间插值**。实验二复用实验一完全相同的 query points、CoTracker 轨迹和层，并按论文 `evaluate_tapvid.py` 的 tracking pipeline 执行。

### 结果

| 区域 | CoTracker 可见率 | 平均误差 | PCK@4 | PCK@8 | PCK@16 |
|---|---:|---:|---:|---:|---:|
| 物体 A：高速 wheel | 56.95% | 9.32 px | 26.36% | 61.90% | 88.39% |
| 物体 B：crate box | 100.00% | 4.05 px | 68.36% | 94.73% | 99.87% |
| 背景 | 98.72% | 4.48 px | 48.35% | 96.83% | 98.94% |

可视化：

- 物体 A：[CoTracker 与 Q/K 叠加视频](/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks_framewise/case_019_wheel_hits_block_base/layer17_step49/object_a/overlay_comparison.mp4)
- 物体 B：[CoTracker 与 Q/K 叠加视频](/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks_framewise/case_019_wheel_hits_block_base/layer17_step49/object_b/overlay_comparison.mp4)
- 背景：[CoTracker 与 Q/K 叠加视频](/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks_framewise/case_019_wheel_hits_block_base/layer17_step49/background/overlay_comparison.mp4)
- 新旧协议代表帧对比：[protocol_visual_comparison.jpg](/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks_framewise/case_019_wheel_hits_block_base/layer17_step49/protocol_visual_comparison.jpg)
- 交互页面：[本地 dashboard](http://127.0.0.1:8765/)

![两种协议的代表帧对比](../../../../../data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks_framewise/case_019_wheel_hits_block_base/layer17_step49/protocol_visual_comparison.jpg)

## 5. 两种完整协议的对比

| 区域 | 整段编码 PCK@8 | 逐帧编码 PCK@8 | PCK@8 变化 | 平均误差变化 |
|---|---:|---:|---:|---:|
| 物体 A：高速 wheel | 11.38% | 61.90% | **+50.52 pp** | 30.86 → 9.32 px |
| 物体 B：crate box | 75.26% | 94.73% | **+19.47 pp** | 6.31 → 4.05 px |
| 背景 | 96.83% | 96.83% | 0.00 pp | 4.02 → 4.48 px |

除 VAE 编码与 chunking 外，两条公开代码路径的 Q/K softmax temperature 也不同：实验一的 `analyze_real` processor 使用 `sqrt(head_dim)=sqrt(64)`，实验二按论文 `evaluate_tapvid.py` 使用 `sqrt(num_heads)=sqrt(30)`。因此下表比较的是**两个完整协议**，不能把全部数值变化严格归因于单一变量；但实验一的时间轴错位和插值伪轨迹不依赖该差异，已经足以说明它不适合逐帧 PCK。

结果分析：

- 物体 A 的大幅提升说明，实验一的 11.38% 不能解释为 Q/K 完全无法跟踪高速物体；时间压缩、插值及 tracking 实现差异共同造成了严重低估。
- 物体 B 在论文 tracking 协议下达到 94.73%，碰撞阶段的时间对应明显优于旧路径。
- 背景 PCK@8 完全不变，符合静态区域几乎不受时间插值影响的预期。
- 逐帧编码后仍存在区域差异：背景 96.83%、物体 B 94.73%、物体 A 61.90%。高速运动、遮挡和碰撞仍是 Q/K correspondence 的主要困难。

## 6. 结论

1. 按论文 tracking 协议，CogVideoX-2B 的 Q/K 包含明确的物体级跨帧对应：物体 B PCK@8 为 94.73%，高速物体 A 为 61.90%。
2. 整段时序 VAE 编码不适合逐帧 correspondence 评估；其时间压缩和轨迹插值会系统性低估运动物体，尤其是高速、碰撞和非线性运动。
3. 全图平均 PCK 仍会被背景主导。正确协议下背景为 96.83%，但高速物体仅 61.90%，因此后续必须按对象/背景分组报告。
4. 物体 A 的 CoTracker 可见率仅 56.95%，当前 PCK 只在可见点上统计；剩余误差同时包含 Q/K 失配和 CoTracker 在高速/遮挡阶段的不确定性，后续应结合 renderer 轨迹进一步验证。

## 7. 结果与复现

- 实验一结果：`/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks/case_019_wheel_hits_block_base/layer17_step49`
- 实验二结果：`/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks_framewise/case_019_wheel_hits_block_base/layer17_step49`
- 协议对比数据：[protocol_comparison.json](/data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks_framewise/case_019_wheel_hits_block_base/layer17_step49/protocol_comparison.json)
- 实验一脚本：[analyze_region_tracks.py](/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/analyze_region_tracks.py)
- 实验二脚本：[rerun_region_tracks_framewise.py](/home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/rerun_region_tracks_framewise.py)

当前 dashboard 服务地址为 `http://127.0.0.1:8765/`。服务停止后，可用以下前台命令重新启动：

```bash
python3 -m http.server 8765 \
  --bind 0.0.0.0 \
  --directory /data/gaoya/agent-data/outputs/difftrack_0718toy/region_tracks_framewise/case_019_wheel_hits_block_base/layer17_step49
```
