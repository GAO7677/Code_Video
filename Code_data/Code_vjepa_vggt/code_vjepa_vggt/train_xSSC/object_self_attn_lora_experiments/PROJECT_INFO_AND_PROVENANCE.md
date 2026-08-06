# xSSC-Wan 项目信息与权重溯源

## 1. 项目概述

本项目以 Wan2.2 TI2V-5B 为视频生成主干，研究 
(1) 物理仿真数据集SFT训练
(2) 视频自监督模型提取 object信息做 Cross-Attn 条件注入
(3) Self-Attention 不同head如何影响视频中object的运动

输入： 8 帧 context video + text
输出： 49 帧、512x896 视频；

训练时采用49帧视频训练，显存约占36.95GiB / 47.99GiB，推理时可以跑189帧


http://127.0.0.1:8092/attention-lora-pck32-seed90094?v=1


### 1.1 相关可视化页面

| 页面 | 简要说明 |
|---|---|
| [Wan2.2 Legacy TI2V Test5](http://127.0.0.1:8092/wan22-ti2v-legacy-test5?v=1) | 历史 Wan2.2-TI2V 基线视频墙：使用 `legacy_diffsynth.WanVideoPipeline`，仅输入 prompt 和首帧，不使用 context video；配置为 seed 42、CFG 5、40 去噪步、49 帧、704x1280。用于检查未做 Head 干预时的原始生成结果。 |


## 2. 训练阶段



### 2.1 v2v适配训练

使wan可以接受任意输入帧数的video，对Wan 的 30 个block微调lora：
- 输入 24 帧，384x672，mixed context sampling 
- 4 GPU，每卡 batch=1，gradient accumulation=4
- Self-Attention, Text Cross-Attention, FFN, LoRA rank=32 
- 可训练参数 80.609M。
- step-010000

### 2.2 仿真数据集微调self-attention LoRA
### 2.3 视频自监督模型微调cross-attn + self-attention LoRA

http://10.176.42.45:8844/physiciq-metrics/

- no-object 训练：仿真数据集占比越大、指标越好，画面质量也好，但是美学下降了
- full head-SA 情况下+ object branch 指标还会下降，object branch 基本没啥用

```text
8-frame context video
  -> SAM2 AMG masks + filtering
  -> pseudo boxes
  -> frozen DINOv3 xSSC
  -> context slots [B, 8, 11, 512]
  -> LayerNorm + Linear(512, 3072) + learned time embedding
  -> object tokens [B, 8*11, 3072] = [B, 88, 3072]
  -> object cross-attention in Wan DiT
  -> 49-frame generated video
```

xSSC 使用 bbox 初始化 frame-0 slots，然后通过 transition 和 SlotAttention 沿 context 时间递推。Wan 训练阶段冻结 SAM2、DINO 和 xSSC，只训练实验配置指定的 projector、gate、object-attention LoRA 或 Self-Attention LoRA。




参数口径：

| 组件 | 参数量 | xSSC 训练状态 | Wan 训练状态 |
|---|---:|---|---|
| DINOv3 ViT-L/16 | 303.130M | 冻结 | 冻结 |
| xSSC 非 Backbone | 81.044M | 可训练 | 冻结 |
| 合计 | 384.174M | 81.044M 可训练 | 全部冻结 |





## 2.4. 实验方案与参数

下表的“冻结参数”按训练涉及组件统计；SAM2 可能作为预处理单独执行，因此该总量不等于同时驻留 GPU 的参数量。

| 方法 | 可训练模块 | 可训练参数 | 冻结参数 | 总涉及参数 | Object/xSSC | 数据 |
|---|---|---:|---:|---:|---|---|
| Object-only | projector/time + object-attn LoRA + gate | 25.459M | 11,994.007M | 12,019.466M | xSSC-26k | 30/30/40 |
| Full-SA + Object | Object-only + 全30层 Self-Attn Q/K/V/O LoRA | 49.052M | 11,994.007M | 12,043.059M | xSSC-26k | 30/30/40 |
| S-head59 + Object | Object-only + S59 compact head LoRA | 34.683M | 11,994.007M | 12,028.690M | xSSC-26k | 30/30/40 |
| T-head70 + Object | Object-only + T70 compact head LoRA | 34.863M | 11,994.007M | 12,028.870M | xSSC-26k | 30/30/40 |
| Full-SA + Object + Dedup | 与 Full-SA + Object 相同；Dedup 零参数 | 49.052M | 11,994.007M | 12,043.059M | xSSC-26k | 30/30/40 |
| Full-SA + Object + Dedup (xSSC-50k) | 同上 | 49.052M | 11,994.007M | 12,043.059M | xSSC-50k | 30/30/40 |
| T-head70 + Object + Dedup (xSSC-50k) | Object-only + T70 compact head LoRA | 34.863M | 11,994.007M | 12,028.870M | xSSC-50k | 30/30/40 |
| Full-SA + No-Object | 全30层 Self-Attn Q/K/V/O LoRA | 23.593M | 11,385.387M | 11,408.980M | 完全关闭 | 30/30/40 |
| Full-SA + No-Object, PyBullet 100% | 同上 | 23.593M | 11,385.387M | 11,408.980M | 完全关闭 | 100/0/0 |
| Full-SA + No-Object, Kubric 100% | 同上 | 23.593M | 11,385.387M | 11,408.980M | 完全关闭 | 0/100/0 |
| T-head70 + No-Object | T70 compact head LoRA | 9.404M | 11,385.387M | 11,394.791M | 完全关闭 | 30/30/40 |
| Motion-head100 + No-Object | Motion Top100 compact head LoRA | 11.076M | 11,385.387M | 11,396.462M | 完全关闭 | 30/30/40 |
| Object-only + Official xSSC | official projector/time + object-attn LoRA + gate | 24.672M | 11,643.882M | 11,668.553M | official DINOv2 xSSC | 30/30/40 |

Slot-Dedup 使用跨时间 slot-track 的 mean-frame cosine，相似度阈值 0.94，模式为 merge，最少保留 3 个 slot。它发生在 xSSC 输出之后、object-token projection 之前，不新增可训练参数。

## 3. Head 分类

### 3.1 原始实验范围

- 模型：Wan+LoRA、Wan+xSSC、PhysRVG。
- 数据：20 个 case。
- 稳定性范围：三个模型都完成的 22 个公共 seed。
- 去噪步：5、15、25、35。
- Head：30 blocks x 24 heads，共 720 个。
- 保存粒度：每个 model/case/seed/denoise-step/block/head 的全部原始特征值、rank、score 和角色。

### 7.2 分类分数

```text
score_S = 0.55 rank(local_enrichment)
        + 0.45 rank(same_frame_mass)

score_T = 0.55 rank(trajectory_selectivity_log2)
        + 0.25 rank(trajectory_enrichment)
        + 0.20 rank(mean_time_distance)

score_P = 0.75 rank(fixed_position_enrichment)
        + 0.25 rank(aligned_enrichment)

score_C = 0.55 rank(object_context_enrichment)
        + 0.25 rank(full_context_enrichment)
        + 0.20 rank(history_bias)

score_G = 0.60 rank(full_entropy)
        + 0.25 rank(full_mean_time_distance)
        + 0.15 rank(-same_frame_mass)
```


最终公共稳定类别要求三个模型得到相同且非 M 的角色：

| 类别 | 含义 | Head 数量 |
|---|---|---:|
| S | 空间局部/同帧 | 159 |
| T | 运动轨迹 | 13 |
| P | 固定位置 | 82 |
| C | Context | 20 |
| G | 全局候选 | 75 |
| M | 混合、跨模型不一致或不稳定 | 371 |




- `http://127.0.0.1:8844/head-evidence/common-stc-all-heads-qk-seed851/`
- `http://127.0.0.1:8844/head-evidence/head-role-depth-distribution/`




http://localhost:8946/s-head-integrated-analysis/index.html#dominant




| 页面 | 简要说明 |
|---|---|
| [Wan+LoRA 50-Seed Attention Sweep](http://127.0.0.1:8092/attention-additive-lora-seed-sweep?v=1&experiment=alpha090&stage=all_steps&group=top100) | 对 `0613pybullet_sample_001460_w002` 的 50 个 seed 比较原始 Wan+LoRA 与 Attention 干预结果。当前链接选择 Top100 Heads、全部 40 个去噪步和 `alpha=0.9` 加性缩放，并展示同 seed 视频及干预前后热力图，用于判断 Head 干预效果是否跨 seed 稳定。 |
| [Neighbor Diagonal Ranking](http://127.0.0.1:8092/neighbor-diagonal-ranking?v=4) | 在 S039 对 30 blocks x 24 heads 共 720 个 Head 排名。严格分数同时要求相邻三帧的时间对角线响应均衡，以及所有目标帧块内部具有稳定的空间对角线纯度；页面提供三模型切换、30x24 矩阵、PCK@32 和逐 Head Q@K 热力图。 |
| [All-Steps Rankings](http://127.0.0.1:8092/all-steps/rankings?v=4) | 覆盖 40 steps x 30 blocks x 24 heads 共 28,800 个组合，按目标物体或背景的 PCK@8/16/32 与平均位置误差排序；同时展示 GT、LoRA、Baseline 三模型等权综合排名、单模型全局排名及逐步 30x24 热力图，用于定位运动追踪能力主要出现在哪个去噪阶段、Block 和 Head。 |

