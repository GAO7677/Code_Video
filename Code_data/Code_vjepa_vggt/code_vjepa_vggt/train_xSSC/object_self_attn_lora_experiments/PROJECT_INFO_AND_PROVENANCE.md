# 物理合理视频续写

## 1. 项目概述

本项目以 Wan2.2 TI2V-5B 为视频生成主干，研究 
(1) 物理仿真数据集SFT训练
(2) 视频自监督模型提取 object信息做 Cross-Attn 条件注入
(3) Self-Attention 不同head如何影响视频中object的运动

输入： 8 帧 context video + text
输出： 49 帧、512x896 视频；

训练时采用49帧视频训练，显存约占36.95GiB / 47.99GiB，推理时可以跑189帧






## 2. 训练阶段



### 2.1 v2v 适配训练

使wan可以接受任意输入帧数的video，对Wan 的 30 个block微调lora：
- 输入 24 帧，384x672，mixed context sampling 
- 通用视频数据80%，20%物理仿真数据集，差不多70K
- 4 GPU，每卡 batch=1，gradient accumulation=4
- Self-Attention, Text Cross-Attention, FFN, LoRA rank=32 
- 可训练参数 80.609M。
- step-010000


【可视化展示】
http://10.176.42.45:8844/formal-physiciq-compare/?page=1




### 2.2 v2v + 仿真数据集微调 self-attention LoRA


### 2.3 v2v + self-attention LoRA + 视频自监督模型微调cross-attn 



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
利用冻结xSSC提取object信息，Wan DiT 生成object cross-attention，再输入到生成视频中。

xSSC 使用 bbox 初始化 frame-0 slots，然后通过 transition 和 SlotAttention 沿 context 时间递推。Wan 训练阶段冻结 SAM2、DINO 和 xSSC，只训练实验配置指定的 projector、gate、object-attention LoRA 或 Self-Attention LoRA。



### 2.4 指标&可视化

http://10.176.42.45:8844/physiciq-average-metrics/ 


http://10.176.42.45:8844/physiciq-checkpoint-curves/

http://10.176.42.45:8844/test5/

- **no-object 训练：仿真数据集占比越大、指标越好，画面质量也好，但是美学下降了**
- **full head-SA 情况下+ object branch 指标还会下降，object branch 基本没啥用**










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

## 3. 基于注意力的Head 分类

### 3.1 原始实验范围

- 模型：Wan+LoRA、Wan+xSSC、PhysRVG。
- 数据：20 个 case。
- 稳定性范围：三个模型都完成的 22 个公共 seed。
- 去噪步：5、15、25、35。
- Head：30 blocks x 24 heads，共 720 个。
- 保存粒度：每个 model/case/seed/denoise-step/block/head 的全部原始特征值、rank、score 和角色。
- **注意力落在同帧，还是其他帧，是否落在空间相邻区域**

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




- 不同head热力图：http://127.0.0.1:8844/head-evidence/common-stc-all-heads-qk-seed851/
- 各类head在block之间的分布：http://127.0.0.1:8844/head-evidence/head-role-depth-distribution/
- 不同类head输出置0对实验的影响 http://localhost:8946/s-head-integrated-analysis/index.html#dominant


## 4. PCK的Head 分类

### 4.1 筛选方法

Wan2.2 每个 DiT 有 30 个 block、每个 block 有 24 个 Head。单个去噪步包含
720 个 `(block, head)`；40 个去噪步共有 28,800 个
`(step, block, head)` 组合。

**对每个 object query，使用该 Head 的 Q@K 最高响应位置作为object 轨迹预测，与 GT 轨迹计算像素距离**
```text 
PCK@32 = 100 * count(distance <= 32 px) / count(valid comparisons)

```


按 PCK@32 降序得到 Top30/Top100，升序得到
Bottom30/Bottom100。



模型： GT teacher-forced、Wan+LoRA、Wan2.2 Baseline，

【可视化】http://127.0.0.1:8092/object-query-top100-mean-overlay?seed=47326&stage=all_steps&v=2

| 页面 | 流程角色 | 与 PCK Head 的关系 |
|---|---|---|
| [All-Steps Rankings](http://127.0.0.1:8092/all-steps/rankings?v=4) | PCK统计与筛选 | 覆盖 40×30×24 组合，提供单模型及三模型综合 Top/Bottom，是主要候选来源。 |
| [Neighbor Diagonal Ranking](http://127.0.0.1:8092/neighbor-diagonal-ranking?v=4) | 注意力结构分析 | 用空间对角纯度、时间连续性和跨帧均衡解释或重新排序同一批 Head，并与 PCK 对照。 |
| [Wan+LoRA 50-Seed Attention Sweep](http://127.0.0.1:8092/attention-additive-lora-seed-sweep?v=1&experiment=alpha090&stage=all_steps&group=top100) | 跨 seed 干预验证 | **干预top100head运动会受影响，干预bottom100head画面崩坏，top100更加关注object query跨帧位置** |


改变PCK head对应query热力图，可以改变运动【热力图】 http://127.0.0.1:8092/object-query-top100-mean-overlay?seed=47326&stage=all_steps&v=2

改变PCK head对应query热力图，可以改变运动 【视频】 
- 【002】 http://127.0.0.1:8092/object-query-frozen-trajectory?v=23&seed=47326&step=9&branch=conditional&viz=reverse&stage=all_steps&heatmap=s09_fixed_3


- 【0025】 http://127.0.0.1:8092/physiq025-object-query-frozen-trajectory?v=1&seed=13161&stage=all_steps&step=9&branch=conditional&variant=all

# 5. 发现&问题&后续
a. 微调的lora版本会出现**重复的新物体**
- 40step的object query PCK head热力图会关注到**错误物体位置**，改变PCK head 对应的热力图，可以改善。 **object的错误运动——pck head的跨帧注意力落在了错误的位置**。 (加一些运动平滑的attention正则项可以改善)
- 微调的lora版本【10step推理】会减轻**重复的新物体**，10step的热力图也不会关注到**重复的新物体**。(扰动PCK head 热力图是影响物体运动的充分条件？)
- **重复的新物体**问题尚未验证**基础模型**中是否存在，有可能是我自己训练的问题。  
- 对于不合理的轨迹，pck head的热力图本身没办法判断。可以借助一些外部规则，或者渲染出一个草图视频交给VLM判断。
b. 现有工作多数都是对于PCK head进行推理干预，降低成本。
- 可以考虑训练。loss加在PCK head 的attention上？VLM做reward？
c. object branch 对物理合理性的影响不大，仿真数据集作用更大
- 还没验证是信息本身没用还是条件注入的方式有问题，先从loss上去验证这个信息到底对指标有用不



