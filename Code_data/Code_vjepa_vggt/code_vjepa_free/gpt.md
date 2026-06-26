先不要实现完整 V-JEPA-guided Wan pipeline。请先做前置验证实验。

目标是验证 V-JEPA2/V-JEPA2.1 中层特征是否能作为 Wan2.2-TI2V-5B 的 training-free attention guidance。请完成以下任务：

1. 用 Wan2.2-TI2V-5B 生成一组 baseline videos，覆盖碰撞、弹跳、遮挡、重力、液体、物体持久性等物理场景。推理脚本和权重记录在/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/1.executed.ipynb，输出结果存放在(/data/gaoya/AAA_test_video/0626vjepa_free/test/{runname})runname，可以写对应代码版本名，比如xxx_v1
2. 对这些视频提取 V-JEPA 的多层 hidden states，至少包括 early、PEZ、middle、late/final 层。（vjepa官方代码仓库在/home/gaoya/Code_Video/vjepa2-main，注意如果需要修改代码，不要在/home/gaoya/Code_Video/vjepa2-main 目录下修改，而是在/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_exp 目录下复制一份相关文件然后修改）
3. 从每层构造 feature affinity、temporal correspondence、motion saliency。如果 predictor 接口可用，再构造 surprise/residual。
4. 评估这些信号是否能定位 Wan baseline 中的物理错误时间窗口和错误区域。
5. 记录 Wan visual self-attention 的轻量统计，比较 Wan attention 与 V-JEPA relation 的相关性和互补性。
6. 给出信号排序：PEZ relation、middle relation、surprise、motion saliency、global embedding、raw attention。
7. 只有当前置验证显示 V-JEPA relation 有效时，再实现最小 training-free 注入：V-JEPA low-rank relation → Wan attn1 Q/K augmentation。
8. 注入只作用在 Wan visual self-attention，不改 cross-attention，不替换 hidden state，不训练 adapter。
9. 输出一份 markdown 报告，明确说明是否值得继续做完整方法，以及推荐使用哪些 V-JEPA 层、哪些 Wan 层、哪种信号形式。


对，应该先把方案改成：**前置验证实验优先，确认 V-JEPA 信号确实和 Wan 的生成失败/attention 行为有关，再做完整注入。** Codex 任务不需要一上来写完整 pipeline，而是按阶段推进，每阶段有 go/no-go 标准。

下面是更适合交给 Codex 的执行版。

---

# Codex 执行方案：V-JEPA → Wan2.2-TI2V 前置验证 + 最小注入

## 目标

验证并实现一个 training-free 方法：

> 从 V-JEPA2 / V-JEPA2.1 中层特征提取时空关系先验，用于调制 Wan2.2-TI2V-5B 的 visual self-attention，而不是训练 adapter、不是 reranking、不是替换 hidden state。

第一阶段重点不是生成效果，而是回答四个问题：

1. **V-JEPA 哪些层真的携带可用物理/运动信号？**
2. **哪种信号形式最适合注入 Wan attention？**
3. **Wan 哪些层/step 的 self-attention 最适合被调制？**
4. **training-free 注入是否会破坏 prompt adherence 和画面质量？**

---

# 阶段 0：建立 baseline 和数据集

## 0.1 Wan baseline 生成集

先固定一组 prompt，用 Wan2.2-TI2V-5B 生成 baseline。每个 prompt 至少 4 个 seed。

优先用容易暴露物理问题的 prompt：

```text
1. A red ball rolls down a wooden ramp and collides with a blue cube.
2. A glass falls from a table and shatters on the floor.
3. A basketball bounces on the ground several times.
4. A toy car drives behind a box and reappears on the other side.
5. A stack of blocks is pushed and topples over.
6. Water is poured from a cup into a bowl.
7. A pendulum swings back and forth.
8. A ball rolls off a table and falls to the ground.
```

I2V prompt 也要准备一组，因为 Wan2.2-TI2V 支持 image-to-video，首帧有 object layout，更适合验证 V-JEPA 的 object persistence / motion prior。

## 0.2 保存内容

每个样本保存：

```text
video.mp4
prompt.txt
seed.txt
Wan intermediate latents or decoded previews at several denoising steps
Wan selected self-attention statistics
```

不要求一开始保存完整 attention map，可以先保存：

```text
attention entropy
top-k attention concentration
temporal attention ratio
spatial-local attention ratio
cross-frame attention ratio
```

## 0.3 baseline 评价

先人工标注每个视频的主要物理错误类型：

```text
motion discontinuity
object teleportation
object deformation
collision failure
wrong bounce
wrong gravity
occlusion failure
object permanence failure
liquid/material failure
prompt mismatch
```

这一步很重要。后续 V-JEPA 信号要证明能定位这些错误。

---

# 阶段 1：V-JEPA 层位和信号形式验证

这一阶段不接 Wan，只分析 V-JEPA 特征。

## 1.1 模型选择

至少测两个：

```text
V-JEPA2-L 或 V-JEPA2-g：主 baseline
V-JEPA2.1：如果环境可用，作为 dense/contact/object prior baseline
```

如果显存紧张，先用 V-JEPA2-L。不要一开始上 gigantic。

## 1.2 接出层位

按模型 depth 比例接层，不要只接 final layer。

对于 V-JEPA2-g / g-384，优先接：

```text
early:  blocks[4], blocks[6]
PEZ:    blocks[12], blocks[13], blocks[14], blocks[15]
middle: blocks[18], blocks[20], blocks[24]
late:   blocks[32] 或 final
```

对于 V-JEPA2-L，优先接：

```text
early:  blocks[3], blocks[5]
PEZ:    blocks[7], blocks[8], blocks[9], blocks[10]
middle: blocks[12], blocks[14], blocks[16]
late:   blocks[20] 或 final
```

层位假设：

| 层段              | 预期功能                                             | 是否适合注入 Wan                |
| --------------- | ------------------------------------------------ | ------------------------- |
| early           | 局部纹理、motion saliency、运动强弱                        | 辅助，不做主 prior              |
| PEZ / 1/3 depth | 方向、短时物理、object-motion binding                    | 最适合 attention relation    |
| middle          | violation-of-expectation、next-frame plausibility | 适合 surprise / scheduler   |
| late/final      | 全局语义、任务目标                                        | 不适合 dense attention prior |

## 1.3 待比较的 V-JEPA 信号

Codex 需要实现五类信号导出，不注入，只保存和分析：

### Signal A：feature affinity

从某层 hidden tokens 得到：

```text
token feature → L2 normalize → pairwise similarity / low-rank embedding
```

用途：构造 Wan attention relation prior。

优先级最高。

### Signal B：temporal correspondence

计算相邻帧或间隔帧 token 的 nearest-neighbor matching：

```text
frame t token i 在 frame t+1 / t+k 中最相似的 token
```

用途：验证 V-JEPA 是否能跟踪同一物体或同一运动区域。

### Signal C：motion saliency

计算跨帧 feature difference：

```text
||h_t - h_{t-1}||
```

用途：找到运动区域。

这是辅助信号，适合 regional gate，不适合作为主 attention prior。

### Signal D：surprise / prediction residual

如果 predictor 接口可用，计算：

```text
V-JEPA predicted future feature vs actual encoded future feature
```

用途：定位物理异常区域，作为 attention guidance strength scheduler。

如果 predictor 接口复杂，第一版可以先跳过，不阻塞主流程。

### Signal E：raw attention map

如果实现方便可以保存；不方便就不做。它不是主方案。

raw attention map 只用于分析，不用于 hard replacement。

---

# 阶段 2：前置验证实验 A：V-JEPA 信号是否能定位物理错误

## 实验目的

验证 V-JEPA 信号是否和 Wan baseline 的物理错误相关。

## 输入

使用阶段 0 生成的 Wan baseline videos。

## 操作

对每个视频：

1. 输入 V-JEPA。
2. 提取不同层的 signal A/B/C/D。
3. 将信号上采样到视频帧。
4. 和人工标注的错误区域/错误时间窗口对齐。

如果没有像素级标注，先做粗粒度：

```text
错误发生帧区间：例如 frame 20-35
错误对象区域：手动 bbox 或粗 mask
```

## 评价指标

### 时间定位

```text
surprise / feature inconsistency peak 是否落在错误时间窗口
```

指标：

```text
Temporal Hit@K
Peak-in-error-window ratio
AUC over frame-level error labels
```

### 空间定位

```text
motion saliency / surprise map 是否覆盖错误物体或接触区域
```

指标：

```text
bbox IoU
pointing game accuracy
foreground/background score ratio
```

### 层位比较

比较：

```text
early vs PEZ vs middle vs late
```

预期结果：

```text
PEZ / middle > early > late
```

## go/no-go 标准

继续做 Wan 注入的最低标准：

```text
1. PEZ 或 middle 层的错误时间定位明显优于 random。
2. PEZ feature affinity 在 motion/collision/object persistence case 上优于 final layer。
3. motion saliency 或 surprise 至少能粗略覆盖主要运动对象。
```

如果这些都不成立，不要继续做 attention 注入，应改成 WMReward-style reranking 或重新选择 video foundation model。

---

# 阶段 3：前置验证实验 B：V-JEPA relation 是否和 Wan self-attention 有互补性

## 实验目的

确认 V-JEPA relation 不是 Wan 已经完全学到的东西。否则注入意义不大。

## 输入

同一批 Wan baseline generation。

## 操作

在 Wan denoising 过程中，不做干预，只记录 selected layers 的 self-attention statistics。

Wan 层选择：

```text
early:      blocks[0-5]
early-mid:  blocks[6-13]
mid:        blocks[14-20]
late:       blocks[21-29]
```

不要保存完整 attention map，先保存轻量统计。如果显存允许，再对低分辨率/短帧实验保存小规模 attention。

## 比较内容

对每个 step/layer/head：

```text
Wan attention relation vs V-JEPA feature affinity
Wan temporal attention ratio vs V-JEPA temporal correspondence confidence
Wan attention entropy vs V-JEPA surprise
```

## 预期发现

理想情况是：

```text
1. Wan early-mid layers 和 V-JEPA PEZ relation 有中等相关性。
2. 物理失败时，Wan attention entropy 或 temporal attention ratio 异常。
3. V-JEPA relation 在失败区域提供了更稳定的 cross-frame correspondence。
```

如果 V-JEPA 和 Wan attention 完全不相关，也不一定坏，说明它可能提供外部先验；但如果 V-JEPA signal 和错误无关，就不该注入。

## go/no-go 标准

继续做注入的标准：

```text
1. Wan blocks[6-13] 或 blocks[10-18] 对结构/运动最敏感。
2. V-JEPA PEZ relation 在错误 case 中比 Wan 自身 attention 更稳定。
3. 不需要 V-JEPA final layer。
```

---

# 阶段 4：前置验证实验 C：信号形式排序

## 目标

确定到底该用哪种 V-JEPA 信号注入。

## 待比较信号

按优先级测试：

```text
A. PEZ feature affinity / low-rank relation
B. PEZ + middle ensemble relation
C. predictor surprise / residual
D. motion saliency
E. global pooled embedding
F. raw V-JEPA attention map
```

## 评价方式

不一定立刻生成完整视频。先做 offline scoring：

```text
1. 错误定位能力
2. temporal correspondence 稳定性
3. object region consistency
4. 和 prompt/object region 的冲突程度
5. 和 Wan attention 的互补性
```

## 预期排序

```text
PEZ+middle relation >= PEZ relation > surprise > motion saliency > global embedding > raw attention
```

如果实验不支持这个排序，也要保留结果。不要强行按假设推进。

---

# 阶段 5：最小 training-free 注入实验

只有前置验证通过后才做。

## 5.1 第一版注入位置

只注入 Wan visual self-attention：

```text
Wan blocks[6-13]: strong injection
Wan blocks[14-19]: weak injection
Wan blocks[0-5], [20-29]: off
```

不要动：

```text
text cross-attention
image condition cross-attention
VAE
scheduler
CFG
```

## 5.2 第一版注入形式

使用：

```text
V-JEPA PEZ/middle low-rank relation → Wan attn1 Q/K augmentation
```

不要做：

```text
V-JEPA raw token → Wan hidden replacement
V-JEPA raw attention → Wan attention hard replacement
V-JEPA token → Wan value replacement
```

原因：training-free 条件下空间不对齐，风险太高。

## 5.3 denoising step 调度

只在中前段启用：

```text
0% - 20% steps: off
20% - 70% steps: on
70% - 85% steps: decay
85% - 100% steps: off
```

理由：

```text
太早：latents 噪声大，V-JEPA preview 不可靠。
中段：结构和运动仍可改变。
太晚：容易破坏细节、纹理和 prompt adherence。
```

## 5.4 guidance strength sweep

先测试：

```text
lambda = 0.0
lambda = 0.1
lambda = 0.2
lambda = 0.35
lambda = 0.5
```

默认主结果用：

```text
lambda = 0.2 或 0.35
```

如果出现过平滑、冻结、prompt 偏离，降低 lambda。

---

# 阶段 6：最小生成验证

## 6.1 prompt set

用 8 个 prompt，每个 4 seed：

```text
32 baseline videos
32 guided videos
```

优先短视频、低分辨率：

```text
480p / 49 frames / 20-30 steps
```

跑通后再上：

```text
704x1280 / 81 frames / 30-50 steps
```

## 6.2 对比组

必须包括：

```text
1. Wan baseline
2. V-JEPA relation guidance
3. V-JEPA relation guidance with wrong layer
4. V-JEPA relation guidance with late Wan layers
5. motion saliency only
6. global modulation only, optional
```

这样能证明不是“随便扰动 Wan attention 都有效”。

## 6.3 人工评价维度

每个视频打分：指标使用/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/single_case/README.md中记录的


# 阶段 7：实验决策树

## 情况 A：V-JEPA signal 定位物理错误有效，注入也改善视频

继续完整化方法：

```text
主方法：PEZ/middle relation → Wan self-attn low-rank prior
辅助：surprise → lambda scheduler
扩展：V-JEPA2.1 for dense/contact cases
```

## 情况 B：V-JEPA signal 有效，但注入破坏画质

改为更软的方式：

```text
1. 只做 attention temperature，不做 relation bias
2. 只调 gate_msa，不改 Q/K
3. 只在高 surprise 区域局部启用
4. 降低 lambda
5. 只在 I2V case 使用
```

## 情况 C：V-JEPA signal 只对错误打分有效，无法稳定注入

转向：

```text
V-JEPA surprise-guided candidate selection
local window regeneration
tree search / particle filtering
```

但这会更接近 WMReward，创新性弱一些。

## 情况 D：V-JEPA signal 对 Wan 错误无明显相关性

停止该方向，换 teacher：

```text
V-JEPA2.1
DINOv3 video/dense feature
Depth/flow/contact-specific model
VideoMAE/InternVideo/MoAlign-style motion encoder
```

---

# Codex 交付物

让 Codex 先交付这些，而不是完整大 pipeline：

```text
1. Wan baseline generation script
2. V-JEPA feature extraction script
3. V-JEPA layer/signal analysis script
4. Wan attention statistics logger
5. Offline correlation analysis notebook/script
6. 前置验证报告 markdown
7. 通过验证后，再实现最小 low-rank Q/K guidance
```

报告必须包含：

```text
1. 哪个 V-JEPA 层最好
2. 哪个信号最好
3. 哪个 Wan 层最适合注入
4. V-JEPA signal 是否能定位物理错误
5. 是否值得继续做 training-free attention injection
```

---
