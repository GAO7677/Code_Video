# xSSC Slot Conditioning Experiment Plan

本文档用于比较两条 xSSC slot 条件路线，目标是判断哪一种更适合物理合理视频生成：

- 方案 A：All-frame slot cross-attention，把完整时间轴 xSSC slots 作为 object tokens。
- 方案 B：Slot-ControlNet，把 slot embedding 转成时空 feature map，以 ControlNet/adapter residual 的方式注入。

核心原则：先做 oracle upper bound，再做 predictor。oracle 阶段允许使用完整视频所有帧的真实 xSSC slots，用来判断该条件形式是否有价值；predictor 阶段只能从 ctx slots 预测未来 slots，才是可推理方案。

## 当前已知结论

1. full ctx slots 表现优于 pooled slots。
2. randomcrop pooled 与 centercrop pooled 在 step-000500 的 test_5 结果非常接近，20 个 case 上 random/center pooled 的平均 SSIM 为 0.9683。
3. centercrop pooled 没有系统性优于 randomcrop pooled：center 比 random 更接近 full ctx 的 case 数为 8/20，平均差值约 -0.0014。
4. 因此 pooled slots 效果一般，主因不太像 random crop，而更可能是时间平均压缩掉了 object trajectory，或 pooled branch 训练方式/条件注入能力不足。

## 统一实验设置

为了让两个方案可比，除被比较的条件注入方式外，其余设置尽量固定：

- 数据：30% 0717 PyBullet，30% PhyCo Kubric，40% OpenVidHD。
- 视频长度：49 frames。
- ctx 帧数：8 frames。
- xSSC：冻结。
- Wan 主体：冻结。
- 可训练参数：只训练新增 slot projector / adapter / ControlNet branch / object cross-attention LoRA。
- 评估集：`/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt`。
- 训练 checkpoint：优先比较 step-500、step-1000、step-1500、step-2000。
- 输出目录：`/data/gaoya/agent-data/outputs/AAA_physv/AAA_xSSC` 或已有 `train_xSSC/test_5` 对应方法目录。

## 方案 A：All-Frame Slot Cross-Attention

### 方法流程

对每个训练视频提取完整 49 帧的 frozen xSSC slots，得到 `[B,49,7,256]`。将这些 slots 对齐到 Wan latent 时间轴：如果 latent temporal length 为 `T_lat`，则每个 latent step 对应原视频中的一个中心帧或一个局部帧窗口，得到 `[B,T_lat,7,256]`；也可以先保留 `[B,49,7,256]` 做全帧 token。随后经过 `LayerNorm + Linear` 投影到 Wan hidden dim，并加入 frame/time embedding，flatten 成 `[B,T*7,3072]` object tokens，作为 object cross-attention 的 K/V 条件。训练时冻结 xSSC 和 Wan 主体，只训练 object cross-attention LoRA、slot projector、time/frame embedding 等 object branch 参数。

### Oracle 阶段

训练时和推理评估时都使用完整目标视频的真实 all-frame xSSC slots。这一阶段有未来信息泄漏，所以不能作为最终方法，只用于回答：

> 如果生成器能看到完整未来 object trajectory，object slot 条件是否能显著提升视频生成质量？

### Predictor 阶段

如果 oracle 阶段明显优于 full ctx / pooled slots，再训练 future slot predictor。输入为 ctx slots `[B,8,7,256]`，输出为未来或完整时间轴 slots `[B,T,7,256]`。训练损失包括：

- slot embedding MSE
- cosine similarity loss
- temporal smoothness loss
- 可选的 slot identity consistency loss

推理时只用 ctx slots，经 predictor 得到 future slots，再输入第一阶段训练好的 object cross-attention 分支。为了减小 oracle slots 与 predicted slots 的分布差，生成器训练阶段建议加入 slot noise / slot dropout，后期可以用 predicted slots 做一次短 finetune。

### 优点

- 改动小，能复用当前 object cross-attention LoRA 框架。
- 很适合快速验证完整时序 slot 是否有用。
- token 形式简单，容易做扰动、置零、时间打乱等消融。

### 风险

- slot token 是全局 object 表示，空间位置主要靠 xSSC slot embedding 和 time embedding 隐式表达。
- 如果 cross-attention 不会自动学到 latent patch 与 slot 的空间对应关系，控制可能仍然弱。
- oracle 到 predictor 存在分布差，predictor 误差可能导致生成质量下降。

## 方案 B：Slot-ControlNet

### 方法流程

对每个训练视频提取完整 49 帧 xSSC slots，同时从 xSSC 中获得 slot-to-patch assignment、slot attention map，或用 slot 与 patch feature 的相似度重建空间权重图。将每个 slot embedding 按空间权重广播回 latent 空间，形成 slot-conditioned feature map，例如 `[B,T,H_lat,W_lat,C_slot]`。再通过轻量 control encoder / zero-conv adapter 投影到 Wan block hidden dim，在多个 denoising blocks 中以 residual 形式注入。训练时冻结 xSSC 和 Wan 主体，只训练 ControlNet branch、zero-conv、slot-map projector，必要时训练少量 LoRA。

### Oracle 阶段

训练和评估时使用完整目标视频的真实 all-frame slot maps。该阶段同样有未来信息泄漏，只用于回答：

> 显式空间对齐的 slot map 是否比纯 cross-attention slot tokens 更有效？

### Predictor 阶段

如果 oracle Slot-ControlNet 显著优于方案 A，再训练 predictor。可选两种 predictor：

1. 预测 future slot embeddings，再结合预测/外推的 spatial maps 构造 slot feature map。
2. 直接预测 future slot feature maps `[B,T,H_lat,W_lat,C]`。

第一种参数更少、可解释性更强；第二种更直接，但更重，也更容易过拟合。初期建议先预测 slot embeddings，并使用从 ctx 得到的 motion/attention prior 外推 spatial maps。

### 优点

- 显式提供空间位置、物体区域和接触区域条件，更符合物理视频生成需求。
- 对局部物体运动、遮挡、碰撞、接触状态可能比 token cross-attention 更强。
- 更接近 ControlNet 范式，条件注入强度更直接。

### 风险

- 工程复杂度明显更高，需要可靠的 slot spatial map。
- xSSC attention map 如果空间分辨率低或不稳定，ControlNet 分支可能学习到噪声。
- 多层 residual 注入的 scale、层位置、zero-conv 初始化都需要调参。
- 如果 oracle 失败，难以判断是 slot 不好、map 不好、注入方式不好，还是训练没收敛。

## 主要假设

### Hypothesis A

如果完整时序 xSSC slots 通过 cross-attention 输入生成器，那么生成结果会优于 pooled slots，因为完整 slots 保留了 object trajectory 和时间阶段信息。

成功条件：

- 在 test_5 上，oracle all-frame cross-attention 明显优于 randomcrop/centercrop pooled slots。
- 至少接近或超过 full ctx slots 的视觉质量。
- slot 置零、时间打乱、slot shuffle 会显著影响结果，说明模型确实使用了 slot 条件。

失败条件：

- oracle all-frame slots 与 pooled slots 差不多。
- slot 扰动对结果影响很小。
- 结果只表现为背景/外观变化，物体运动没有改善。

### Hypothesis B

如果将 xSSC slots 转成时空 feature map 并用 ControlNet/adapter residual 注入，那么生成器会比纯 token cross-attention 更好地利用 object spatial condition，因为每个 latent patch 能获得局部物体条件。

成功条件：

- oracle Slot-ControlNet 明显优于 oracle all-frame cross-attention。
- 物体位置、接触区域、遮挡关系、运动方向更稳定。
- 控制分支置零或 slot map 时间打乱会显著破坏结果。

失败条件：

- 相比方案 A 没有明显提升。
- 结果出现空间伪影、过强绑定输入图像、或运动变静态。
- slot attention map 与真实物体区域不一致，导致注入噪声。

## 推荐执行顺序

### Step 1：离线缓存 all-frame xSSC slots

为训练集和 test_5 输入视频缓存 frozen xSSC slots：

- 输入：完整 49 帧视频。
- 输出：`[T=49,K=7,D=256]` slots。
- 同时保存 xSSC preprocess metadata：crop mode、resize size、frame indices。
- 输出位置应放在 `/data/gaoya/agent-data/cache/xssc_slots` 或任务专属缓存目录。

验收标准：

- 随机抽样 20 个视频，slot shape 正确。
- 同一视频重复提取结果一致。
- frame index 与原视频帧严格对应。

### Step 2：方案 A oracle all-frame cross-attention

新建训练脚本，复制当前 full ctx/object cross-attention 训练脚本，改为读取缓存的 all-frame slots，并按 latent 时间轴构造 object tokens。

建议先做两个变体：

- A1：使用 `[B,49,7,256] -> [B,343,3072]` 全帧 tokens。
- A2：对齐到 latent 时间轴 `[B,T_lat,7,256] -> [B,T_lat*7,3072]`。

优先跑 A2，因为 token 数更少，时间对齐更清晰。

验收标准：

- smoke train 能在 batch size 1/2 跑通。
- 训练 loss 正常下降。
- step-500 自动跑 test_5。
- slot zero / time shuffle 消融能改变结果。

### Step 3：方案 A predictor

只有在 A oracle 明显有效时启动。训练 predictor 从 ctx slots 预测 full/latent-time slots。

验收标准：

- predictor 在 held-out clips 上 slot cosine similarity 明显高于 naive repeat/mean baseline。
- predicted-slot 推理结果优于 pooled slots。
- predicted-slot 与 oracle-slot 之间的质量差距可接受。

### Step 4：方案 B oracle Slot-ControlNet

只有在以下任一情况成立时启动：

- 方案 A oracle 有提升，但物体空间控制仍弱。
- attention 可视化显示 cross-attention 没有稳定对齐物体区域。
- slot maps 与物体区域视觉上足够一致。

先做最小版本：

- 单层或少数层 zero-conv residual injection。
- 只注入中层 blocks，避免过早破坏低层纹理或过晚影响不足。
- control scale 从 0 初始化，逐步学习。

验收标准：

- oracle Slot-ControlNet 在物体位置/接触/运动方面优于方案 A oracle。
- 没有明显空间伪影或静态化。

### Step 5：方案 B predictor

只有在 B oracle 明显强于 A oracle 时启动。优先预测 slot embeddings，而不是直接预测 dense feature map。

验收标准：

- predicted Slot-ControlNet 优于方案 A predicted slots。
- 对空间扰动、crop 变化更稳定。

## 评估指标

### 主指标

人工视觉排序：每个 case 比较 full ctx、pooled、A oracle、A predicted、B oracle、B predicted，重点看：

- 物体运动方向是否合理。
- 接触/碰撞/遮挡是否合理。
- 目标物体是否保持身份和形状。
- 后半段是否静止或漂移。

### 辅助指标

- 生成视频与参考/输入的 temporal frame difference。
- slot perturbation sensitivity：zero slots、shuffle slots、time reverse slots。
- object cross-attention concentration：是否集中到物体相关区域。
- predicted slots vs oracle slots 的 cosine similarity、MSE、temporal smoothness。
- test_5 grouped viewer 横向对比。

## 决策规则

1. 如果 A oracle 不明显优于 pooled slots，则暂停 predictor，优先检查 slot 表征和 cross-attention 接入。
2. 如果 A oracle 明显优于 pooled slots，但 A predicted 明显退化，则重点改 predictor 和 scheduled sampling。
3. 如果 A oracle 有效但空间控制弱，再做 B oracle。
4. 如果 B oracle 不优于 A oracle，不继续做 B predictor。
5. 如果 B oracle 明显优于 A oracle，再投入 B predictor 和 ControlNet 多层注入调参。

## 最小可行实验

第一轮只做：

1. 缓存 test_5 和小训练子集的 all-frame xSSC slots。
2. 训练 A2：latent-time aligned all-frame slot cross-attention oracle。
3. 跑 step-500 test_5。
4. 做 slot zero / time shuffle / time reverse 消融。
5. 与 full ctx、randomcrop pooled、centercrop pooled 对比。

第一轮完成后再决定是否训练 predictor 或启动 Slot-ControlNet。

