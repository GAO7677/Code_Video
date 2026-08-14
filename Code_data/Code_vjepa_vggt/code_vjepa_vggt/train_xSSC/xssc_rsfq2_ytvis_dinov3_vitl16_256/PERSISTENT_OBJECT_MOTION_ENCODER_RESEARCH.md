# V-JEPA xSSC：持久物体与运动表示方案调研

更新日期：2026-08-14

## 结论先行

当前路线的核心问题不是“再多训练一些 step”，而是训练目标与期望输出不一致：模型主要通过 V-JEPA feature reconstruction 学习 slot，ARI/mBO/mIoU 只被当作验证指标；一个 512 维 slot 同时承担身份、外观、位置和运动，且时序 transition 在输入处执行 `detach()`。因此，重建 loss 可以继续下降，但没有足够约束保证同一物体跨时间保持同一 slot，也没有保证某个子空间能被解释为运动。

推荐把下一版定义为 **Persistent Motion xSSC（PM-xSSC，暂定名）**：保留冻结的 V-JEPA video encoder 和大部分现有 xSSC 聚合/解码代码，但将每个对象状态拆成“慢变化身份 + 当前外观 + 显式几何/运动 + 生命周期”，采用 SAVi 式 predictor-corrector 时序更新、clip 级固定身份匹配、运动与多步预测监督。先做一个由 GT/SAM2 mask 提供对象区域的 tracker-assisted 强基线，验证 V-JEPA feature 本身是否足以支持身份和运动；基线通过后再训练 prompt-free slot discovery，能显著降低一次性大改的风险。

## 已暂停的训练

- GPU5/6 的 MOVi-C 非因果、10-frame、clip-norm=2.0 训练已停止；最后一个完整 checkpoint 是：
  `/data/gaoya/agent-data/checkpoints/xssc_vjepa2_1_video_noncausal_movi_c_10f_transfer16000_clip2_steps50000/rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-10f-slot512-transfer16000-clip2/42/step-035000.pth`。
- 停止发生在 optimizer step 35186，35186 没有单独保存；原 checkpoint、日志和 W&B run 均未覆盖。
- 同项目残留的 GPU0 `xssc_stage1_causal_state_from25000` 也已停止。它从 step-25000 启动，停止时约为 step-25708，尚未写出新的 `step-*.pth`，所以可恢复点仍是其输入 checkpoint step-25000。
- 当前没有启动新的 xSSC GPU 训练。

## 当前实现为什么会换 slot 或塌缩

### 1. 优化目标没有直接约束身份

当前 MOVi-C 配置的训练 loss 只有 V-JEPA feature reconstruction MSE；ARI、FG-ARI、mBO、mIoU 只是验证指标。重建是集合层面的：只要所有 slots 合起来能解释 feature，交换两个 slot 的编号、让多个空 slot 变成同一个向量，loss 都可能几乎不变。

### 2. 同一个向量被要求同时“不变”和“变化”

理想身份表示应对位置、姿态和短时遮挡稳定，运动表示则必须对位移、速度、加速度敏感。让同一个 512 维 slot 同时满足两者，会产生目标冲突：slot 足够稳定时不含运动，足够响应运动时相邻帧相似度又会下降。较新的 Dual-State Slot Attention 也把“局部外观敏感”与“稳定身份不变”之间的冲突作为其核心问题，但其代码尚不适合作为本项目的复现依赖，只能作为架构启发。[DSSA](https://arxiv.org/abs/2606.12601)

### 3. 时间反传被切断

当前 transition 在接收历史 slot 时使用 `slotz.detach() + temporal_embedding`。这能降低显存，但跨时间的身份错误不能沿完整链路反传，模型更容易学成“每个时间点能重建”，而不是“同一个对象状态持续传播”。

### 4. 无效 slot 的初始化具有完全对称性

test_5 的首帧条件不足 11 个对象时，未使用 bbox slot 被同样地补零。相同输入、共享权重和缺少 valid/alive mask 会让这些 slots 产生近乎相同的 embedding。现有完整视频审计中，每个 tubelet 都出现重复 slot；大量帧内跨 slot cosine 接近 1，说明这不是单纯的可视化配色问题。

### 5. overlay 不是独立训练的对象 mask

当前可视化用 decoder attention 的 `argmax` 产生 slot label。decoder attention 主要服务 feature reconstruction，并没有 Dice/BCE 或 persistent-ID loss 直接要求它成为物体分割，所以颜色跳变只能作为异常信号，不能被等同于严格的 tracking 输出。

### 6. 训练时域与测试时域不匹配

当前 10 个原始帧经 V-JEPA tubelet size 2 后只有 5 个 encoder 时间步，而完整视频常有 15–114 个 tubelet 时间步。模型几乎没有在长遮挡、物体出入、长程漂移条件下得到监督。

### 7. 训练曲线支持“目标错位”而不是“训练不够”

step-35000 的 reconstruction validation 达到当前最好值约 0.1425，但对象指标已平台化：ARI 约 0.491、FG-ARI 约 0.597、mBO 约 0.211、mIoU 约 0.197，部分最好值反而出现在更早 checkpoint。这说明继续单独优化 reconstruction 很可能只会改善 feature 拟合，而不会自动修复身份一致性。

## 第一手工作对方案的启示

- [SAVi 官方实现](https://github.com/google-research/slot-attention-video)采用 sequential predictor-corrector：先从旧 slot 预测新状态，再用当前帧 feature 修正；上一时刻状态直接初始化下一时刻 slot。它也表明首帧 box/center cue 与 optical flow supervision 能明显帮助分组和跟踪。[SAVi 项目页](https://slot-attention-video.github.io/) / [ICLR 论文](https://openreview.net/pdf?id=aD7uesX1GF_)
- [VideoSAUR](https://github.com/martius-lab/videosaur)在 feature reconstruction 之外加入 temporal feature similarity，用运动偏置帮助 patch 在视频中保持一致分组，覆盖 MOVi 与 YouTube-VIS。[论文](https://arxiv.org/abs/2306.04829)
- [SlotFormer](https://github.com/pairlab/SlotFormer)不重新承担 object discovery，而是在已学好的 slots 上训练自回归 Transformer dynamics，适合在 slot 已经可靠后建模交互和未来状态。[论文](https://arxiv.org/abs/2210.05861)
- [SAM 2](https://ai.meta.com/research/publications/sam-2-segment-anything-in-images-and-videos/)使用 streaming memory 维持视频对象状态，适合成为“对象区域已知”强基线或现实视频的 pseudo-track 生成器；SAM 2.1 进一步针对长序列和遮挡做了改进。[项目页](https://ai.meta.com/research/sam2/) / [SA-V 数据集](https://ai.meta.com/datasets/segment-anything-video/)
- [TSA](https://arxiv.org/abs/2606.13714)给每个 slot 每帧增加 activation：inactive slot 保留历史状态且不参与当前解码，直接针对遮挡、物体进入/离开和长期 slot 生命周期。这一机制适合吸收到本项目，但不应整套照搬。
- [SlotContrast](https://arxiv.org/abs/2412.14295)表明 object-level temporal contrastive objective 可用于强化相同对象的时序一致性，适合作为 identity 子空间的辅助 loss。
- [DDLP 官方实现](https://github.com/taldatech/ddlp)把对象状态表示为带位置、大小等信息的动态 latent particles，说明显式几何变量比要求一个无结构 slot 自动“涌现”运动更容易解释和评估。
- [xSSC](https://arxiv.org/abs/2605.31508)通过静态/动态通道与跨时间重建鼓励一致性；当前实现已有 decoder dynamic ratio=0.25，但“通道被随机跨时间混合”不等于该通道在语义上就是速度，也不保证 persistent identity。

## 候选方案排序

| 优先级 | 方案 | 身份稳定 | 运动可解释 | 改造量 | 作用 |
|---|---|---:|---:|---:|---|
| 0 | GT/SAM2 masks + V-JEPA mask pooling + temporal head | 很高 | 很高 | 小 | 强基线；先验证 backbone feature 是否够用 |
| 1 | PM-xSSC：dual-state + lifecycle + motion supervision | 高 | 高 | 中 | 推荐主方案；保留现有工程主体 |
| 2 | SAVi/VideoSAUR 风格 recurrent slots + temporal similarity | 中高 | 中 | 中 | 监督更少的备选，但运动仍主要是隐变量 |
| 3 | PM-xSSC slots 上再训练 SlotFormer/C-JEPA dynamics | 依赖上游 | 高 | 中 | 第二阶段世界模型，不负责修复 slot discovery |
| 4 | DDLP 式几何粒子表示 | 高 | 很高 | 大 | 可解释运动基线，但真实视频外观能力可能较弱 |

不建议现在直接做第 3 项。若输入 slots 不稳定，后置 dynamics Transformer 只会学习并放大错误对应关系。

## 推荐主方案：PM-xSSC

### 1. 输入与时间轴

输入视频为 `video: [B, 3, T, H, W]`。V-JEPA tubelet size 为 2，因此 encoder 时间轴为 `T' = T / 2`（训练时使用偶数 T）；feature 为 `feature: [B, T', N, D_v]`，其中 `N=(H/16)*(W/16)`。

每个 tubelet `tau` 应被视作原视频区间 `[2*tau, 2*tau+1]`，而不是武断地声明只对应第二帧。建议监督定义为：

- `g_tau`：两帧对象几何中心/尺度的均值；
- `u_tau`：tubelet 内位移 `g[2*tau+1] - g[2*tau]`；
- `v_tau`：相邻 tubelet 几何中心差；若原视频 FPS 为 `F`，物理时间间隔是 `2/F` 秒。

这样运动目标严格沿 V-JEPA tubelet 时间轴计算，同时不丢掉 tubelet 内的运动。

### 2. 每个 slot 的结构化输出

保留兼容的 512 维 `slot`，但明确拆分：

```text
identity    i: [B, T', S, 256]  慢变化、用于同一对象关联
appearance  a: [B, T', S, 128]  颜色/纹理/姿态等局部状态
motion      m: [B, T', S, 128]  速度、加速度和未来变化的可学习表示
geometry    g: [B, T', S,   8]  cx, cy, logw, logh, vx, vy, ax, ay
alive       p: [B, T', S,   1]  存在/遮挡/未激活概率
mask_logits : [B, T', S, H', W']
```

`concat(i,a,m)` 仍是 512 维，可尽量复用现有 decoder；下游若要物体身份使用 `i`，若要运动使用 `concat(m,g)`，不再把整条 slot cosine 当作身份依据。

### 3. 时序更新

对每个时间步执行：

1. **Predict**：上一时刻 `identity/appearance/geometry/motion` 经 slot-interaction Transformer 预测当前位置与状态；slot 间 self-attention 表达碰撞、遮挡和相互作用。
2. **Correct**：预测位置提供 spatial prior，当前 V-JEPA patch features 通过竞争式 cross-attention 修正 slot。
3. **Lifecycle gate**：`alive` 控制当前 slot 是否吸收新观测及是否参与 mask/decoder；遮挡时保留 identity memory，真正离场后才释放。每个空 slot 使用不同 learned null query，并显式携带 valid/alive mask，彻底打破零 bbox 对称性。
4. **有限 BPTT**：去掉每一步无条件 `detach()`，先用 4–8 个 tubelet 的 truncated BPTT 控制显存。隐藏状态可在截断边界 detach，而不是每个 transition 都 detach。

主模型采用**因果**更新，保证推理时 `z_tau` 只依赖 `<=tau` 的视频。非因果模型可作为离线 teacher/smoother，但不能把它的指标与在线因果编码器混为一谈。

### 4. 身份对齐原则

MOVi-C 提供贯穿视频的 instance mask，因此训练中只在 clip 开头做一次 Hungarian matching，或对整个 clip 计算一次联合 assignment；随后同一 GT instance 始终监督同一个 slot。禁止逐帧独立 Hungarian matching，否则模型即使每帧交换 slot 也不会受罚。

真实视频阶段可以用首帧 mask/box 加 SAM2 track 产生 pseudo IDs。为最终支持纯视频输入，训练时逐步增加 condition dropout，并让 learned object queries 负责新对象 birth；但“prompted 稳定模式”和“prompt-free discovery 模式”应分别报告。

### 5. Loss 设计

总损失建议从以下可审计项组成，而不是一次启用所有项：

```text
L = L_feature_recon
  + lambda_mask   * (L_dice + L_bce)
  + lambda_id     * L_track_contrast
  + lambda_geom   * smoothL1(g_pred, g_gt)
  + lambda_motion * smoothL1(motion_readout, velocity/acceleration_gt)
  + lambda_future * sum_h L_predict(z[t+h]), h in {1,2,4}
  + lambda_alive  * BCE(alive, visible_or_occluded_state)
  + lambda_div    * L_active_slot_diversity
```

- `L_feature_recon` 保留，维持与现有 xSSC 的连接，但降为辅助目标。
- `L_track_contrast` 只作用于 identity 子空间：同一 instance 跨时间为正样本，同视频其他对象为 hard negatives；加入 variance/covariance 或 prototype 约束防止全 slot 同向塌缩。
- motion 不能只靠 `slot[t]-slot[t-1]` 事后解释；训练中应显式回归位置差、速度/加速度，并做多步未来预测。
- active-slot diversity 只约束 `alive=1` 的 slots；对无效 slots 强行互斥会制造虚假对象。

### 6. 数据课程

1. MOVi-C 10 raw frames / 5 tubelets：用于代码 smoke test，不用于得出长期跟踪结论。
2. MOVi-C 24 raw frames / 12 tubelets：训练 predictor-corrector、identity 和 motion。
3. MOVi-C 48 raw frames / 24 tubelets：加入遮挡、生命周期与长时一致性；使用随机长度和随机起点。
4. YouTube-VIS 与 SA-V/SAM2 pseudo tracks：迁移到真实外观与遮挡；保持同一个 instance 的 clip-level identity。
5. test_5：只做 held-out qualitative/diagnostic，不参与调参监督。

## 最小验证路线

### P0：不训练/轻量 probe，1–2 天

用 MOVi-C GT mask 和 test_5 的 SAM2 mask 对冻结 V-JEPA patch feature 做 mask pooling，得到 `[B,T',S,D]`；仅训练很小的 identity projection 与 motion readout。比较：

- 同一对象跨时间检索准确率、不同对象 separation；
- 速度/位移 linear-probe R² 与 MAE；
- 遮挡前后 identity retrieval；
- 直接 V-JEPA pooled feature、当前 xSSC slot、结构化 probe 三者。

如果这个基线都无法稳定区分对象，问题在 V-JEPA feature/区域信息或数据，而不是 Slot Attention；此时不应先投入主模型训练。

### P1：最小身份修复，约 2k steps

在现有模型上只加入 unique null queries、valid/alive mask、专用 mask head、clip-level persistent matching 和 identity contrastive loss。先用 10/24 帧小规模验证 slot duplication 和 ID switch 是否显著下降。

### P2：显式运动，约 3–5k steps

加入 dual-state、geometry/motion heads、1/2/4-step future prediction，训练 24 帧，BPTT=4 tubelets。只在 P1 通过后运行。

### P3：长视频与真实视频

训练 24→48 帧 curriculum，加入 lifecycle 和 condition dropout，再迁移 YouTube-VIS/SA-V。最后才考虑在冻结 slots 上训练 SlotFormer/C-JEPA 式 dynamics。

## 评价指标与停机门槛

不能再只看 reconstruction、ARI 和 mBO。每次 validation 至少同时记录：

- **分组**：FG-ARI、mBO、mIoU；
- **身份**：IDF1、HOTA/AssA、ID switches / 100 frames、遮挡恢复准确率；
- **塌缩**：active slots 数、帧内跨 slot cosine、duplicate-slot rate、slot utilization entropy；
- **运动**：中心/尺度 velocity MAE、acceleration MAE、1/2/4-step ADE/FDE、motion linear-probe R²；
- **长期稳定性**：5、12、24、50 tubelet 时域下的指标退化曲线；
- **模式拆分**：prompted 与 prompt-free 分开报告。

建议的首个 go/no-go 门槛是：相较 step-35000，active duplicate-slot rate 降低至少 80%，ID switches / 100 frames 降低至少 50%，且 FG-ARI/mBO 不下降；运动 head 必须显著优于“最后位置不动”基线，并与 constant-velocity baseline 同表比较。未达到门槛就不进入长视频大规模训练。

## checkpoint 复用建议

- `step-035000.pth` 可复用 frozen V-JEPA、encoder projection 和部分 decoder 权重。
- 不建议原样继承已经高度相似的 null-slot 状态；unique queries、alive head、identity/motion heads 应重新初始化。
- transition 可以部分加载，但必须移除逐步 `detach()` 并重新验证梯度/显存。
- 新方案使用独立 save-dir 和 W&B project/run，不覆盖现有 checkpoints。

## 两句话项目表述

现有 xSSC 用单一重建驱动 slot 同时表示身份、外观和运动，导致跨帧换槽、无效 slot 塌缩以及运动不可解释。PM-xSSC 在冻结 V-JEPA 特征上引入带生命周期的 predictor-corrector object slots，把持久 identity 与时变 appearance/geometry/motion 解耦，并通过 clip 级身份对齐和多步动力学监督产生可跟踪、可读出的对象时序表示。

## 推荐的下一项动作

先实施 P0 tracker-assisted probe，而不是立刻恢复 50k-step 训练。P0 是最便宜且信息量最大的决策实验：它可以直接区分“V-JEPA feature 不适合对象身份/运动”与“当前 slot 学习目标设计错误”这两种根因。
