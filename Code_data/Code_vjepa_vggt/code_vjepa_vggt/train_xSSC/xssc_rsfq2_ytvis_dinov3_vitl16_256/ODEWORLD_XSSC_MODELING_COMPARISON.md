# ODEWorld 与当前 V-JEPA xSSC 的建模方式对照

## 分析范围

- ODEWorld：`third_party/ODEWorld`，固定在 commit
  `a15276659a21bd079ee4f5ebeca2512937b89228`。
- xSSC：本项目当前运行的非因果 V-JEPA2.1 ViT-L/16 + RandSFQ2，
  10 个原始视频帧、tubelet size 2、5 个 xSSC 时间步；先在 YTVIS-HQ
  训练，再从 step-16000 迁移到 MOVi-C。
- 本文主要依据实际代码。ODEWorld 仓库目前公开了模型、demo 和
  dataloader，但没有完整训练入口，因此优化器、训练轮数等不从代码外推。

## 一句话结论

ODEWorld 是一个**目标条件、连续时间、整体场景级 latent 动力学模型**；
xSSC 是一个**视频条件、离散 tubelet 时间、对象 slot 分解与时序一致性模型**。
ODEWorld 预测“状态如何沿物理时间流动”，xSSC 学习“哪些 patch 属于同一对象，
以及对象表征如何在观测帧间延续”。二者互补，但当前实现不能直接互换。

## 端到端建模流程

### ODEWorld / PT-Flow

以公开 LIBERO 配置为例，图像先统一到 256×256，随后在
`DinoPatchLatentEncoder` 内再次 bicubic resize 到 224×224；冻结的
DINOv2-with-registers ViT-B/14 去掉 CLS 与 4 个 register token，得到：

```text
obs_s0, obs_sg, obs_st_chunk: RGB images
        │
        ▼ frozen DINOv2-B/14
s0, sg, st: [B, 256 patches, 768]
        │
        ├─ delta_decouple(s0, st): learned delta queries cross-attend to [s0; st]
        ▼
z0, zτ, zg: [B, K_delta, 768]，公开 checkpoint 中 K_delta=1
        │
        ▼ FiLM MLP vθ([z0,zτ,zg], τ)
predicted latent velocity: [B, 1, 768]
        │
        ▼ torchdiffeq.odeint (RK4)
z(τ1...τN): [B, N, 1, 768]
        │
        ▼ delta_decode(s0, zτ)
reconstructed DINO patch latents: [B, N, 256, 768]
        │
        ▼ separately trained RAE decoder
RGB rollout: [B, N, 3, 224, 224]（概念 shape）
```

其中局部物理时间由数据集直接定义：

```text
τ = (target_idx - current_idx) / max_time_length
```

训练时用以目标时刻为中心的奇数帧窗口，在 DINO patch latent 上通过一阶
FIR 估计 `ds/dt`；再用 JVP 把它推送到 delta latent，得到
`dz/dτ` 的监督。速度模型学习：

```text
vθ(z0, zτ, zg, τ) ≈ dz/dτ
```

公开实现的 PT-Flow loss 是：

```text
L = rec_weight · MSE(delta_decode(s0, zτ), st)
  + dyn_enc_weight · MSE(vθ(z0,zτ,zg,τ), JVP(delta_decouple, ds/dt))
```

推理时可用真实目标图像 `sg`，也可先用起始图像和语言预测目标 latent；
ODE solver 的积分网格决定输出时间分辨率，因此可在训练采样点之间查询，
也具备反向积分的结构基础。

### 当前 V-JEPA xSSC

输入为连续 10 帧。冻结 V-JEPA2.1 ViT-L/16 使用 tubelet size 2，把每两帧
编码成一个时空 token slice：

```text
video: [B, 10, 3, H, W]
        │
        ▼ frozen V-JEPA2.1, tubelet=2, patch=16
feature: [B, 5, 1024, H/16, W/16]
        │ flatten spatial grid + MLP projection
encode: [B, 5, (H/16·W/16), 1024]
        │
        ├─ t=0: 7-slot learned Gaussian initializer (YTVIS)
        │        or 11 bbox-conditioned slot queries (MOVi-C)
        ├─ t>0: RSFQTransit(previous slots, observed features up to current t)
        ▼
Slot Attention with inverted attention competition
slotz: [B, 5, S, 512]
attenta: [B, 5, S, H/16, W/16]
        │
        ▼ Markov random autoregressive decoder / CTR
recon: [B, 5, 1024, H/16, W/16]
attentd: [B, 5, S, H/16, W/16]
```

当前非因果版本先对完整 10 帧做一次 V-JEPA 编码，所以每个 tubelet token
可能已包含整段 clip 的时序上下文；随后 RandSFQ2 按 5 个 tubelet index
递推 slot。GT segmentation 选择原始帧 `[1,3,5,7,9]`，只用于评估及
MOVi-C bbox 条件构造，不进入 reconstruction loss。

xSSC 的关键不是额外 slot 对比损失，而是：

1. `RSFQTransit` 用历史 slot 和截至当前 tubelet 的观测特征产生下一步 query；
2. Chrono-Channel Decomposition 把 slot channel 分为 static/dynamic 两部分；
3. Cross-Temporal Reconstruction 在训练时随机选择当前或较早时间索引，
   交换/组合相应 channel 后重建冻结 backbone feature；
4. 唯一训练目标仍是 detached V-JEPA feature reconstruction MSE。

```text
L_xSSC = MSE(recon, stop_gradient(selected V-JEPA feature))
```

ARI、ARI-FG、mBO 和 mIoU 是由 decoder attention 得到的分割评估指标，
不是训练 loss。

## 逐项异同

| 维度 | ODEWorld | 当前 V-JEPA xSSC |
|---|---|---|
| 主要任务 | 连续时间 latent rollout、视频生成、机器人规划 | 视频对象发现、对象 slot 表征与跨帧一致性 |
| 基本状态单位 | 整体场景变化的 `K_delta` 个 delta token；公开配置 `K=1` | `S` 个对象候选 slot；YTVIS 7、MOVi-C 11 |
| 空间表征 | DINOv2 patch grid 被 delta token 全局汇聚，再解回 patch grid | patch grid 通过 inverted Slot Attention 竞争分配给对象 slots |
| 视觉编码器 | 冻结的逐帧 DINOv2-B/14，224² → 16×16×768 | 冻结的 V-JEPA2.1 ViT-L/16 视频编码器，10 帧 → 5×空间网格×1024 |
| 时间变量 | 显式连续标量 `τ`，由真实 frame-index gap 归一化 | 离散 tubelet index 0…4 与 learned integer time embedding |
| 动力学 | 一阶 ODE `dz/dτ=vθ(z0,zτ,z_goal,τ)` | Transformer transition 产生下一 tubelet 的 slot queries |
| 条件 | 起始状态 + 目标图像 latent，或起始状态 + 语言目标 | 整段观测视频；首步可无条件或由 GT/AMG bbox 初始化 |
| 监督 | delta reconstruction + latent velocity matching | frozen feature reconstruction；无显式 SSC/速度 loss |
| 对象归纳偏置 | 没有显式对象分解；delta token 可能混合多个物体与背景 | slot competition、固定 slot budget、bbox/learned queries |
| 输出 | 任意积分网格上的 latent trajectory，可经 RAE 解码 RGB | 每个观测 tubelet 的 slots、attention masks、feature reconstruction |
| 插值/外推 | ODE solver 原生支持改变积分步数、查询连续时间 | 时间 embedding 和训练 clip 长度绑定，不能直接宣称连续插值 |
| 因果性 | rollout 从 `z0` 积分，但通常目标条件已包含未来目标 | 当前 backbone 为 noncausal；另有 prefix-causal 实现 |
| 时间一致性 | 由单个连续速度场与 ODE trajectory 约束 | 由 slot recurrence + CCD + CTR 隐式内化 |
| 物理含义 | “physical time”是 frame gap 归一化后的连续坐标；代码未显式守恒或接触方程 | 没有显式物理方程；dynamic channel 是结构性表征假设 |

## 真正相似的部分

1. **都在冻结视觉表征空间中建模，而非直接预测像素。** 这降低了动力学和
   对象发现对低层纹理的负担。
2. **都把静态参考和动态变化分开。** ODEWorld 用 `s0` 与 delta token；xSSC
   用 static/dynamic channel decomposition。
3. **都用少量 latent token 作为瓶颈并重建 patch feature。** ODEWorld 的
   delta token 与 xSSC slot 在形式上都是 learned queries + cross-attention，
   但语义不相同。
4. **两者都不依赖显式像素级运动标注。** ODEWorld 从邻帧 latent 估计速度，
   xSSC 从重建结构中获得对象一致性。

## 最容易混淆但本质不同的部分

### “delta token”不等于“object slot”

ODEWorld 默认只有一个 delta token，它表示从起始场景到当前/目标场景的
全局变化；代码没有 slot-wise competition，也没有保证一个 token 对应一个
物体。xSSC 的多个 slot 对空间 patch 进行竞争分配，目标就是产生对象级分组。

### 两个系统中的“时间”不是同一个量

ODEWorld 的 `τ` 是数据 frame gap 除以 `max_time_length` 后的连续标量，进入
FiLM velocity field，并由 ODE solver 积分。xSSC 的时间首先由 V-JEPA tubelet
定义：2 个 RGB 帧成为 1 个 encoder step；之后 transition 和 CTR 使用整数
tubelet 距离。增加 ODE solver step 不会自动增加 xSSC 的观测时间分辨率。

### ODEWorld 的 rollout 与 xSSC 的 forward 范围不同

ODEWorld 从起点和目标生成未观测的中间/未来 latent。当前 xSSC forward
编码整段已给定视频并解释其中对象；它并没有独立从初始 slot 自由 rollout
未来对象状态的公开接口。

## 对本项目最有价值的结合方式

最合理的研究路线不是用 ODEWorld 替换 xSSC，而是把 PT-Flow 放到 xSSC 的
对象状态空间：

```text
video → V-JEPA → xSSC slots Z(t)=[B,T,S,512]
                    │
                    ▼ slot-wise / interaction-aware velocity field
       dZ/dτ = fθ(Z0, Zτ, Zgoal, τ, object interactions)
                    │ ODE integration
                    ▼
             future object slots → xSSC decoder / downstream policy
```

推荐按风险从低到高分三步验证：

1. **离线 probe：** 冻结现有 xSSC，使用 MOVi-C GT identity 对 slot 做
   Hungarian 对齐，在对齐后的 slot trajectory 上训练小型 velocity MLP；先验证
   ODE rollout 是否优于离散 MLP/Transformer baseline。
2. **slot-wise ODE：** 每个对象 slot 共享 velocity 网络，同时通过 slot-set
   attention 建模物体交互；对 permutation 使用 matching-invariant loss。
3. **联合训练：** 保留 xSSC feature reconstruction，增加 ODE velocity matching
   与 multi-step rollout consistency；最后才考虑端到端更新 slot encoder。

最低限度的公平基线必须包括：

- 相同 xSSC slots 上的 one-step MLP；
- 相同参数量的离散 Transformer transition；
- ODE solver 不同步数下的误差/耗时曲线；
- 对象 matching 前后、单物体/多物体、遮挡与碰撞分组结果；
- ADE/FDE、slot ARI/mBO、feature reconstruction、长时 rollout drift。

## 当前代码层面的限制与风险

- ODEWorld 公开仓库没有训练 launcher/config，仅凭当前代码不能一键复现论文
  完整训练。
- `DINOv2PTFlow.py` 的默认构造参数 `max_time_length=200`，但公开 LIBERO 与
  AgiBot checkpoint config 均为 50；分析或复现时必须以 checkpoint config 为准。
- 公开 PT-Flow checkpoint 的 `disable_encoder_detach=true`；尽管 DINO encoder
  本身冻结，这会让 velocity loss 通过 JVP 路径更新 delta encoder。不能简单描述
  为“所有 target 全部 stop-gradient”。
- ODEWorld demo 接受起始图与目标图；语言目标仅在 LIBERO 提供独立 goal
  predictor。AgiBot demo 不具有同一条语言目标链路。
- 当前 MOVi-C xSSC 的首步依赖 bbox；对无 GT 外部视频使用 SAM2-AMG pseudo
  boxes，因此其对象分解质量会同时受到初始化器与分割器影响。
- 当前 xSSC 为 noncausal V-JEPA backbone。若用于在线 planning，应先切换并验证
  prefix-causal encoder，否则观测 slot 可能包含未来 clip 信息。

## 代码索引

### ODEWorld

- 总览与 checkpoint 下载：`third_party/ODEWorld/README.md`
- 图像 latent：`third_party/ODEWorld/models/DINOv2Latent.py`
- PT-Flow、velocity target 与 ODE rollout：
  `third_party/ODEWorld/models/DINOv2PTFlow.py`
- RGB reconstruction decoder：`third_party/ODEWorld/models/DINOv2RAE.py`
- 语言目标预测：`third_party/ODEWorld/models/DINOv2GoalPred.py`
- 训练数据的物理时间采样：`third_party/ODEWorld/dataloader/ImgVelEngine.py`
- 推理调用与 PCA velocity field：`third_party/ODEWorld/demo_infer.py`

### 当前 V-JEPA xSSC

- V-JEPA tubelet adapter：
  `upstream/object_centric_bench/model/vjepa2_1_video_backbone.py`
- tubelet-to-slot forward：
  `upstream/object_centric_bench/model/randsfq2_vjepa_video.py`
- slot transition：`upstream/object_centric_bench/model/randsfq.py`
- Slot Attention：`upstream/object_centric_bench/model/ocl.py`
- CCD/CTR decoder：`upstream/object_centric_bench/model/randsfq2.py`
- 当前 MOVi-C 10-frame config：
  `upstream/config-randsfq/rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-10f-slot512-transfer16000-clip2.py`

