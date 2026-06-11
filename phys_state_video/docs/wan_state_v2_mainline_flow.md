# 2026-06-04 Mainline Flow: `wan_state_v2_latent_time` + Wan State-Condition Backbones

这份文档描述当前仓库里已经落地的 v2 主线实现：`wan_state_v2_latent_time predictor -> state_condition bundle -> Wan state adapter -> Wan video backbone inference`。当前推荐主线已经明确拆成两条下游 backbone：

- `Wan I2V clean-prefix infill`
- `Wan TI2V clean-prefix infill`

同时保留一条 legacy 对照线：

- `Wan TI2V first-frame conditioning`

这里不再沿用旧版 `future_state_latents / prompt_token_ids` 的叙述，统一以当前代码中的真实接口、真实 shape、真实脚本为准。

## 1. 一句话流程

输入一个 episode 后，先把 `context_frames` 通过冻结的 Wan VAE 编码到 Wan latent 时间轴，再用 `WanStateLatentPredictorV2` 在 latent 时间轴上预测未来的 `future_state_maps`；随后把它转成 Wan 侧消费的 canonical 条件表示 `condition_maps`，并可选附带 `memory_tokens`；下游 Wan backbone 不再把 `state_tokens` 当成主条件输入，而是统一先把 `condition_maps + memory_tokens` 经过训练好的 `state adapter` 编成 `state_context`，再注入对应的 Wan DiT。

## 2. 记号与时间轴

- `K`: context 原始帧数
- `T`: future 原始帧数
- `L_ctx`: context 在 Wan VAE latent 时间轴上的步数
- `L_future`: future 在 Wan VAE latent 时间轴上的步数
- `L = L_ctx + L_future`: 总 latent 时间步
- `C_w`: Wan latent channel 数
- `H_w, W_w`: predictor 输入 latent 的空间尺寸
- `H'_w, W'_w`: Wan 正式推理时目标分辨率对应的 latent 空间尺寸
- `D_s`: predictor 内部 state latent 维度，默认 `128`
- `H_s, W_s`: predictor 输出 state map 的空间尺寸，默认 `2, 2`
- `N`: object slot / 监督 object 数上限
- `C_cam`: camera 特征维度
- `L_prompt`: 冻结 Wan T5 encoder 输出的 token 数
- `D_prompt`: 冻结 Wan T5 token hidden dim

latent 时间步由 [wan_state_v2_helpers.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/wan_state_v2_helpers.py:11) 中的逻辑决定：

- `L_ctx = 1 + floor((K - 1) / temporal_stride)`
- `L_total = 1 + floor((K + T - 1) / temporal_stride)`
- `L_future = L_total - L_ctx`

## 3. 主线分段

当前主线代码链路如下：

1. [train_predictor_wan_state_v2.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_predictor_wan_state_v2.py:1)
   训练 latent-time predictor。当前 v2 主线固定只使用 Wan VAE latents。
2. [export_wan_state_condition_dataset.py](/home/gaoya/Code_Video/phys_state_video/scripts/export_wan_state_condition_dataset.py:1)
   导出 `condition_maps` 主条件、`memory_tokens` 与状态预测。
3. [train_wan_state_adapter_prefix_local.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_wan_state_adapter_prefix_local.py:1)
   在 I2V clean-prefix 语义下训练 Wan state adapter。
4. [run_inference_wan_state.py](/home/gaoya/Code_Video/phys_state_video/scripts/run_inference_wan_state.py:1)
   用 predictor + Wan I2V backend 做正式推理。
5. [train_wan_state_adapter_ti2v_prefix_local.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_wan_state_adapter_ti2v_prefix_local.py:1)
   在 TI2V clean-prefix 语义下训练 Wan state adapter。
6. [run_inference_wan_state_ti2v_prefix.py](/home/gaoya/Code_Video/phys_state_video/scripts/run_inference_wan_state_ti2v_prefix.py:1)
   用 predictor + Wan TI2V clean-prefix backend 做正式推理。
7. [train_wan_state_adapter_local.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_wan_state_adapter_local.py:1)
   legacy：在 TI2V 首帧条件语义下训练 Wan state adapter。
8. [run_inference_wan_state_ti2v.py](/home/gaoya/Code_Video/phys_state_video/scripts/run_inference_wan_state_ti2v.py:1)
   legacy：用 predictor + Wan TI2V first-frame backend 做正式推理。

## 4. Predictor 输入、输出与 shape

### 4.1 训练/推理输入

单个 batch 的原始输入通常是：

- `context_frames ∈ R^{B×K×3×H×W}`
- `camera ∈ R^{B×K×C_cam}`
- `context_states ∈ R^{B×K×N×10}`
- `future_states ∈ R^{B×T×N×10}`
- `prompts: list[str]`

其中真正进入 predictor 主干的只有三类输入：

- `context_latents ∈ R^{B×L_ctx×C_w×H_w×W_w}`
- `camera_latent ∈ R^{B×L_ctx×C_cam}`
- `prompt_context ∈ R^{B×L_prompt×D_prompt}`
- `prompt_mask ∈ R^{B×L_prompt}`

`context_states / future_states` 不进入 predictor 主干，只作为监督目标重采样到 latent 时间轴：

- `context_target ∈ R^{B×L_ctx×N×10}`
- `future_target ∈ R^{B×L_future×N×10}`

### 4.2 prompt 路径

当前 v2 主线不再使用仓库内部的 bag-of-hash `PromptEncoder`。训练和推理都改成：

1. 文本 prompt 先通过冻结的 Wan tokenizer + T5 encoder。
2. 得到 token-level 上下文：
   `prompt_context ∈ R^{B×L_prompt×D_prompt}`。
3. 同时生成 padding mask：
   `prompt_mask ∈ R^{B×L_prompt}`。

实现位于 [wan_state_v2_helpers.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/wan_state_v2_helpers.py:178) 的 `WanPromptContextEncoder`。

这比旧的 bag-of-words prompt 分支更合理，因为：

- 保留了 token-level 语义结构，而不是只保留一个均值 embedding。
- 与 Wan 本体使用的文本编码器一致，减少 predictor 和 Wan 之间的文本域差。

### 4.3 Predictor 内部模块

主干定义在 [predictor_wan_state_v2.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/predictor_wan_state_v2.py:27)。

#### 模块 A: `prompt_token_proj + prompt_summary_proj`

输入：

- `prompt_context ∈ R^{B×L_prompt×D_prompt}`
- `prompt_mask ∈ R^{B×L_prompt}`

输出：

- `prompt_tokens ∈ R^{B×L_prompt×D_s}`
- `prompt_summary ∈ R^{B×D_s}`

作用：

- `prompt_tokens` 保留 token-level 条件，供 cross-attention 使用。
- `prompt_summary` 提供一个全局文本语义向量，直接加到 context state map 上。

#### 模块 B: `visual_stem`

输入：

- `context_latents ∈ R^{B×L_ctx×C_w×H_w×W_w}`

输出：

- `visual_maps ∈ R^{B×L_ctx×H_s×W_s×D_s}`

过程：

1. 每个 latent step 先经过 2 层 `Conv2d`。
2. 再做 `adaptive_avg_pool2d` 到 `H_s×W_s`。
3. 最终排列成 `R^{B×L_ctx×H_s×W_s×D_s}`。

#### 模块 C: `camera_proj`

输入：

- `camera_latent ∈ R^{B×L_ctx×C_cam}`

输出：

- `camera_embed ∈ R^{B×L_ctx×1×1×D_s}`

作用是把每个时间步的 camera 条件加到对应的 state map 上。

#### 模块 D: `context_encoder`

输入前，先把以下量相加：

- `visual_maps ∈ R^{B×L_ctx×H_s×W_s×D_s}`
- `camera_embed ∈ R^{B×L_ctx×1×1×D_s}`
- `prompt_summary ∈ R^{B×1×1×1×D_s}`
- `context_time_pos_embed ∈ R^{1×L_ctx×1×1×D_s}`
- `spatial_pos_embed ∈ R^{1×1×H_s×W_s×D_s}`

相加后当前会先经过一层 `LayerNorm`，再作为 encoder 输入。得到：

- `state_maps ∈ R^{B×L_ctx×H_s×W_s×D_s}`

然后 flatten 成完整时空 token 序列：

- `context_tokens ∈ R^{B×(L_ctx·H_s·W_s)×D_s}`

再送入 `TransformerEncoder`，输出仍是：

- `context_tokens ∈ R^{B×(L_ctx·H_s·W_s)×D_s}`

最后 reshape 回：

- `context_state_maps ∈ R^{B×L_ctx×H_s×W_s×D_s}`

这是这次重构里最关键的设计修正之一。旧实现把每个 spatial cell 独立看成一条时间序列，缺乏跨空间通信；现在 encoder 直接在完整时空 token 序列上做 self-attention，空间与时间都能交互。同时，融合后的 `LayerNorm` 也让 `visual / camera / prompt / pos` 几路特征的初始尺度更稳定。

#### 模块 E: `context_prompt_cross_attn`

输入：

- query: `context_tokens ∈ R^{B×(L_ctx·H_s·W_s)×D_s}`
- key/value: `prompt_tokens ∈ R^{B×L_prompt×D_s}`

输出：

- `attended_prompt ∈ R^{B×(L_ctx·H_s·W_s)×D_s}`

作用：

- 让 context 视频表示与 token-level 文本语义直接交互，而不只是加一个全局文本均值。

#### 模块 F: `future_decoder`

输入 memory：

- `context_memory ∈ R^{B×(L_ctx·H_s·W_s)×D_s}`
- 再与 `prompt_tokens ∈ R^{B×L_prompt×D_s}` 拼接，形成 decoder memory。

query 来自：

- `future_time_queries ∈ R^{1×L_future×1×1×D_s}`
- `spatial_pos_embed ∈ R^{1×1×H_s×W_s×D_s}`

展开后：

- `future_queries ∈ R^{B×(L_future·H_s·W_s)×D_s}`

输出：

- `future_state_maps ∈ R^{B×L_future×H_s×W_s×D_s}`

注意，这里主输出已经不是旧文档里写的 `future_state_latents ∈ R^{B×L_future×D_s}`，而是一个 5D tensor。

#### 模块 G: `SpatialObjectQueryDecoder`

输入：

- `context_state_maps ∈ R^{B×L_ctx×H_s×W_s×D_s}`
- `future_state_maps ∈ R^{B×L_future×H_s×W_s×D_s}`

输出：

- `debug_context_object_slots ∈ R^{B×L_ctx×N×D_s}`
- `debug_future_object_slots ∈ R^{B×L_future×N×D_s}`

作用：

- 用 object query 对每个时间步的空间 state map 做 cross-attention，提取 object slots。
- 这些 slots 只在 predictor 内部用于显式状态监督和 memory token 构建，不直接送入 Wan。

#### 模块 H: `MemoryTokenHead`

输入：

- `debug_context_object_slots ∈ R^{B×L_ctx×N×D_s}`

输出：

- `memory_tokens ∈ R^{B×N×D_s}`

当前实现不再只取最后一步，而是对每个 object 的时间序列做 attention pooling。这样相比“只看最后一步”，能保留速度、周期、加速度等多步历史信息。

#### 模块 I: `StateTokenHead`

输入：

- `future_state_maps ∈ R^{B×L_future×H_s×W_s×D_s}`

输出：

- `debug_projected_future_state_maps ∈ R^{B×L_future×H_s×W_s×D_s}`

随后在前向中会导出两个视图：

- `condition_maps ∈ R^{B×L_future×D_s×H_s×W_s}`
- `state_tokens ∈ R^{B×(L_future·H_s·W_s)×D_s}`

这里必须强调：

- `condition_maps` 是当前 v2 主线的 canonical 条件表示。
- `state_tokens` 只是由 `condition_maps` flatten 得到的兼容视图，不再是主线训练/推理接口。
- 因此旧文档里“`state_tokens ∈ R^{L_future×D_s}`”是错误的；真实 shape 是 `R^{B×(L_future·H_s·W_s)×D_s}`。

#### 模块 J: `GroupedStateHeads`

输入：

- `debug_context_object_slots ∈ R^{B×L_ctx×N×D_s}`
- `debug_future_object_slots ∈ R^{B×L_future×N×D_s}`

输出：

- `context_state_predictions ∈ R^{B×L_ctx×N×10}`
- `future_state_predictions ∈ R^{B×L_future×N×10}`

三个分组头分别预测：

- `geom`: `center_x, center_y, depth, log_scale`
- `motion`: `vel_x, vel_y, depth_vel`
- `vis`: `visibility, existence, confidence`

其中 `vis` 经过 sigmoid，显式改成有界概率输出。

### 4.4 Predictor 返回值

当前 v2 前向的核心输出是：

- `context_state_maps ∈ R^{B×L_ctx×H_s×W_s×D_s}`
- `future_state_maps ∈ R^{B×L_future×H_s×W_s×D_s}`
- `condition_maps ∈ R^{B×L_future×D_s×H_s×W_s}`
- `state_tokens ∈ R^{B×(L_future·H_s·W_s)×D_s}`
- `memory_tokens ∈ R^{B×N×D_s}`
- `context_state_predictions ∈ R^{B×L_ctx×N×10}`
- `future_state_predictions ∈ R^{B×L_future×N×10}`

调试输出是：

- `debug_context_object_slots ∈ R^{B×L_ctx×N×D_s}`
- `debug_future_object_slots ∈ R^{B×L_future×N×D_s}`
- `debug_projected_future_state_maps ∈ R^{B×L_future×H_s×W_s×D_s}`
- `debug_prompt_tokens ∈ R^{B×L_prompt×D_s}`

## 5. Predictor 训练目标

loss 位于 [predictor_wan_state_v2.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/predictor_wan_state_v2.py:465)。

分为三部分：

1. `context_state_predictions` 对 `context_target` 的监督。
2. `future_state_predictions` 对 `future_target` 的监督。
3. `future_state_maps` 时间平滑正则：
   对 `future_state_maps[:, 1:] - future_state_maps[:, :-1]` 做平方约束。

训练阶段：

- `context_only`
- `future_only`
- `joint_finetune`

阶段切换逻辑在训练脚本中控制冻结/解冻。当前实现仍会在阶段切换时重建 optimizer，这一点在工程上不是最理想，因为会丢掉动量状态，但在“可训练参数集合变化”时仍属常见做法。如果后续要进一步优化，可以把参数组做成稳定 superset，而不是每阶段重建 optimizer。

## 6. state_condition bundle 导出

导出脚本位于 [export_wan_state_condition_dataset.py](/home/gaoya/Code_Video/phys_state_video/scripts/export_wan_state_condition_dataset.py:1)。

当前推荐导出内容为：

- `condition_maps ∈ R^{L_future×D_s×H_s×W_s}`
- `memory_tokens ∈ R^{N×D_s}`
- `predicted_states ∈ R^{L_future×N×10}`

以及 metadata：

- `future_latent_steps`
- `future_state_map_spatial_shape`
- `temporal_stride`

这里的设计原则是：

- adapter/video-side 以 `condition_maps` 为主。
- `state_tokens` 仍然可以从 `condition_maps` 派生，但不再作为推荐导出主条件。

## 7. Wan I2V clean-prefix adapter 训练

训练脚本位于 [train_wan_state_adapter_prefix_local.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_wan_state_adapter_prefix_local.py:1)。

单个样本的关键张量是：

- `full_video ∈ R^{F×3×H×W}`，其中 `F = align_wan_frame_num(K + T)`
- `full_latents ∈ R^{C_w×L×H'_w×W'_w}`
- `clean_prefix_latents ∈ R^{C_w×L_ctx×H'_w×W'_w}`
- `condition_maps ∈ R^{1×L_future×D_s×H_s×W_s}`，必要时按 `L_future` 做时间重采样
- `memory_tokens ∈ R^{1×N×D_s}`，如果 bundle 中存在

训练时现在统一使用共享 helper：

- `build_state_condition_payload_from_condition_maps(...)`

构造 canonical payload，再交给：

- `filter_state_condition_payload_for_adapter(...)`

适配 Wan adapter 实际启用的分支。

clean-prefix 训练语义是：

1. 对完整视频编码得到 `full_latents`。
2. 单独对 context 编码得到 `clean_prefix_latents`。
3. 只对 future latent 部分加噪和回归。
4. 每个 step 前后都用 `clean_prefix_latents` 覆盖前缀。

这样与正式推理语义一致，不再是“首帧 clean”的简化版本。

## 8. Wan I2V 正式推理

推理入口在 [run_inference_wan_state.py](/home/gaoya/Code_Video/phys_state_video/scripts/run_inference_wan_state.py:1)，Wan backend 主逻辑在 [wan_bridge.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/wan_bridge.py:1)。

正式推理时：

1. predictor 先输出 `condition_maps` 与 `memory_tokens`。
2. `context_frames` 被编码成 `clean_prefix_latents ∈ R^{C_w×L_ctx×H'_w×W'_w}`。
3. 构造总 latent 噪声 `noise ∈ R^{C_w×L×H'_w×W'_w}`。
4. 在每个 diffusion step 前后都覆盖 `:L_ctx` 前缀。
5. future 段使用 state adapter 编成的 `state_context` 做条件注入。

当前 v2 语义下，`WanImageToVideoBackend.generate()` 已经把 `condition_maps` 设为必选输入，不再允许主线只传 `state_tokens` 的危险 fallback。

## 9. Wan TI2V clean-prefix adapter 训练

训练脚本位于 [train_wan_state_adapter_ti2v_prefix_local.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_wan_state_adapter_ti2v_prefix_local.py:1)。

这条 backbone 的语义是 TI2V clean-prefix continuation：

1. 用完整训练视频 `context_frames + future_frames` 构造 TI2V clean-prefix 监督视频。
2. 对完整训练视频编码得到 `full_latents`。
3. 用 context 对应的 latent prefix 作为 clean anchor，只对非-context latent 步做噪声回归。
4. TI2V backbone 仍保留首帧图像条件 `y`，但多帧 context 通过 clean prefix latent 直接固定在主 latent 序列里。
5. 将 predictor 导出的 `condition_maps + memory_tokens` 重采样到 padded `future_latent_steps`，再构造成 canonical payload 后送入 state adapter。
6. loss 只在非-context latent steps 上计算，并显式 mask 掉 `4n+1` 对齐带来的无效 padded future latent steps。

单个样本的关键张量是：

- `full_video ∈ R^{F×3×H×W}`，其中 `F = align_wan_frame_num(K + T)`
- `full_latents ∈ R^{C_w×L×H'_w×W'_w}`
- `clean_prefix_latents ∈ R^{C_w×L_ctx×H'_w×W'_w}`
- `condition_maps ∈ R^{1×L_future,padded×D_s×H_s×W_s}`，由 bundle 条件重采样到 TI2V future latent 时间轴
- `memory_tokens ∈ R^{1×N×D_s}`，如果 bundle 中存在

与 I2V 侧一样，TI2V 训练也统一通过：

- `build_state_condition_payload_from_condition_maps(...)`
- `filter_state_condition_payload_for_adapter(...)`

构造并筛选真正送入 adapter 的 payload。

## 10. Wan TI2V clean-prefix 正式推理

推理入口在 [run_inference_wan_state_ti2v_prefix.py](/home/gaoya/Code_Video/phys_state_video/scripts/run_inference_wan_state_ti2v_prefix.py:1)，Wan backend 主逻辑在 [wan_bridge.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/wan_bridge.py:692) 的 `WanTextImageToVideoPrefixBackend`。

正式推理时：

1. predictor 仍先读取完整 `context_frames`，输出 `condition_maps` 与 `memory_tokens`。
2. TI2V backbone 接收完整 `context_frames`，并把它们编码成 `clean_prefix_latents ∈ R^{C_w×L_ctx×H'_w×W'_w}`。
3. 采样从总 latent 噪声 `noise ∈ R^{C_w×L×H'_w×W'_w}` 开始，但在每个 diffusion step 后都把 `:L_ctx` 覆盖回 clean prefix。
4. TI2V 主干仍保留首帧图像条件 `y` 作为兼容锚点。
5. `condition_maps` 被重采样到 `future_latent_steps = L - L_ctx`。
6. `condition_maps + memory_tokens` 经 state adapter 编成 `state_context`，注入 `WanTI2V`。
7. 默认总输出帧数是 `context_steps + future_steps`，因为 clean-prefix 版本显式保留了整段 context video。

这里 predictor 和 TI2V backbone 的时间语义需要区分：

- predictor 仍然基于 `K` 帧 context video 建模。
- TI2V clean-prefix backbone 既看到首帧图像锚点，也看到完整 context latent prefix。
- 二者共享的是 future state condition，并且现在也共享多帧 context 的时间边界。

## 10.1 legacy：Wan TI2V first-frame 训练与推理

legacy 训练脚本位于 [train_wan_state_adapter_local.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_wan_state_adapter_local.py:1)，legacy 推理脚本位于 [run_inference_wan_state_ti2v.py](/home/gaoya/Code_Video/phys_state_video/scripts/run_inference_wan_state_ti2v.py:1)。

这条 legacy 线的语义是：

1. 用 episode 的首帧作为 TI2V 图像条件。
2. 用 `first_frame + future_frames` 构造监督视频。
3. 用首帧 latent 作为 clean anchor，只对后续 latent 步做噪声回归。
4. predictor 仍然可以读取完整 `context_frames`，但 TI2V backbone 只直接消费首帧图像。

## 11. 可训练模块与冻结模块

### 11.1 predictor 训练阶段可训练模块

当前 v2 predictor 中可训练的主要模块是：

- `prompt_token_proj`
- `prompt_summary_proj`
- `camera_proj`
- `visual_stem`
- `context_encoder`
- `context_prompt_cross_attn`
- `future_decoder`
- `object_query_decoder`
- `state_token_head`
- `memory_token_head`
- `state_heads`
- 各类位置参数与 query 参数

### 11.2 predictor 训练阶段冻结模块

- `WanPromptContextEncoder` 内的 Wan tokenizer / T5 encoder
- `WanLatentExtractor` 内的 Wan VAE

也就是说，prompt 与 visual latent 都来自冻结的 Wan 模块，predictor 只学习“如何把这些条件映射成 future state map / state condition”。

### 11.3 adapter 训练阶段可训练模块

- `pipeline.state_adapter`
- Wan DiT 内部已经显式暴露出来的 `state_adapter_*` 注入权重

对 I2V 是 `low_noise_model / high_noise_model` 中的注入权重；
对 TI2V 是 `model` 中的注入权重。

### 11.4 adapter 训练阶段冻结模块

- Wan text encoder
- Wan VAE
- Wan 主干里除 `state_adapter_*` 外的参数

## 12. 这轮 review 后确认的问题与修正

这轮对照代码后，以下问题确实存在且已经按更合理的方向修正：

1. `state_tokens` 的 shape 旧文档写错了。
   现在统一以 `R^{B×(L_future·H_s·W_s)×D_s}` 为准。
2. `future_state_latents` 旧叙述与代码不一致。
   现在统一为 `future_state_maps ∈ R^{B×L_future×H_s×W_s×D_s}`，Wan 侧 canonical 输出为 `condition_maps ∈ R^{B×L_future×D_s×H_s×W_s}`。
3. prompt 路径旧文档不准确。
   当前主线已经改成冻结 Wan T5 的 token-level 上下文。
4. context encoder 缺乏跨空间通信。
   当前 encoder 已改为完整时空 token 序列建模。
5. `MemoryTokenHead` 只取最后一步。
   当前已改为时间 attention pooling。
6. `state_tokens`-only fallback 危险。
   当前 v2 I2V/TI2V 主线都已经把 `condition_maps` 设为 canonical 输入。
7. Wan bridge / adapter training 里的重复 helper。
   当前已经集中到共享 helper 模块中复用。
8. `latent_source` 混乱。
   当前 v2 主线固定只使用 Wan VAE latents，不再推荐 `MockLatentExtractor` 作为训练/部署分布。
9. I2V 与 TI2V 原先没有统一的 state-condition backbone 适配层。
   当前已经分别收敛到 `WanImageToVideoBackend` 与 `WanTextImageToVideoBackend` 两个独立 backend，但共享相同的 payload helper。

仍然保留但需要后续再看的一点是：

- `MockLatentExtractor` 与真实 Wan VAE 仍存在域差；它适合 CPU 单测和 smoke，但不应被当作当前 v2 主线的训练/部署分布。

## 13. 当前主线最简结论

当前仓库里更准确的一句话不是“predictor 预测 `future_state_latents` 再送给 Wan”，而是：

`context video + frozen Wan T5 prompt tokens -> predictor 预测 future_state_maps -> 导出 condition_maps/memory_tokens -> Wan 将其编码成 state_context，再交给 I2V clean-prefix 或 TI2V first-frame backbone 生成 future video。`
