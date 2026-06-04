# 2026-06-04 Mainline Flow: `wan_state_v2_latent_time` + Wan Clean-Prefix Infill

这份文档只描述当前推荐主线的真实代码流程，不回顾旧版 `baseline_v1 / latent_v1 / latent_v2` 的历史设计。主线由 4 段组成：

1. `latent-time predictor` 训练
2. `state_condition bundle` 导出
3. `Wan I2V clean-prefix state adapter` 训练
4. `formal Wan inference` 正式推理

核心思想是：

- predictor 不再在原始视频帧时间轴 `K / T` 上预测，而是在 Wan VAE 压缩后的 latent 时间轴 `L_ctx / L_future` 上工作
- Wan 正式推理里，context video 会被编码成 `clean_prefix_latents`，并且在每个 diffusion step 前后都覆盖回主 latent 序列
- 真正被去噪更新的只有 future latent 部分
- predictor 输出的 `future_state_latents` 会作为 `state_tokens` 送进训练好的 Wan state adapter，再变成 Wan DiT 可以消费的 `state_context`

## 1. 记号和时间轴

为了避免把原视频时间轴和 Wan latent 时间轴混淆，当前主线里固定使用下面两套记号：

- `K`: context video 的原始帧数
- `T`: future video 的原始帧数
- `L_ctx`: context video 编码到 Wan latent 后的时间步数
- `L_future`: future 部分在 Wan latent 时间轴上的步数
- `L`: 总 Wan latent 时间步数，满足 `L = L_ctx + L_future`
- `C_w`: Wan latent channel 数
- `H_w, W_w`: predictor 使用的 Wan latent 空间尺寸
- `H'_w, W'_w`: 正式 Wan 推理时按目标分辨率编码后的 latent 空间尺寸
- `D_s`: predictor 输出的 state latent 维度，默认 `128`
- `N`: 单个场景中的最大物体数
- `C_cam`: camera 特征维度

代码中 latent 时间步通过 `temporal_stride` 计算，定义在 [wan_state_v2_helpers.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/wan_state_v2_helpers.py:9)：

- `L_ctx = 1 + floor((K - 1) / temporal_stride)`
- `L_total = 1 + floor((K + T - 1) / temporal_stride)`
- `L_future = L_total - L_ctx`

## 2. 整体总流程

单个 episode 的基础输入来自 `.npz`：

- `context_frames ∈ R^{K×3×H×W}`
- `future_frames ∈ R^{T×3×H×W}`
- `context_states ∈ R^{K×N×10}`
- `future_states ∈ R^{T×N×10}`
- `camera ∈ R^{K×C_cam}`
- `prompt ∈ string`

一个 batch 之后，训练脚本里通常写成：

- `context_frames ∈ R^{B×K×3×H×W}`
- `future_states ∈ R^{B×T×N×10}`
- `camera ∈ R^{B×K×C_cam}`

主线的四段代码链路如下：

1. [train_predictor_wan_state_v2.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_predictor_wan_state_v2.py:1)
   把 `context_frames` 编码成 Wan latent-time 输入，训练 predictor 输出 `future_state_latents`
2. [export_wan_state_condition_dataset.py](/home/gaoya/Code_Video/phys_state_video/scripts/export_wan_state_condition_dataset.py:1)
   把 predictor 输出写成 Wan 可消费的 `state_tokens` bundle
3. [train_wan_state_adapter_prefix_local.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_wan_state_adapter_prefix_local.py:1)
   在正式 `I2V clean-prefix` 语义下训练 Wan state adapter
4. [run_inference_wan_state.py](/home/gaoya/Code_Video/phys_state_video/scripts/run_inference_wan_state.py:1)
   调 predictor 预测 `future_state_latents`，再调 Wan backend 做 clean-prefix future infill

## 3. Stage A: Predictor 训练

### 3.1 输入如何进入 predictor

入口在 [train_predictor_wan_state_v2.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_predictor_wan_state_v2.py:1) 的 `run_epoch()`。

训练时每个 batch 会先取：

- `context_frames ∈ R^{B×K×3×H×W}`
- `camera ∈ R^{B×K×C_cam}`
- `context_states ∈ R^{B×K×N×10}`
- `future_states ∈ R^{B×T×N×10}`

然后通过 latent extractor 把 `context_frames` 压到 Wan latent 时间轴：

- `context_latents = latent_extractor.encode_context_frames_raw(context_frames)`
- shape: `context_latents ∈ R^{B×L_ctx×C_w×H_w×W_w}`

如果是 `MockLatentExtractor`，逻辑在 [wan_state_v2_helpers.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/wan_state_v2_helpers.py:30)：

- 先在时间维把 `K` 压到 `L_ctx`
- 再在空间维把 `H×W` 压到 `H_w×W_w`
- 最终得到模拟的 `Wan latent`

如果是 `WanLatentExtractor`，逻辑在 [wan_bridge.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/wan_bridge.py:224)：

- 用真实 Wan VAE 编码整段 context clip
- 输出原生 Wan latent
- 每个样本从 `C_w×L_ctx×H_w×W_w` 转成 `L_ctx×C_w×H_w×W_w`
- batch 后得到 `R^{B×L_ctx×C_w×H_w×W_w}`

与此同时：

- `camera` 会被重采样到 latent 时间轴
- `camera_latent = resample_camera_to_latent_steps(camera, L_ctx)`
- `camera_latent ∈ R^{B×L_ctx×C_cam}`

监督目标也会被重采样到 latent 时间轴：

- `context_target = resample_temporal_states(context_states, L_ctx)`
- `context_target ∈ R^{B×L_ctx×N×10}`
- `future_target = resample_temporal_states(future_states, L_future)`
- `future_target ∈ R^{B×L_future×N×10}`

这里有个关键点：

- `context_states / future_states` 只是监督目标
- 它们不进入 predictor 主干前向
- predictor 真正看的输入只有 `context_latents + camera_latent + prompt`

### 3.2 Predictor 内部模块和 shape

主干定义在 [predictor_wan_state_v2.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/predictor_wan_state_v2.py:23)。

#### 模块 1: `PromptEncoder`

输入：

- `prompt_token_ids ∈ R^{B×L_prompt}`
- `prompt_token_mask ∈ R^{B×L_prompt}`

输出：

- `prompt_embed ∈ R^{B×D_prompt}`
- 经过 `prompt_proj` 后为 `R^{B×D_h}`

含义：

- 把文本 prompt 压成一个全局语义向量
- 后续会沿时间维 broadcast 到所有 context latent step

#### 模块 2: `_latent_features()`

输入：

- `context_latents ∈ R^{B×L_ctx×C_w×H_w×W_w}`

输出：

- `latent_features ∈ R^{B×L_ctx×D_lat}`

其中：

- `D_lat = C_w × (latent_pool_side^2 + 2)`
- 默认 `latent_pool_side = 2`
- 所以默认 `D_lat = C_w × 6`

构成方式：

1. 对每个 latent step 做 `adaptive_avg_pool2d(..., 2×2)`
2. 展平成 `C_w×4`
3. 再拼接每个 channel 的 `mean` 和 `std`
4. 得到 `4C_w + C_w + C_w = 6C_w`

它表达的是：

- pooled latent 保留粗空间布局
- `mean/std` 提供全局外观和能量统计

#### 模块 3: `context_input_proj`

输入拼接：

- `latent_features ∈ R^{B×L_ctx×D_lat}`
- `camera_latent ∈ R^{B×L_ctx×C_cam}`
- `prompt_context ∈ R^{B×L_ctx×D_h}`

拼接后：

- `encoder_input_raw ∈ R^{B×L_ctx×(D_lat + C_cam + D_h)}`

经过 MLP 投影：

- `encoder_input ∈ R^{B×L_ctx×D_h}`

再加位置编码：

- `encoder_input + context_pos_embed[:, :L_ctx]`

#### 模块 4: `context_encoder`

输入：

- `encoder_input ∈ R^{B×L_ctx×D_h}`

输出：

- `context_hidden ∈ R^{B×L_ctx×D_h}`

作用：

- 在 latent 时间轴上建模 context 的时序关系
- 融合视觉、camera、prompt 三类信息

#### 模块 5: `context_state_latent_proj`

输入：

- `context_hidden ∈ R^{B×L_ctx×D_h}`

输出：

- `context_state_latents ∈ R^{B×L_ctx×D_s}`

含义：

- 这是“上下文时刻”的 state latent
- 更多是辅助监督和共享表示，不直接进 Wan

#### 模块 6: `future_decoder`

构造：

- `future_queries ∈ R^{1×L_future×D_h}`
- `future_pos_embed ∈ R^{1×L_future×D_h}`
- `global_context = mean(context_hidden, dim=1, keepdim=True) ∈ R^{B×1×D_h}`

解码输入：

- `future_queries + future_pos_embed + global_context`
- shape: `R^{B×L_future×D_h}`

memory：

- `context_hidden ∈ R^{B×L_ctx×D_h}`

输出：

- `future_hidden ∈ R^{B×L_future×D_h}`
- 经过 `future_state_latent_proj` 后：
- `future_state_latents ∈ R^{B×L_future×D_s}`

这就是当前主线最关键的 predictor 输出。

#### 模块 7: `GroupedStateHeads`

输入：

- `context_state_latents ∈ R^{B×L_ctx×D_s}`
- `future_state_latents ∈ R^{B×L_future×D_s}`

输出：

- `context_geom_predictions ∈ R^{B×L_ctx×N×4}`
- `context_motion_predictions ∈ R^{B×L_ctx×N×3}`
- `context_vis_predictions ∈ R^{B×L_ctx×N×3}`
- `context_state_predictions ∈ R^{B×L_ctx×N×10}`
- `future_geom_predictions ∈ R^{B×L_future×N×4}`
- `future_motion_predictions ∈ R^{B×L_future×N×3}`
- `future_vis_predictions ∈ R^{B×L_future×N×3}`
- `future_state_predictions ∈ R^{B×L_future×N×10}`

三组 head 的语义：

- `geom = [center_x, center_y, depth, log_scale]`
- `motion = [vel_x, vel_y, depth_vel]`
- `vis = [visibility, existence, confidence]`

实现细节：

- `vis` 分支在前向里先输出 `vis_logits`
- 训练损失使用 `BCEWithLogits`
- 对外暴露的 `vis_predictions` 则已经过 sigmoid

### 3.3 Predictor 损失

损失定义在 [predictor_wan_state_v2.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/predictor_wan_state_v2.py:288)。

分成三块：

1. `context_loss`
   对 `context_state_predictions` 和 `context_target` 做监督
2. `future_loss`
   对 `future_state_predictions` 和 `future_target` 做监督
3. `latent_smooth`
   对 `future_state_latents[:, 1:] - future_state_latents[:, :-1]` 做平滑约束

其中：

- `geom` 和 `motion` 使用 MSE
- `vis` 使用 BCE-with-logits
- 总损失里 `vis` 权重是 `0.5`

### 3.4 三阶段训练和冻结关系

训练日程在 [train_predictor_wan_state_v2.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_predictor_wan_state_v2.py:115)。

#### `context_only`

目标：

- 先把显式 head 稳定下来
- 让 `context_state_latents` 学会解释当前物理状态

参数状态：

- 整个 predictor 可训练
- `state_heads` 明确解冻

损失：

- 只优化 `context_loss`

#### `future_only`

目标：

- 重点学习 future rollout 的 latent dynamics

参数状态：

- `state_heads` 冻结
- 其它模块继续训练

损失：

- `future_loss + latent_smooth`
- 如果启用 adapter 对齐，还会加 `adapter_align_scale * adapter_align`

#### `joint_finetune`

目标：

- 联合微调 context 表达、future latent 和显式 head

参数状态：

- `state_heads` 解冻
- 整个 predictor 联合优化

损失：

- `context_loss + future_loss + latent_smooth`
- 可选再加 `adapter_align`

### 3.5 Adapter-space alignment 是怎么接进来的

逻辑在 [train_predictor_wan_state_v2.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_predictor_wan_state_v2.py:186)。

可选加载两样冻结模块：

- 一个冻结 teacher predictor
- 一个冻结的 Wan state adapter encoder

流程是：

1. 当前 predictor 输出 `future_state_latents`
2. teacher predictor 也输出一份 `teacher_future_state_latents`
3. 两者分别送入同一个冻结 adapter encoder
4. 得到：
   - `predicted_state_context`
   - `teacher_state_context`
5. 增加：
   - `adapter_align = mean((predicted_state_context - teacher_state_context)^2)`

这个 loss 的意义不是直接监督视频，而是把 predictor latent 空间往“adapter 真能消费的空间”上拉。

## 4. Stage B: `state_condition` bundle 导出

入口在 [export_wan_state_condition_dataset.py](/home/gaoya/Code_Video/phys_state_video/scripts/export_wan_state_condition_dataset.py:1)。

如果用当前主线 predictor 导出，每个样本会生成：

- `input_image.png`
- `state_condition.npz`
- `meta.json`
- `prompt.txt`

其中 predictor 模式下的关键内容是：

- `state_tokens ∈ R^{L_future×D_s}`
- `predicted_states ∈ R^{L_future×N×10}`
- `context_state_predictions ∈ R^{L_ctx×N×10}`

这一步的职责很明确：

- 它不做 Wan 推理
- 只是在 predictor 和 Wan adapter 之间建立稳定的数据接口

`meta.json` 还会记录：

- `context_latent_steps`
- `future_latent_steps`
- `temporal_stride`
- `predictor_version`

所以 adapter 训练时不需要重新猜时间轴语义。

## 5. Stage C: Wan I2V Clean-Prefix State Adapter 训练

推荐入口是 [train_wan_state_adapter_prefix_local.py](/home/gaoya/Code_Video/phys_state_video/scripts/train_wan_state_adapter_prefix_local.py:1)。

### 5.1 训练样本怎么构造

每个 bundle 会回溯到原始 episode，然后拼出完整训练视频：

- `training_video = concat(context_frames, future_frames)`
- shape: `training_video ∈ R^{F×3×H×W}`，其中 `F = K + T`

再通过 `align_wan_frame_num()` 补到 `4n + 1`：

- `training_video_aligned ∈ R^{F'×3×H×W}`

这一步的目的：

- 保证视频长度符合 Wan 的时间压缩约束

### 5.2 训练时进入 Wan 的核心张量

在 `run_step()` 里：

1. `full_video`
   - 输入：`R^{F'×3×H×W}`
   - resize 后编码成：
   - `full_latents ∈ R^{C_w×L×H_w×W_w}`

2. `context_frames`
   - 输入：`R^{K×3×H×W}`
   - resize 后单独编码成：
   - `clean_prefix_latents ∈ R^{C_w×L_ctx×H_w×W_w}`

3. `state_tokens`
   - 从 bundle 读取
   - 原始 shape: `R^{L_future_src×D_s}` 或 `R^{1×L_future_src×D_s}`
   - 重采样后：
   - `state_tokens_resampled ∈ R^{1×L_future×D_s}`

其中：

- `prefix_len = L_ctx`
- `future_latent_steps = L - L_ctx`

### 5.3 Clean-prefix 训练语义

当前主线最重要的一点就是这里。

训练时并不是把整个 latent 都当成普通 diffusion sample，而是：

1. 对 `full_latents` 采样噪声：
   - `noise ∈ R^{C_w×L×H_w×W_w}`
2. 采样 timestep：
   - `timestep ∈ R^{1}`
3. 得到加噪结果：
   - `noised_latents ∈ R^{C_w×L×H_w×W_w}`
4. 用 `clean_prefix_latents` 覆盖前缀：
   - `noised_latents[:, :L_ctx] = clean_prefix_latents`

因此训练时主模型看到的是：

- 前 `L_ctx` 步是干净 context latent
- 后 `L_future` 步是带噪 future latent

loss 也只在 future 部分计算：

- `loss = mse(noise_pred[:, L_ctx:], training_target[:, L_ctx:])`

这和正式推理的语义是一致的。

### 5.4 `y` 和 `state_context`

当前 `WanI2V` 主干仍需要一个 `y` 输入，因此脚本保留了一份首帧兼容条件：

- `i2v_video ∈ R^{3×F'×H_out×W_out}`
- 只有第 1 帧是真图，其余帧为 0
- 编码得到 `i2v_latent`
- 再配合 `i2v_mask`
- 拼成 `y`

但要明确：

- 这份 `y` 现在只是为了兼容 Wan I2V 接口
- 真正的 context video 条件不是靠 `y`
- 而是靠 `clean_prefix_latents` 直接覆盖主 latent 前缀

另外一条真正的状态条件支路是：

- `state_tokens_resampled ∈ R^{1×L_future×D_s}`
- `pipeline._build_state_context({"state_tokens": state_tokens_resampled})`
- 得到 `state_context`

这个 `state_context` 会通过 Wan 的 state adapter 注入 DiT。

### 5.5 哪些参数在训练

可训练参数选择在 [wan_adapter_training.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/wan_adapter_training.py:291)。

当前 `I2V clean-prefix` 路线里：

- `text_encoder` 冻结
- `vae` 冻结
- `low_noise_model` 主体冻结
- `high_noise_model` 主体冻结
- `pipeline.state_adapter` 全部可训练
- `low_noise_model` 中名字带 `state_adapter_` 的参数可训练
- `high_noise_model` 中名字带 `state_adapter_` 的参数可训练

也就是说，这一阶段不是训练整个 Wan，而是只训练 state adapter 分支和它在 DiT 中对应的注入权重。

## 6. Stage D: 正式 Wan 推理

入口在 [run_inference_wan_state.py](/home/gaoya/Code_Video/phys_state_video/scripts/run_inference_wan_state.py:1)。

### 6.1 Predictor 部分

当 checkpoint 是 `wan_state_v2_latent_time` 时：

1. `context_frames ∈ R^{1×K×3×H×W}`
2. `context_latents = encode_context_frames_raw(context_frames)`
3. `context_latents ∈ R^{1×L_ctx×C_w×H_w×W_w}`
4. `camera_latent = resample_camera_to_latent_steps(camera, L_ctx)`
5. `camera_latent ∈ R^{1×L_ctx×C_cam}`
6. predictor 输出：
   - `future_state_latents ∈ R^{1×L_future×D_s}`
   - `future_state_predictions ∈ R^{1×L_future×N×10}`

然后：

- `state_tokens = future_state_latents[0] ∈ R^{L_future×D_s}`

### 6.2 Wan backend 部分

正式逻辑在 [wan_bridge.py](/home/gaoya/Code_Video/phys_state_video/src/phys_state_video/wan_bridge.py:278) 的 `WanImageToVideoBackend.generate()`。

输入：

- `prompt ∈ string`
- `context_frames ∈ R^{K×3×H×W}`
- `state_tokens ∈ R^{L_future_src×D_s}` 或 `R^{1×L_future_src×D_s}`

先做几件事：

1. 按目标分辨率 resize context
2. 编码得到：
   - `clean_prefix_latents ∈ R^{C_w×L_ctx×H'_w×W'_w}`
3. 构造总噪声：
   - `noise ∈ R^{C_w×L×H'_w×W'_w}`
4. 初始化主 latent：
   - `latent = apply_clean_prefix(noise, clean_prefix_latents)`
5. 计算：
   - `prefix_len = L_ctx`
   - `future_latent_steps = L - L_ctx`
6. 把 predictor 提供的 `state_tokens` 重采样到当前 Wan 需要的 future 步数：
   - `condition_tokens ∈ R^{1×L_future×D_s}`

### 6.3 推理时 prefix 是否真的固定

是的，当前正式链路里 prefix latent 是真的固定的，不只是“有一个 mask”。

具体实现是两次覆盖：

1. 初始化时：
   - `latent = _apply_clean_prefix_to_latent(noise, clean_prefix_latents)`
2. 每个 diffusion step 更新后：
   - `latent = _apply_clean_prefix_to_latent(latent, clean_prefix_latents)`

所以当前这条正式 `Wan I2V clean-prefix continuation` 路线的真实语义是：

- `0:L_ctx` 对应的 context latent 保持干净不变
- 只有 `L_ctx:L` 对应的 future latent 会被采样更新

### 6.4 State adapter 在推理时如何注入

如果没有提供训练好的 `state_adapter_ckpt`，当前代码会直接报错，不再继续“空跑”。

有了 checkpoint 之后：

1. `pipeline.load_state_adapter(...)`
2. `state_context = pipeline._build_state_context({"state_tokens": condition_tokens})`
3. 在每个 timestep 调 Wan model 时，把 `state_context` 作为额外条件传入

### 6.5 当前 state-aware CFG

推理里不是简单的 cond/uncond 两支，而是三支：

1. `noise_pred_uncond`
   - 无文本、无状态
2. `noise_pred_text_only`
   - 有文本、无状态
3. `noise_pred_text_state`
   - 有文本、有状态

最终组合：

- `noise_pred = noise_pred_uncond`
- `+ guide_scale * (noise_pred_text_only - noise_pred_uncond)`
- `+ state_guidance_scale * (noise_pred_text_state - noise_pred_text_only)`

优点是：

- 文本引导和状态引导被拆开了
- `state_guidance_scale` 可以单独控制 state condition 的影响强度

## 7. 各模块输入输出速查表

### `WanLatentExtractor.encode_context_frames_raw`

- 输入：`context_frames ∈ R^{B×K×3×H×W}`
- 输出：`context_latents ∈ R^{B×L_ctx×C_w×H_w×W_w}`
- 训练性：冻结，不训练

### `WanStateLatentPredictorV2`

- 输入：`context_latents ∈ R^{B×L_ctx×C_w×H_w×W_w}`
- 输入：`camera_latent ∈ R^{B×L_ctx×C_cam}`
- 输入：`prompt_token_ids / prompt_token_mask`
- 输出：`context_state_latents ∈ R^{B×L_ctx×D_s}`
- 输出：`future_state_latents ∈ R^{B×L_future×D_s}`
- 输出：`context_state_predictions ∈ R^{B×L_ctx×N×10}`
- 输出：`future_state_predictions ∈ R^{B×L_future×N×10}`
- 训练性：可训练

### `export_wan_state_condition_dataset.py`

- 输入：predictor 输出
- 输出：`state_tokens ∈ R^{L_future×D_s}`
- 输出：`predicted_states ∈ R^{L_future×N×10}`
- 训练性：不训练，只导出

### `train_wan_state_adapter_prefix_local.py`

- 输入：`full_latents ∈ R^{C_w×L×H_w×W_w}`
- 输入：`clean_prefix_latents ∈ R^{C_w×L_ctx×H_w×W_w}`
- 输入：`state_tokens ∈ R^{1×L_future×D_s}`
- 输出：训练好的 `state_adapter_ckpt`
- 训练性：只训练 state adapter 相关参数

### `WanImageToVideoBackend.generate`

- 输入：`context_frames ∈ R^{K×3×H×W}`
- 输入：`state_tokens ∈ R^{L_future×D_s}`
- 中间：`clean_prefix_latents ∈ R^{C_w×L_ctx×H'_w×W'_w}`
- 中间：`noise / latent ∈ R^{C_w×L×H'_w×W'_w}`
- 输出：`video ∈ R^{3×F×H_out×W_out}`
- 训练性：推理时不训练

## 8. 设计上最值得记住的 6 个点

1. predictor 主时间轴已经从 `K/T` 切到 `L_ctx/L_future`，这是当前主线最关键的变化。
2. predictor 主输出是 `future_state_latents ∈ R^{B×L_future×D_s}`，显式 `N×10` 状态只是监督头。
3. 正式 Wan 推理时，context video 不是作为 noisy latent 一起采样，而是作为 clean latent prefix 被持续覆盖。
4. 正式去噪只发生在 future latent 区间。
5. state 条件不是直接拼到主 latent 上，而是经 `state_adapter -> state_context` 以 adapter/cross-attention 形式进入 Wan DiT。
6. 当前仓库已经把 predictor 训练、bundle 导出、prefix adapter 训练、正式推理 4 段主线代码闭环补齐了。
