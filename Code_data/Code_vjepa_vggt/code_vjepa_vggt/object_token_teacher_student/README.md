# Object Token Teacher-Student

这个子目录用于放一条新的 teacher-student 实验分支，目标是最大化复用当前
`code_vjepa_vggt` 里的现有 object branch、Wan 注入链路、dataset、监督函数和训练 runner，
避免把 `train_v_newtrain.py` 或默认分支再复制一遍。

## 默认方案

默认只考虑三阶段主线，其他备选分支放在文末。

### Stage 1A: Full-Token Teacher

- 输入视频：
  - full video: `[B, 3, T, H, W]`
- object slots:
  - `O = 4`
  - `Q = 8`
- 目标：
  - 用 full video 构造 `oracle object latent tokens`
  - 直接验证这个 token 空间本身能否回归 full-time object supervision
  - 暂时不训练 Wan object cross-attn

teacher token 语义保持和当前 object branch 一致：

- `oracle_context_tokens`: `[B, T_ctx_lat, O, 4096]`
- `oracle_full_tokens`: `[B, T_full_lat, O, 4096]`
- 其中通常：
  - `T_ctx_lat = 2`
  - `T_full_lat = 6`

这条路径复用：

- `PhysStateEpisodeDataset`
- `JEPAPatchAdapter`
- `CoTrackerAdapter`
- `VGGTTrackAdapter` 或 `VGGT cache`
- `ObjectTubeProjector`
- `ObjectAuxHeads`
- `ContextVideoTrainer`

### Stage 1B: Oracle Injection

- 输入：
  - `oracle_full_tokens: [B, T_full_lat, O, 4096]`
- 路径：
  - `oracle_full_tokens -> ObjectConditionAdapter -> object_context -> Wan object_cross_attn`
- 目标：
  - 训练 Wan 的 object injection 分支
  - 验证 Wan 是否会使用 full-time object token

Stage 1B 复用：

- `ObjectConditionAdapter`
- `WanContextVideoModel`
- `ContextVideoTrainer`
- `OracleObjectTokenEncoder`

### Stage 1C: Joint Alignment

- 输入：
  - `oracle_full_tokens -> ObjectConditionAdapter -> Wan object_cross_attn`
- 目标：
  - 在 Stage 1B 稳定后，小范围重新开放 token tail / aux heads
  - 让 token teacher 与 Wan 的消费方式重新对齐

Stage 1C 当前第一版默认会继续冻结大部分 backbone，只打开：

- `ObjectConditionAdapter`
- Wan `object_embedding / object_cross_attn / object_gate / norm4`
- `ObjectTubeProjector` 的少量 tail 模块
- `ObjectAuxHeads`

`Wan LoRA` 仍默认关闭，需要显式在配置里打开。

### Stage 2: Future Predictor

- 输入：
  - `context object latent tokens`: `[B, 2, 4, 4096]`
- 预测：
  - `future object latent tokens`: `[B, 4, 4, 4096]`

默认第一版只做三类监督：

- token distillation
- future track supervision
- future box supervision

当前第一版不强制 future depth supervision，原因是：

- predictor 先需要学会 time rollout 和 object identity 对齐
- token + track + box 已足够形成第一版几何监督闭环
- future depth 容易把新的 target pipeline 再拉复杂

### Stage 3: Bridge Finetune

- 把 predictor 预测出来的 future tokens 接回 Wan object branch
- teacher / predictor token 之间做 mix
- teacher forcing ratio 后续可退火

第一版脚本里会保留 Stage 3 的接口和占位说明，但重点先实现 Stage 1 / Stage 2。

## 代码组织

- `common.py`
  - 通用 shape / mask / slot / future 切片 helper
- `oracle_encoder.py`
  - full-video oracle token builder
- `predictor.py`
  - context-to-future token predictor
- `future_heads.py`
  - future track / future box heads
- `losses.py`
  - token / track / box loss
- `runtime_stage1a_full_token.py`
  - Stage 1A full-token teacher trainer
- `runtime_stage1_common.py`
  - Stage 1 shared oracle token-building helper
- `runtime.py`
  - Stage 2 predictor trainer
- `runtime_stage1.py`
  - Stage 1B oracle injection trainer
- `runtime_stage1c_joint.py`
  - Stage 1C joint finetune trainer
- `train_stage1a_full_token.py`
  - Stage 1A 入口
- `train_stage1_oracle_injection.py`
  - Stage 1B 入口
- `train_stage1b_oracle_cross_attn.py`
  - Stage 1B 别名入口
- `train_stage1c_joint.py`
  - Stage 1C 入口
- `train_stage2_predictor.py`
  - Stage 2 入口
- `train_stage3_bridge.py`
  - 当前暂时复用 Stage 1C joint finetune trainer

## 当前实现边界

这版优先保证：

- 新分支代码和默认分支隔离
- 尽量 import 现有模块
- Stage 1A 会真实用 `full video -> oracle_object_latent_tokens`
  去训练 token teacher
- Stage 1B 会真实用 `full video -> oracle_object_context`
  替换默认 `context object_context`
- Stage 2 可以独立读取 `context_* / future_*` supervision
- 训练入口沿用现有 config / runner 风格

### 当前 Stage 1A 实现说明

当前 `train_stage1a_full_token.py` 会走：

- `FullTokenTeacherTrainer`
- 直接用 `full video + full GT boxes`
  通过 `OracleObjectTokenEncoder` 重建 full-time object token 链
- 不跑 Wan DiT 去噪
- 只训练：
  - `ObjectTubeProjector`
  - `ObjectAuxHeads`

当前 Stage 1A 的 loss 是：

- `loss_full_track`
- `loss_full_box`
- `loss_full_depth`

当前 Stage 1A 显式冻结：

- `ObjectConditionAdapter`
- `JEPAPatchAdapter`
- `CoTrackerAdapter`
- `VGGTTrackAdapter`
- Wan VAE / text encoder / DiT / LoRA

当前 Stage 1A 默认 backbone 固定为：

- Wan base:
  - `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- frozen LoRA:
  - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`

### 当前 Stage 1B 实现说明

当前 `train_stage1_oracle_injection.py` 已经不再只是空壳入口。

它现在会走：

- `OracleInjectionTrainer`
- 内部先复用 `ContextVideoTrainer._prepare_batch()` 产出默认 batch
- 再调用 `oracle_encoder.forward_from_batch(..., use_full_video_as_context=True)`
- 用 full-video oracle 结果替换：
  - `prepared["object_context"]`
  - `prepared["object_latent_tokens"]`

因此送进 Wan DiT 的 object condition 已经是：

- `oracle_object_context`: `[B, T_full_lat * O, 4096]`

而不是默认 context-only 的：

- `context_object_context`: `[B, T_ctx_lat * O, 4096]`

当前 Stage 1B 可训练模块被显式限制为 Wan object injection 分支加 adapter：

- `ObjectConditionAdapter`
- `object_embedding`
- `object_cross_attn`
- `object_gate`
- `norm4`

其余 teacher token builder 路径都冻结：

- `JEPAPatchAdapter`
- `CoTrackerAdapter`
- `VGGTTrackAdapter`
- `ObjectTubeProjector`
- `ObjectAuxHeads`
- Wan VAE / text encoder / 非 object DiT / LoRA

当前 Stage 1B 默认 backbone 固定为：

- Wan base:
  - `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- frozen LoRA:
  - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`

### 当前 Stage 1C 实现说明

当前 `train_stage1c_joint.py` 和 `train_stage3_bridge.py` 都会走：

- `Stage1CJointTrainer`

它是在 Stage 1B 的 oracle injection runtime 之上，继续打开一小部分 token-side 模块：

- `ObjectConditionAdapter`
- `ObjectAuxHeads`
- `ObjectTubeProjector` 的 tail 模块：
  - `modal_refine`
  - `out_norm`
  - `jepa_router_score`
  - `latent_router_score`
  - `track_geometry_router_score`
  - `appearance_router_score`

可选地，也可以通过配置把 Wan LoRA 再打开，但默认仍关闭。

当前 Stage 1C 默认 backbone 也固定为：

- Wan base:
  - `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- frozen LoRA:
  - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`

这版暂不做：

- 直接重写 Wan 完整训练循环
- 新建另一套 object token 语义
- 强制 future depth supervision

## 附加说明：备选分支

- 可以后续给 Stage 2 增加 future depth supervision：
  - 优先用 `future_states[..., depth_index]`
  - 再考虑更重的 future depth cache / online depth
- 可以后续给 Stage 3 增加 teacher forcing schedule
- 可以后续把 future predictor 接 DiT 中间特征，而不只是 context object tokens
