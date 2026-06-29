# Object Token Teacher-Student

这个子目录用于放一条新的 teacher-student 实验分支，目标是最大化复用当前
`code_vjepa_vggt` 里的现有 object branch、Wan 注入链路、dataset、监督函数和训练 runner，
避免把 `train_v_newtrain.py` 或默认分支再复制一遍。

## 默认方案

默认只考虑三阶段主线，其他备选分支放在文末。

### Stage 1: Oracle Injection

- 输入视频：
  - full video: `[B, 3, T, H, W]`
  - context video: `[B, 3, 8, H, W]`
- object slots:
  - `O = 4`
  - `Q = 8`
- 目标：
  - 用 full video 构造 `oracle object latent tokens`
  - 把它作为 Wan object cross-attn 的 teacher condition
  - 训练 Wan 的 object injection 分支

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
- `ObjectConditionAdapter`
- `ContextVideoTrainer` / `WanContextVideoModel`

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
- `runtime.py`
  - 基于现有 `ContextVideoTrainer` 的薄封装 trainer
- `train_stage1_oracle_injection.py`
  - Stage 1 入口
- `train_stage2_predictor.py`
  - Stage 2 入口
- `train_stage3_bridge.py`
  - Stage 3 占位入口

## 当前实现边界

这版优先保证：

- 新分支代码和默认分支隔离
- 尽量 import 现有模块
- Stage 2 可以独立读取 `context_* / future_*` supervision
- 训练入口沿用现有 config / runner 风格

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
