# 0624 训练链路排查记录

## 当前目标

- 让 object 条件分支真正参与主去噪损失回传，而不是只靠 `object_aux_heads`
- 在正式训练前，把每一步问题、原因、修复和验证记录清楚

## 阶段 1：先把 smoke test 跑通

### 问题

- 初始 smoke test 卡在 `construct_trainer`
- 之后又遇到 `meta tensor`、分布式初始化、dataset 冷启动过慢等问题

### 原因

- WAN 新增 object 分支参数不在原 checkpoint 中，`from_pretrained(..., low_cpu_mem_usage=True)` 时会残留 `meta`
- smoke 脚本误调用 `trainer.train()`，触发 `accelerate` 的分布式入口
- `PhysStateEpisodeDataset` 初始化时会扫描全部 3600 个样本做 context 过滤，单次 smoke 冷启动过重

### 解决

- `WanModel.from_pretrained` 改成 `low_cpu_mem_usage=False`
- smoke 脚本改成只切 `train mode`，不走真正的训练 launcher
- 增加 `init_scan_limit`，只在 smoke 时限制 dataset 初始化扫描范围

### 证据

- `/data/gaoya/AAA_test_video/0623/train/smoke_test/wan_init_profile.json`
- `/data/gaoya/AAA_test_video/0623/train/smoke_test/smoke_report.json`

## 阶段 2：确认 object 分支没有形成有效主损失梯度

### 现象

- smoke 可以完整跑到 `forward/backward`
- `object_pooler` 和 `object_aux_heads` 有梯度
- `object_adapter` 的梯度统计为 0

### 初步怀疑

- `object_gate` 初值为 0，object 分支被门控完全关死
- WAN 内新增 object 分支参数虽然被标记为 trainable，但实际没有进入主损失路径

### 处理

- 给 `object_gate` 设置小的非零初值 `0.1`
- 将 `object_cross_attn / norm4 / object_gate / object_embedding` 从冻结主干中单独解冻

### 结果

- WAN object 分支参数被纳入 trainable 集合
- 但 `object_adapter` 仍然没有拿到有效主损失梯度

## 阶段 3：把断点从“猜测”收敛成证据

### 诊断 1：最小梯度链路

#### 观测

- `object_latent_tokens.grad_abs_sum > 0`
- `object_context.grad_abs_sum == 0`

#### 结论

- 梯度不是丢在 `object_pooler`
- 梯度断在 `object_latent_tokens -> object_adapter -> object_context -> WAN object branch` 的后半段

### 诊断 2：WAN object 分支参数梯度

#### 观测

- `object_cross_attn.q/k/v`、`norm4`、`object_embedding` 梯度几乎全 0
- 只有 `object_cross_attn.o.bias` 有明显非零梯度

#### 结论

- 主损失确实进入了 object 分支末端残差位置
- 但 object attention 主体没有形成正常前向输出

### 诊断 3：前向数值检查

#### 观测

- `object_embedding_in_absmax ≈ 4.5`
- `object_embedding_out_absmax = 0.0`
- `block0_q_absmax ≈ 5.7e-31`
- `block0_k_absmax = 0.0`
- `block0_v_absmax = 0.0`
- `block0_object_delta_absmax = 0.0`

#### 结论

- 不是 `object_context` 没信息
- 是 `object_embedding` 前向直接输出了全 0，导致整个 object attention 塌掉

### 诊断 4：参数本体检查

#### 观测

- `object_embedding.0.weight/bias = 0`
- `object_embedding.2.weight/bias = 0`
- `blocks.0.object_cross_attn.q/k/v/o.base_layer.weight` 量级约 `1e-36`

#### 结论

- 问题不是 dtype 把正常值压没
- 是“checkpoint 中不存在的新增 object 分支参数”在加载后几乎变成全 0

## 当前根因判断

- WAN 主 checkpoint 不包含 object 分支
- 这些缺失参数在当前 `from_pretrained` 路径里没有得到可靠初始化
- 结果是：
  - `object_embedding` 前向输出全 0
  - `object_cross_attn` 的 `q/k/v` 全 0
  - 主损失对 object 分支只有极弱的末端偏置梯度
  - `object_context` 无法形成有效梯度回传

## 当前修复方案

- 在加载 WAN checkpoint 之后，显式重初始化缺失的 object 分支参数：
  - `object_embedding`
  - 每个 block 的 `object_cross_attn`
  - 每个 block 的 `norm4`
  - 每个 block 的 `object_gate`

### 初始化策略

- `object_embedding`：按 upstream `text_embedding/object_embedding` 逻辑，用 `normal_(std=0.02)`
- `object_cross_attn.q/k/v/o`：按 upstream 线性层逻辑，用 `xavier_uniform_`
- `norm4.weight=1, norm4.bias=0`
- `object_gate=0.1`

## 下一步验证项

- 重新跑 smoke，确认：
  - `object_embedding_out_absmax > 0`
  - `block0_object_delta_absmax > 0`
  - `object_context.grad_abs_sum > 0`
  - `object_adapter` 梯度不再为 0
- 如果通过，再上 `gpu6,7` 正式训练

## 阶段 4：修复后 smoke 复验结果

### 实际修改

- 在 `wan_context_model.py` 的 WAN checkpoint 加载后流程中，新增 `_reinitialize_missing_object_branch(base_dit)`
- 对 checkpoint 中缺失的 object 分支显式初始化：
  - `object_embedding` 用 `normal_(std=0.02)`
  - `object_cross_attn.q/k/v/o` 用 `xavier_uniform_`
  - `norm4.weight=1, norm4.bias=0`
  - `object_gate=0.1`

### 复验命令

```bash
CUDA_VISIBLE_DEVICES=7 LOCAL_RANK=0 CODEX_DEBUG_TRAINER_INIT=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python -u \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/smoke_train_forward_backward.py \
  --output-dir /data/gaoya/AAA_test_video/0623/train/smoke_test \
  --resolution 128 224 \
  --init-scan-limit 1
```

### 复验证据

- `smoke_report.json` 状态为 `ok`
- `train/loss_total = 0.10992655158042908`
- `train/loss_main = 0.08561114966869354`
- `object_adapter.grad_norm_sum = 9417.615341186523`
- `bundle.dit.base_model.model.object_embedding.0.weight`
- `bundle.dit.base_model.model.object_embedding.2.weight`
- `bundle.dit.base_model.model.blocks.0.object_cross_attn.v.base_layer.weight`
- `bundle.dit.base_model.model.blocks.0.object_cross_attn.o.base_layer.weight`

上述参数都出现在 top gradients 中，说明主损失已经穿过 WAN object branch，而不再只是停在 `object_aux_heads`。

### 结论

- 根因不是 `object_pooler` 或 `object_adapter` 本身断梯度
- 根因是 WAN 中新增但不在原 checkpoint 内的 object 分支没有得到可靠初始化，前向近似塌成 0
- 显式重初始化后：
  - `object_context` 不再是“有输入、无回传”的死分支
  - `object_adapter` 拿到显著主损失梯度
  - WAN object branch 的 `object_embedding / object_cross_attn` 也拿到稳定非零梯度
- 可以进入 `gpu6,7` 双卡真实训练验证阶段

## 阶段 5：真实双卡 pilot 的新问题

### 已确认通过的部分

- `gpu6,7` 双卡 `accelerate launch` 可以正常启动
- `W&B` 可以成功创建 run
- `ContextVideoTrainer`、`WAN DiT`、`JEPA`、`VGGT`、`CoTracker`、`SAM2 prior` 都能完成初始化
- 真实训练能跑到：
  - `first batch fetched`
  - `first forward done`
  - `first backward done`

### 新暴露的问题

- 第 1 个优化步之前，`nonfinite_probe` 检测到梯度非有限
- 首个报错参数：
  - `bundle.dit.base_model.model.object_embedding.0.weight`
- 错误形态：
  - `bad_count = 12582912`
  - `has_nan = True`
  - `has_posinf = False`
  - `has_neginf = False`

### 结论

- 这不是“训练起不来”问题
- 主链路已经真实走到 object branch 的 forward/backward
- 当前阻塞点是：
  - object 分支在 smoke 下数值稳定
  - 但在真实双卡、真实分辨率、真实 batch 前向里，第 1 步就出现 `object_embedding` 梯度 NaN

### 当前判断方向

- 优先排查 object 分支的数值尺度，而不是分布式、W&B、checkpoint 或 dataloader
- 重点看：
  - `object_context / object_latent_tokens / object_embedding` 的幅值是否在真实训练里过大
  - `object_gate`、`object_cross_attn`、`norm4` 的混精度路径是否导致溢出
  - 是否需要在 object 支路增加更强的归一化、裁剪或更保守的初始化/门控

### 后续定位结果

- 单卡、真实分辨率 smoke 复现结果：
  - `512x896`
  - `enable_sam2_priors=true`
  - `track_source=cotracker`
  - `object_embedding` 梯度有限
  - `object_adapter / object_pooler / object_cross_attn` 梯度也都有限
- 因此问题不是“真实分辨率必炸”
- 当前更像是双卡训练配置里的混精度不匹配：
  - config 里原来设置的是 `mixed_precision: fp16`
  - 但 WAN patch 里的关键路径实际大量依赖 `bfloat16 / float32` autocast
  - 双卡真实训练第 1 步 NaN，而单卡 smoke 无 NaN，最合理的优先修正是把训练 launcher 精度切到 `bf16`

### 当前修正

- `train_0624pybullet_wan_lora_monitor_gpu67_pilot.yaml`
- `train_0624pybullet_wan_lora_monitor_gpu67.yaml`

都已从：

- `mixed_precision: fp16`

改为：

- `mixed_precision: bf16`

## 阶段 6：bf16 后的新问题

### 现象

- `bf16` 双卡 pilot 已经不再出现第 1 步 `object_embedding` 梯度 NaN
- 第 1 个 `forward/backward` 可以完成
- 但当第 1 次 `optimizer.step()` 真正执行 AdamW 状态初始化时，显存仍然 OOM
- 即使把 `grad_accum_steps` 从 `2` 降到 `1`，OOM 仍然发生在 `optimizer.step()`，而不是前向或反向

### 结论

- `fp16` 导致的 NaN 问题已经独立解决，和当前故障不是同一个问题
- 当前新的真实训练阻塞点是优化器状态内存，而不是 object 分支数值稳定性
- 由于训练分辨率是 `512x896`、trainable 参数量又比较大，标准 `torch.optim.AdamW` 在第 1 次 step 分配 `exp_avg/exp_avg_sq` 时把单卡 48GB 显存顶满

### 进一步验证

- 单卡全分辨率 smoke 在 `512x896 + cotracker + SAM2 priors` 下可以稳定完成 `forward/backward/step`
- 说明主链路、梯度回传、loss 数值本身没有崩
- 双卡真实训练里的最后瓶颈是 optimizer state memory，而不是模型图结构错误

## 阶段 7：用 8-bit optimizer 修复 optimizer.step() OOM

### 处理思路

- 优先不改训练语义，不减少 trainable 模块
- 先把 optimizer 做成可配置项，最小化修改范围
- 保持学习率、`betas`、`eps`、`weight_decay` 不变，只切换 optimizer 实现

### 实际修改

- `training/runner.py`
  - 新增 `_build_optimizer(...)`
  - 支持：
    - `adamw`
    - `adamw8bit`
    - `paged_adamw8bit`
- `trainers/context_video_trainer.py`
  - 将 `optimizer_type / betas / eps` 从 YAML 传入 `launch_training_task(...)`
- `train_0624pybullet_wan_lora_monitor_gpu67_pilot.yaml`
- `train_0624pybullet_wan_lora_monitor_gpu67.yaml`
  - `optimization.optimizer_type` 改为 `paged_adamw8bit`

### 环境确认

- `bitsandbytes==0.49.2`
- `bitsandbytes.optim.AdamW8bit` 可用
- `bitsandbytes.optim.PagedAdamW8bit` 可用

### 单卡 smoke 复验

- 使用 `train_0624pybullet_wan_lora_monitor_gpu67_pilot.yaml` 直接跑全分辨率 smoke
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/smoke_test_fullres_pilot_bnb`
- 结果：
  - `status = ok`
  - `nonfinite_grad_tensors = []`
  - `object_adapter / object_pooler / WAN object branch` 梯度都保持有限
  - 说明切换到 `PagedAdamW8bit` 没有引入新的数值问题

### 双卡真实 pilot 命令

```bash
CUDA_VISIBLE_DEVICES=6,7 CODEX_DEBUG_TRAINER_INIT=1 CODEX_DEBUG_RUNNER_INIT=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate launch \
  --multi_gpu --num_processes 2 --gpu_ids 6,7 --mixed_precision bf16 \
  --main_process_port 29524 \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_context_video_wan.py \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67_pilot.yaml
```

### 双卡真实 pilot 结果

- 新 W&B run:
  - `qbygdrdk`
- 关键日志已经明确出现：
  - `first forward done`
  - `first backward done`
  - `first optimizer.step done`
- 训练完整跑完 `max_steps = 6`
- 最终没有出现 NaN、OOM 或 checkpoint 保存失败

### 产物证据

- checkpoint 目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67_pilot`
- 已保存：
  - `step_0000002.pt`
  - `step_0000004.pt`
  - `step_0000006.pt`
- W&B 本地日志：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/logs/wandb/wandb`

### 当前结论

- 真实训练里的两个问题已经被成功拆开并分别修复：
  - `fp16` 导致 object branch 第 1 步梯度 NaN
  - `torch AdamW` 导致 optimizer state 初始化 OOM
- 当前稳定方案是：
  - `mixed_precision: bf16`
  - `grad_accum_steps: 1`
  - `optimizer_type: paged_adamw8bit`

## 阶段 8：验证训练 checkpoint 能被 inference 使用

### 验证目标

- 确认训练产出的 `step_*.pt` 不只是“能保存”
- 还必须能被 `infer_context_video_wan.py` 的 trainable state 加载逻辑正常接住

### 验证命令

```bash
CUDA_VISIBLE_DEVICES=7 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python -u \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py \
  --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67_pilot \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67_pilot.yaml \
  --prompt "industrial rigid body simulation sphere" \
  --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 \
  --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_pilot \
  --sampling-steps 2 \
  --num-frames 24 \
  --sampling-mode prefix
```

### 验证结果

- inference 入口成功构建 trainer
- 成功加载 pilot checkpoint
- `unexpected_keys = 0`
- `model_state_key_count = 1272`
- `checkpoint_key_count = 1272`
- 采样成功完成并打印 `sampling finished`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_pilot`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_pilot/result.json`

### 关于 `missing_keys`

- 推理日志里会看到 `load_state_dict(strict=False)` 返回很多 `missing_keys`
- 这些并不是 trainable checkpoint 不兼容
- 原因是：
  - `step_*.pt` 只保存 trainable state
  - 冻结的 WAN / VGGT / CoTracker 等基础权重来自各自预训练 checkpoint，不在 `step_*.pt` 内
- 真正的 trainable 集合一致性检查是 `_load_trainable_state_into_model(...)` 前面的 normalized key 对齐：
  - `model_state_key_count = checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
- 因此可以认定这份训练 checkpoint 与当前 inference 代码兼容

## 当前总状态

- object 条件主损失梯度问题：已修复
- `fp16` 下的双卡 NaN：已修复
- `optimizer.step()` 的显存 OOM：已修复
- 双卡真实 pilot：已成功跑通并保存 checkpoint
- inference 加载训练 checkpoint：已验证通过
- 训练真实进入第 1 步 optimizer 累积过程
- 但在下一次前向过程中触发显存 OOM：
  - GPU 约 `47.37 GiB` 已占满
  - 报错位置出现在第 2 个 micro-step 的 forward 链路中

### 原因判断

- 当前配置是：
  - `batch_size: 1`
  - `grad_accum_steps: 2`
  - `resolution: 512x896`
  - `track_source: cotracker`
  - `enable_sam2_priors: true`
- 第 1 个 micro-step 反向后梯度已常驻显存
- 第 2 个 micro-step 再做一轮完整前向时，峰值显存超过单卡 48GB
- 所以这不是数值稳定性问题，而是 accumulation 带来的峰值显存问题

### 当前修正

- `train_0624pybullet_wan_lora_monitor_gpu67_pilot.yaml`
- `train_0624pybullet_wan_lora_monitor_gpu67.yaml`

都已从：

- `grad_accum_steps: 2`

改为：

- `grad_accum_steps: 1`
