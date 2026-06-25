# 0624 训练链路排查记录
换了一个训练架构

## 2026-06-25 补充

- 当前 `train_v_newtrain.py` 的 object-heads-only strict 方案里，`object_pooler.depth_proj.*` 和 `object_pooler.world_proj.*` 已显式冻结。
- 原因不是这两组参数有 bug，而是当前训练前向没有向 `ObjectTubeProjector.forward()` 传入 `vggt_depth / vggt_world_points`，所以这条 VGGT 几何分支根本不会被执行。
- 现在的文档口径统一为：
  - `object_pooler` 里真正参与当前 strict 训练的是 `jepa_proj / latent_proj / track_geom_proj / out_norm`
  - `depth_proj / world_proj` 属于“结构上存在、当前方案未接线”的子模块，不再计入有效 trainable 集合

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

## 阶段 9：正式长训启动与实时监控

### 正式训练命令

```bash
CUDA_VISIBLE_DEVICES=6,7 CODEX_DEBUG_TRAINER_INIT=1 CODEX_DEBUG_RUNNER_INIT=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate launch \
  --multi_gpu --num_processes 2 --gpu_ids 6,7 --mixed_precision bf16 \
  --main_process_port 29525 \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_context_video_wan.py \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml
```

### 启动时配置确认

- 使用配置：
  - `train_0624pybullet_wan_lora_monitor_gpu67.yaml`
- 关键参数：
  - `mixed_precision: bf16`
  - `grad_accum_steps: 1`
  - `optimizer_type: paged_adamw8bit`
  - `save_every: 20`
  - `max_steps: 20000`

### 正式训练实时观测

- 新 W&B run:
  - `flslwgvw`
- 启动日志确认：
  - `accelerator.prepare done`
  - `first batch fetched`
  - `first forward done`
  - `first backward done`
  - `first optimizer.step done`
- 说明正式训练已经越过 pilot 中曾经失败过的两个关键点：
  - 不再出现 `fp16` 下第 1 步梯度 NaN
  - 不再出现 `torch AdamW` 第 1 次 `optimizer.step()` OOM

### 早期训练行为

- 训练在监控期间连续推进到至少 `step 20`
- 期间进度条持续更新，loss 不是常数，说明主损失和条件分支并没有“表面跑动、实际不更新”
- 监控中观测到的 loss 序列样本包括：
  - `0.0812`
  - `0.6298`
  - `0.1222`
  - `0.4469`
  - `0.5522`
  - `1.0167`
  - `0.4789`
  - `1.2983`
  - `1.1436`
  - `0.1985`
  - `2.7593`
  - `3.3281`
- 这些波动说明训练在真实数据上正常前进，而不是 loss 静态卡死或非数值异常后被静默吞掉

### 正式 checkpoint 保存证据

- checkpoint 目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67`
- 已确认保存：
  - `step_0000020.pt`
- 文件大小约：
  - `5.2G`

### 当前判断

- 截至 `step 20`，正式训练满足以下条件：
  - 启动成功
  - 双卡同步正常
  - loss 持续变化
  - 权重在更新，否则不会进入稳定的多步训练和 checkpoint 保存
  - checkpoint 保存链路正常
- 下一步继续抽检 `step_0000020.pt` 的 inference 兼容性，确保“正式训练 checkpoint”也能被推理脚本直接使用

### 抽检过程中的附加问题

- 第一次对 `step_0000020.pt` 做 inference 抽检时，误把推理任务也放到了训练正在占用的 GPU 上
- 结果在 `ContextVideoTrainer` 初始化 `WAN VAE` 时直接触发 CUDA OOM
- 这不是 checkpoint 不兼容，也不是训练权重损坏
- 根因是：
  - 正式训练已经独占 `gpu6,7`
  - 推理脚本又尝试在同一张卡上初始化完整 WAN/VAE/T5
  - 导致资源冲突

### 处理

- 不打断正式训练
- 将 inference 抽检改到空闲 GPU 上单独执行
- 这样可以同时满足：
  - 训练持续跑
  - 正式 checkpoint 仍然可以独立验证推理兼容性

### 正式 checkpoint 抽检结果

- 抽检文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000020.pt`
- 抽检输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step20`
- 抽检结果：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `sampling finished`
- 说明：
  - 正式长训产出的 checkpoint 可以被当前 `infer_context_video_wan.py` 正常加载
  - trainable state 和 inference 侧期望的 trainable key 集合是一致的
  - 日志中的大量 `missing_keys` 仍然只是“冻结底座权重不保存在 step checkpoint 中”的预期现象，不是 checkpoint 损坏

### 当前阶段性结论

- 正式训练已经在 `gpu6,7` 上稳定启动并持续推进
- 训练期间 loss 持续变化、checkpoint 正常保存、推理兼容性抽检通过
- 到这一阶段，可以认为：
  - 主损失梯度链路已经有效形成
  - 训练和推理两端的 trainable checkpoint 交接链路已经打通

## 阶段 10：验证正式训练不是“假更新”

### 新证据

- 正式训练继续推进后，已经生成：
  - `step_0000040.pt`
- checkpoint 目录当前包含：
  - `step_0000020.pt`
  - `step_0000040.pt`

### 权重更新核查方法

- 直接加载：
  - `step_0000020.pt`
  - `step_0000040.pt`
- 对它们的 `model` trainable state 做逐 tensor 差分，而不是只看 loss 曲线

### 核查结果

- `keys20 = 1272`
- `keys40 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 197705.3647`
- 最大单元素变化：
  - `max_abs_diff = 0.0026037730`
  - `max_abs_key = bundle.dit.base_model.model.object_embedding.2.weight`

### 变化样例

- `bundle.dit.base_model.model.blocks.0.cross_attn.k.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.k.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.o.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.o.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.q.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.q.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.v.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.v.lora_B.default.weight`

### 结论

- 这说明正式训练中的 trainable parameters 确实在持续更新
- 不是只有 loss 在打印，也不是只有 optimizer step 在形式上执行
- object branch 相关参数也在变化：
  - 最大变化张量就是 `object_embedding.2.weight`
- 因此“有效主损失梯度已经形成并驱动参数更新”这一点，在正式训练阶段也有了直接权重证据
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

## 阶段 11：正式训练持续监控补充（2026-06-23 19:45 UTC）

### 本轮监控目标

- 确认正式训练在完成早期稳定性验证后，是否仍然持续向前推进
- 排除“进程还活着但实际不再训练”的假活跃状态
- 观察在尚未到下一个 checkpoint 保存点前，loss、step、W&B 本地日志是否仍同步更新

### 监控结果

- 前台训练会话持续输出，没有出现新的 Python 异常、CUDA OOM、NCCL 异常或显式卡死
- 已经观察到 step 从此前确认过的 `54` 继续推进到：
  - `55`
  - `56`
  - `57`
- 对应进度输出样例：
  - `55/20000 ... loss=0.3443`
  - `56/20000 ... loss=1.7659`
  - `57/20000 ... loss=0.3609`
- loss 仍在变化，不是固定值重复打印

### 对“是否真的还在训练”的交叉验证

- 训练主进程和两个 worker 仍然存活，两个 worker 维持高 CPU 占用
- GPU 侧仍由 `gpu6` 和 `gpu7` 占用大显存：
  - `gpu6`: `46757 MiB`
  - `gpu7`: `48197 MiB`
- W&B 本地 run 目录继续刷新：
  - `debug-internal.log` 更新时间到 `2026-06-23 19:45:02 UTC`
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 19:45:05 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 19:45:10 UTC`

### 关于为什么此时还没有新 checkpoint

- 当前 checkpoint 目录仍只有：
  - `step_0000020.pt`
  - `step_0000040.pt`
- 这与当前已观测 step `57` 并不矛盾
- 原因很直接：
  - checkpoint 保存间隔是按固定 step 间隔触发
  - 既然已有 `step_0000040.pt`，那么下一次正常保存应在更后的保存点
  - 当前监控时刻尚未推进到下一个保存点，因此“没有新 checkpoint”不能解释为训练停滞

### 当前结论

- 正式训练在修复：
  - fp16 数值不稳定
  - AdamW optimizer-state OOM
  - grad accumulation 峰值显存问题
 之后，仍然在继续稳定推进
- 到 `2026-06-23 19:45 UTC` 为止，没有观察到新的中断信号
- 目前最合理判断是：
  - 训练仍健康运行
  - 只是单 step 耗时较长，且尚未到下一个 checkpoint 保存点
- 下一轮应继续等待更高 step 或新的 checkpoint（如 `step_0000060.pt`）出现，再做下一次权重变化或推理兼容性抽检

## 阶段 12：step_0000060.pt 验证（2026-06-23 19:50 UTC）

### 新进展

- 正式训练继续向前推进，并已生成新的 checkpoint：
  - `step_0000060.pt`
- 前台训练输出已明确出现：
  - `60/20000 ... loss=0.8774`
- 说明训练不仅越过了此前的 `step 57`，而且已经抵达下一个 checkpoint 保存点

### 1. 权重继续更新核查：`step40 -> step60`

- 本轮继续直接比较：
  - `step_0000040.pt`
  - `step_0000060.pt`
- 仍然只比较 trainable `model` state，而不是依赖 loss 曲线做间接判断

### 差分结果

- `keys40 = 1272`
- `keys60 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 224303.6667`
- `max_abs_diff = 0.0022221070`
- `max_abs_key = bundle.dit.base_model.model.blocks.27.object_cross_attn.o.lora_B.default.weight`

### 变化样例

- `bundle.dit.base_model.model.blocks.0.cross_attn.k.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.k.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.o.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.o.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.q.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.q.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.v.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.v.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.ffn.0.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对权重差分的解释

- 与此前 `step20 -> step40` 一样，这一轮仍然是 `1270/1272` 个 trainable tensor 发生变化
- 而且本轮最大变化张量已经落在：
  - `object_cross_attn.o.lora_B.default.weight`
- 这进一步加强了判断：
  - object 条件分支相关参数并不是“挂着不用”
  - 它们在正式训练中持续被主损失驱动更新

### 2. `step_0000060.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮推理抽检放在 `gpu0` 上执行
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step60`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step60/result.json`

### 关键结果

- 推理日志明确出现：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
- `result.json` 中记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`

### 关于 `missing=3306` 的说明

- 这一现象与此前 `step20`、`step40` 的推理验证完全一致
- 原因不是 checkpoint 损坏，而是：
  - `step_*.pt` 只保存 trainable state
  - 基座模型和冻结模块的参数不在 trainable checkpoint 内
- 因此真正的兼容性判断标准仍然是：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
- 本轮 `step_0000060.pt` 完全满足这个标准

### 3. 本轮结论

- 正式训练已经持续推进到并越过新的保存点
- `step_0000060.pt` 成功保存
- `step40 -> step60` 间 trainable 权重继续大范围变化
- 最大变化项已经直接落在 object cross-attn 参数上
- `step_0000060.pt` 也已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 19:50 UTC` 为止，已有连续证据证明：
  - 主损失梯度链路有效
  - 正式训练持续正常进行
  - 权重持续真实更新
  - 新保存的 checkpoint 能够被推理脚本正确接收并生成采样结果
- 下一步继续监控更后续的 step 和 checkpoint；如果中途再出现异常，再继续定位和修正

## 阶段 13：正式训练持续推进监控补充（2026-06-23 19:52 UTC）

### 本轮监控重点

- 在 `step_0000060.pt` 已完成推理验证之后，继续确认正式训练是否仍然稳定推进
- 重点排除两类假象：
  - 进程仍在但 step 不再增长
  - W&B 或日志仍刷新，但实际训练线程已经停住

### 本轮观察结果

- 前台训练会话继续正常输出，没有出现新的：
  - Python traceback
  - CUDA OOM
  - NCCL/accelerate 异常
  - 显式卡死
- 已继续观察到 step 从 `70` 之后推进到：
  - `71`
  - `72`
  - `73`
  - `74`
- 对应 loss 仍在变化：
  - `71/20000 ... loss=0.1493`
  - `72/20000 ... loss=0.9847`
  - `73/20000 ... loss=0.3522`
  - `74/20000 ... loss=0.1869`

### 交叉验证

- GPU 仍由正式训练主占用：
  - `gpu6`: `46757 MiB`
  - `gpu7`: `48197 MiB`
- W&B 本地 run 文件继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 19:50:53 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 19:51:27 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 19:51:33 UTC`

### 关于为什么还没有新 checkpoint

- 截止本轮监控时，checkpoint 目录仍为：
  - `step_0000020.pt`
  - `step_0000040.pt`
  - `step_0000060.pt`
- 这与当前已经推进到 `step 74` 并不冲突
- 原因仍然是：
  - checkpoint 保存是固定 step 间隔触发
  - 当前只是还没有到下一个保存点
- 因此“暂时没有新 checkpoint”在本轮不能解释为训练异常

### 本轮结论

- 到 `2026-06-23 19:52 UTC` 为止，正式训练仍在继续健康推进
- loss 持续变化，step 持续增长，W&B 本地落盘持续刷新
- 目前没有新的故障信号
- 下一轮继续等待新的 checkpoint 出现后，再补一轮：
  - 权重差分验证
  - 推理兼容性抽检

## 阶段 14：step_0000080.pt 验证（2026-06-23 19:57 UTC）

### 新进展

- 正式训练继续推进并生成新的 checkpoint：
  - `step_0000080.pt`
- 前台训练输出已明确出现：
  - `79/20000 ... loss=1.8112`
  - `80/20000 ... loss=0.2624`
- 说明训练已经稳定推进到下一个保存点并完成保存

### 1. 权重继续更新核查：`step60 -> step80`

- 本轮直接比较：
  - `step_0000060.pt`
  - `step_0000080.pt`
- 继续只比较 trainable `model` state

### 差分结果

- `keys60 = 1272`
- `keys80 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 206325.8813`
- `max_abs_diff = 0.0018013896`
- `max_abs_key = bundle.dit.base_model.model.blocks.29.object_cross_attn.o.lora_B.default.weight`

### 变化样例

- `bundle.dit.base_model.model.blocks.0.cross_attn.k.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.k.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.o.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.o.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.q.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.q.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.v.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.v.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.ffn.0.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80` 都呈现出同样模式：
  - `1270/1272` 个 trainable tensor 在持续变化
- 本轮最大变化项再次落在：
  - `object_cross_attn.o.lora_B.default.weight`
- 这说明 object 条件分支不是偶然更新一次，而是在正式训练中持续获得主损失驱动

### 2. `step_0000080.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step80`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step80/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
- `result.json` 中记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`

### 关于 `missing=3306` 的说明

- 与 `step20`、`step40`、`step60` 的抽检结果一致
- 这仍然是“checkpoint 只保存 trainable state”的预期现象
- 因此兼容性判断标准仍然是：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
- `step_0000080.pt` 完全满足这一标准

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000080.pt` 成功保存
- `step60 -> step80` 之间 trainable 权重继续显著更新
- object cross-attn 相关权重再次成为最大变化项
- `step_0000080.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 19:57 UTC` 为止，已经形成连续多轮闭环证据：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 每个新 checkpoint 都能被推理脚本接住并完成采样
- 下一步继续保持监控；如果后续出现新异常，再继续定位并修复

### 补充监控：`step80` 验证后训练继续推进

- `step_0000080.pt` 验证完成后，正式训练没有停在保存点
- 前台继续观测到 step 向前推进到：
  - `81`
  - `82`
  - `83`
  - `86`
  - `87`
  - `88`
  - `89`
- 对应 loss 仍持续变化，例如：
  - `81/20000 ... loss=0.2794`
  - `82/20000 ... loss=0.3538`
  - `83/20000 ... loss=1.4464`
  - `86/20000 ... loss=0.7142`
  - `87/20000 ... loss=1.2498`
  - `88/20000 ... loss=1.5316`
  - `89/20000 ...`

### 补充结论

- 这进一步说明：
  - checkpoint 保存和推理抽检不会破坏正式训练主流程
  - 正式训练在 `step80` 之后仍持续健康推进

## 阶段 15：step_0000100.pt 验证（2026-06-23 20:06 UTC）

### 新进展

- 正式训练继续推进并生成新的 checkpoint：
  - `step_0000100.pt`
- 前台训练输出已明确出现：
  - `98/20000 ... loss=1.5453`
  - `99/20000 ... loss=1.0880`
  - `100/20000 ... loss=1.8301`

### 1. 权重继续更新核查：`step80 -> step100`

- 本轮直接比较：
  - `step_0000080.pt`
  - `step_0000100.pt`
- 继续只比较 trainable `model` state

### 差分结果

- `keys80 = 1272`
- `keys100 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 128441.4915`
- `max_abs_diff = 0.0029494218`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 变化样例

- `bundle.dit.base_model.model.blocks.0.cross_attn.k.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.k.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.o.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.o.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.q.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.q.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.v.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.cross_attn.v.lora_B.default.weight`
- `bundle.dit.base_model.model.blocks.0.ffn.0.lora_A.default.weight`
- `bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项落在 `ffn.0.lora_B.default.weight`
- 同时，object cross-attn 相关权重仍持续出现在变化样例中
- 这说明训练不是只更新局部少量参数，而是整个 trainable 子空间都在持续被主损失驱动

### 2. `step_0000100.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step100`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step100/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
- `result.json` 中记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`

### 关于 `missing=3306` 的说明

- 与此前 `step20`、`step40`、`step60`、`step80` 的结果完全一致
- 仍然是“checkpoint 只保存 trainable state”的预期现象
- 因此兼容性判断标准仍然是：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
- `step_0000100.pt` 完全满足这一标准

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000100.pt` 成功保存
- `step80 -> step100` 之间 trainable 权重继续大范围真实更新
- `step_0000100.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 20:06 UTC` 为止，已经形成更长链条的连续证据：
  - 训练稳定推进到 `step 100`
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 补充监控：`step100` 验证后训练继续推进

- `step_0000100.pt` 验证完成后，正式训练没有停在保存点
- 前台继续观测到 step 向前推进到：
  - `101`
  - `102`
  - `106`
  - `107`
  - `108`
  - `109`
- 对应 loss 仍持续变化，例如：
  - `101/20000 ... loss=0.4393`
  - `102/20000 ... loss=1.7394`
  - `106/20000 ... loss=0.4599`
  - `107/20000 ... loss=1.9186`
  - `108/20000 ... loss=1.7418`
  - `109/20000 ... loss=1.2166`

### 补充结论

- 这进一步说明：
  - `step_0000100.pt` 的保存与推理抽检没有干扰正式训练主流程
  - 训练在超过 `step 100` 后仍持续健康推进

### 继续监控补充（2026-06-23 20:08 UTC）

- 在 `step_0000100.pt` 验证完成后，正式训练继续推进到：
  - `110`
  - `111`
- 对应 loss 仍在变化：
  - `110/20000 ... loss=1.7574`
  - `111/20000 ... loss=0.4366`
- W&B 本地 run 文件仍持续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:08:11 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:08:32 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:08:36 UTC`

### 本轮结论

- 到 `2026-06-23 20:08 UTC` 为止，训练在 `step100` 验证之后仍保持健康推进
- 当前没有新的错误、停滞或 checkpoint 兼容性回归信号

### 继续监控补充（2026-06-23 20:12 UTC）

- 正式训练继续推进并生成新的 checkpoint：
  - `step_0000120.pt`
- 在 `step_0000100.pt` 验证完成后，前台继续明确观测到：
  - `112`
  - `118`
  - `119`
  - `120`
- 对应 loss 仍持续变化，例如：
  - `119/20000 ... loss=0.3406`
  - `120/20000 ... loss=0.1996`
- W&B 本地 run 文件继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:12:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:12:45 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:12:47 UTC`

## 阶段 16：step_0000120.pt 验证（2026-06-23 20:16 UTC）

### 新进展

- 正式训练继续推进并稳定越过新的保存点
- 在本轮核查过程中，前台训练输出继续推进到：
  - `121`
  - `122`
  - `123`
  - `124`
  - `126`
  - `127`
  - `128`
  - `129`
- 对应 loss 仍持续变化，例如：
  - `121/20000 ... loss=0.2423`
  - `122/20000 ... loss=0.0333`
  - `123/20000 ... loss=0.8597`
  - `124/20000 ... loss=0.2422`
  - `126/20000 ... loss=1.2978`
  - `127/20000 ... loss=1.5864`
  - `128/20000 ... loss=0.8677`
  - `129/20000 ... loss=1.7380`

### 1. 权重继续更新核查：`step100 -> step120`

- 本轮直接比较：
  - `step_0000100.pt`
  - `step_0000120.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys100 = 1272`
- `keys120 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 32878.2380`
- `max_abs_diff = 0.0027459145`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000120.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“只保存 checkpoint 但参数不再动”的回归迹象

### 2. `step_0000120.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step120`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step120/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
- `result.json` 中记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`

### 关于 `missing=3306` 的说明

- 与此前 `step20`、`step40`、`step60`、`step80`、`step100` 的结果完全一致
- 仍然是“checkpoint 只保存 trainable state”的预期现象
- 因此兼容性判断标准仍然是：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
- `step_0000120.pt` 完全满足这一标准

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000120.pt` 成功保存
- `step100 -> step120` 之间 trainable 权重继续大范围真实更新
- `step_0000120.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 20:16 UTC` 为止，连续闭环证据已进一步延长到 `step 120+`：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果
- W&B 本地 run 文件也继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:15:17 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:16:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:16:06 UTC`

### 本轮补充结论

- `step_0000120.pt` 的保存与推理抽检同样没有干扰正式训练主流程
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常或 checkpoint 兼容性回归信号

### 继续监控补充（2026-06-23 20:17 UTC）

- `step_0000120.pt` 验证完成后，正式训练继续健康推进，前台继续明确观测到：
  - `132`
  - `133`
  - `134`
- 对应 loss 仍持续变化，例如：
  - `132/20000 ... loss=0.6317`
  - `133/20000 ... loss=0.3460`
  - `134/20000 ... loss=0.5021`
- checkpoint 目录在本轮时刻仍停留在：
  - `step_0000120.pt`
- 这与当前每 `20` step 保存一次 checkpoint 的行为一致，没有出现“训练前进但保存异常”的信号
- W&B 本地 run 文件继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:17:35 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:17:44 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:17:47 UTC`

### 本轮结论

- 到 `2026-06-23 20:17 UTC` 为止，正式训练在 `step120` 验证之后仍持续正常推进
- 当前没有新的错误、停滞、保存失败、W&B 中断或 checkpoint 兼容性回归信号

### 继续监控补充（2026-06-23 20:19 UTC）

- 正式训练在本轮继续向前推进，前台继续明确观测到：
  - `135`
  - `136`
  - `137`
- 对应 loss 仍持续变化，例如：
  - `135/20000 ... loss=1.0913`
  - `136/20000 ... loss=0.0579`
  - `137/20000 ... loss=0.8703`
- checkpoint 目录在本轮时刻仍停留在：
  - `step_0000120.pt`
- 这仍与当前每 `20` step 保存一次 checkpoint 的行为一致，没有出现保存节奏异常
- W&B 本地 run 文件继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:19:32 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:19:32 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:19:31 UTC`

### 本轮结论

- 到 `2026-06-23 20:19 UTC` 为止，正式训练在 `step120` 验证之后仍持续健康推进
- 当前没有新的错误、停滞、OOM、NCCL 异常、保存失败或 W&B 中断信号

### 继续监控补充（2026-06-23 20:20 UTC）

- 正式训练继续向前推进并达到新的保存点附近，前台继续明确观测到：
  - `138`
  - `139`
  - `140`
- 对应 loss 仍持续变化，例如：
  - `138/20000 ... loss=0.0100`
  - `139/20000 ... loss=0.8028`
  - `140/20000 ... loss=1.1838`
- checkpoint 目录在本轮时刻成功生成：
  - `step_0000140.pt`
- W&B 本地 run 文件继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:20:45 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:20:48 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:21:02 UTC`

## 阶段 17：step_0000140.pt 验证（2026-06-23 20:23 UTC）

### 新进展

- 正式训练继续推进并稳定越过新的保存点
- 在本轮核查过程中，前台训练输出继续推进到：
  - `141`
  - `142`
  - `143`
  - `144`
  - `145`
- 对应 loss 仍持续变化，例如：
  - `141/20000 ... loss=0.4342`
  - `142/20000 ... loss=0.8212`
  - `143/20000 ... loss=0.7589`
  - `144/20000 ... loss=1.6112`
  - `145/20000 ... loss=0.1264`

### 1. 权重继续更新核查：`step120 -> step140`

- 本轮直接比较：
  - `step_0000120.pt`
  - `step_0000140.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys120 = 1272`
- `keys140 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 22539.2483`
- `max_abs_diff = 0.0024811565`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000140.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 正常落盘但参数已不再变化”的回归迹象

### 2. `step_0000140.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step140`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step140/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
- `result.json` 中记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`

### 关于 `missing=3306` 的说明

- 与此前 `step20`、`step40`、`step60`、`step80`、`step100`、`step120` 的结果完全一致
- 仍然是“checkpoint 只保存 trainable state”的预期现象
- 因此兼容性判断标准仍然是：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
- `step_0000140.pt` 完全满足这一标准

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000140.pt` 成功保存
- `step120 -> step140` 之间 trainable 权重继续大范围真实更新
- `step_0000140.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 20:23 UTC` 为止，连续闭环证据已进一步延长到 `step 140+`：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果
- W&B 本地 run 文件也继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:21:47 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:22:47 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:22:53 UTC`

### 本轮补充结论

- `step_0000140.pt` 的保存与推理抽检同样没有干扰正式训练主流程
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

### 继续监控补充（2026-06-23 20:25 UTC）

- `step_0000140.pt` 验证完成后，正式训练继续健康推进，前台继续明确观测到：
  - `150`
  - `151`
- 对应 loss 仍持续变化，例如：
  - `150/20000 ... loss=1.6343`
  - `151/20000 ... loss=1.4673`
- checkpoint 目录在本轮时刻仍停留在：
  - `step_0000140.pt`
- 这仍与当前每 `20` step 保存一次 checkpoint 的行为一致，没有出现保存节奏异常
- W&B 本地 run 文件继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:25:17 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:25:17 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:25:29 UTC`

### 本轮结论

- 到 `2026-06-23 20:25 UTC` 为止，正式训练在 `step140` 验证之后仍持续健康推进
- 当前没有新的错误、停滞、OOM、NCCL 异常、保存失败或 W&B 中断信号

### 继续监控补充（2026-06-23 20:26 UTC）

- `step_0000140.pt` 验证完成后，正式训练继续健康推进，前台继续明确观测到：
  - `152`
  - `153`
  - `154`
  - `155`
- 对应 loss 仍持续变化，例如：
  - `152/20000 ... loss=0.1957`
  - `153/20000 ... loss=0.3374`
  - `154/20000 ... loss=0.6748`
  - `155/20000 ... loss=1.8875`
- checkpoint 目录在本轮时刻仍停留在：
  - `step_0000140.pt`
- 这仍与当前每 `20` step 保存一次 checkpoint 的行为一致，没有出现保存节奏异常
- W&B 本地 run 文件继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:26:13 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:26:33 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:26:35 UTC`

### 本轮结论

- 到 `2026-06-23 20:26 UTC` 为止，正式训练在 `step140` 验证之后仍持续健康推进
- 当前没有新的错误、停滞、OOM、NCCL 异常、保存失败或 W&B 中断信号

### 继续监控补充（2026-06-23 20:29 UTC）

- 正式训练继续向前推进并达到新的保存点附近，前台继续明确观测到：
  - `156`
  - `157`
  - `158`
  - `159`
  - `160`
- 对应 loss 仍持续变化，例如：
  - `156/20000 ... loss=0.2879`
  - `157/20000 ... loss=0.4640`
  - `158/20000 ... loss=0.1814`
  - `159/20000 ... loss=1.9135`
  - `160/20000 ... loss=1.7635`
- checkpoint 目录在本轮时刻成功生成：
  - `step_0000160.pt`
- W&B 本地 run 文件继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:29:17 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:29:08 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:29:17 UTC`

## 阶段 18：step_0000160.pt 验证（2026-06-23 20:31 UTC）

### 新进展

- 正式训练继续推进并稳定越过新的保存点
- 在本轮核查过程中，前台训练输出继续推进到：
  - `161`
  - `162`
  - `163`
  - `165`
  - `166`
  - `167`
- 对应 loss 仍持续变化，例如：
  - `161/20000 ... loss=1.6345`
  - `162/20000 ... loss=1.7791`
  - `163/20000 ... loss=1.3647`
  - `165/20000 ... loss=1.5943`
  - `166/20000 ... loss=1.5435`
  - `167/20000 ... loss=0.3791`

### 1. 权重继续更新核查：`step140 -> step160`

- 本轮直接比较：
  - `step_0000140.pt`
  - `step_0000160.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys140 = 1272`
- `keys160 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 25539.5189`
- `max_abs_diff = 0.0024101962`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000160.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 正常落盘但参数已不再变化”的回归迹象

### 2. `step_0000160.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step160`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step160/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
- `result.json` 中记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`

### 关于 `missing=3306` 的说明

- 与此前 `step20`、`step40`、`step60`、`step80`、`step100`、`step120`、`step140` 的结果完全一致
- 仍然是“checkpoint 只保存 trainable state”的预期现象
- 因此兼容性判断标准仍然是：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
- `step_0000160.pt` 完全满足这一标准

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000160.pt` 成功保存
- `step140 -> step160` 之间 trainable 权重继续大范围真实更新
- `step_0000160.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 20:31 UTC` 为止，连续闭环证据已进一步延长到 `step 160+`：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果
- W&B 本地 run 文件也继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:30:21 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:31:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:30:57 UTC`

### 本轮补充结论

- `step_0000160.pt` 的保存与推理抽检同样没有干扰正式训练主流程
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

### 继续监控补充（2026-06-23 20:30 UTC）

- `step_0000160.pt` 验证完成后，正式训练继续健康推进，前台继续明确观测到：
  - `168`
  - `170`
- 对应 loss 仍持续变化，例如：
  - `168/20000 ... loss=0.8195`
  - `170/20000 ... loss=0.0124`
- W&B 本地 run 文件继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:30:21 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:31:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:30:57 UTC`

### 本轮结论

- 到 `2026-06-23 20:31 UTC` 之后的继续观测为止，正式训练在 `step160` 验证之后仍持续健康推进
- 当前没有新的错误、停滞、OOM、NCCL 异常、保存失败或 W&B 中断信号

### 继续监控补充（2026-06-23 20:34 UTC）

- `step_0000160.pt` 验证完成后，正式训练继续健康推进，前台继续明确观测到：
  - `172`
  - `173`
  - `174`
- 对应 loss 仍持续变化，例如：
  - `172/20000 ... loss=0.1761`
  - `173/20000 ... loss=1.5718`
  - `174/20000 ... loss=1.6814`
- checkpoint 目录在本轮时刻仍停留在：
  - `step_0000160.pt`
- 这仍与当前每 `20` step 保存一次 checkpoint 的行为一致，没有出现保存节奏异常
- W&B 本地 run 文件继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:34:24 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:34:47 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:34:52 UTC`

### 本轮结论

- 到 `2026-06-23 20:34 UTC` 为止，正式训练在 `step160` 验证之后仍持续健康推进
- 当前没有新的错误、停滞、OOM、NCCL 异常、保存失败或 W&B 中断信号

## 阶段 19：step_0000180.pt 验证（2026-06-23 20:41 UTC）

### 新进展

- 正式训练继续推进并稳定越过新的保存点
- 在本轮核查启动时，checkpoint 目录已经新增：
  - `step_0000180.pt`
- 本轮核查过程中，前台训练输出继续推进到：
  - `179`
  - `180`
  - `181`
  - `182`
  - `184`
- 对应 loss 仍持续变化，例如：
  - `179/20000 ... loss=1.9152`
  - `180/20000 ... loss=1.8360`
  - `181/20000 ... loss=0.0083`
  - `182/20000 ... loss=1.9226`
  - `184/20000 ... loss=0.2022`

### 1. 权重继续更新核查：`step160 -> step180`

- 本轮直接比较：
  - `step_0000160.pt`
  - `step_0000180.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys160 = 1272`
- `keys180 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 22266.2260`
- `max_abs_diff = 0.0022789692`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000180.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 正常落盘但参数已停止更新”的回归迹象

### 2. `step_0000180.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step180`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step180/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
- `result.json` 中记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`

### 关于本轮推理日志中额外告警的说明

- 本轮推理过程还出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - Wan VAE / SDP kernel 的 FutureWarning 或 fallback 提示
  - 基础 Wan 权重加载后对 object cross-attn 相关模块的 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断仍然以：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
  为准
- `step_0000180.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000180.pt` 成功保存
- `step160 -> step180` 之间 trainable 权重继续大范围真实更新
- `step_0000180.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 20:41 UTC` 为止，连续闭环证据已进一步延长到 `step 180+`：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果
- W&B 本地 run 文件也继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:40:32 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:41:32 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:41:37 UTC`

### 本轮补充结论

- `step_0000180.pt` 的保存与推理抽检同样没有干扰正式训练主流程
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

### 继续监控补充（2026-06-23 20:42 UTC）

- `step_0000180.pt` 验证完成后，正式训练继续健康推进，前台继续明确观测到：
  - `185`
  - `188`
  - `189`
- 对应 loss 仍持续变化，例如：
  - `185/20000 ... loss=0.3608`
  - `188/20000 ... loss=0.2687`
  - `189/20000 ... loss=1.1565`
- checkpoint 目录在本轮时刻最新仍已包含：
  - `step_0000180.pt`
- 这仍与当前每 `20` step 保存一次 checkpoint 的行为一致，没有出现保存节奏异常

### 本轮结论

- 到 `2026-06-23 20:42 UTC` 为止，正式训练在 `step180` 验证之后仍持续健康推进
- 当前没有新的错误、停滞、OOM、NCCL 异常、保存失败或 W&B 中断信号

## 阶段 20：重新核实“主损失是否真的在回传” （2026-06-23 20:48 UTC）

### 为什么还要再核实一次

- 虽然前面已经看到 checkpoint 持续更新、推理 checkpoint 持续可用
- 但这仍然不能单独证明“object 条件分支拿到的是主去噪损失梯度”，因为理论上也可能主要是 aux head 在更新
- 所以本轮做一次更直接的分项 backward 对照：
  - 只对 `loss_main` backward
  - 只对 aux loss backward
  - 再对 `loss_total` backward
- 核心目标是把“谁在被哪一部分 loss 驱动”从现象判断变成直接证据

### 本轮代码调整

- 在 `context_video_trainer.py` 中新增 `self.last_loss_breakdown`
- 每次 `forward` 后缓存：
  - `loss_main`
  - `track_aux_loss`
  - `box_aux_loss`
  - `depth_aux_loss`
  - `loss_total`
  - 以及对应 `lambda_*`
- 这一步没有改训练语义，只是把分项 loss 暴露给诊断脚本，便于做对照 backward

### 本轮诊断脚本

- 新增：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/analyze_main_loss_gradients.py`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/smoke_test_main_loss_diag`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/smoke_test_main_loss_diag/main_loss_grad_report.json`

### 诊断方式

- 使用正式训练配置：
  - `train_0624pybullet_wan_lora_monitor_gpu67.yaml`
- 仍然保持：
  - `512x896`
  - `track_source=cotracker`
  - `enable_sam2_priors=true`
- 为了缩短 dataset 冷启动，本轮只额外设置：
  - `init_scan_limit=1`
- 不改正式训练本身，也不碰 `gpu6,7`
- 诊断只在 `gpu0` 上进行

### 前向分项数值

- `loss_total = 0.0792993307`
- `loss_main = 0.0557915345`
- `loss_track_aux = 0.0700420365`
- `loss_box_aux = 0.1650359035`
- `loss_depth_aux = 0.0`
- 当前分项权重：
  - `lambda_main = 1.0`
  - `lambda_track_aux = 0.1`
  - `lambda_box_aux = 0.1`
  - `lambda_depth_aux = 0.0`

### 1. 只对 `loss_main` backward 的结果

- `bundle.dit`：
  - `params_with_grad = 1234`
  - `grad_norm_sum = 55704.5399`
  - `grad_abs_max = 330.0`
- `object_pooler`：
  - `params_with_grad = 18`
  - `grad_norm_sum = 12271.5968`
  - `grad_abs_max = 24.7919`
- `object_adapter`：
  - `params_with_grad = 8`
  - `grad_norm_sum = 5175.3986`
  - `grad_abs_max = 13.6596`
- `object_aux_heads`：
  - `params_with_grad = 0`
  - `grad_norm_sum = 0.0`

### 对 `loss_main` 结果的解释

- 这说明在完全不依赖 aux head 的情况下：
  - WAN DiT 主干 trainable 部分拿到强梯度
  - `object_pooler` 拿到强梯度
  - `object_adapter` 也拿到强梯度
- 并且 `object_aux_heads` 在这一轮没有参与梯度
- 所以可以直接排除一种错误理解：
  - 不是 `object_aux_heads` 假装带动了 object 分支
  - 是主去噪损失本身就已经穿过 `object_context -> WAN object branch`

### 2. 只对 aux loss backward 的结果

- `bundle.dit`：
  - `params_with_grad = 0`
  - `grad_norm_sum = 0.0`
- `object_pooler`：
  - `params_with_grad = 18`
  - `grad_norm_sum = 0.401526`
  - `grad_abs_max = 0.0008672862`
- `object_aux_heads`：
  - `params_with_grad = 12`
  - `grad_norm_sum = 0.841698`
  - `grad_abs_max = 0.0139678102`
- `object_adapter`：
  - `params_with_grad = 0`
  - `grad_norm_sum = 0.0`
- `trainable_missing_grad_count = 1242`

### 对 aux-only 结果的解释

- aux loss 主要只驱动：
  - `object_aux_heads`
  - 与其直接相连的一部分 `object_pooler`
- 它不会把梯度送回 `bundle.dit`
- 也不会把梯度送回 `object_adapter`
- 所以当前架构下：
  - `object_adapter` 是否被训练，几乎完全取决于主损失
  - 这反过来证明：前面在正式训练中看到 `object_adapter` 权重持续变化，不可能只是 aux loss 造成的

### 3. 对 `loss_total` backward 的结果

- `bundle.dit`：
  - `params_with_grad = 1234`
  - `grad_norm_sum = 64888.6501`
  - `grad_abs_max = 458.0`
- `object_pooler`：
  - `params_with_grad = 18`
  - `grad_norm_sum = 13262.9626`
  - `grad_abs_max = 33.7859`
- `object_aux_heads`：
  - `params_with_grad = 12`
  - `grad_norm_sum = 0.841698`
  - `grad_abs_max = 0.0139678102`
- `object_adapter`：
  - `params_with_grad = 8`
  - `grad_norm_sum = 5648.6399`
  - `grad_abs_max = 18.7636`
- `trainable_missing_grad_count = 0`
- `nonfinite_grad_count = 0`

### 对 `loss_total` 结果的解释

- 总 loss 下所有 trainable tensor 都有梯度，且都有限
- `object_adapter` 的梯度强度和 `loss_main_only` 一致保持在明显非零量级
- `object_aux_heads` 只是在总 loss 下附加进来，并没有掩盖主损失的作用方向

### 本轮最关键的结论

- 现在可以更严格地说：
  - 当前训练代码里，object 条件分支已经形成有效主损失梯度
  - 这个结论不是靠 checkpoint 在变、也不是靠 aux head 在变来间接推断
  - 而是由 `loss_main_only` backward 直接证明的
- 更具体地说：
  - `object_adapter` 只在 `loss_main` 和 `loss_total` 下拿到梯度
  - 在 `aux_only` 下它完全没有梯度
- 所以当前 object 条件分支的训练主驱动力确实是去噪主损失，而不是辅助监督

### 对“当前还需要不要继续改方案”的判断

- 从“主损失梯度是否有效”这个问题本身看：
  - 当前没有再发现新的断梯度问题
  - 也没有发现 aux 伪装成主训练信号的问题
- 因此这一轮不需要为了“修主损失梯度”再做结构性改动
- 目前更合理的策略是：
  - 保持当前训练方案继续正式训练
  - 持续监控 checkpoint 权重更新、W&B、推理兼容性
  - 只在后续出现新的数值异常、loss 异常或推理兼容性回归时再修改代码

### 与正式训练现场状态的对照

- 在这轮离线诊断期间，正式训练前台继续推进到：
  - `200`
  - `205`
  - `206`
- 对应 loss 仍持续变化，例如：
  - `200/20000 ... loss=0.6670`
  - `205/20000 ... loss=1.8082`
  - `206/20000 ... loss=1.4967`
- W&B 本地 run 文件继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:48:08 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:48:36 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:48:47 UTC`

### 本轮结论

- 到 `2026-06-23 20:48 UTC` 为止：
  - 主损失对 object 条件分支的有效梯度已被再次直接证实
  - 当前没有发现需要为“主损失断梯度”继续修改代码的新证据
  - 正式训练仍在健康推进，且没有被这轮诊断打扰

## 阶段 21：step_0000200.pt 验证（2026-06-23 20:53 UTC）

### 新进展

- 正式训练继续推进并稳定越过新的保存点
- checkpoint 目录已经新增：
  - `step_0000200.pt`
- 本轮核查过程中，前台训练输出继续推进到：
  - `207`
  - `208`
  - `209`
  - `210`
  - `213`
  - `216`
- 对应 loss 仍持续变化，例如：
  - `207/20000 ... loss=0.9767`
  - `208/20000 ... loss=0.3506`
  - `209/20000 ... loss=0.1310`
  - `210/20000 ... loss=0.5599`
  - `213/20000 ... loss=0.3895`
  - `216/20000 ... loss=0.0350`

### 1. 权重继续更新核查：`step180 -> step200`

- 本轮直接比较：
  - `step_0000180.pt`
  - `step_0000200.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys180 = 1272`
- `keys200 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 22539.4465`
- `max_abs_diff = 0.0024592392`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000200.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“只有 checkpoint 在保存，但参数更新已经停掉”的回归迹象

### 2. `step_0000200.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step200`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step200/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
- `result.json` 中记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`

### 关于本轮推理日志中额外告警的说明

- 本轮推理过程仍然出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - Wan VAE / SDP kernel 的 FutureWarning 或 fallback 提示
  - 基础 Wan 权重加载后对 object cross-attn 相关模块的 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准仍然保持不变：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000200.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000200.pt` 成功保存
- `step180 -> step200` 之间 trainable 权重继续大范围真实更新
- `step_0000200.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 20:53 UTC` 为止，连续闭环证据已进一步延长到 `step 200+`：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 主损失对 object 条件分支的有效梯度已被单独 backward 诊断直接证实
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果
- W&B 本地 run 文件也继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:50:07 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:50:16 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:50:17 UTC`

### 本轮补充结论

- `step_0000200.pt` 的保存与推理抽检同样没有干扰正式训练主流程
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 阶段 22：step_0000220.pt 验证（2026-06-23 20:57 UTC）

### 新进展

- 正式训练继续推进并稳定越过新的保存点
- checkpoint 目录已经新增：
  - `step_0000220.pt`
- 本轮核查过程中，前台训练输出继续推进到：
  - `217`
  - `218`
  - `219`
  - `220`
  - `221`
  - `224`
  - `226`
- 对应 loss 仍持续变化，例如：
  - `217/20000 ... loss=0.0290`
  - `218/20000 ... loss=0.9261`
  - `219/20000 ... loss=0.0301`
  - `220/20000 ... loss=0.9298`
  - `221/20000 ... loss=0.3215`
  - `224/20000 ... loss=1.6894`
  - `226/20000 ... loss=1.9183`

### 1. 权重继续更新核查：`step200 -> step220`

- 本轮直接比较：
  - `step_0000200.pt`
  - `step_0000220.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys200 = 1272`
- `keys220 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 25836.0871`
- `max_abs_diff = 0.0027137604`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000220.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 继续保存但参数实际已经不再变化”的回归迹象

### 2. `step_0000220.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step220`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step220/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
- `result.json` 中记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`

### 关于本轮推理日志中额外告警的说明

- 本轮推理过程仍然出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - Wan VAE / SDP kernel 的 FutureWarning 或 fallback 提示
  - 基础 Wan 权重加载后对 object cross-attn 相关模块的 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准仍然保持不变：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000220.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000220.pt` 成功保存
- `step200 -> step220` 之间 trainable 权重继续大范围真实更新
- `step_0000220.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 20:57 UTC` 为止，连续闭环证据已进一步延长到 `step 220+`：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 主损失对 object 条件分支的有效梯度已被单独 backward 诊断直接证实
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果
- W&B 本地 run 文件也继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 20:54:18 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 20:54:50 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 20:55:02 UTC`

### 本轮补充结论

- `step_0000220.pt` 的保存与推理抽检同样没有干扰正式训练主流程
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 阶段 23：step_0000240.pt 验证（2026-06-23 21:08 UTC）

### 新进展

- 正式训练继续推进并稳定越过新的保存点
- checkpoint 目录已经新增：
  - `step_0000240.pt`
- 本轮核查过程中，前台训练输出继续推进到：
  - `236`
  - `237`
  - `238`
  - `239`
  - `240`
  - `241`
  - `242`
  - `243`
  - `246`
  - `247`
- 对应 loss 仍持续变化，例如：
  - `236/20000 ... loss=1.4872`
  - `237/20000 ... loss=1.9208`
  - `238/20000 ... loss=0.6932`
  - `239/20000 ... loss=1.0177`
  - `240/20000 ... loss=1.1086`
  - `241/20000 ... loss=0.6404`
  - `242/20000 ... loss=1.2659`
  - `243/20000 ... loss=1.8468`
  - `246/20000 ... loss=0.1241`
  - `247/20000 ... loss=0.8306`

### 1. 权重继续更新核查：`step220 -> step240`

- 本轮直接比较：
  - `step_0000220.pt`
  - `step_0000240.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys220 = 1272`
- `keys240 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 25091.4704`
- `max_abs_diff = 0.0024072216`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000240.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 继续保存但参数实际已经不再变化”的回归迹象

### 2. `step_0000240.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step240`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step240/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`

### 关于本轮推理日志中额外告警的说明

- 本轮推理过程仍然出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - Wan VAE / SDP kernel 的 FutureWarning 或 fallback 提示
  - 基础 Wan 权重加载后对 object cross-attn 相关模块的 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准仍然保持不变：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000240.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000240.pt` 成功保存
- `step220 -> step240` 之间 trainable 权重继续大范围真实更新
- `step_0000240.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 21:08 UTC` 为止，连续闭环证据已进一步延长到 `step 240+`：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 主损失对 object 条件分支的有效梯度已被单独 backward 诊断直接证实
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果
- W&B 本地 run 文件也继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 21:07:40 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 21:07:47 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 21:07:47 UTC`

### 本轮补充结论

- `step_0000240.pt` 的保存与推理抽检同样没有干扰正式训练主流程
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 阶段 24：step_0000260.pt 验证（2026-06-23 21:16 UTC）

### 新进展

- 正式训练继续推进并稳定越过新的保存点
- checkpoint 目录已经新增：
  - `step_0000260.pt`
- 本轮核查过程中，前台训练输出继续推进到：
  - `251`
  - `252`
  - `253`
  - `254`
  - `255`
  - `256`
  - `257`
  - `258`
  - `259`
  - `260`
  - `262`
  - `263`
  - `264`
  - `266`
- 对应 loss 仍持续变化，例如：
  - `251/20000 ... loss=0.4401`
  - `252/20000 ... loss=0.0509`
  - `253/20000 ... loss=1.5103`
  - `254/20000 ... loss=0.1503`
  - `255/20000 ... loss=0.2822`
  - `256/20000 ... loss=0.0070`
  - `257/20000 ... loss=1.5429`
  - `258/20000 ... loss=1.9130`
  - `259/20000 ... loss=1.9377`
  - `260/20000 ... loss=0.5890`
  - `262/20000 ... loss=1.5975`
  - `263/20000 ... loss=1.7904`
  - `264/20000 ... loss=0.0975`
  - `266/20000 ... loss=0.8158`

### 1. 权重继续更新核查：`step240 -> step260`

- 本轮直接比较：
  - `step_0000240.pt`
  - `step_0000260.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys240 = 1272`
- `keys260 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 27043.3220`
- `max_abs_diff = 0.0029285662`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000260.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 继续保存但参数实际已经不再变化”的回归迹象

### 2. `step_0000260.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step260`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step260/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`

### 关于本轮推理日志中额外告警的说明

- 本轮推理过程仍然出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - Wan VAE / SDP kernel 的 FutureWarning 或 fallback 提示
  - 基础 Wan 权重加载后对 object cross-attn 相关模块的 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准仍然保持不变：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000260.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000260.pt` 成功保存
- `step240 -> step260` 之间 trainable 权重继续大范围真实更新
- `step_0000260.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 21:16 UTC` 为止，连续闭环证据已进一步延长到 `step 260+`：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 主损失对 object 条件分支的有效梯度已被单独 backward 诊断直接证实
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果
- W&B 本地 run 文件也继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 21:15:15 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 21:15:50 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 21:15:47 UTC`

### 本轮补充结论

- `step_0000260.pt` 的保存与推理抽检同样没有干扰正式训练主流程
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 阶段 25：step_0000280.pt 验证（2026-06-23 21:25 UTC）

### 新进展

- 正式训练继续推进并稳定越过新的保存点
- checkpoint 目录已经新增：
  - `step_0000280.pt`
- 本轮核查过程中，前台训练输出继续推进到：
  - `270`
  - `271`
  - `272`
  - `273`
  - `274`
  - `275`
  - `276`
  - `277`
  - `278`
  - `279`
  - `280`
  - `281`
  - `282`
  - `283`
  - `284`
  - `285`
  - `287`
- 对应 loss 仍持续变化，例如：
  - `270/20000 ... loss=1.3829`
  - `271/20000 ... loss=0.3201`
  - `272/20000 ... loss=0.7863`
  - `273/20000 ... loss=0.8805`
  - `274/20000 ... loss=0.1198`
  - `275/20000 ... loss=0.6439`
  - `276/20000 ... loss=0.1942`
  - `277/20000 ... loss=0.2831`
  - `278/20000 ... loss=0.3940`
  - `279/20000 ... loss=1.7102`
  - `280/20000 ... loss=1.6368`
  - `281/20000 ... loss=0.3966`
  - `282/20000 ... loss=0.1019`
  - `283/20000 ... loss=0.1691`
  - `284/20000 ... loss=0.5885`
  - `285/20000 ... loss=0.5903`
  - `287/20000 ... loss=0.0973`

### 1. 权重继续更新核查：`step260 -> step280`

- 本轮直接比较：
  - `step_0000260.pt`
  - `step_0000280.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys260 = 1272`
- `keys280 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 23225.7662`
- `max_abs_diff = 0.0026418956`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000280.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 继续保存但参数实际已经不再变化”的回归迹象

### 2. `step_0000280.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step280`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step280/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`

### 关于本轮推理日志中额外告警的说明

- 本轮推理过程仍然出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - Wan VAE / SDP kernel 的 FutureWarning 或 fallback 提示
  - 基础 Wan 权重加载后对 object cross-attn 相关模块的 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准仍然保持不变：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000280.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000280.pt` 成功保存
- `step260 -> step280` 之间 trainable 权重继续大范围真实更新
- `step_0000280.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 21:25 UTC` 为止，连续闭环证据已进一步延长到 `step 280+`：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 主损失对 object 条件分支的有效梯度已被单独 backward 诊断直接证实
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果
- W&B 本地 run 文件也继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 21:24:18 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 21:25:13 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 21:25:17 UTC`

### 本轮补充结论

- `step_0000280.pt` 的保存与推理抽检同样没有干扰正式训练主流程
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 阶段 26：step_0000300.pt 验证（2026-06-23 21:33 UTC）

### 新进展

- 正式训练继续推进并稳定越过新的保存点
- checkpoint 目录已经新增：
  - `step_0000300.pt`
- 本轮核查过程中，前台训练输出继续推进到：
  - `292`
  - `293`
  - `294`
  - `295`
  - `296`
  - `297`
  - `298`
  - `299`
  - `300`
  - `301`
  - `302`
  - `304`
- 对应 loss 仍持续变化，例如：
  - `292/20000 ... loss=0.0506`
  - `293/20000 ... loss=0.0730`
  - `294/20000 ... loss=0.0053`
  - `295/20000 ... loss=0.6585`
  - `296/20000 ... loss=0.0684`
  - `297/20000 ... loss=0.3520`
  - `298/20000 ... loss=0.9457`
  - `299/20000 ... loss=0.1914`
  - `300/20000 ... loss=1.5625`
  - `301/20000 ... loss=1.5533`
  - `302/20000 ... loss=0.1802`
  - `304/20000 ... loss=0.3734`

### 1. 权重继续更新核查：`step280 -> step300`

- 本轮直接比较：
  - `step_0000280.pt`
  - `step_0000300.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys280 = 1272`
- `keys300 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 22983.7449`
- `max_abs_diff = 0.0024036607`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000300.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 继续保存但参数实际已经不再变化”的回归迹象

### 2. `step_0000300.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step300`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step300/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`

### 关于本轮推理日志中额外告警的说明

- 本轮推理过程仍然出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - Wan VAE / SDP kernel 的 FutureWarning 或 fallback 提示
  - 基础 Wan 权重加载后对 object cross-attn 相关模块的 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准仍然保持不变：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000300.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000300.pt` 成功保存
- `step280 -> step300` 之间 trainable 权重继续大范围真实更新
- `step_0000300.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 21:33 UTC` 为止，连续闭环证据已进一步延长到 `step 300+`：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 主损失对 object 条件分支的有效梯度已被单独 backward 诊断直接证实
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果
- W&B 本地 run 文件也继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 21:32:50 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 21:33:34 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 21:33:32 UTC`

### 本轮补充结论

- `step_0000300.pt` 的保存与推理抽检同样没有干扰正式训练主流程
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 阶段 27：step_0000320.pt 验证（2026-06-23 21:42 UTC）

### 新进展

- 正式训练继续推进并稳定越过新的保存点
- checkpoint 目录已经新增：
  - `step_0000320.pt`
- 本轮核查过程中，前台训练输出继续推进到：
  - `314`
  - `315`
  - `316`
  - `317`
  - `318`
  - `319`
  - `320`
  - `321`
  - `324`
  - `325`
  - `326`
  - `327`
  - `328`
- 对应 loss 仍持续变化，例如：
  - `314/20000 ... loss=0.7530`
  - `315/20000 ... loss=0.2584`
  - `316/20000 ... loss=1.8866`
  - `317/20000 ... loss=0.4497`
  - `318/20000 ... loss=1.4279`
  - `319/20000 ... loss=1.7647`
  - `320/20000 ... loss=1.6386`
  - `321/20000 ... loss=1.6959`
  - `324/20000 ... loss=1.9142`
  - `325/20000 ... loss=1.0415`
  - `326/20000 ... loss=1.7766`
  - `327/20000 ... loss=1.8328`
  - `328/20000 ... loss=1.9438`

### 1. 权重继续更新核查：`step300 -> step320`

- 本轮直接比较：
  - `step_0000300.pt`
  - `step_0000320.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys300 = 1272`
- `keys320 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 29223.0675`
- `max_abs_diff = 0.0027167057`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000320.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 继续保存但参数实际已经不再变化”的回归迹象

### 2. `step_0000320.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step320`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step320/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step320`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 本轮 `result.json` 也已经成功落盘，没有出现 checkpoint 可加载但采样结束后目录未写出的异常

### 关于本轮推理日志中额外告警的说明

- 本轮推理过程仍然出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - 基础 Wan 权重加载后对 object cross-attn 相关模块的 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准仍然保持不变：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000320.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000320.pt` 成功保存
- `step300 -> step320` 之间 trainable 权重继续大范围真实更新
- `step_0000320.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 21:42 UTC` 为止，连续闭环证据已进一步延长到 `step 320+`：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 主损失对 object 条件分支的有效梯度已被单独 backward 诊断直接证实
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果
- W&B 本地 run 文件也继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 21:42:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 21:42:37 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 21:42:32 UTC`

### 本轮补充结论

- `step_0000320.pt` 的保存与推理抽检同样没有干扰正式训练主流程
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 阶段 28：step_0000340.pt 验证（2026-06-23 21:51 UTC）

### 新进展

- 正式训练继续推进并稳定越过新的保存点
- checkpoint 目录已经新增：
  - `step_0000340.pt`
- 本轮核查过程中，前台训练输出继续推进到：
  - `332`
  - `333`
  - `334`
  - `335`
  - `336`
  - `337`
  - `338`
  - `339`
  - `340`
  - `341`
  - `342`
  - `343`
  - `344`
  - `345`
  - `346`
  - `347`
  - `348`
- 对应 loss 仍持续变化，例如：
  - `332/20000 ... loss=0.4173`
  - `333/20000 ... loss=1.7255`
  - `334/20000 ... loss=1.2058`
  - `335/20000 ... loss=0.4414`
  - `336/20000 ... loss=1.4075`
  - `337/20000 ... loss=1.1025`
  - `338/20000 ... loss=0.7132`
  - `339/20000 ... loss=0.7290`
  - `340/20000 ... loss=1.3958`
  - `341/20000 ... loss=0.4214`
  - `342/20000 ... loss=1.6851`
  - `343/20000 ... loss=0.7154`
  - `344/20000 ... loss=0.5266`
  - `345/20000 ... loss=1.8452`
  - `346/20000 ... loss=0.6250`
  - `347/20000 ... loss=0.0544`
  - `348/20000 ... loss=0.2827`

### 1. 权重继续更新核查：`step320 -> step340`

- 本轮直接比较：
  - `step_0000320.pt`
  - `step_0000340.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys320 = 1272`
- `keys340 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 32491.7073`
- `max_abs_diff = 0.0032491339`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000340.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 继续保存但参数实际已经不再变化”的回归迹象

### 2. `step_0000340.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step340`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step340/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step340`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 本轮 `result.json` 同样成功落盘，没有出现 checkpoint 可加载但采样结束后目录未写出的异常

### 关于本轮推理日志中额外告警的说明

- 本轮推理过程仍然出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - 基础 Wan 权重加载后对 object cross-attn 相关模块的 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准仍然保持不变：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000340.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000340.pt` 成功保存
- `step320 -> step340` 之间 trainable 权重继续大范围真实更新
- `step_0000340.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 21:51 UTC` 为止，连续闭环证据已进一步延长到 `step 340+`：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 主损失对 object 条件分支的有效梯度已被单独 backward 诊断直接证实
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果
- W&B 本地 run 文件也继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 21:51:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 21:50:58 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 21:51:02 UTC`

### 本轮补充结论

- `step_0000340.pt` 的保存与推理抽检同样没有干扰正式训练主流程
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 阶段 29：step_0000360.pt 验证（2026-06-23 21:59 UTC）

### 新进展

- 正式训练继续推进并稳定越过新的保存点
- checkpoint 目录已经新增：
  - `step_0000360.pt`
- 本轮核查过程中，前台训练输出继续推进到：
  - `351`
  - `352`
  - `353`
  - `354`
  - `355`
  - `356`
  - `357`
  - `358`
  - `359`
  - `360`
  - `361`
  - `364`
  - `365`
  - `366`
  - `367`
- 对应 loss 仍持续变化，例如：
  - `351/20000 ... loss=1.8967`
  - `352/20000 ... loss=0.9617`
  - `353/20000 ... loss=0.8827`
  - `354/20000 ... loss=1.4248`
  - `355/20000 ... loss=1.5370`
  - `356/20000 ... loss=1.9116`
  - `357/20000 ... loss=1.8532`
  - `358/20000 ... loss=0.1634`
  - `359/20000 ... loss=1.8649`
  - `360/20000 ... loss=1.7093`
  - `361/20000 ... loss=0.6940`
  - `364/20000 ... loss=0.9952`
  - `365/20000 ... loss=1.7650`
  - `366/20000 ... loss=1.1145`
  - `367/20000 ... loss=1.3529`

### 1. 权重继续更新核查：`step340 -> step360`

- 本轮直接比较：
  - `step_0000340.pt`
  - `step_0000360.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys340 = 1272`
- `keys360 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 37085.8050`
- `max_abs_diff = 0.0035082884`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000360.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 继续保存但参数实际已经不再变化”的回归迹象

### 2. `step_0000360.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step360`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step360/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step360`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 本轮 `result.json` 同样成功落盘，没有出现 checkpoint 可加载但采样结束后目录未写出的异常

### 关于本轮推理日志中额外告警的说明

- 本轮推理过程仍然出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - 基础 Wan 权重加载后对 object cross-attn 相关模块的 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准仍然保持不变：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000360.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 3. 本轮结论

- 正式训练已继续推进到新的 checkpoint 保存点
- `step_0000360.pt` 成功保存
- `step340 -> step360` 之间 trainable 权重继续大范围真实更新
- `step_0000360.pt` 已被推理脚本成功加载并完成采样

### 当前总体判断

- 到 `2026-06-23 21:59 UTC` 为止，连续闭环证据已进一步延长到 `step 360+`：
  - 正式训练稳定推进
  - loss 持续变化
  - 权重持续真实更新
  - object 条件分支持续参与更新
  - 主损失对 object 条件分支的有效梯度已被单独 backward 诊断直接证实
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果
- W&B 本地 run 文件也继续刷新到本轮时刻附近：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 21:59:47 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 21:59:50 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 21:59:47 UTC`

### 本轮补充结论

- `step_0000360.pt` 的保存与推理抽检同样没有干扰正式训练主流程
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-23 22:09 UTC 左右继续监控：phase 30，正式训练推进到 `step_0000380.pt`

### 0. 当前运行状态

- 正式训练前台 session `27065` 持续存活，没有重启
- checkpoint 目录已经新增：
  - `step_0000380.pt`
- `step_0000380.pt` 文件大小已稳定为：
  - `5.2G`
- 本轮核查期间，前台训练继续推进到：
  - `381`
  - `382`
  - `383`
  - `384`
  - `385`
  - `386`
  - `387`
  - `388`
  - `390`
- 对应 loss 仍持续变化，例如：
  - `381/20000 ... loss=1.2552`
  - `382/20000 ... loss=1.0549`
  - `383/20000 ... loss=1.6466`
  - `384/20000 ... loss=1.6163`
  - `385/20000 ... loss=0.5775`
  - `386/20000 ... loss=0.3972`
  - `387/20000 ... loss=0.0738`
  - `388/20000 ... loss=0.2609`
  - `390/20000 ... loss=1.5241`

### 1. 权重继续更新核查：`step360 -> step380`

- 本轮直接比较：
  - `step_0000360.pt`
  - `step_0000380.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys360 = 1272`
- `keys380 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 32531.1768`
- `max_abs_diff = 0.0023604427`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000380.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 继续保存但参数实际已冻结”的回归迹象

### 2. `step_0000380.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step380`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step380/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step380`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step360` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- 这与前面代码改造后的目标设计一致：
  - 训练与推理链路当前默认走 CoTracker 轨迹
  - object 条件保留 latent-time 维度后，再展平成 `T_lat * K = 2 * 8 = 16` 个 object token 送入 Wan object cross-attn

### 4. 关于本轮推理日志中额外告警的说明

- 本轮推理过程仍然出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA/Flash attention fallback 到较慢 kernel 的提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准仍然保持不变：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000380.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 本地刷新状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 22:09:01 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 22:09:17 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 22:09:25 UTC`
- 这说明本地 W&B 记录端没有出现明显中断

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000380.pt`
- `step360 -> step380` 之间 trainable 权重继续大范围真实更新
- `step_0000380.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-23 22:09 UTC` 为止，连续闭环证据已进一步延长到 `step 380+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-23 22:16 UTC 左右继续监控：phase 31，正式训练推进到 `step_0000400.pt`

### 0. 当前运行状态

- 正式训练前台 session `27065` 持续存活，没有重启
- checkpoint 目录已经新增：
  - `step_0000400.pt`
- `step_0000400.pt` 文件大小已稳定为：
  - `5.2G`
- 本轮核查期间，前台训练继续推进到：
  - `397`
  - `398`
  - `399`
  - `400`
  - `401`
  - `402`
  - `407`
- 对应 loss 仍持续变化，例如：
  - `397/20000 ... loss=1.9192`
  - `399/20000 ... loss=1.6576`
  - `400/20000 ... loss=0.2348`
  - `401/20000 ... loss=0.3810`
  - `402/20000 ... loss=0.2354`
  - `407/20000 ... loss=0.8488`

### 1. 权重继续更新核查：`step380 -> step400`

- 本轮直接比较：
  - `step_0000380.pt`
  - `step_0000400.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys380 = 1272`
- `keys400 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 26854.2349`
- `max_abs_diff = 0.0017482769`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000400.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 继续保存但参数实际已经不再变化”的回归迹象

### 2. `step_0000400.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step400`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step400/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step400`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step380` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- 这与前面代码改造后的目标设计一致：
  - 训练与推理链路当前默认走 CoTracker 轨迹
  - object 条件保留 latent-time 维度后，再展平成 `T_lat * K = 2 * 8 = 16` 个 object token 送入 Wan object cross-attn

### 4. 关于本轮推理日志中额外告警的说明

- 本轮推理过程仍然出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA/Flash attention fallback 到较慢 kernel 的提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准仍然保持不变：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000400.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 本地刷新状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 22:16:20 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 22:16:44 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 22:16:47 UTC`
- 这说明本地 W&B 记录端没有出现明显中断

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000400.pt`
- `step380 -> step400` 之间 trainable 权重继续大范围真实更新
- `step_0000400.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-23 22:16 UTC` 为止，连续闭环证据已进一步延长到 `step 400+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-23 22:25 UTC 左右继续监控：phase 32，正式训练推进到 `step_0000420.pt`

### 0. 当前运行状态

- 正式训练前台 session `27065` 持续存活，没有重启
- checkpoint 目录已经新增：
  - `step_0000420.pt`
- `step_0000420.pt` 文件大小已稳定为：
  - `5.2G`
- 本轮核查期间，前台训练继续推进到：
  - `417`
  - `418`
  - `419`
  - `420`
  - `421`
  - `422`
  - `428`
- 对应 loss 仍持续变化，例如：
  - `417/20000 ... loss=0.0938`
  - `418/20000 ... loss=0.2619`
  - `419/20000 ... loss=0.5297`
  - `420/20000 ... loss=0.3495`
  - `421/20000 ... loss=0.4387`
  - `422/20000 ... loss=0.1697`
  - `428/20000 ... loss=1.4245`

### 1. 权重继续更新核查：`step400 -> step420`

- 本轮直接比较：
  - `step_0000400.pt`
  - `step_0000420.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys400 = 1272`
- `keys420 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 35560.9279`
- `max_abs_diff = 0.0023396583`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400`、`step400 -> step420` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000420.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 继续保存但参数实际已经不再变化”的回归迹象

### 2. `step_0000420.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step420`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step420/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step420`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step400` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- 这与前面代码改造后的目标设计一致：
  - 训练与推理链路当前默认走 CoTracker 轨迹
  - object 条件保留 latent-time 维度后，再展平成 `T_lat * K = 2 * 8 = 16` 个 object token 送入 Wan object cross-attn

### 4. 关于本轮推理日志中额外告警的说明

- 本轮推理过程仍然出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA/Flash attention fallback 到较慢 kernel 的提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准仍然保持不变：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000420.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 本地刷新状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `files/output.log` 更新时间到 `2026-06-23 22:25:00 UTC`
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 22:25:02 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 22:25:02 UTC`
- 这说明本地 W&B 记录端没有出现明显中断

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000420.pt`
- `step400 -> step420` 之间 trainable 权重继续大范围真实更新
- `step_0000420.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-23 22:25 UTC` 为止，连续闭环证据已进一步延长到 `step 420+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-23 22:33 UTC 左右继续监控：phase 33，正式训练推进到 `step_0000440.pt`

### 0. 当前运行状态

- 正式训练前台 session `27065` 持续存活，没有重启
- checkpoint 目录已经新增：
  - `step_0000440.pt`
- `step_0000440.pt` 文件大小已稳定为：
  - `5.2G`
- 本轮核查期间，前台训练继续推进到：
  - `431`
  - `432`
  - `436`
  - `437`
  - `438`
  - `439`
  - `440`
  - `441`
  - `443`
  - `449`
- 对应 loss 仍持续变化，例如：
  - `431/20000 ... loss=1.0397`
  - `432/20000 ... loss=0.5647`
  - `436/20000 ... loss=1.6815`
  - `437/20000 ... loss=0.0502`
  - `438/20000 ... loss=0.0946`
  - `439/20000 ... loss=1.4463`
  - `440/20000 ... loss=0.1394`
  - `441/20000 ... loss=1.8636`
  - `443/20000 ... loss=0.7025`
  - `449/20000 ... loss=0.4498`

### 1. 权重继续更新核查：`step420 -> step440`

- 本轮直接比较：
  - `step_0000420.pt`
  - `step_0000440.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys420 = 1272`
- `keys440 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 35097.2775`
- `max_abs_diff = 0.0021617729`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400`、`step400 -> step420`、`step420 -> step440` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000440.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 继续保存但参数实际已经不再变化”的回归迹象

### 2. `step_0000440.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮依旧在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step440`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step440/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step440`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step420` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- 这与前面代码改造后的目标设计一致：
  - 训练与推理链路当前默认走 CoTracker 轨迹
  - object 条件保留 latent-time 维度后，再展平成 `T_lat * K = 2 * 8 = 16` 个 object token 送入 Wan object cross-attn

### 4. 关于本轮推理日志中额外告警的说明

- 本轮推理过程仍然出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA/Flash attention fallback 到较慢 kernel 的提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准仍然保持不变：
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000440.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 本地刷新状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 22:33:17 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 22:33:32 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 22:33:42 UTC`
- 这说明本地 W&B 记录端没有出现明显中断

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000440.pt`
- `step420 -> step440` 之间 trainable 权重继续大范围真实更新
- `step_0000440.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-23 22:33 UTC` 为止，连续闭环证据已进一步延长到 `step 440+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-23 22:44 UTC: phase 34, `step_0000460.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000460.pt`
- `step_0000460.pt` 文件大小已稳定为：
  - `5.2G`
- 本轮核查期间，前台训练继续推进到：
  - `469`
  - `470`
  - `471`
  - `472`
- 对应 loss 继续变化，例如：
  - `469/20000 ... loss=1.8893`
  - `470/20000 ... loss=1.3024`
  - `471/20000 ... loss=0.1801`
  - `472/20000 ... loss=1.7943`
- 这说明在 `step_0000460.pt` 产生之后，正式训练仍继续向前推进，没有在 checkpoint 保存点附近卡住

### 1. 权重继续更新核查：`step440 -> step460`

- 本轮直接比较：
  - `step_0000440.pt`
  - `step_0000460.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys440 = 1272`
- `keys460 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 26774.9081`
- `max_abs_diff = 0.0012732111`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.ffn.0.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400`、`step400 -> step420`、`step420 -> step440`、`step440 -> step460` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000460.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项依旧落在 LoRA 可训练子空间内部，没有出现“checkpoint 在保存、但训练参数已实质冻结”的回归迹象

### 2. `step_0000460.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step460`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step460/result.json`

### 关键结果

- 推理日志明确出现：
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step460`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step440` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- 这继续和当前代码设计一致：
  - 训练与推理链路默认走 CoTracker 轨迹
  - object 条件保留 latent-time 维度后，展平成 `T_lat * K = 2 * 8 = 16` 个 object token 送入 Wan object cross-attn

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 到较慢 kernel 的提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000460.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 与资源占用状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 22:42:32 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 22:43:17 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 22:43:17 UTC`
- GPU 快照显示：
  - `gpu6 = 46757 MiB`
  - `gpu7 = 48197 MiB`
- 当前没有看到新的 OOM、NCCL 或训练停滞信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000460.pt`
- `step440 -> step460` 之间 trainable 权重继续大范围真实更新
- `step_0000460.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-23 22:44 UTC` 为止，连续闭环证据已进一步延长到 `step 460+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-23 22:51 UTC: phase 35, `step_0000480.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000480.pt`
- `step_0000480.pt` 文件大小已稳定为：
  - `5.2G`
- 本轮核查期间，前台训练继续推进到：
  - `480`
  - `481`
  - `482`
  - `483`
  - `484`
  - `485`
  - `486`
  - `487`
  - `488`
  - `489`
- 对应 loss 继续变化，例如：
  - `480/20000 ... loss=0.1625`
  - `481/20000 ... loss=0.1810`
  - `482/20000 ... loss=0.8257`
  - `483/20000 ... loss=0.9834`
  - `484/20000 ... loss=0.5868`
  - `485/20000 ... loss=1.1131`
  - `486/20000 ... loss=1.3473`
  - `487/20000 ... loss=0.3396`
  - `488/20000 ... loss=1.8522`
  - `489/20000 ... loss=0.6511`
- 这说明在 `step_0000480.pt` 产生之后，正式训练仍继续向前推进，没有在 checkpoint 保存点附近卡住

### 1. 权重继续更新核查：`step460 -> step480`

- 本轮直接比较：
  - `step_0000460.pt`
  - `step_0000480.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys460 = 1272`
- `keys480 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 30281.9675`
- `max_abs_diff = 0.0013349818`
- `max_abs_key = bundle.dit.base_model.model.blocks.29.self_attn.q.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400`、`step400 -> step420`、`step420 -> step440`、`step440 -> step460`、`step460 -> step480` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000480.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项仍落在 LoRA 可训练子空间内部，没有出现“checkpoint 在保存、但训练参数已实质冻结”的回归迹象

### 2. `step_0000480.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000480.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step480 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step480`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step480/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step480`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step460` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- `result.json` 中 `sample_debug.loss = 116057.3125`
- 这继续和当前代码设计一致：
  - 训练与推理链路默认走 CoTracker 轨迹
  - object 条件保留 latent-time 维度后，展平成 `T_lat * K = 2 * 8 = 16` 个 object token 送入 Wan object cross-attn

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 到较慢 kernel 的提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000480.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 与资源占用状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 22:50:04 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 22:51:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 22:50:59 UTC`
- `step480` 推理兼容性验证期间，`gpu0` 上单独出现推理进程，占用约：
  - `3184 MiB`
- 当前没有看到新的 OOM、NCCL 或训练停滞信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000480.pt`
- `step460 -> step480` 之间 trainable 权重继续大范围真实更新
- `step_0000480.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-23 22:51 UTC` 为止，连续闭环证据已进一步延长到 `step 480+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-23 22:59 UTC: phase 36, `step_0000500.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000500.pt`
- `step_0000500.pt` 文件大小已稳定为：
  - `5.2G`
- 本轮核查期间，前台训练继续推进到：
  - `500`
  - `501`
  - `502`
  - `503`
  - `504`
  - `505`
  - `506`
  - `507`
  - `508`
  - `509`
- 对应 loss 继续变化，例如：
  - `500/20000 ... loss=0.0876`
  - `501/20000 ... loss=0.4778`
  - `502/20000 ... loss=1.1632`
  - `503/20000 ... loss=1.0829`
  - `504/20000 ... loss=0.1975`
  - `505/20000 ... loss=0.1167`
  - `506/20000 ... loss=1.3583`
  - `507/20000 ... loss=0.1263`
  - `508/20000 ... loss=1.8570`
  - `509/20000 ... loss=0.2938`
- 这说明在 `step_0000500.pt` 产生之后，正式训练仍继续向前推进，没有在 checkpoint 保存点附近卡住

### 1. 权重继续更新核查：`step480 -> step500`

- 本轮直接比较：
  - `step_0000480.pt`
  - `step_0000500.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys480 = 1272`
- `keys500 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 39814.1091`
- `max_abs_diff = 0.0015970804`
- `max_abs_key = bundle.dit.base_model.model.blocks.7.self_attn.k.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400`、`step400 -> step420`、`step420 -> step440`、`step440 -> step460`、`step460 -> step480`、`step480 -> step500` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000500.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项仍落在 LoRA 可训练子空间内部，没有出现“checkpoint 在保存、但训练参数已实质冻结”的回归迹象

### 2. `step_0000500.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000500.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step500 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step500`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step500/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step500`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step480` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- `result.json` 中 `sample_debug.loss = 142088.34375`
- 这继续和当前代码设计一致：
  - 训练与推理链路默认走 CoTracker 轨迹
  - object 条件保留 latent-time 维度后，展平成 `T_lat * K = 2 * 8 = 16` 个 object token 送入 Wan object cross-attn

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 到较慢 kernel 的提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000500.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 与资源占用状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 22:58:32 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 22:58:47 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 22:58:57 UTC`
- `step500` 推理兼容性验证期间，`gpu0` 上单独出现推理进程，占用约：
  - `23288 MiB`
- 当前没有看到新的 OOM、NCCL 或训练停滞信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000500.pt`
- `step480 -> step500` 之间 trainable 权重继续大范围真实更新
- `step_0000500.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-23 22:59 UTC` 为止，连续闭环证据已进一步延长到 `step 500+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-23 23:09 UTC: phase 37, `step_0000520.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000520.pt`
- `step_0000520.pt` 文件时间与大小稳定为：
  - `2026-06-23 23:03:10 UTC`
  - `5533953209 bytes`
- 本轮核查时，旧前台会话 `27065` 已不存在，但训练主进程仍在继续运行：
  - `4020150 /home/gaoya/miniconda3/envs/wan-cu128/bin/python /home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate launch --multi_gpu --num_processes 2 --gpu_ids 6,7 --mixed_precision bf16 --main_process_port 29525 /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_context_video_wan.py --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml`
- W&B `output.log` 尾部显示训练已继续推进到：
  - `533/20000 ... loss=1.0978`
- 上一轮观测中已经看到 `522` 到 `528` 的 loss 持续变化，本轮再确认到 `533`，说明 `step_0000520.pt` 产生后训练仍持续前进，没有在保存点附近停滞

### 1. 权重继续更新核查：`step500 -> step520`

- 本轮直接比较：
  - `step_0000500.pt`
  - `step_0000520.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys500 = 1272`
- `keys520 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 41120.62910064694`
- `max_abs_diff = 0.0012030084617435932`
- `max_abs_key = bundle.dit.base_model.model.blocks.6.self_attn.k.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400`、`step400 -> step420`、`step420 -> step440`、`step440 -> step460`、`step460 -> step480`、`step480 -> step500`、`step500 -> step520` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 这说明在 `step_0000520.pt` 保存之前，训练权重仍在持续真实更新
- 最大变化项仍落在 LoRA 可训练子空间内部，没有出现“checkpoint 在保存、但训练参数已实质冻结”的回归迹象

### 2. `step_0000520.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000520.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step520 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step520`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step520/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step520`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step500` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- `result.json` 中 `sample_debug.loss = 116658.546875`
- 这继续和当前代码设计一致：
  - 训练与推理链路默认走 CoTracker 轨迹
  - object 条件保留 latent-time 维度后，展平成 `T_lat * K = 2 * 8 = 16` 个 object token 送入 Wan object cross-attn

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 到较慢 kernel 的提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
- 这些信息没有阻止 checkpoint 加载，也没有阻止采样完成
- 兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000520.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 与资源占用状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 23:08:02 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 23:08:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 23:08:11 UTC`
- 本轮 GPU 快照显示：
  - `gpu6 = 46757 MiB`
  - `gpu7 = 48197 MiB`
- 这与正式训练仍驻留在 `gpu6,7` 的状态一致；本轮没有看到新的 OOM、NCCL 或训练停滞信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000520.pt`
- `step500 -> step520` 之间 trainable 权重继续大范围真实更新
- `step_0000520.pt` 已被推理脚本成功加载并完成采样
- 虽然最早记录的前台会话 PID `27065` 已结束，但训练任务本身已经由新的进程组继续推进，当前没有证据表明训练中断
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-23 23:09 UTC` 为止，连续闭环证据已进一步延长到 `step 520+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-23 23:15 UTC: phase 38, `step_0000540.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000540.pt`
- `step_0000540.pt` 文件时间与大小稳定为：
  - `2026-06-23 23:11:39 UTC`
  - `5533953209 bytes`
- 本轮等待新 checkpoint 时，训练已从 `538` 推进到 `541`
- 完成本轮 `gpu0` 推理兼容性抽检后，再次从 W&B `output.log` 确认训练继续推进到：
  - `548/20000 ... loss=0.4260`
- 这说明在 `step_0000540.pt` 产生后，正式训练仍在持续前进，没有在保存或抽检期间卡住

### 1. 权重继续更新核查：`step520 -> step540`

- 本轮直接比较：
  - `step_0000520.pt`
  - `step_0000540.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys520 = 1272`
- `keys540 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 199302.80548778316`
- `max_abs_diff = 0.001732584205456078`
- `max_abs_key = bundle.dit.base_model.model.blocks.18.object_cross_attn.o.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400`、`step400 -> step420`、`step420 -> step440`、`step440 -> step460`、`step460 -> step480`、`step480 -> step500`、`step500 -> step520`、`step520 -> step540` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 与上一轮相比，这次最大变化项直接落在：
  - `bundle.dit.base_model.model.blocks.18.object_cross_attn.o.lora_B.default.weight`
- 这说明 object 条件分支对应的 LoRA 可训练子空间也在持续真实更新，不只是文本 / 自注意力分支在变化
- 因而“主损失没有有效回传到 object-conditioned 主干”的担忧，在当前正式训练链路上没有得到证据支持

### 2. `step_0000540.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000540.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step540 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step540`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step540/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step540`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step520` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- `result.json` 中 `sample_debug.loss = 54165.3046875`
- 推理采样时打印出的张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 这继续和当前代码设计一致：
  - 训练与推理链路默认走 CoTracker 轨迹
  - object 条件保留 latent-time 维度后，展平成 `T_lat * K = 2 * 8 = 16` 个 object token 送入 Wan object cross-attn

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 到较慢 kernel 的提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条出现在推理采样路径中，属于 inference/no-grad 环境下的 checkpoint 包装提示，不代表正式训练图上的主损失梯度中断
- 兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000540.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 23:14:31 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 23:15:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 23:15:08 UTC`
- W&B `output.log` 尾部已经推进到：
  - `548/20000 ... loss=0.4260`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000540.pt`
- `step520 -> step540` 之间 trainable 权重继续大范围真实更新
- 且本轮最大变化项直接落在 object cross-attn 的 LoRA 权重上，进一步支持：
  - object 条件分支参与了有效优化
  - 主损失梯度并非只在无条件或文本条件子路径内循环
- `step_0000540.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - object-conditioned LoRA 子空间也在持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-23 23:15 UTC` 为止，连续闭环证据已进一步延长到 `step 540+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-23 23:24 UTC: phase 39, `step_0000560.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000560.pt`
- `step_0000560.pt` 文件时间与大小稳定为：
  - `2026-06-23 23:20:30 UTC`
  - `5533953209 bytes`
- 本轮持续观察中，训练先推进到：
  - `551/20000 ... loss=0.1112`
  - `558/20000 ... loss=1.9503`
- `step_0000560.pt` 生成后，训练继续推进到：
  - `561/20000 ... loss=1.5698`
- 这说明在 `step_0000560.pt` 产生前后，正式训练都在稳定前进，没有因为 checkpoint 保存或侧路验证而卡住

### 1. 权重继续更新核查：`step540 -> step560`

- 本轮直接比较：
  - `step_0000540.pt`
  - `step_0000560.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys540 = 1272`
- `keys560 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 159017.85407623055`
- `max_abs_diff = 0.001536418916657567`
- `max_abs_key = bundle.dit.base_model.model.object_embedding.2.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400`、`step400 -> step420`、`step420 -> step440`、`step440 -> step460`、`step460 -> step480`、`step480 -> step500`、`step500 -> step520`、`step520 -> step540`、`step540 -> step560` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 与上一轮相比，这次最大变化项进一步落在：
  - `bundle.dit.base_model.model.object_embedding.2.weight`
- 这比“object cross-attn 的 LoRA 在更新”更进一步，说明 object 条件入口投影本身也在持续被主损失优化
- 因而从参数更新角度看，当前正式训练链路已经同时覆盖：
  - object 条件入口 `object_embedding`
  - object-conditioned attention 分支 `object_cross_attn`
  - 主干 LoRA 子空间

### 2. `step_0000560.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000560.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step560 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step560`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step560/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step560`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step540` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- `result.json` 中 `sample_debug.loss = 84322.578125`
- 推理采样时打印出的张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 与 `step540` 相比，本轮 `object_context` 的统计范围继续变化：
  - `min = -7.2808`
  - `max = 6.4834`
  - `std = 1.8020`
- 这再次说明 object 条件通道不是静态常量分支，而是在随训练更新

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 到较慢 kernel 的提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍出现在 inference/no-grad 采样路径中，不代表正式训练图的主损失梯度中断
- 兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000560.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 23:24:00 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 23:24:17 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 23:24:22 UTC`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号
- W&B `output.log` 尾部在本轮读取时被新的 propagate/frame loading 日志覆盖，没有直接显示最新 step 行；但前一轮已明确看到：
  - `561/20000 ... loss=1.5698`
- 同时 W&B 文件时间仍在持续刷新，因此训练活性仍被直接证实

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000560.pt`
- `step540 -> step560` 之间 trainable 权重继续大范围真实更新
- 且本轮最大变化项直接落在 `object_embedding.2.weight`，进一步支持：
  - object 条件入口本身参与了有效优化
  - 主损失梯度不仅到达 object-conditioned attention，也到达 object token 的入口映射
- `step_0000560.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - object-conditioned LoRA 与 object embedding 入口都在持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-23 23:24 UTC` 为止，连续闭环证据已进一步延长到 `step 560+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-23 23:33 UTC: phase 40, `step_0000580.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000580.pt`
- `step_0000580.pt` 文件时间与大小稳定为：
  - `2026-06-23 23:29:14 UTC`
  - `5533953209 bytes`
- 本轮监控期间，训练先推进到：
  - `573/20000 ... loss=0.4624`
  - `579/20000 ... loss=0.2330`
- `step_0000580.pt` 生成后，训练继续推进到：
  - `582/20000 ... loss=1.5429`
  - `589/20000 ... loss=0.3014`
- 这说明 `step_0000580.pt` 产生前后，正式训练都在稳定前进，没有在保存点附近停住

### 1. 权重继续更新核查：`step560 -> step580`

- 本轮直接比较：
  - `step_0000560.pt`
  - `step_0000580.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys560 = 1272`
- `keys580 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 130124.10849526734`
- `max_abs_diff = 0.0015374226495623589`
- `max_abs_key = bundle.dit.base_model.model.object_embedding.2.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400`、`step400 -> step420`、`step420 -> step440`、`step440 -> step460`、`step460 -> step480`、`step480 -> step500`、`step500 -> step520`、`step520 -> step540`、`step540 -> step560`、`step560 -> step580` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项再次落在：
  - `bundle.dit.base_model.model.object_embedding.2.weight`
- 这和上一轮 `step540 -> step560` 的结果一致，说明 object 条件入口映射持续被主损失稳定优化，不是偶然跳变
- 因而“主损失没有真正穿过 object 条件入口”这一假设，在当前正式训练链路上进一步缺乏支持

### 2. `step_0000580.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000580.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step580 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step580`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step580/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step580`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step560` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- `result.json` 中 `sample_debug.loss = 101894.34375`
- 推理采样时打印出的张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 统计仍然呈现非静态更新：
  - `min = -7.3117`
  - `max = 6.2418`
  - `std = 1.6849`
- 这和前几轮一样，继续支持 object 条件通道处于活跃更新状态

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 到较慢 kernel 的提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000580.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 23:32:47 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 23:33:11 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 23:33:17 UTC`
- 最新 step 记录已明确推进到：
  - `589/20000 ... loss=0.3014`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000580.pt`
- `step560 -> step580` 之间 trainable 权重继续大范围真实更新
- 且本轮最大变化项再次落在 `object_embedding.2.weight`，进一步支持：
  - object 条件入口本身正在被主损失持续、稳定地优化
  - 当前主损失梯度不是只到达 LoRA 主干或 object cross-attn，而是持续穿过 object token 的入口映射
- `step_0000580.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - object-conditioned LoRA、object cross-attn 与 object embedding 入口都在持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-23 23:33 UTC` 为止，连续闭环证据已进一步延长到 `step 580+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-23 23:41 UTC: phase 41, `step_0000600.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000600.pt`
- `step_0000600.pt` 文件时间与大小稳定为：
  - `2026-06-23 23:37:58 UTC`
  - `5533953209 bytes`
- 本轮起始时已确认训练推进到：
  - `593/20000 ... loss=1.5820`
- 在等待 `step_0000600.pt` 生成期间，W&B 文件持续刷新，`gpu6,7` 利用率从短暂空档恢复到：
  - `gpu6 = 61%`
  - `gpu7 = 61%`
- 完成本轮侧路推理抽检后，再次从日志确认训练继续推进到：
  - `609/20000 ... loss=0.0492`
- 这说明 `step_0000600.pt` 产生前后，正式训练都在持续推进，没有因为 checkpoint 保存或侧路验证而中断

### 1. 权重继续更新核查：`step580 -> step600`

- 本轮直接比较：
  - `step_0000580.pt`
  - `step_0000600.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys580 = 1272`
- `keys600 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 154782.4956314942`
- `max_abs_diff = 0.0016044415533542633`
- `max_abs_key = bundle.dit.base_model.model.blocks.17.self_attn.k.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400`、`step400 -> step420`、`step420 -> step440`、`step440 -> step460`、`step460 -> step480`、`step480 -> step500`、`step500 -> step520`、`step520 -> step540`、`step540 -> step560`、`step560 -> step580`、`step580 -> step600` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项回到：
  - `bundle.dit.base_model.model.blocks.17.self_attn.k.lora_B.default.weight`
- 这说明当前优化并不是只停留在 object 条件入口或 object cross-attn，主干 self-attn 的 LoRA 子空间同样在持续被有效更新
- 结合前两轮 `step540 -> step560`、`step560 -> step580` 最大变化项都落在 `object_embedding.2.weight` 的事实，可以更完整地判断：
  - 主损失更新已经覆盖 object 条件入口
  - 主损失更新也覆盖 object-conditioned 分支
  - 主损失更新同时覆盖 DiT 主干 LoRA 子空间

### 2. `step_0000600.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000600.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step600 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step600`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step600/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step600`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step580` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- `result.json` 中 `sample_debug.loss = 114108.21875`
- 推理采样时打印出的张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 统计进一步扩大：
  - `min = -9.4517`
  - `max = 7.9093`
  - `std = 2.2462`
- 这继续支持 object 条件通道处于活跃更新状态，而非固定常量分支

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 到较慢 kernel 的提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000600.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 23:41:39 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 23:41:48 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 23:41:49 UTC`
- 最新 step 记录已明确推进到：
  - `609/20000 ... loss=0.0492`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000600.pt`
- `step580 -> step600` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项回到主干 self-attn LoRA 权重，与前两轮 object embedding 最大变化项结合，进一步支持：
  - 当前主损失梯度覆盖的是一条完整的多路径训练链路，而不是局限在某个局部支路
  - object 条件入口、object-conditioned 分支、DiT 主干 LoRA 均处在持续真实优化中
- `step_0000600.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - object-conditioned LoRA、object cross-attn、object embedding 入口与主干 LoRA 都在持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-23 23:41 UTC` 为止，连续闭环证据已进一步延长到 `step 600+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-23 23:50 UTC: phase 42, `step_0000620.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000620.pt`
- `step_0000620.pt` 文件时间与大小稳定为：
  - `2026-06-23 23:46:23 UTC`
  - `5533953209 bytes`
- 本轮开始时，训练日志中已明确看到：
  - `613/20000 ... loss=1.6616`
- 等待 `step_0000620.pt` 落盘期间，W&B 本地文件持续刷新，没有中断
- 本轮结束时，`rg` 抽取 step 行被新的中间日志冲掉，只留下最早的 `1/20000` 记录；但新的 `step_0000620.pt` 已经生成，且 W&B 文件仍在持续刷新，因此训练已经越过 `620` 并继续活跃，这一点由 checkpoint 生成与 W&B 心跳直接证明

### 1. 权重继续更新核查：`step600 -> step620`

- 本轮直接比较：
  - `step_0000600.pt`
  - `step_0000620.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys600 = 1272`
- `keys620 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 119363.96914979548`
- `max_abs_diff = 0.001849571242928505`
- `max_abs_key = bundle.dit.base_model.model.blocks.0.self_attn.q.lora_A.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400`、`step400 -> step420`、`step420 -> step440`、`step440 -> step460`、`step460 -> step480`、`step480 -> step500`、`step500 -> step520`、`step520 -> step540`、`step540 -> step560`、`step560 -> step580`、`step580 -> step600`、`step600 -> step620` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项落在：
  - `bundle.dit.base_model.model.blocks.0.self_attn.q.lora_A.default.weight`
- 这比前一轮最大变化落在较深层 `blocks.17.self_attn.k.lora_B` 更进一步，说明主损失更新不仅覆盖深层主干，也覆盖到更前层的 DiT 主干可训练子空间
- 结合前几轮 object embedding 与 object cross-attn 的证据，可以更完整地判断：
  - 主损失更新已经覆盖 object 条件入口
  - 主损失更新覆盖 object-conditioned 分支
  - 主损失更新覆盖 DiT 主干从前层到后层的 LoRA 子空间

### 2. `step_0000620.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000620.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step620 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step620`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step620/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step620`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step600` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- `result.json` 中 `sample_debug.loss = 86977.5546875`
- 推理采样时打印出的张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 统计进一步扩大：
  - `min = -10.2262`
  - `max = 8.8659`
  - `std = 2.5354`
- 这继续支持 object 条件通道处于活跃更新状态，而非固定常量分支

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 到较慢 kernel 的提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000620.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-23 23:49:21 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-23 23:50:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-23 23:50:11 UTC`
- 虽然 `rg` 在这一轮没有直接抽出新的 step 行，但：
  - `step_0000620.pt` 已成功生成
  - W&B 三类文件仍持续刷新
- 因而训练活性仍被直接证实，且活性证据强于单条 step 文本匹配
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000620.pt`
- `step600 -> step620` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项落在 `blocks.0.self_attn.q.lora_A.default.weight`，进一步支持：
  - 当前主损失梯度不仅到达 object 支路和深层主干，也到达更前层主干可训练权重
  - 当前优化路径是广覆盖、非局部、非偶然的
- `step_0000620.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - object-conditioned LoRA、object cross-attn、object embedding 入口与主干前后层 LoRA 都在持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-23 23:50 UTC` 为止，连续闭环证据已进一步延长到 `step 620+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 00:00 UTC: phase 43, `step_0000640.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000640.pt`
- `step_0000640.pt` 文件时间与大小稳定为：
  - `2026-06-23 23:55:03 UTC`
  - `5533953209 bytes`
- 本轮开始时，训练日志中已明确看到：
  - `633/20000 ... loss=1.0794`
- 等待 `step_0000640.pt` 落盘期间，W&B 本地文件持续刷新，没有中断
- 本轮结束时，`rg` 再次没有抽出新的有效 step 行，只剩最早的 `1/20000`；但新的 `step_0000640.pt` 已经生成，且 W&B 文件继续刷新到 `2026-06-24 00:00 UTC`，因此训练已经越过 `640` 并继续活跃，这一点由 checkpoint 与 W&B 心跳直接证明

### 1. 权重继续更新核查：`step620 -> step640`

- 本轮直接比较：
  - `step_0000620.pt`
  - `step_0000640.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys620 = 1272`
- `keys640 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 80459.35117940593`
- `max_abs_diff = 0.002235744148492813`
- `max_abs_key = bundle.dit.base_model.model.blocks.22.self_attn.q.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400`、`step400 -> step420`、`step420 -> step440`、`step440 -> step460`、`step460 -> step480`、`step480 -> step500`、`step500 -> step520`、`step520 -> step540`、`step540 -> step560`、`step560 -> step580`、`step580 -> step600`、`step600 -> step620`、`step620 -> step640` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项落在：
  - `bundle.dit.base_model.model.blocks.22.self_attn.q.lora_B.default.weight`
- 这说明在 `step600 -> step620` 看到前层主干 LoRA 明显更新之后，本轮又观察到中后层主干 LoRA 明显更新
- 与更早几轮 object embedding / object cross-attn 的证据结合，进一步说明：
  - 当前主损失更新路径覆盖 object 条件入口
  - 覆盖 object-conditioned 分支
  - 覆盖主干前层与中后层 LoRA 子空间

### 2. `step_0000640.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000640.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step640 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step640`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step640/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step640`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step620` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- `result.json` 中 `sample_debug.loss = 127944.0625`
- 推理采样时打印出的张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 统计继续扩大：
  - `min = -10.6557`
  - `max = 9.3829`
  - `std = 2.7584`
- 这继续支持 object 条件通道处于活跃更新状态，而非固定常量分支

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 到较慢 kernel 的提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000640.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 00:00:17 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 00:00:29 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 00:00:32 UTC`
- 虽然 `rg` 在这一轮没有直接抽出新的 step 行，但：
  - `step_0000640.pt` 已成功生成
  - W&B 三类文件仍持续刷新
- 因而训练活性仍被直接证实，且活性证据强于单条 step 文本匹配
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000640.pt`
- `step620 -> step640` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项落在 `blocks.22.self_attn.q.lora_B.default.weight`，进一步支持：
  - 当前主损失梯度在主干中后层仍保持有效
  - 结合前几轮前层主干、object embedding、object cross-attn 的证据，当前优化路径是广覆盖、稳定、持续的
- `step_0000640.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - object-conditioned LoRA、object cross-attn、object embedding 入口与主干前后层 LoRA 都在持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 00:00 UTC` 为止，连续闭环证据已进一步延长到 `step 640+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 00:08 UTC: phase 44, `step_0000660.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000660.pt`
- `step_0000660.pt` 文件时间与大小稳定为：
  - `2026-06-24 00:03:35 UTC`
  - `5533953209 bytes`
- 本轮开始时，训练日志中已明确看到：
  - `657/20000 ... loss=1.8768`
- 在短窗口等待后，已直接确认训练继续推进到：
  - `662/20000 ... loss=0.1223`
- 完成本轮侧路推理抽检后，再次确认训练继续推进到：
  - `670/20000 ... loss=0.8972`
- 这说明 `step_0000660.pt` 产生前后，正式训练都在持续推进，没有在保存点或推理抽检期间中断

### 1. 权重继续更新核查：`step640 -> step660`

- 本轮直接比较：
  - `step_0000640.pt`
  - `step_0000660.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys640 = 1272`
- `keys660 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 47896.29369144235`
- `max_abs_diff = 0.0029779693577438593`
- `max_abs_key = bundle.dit.base_model.model.blocks.12.self_attn.q.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40`、`step40 -> step60`、`step60 -> step80`、`step80 -> step100`、`step100 -> step120`、`step120 -> step140`、`step140 -> step160`、`step160 -> step180`、`step180 -> step200`、`step200 -> step220`、`step220 -> step240`、`step240 -> step260`、`step260 -> step280`、`step280 -> step300`、`step300 -> step320`、`step320 -> step340`、`step340 -> step360`、`step360 -> step380`、`step380 -> step400`、`step400 -> step420`、`step420 -> step440`、`step440 -> step460`、`step460 -> step480`、`step480 -> step500`、`step500 -> step520`、`step520 -> step540`、`step540 -> step560`、`step560 -> step580`、`step580 -> step600`、`step600 -> step620`、`step620 -> step640`、`step640 -> step660` 均显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项落在：
  - `bundle.dit.base_model.model.blocks.12.self_attn.q.lora_B.default.weight`
- 这与前面 `blocks.0`、`blocks.17`、`blocks.22` 的结果一起，形成了从前层、中层到后层主干 LoRA 全部被持续更新的连续证据
- 再结合更早几轮 object embedding / object cross-attn 的结果，可以更强地判断：
  - 主损失更新覆盖 object 条件入口
  - 覆盖 object-conditioned 分支
  - 覆盖主干前中后层 LoRA 子空间

### 2. `step_0000660.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000660.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step660 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step660`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step660/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step660`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step640` 的兼容性模式完全一致，说明 trainable-only checkpoint 仍可被当前推理脚本稳定接收

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `prep_debug.object_context = [1, 16, 4096]`
- `result.json` 中 `sample_debug.loss = 102366.796875`
- 推理采样时打印出的张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 统计保持在较高动态范围：
  - `min = -10.6172`
  - `max = 9.3327`
  - `std = 2.7243`
- 这继续支持 object 条件通道处于活跃更新状态，而非固定常量分支

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 到较慢 kernel 的提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000660.pt` 完全满足这一标准，没有出现新的结构不匹配回归

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 00:07:17 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 00:07:57 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 00:08:02 UTC`
- 最新 step 记录已明确推进到：
  - `670/20000 ... loss=0.8972`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000660.pt`
- `step640 -> step660` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项落在 `blocks.12.self_attn.q.lora_B.default.weight`，与前几轮不同深度的主干 LoRA 最大变化项一起，进一步支持：
  - 当前主损失梯度在主干前中后层都保持有效
  - 结合 object embedding 与 object cross-attn 的证据，当前优化路径是广覆盖、稳定、持续的
- `step_0000660.pt` 已被推理脚本成功加载并完成采样
- 当前训练链路继续满足之前已经建立的闭环判断：
  - loss 持续变化
  - 主损失有效梯度已被独立 backward 诊断直接证实
  - trainable 权重持续真实更新
  - object-conditioned LoRA、object cross-attn、object embedding 入口与主干前中后层 LoRA 都在持续真实更新
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 00:08 UTC` 为止，连续闭环证据已进一步延长到 `step 660+`
- 到当前观测点为止，没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 00:16 UTC: phase 45, `step_0000680.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000680.pt`
- 本轮核查开始时，上一轮已知训练日志位置已经推进到：
  - `689/20000 ... loss=0.1219`
- 在完成本轮侧路推理抽检后，再次确认训练继续推进到：
  - `692/20000 ... loss=1.6940`
- 这说明 `step_0000680.pt` 产生前后，正式训练持续前进，没有在保存点或侧路验证期间停住

### 1. 权重继续更新核查：`step660 -> step680`

- 本轮直接比较：
  - `step_0000660.pt`
  - `step_0000680.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys660 = 1272`
- `keys680 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 62399.24379369919`
- `max_abs_diff = 0.003929849248379469`
- `max_abs_key = bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step660 -> step680` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项落在：
  - `bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`
- 这进一步补强了先前已经看到的 `blocks.0`、`blocks.12`、`blocks.17`、`blocks.22` 等不同深度主干 LoRA 更新证据
- 当前最稳妥的解释仍然是：
  - 主损失更新没有塌成“只动 object 入口”
  - 也没有塌成“只动某一层 LoRA”
  - 而是在 object 条件入口、object-conditioned 分支和主干 DiT 前中后层之间持续形成广覆盖更新

### 2. `step_0000680.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000680.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step680 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step680`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step680/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step680`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step660` 的兼容性模式完全一致，说明当前 checkpoint 结构没有出现回归

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `result.json` 中 `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `result.json` 中 `prep_debug.object_context = [1, 16, 4096]`
- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 数值统计为：
  - `min = -11.0551`
  - `max = 9.9757`
  - `std = 2.9405`
- `sample_debug.loss = 100757.234375`
- 这些结果继续支持：
  - object 条件通道在推理侧是活跃的、非常量的
  - 当前 `track_source = cotracker` 的训练设计确实被推理脚本沿同一路径接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000680.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 00:15:32 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 00:15:47 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 00:15:59 UTC`
- 最新 step 记录已明确推进到：
  - `692/20000 ... loss=1.6940`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000680.pt`
- `step660 -> step680` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项落在 `blocks.2.self_attn.k.lora_B.default.weight`，继续补强“主干前中后层 LoRA 都在持续更新”的证据链
- `step_0000680.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 00:16 UTC` 为止，连续闭环证据已进一步延长到 `step 680+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 00:24 UTC: phase 46, `step_0000700.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000700.pt`
- `step_0000700.pt` 文件时间与大小稳定为：
  - `2026-06-24 00:20:40 UTC`
  - `5533953209 bytes`
- 本轮开始时，训练日志已经明确推进到：
  - `700/20000 ... loss=0.1852`
- 完成本轮侧路推理抽检后，再次确认训练继续推进到：
  - `707/20000 ... loss=1.5831`
- 这说明 `step_0000700.pt` 生成前后，正式训练仍在持续前进，没有在保存点或抽检期间停住

### 1. 权重继续更新核查：`step680 -> step700`

- 本轮直接比较：
  - `step_0000680.pt`
  - `step_0000700.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys680 = 1272`
- `keys700 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 51495.856478817295`
- `max_abs_diff = 0.005619422532618046`
- `max_abs_key = bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step680 -> step700` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项仍然落在：
  - `bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`
- 结合前面落在 `blocks.0`、`blocks.12`、`blocks.17`、`blocks.22` 的结果，可以继续支持：
  - 主损失更新不是只停留在 object 入口或某一层 LoRA
  - object 条件入口、object-conditioned 分支以及主干 DiT 多层 LoRA 仍在持续形成真实更新

### 2. `step_0000700.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000700.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step700 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step700`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step700/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step700`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step680` 的兼容性模式完全一致，说明当前 checkpoint 结构仍然稳定，没有出现新回归

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `result.json` 中 `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `result.json` 中 `prep_debug.object_context = [1, 16, 4096]`
- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 数值统计为：
  - `min = -10.8617`
  - `max = 9.8173`
  - `std = 2.8447`
- `sample_debug.loss = 97231.484375`
- 这些结果继续支持：
  - object 条件通道在推理侧保持活跃、非常量
  - 当前 `track_source = cotracker` 的训练设计仍被推理脚本沿同一路径正确接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000700.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 00:22:47 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 00:23:43 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 00:23:47 UTC`
- 最新 step 记录已明确推进到：
  - `707/20000 ... loss=1.5831`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000700.pt`
- `step680 -> step700` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项依旧落在 `blocks.2.self_attn.k.lora_B.default.weight`，继续支持“主干 LoRA 多层持续更新”的判断
- `step_0000700.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 00:24 UTC` 为止，连续闭环证据已进一步延长到 `step 700+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 00:32 UTC: phase 47, `step_0000720.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000720.pt`
- `step_0000720.pt` 文件时间与大小稳定为：
  - `2026-06-24 00:29:24 UTC`
  - `5533953209 bytes`
- 本轮开始时，训练日志已经明确推进到：
  - `720/20000`
- 完成本轮侧路推理抽检后，再次确认训练继续推进到：
  - `726/20000`
- 这说明 `step_0000720.pt` 生成前后，正式训练仍在持续前进，没有在保存点或抽检期间停住

### 1. 权重继续更新核查：`step700 -> step720`

- 本轮直接比较：
  - `step_0000700.pt`
  - `step_0000720.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys700 = 1272`
- `keys720 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 78118.47006374807`
- `max_abs_diff = 0.006666284985840321`
- `max_abs_key = bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step700 -> step720` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项仍然落在：
  - `bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`
- 结合前面落在 `blocks.0`、`blocks.12`、`blocks.17`、`blocks.22` 的结果，可以继续支持：
  - 主损失更新不是只停留在 object 入口或某一层 LoRA
  - object 条件入口、object-conditioned 分支以及主干 DiT 多层 LoRA 仍在持续形成真实更新

### 2. `step_0000720.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000720.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step720 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step720`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step720/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step720`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step700` 的兼容性模式完全一致，说明当前 checkpoint 结构仍然稳定，没有出现新回归

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `result.json` 中 `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `result.json` 中 `prep_debug.object_context = [1, 16, 4096]`
- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 数值统计为：
  - `min = -10.9131`
  - `max = 9.8656`
  - `std = 2.8207`
- `sample_debug.loss = 87025.765625`
- 这些结果继续支持：
  - object 条件通道在推理侧保持活跃、非常量
  - 当前 `track_source = cotracker` 的训练设计仍被推理脚本沿同一路径正确接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000720.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 00:31:47 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 00:32:03 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 00:32:04 UTC`
- 最新 step 记录已明确推进到：
  - `726/20000`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000720.pt`
- `step700 -> step720` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项依旧落在 `blocks.2.self_attn.k.lora_B.default.weight`，继续支持“主干 LoRA 多层持续更新”的判断
- `step_0000720.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 00:32 UTC` 为止，连续闭环证据已进一步延长到 `step 720+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 00:40 UTC: phase 48, `step_0000740.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000740.pt`
- `step_0000740.pt` 文件时间与大小稳定为：
  - `2026-06-24 00:37:53 UTC`
  - `5533953209 bytes`
- 本轮开始时，训练日志已经明确推进到：
  - `740/20000`
- 完成本轮侧路推理抽检后，再次确认训练继续推进到：
  - `745/20000`
- 这说明 `step_0000740.pt` 生成前后，正式训练仍在持续前进，没有在保存点或抽检期间停住

### 1. 权重继续更新核查：`step720 -> step740`

- 本轮直接比较：
  - `step_0000720.pt`
  - `step_0000740.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys720 = 1272`
- `keys740 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 83239.0258610479`
- `max_abs_diff = 0.0059608882293105125`
- `max_abs_key = bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step720 -> step740` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项仍然落在：
  - `bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`
- 结合前面落在 `blocks.0`、`blocks.12`、`blocks.17`、`blocks.22` 的结果，可以继续支持：
  - 主损失更新不是只停留在 object 入口或某一层 LoRA
  - object 条件入口、object-conditioned 分支以及主干 DiT 多层 LoRA 仍在持续形成真实更新

### 2. `step_0000740.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000740.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step740 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step740`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step740/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step740`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step720` 的兼容性模式完全一致，说明当前 checkpoint 结构仍然稳定，没有出现新回归

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `result.json` 中 `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `result.json` 中 `prep_debug.object_context = [1, 16, 4096]`
- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 数值统计为：
  - `min = -10.6344`
  - `max = 9.5415`
  - `std = 2.6682`
- `sample_debug.loss = 122329.1484375`
- 这些结果继续支持：
  - object 条件通道在推理侧保持活跃、非常量
  - 当前 `track_source = cotracker` 的训练设计仍被推理脚本沿同一路径正确接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000740.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 00:40:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 00:40:12 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 00:40:17 UTC`
- 最新 step 记录已明确推进到：
  - `745/20000`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000740.pt`
- `step720 -> step740` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项依旧落在 `blocks.2.self_attn.k.lora_B.default.weight`，继续支持“主干 LoRA 多层持续更新”的判断
- `step_0000740.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 00:40 UTC` 为止，连续闭环证据已进一步延长到 `step 740+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 00:51 UTC: phase 49, `step_0000760.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000760.pt`
- `step_0000760.pt` 文件时间与大小稳定为：
  - `2026-06-24 00:47:09 UTC`
  - `5533953209 bytes`
- 本轮开始时，训练日志已经明确推进到：
  - `760/20000`
- 完成本轮侧路推理抽检后，继续确认训练仍在前进：
  - 早先抽检时已推进到 `762/20000`
  - 当前复核时已进一步推进到 `769/20000`
- 这说明 `step_0000760.pt` 生成前后，正式训练没有在保存点或抽检期间停住

### 1. 权重继续更新核查：`step740 -> step760`

- 本轮直接比较：
  - `step_0000740.pt`
  - `step_0000760.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys740 = 1272`
- `keys760 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 90340.58774209628`
- `max_abs_diff = 0.004190660081803799`
- `max_abs_key = bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step740 -> step760` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项仍然落在：
  - `bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`
- 结合前面落在 `blocks.0`、`blocks.12`、`blocks.17`、`blocks.22` 与当前 `blocks.2` 的结果，可以继续支持：
  - 主损失更新不是只停留在 object 入口或某一层 LoRA
  - object 条件入口、object-conditioned 分支以及主干 DiT 多层 LoRA 仍在持续形成真实更新

### 2. `step_0000760.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000760.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step760 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step760`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step760/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step760`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step740` 的兼容性模式完全一致，说明当前 checkpoint 结构仍然稳定，没有出现新回归

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `result.json` 中 `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `result.json` 中 `prep_debug.object_context = [1, 16, 4096]`
- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 数值统计为：
  - `min = -10.8679`
  - `max = 9.5564`
  - `std = 2.7157`
- `sample_debug.loss = 126325.3828125`
- 这些结果继续支持：
  - object 条件通道在推理侧保持活跃、非常量
  - 当前 `track_source = cotracker` 的训练设计仍被推理脚本沿同一路径正确接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000760.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 00:50:47 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 00:51:03 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 00:51:14 UTC`
- 最新 step 记录已明确推进到：
  - `769/20000`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000760.pt`
- `step740 -> step760` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项依旧落在 `blocks.2.self_attn.k.lora_B.default.weight`，继续支持“主干 LoRA 多层持续更新”的判断
- `step_0000760.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 00:51 UTC` 为止，连续闭环证据已进一步延长到 `step 760+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 01:00 UTC: phase 50, `step_0000780.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000780.pt`
- `step_0000780.pt` 文件时间与大小稳定为：
  - `2026-06-24 00:56:19 UTC`
  - `5533953209 bytes`
- 本轮开始时，训练日志已经明确推进到：
  - `780/20000`
- 完成本轮侧路推理抽检后，继续确认训练仍在前进：
  - 当前复核时已进一步推进到 `789/20000`
- 这说明 `step_0000780.pt` 生成前后，正式训练没有在保存点或抽检期间停住

### 1. 权重继续更新核查：`step760 -> step780`

- 本轮直接比较：
  - `step_0000760.pt`
  - `step_0000780.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys760 = 1272`
- `keys780 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 82808.84362279624`
- `max_abs_diff = 0.004333448596298695`
- `max_abs_key = bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step760 -> step780` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项仍然落在：
  - `bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`
- 结合前面跨多轮落在 `blocks.0`、`blocks.12`、`blocks.17`、`blocks.22` 与当前 `blocks.2` 的结果，可以继续支持：
  - 主损失更新不是只停留在 object 入口或某一层 LoRA
  - object 条件入口、object-conditioned 分支以及主干 DiT 多层 LoRA 仍在持续形成真实更新

### 2. `step_0000780.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000780.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step780 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step780`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step780/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step780`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step760` 的兼容性模式完全一致，说明当前 checkpoint 结构仍然稳定，没有出现新回归

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `result.json` 中 `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `result.json` 中 `prep_debug.object_context = [1, 16, 4096]`
- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 数值统计为：
  - `min = -11.0757`
  - `max = 9.6477`
  - `std = 2.7558`
- `sample_debug.loss = 107758.96875`
- 这些结果继续支持：
  - object 条件通道在推理侧保持活跃、非常量
  - 当前 `track_source = cotracker` 的训练设计仍被推理脚本沿同一路径正确接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000780.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 00:59:34 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 01:00:01 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 01:00:03 UTC`
- 最新 step 记录已明确推进到：
  - `789/20000`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000780.pt`
- `step760 -> step780` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项依旧落在 `blocks.2.self_attn.k.lora_B.default.weight`，继续支持“主干 LoRA 多层持续更新”的判断
- `step_0000780.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 01:00 UTC` 为止，连续闭环证据已进一步延长到 `step 780+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 01:08 UTC: phase 51, `step_0000800.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000800.pt`
- 首次观察到 `step_0000800.pt` 时，文件大小短暂显示为：
  - `2278243712 bytes`
- 继续复核约 `10s` 后，文件已稳定回到完整大小：
  - `2026-06-24 01:04:52 UTC`
  - `5533953209 bytes`
- 随后只读加载 checkpoint 也成功通过：
  - `load_ok = true`
  - `model_key_count = 1272`
- 这说明本轮出现的“文件明显偏小”只是保存过程中的瞬时中间态，不是新的损坏 checkpoint 现象
- 本轮开始时，训练日志已经明确推进到：
  - `800/20000`
- 完成本轮侧路推理抽检后，继续确认训练仍在前进：
  - 当前复核时已进一步推进到 `809/20000`
- 这说明 `step_0000800.pt` 生成前后，正式训练没有在保存点或抽检期间停住

### 1. 权重继续更新核查：`step780 -> step800`

- 本轮直接比较：
  - `step_0000780.pt`
  - `step_0000800.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys780 = 1272`
- `keys800 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 62173.61862022744`
- `max_abs_diff = 0.002634427510201931`
- `max_abs_key = bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step780 -> step800` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项仍然落在：
  - `bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`
- 这继续支持：
  - 主损失更新不是只停留在 object 入口或某一层 LoRA
  - object 条件入口、object-conditioned 分支以及主干 DiT 多层 LoRA 仍在持续形成真实更新

### 2. `step_0000800.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000800.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step800 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step800`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step800/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step800`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step780` 的兼容性模式完全一致，说明当前 checkpoint 结构仍然稳定，没有出现新回归

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `result.json` 中 `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `result.json` 中 `prep_debug.object_context = [1, 16, 4096]`
- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 数值统计为：
  - `min = -11.8374`
  - `max = 10.1017`
  - `std = 2.9287`
- `sample_debug.loss = 91544.875`
- 这些结果继续支持：
  - object 条件通道在推理侧保持活跃、非常量
  - 当前 `track_source = cotracker` 的训练设计仍被推理脚本沿同一路径正确接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000800.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 01:08:02 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 01:08:33 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 01:08:38 UTC`
- 最新 step 记录已明确推进到：
  - `809/20000`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000800.pt`
- `step780 -> step800` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项依旧落在 `blocks.2.self_attn.k.lora_B.default.weight`，继续支持“主干 LoRA 多层持续更新”的判断
- `step_0000800.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 01:08 UTC` 为止，连续闭环证据已进一步延长到 `step 800+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 01:17 UTC: phase 52, `step_0000820.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000820.pt`
- `step_0000820.pt` 文件时间与大小稳定为：
  - `2026-06-24 01:13:25 UTC`
  - `5533953209 bytes`
- 本轮开始时，训练日志已经明确推进到：
  - `820/20000`
- 完成本轮侧路推理抽检后，继续确认训练仍在前进：
  - 当前复核时已进一步推进到 `830/20000`
- 这说明 `step_0000820.pt` 生成前后，正式训练没有在保存点或抽检期间停住

### 1. 权重继续更新核查：`step800 -> step820`

- 本轮直接比较：
  - `step_0000800.pt`
  - `step_0000820.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys800 = 1272`
- `keys820 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 78740.71455280646`
- `max_abs_diff = 0.003492050338536501`
- `max_abs_key = bundle.dit.base_model.model.blocks.11.self_attn.q.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step800 -> step820` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项落在：
  - `bundle.dit.base_model.model.blocks.11.self_attn.q.lora_B.default.weight`
- 这相对前面多轮经常落在 `blocks.2.self_attn.k.lora_B.default.weight` 的结果，进一步支持：
  - 更新并没有塌缩到单一 LoRA 位置
  - 主干 DiT 多层 LoRA 仍在参与主损失优化
  - object 条件入口、object-conditioned 分支以及主干 DiT 仍在持续形成真实更新

### 2. `step_0000820.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000820.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step820 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step820`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step820/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step820`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step800` 的兼容性模式完全一致，说明当前 checkpoint 结构仍然稳定，没有出现新回归

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `result.json` 中 `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `result.json` 中 `prep_debug.object_context = [1, 16, 4096]`
- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 数值统计为：
  - `min = -12.1802`
  - `max = 10.1431`
  - `std = 2.9874`
- `sample_debug.loss = 132149.03125`
- 这些结果继续支持：
  - object 条件通道在推理侧保持活跃、非常量
  - 当前 `track_source = cotracker` 的训练设计仍被推理脚本沿同一路径正确接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000820.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 01:16:32 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 01:17:18 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 01:17:27 UTC`
- 最新 step 记录已明确推进到：
  - `830/20000`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000820.pt`
- `step800 -> step820` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项切换到 `blocks.11.self_attn.q.lora_B.default.weight`，继续支持“主干 LoRA 多层持续更新，而不是卡死在单一位置”的判断
- `step_0000820.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 01:17 UTC` 为止，连续闭环证据已进一步延长到 `step 820+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 01:25 UTC: phase 53, `step_0000840.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000840.pt`
- `step_0000840.pt` 文件时间与大小稳定为：
  - `2026-06-24 01:22:13 UTC`
  - `5533953209 bytes`
- 本轮开始时，训练日志已经明确推进到：
  - `840/20000`
- 完成本轮侧路推理抽检后，继续确认训练仍在前进：
  - 当前复核时已进一步推进到 `848/20000`
- 这说明 `step_0000840.pt` 生成前后，正式训练没有在保存点或抽检期间停住

### 1. 权重继续更新核查：`step820 -> step840`

- 本轮直接比较：
  - `step_0000820.pt`
  - `step_0000840.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys820 = 1272`
- `keys840 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 80464.7339969736`
- `max_abs_diff = 0.0031177159398794174`
- `max_abs_key = bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step820 -> step840` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项回到：
  - `bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`
- 结合上一轮落在 `blocks.11.self_attn.q.lora_B.default.weight` 的结果，可以继续支持：
  - 更新并没有塌缩到单一 LoRA 位置
  - 主干 DiT 多层 LoRA 仍在持续参与主损失优化
  - object 条件入口、object-conditioned 分支以及主干 DiT 仍在持续形成真实更新

### 2. `step_0000840.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000840.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step840 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step840`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step840/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step840`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step820` 的兼容性模式完全一致，说明当前 checkpoint 结构仍然稳定，没有出现新回归

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `result.json` 中 `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `result.json` 中 `prep_debug.object_context = [1, 16, 4096]`
- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 数值统计为：
  - `min = -13.0025`
  - `max = 10.6485`
  - `std = 3.1841`
- `sample_debug.loss = 93359.4375`
- 这些结果继续支持：
  - object 条件通道在推理侧保持活跃、非常量
  - 当前 `track_source = cotracker` 的训练设计仍被推理脚本沿同一路径正确接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000840.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 01:25:44 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 01:25:45 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 01:25:47 UTC`
- 最新 step 记录已明确推进到：
  - `848/20000`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000840.pt`
- `step820 -> step840` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项回到 `blocks.2.self_attn.k.lora_B.default.weight`，结合上一轮 `blocks.11` 的结果，继续支持“主干 LoRA 多层持续更新，而不是卡死在单一位置”的判断
- `step_0000840.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 01:25 UTC` 为止，连续闭环证据已进一步延长到 `step 840+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 01:35 UTC: phase 54, `step_0000860.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000860.pt`
- 首次观察到 `step_0000860.pt` 时，文件大小短暂显示为：
  - `4199772160 bytes`
- 继续复核约 `1s` 后，文件已稳定回到完整大小：
  - `2026-06-24 01:31:03 UTC`
  - `5533953209 bytes`
- 随后只读加载 checkpoint 也成功通过：
  - `load_ok = true`
  - `model_key_count = 1272`
- 这说明本轮出现的“文件偏小”依旧只是保存过程中的瞬时中间态，不是新的损坏 checkpoint 现象
- 本轮开始时，训练日志已经明确推进到：
  - `860/20000`
- 完成本轮侧路推理抽检后，继续确认训练仍在前进：
  - 当前复核时已进一步推进到 `869/20000`
- 这说明 `step_0000860.pt` 生成前后，正式训练没有在保存点或抽检期间停住

### 1. 权重继续更新核查：`step840 -> step860`

- 本轮直接比较：
  - `step_0000840.pt`
  - `step_0000860.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys840 = 1272`
- `keys860 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 86884.65277267306`
- `max_abs_diff = 0.005202052649110556`
- `max_abs_key = bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step840 -> step860` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项落在：
  - `bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`
- 结合前面 `blocks.11` 和本轮 `blocks.2` 的结果，可以继续支持：
  - 更新并没有塌缩到单一 LoRA 位置
  - 主干 DiT 多层 LoRA 仍在持续参与主损失优化
  - object 条件入口、object-conditioned 分支以及主干 DiT 仍在持续形成真实更新

### 2. `step_0000860.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000860.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step860 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step860`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step860/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step860`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step840` 的兼容性模式完全一致，说明当前 checkpoint 结构仍然稳定，没有出现新回归

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `result.json` 中 `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `result.json` 中 `prep_debug.object_context = [1, 16, 4096]`
- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 数值统计为：
  - `min = -13.5953`
  - `max = 11.0559`
  - `std = 3.3723`
- `sample_debug.loss = 167767.71875`
- 这些结果继续支持：
  - object 条件通道在推理侧保持活跃、非常量
  - 当前 `track_source = cotracker` 的训练设计仍被推理脚本沿同一路径正确接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000860.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 01:34:45 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 01:35:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 01:35:12 UTC`
- 最新 step 记录已明确推进到：
  - `869/20000`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000860.pt`
- `step840 -> step860` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项仍落在 `blocks.2.self_attn.k.lora_B.default.weight`，结合前面 `blocks.11` 的结果，继续支持“主干 LoRA 多层持续更新，而不是卡死在单一位置”的判断
- `step_0000860.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 01:35 UTC` 为止，连续闭环证据已进一步延长到 `step 860+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 01:43 UTC: phase 55, `step_0000880.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000880.pt`
- 首次观察到 `step_0000880.pt` 时，文件大小短暂显示为：
  - `714141696 bytes`
- 继续复核约 `1s` 后，文件已稳定回到完整大小：
  - `2026-06-24 01:39:42 UTC`
  - `5533953209 bytes`
- 随后只读加载 checkpoint 也成功通过：
  - `load_ok = true`
  - `model_key_count = 1272`
- 这说明本轮出现的“文件特别偏小”依旧只是保存过程中的瞬时中间态，不是新的损坏 checkpoint 现象
- 本轮开始时，训练日志已经明确推进到：
  - `880/20000`
- 完成本轮侧路推理抽检后，继续确认训练仍在前进：
  - 当前复核时已进一步推进到 `889/20000`
- 这说明 `step_0000880.pt` 生成前后，正式训练没有在保存点或抽检期间停住

### 1. 权重继续更新核查：`step860 -> step880`

- 本轮直接比较：
  - `step_0000860.pt`
  - `step_0000880.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys860 = 1272`
- `keys880 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 73356.56576541858`
- `max_abs_diff = 0.002141970209777355`
- `max_abs_key = bundle.dit.base_model.model.blocks.14.object_cross_attn.q.base_layer.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step860 -> step880` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项切换到：
  - `bundle.dit.base_model.model.blocks.14.object_cross_attn.q.base_layer.weight`
- 这相对前面多轮落在 `blocks.2.self_attn.k.lora_B.default.weight` 和 `blocks.11.self_attn.q.lora_B.default.weight` 的结果，进一步支持：
  - 更新并没有塌缩到单一层或单一条件入口
  - 主干 DiT 和 object-conditioned 分支都仍在参与主损失优化
  - object 条件入口、object-conditioned 分支以及主干 DiT 多层 LoRA 仍在持续形成真实更新

### 2. `step_0000880.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000880.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step880 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step880`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step880/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step880`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- 这与前面 `step20` 到 `step860` 的兼容性模式完全一致，说明当前 checkpoint 结构仍然稳定，没有出现新回归

### 3. 本轮附带观察到的推理侧结构信息

- `result.json` 中 `prep_debug.track_source = cotracker`
- `result.json` 中 `prep_debug.object_latent_tokens = [1, 2, 8, 4096]`
- `result.json` 中 `prep_debug.object_context = [1, 16, 4096]`
- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 本轮 `object_context` 数值统计为：
  - `min = -13.5455`
  - `max = 11.1362`
  - `std = 3.3578`
- `sample_debug.loss = 102898.578125`
- 这些结果继续支持：
  - object 条件通道在推理侧保持活跃、非常量
  - 当前 `track_source = cotracker` 的训练设计仍被推理脚本沿同一路径正确接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000880.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 01:43:02 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 01:43:47 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 01:43:56 UTC`
- 最新 step 记录已明确推进到：
  - `889/20000`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000880.pt`
- `step860 -> step880` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项切换到 `blocks.14.object_cross_attn.q.base_layer.weight`，进一步支持“主干 LoRA 与 object-conditioned 分支都在持续更新，而不是卡死在单一位置”的判断
- `step_0000880.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 01:43 UTC` 为止，连续闭环证据已进一步延长到 `step 880+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 01:55 UTC: phase 56, `step_0000900.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000900.pt`
- 首次观察到 `step_0000900.pt` 时，文件大小短暂显示为：
  - `2472488960 bytes`
- 继续复核几秒后，文件已稳定回到完整大小：
  - `2026-06-24 01:48:48 UTC`
  - `5533953209 bytes`
- 随后只读加载 checkpoint 也成功通过：
  - `load_ok = true`
  - `top_keys = ["step", "model"]`
  - `model_key_count = 1272`
- 这说明本轮出现的“小文件”仍然只是保存过程中的瞬时中间态，不是新的损坏 checkpoint 现象
- 本轮落盘时，训练日志已经明确推进到：
  - `900/20000`
- 完成本轮侧路推理抽检后继续复核训练日志，训练仍在前进：
  - 当前复核时已进一步推进到 `915/20000`
- 这说明 `step_0000900.pt` 生成前后，正式训练没有在保存点或抽检期间停住

### 1. 权重继续更新核查：`step880 -> step900`

- 本轮直接比较：
  - `step_0000880.pt`
  - `step_0000900.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys880 = 1272`
- `keys900 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 81402.48943752854`
- `max_abs_diff = 0.002498270943760872`
- `max_abs_key = bundle.dit.base_model.model.blocks.14.object_cross_attn.q.base_layer.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step880 -> step900` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项继续落在：
  - `bundle.dit.base_model.model.blocks.14.object_cross_attn.q.base_layer.weight`
- 结合上一轮 `step860 -> step880` 的结果，本轮形成了连续两轮相同的最大变化项，这进一步支持：
  - object-conditioned 分支本身在持续随主损失更新
  - 更新并没有塌缩到单一 LoRA 小块，也没有退化为“只有主干某个固定层在动”
  - object 条件入口、object-conditioned 分支以及主干 DiT 多层 LoRA 仍在持续形成真实更新

### 2. `step_0000900.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000900.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step900 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step900`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step900/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step900`
- `result.json` 中 `load_state_missing` 记录：
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
- `result.json` 中 `prep_debug` 关键字段：
  - `track_source = cotracker`
  - `object_latent_tokens = [1, 2, 8, 4096]`
  - `object_context = [1, 16, 4096]`
- `result.json` 中 `sample_debug.loss = 118224.671875`
- 这与前面 `step20` 到 `step880` 的兼容性模式完全一致，说明当前 checkpoint 结构仍然稳定，没有出现新回归

### 3. 本轮附带观察到的推理侧结构信息

- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 这些结果继续支持：
  - object 条件通道在推理侧保持活跃、非常量
  - 当前 `track_source = cotracker` 的训练设计仍被推理脚本沿同一路径正确接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
- 最后一条仍属于 inference/no-grad 采样路径提示，不代表正式训练图上的主损失梯度中断
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000900.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 01:54:32 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 01:55:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 01:55:10 UTC`
- 最新 step 记录已明确推进到：
  - `915/20000`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000900.pt`
- `step880 -> step900` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项继续落在 `blocks.14.object_cross_attn.q.base_layer.weight`，并且连续两轮都落在同一个 object-conditioned 关键权重上，这进一步支持“object 分支不是摆设，而是在持续随主损失共同更新”的判断
- `step_0000900.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 01:55 UTC` 为止，连续闭环证据已进一步延长到 `step 900+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 01:59 UTC: phase 57, `step_0000920.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000920.pt`
- 首次观察到 `step_0000920.pt` 时，文件大小短暂显示为：
  - `1617121280 bytes`
- 继续复核约 `4s` 后，文件已稳定回到完整大小：
  - `2026-06-24 01:57:31 UTC`
  - `5533953209 bytes`
- 随后只读加载 checkpoint 也成功通过：
  - `load_ok = true`
  - `top_keys = ["step", "model"]`
  - `model_key_count = 1272`
- 这说明本轮出现的“小文件”仍然只是保存过程中的瞬时中间态，不是新的损坏 checkpoint 现象
- 本轮落盘时，训练日志已经明确推进到：
  - `920/20000`

### 1. 权重继续更新核查：`step900 -> step920`

- 本轮直接比较：
  - `step_0000900.pt`
  - `step_0000920.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys900 = 1272`
- `keys920 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 163748.9624522907`
- `max_abs_diff = 0.0032938262447714806`
- `max_abs_key = bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step900 -> step920` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项从前两轮的 object-conditioned 分支切回：
  - `bundle.dit.base_model.model.blocks.2.self_attn.k.lora_B.default.weight`
- 这进一步支持：
  - 更新并没有塌缩到 object 分支或主干的单一点位
  - 主干 DiT LoRA 与 object-conditioned 分支是在轮换地主导单轮最大变化项
  - 当前主损失回传仍然在驱动较大范围的真实参数更新

### 2. `step_0000920.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000920.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step920 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step920`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step920/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step920`
- `result.json` / 推理输出中的关键字段继续保持稳定：
  - `track_source = cotracker`
  - `object_latent_tokens = [1, 2, 8, 4096]`
  - `object_context = [1, 16, 4096]`
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
  - `sample_debug.loss = 103395.671875`
- 本轮采样侧关键张量统计也保持合理、非退化：
  - `object_context std = 3.0327`
  - `pred_step_0 std = 0.8566`
  - `pred_step_1 std = 0.8019`
- 这与前面 `step20` 到 `step900` 的兼容性模式完全一致，说明当前 checkpoint 结构仍然稳定，没有出现新回归

### 3. 本轮附带观察到的推理侧结构信息

- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 这些结果继续支持：
  - object 条件通道在推理侧保持活跃、非常量
  - 当前 `track_source = cotracker` 的训练设计仍被推理脚本沿同一路径正确接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
  - `npz_io.py` 关于 non-writable buffer 的 UserWarning
- 这些都没有阻止：
  - `trainer constructed`
  - `checkpoint loaded`
  - `sampling finished`
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000920.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 01:58:47 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 01:59:02 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 01:59:04 UTC`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000920.pt`
- `step900 -> step920` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项切回主干 LoRA `blocks.2.self_attn.k.lora_B.default.weight`，结合前两轮 object-conditioned 分支的最大变化项，继续支持“更新在主干与 object 分支之间共同流动，而不是卡死在单一位置”的判断
- `step_0000920.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 01:59 UTC` 为止，连续闭环证据已进一步延长到 `step 920+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 02:08 UTC: phase 58, `step_0000940.pt` 已生成并完成推理兼容性核查

### 0. 本轮先验现象

- checkpoint 目录已经新增：
  - `step_0000940.pt`
- 本轮首次复核时，`step_0000940.pt` 已经是完整大小：
  - `2026-06-24 02:05:28 UTC`
  - `5533953209 bytes`
- 随后只读加载 checkpoint 也成功通过：
  - `load_ok = true`
  - `top_keys = ["step", "model"]`
  - `model_key_count = 1272`
- 本轮落盘时，训练日志已经明确推进到：
  - `940/20000`
- 继续复核训练日志与 W&B 刷新状态时，正式训练仍在前进：
  - 当前侧路推理抽检期间已进一步推进到 `943/20000`

### 1. 权重继续更新核查：`step920 -> step940`

- 本轮直接比较：
  - `step_0000920.pt`
  - `step_0000940.pt`
- 继续只比较 checkpoint 中 trainable `model` state

### 差分结果

- `keys920 = 1272`
- `keys940 = 1272`
- `common = 1272`
- `changed_tensors = 1270`
- `total_abs_diff = 104176.42567288876`
- `max_abs_diff = 0.0031138930935412645`
- `max_abs_key = bundle.dit.base_model.model.blocks.9.self_attn.q.lora_B.default.weight`

### 对差分结果的解释

- 到这一轮为止，`step20 -> step40` 一直到 `step920 -> step940` 的所有闭环抽检都显示：
  - `1270/1272` 个 trainable tensor 持续发生变化
- 本轮最大变化项继续落在主干 DiT LoRA：
  - `bundle.dit.base_model.model.blocks.9.self_attn.q.lora_B.default.weight`
- 结合前几轮最大变化项分别落在：
  - `blocks.14.object_cross_attn.q.base_layer.weight`
  - `blocks.2.self_attn.k.lora_B.default.weight`
  - `blocks.9.self_attn.q.lora_B.default.weight`
- 这进一步支持：
  - object-conditioned 分支与主干 LoRA 都在轮换地主导单轮最大变化项
  - 更新没有塌缩到某一层、某一分支或某个固定参数块
  - 当前主损失回传仍然在驱动较大范围的真实参数更新

### 2. `step_0000940.pt` 推理兼容性验证

- 为避免干扰 `gpu6,7` 的正式训练，本轮继续在 `gpu0` 上完成推理抽检
- 使用命令：
  - `CUDA_VISIBLE_DEVICES=0 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt /home/gaoya/miniconda3/envs/wan-cu128/bin/python -u /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py --checkpoint-dir /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000940.pt --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml --prompt 'industrial rigid body simulation sphere' --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step940 --sampling-steps 2 --num-frames 24 --sampling-mode prefix`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step940`
- 结果文件：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step940/result.json`

### 关键结果

- 推理日志明确出现：
  - `trainer constructed`
  - `checkpoint loaded: missing=3306 unexpected=0`
  - `sampling finished`
  - `output_dir: /data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step940`
- `result.json` / 推理输出中的关键字段继续保持稳定：
  - `track_source = cotracker`
  - `object_latent_tokens = [1, 2, 8, 4096]`
  - `object_context = [1, 16, 4096]`
  - `model_state_key_count = 1272`
  - `checkpoint_key_count = 1272`
  - `unexpected_keys = 0`
  - `missing_keys = 3306`
  - `sample_debug.loss = 126256.984375`
- 本轮采样侧关键张量统计也保持合理、非退化：
  - `object_context std = 3.0064`
  - `pred_step_0 std = 1.2095`
  - `pred_step_1 std = 0.9041`
- 这与前面 `step20` 到 `step920` 的兼容性模式完全一致，说明当前 checkpoint 结构仍然稳定，没有出现新回归

### 3. 本轮附带观察到的推理侧结构信息

- 推理侧打印出的关键张量摘要继续符合预期：
  - `context_latents = [48, 2, 32, 56]`
  - `text_context = [7, 4096]`
  - `object_context = [16, 4096]`
  - `pred_step_0 = [48, 3, 32, 56]`
  - `pred_step_1 = [48, 3, 32, 56]`
- 这些结果继续支持：
  - object 条件通道在推理侧保持活跃、非常量
  - 当前 `track_source = cotracker` 的训练设计仍被推理脚本沿同一路径正确接收并使用

### 4. 本轮观察到的告警与兼容性判断

- 推理过程仍出现了几类非阻塞告警：
  - `timm.models.layers` 的 FutureWarning
  - `torch.cuda.amp.autocast` 的 FutureWarning
  - SDPA / Flash attention fallback 提示
  - `torch.meshgrid` 与 checkpoint API 的兼容性提示
  - object cross-attn 相关基础 Wan 权重 “newly initialized” 提示
  - `torch.utils.checkpoint` 关于 `use_reentrant` 的 FutureWarning
  - `None of the inputs have requires_grad=True. Gradients will be None`
  - `npz_io.py` 关于 non-writable buffer 的 UserWarning
- 这些都没有阻止：
  - `trainer constructed`
  - `checkpoint loaded`
  - `sampling finished`
- 本轮兼容性判断标准维持不变：
  - `checkpoint_key_count == model_state_key_count == 1272`
  - `unexpected_keys == 0`
  - `sampling finished`
- `step_0000940.pt` 完全满足这一标准，没有看到新的结构不匹配问题

### 5. W&B 与训练活性状态

- 本轮核查时，W&B 本地 run 文件继续刷新到：
  - `run-flslwgvw.wandb` 更新时间到 `2026-06-24 02:07:47 UTC`
  - `logs/debug-internal.log` 更新时间到 `2026-06-24 02:08:17 UTC`
  - `files/output.log` 更新时间到 `2026-06-24 02:08:06 UTC`
- 训练日志尾部在本轮抽检期间已明确推进到：
  - `943/20000`
- 当前没有看到新的 OOM、NCCL、进程退出或 W&B 中断信号

### 6. 本轮结论

- 正式训练已稳定推进并保存出：
  - `step_0000940.pt`
- `step920 -> step940` 之间 trainable 权重继续大范围真实更新
- 本轮最大变化项落在主干 LoRA `blocks.9.self_attn.q.lora_B.default.weight`，结合前面 object-conditioned 分支和其他主干层的最大变化项，继续支持“更新在主干与 object 分支之间共同流动，而不是卡死在单一位置”的判断
- `step_0000940.pt` 已被当前推理脚本成功加载并完成采样
- 到这一轮为止，现有证据仍然一致支持：
  - loss 持续变化
  - trainable 权重持续真实更新
  - object 条件入口、object-conditioned 分支和主干 LoRA 都在持续参与优化
  - 新 checkpoint 持续可被推理脚本正确接收并生成采样结果

### 当前总体判断

- 到 `2026-06-24 02:08 UTC` 为止，连续闭环证据已进一步延长到 `step 940+`
- 到当前观测点为止，仍然没有新的报错、停滞、OOM、NCCL 异常、checkpoint 兼容性回归或 W&B 中断信号

## 2026-06-24 02:14 UTC: phase 59, `step_0000960.pt` 保存失败，根因定位到 `/data` 磁盘写满

### 0. 本轮异常现象

- 训练日志已经推进到：
  - `960/20000`
- checkpoint 目录里出现了新的：
  - `step_0000960.pt`
- 但该文件大小只有：
  - `76570624 bytes`
  - 约 `74M`
- 这与前面所有完整 checkpoint 的典型大小：
  - `5533953209 bytes`
  - 约 `5.2G`
- 明显不一致，说明本轮不是“暂时的小文件中间态后续会长成完整文件”，而是保存过程中真正失败

### 1. 训练日志中的直接报错

- `output.log` 在 `960/20000` 附近直接出现：
  - `RuntimeError: [enforce fail at inline_container.cc:857] . PytorchStreamWriter failed writing file data/2: file write failed`
- 这说明出错点发生在：
  - `torch.save(...)` 的 checkpoint 落盘阶段
  - 不是 forward / backward / optimizer step 本身先报错

### 2. 损坏 checkpoint 的二次核实

- 对当前 `step_0000960.pt` 执行 `torch.load(map_location='cpu')`，结果为：
  - `load_ok = false`
  - `error_type = RuntimeError`
  - `error = PytorchStreamReader failed reading zip archive: failed finding central directory`
- 这进一步确认：
  - 当前 `step_0000960.pt` 是不完整的损坏文件
  - 不能作为恢复训练或推理验证的可用 checkpoint

### 3. 根因定位：不是梯度链路问题，而是存储空间耗尽

- 对目标存储挂载点执行 `df -h`，结果为：
  - `/dev/sda1 3.6T used 3.4T avail 0 use% 100% mounted on /data`
- 对 inode 执行 `df -i`，结果为：
  - `IUse% = 3%`
- 这说明：
  - 不是 inode 耗尽
  - 是 `/data` 所在文件系统可用字节数已经归零
- 因为 checkpoint 落盘目录就在 `/data` 下，所以 `PytorchStreamWriter file write failed` 与磁盘写满完全吻合

### 4. 训练进程状态

- 本轮复核进程列表时，没有再看到正式训练命令对应的存活进程：
  - `train_context_video_wan.py`
  - `train_0624pybullet_wan_lora_monitor_gpu67`
- 结合上面的保存异常，可以判断：
  - 正式训练已经因为 checkpoint 写失败而中断退出

### 5. 当前空间占用概况

- `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints` 当前约占：
  - `258G`
- 其中 `pybullet0624_wan_lora_monitor_gpu67` 目录下从 `step_0000560.pt` 到 `step_0000940.pt` 的大量 checkpoint 都是：
  - 单个约 `5.2G`
- 同目录下的 `infer_verify_step*` 结果总体都很小：
  - 多数约 `248K`
  - 日志侧整体约 `15M`
- 因此这次存储危机的主因很明确：
  - checkpoint 长期积累占用过大
  - 不是推理抽检目录或 wandb 日志导致

### 6. 对“主损失梯度是否有效”的影响判断

- 到 `step_0000940.pt` 为止，我们已经有连续多轮闭环证据支持：
  - loss 持续变化
  - `1270/1272` 个 trainable tensor 持续真实更新
  - object-conditioned 分支和主干 LoRA 都在参与更新
  - 新 checkpoint 可被推理脚本加载并完成采样
- 本轮 `step_0000960.pt` 的异常没有提供任何新证据去推翻这些结论
- 当前新增问题是：
  - 存储层故障打断了训练连续性
  - 不是新的“主损失梯度失效”证据

### 7. 当前恢复前提

- 由于 `/data` 可用空间已经是：
  - `0`
- 在不释放或迁移足够空间之前：
  - 不能重新启动正式训练
  - 否则新的 checkpoint 落盘会再次失败
- 考虑到单个完整 checkpoint 约为：
  - `5.2G`
- 实际恢复时至少需要：
  - 删除损坏的 `step_0000960.pt`
  - 再额外释放至少一个完整 checkpoint 以上的可用空间
- 更稳妥的恢复空间建议是：
  - 预留 `10G+`
  - 避免训练刚恢复又在下一次落盘点再次写满

### 8. 针对重复故障的代码级修复

- 本轮已经对训练保存逻辑做了最小且直接的修复，位置：
  - `code_vjepa_vggt/training/runner.py`
  - `code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml`
- 修复点 1：checkpoint 改为原子写入
  - 先写入临时文件
  - 成功后再 `os.replace(...)` 覆盖成正式 `step_XXXXXXX.pt`
  - 如果写失败，会清理临时文件，避免留下新的“半拉子 checkpoint”
- 修复点 2：增加 checkpoint 轮转保留策略
  - 新配置：
    - `logging.max_checkpoints: 8`
  - 训练在保存新 checkpoint 前，会自动删除最旧的完整 `step_*.pt`
  - 从而把占用控制在最近 8 个完整 checkpoint 左右，而不是无限累计
- 修复点 3：增加保存前剩余空间阈值
  - 新配置：
    - `logging.min_checkpoint_free_gb: 12`
  - 若保存前剩余空间低于阈值，会在写 checkpoint 之前显式报错
  - 这样能避免继续写出损坏的部分文件

### 9. 修复后的轻量级自测

- `runner.py` 与 `train_context_video_wan.py` 已通过：
  - `python -m py_compile`
- checkpoint 轮转逻辑已单独验证：
  - 在临时目录连续保存 `step20/40/60`
  - `keep_last=2` 时会自动删除最早的 `step_0000020.pt`
  - 最终只保留：
    - `step_0000040.pt`
    - `step_0000060.pt`
- 剩余空间阈值逻辑也已单独验证：
  - 人为设置极高 `min_free_gb`
  - 保存前会直接报：
    - `insufficient free disk space before checkpoint save`

### 10. 当前最稳恢复点与恢复命令

- 当前最后一个完整、已验证可推理的 checkpoint 仍然是：
  - `step_0000940.pt`
- 恢复训练时，应当从它继续，而不是使用损坏的：
  - `step_0000960.pt`
- 恢复命令应使用：

```bash
CUDA_VISIBLE_DEVICES=6,7 CODEX_DEBUG_TRAINER_INIT=1 CODEX_DEBUG_RUNNER_INIT=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate launch \
  --multi_gpu --num_processes 2 --gpu_ids 6,7 --mixed_precision bf16 \
  --main_process_port 29525 \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_context_video_wan.py \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_wan_lora_monitor_gpu67.yaml \
  --resume-checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_wan_lora_monitor_gpu67/step_0000940.pt
```

### 11. 现在还差什么

- 当前仍然不能立即重启正式训练，因为 `/data` 还是：
  - `Avail 0`
- 但与上一阶段相比，恢复路径已经明确为：
  - 先释放足够空间
  - 删除损坏的 `step_0000960.pt`
  - 从 `step_0000940.pt` 恢复
  - 之后由新的 checkpoint 轮转和空间阈值机制接管，避免再次无限堆积到磁盘打满

### 12. 为恢复动作补充的工具脚本

- 为避免后续手工敲高风险命令，本轮新增了两个恢复工具：
  - `code_vjepa_vggt/manage_train0624_checkpoints.py`
  - `code_vjepa_vggt/restart_train0624_from_step940.sh`
- 其中：
  - `manage_train0624_checkpoints.py` 支持：
    - `list`
    - `move`
    - `delete`
    - `--dry-run`
  - `restart_train0624_from_step940.sh` 固定封装了：
    - `gpu6,7`
    - `accelerate launch`
    - `train_0624pybullet_wan_lora_monitor_gpu67.yaml`
    - `--resume-checkpoint step_0000940.pt`

### 13. 工具脚本的轻量验证

- `manage_train0624_checkpoints.py` 已通过：
  - `python -m py_compile`
- 对计划中的“迁移最老 4 个 checkpoint”做过 dry-run：
  - `step_0000020.pt`
  - `step_0000040.pt`
  - `step_0000060.pt`
  - `step_0000080.pt`
- dry-run 汇总结果为：
  - `total_selected_gib = 20.62`
- `restart_train0624_from_step940.sh` 已通过：
  - `bash -n`

### 14. 当前推荐恢复动作

- 不再建议继续保留“无限 checkpoint 累积”的旧策略
- 当前最稳妥的恢复顺序应为：
  - 用 `manage_train0624_checkpoints.py` 先迁移或删除最老一批 checkpoint
  - 删除损坏的 `step_0000960.pt`
  - 确认 `/data` 恢复出至少 `12G` 以上剩余空间
  - 执行 `restart_train0624_from_step940.sh`
  - 训练恢复后继续沿用前面的监控闭环：
    - loss
    - 权重差分
    - W&B 刷新
    - 新 checkpoint 推理兼容性

## 2026-06-24 02:30 UTC: phase 60, 首次恢复训练失败，根因定位到 object pooler latent dim 推断 key 写错

### 0. 本轮现象

- 在清理空间并从 `step_0000940.pt` 重新启动正式训练后，进程没有进入真正的训练循环
- 恢复日志在 `load_state_dict` 阶段直接报错：
  - `size mismatch for object_pooler.latent_proj.weight`
  - checkpoint 中权重 shape：
    - `[4096, 48]`
  - 当前新建模型中的权重 shape：
    - `[4096, 16]`

### 1. 为什么这不是 checkpoint 损坏

- `step_0000940.pt` 本身仍可正常 `torch.load`
- checkpoint 内记录：
  - `step = 940`
  - `model_keys = 1272`
- 单独读取 checkpoint 中相关参数，确认实际 key 与 shape 为：
  - `object_pooler.latent_proj.weight`
  - `(4096, 48)`
- 因此问题不是 checkpoint 坏了，而是恢复时构造出来的新模型维度不匹配

### 2. 根因

- 恢复前，训练入口会调用：
  - `_load_trainable_state(...)`
  - `_infer_object_pooler_latent_dim(...)`
- 但原始实现只检查了：
  - `bundle.object_pooler.latent_proj.weight`
- 而当前 `step_0000940.pt` 中真实存在的 key 是：
  - `object_pooler.latent_proj.weight`
- 结果是：
  - latent dim 推断失败
  - 回退到了默认值 `16`
  - 新建 `ObjectTubeProjector.latent_proj` 仍是 `Linear(16, 4096)`
  - 最终在恢复 `load_state_dict` 时因 `[4096, 48] -> [4096, 16]` 不匹配而退出

### 3. 证据

- 修复前单独验证 `_infer_object_pooler_latent_dim(...)` 对 `step_0000940.pt` 的结果：
  - `inferred_dim = 16`
- 同时打印 checkpoint 中真实相关 key：
  - `['object_pooler.latent_proj.weight']`
- 这两条证据合在一起，直接证明：
  - 不是 checkpoint 中没有这个权重
  - 而是推断函数盯错了 key 名

### 4. 修复

- 已修改：
  - `code_vjepa_vggt/infer_context_video_wan.py`
- 当前 `_infer_object_pooler_latent_dim(...)` 同时支持：
  - `object_pooler.latent_proj.weight`
  - `bundle.object_pooler.latent_proj.weight`
- 修复后再次单独验证：
  - `inferred_dim = 48`

### 5. 对主损失梯度问题的影响判断

- 这次恢复失败仍然不是“主损失梯度无效”的新证据
- 当前新增问题属于：
  - resume 构图前的 checkpoint 兼容性 bug
- 训练在恢复前向循环之前就退出了，因此：
  - 还没有产生新的 loss / 梯度 / 权重更新证据

### 6. 当前下一步

- 现在可以基于修复后的 latent dim 推断逻辑，再次从：
  - `step_0000940.pt`
- 重试正式训练恢复
- 若恢复成功，继续回到原先的监控闭环：
  - W&B 刷新
  - step 继续推进
  - 新 checkpoint 落盘
  - 推理兼容性验证

### 12. 当前最小风险空间恢复方案

- 目前已经量化出的最小风险释放方案是：
  - 将以下最老的 4 个 checkpoint 从 `/data` 迁移到 `/home` 归档目录，而不是直接删除：
    - `step_0000020.pt`
    - `step_0000040.pt`
    - `step_0000060.pt`
    - `step_0000080.pt`
- 这 4 个文件合计大小约为：
  - `22135812836 bytes`
  - `20.62G`
- 归档目标目录已准备好：
  - `/home/gaoya/AAA_train0624_checkpoint_archive`
- 该目录所在磁盘当前可用空间约为：
  - `92.36G`
- 因此这一方案的特点是：
  - 能明显缓解 `/data` 空间压力
  - 不丢失早期 checkpoint 资产
  - 为删除损坏的 `step_0000960.pt` 和从 `step_0000940.pt` 恢复训练创造足够空间条件
- 若执行这一方案，随后应继续做：
  - 删除损坏的 `step_0000960.pt`
  - 再次确认 `/data` 可用空间
  - 用 `--resume-checkpoint step_0000940.pt` 在 `gpu6,7` 上恢复正式训练

## 2026-06-24 03:46 UTC: phase 61, freeze-LoRA 新 run 已稳定训练并完成首个推理闭环

### 0. 当前 run 标识

- `tmux` 会话：
  - `train0624_freeze_lora`
- 配置：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_freeze_lora_other_modules_gpu67.yaml`
- 启动脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/run_train_0624_freeze_lora_other_modules_gpu67.sh`
- 日志：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/logs/freeze_lora/train0624_freeze_lora_20260624_032723.log`
- W&B：
  - project: `vjepa_vggt_wan`
  - run: `pybullet0624_freeze_lora_other_modules_gpu67`
  - run id: `xkws0bla`

### 1. 已验证训练正常进入正式循环

- rank0 / rank1 都已经走到：
  - `first batch fetched`
  - `first forward start`
  - `first forward done`
  - `first backward start`
  - `first backward done`
  - `first optimizer.step done`
- 首个 step 的日志证据：
  - rank0:
    - `first forward done` at runner `+36.73s`
    - `first backward done` at runner `+44.75s`
    - `first optimizer.step done` at runner `+45.20s`
  - rank1:
    - `first forward done` at runner `+42.21s`
    - `first backward done` at runner `+43.39s`
    - `first optimizer.step done` at runner `+43.83s`

### 2. 已验证 loss / W&B / checkpoint 正常

- W&B 已正常登录并同步：
  - 本地目录：
    - `/data/gaoya/AAA_test_video/0623/train/train0624/logs/wandb/wandb/run-20260624_033004-xkws0bla`
  - 页面：
    - `https://wandb.ai/875222004-gy/vjepa_vggt_wan/runs/xkws0bla`
- 训练过程中已有连续 loss：
  - step 1:
    - `loss=0.1119`
  - step 2:
    - `loss=0.8549`
  - 后续日志继续推进到：
    - `step 47`
    - 最近一条可见进度：
      - `loss=0.4984`
- checkpoint 已正常落盘：
  - `step_0000020.pt`
  - `step_0000040.pt`
- 当前 checkpoint 目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_freeze_lora_other_modules_gpu67`

### 3. 已验证 trainable checkpoint 结构正确

- `step_0000020.pt` 可正常 `torch.load`
- 顶层 key：
  - `['model', 'step']`
- `step` 值：
  - `20`
- trainable tensor 数量：
  - `432`
- 说明：
  - 当前保存格式是“只导出 trainable modules 的状态”，不是整模型 full-state checkpoint
  - 因此用于推理时会看到大量 `missing_keys` 指向冻结的上游 backbone / adapter 权重，这属于预期行为，不代表 checkpoint 损坏

### 4. 已验证权重在持续更新，不是空转

- 对比：
  - `step_0000020.pt`
  - `step_0000040.pt`
- 差异统计结果：
  - `key_sets_equal = True`
  - `num_common = 432`
  - `changed_tensors = 430`
  - `unchanged_tensors = 2`
  - `sum_abs_diff = 187676.77407554176`
  - `max_abs_diff = 0.0008077044039964676`
  - `max_abs_diff_name = object_pooler.depth_proj.2.weight`
- 结论：
  - trainable 权重不是只在第一步发生一次更新
  - 至少从 step20 到 step40 期间，绝大多数 trainable tensors 持续发生了参数变化

### 5. 已验证 checkpoint 能用当前推理脚本生成视频

- 推理命令使用：
  - checkpoint:
    - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0624_freeze_lora_other_modules_gpu67/step_0000020.pt`
  - config:
    - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0624pybullet_freeze_lora_other_modules_gpu67.yaml`
  - context video:
    - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4`
  - prompt:
    - `industrial rigid body simulation sphere`
- 推理输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/infer_verify_step20`
- 实际产物：
  - `prediction.mp4`
  - `prediction.browser.mp4`
  - `result.json`
- `result.json` 中已记录：
  - `prediction_video_raw`
  - `prediction_video`
  - `target_num_frames = 24`
  - `configured_num_context_frames = 8`
- 结论：
  - `step_0000020.pt` 已经能被当前 `infer_context_video_wan.py` 正常加载并生成视频

### 6. 当前仍需继续监控的点

- 训练虽然已经证明：
  - 能正常启动
  - 有有效 backward
  - 有 optimizer step
  - 权重持续更新
  - checkpoint 可推理
- 但仍需继续长期盯住：
  - 更高 step 的 loss 是否稳定
  - checkpoint 数量与磁盘空间是否继续可控
  - 更高 step checkpoint 的推理结果是否继续可用

### 7. 追加监控结果：训练继续推进到 step60，权重仍持续更新

- 后续新 checkpoint 已继续落盘：
  - `step_0000060.pt`
- 训练进度日志已继续推进到：
  - `step 56`
  - 可见最近一条 loss：
    - `loss=1.5115`
- 再次对比：
  - `step_0000040.pt`
  - `step_0000060.pt`
- 差异统计结果：
  - `key_sets_equal = True`
  - `num_common = 432`
  - `changed_tensors = 430`
  - `unchanged_tensors = 2`
  - `sum_abs_diff = 158091.30594216613`
  - `max_abs_diff = 0.000709090381860733`
  - `max_abs_diff_name = bundle.dit.base_model.model.blocks.20.object_cross_attn.o.base_layer.weight`
- 结论：
  - 从 `step20 -> step40` 到 `step40 -> step60`，trainable 权重都在持续变化
  - 当前没有出现“只在最初几步更新、后面冻结不动”的迹象
