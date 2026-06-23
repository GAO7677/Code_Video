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
