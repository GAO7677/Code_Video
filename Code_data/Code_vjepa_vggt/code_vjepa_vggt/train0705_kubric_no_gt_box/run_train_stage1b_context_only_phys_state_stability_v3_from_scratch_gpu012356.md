# Stage1B PhysState Stability V3 Training

对应训练脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_train_stage1b_context_only_phys_state_stability_v3_from_scratch_gpu012356.sh
```

## 1. 训练目标

该脚本保留当前 Stage1B stability-v3 的模型结构、object branch、regularization、
ratio guard 和初始化方案，仅将训练数据后端从 `KubricNoGTBoxDataset` 切换为：

```python
code_vjepa_vggt.data.phys_state_dataset.PhysStateEpisodeDataset
```

训练入口通过以下参数实例化该数据集：

```bash
--dataset_type phys_state_episode
--phys_state_root "${PHYS_STATE_ROOT}"
--phys_state_split "${PHYS_STATE_SPLIT}"
```

## 2. 默认训练数据

默认数据根目录：

```text
/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500
```

默认 split：

```text
train
```

2026-07-13 实际检查结果：

| 项目 | 数值 |
| --- | ---: |
| JSON 元数据文件 | 3600 |
| NPZ 数据文件 | 3600 |
| `PhysStateEpisodeDataset` 有效样本 | 3600 |
| 单条完整视频帧数 | 24 |
| 单条 context 帧数 | 8 |
| resize 后视频形状 | `[3, 24, 512, 896]` |
| resize 后 context 形状 | `[3, 8, 512, 896]` |
| train split 磁盘大小 | 约 8.2 GB |

有效样本数通过实际构造 `PhysStateEpisodeDataset` 得到，不只是统计目录文件数。
当前 3600 条数据全部满足 `num_context_frames=8` 和
`context_fraction=0.5` 的过滤要求。

数据文件名中的 `s8_f16` 对应 8 帧 context 和 16 帧 future，因此每条样本共
24 帧。`--num_frames 69` 是共享训练入口保留的参数，不会将 PhysState NPZ
样本扩展为 69 帧；该数据集返回的实际视频长度由 NPZ 中的
`context_frames + future_frames` 决定。

默认使用固定的前 8 帧作为 context：

```python
context_fraction=0.5
random_context_frames=False
num_context_frames=8
```

可覆盖数据根目录和 split：

```bash
PHYS_STATE_ROOT=/path/to/phys_state_dataset \
PHYS_STATE_SPLIT=train \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_train_stage1b_context_only_phys_state_stability_v3_from_scratch_gpu012356.sh
```

## 3. 模型初始化

Stage1A 的 `object_pooler` 和 `object_aux_heads` 初始化权重：

```text
/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt
```

冻结的 Wan base LoRA：

```text
/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors
```

Wan 基础模型：

```text
/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B
```

该版本不加载 Stage1B resume。object adapter 和 DiT object branch 从头训练，
Stage1A token builder 只作为冻结初始化。

## 4. 可训练与冻结模块

可训练模块：

- `object_adapter`
- DiT blocks 中的 `object_cross_attn`
- DiT blocks 中的 `object_gate`
- 对应 object branch normalization 参数

冻结模块：

- Wan DiT base
- base LoRA
- VAE
- text encoder
- Stage1A `object_pooler`
- Stage1A `object_aux_heads`
- JEPA、CoTracker、VGGT 和 grounding/SAM2 特征模块

## 5. Stability V3 配置

主要 object branch 配置：

```text
object_num_queries                   = 8
aux_max_objects                     = 4
compact_object_context_slots        = true
object_slot_dropout_prob            = 0.35
full_slot_loss_weight               = 1.0
object_gate_init                    = 0.1
```

训练期 regularization：

```text
lambda_object_context_reg           = 1e-2
lambda_object_gate_reg              = 1e-1
object_gate_reg_target              = 0.08
lambda_object_adapter_mlp_reg       = 1e-1
object_adapter_mlp_reg_target       = 2.5
object_adapter_mlp_residual_max_ratio = 3.0
```

object branch 响应保护与诊断：

```text
object_branch_train_trace           = true
object_branch_ratio_guard_max_ratio = 0.30
object_branch_ratio_guard_max_block_id = -1
debug_print_object_regularization   = true
```

优化参数：

```text
learning_rate                       = 1e-4
weight_decay                        = 0.01
gradient_accumulation_steps         = 1
optimizer_type                      = paged_adamw8bit
max_grad_norm                       = 1.0
mixed_precision                     = bf16
max_train_steps                     = 10000
save_steps                          = 500
max_checkpoints_keep                = 10
```

## 6. Grounding 配置

```text
proposal source                     = gdino_only
prompt frame                        = first
GDINO box threshold                 = 0.20
GDINO text threshold                = 0.15
SAM2 segment length                 = 8
track dedupe IoU                    = 0.75
caption-derived terms               = disabled
```

默认 grounding prompt：

```text
box . cube . block . cylinder . capsule . sphere . ball .
```

## 7. GPU 分配

默认需要 6 张可见 GPU：

- 前 4 张：4 个 distributed training process。
- 后 2 张：JEPA、CoTracker、VGGT、SAM2 等 object auxiliary 前向。
- `OBJECT_AUX_DEVICES=cuda:4,cuda:4,cuda:5,cuda:5` 按训练 rank 分配。
- 脚本显式拒绝已知故障的物理 GPU4。

默认 UUID 配置保持当前正式 stability-v3 脚本使用的物理 GPU
`0,1,2,3,5,6` 映射。

## 8. 输出与 W&B

默认输出目录：

```text
/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_phys_state_stability_v3_from_scratch_<UTC时间戳>
```

每次执行默认创建新的 W&B run：

```text
project: vjepa_vggt_wan
name: stage1b_phys_state_stability_v3_from_scratch_<UTC时间戳>
mode: online
```

可以显式指定稳定的 run tag、输出目录和 W&B name：

```bash
RUN_TAG=phys_state_formal_v1 \
OUTPUT_DIR=/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_phys_state_stability_v3_formal_v1 \
WANDB_NAME=stage1b_phys_state_stability_v3_formal_v1 \
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_train_stage1b_context_only_phys_state_stability_v3_from_scratch_gpu012356.sh
```

## 9. 启动命令

前台启动：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_train_stage1b_context_only_phys_state_stability_v3_from_scratch_gpu012356.sh
```

tmux 后台启动：

```bash
tmux new-session -d \
  -s stage1b_phys_state_stability_v3 \
  "bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_kubric_no_gt_box/run_train_stage1b_context_only_phys_state_stability_v3_from_scratch_gpu012356.sh"
```

查看 tmux：

```bash
tmux attach -t stage1b_phys_state_stability_v3
```

当前只创建并验证了训练脚本，没有自动启动训练。
