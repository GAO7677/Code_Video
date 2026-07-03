# DiffSynth-Native Training Script 使用指南

## 概述

本文档说明如何使用新创建的 `train_stage1b_diffsynth_native.py` 训练脚本。

## 创建的文件

1. **核心模块**
   - `models/diffsynth_object_injection.py` - Monkey-patch DiTBlock 添加 object_cross_attn 支持
   - `trainers/diffsynth_context_trainer.py` - DiffSynth-native trainer，集成 object conditioning
   
2. **训练脚本**
   - `train_stage1b_diffsynth_native.py` - 主训练入口
   - `object_token_teacher_student/config_stage1b_diffsynth_native_test.yaml` - 测试配置
   
3. **测试脚本**
   - `test_diffsynth_injection.py` - 验证 object branch 注入是否正常

## 关键架构差异

### 原有训练脚本 (train_stage1b_context_only_no_gt_box_diffsynth.py)
```
code_vjepa_vggt.WanContextVideoModel (wrapper)
  └─> bootstrap.py monkey-patch Wan2.2
      └─> 添加 object_cross_attn 到 blocks
      └─> LoRA 封装产生 .base_layer. 键名

问题：训练权重键名 (object_cross_attn.q.base_layer.weight) 
     与 DiffSynth pipe() 推断路径不匹配 (object_cross_attn.q.weight)
```

### 新训练脚本 (train_stage1b_diffsynth_native.py)
```
DiffSynth WanVideoPipeline (原生)
  └─> inject_object_branch_to_dit()
      └─> Monkey-patch DiTBlock.forward
      └─> 直接添加 object_cross_attn (无 LoRA 封装)
      └─> 键名与 DiffSynth 推断路径完全一致

优势：训练权重可直接用于 DiffSynth pipe() 推断
```

## 使用步骤

### 步骤 1：测试 object branch 注入

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt

# 激活 conda 环境（需要有 torch、diffsynth 等依赖）
conda activate <your_env>

# 运行测试脚本
python test_diffsynth_injection.py
```

预期输出：
```
================================================================================
Testing DiffSynth-Native Object Branch Injection
================================================================================

1. Importing modules...
   ✓ Imports successful

2. Loading WanVideoPipeline...
   ✓ Pipeline loaded from /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B

3. Checking original DiT blocks...
   Has object_cross_attn: False
   Has norm4: False
   Has object_gate: False

4. Injecting object branch...
   ✓ Object branch injected

5. Verifying injection...
   Has object_cross_attn: True
   Has norm4: True
   Has object_gate: True
   Has object_embedding (global): True
   object_gate value: 0.1000

6. Counting added parameters...
   Object branch parameters: XX,XXX,XXX

7. Testing forward pass with object_context...
   ✓ Forward pass successful, output shape: torch.Size([...])

================================================================================
✅ All tests passed! Architecture is ready for training.
================================================================================
```

### 步骤 2：运行训练（测试模式）

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt

# 使用测试配置运行 100 步
python train_stage1b_diffsynth_native.py \
    --config object_token_teacher_student/config_stage1b_diffsynth_native_test.yaml \
    --gpu 7
```

预期输出：
```
================================================================================
Stage1b Context-Only Training (DiffSynth-Native)
================================================================================
Config: object_token_teacher_student/config_stage1b_diffsynth_native_test.yaml
GPU: 7
Output: /data/gaoya/AAA_test_video/stage1b_diffsynth_native/test_run
================================================================================
Loading WanVideoPipeline from /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B...
Injecting object branch to DiT blocks...
VAE frozen
Text encoder frozen
Injecting LoRA (rank=32, alpha=32) to DiT...
Loading LoRA weights from /data/gaoya/AAA_test_video/.../checkpoint.safetensors...
...
Training: 100%|█████████████████████| 100/100 [XX:XX<00:00, X.XXit/s, loss=X.XXXX]
Checkpoint saved: .../checkpoints/step-000100/checkpoint.safetensors (XXX keys)
Training completed: 100 steps
```

### 步骤 3：验证权重加载

训练完成后，修改推断脚本以使用新的 checkpoint：

```python
# 在 wan_stage1b_context_only_no_gt_box_diffsynth_v2v.py 中

# 1. 导入 inject_object_branch_to_dit
from code_vjepa_vggt.models.diffsynth_object_injection import inject_object_branch_to_dit

# 2. 在加载 checkpoint 前注入 object branch
pipe = WanVideoPipeline.from_pretrained(...)
inject_object_branch_to_dit(pipe.dit, object_cross_attn_dim=4096)

# 3. 加载 checkpoint
state_dict = load_file("path/to/step-000100/checkpoint.safetensors")
pipe.dit.load_state_dict(state_dict, strict=False)

# 4. 推断时传入 object_context
output = pipe(
    ...,
    object_context=object_context,  # [B, T*O, D]
)
```

预期结果：
- 权重键名完全匹配（406/406 keys loaded）
- pred_norm 正常（约 425，不会发散到 1500+）
- 生成视频无噪声

## 配置文件说明

`config_stage1b_diffsynth_native_test.yaml` 关键参数：

```yaml
model:
  wan_ckpt_dir: /data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B  # Wan2.2 预训练权重
  freeze_wan_dit: true         # 冻结 DiT base 参数
  freeze_wan_lora: true        # 冻结 LoRA 参数
  freeze_object_pooler: false  # 训练 ObjectTubeProjector
  wan_lora_rank: 32            # LoRA rank（从 stage1a 加载）
  cond_proj_dim: 4096          # Object token 维度
  object_num_queries: 4        # 每帧 object 数量

data:
  num_context_frames: 8        # Context 帧数
  resolution: [512, 896]       # 视频分辨率
  batch_size: 1                # Batch size

optimization:
  max_steps: 100               # 测试模式：只训练 100 步
  lr: 1.0e-4                   # 学习率
  mixed_precision: bf16        # 使用 bfloat16 混合精度
```

## 常见问题

### Q1: ImportError: No module named 'diffsynth'

**原因**：DiffSynth-Studio-main 未在 PYTHONPATH 中。

**解决**：训练脚本已自动添加路径，无需手动设置。

### Q2: 训练时 OOM

**原因**：GPU 内存不足。

**解决**：
- 减小 batch_size
- 减少 num_context_frames
- 降低 resolution
- 使用 freeze_object_pooler: true

### Q3: 权重加载失败（keys mismatch）

**原因**：checkpoint 是旧版 bootstrap.py 训练的，有 .base_layer. 键名。

**解决**：使用新训练脚本重新训练，或写转换脚本去掉 .base_layer.

### Q4: 推断时 pred_norm 仍然很大

**检查清单**：
1. 是否在加载 checkpoint 前调用 `inject_object_branch_to_dit()`？
2. 是否使用 cfg_scale=1.0（模型未经 CFG 训练）？
3. 是否正确传入 object_context 参数？

## 下一步

1. **验证测试**：运行 test_diffsynth_injection.py 确认架构正常
2. **短训练测试**：运行 100 步训练，检查 loss 是否下降
3. **推断验证**：加载 step-100 checkpoint，推断检查 pred_norm
4. **完整训练**：如果验证通过，修改 max_steps 到 20000 进行完整训练

## 文件路径总结

```
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/
├── models/
│   └── diffsynth_object_injection.py          # Object branch 注入逻辑
├── trainers/
│   └── diffsynth_context_trainer.py           # DiffSynth-native trainer
├── train_stage1b_diffsynth_native.py          # 主训练脚本
├── test_diffsynth_injection.py                # 测试脚本
└── object_token_teacher_student/
    └── config_stage1b_diffsynth_native_test.yaml  # 测试配置

输出目录：
/data/gaoya/AAA_test_video/stage1b_diffsynth_native/test_run/
└── checkpoints/
    ├── step-000050/
    │   └── checkpoint.safetensors
    └── step-000100/
        └── checkpoint.safetensors
```
