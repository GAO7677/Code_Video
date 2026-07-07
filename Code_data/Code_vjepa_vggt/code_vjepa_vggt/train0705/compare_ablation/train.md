# train0705 compare_ablation

这个目录现在包含两套消融：

- `signal ablation`
  - 通过占位输入/置零输入做快速训练级对照
  - 现有文件前缀：`run_train_stage1b_*` / `train_stage1b_context_only_no_gt_box_v_newtrain_ablation.py`
- `structure ablation`
  - 真正把模块从训练结构里拿掉，然后重新训练
  - 新文件前缀：`structure_ablation*`


## 1. 结构消融文件

- `structure_ablation_train_stage1b_context_only_no_gt_box_v_newtrain.py`
  - 结构消融专用训练入口
  - 不改原始训练文件
  - 新增参数：
    - `--structure_ablation_type none`
    - `--structure_ablation_type wo_cotracker`
    - `--structure_ablation_type wo_jepa`
    - `--structure_ablation_type wo_vggt`

- `structure_ablation_base_gpu0235.sh`
  - 结构消融公共基座脚本

- `structure_ablation_no_stage1a_init_gpu0235.sh`
- `structure_ablation_wo_cotracker_gpu0235.sh`
- `structure_ablation_wo_jepa_gpu0235.sh`
- `structure_ablation_wo_vggt_gpu0235.sh`
  - 四个具体实验入口


## 2. 结构消融定义

### 2.1 No Stage1A init

含义：

- 不加载 `Stage1A` 的 `object_pooler / object_aux_heads` 初始化
- 结构不变
- 这是重新训练版，不是推理时置零

加载权重：

- Wan base
  - `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- Base LoRA
  - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`
- 不加载 Stage1A init
- JEPA
  - `/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth`
- CoTracker
  - `/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth`
- VGGT
  - `/data/gaoya/ckpt/facebook-VGGT-1B`

冻结：

- Wan DiT base
- Base LoRA
- VAE
- Text encoder
- `object_pooler`
- `object_aux_heads`
- JEPA / CoTracker / VGGT 特征提取器

训练：

- DiT object 注入分支
  - `object_embedding`
  - `object_cross_attn`
  - `object_gate`
  - `norm4`
- `object_adapter`

运行命令：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/compare_ablation/structure_ablation_no_stage1a_init_gpu0235.sh
```


### 2.2 w/o CoTracker

含义：

- 从结构上移除 `CoTracker` 模块
- 同时移除 `object_pooler` 内的 motion branch
- 不再使用真实轨迹
- 只保留 viewer grounding 给出的静态 query prior 作为 slot anchor，用来做 JEPA / latent / VGGT 的局部采样

加载权重：

- Wan base
  - `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- Base LoRA
  - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`
- Stage1A init
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt`
- JEPA
  - `/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth`
- VGGT
  - `/data/gaoya/ckpt/facebook-VGGT-1B`
- 不使用 CoTracker 权重，即使脚本里保留了原路径参数
  - `/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth`

冻结：

- Wan DiT base
- Base LoRA
- VAE
- Text encoder
- Stage1A `object_pooler / object_aux_heads`
- JEPA / VGGT 特征提取器

训练：

- DiT object 注入分支
- `object_adapter`

运行命令：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/compare_ablation/structure_ablation_wo_cotracker_gpu0235.sh
```


### 2.3 w/o JEPA

含义：

- 从结构上移除 `JEPA` 模块
- 同时移除 `object_pooler` 内的 JEPA appearance branch
- appearance 只来自 latent branch

加载权重：

- Wan base
  - `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- Base LoRA
  - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`
- Stage1A init
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt`
- CoTracker
  - `/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth`
- VGGT
  - `/data/gaoya/ckpt/facebook-VGGT-1B`
- 不使用 JEPA 权重，即使脚本里保留了原路径参数
  - `/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth`

冻结：

- Wan DiT base
- Base LoRA
- VAE
- Text encoder
- Stage1A `object_pooler / object_aux_heads`
- CoTracker / VGGT 特征提取器

训练：

- DiT object 注入分支
- `object_adapter`

运行命令：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/compare_ablation/structure_ablation_wo_jepa_gpu0235.sh
```


### 2.4 w/o VGGT

含义：

- 从结构上移除 `VGGT` 模块
- 同时移除 `object_pooler` 内的 VGGT geometry / depth branch
- 几何只保留轨迹摘要分支

加载权重：

- Wan base
  - `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- Base LoRA
  - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`
- Stage1A init
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt`
- JEPA
  - `/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth`
- CoTracker
  - `/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth`
- 不使用 VGGT 权重，即使脚本里保留了原路径参数
  - `/data/gaoya/ckpt/facebook-VGGT-1B`

冻结：

- Wan DiT base
- Base LoRA
- VAE
- Text encoder
- Stage1A `object_pooler / object_aux_heads`
- JEPA / CoTracker 特征提取器

训练：

- DiT object 注入分支
- `object_adapter`

运行命令：

```bash
bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/compare_ablation/structure_ablation_wo_vggt_gpu0235.sh
```


## 3. 默认输出

```text
/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705_structure_ablation/
```


## 4. 备注

- 默认使用 `GPU_SET=0,2,3,5`
- 明确不使用 `gpu4`
- 结构消融推荐用于论文表格中的正式 retrain ablation
- 旧的 `run_train_stage1b_*` 那套更适合快速 signal ablation
