# train0705 权重来源总链路

这份文档按真实依赖顺序整理 `train0705` 涉及到的训练权重，从最上游开始写，一直写到 `train0705` 自己产出的 Stage1B checkpoint。

目标是回答三个问题：

1. 最上游基础模型是什么。
2. 中间每一段权重是被哪条训练脚本产出的。
3. `train0705` 最终到底接了哪些上游权重，自己又训练出了什么。


## 0. 最上游基础模型

最上游不是这个项目训练出来的，而是外部 Wan 基础模型：

- `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`

这份权重的角色是：

- Wan 2.2 TI2V-5B base model
- 后续所有 `wan2.2` 这条线的 LoRA 训练，都是在它上面做
- `train0705` 自己也会通过 `--wan_root` 加载它


## 1. 第一段：OpenVid + MOVI-D + Genesis rigid 混合数据 LoRA 预训练



### 1.1 训练脚本

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/run_train.sh`

### 1.2 训练输入

- Wan base:
  - `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- 数据配置：
  - `OpenVid + MOVI-D + Genesis rigid` 混合数据

### 1.3 训练输出

输出目录：

- `/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora`

当前可见的一组主 checkpoint：

- `step-001000/checkpoint.safetensors`
- `step-002000/checkpoint.safetensors`
- `step-003000/checkpoint.safetensors`
- `step-004000/checkpoint.safetensors`
- `step-005000/checkpoint.safetensors`
- `step-006000/checkpoint.safetensors`
- `step-007000/checkpoint.safetensors`
- `step-008000/checkpoint.safetensors`
- `step-009000/checkpoint.safetensors`
- `step-010000/checkpoint.safetensors`

这一步里，后续最关键、被下游继续使用的是：

- `/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors`

### 1.4 它的作用

这份 `step-010000` 是后续 phys-state continuation LoRA 的初始化权重。


## 2. 第二段：phys-state 数据继续训练 LoRA

这一段产出的就是 `train0705` 直接依赖的基础 LoRA。

### 2.1 训练脚本

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/run_train_phys_state_lora_continue.sh`

### 2.2 训练输入

这条脚本的输入有两部分：

- Wan base:
  - `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- 初始化 LoRA:
  - `/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors`

也就是说，这一步不是从空 LoRA 开始，而是：

- 先拿 OpenVid mixed LoRA step-010000
- 再在 phys-state 数据上继续训练

### 2.3 训练输出

输出目录：

- `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24`

当前可见 checkpoint：

- `step-000500/checkpoint.safetensors`
- `step-001000/checkpoint.safetensors`

其中后续 `train0705` 默认真正使用的是：

- `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`

### 2.4 它的作用

这份 `step-000500` 后面被两条线共同拿去初始化：

1. teacher-student Stage1A
2. `train0705` 自己的 Stage1B DiffSynth-native 训练


## 3. 第三段：teacher-student Stage1A

这一段训练的是 object token builder 相关模块，给后续 Stage1B 提供冻结初始化。

### 3.1 训练脚本入口

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_teacher_student_stage1a_gpu67.sh`

### 3.2 训练配置

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/object_token_teacher_student/config_stage1a_full_token_template.yaml`

### 3.3 训练输入

这一步会加载：

- Wan base:
  - `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- 初始化 LoRA:
  - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`

也就是说，Stage1A 是建立在 `raw_phys_state step-000500` 这份 LoRA 之上的。

### 3.4 标准输出目录

配置里标准 `output_dir` 是：

- `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token`

### 3.5 train0705 实际引用的 Stage1A 权重

`train0705` 默认吃的不是上面这个标准目录里的最新文件，而是一个历史 old run：

- `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt`

这份权重的作用是：

- 提供 `object_pooler.*`
- 提供 `object_aux_heads.*`
- 在 `train0705` 里作为冻结初始化加载


## 4. 第四段：老的 Stage1B no-GT-box teacher-student 训练线

这一段不是 `train0705` 自己的输出，但它是 `train0705` 头注释里明确写的“原训练脚本路径1”。

### 4.1 原训练脚本路径1

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_teacher_student_stage1b_context_only_no_gt_box.sh`

### 4.2 它对应的训练模块

- `code_vjepa_vggt.object_token_teacher_student.train_stage1b_context_only_no_gt_box`

### 4.3 它对应的老权重线

这条老线默认对应的是：

- `pybullet0629_teacher_student/stage1b_context_only_no_gt_box`

它的意义是：

- `train0705` 的训练逻辑不是凭空发明的
- 而是把这条老 teacher-student `stage1b context-only no-GT-box` 训练逻辑迁到了 DiffSynth-native `v_newtrain` 框架

所以如果问：

- “原训练脚本路径1是什么？”

答案就是：

- `run_train_teacher_student_stage1b_context_only_no_gt_box.sh`


## 5. 第五段：train0705 自己的 Stage1B DiffSynth-native 训练

这一步才是 `train0705` 目录本身真正做的事情。

### 5.1 主训练脚本

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/train_stage1b_context_only_no_gt_box_v_newtrain.py`

### 5.2 常用启动脚本

- 单卡：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_train_stage1b_context_only_no_gt_box_v_newtrain0705.sh`
- 多卡：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_train_stage1b_context_only_no_gt_box_v_newtrain0705_gpu0235.sh`

### 5.3 它加载的上游权重

`train0705` 默认会同时加载三类上游权重：

1. Wan base
   - `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
2. 基础 LoRA
   - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`
3. Stage1A token builder
   - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt`

### 5.4 它训练什么

这一步不是训练全模型，而是只训练 Stage1B 的 object 分支相关模块：

- `DiT object 注入分支`
- `ObjectConditionAdapter`

而下面这些是冻结的：

- Wan base
- 基础 LoRA
- VAE
- Text encoder
- Stage1A token builder
- JEPA / CoTracker / VGGT 特征提取分支

### 5.5 它的输出目录

默认输出根目录：

- `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705`

正式训练目录：

- `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints`

当前可见正式 checkpoint：

- `step-002500/checkpoint.safetensors`
- `step-003000/checkpoint.safetensors`
- `step-003500/checkpoint.safetensors`
- `step-004000/checkpoint.safetensors`
- `step-004500/checkpoint.safetensors`
- `step-005000/checkpoint.safetensors`
- `step-005500/checkpoint.safetensors`
- `step-006000/checkpoint.safetensors`
- `step-006500/checkpoint.safetensors`
- `step-007000/checkpoint.safetensors`

这些就是 `train0705` 这条线自己训练出来的最终 Stage1B 权重。


## 6. 整条链压成一句话

完整顺序是：

1. `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
2. `train0419_reference/run_train.sh`
3. `/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors`
4. `train0419_reference/run_train_phys_state_lora_continue.sh`
5. `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`
6. `run_train_teacher_student_stage1a_gpu67.sh`
7. `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt`
8. `train0705/run_train_stage1b_context_only_no_gt_box_v_newtrain0705.sh`
9. `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-*`


## 7. 最后单独回答两个最常问的问题

### 7.1 `/data/gaoya/AAA_test_video/0529/.../step-000500/checkpoint.safetensors` 是怎么得到的

答案是：

- 它由 `train0419_reference/run_train_phys_state_lora_continue.sh` 训练得到
- 它不是从零开始
- 它是从 `openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors` 继续训出来的


### 7.2 `train0705` 的“原训练脚本路径1”是什么

答案是：

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_teacher_student_stage1b_context_only_no_gt_box.sh`

它对应的是老 teacher-student 的 `stage1b context-only no-GT-box` 训练线；
`train0705` 则是把这条老线迁到 DiffSynth-native `v_newtrain` 框架后的新实现。
