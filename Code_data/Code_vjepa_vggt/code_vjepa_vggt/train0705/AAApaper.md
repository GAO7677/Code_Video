# AAApaper

生成日期：2026-07-08

这份文件不是论文正文，而是给 GPT 写论文时使用的“事实底稿”。它按照 `train0705/AAAtrain.md` 的训练链顺序整理整个项目，并且把我已经核查到的事实、基于代码的保守归纳、以及目前不能安全写进论文的内容分开。

使用原则：

1. GPT 只能把“已核实事实”写成确定陈述。
2. “保守归纳”只能写成方法层面的解释，不能写成实验结论。
3. “待补证据”里的内容不能写成论文结果，除非你另外补表格、日志或人工确认。


## 0. 这份底稿覆盖什么

这份底稿覆盖当前仓库里和“当前正式 Stage1B 主线”最直接相关的四部分：

- 上游权重来源链路
- 当前正式 Stage1B 的 Kubric 主训练方法
- 当前正式 Stage1B 的推理与评测工具链
- 早期 `train0705` phys-state Stage1B 实现与它和 Kubric 版的关系

这份底稿的中心对象现在改为 `train0705_kubric_no_gt_box` 这条 Stage1B 主线。早期 `train0705` 物理视频分支保留为前序实现和方法来源说明，而不是当前项目里默认代表 Stage1B 的最终口径。


## 1. 已核查的源文件

下面这些文件和路径已经实际检查过：

- `code_vjepa_vggt/train0705/AAAtrain.md`
- `code_vjepa_vggt/train0705/AAAinfer.md`
- `code_vjepa_vggt/train0705/AAAbench.md`
- `code_vjepa_vggt/train0419_reference/run_train.sh`
- `code_vjepa_vggt/train0419_reference/run_train_phys_state_lora_continue.sh`
- `code_vjepa_vggt/run_train_teacher_student_stage1a_gpu67.sh`
- `code_vjepa_vggt/run_train_teacher_student_stage1b_context_only_no_gt_box.sh`
- `code_vjepa_vggt/object_token_teacher_student/config_stage1a_full_token_template.yaml`
- `code_vjepa_vggt/object_token_teacher_student/README.md`
- `code_vjepa_vggt/train0705/train_stage1b_context_only_no_gt_box_v_newtrain.py`
- `code_vjepa_vggt/train0705/run_train_stage1b_context_only_no_gt_box_v_newtrain0705.sh`
- `code_vjepa_vggt/train0705/run_train_stage1b_context_only_no_gt_box_v_newtrain0705_gpu0235.sh`
- `code_vjepa_vggt/train0705/infer_stage1b_context_only_no_gt_box_v_newtrain0705.py`
- `code_vjepa_vggt/train0705/bench_ti2v_t2v.py`
- `code_vjepa_vggt/train_v_newtrain.py`
- `code_vjepa_vggt/data/phys_state_dataset.py`
- `code_vjepa_vggt/train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py`
- `code_vjepa_vggt/train0705_kubric_no_gt_box/run_train_stage1b_context_only_no_gt_box_v_newtrain_kubric.sh`

下面这些关键权重或输出目录也已经检查存在：

- `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- `/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors`
- `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`
- `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt`
- `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints`


## 2. 一句话项目定义

可以安全写成下面这句话：

`code_vjepa_vggt` 在 Wan 2.2 TI2V-5B 基础上构建了一条面向物理视频的 object-conditioned 训练链，其中当前项目的正式 Stage1B 口径采用 `train0705_kubric_no_gt_box` 这版实现；它继承了早期 `train0705` no-GT-box 逻辑，并在 DiffSynth-native `v_newtrain` 框架上保留了 JEPA、CoTracker、VGGT、ObjectTubeProjector 和 ObjectConditionAdapter 这条对象条件分支。

这句话来自以下已核实信息：

- `train0705/train_stage1b_context_only_no_gt_box_v_newtrain.py` 的文件头明确写了这是把老 `run_train_teacher_student_stage1b_context_only_no_gt_box.sh` 迁到 `train_v_newtrain.WanTrainingModule`
- `train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py` 的文件头明确表明它沿用同一 no-GT-box Stage1B 结构，只把数据接口切到 Kubric
- `train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py` 和 `train0705_kubric_no_gt_box/infer_stage1b_context_only_no_gt_box_v_newtrain_kubric.py` 都明确列出了与 Kubric 训练一致的 object-conditioning 路径


## 3. 已核实的训练链

### 3.1 最上游基础模型

已核实事实：

- 最上游基础模型不是本项目训练出来的，而是外部 Wan 模型目录：
  - `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- `train0705` 和 Kubric 分支都通过 `--wan_root` 使用这份基础模型。

可安全写法：

- 整条方法线建立在 Wan 2.2 TI2V-5B 基座之上。


### 3.2 第一段：混合数据 LoRA 预训练

已核实事实：

- 训练脚本是 `code_vjepa_vggt/train0419_reference/run_train.sh`
- 脚本里 `--wan_root` 指向 `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- 脚本里 `--dataset_base_path` 指向 `dataset_mix_config.json`
- 脚本注释明确写的是 `OpenVid + MOVI-D + Genesis rigid` 混合数据
- 输出目录是：
  - `/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora`
- `AAAtrain.md` 指向的关键下游初始化权重是：
  - `/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors`
- 这份 `step-010000` 文件已检查存在。

可安全写法：

- 项目先进行了一个 Wan LoRA 预训练阶段，数据源是 OpenVid、MOVI-D 和 Genesis rigid 的混合数据，产出后续 continuation 阶段的初始化 LoRA。


### 3.3 第二段：phys-state continuation LoRA

已核实事实：

- 训练脚本是 `code_vjepa_vggt/train0419_reference/run_train_phys_state_lora_continue.sh`
- 脚本里 `INIT_LORA` 指向上一阶段的：
  - `openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors`
- 这说明这一阶段不是从空 LoRA 开始，而是从上一阶段 LoRA 继续训练
- 输出目录是：
  - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24`
- 当前主线实际引用的 LoRA 是：
  - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`
- 这份 `step-000500` 文件已检查存在。

可安全写法：

- 第二阶段在 phys-state 原始仿真视频上继续训练 LoRA，并把第一阶段的混合数据 LoRA 作为初始化。


### 3.4 第三段：teacher-student Stage1A

已核实事实：

- 训练入口脚本是 `code_vjepa_vggt/run_train_teacher_student_stage1a_gpu67.sh`
- 它实际调用：
  - `-m code_vjepa_vggt.object_token_teacher_student.train_stage1a_full_token`
- 配置文件是：
  - `code_vjepa_vggt/object_token_teacher_student/config_stage1a_full_token_template.yaml`
- 该配置文件中：
  - `output_dir` 是 `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token`
  - `wan_ckpt_dir` 指向 Wan base
  - `init_wan_lora_from_checkpoint` 指向 phys-state continuation 的 `step-000500`
  - `freeze_vae: true`
  - `freeze_text_encoder: true`
  - `freeze_wan_dit: true`
  - `freeze_wan_lora: true`
  - `freeze_object_pooler: false`
- `AAAtrain.md` 中指出 `train0705` 默认引用的 Stage1A 权重不是标准目录下的最新文件，而是：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt`
- 这份 `step_0005000.pt` 已检查存在。

保守归纳：

- Stage1A 的功能是训练 object token builder 相关模块，为后续 Stage1B 提供冻结初始化。
- 这里最安全的具体表述是：`train0705` 后续会把其中的 `object_pooler.*` 和 `object_aux_heads.*` 作为冻结初始化加载。


### 3.5 第四段：老的 Stage1B no-GT-box teacher-student 训练线

已核实事实：

- `AAAtrain.md` 中的“原训练脚本路径1”是：
  - `code_vjepa_vggt/run_train_teacher_student_stage1b_context_only_no_gt_box.sh`
- 这条脚本实际调用：
  - `-m code_vjepa_vggt.object_token_teacher_student.train_stage1b_context_only_no_gt_box`
- 默认 `INIT_FROM` 是：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt`

可安全写法：

- `train0705` 并不是从零重新发明一条 Stage1B 方法线，而是把旧的 teacher-student `stage1b context-only no-GT-box` 思路迁到了新的 DiffSynth-native 框架。


### 3.6 第五段：早期 `train0705` phys-state Stage1B 实现

已核实事实：

- 主训练脚本是：
  - `code_vjepa_vggt/train0705/train_stage1b_context_only_no_gt_box_v_newtrain.py`
- 常用启动脚本有两个：
  - 单卡：`code_vjepa_vggt/train0705/run_train_stage1b_context_only_no_gt_box_v_newtrain0705.sh`
  - 多卡：`code_vjepa_vggt/train0705/run_train_stage1b_context_only_no_gt_box_v_newtrain0705_gpu0235.sh`
- 单卡脚本显式加载三类上游权重：
  - Wan base：`/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
  - 基础 LoRA：`raw_phys_state_wan_lora_continue_576x1024_f24/.../step-000500/checkpoint.safetensors`
  - Stage1A：`stage1a_full_token_old/step_0005000.pt`
- 单卡脚本的主要训练参数是：
  - `--dataset_type phys_state_episode`
  - `--height 512`
  - `--width 896`
  - `--num_frames 24`
  - `--fixed_num_context_frames 8`
  - `--save_steps 500`
  - `--max_checkpoints_keep 10`
  - `--enable_object_branch`
  - `--freeze_non_object_trainables`
  - `--train_object_adapter`
  - `--train_object_dit_branch`
  - `--lambda_main 1.0`
  - `--lambda_track_aux 0.0`
  - `--lambda_box_aux 0.0`
  - `--lambda_depth_aux 0.0`
- 多卡脚本把物理 GPU 固定在 `0,2,3,5`，默认输出目录是：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703`
- 该正式训练目录已经检查存在，`checkpoints` 下当前能看到：
  - `step-002500`
  - `step-003000`
  - `step-003500`
  - `step-004000`
  - `step-004500`
  - `step-005000`
  - `step-005500`
  - `step-006000`
  - `step-006500`
  - `step-007000`

可安全写法：

- 早期 `train0705` 是一条基于 DiffSynth-native `v_newtrain` 框架的 Stage1B 训练线，加载 Wan base、phys-state continuation LoRA 和冻结的 Stage1A token builder，然后只训练 object branch 相关模块。


### 3.7 第六段：当前正式 `train0705_kubric_no_gt_box` Stage1B

已核实事实：

- 当前正式 Stage1B 训练脚本是：
  - `code_vjepa_vggt/train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py`
- 当前正式训练启动脚本是：
  - `code_vjepa_vggt/train0705_kubric_no_gt_box/run_train_stage1b_context_only_no_gt_box_v_newtrain_kubric.sh`
- 当前正式推理入口是：
  - `code_vjepa_vggt/train0705_kubric_no_gt_box/infer_stage1b_context_only_no_gt_box_v_newtrain_kubric.py`
- 训练脚本文件头明确写了：
  - 仍然是 no-GT-box Stage1B
  - query priors 来自 `ViewerGroundingBoxProvider`
  - GT-box aux losses 被去掉
  - 训练集合仍然是 `DiT object-injection branch + ObjectConditionAdapter`
- 启动脚本当前显式加载的上游权重仍然是：
  - Wan base：`/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
  - 基础 LoRA：`raw_phys_state_wan_lora_continue_576x1024_f24/.../step-000500/checkpoint.safetensors`
  - Stage1A：`stage1a_full_token_old/step_0005000.pt`
- 启动脚本当前显式设置：
  - `--dataset_type kubric_no_gt_box`
  - `--num_frames 69`
  - `--fixed_num_context_frames 20`
  - `--ctx_max_length 20`
  - `--context_length_sampling short_biased`
  - `--save_steps 500`
  - `--max_checkpoints_keep 20`
  - `--output_path /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708`
- 推理脚本文件头明确写了它复用 `train0705` 单样本推理入口，但把训练模块切到 Kubric 版本，以保证 loaded weights、viewer grounding、JEPA/CoTracker/VGGT runners 和 object branch 布局与 Kubric 训练完全一致。

可安全写法：

- 当前项目的正式 Stage1B 口径应当写成 `train0705_kubric_no_gt_box` 版本；早期 `train0705` phys-state 版更适合作为方法来源或前序实现，而不是当前主结果的默认代称。


## 4. 已核实的方法描述

### 4.1 当前 Stage1B 的对象条件路径

已核实事实：

- `train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py` 文件头明确写了：
  - 对象 query-point 和 box prior 不来自数据集 GT box
  - 而来自 `ViewerGroundingBoxProvider` 的 viewer-style pseudo-box 流程
- 同一个文件和 Kubric 推理脚本都明确描述了主链路：
  - `context video -> viewer grounding pseudo boxes -> CoTracker / VGGT / JEPA -> ObjectTubeProjector -> ObjectConditionAdapter -> Wan object branch`
- 该训练脚本还明确写了：
  - 取消所有 GT-box 相关 aux loss
  - 只优化注入 `object_context` 后的主 flow-match loss

可安全写法：

- 当前正式 Stage1B 的核心方法不是直接使用人工 GT box 监督，而是通过 viewer-style grounding 产生 pseudo object priors，再结合 JEPA、CoTracker、VGGT 和 object token builder 构造 object context，最后注入 Wan 的 object branch。


### 4.2 当前 Stage1B 真正训练哪些模块

已核实事实：

- `train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py` 文件头明确写了当前可训练集合是：
  - `DiT object-injection branch`
  - `ObjectConditionAdapter`
- `train_v_newtrain.py` 里当 `enable_object_branch` 打开时，会给 DiT 加 object branch，并创建：
  - `JEPAPatchAdapter`
  - `CoTrackerAdapter`
  - `VGGTTrackAdapter`
  - `ObjectTubeProjector`
  - `ObjectAuxHeads`
  - `ObjectConditionAdapter`
- `train_v_newtrain.py` 中 `freeze_non_object_trainables` 会先把 `pipe.dit` 全部参数冻结
- 随后只把这些名字匹配的参数按 `train_object_dit_branch` 打开：
  - `object_embedding`
  - `object_cross_attn`
  - `object_gate`
  - `norm4`
- 运行脚本显式传了：
  - `--freeze_non_object_trainables`
  - `--train_object_adapter`
  - `--train_object_dit_branch`
- 运行脚本没有传 `--train_object_pooler` 或 `--train_object_aux_heads`

可安全写法：

- 当前正式 Stage1B 并不是 end-to-end 训练所有感知模块，而是冻结 Wan 主干和多种辅助感知模块，只训练 Wan 的 object injection 子分支以及 `ObjectConditionAdapter`。


### 4.3 当前 Stage1B 哪些模块是冻结的

已核实事实：

- `train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py` 文件头明确写了冻结对象包括：
  - base Wan DiT
  - raw-phys LoRA
  - VAE
  - text encoder
  - Stage1A token builder
- `train_v_newtrain.py` 中 JEPA、CoTracker、VGGT adapters 默认是 `trainable=False` 或被放入 `FrozenAuxRunner`

可安全写法：

- 当前正式 Stage1B 训练把 Wan base、其基础 LoRA、VAE、文本编码器、Stage1A token builder 以及 JEPA/CoTracker/VGGT 感知分支视为冻结组件。


### 4.4 早期 `train0705` phys-state 版本使用的数据接口

已核实事实：

- `data/phys_state_dataset.py` 定义了 `PhysStateEpisodeDataset`
- 该数据集返回：
  - `video`
  - `context_video`
  - `caption`
  - `context_boxes`
  - `future_boxes`
  - `context_states`
  - `future_states`
- 该数据集会把 `context_frames` 和 `future_frames` 拼成完整视频，再从前 `context_fraction` 比例范围里选择 `num_context_frames`
- 当 `random_context_frames=False` 时，直接取前 `num_context_frames` 帧
- Stage1A 配置文件中：
  - `dataset_type: phys_state_episode`
  - `num_context_frames: 8`
  - `context_fraction: 0.5`
  - `random_context_frames: false`
- `train0705` 单卡/多卡启动脚本中：
  - `--dataset_type phys_state_episode`
  - `--fixed_num_context_frames 8`
  - 没有显式设置 `--ctx_max_length`
- `train_v_newtrain.py` 中 `sample_context_spec()` 的逻辑是：
  - 如果 `ctx_max_length` 被设置，则从完整视频前缀采样
  - 否则如果 `raw_sample["context_video"]` 存在，则从 dataset 提供的 `context_video` pool 继续采样

可安全写法：

- 早期 `train0705` phys-state 版本仍然依赖 `PhysStateEpisodeDataset` 提供完整视频和 context pool；该版本没有启用 `ctx_max_length`，因此它沿用 dataset 提供的 `context_video` 作为上下文候选来源。


### 4.5 当前正式 Kubric Stage1B 的数据接口与 context 采样

已核实事实：

- Kubric 训练脚本显式导入：
  - `KubricNoGTBoxDataset`
- Kubric 启动脚本显式设置：
  - `--dataset_type kubric_no_gt_box`
  - `--fixed_num_context_frames 20`
  - `--ctx_max_length 20`
  - `--context_length_sampling short_biased`
- `train_v_newtrain.py` 当前在 `ctx_max_length` 被设置时，会直接从完整视频前缀采样 context，并记录：
  - `ctx_max_length`
  - `sampled_ctx_last_index`
  - `sampled_ctx_num_frames`

可安全写法：

- 当前正式 Kubric Stage1B 采用 `KubricNoGTBoxDataset`，并启用基于完整视频前缀的 context 长度采样，而不是沿用早期 phys-state 版的固定 context pool 二次采样口径。


## 5. 已核实的推理工具链

### 5.1 当前正式 Stage1B 的单 case 推理

已核实事实：

- 单 case 推理脚本是：
  - `code_vjepa_vggt/train0705_kubric_no_gt_box/infer_stage1b_context_only_no_gt_box_v_newtrain_kubric.py`
- 该脚本要求的核心输入参数是：
  - 它复用 `train0705` 单 case 推理入口的参数体系
- 文件头明确写了它会把 runtime model 切换到 Kubric 训练模块，使 loaded weights、viewer grounding、JEPA/CoTracker/VGGT runners 和 object branch 布局与 Kubric 训练一致

可安全写法：

- 项目已经提供了与当前正式 Kubric Stage1B 训练一致的单样本推理脚本，能够用相同的 object-conditioning 路径重建推理。


### 5.2 VJEPA guidance 是可选推理增强

已核实事实：

- 早期 `train0705` 推理脚本里定义了大量 `--vjepa-*` 参数
- Kubric 推理脚本直接复用该推理入口

可安全写法：

- VJEPA guidance 在当前仓库里是推理期可选能力，而不是当前正式 Stage1B 主训练目标本身。


### 5.3 批量推理和可视化

已核实事实：

- 早期 `train0705` 有批量 v2v 包装脚本：
  - `code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py`
- Kubric 版本额外提供了：
  - `code_vjepa_vggt/train0705_kubric_no_gt_box/wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v.py`
- Kubric 推理包装脚本和单 case 推理脚本都明确写了它们保持 Kubric 训练路径一致
- 可视化脚本是：
  - `code_vjepa_vggt/train0705/inspect_stage1b_prepipe_overlay.py`
- `AAAinfer.md` 中说明这个可视化会展示：
  - viewer grounding boxes
  - query points
  - CoTracker 轨迹等 pre-pipe overlay

可安全写法：

- 项目不只有训练脚本，也包含了与当前正式 Kubric Stage1B 对齐的单样本推理、批量推理和检查工具，便于检查 pseudo boxes、query points 和轨迹条件是否合理。


## 6. 已核实的评测工具链

### 6.1 `bench_ti2v_t2v.py` 当前直接实现的指标

已核实事实：

- `code_vjepa_vggt/train0705/bench_ti2v_t2v.py` 中 `builders` 当前直接实现了这些指标：
  - `wmreward`
  - `videophy2`
  - `cosmos_reason1`
  - `physics_iq`
  - `physics_iq_with_context`
  - `pmf_with_context`

可安全写法：

- 当前 `train0705` 仓库里存在一个专门面向 `ti2v/t2v` 结果目录的指标脚本，覆盖 WMReward、VideoPhy2、Cosmos-Reason1、Physics-IQ 及带 context 的 PMF。


### 6.2 `AAAbench.md` 记录了更宽的 bench 工作流

已核实事实：

- `AAAbench.md` 记录了：
  - 指标回填流程
  - 单独重跑某一指标的方法
  - 汇总 `generated_folder_metric_summary.md/csv` 的脚本
  - 合并 HTML 报告的脚本入口

写作建议：

- 可以写“项目包含成套评测与报告生成脚本”
- 不要在没有表格和数字的情况下写具体分数提升


## 7. 当前正式 Stage1B 与早期 Stage1B 的关系

这里不再把 Kubric 版写成“扩展分支”。按当前项目口径，Kubric 版就是正式 Stage1B；早期 `train0705` phys-state 版是它的前序实现与方法来源。

### 7.1 当前正式 Kubric Stage1B 的基本定位

已核实事实：

- Kubric 主训练脚本是：
  - `code_vjepa_vggt/train0705_kubric_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_kubric.py`
- 该文件头的整体描述和 `train0705` 主线一致，仍然是：
  - no-GT-box
  - viewer grounding pseudo boxes
  - `CoTracker / VGGT / JEPA -> ObjectTubeProjector -> ObjectConditionAdapter`
  - 训练 `DiT object-injection branch + ObjectConditionAdapter`
- 不同点在于它把数据集切换成：
  - `KubricNoGTBoxDataset`
  - `dataset_type kubric_no_gt_box`

可安全写法：

- 当前项目的正式 Stage1B 采用 Kubric/PhyCo 数据分支，核心 object-conditioning 结构与早期 `train0705` no-GT-box 版保持一致，主要变化是数据适配与 context 采样策略。


### 7.2 当前正式 Kubric Stage1B 的训练配置

已核实事实：

- Kubric 启动脚本是：
  - `code_vjepa_vggt/train0705_kubric_no_gt_box/run_train_stage1b_context_only_no_gt_box_v_newtrain_kubric.sh`
- 该脚本当前显式设置：
  - `PYTHONNOUSERSITE=1`
  - `--dataset_type kubric_no_gt_box`
  - `--num_frames 69`
  - `--fixed_num_context_frames 20`
  - `--ctx_max_length 20`
  - `--min_context_frames 0`
  - `--max_context_ratio 1.0`
  - `--context_length_sampling short_biased`
  - `--no_context_ratio 0.0`
  - `--save_steps 500`
  - `--max_checkpoints_keep 20`
  - `--output_path /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_kubric0708`

### 7.3 当前正式 Kubric Stage1B 的 context 采样语义

已核实事实：

- `train_v_newtrain.py` 当前支持在设置 `ctx_max_length` 时：
  - 直接从完整视频前缀采样 context
  - 导出 `ctx_max_length`
  - 导出 `sampled_ctx_last_index`
  - 导出 `sampled_ctx_num_frames`
- Kubric 启动脚本当前把：
  - `ctx_max_length=20`
  - `context_length_sampling=short_biased`
  - `max_context_ratio=1.0`

可安全写法：

- 当前正式 Kubric Stage1B 已经接入“基于完整视频前缀的 context 长度采样”，而不是只依赖固定 context pool。


## 8. GPT 写论文时可以直接使用的主张

下面这些主张都可以直接写，只要措辞保持保守：

- 该项目建立在 Wan 2.2 TI2V-5B 上，并通过多阶段训练形成最终的 physics-oriented Stage1B 模型。
- 上游权重链路是：Wan base -> 混合数据 LoRA 预训练 -> phys-state continuation LoRA -> Stage1A token builder -> 当前正式 Kubric Stage1B。
- 当前正式 Stage1B 沿用了早期 `train0705` no-GT-box 思路，并把它落实到 Kubric/PhyCo 数据分支。
- 当前正式 Stage1B 的 object-conditioning 路径使用 pseudo object priors，而不是依赖 GT boxes。
- 当前正式 Stage1B 只训练 Wan 的 object injection 子分支和 `ObjectConditionAdapter`，大部分 backbone 与感知组件保持冻结。
- 仓库已经提供了与当前正式 Stage1B 路径一致的推理、批量推理、可视化和指标回填脚本。


## 9. GPT 不应该擅自写的内容

下面这些内容当前没有在这份底稿里被充分证实，GPT 不应直接写成事实：

- “方法显著优于某某 baseline”：
  - 这里没有附带最终数字表格。
- “JEPA / VGGT / CoTracker 在当前正式 Stage1B 中被联合训练”：
  - 当前更接近“被冻结或作为 frozen aux 使用”。
- “Stage1A、Stage1B、Kubric 分支是统一 end-to-end 联训”：
  - 当前代码和脚本更像分阶段训练与冻结加载。
- “所有 Stage1B 版本都完全不使用 dataset context pool”：
  - 这只对当前 Kubric 口径成立；早期 `train0705` phys-state 版没有启用 `ctx_max_length`。
- “VJEPA guidance 是训练核心模块”：
  - 当前更安全的说法是推理时可选 guidance。
- “Kubric 分支已经在所有最长 context 长度上做了完整稳定性实验并报告结果”：
  - 这类 runtime 结论需要单独附日志或表格。


## 10. 最适合让 GPT 采用的论文结构

如果要让 GPT 写论文，建议它按照下面的结构写：

### 10.1 问题定义

- 目标是做面向物理场景的视频生成或视频条件生成
- 难点是不依赖 GT box 的情况下，仍然为生成模型提供稳定的对象级条件

### 10.2 方法概述

- 基座：Wan 2.2 TI2V-5B
- 多阶段权重链：
  - 混合数据 LoRA 预训练
  - phys-state continuation LoRA
  - Stage1A object token builder
  - 当前正式 Kubric Stage1B object-conditioned no-GT-box 训练
- Stage1B 主路径：
  - viewer grounding pseudo boxes
  - CoTracker / VGGT / JEPA
  - ObjectTubeProjector
  - ObjectConditionAdapter
  - Wan object branch

### 10.3 训练策略

- 分阶段训练而不是一次性端到端联训
- 冻结大部分 backbone 和辅助感知模块
- 只开放 object injection 子分支和 adapter

### 10.4 推理与评测

- 单 case 推理
- 批量 v2v 推理
- 可选 VJEPA guidance
- 评测脚本支持物理相关指标回填与 HTML 报告生成

### 10.5 扩展方向

- 从早期 phys-state Stage1B 过渡到当前 Kubric/PhyCo Stage1B
- 在当前正式 Stage1B 中引入完整视频前缀 context 采样


## 11. 让 GPT 写作时最稳的一段摘要素材

下面这段可以直接发给 GPT，当作摘要或引言的事实起点：

本项目基于 Wan 2.2 TI2V-5B 构建了一条面向物理视频场景的 object-conditioned 训练链。其权重来源按时间顺序包括：混合数据 LoRA 预训练、phys-state continuation LoRA、teacher-student Stage1A object token builder，以及当前正式采用的 `train0705_kubric_no_gt_box` Stage1B no-GT-box 训练。当前正式 Stage1B 的核心做法不是依赖数据集 GT boxes，而是通过 viewer-style grounding 生成 pseudo object priors，再结合 CoTracker、VGGT、JEPA 和 ObjectTubeProjector 构造 object context，并通过 ObjectConditionAdapter 注入 Wan 的 object branch。该训练线保持分阶段设计，冻结 Wan base、基础 LoRA、VAE、文本编码器、Stage1A token builder 和多种辅助感知模块，只训练 object injection 子分支与 adapter。仓库同时提供了与该训练链一致的推理、批量推理、可视化和物理指标评测脚本。早期 `train0705` phys-state Stage1B 可以作为这条正式 Kubric 训练线的方法来源与前序实现来描述。


## 12. 还需要你手工补给 GPT 的材料

如果你希望 GPT 继续往论文正文推进，最缺的是下面这些外部证据：

- 最终实验表格：
  - 各 checkpoint 或各模型的定量结果
- 基线列表：
  - 要和哪些方法比较
- 最终主模型选择：
  - 当前正式 Kubric Stage1B 应以哪个 checkpoint 作为论文主结果
- 数据集统计：
  - 训练/验证/测试样本数
  - 各数据分支的场景类型和数量
- 图表素材：
  - 代表性生成结果
  - pre-pipe overlay 图
  - 失败案例

没有这些材料时，GPT 最多只能把方法和工程链路写清楚，不能负责任地补出“结果更好”的论文版本。
