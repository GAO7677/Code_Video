# 1. Wan 2.2 TI2V 5B 训练 V2V 当前摘要

当前以以下代码为准：

- 训练入口：[run_train.sh](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/run_train.sh:1)
- 训练主逻辑：[train.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/train.py:1)
- 数据封装：[dataset.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/dataset.py:1)
- context 逻辑：[context_wan.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/context_wan.py:1)
- 推理与 benchmark：[batch_eval_lora.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/batch_eval_lora.py:1)
- validation + VBench：[run_validation_vbench.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/run_validation_vbench.py:1)

## 1.1 训练目标

- 基座模型：`Wan 2.2 TI2V 5B`
- 任务：多帧 context 条件下的 V2V / context-aware video continuation
- 训练形式：LoRA

## 1.2 训练数据

混合数据配置：

- 配置文件：[dataset_mix_config.json](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/dataset_mix_config.json:1)
- 数据集：`OpenVidParquetDataset`、`MoviDTFRecordDataset`、`GenesisRigidDataset`

当前配比：

- OpenVid：`repeat=1`
- MOVI-D train：`repeat=2`
- Genesis rigid train：`repeat=2`

当前有效样本统计：

- OpenVid：`65975`
- MOVI-D train：`6166`
- Genesis rigid train：`4876`
- total effective：`77017`

统一训练目标：

- 分辨率：`384x672`
- 目标帧数：`24`

数据侧处理：

- OpenVid：至少 `24` 帧，随机截连续 `24` 帧，`crop + resize`
- MOVI-D：`24` 帧，按比例后 `pad` 到 `384x672`
- Genesis rigid：保留原始 `13/16` 帧，`crop + resize`

Genesis 训练规则：

- held-out object groups：`8`
- held-out test samples：`205`
- 训练不看到 held-out 对象

## 1.3 Context 训练策略

当前使用 `mixed_modes` 采样，定义见 [train.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/train.py:289)。

采样概率：

- `prefix`：`55%`
- `first_frame`：`20%`
- `sparse`：`15%`
- `random`：`5%`
- `text_only`：`5%`

在当前 `24` 帧训练下，参考前缀集合缩放后实际为：

- `prefix`：`{1, 2, 4, 6, 8}`
- `sparse`：`{2, 4, 6, 8}`
- `random`：`{2, 4, 6, 8}`

模式含义：

- `prefix`：前 `K` 帧
- `first_frame`：仅首帧
- `sparse`：全段均匀采样若干帧
- `random`：包含首帧，再随机补若干帧
- `text_only`：无视觉条件

训练核心约束：

- context latent 保持 clean
- 非 context latent 加噪并参与 loss
- Wan 时间维要求 `4n+1`
- 请求 `24` 帧时，内部会对齐到 `25`

## 1.4 当前训练配置

启动脚本：[run_train.sh](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/run_train.sh:1)

主要配置：

- GPU：`CUDA_VISIBLE_DEVICES=0,1,2,3`
- per-GPU batch：`1`
- `gradient_accumulation_steps=4`
- effective batch：`16`
- 分辨率：`384x672`
- 目标帧数：`24`
- `learning_rate=1e-4`
- `weight_decay=0.01`
- `max_train_steps=8000`
- `num_epochs=10`
- `save_steps=1000`
- `dataset_num_workers=0`
- `wandb_mode=offline`

LoRA：

- `lora_base_model=dit`
- `lora_target_modules=q,k,v,o,ffn.0,ffn.2`
- `lora_rank=32`

额外条件：

- `extra_inputs=input_image`
- 即非 `text_only` 情况下，额外给首帧图像

输出目录：

- `/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora`

恢复策略：

- `run_train.sh` / `train.py` 会自动寻找最新可恢复状态
- checkpoint 统一使用零填充 step 命名，如 `step-001000`

## 1.5 训练期 benchmark

固定 benchmark 每 `1000` steps` 运行一次。

样本列表：

- [benchmark_meta_json_paths_fixed24.txt](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_fixed24.txt:1)

样本组成：

- OpenVid：`12`
- MOVI-D：`6`
- Genesis：`6`
- total：`24`

生成参数：

- 分辨率：`384x672`
- 输出帧数：`24`
- `context_frames=8`
- `fps=8`
- `num_inference_steps=50`
- `cfg_scale=5.0`
- `seed=42`

说明：

- Wan 内部会把 `24` 帧对齐到 `25`
- 最终保存仍为前 `24` 帧

## 1.6 Validation

validation 每 `2000` steps 运行一次。

样本列表：

- [benchmark_meta_json_paths_validation100.txt](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_validation100.txt:1)

样本组成：

- OpenVid：`50`
- MOVI-D：`25`
- Genesis：`25`
- total：`100`

生成参数：

- 分辨率：`384x672`
- 输出帧数：`24`
- `fps=8`
- `num_inference_steps=50`
- `cfg_scale=5.0`
- context sweep：`0,1,2,4,6,8`

输出包括：

- 各 context 设置的生成结果
- `summary.json`
- `context_curve.csv`
- VBench short 指标
- GT-based future 指标：`future_psnr / future_ssim / future_lpips / future_dino`
