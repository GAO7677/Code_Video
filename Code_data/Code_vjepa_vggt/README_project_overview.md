# Code VJEPA VGGT 项目说明

## 项目做什么

这个项目的目标是把 `context video` 作为输入条件，训练一个基于 `Wan2.2 TI2V-5B` 的视频生成模型，让模型看到前面一段视频以后继续生成后续视频。

整体思路不是直接把整段视频和文本丢给 Wan，而是先从上下文视频里提取更结构化的条件信息，再把这些条件送给 Wan 的 DiT 主干。现在用到的主要条件来源有三类：

- 文本条件：caption
- 局部视觉条件：V-JEPA patch token
- 物体运动与几何条件：VGGT tracks、visibility、confidence、depth、world_points

训练时，`context video` 会同时送入 V-JEPA、VGGT、VAE 和文本编码器，后面通过 `ObjectTubeProjector` 和 `ContextTokenFuser` 把这些信息融合成 Wan 可消费的条件 token，然后只训练 Wan DiT 上的 LoRA 参数和少量条件融合模块。

## 主要模型

- `Wan2.2 TI2V-5B`
  - 用作主生成模型
  - 当前不是全量微调，而是 LoRA 微调
- `V-JEPA2`
  - 用作上下文视频的 patch 级表征提取器
  - 当前冻结，只做前向
- `VGGT`
  - 用作 query-point tracking 和几何提取器
  - 输出 tracks、visibility、confidence、depth、world_points
  - 当前冻结，只做前向
- `SAM2 / GroundingDINO`
  - 用于生成 query priors
  - 当前作为前处理和先验，不参与训练

## 代码放在哪里

- 项目根目录：
  - [Code_Video/Code_data/Code_vjepa_vggt](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt)
- 主代码目录：
  - [code_vjepa_vggt](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt)

关键代码文件如下：

- 训练入口：
  - [train_context_video_wan.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_context_video_wan.py)
- 主训练逻辑：
  - [context_video_trainer.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/trainers/context_video_trainer.py)
- 训练循环与 wandb 记录：
  - [runner.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/training/runner.py)
- Wan 封装：
  - [wan_context_model.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/models/wan_context_model.py)
- 条件融合模块：
  - [object_tokens.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/models/object_tokens.py)
  - [context_fuser.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/models/context_fuser.py)
- 数据集读取：
  - [phys_state_dataset.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/data/phys_state_dataset.py)
- 推理脚本：
  - [infer_context_video_wan.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py)
  - [infer_context_video_wan_testset.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan_testset.py)

## 数据集和权重放在哪里

训练数据当前主要用的是：

- 训练数据根目录：
  - `/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500`

这个目录下面按 `train / val / test` 分子集，每个样本通常由：

- 一个 `.json` 元数据文件
- 一个同名 `.npz` 张量文件

当前训练用的是 `train` 子集，测试脚本可直接读取 `val` 或 `test` 子集。

相关模型权重目录如下：

- Wan：
  - `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B`
- V-JEPA2：
  - `/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384`
- VGGT：
  - `/data/gaoya/ckpt/facebook-VGGT-1B`
- CoTracker：
  - `/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth`

## 训练配置

当前主要训练配置文件是：

- [train_0613pybullet_wan_lora_gpu67.yaml](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_wan_lora_gpu67.yaml)

这份配置的核心含义如下：

- 主模型：
  - `wan_task: ti2v-5B`
  - 使用 Wan2.2 TI2V-5B 作为视频生成主干
- 冻结策略：
  - `freeze_vae: true`
  - `freeze_text_encoder: true`
  - `freeze_wan_dit: true`
  - 这里的 `freeze_wan_dit: true` 指原始 DiT 权重冻结，但 LoRA 参数仍然可训练
- LoRA：
  - `wan_lora_rank: 32`
  - `wan_lora_alpha: 32`
  - `wan_lora_dropout: 0.0`
- 上下文长度：
  - `num_context_frames: 12`
  - `context_fraction: 0.5`
  - `random_context_frames: false`
  - 当前使用前缀上下文，不是随机抽帧
- 分辨率：
  - `resolution: [704, 1280]`
- batch size：
  - `batch_size: 2`
- 优化器：
  - `lr: 2e-5`
  - `weight_decay: 0.01`
  - `max_grad_norm: 0.5`
  - `mixed_precision: bf16`
- loss：
  - `lambda_main: 1.0`
  - `lambda_vggt_align: 0.0`
  - `lambda_vggt_iou: 0.0`
  - 当前主要先保证主 flow-matching loss 稳定，VGGT box 辅助损失暂时关闭
- 日志和保存：
  - `log_every: 1`
  - `save_every: 200`
  - `use_wandb: true`

## 现在的训练流程

一条样本进入训练后，大致经过下面几步：

1. 读取完整视频 `video`，并切出上下文视频 `context_video`
2. 文本 `caption` 送入 Wan text encoder，得到文本条件
3. `video` 和 `context_video` 分别送入 Wan VAE，得到完整 latent 和上下文 latent
4. `context_video` 送入 V-JEPA，得到 patch tokens
5. `context_video` 在 query priors 的引导下送入 VGGT，得到 tracks、visibility、confidence、depth、world_points
6. `ObjectTubeProjector` 沿 tracks 从 JEPA token、latent、几何特征里采样并池化，得到 object tokens
7. `ContextTokenFuser` 把文本 token 和 object tokens 融合成 `fused_context`
8. 训练时对完整 latent 加噪，但 context 部分保持干净，只对 future 部分做预测
9. Wan DiT 在 `fused_context` 条件下预测 future latent 的 flow-matching target
10. 计算主 loss，并更新 LoRA 与条件融合模块

## 当前可训练模块

当前真正参与训练的主要是：

- Wan DiT 上的 LoRA 参数
- `ObjectTubeProjector`
- `ContextTokenFuser`

当前冻结不训练的主要是：

- Wan VAE
- Wan text encoder
- V-JEPA2
- VGGT
- SAM2 / GroundingDINO

## 输出结果放在哪里

训练输出目录当前是：

- checkpoint：
  - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/pybullet0613_wan_lora_gpu67`
- wandb 日志：
  - `/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/logs/wandb`

测试输出目录当前默认是：

- `/data/gaoya/AAA_test_video/0529/vjepa_vggt/test`

`infer_context_video_wan_testset.py` 现在会把每个 case 的结果扁平保存为：

- `basename.json`
- `basename_input.mp4`
- `basename_input_context.mp4`
- `basename.mp4`

其中 json 里目前只保留：

- `checkpoint_dir`
- `seed`
- `input_caption`
- `input_video`
- `input_context_video`
- `output_video`

## 常用命令

训练启动命令：

```bash
CUDA_VISIBLE_DEVICES=6,7 PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python -m accelerate.commands.launch \
  --multi_gpu --num_processes 2 \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_context_video_wan.py \
  --config /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/configs/train_0613pybullet_wan_lora_gpu67.yaml


```

测试集推理命令：

```bash
CUDA_VISIBLE_DEVICES=2
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan.py \
  --checkpoint-dir /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/pybullet0613_wan_lora_gpu67/step_0000200.pt \
  --prompt "A sphere rolls after landing on the platform and leaves the support surface, testing support switching." \
  --context-video /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500/val/F5_drop_support/sample_000335/context_video.mp4 \
  --output-dir /data/gaoya/AAA_test_video/0529/vjepa_vggt/tmp/infer_context_video_wan \
  --num-frames 24 \
  --sampling-mode prefix \
  --save-raw


`--checkpoint-dir` 现在既可以传权重目录，也可以直接传单个 `step_XXXXXXX.pt` 文件的绝对路径；如果传目录，脚本会自动加载该目录下最新的 `step_*.pt`。

PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_context_video_wan_testset.py \
  --checkpoint-dir /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/pybullet0613_wan_lora_gpu67 \
  --split test \
  --dataset-root /data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500 \
  --output-dir /data/gaoya/AAA_test_video/0529/vjepa_vggt/test \
  --num-cases 4 \
  --save-raw


```
