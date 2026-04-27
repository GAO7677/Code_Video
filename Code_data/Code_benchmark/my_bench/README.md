# My Bench

这个目录是给 `prompt + generated video + optional context frames` 的自定义 benchmark harness，专门适配你现在这种：

- 文本条件仍然保留
- 额外给前几帧 context latent
- 想同时看通用视频质量、TI2V 式锚定能力、以及 continuation 专项指标

## 支持的输入

统一输入 manifest。每条样本至少需要：

- `prompt`
- `video_path`

可选字段：

- `context_frames_dir`
- `context_frame_paths`
- `image_path`
- `gt_video_path`
- `generated_start_frame`
- `gt_start_frame`

推荐 manifest 格式是 `jsonl`。示例见 [example_manifest.jsonl](/home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/manifests/example_manifest.jsonl)。

### 字段说明

- `prompt`: 文本提示词。
- `video_path`: 你模型生成的视频。默认假设这是“只包含未来生成段”的视频。
- `context_frames_dir` / `context_frame_paths`: 前几帧 context。跑 I2V 指标时，会自动取最后一帧作为参考图 `I`。
- `image_path`: 如果你已经单独存好了参考图，也可以直接给这个字段，不必再给 context frames。
- `gt_video_path`: continuation 指标的 ground-truth 视频，可选。
- `generated_start_frame`: 如果 `video_path` 不是纯 future，而是“包含 context + future”的整段视频，可以用这个字段指定从哪一帧开始算生成段。
- `gt_start_frame`: 如果 `gt_video_path` 是整段真实视频，可以用这个字段指定未来段起点；常见设置就是 `len(context_frames)`。

## 已提供的脚本

- [run_vbench.py](/home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/scripts/run_vbench.py)
  运行原始 VBench 短视频质量评测。
- [run_i2v.py](/home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/scripts/run_i2v.py)
  运行 VBench-I2V。自定义输入模式下，会用最后一个 context frame 当参考图。
- [run_long.py](/home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/scripts/run_long.py)
  运行 VBench-Long。
- [run_continuation.py](/home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/scripts/run_continuation.py)
  运行 continuation 专项指标。
- [run_all.py](/home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/scripts/run_all.py)
  顺序跑推荐组合。
- [link_manual_weights.py](/home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/scripts/link_manual_weights.py)
  把你手头已有的权重路径软链接到 VBench 期望的 cache 布局。

## 推荐先跑哪些指标

### 1. 原始 VBench

默认会跑这些核心维度：

- `subject_consistency`
- `background_consistency`
- `motion_smoothness`
- `temporal_flickering`
- `dynamic_degree`
- `imaging_quality`
- `aesthetic_quality`
- `overall_consistency`
- `temporal_style`

### 2. VBench-I2V

默认会跑：

- `i2v_subject`
- `i2v_background`
- `camera_motion`
- `subject_consistency`
- `background_consistency`
- `motion_smoothness`
- `dynamic_degree`
- `imaging_quality`
- `aesthetic_quality`

这里 `i2v_subject` / `i2v_background` 会自动使用最后一个 context frame 作为参考图。

### 3. VBench-Long

默认会跑：

- `subject_consistency`
- `background_consistency`
- `motion_smoothness`
- `temporal_flickering`
- `dynamic_degree`
- `imaging_quality`
- `aesthetic_quality`

### 4. Continuation 专项

当前实现了两类：

- `boundary_*`: 最后一张 context frame 和第一张生成帧之间的连续性
- `future_*`: 生成 future 和 GT future 的逐帧 PSNR / SSIM / MSE

如果你安装了 `lpips`，还可以打开 LPIPS。

## 使用方式

### 1. 复制配置

先复制一份示例配置：

```bash
cp /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/configs/paths.example.yaml \
   /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/configs/paths.local.yaml
```

然后填：

- `vbench_repo_root`
- `work_root`
- `vbench_cache_dir`
- `vbench2_cache_dir`
- `weights_paths`
- `dataset_paths`

这里我已经把示例配置的 cache 默认值改成了 `work_root/cache/...`，不会默认写到 `~/.cache`。

### 2. 跑短视频 VBench

```bash
python3 /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/scripts/run_vbench.py \
  --config /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/configs/paths.local.yaml \
  --manifest /path/to/manifest.jsonl \
  --output-dir /path/to/output/vbench_short
```

### 3. 跑 I2V 指标

```bash
python3 /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/scripts/run_i2v.py \
  --config /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/configs/paths.local.yaml \
  --manifest /path/to/manifest.jsonl \
  --output-dir /path/to/output/vbench_i2v \
  --resolution 1-1
```

### 4. 跑长视频指标

```bash
python3 /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/scripts/run_long.py \
  --config /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/configs/paths.local.yaml \
  --manifest /path/to/manifest.jsonl \
  --output-dir /path/to/output/vbench_long
```

### 5. 跑 continuation 指标

```bash
python3 /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/scripts/run_continuation.py \
  --config /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/configs/paths.local.yaml \
  --manifest /path/to/manifest.jsonl \
  --output-dir /path/to/output/continuation
```

### 6. 一次全跑

```bash
python3 /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/scripts/run_all.py \
  --config /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/configs/paths.local.yaml \
  --manifest /path/to/manifest.jsonl \
  --output-dir /path/to/output/all \
  --resolution 1-1
```

## 需要下载哪些权重

### A. 只跑当前默认核心维度时，最关键的权重

- `clip_vit_b32`
  供 `background_consistency` 等维度使用。
- `clip_vit_l14`
  供 `aesthetic_quality` 使用。
- `dino_vitb16`
  供 `subject_consistency` 使用。
- `dino_repo_dir`
  如果你开 `load_ckpt_from_local: true`，原始 VBench 还需要 DINO repo 本地目录。
- `amt_s`
  供 `motion_smoothness` 使用。
- `raft_things`
  供 `dynamic_degree` 使用。
- `musiq_spaq`
  供 `imaging_quality` 使用。

### A1. 这些键对应的 repo-id / 官方来源

- `clip_vit_b32`
  来源：`openai/CLIP`
  实际模型名：`ViT-B/32`
- `clip_vit_l14`
  来源：`openai/CLIP`
  实际模型名：`ViT-L/14`
- `dino_vitb16`
  来源：`facebookresearch/dino`
  权重文件：`dino_vitbase16_pretrain.pth`
- `dino_repo_dir`
  来源仓库：`https://github.com/facebookresearch/dino`
- `amt_s`
  来源：`lalala125/AMT`
  权重文件：`amt-s.pth`
- `raft_things`
  来源：RAFT 官方 Things checkpoint
  VBench 代码里是通过 Dropbox 的 `models.zip` 下载，不是 Hugging Face repo-id
- `musiq_spaq`
  来源：`chaofengc/IQA-PyTorch`
  release 文件：`musiq_spaq_ckpt-358bb6af.pth`

### B. 如果你扩展去跑更多语义维度，还需要

- `umt_human_action`
  供 `human_action` 使用。
- `grit_densecap`
  供 `object_class` / `multiple_objects` / `color` / `spatial_relationship` 使用。
- `tag2text_swin`
  供 `scene` 使用。
- `viclip_pretrain`
  供 `overall_consistency` / `temporal_style` 等文本视频对齐维度使用。

### B1. 扩展键对应的 repo-id / 官方来源

- `umt_human_action`
  来源：`OpenGVLab/VBench_Used_Models`
  文件：`l16_ptk710_ftk710_ftk400_f16_res224.pth`
- `grit_densecap`
  来源：`OpenGVLab/VBench_Used_Models`
  文件：`grit_b_densecap_objectdet.pth`
- `tag2text_swin`
  来源：`xinyu1205/recognize-anything`
  文件：`tag2text_swin_14m.pth`
- `viclip_pretrain`
  来源：`OpenGVLab/VBench_Used_Models`
  文件：`ViClip-InternVid-10M-FLT.pth`

### B2. VBench-2.0 对应键

- `clip_vit_b32_vbench2`
  来源：`openai/CLIP`
- `clip_vit_l14_vbench2`
  来源：`openai/CLIP`
- `dino_vitb16_vbench2`
  来源：`facebookresearch/dino`
- `dino_repo_dir_vbench2`
  来源仓库：`https://github.com/facebookresearch/dino`
- `amt_s_vbench2`
  来源：`lalala125/AMT`
- `raft_things_vbench2`
  来源：RAFT 官方 Things checkpoint
- `musiq_spaq_vbench2`
  来源：`chaofengc/IQA-PyTorch`

### C. I2V 的额外注意点

- 自定义输入模式不需要下载官方 I2V image suite。
- 标准 benchmark 模式才需要 `vbench2_beta_i2v/data/crop/{ratio}` 下面那套图片。

### D. continuation 指标

- `run_continuation.py` 不依赖 VBench 模型。
- 如果要开启 LPIPS，需要额外安装：

```bash
pip install lpips
```

## 你给我路径时，优先给这些

如果你后面把本地路径发给我，我最希望先拿到：

- `clip_vit_b32`
- `clip_vit_l14`
- `dino_vitb16`
- `dino_repo_dir`
- `amt_s`
- `raft_things`
- `musiq_spaq`

把这些填进配置后，可以先运行：

```bash
python3 /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/scripts/link_manual_weights.py \
  --config /home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench/configs/paths.local.yaml
```

它会把你的现成权重软链接到 VBench 默认 cache 路径。

## 额外说明

- 这个 harness 当前是“输入已经生成好的视频”，不负责调用你的 Wan2.2 推理。
- 自定义 I2V 评测时，脚本会把 `context_frames` 的最后一帧自动当作参考图。
- 如果一个样本没有 context frames，但你还要跑 I2V 指标，可以单独填 `image_path`。
- 默认假设 `video_path` 是“未来生成段”。如果你保存的是“context + future”的整段视频，请在 manifest 里填 `generated_start_frame`。
