# OpenVid -> Wan2.1-1.3B 训练数据准备

## 作用

这条链路用于把新下载的 OpenVid parquet 数据整理成 `wan2.1-1.3b` 当前训练代码可以直接使用的格式。

脚本会完成这几件事：

1. 读取新的 OpenVid parquet 目录。
2. 按 `wan2.1-1.3b` 的 OpenVid LoRA 配方过滤样本。
   - 当前按 `24` 帧门槛过滤。
   - 过滤规则与训练 loader 对齐：`caption 非空`、`视频可解码`、`帧数 >= num_frames`。
3. 导出新的 parquet 子集到新的 `train/` 目录。
4. 生成可直接喂给训练脚本的 config。
5. 用训练侧 `WanTI2VDataset` 做一次 smoke 读取，确认 loader 可正常读取。

## 相关脚本

- 准备脚本：
  [prepare_openvid_wan21_13b_dataset.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/prepare_openvid_wan21_13b_dataset.py)
- 一键 shell：
  [run_prepare_openvid_wan21_13b_dataset.sh](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_prepare_openvid_wan21_13b_dataset.sh)
- 训练脚本：
  [run_train_openvid_mixed_ctx24_384x672_lora_wan21_13b_gpu0235.sh](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_train_openvid_mixed_ctx24_384x672_lora_wan21_13b_gpu0235.sh)

## 一键运行

默认输入路径：

- `/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train`

默认输出路径：

- `/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train_wan21_13b_ready_ctx24`

直接运行：

```bash
sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_prepare_openvid_wan21_13b_dataset.sh
```

指定输入输出路径运行：

```bash
INPUT_ROOT=/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train \
OUTPUT_ROOT=/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train_wan21_13b_ready_ctx24 \
sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_prepare_openvid_wan21_13b_dataset.sh
```

## 调试运行

只扫少量文件做 smoke：

```bash
INPUT_ROOT=/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train \
OUTPUT_ROOT=/data/gaoya/agent-data/outputs/openvid_prepare_smoke_ctx24 \
MAX_FILES=1 \
MAX_ROWS_PER_FILE=10 \
sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_prepare_openvid_wan21_13b_dataset.sh
```

如果只想做过滤导出，不跑 loader smoke：

```bash
INPUT_ROOT=/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train \
OUTPUT_ROOT=/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train_wan21_13b_ready_ctx24 \
SKIP_SMOKE=1 \
sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_prepare_openvid_wan21_13b_dataset.sh
```

## 输出内容

准备完成后，`OUTPUT_ROOT` 下会包含：

- `train/`
  - 过滤后的 parquet 子集，训练 loader 直接读取这个目录。
- `reports/`
  - `summary.json`
  - `accepted_rows.jsonl`
  - `skipped_rows.jsonl`
  - `file_summaries.json`
- `configs/`
  - `openvid_only_config.json`
  - `dataset_mix_config_wan21_13b.json`
- `meta/`
  - `prepare_summary.json`
  - `smoke_summary.json`

## 如何接训练

准备完成后，直接把生成的 mixed config 传给现有训练脚本即可。

```bash
DATASET_CONFIG=/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train_wan21_13b_ready_ctx24/configs/dataset_mix_config_wan21_13b.json \
CUDA_VISIBLE_DEVICES=3,5,6,7 \
sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_train_openvid_mixed_ctx24_384x672_lora_wan21_13b_gpu0235.sh
```

说明：

- 训练脚本现在支持 `DATASET_CONFIG=...` 覆盖默认配置路径。
- 默认 mixed config 会保留原有 `raw_phys_state_video` 和 `genesis_rigid` 条目，只替换其中的 OpenVid 路径。
- 如果只想做纯 OpenVid 训练，可使用：
  [openvid_only_config.json](/data/gaoya/agent-data/outputs/openvid_prepare_smoke_ctx24/configs/openvid_only_config.json)

## 当前配方

- 目标模型：`Wan2.1-T2V-1.3B`
- OpenVid 训练 clip 长度：`24` 帧
- 训练分辨率：`384x672`
- 当前过滤门槛：`num_frames >= 24`
