# WISA Stage1b 训练 / 检查脚本

## 1. 目录与脚本

训练主脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_wisa_no_gt_box/train_stage1b_context_only_no_gt_box_v_newtrain_wisa.py
```

正式训练启动脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_wisa_no_gt_box/run_train_stage1b_context_only_no_gt_box_v_newtrain_wisa.sh
```

smoke 启动脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_wisa_no_gt_box/run_smoke_stage1b_context_only_no_gt_box_v_newtrain_wisa.sh
```

前向 + aux overlay 检查脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_wisa_no_gt_box/inspect_wisa_train_forward_aux_overlay.py
```

预处理 overlay 输入抽样脚本：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_wisa_no_gt_box/sample_wisa_overlay_inputs.py
```

数据集 adapter：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/data/wisa_no_gt_box_dataset.py
```


## 2. 已下载的 HF metadata

实际下载命令：

```bash
env -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY -u all_proxy -u ALL_PROXY \
PATH=/home/gaoya/miniconda3/envs/wan-cu128/bin:$PATH \
HF_ENDPOINT=https://hf-mirror.com \
HF_TOKEN='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' \
hf download qihoo360/WISA-80K \
  --repo-type dataset \
  --local-dir /data/gaoya/dataset/qihoo360-WISA-80K
```

当前落盘目录：

```text
/data/gaoya/dataset/qihoo360-WISA-80K
```

注意：

- 这个 HF dataset repo 当前只有 `README.md`、`.gitattributes`、`data/wisa-80k.json`。
- 没有实际视频文件，所以训练前还需要把 mp4 放到：

```text
/data/gaoya/dataset/qihoo360-WISA-80K/videos
```

- 或者运行时显式指定 `WISA_VIDEOS_ROOT=/path/to/mp4_dir`。


## 3. 训练默认路径

metadata 根目录：

```text
/data/gaoya/dataset/qihoo360-WISA-80K
```

默认视频目录：

```text
/data/gaoya/dataset/qihoo360-WISA-80K/videos
```

默认 cache：

```text
/data/gaoya/agent-data/cache/wisa_no_gt_box_dataset
```

默认 checkpoint 输出：

```text
/data/gaoya/agent-data/checkpoints/train_stage1b_diffsynth_native0705_wisa
```


## 4. 启动示例

正式训练：

```bash
GPU=5 \
WISA_VIDEOS_ROOT=/data/gaoya/dataset/qihoo360-WISA-80K/videos \
OUTPUT_DIR=/data/gaoya/agent-data/checkpoints/stage1b_wisa_no_gt_box_gpu5 \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_wisa_no_gt_box/run_train_stage1b_context_only_no_gt_box_v_newtrain_wisa.sh
```

smoke：

```bash
GPU=5 \
WISA_VIDEOS_ROOT=/data/gaoya/dataset/qihoo360-WISA-80K/videos \
WISA_INIT_SCAN_LIMIT=64 \
OUTPUT_DIR=/data/gaoya/agent-data/checkpoints/stage1b_wisa_no_gt_box_smoke_gpu5 \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_wisa_no_gt_box/run_smoke_stage1b_context_only_no_gt_box_v_newtrain_wisa.sh
```

抽取 overlay 输入：

```bash
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705_wisa_no_gt_box/sample_wisa_overlay_inputs.py \
  --dataset-root /data/gaoya/dataset/qihoo360-WISA-80K \
  --videos-root /data/gaoya/dataset/qihoo360-WISA-80K/videos \
  --count 10
```


## 5. 适配说明

- 这条分支沿用 `train0705_kubric_no_gt_box` 的 no-GT-box 训练逻辑。
- 主要改动在于数据 adapter：WISA 使用单个 `wisa-80k.json` 建索引，并按 `video_name` 去视频目录匹配 mp4。
- 仍然使用稳定哈希划分 `train/val/test`。
- 如果视频尚未补齐，dataset 初始化会直接报错提示 `videos_root` 缺失，这是预期行为。
