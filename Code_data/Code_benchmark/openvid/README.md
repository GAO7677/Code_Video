# OpenVid Val Utilities

这个目录用于读取并整理 `mvp-lab/OpenVidHD-0.4M-720p-48fps` 的 `val` split。

## 原始数据格式

`val/` 目录下是多个 parquet shard，每条样本原始上只有两列：

- `info`: `binary`，里面是 `torch.load(...)` 可反序列化出的 Python `dict`
- `raw_video`: `binary`，对应样本的原始 mp4 字节流

当前实际观测到的 `info` 字段为：

- `video`
- `caption`
- `aesthetic score`
- `motion score`
- `temporal consistency score`
- `camera motion`
- `frame`
- `fps`
- `seconds`

训练侧在 [/home/gaoya/Code_Video/Code_data/Code_train/train_0419/dataset.py](/home/gaoya/Code_Video/Code_data/Code_train/train_0419/dataset.py) 里会把它读成：

```python
{
  "video": list[PIL.Image.Image],
  "prompt": str,
}
```

也就是：

- `prompt <- info["caption"]`
- `video <- decode(raw_video)` 后抽帧并 resize/crop 得到的帧列表

## 处理脚本

脚本：[/home/gaoya/Code_Video/Code_data/Code_benchmark/openvid/prepare_openvid_val.py](/home/gaoya/Code_Video/Code_data/Code_benchmark/openvid/prepare_openvid_val.py)

默认会把 `val/*.parquet` 整理成：

- `manifest.jsonl`: 每行一个样本的规范化 metadata
- `summary.json`: 样本数和输出位置
- `dataset_format.json`: 原始 schema 和 manifest 字段说明

默认命令：

```bash
source /home/gaoya/miniconda3/bin/activate flux
python /home/gaoya/Code_Video/Code_data/Code_benchmark/openvid/prepare_openvid_val.py --overwrite
```

如果还想把 parquet 里的 mp4 字节流直接导出成视频文件：

```bash
source /home/gaoya/miniconda3/bin/activate flux
python /home/gaoya/Code_Video/Code_data/Code_benchmark/openvid/prepare_openvid_val.py \
  --overwrite \
  --export-videos
```

规范化后的 `manifest.jsonl` 每条记录包含：

- `sample_id`
- `dataset_id`
- `split`
- `global_index`
- `parquet_file`
- `parquet_path`
- `parquet_row_index`
- `source_video_name`
- `caption`
- `fps`
- `num_frames`
- `duration_seconds`
- `camera_motion`
- `aesthetic_score`
- `motion_score`
- `temporal_consistency_score`
- `raw_video_num_bytes`
- `raw_video_path`
- `dataset_root`
