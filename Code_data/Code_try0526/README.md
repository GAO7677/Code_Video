# Try0526 Candidate Reranking

这个目录用于做“同一 `context video + prompt` 下，多候选生成、三路物理分数打分、重排序选择最优视频”的尝试。

当前实现目标：

- 统一生成候选：
  - `Wan` context-aware
  - `VACE` video-conditioned
  - `CogVideoX` image-to-video（默认退化为只使用 `context` 首帧）
- 统一三类分数：
  - `latent_motion`
  - `geometry_proxy`
  - `jepa_predictive`
- 统一输出：
  - 每个 candidate 的视频、原始分数、归一化分数、总分
  - 最终最佳视频
  - 运行清单和可复现配置

## 目录约定

代码：

- `run_rerank_pipeline.py`
- `rerank_video/`

运行输出默认写到：

- `/data/gaoya/AAA_test_video/Output_try0526/runs/<run_name>`

临时文件默认写到：

- `/data/gaoya/AAA_test_video/Output_try0526/tmp/<run_name>`

## 当前几何分数说明

当前 `geometry_proxy` 先实现为“PDI-style 近似几何审计”：

- 用最后一帧 context 作为 anchor
- 用帧差分提取主要运动目标
- 估计 bbox / area / centroid / 伪深度
- 计算：
  - scale-depth consistency
  - trajectory smoothness
  - local rigidity stability

这不是完整 PDI-Bench，但接口已经预留成后端式设计，后续可以补全成 `full_pdi`。

## 示例

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_try0526/run_rerank_pipeline.py \
  --config /home/gaoya/Code_Video/Code_data/Code_try0526/example_config.json
```

建议直接在 `wan` 环境里运行，因为当前实现依赖：

- `torch`
- `diffusers`
- `transformers`
- 你本地的 `DiffSynth-Studio`
- 你本地的 `vjepa2-main`
