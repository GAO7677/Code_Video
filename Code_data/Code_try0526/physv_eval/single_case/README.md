# physv_eval.single_case

这一个目录放的是单样本评测入口。

设计目标：

- 每个指标都提供一个独立的 `score_case(...)` 函数，供其他脚本直接 `import`
- 每个指标都提供一个 `python -m physv_eval.single_case.<name>` 的命令行入口
- 单 case 输入格式尽量统一
- 命令行默认直接输出 Python `dict`

当前保留的指标入口：

- `pdi`
- `wmreward`
- `proxy`
- `videophy2`
- `phyground`
- `cosmos_reason1`
- `physics_iq`
- `pmf`
- `vbench`
- `vbench2`

## 通用输入格式

单 case 入口统一支持两类输入：

### 1. 直接传视频

```bash
python -m physv_eval.single_case.wmreward --video /path/to/video.mp4
```

### 2. 传 case JSON

```bash
python -m physv_eval.single_case.pdi --input-json /path/to/case.json
```

通用字段约定：

- 视频路径字段：`video`、`video_path`、`output_video`、`paths.output_video_path`
- 上下文视频字段：`context_video`、`context_video_path`、`paths.context_video_path`
- 参考视频字段：`source_video`、`source_video_path`、`source`
- caption 字段：`caption`、`text_prompt`、`prompt`、`description`、`scenario`、`experiment`、`target_object`、`clip_name`、`name`
- 规则字段：`rule`、`physical_law`、`law`

最小 case JSON 示例：

```json
{
  "video": "/path/to/video.mp4",
  "caption": "a ball rolls down a ramp"
}
```

命令行输出说明：

- stdout 默认输出 Python `dict`
- 如果传了 `--output-json`，会额外把结果写成 JSON 文件

## 代码复用方式

所有模块都优先通过 `score_case(...)` 复用。

示例：

```python
from physv_eval.single_case.videophy2 import score_case

result = score_case(
    "/path/to/video.mp4",
    task="pc",
)
print(result)
```

`case` 支持：

- `str`
- `Path`
- `dict`
- `EvalCase`

## 1. PDI

入口：

- 模块：`physv_eval.single_case.pdi`
- 函数：`score_case(case, text_query=None, refresh=False, runner=None)`

指标定义：

- 对齐官方 PDI-Bench 的几何一致性 / 结构一致性评测
- 更适合看分割、跟踪、深度、投影关系是否稳定
- 它不是纯物理规律打分，仍会受到画面内容和跟踪质量影响

指标属性：

- 类型：连续值
- 主方向：`pdi_score` 越低越好
- 子项：
  - `scale_component`
  - `traj_component`
  - `epsilon_rigidity`
  - `vp_component`

函数返回：

```python
{
    "pdi_score": float | None,
    "grade": str | None,
    "scale_component": float | None,
    "traj_component": float | None,
    "epsilon_rigidity": float | None,
    "rigidity_strategy": str | None,
    "vp_component": float | None,
    "ra_math_pass": str | None,
    "ra_ground_rmse": float | None,
    "ra_scale_jump": float | None,
    "ra_reproj_err": float | None,
    "ra_overall_pass": str | None,
    "raw_report_path": str,
}
```

运行示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.pdi \
  --video /path/to/video.mp4 \
  --caption "a ball rolls on a ramp"
```

## 2. WMReward

入口：

- 模块：`physv_eval.single_case.wmreward`
- 函数：`score_case(case, runner=None)`

指标定义：

- 对齐官方 `compute_wmreward.py` 默认口径
- 本质上是 V-JEPA 的滑窗未来预测误差
- 更接近短时可预测性，不是显式物理规则判定器

指标属性：

- 类型：连续值
- 主字段：
  - `surprise` 越低越好
  - `similarity = 1 - surprise`，越高越好

函数返回：

```python
{
    "surprise": float,
    "similarity": float,
    "method": str,
    "model": str,
    "img_size": int,
    "window_size": int,
    "context_frames": int,
    "stride": int,
    "seed": int,
}
```

运行示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.wmreward \
  --video /path/to/video.mp4
```

## 3. Proxy

入口：

- 模块：`physv_eval.single_case.proxy`
- 函数：`score_case(case, context_video_path=None, runner=None)`

指标定义：

- 项目内的几何 / 时序代理诊断量
- 本质上不是官方 benchmark 主分数
- 更适合看趋势、相对变化和失败案例

指标属性：

- 类型：连续诊断值
- 主字段：`score`
- 细节字段放在 `details`

函数返回：

```python
{
    "score": float,
    "context_frames": int,
    "future_frames": int,
    "context_video": str,
    "details": dict,
}
```

说明：

- 如果视频过短或无法构造有效上下文，函数可能返回 `None`

运行示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.proxy \
  --video /path/to/candidate.mp4 \
  --context-video /path/to/context.mp4
```

## 4. VideoPhy-2

入口：

- 模块：`physv_eval.single_case.videophy2`
- 函数：`score_case(case, task="pc", caption=None, rule=None, runner=None)`

指标定义：

- VideoPhy-2 自动评测器
- `sa` 偏 caption 对齐
- `pc` 偏物理 commonsense
- `rule` 偏显式规则约束判断

指标属性：

- 类型：离散 judge 分数
- 主字段：`score`
- 任务字段：`task`
- 常见分值范围：1 到 5

函数返回：

```python
{
    "task": str,
    "score": int,
    "raw_output": str,
    "num_frames": int,
    "checkpoint": str,
}
```

运行示例：

物理常识任务：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.videophy2 \
  --task pc \
  --video /path/to/video.mp4
```

caption 对齐任务：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.videophy2 \
  --task sa \
  --video /path/to/video.mp4 \
  --caption "a ball rolls and stops near the wall"
```

rule 任务：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.videophy2 \
  --task rule \
  --video /path/to/video.mp4 \
  --rule "objects should not pass through solid barriers"
```

## 5. PhyGround

入口：

- 模块：`physv_eval.single_case.phyground`
- 函数：
  `score_case(case, caption=None, metrics=None, laws=None, criteria_overrides=None, runner=None)`

指标定义：

- PhyGround judge-style 评测
- 一般分成两部分：
  - `general`
  - `physical_laws`

指标属性：

- 类型：离散 judge 分数
- `general_avg` 越高越好
- `physical_avg` 越高越好

函数返回：

```python
{
    "general": dict[str, int | None],
    "general_avg": float | None,
    "physical_laws": dict[str, int | None],
    "physical_avg": float | None,
    "coverage": float | None,
    "raw": {
        "general": dict[str, str],
        "physical_laws": dict[str, str],
    },
    "method": str,
    "adapter_dir": str,
    "infer_script": str,
    "max_new_tokens": int,
    "temperature": float,
    "fps": float,
    "max_pixels": int,
}
```

运行示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.phyground \
  --video /path/to/video.mp4 \
  --caption "a ball bounces and comes to rest"
```

只跑 general 部分：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.phyground \
  --input-json /path/to/case.json \
  --general-only
```

## 6. Cosmos-Reason1

入口：

- 模块：`physv_eval.single_case.cosmos_reason1`
- 函数：`score_case(case, runner=None)`

指标定义：

- Cosmos-Reason1 physical plausibility judge
- 偏整体物理合理性判断
- 更像粗粒度裁判分数，而不是连续误差

指标属性：

- 类型：离散 judge 分数
- 主字段：`score`
- 常见分值范围：1 到 5

函数返回：

```python
{
    "score": int | None,
    "raw": str,
    "method": str,
    "model": str,
    "prompt_path": str,
    "fps": int,
    "total_pixels": int,
    "max_new_tokens": int,
    "temperature": float,
    "top_k": int,
    "top_p": float,
    "repetition_penalty": float,
    "seed": int,
    "attempt_count": int,
}
```

运行示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.cosmos_reason1 \
  --video /path/to/video.mp4
```

## 建议

- 如果你是在代码里复用，优先直接 `import score_case`
- 如果你是在做批处理调度，优先从 `physv_eval/pipeline.py`、`physv_eval/*_batch.py` 这些外层脚本进入
- 不要把 `proxy` 这类诊断量直接当成最终 benchmark 总分

## 7. Physics-IQ 单视角近似版

入口：

- 模块：`physv_eval.single_case.physics_iq`
- 函数：
  `score_case(case, source_video_path=None, threshold_value=10, downsample_factor=4, context_mode="with_context", context_frames=None, aligned_video_dir=None)`

指标定义：

- 基于官方 Physics-IQ 底层分量做的单视角近似版
- 只比较一条 `output_video` 和一条 `source_video`
- 支持两种上下文模式：
  - `with_context`
  - `without_context`
- 保留：
  - `mse_mean`
  - `spatiotemporal_iou_mean`
  - `spatial_iou`
  - `weighted_spatial_iou`
- 明确去掉官方必须依赖的：
  - 多视角聚合
  - 第二条 real take 的 physical variance 归一化
- 会另存两条真正参与评分的视频：
  - 时间轴已对齐到公共窗口
  - `without_context` 时会先从输出视频和参考视频同时丢掉前 `context_frames`
  - 如果参考视频更长，会再截到和输出窗口同样的帧数
  - 参考视频会先被直接 resize 到输出视频空间尺寸，再按 `downsample_factor` 缩放到评分尺寸
  - 路径写回 `scored_output_video`、`scored_source_video`
  - 另存一个并排预览 `compare_side_by_side`

模式语义：

- `with_context`
  - 从第 `0` 帧开始比较
  - 更适合“输出视频本身包含 context 前缀”的场景
- `without_context`
  - 先从输出视频和参考视频同时丢掉前 `context_frames`
  - 更适合 v2v / future prediction 任务，尤其是输出前缀本身就是上下文时
  - `context_frames` 可以显式传入，也可以尝试从 case 元数据中推断，例如 `context_frames`、`used_context_frames`、`model_args.context_frames`

指标属性：

- 类型：连续值近似分
- 主字段：`score`
- 范围：`0` 到 `100`
- 注意：这不是官方 Physics-IQ 总分，不能直接和官方榜单横向对比

函数返回：

```python
{
    "score": float,
    "physics_iq_score": float,
    "official": bool,
    "method": str,
    "reference_video": str,
    "context_mode": str,
    "context_frames_used": int,
    "output_start_frame": int,
    "source_start_frame": int,
    "output_frames_after_context_clip": int,
    "source_frames_after_context_clip": int,
    "scored_output_video": str,
    "scored_source_video": str,
    "compare_side_by_side": str,
    "mse_mean": float,
    "spatiotemporal_iou_mean": float,
    "spatial_iou": float,
    "weighted_spatial_iou": float,
    "raw_score": float,
    "num_frames_compared": int,
    "compare_duration_sec": float,
    "compare_fps": float,
    "output_fps": float,
    "source_fps": float,
    "output_duration_sec": float,
    "source_duration_sec": float,
    "output_spatial_size": list[int],
    "source_aligned_size": list[int],
    "target_size": list[int],
    "downsample_factor": int,
    "threshold_value": int,
    "frame_alignment": str,
    "spatial_alignment": str,
    "score_formula": str,
    "notes": str,
}
```

运行示例：

直接传两条视频：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.physics_iq \
  --video /path/to/output.mp4 \
  --source-video /path/to/source.mp4
```

future-only 比较：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.physics_iq \
  --video /path/to/output.mp4 \
  --source-video /path/to/source.mp4 \
  --context-mode without_context \
  --context-frames 8
```

指定另存目录：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.physics_iq \
  --video /path/to/output.mp4 \
  --source-video /path/to/source.mp4 \
  --aligned-video-dir /data/gaoya/agent-data/outputs/my_physics_iq_case
```

传 case JSON：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.physics_iq \
  --input-json /path/to/case.json
```

## 8. PhysInOne PMF

入口：

- 模块：`physv_eval.single_case.pmf`
- 函数：
  `score_case(case, source_video_path=None, context_mode="with_context", context_frames=None, device="cpu", aligned_video_dir=None)`

指标定义：

- 对齐 PhysInOne 的 PMF
- 基于 3D FFT 频域能量分布的相似度
- 更适合看整体运动频谱是否和 GT 接近
- 支持两种上下文模式：
  - `with_context`
  - `without_context`
- `without_context` 时会先从输出视频和参考视频同时丢掉前 `context_frames`
- 如果参考视频更长，会再截到和输出窗口同样的帧数
- 参考视频会先被直接 resize 到输出视频空间尺寸，再调用官方 PMF 实现
- 会另存三条真正参与评测的可视化视频：
  - `pred_used_for_pmf`
  - `gt_used_for_pmf`
  - `compare_side_by_side`

指标属性：

- 类型：连续值
- 主字段：`score`
- 主方向：越高越好

函数返回：

```python
{
    "score": float,
    "pmf_score": float,
    "official": bool,
    "method": str,
    "reference_video": str,
    "context_mode": str,
    "context_frames_used": int,
    "output_start_frame": int,
    "source_start_frame": int,
    "output_frames_after_context_clip": int,
    "source_frames_after_context_clip": int,
    "pred_used_for_pmf": str,
    "gt_used_for_pmf": str,
    "compare_side_by_side": str,
    "video_codec": str,
    "metric_direction": str,
    "device": str,
    "output_fps": float,
    "source_fps": float,
    "compare_fps": float,
    "output_spatial_size": list[int],
    "source_aligned_size": list[int],
    "used_shape": list[int],
    "frame_alignment": str,
    "spatial_alignment": str,
    "notes": str,
}
```

运行示例：

直接传两条视频：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.pmf \
  --video /path/to/output.mp4 \
  --source-video /path/to/source.mp4
```

future-only 比较：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.pmf \
  --video /path/to/output.mp4 \
  --source-video /path/to/source.mp4 \
  --context-mode without_context \
  --context-frames 8
```

传 case JSON：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.pmf \
  --input-json /path/to/case.json
```

## 9. VBench

入口：

- 模块：`physv_eval.single_case.vbench`
- 函数：
  `score_case(case, dimension, caption=None, output_path=None, runner=None)`

指标定义：

- 对齐官方 VBench 的 `custom_input` 单维度评测
- 适合把已有视频接入 VBench 的单项能力评分
- 当前这个 `single_case` 入口只封装官方明确支持自定义视频的维度
- 官方结果会先落盘，再被适配层读回并整理成统一 `dict`

当前支持维度：

- `subject_consistency`
- `background_consistency`
- `motion_smoothness`
- `dynamic_degree`
- `aesthetic_quality`
- `imaging_quality`

指标属性：

- 类型：连续值
- 主字段：`score`
- 主方向：通常越高越好
- 原始逐视频结果保留在 `raw_results`

函数返回：

```python
{
    "score": float | None,
    "dimension": str,
    "metric_direction": str,
    "official": bool,
    "method": str,
    "supported_dimensions": list[str],
    "raw_dimension_score": float | None,
    "raw_results": list[dict],
    "video": str,
    "caption_used": str | None,
    "result_json": str,
    "full_info_json": str,
    "output_path": str,
    "cache_dir": str,
    "device": str,
    "mode": str,
}
```

说明：

- 该入口固定走官方 `custom_input` 模式
- 一次只评一个维度
- 如果不传 `caption`，默认复用通用 case 字段里的 prompt/caption
- 官方输出目录默认放在 `/data/gaoya/agent-data/outputs/vbench_single_case/...`

运行示例：

直接传视频：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.vbench \
  --video /path/to/video.mp4 \
  --dimension motion_smoothness
```

带 caption 覆盖：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.vbench \
  --video /path/to/video.mp4 \
  --dimension subject_consistency \
  --caption "a red ball rolling on the floor"
```

传 case JSON：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.vbench \
  --input-json /path/to/case.json \
  --dimension aesthetic_quality
```

## 10. VBench-2.0

入口：

- 模块：`physv_eval.single_case.vbench2`
- 函数：
  `score_case(case, dimension, caption=None, output_path=None, runner=None)`

指标定义：

- 对齐官方 VBench-2.0 的 `custom_input` 单维度评测
- 更偏下一代视频模型的 intrinsic faithfulness 能力
- 当前这个 `single_case` 入口只封装官方明确支持自定义视频的维度
- 官方结果会先落盘，再被适配层读回并整理成统一 `dict`

当前支持维度：

- `Human_Anatomy`
- `Human_Identity`
- `Human_Clothes`
- `Diversity`
- `Multi-View_Consistency`

指标属性：

- 类型：连续值
- 主字段：`score`
- 主方向：通常越高越好
- 原始逐视频结果保留在 `raw_results`

函数返回：

```python
{
    "score": float | None,
    "dimension": str,
    "metric_direction": str,
    "official": bool,
    "method": str,
    "supported_dimensions": list[str],
    "raw_dimension_score": float | None,
    "raw_results": list[dict],
    "video": str,
    "caption_used": str | None,
    "result_json": str,
    "full_info_json": str,
    "output_path": str,
    "cache_dir": str,
    "device": str,
    "mode": str,
}
```

说明：

- 该入口固定走官方 `custom_input` 模式
- 一次只评一个维度
- `Diversity` 比较特殊：
  - 需要至少 20 个视频
  - 命名格式应为 `prompt-index.mp4`
  - 这里的 `--video` 或 case JSON 视频字段应传“目录路径”而不是单个 mp4
- `Human_Anatomy`、`Human_Identity`、`Human_Clothes`、`Multi-View_Consistency` 可直接传单视频
- 官方输出目录默认放在 `/data/gaoya/agent-data/outputs/vbench2_single_case/...`

运行示例：

直接传视频：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.vbench2 \
  --video /path/to/video.mp4 \
  --dimension Human_Anatomy
```

多样性目录评测：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.vbench2 \
  --video /path/to/diversity_case_folder \
  --dimension Diversity
```

传 case JSON：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.vbench2 \
  --input-json /path/to/case.json \
  --dimension Multi-View_Consistency
```
