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
- 函数：`score_case(case, source_video_path=None, threshold_value=10, downsample_factor=4)`

指标定义：

- 基于官方 Physics-IQ 底层分量做的单视角近似版
- 只比较一条 `output_video` 和一条 `source_video`
- 保留：
  - `mse_mean`
  - `spatiotemporal_iou_mean`
  - `spatial_iou`
  - `weighted_spatial_iou`
- 明确去掉官方必须依赖的：
  - 多视角聚合
  - 第二条 real take 的 physical variance 归一化

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
    "target_size": list[int],
    "downsample_factor": int,
    "threshold_value": int,
    "frame_alignment": str,
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

传 case JSON：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.physics_iq \
  --input-json /path/to/case.json
```
