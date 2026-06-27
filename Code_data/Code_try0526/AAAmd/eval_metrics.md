# Code_try0526 指标整理与用法

说明：
- 这一版只整理“指标脚本怎么用”，不再展开数据集分组和历史背景。
- 现在单 case 评测入口统一放在 `physv_eval/single_case/` 下，其他推理脚本如果要复用，优先 `import` 这些模块里的 `score_case(...)`。
- 如果一个指标脚本同时支持 `--input-json` 和 `--video`，优先把单样本整理成统一的 case JSON，这样更适合和其他脚本复用。

## 目录约定

推荐的单 case 入口：

- `python -m physv_eval.single_case.pdi`
- `python -m physv_eval.single_case.wmreward`
- `python -m physv_eval.single_case.proxy`
- `python -m physv_eval.single_case.videophy2`
- `python -m physv_eval.single_case.phyground`
- `python -m physv_eval.single_case.cosmos_reason1`

这些入口都支持最少一个视频路径；其中一部分还支持 caption、rule、context video 等额外输入。  
对需要批量跑目录或整套 benchmark 的脚本，可以继续保留在各自仓库目录里，但建议把真正的单样本打分逻辑统一收敛到上面的 `score_case(...)`。

## 通用单 case 输入格式

### 1) 直接传视频

最简单的输入就是一个视频路径：

```bash
python -m physv_eval.single_case.wmreward --video /path/to/video.mp4
```

### 2) 传 case JSON

如果你已经有统一的样本描述，推荐用 `--input-json`。脚本会自动从 JSON 里提取视频、caption、rule 和 context video。

当前通用字段如下：

- 视频路径字段：`video`、`video_path`、`output_video`、`paths.output_video_path`
- 上下文视频字段：`context_video`、`context_video_path`、`paths.context_video_path`
- caption 字段：`caption`、`text_prompt`、`prompt`、`description`、`scenario`、`experiment`、`target_object`、`clip_name`、`name`
- 规则字段：`rule`、`physical_law`、`law`

一个最小 JSON 可以长这样：

```json
{
  "video": "/path/to/video.mp4",
  "caption": "a ball rolls down a ramp"
}
```

带上下文视频时可以这样写：

```json
{
  "video": "/path/to/candidate.mp4",
  "context_video": "/path/to/context.mp4",
  "caption": "the candidate video should follow the same setup"
}
```

### 3) 代码里复用

如果其他推理脚本要复用单 case 评测，直接导入对应模块即可：

```python
from physv_eval.single_case.videophy2 import score_case

result = score_case(case, task="pc", caption="a ball rolls down a slope")
```

这里的 `case` 可以是：

- `Path` / `str`，表示视频路径
- `dict`，表示 case JSON 解析后的字典
- `EvalCase`，表示已经标准化过的输入对象

## 指标方向速查

这一节只说明“数值变大/变小代表什么”，口径与 `AAAmd/physv_sim_eval.md` 保持一致。

| 指标 | 推荐主字段 | 方向 | 说明 |
|---|---|---|---|
| PDI-Bench Official | `pdi.pdi_score` | ↓ | 主分数是误差型分数，越低越好。 |
| WMReward | `wmreward.surprise` | ↓ | 主口径看 `surprise`。`similarity = 1 - surprise` 只是派生量，若单独看 `similarity` 则是 ↑。 |
| Geometry Proxy / VJEPA Proxy | `proxy.temporal_relation_raw_error` / `proxy.delta_relation_raw_error` / `proxy.delta_profile_error` | ↓ | 更推荐看这三个展开误差项。兼容字段 `proxy.score = exp(-(三项误差之和))` 是 ↑，但不建议把它当主结论。 |
| VideoPhy-2 | `videophy2.score` | ↑ | 离散 judge 分数，分数越高越好。 |
| PhyGround | `phyground.general_avg` / 其他平均分 | ↑ | judge 均分口径，越高越好。 |
| Cosmos-Reason1 | `cosmos_reason1.score` | ↑ | 1 到 5 的离散 judge 分数，越高越好。 |
| FID | `fid` | ↓ | 分布距离，越低越好。 |
| FVD | `fvd` | ↓ | 视频分布距离，越低越好。 |
| CSE / TSE | `cse` / `tse` | ↓ | Sampson error 类指标，越低越好。 |
| Accuracy | `accuracy` | ↑ | 正确率，越高越好。 |
| Correlation | `pearson_correlation` | ↑ | 正相关越强通常越好；如果出现负相关，一般说明排序方向有问题。 |

## 1. PDI-Bench 官方分数

- 推荐入口：`python -m physv_eval.single_case.pdi`
- 典型用途：对单个视频跑官方 PDI 评测，输入里最好有视频和对应文本描述
- 输入格式：
  - `--video /path/to/video.mp4`
  - 或 `--input-json case.json`
  - 可选 `--caption "..."`，作为 text query
- 输出格式：标准输出一份 JSON，常见字段包括 `pdi_score`、`grade`、`scale_error`、`traj_error`、`rigidity_error`、`vp_error`
- 方向：`pdi_score ↓`

命令示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.pdi \
  --video /path/to/video.mp4 \
  --caption "a ball rolls on a ramp" \
  --output-json /tmp/pdi_single_case.json
```

如果你已经有 case JSON：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.pdi \
  --input-json /path/to/case.json \
  --output-json /tmp/pdi_single_case.json
```

补充说明：
- 这一层的核心逻辑已经收敛到 `physv_eval/single_case/pdi.py`。
- 旧的批处理脚本如果还在用，建议只当外层调度，不要再复制一份评分逻辑。

## 2. WMReward

- 推荐入口：`python -m physv_eval.single_case.wmreward`
- 典型用途：对单个视频计算 WMReward / surprise / similarity
- 输入格式：
  - `--video /path/to/video.mp4`
  - 或 `--input-json case.json`
  - 不要求 caption，主要看视频本身
- 输出格式：标准输出 JSON，通常会包含 `surprise`、`similarity`、`wmreward` 一类字段
- 方向：主口径看 `surprise ↓`；`similarity` 只是 `1 - surprise` 的派生量，对应 `similarity ↑`

命令示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.wmreward \
  --video /path/to/video.mp4 \
  --output-json /tmp/wmreward_single_case.json
```

如果是 case JSON：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.wmreward \
  --input-json /path/to/case.json \
  --output-json /tmp/wmreward_single_case.json
```

## 3. Geometry Proxy / VJEPA Proxy

- 推荐入口：`python -m physv_eval.single_case.proxy`
- 典型用途：对单个视频和可选 context video 跑几何代理或时序代理分数
- 输入格式：
  - `--video /path/to/video.mp4`
  - 可选 `--context-video /path/to/context.mp4`
  - 或 `--input-json case.json`
- 输出格式：标准输出 JSON，通常会返回代理分数以及相关诊断项；如果当前 case 无法打分，脚本会报错
- 方向：优先看 `temporal_relation_raw_error / delta_relation_raw_error / delta_profile_error ↓`；兼容总分 `score ↑`

命令示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.proxy \
  --video /path/to/candidate.mp4 \
  --context-video /path/to/context.mp4 \
  --output-json /tmp/proxy_single_case.json
```

如果是 case JSON：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.proxy \
  --input-json /path/to/case.json \
  --output-json /tmp/proxy_single_case.json
```

补充说明：
- 这个指标更适合看趋势和失败样本，不建议直接当主 benchmark 总分。

## 4. VideoPhy-2 AutoEval

- 推荐入口：`python -m physv_eval.single_case.videophy2`
- 典型用途：对单个视频跑 VideoPhy-2 自动评测
- 输入格式：
  - `--task sa|pc|rule`
  - `--video /path/to/video.mp4`
  - 对 `sa` 通常配 `--caption`
  - 对 `rule` 通常配 `--rule`
  - 也支持 `--input-json case.json`
  - 可选 `--context-video /path/to/context.mp4`
- 输出格式：标准输出 JSON，通常会返回该 task 的 judge 分数和相关元信息
- 方向：常用 `score ↑`

命令示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.videophy2 \
  --task pc \
  --video /path/to/video.mp4 \
  --output-json /tmp/videophy2_pc.json
```

如果是 caption 对齐任务：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.videophy2 \
  --task sa \
  --video /path/to/video.mp4 \
  --caption "a ball rolls and stops near the wall" \
  --output-json /tmp/videophy2_sa.json
```

如果是 rule 任务：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.videophy2 \
  --task rule \
  --video /path/to/video.mp4 \
  --rule "objects should not pass through solid barriers" \
  --output-json /tmp/videophy2_rule.json
```

补充说明：
- `SA` 更偏 caption 对齐。
- `PC` 更偏物理 commonsense。
- `rule` 适合显式规则约束场景。

## 5. PhyGround

- 推荐入口：`python -m physv_eval.single_case.phyground`
- 典型用途：对单个视频跑 PhyGround 评测
- 输入格式：
  - `--video /path/to/video.mp4`
  - `--caption "..."`，用于描述视频内容
  - 或 `--input-json case.json`
  - 可选 `--general-only`，只跑 general metrics
- 输出格式：标准输出 JSON，通常包含 `phyground_general_avg` 以及更细的子项
- 方向：`phyground_general_avg ↑`

命令示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.phyground \
  --video /path/to/video.mp4 \
  --caption "a ball bounces and comes to rest" \
  --output-json /tmp/phyground_single_case.json
```

只跑 general 部分时：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.phyground \
  --input-json /path/to/case.json \
  --general-only \
  --output-json /tmp/phyground_general_only.json
```

补充说明：
- 这类 judge 分数会受视频清晰度、提示词表达、画面风格影响。

## 6. Cosmos-Reason1 物理合理性分数

- 推荐入口：`python -m physv_eval.single_case.cosmos_reason1`
- 典型用途：对单个视频跑 Cosmos-Reason1 物理合理性评测
- 输入格式：
  - `--video /path/to/video.mp4`
  - 或 `--input-json case.json`
  - 不强制要求 caption，但 case JSON 里有 caption 也会被保留到输出记录中
- 输出格式：标准输出 JSON，常见核心字段是 `cosmos_reason1`
- 方向：`cosmos_reason1.score ↑`

命令示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.cosmos_reason1 \
  --video /path/to/video.mp4 \
  --output-json /tmp/cosmos_reason1_single_case.json
```

如果是 case JSON：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python -m physv_eval.single_case.cosmos_reason1 \
  --input-json /path/to/case.json \
  --output-json /tmp/cosmos_reason1_single_case.json
```

补充说明：
- 这个分数更适合判断“整体像不像真的”，对明显不合理运动和碰撞更敏感。

## 7. FID

- 脚本：`cosmos-cookbook/scripts/metrics/qualitative/fvd_fid/compute_fid_single_view.py`
- 典型用途：比较预测视频和 GT 视频的单帧图像分布距离
- 输入格式：
  - `--pred_video_paths "/path/pred/*.mp4"`
  - `--gt_video_paths "/path/gt/*.mp4"`
  - 两边会分别按 glob 展开后排序，数量必须一致
  - 可选 `--num_frames N`
  - 可选 `--output_file /tmp/fid_results.json`
- 输出格式：终端打印 FID，另写一个 JSON 结果文件
- 方向：`FID ↓`

命令示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python cosmos-cookbook/scripts/metrics/qualitative/fvd_fid/compute_fid_single_view.py \
  --pred_video_paths "/path/pred/*.mp4" \
  --gt_video_paths "/path/gt/*.mp4" \
  --output_file /tmp/fid_results.json
```

补充说明：
- FID 只看 frame-level 外观分布，不直接衡量物理合理性。

## 8. FVD

- 脚本：`cosmos-cookbook/scripts/metrics/qualitative/fvd_fid/compute_fvd_single_view.py`
- 典型用途：比较预测视频和 GT 视频的视频分布距离
- 输入格式：
  - `--pred_video_paths "/path/pred/*.mp4"`
  - `--gt_video_paths "/path/gt/*.mp4"`
  - 两边会分别按 glob 展开后排序，数量必须一致
  - 可选 `--num_frames N`
  - 可选 `--batch_size N`
  - 可选 `--target_size H W`
  - 可选 `--output_file /tmp/fvd_results.json`
- 输出格式：终端打印 FVD，另写一个 JSON 结果文件
- 方向：`FVD ↓`

命令示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python cosmos-cookbook/scripts/metrics/qualitative/fvd_fid/compute_fvd_single_view.py \
  --pred_video_paths "/path/pred/*.mp4" \
  --gt_video_paths "/path/gt/*.mp4" \
  --output_file /tmp/fvd_results.json
```

补充说明：
- FVD 比 FID 更适合视频，但仍然是通用质量指标，不等于物理正确性。

## 9. CSE / TSE

- 脚本：`cosmos-cookbook/scripts/metrics/geometrical_consistency/sampson/run_cse_tse.py`
- 典型用途：对单个视频或一个目录做 cross-view / temporal Sampson error 评测
- 输入格式：
  - `--input /path/to/video_or_dir`
  - 输入既可以是单个视频文件，也可以是目录
  - 如果是目录，可以用 `--pattern "*.mp4"` 控制匹配模式
  - 可选 `--output /tmp/cse_tse_eval`
  - 可选 `--verbose`
- 输出格式：每个视频会输出单独结果，同时生成聚合统计
- 方向：`CSE ↓`，`TSE ↓`

命令示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python cosmos-cookbook/scripts/metrics/geometrical_consistency/sampson/run_cse_tse.py \
  --input /path/to/video_or_dir \
  --output /tmp/cse_tse_eval
```

如果输入是目录，可以显式指定 pattern：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python cosmos-cookbook/scripts/metrics/geometrical_consistency/sampson/run_cse_tse.py \
  --input /path/to/video_dir \
  --pattern "*.mp4" \
  --output /tmp/cse_tse_eval
```

补充说明：
- `TSE` 更偏时间稳定性。
- `CSE` 更偏跨视角一致性。
- 如果数据本身不是多视角，这两个量只能当 proxy 看。

## 10. Accuracy / Correlation

这两个脚本不是同一种输入格式，不能混用。

### 10.1 `compute_metrics.py`

- 脚本：`cosmos-cookbook/scripts/examples/reason2/physical-plausibility-check/video_critic/compute_metrics.py`
- 典型用途：从一批 JSON 文件里读 `ground_truth` 和 `pred_score`，计算 accuracy 和 Pearson correlation
- 输入格式：
  - 传一个目录参数 `output_dir`
  - 目录下每个 JSON 文件应该是一个对象，至少包含：
    - `ground_truth`
    - `pred_score`
  - 脚本会忽略 `summary.json`
- 输出格式：打印 `accuracy`、`pearson_correlation`、`num_samples`，并把汇总写回 `summary.json`
- 方向：`accuracy ↑`，`pearson_correlation ↑`

命令示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python cosmos-cookbook/scripts/examples/reason2/physical-plausibility-check/video_critic/compute_metrics.py \
  /path/to/output_dir
```

### 10.2 `calculate_accuracy.py`

- 脚本：`cosmos-reason1/examples/benchmark/tools/eval/calculate_accuracy.py`
- 典型用途：从一个结果目录里递归扫描 JSON 文件，统计 `is_correct`
- 输入格式：
  - 传 `--result_dir /path/to/result_dir`
  - 每个 JSON 文件顶层应该是一个 list
  - list 中每个 item 应该是一个 dict，并且包含 `is_correct`
- 输出格式：打印总样本数、正确样本数和平均 accuracy
- 方向：`accuracy ↑`

命令示例：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python cosmos-reason1/examples/benchmark/tools/eval/calculate_accuracy.py \
  --result_dir /path/to/result_dir
```

补充说明：
- `compute_metrics.py` 处理的是“每个文件一条记录”的 JSON。
- `calculate_accuracy.py` 处理的是“每个文件是一个结果列表”的 JSON。

## 快速选型

- 只想跑单个视频的物理/语义评测：优先用 `physv_eval/single_case/*`
- 想跑预测视频 vs GT 的分布距离：用 FID / FVD
- 想看几何或时序一致性：用 CSE / TSE
- 想看汇总统计：再用 Accuracy / Correlation
