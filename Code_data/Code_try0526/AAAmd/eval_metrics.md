# Code_try0526 指标整理

说明：这一份只看“指标”，不展开数据集分组。

补充说明：
- `越高越好` / `越低越好` 只表示常规排序方向，不代表它一定是“更物理”的唯一标准。
- `官方` 表示仓库里直接对齐了对应官方代码或官方评测思路。
- `自定义/代理` 表示更适合做诊断或补充观察，不建议单独当总分。

## 指标一览表

| 指标 | 常见字段 | 指标属性 | 方向 | 简要解读 |
|---|---|---|---|---|
| PDI-Bench 官方分数 | `official_pdi`、`scale_component`、`traj_component`、`epsilon_rigidity`、`vp_component` | 官方主分数；连续值 | 越低越好 | 更像几何审计和结构一致性分数，适合看分割、跟踪、深度、投影关系是否稳定，不等于纯物理分数 |
| WMReward | `surprise`、`similarity`、`wmreward` | 官方口径；连续值 | `surprise` 越低越好，`similarity` 越高越好 | 本质是 V-JEPA 滑窗未来预测误差，更接近短时可预测性；`similarity` 是 `1 - surprise` 的派生量 |
| Geometry Proxy / VJEPA Proxy | `geometry_score`、`vjepa_proxy` | 自定义/代理；诊断量 | 通常越低越好 | 更适合看相对趋势和失败样本，不建议单独当主分数 |
| VideoPhy-2 AutoEval | `videophy2_auto_sa`、`videophy2_auto_pc` | 官方口径；离散 judge 分数 | 越高越好 | `SA` 更偏 caption 对齐，`PC` 更偏物理 commonsense，适合抓明显违和 |
| PhyGround | `phyground_general_avg` | 官方口径；judge 分数 | 越高越好 | 常见 general metrics 是 `SA / PTV / persistence`，更适合看语义对齐、时间变化合理性和持续存在性 |
| Cosmos-Reason1 | `cosmos_reason1` | 官方口径；离散 judge 分数 | 越高越好 | 更像整体 physical plausibility 裁判，适合看视频是否“像真的” |
| JEPA 分数 | `jepa_score` | 项目内评测量；连续值 | 当前用法里越高越好 | 更像受控实验下的局部预测 plausibility 信号，适合作为补充量 |
| FID | `fid` | 通用生成质量指标；连续值 | 越低越好 | 看单帧外观质量和分布接近度，不直接衡量物理规律 |
| FVD | `fvd` | 通用生成质量指标；连续值 | 越低越好 | 同时看外观和时序动态，比 FID 更适合视频，但仍不是物理主分数 |
| CSE / TSE | `cse`、`tse` | 官方 family；连续误差 | 越低越好 | 多视角几何一致性指标，`TSE` 看时序稳定性，`CSE` 看跨视角一致性；单视角时通常只能当 proxy |
| Accuracy / Correlation | `accuracy`、`pearson_correlation` | 汇总统计 | `accuracy` 越高越好，`pearson` 越接近 `1` 越好 | 只有在有明确标签或可靠参考分数时才有直接意义，否则更适合作为诊断 |

## 1. PDI-Bench 官方分数

- 含义：物理/几何一致性主分数，常见字段有 `official_pdi`、`scale_component`、`traj_component`、`epsilon_rigidity`、`vp_component`
- 指标属性：官方主分数；连续值；`official_pdi` 越低越好，各 component 一般也按误差理解，越低越好
- 解读：更像“几何审计 + 结构一致性”分数，通常适合看分割、跟踪、深度、投影关系是否稳定。它不是纯物理分数，外观变化、遮挡、相机变化也会影响结果
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/run_pdi_official_eval.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/evaluate_pdi_benchmark_methods.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/eval_benchmark_dir_metrics.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/pipeline.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python run_pdi_official_eval.py \
  --summary_json /data/gaoya/AAA_test_video/Output_try0526/runs/pdi_proxy_eval_demo/report/summary.json \
  --providers wan vace gt \
  --run_name pdi_official_eval_demo
```

## 2. WMReward

- 含义：视频奖励模型分数，脚本里常见字段是 `surprise`、`similarity`，汇总时通常记作 `wmreward`
- 指标属性：官方口径；连续值；`surprise` 越低越好，`similarity` 越高越好，但 `similarity` 本质上只是 `1 - surprise` 的派生量
- 解读：本质是 V-JEPA 滑窗未来预测误差，更接近“短时可预测性”而不是明确的物理法则评分。它通常能看出时间预测难度，但区分度未必很强
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/eval_pdibench_wmreward.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/eval_wmreward_jepa.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/eval_benchmark_dir_metrics.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/pipeline.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python physics_sim/eval_pdibench_wmreward.py
```

## 3. Geometry Proxy / VJEPA Proxy

- 含义：几何代理指标，脚本里常见字段有 `geometry_score`、`vjepa_proxy`
- 指标属性：自定义/代理；通常按误差理解为越低越好；更适合拆开看子项，不建议当单一主分数
- 解读：这是项目内诊断量，不是官方 benchmark 主指标。更适合用来比较相对趋势，或者分析哪些视频在几何/时序预测上更不稳定
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/run_pdi_proxy_eval.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/eval_benchmark_dir_metrics.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/pipeline.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python run_pdi_proxy_eval.py \
  --run_name pdi_proxy_eval \
  --device cuda
```

## 4. VideoPhy-2 AutoEval

- 含义：VideoPhy-2 自动评测，支持 `sa / pc / rule`，常见字段有 `videophy2_auto_sa`、`videophy2_auto_pc`
- 指标属性：官方口径；离散 judge 分数；通常是 1 到 5 分，越高越好
- 解读：`SA` 更偏 caption 对齐，`PC` 更偏物理 commonsense。它适合抓明显违和或明显合理的样本，但离散分数不太适合做很细的排序
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/eval_videophy2_auto.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/videophy2_auto.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/eval_benchmark_dir_metrics.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/pipeline.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python physics_sim/eval_videophy2_auto.py \
  --task pc \
  --groups A B1 B2 B3 C
```

## 5. PhyGround

- 含义：PhyGround 的 general score + physical laws score，当前常见字段有 `phyground_general_avg`
- 指标属性：官方口径；judge 分数；general metrics 常见是 `SA / PTV / persistence`，1 到 5 分，平均后越高越好
- 解读：更适合看语义对齐、时间变化合理性、物体是否持续存在。它本质上也是 VLM-as-judge 评分，所以结果会受视频外观、清晰度、提示词表达影响
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/eval_phyground.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/phyground_batch.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/PhyGround/evals/vlm_eval.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/pipeline.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python physics_sim/eval_phyground.py \
  --groups A B1 B2 B3 C
```

## 6. Cosmos-Reason1 物理合理性分数

- 含义：Cosmos-Reason1 的 physical plausibility score，常见字段是 `cosmos_reason1`
- 指标属性：官方口径；离散 judge 分数；通常是 1 到 5 分，越高越好
- 解读：适合判断视频整体是否“像真的”，对明显不合理的运动、碰撞、时序违和较敏感。但它粒度比较粗，也会受画面质量和呈现风格影响
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/eval_cosmos_reason1.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/cosmos_reason1_batch.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/eval_benchmark_dir_metrics.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/pipeline.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python physics_sim/eval_cosmos_reason1.py \
  --groups A B1 B2 B3 C
```

## 7. JEPA 分数

- 含义：主要出现在 `ball_block` 评测里，脚本里是 `jepa_score`
- 指标属性：项目内评测量；连续值；当前本地 ball-block 报表里按“越高越好”的预测合理性分数使用
- 解读：它更像局部预测 plausibility 信号，适合在受控实验里做补充对比，不建议把它和官方 benchmark 主分数直接等价
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/eval_ball_block.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim
python eval_ball_block.py --gpu 0 --port 18703
```

## 8. FID

- 含义：生成视频和 GT 的图像分布距离
- 指标属性：通用生成质量指标；连续值；越低越好
- 解读：FID 只在 frame level 上比较图像特征分布，主要看单帧外观质量和分布接近度，不直接衡量物理规律，也不直接保证时间连续性
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/cosmos-cookbook/scripts/metrics/qualitative/fvd_fid/compute_fid_single_view.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python cosmos-cookbook/scripts/metrics/qualitative/fvd_fid/compute_fid_single_view.py \
  --pred_video_paths "/path/pred/*.mp4" \
  --gt_video_paths "/path/gt/*.mp4" \
  --output_file /tmp/fid_results.json
```

## 9. FVD

- 含义：生成视频和 GT 的视频分布距离
- 指标属性：通用生成质量指标；连续值；越低越好
- 解读：FVD 会同时看外观和时序动态，比 FID 更适合视频，但它仍然是通用视频质量指标，不等于物理正确性
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/cosmos-cookbook/scripts/metrics/qualitative/fvd_fid/compute_fvd_single_view.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python cosmos-cookbook/scripts/metrics/qualitative/fvd_fid/compute_fvd_single_view.py \
  --pred_video_paths "/path/pred/*.mp4" \
  --gt_video_paths "/path/gt/*.mp4" \
  --output_file /tmp/fvd_results.json
```

## 10. CSE / TSE

- 含义：几何一致性指标，`CSE` 是 cross-view sampson error，`TSE` 是 temporal sampson error
- 指标属性：官方 family 是多视角几何一致性指标；连续误差；越低越好
- 解读：`TSE` 看单个视角随时间的几何稳定性，`CSE` 看同一时刻不同视角之间的几何一致性。注意它更适合多视角数据；如果在单视角项目里做简化替代，结果只能当 proxy 看
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/cosmos-cookbook/scripts/metrics/geometrical_consistency/sampson/run_cse_tse.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python cosmos-cookbook/scripts/metrics/geometrical_consistency/sampson/run_cse_tse.py \
  --input /path/to/videos \
  --output /tmp/cse_tse_eval
```

## 11. Accuracy / Correlation

- 含义：一些附带 benchmark 里会统计准确率、Pearson 相关系数
- 指标属性：汇总统计，不是物理主分数；`accuracy` 越高越好，`pearson_correlation` 越接近 `1` 越好
- 解读：只有在存在明确 ground truth label 或可靠打分参考时，这两个量才有直接意义。如果是用 pseudo label 或排序结果构造出来的统计，它们更适合作为仓库内诊断，不适合当成官方结论
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/cosmos-cookbook/scripts/examples/reason2/physical-plausibility-check/video_critic/compute_metrics.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/cosmos-reason1/examples/benchmark/tools/eval/calculate_accuracy.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python cosmos-reason1/examples/benchmark/tools/eval/calculate_accuracy.py \
  --result_dir /path/to/result_dir
```
