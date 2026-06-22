# Code_try0526 指标整理

说明：这一份只看“指标”，不展开数据集分组。

## 1. PDI-Bench 官方分数

- 含义：物理/几何一致性主分数，常见字段有 `official_pdi`、`scale_component`、`traj_component`、`epsilon_rigidity`、`vp_component`
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

- 含义：PhyGround 的 general score + physical laws score，常见字段有 `phyground_general_avg`
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
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim/eval_ball_block.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526/physics_sim
python eval_ball_block.py --gpu 0 --port 18703
```

## 8. FID

- 含义：生成视频和 GT 的图像分布距离
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
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/cosmos-cookbook/scripts/metrics/geometrical_consistency/sampson/run_cse_tse.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python cosmos-cookbook/scripts/metrics/geometrical_consistency/sampson/run_cse_tse.py \
  --video_dir /path/to/videos \
  --output_dir /tmp/cse_tse_eval
```

## 11. Accuracy / Correlation

- 含义：一些附带 benchmark 里会统计准确率、Pearson 相关系数
- 常用脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/cosmos-cookbook/scripts/examples/reason2/physical-plausibility-check/video_critic/compute_metrics.py`
  - `/home/gaoya/Code_Video/Code_data/Code_try0526/cosmos-reason1/examples/benchmark/tools/eval/calculate_accuracy.py`
- 命令示例：
```bash
cd /home/gaoya/Code_Video/Code_data/Code_try0526
python cosmos-reason1/examples/benchmark/tools/eval/calculate_accuracy.py \
  --result_dir /path/to/result_dir
```
