## PhysV Metric Dashboard

这个目录放一个独立工具，用来把一批 `GT + 多个方法视频` 组织成单页可视化，并在每个视频卡片上标注：

- `WMReward surprise`
- `VideoPhy2-SA`
- `VideoPhy2-PC`
- `Cosmos Reason1`

设计原则：

- 不修改原有 benchmark / 评测仓库里的脚本
- 优先复用 case sidecar JSON 中已经存在的指标
- 只有在显式打开 `--compute-missing` 时，才调用 `physv_eval.single_case` 补算缺失指标
- 页面输出和缓存都写到 `/data/gaoya/agent-data/outputs`

示例：

```bash
python3 /home/gaoya/Code_Video/DiffTrack-main/AAA_my_test/physv_metric_dashboard/build_metric_annotated_dashboard.py \
  --gt-dir /data/gaoya/AAA_test_video/Output_try0526/ABD_test/B/GT \
  --method wan22-5B-TI2V=/data/gaoya/AAA_test_video/Output_try0526/ABD_test/B/wan22-5B-TI2V \
  --method VACE_1p3B_TI2V=/data/gaoya/AAA_test_video/Output_try0526/ABD_test/B/VACE_1p3B_TI2V \
  --output-dir /data/gaoya/agent-data/outputs/physv_metric_dashboard/abd_b_wan_vs_vace_ti2v
```
