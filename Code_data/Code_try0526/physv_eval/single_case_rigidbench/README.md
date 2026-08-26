# `single_case_rigidbench`

这里的模块只接收“指标计算所需的实际输入”，不接收 task ID、case ID，也不负责扫描目录或修改结果 JSON。test70 的目录扫描和增量回填由：

```text
/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_metric_backfill.py
```

或一键脚本：

```bash
/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_metric_backfill.sh
```

## 输入规范总表

| 输入 | shape | dtype / 数值约定 |
|---|---:|---|
| mask | `(T,N,H,W)`，也接受单 actor `(T,H,W)` | `bool`；整数只允许 `0/1` 或 `0/255`；浮点只允许 `[0,1]`，阈值为 `0.5` |
| RGB frames | `(T,H,W,3)` | `uint8` `[0,255]`；浮点可为 `[0,1]` 或 `[0,255]`，模块会转为 `uint8`；通道顺序 RGB |
| 2D tracks | `(N,T,2)` | 有限浮点像素坐标，最后一维为 `(x,y)`，不归一化 |
| visibility | `(N,T)` | `bool`；只在轨迹相关指标中使用 |
| actor offsets | `(A+1,)` | `int64`，首值 `0`、末值为轨迹点数 `N` |
| depth | `(T,H,W)` | 有限正浮点；GT 是 CYCLES metric depth（米） |
| predicted disparity | `(T,H,W)` | 有限正浮点；是预测逆深度/视差，不要预先归一化，官方函数会按视频做 affine alignment |
| 3D centroid | list of `N` arrays `(T,3)` | 世界坐标、米；不做额外归一化 |

时间维 `T` 必须对应同一段评测窗口；test70 当前窗口为 49 帧，native CYCLES 为 30 FPS、`896×512`。mask 的空间 shape 必须完全一致；深度预测空间分辨率可以不同，官方 SI-MSE 会 resize 到 GT 分辨率。

## 模块接口

| 文件 | Python 接口 | 必需输入 |
|---|---|---|
| `iou.py` | `score_case(gt_mask, pred_mask)` | `(T,N,H,W)` mask |
| `l2.py` | `score_case(gt_mask, pred_mask)` | `(T,N,H,W)` mask；输出质心距离 `/ H` |
| `chamfer.py` | `score_case(gt_mask, pred_mask)` | `(T,N,H,W)` mask；输出边界点距离 `/ H` |
| `ate.py` | `score_case(gt_tracks, pred_tracks, image_height, visibility)` | `(N,T,2)` tracks、`(N,T)` visibility |
| `si_mse.py` | `score_case(gt_depth, pred_disparity)` | GT depth 和预测 disparity |
| `ssim.py` | `score_case(gt_frames, pred_frames, device)` | RGB uint8 frames |
| `lpips.py` | `score_case(gt_frames, pred_frames, model, device)` | RGB uint8 frames、已加载 LPIPS 模型 |
| `ate3d.py` | `score_case(pred_centroids, gt_trajectories, actors)` | 世界坐标 3D centroid、GT trajectory dict |
| `iddrift.py` | `score_case(gt_frames, pred_frames, gt_tracks, pred_tracks, visibility, actor_offsets, dinov2_model, device)` | RGB、2D tracks、visibility、DINOv2 |
| `bgdrift.py` | `score_case(pred_frames, foreground_mask, cotracker_model, device)` | 生成 RGB、前景 mask、CoTracker |

每个返回值至少包含对应的标量字段；有逐帧定义的指标还返回 `per_frame`。例如 `iou.py` 返回 `{"iou": float, "per_frame": (T,)}`。

## 命令行示例

命令行同样传真实指标输入路径，不传 case ID：

```bash
export PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_try0526:/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src
python -m physv_eval.single_case_rigidbench.iou \
  --gt-mask /path/to/gt/masks.npz \
  --pred-mask /path/to/pred/mask.npz

python -m physv_eval.single_case_rigidbench.si_mse \
  --gt-depth /path/to/gt/depth.npz \
  --pred-disparity /path/to/pred/depth.npz

python -m physv_eval.single_case_rigidbench.ssim \
  --gt-video /path/to/gt/video.mp4 \
  --pred-video /path/to/pred/video.mp4
```

## test70 补测逻辑

一键脚本按指标外层循环：先收集所有 task 的所有缺失 `iou` case 并逐个执行，再处理 `l2`，依次到 `bgdrift`。每完成一个 case：

1. 只向 `metrics/<sample>.json` 增加当前指标字段，不覆盖其他字段；
2. 有逐帧输出时，只向对应 `metrics_per_frame/<sample>.npz` 增加当前指标数组；
3. 使用临时文件原子替换，避免中途退出留下半个 JSON/NPZ；
4. 当前指标批次结束后刷新 test70 汇总快照。

缺少生成视频、预测 mask、tracks、depth 或 GT 文件时，该 case 会保留为 pending 并记录失败原因；本脚本不自动重新生成 tracker 中间结果。

预览待测项而不执行：

```bash
/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_metric_backfill.sh \
  --dry-run
```

只补一个指标：

```bash
/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_metric_backfill.sh \
  --metrics lpips
```
