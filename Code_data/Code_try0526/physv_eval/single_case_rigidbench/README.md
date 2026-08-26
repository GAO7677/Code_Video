# `single_case_rigidbench`

这里的模块只接收“一个指标计算所需的 GT 信息”和“生成视频”。预测侧的
mask、tracks、disparity/depth 不再作为输入缓存传入，而是在指标函数内部从
生成视频提取。模型对象由外层指标 worker 传入，worker 对一个指标的全部
case 只加载一次模型。

目录扫描、GT case 组织、指标 JSON 回填由：

```text
/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_metric_backfill.py
```

## 输入约定

| 输入 | shape / 类型 | 说明 |
|---|---|---|
| GT video | `(T,H,W,3)`，RGB `uint8` `[0,255]` | strict CYCLES reference video |
| GT mask | `(T,N,H,W)`，`bool` | active-object 的真值由 `metadata.json` 的 `role=active` 选择 |
| GT tracks | `(N,T,2)` | 像素坐标 `(x,y)`，不归一化；按官方 projector 从 strict truth 得到 |
| GT visibility | `(N,T)`，`bool` | GT track 可见性 |
| GT depth | `(T,H,W)`，正浮点，米 | CYCLES metric depth |
| camera | `intrinsics + extrinsics` | 用于 ATE-3D 的世界坐标重建 |
| GT trajectory | `<actor>_positions: (T,3)` | 世界坐标、米，用于 ATE-3D |
| generated video | `(T,H,W,3)`，RGB | 预测侧唯一输入；兼容当前 test70 的 jpg/png frame directory |

mask 的空间尺寸必须与 GT 第一帧一致；tracks 的最后一维必须是 `(x,y)` 像素坐标；
depth/disparity 不要预先归一化。时间长度按 RigidBench 当前协议截取到共同窗口。

## 预测侧提取协议

预测侧严格复用 RigidBench 官方 tracker 逻辑：

| 预测中间结果 | 内部模型 / 初始化 | 使用它的指标 |
|---|---|---|
| predicted mask | SAM2；用 GT 第一帧 active-object mask 初始化，再在 generated video 上传播 | IoU、L2、Chamfer、BG-Drift |
| predicted tracks | CoTracker3；用 GT tracks 的首帧点作为 query points | ATE、ID-Drift、ATE-3D |
| predicted disparity | Video-Depth-Anything-Large，`target_fps=24` 和官方 `DEPTH_INPUT_SIZE` | SI-MSE、ATE-3D |
| DINO features | DINOv2 ViT-L/14 | ID-Drift |
| LPIPS features | AlexNet-LPIPS | LPIPS |
| background tracks | CoTracker3；从 SAM2 预测的首帧 foreground mask 排除前景后检测背景角点 | BG-Drift |

因此，SAM2/CoTracker/VDA 不使用生成侧已有的 `mask.npz`、`tracks.npz` 或
`depth.npz`；这些文件不再是新指标接口的前置条件。它们仍可作为旧结果回归
对照，但不会被新 runner 读取。

## Python 接口

```python
from physv_eval.single_case_rigidbench import iou, ate, si_mse
from physv_eval.single_case_rigidbench.prediction import (
    load_sam2_model, load_cotracker_model, load_vda_model,
)

sam2 = load_sam2_model("cuda")
result_iou = iou.score_case(
    gt_mask, generated_video, sam2, active_actor_indices=[0],
)

cotracker = load_cotracker_model("cuda")
result_ate = ate.score_case(
    gt_tracks, generated_video, image_height, cotracker, gt_visibility,
)

vda = load_vda_model("cuda")
result_si_mse = si_mse.score_case(
    gt_depth, generated_video, vda, "cuda",
)
```

各指标接口如下：

| 文件 | 接口核心参数 | 内部提取 |
|---|---|---|
| `iou.py` | `gt_mask, pred_video, sam2_model` | SAM2 mask |
| `l2.py` | `gt_mask, pred_video, sam2_model` | SAM2 mask |
| `chamfer.py` | `gt_mask, pred_video, sam2_model` | SAM2 mask |
| `ate.py` | `gt_tracks, pred_video, image_height, cotracker_model, visibility` | CoTracker tracks |
| `si_mse.py` | `gt_depth, pred_video, vda_model` | VDA disparity |
| `ssim.py` | GT frames, generated frames | 无模型 |
| `lpips.py` | GT frames, generated frames, LPIPS model | LPIPS |
| `ate3d.py` | GT tracks/depth/camera/trajectory + generated video + VDA/CoTracker | disparity + tracks |
| `iddrift.py` | GT frames/tracks/visibility/offsets + generated video + DINO/CoTracker | tracks + DINO features |
| `bgdrift.py` | GT mask + generated video + SAM2/CoTracker | foreground mask + background tracks |

## 单指标命令行示例

```bash
export PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_try0526:/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src

python -m physv_eval.single_case_rigidbench.iou \
  --gt-mask /path/to/strict/rigidbench/masks.npz \
  --pred-video /path/to/generated/video-or-frame-dir \
  --device cuda

python -m physv_eval.single_case_rigidbench.ate \
  --gt-tracks /path/to/gt_tracks.npz \
  --pred-video /path/to/generated/video-or-frame-dir \
  --image-height 512 --device cuda

python -m physv_eval.single_case_rigidbench.si_mse \
  --gt-depth /path/to/strict/rigidbench/depth.npz \
  --pred-video /path/to/generated/video-or-frame-dir \
  --device cuda
```

## test70 回填

批处理 runner 现在按指标外层循环，并在每个指标进程中只加载一次该指标所需
模型；它只要求生成目录存在和 strict GT 文件齐全，不检查生成侧的 mask、tracks、
depth 缓存：

```bash
/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_metric_backfill.sh \
  --metric bgdrift --no-build
```

每个 case 完成后，runner 只增量更新对应指标 JSON 字段和逐帧 NPZ，并使用 case
锁、临时文件和原子替换。已有旧结果不会因为回归测试被覆盖；只有正式补测流程
才会回填 pending 指标。
