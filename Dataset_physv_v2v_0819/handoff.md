# PhysV V2V 0819 / CYCLES-RigidBench Handoff

截至 2026-08-26，本项目已经完成 `physv_v2v_0819` 数据集的 CYCLES 对齐真值导出、RigidBench adapter 构建，以及一个单 case 的 RigidBench-style 原生分辨率评测。本文档用于后续接手、复现和扩展。

## 1. 项目范围

项目的核心数据集是 PhysV V2V 0819：每个 case 来自 PyBullet/仿真场景，并包含原始 RGB、CYCLES 重渲染 RGB、动态物体 mask、轨迹、深度/相机和物理状态等数据。当前 RigidBench 相关工作只把 CYCLES RGB 和与其同一相机坐标系重新渲染的真值用于评测；原始仿真坐标系的真值没有直接混入 CYCLES 像素评测。

## 2. 关键路径

| 内容 | 路径 |
|---|---|
| 项目仓库 | `/home/gaoya/Code_Video/Dataset_physv_v2v_0819` |
| 原始数据集 | `/data/gaoya/AAA_test_video/physv_v2v_0819` |
| CYCLES 对齐真值 v2 | `/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v2_rigidbench` |
| 已验证 case | `cases/difficulty_l2_f11_h030_sr048` |
| RigidBench adapter | `cases/difficulty_l2_f11_h030_sr048/rigidbench` |
| RigidBench 源码 | `/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench` |
| VDA 官方源码 vendored 目录 | `RigidBench/vendor/Video-Depth-Anything` |
| 单 case GT-vs-GT 结果 | `evaluations/gt_oracle_single_case/results.json` |
| 单 case SAM2/CoTracker/VDA 结果 | `evaluations/rigidbench_tracker_gt_native_single_case/cycles_gt_native_sam2_cotracker3_vda/results.json` |
| 评测日志 | `evaluations/logs/rigidbench_gt_native_eval_gpu0.log` |
| 真值可视化页面 | `http://localhost:8860/` |

当前真值页面的前台启动命令是：

```bash
/usr/bin/python3 -m http.server 8860 \
  --bind 0.0.0.0 \
  --directory /data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v2_rigidbench/visualization
```

不要把该服务改成后台 daemon；需要重启时在前台执行上述命令。当前服务若仍在运行，可先确认 `ss -ltnp | rg ':8860'`。

## 3. 数据契约与来源

### 3.1 实际评测视频

本次 RigidBench adapter 的 `video.mp4` 指向：

```text
/data/gaoya/AAA_test_video/physv_v2v_0819/samples/difficulty_l2_f11_h030_sr048/videos/rgb_cycles.mp4
```

也就是说，本次评测实际使用的是 `rgb_cycles.mp4`，不是 `rgb.mp4`。`rgb.mp4` 是原始仿真 RGB；`rgb_cycles.mp4` 是使用同一场景、相机和轨迹重新用 CYCLES 渲染的参考视频。当前 case 的视频属性是：

- 90 帧；
- 30 FPS；
- 原生分辨率 `896×512`；
- 时长约 3 秒。

### 3.2 RigidBench adapter 文件

`rigidbench/` 中包含：

- `video.mp4`：CYCLES RGB 视频的 adapter 入口；
- `masks.npz`：形状 `[T,N,H,W]`，当前 case 的动态 actor mask；
- `depth.npz`：CYCLES Depth/Z pass，和 CYCLES RGB 使用同一像素坐标系；
- `trajectories.npz`：actor 世界坐标轨迹；
- `metadata.json`：sample id、actor role、相机内参 K、外参和坐标约定。

原始 case 目录中的 `contacts.json`、`physics_supervision.npz`、`raw/trajectories.npz` 等仍然保留，主要用于物理事件/训练监督；它们不是本次 RigidBench 十项指标的直接输入。`raw/masks.npz` 是原始仿真相机坐标系的 mask，也不能直接替代 CYCLES 坐标系的 `rigidbench/masks.npz`。

### 3.3 数据生成代码

主要脚本为：

- `scripts/render_physv_cycles.py`：CYCLES RGB 渲染；
- `scripts/render_physv_cycles_aligned_truth.py`：按 CYCLES 相机/分辨率重新生成动态 mask、Depth/Z、轨迹和 metadata；
- `scripts/generate_physv_cycles_aligned_truth.py`：组织 CYCLES 对齐真值导出；
- `scripts/build_cycles_rigidbench_truth_viewer.py`：生成真值检查页面；
- `scripts/run_cycles_gt_rigidbench_tracker_eval.py`：运行本次原生 CYCLES RigidBench-style 评测；
- `scripts/evaluate_cycles_gt_oracle.py`：运行确定性的 GT-vs-GT 上限检查。

## 4. 评测协议

RigidBench 官方评测包含以下十项指标：

`IoU`、`L2`、`Chamfer`、`ATE`、`SI-MSE`、`LPIPS`、`SSIM`、`ATE3D`、`IdDrift`、`BGDrift`。

本次评测采用 `rigidbench-code-native-cycles` 协议：

- 保留 CYCLES 的 `30 FPS / 896×512 / 90 frames`；
- 不做官方代码中的 30→24 FPS 时间重采样；
- 不把单 case 结果标记为官方分数；
- 当前只验证了 1 个 case：`difficulty_l2_f11_h030_sr048`。

官方 RigidBench 协议使用固定的 24 FPS、`1280×704` 和完整 100-case eval 集合。因此当前结果应称为 **RigidBench-style · CYCLES · native-resolution · single-case**，不能写成官方 full benchmark score，也不能与官方 100-case 分数直接比较。

## 5. 已验证结果

### 5.1 GT-vs-GT oracle

结果文件：

```text
/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v2_rigidbench/evaluations/gt_oracle_single_case/results.json
```

该检查把同一份 CYCLES GT 同时作为 reference 和 prediction，用于验证 adapter、mask、depth、相机和轨迹的内部一致性；它不运行 SAM2、CoTracker、VDA、DINO 或 LPIPS。

| 指标 | 值 |
|---|---:|
| IoU | 1.0 |
| L2 | 0 |
| Chamfer | 0 |
| ATE | 0 |
| SI-MSE | `2.7204665e-15` |
| LPIPS | 0 |
| SSIM | 1.0 |
| ATE3D | 0 |
| IdDrift | 0 |
| BGDrift | 0 |

### 5.2 SAM2/CoTracker/VDA 实际 pipeline

本次已经重新运行并成功完成：

- SAM2.1 Hiera-Large：生成动态 mask；
- CoTracker3 offline：生成 2D tracks；
- Video Depth Anything Large：生成预测 disparity/depth；
- RigidBench scorer：计算十项指标。

结果文件：

```text
/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v2_rigidbench/evaluations/rigidbench_tracker_gt_native_single_case/cycles_gt_native_sam2_cotracker3_vda/results.json
```

当前单 case 数值：

| 指标 | 值 |
|---|---:|
| IoU | 0.9744643 |
| L2 | 0.0007849 |
| Chamfer | 0.00005338 |
| ATE | 0.0330741 |
| SI-MSE | 0.0006483 |
| LPIPS | 0.0066419 |
| SSIM | 0.9875908 |
| ATE3D | 0.1705630 |
| IdDrift | 0.2229415 |
| BGDrift | 0.0004303 |

该结果不是 GT-vs-GT 上限：mask、tracks、depth 和相关 learned features 由实际模型重新估计。由于 reference video 和 prediction video 在这个 sanity check 中是同一份 CYCLES 视频，LPIPS/SSIM 主要验证 scorer 链路，不能代表生成模型质量。

## 6. 可复现环境与命令

推荐使用 `sam` 环境，并禁用用户 site package，避免全局 NumPy 2.x 与当前 Torch ABI 产生干扰：

```bash
cd /home/gaoya/Code_Video/Dataset_physv_v2v_0819

export PYTHONNOUSERSITE=1
export TORCH_HOME=/home/gaoya/.cache/torch
export PYTHONPATH=/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src:/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/vendor/Video-Depth-Anything
export CUDA_VISIBLE_DEVICES=0
export RB_PYTHON=/home/gaoya/miniconda3/envs/sam/bin/python
```

### 6.1 运行 GT-vs-GT 检查

```bash
"$RB_PYTHON" scripts/evaluate_cycles_gt_oracle.py \
  --case-dir /data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v2_rigidbench/cases/difficulty_l2_f11_h030_sr048 \
  --output-dir /data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v2_rigidbench/evaluations/gt_oracle_single_case
```

### 6.2 运行实际 SAM2/CoTracker/VDA pipeline

```bash
"$RB_PYTHON" scripts/run_cycles_gt_rigidbench_tracker_eval.py \
  --case-dir /data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v2_rigidbench/cases/difficulty_l2_f11_h030_sr048 \
  --output-dir /data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v2_rigidbench/evaluations/rigidbench_tracker_gt_native_single_case
```

使用到的本地 checkpoint：

- SAM2：`/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt`；
- CoTracker3：`/home/gaoya/.cache/torch/hub/checkpoints/cotracker3_scaled_offline.pth`；
- VDA Large：`/data/gaoya/ckpt/Video-Depth-Anything-Large/video_depth_anything_vitl.pth`。

脚本显式加载这些本地 checkpoint，不依赖 Hugging Face 或 torch.hub 在线下载。RigidBench scorer 内部的 BGDrift 路径还会二次调用 CoTracker；当前 runner 已将该调用复用到同一个本地 CoTracker checkpoint，避免 `torch.hub` 的 401/联网失败。

## 7. 常见问题

1. **视频无法在浏览器播放**：优先使用真值页面中生成的 H.264/`avc1` 视频，不要把 `mp4v` 文件直接作为浏览器 `<video>` 源。
2. **NumPy/Torch ABI 警告或导入异常**：确认使用 `/home/gaoya/miniconda3/envs/sam/bin/python`，并设置 `PYTHONNOUSERSITE=1`。
3. **SAM2 的 `_C` warning**：若只出现扩展加载 warning 但 mask 阶段正常完成，先以阶段结果和最终 `results.json` 为准；若 mask 阶段失败，再检查 SAM2 编译环境。
4. **torch.hub 401**：不要恢复默认在线 CoTracker 加载；使用当前 runner 的本地 checkpoint patch。
5. **官方 CLI 找不到 sample**：官方 loader 要求固定 manifest/目录结构；当前 runner 已构建 `rigidbench_dataset/samples/<sample_id>` adapter，并直接指定 data root 和 sample id。
6. **不要混淆分辨率协议**：当前结果是 native `896×512/30 FPS`；如果改成官方 `1280×704/24 FPS`，必须新建输出目录并在报告中明确标注重采样和 resize。

## 8. 当前限制与下一步

- 当前 learned pipeline 只跑通了 1 个 case；要形成 CYCLES 数据集结果，需要批量生成 70 个 case 的 CYCLES RigidBench adapter，并按 case 汇总均值、标准差和失败清单。
- 即使完成 70 case，也只能称为 `RigidBench-style · CYCLES · 70-case`，不能称为官方 100-case `official=true`。
- 批量评测前应先确定唯一协议：推荐保留 CYCLES native 分辨率/FPS；如果需要和官方数值对照，则另建严格官方协议输出，避免两类结果混在同一张表中。
- 如果评测生成视频，必须保证生成视频和 `rgb_cycles.mp4` 使用同一 case、同一帧范围和明确的 FPS；随后将 prediction video 放入对应 RigidBench adapter 的生成目录，再运行 scorer。
- 新增或重跑大规模任务时使用 GPU `0,1,2,3,5,6,7`，不要使用 GPU4；大型数据、checkpoint、cache 和评测输出继续放在 `/data/gaoya`，不要写入 `/home/gaoya`。

## 9. 交接前检查

```bash
cd /home/gaoya/Code_Video/Dataset_physv_v2v_0819
git diff --check
/home/gaoya/miniconda3/envs/sam/bin/python -m py_compile \
  scripts/evaluate_cycles_gt_oracle.py \
  scripts/run_cycles_gt_rigidbench_tracker_eval.py
```

接手者首先应检查：

1. `evaluations/.../results.json` 是否存在且包含 `official: false`；
2. `native_protocol.json` 是否仍记录 `rgb_cycles.mp4`、30 FPS 和 `896×512`；
3. 重新跑批量评测前是否有其他用户任务占用 GPU；
4. 不要覆盖 `samples/` 原始数据和已有 v1/v2 真值目录；
5. 任何新结果使用独立的 `evaluations/<protocol>/<model>/` 目录，并把协议写入 JSON/README。

