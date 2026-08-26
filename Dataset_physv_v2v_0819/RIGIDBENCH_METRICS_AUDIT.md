# RigidBench 指标与 PhysV V2V 0819 CYCLES 数据集对照审查

审查日期：2026-08-26 UTC  
RigidBench 本地版本：`f0b8f298f7609e05cd3bf2bc1897e6c37ac32515`（Initial release）  
官方仓库：[swarnim-j/RigidBench](https://github.com/swarnim-j/RigidBench)  
本地副本：`/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench`

## 结论

`physv_v2v_0819` 的 CYCLES 数据已经具备大部分物理评估所需的原始真值，但目前不能直接执行官方的：

```bash
rigidbench evaluate <prediction-dir>
```

主要原因不是缺少动态物体 mask，而是：

1. CYCLES 对齐真值目前没有独立的 CYCLES 像素坐标系深度 GT；现有 `raw/depth.npz` 是原始 PyRender 相机的 `1280×720` 深度。
2. 我们的 actor 角色写成 `dynamic`，RigidBench evaluator 的源码只按 `role == "active"` 选择 actor。
3. CYCLES 视频是 90 帧、30 FPS、3 秒，分辨率为 `896×512` 或 `640×360`；RigidBench 的固定协议是 24 FPS、2 秒、官方 GT 分辨率 `1280×704`。
4. CYCLES 对齐 mask 的字段和布局与 RigidBench 原生 `masks.npz` 不同，需要转换。
5. RigidBench 只有在完整的 100 个 eval sample 和全部十项指标存在时才标记 `official=true`；我们的数据集是 70 个 case，因此即使适配成功，也只能是适配版/子集结果。

因此，当前数据适合做 **RigidBench-style CYCLES 评估**；若要声称严格的官方 RigidBench 分数，需要先建立一套明确的 CYCLES 适配协议，并标注为非官方 70-case 结果。

本次已在不覆盖 v1 的前提下完成一个 CYCLES 适配原型并通过链路验证：
`physv_v2v_0819_cycles_aligned_truth_v2_rigidbench/cases/difficulty_l2_f11_h030_sr048/`
包含 CYCLES Depth/Z pass、显式 K/外参和嵌套的 RigidBench 标准样本。RigidBench 官方 loader 能加载该样本，官方 GT 20 点轨迹函数也能正常运行；70 个 case 尚未批量生成。

## 1. RigidBench 官方评估协议

官方 README 要求每个样本提交一个按 sample ID 命名的视频，覆盖 `t=0` 到 `t=2.0 s`；评估器会记录并对齐输入视频的 FPS 和分辨率，完整 benchmark 包含 100 个 eval examples。[官方评估说明](https://github.com/swarnim-j/RigidBench/blob/main/README.md#generate-one-video-per-example)

源码中的协议常量是：

- `GT_FPS = 24`
- `GT_RESOLUTION = (1280, 704)`
- `DURATION_SECONDS = 2.0`
- `EVALUATION_SIZE = 100`

见本地 [constants.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src/rigidbench/core/constants.py:1) 和 [benchmark.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src/rigidbench/benchmark.py:13)。

生成视频在预处理阶段会被抽帧到评估目录，并直接 resize 到 `1280×704`；时间上会对生成结果做 FPS 对齐，但 reference GT 必须与 evaluator 数据目录中的 sample 结构相匹配。见 [prepare.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src/rigidbench/eval/prepare.py:128)。

## 2. 十项指标及其 GT 依赖

RigidBench 不把十项指标合并为一个总分，而是分别报告 IoU、L2、Chamfer、ATE、ATE-3D、SI-MSE、SSIM、LPIPS、IdDrift 和 BGDrift。[官方指标列表](https://github.com/swarnim-j/RigidBench#measurements)

| 指标 | evaluator 实际计算方式 | 必需 GT/参考数据 | 我们的 CYCLES 状态 |
| --- | --- | --- | --- |
| IoU | 逐 actor、逐帧比较 GT mask 与生成视频上由 SAM2 传播出的 mask | 每帧 active actor mask；首帧 mask 用于初始化 SAM2 | **有，需字段转换** |
| L2 | GT/预测 actor mask 的质心距离，除以图像高度 | 每帧 active actor mask | **有，需字段转换** |
| Chamfer | GT/预测 actor mask 的双向最近邻距离 | 每帧 active actor mask | **有，需字段转换** |
| ATE | CoTracker 预测点轨迹与 GT 点轨迹的 2D 误差 | 首帧 actor mask、actor 3D 轨迹、逐帧深度、相机内外参 | **部分具备，CYCLES 相机/深度需适配** |
| ATE-3D | 将预测轨迹和预测深度重建为 3D centroid，再与 GT 世界坐标轨迹比较 | actor 世界坐标轨迹、相机内外参、GT depth、预测 tracks/depth | **缺 CYCLES depth** |
| SI-MSE | 预测 Video Depth Anything disparity 与 GT depth 做 affine 对齐后计算 scale-invariant MSE | 每帧 reference depth | **缺 CYCLES depth** |
| SSIM | reference RGB 与生成 RGB 的逐帧结构相似度 | 完整 reference RGB 帧 | **有 `rgb_cycles.mp4`** |
| LPIPS | reference RGB 与生成 RGB 的 LPIPS | 完整 reference RGB 帧 | **有 `rgb_cycles.mp4`** |
| IdDrift | 在 GT/生成轨迹位置截取 RGB patch，用 DINOv2 比较 actor identity | reference RGB、GT/预测 tracks、共同可见性 | **有基础数据，依赖 tracks 适配** |
| BGDrift | 在生成首帧的前景 mask 外检测背景角点，用 CoTracker 测量背景非刚性漂移 | 生成视频、生成前景 mask；不需要静态 GT mask | **可用，依赖 mask pipeline** |

### 2.1 Mask 指标

`ScoreContext.masks` 读取 `masks.npz` 中的：

```text
masks: (T, N, H, W)
object_names: (N,)
```

然后根据 `metadata["actors"][name]["role"] == "active"` 选出 actor。IoU、L2、Chamfer 都是逐 actor 计算，再对 actor 和帧做平均。见 [context.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src/rigidbench/eval/score/context.py:48) 和 [mask.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src/rigidbench/eval/score/mask.py:8)。

这说明 RigidBench 需要的是 **运动 actor 的逐帧 mask**，不是静态场景中所有物体的 union mask。静态物体 mask 不是这十项指标的必需输入。

### 2.2 2D/3D 轨迹指标

官方 evaluator 不是直接使用一个 actor 中心点文件计算 ATE。它会：

1. 在首帧 actor mask 内采样 20 个 query points；
2. 用首帧 GT depth 和相机参数把像素点反投影到世界坐标；
3. 按 actor 的世界位置和旋转轨迹重新投影到每一帧；
4. 用逐帧 actor mask 和深度判断点是否可见；
5. 将这些投影结果作为 GT tracks，与生成视频上的 CoTracker tracks 对比。

对应实现见 [gt.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src/rigidbench/eval/track/gt.py:13) 和 [cotracker3.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src/rigidbench/eval/track/cotracker3.py:48)。

ATE-3D 还需要使用预测 disparity 与 GT depth 做尺度/偏置对齐，再使用相机内外参将预测轨迹反投影到世界坐标。见 [context.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src/rigidbench/eval/score/context.py:107) 和 [trajectory.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src/rigidbench/eval/score/trajectory.py:107)。因此，CYCLES 的 `trajectory_pixels.npz` 中心点轨迹不能单独替代官方所需的 actor 级 world trajectory + camera + depth。

### 2.3 深度、RGB 和背景指标

- SI-MSE 读取 GT depth，并将预测 Video Depth Anything disparity 与其对齐，见 [depth.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src/rigidbench/eval/score/depth.py:8)。
- SSIM/LPIPS 使用完整 reference RGB 帧，不需要静态物体 mask，见 [frame.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src/rigidbench/eval/score/frame.py:8)。
- IdDrift 使用 reference/生成 RGB patch 和共同可见的 actor tracks，见 [identity.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src/rigidbench/eval/score/identity.py:38)。
- BGDrift 在生成帧上检测背景角点，并使用生成前景 mask 排除 actor 区域；它不读取静态物体 GT mask，见 [background.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src/rigidbench/eval/score/background.py:15)。

`contacts.json`、`physics_supervision.npz`、碰撞法向力以及 caption 事件信息不在 RigidBench 这十项 evaluator 的输入链路中；它们可以用于我们自己的物理事件指标，但不是 RigidBench 的必需 GT。

## 3. 我们的 CYCLES 数据实际具备什么

本次对 `/data/gaoya/AAA_test_video/physv_v2v_0819` 的 70 个 sample 和 `physv_v2v_0819_cycles_aligned_truth_v1` 做了逐 case 检查：

| 数据项 | 实际状态 |
| --- | --- |
| CYCLES reference RGB | 70/70 有 `videos/rgb_cycles.mp4`，均为 90 帧、30 FPS、3 秒 |
| CYCLES 分辨率 | 60 个 `896×512`，10 个 `640×360` |
| CYCLES dynamic mask | 70/70 有 `dynamic_masks.npz`；65 个 case 为 1 个动态 actor，5 个 case 为 5 个动态 actor |
| CYCLES mask 字段 | `masks_thw: (N,T,H,W)`、`union_thw: (T,H,W)`、`object_names`、`object_indices` |
| 原始 simulator mask | 70/70 有 `raw/masks.npz`，形状为 `(T,N,720,1280)`，只含动态 actor |
| 原始 simulator depth | 70/70 有 `raw/depth.npz`，形状为 `(90,720,1280)` |
| 原始 world trajectory | 70/70 有 `raw/trajectories.npz`，90 帧位置和旋转 |
| CYCLES pixel trajectory | 70/70 有 `trajectory_pixels.npz`，保存动态 actor 中心投影，不是 20 点 tracks |
| CYCLES depth GT | 当前 0/70 有独立 CYCLES depth 文件 |
| actor role 字段 | 70/70 使用 `dynamic`，没有 `active` |
| test 列表 | 两个 test70 列表均为 70 条、无重复 |

新增的 v2 单 case 原型额外验证了：CYCLES Depth/Z pass 为 `(90,H,W)`、mask 与 depth 使用相同的左下到左上垂直翻转、CYCLES 相机 K/E 已写入 metadata，且 `rigidbench_dataset/samples/<sample_id>` 可被官方 sample loader 发现。

CYCLES 对齐真值的字段和来源见 [CYCLES 对齐真值 README](/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v1/README.md:13)；原始 mask、深度和轨迹的实际布局见 [数据集 README](/data/gaoya/AAA_test_video/physv_v2v_0819/README.md:14)。

## 4. 兼容性判断

| 目标 | 判断 | 原因 |
| --- | --- | --- |
| 直接把原始 CYCLES 目录传给官方 evaluator | **不兼容** | 目录结构、mask 字段、actor role、分辨率/FPS协议和 CYCLES depth 均不匹配 |
| 使用新增 v2 adapter 跑单 case 的官方 GT 准备链路 | **已验证** | 官方 loader、CYCLES depth、相机参数、world trajectory 和 GT point-track 函数均可读取；这不是官方 100-case 分数 |
| 只做 CYCLES IoU/L2/Chamfer | **可以适配** | 已有逐帧动态 actor mask；需转置 `masks_thw`、映射 `dynamic -> active`，并用首帧 mask 初始化生成视频上的 SAM2 |
| 做 CYCLES ATE | **可以适配但不能直接运行** | 已有 world trajectory 和动态 mask；需补齐 CYCLES 相机内外参与同坐标系深度，或改为使用已有 pixel trajectory 的自定义 2D 评估 |
| 做 CYCLES ATE-3D | **当前不具备直接条件** | 缺少 CYCLES 像素坐标系 depth；raw depth 与 CYCLES RGB 的相机/分辨率不一致 |
| 做 CYCLES SI-MSE | **当前不具备直接条件** | 缺少与 `rgb_cycles.mp4` 对齐的 GT depth |
| 做 CYCLES SSIM/LPIPS | **可以适配** | CYCLES reference RGB 已有；需要统一抽帧和输出视频时间范围 |
| 做 CYCLES IdDrift/BGDrift | **可以适配** | 依赖 RGB、actor tracks 和生成前景 mask，不要求静态物体 GT |
| 生成官方 `official=true` 分数 | **不可能直接成立** | evaluator 将完整官方集合固定为 100 个 eval sample，而我们当前是 70 个 case |

## 5. 适配实现与后续批量方案

当前实现新增独立输出目录，不覆盖现有数据：

```text
/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v2_rigidbench/
├── cases/<case_id>/
│   ├── dynamic_masks.npz          # 项目格式，(N,T,H,W)
│   ├── cycles_depth.npz           # CYCLES Depth/Z pass，(T,H,W)
│   ├── trajectory_pixels.npz      # CYCLES 中心投影
│   ├── truth_metadata.json        # CYCLES K/E 与对齐说明
│   └── rigidbench/                # RigidBench 标准单样本
│       ├── video.mp4              # 指向 rgb_cycles.mp4 的链接
│       ├── masks.npz              # (T,N,H,W)，dynamic -> active
│       ├── depth.npz
│       ├── trajectories.npz
│       └── metadata.json
└── rigidbench_dataset/samples/<case_id>  # 指向上面 rigidbench/ 的链接
```

单 case 原型使用的脚本为：

- [render_physv_cycles_aligned_truth.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/render_physv_cycles_aligned_truth.py)
- [generate_physv_cycles_aligned_truth.py](/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/generate_physv_cycles_aligned_truth.py)

单 case 验证命令已经实际运行过，输出为 `difficulty_l2_f11_h030_sr048`。需要批量生成时，再用非 GPU 4 的 Blender worker 执行全部 70 个 case；预计按单 case 约 1–2 分钟、并行度取决于空闲 GPU 和磁盘吞吐，批量输出会明显增加存储占用。

后续指标执行建议分两步：

1. **先做 3 个 RGB/mask 指标**：将 CYCLES `masks_thw` 转成 `(T,N,H,W)`，把 `dynamic` 映射成 `active`，用 CYCLES 视频首帧初始化 SAM2，先验证 IoU、L2、Chamfer 的链路。
2. **再运行 RigidBench-style 轨迹/深度指标**：v2 原型已经补上 CYCLES Z pass、相机内外参和 RigidBench GT 文件，因此批量完成后可继续接入 SI-MSE、ATE 和 ATE-3D；ATE 仍建议使用同一套 CYCLES camera/depth 生成 GT point tracks。

如果只是为了与官方代码保持结构兼容，可以将 CYCLES 视频和真值统一转换为 24 FPS、2 秒、`1280×704` 的 adapter 数据；但这会引入 resize/重采样，不应与原生 CYCLES 分辨率结果混称。更合理的做法是给 evaluator 增加 `native-resolution` 模式，并将结果标记为 `RigidBench-style · CYCLES · 70-case`。

## 6. 当前不需要补充的 GT

以下数据对 RigidBench 的十项指标不是必需项：

- 静态场景所有物体的 union mask；
- `raw/instance_ids.npz` 的全物体实例 ID；
- `contacts.json` 接触点/法向力；
- `physics_supervision.npz` 事件和状态标签；
- `collision_supervision` cache。

它们对我们自己的物理事件、碰撞和场景理解指标有价值，但不属于 RigidBench evaluator 的输入依赖。
