# Future Query Predictor 场景几何实验总报告

日期：2026-09-20  
代码快照：`visual_scene_context8_20260918`  
状态：已完成 CPU 诊断实验整理；**不是正式大规模训练，也不是正式 OOD/视觉泛化结论**。

## 1. 本报告范围

本报告整理当前 Future Query Predictor 的场景几何路线及其已执行实验。复制到本目录的只有代码、测试、配置和说明；原始数据、缓存、checkpoint、预测结果和视频没有移动，仍位于 `/data/gaoya/agent-data`。

当前 Predictor 保持原接口：观察 RGB0--RGB7 对应的运动 context，预测 RGB8--RGB48 的 41 个位置。核心实现见 [`code/future_query_predictor.py`](../code/future_query_predictor.py)。本轮 CPU 诊断没有联合 DiT、VAE、VGGT、SAM2 或 Utonia；几何实验使用的是特权解析任务几何或已有几何缓存，不能称为纯 RGB 结果。

## 2. 共同协议

- 运动输入、`last_position`、`last_velocity`、`future_dt` 和 position/interval-velocity SmoothL1 loss 保持一致。
- 几何分支使用 1792 个槽位；任务几何诊断中 512 个有效点，其余为 masked padding。
- 默认场景参考点仍是 `last_position`，没有 two-pass、未来位置重定位、接触 head 或新物理约束 loss。
- geometry-only 分支忽略 `scene_features`；因此下面的 task-geometry 结果不是 Utonia 增益。
- 训练结果均为有限步数 CPU 诊断，不能替代正式多 seed 训练。

## 3. 实验总览

| 实验 | 目的 | 数据/预算 | 状态 | 主要结论 |
|---|---|---|---|---|
| 几何尺度 transfer | 检查 observed sphere scale 是否真正进入 Predictor | door 40例，500 steps，10例 development-val | `EXECUTED` | 新尺度确实改变输入和输出，但 ADE 几乎不变，物理指标未改善 |
| paired learnability | 在相同历史、不同未来下检查场景响应是否可拟合 | control_door_00--03，1000 steps | `EXECUTED` | dense 特权几何可拟合，sparse 受关键区域覆盖影响；固定采样不稳定 |
| sampling invariance/resampling | 区分 token 顺序问题与采样支持问题，测试每步重采样 | 同四例，1000 steps，20 个采样 seed | `EXECUTED` | 顺序不是主因；重采样显著改善响应和负对照，但未达到全 seed 准入 |
| independent history holdout | 在新的 history group 上比较 motion-only 与 geometry branch | 24 physics-only episode；6 train/2 dev group；1500 steps | `EXECUTED` | dev pair 全部等未来，无法证明正向场景响应；不能宣称泛化 |

## 4. 实验一：尺度 transfer

报告与结果：[`predictor_geometry_transfer_paired_report.md`](/data/gaoya/agent-data/outputs/a3_training_redesign_20260908/visual_scene_context8_20260918/docs/predictor_geometry_transfer_paired_report.md)。

对照包括旧 AABB 尺度、新 `observed_sphere_surface` 尺度、partial visible oracle、motion-only 和 analytic constant-velocity。四个学习臂使用相同初始化、500 optimizer steps、effective batch 2。

| arm | Val ADE | Val FDE | interval-v MAE | physical loss | contact accuracy |
|---|---:|---:|---:|---:|---:|
| analytic CV | 0.11932 | 0.44426 | 0.12733 | 0.01898 | 1.00 |
| motion-only | 0.05910 | 0.25956 | 0.09783 | 0.00793 | 0.80 |
| old-scale geometry | 0.04922 | 0.22826 | 0.08139 | 0.00698 | 0.80 |
| new sphere-scale geometry | 0.04916 | 0.22731 | 0.08235 | 0.00721 | 0.80 |
| partial oracle geometry | 0.04821 | 0.22157 | 0.07966 | 0.00675 | 0.80 |

新尺度相对旧 AABB 仅改善 `0.067 mm` ADE，且 velocity/physical loss 略差。相同 checkpoint 替换缓存时，`scene_xyz` 平均改变 11.44 cm、预测平均改变 6.60 mm，证明新尺度没有停留在缓存字段中。结论是：**尺度修正能进入计算路径，但 raw metric scale 不是解决场景响应的充分条件。**

穿透诊断也没有随 ADE 单调改善：motion-only 的 5 mm 穿透帧率反而最低。因此 ADE 下降不能解释成物理安全性提高。

## 5. 实验二：paired learnability

报告与结果：[`paired_fit_report.md`](/data/gaoya/agent-data/outputs/a3_training_redesign_20260908/visual_scene_context8_20260918/docs/paired_fit_report.md)。四个 `control_door_00..03` 共享 bitwise-identical 的观察运动状态，只改变门宽并重新仿真。六个 pair 中 5 个非退化，`02--03` 是等未来负对照。

| arm | final loss | ADE | FDE | interval-v MAE | 结论 |
|---|---:|---:|---:|---:|---|
| motion-only | 0.006650732 | 53.92 mm | 263.58 mm | 0.08299 | 未拟合；对不同场景完全不响应 |
| geometry-sparse-task | 0.000394741 | 11.13 mm | 38.80 mm | 0.02145 | 关键 0.50/0.62 m pair 未过响应阈值 |
| geometry-dense-task | 0.000065710 | 4.04 mm | 16.52 mm | 0.00998 | 在已消费的四例上达到预设 fit 阈值 |

固定采样下，dense arm 的 5 个非退化 pair 都达到 `R_delta <= 0.25`，但只代表这四个开发样本上的拟合能力，不代表留出泛化。dense 的关键左右门框各采 192 点，sparse 各采 4 点；因此关键区域覆盖是明确混杂因素。

## 6. 实验三：采样不变性与每步重采样

报告与结果：[`resampling_fit_report.md`](/data/gaoya/agent-data/outputs/a3_training_redesign_20260908/visual_scene_context8_20260918/docs/resampling_fit_report.md)。

首先做同步点集重排：只改变 token 顺序，输出最大变化约 `3.58e-7 m`，低于 `1e-5 m` 容差，说明大幅波动不是简单的顺序问题。随后修正 surface pool 的 face-edge 重复点，并比较固定采样和每步重采样训练。

| 指标 | dense_fixed | dense_resample_train |
|---|---:|---:|
| final loss | 2.32e-5 | 3.15e-4 |
| 五个非退化 pair 的响应通过数 | 10/100 | 88/100 |
| 等未来负对照通过数 | 3/20 | 20/20 |
| 新 seed 严格 combined gate | 0/16 | 8/16 |

每步重采样显著提高采样稳定性，但 case 01 的尾部仍未过均值稳定性阈值，且穿透率没有单调改善。因此结论是：**重采样是必要的鲁棒化措施，但不是完整的物理约束或场景读取解决方案。**

## 7. 实验四：独立 history group 留出

结果：[`group_holdout_report.md`](/data/gaoya/agent-data/outputs/a3_training_redesign_20260908/visual_scene_context8_20260918/validation_20260920/group_holdout_training_1500_v1/group_holdout_report.md)。数据 manifest：[`paired_history_data_manifest.json`](/data/gaoya/agent-data/outputs/a3_training_redesign_20260908/visual_scene_context8_20260918/validation_20260920/paired_history_physics_v1/paired_history_data_manifest.json)。

共 24 条 physics-only episode：`g00..g05` 为 6 个 train history group，`g06..g07` 为 2 个 development history group，每组门宽为 0.46/0.58/0.70 m。16 个 sampling seed 是重复测量，不是额外 history。

### 普通轨迹指标

| split/arm | ADE | FDE | interval-v MAE | physical loss | >5 mm case fraction | mean max penetration |
|---|---:|---:|---:|---:|---:|---:|
| train/motion-only | 0.01541 m | 0.06826 m | 0.04342 | 0.00165 | 50.0% | 9.95 mm |
| train/geometry-dense-resample | 0.01694 m | 0.08143 m | 0.03304 | 0.00130 | 33.3% | 9.41 mm |
| dev/motion-only | 0.11238 m | 0.47421 m | 0.15330 | 0.02578 | 50.0% | 34.21 mm |
| dev/geometry-dense-resample | 0.10640 m | 0.47377 m | 0.13791 | 0.02665 | 33.3% | 17.64 mm |

### 场景响应

train 中 144 条非退化 pair 的平均真值差为 `31.91 mm`。geometry-dense-resample 的平均预测差只有 `0.351 mm`，`R_delta=1.00187`，响应通过率为 0；motion-only 的预测差为 0。dev 的所有 pair 都是等未来，只有负对照，不能用于正向场景响应结论。

因此独立 history 结果只能支持：geometry branch 在普通 ADE 和穿透 proxy 上有局部变化，但**尚未学会按不同静态几何产生正确的未来运动**。当前失败不能唯一归因于 sampling tail；更主要的问题已经在 train non-degenerate pair 上出现。下一步应先生成至少一个具有非退化未来差异的独立 dev history，再决定改数据覆盖还是改 geometry readout。

## 8. 结论

1. 场景几何确实能改变 Predictor 的中间计算和输出；它不是只存在于接口中。
2. 仅修尺度不能解决场景条件化；partial oracle 也没有在旧 zero-shot pair 上产生足够响应。
3. 关键表面采样覆盖和每步重采样对结果影响很大。固定采样的成功容易混入 token/support 记忆。
4. paired supervision 能让 dense 特权几何在四例上拟合出不同未来，但该能力在采样变化和独立 history 上尚未稳定。
5. 当前没有证据支持“视觉场景输入已经实现物理泛化”。原因包括：独立 dev pair 退化、task geometry 不是 RGB-visible geometry、尚未完成全前端 RGB/VGGT/SAM2/Utonia 重放，以及接触/穿透指标仍是离线 proxy。
6. 不应基于这些诊断直接启动全量正式训练或联合 DiT；优先补充非退化独立 history，并固定关键几何 support，再做视觉 geometry-only 与 Utonia 增量对照。

## 9. 可视化入口

| 页面/文件 | 内容 | 状态 |
|---|---|---|
| [http://localhost:8911/](http://localhost:8911/) | 当前 independent-history 代表 pair 的 H.264 overlay；青色 GT、红色 geometry、绿色 motion-only | 当前服务中 |
| [`overlay_pair...mp4`](/data/gaoya/agent-data/outputs/a3_training_redesign_20260908/visual_scene_context8_20260918/validation_20260920/paired_pair_overlay_v1/overlay_pair_g01_w46_w70_seed20262001.mp4) | `history_door_g01`，0.46↔0.70 m，CPU canonical rerender + trajectory overlay | 已生成；不是原始 RGB 视频 |
| [`resampling_overlay_view/index.html`](/data/gaoya/agent-data/outputs/a3_training_redesign_20260908/visual_scene_context8_20260918/validation_20260920/resampling_overlay_view/index.html) | 固定采样/重采样轨迹对比 | 静态页面，未单独启动服务 |
| [旧版 P4 SG-O overlay](http://localhost:8844/generic-full-2175-lora1500-lineage/p4-v2-sg-o-overlay/?case=difficulty_l2_f11_h030_sr048&frame=8) | 旧版 P4 V2 的 RGB 轨迹 overlay | 旧页面；不是本轮独立 history 结果 |
| [`two_round_release/reports/index.html`](/data/gaoya/agent-data/outputs/a3_training_redesign_20260908/visual_scene_context8_20260918/two_round_release/reports/index.html) | 前两轮 predictor release 的静态报告页 | 文件存在；需自行用 HTTP 服务打开 |

当前 overlay 的背景是 physics-only episode 用同一门场景和相机做的 CPU canonical rerender，因为该批样本没有原始 RGB/MP4。页面中已明确标注这一限制。

## 10. 未解决的不确定性

- 独立 history 的 post-RGB7 外部 action/state rewrite 仍需沿真实 replay 协议最终核实。
- task geometry 是声明的碰撞几何，不等同于 RGB 可见表面；视觉前端重放和 Utonia 增量尚未在这批 history 上执行。
- dev `g06/g07` 没有非退化 pair，不能作为正向场景响应验证集。
- 穿透、接触和 collision-time 结果使用离线 OBB/ground proxy，不能解释为 simulator impulse 真值。
- 代码快照中的脚本保留原始 workspace 路径约定；本目录没有复制数据，直接运行前需要按报告设置 `PROJECT`、`ROOT` 和结果路径。

## 11. 结果原始位置

完整结果仍在：

`/data/gaoya/agent-data/outputs/a3_training_redesign_20260908/visual_scene_context8_20260918/validation_20260920/`

关键子目录：`paired_task_fit_1000/`、`resampling_train_compare_v2/`、`leaveout_geometry_500/`、`paired_history_physics_v1/`、`group_holdout_training_1500_v1/`、`paired_pair_overlay_v1/`。
