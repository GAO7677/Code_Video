# Object Query Ablation Metrics · 定义、计算与实现一览

本文档以当前代码和已生成的 `report.json` 为准，区分四套不同用途的评价。它们不应经过任意权重合成一个“总质量分”。

## 1. 评价流水线总览

| 流水线 | 主要回答 | Reference | 主报告 | 入口与核心实现 |
|---|---|---|---|---|
| Complete 25-family benchmark | 消融对轨迹、物理交互、形状、光流、外观和全局质量的影响 | 同 seed Baseline、source render、simulator GT | `<output>/<case>/seed_<seed>/report.json` | `bench.sh`; `compute_metrics_and_overlays.py`; `metric_definitions.py` |
| Head-Scope fast proxy | 已生成 M1/M2/M3 视频相对 Baseline 的快速像素/结构变化 | 同 seed Baseline | `head_scope_baseline_fast/<case>/seed_<seed>/report.json` | `compute_head_scope_baseline_metrics.py` |
| Head-Scope trajectory | 对象轨迹实际改变多少，以及 CoTracker 可观测性是否丢失 | 同 seed Baseline | `head_scope_trajectory/<case>/seed_<seed>/report.json` | `compute_head_scope_trajectory_metrics.py` |
| Head-Scope object survival | 对象是否仍保留正确身份、合理大小与非空 mask | 同 seed Baseline | `head_scope_trajectory/<case>/seed_<seed>/object_survival_report.json` | `compute_head_scope_object_survival_metrics.py` |

### 通用符号

| 符号 | 含义 |
|---|---|
| `abl`, `base`, `src`, `GT` | 消融视频、同 seed 未消融视频、source render、simulator 投影 GT |
| `p(i,t)`, `c(t)` | CoTracker 第 `i` 个点和对象中心；中心为该帧至少 4 个可见点的坐标中位数 |
| `D0` | Baseline F00 对象 mask 的 bbox 对角线，用于跨对象尺度归一化 |
| `T` | Candidate 和 reference 都有效的共同帧集合 |
| `t*` | 最后一个共同有效帧；FDE 的取值位置 |
| `v(t)` | 四帧差分速度：`[c(t+4)-c(t)]/4` |
| `M(t)` | SAM2 对象 mask |
| `u(t,x)` | 相邻帧 RAFT 光流 |

## 2. Complete benchmark：25 个指标族

优先级是实验解释优先级，不是官方排名。`P0` 首先解释物理/因果效应，`P3` 主要用于 sanity check。

| # | 优先级 | 指标与 JSON 字段 | 定义与计算 | 数值方向 | 实现路径／函数 |
|---:|---|---|---|---|---|
| 1 | P0 | GT Center-ADE Change<br>`simulator_gt_reference.center_ade_change_vs_baseline_norm` | `mean_T ‖c_abl-c_GT‖/D0 - mean_T ‖c_base-c_GT‖/D0` | `0` 表示未改变 Baseline 的 GT 误差；正值更差，负值改善 | `compute_metrics_and_overlays.py::exact_gt_trajectory_metrics()` 及 `main()` |
| 2 | P0 | Baseline-relative Center-ADE<br>`baseline_reference.center_ade_px/norm` | `mean_T ‖c_abl(t)-c_base(t)‖`；`norm` 再除以 `D0` | 越大表示消融的轨迹效应越强，不表示物理更差 | `compute_metrics_and_overlays.py::trajectory_metrics()` |
| 3 | P0 | GT Velocity Error Change<br>`velocity_vector_error_change_vs_baseline_px_per_frame` | `mean ‖v_abl-v_GT‖ - mean ‖v_base-v_GT‖` | 正值表示相对 GT 的速度向量误差增加 | `compute_metrics_and_overlays.py::exact_gt_trajectory_metrics()` 及 `main()` |
| 4 | P0 | Contact-time Error Change<br>`interaction.contact_time_error_change_frames` | Candidate mask 首次连续 2 帧接触；`Δ=abs(τ_abl-τ_GT)-abs(τ_base-τ_GT)` | 越小越好；正值表示接触时刻更差；未持续接触为 `N/A` | `compute_metrics_and_overlays.py::mask_contact()` 及 `main()` |
| 5 | P0 | Post-contact Velocity Error Change<br>`interaction.post_contact_velocity_error_change_px_per_frame` | GT 接触后 8 帧内，`mean ‖v_abl-v_GT‖ - mean ‖v_base-v_GT‖` | 越小越好；正值表示碰撞后运动更差 | `compute_metrics_and_overlays.py::post_contact_velocity_error()` |
| 6 | P0 | Other-object Center-ADE<br>`other_object.center_ade_px/norm` | 单对象消融后，未选中对象相对 Baseline 的 Center-ADE | 越大表示跨对象传播/spillover 越强 | `compute_metrics_and_overlays.py::trajectory_metrics()` 及 `main()` |
| 7 | P1 | Center-FDE<br>`center_fde_px/norm` | `‖c_abl(t*)-c_ref(t*)‖`；`norm` 再除以 `D0` | 越小越接近 reference；只看最后共同有效帧 | `compute_metrics_and_overlays.py::trajectory_metrics()` / `exact_gt_trajectory_metrics()` |
| 8 | P1 | Object-normalized PCK@5/10/20%<br>`pck_normalized.{0.05,0.1,0.2}` | `mean_(i,t) 1[‖p_abl-p_ref‖ < αD0]` | 越大越接近 reference；`1` 为全部命中 | `compute_metrics_and_overlays.py::trajectory_metrics()` |
| 9 | P1 | Native PCK@16/32/64<br>`pck_native.{16,32,64}` | 1280×704 坐标中 `mean 1[‖p_abl-p_ref‖<τ px]` | 越大越接近 reference；不是 Attention Q→K PCK | `compute_metrics_and_overlays.py::trajectory_metrics()` |
| 10 | P1 | Point-ADE<br>`point_ade_px/norm` | 所有共同可见 CoTracker 表面点的 `mean ‖p_abl-p_ref‖` | 越小越接近；球体滚动时会受表面点对应影响 | `compute_metrics_and_overlays.py::trajectory_metrics()` |
| 11 | P1 | Velocity Vector / Speed / Direction Error<br>`velocity_*_error_*` | `mean ‖v_abl-v_ref‖`；`mean abs(‖v_abl‖-‖v_ref‖)`；`mean acos(cos(v_abl,v_ref))` | 均越小越接近；方向误差只统计双方速度至少 `0.25 px/frame` 的帧 | `compute_metrics_and_overlays.py::trajectory_metrics()` / `exact_gt_trajectory_metrics()` |
| 12 | P1 | Center-aligned Shape IoU<br>`shape_*.center_aligned_iou_mean` | 仅平移 Candidate mask 使质心对齐 reference，不缩放，再计算 IoU | 越大表示形状越接近；`1` 完全一致 | `compute_metrics_and_overlays.py::shape_metrics()` |
| 13 | P1 | Area / Aspect / Circularity Error<br>`area_log_ratio_error_mean`<br>`aspect_log_ratio_error_mean`<br>`circularity_error_mean` | `abs(log(A_abl/A_ref))`；`abs(log(aspect_abl/aspect_ref))`；`abs(circ_abl-circ_ref)`，其中 `circ=4πA/P²` | 均越小越接近 reference；`0` 最接近 | `compute_metrics_and_overlays.py::mask_geometry()` / `shape_metrics()` |
| 14 | P1 | RAFT ROI Flow EPE<br>`raft.<scope>.<ref>.flow_epe_mean_px` | 在 reference 冻结 ROI 内 `mean_(t,x) ‖u_abl-u_ref‖` | 越小越接近；是两个估计光流的 disagreement，不是 flow GT | `AAA_my_test/analyze_legacy_ti2v_object_ablation_raft_motion.py::compare_flow()` |
| 15 | P1 | RAFT Motion Magnitude Ratio<br>`motion_magnitude_ratio` | `mean ‖u_abl‖ / mean ‖u_ref‖` | 接近 `1` 表示运动量接近；reference 近静止时不稳定 | `AAA_my_test/analyze_legacy_ti2v_object_ablation_raft_motion.py::compare_flow()` |
| 16 | P2 | Object DINOv2 Similarity<br>`perceptual.<ref>.dino_cosine_mean` | 固定物理 crop 尺寸、质心平移对齐、mask-pooling DINOv2 ViT-L/14 patch feature 后求 cosine | 越大表示对象身份/语义外观越接近 | `compute_perceptual.py::dino_tokens()` 及 `main()` |
| 17 | P2 | Object LPIPS<br>`perceptual.<ref>.lpips_mean` | 质心对齐、固定 crop、mask 外置灰后计算 LPIPS-Alex | 越小表示局部外观越接近；`0` 最接近 | `compute_perceptual.py::lpips_maps()` 及 `main()` |
| 18 | P2 | Outside-object LPIPS<br>`outside_object_lpips.<ref>.outside_object_lpips_mean` | 排除 Candidate/reference 对象 mask 膨胀并集，其余区域计算 LPIPS | 越小表示背景/非对象 spillover 越弱 | `compute_perceptual.py::outside_frames()` / `lpips_maps()` |
| 19 | P2 | Raw-mask IoU<br>`shape_*.raw_iou_mean` | 不做对齐，直接计算 Candidate/reference mask IoU | 越大越接近；同时混合位置和形状变化 | `compute_metrics_and_overlays.py::shape_metrics()` |
| 20 | P3 | VBench Subject Consistency<br>`vbench_subject_consistency.score/delta` | 官方 VBench 主体跨帧一致性；`delta=score_abl-score_base` | score 越大越一致；冻结视频也可能得高分，只作 sanity | `prepare_multiseed_vbench.py`; `/home/gaoya/.../AAAinfer/bench.py` |
| 21 | P3 | VBench Motion Smoothness<br>`vbench_motion_smoothness.score/delta` | 官方 VBench/AMT 运动平滑度 | 越大越平滑；不证明运动方向/物理正确 | 同上 |
| 22 | P3 | VBench Dynamic Degree<br>`vbench_dynamic_degree.score/delta` | 官方 VBench 运动充足性判定 | 越大越动态；仅用于发现冻结 | 同上 |
| 23 | P3 | VBench Quality Suite<br>`background_consistency` / `temporal_flickering` / `imaging_quality` / `aesthetic_quality` | 官方背景一致、时间闪烁、成像质量、美学质量 | 官方 score 均通常越大越好；仅检查生成崩坏/视觉质量 | 同上 |
| 24 | P3 | Full-frame SSIM / PSNR / MAE<br>`pixel.<ref>.ssim_mean/psnr_db/mae_0_1` | 统一分辨率后逐帧比较；PSNR=`10log10(255²/MSE)` | SSIM/PSNR 越大、MAE 越小表示像素更相似；静态背景会稀释对象差异 | `compute_metrics_and_overlays.py::pixel_metrics()` |
| 25 | P3 | Temporal Delta-MAE<br>`pixel.<ref>.temporal_delta_mae_0_1` | `mean abs(ΔI_abl-ΔI_ref)/255` | 越小表示逐帧像素变化模式更接近；不能当作轨迹指标 | `compute_metrics_and_overlays.py::pixel_metrics()` |

### Complete report 中的 RAFT 完整叶子指标

下列字段都由 `AAA_my_test/analyze_legacy_ti2v_object_ablation_raft_motion.py::compare_flow()` 实现，在 `object_A` / `object_B` / `all_objects` ROI 上分别对 `vs_baseline` 和 `vs_source` 计算。

| JSON 叶子字段 | 计算 | 读法 |
|---|---|---|
| `flow_epe_mean_px` | `mean ‖u_abl-u_ref‖` | 越小越接近 |
| `flow_epe_frame_p95_px` | 每帧平均 EPE 的 P95 | 越小越接近；强调高误差帧 |
| `flow_epe_over_reference_magnitude` | `mean EPE / mean ‖u_ref‖` | 越小越接近；reference 近静止时为 `N/A` |
| `flow_vector_cosine` | 所有 ROI 光流向量整体 cosine | 越接近 `1` 方向越一致 |
| `active_direction_cosine` | 双方幅值均至少 `0.25 px` 的 active pixel cosine 均值 | 越接近 `1` 越一致 |
| `active_pixel_fraction` | active pixel 数 / ROI pixel 数 | 运动支持范围诊断，无单调好坏 |
| `magnitude_mae_px` | `mean abs(‖u_abl‖-‖u_ref‖)` | 越小运动幅值越接近 |
| `reference_mean_magnitude_px` | `mean ‖u_ref‖` | reference 运动量，是分母/审计量 |
| `candidate_mean_magnitude_px` | `mean ‖u_abl‖` | Candidate 运动量，无单调好坏 |
| `motion_magnitude_ratio` | `candidate_mean/reference_mean` | 目标值 `1` |
| `motion_profile_correlation` | Candidate/reference 的逐帧平均光流幅值 Pearson 相关 | 越接近 `1` 时序运动轮廓越一致 |

### Complete report 审计字段

| 字段 | 含义 | 实现 |
|---|---|---|
| `simulator_gt_reference.center_ade_px/norm` | Candidate 相对 simulator GT 的绝对 Center-ADE；#1 的 Change 是再减去 Baseline 的该值 | `exact_gt_trajectory_metrics()` |
| `simulator_gt_reference.center_fde_change_vs_baseline_norm` | Candidate 相对 GT 的 Center-FDE，减去 Baseline 相对 GT 的 Center-FDE | `exact_gt_trajectory_metrics()` 及 `main()` |
| `source_video_reference.*` | 用 source render 作 reference 重复 Center/Point ADE、FDE、PCK 和速度误差 | `trajectory_metrics()` |
| `center_valid_frames`, `last_common_visible_frame` | Center-ADE/FDE 实际使用的帧数与末帧 | `trajectory_metrics()` |
| `point_valid_count`, `velocity_valid_count`, `direction_valid_count` | 对应指标的有效样本数 | `trajectory_metrics()` |
| `candidate_nonempty_rate`, `reference_nonempty_rate` | SAM2 mask 非空帧比例 | `shape_metrics()` |
| `candidate_contact_by_frame`, `candidate_mask_gap_px` | 逐帧接触布尔值和两对象 mask 间隙 | `mask_contact()` |
| `vbench.<metric>.baseline` | 同 seed Baseline 的官方 VBench 分数；`delta=score-baseline` | `vbench_scores()` |
| `series.*`, `*_by_frame`, `*_by_anchor` | 用于 overlay 和时序审计的逐帧/锚点值 | 各指标函数 |

## 3. Head-Scope true trajectory 与 Track Loss

这一组只使用同 seed Baseline 作 reference，不回答是否更接近 simulator GT。实现集中在 `compute_head_scope_trajectory_metrics.py::object_trajectory_metrics()`。

| 指标／JSON 字段 | 定义与计算 | 数值方向／用途 |
|---|---|---|
| `trajectory_impact_percent_d0` | `100 × mean_selected_objects(center_ade_norm)` | 越大表示相对 Baseline 的轨迹影响越强；可超过 100 |
| `target_center_ade_norm` | 单对象取该对象 Center-ADE；`all_objects` 对 A/B 宏平均 | 主排名原始值 |
| `center_ade_px/norm` | `mean_T ‖c_abl-c_base‖`；`norm` 除以 `D0` | 越大轨迹变化越强 |
| `center_fde_px/norm` | `‖c_abl(t*)-c_base(t*)‖`；`norm` 除以 `D0` | 越大表示最终共同有效位置差异越大 |
| `velocity_vector_error_px_per_frame/norm_per_frame` | 四帧差分速度向量的平均 L2 差；`norm` 除以 `D0` | 越大表示速度/方向改变越强 |
| `point_ade_px/norm` | 共同可见表面点的平均距离 | 越大表面点轨迹变化越强 |
| `pck_normalized.{0.05,0.1,0.2}` | `mean 1[point distance < αD0]` | 越小表示与 Baseline 轨迹一致性越低 |
| `common_center_coverage` | 共同中心有效帧 / Baseline 中心有效帧 | 越大证据覆盖越完整 |
| `track_retention_score_0_100` | `100 × common_center_coverage` | 越大表示 CoTracker 可观测性保留越好 |
| `track_loss_score_0_100` | `100 × (1-common_center_coverage)` | 越大表示可跟踪中心帧丢失越多；不能单独证明对象消失 |
| `target_mean_track_loss_score_0_100` | 所选对象 Track Loss 的均值 | 用于报告总体可观测性 |
| `target_worst_track_loss_score_0_100` | 所选对象 Track Loss 的最大值 | Track Loss 主排名，防止 A/B 平均掩盖单对象丢失 |
| `quality_pass` | 至少 4 个共同中心帧且 `coverage≥0.8`；`all_objects` 要求 A/B 都通过 | 失败时 ADE/FDE/PCK 不排名，记 `N/A` |
| `baseline_center_valid_frames` / `common_center_valid_frames` | Baseline 可定义中心的帧数，以及 Candidate/Baseline 共同有效帧数 | 质量门控的原始计数 |
| `last_common_visible_frame` | 最后共同有效帧索引 | FDE 实际取值时刻 |
| `velocity_valid_count` / `point_valid_count` | 速度误差和 Point-ADE/PCK 使用的有效样本数 | 统计支持量审计 |
| `series.center_distance_px` / `series.velocity_vector_error_px_per_frame` | 49 帧中的逐帧中心距离和四帧差分速度误差；无效位置为 `null` | 用于定位偏差从何时出现 |
| `trajectory_rank_within_case_seed` | 对通过门控的 `trajectory_impact_percent_d0` 降序 | `1` 表示轨迹改变最大 |
| `track_loss_rank_within_case_seed` | 对 `target_worst_track_loss_score_0_100` 降序 | `1` 表示 CoTracker 可观测丢失最大 |

`ADE` 使用整段共同有效轨迹的平均距离，回答“整个过程改变多少”；`FDE` 只使用最后共同有效帧，回答“最终结果偏到哪里”。

## 4. Head-Scope object retention / disappearance

实现集中在 `compute_head_scope_object_survival_metrics.py::object_survival_metrics()`。对对象 `o` 和帧 `t`：

`alive(o,t) = mask_nonempty AND identity_cosine≥threshold_o AND 0.25≤area_abl/area_base≤4.0`。

| 指标／JSON 字段 | 定义与计算 | 数值方向／用途 |
|---|---|---|
| `f00_prompt_iou` | SAM2 F00 预测 mask 与冻结 prompt mask 的 IoU | 初始化质量审计；至少 `0.50` 才通过 |
| `identity_similarity_mean` | Candidate 和同帧 Baseline 对象的 mask-pooled DINOv2 cosine 均值 | 越大表示对象身份越接近 |
| Identity threshold | 若 Baseline 同对象时序 cosine Q05 高于跨对象 cosine Q95，取两者中点；否则取时序 Q05；实现为 `compute_head_scope_object_survival_metrics.py::calibrate_identity_thresholds()` | 只用 Baseline 标定，不读取消融结果 |
| `identity_failure_rate` | `mean 1[identity_cosine<threshold]` | 越大表示身份不一致帧越多 |
| `area_failure_rate` | `mean 1[area_ratio∉[0.25,4.0]]` | 越大表示极端尺寸破坏帧越多 |
| `empty_mask_rate` | `mean 1[mask area=0]` | 越大表示 SAM2 无 mask 帧越多；仍可受 SAM2 失败影响 |
| `survival_rate` | `mean_t alive(t)` | 越大对象保留越好 |
| `retention_score_0_100` | `100 × survival_rate` | 越大保留越好 |
| `disappearance_score_0_100` | `100 × (1-survival_rate)` | 越大失败越严重；包含真消失、身份替换和极端大小破坏 |
| `target_mean_disappearance_score_0_100` | 所选对象 disappearance score 的均值 | 宏平均审计量 |
| `target_worst_disappearance_score_0_100` | 所选对象 disappearance score 的最大值 | Object Retention Failure 主排名 |
| `target_mean_mask_absence_score_0_100` | `100 ×` 所选对象 `empty_mask_rate` 均值 | 平均 mask absence |
| `target_worst_mask_absence_score_0_100` | `100 ×` 所选对象 `empty_mask_rate` 最大值 | 更接近“纯消失”的主排名，但仍非绝对 GT |
| `quality_pass` | 所选对象的 `f00_prompt_iou` 都至少 `0.50` | 失败时 target 级保留/absence 分数为 `N/A` |
| `alive_frame_count` / `frame_count` | `alive=true` 帧数和总帧数 | `survival_rate` 的原始计数 |
| `first_sustained_loss_frame` | 首个连续 3 帧 `alive=false` 的起始帧 | 越早表示持续丢失更早发生 |
| `terminal_missing_rate` | 最后 8 帧中 `alive=false` 的比例 | 越大表示结尾保留越差 |
| `series.alive` / `series.identity_similarity` / `series.area_ratio_vs_baseline` | 逐帧 alive 判定、DINOv2 cosine 和 mask 面积比 | 用于区分无 mask、身份替换和尺寸破坏 |
| `disappearance_rank_within_case_seed` | `target_worst_disappearance_score_0_100` 降序 | `1` 表示对象保留失败最严重 |
| `mask_absence_rank_within_case_seed` | `target_worst_mask_absence_score_0_100` 降序 | `1` 表示 mask absence 最严重 |

## 5. Head-Scope fast proxy 全部指标

这套指标在 320×176 上计算，用于快速筛选和页面排名。它可能被外观、形变和闪烁放大，所以 **Temporal Delta-MAE 不是轨迹相似度**。实现集中在 `compute_head_scope_baseline_metrics.py::compare_frames()` 和 `assign_ranks()`。

| 指标／JSON 字段 | 计算 | 数值方向／用途 |
|---|---|---|
| `global.ssim_mean` | 49 帧全画面 `mean SSIM(I_abl,I_base)` | 越小结构变化越大 |
| `global.psnr_db` | `10log10(255²/MSE)` | 越小像素变化越大 |
| `global.mae_0_1` | `mean abs(I_abl-I_base)/255` | 越大全局像素变化越大 |
| `global.temporal_delta_mae_0_1` | `mean abs(ΔI_abl-ΔI_base)/255` | 越大时序外观变化越大；不可解释为轨迹差 |
| `target_roi.mae_0_1` | Baseline 冻结 CoTracker tube 凸包 ROI 内 RGB MAE | 越大目标位置/外观变化越大 |
| `target_roi.temporal_delta_mae_0_1` | 目标 ROI 相邻帧并集内 Delta-MAE | 越大目标区动态像素变化越大 |
| `outside_objects.mae_0_1` | 排除全部冻结对象 ROI 后的 RGB MAE | 越大静态 spillover 越强 |
| `outside_objects.temporal_delta_mae_0_1` | 对象 ROI 外 Delta-MAE | 越大动态 spillover 越强 |
| `global_appearance` | `100[0.5(1-SSIM)+0.5 global_MAE]` | 全局外观影响排名；越大影响越强 |
| `target_local` | `100 × target_ROI_MAE` | 目标局部影响排名 |
| `temporal_appearance` | `100[0.4 global_DeltaMAE+0.6 target_ROI_DeltaMAE]` | 时序外观影响排名，不是轨迹排名 |
| `outside_spillover` | `100 × mean(outside_MAE,outside_DeltaMAE)` | 对象外传播影响排名 |
| `spillover_score_0_100` | 与 `outside_spillover` 同一公式 | 兼容性字段；越大 spillover 越强 |
| `impact_score_0_100` | `100[0.20(1-SSIM)+0.15 global_MAE+0.15 global_DeltaMAE+0.30 target_MAE+0.20 target_DeltaMAE]` | 仅表示可见干预强度，不是生成质量/物理正确性 |
| `*_rank_within_case_seed` | 对对应影响分数降序，并平均化 ties | `1` 表示该类影响最强 |
| `impact_percentile_within_case_seed` / `category_percentiles_within_case_seed.*` | 由对应 rank 线性映射到 0–100 percentile | 越大表示在当前 case/seed 中该类影响越强 |

`target_roi.mean_area_fraction` 和 `outside_objects.mean_area_fraction` 是计算支持区域占全帧的比例，属于审计量，不是效果好坏分数。

## 6. 实现与中间产物路径

| 功能 | 代码路径 | 主要产物 |
|---|---|---|
| 一键调度 | `AAA_my_test/object_query_ablation_metrics/bench.sh` | 串行调度 VBench、CoTracker、SAM2、RAFT、DINO/LPIPS、统计、overlay、验证与聚合 |
| 25 指标族展示元数据 | `AAA_my_test/object_query_ablation_metrics/metric_definitions.py` | `METRIC_DEFINITIONS` |
| CoTracker 轨迹提取 | `AAA_my_test/object_query_ablation_metrics/extract_tracks.py` | `tracks/*.npz` |
| SAM2 mask 提取 | `AAA_my_test/object_query_ablation_metrics/extract_masks.py` | `masks/*.npz` |
| Candidate/Baseline RAFT | `AAA_my_test/analyze_legacy_ti2v_object_ablation_raft_motion.py` | `raft_motion_top100_v1/flows/*.npy` |
| Source-render RAFT | `AAA_my_test/object_query_ablation_metrics/extract_source_raft.py` | `raft/source_gt_video.npy` |
| DINOv2 / LPIPS | `AAA_my_test/object_query_ablation_metrics/compute_perceptual.py` | `perceptual/perceptual_metrics.json` 及 montages |
| Complete metrics 与 overlay | `AAA_my_test/object_query_ablation_metrics/compute_metrics_and_overlays.py` | `report.json`; trajectory/mask/pixel/RAFT overlays |
| 官方 VBench 输入组织 | `AAA_my_test/object_query_ablation_metrics/prepare_multiseed_vbench.py` | `vbench/index`; per-video manifests |
| 官方 VBench driver | `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py` | 7 个 `eval_summary_*.json` |
| 完整性验证 | `AAA_my_test/object_query_ablation_metrics/validate_outputs.py` | 缺失/非法指标报错 |
| 六 seed 共同 cohort 聚合 | `AAA_my_test/object_query_ablation_metrics/aggregate_reports.py` | `<case>/aggregate/report.json` |
| Head-Scope 快速代理 | `AAA_my_test/object_query_ablation_metrics/compute_head_scope_baseline_metrics.py` | `head_scope_baseline_fast/.../report.json` |
| Head-Scope 真轨迹 | `AAA_my_test/object_query_ablation_metrics/compute_head_scope_trajectory_metrics.py` | `head_scope_trajectory/.../report.json`; trajectory overlays |
| Head-Scope 对象保留 | `AAA_my_test/object_query_ablation_metrics/compute_head_scope_object_survival_metrics.py` | `object_survival_report.json`; masks/features/survival overlays |
| Head-Scope 排名 Markdown | `AAA_my_test/object_query_ablation_metrics/build_head_scope_trajectory_ranking_md.py` | `TRAJECTORY_METRICS_COMPLETE_RANKING.md` |
| 8092 页面 | `AAA_my_test/serve_latent_block_head_viewer_with_metrics.py` | `/object-query-ablation-metrics` 与 M1/M2/M3 视频隐藏指标栏 |

## 7. 解读约束

1. `vs Baseline` 衡量“消融产生多大效果”，不自动等于“生成变差”。
2. `vs source/simulator GT` 才能用于讨论物理忠实度；source render 也不等于点对应 GT。
3. Center-ADE 通过质量门控的子集存在选择效应；必须同时查看 Track Loss、Object Retention Failure 和 Mask Absence。
4. RAFT 比较的是两个光流估计，不是真实光流 GT。
5. DINOv2、LPIPS、SSIM、MAE 是外观/像素相似度，不能代替轨迹和接触时序评价。
6. VBench 七项指标主要用于发现冻结、闪烁或全局崩坏，不应作为消融的核心物理结论。
