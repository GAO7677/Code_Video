# V-JEPA 引导 → WMReward 优化：进展与后续计划

最后更新：2026-07-03

## 目标

找到一组 V-JEPA 引导配置，能在生成视频上**实际提升 wmreward 物理指标**
（`surprise` 越低越好 / `similarity` 越高越好）。用 context-anchored 能量变化
作为**过程信号**（引导是否真的扰动了采样），用 wmreward 作为**结果信号**
（扰动是否改善了物理）。

约束：
- **不修改** `physv_eval/` 下的指标代码，只 `import score_case` / `WMRewardRunner` 调用。
- 复用同一个常驻的 `WMRewardRunner`（它会缓存 V-JEPA 模型），批量打分。
- 先在单个 case (`025_Solid_Mechanics_0002`) 上跑通，再在 3–5 个 case 上验证。
- 新脚本放在 `code_vjepa_free/vjepa_guidance/`，输出到
  `/data/gaoya/agent-data/outputs/probe_sweep/`。

## 关键背景（此前实验的结论）

- context-anchored 能量的方向是**对的**（line-search 确认），但**基本是平的**：
  baseline 能量在 40 步里都是 0.484–0.506（cosine-sim ≈ 0.50，噪声带 ±0.01）。
- 引导只能把能量降 ~0.005，落在噪声底噪里。
- 步长必须很小（~0.005），0.02 单步会 overshoot（能量反而上升）。
  详见 memory: `vjepa-guidance-step-size`、`vjepa-guidance-context-anchored`。

## 已完成

### Phase 0a — 建立耦合关系（决定性的便宜实验）✅

脚本：`score_guided_videos.py`（只读，包一层 `WMRewardRunner`，单进程缓存模型）。

对 phase4 已有的 baseline + 6 个引导视频打 wmreward，并与记录的
`mean_delta_post`（能量持续性）做联表：

| label | mean_delta_post | surprise | Δsurprise vs base |
|---|---|---|---|
| anch_dense20_bt   | -0.00691 | 0.6898 | +0.0013 |
| anch_dense20_s003 | -0.00538 | 0.6886 | +0.0001 |
| anch_dense12_bt   | -0.00234 | 0.6901 | +0.0016 |
| anch_dense12_s005 | -0.00211 | 0.6887 | +0.0002 |
| anch_dense6_s005  | -0.00192 | 0.6884 | -0.0001 |
| anch_single_s005  | -0.00033 | 0.6884 | -0.0002 |

baseline surprise = 0.6885。

**结论（决定性）：**
- wmreward 在所有引导视频上几乎不动，全体落在 ±0.0016 的噪声带里。
- 若有任何趋势，反而是**反相关**：能量降得最多的两个（dense20_bt、dense12_bt）
  surprise 最高（更差）。所以「把 anchored 能量降 ~0.005」对 wmreward 没有好处。
- 唯一的正向线索：`anch_span3_s05`（步长 0.05，此前那批里最大）给出最大 wmreward
  增益（-0.0016）。**是强度、而不是能量持续性，才动了结果。**
- 诚实的判断：这些视频彼此太像，根本无法测出耦合——引导太弱了。

→ 明确指向 **Phase 1（强度阶梯）**，而不是 Phase 3（换能量目标）。
在拿到「真的和 baseline 不同」的视频之前，无法诊断能量目标对不对。

记录在 memory: `vjepa-guidance-wmreward-coupling`。

产物：`/data/gaoya/agent-data/outputs/probe_sweep/phase4/wmreward_scores.json`

### Phase 0b — 量化引导对 latent 的实际写入量 ✅（已埋点，待随 Phase 1 采集）

在 `wan_openvid_0613pybullet_lorav2v_vjepa.py` 的 `_apply_context_anchored_guidance`
的 stats 里加了三个量（不改任何指标代码）：
- `correction_l2` = ‖step · grad‖（这一步实际施加的位移）
- `latent_l2` = ‖latent_xt‖（被扰动对象的尺度）
- `correction_ratio` = correction_l2 / latent_l2（相对扰动；比值极小 ⇒ 后续去噪几乎必然抹掉）

并把这三个量加进了每个引导步的 `logging.info` 行。运行 Phase 1 时会自动落进日志/trace。

## 待办

### Phase 1 — 强度阶梯（找到「视觉上开始变化」的拐点）✅ 已跑通单 case

已在 `probe_energy_persistence.py` 里实现为 **`--phase 5`**（`_phase5_conditions`）：

- 时序固定在密集带 p35→p80（12 步，phase4 里能量降得最持续的那段）。
- 只扫强度轴：`latent_step_size ∈ {0.01, 0.02, 0.05, 0.1, 0.2}`（rms 归一化）。
- 外加两个归一化变体（在 0.05 处）：`grad_norm="none"`、`grad_norm="l2"`，
  测试是不是 rms rescaling 把效果压住了。

配套代码改动（均已完成、语法通过）：
- `_run_condition` / `_run_phase` 新增 `grad_norm` 逐条件覆盖参数。
- CLI `--phase` 增加选项 5；main 里加 phase 5 分派（用 anchor_timing/anchor_step_size 定心）。

**运行命令**（参照 phase4，改 `--phase 5`）：
```bash
PYTHONPATH=.../Code_vjepa_vggt:.../DiffSynth-Studio-main:.../train_0419 \
CUDA_VISIBLE_DEVICES=<gpus> \
/data/gaoya/miniconda3/envs/wan/bin/python probe_energy_persistence.py \
  --weights-root .../step-000500 \
  --input-json .../physicIQ_025_...trimmed.json \
  --context-path .../context_video_8f.mp4 \
  --output-dir /data/gaoya/agent-data/outputs/probe_sweep \
  --phase 5 --anchor-timing 0.35 --anchor-step-size 0.005 \
  --device cuda:0 --vjepa-device cuda:1 \
  --num-inference-steps 40 --seed 42
```
（注意：生成/probe 用的 python 需带 diffusers+decord，`wan` 或 `vjepa2` 环境都行；
`vphy` 环境缺 diffusers，不能用来跑 wmreward runner。）

单 case 结果（`025_Solid_Mechanics_0002`）已跑完，产物：
- `phase5/phase5_summary.json`
- `phase5/wmreward_scores.json`
- `phase5/videos/*.mp4`

关键信号：
- `ladder_s01`: `mean_delta_post = 0.002446`，wmreward surprise `0.6935`，比 baseline `0.6928` 略差。
- `ladder_s02`: `mean_delta_post = 0.007497`，wmreward surprise `0.6945`，略差。
- `ladder_s05`: `mean_delta_post = 0.024246`，wmreward surprise `0.6993`，明显变差。
- `ladder_s10`: `mean_delta_post = 0.098149`，wmreward surprise `0.6916`，首次略优于 baseline。
- `ladder_s20`: `mean_delta_post = 0.116529`，wmreward surprise `0.6485`，`similarity = 0.3515`，
  显著优于 baseline 的 `0.3072`；`physics_iq = 16.10`，也高于 baseline 的 `13.26`。

机制结论：
- `rms` 归一化不是在压效果，而是在把极弱梯度放大到可用区间。
- `grad_norm="none"` / `"l2"` 在 `0.05` 处几乎写不进 latent，`mean_delta_post` 仅
  `0.000131 / 0.000304`，说明当前有效强度来自 `rms` 重标定。
- `correction_ratio` 在有效配置上已进入可见量级：
  `s01 ~1%`，`s02 ~2%`，`s05 ~6%`，`s10 ~12%`，`s20 ~20%`。

**结论（当前最重要）：**
- “方向对但太弱”这个 Phase 0 判断已被数据证实。
- 当强度足够大时，wmreward **会响应**，所以此时**不应**优先回头换能量目标。
- 当前最有希望的区域在 `latent_step_size ≈ 0.10–0.20`，其中 `0.20` 在这个 case 上
  给出了最强的正向 wmreward / similarity / physics_iq 改善。

**目标**：找到最小的、能让 (a) 解码视频肉眼可辨地偏离 baseline **且** (b) wmreward
移动的步长。预期三段：太弱（无变化）→ 有用（物理改善）→ 太强（artifact、wmreward 变差）。
要的是那个拐点。

### Phase 2 — 拐点附近的 timing × inner-K 精调 ✅ 单 case 已完整跑分

Phase 1 找到能动结果的步长后：
- **timing**：早（高噪声 p10–p40）vs 中（p35–p65）vs 晚（p60–p90）。平的能量暗示
  可能需要更早、更强的推动。
- **inner_k**（每步重复修正次数）：{1, 2, 4}，比加步数便宜。已在 pipeline 里支持
  （`vjepa_inner_k`，`_run_condition` 的 `inner_k`）。
- **backtracking** 开/关（步长够大到会 overshoot 时才有意义）。
每个配置都用 wmreward 打分，能量轨迹 + 像素差留作诊断。

已在 `probe_energy_persistence.py` 里加了 **`--phase 6`**（`_phase6_conditions`）：
- `knee_mid_s12 / s15 / s18`
- `knee_mid_s15_bt`
- `knee_early_s15`
- `knee_late_s15`
- `knee_mid_s10_k2`
- `knee_mid_s075_k2`

另外，`phase5_summary.json` / `phase6_summary.json` 这类摘要在 phase 5/6 下改成按
`abs(mean_delta_post)` 优先排序，避免“最没写进去的弱扰动”因为 persistence 高而被误判为 best。

单 case 结果（`025_Solid_Mechanics_0002`）现已完整跑完，产物：
- `phase6/phase6_summary.json`
- `phase6/wmreward_scores.json`
- `phase6/phase6_multimetric_scores.json`
- `phase6/videos/*.mp4`

关键信号：
- `knee_mid_s15_bt`：`mean_delta_post = -0.002003`，`surprise = 0.6942`，比 baseline `0.6928`
  略差；这再次证明 backtracking 会退回“几乎没写进去”的弱扰动区。
- `knee_mid_s075_k2`：`mean_delta_post = 0.096013`，`surprise = 0.6756`，已优于 baseline，
  但改善幅度明显小于更强配置。
- `knee_mid_s10_k2`：`mean_delta_post = 0.111499`，`surprise = 0.6626`，`similarity = 0.3374`，
  `physics_iq = 15.36`，`videophy2_pc = 5`。
- `knee_early_s15`：`mean_delta_post = 0.118585`，`surprise = 0.6623`，`similarity = 0.3377`，
  `physics_iq = 15.35`，`videophy2_pc = 5`。
- `knee_mid_s15`：`mean_delta_post = 0.114565`，`surprise = 0.6684`，`physics_iq = 15.47`，
  `videophy2_pc = 5`。
- `knee_mid_s18`：`mean_delta_post = 0.118087`，`surprise = 0.6582`，`similarity = 0.3418`，
  `physics_iq = 16.44`，`videophy2_pc = 5`。
- `knee_late_s15`：`mean_delta_post = 0.110535`，`surprise = 0.6663`，`physics_iq = 17.06`，
  但 `videophy2_pc = 4`，没有随 wmreward 同步上涨。
- `cosmos_reason1` 在所有 phase6 视频上都为 `1`，没有提供可分辨排序力。

和 phase5 的最优 `ladder_s20` 对比：
- `ladder_s20` 仍是当前 **最佳 wmreward**：`surprise = 0.6485`，`similarity = 0.3515`，
  `physics_iq = 16.10`，`videophy2_pc = 5`。
- Phase 6 中最接近它的是 `knee_mid_s18`：`surprise = 0.6582`，`similarity = 0.3418`，
  `physics_iq = 16.44`，`videophy2_pc = 5`。
- `knee_early_s15` / `knee_mid_s10_k2` 也很稳，说明“更早时机”与“较小步长 + inner_k=2”
  都能进入有效区，但目前还没有超过 `s20` 的中段强推。

**Phase 2 结论：**
- strong fixed-step 配置在 timing / inner-K 上确实都会写进结果视频，并且多数会带来
  一致的 wmreward 改善。
- backtracking 在当前能量地形下不合适；它过于保守，等价于把 guidance 强度降回无效区。
- 当前最值得进入多 case 验证的候选是：
  - `ladder_s20`（phase5 总冠军）
  - `knee_mid_s18`
  - `knee_early_s15`
  - `knee_mid_s10_k2`

### Phase 3 —（仅当 Phase 0/1 显示解耦时）重审能量目标 ⏳ 条件触发

如果 wmreward 对 anchored 能量下降**没反应**，说明固定 future-feature anchor 可能
不是对的目标。按代价从低到高：
- `reduction="max"`（最差窗口）替代 `"mean"`——平的均值可能把唯一的坏窗口平均掉了。
- 加宽 future 视野（`window_size` / `context_frames` 切分），让能量看到更多生成的运动。
- 用 **surprise**（自一致性）能量在大步长下做对照——若它能动 wmreward 而 anchored 不能，
  问题就在 anchor。
只在有证据时触发，不投机做。

当前更新（2026-07-03, pilot3 完成后）：
- pilot3 表明强 fixed-step guidance 已经**稳定地**推动了 `wmreward`，所以严格说
  “wmreward 对强 guidance 不响应”这一触发条件并不成立。
- 但 pilot3 也表明当前 top-4 fixed-step 配置之间继续微调强弱，**并不能**缓解
  `physics_iq / cosmos_reason1` 的张力。因此，若后续目标是“保住 wmreward 提升并减少
  cross-metric 冲突”，优先级已经转向 Phase 3。
- 一个重要实现细节：当前 `reduction="max"` 只对**滑窗 self-consistency surprise**
  是直接有意义的；现有 `context_anchored` loss 是单个 anchored clip 的 feature mismatch，
  还没有“多窗口取 max”的结构。所以当前最低成本的 Phase 3 入口不是先切 `reduction=max`，
  而是**固定 `ladder_s20` 调度，先加宽 `window_size`**，让 anchored target 看更长的未来。
- 已在 `probe_energy_persistence.py` 中加入 **Phase 7**：
  - `target_w16` / `target_w24` / `target_w32`
  - 都固定用 `ladder_s20` 的强 guidance 时序与步长
  - 只比较 anchored future horizon 的改变
  - 由于不同 `window_size` 对应的是不同 energy signature，脚本现在会为每种 signature
    单独缓存 baseline，并在 summary 里写 `baseline_signature_key` / `energy_signature`，
    避免把不同基准下的 `mean_delta_post` 混为一谈

### Phase 4 — 在 3–5 个 case 上验证胜出配置 ⏳ 已启动，已有首批多 case 结果

拿 Phase 1–2 的最优配置，对 3–5 个 v2v case 跑 baseline vs 引导。报告每个 case 及
平均 wmreward delta，再加一个副指标（`videophy2 --task pc` 和/或 `cosmos_reason1`）
确认 wmreward 的提升不是在钻单一指标的空子。用已有的 HTML compare viewer 做可视检查。
（本 case 有真实续帧 `physicIQ_0002_clip_2p5s_3p5s.mp4`，Phase 4 可用 `physics_iq` 做参照打分。）

目前已补齐 Phase 4 所需脚本：
- `run_phase4_multicase.py`
  - 复用 Wan2.2 LoRA 批量入口，直接跑当前 top 配置：
    `baseline / ladder_s20 / knee_mid_s18 / knee_early_s15 / knee_mid_s10_k2`
  - 用 **exact target_step_indices** 复现 phase5/6 里真正验证过的 winner，
    不再用百分比近似回推步号。
  - 显式传 `--vjepa-guidance-mode context_anchored`，避免误回退到 `surprise` 路径。
- `score_multicase_methods.py`
  - 按方法目录逐 case 打分；
  - 每个视频从 sidecar 里的 `input_json` 反查 `source_video`，从而逐 case 调用
    `physics_iq`；
  - 只读调用 `WMRewardRunner` / `score_case`，不修改任何指标代码。

首批 pilot case（3 个）：
- `physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed`
- `0613pybullet_sample_001460_w002`
- `phyco_kubric_cube_deform_soft_v2_noeff_2025-09-08_fe6f35`

**已完成的 pilot3 全量多 case 对比：baseline + 4 个候选（3/3 cases）**

产物：
- `phase4_pilot3_baseline_vs_ladder_s20_scores.json`
- `phase4_pilot3_all_methods_scores.json`

方法均值（按 `mean_delta_surprise_vs_baseline` 排序）：
- baseline:
  - `mean_surprise = 0.69585`
  - `mean_similarity = 0.30415`
  - `mean_physics_iq = 34.25`
  - `mean_videophy2_pc = 3.00`
  - `mean_cosmos_reason1 = 1.67`
- `ladder_s20`（当前 pilot3 总冠军）:
  - `mean_surprise = 0.65595`
  - `mean_similarity = 0.34405`
  - `mean_delta_surprise_vs_baseline = -0.03990`
  - `mean_delta_similarity_vs_baseline = +0.03990`
  - `mean_delta_videophy2_pc_vs_baseline = +0.67`
  - `mean_delta_physics_iq_vs_baseline = -30.87`
  - `mean_delta_cosmos_reason1_vs_baseline = -0.67`
- `knee_early_s15`:
  - `mean_surprise = 0.66846`
  - `mean_similarity = 0.33154`
  - `mean_delta_surprise_vs_baseline = -0.02739`
  - `mean_delta_similarity_vs_baseline = +0.02739`
  - `mean_delta_videophy2_pc_vs_baseline = +0.33`
  - `mean_delta_physics_iq_vs_baseline = -30.99`
  - `mean_delta_cosmos_reason1_vs_baseline = -0.67`
- `knee_mid_s18`:
  - `mean_surprise = 0.66853`
  - `mean_similarity = 0.33147`
  - `mean_delta_surprise_vs_baseline = -0.02732`
  - `mean_delta_similarity_vs_baseline = +0.02732`
  - `mean_delta_videophy2_pc_vs_baseline = +0.33`
  - `mean_delta_physics_iq_vs_baseline = -30.36`
  - `mean_delta_cosmos_reason1_vs_baseline = -0.67`
- `knee_mid_s10_k2`:
  - `mean_surprise = 0.66936`
  - `mean_similarity = 0.33064`
  - `mean_delta_surprise_vs_baseline = -0.02648`
  - `mean_delta_similarity_vs_baseline = +0.02648`
  - `mean_delta_videophy2_pc_vs_baseline = +0.33`
  - `mean_delta_physics_iq_vs_baseline = -29.67`
  - `mean_delta_cosmos_reason1_vs_baseline = -0.67`

逐 case 的 `wmreward` 变化（`ladder_s20`）：
- `0613pybullet_sample_001460_w002`: `0.6948 -> 0.6604` (`Δsurprise = -0.0344`)
- `phyco_kubric_cube_deform_soft_v2_noeff_2025-09-08_fe6f35`:
  `0.6854 -> 0.6591` (`Δsurprise = -0.0262`)
- `physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed`:
  `0.7074 -> 0.6483` (`Δsurprise = -0.0591`)

逐 case 的 `wmreward` 变化（其余 3 个候选）：
- `knee_mid_s18`
  - `0613pybullet_sample_001460_w002`: `0.6948 -> 0.6657` (`Δsurprise = -0.0291`)
  - `phyco_kubric_cube_deform_soft_v2_noeff_2025-09-08_fe6f35`: `0.6854 -> 0.6732` (`Δsurprise = -0.0122`)
  - `physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed`: `0.7074 -> 0.6667` (`Δsurprise = -0.0407`)
- `knee_early_s15`
  - `0613pybullet_sample_001460_w002`: `0.6948 -> 0.6743` (`Δsurprise = -0.0205`)
  - `phyco_kubric_cube_deform_soft_v2_noeff_2025-09-08_fe6f35`: `0.6854 -> 0.6627` (`Δsurprise = -0.0226`)
  - `physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed`: `0.7074 -> 0.6684` (`Δsurprise = -0.0390`)
- `knee_mid_s10_k2`
  - `0613pybullet_sample_001460_w002`: `0.6948 -> 0.6718` (`Δsurprise = -0.0230`)
  - `phyco_kubric_cube_deform_soft_v2_noeff_2025-09-08_fe6f35`: `0.6854 -> 0.6665` (`Δsurprise = -0.0189`)
  - `physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed`: `0.7074 -> 0.6698` (`Δsurprise = -0.0376`)

当前解读：
- 这是一个**稳定的正向多 case 信号**：4 个 guided 配置在这 3 个 pilot case 上的
  `wmreward` 都比 baseline 更好，没有出现单 case 反向。
- 当前排序已经比较稳定：`ladder_s20` 明显领先，其余 3 个候选彼此非常接近，
  且都落后于 `ladder_s20` 约 `0.0125–0.0134` 的 mean surprise 改善幅度。
- `videophy2_pc` 也与 `wmreward` 大体同向：
  - `ladder_s20` 从 baseline 的 `3.00` 提升到 `3.67`
  - 其余 3 个候选都提升到 `3.33`
- 但 `physics_iq` 和 `cosmos_reason1` 的张力没有被更弱配置消解：
  - 4 个 guided 配置的 `mean_cosmos_reason1` 都从 baseline 的 `1.67` 降到 `1.00`
  - 4 个 guided 配置的 `mean_physics_iq` 都从 baseline 的 `34.25` 大幅降到 `3.26–4.58`
- 这意味着当前 pilot3 的结论不是“存在一个更温和的 config，能同时保住 wmreward 提升并显著少伤 `physics_iq` / `cosmos_reason1`”；
  相反，更像是：
  - `wmreward` / `videophy2_pc` 所偏好的 reference-free predictive plausibility
    与 GT-reference judge (`physics_iq`) 存在结构性张力；
  - 在当前 4 个候选里，**最强 `wmreward` 点仍然就是 `ladder_s20`**，
    没有出现更平衡的明显替代者。
- 实务上，Phase 4 的 pilot3 已经给出一个清晰结论：
  - 若主目标是 `wmreward`，当前应该继续以 `ladder_s20` 为主配置；
  - 若后续要缓解 cross-metric 冲突，优先级不再是继续在这 4 个 fixed-step 配置里微调强弱，
    而是要么扩展 case 集进一步确认张力是否稳定存在，要么进入 Phase 3 重审能量目标/窗口定义。

## 交付物

- `score_guided_videos.py` — 只读批量打分器（各 phase 复用）。✅
  现在支持：
  - `wmreward`
  - `physics_iq`
  - `videophy2-task pc`
  - `cosmos-reason1`
  - `--merge-from-json` + `--skip-wmreward` 复用已有 wmreward 结果，只补交叉指标
  - 从 `phaseN_summary.json` 继承每个条件的元数据；对 Phase 7 这类多 energy-signature
    phase，会把 `baseline_signature_key` / `energy_signature` 一并写进评分结果，避免后续解释时
    丢失“这条视频对应哪种 anchored target”的上下文
- `run_phase7_target_shape.py` — Phase 7 单 case 一键 runner（先跑生成，再对
  `phase7/videos` 目录做 `wmreward + physics_iq + videophy2-pc + cosmos_reason1` 打分）。✅
- `wait_for_phase7_gpus.py` — Phase 7 前台等待器：轮询指定物理 GPU，等显存/利用率连续
  低于阈值后，再自动启动 `run_phase7_target_shape.py`。✅
  作用不是后台偷跑，而是在当前外部训练 / probe 作业结束时间不稳定时，避免人工持续盯卡。
  当前状态（2026-07-04 00:16 UTC 起）：
  - 已在前台启动，监控物理 `gpu6/7`
  - 条件：`memory_used <= 8000 MiB` 且 `utilization <= 20%`
  - 需要连续 `3` 次轮询满足条件，轮询间隔 `30s`
  - 启动后首个观测仍为 busy：`gpu6 ~3513 MiB @ 96%`，`gpu7 ~3617 MiB @ 96%`
- 引导 stats 里的能量/修正量埋点（不改指标代码）。✅
- 每个 phase 一张结果表（JSON + 简短 markdown）：配置 → 能量 delta → 修正 L2 →
  像素差 → wmreward（Phase 4 再加 judge 分）。
- 更新 memory：强度阈值发现、能量↔wmreward 耦合结论。✅（耦合已记）

## 相关文件

代码（都在 `code_vjepa_free/vjepa_guidance/`）：
- `probe_energy_persistence.py` — 扫描驱动（Phase 1 = `--phase 5`，Phase 2 = `--phase 6`）
- `run_phase7_target_shape.py` — Phase 7 单 case 编排器（生成 + 打分）
- `wait_for_phase7_gpus.py` — Phase 7 GPU 等待器（前台轮询后自动接力运行）
- `wan_openvid_0613pybullet_lorav2v_vjepa.py` — 引导 pipeline（Phase 0b 埋点在此）
- `vjepa_surprise.py` — 能量计算（`context_anchored` / `precompute_future_prediction`）
- `score_guided_videos.py` — wmreward 批量打分（只读调用指标）

指标（只调用，**不改**）：`Code_try0526/physv_eval/single_case/`（wmreward / physics_iq / videophy2 / cosmos_reason1 ...）

完整规划：`~/.claude/plans/vjepa-guidance-wmreward-search.md`
