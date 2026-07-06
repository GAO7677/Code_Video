# V-JEPA Guidance 实验记录

创建日期：2026-07-05
维护者：Codex + gaoya
范围：`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance`

这个文件是 `vjepa_guidance/` 目录下实验的统一登记册。从现在开始，凡是在这个项目下执行的新实验，都应在这里补充记录。

每条实验记录至少包含：

- 日期或日期范围
- 实验名称 / 目标
- 主要代码入口
- 模型 / preset / 输入范围
- 输出目录
- 分数文件或汇总文件
- 当前状态
- 简短结论

相关配套文档：

- 代码概览：
  `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/README.md`
- 进度备忘：
  `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/PROGRESS_wmreward_guidance.md`

## 后续更新模板

后续新增实验时，复制下面这个模板填写：

```md
## YYYY-MM-DD — <实验名称>

- 目标：
- 代码：
  - /abs/path/to/script.py
- 输入：
  - /abs/path/to/input_manifest_or_json.txt
- 输出：
  - /abs/path/to/output_root
- 分数 / 汇总：
  - /abs/path/to/summary.json
  - /abs/path/to/scores.json
- 状态：
- 结论：
```

## 历史实验回填

下面的条目是在 2026-07-05 根据现有代码、输出目录和 summary 文件回填整理的。

如果无法精确恢复实验开始日期，会明确标注为“按目录名/进度文档推断”。

## 2026-06-26 到 2026-06-30（推断）— 早期 LoRA 7 模式与批量 case 原型实验

- 目标：
  - 验证简单的 training-free V-JEPA guidance preset 是否会改变 Wan2.2 + LoRA v2v 在 `test_5` 和更大 `v2v_jsons` 批次上的输出。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan_openvid_0613pybullet_lorav2v_vjepa.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_lora_vjepa_modes.py`
  - 历史辅助脚本：
    `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/archive/2026-07-cleanup/run_mode_smoke_suite.py`
    和
    `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/archive/2026-07-cleanup/run_manifest_all_cases.py`
- 输入：
  - `/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt`
  - `/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons`
- 输出：
  - `/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results/lora_test5`
  - `/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results/v2v_jsons_full_wan22`
- 产物中出现的主要模式：
  - `baseline`
  - `g1_mid1_s001`
  - `g2_mid2_s001`
  - `g3_mid2_s002`
  - `g4_wide4_s001`
  - `g5_wide4_s002`
  - `g6_wide6_s002`
- 分数 / 汇总：
  - 各 mode 目录下的逐视频 sidecar JSON，例如：
    `/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results/lora_test5/wan_openvid_0613pybullet_lorav2v_step000500_test5_vjepa_baseline/0613pybullet_sample_000301_w000.json`
  - suite 配置：
    `/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results/v2v_jsons_full_wan22/suite_config.json`
  - manifest：
    `/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results/v2v_jsons_full_wan22/manifests/manifest.csv`
- 状态：
  - 已完成，属于历史原型批次。
- 结论：
  - 建立了最初的 7-mode 对比形态，后续演化成当前的 `experiment_presets.py` 和 `run_lora_vjepa_modes.py` 工作流。

## 2026-07-01 到 2026-07-02（推断）— 单 case timestep / step-index 可视化诊断

- 目标：
  - 通过只在选定去噪步上施加 guidance，观察 timing 是否显著影响中间结果和最终结果。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/build_timestep_sweep_viewer.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/compare_guidance_videos.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/diag_anchored_onestep.py`
- 输入：
  - 单 case 清单保存在：
    `/data/gaoya/agent-data/outputs/vjepa_timestep_sweep/one_case.txt`
    和
    `/data/gaoya/agent-data/outputs/vjepa_stepindex_sweep/one_case.txt`
- 输出：
  - `/data/gaoya/agent-data/outputs/vjepa_timestep_sweep`
  - `/data/gaoya/agent-data/outputs/vjepa_timestep_sweep_1460`
  - `/data/gaoya/agent-data/outputs/vjepa_stepindex_sweep`
- 分数 / 汇总：
  - 运行脚本记录：
    `/data/gaoya/agent-data/outputs/vjepa_timestep_sweep/run_sweep.sh`
    和
    `/data/gaoya/agent-data/outputs/vjepa_stepindex_sweep/run_sweep.sh`
  - 可视化页面：
    `/data/gaoya/agent-data/outputs/vjepa_stepindex_sweep/index.html`
- 状态：
  - 以诊断为主，主要是视觉/定性分析。
- 结论：
  - 帮助缩小了后续单 case `phase5/phase6/phase7` 重点探索的 guidance 区域。

## 2026-07-02 — Probe Sweep Phase 4：单 case 上的 wmreward 耦合检查

- 目标：
  - 测试小幅 anchored energy 下降是否真的会推动 `wmreward`。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/probe_energy_persistence.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_guided_videos.py`
- 输入：
  - 单 case：`physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed`
- 输出：
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase4`
- 分数 / 汇总：
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase4/phase4_summary.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase4/wmreward_scores.json`
- 状态：
  - 已完成。
- 结论：
  - 小幅 anchored energy 降低基本落在噪声区间内，不能稳定转化为 `wmreward` 提升。

## 2026-07-03 — Probe Sweep Phase 5：强 fixed-step 强度阶梯

- 目标：
  - 扫更大的 latent step size，找出 guidance 真正“写进” latent 并影响最终指标的阈值。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/probe_energy_persistence.py`
- 输出：
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase5`
- 分数 / 汇总：
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase5/phase5_summary.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase5/wmreward_scores.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase5/phase5_multimetric_scores.json`
- 状态：
  - 已完成。
- 结论：
  - 证实前面的问题主要不是方向错，而是写入太弱。`ladder_s20` 成为首个在单 case 上明显正向推动 `wmreward` 的候选。

## 2026-07-03 — Probe Sweep Phase 6：timing × inner-k 精调

- 目标：
  - 在 phase5 的“膝点”附近继续细调，比较更早/更晚 timing、重复 inner correction、以及 backtracking 变体。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/probe_energy_persistence.py`
- 输出：
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase6`
- 分数 / 汇总：
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase6/phase6_summary.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase6/wmreward_scores.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase6/phase6_multimetric_scores.json`
  - regression-light 补充结果：
    `/data/gaoya/agent-data/outputs/probe_sweep/phase6/phase6_multimetric_scores_regression_light.json`
    和
    `/data/gaoya/agent-data/outputs/probe_sweep/phase6/phase6_multimetric_scores_regression_light2.json`
- 状态：
  - 已完成。
- 结论：
  - backtracking 太保守，不适合当前能量地形。
    `knee_mid_s18`、`knee_early_s15`、`knee_mid_s10_k2` 成为强候选，
    但单 case 最强 `wmreward` 结果仍是 `ladder_s20`。

## 2026-07-04 — Probe Sweep Phase 7：target window size / future horizon

- 目标：
  - 固定强 guidance timing，仅比较 anchored future window 的大小。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/probe_energy_persistence.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_phase7_target_shape.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wait_for_phase7_gpus.py`
- 输出：
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase7`
- 分数 / 汇总：
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase7/phase7_summary.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase7/phase7_multimetric_scores.json`
- 状态：
  - 已完成。
- 结论：
  - `target_w24` 成为最优的 target-shape 单 case 变体，优于 `w16` 和 `w32`。

## 2026-07-04（推断）— Probe Sweep Phase 8 后续实验

- 目标：
  - phase7 之后的后续 probe；精确文字结论尚未完全补回。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/probe_energy_persistence.py`
- 输出：
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase8`
- 分数 / 汇总：
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase8/phase8_summary.json`
  - `/data/gaoya/agent-data/outputs/probe_sweep/phase8/phase8_multimetric_scores.json`
- 状态：
  - 有产物，但文字总结尚未完整回填。
- 结论：
  - 暂时保留为已记录的实验产物集合，待后续恢复详细实验说明。

## 2026-07-03 到 2026-07-04 — Phase4 multicase pilot3：LoRA baseline + guided preset 小规模多 case 对比

- 目标：
  - 在转入更重的 `train0705` 批次前，先在小规模多 case 子集上比较主要 guided 候选。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_phase4_multicase.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_multicase_methods.py`
- 输出：
  - `/data/gaoya/agent-data/outputs/vjepa_phase4_multicase`
- 产物中的主要方法：
  - `phase4_pilot3_baseline`
  - `phase4_pilot3_ladder_s20`
  - `phase4_pilot3_knee_early_s15`
  - `phase4_pilot3_knee_mid_s18`
  - `phase4_pilot3_knee_mid_s10_k2`
- 分数 / 汇总：
  - `/data/gaoya/agent-data/outputs/vjepa_phase4_multicase/phase4_pilot3_all_methods_scores.json`
  - `/data/gaoya/agent-data/outputs/vjepa_phase4_multicase/phase4_pilot3_baseline_vs_ladder_s20_scores.json`
  - 各方法对应的 `*_runtime/summary.json`
- 状态：
  - 已完成。
- 结论：
  - 这是从单 case 调参过渡到子集级 A/B 的桥接实验，也直接喂给了后面的 `train0705 current modes` preset 家族。

## 2026-07-03 — train0705 current modes：pilot3 round1

- 目标：
  - 将单 case 上表现最好的 preset 移植到自定义 `train0705 stage1b` 分支，并在 3-case pilot 上对比。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_current_modes.py`
- 输出：
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/pilot3_round1`
  - 对比报告：
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/pilot3_round1_compare_partial`
    和
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/pilot3_round1_compare_full`
- 分数 / 汇总：
  - 各方法 summary，例如：
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/pilot3_round1/train0705_pilot3_round1_ladder_s20/summary.json`
  - 汇总对比：
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/pilot3_round1_compare_full/aggregate_summary.md`
    和
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/pilot3_round1_compare_full/aggregate_summary.json`
- 状态：
  - 已完成。
- 结论：
  - `ladder_s20` 在 3-case pilot 上的平均 `Δsurprise` 最好，
    `knee_mid_s18` 在跨指标稳定性上更均衡。

## 2026-07-06 — 4-family freqguide A/B 状态审计

- 目标：
  - 核对当前 4-family 频域 guidance 目标和现有输出根目录是否一致，避免把历史 `step-005000` 结果误当成当前目标 `step-007000`。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/audit_model_weight_ab_status.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_model_weight_ab_test5.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_model_weight_ab_test5_freqguide.py`
- 输入：
  - 历史 root：
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705`
  - 当前目标 root：
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_freqguide_20260706`
- 输出：
  - `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/status_audit.json`
  - `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/status_audit.md`
  - `/data/gaoya/agent-data/outputs/model_weight_ab_test5_freqguide_20260706/status_audit.json`
  - `/data/gaoya/agent-data/outputs/model_weight_ab_test5_freqguide_20260706/status_audit.md`
- 状态：
  - 已完成审计。
- 结论：
  - 历史 root `model_weight_ab_test5_20260705` 已完整覆盖：
    `wan22_official_ti2v5b`、`wan22_early_lora_step000500`、`train0705_step002500`，
    以及一个额外的历史 family `train0705_step005000`。
  - 它不包含当前目标里的 `train0705_step007000`。
  - 新目标 root `model_weight_ab_test5_freqguide_20260706` 目前还是空目录，4 个目标 family 都尚未开始生成。
  - 因此，当前“脚本目标”已经切到
    `wan22_official_ti2v5b / wan22_early_lora_step000500 / train0705_step002500 / train0705_step007000`，
    但“已验证落盘结果”还没有覆盖到这个新目标组合。

## 2026-07-06 — 现有 A/B root 的 full-metric 重打分脚本

- 目标：
  - 为历史 root 和未来的新 root 提供统一的全指标重打分入口，避免继续依赖早期只含部分指标的 `scores/*.json`。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/rescore_model_weight_ab_root.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_multicase_allmetrics.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/export_model_weight_ab_markdown.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/visualize_model_weight_ab.py`
- 输入：
  - 任意已有 baseline/guided family 目录的 A/B root，例如：
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705`
- 输出：
  - 对应 root 下新的 `scores/*.json`
  - `ab_report/model_weight_ab_report.md`
  - `ab_dashboard/index.html`
- 状态：
  - 脚本已创建并通过 `py_compile` 静态检查，尚未启动完整重打分。
- 结论：
  - 后续如果先重打历史 root，可以直接验证当前 full-metric 评分链条是否完整；
    等 `model_weight_ab_test5_freqguide_20260706` 有生成结果后，也可以复用同一脚本做统一汇总。

### 2026-07-06 当天 smoke 追记

- 1-case / 1-family smoke 使用：
  - `wan22_official_ti2v5b`
  - `limit-cases=1`
  - root:
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705`
- 观察到的具体问题：
  - `Official PDI` 最初不是纯慢，而是直接报错：
    `third_party/mega_sam/work_space` 目录缺失。
    现已在
    `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/official_pdi.py`
    中补了启动前的目录创建。
  - `WMReward` 最初因为新版 PyTorch 的 `torch.load` 默认行为与上游
    `WMReward-main/utils.py` 不兼容而失败。
    现已在
    `/home/gaoya/Code_Video/Code_data/Code_try0526/physv_eval/wmreward_official.py`
    中补了本地 trusted checkpoint 的兼容层，使其默认走
    `weights_only=False`。
  - 修复后再次 smoke，`wmreward` 已经进入大权重加载阶段，不再是先前的
    `weights_only` 异常；当前剩余问题更偏向运行时长，而不是接口错误。

## 2026-07-03 — train0705 round2：完整 test_5 上 baseline vs ladder_s20 vs knee_mid_s18

- 目标：
  - 检查 pilot3 的优胜 preset 在 17-case `test_5` 子集上是否还能成立。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_current_modes.py`
- 输出：
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round2_test5`
- 分数 / 汇总：
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round2_test5/round2_test5_compare_summary.md`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round2_test5/round2_test5_compare_summary.json`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round2_test5/round2_test5_scores.json`
- 状态：
  - 已完成。
- 结论：
  - `ladder_s20` 和 `knee_mid_s18` 都没有在 17-case baseline 上取得平均 `wmreward surprise` 的正向提升。
    最后保留 `knee_mid_s18` 作为更稳的折中 preset。

## 2026-07-03 — train0705 round3：overlap-5 子集上的 guard ablation

- 目标：
  - 比较旧版 dense target guidance 与 ratio-cap、L1 guard 变体。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_guard_ablation.py`
- 输出：
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round3_guard_ablation`
- 分数 / 汇总：
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round3_guard_ablation/round3_guard_ablation_compare_summary.md`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round3_guard_ablation/round3_guard_ablation_compare_summary.json`
  - 例如：
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round3_guard_ablation/target_w24_ratio_005_vs_baseline5_scores.json`
- 状态：
  - 已完成。
- 结论：
  - `target_w24_old` 虽然最能拉动 `wmreward`，但会明显破坏 `physics_iq`。
    `target_w24_ratio_005` 因此成为后续多轮实验里的“相对不坏”的诊断锚点。

## 2026-07-03 到 2026-07-04 — train0705 round4/5/6：围绕 ratio-cap 与 s15 的局部细化

- 目标：
  - 围绕 `target_w24_ratio_005`，以及后续 `s15` 家族，对 dense mid-band `context_anchored` guidance 继续做局部精调。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_ratio_cap_sweep.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_s15_local_sweep.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_train0705_s15_local_sweep.py`
- 输出：
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round4_test5_ratio_only`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round5_ratio_cap_sweep_overlap5`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round6_s15_local_sweep_overlap5`
- 分数 / 汇总：
  - round4：
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round4_test5_ratio_only/round4_test5_ratio_only_scores.json`
  - round5：
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round5_ratio_cap_sweep_overlap5/overlap5_target_w24_s20_ratio_010_scores.json`
  - round6：
    `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round6_s15_local_sweep_overlap5/round6_full_scores.json`
- 状态：
  - 已完成。
- 结论：
  - 这些回合逐步建立了当前围绕 `target_w24_s15_ratio_003` 及附近 ratio-cap 变体的本地候选族。

## 2026-07-04 — train0705 round7：在完整 test_5 上扩展验证

- 目标：
  - 在完整 17-case 子集上重新验证当前 dense-mid 本地候选。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_round7_expansion.py`
- 输出：
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round7_test5_expansion`
- 分数 / 汇总：
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round7_test5_expansion/round7_test5_scores.json`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round7_test5_expansion/round7_baseline_vs_0025_scores.json`
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round7_test5_expansion/round7_baseline_vs_0035_scores.json`
- 状态：
  - 已完成。
- 结论：
  - 说明 dense-mid 本地候选仍是活跃方向，但还没有形成一个在完整子集上“干净取胜”的配置。

## 2026-07-04 到 2026-07-05 — train0705 round8：step-005000 guided 复跑

- 目标：
  - 在更晚的 `train0705 step-005000` checkpoint 上重新跑当前 guided 家族。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_current_modes.py`
- 输出：
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round8_step5000_test5_guided`
- 分数 / 汇总：
  - `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round8_step5000_test5_guided/train0705_round8_step5000_test5_target_w24_s15_ratio_0025/summary.json`
- 状态：
  - 历史产物存在，但完整对比结论尚未完全回填。
- 结论：
  - 记录了 step-005000 guided 分支，后面也被纳入更大的 model-weight A/B 测试。

## 2026-07-05 — 多模型权重统一 A/B：test_5 上 baseline vs guided

- 目标：
  - 在多个模型线上统一执行 baseline vs guided 对比：
    Wan2.2 official TI2V、早期 LoRA、`train0705 step-002500`、
    `train0705 step-005000`、以及 Wan2.1 T2V 1.3B 原型线。
- 代码：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_model_weight_ab_test5.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_multicase_methods.py`
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan21_t2v_1_3b_batch.py`
  - 支撑生成脚本：
    `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wanti2v.py`
    和
    `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan_openvid_0613pybullet_lorav2v_vjepa.py`
- 输入：
  - `/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt`
  - 去重后清单：
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/inputs/test5_unique.txt`
- 输出：
  - `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705`
- 分数 / 汇总：
  - 现有 summary / runtime summary：
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step002500/baseline/summary.json`
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step002500/guided/summary.json`
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step005000/baseline/summary.json`
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step005000/guided/summary.json`
  - baseline 复用报告：
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/reuse_reports/lora_baseline_reuse.json`
    和
    `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/reuse_reports/official_baseline_reuse.json`
- 状态：
  - 截至 2026-07-05 仍在进行中。
- 当前结论：
  - LoRA baseline 可以安全复用一部分 `/data/gaoya/AAA_test_video/0623/test/v2v` 里的历史结果。
  - Official Wan2.2 baseline 没有找到字节级一致的历史匹配，仍需在当前 A/B 树下重新生成。
