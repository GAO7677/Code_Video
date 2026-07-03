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

### Phase 2 — 拐点附近的 timing × inner-K 精调 ⏳ 已建脚本入口，待完整跑分

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

### Phase 3 —（仅当 Phase 0/1 显示解耦时）重审能量目标 ⏳ 条件触发

如果 wmreward 对 anchored 能量下降**没反应**，说明固定 future-feature anchor 可能
不是对的目标。按代价从低到高：
- `reduction="max"`（最差窗口）替代 `"mean"`——平的均值可能把唯一的坏窗口平均掉了。
- 加宽 future 视野（`window_size` / `context_frames` 切分），让能量看到更多生成的运动。
- 用 **surprise**（自一致性）能量在大步长下做对照——若它能动 wmreward 而 anchored 不能，
  问题就在 anchor。
只在有证据时触发，不投机做。

### Phase 4 — 在 3–5 个 case 上验证胜出配置 ⏳

拿 Phase 1–2 的最优配置，对 3–5 个 v2v case 跑 baseline vs 引导。报告每个 case 及
平均 wmreward delta，再加一个副指标（`videophy2 --task pc` 和/或 `cosmos_reason1`）
确认 wmreward 的提升不是在钻单一指标的空子。用已有的 HTML compare viewer 做可视检查。
（本 case 有真实续帧 `physicIQ_0002_clip_2p5s_3p5s.mp4`，Phase 4 可用 `physics_iq` 做参照打分。）

## 交付物

- `score_guided_videos.py` — 只读批量打分器（各 phase 复用）。✅
  现在支持：
  - `wmreward`
  - `physics_iq`
  - `videophy2-task pc`
  - `cosmos-reason1`
  - `--merge-from-json` + `--skip-wmreward` 复用已有 wmreward 结果，只补交叉指标
- 引导 stats 里的能量/修正量埋点（不改指标代码）。✅
- 每个 phase 一张结果表（JSON + 简短 markdown）：配置 → 能量 delta → 修正 L2 →
  像素差 → wmreward（Phase 4 再加 judge 分）。
- 更新 memory：强度阈值发现、能量↔wmreward 耦合结论。✅（耦合已记）

## 相关文件

代码（都在 `code_vjepa_free/vjepa_guidance/`）：
- `probe_energy_persistence.py` — 扫描驱动（Phase 1 = `--phase 5`，Phase 2 = `--phase 6`）
- `wan_openvid_0613pybullet_lorav2v_vjepa.py` — 引导 pipeline（Phase 0b 埋点在此）
- `vjepa_surprise.py` — 能量计算（`context_anchored` / `precompute_future_prediction`）
- `score_guided_videos.py` — wmreward 批量打分（只读调用指标）

指标（只调用，**不改**）：`Code_try0526/physv_eval/single_case/`（wmreward / physics_iq / videophy2 / cosmos_reason1 ...）

完整规划：`~/.claude/plans/vjepa-guidance-wmreward-search.md`
