# PhysV Simulation Benchmarking

以当前各视频 JSON 中的 `metric_results` 为准。旧的 `result/metrics.csv` 和本文件早期版本里有一部分数值与口径已经过期，不再作为主依据。

## 组别与样本

| 组 | 当前定义 | 数量 | 说明 |
|---|---|---:|---|
| A | PDI-Bench 生成视频 | 60 | 4 个方法 × 15 个 case，来自 5 个子类：`partial_occlusion / Biological_Motion / Dynamic_Tracking / Longitudinal_Convergence / Curved_Motion` |
| B1 | Ball-Block 物理参数 | 8 | 固定外观，只改恢复系数 / 摩擦 / 球质量 |
| B2 | JEPA 运动敏感性 | 20 | 固定外观，系统改变速度 / 质量 / 重力 / 碰撞 / 方向 |
| B3 | 外观敏感性 | 24 | 同一物理轨迹，只改渲染外观与光照 |
| C | 帧序打乱 sanity check | 21 个文件，可形成 20 对 | 目录里不是“2 个样本”，而是 20 个 `*_shuffled` + 1 个 `gt_ball_original`；其余原始视频需要回到 A/B 根目录配对 |

`A` 是自然视频与生成视频的方法对比集，用来比较不同方法在复杂真实场景下的整体表现。`B1/B2/B3/C` 都属于项目内自建的 `Dataset_physV` 仿真评测集：`B1` 是基础刚体碰撞控制变量实验，`B2` 是更系统的运动参数扫描，`B3` 固定物理只改外观，`C` 则只打乱时间顺序做 sanity check。

## 指标口径

| 指标 | 当前口径 | 方向 | 备注 |
|---|---|---|---|
| Official PDI | `metric_results.official_pdi` | ↓ | 当前实现保存总分和 `scale / traj / rigidity / vp` 子项；本项目现有结果里 `traj_component` 基本全为 `0.0`，所以排序主要由其余三项决定 |
| WMReward | `metric_results.wmreward_jepa.surprise` | ↓ | 当前代码已对齐官方 `compute_wmreward.py` 默认口径；`similarity = 1 - surprise` 只是派生字段，不建议再作为主表指标 |
| V-JEPA Proxy | `RelRaw / DeltaRel / DeltaProf` | ↓ | 这是项目内自定义诊断，不是官方 benchmark；兼容字段里的总分只是 `exp(-(三项误差之和))`，不应再当作主结论 |
| Cosmos Reason1 | `metric_results.cosmos_reason1.score` | ↑ | 1 到 5 的离散 LLM judge 分数，解释性强，但稳定性一般 |
| VideoPhy-2 | `sa_score / pc_score / joint` | ↑ | `joint = 1[SA>=4 且 PC>=4]`，属于较粗的离散判断 |

`Official PDI` 是几何一致性审计分数，依赖分割、跟踪、深度和投影关系。`WMReward` 是基于 V-JEPA 的滑窗未来预测误差，这里统一看官方 `surprise`。`V-JEPA Proxy` 是项目内自定义的未来结构误差诊断。`Cosmos Reason1` 是 LLM judge 直接看视频给出的 1 到 5 分。`VideoPhy-2` 是离散自动评测器，这里主看 `SA / PC / Joint`。

## A 组：PDI-Bench 生成视频

当前 60 个视频的均值如下：

| method | PDI ↓ | WMReward Surprise ↓ | Cosmos ↑ | RelRaw ↓ | DeltaRel ↓ | DeltaProf ↓ | VPhy-PC ↑ | VPhy-Joint ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GT | **0.1416** | **0.4270** | **4.4667** | 0.0536 | **0.4157** | 0.2735 | **4.1333** | **0.7333** |
| VACE_1p3B_TI2V | 0.3855 | 0.4272 | 3.0000 | **0.0318** | 0.4793 | **0.2318** | 3.6667 | 0.4000 |
| VACE_1p3B_ctx08 | 0.5324 | 0.4307 | 3.1333 | 0.0455 | 0.4379 | 0.2543 | 4.0667 | 0.6667 |
| wan22-5B-TI2V | 0.4314 | 0.4276 | 3.5333 | 0.0442 | 0.5078 | 0.2515 | 3.8000 | 0.6000 |

修正点：

- 旧版把 `WMReward JEPA` 写成主表 `↑ similarity`，现在应明确主口径是 `surprise ↓`。
- 旧版写成 “PDI 与 WMReward 排序相反” 已经不准确。当前 A 组里 `WMReward Surprise` 四个方法非常接近，整体跨度只有 `0.0037`，不能据此得出稳定的反向排序结论。
- 旧版把 `vjepa_proxy` 当成单一“加权总分”来解读也不准确。当前更有意义的是三个展开子指标：`RelRaw / DeltaRel / DeltaProf`。

就当前数值看：

- A 组里 GT 仍然明显更优的是 `PDI / Cosmos / VideoPhy-2`。
- `WMReward` 在 A 组几乎打平，更像短窗可预测性，而不是稳健的方法排序指标。
- `V-JEPA Proxy` 子指标出现分裂：`RelRaw` 更偏好 `VACE_1p3B_TI2V`，但 `DeltaRel` 仍由 GT 最优，说明它更容易奖励“平滑、容易预测”的未来，而不一定是“更真实的物理”。

## B 组：仿真视频

### B1 物理参数

- `PDI` 范围很大：`0.0223 -> 2.0636`，能区分出极端 case。
- 但它主要由 `scale_component` 拉开；例如 `e07_mu05_m01` 的 `PDI=2.0636`，其中 `scale_component=4.7871`。
- `WMReward Surprise` 只在 `0.4019 -> 0.4202` 之间变化，跨度 `0.0184`。
- `Cosmos` 在 8 个样本上全部给 `2`，完全没有区分力。
- `VideoPhy-2 PC` 在 8 个样本上全部是 `4`，`Joint` 全为 `0`，也没有排序能力。

### B2 运动敏感性

- `PDI` 范围更大：`0.0295 -> 5.1678`，均值 `1.3065`。
- 这里同样主要是 `scale_component` 在工作；`traj_component` 仍是 `0.0`。
- `WMReward Surprise` 只有 `0.3796 -> 0.4051`，跨度 `0.0255`。
- `RelRaw` 只有 `0.0107 -> 0.0255`，`DeltaRel` 只有 `0.2524 -> 0.4646`，都是弱变化。
- `Cosmos` 和 `VideoPhy-2` 能给出一些离散差异，但粒度很粗，且不总是与其它指标一致。

### B3 外观敏感性

这个子组最关键，因为它不改物理，只改渲染。

- `PDI` 在同一物理轨迹下仍能变化 `5.3x ~ 12.6x`。
  - `e05_mu05_m1`: `0.0197 -> 0.2486`，比例 `12.6x`
  - `e07_mu05_m1`: `0.0218 -> 0.2209`，比例 `10.1x`
- 同一批 case 上，`WMReward Surprise` 的变化只有 `0.0045 ~ 0.0095`。
- `Cosmos` 甚至会在纯外观变化下从 `1` 跳到 `5`。

修正点：

- 可以说 `PDI` 在仿真里“有区分度”，但不能再直接说它在这里测的是“纯几何质量”或“纯物理正确性”。B3 已经说明它对渲染风格高度敏感。

## C 组：帧序打乱 sanity check

当前可配对的打乱样本是 `20` 对，不是旧文档里的“2 个样本”。

按配对后的结果看：

- `PDI` 在 `20` 对里有 `17` 对在打乱后反而更好，平均变化 `+0.0953`（按 `shuffled - original` 记，负值才算更好）。
  - 例子：`sim_e03_mu05_m1` 从 `0.2730 -> 0.0120`
  - 例子：`gt_ball` 从 `0.1748 -> 0.1019`
- `WMReward Surprise` 在 `20` 对里有 `19` 对打乱后变差，方向基本正确，但平均只增加 `0.0181`，幅度不大。
- `V-JEPA RelRaw` 在 `19` 个有效对里有 `13` 对打乱后变差，平均只增加 `0.0071`，也是弱敏感。
- `VideoPhy-2 PC` 在 `20` 对里有 `7` 对下降、`13` 对不变，没有出现“打乱后更高”，但大量并列说明它分辨率偏粗。
- `Cosmos` 在 `20` 对里 `14` 对完全不变，另外少数 case 还会出现大幅跳分，稳定性一般。

修正点：

- 旧版说 “WMReward 方向不一致，部分反而上升”，那是基于 `similarity ↑` 的写法。按当前主口径 `surprise ↓` 来看，C 组里 WMReward 基本是朝正确方向变化的，只是幅度偏小。
- 旧版对 PDI 的核心判断仍成立，而且更强：它不是“偶尔失效”，而是在当前 C 组里大多数配对上都未通过时序 sanity check。

## 当前更稳妥的结论

1. `Official PDI` 在本项目里更像“几何前端 + 渲染条件 + 局部结构稳定性”的综合结果，不等于纯物理分数。
2. `WMReward Surprise` 已经统一到官方默认口径，但它主要反映短窗预测难度；在 A/B 组上的排序能力都偏弱。
3. `V-JEPA Proxy` 适合作为诊断展开项，不适合作为单一总分下结论。
4. `Cosmos Reason1` 和 `VideoPhy-2` 有一定可解释性，但当前都是离散粗分，适合看明显错误，不适合细粒度排序。
5. 如果要把某个 benchmark 当“物理评测集”，至少要同时过两类检查：`B3` 的外观鲁棒性检查，以及 `C` 的时序打乱检查。当前没有任何一个单指标同时过这两关。
