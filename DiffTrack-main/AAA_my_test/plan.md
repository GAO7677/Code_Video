# Wan2.2-TI2V-5B Motion 实验计划

本实验的完整协议、单 case gate、批量统计规则、存储清理策略和解释边界见 [wan_motion_plan.md](wan_motion_plan.md)。

当前执行状态：

- case 019 的 6 layers x 5 steps gate 已通过。
- 50 个 base case 已在 tmux 会话 `wan_motion_batch` 中完成，产出 50 个 complete markers 和 250 个 step files。
- 多样本统计选择 `layer 17 / step 49 (t=92, sigma=0.0925)` 为主配置，`layer 17 / step 36 (t=660, sigma=0.6601)` 为中噪声对照。
- 结果与分析见 [wan_motion_results.md](wan_motion_results.md)。
- 后续 150 个 appearance/shape/background variants 只用于固定候选配置的鲁棒性验证，不重新搜索 layer/step。
