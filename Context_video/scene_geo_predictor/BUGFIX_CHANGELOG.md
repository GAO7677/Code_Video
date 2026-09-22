# 代码 Bug 修复清单

本清单采用追加方式维护。以后本项目每次修复代码 bug，必须追加日期、原因、修改文件、验证结果和限制，不覆盖历史记录。项目规则见 `AGENTS.md`。

## 2026-09-22 — 缺失支撑准入检查与平台射线投影实验

### 已确认的问题

- `pybullet_initial_contact.py` 在穿插未超过 1 mm 时直接放行，未检查调用者声明的支撑接触是否实际存在。旧 support_edge_g03_v0620 的 C 因估计平台横向偏移没有初始支撑，却被当作有效轨迹推进。
- 平台拟合用球体估计高度修正顶面 z 后，继续沿用原深度点的 x/y 范围，导致顶面几何与观测像素射线不一致。
- 剩余尺度偏差未解决；不能把初始化拒绝当作成功恢复几何。

### 修改清单

| 文件 | 修改 | 启用状态 |
|---|---|---|
| `code/pybullet_initial_contact.py` | 无明显穿插时仍校验已声明支撑、朝上法向及球心是否位于有限 AABB 内；缺失时拒绝且不推进时间 | 默认启用；支撑必须由调用者声明 |
| `code/evaluate_context_rgb_pybullet.py` | 固定 support_edge pilot 自动声明 RGB0–7 连续左平台支撑前提，防止遗漏传参 | 默认启用，仅该 pilot 协议 |
| `code/fit_context_collision_primitives.py` | 增加 `support_ray_projection=True` 实验：将已选顶面像素射线与接触平面求交，再拟合有限边界；记录原边界及位移 | 显式实验，默认关闭，无自动 fallback |
| `tests/test_initial_contact_gate.py` | 增加缺失支撑拒绝测试；保留穿插、对齐、速度、多 client 测试 | 7/7 通过 |
| `code/validate_support_projection.py` | 对 12 个平台 case 冻结估计后再读 GT，评测严格 C、接触对齐 C、严格 D；保存事件、几何误差与 overlay | EXECUTED |
| `AGENTS.md` | 后续 bug 修复必须追加本清单 | 项目规则 |

### 验证命令

在项目根目录执行（版本输出目录必须不存在）：

```bash
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B tests/test_initial_contact_gate.py
CUDA_VISIBLE_DEVICES='' OPENBLAS_NUM_THREADS=2 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B code/validate_support_projection.py
```

输出：`/data/gaoya/agent-data/outputs/physvideo_context_rgb_to_pybullet_20260921_v1/support_projection_fix_20260922_v1`。

### 实际结果与限制

- 初始化回归测试 7/7 通过；12 个平台 case × 3 模式全部完成检查。CPU、两线程，无新 GPU 推理、无训练、无未来 RGB 输入。
- 新几何下严格 C 12/12 拒绝；显式对齐 C 6/12 可推进、6/12 拒绝；D 12/12 初始支撑成立并可推进。BLOCKED 无预测轨迹，误差不填零。
- 目标 case 平台 y 上边界从 -0.00830 m 改为 +0.18544 m；GT 为 +0.520 m。GT 球心 y=0.21783 m 仍越界，接触为斜向边缘穿插，因此严格 C 与接触对齐 C 均拒绝。没有横向移动 GT 球或扩大平台。
- 目标缺口宽度由 0.525 m 改为 0.540 m，GT 为 0.620 m；仍不准确。
- 目标 D ADE 0.340197 → 0.339761 m，变化很小；12-case D 平均 ADE 0.296555 → 0.296594 m，未改善，因此新几何策略不默认上线。
- 目标 D 在 0.92083 s 后离开左平台（GT 0.875 s），说明未通过无限扩大平台阻止下落。但后续接触事件仍不匹配（预测接触右平台，GT 落地），不能宣称物理预测已修好。
- 本轮完成准入 bug 修复及可复现几何实验；完整米制状态/几何联合估计仍为 PARTIAL。后续需诊断球轮廓/透视模型与尺度来源，再验证平台区域和多帧支撑约束，不得用 GT 输入强行对齐。

## 2026-09-22 — 全 36 case / 144 显示分支回归复核

- 新增 `code/recheck_all_viewer_cases.py`。使用修复后准入代码重新执行页面中全部 A/B/C/D、GT omega 分支；冻结 VGGT 输入和 primitives_v3，实验射线几何关闭。C 保留 v2 原来的显式接触对齐策略，非自动 fallback。
- 命令：`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B code/recheck_all_viewer_cases.py`。
- 预先固定轨迹一致性容差 1e-6 m。114 组轨迹一致，最大差异 1.07624e-7 m；9 组仍被拦截；21 组新增缺失支撑拒绝。A 36/36 不变，B 24 不变/12 新拒绝，C 18 不变/9 旧拒绝/9 新拒绝，D 36/36 不变。
- 所有拒绝分支实测 stepSimulation 调用为 0；可推进分支每组 328 次。额外验证了 v2 接触对齐 C 的严格模式先拒绝。
- 以前标 PASS 的 21 组实际缺少初始左平台接触；本轮改变的是准入判定，不能当作模型性能改善。未发现被保留轨迹数值退化；未宣称所有旧 PASS 预测正确。
- 原 support_edge_g03_v0620 C 现在缺失支撑即拒绝，错误自由落体不再作为新预测展示；估计几何本身仍未修复。18 组旧 C 接触对齐结果未变；其余 9 组旧拒绝仍未解决。
- 新结果：`all36_recheck_20260922_v1`；新页面：`overlay_viewer_v3`。旧 v2 数据及页面保留用于对照。所有结果位于原输出根目录下。

