# Wan GT Spatiotemporal Correspondence Guidance

## 实验目的

这是 **oracle inference-time diagnostic**，不是训练，也不是 M1/M2/M3 消融。实验用 source/simulator future 提供的对象轨迹引导 Wan 去噪，检验 latest3350 Top100 high-PCK heads 中的 Q/K correspondence 是否能作为可微控制信号，让生成对象轨迹更接近 GT。

主脚本：`AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py`。

## 固定输入与实验单位

- case 列表：`/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt`。
- 原列表 22 行，按 JSON 绝对路径稳定去重后为 **20 cases**。
- GT：每个 JSON 的 `source_video` 前 49 帧；不足 49 帧时使用全部帧并线性映射到 13 个 latent 时刻，且在 manifest 中显式标记。
- 条件：与现有 legacy Wan2.2 TI2V baseline 一致，只使用 `input_image`。JSON 中的多帧 `input_video/context video` 不进入该 TI2V pipeline；不能把结果表述成“8-frame context conditioned Wan”。
- head：`pck_head_scopes_s039_latest3350.json` 前 100 个物理 layer-head，严格校验为完整 30×24 排名的 Top100。
- 去噪：40 steps，loss 只取 positive/conditional CFG forward；negative branch 不构造 loss。
- 默认 seed：47326。

## GT tube 构造

对 source video 执行：

1. caption physical noun phrases → GroundingDINO boxes；
2. SAM2 从首帧 box 向整个 source prefix 传播 object masks；
3. 每个首帧 object mask 内确定性采样 8 个点；
4. CoTracker 从首帧跟踪这些点；
5. 取 source pixel frames `F00,F04,…,F48`，得到 13 个 mask/point tube：

\[
R^*=(R_0^*,R_1^*,\ldots,R_{12}^*).
\]

每个对象单独成为一个 guidance target。若至少两个对象的 robust displacement 超过首帧 bbox 对角线的 5%，额外构造 `moving_union`；该阈值可用 `--moving-threshold-d0` 修改。SAM2/GroundingDINO 与 CoTracker 不同时驻留 GPU。

注意：这里使用的是 source future oracle，不能用于无 GT 的真实推理，也不能用于证明模型“自然知道”未来轨迹；它只回答该 Q/K 通道能否接受轨迹约束。

## 两类 correspondence loss

设第 `l` 层第 `h` 个 head 的归一化、RoPE 后向量为 `q_lh`、`k_lh`，每个 latent frame 有 `H×W` 个空间 token。所有 loss 都覆盖有序跨时刻对 `t_q != t_k`，不包含 same-time 项。

### 1. Region tube-mass loss

SAM2 mask 通过 adaptive max pooling 映射到 token region `R_t*`。对 `R_tq*` 中每个 Query，仅在目标 latent frame `t_k` 的 `H×W` 个 Key 上归一化：

\[
L_{region}=-\frac{1}{|\mathcal P|}
\sum_{(t_q,t_k):t_q\ne t_k}
\operatorname{mean}_{q\in R_{t_q}^*}
\log\frac{\sum_{k\in R_{t_k}^*}\exp(q^\top k/\sqrt d)}
{\sum_{k\in \Omega_{t_k}}\exp(q^\top k/\sqrt d)}.
\]

含义：要求对象区域 Query 在另一个时刻把更多概率质量放到 GT 对象区域；不要求 object 内部点身份一一对应。

### 2. Point Gaussian correspondence loss

同一 CoTracker point 在 `t_q` 的 token 作为 Query，在 `t_k` 的 GT point token 周围构造标准差默认 1.5 token 的二维 Gaussian target `G`：

\[
L_{point}=-\frac{1}{|\mathcal V|}
\sum_{(t_q,t_k,n)\in\mathcal V}
\sum_{k\in\Omega_{t_k}}G(k;p^*_{t_k,n})
\log\operatorname{softmax}_k(q_{t_q,n}^\top k/\sqrt d).
\]

`V` 只包含 source CoTracker 在两端均可见的点。含义：不仅约束 object region，还约束同一表面点的跨时刻 correspondence；球体滚动、遮挡和跟踪漂移会让它比 region loss 更敏感。

两类 loss 默认分别生成视频，不混在同一次运行。脚本也提供显式的 `combined` 研究选项。

## 梯度与采样更新

每个 step 只对当前 noisy latent 建叶子节点：

\[
g_s=\nabla_{x_s}L_{STC}(x_s),\qquad \nabla_\theta L=0.
\]

模型全部 `requires_grad=False`。默认把梯度 RMS 对齐到 conditional noise prediction RMS，并设置最大比例保护。对 FlowMatch velocity 使用：

\[
v_s^{guided}=v_s^{CFG}+\lambda\,\sigma_s\,\widehat g_s,
\]

再执行原 scheduler：

\[
x_{s-1}=x_s+(\sigma_{s-1}-\sigma_s)v_s^{guided}.
\]

因为 `sigma_{s-1}-sigma_s<0`，在 velocity 中加入正的 `grad(L)` 会在 latent 更新中沿 `-grad(L)` 移动。首个条件 latent 的梯度被强制置零，scheduler 后仍恢复 `first_frame_latents`。

## 输出与判定

输出根目录默认是：

`/data/gaoya/agent-data/outputs/wan_gt_spatiotemporal_correspondence_guidance/latest3350_top100_v1`

每个生成 variant 的目录包含：

- `generated.mp4`；
- `manifest.json`：精确 loss、target、head、梯度、每步 loss/gradient audit；
- `trajectory_metrics.json`：生成视频相对 source GT 的 ADE/FDE、PCK@10/20%D0、可见率；
- case/seed 根目录的 `comparison_to_baseline.json`：guided minus baseline。

主要成功标准：

- `delta_ADE < 0`、`delta_FDE < 0`；
- `delta_PCK > 0`；
- 可见率不能明显下降，否则“轨迹改善”可能只是困难帧被跟踪门控删除。

Baseline 仍需和现有 legacy baseline 做像素 hash/数值 parity spot-check；单元测试只能证明 loss 方向和 scheduler 符号，不能替代一次 GPU smoke test。

## 命令

先做无 GPU 任务展开检查：

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
  --stage all --dry-run
```

建议先只跑一个 case，并分阶段执行：

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
CUDA_VISIBLE_DEVICES=0 /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
  --stage prepare --device cuda:0 \
  --case-keys 0613pybullet_sample_001460_w002

CUDA_VISIBLE_DEVICES=0 /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
  --stage generate --device cuda:0 \
  --case-keys 0613pybullet_sample_001460_w002 \
  --loss-modes region point --guidance-scale 0.1

CUDA_VISIBLE_DEVICES=0 /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  AAA_my_test/run_wan_gt_spatiotemporal_correspondence_guidance.py \
  --stage evaluate --device cuda:0 \
  --case-keys 0613pybullet_sample_001460_w002
```

GPU 4 禁止使用。第一次正式批量运行前，建议在同一 case 上比较 `lambda=0,0.05,0.1,0.2`，确认梯度 audit 无爆炸且 `lambda=0` 与 baseline 一致；variant 路径包含 lambda，不会互相覆盖。

## 已实现的 CPU 测试

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
/home/gaoya/miniconda3/envs/wan-cu128/bin/python -m unittest -v \
  AAA_my_test.test_wan_gt_spatiotemporal_correspondence_guidance
```

覆盖：列表去重、13-anchor 策略、region/point 正负样例、梯度有限性、RMS 归一化、FlowMatch 符号、non-reentrant checkpoint 的 side-loss 反传行为。
