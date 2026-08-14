# MVTrack4Gen 与本地 Noise-Gated Correspondence Loss 对比

> 调研日期：2026-08-14。结论基于 MVTrack4Gen 官方项目页、arXiv v1 论文及官方 GitHub 仓库；本地实现以同目录下的 `noise_gated_correspondence.py` 和 `run_pybullet_correspondence_diagnostics.py` 为准。

## 结论

**不是同一个 loss。** 两者共享的核心思想是：从视频 DiT self-attention 的 query/key 构造逐帧空间匹配分布，再用点轨迹提供 correspondence 监督。MVTrack4Gen 的 `L_corr` 因而与本地 Gaussian soft-label CE 是最接近的部分；但 MVTrack4Gen 使用**单个正确 token 的 hard-label CE**，覆盖 reference/target 两个视图和全部共可见帧，并额外训练一个完整的 multi-view tracking head。当前本地方案使用 **Gaussian soft labels、固定单源帧到未来帧、跨层 Top100 PCK 加权、attention soft-argmax coordinate Huber，以及显式 SNR gate/cutoff**。这些关键设计均不是 MVTrack4Gen 论文披露的 `L_corr`。

更重要的是，二者中出现的 Huber loss 作用位置不同：MVTrack4Gen 的 Huber 是 tracking head 输出轨迹坐标与 GT 轨迹之间的 `L_seq`；本地 Huber 是聚合 QK attention 的 soft-argmax 坐标与伪轨迹坐标之间的损失。二者虽然都是坐标回归，却具有不同的预测器、计算图和梯度路径，不能视为相同 loss。

## 一、MVTrack4Gen 的官方训练目标

### 1. 总目标和扩散目标

论文给出的联合目标为：

$$
\mathcal L_{\mathrm{total}}
=\mathcal L_{\mathrm{diff}}
+\lambda_{\mathrm{track}}\mathcal L_{\mathrm{track}}
+\lambda_{\mathrm{corr}}\mathcal L_{\mathrm{corr}},
\qquad
\lambda_{\mathrm{track}}=\lambda_{\mathrm{corr}}=0.01.
$$

扩散部分采用 rectified flow：

$$
\mathcal L_{\mathrm{diff}}
=w(t)\left\|
v_\theta([z_{\mathrm{ref}},z_{\mathrm{tgt}}],t,c,
\mathrm{cam}_{\mathrm{ref}},\mathrm{cam}_{\mathrm{tgt}})
-(\epsilon-x_0)
\right\|^2_{\mathrm{tgt}},
$$

其中 $t\sim\mathcal U(0,1)$、$\epsilon\sim\mathcal N(0,I)$，噪声输入为 $x_t=(1-t)x_0+t\epsilon$；平方误差只计算 target-view tokens，reference-view tokens 仅作为条件。[论文 Sec. 5.3](https://arxiv.org/html/2606.26087v1#S5.SS3)、[Appendix B.5, Eq. 17](https://arxiv.org/html/2606.26087v1#A2.E17)

### 2. Q/K correspondence tensor

reference 和 target latent tokens 被拼接后共同进入 3D self-attention。在 transformer layer $l$、flow-matching timestep $t$，每个视图 $v\in\{\mathrm{ref},\mathrm{tgt}\}$ 的第 $i$ 个 latent frame 产生：

$$
Q_i^v,K_i^v\in\mathbb R^{hw\times d_{\mathrm{head}}}.
$$

任意 source frame/view 到 target frame/view 的空间对应矩阵为：

$$
\mathcal C_{i,j}^{v_1,v_2}
=\operatorname{Softmax}\left(
\frac{Q_i^{v_1}(K_j^{v_2})^\top}{\sqrt{d_{\mathrm{head}}}}
\right),
\qquad
v_1,v_2\in\{\mathrm{ref},\mathrm{tgt}\}.
$$

每一行是对第 $j$ 帧空间 keys 的逐帧 softmax 分布，而不是对整段视频所有 keys 一次归一化。论文明确说为简洁省略了 attention-head 下标，但没有说明实际训练时如何在 heads 之间归约。[论文 Sec. 3, Eq. 2-3](https://arxiv.org/html/2606.26087v1#S3.E3)

### 3. Multi-view correspondence loss

对随机采样的 query point $p_i^{v_1}$，轨迹在每个 target frame/view 给出一个 GT 位置 $p_j^{v_2,\mathrm{GT}}$ 和 visibility $o_j^{v_2,\mathrm{GT}}\in\{0,1\}$。论文把每个可见帧中的匹配定义为**单标签分类**：

$$
\mathcal L_{\mathrm{corr}}
=\sum_{v_2}\sum_j
o_j^{v_2,\mathrm{GT}}
\operatorname{CE}\left(
\mathcal C_{i,j}^{v_1,v_2}(p_i^{v_1},\cdot),
p_j^{v_2,\mathrm{GT}}
\right).
$$

其含义是让 source query token 在每个共可见 target frame 中把最大概率分配给唯一的 GT token。监督同时覆盖：reference-view 内时间对应、target-view 内时间对应、reference-target 跨视图对应。论文正文说最终对全部共可见帧取平均，虽然 Eq. 20 写成求和；由于代码未发布，精确 reduction/normalization 目前不可核验。[论文 Appendix B.5, Eq. 20](https://arxiv.org/html/2606.26087v1#A2.E20)

这里没有 Gaussian 邻域标签、soft-argmax coordinate loss、PCK head weight、future-only mask 或噪声可靠性 gate。轨迹位置在 latent grid 上作为唯一 class label；论文还明确描述为“single-label classification”。[论文 Sec. 5.3](https://arxiv.org/html/2606.26087v1#S5.SS3)

### 4. Multi-view tracking loss

MVTrack4Gen 不是只监督 attention map。它还从 Q/K 构造 multi-scale local 4D correlation volumes，并输入迭代式 transformer tracking head。局部 query/key 特征在 query point 和当前预测点周围双线性采样，四个尺度 $s\in\{1,2,3,4\}$ 的 correlation volume 被拼接；head 每轮预测 point、visibility、confidence 的残差 $(\Delta P,\Delta V,\Delta C)$。[论文 Sec. 5.2](https://arxiv.org/html/2606.26087v1#S5.SS2)、[Appendix B.2](https://arxiv.org/html/2606.26087v1#A2.SS2)

tracking 目标为：

$$
\mathcal L_{\mathrm{track}}
=\lambda_{\mathrm{seq}}\mathcal L_{\mathrm{seq}}
+\lambda_{\mathrm{conf}}\mathcal L_{\mathrm{conf}}
+\lambda_{\mathrm{vis}}\mathcal L_{\mathrm{vis}},
$$

$$
\lambda_{\mathrm{seq}}=0.05,
\qquad
\lambda_{\mathrm{conf}}=\lambda_{\mathrm{vis}}=1.0.
$$

轨迹坐标项是 tracking-head 输出的 visibility-weighted Huber：

$$
\mathcal L_{\mathrm{seq}}
=\frac{1}{\sum_{v,n,i}o_{n,i}^{v,\mathrm{GT}}}
\sum_{v,n,i}o_{n,i}^{v,\mathrm{GT}}
\rho\!\left(\hat p_{n,i}^v-p_{n,i}^{v,\mathrm{GT}}\right).
$$

`L_conf` 将预测 confidence 与实际回归误差校准，预测点落在 GT 小半径内时目标为高 confidence；`L_vis` 是预测 visibility logits 的 binary cross-entropy。[论文 Appendix B.5, Eq. 18-19](https://arxiv.org/html/2606.26087v1#A2.E18)

论文没有给出 `L_conf` 的完整概率公式、半径值、`L_vis` 的显式公式、Huber delta、迭代轮次 loss aggregation 等实现细节；官方代码尚未发布，因此不能从第一方材料恢复这些数值。

### 5. 噪声/timestep 处理

训练在 $t\sim\mathcal U(0,1)$ 的 noisy target latent 上运行，所以 `L_track` 和 `L_corr` 使用的 Q/K 会随 timestep 改变。论文只为 `L_diff` 明确写出 timestep weight $w(t)$；总目标中 `L_track` 和 `L_corr` 仅乘固定的 0.01，论文没有给这两个辅助损失增加 SNR weighting、high-noise cutoff 或其他 $g(t)$。[论文 Appendix B.5](https://arxiv.org/html/2606.26087v1#A2.SS5)

论文的 layer-timestep/PCK 图是用于发现 correspondence-specialized layer 的**分析**，不能当作训练中的 PCK head weighting 或 timestep gating。官方项目页也把 cycle consistency、PCK、attention score 和 confidence score描述为 probing/analysis 指标。[官方项目页 Analysis](https://cvlab-kaist.github.io/MVTrack4Gen/#analysis)

### 6. 哪些参数接收梯度

论文明确说明：训练时 fine-tune backbone 的 **3D attention layers** 和 **camera encoder**，冻结其余 backbone 参数；multi-view tracking head 也是联合训练的。tracking head 使用 DiT **第 18 层**的 Q/K 特征，因而 `L_track` 会经 local correlation volume 反传至共享 attention features；`L_corr` 直接通过 $\mathcal C(Q,K)$ 反传至产生 Q/K 的 attention 路径。[论文 Sec. 6.1](https://arxiv.org/html/2606.26087v1#S6.SS1)、[论文 Sec. 5.2-5.3](https://arxiv.org/html/2606.26087v1#S5.SS2)

可可靠确认的 layer/head 范围如下：

| 项目 | 官方材料能确认的内容 |
|---|---|
| Tracking head layer | 使用第 18 个 DiT layer 的 Q/K。 |
| Tracking multi-scale | 同一 Q/K feature pyramid 插值为 4 个尺度，不是 Top100 heads。 |
| `L_corr` layer | 论文说直接监督 correspondence-specialized 3D attention map，但没有像 tracking head 那样明确写出实际 layer 列表；实验和可视化聚焦第 18 层。 |
| Attention heads | 公式明确“省略 head 下标”；未披露各 head 分别算 loss、先平均 attention、还是其他聚合。 |
| Layer/head weights | 未披露任何 PCK 加权、Top-K head 选择或跨层加权。 |

因此，不能严谨地声称 MVTrack4Gen 的 `L_corr` 就是“第 18 层所有 heads 等权平均”。这很可能需要官方代码才能确定。

## 二、本地方案的精确定义

本地 loss core 位于 [`noise_gated_correspondence.py`](./noise_gated_correspondence.py)，Top100 聚合和诊断入口位于 [`run_pybullet_correspondence_diagnostics.py`](./run_pybullet_correspondence_diagnostics.py)。

### 1. 标签和 attention CE

本地首先把 CoTracker 轨迹从 pixel 坐标线性映射为连续 token-center 坐标 $\mu_{t,n}\in\mathbb R^2$，再构造归一化 Gaussian soft label：

$$
Y_{t,n}(s)=
\frac{\exp\left(-\|g_s-\mu_{t,n}\|^2/(2\sigma_Y^2)\right)}
{\sum_{s'}\exp\left(-\|g_{s'}-\mu_{t,n}\|^2/(2\sigma_Y^2)\right)},
\qquad \sigma_Y=1.0\ \text{token}.
$$

source point 只在选择 query row 时 round 到 token；target 保持连续并通过 Gaussian 表达。对每个选中 layer/head $h$：

$$
A_{h,t,n}(s)=\operatorname{Softmax}_s
\left(\frac{Q_h(q_n)K_{h,t}(s)^\top}{\sqrt d}\right),
$$

$$
\mathcal L_{\mathrm{softCE}}
=\operatorname{mean}_{(t,n)\in\mathcal V}
\sum_h w_h\left[-\sum_sY_{t,n}(s)\log A_{h,t,n}(s)\right].
$$

有效集合要求 source/target 都 visible，并且只保留固定 source `L01/F04` 之后的 `L02/F08 ... L12/F48`。它是单视图时间对应，不包含 MVTrack4Gen 的 reference-target multi-view token blocks。[本地 core lines 90-192](./noise_gated_correspondence.py#L90)、[collector lines 318-365](./run_pybullet_correspondence_diagnostics.py#L318)

### 2. Top100 layer/head 聚合

本地从 Wan2.2-TI2V-5B 的 30 blocks × 24 heads 中固定选择 100 个 `(block, head)`，这些 heads 按既有 Physics-IQ67 `pck32` ranking 选出，分布在 25 个 blocks。默认权重是归一化 `pck32`：

$$
w_h=\frac{\mathrm{PCK32}_h}{\sum_{h'}\mathrm{PCK32}_{h'}},
\qquad \sum_hw_h=1.
$$

CE 是先对每个 head 的独立 softmax 分布求 soft-label contribution，再按 $w_h$ 加权。坐标预测则先聚合概率：

$$
\bar A_{t,n}(s)=\sum_hw_hA_{h,t,n}(s).
$$

这与 MVTrack4Gen “第 18 层 tracking feature + 未披露的 `L_corr` head reduction”明显不同，也不能把 MVTrack4Gen 用于 layer discovery 的 PCK 分析理解为这种训练权重。[本地 head weight loader](./frozen_motion_probe.py#L66)、[本地 head config](../configs/physiciq67_pck32_s039_latest3350_top100_heads.json)

### 3. Attention soft-argmax coordinate Huber

本地直接从聚合 attention 计算 soft-argmax/期望坐标：

$$
\hat p_{t,n}=\sum_s\bar A_{t,n}(s)g_s,
$$

再计算 token-space Smooth L1：

$$
\mathcal L_{\mathrm{coord}}
=\operatorname{mean}_{(t,n)\in\mathcal V}
\operatorname{SmoothL1}_{\beta=0.5}(\hat p_{t,n},\mu_{t,n}).
$$

它直接约束 QK attention 的一阶空间期望，没有独立 tracking transformer，也没有 visibility/confidence prediction heads。[本地 finalize lines 411-449](./run_pybullet_correspondence_diagnostics.py#L411)

### 4. 显式噪声门控

对 scheduler sigma，当前本地公式是：

$$
\operatorname{SNR}(\sigma)=\frac{(1-\sigma)^2}{\max(\sigma^2,10^{-8})},
$$

$$
g(\sigma)=
\begin{cases}
\dfrac{\operatorname{SNR}(\sigma)}{\operatorname{SNR}(\sigma)+\gamma},
&\sigma<0.75,\\
0,&\sigma\ge 0.75,
\end{cases}
\qquad \gamma=1.
$$

最终诊断量为：

$$
\mathcal L_{\mathrm{local}}
=g(\sigma)\left(
\mathcal L_{\mathrm{softCE}}+0.25\mathcal L_{\mathrm{coord}}
\right).
$$

MVTrack4Gen 没有披露对应的辅助 loss gate/cutoff。[本地 gate lines 69-87](./noise_gated_correspondence.py#L69)、[本地 total lines 428-449](./run_pybullet_correspondence_diagnostics.py#L428)

### 5. 当前梯度状态

当前 `run_pybullet_correspondence_diagnostics.py` 是 **forward-only diagnostic**：它执行 `pipe.dit.requires_grad_(False)`，model forward 位于 `torch.no_grad()`，并且没有 optimizer step。因此当前脚本中**没有任何模型参数实际接收该 loss 的梯度**；`total` 是诊断数值，不是已接入训练循环的 objective。[本地 run lines 483-500, 594-626](./run_pybullet_correspondence_diagnostics.py#L483)

`noise_gated_correspondence.py` 的数学操作本身可微；如果未来在 autograd forward 中接入，梯度会流向所选 heads 的 Q/K 及其上游可训练参数。但最终训练哪些 Wan 参数或 adapter 参数，取决于尚未接入的训练器配置，不能从当前 diagnostic 脚本推出。

## 三、逐项对照

| 维度 | MVTrack4Gen | 本地方案 | 是否相同 |
|---|---|---|---|
| 核心 correspondence | 对逐帧 QK spatial softmax 做监督 | 对逐帧 QK spatial softmax 做监督 | 思想相同 |
| CE target | 唯一 GT token，hard single-label | 连续中心的 Gaussian soft label，$\sigma_Y=1$ token | 不同 |
| 视图范围 | reference/reference、target/target、跨 reference-target | 当前仅单视频时间对应 | 不同 |
| 时间范围 | 随机 query；两个视图中的所有共可见 latent frames | 固定 `L01/F04` query；仅未来帧 | 不同 |
| 可见性 | target visibility 加权，query 被假定为有效 | 显式要求 source 和 target 均 visible | 接近但不等价 |
| Layer/head | tracking head 明确 layer 18；`L_corr` heads/reduction 未披露 | 跨 25/30 blocks 的 Top100 heads | 不同 |
| Head weighting | 未披露；没有 PCK training weight 证据 | 固定 PCK32 归一化权重 | 不同 |
| Coordinate loss | tracking-head 轨迹输出的 Huber | attention soft-argmax 的 Huber | 不同计算图 |
| Confidence/visibility | 有 `L_conf` 和 `L_vis` | 无对应预测头或 loss | 不同 |
| Tracking module | multi-scale local 4D correlation + iterative transformer | 无独立 tracking head | 不同 |
| Noise treatment | sampled $t$；只有 `L_diff` 明确使用 $w(t)$；aux loss 无披露 gate | SNR soft gate，$\sigma\ge0.75$ hard zero | 不同 |
| Aux weight | `0.01 L_track + 0.01 L_corr` | `g(σ)(L_softCE + 0.25 L_coord)` | 不同 |
| 轨迹监督来源 | 训练数据的 multi-view GT/dense tracks；MV-TAP pseudo tracks 用于论文分析 | CoTracker pseudo-GT，F04 SAM2 mask 初始化 | 不同 |
| 当前参数更新 | 训练 3D attention、camera encoder、tracking head | diagnostic 中全部冻结，无 optimizer | 不同 |

## 四、最接近的对应关系

可以把本地方案理解为借鉴了 MVTrack4Gen `L_corr` 的基本方向，但做了以下实质改造：

1. 将 hard token CE 改为连续坐标 Gaussian soft-label CE。
2. 将单层/未披露 head aggregation 改为跨层 Top100 PCK-weighted heads。
3. 将 tracking-head trajectory Huber 改成 attention soft-argmax Huber。
4. 将全视图、全共可见帧监督缩小为固定 source 到 future frames。
5. 增加论文没有的 SNR reliability gate 和 high-noise hard cutoff。
6. 目前只做 forward diagnosis，尚未实现 MVTrack4Gen 的联合训练和 tracking head。

因此，最准确的表述是：**本地 loss 与 MVTrack4Gen `L_corr` 属于同一类 attention-correspondence supervision，但不是其复现；它是一个带 soft targets、head selection、coordinate moment regression 和 noise gating 的新变体。**

## 五、可验证性边界

截至调研日期，官方仓库 `main` 分支只有 README，并明确写着 “Code Coming Soon”；仓库 HEAD 为 `3685b0dba30e96ac3d827fe4f865c1d43c5d3b8f`。因此以下问题无法从第一方代码核验：

- `L_corr` 具体监督哪些 layer。
- multi-head attention 的 loss/reduction 方式。
- hard GT point 到 latent token 的离散化细节。
- `L_corr` 求和/平均和 batch/query normalization 的实现。
- `L_conf` 的概率公式、阈值，以及 Huber delta。
- tracking head 各 refinement iteration 是否都计算 loss。

这些未知项不影响“不采用相同 loss”的总体判断，因为 Gaussian soft labels、Top100 PCK weighting、attention soft-argmax Huber、SNR gate/cutoff 均明确存在于本地代码，而在 MVTrack4Gen 官方论文目标中不存在。

## Sources

1. [MVTrack4Gen official project page](https://cvlab-kaist.github.io/MVTrack4Gen/)
2. [MVTrack4Gen arXiv abstract, v1](https://arxiv.org/abs/2606.26087v1)
3. [MVTrack4Gen arXiv HTML, v1](https://arxiv.org/html/2606.26087v1)
4. [MVTrack4Gen paper PDF, v1](https://arxiv.org/pdf/2606.26087v1)
5. [Official GitHub repository](https://github.com/cvlab-kaist/MVTrack4Gen)
6. [Official repository README at inspected commit](https://github.com/cvlab-kaist/MVTrack4Gen/blob/3685b0dba30e96ac3d827fe4f865c1d43c5d3b8f/README.md)
