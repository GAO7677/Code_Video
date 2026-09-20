# Utonia + 41-query predictor: 三例接线与拟合诊断

日期：2026-09-18。状态：完成三例冻结特征与两组各200步拟合；没有独立验证/测试，正式渲染与训练规模待确认。

本文件记录本次新增的 `scene_token_cache.py` 和 `future_query_predictor.py` 实验。工作目录同时出现的 `predictor_scene_adapter.py` 是复用旧预测头的另一条接口准备记录，本次未改动它及其主README；不能混用两个版本的参数量、测试数、训练状态或结论。

## 目的与范围

先把实际可获得的 motion + scene 信息接入一个能训练的预测器，检查数据对应、梯度、输出时间和保存/重载；不先增加更多物理loss。

- 复用已有三例RGB0-7及VGGT粗尺度点图，不新增渲染，不重新加载VGGT。
- 冻结Utonia，提取并缓存场景特征。
- 两层四头注意力用41个时间query读取场景，输出RGB8-48位置。
- 同构固定场景内容参照与视觉场景各训练200步。
- 仅位置SmoothL1 + 0.2倍区间速度SmoothL1；不加载DiT/VAE，不用配对loss，不加穿透loss或事件头。
- 仅单物体球体/直立puck接线，不包含角运动和多物体交互，不声称是完整刚体状态模型。

## 实际输入与来源

| 输入 | shape | 来源 |
|---|---|---|
| 观察位置 | [B,8,K,3] | 已有动态context，RGB0-7 |
| 尺寸 | [B,K,3] | 动态context |
| 观察速度 | [B,8,K,3] | 位置差分，首个区间重复一次；不取未来状态 |
| motion特征 | [B,K,65] | 相对位置24 + 速度24 + 尺寸3 + 末位置3 + 观察时间8 + 重力3 |
| scene_xyz | [B,1792,3] | 观察深度的粗尺度中点，经实际相机反投影 |
| scene_features | [B,1792,1386] | 动态区域排除后的点、RGB、局部法线，冻结Utonia |
| scene_mask | [B,1792] | 观察有效点mask，不是GT碰撞体标签 |
| future_dt | [B,41] | 末观察时刻之后1/30至41/30秒 |
| 未来位置输出 | [B,K,41,3] | 当前仅K=1 |

没有额外读取物体GT材质/四元数、静态实体尺寸/旋转/材质或未来图像作为前向输入。
这是观察RGB加动态位置/尺寸和实际相机辅助，不是严格RGB-only。
未来位置只由监督构建器读取，并与同样本观察位置和时刻核对；predictor前向拒绝target/oracle等非声明字段。

## 场景token对应关系

1. 校验既有coarse_static_points.npz及其绑定的raw_vggt.npz的hash。
2. 在同一世界坐标系用局部差分估计朝向实际相机的法线，排除无效邻域和明显深度断点。
3. 确定性抽取最多65536个有效观察像素，使用官方Utonia变换，scale=1，不做旧高度归一化。
4. full-upcast后通过root_inverse找同一root内depth confidence最高的来源像素，不使用静态GT射线桥接。
5. 同时保留特征、对应观察世界位置、frame/y/x和编码器坐标；编码坐标不直接当世界坐标。
6. 最多保存1792个token。三例原生root数为50272/52510/57953。此采样预算只用于控制小试验开销，尚未证明关键边缘覆盖充分。

Utonia共137.25M参数，冻结。使用物理GPU0，模型加载约4.73秒，进程内记录总耗时约9.74秒，峰值allocated约4.09GiB。此计时不含之前的渲染、VGGT、SAM及进程启动。

`static_valid`仅表示正深度且未被观察动态mask排除，不等于准确的静态碰撞体。不可见区域仍未知。depth confidence只用于排序，不是准确概率。
VGGT上游看过完整观察RGB，过滤动态点不等于两条表示完全独立。粗点图仍有尺度、深度和分割误差，不能当作精密SDF。

## 网络与训练

```text
motion65 -> MLP -> h[128]
future_dt -> time MLP -> e_t[128]
q_t = h + e_t
scene1386 -> LayerNorm + Linear -> scene128
q_t -> 两层四头scene cross-attention（含运动残差和FFN）
    -> 共享位置残差头
    -> 末位置 + 观察末区间速度*dt + 残差*dt
```

41个query共享场景K/V，不复制[B,K,41,N,128]特征。
时间直接对应实际30Hz，未调用旧24Hz canonical插值。
query含观察运动与时间，不含未来GT位置，也不是沿预测位置自回归读取。
本轮没有新增时间self-attention。

两组结构、初始化一致，均572831参数：
- constant：场景内容和相对关系取固定值，保留网络与运动残差，不能读取实例场景差异。
- visual：读取真实对应的观察视觉缓存。

AdamW，lr=3e-4，weight_decay=0.01，clip=1，seed42；每步同时读取3例，有效batch3，各200步。
这是短拟合检查，不是此前讨论的2000step/batch64正式实验。
归一化统计仅在3个拟合样例上计算，单列为fit_only_stats，不得复用于正式划分。

## 结果：仅训练拟合

| 模式 | 初始loss | 最终loss | 训练ADE | 训练FDE |
|---|---:|---:|---:|---:|
| constant | 0.035411611 | 0.000196928 | 0.004220919 m | 0.003382159 m |
| visual | 0.035411611 | 0.000038913 | 0.002390074 m | 0.000304862 m |

视觉组从step2开始有非零场景投影梯度。输出头零初始化，因此第一步场景梯度为零是初始化结果，不是接口断路。
交换三个不同族样本的场景后，constant输出不变，visual平均位置变化约0.243343m。该交换不是物理匹配的反事实实验，仅证明输入内容影响计算。

![三例训练拟合，不是独立测试](predictor_fit_pilot/training_fit.png)

能说明：接线、梯度、优化及保存/重载可用，模型能拟合3个已见例子。
不能说明：场景引起的运动变化正确、碰撞/穿透已修好、41-query优于单query、视觉模型具有独立测试优势或泛化能力。
两个版本都可能利用不同motion记住3个样例，毫米级训练ADE不代表深度几何达到毫米级精度。

## 验证

- 本次运行原65项及新增14项CPU测试，共79项通过。这一计数不包含同时新增的另一适配器分支测试。
- 覆盖未来信息拒绝、动态点排除、像素/世界点对应、相机旋转后的法线一致性、空场景有限性、初始化CV、场景梯度和constant内容不变性。
- 三例1792个token均通过对应观察静态mask，均覆盖RGB0-7。
- root代表像素到编码器代表点最大距离分别12.922/10.199/10.812mm，符合1cm网格对角线范围；不是对GT几何的误差度量。
- CPU重新加载权重与保存GPU输出的最大坐标差：constant 1.49e-6m，visual 6.56e-7m。
- 源码hash、输入像素、观察几何、特征和监督来源均核对通过。
- GPU任务均已结束，工作负载只使用物理GPU0；GPU4未使用。

结果：[训练JSON](predictor_fit_pilot/report.json)、[CPU复核](predictor_fit_pilot/cpu_verification.json)、[特征提取摘要](utonia_feature_pilot/summary.json)。

## 下一步

正式数据规模尚待选择，未启动批量渲染或正式训练。建议先120条小规模功能试验；600条需要更多渲染资源。
原4200条物理bank没有配套RGB，必须按同一蓝图和随机种子重放、核对轨迹，再渲染观察RGB0-7，实际相机取render_metadata。
当前三例是开发拟合样例，不能充当测试集，也不能把其图像拼接到4200 bank。
按此前两例8帧Cycles约61-88秒粗估，120条仅渲染约2-3 GPU小时，600条约10-15 GPU小时；不含提取、重试，不是完成时间保证。
确认规模后锁定物理来源、训练/验证分组、训练步数和评测。小样本不应机械照搬2000step当成已收敛要求。保持同一数据与预算的两组对照，先不加新loss。

## 复现

全部前台运行；不要覆盖已有输出。GPU命令执行前确认GPU0空闲，禁止GPU4。

```bash
cd /data/gaoya/agent-data/outputs/a3_training_redesign_20260908/visual_scene_context8_20260918
env CUDA_VISIBLE_DEVICES='' TMPDIR=/data/gaoya/agent-data/cache/sg-overlay/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B -m unittest -v test_prepare_context test_probe_vggt test_context_geometry test_observed_masks test_align_observed_depth test_predictor_interface
```

冻结特征提取到新目录：

```bash
timeout --signal=TERM 600s env CUDA_VISIBLE_DEVICES=GPU-34579b7b-23fc-35ea-539f-1eac72fb7fa5 TMPDIR=/data/gaoya/agent-data/cache/sg-overlay/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME=/data/gaoya/agent-data/cache/huggingface XDG_CACHE_HOME=/data/gaoya/agent-data/cache CUDA_CACHE_PATH=/data/gaoya/agent-data/cache/cuda /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B scene_token_cache.py --cases pilot_gap_0000 pilot_barrier_0004 pilot_door_ball_0006 --output utonia_features_repeat --gpu-uuid GPU-34579b7b-23fc-35ea-539f-1eac72fb7fa5
```

三例拟合到新目录：

```bash
timeout --signal=TERM 600s env CUDA_VISIBLE_DEVICES=GPU-34579b7b-23fc-35ea-539f-1eac72fb7fa5 TMPDIR=/data/gaoya/agent-data/cache/sg-overlay/tmp PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 XDG_CACHE_HOME=/data/gaoya/agent-data/cache CUDA_CACHE_PATH=/data/gaoya/agent-data/cache/cuda /data/gaoya/agent-data/envs/physrvg-full-sa/bin/python -B predictor_pilot.py --features utonia_feature_pilot --cases pilot_gap_0000 pilot_barrier_0004 pilot_door_ball_0006 --output predictor_fit_repeat --gpu-uuid GPU-34579b7b-23fc-35ea-539f-1eac72fb7fa5 --steps 200
```

代码：[scene_token_cache.py](scene_token_cache.py)、[future_query_predictor.py](future_query_predictor.py)、[predictor_pilot.py](predictor_pilot.py)、[test_predictor_interface.py](test_predictor_interface.py)、[verify_predictor_pilot.py](verify_predictor_pilot.py)。
