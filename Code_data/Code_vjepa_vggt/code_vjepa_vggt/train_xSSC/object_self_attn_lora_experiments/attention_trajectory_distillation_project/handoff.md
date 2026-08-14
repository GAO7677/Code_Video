# Frozen Motion Probe Attention-Trajectory Distillation · Handoff

更新时间：2026-08-14

## 1. 项目目标

本项目建立一个独立的 Wan2.2-TI2V-5B 训练入口，在不加载任何历史
LoRA checkpoint 的前提下，保留原始 flow-matching loss，并通过冻结的
Wan2.2 baseline DiT 测量 GT 与 Student 预测之间的 object-query motion
correspondence 差异。

核心目标是让 attention trajectory loss 只能通过改善第一次 Student DiT
产生的 `x0_pred` 来下降，不能通过更新第二次 probe forward 的 Q/K 参数走
捷径。

## 2. 已确认的训练定义

### 2.1 Main Student

```text
x_t
→ Main Student DiT
→ v_pred
→ x0_pred = x_t - sigma_t * v_pred
```

- Main Student 从官方 `Wan-AI-Wan2.2-TI2V-5B` baseline 初始化。
- 不加载 OpenVid LoRA、preset LoRA 或 resume LoRA。
- 不进行 Wan 全参数微调。
- 仍会从 baseline 新注入零初始化的 `full_sa` 或 `t_head`
  self-attention adapter；“不加载 LoRA”指不加载已有 LoRA 权重。
- 原始 flow-matching loss 的实现和计算形式保持不变。

### 2.2 Shared Frozen Motion Probe

Teacher 和 Student 使用同一个额外加载的官方 Wan2.2 baseline DiT：

- 所有 probe 参数均为 `requires_grad=False`；
- probe 始终保持 `eval()`；
- probe 不注册到 Main Student 的 `nn.Module` tree；
- probe 不进入 optimizer、DDP state 或 Student checkpoint；
- Teacher probe forward 在 `torch.no_grad()` 下执行；
- Student probe forward 保留对输入 `x0_pred` 的梯度。

入口会检查 Main Student 和 frozen probe 使用完全相同的三个 Wan2.2 DiT
safetensor shards，并拒绝其他 DiT 权重。

### 2.3 Fixed object query

默认沿用已有 attention overlay 协议：

- fixed query pixel frame：F04；
- fixed query latent frame：latent-1；
- heads：latest3350 PCK ranking 的 Top100；
- 当前配置包含 100 个 layer-head，分布在 25 个 transformer blocks；
- full-mask token membership：只要 object mask 的任意像素与 latent cell
  相交，该 cell 就属于 query region。

每个选中 layer-head 的 query 采用更严格定义：

1. Teacher 在 GT probe input 上提取固定 query rows 的 Q；
2. Teacher Q 执行 `detach()`；
3. Student heatmap 使用完全相同的 Teacher GT-Q 与 Student K 计算。

因此 Student 既不能选择 query 位置，也不能提供 loss 所使用的 Q 表征：

```text
Teacher map = softmax(Q_GT @ K_GT^T / sqrt(d))
Student map = softmax(stopgrad(Q_GT) @ K_Student^T / sqrt(d))
```

多个 query rows 先在每个 physical head 内聚合，再对 Top100 heads 等权平均。
已有可视化代码使用 query-row sum；当前训练代码使用 mean。两者仅相差固定
常数，进行 heatmap 概率归一化后结果相同。

### 2.4 Shared probe corruption

对 GT `x0` 和 Student `x0_pred` 采样一次共享的 `epsilon_p`：

```text
x_probe = (1 - probe_noise_level) * x0
        + probe_noise_level * epsilon_p
```

Teacher 与 Student 使用完全相同的：

- `epsilon_p`；
- `probe_noise_level`；
- `probe_timestep`；
- text/TI2V conditioning；
- clean conditioning latent frames。

`probe_timestep` 不随机改变。`probe_noise_level` 与 scheduler 对应 timestep 的
sigma 会同时记录，避免二者不一致时无法审计。

### 2.5 Loss

```text
L = L_flow
  + lambda_heatmap * KL(StudentAttention || TeacherAttention)
  + lambda_trajectory * Huber(StudentTrajectory, TeacherTrajectory)
```

- `L_flow`：原始 Wan flow-matching loss；
- heatmap loss：对归一化 Top100 aggregate heatmap 计算
  `KL(student || teacher)`；
- trajectory loss：每个 latent frame 对 heatmap 进行空间归一化和
  soft-argmax，得到 `[B, 13, 2]` 轨迹，再计算 Huber loss；
- trajectory 坐标归一化到 `[0, 1]`。

## 3. 梯度路径

预期且已经由单元测试验证的路径为：

```text
L_heatmap / L_traj
→ explicit QK-map checkpoint output
→ Student K field in frozen probe
→ frozen probe input
→ x0_pred
→ v_pred
→ first Main Student DiT
```

不允许出现：

```text
loss → update frozen probe parameters
```

训练入口还会按配置周期性执行：

```python
torch.autograd.grad(auxiliary_loss, v_pred)
```

如果 auxiliary loss 无法回传到第一次 Student forward 的 `v_pred`，训练会
直接报错。

## 4. SAM2 是否可以替代 simulator GT mask

可以。训练真正要求的是 **frozen external query mask**，不要求一定来自
simulator GT。

允许的方案：

```text
GT clean training video 的 F04
→ SAM2 / SAM2 AMG
→ 固定目标 mask
→ 固定 Wan query rows
→ Teacher 和 Student 共用
```

不允许的方案：

```text
Student x0_pred
→ SAM2
→ Student 自己产生 mask/query rows
```

后一种方案会让对象位置、外观、SAM2 分割误差和 correspondence loss 同时
变化，破坏固定 query 的控制变量。

SAM2 的关键问题不是能否使用，而是如何确定哪个候选 mask 对应目标对象。
推荐优先级：

1. simulator object ID / GT point 选择 SAM2 mask；
2. 固定 tracking point 或 bbox 选择包含该位置的 SAM2 mask；
3. 使用跨帧 track ID 选择目标；
4. 仅当数据没有目标身份要求时，才按面积、稳定性等固定规则选择 AMG mask。

不建议在训练 iteration 内重复运行 SAM2。应在 clean GT video 上离线提取并
缓存 mask/token rows，保证速度、确定性和可复现性。

## 5. 数据接口

每个 raw training sample 必须直接或在 `metadata` 内提供至少一种：

| Key | 支持形状 | 含义 |
|---|---|---|
| `object_query_token_indices` | `[Q]` | 已计算的固定 Wan flattened token rows |
| `object_query_mask` | `[H,W]`、`[T,H,W]`、`[O,T,H,W]` | simulator 或 frozen SAM2 binary mask |
| `object_query_points` | `[P,2]`、`[T,P,2]`、`[O,T,P,2]` | 归一化或 pixel-space tracking points |

优先级为 token indices → mask → points。缺少所有 query source 时直接报错，
不存在 Student-derived fallback。

当前 `xssc_replay_mix` no-GT-box 数据并不自动提供这些字段。正式训练前必须：

- 修改 dataset adapter，使其读取 SAM2/simulator query cache；或
- 在样本 metadata 中写入上述字段。

当前入口强制 per-GPU batch size 为 1，避免一个 batch 内多个 query set 的
不明确合并。

## 6. 已实现文件

| 文件 | 作用 |
|---|---|
| `train_xssc_object_self_attn_lora_frozen_motion_probe.py` | 独立训练入口、baseline/probe 加载、x0 重建、双 probe forward、loss 与梯度诊断 |
| `frozen_motion_probe.py` | 固定 GT-Q/Student-K heatmap、QK capture、soft-argmax、KL、Huber、mask/point 到 token 映射 |
| `test_frozen_motion_probe.py` | 冻结参数、输入梯度、共享噪声、checkpoint、固定 GT-Q、`v_pred` 梯度测试 |
| `README.md` | 配置、数据契约、训练命令模板和旧 Scheme B 对比 |
| `handoff.md` | 当前项目状态与下一步执行说明 |

项目目录：

```text
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/
train_xSSC/object_self_attn_lora_experiments/
attention_trajectory_distillation_project/
```

## 7. 已完成验证

已通过 6 项小模型测试：

1. frozen probe 参数不产生梯度；
2. Student probe input 保留梯度；
3. Teacher heatmap stop-gradient；
4. Teacher/Student 使用完全相同的 probe noise；
5. activation checkpoint 显式返回 QK map 时可以正确反传；
6. 固定 Teacher GT-Q 后，loss 可以回传到第一次 forward 的 `v_pred`。

同时完成：

- Python `py_compile`；
- 训练入口 `--help` 导入检查；
- 三个 baseline shard 存在性检查；
- latest3350 Top100 配置检查：100 heads / 25 blocks；
- Main Student 与 frozen probe 同源 baseline 检查；
- 非 baseline DiT、旧 LoRA 和 resume LoRA fail-closed 检查。

测试命令：

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/attention_trajectory_distillation_project
/home/gaoya/miniconda3/envs/wan-cu128/bin/python - <<'PY'
import test_frozen_motion_probe as tests
names = sorted(name for name in dir(tests) if name.startswith("test_"))
for name in names:
    getattr(tests, name)()
print("PASS", len(names), "tests")
PY
```

## 8. 尚未执行

- 尚未启动完整 Wan2.2 5B training smoke test；
- 尚未把 SAM2 query cache 接入当前训练 dataset；
- 尚未运行训练集 case 级 diagnostic；
- 尚未生成 GT/x0/noisy heatmap 的可视化页面；
- 尚未确定首批训练 case 和 SAM2 多候选 mask 的目标选择规则；
- 尚未对 `lambda_heatmap`、`lambda_trajectory`、`probe_timestep`、
  `probe_noise_level` 做超参数扫描。

## 9. 下一步：训练前 case 诊断

正式训练前应先随机选择 3–5 个训练集 case，执行只前向、不更新参数的
diagnostic。每个 case 至少展示以下内容。

### 9.1 输入与 SAM2

1. 原始 GT video；
2. F04 原图；
3. SAM2 AMG 全部候选 masks；
4. 目标 mask 的选择依据和置信信息；
5. 最终 frozen mask overlay；
6. 映射到 `13 × 22 × 40` grid 的 fixed query cells；
7. query token 数量与 flattened indices。

### 9.2 x0 重建

1. 原始 GT video；
2. VAE encode/decode 后的 GT `x0`，用于分离 VAE reconstruction error；
3. 训练 timestep 的 `x_t`；
4. baseline Main Student 的 `v_pred`；
5. 解码后的 `x0_pred`；
6. GT `x0` 与 `x0_pred` 的并排和差分可视化。

### 9.3 Frozen Motion Probe

Teacher 和 Student 使用同一 `epsilon_p`、`probe_noise_level` 和
`probe_timestep`，展示：

1. GT noisy probe input；
2. Student noisy probe input；
3. Teacher Top100 mean heatmap overlay；
4. Student Top100 mean heatmap overlay；
5. Teacher/Student heatmap difference；
6. 13 个 latent frames 的 soft-argmax trajectories；
7. 两条轨迹叠加；
8. query rows 与 Teacher GT-Q/Student-K 计算定义；
9. `L_flow`、heatmap KL、trajectory Huber、total loss；
10. auxiliary loss 到 `v_pred` 的 gradient norm。

### 9.4 推荐页面结构

每个 case 使用独立 section 或子页面：

```text
Case metadata
├── GT / VAE-GT / x0_pred videos
├── SAM2 candidates / selected mask / query cells
├── Teacher heatmap video
├── Student heatmap video
├── Heatmap difference video
├── Teacher vs Student trajectory overlay
└── Loss and gradient audit table
```

建议将中间结果写入：

```text
/data/gaoya/agent-data/outputs/frozen_motion_probe_training_diagnostics/
```

代码和轻量配置保留在项目目录，不要把视频、SAM2 cache 或模型输出写到
`/home/gaoya`。

## 10. 运行风险与审查重点

1. **显存**：每个 rank 同时持有 Main Student 5B DiT 和 frozen probe 5B
   DiT；必须先做单卡 batch-1 smoke test并记录峰值显存。
2. **SAM2 object identity**：如果没有 point/bbox/object ID，AMG mask 的目标
   选择可能错误，必须在页面中展示全部候选和选择依据。
3. **query mask 对齐**：SAM2 mask 必须对应训练预处理后的空间坐标；不能把
   未经过相同 resize/crop 的 mask 直接映射到 Wan grid。
4. **固定 Q 的解释**：Student map 是 `Q_GT @ K_Student^T`，不是 Student
   probe 原始 self-attention 的 `Q_Student @ K_Student^T`。报告时必须明确。
5. **轨迹不是 GT 物理轨迹**：soft-argmax trajectory 是 Top100 QK
   correspondence trajectory，只能作为内部 motion proxy。
6. **probe noise/timestep**：二者可以独立配置，但需要报告 scheduler sigma，
   并通过后续消融确认最佳 probe noise regime。
7. **当前尚无 5B end-to-end 证据**：现阶段只能确认实现与小模型梯度逻辑，
   不能声称训练效果已经改善。

## 11. 旧 Scheme B 与当前版本

| 项目 | 旧 Scheme B | 当前 Frozen Motion Probe |
|---|---|---|
| Student measurement pass | trainable Student DiT | 独立 frozen baseline DiT |
| Teacher/Student probe 参数 | 不完全相同或 Student 可更新 | 完全相同且固定 |
| Student query | Student Q representation 可变化 | detached Teacher GT-Q |
| Student correspondence field | Student Q/K | GT-Q × Student-K |
| 降低 loss 的捷径 | 可更新 probe Q/K 参数 | probe 参数不可更新 |
| 有效梯度出口 | probe 参数和 `x0_pred` | 只能通过 `x0_pred` 回到 Main Student |
| 主 loss | flow + attention auxiliary | flow + KL + trajectory Huber |

## 12. 当前结论

代码层面的 Frozen Motion Probe 与梯度约束已经完成，并通过小模型测试；
下一步不应直接启动长训练，而应先完成 SAM2 cache 接入和 3–5 个训练 case
的端到端可视化诊断。只有确认 object mask、query cells、`x0_pred`、Teacher/
Student heatmap、trajectory 和梯度范数均符合预期后，才适合运行短训练 smoke
test 和后续超参数实验。
