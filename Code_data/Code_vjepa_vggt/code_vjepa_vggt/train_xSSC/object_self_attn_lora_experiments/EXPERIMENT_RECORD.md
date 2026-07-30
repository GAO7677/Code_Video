# Wan+xSSC Object/Self-Attention LoRA 实验记录

> 本文件是当前四组训练实验的唯一动态记录入口。方法实现细节见
> [README.md](README.md)，参数真值以 `configs/*.json` 和每次 run 内的
> `resolved_experiment_config.json` 为准；本文件只记录实验决策、运行身份、
> 权重、评测结果和方案变更，避免复制整份配置。

- 最近更新：`2026-07-30 04:01 UTC`
- 当前状态：四组正式训练运行中，首个正式 checkpoint 尚未生成
- 正式 run tag：`formal_20260729T184553Z`
- tmux：`xssc_lora_formal_20260729T184553Z`

## 1. 当前研究问题

在同一个 Wan2.2 TI2V 5B、OpenVid 物理 LoRA 初始化、DINOv3 xSSC object
condition 和训练数据设置下，对比以下四种参数高效训练范围：

| 实验 | 新增可训练模块 | Self-attention 范围 | 可训练参数 |
|---|---|---:|---:|
| Object-only | xSSC 投影、时间嵌入、object gates、object cross-attention LoRA | 无 | 25.459 M |
| Full-SA | Object-only + self-attention q/k/v/o LoRA | 30 层全部 head | 49.052 M |
| S-head | Object-only + head-specific q/k/v/o LoRA | 59 个 `same_frame_mass` S heads，分布于 21 层 | 34.683 M |
| T-head | Object-only + head-specific q/k/v/o LoRA | 70 个 common T heads，分布于 21 层 | 34.863 M |

四组实验的变量只应是 self-attention LoRA 的启用范围。数据、初始化权重、
xSSC object branch、优化器和训练长度保持一致。

## 2. 方法流程

1. 加载 Wan2.2 TI2V 5B，并加载 OpenVid/MOVi-D/Genesis 的
   `step-010000` LoRA。
2. 将该 LoRA delta 合并进 Wan 权重，再卸载旧 PEFT wrapper。合并后的
   OpenVid 更新仍保留，但旧 LoRA A/B 不再继续训练。
3. 从 49 帧训练视频中取 8 帧 context；训练前处理使用 center crop。
4. 用 SAM2 AMG 在 context 上生成并过滤 pseudo masks/boxes；
   `selected_masks=0` 的样本重采样并过滤。
5. frozen DINOv3 xSSC 输出 slots `[B, 8, 11, 512]`。
6. slots 经 `LayerNorm -> Linear(512, 3072) + learned time embedding`，
   展平为 object tokens `[B, 88, 3072]`。
7. object tokens 输入 Wan 30 层 object cross-attention。object branch 的
   q/k/v/o LoRA、gate、投影和时间嵌入参与训练；xSSC 与 Wan 主干保持冻结。
8. 按实验模式额外训练无 self-attention LoRA、全部 self-attention LoRA、
   59 个 S heads LoRA 或 70 个 T heads LoRA。

训练时以 slot track 为单位做 `0.1` dropout，即同一 object slot 在 8 个
context 时间步上共同保留或共同丢弃。

## 3. 固定正式配置

| 类别 | 当前值 |
|---|---|
| 生成规格 | 49 帧，512x896，bf16 |
| Context | 8 帧 |
| 数据混合 | 30% 0717 PyBullet + 30% PhyCo Kubric + 40% OpenVidHD |
| 单卡 batch | 1 |
| 梯度累计 | 8 |
| 单实验有效 batch | 8 |
| 优化器 | paged AdamW 8-bit |
| 学习率 / weight decay | `1e-4` / `0.01` |
| 梯度裁剪 | L2 norm `1.0` |
| 训练长度 | 20,000 optimizer steps，最多 100 epochs |
| 随机种子 | 42 |
| 保存策略 | 每 500 optimizer steps，最多保留 10 个 |
| W&B project | `xssc_wan_head_sparse_training` |
| 定期验证 | 当前关闭；正式质量比较由固定 checkpoint 推理和统一指标完成 |

配置继承关系：

| 实验 | 正式配置 | 方法配置 |
|---|---|---|
| Object-only | `configs/formal_object_only_gpu1.json` | `configs/object_only.json` |
| Full-SA | `configs/formal_full_sa_gpu2.json` | `configs/full_sa.json` |
| S-head | `configs/formal_s_head_gpu6.json` | `configs/s_head.json` |
| T-head | `configs/formal_t_head_gpu7.json` | `configs/t_head.json` |

head 身份真值不在本文件重复列出：

- S-head：`configs/same_frame_mass_heads_full59.json`
- T-head：`configs/common_t_heads_full70.json`

S/T checkpoint 内保存排序后的 `[block, head]`、配置 SHA256；run 目录同时保存
`head_selection_config.json`。恢复训练时严格验证身份，防止同 shape 权重加载到
另一组 heads。

## 4. 初始化权重

| 组件 | 权重路径 | 训练状态 |
|---|---|---|
| Wan2.2 TI2V 5B | `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B` | frozen |
| OpenVid 物理 LoRA | `/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors` | 先 merge，旧 A/B 不训练 |
| DINOv3 xSSC | `/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/dinov3_xSSC/restart_save1000_20260720T140029Z/movi_c_transfer15000_b64_acc3_20260721T134713Z/rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42/step-026000.pth` | frozen |
| DINOv3 ViT-L/16 | `/data/gaoya/ckpt/facebook-dinov3-vitl16-pretrain-lvd1689m/model.safetensors` | frozen |
| SAM2.1 Hiera-L | `/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt` | frozen，仅生成 pseudo boxes |

## 5. 正式 Run 注册

状态快照为 `2026-07-30 04:01 UTC`；训练仍在继续，step 只作为排障快照，
不作为实验结论。

| 实验 | GPU | 快照 step | 输出目录 | W&B |
|---|---:|---:|---|---|
| Object-only | 1 | 448 | `/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/object_only_gpu1_formal/formal_20260729T184553Z` | [oiykukey](https://wandb.ai/875222004-gy/xssc_wan_head_sparse_training/runs/oiykukey) |
| Full-SA | 2 | 442 | `/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/full_sa_gpu2_formal/formal_20260729T184553Z` | [hrajh7ie](https://wandb.ai/875222004-gy/xssc_wan_head_sparse_training/runs/hrajh7ie) |
| S-head | 6 | 448 | `/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/same_frame_s_head_full59_gpu6_formal/formal_20260729T184553Z` | [rtthg4ds](https://wandb.ai/875222004-gy/xssc_wan_head_sparse_training/runs/rtthg4ds) |
| T-head | 7 | 448 | `/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora/common_t_head_full70_gpu7_formal/formal_20260729T184553Z` | [wx1g5jny](https://wandb.ai/875222004-gy/xssc_wan_head_sparse_training/runs/wx1g5jny) |

控制日志：

`/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_control/20260729T184553Z/logs`

查看 tmux：

```bash
tmux attach -t xssc_lora_formal_20260729T184553Z
```

checkpoint 只在 `accelerator.sync_gradients=true` 的完整 optimizer update 后
保存，不会在一次梯度累计中的多个 micro-steps 重复保存。

## 6. 权重注册

### 6.1 Smoke 权重

以下四个权重均完成 1 个 optimizer step，并通过一条样本、2 个去噪步的推理
闭环。它们只用于代码正确性验证，不用于比较生成质量。

| 实验 | Smoke checkpoint |
|---|---|
| Object-only | `/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora_smoke/smoke_object_only_gpu1/smoke_20260729T184553Z/checkpoints/step-000001/checkpoint.safetensors` |
| Full-SA | `/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora_smoke/smoke_full_sa_gpu2/smoke_20260729T184553Z/checkpoints/step-000001/checkpoint.safetensors` |
| S-head | `/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora_smoke/smoke_same_frame_s_head_full59_gpu6/smoke_20260729T184553Z/checkpoints/step-000001/checkpoint.safetensors` |
| T-head | `/data/gaoya/agent-data/checkpoints/xssc_object_self_attn_lora_smoke/smoke_common_t_head_full70_gpu7/smoke_20260729T184553Z/checkpoints/step-000001/checkpoint.safetensors` |

Smoke 推理输出：

`/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_smoke/smoke_20260729T184553Z`

### 6.2 正式权重

| 实验 | step | checkpoint | 用途 | 状态 |
|---|---:|---|---|---|
| 四组 | 500 | 各正式 run 的 `checkpoints/step-000500/checkpoint.safetensors` | 首轮统一推理 | 等待生成 |

后续只在此表登记 `首个 / 阶段性 / best / final` 权重，不逐条抄录所有周期
checkpoint。每个权重必须同时记录用途和对应评测行。

## 7. 推理与评测

### 7.1 重要入口

| 用途 | 文件 |
|---|---|
| 训练主体 | `train_xssc_object_self_attn_lora.py` |
| 配置解析与启动 | `launch_from_config.py` |
| 单配置训练入口 | `run_train_from_config.sh` |
| checkpoint 推理主体 | `infer_xssc_object_self_attn_lora.py` |
| 按 run 配置恢复推理 | `run_infer_from_experiment.sh` |
| Smoke -> 推理 -> 正式训练门禁 | `run_smoke_then_formal_gpu1267.sh` |
| 组件测试 | `test_experiment_components.py` |

推理必须从 checkpoint 所属 run 的 `resolved_experiment_config.json` 重建模型，
尤其不能手工猜测 S/T head 列表。标准调用：

```bash
NUM_INFERENCE_STEPS=40 \
TEST_LIST=/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
bash run_infer_from_experiment.sh /path/to/run/checkpoints/step-000500 0 \
  /data/gaoya/agent-data/outputs/xssc_object_self_attn_lora
```

默认推理规格为 49 帧、512x896、8 帧 context、prefix context sampling、
center crop，并保存 numeric traces。GPU 4 禁止使用。

### 7.2 评测结果

目前尚无可用于四组方法比较的正式 benchmark 结果。Smoke loss 和单条两步
视频仅证明训练、保存、严格加载和推理链路可运行，不能作为质量结论。

| 评测批次 | Checkpoints | 数据 / 推理配置 | 指标结果 | 原始结果目录 | 状态 |
|---|---|---|---|---|---|
| 首轮正式评测 | 四组 step-000500 | 待固定 | 待计算 | 待登记 | pending |

后续每次评测只追加一行：记录 checkpoint 集合、测试集版本、推理参数、指标
汇总和原始目录。详细逐 case JSON 留在 `/data`，不粘贴进本文件。

## 8. 已完成核查

- 四组配置继承与 expected trainable parameter 数通过。
- OpenVid LoRA 在四组中使用相同 merge 初始化；实验 LoRA 均为 fresh delta。
- S-head checkpoint 绑定 59 个 head 身份，T-head 绑定 70 个 head 身份。
- 梯度累计期间 checkpoint 只在 optimizer step 后保存。
- 四组 Smoke train、checkpoint 严格加载和生成视频均成功。
- 正式训练已在 GPU 1/2/6/7 启动并上传 W&B。

## 9. 方案变更日志

| 日期 | 变更 | 原因 | 影响范围 |
|---|---|---|---|
| 2026-07-29 | 建立 Object-only、Full-SA、S-head、T-head 四组受控实验 | 比较 object branch 与不同 self-attention LoRA 范围 | 全部当前 run |
| 2026-07-29 | S-head 扩展为完整 59 个 `same_frame_mass` heads | 使用完整目标子类，不再只用 32-head replication 子集 | S-head |
| 2026-07-29 | 新增 70 个 common T-head 配置 | 增加时间/轨迹 head 对照 | T-head |
| 2026-07-29 | checkpoint 保存绑定 optimizer step | 避免梯度累计期间重复保存 | 全部实验 |
| 2026-07-29 | S/T checkpoint 写入 head identity 与配置 hash | 防止同 shape、不同 head 列表误加载 | S-head、T-head |
| 2026-07-29 | 四组 Smoke 与推理通过后启动正式训练 | 建立正式运行门禁 | 全部当前 run |

## 10. 后续维护约定

1. 方法或超参数调整：先改 `configs/*.json`；本文件只在变更日志记录差异、
   理由和新 run tag，不复制完整 JSON。
2. 每个正式 run 都登记 config、GPU、输出目录、W&B ID 和初始化 checkpoint；
   不覆盖历史 run。
3. 权重表只登记有明确用途的 milestone/best/final checkpoint。
4. 指标表只存汇总和原始目录；逐 case 结果、视频、trace 均保留在 `/data`。
5. 结论必须绑定具体 checkpoint 和评测批次；训练 loss、Smoke 结果不直接作为
   方法优劣证据。
6. 任何 S/T head 列表调整都必须创建新的 head 配置文件和 subset ID，不能
   原地改写已用于训练的列表。
