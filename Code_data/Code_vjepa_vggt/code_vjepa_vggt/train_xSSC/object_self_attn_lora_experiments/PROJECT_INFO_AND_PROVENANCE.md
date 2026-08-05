# xSSC-Wan 项目信息与权重溯源

## 1. 项目概述

本项目以 Wan2.2 TI2V-5B 为视频生成主干，研究 
(1) 物理仿真数据集SFT训练
(2) 视频自监督模型提取 object信息做 Cross-Attn 条件注入
(3) Self-Attention 不同head如何影响视频中object的运动

输入： 8 帧 context video + text
输出： 49 帧、512x896 视频；

训练时采用49帧视频训练，显存约占36.95GiB / 47.99GiB，推理时可以跑189帧



## 2. 训练阶段

```text
Wan2.2 TI2V-5B (Baseline)
  -> 通用数据+仿真数据 进行v2v适配：OpenVid(85%) + MOVI-D (8%) + Genesis rigid(6%) LoRA, step-010000
  -> 仿真数据集继续微调
```


### 2.1 v2v适配训练

OpenVid LoRA 覆盖 Wan 的 30 个block：
- 输入 24 帧，384x672，mixed context sampling 
- LoRA rank=32 
- 4 GPU，每卡 batch=1，gradient accumulation=4
- Self-Attention：Q/K/V/O LoRA。
- 文本 Cross-Attention：Q/K/V/O LoRA。
- FFN：`ffn.0` 和 `ffn.2` LoRA。
- 可训练参数 80.609M。
- step-010000

### 2.2 仿真数据集微调self-attention LoRA


### 2.3 视频自监督模型微调cross-attn + self-attention LoRA

当前主前向流程为：

```text
8-frame context video
  -> SAM2 AMG masks + filtering
  -> pseudo boxes
  -> frozen DINOv3 xSSC
  -> context slots [B, 8, 11, 512]
  -> optional slot-track deduplication
  -> LayerNorm + Linear(512, 3072) + learned time embedding
  -> object tokens [B, 8*11, 3072] = [B, 88, 3072]
  -> object cross-attention in Wan DiT
  -> 49-frame generated video
```

xSSC 使用 bbox 初始化 frame-0 slots，然后通过 transition 和 SlotAttention 沿 context 时间递推。Wan 训练阶段冻结 SAM2、DINO 和 xSSC，只训练实验配置指定的 projector、gate、object-attention LoRA 或 Self-Attention LoRA。


### 4.1 DINOv3 MOVi-C 版本

| 配置 | 数值 |
|---|---|
| Backbone | DINOv3 ViT-L/16 LVD-1689M，冻结 |
| 输入 | 256x256，16x16 patch-token 网格 |
| Backbone feature | 1024 |
| Slot | 11 slots，slot dim=512 |
| 数据 | MOVi-C train；每次随机抽取 6 帧 clip |
| 增强 | random crop scale 0.75-1.0、resize、horizontal flip=0.5 |
| 迁移训练 | 从 global step 15k 接续到 50k |
| 并行 | 2 GPU，每卡 batch=64，accumulation=3，有效 batch=384 |
| 优化器 | Adam，lr=5e-5，gradient clip=0.05 |
| 保存/验证 | checkpoint 每 1,000 step；val 每 1,250 step |

训练模块：

- Encoder projection MLP。
- bbox initialization MLP。
- SlotAttention，包括 Q/K/V、GRU 和 FFN。
- Transition，包括时间 embedding；transition dropout=0.5。
- 4 层 decoder；decoder dropout=0。
- DINOv3 backbone 冻结。

参数口径：

| 组件 | 参数量 | xSSC 训练状态 | Wan 训练状态 |
|---|---:|---|---|
| DINOv3 ViT-L/16 | 303.130M | 冻结 | 冻结 |
| xSSC 非 Backbone | 81.044M | 可训练 | 冻结 |
| 合计 | 384.174M | 81.044M 可训练 | 全部冻结 |

### 4.2 Official DINOv2 版本

官方对照使用 `rsfq2_r-ytvis.py` 和 `42-0130.pth`，backbone 为 DINOv2 ViT-S/14 register-token 版本。checkpoint 共 34.048M 参数；在 Wan Object-only 对照中整体冻结。

## 5. Wan 训练公共配置

| 配置 | 数值 |
|---|---|
| 输入/输出 | 8 帧 context，生成 49 帧，512x896 |
| 默认数据 | PyBullet 30%、PhyCo Kubric 30%、OpenVidHD 40% |
| 每卡 batch | 1 |
| Gradient accumulation | 4；具体有效 batch 还需乘 GPU 数 |
| 优化器 | paged AdamW 8-bit，lr=1e-4，weight decay=0.01 |
| 精度 | bf16 |
| Save interval | 500 optimizer steps |
| Object LoRA | rank=32，alpha=32，dropout=0.05 |
| Self-Attention LoRA | rank=32，alpha=32，dropout=0 |
| Slot-track dropout | 0.1 |
| 空 AMG 样本 | 正式 Object 训练过滤，最多重采样 20 次 |

Object 条件模块的可训练参数拆分：

| 模块 | 参数量 |
|---|---:|
| Slot LayerNorm/projector/time embedding | 1,773,568 |
| Object cross-attention LoRA | 23,592,960 |
| Object gates | 92,160 |
| Object-only 合计 | 25,458,688 |

## 6. 实验方案与参数

下表的“冻结参数”按训练涉及组件统计；SAM2 可能作为预处理单独执行，因此该总量不等于同时驻留 GPU 的参数量。

| 方法 | 可训练模块 | 可训练参数 | 冻结参数 | 总涉及参数 | Object/xSSC | 数据 |
|---|---|---:|---:|---:|---|---|
| Object-only | projector/time + object-attn LoRA + gate | 25.459M | 11,994.007M | 12,019.466M | xSSC-26k | 30/30/40 |
| Full-SA + Object | Object-only + 全30层 Self-Attn Q/K/V/O LoRA | 49.052M | 11,994.007M | 12,043.059M | xSSC-26k | 30/30/40 |
| S-head59 + Object | Object-only + S59 compact head LoRA | 34.683M | 11,994.007M | 12,028.690M | xSSC-26k | 30/30/40 |
| T-head70 + Object | Object-only + T70 compact head LoRA | 34.863M | 11,994.007M | 12,028.870M | xSSC-26k | 30/30/40 |
| Full-SA + Object + Dedup | 与 Full-SA + Object 相同；Dedup 零参数 | 49.052M | 11,994.007M | 12,043.059M | xSSC-26k | 30/30/40 |
| Full-SA + Object + Dedup (xSSC-50k) | 同上 | 49.052M | 11,994.007M | 12,043.059M | xSSC-50k | 30/30/40 |
| T-head70 + Object + Dedup (xSSC-50k) | Object-only + T70 compact head LoRA | 34.863M | 11,994.007M | 12,028.870M | xSSC-50k | 30/30/40 |
| Full-SA + No-Object | 全30层 Self-Attn Q/K/V/O LoRA | 23.593M | 11,385.387M | 11,408.980M | 完全关闭 | 30/30/40 |
| Full-SA + No-Object, PyBullet 100% | 同上 | 23.593M | 11,385.387M | 11,408.980M | 完全关闭 | 100/0/0 |
| Full-SA + No-Object, Kubric 100% | 同上 | 23.593M | 11,385.387M | 11,408.980M | 完全关闭 | 0/100/0 |
| T-head70 + No-Object | T70 compact head LoRA | 9.404M | 11,385.387M | 11,394.791M | 完全关闭 | 30/30/40 |
| Motion-head100 + No-Object | Motion Top100 compact head LoRA | 11.076M | 11,385.387M | 11,396.462M | 完全关闭 | 30/30/40 |
| Object-only + Official xSSC | official projector/time + object-attn LoRA + gate | 24.672M | 11,643.882M | 11,668.553M | official DINOv2 xSSC | 30/30/40 |

Slot-Dedup 使用跨时间 slot-track 的 mean-frame cosine，相似度阈值 0.94，模式为 merge，最少保留 3 个 slot。它发生在 xSSC 输出之后、object-token projection 之前，不新增可训练参数。

## 7. Head 分类

### 7.1 原始实验范围

- 模型：Wan+LoRA、Wan+xSSC、PhysRVG。
- 数据：20 个 case。
- 稳定性范围：三个模型都完成的 22 个公共 seed。
- 去噪步：5、15、25、35。
- Head：30 blocks x 24 heads，共 720 个。
- 保存粒度：每个 model/case/seed/denoise-step/block/head 的全部原始特征值、rank、score 和角色。

### 7.2 分类分数

```text
score_S = 0.55 rank(local_enrichment)
        + 0.45 rank(same_frame_mass)

score_T = 0.55 rank(trajectory_selectivity_log2)
        + 0.25 rank(trajectory_enrichment)
        + 0.20 rank(mean_time_distance)

score_P = 0.75 rank(fixed_position_enrichment)
        + 0.25 rank(aligned_enrichment)

score_C = 0.55 rank(object_context_enrichment)
        + 0.25 rank(full_context_enrichment)
        + 0.20 rank(history_bias)

score_G = 0.60 rank(full_entropy)
        + 0.25 rank(full_mean_time_distance)
        + 0.15 rank(-same_frame_mass)
```

单个样本先对四个去噪步求平均。第一、第二名 margin 小于 0.08，或四步角色一致率低于 0.75，则标为 M。轨迹少于 8 个有效时间点或有效率低于 0.8 时，禁用 T/P 判定。

最终公共稳定类别要求三个模型得到相同且非 M 的角色：

| 类别 | 含义 | Head 数量 |
|---|---|---:|
| S | 空间局部/同帧 | 159 |
| T | 运动轨迹 | 13 |
| P | 固定位置 | 82 |
| C | Context | 20 |
| G | 全局候选 | 75 |
| M | 混合、跨模型不一致或不稳定 | 371 |

### 7.3 实际训练子集

- **S-head59**：公共稳定 S159 中的 same-frame-dominant 59 个；另有 local-dominant 100 个。二者互斥且并集为 S159。
- **T-head70**：用户提供并冻结的 70-head 训练清单。它不等于公共稳定 T13；名单和 SHA256 可追溯，但上游阈值与完整筛选链未记录完整。
- **Motion-head100**：从单个 case、单个 seed 的 Wan+LoRA Object PCK@32 排名选出的 Top100；用于运动敏感 Head 实验，不是语义 T 类。

Head 数据：

- 最终 CSV：`/data/gaoya/agent-data/outputs/head_classification_csv/common22_public_stable/head_classification_all_720.csv`
- 原始特征：`/data/gaoya/agent-data/outputs/wan_dit_head_role_dose_control/head_classification`
- S59 配置：`configs/same_frame_mass_heads_full59.json`
- T70 配置：`configs/common_t_heads_full70.json`
- Motion100 配置：`configs/lora_pck32_top100_heads.json`

可视化入口：

- `http://127.0.0.1:8844/head-evidence/fulltoken-head-classification.html`
- `http://127.0.0.1:8844/head-evidence/head_roles_50seeds/`
- `http://127.0.0.1:8844/head-evidence/common-stc-all-heads-qk-seed851/`
- `http://127.0.0.1:8844/head-evidence/head-role-depth-distribution/`

## 8. Checkpoint 追溯

每个正式 run 根目录保存 `resolved_experiment_config.json`，包含：

- 完整配置继承链 `config_sources`。
- 解析后的训练配置 `resolved_config`。
- Head 身份快照、来源和 SHA256。
- GPU、并行方式、训练模块和启动摘要。
- xSSC、DINO、SAM2、OpenVid LoRA 的实际路径。
- `resume_from` 父 checkpoint。

当前所有 watcher 方法、run 根目录、checkpoint steps、resume 父权重和 Head SHA256 自动汇总到：

```text
/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/project-info/checkpoint_inventory.csv
```

该 CSV 和 `provenance.json` 由 `build_project_provenance_page.py` 生成，避免手写 checkpoint 清单与实际目录不同步。

## 9. 代码与页面入口

- 项目页：`http://127.0.0.1:8844/project-info/`
- 总入口：`http://127.0.0.1:8844/`
- 项目页生成器：`build_project_provenance_page.py`
- Dashboard 生成器：`build_xssc_lora_checkpoint_dashboard.py`
- 公共配置：`configs/base.json`
- 上游权重记录：`/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/AAAtrain.md`

前台启动 8844 服务：

```bash
cd /data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub
/home/gaoya/miniconda3/envs/wan-cu128/bin/python -m http.server 8844 \
  --bind 0.0.0.0 --directory .
```
