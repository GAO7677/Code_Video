# Full-Object Teacher-Student 训练说明与监控记录 (0630)

> 目标：监控 1A→1B→1C→2(→3) 各阶段训练；wandb 上传；每阶段用**最好权重**作为下一阶段对应模块的 init；
> 确保 loss/梯度数值正常、梯度回传正常、对应模块权重正常更新；异常及时排查重启；
> 用 gpu2/3/6/7 尽量提util但不 OOM。

---

## 1. Pipeline 与模块依赖

```
1A (token teacher: object_pooler + aux_heads, 无DiT)
   └─pooler/aux─> 1B (oracle 注入: DiT object_* + adapter)
                     └─DiT注入+adapter─> 1C (joint: 1B + pooler tail/aux 联调)
   └─pooler─────> 2  (predictor + future_heads, 蒸馏 future token)
1C + 2 ─────────> 3  (bridge, teacher-forcing 退火; 接口待补)
```

权重 handoff 通过新加的 `--init-from <ckpt.pt>`：strict=False 只载匹配的 trainable 权重、**不恢复 step**。

## 2. 各阶段基础配置（模板）

| 阶段 | trainable | 优化器 | 主 loss | max_steps | 输出子目录 |
|------|-----------|--------|---------|-----------|-----------|
| 1A | object_pooler + aux_heads | adamw | track+box+depth L1 (各1.0) | 20000 | stage1a_full_token |
| 1B | DiT object_*/norm4 + adapter | paged_adamw8bit | denoising (aux=0) | 20000 | stage1b_oracle_cross_attn |
| 1C | 1B + pooler tail + aux | paged_adamw8bit | denoising + aux(0.02) | 20000 | stage1c_joint |
| 2 | predictor + future_heads | adamw | token蒸馏(1.0)+track/box(0.1) | 20000 | stage2_predictor |

公共：res[512,896], num_context_frames 8, batch 1/card, bf16, save_every 500, max_checkpoints 10,
wandb project=vjepa_vggt_wan, 输出根=
`/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/`

## 3. 最好权重选择策略

无独立 val loop。采用**平滑训练 loss 最低**的 checkpoint 作为 best（每 500 步存一个，留最近 10 个）。
后续若加 val 集再升级。

## 4. 健康判据（每阶段巡检）

- loss 有限、整体下降或在合理区间震荡；无 NaN/Inf。
- runner 已内置 non-finite grad / non-finite param 检测，异常会直接 raise 中止。
- trainable_param_abs_max 随训练缓慢变化（说明权重在更新）。
- GPU 利用率高、显存不 OOM。

## 5. 进度与状态

| 时间 | 阶段 | 状态 | 备注 |
|------|------|------|------|
| 0630 初 | 1A | 启动 2 卡(gpu6/7), step~48, loss 1.0→0.31 正常 | 拟迁 4 卡提速 |
| 0630 | 1A | 迁 4 卡(gpu2/3/6/7), 100%util/~45GB each 不OOM; step34 loss0.10 ema0.21 finite | 健康 |
| 0630 巡检2 | 1A | step325 loss0.097 ema0.233 finite 0err; 4卡util在100%↔0%间(DDP barrier正常) | 健康, 首ckpt@500未到 |
| 0630 巡检3 | 1A | step554 loss0.051 ema0.141 finite 0err; 首ckpt step_0000500.pt 已存 | 健康, ema仍降未收敛 |

### Handoff 预校验 (巡检3)
step_0000500.pt: 63 张量 = object_pooler.*(42) + object_aux_heads.*(21), 全 full-path key,
strict=False 加载进 1B/1C/2 同名子模块可匹配 → --init-from 机制已 de-risk。

| 0630 巡检4 | 1A | step685 inst0.28 ema0.204 finite 0err; 4卡100%util 不OOM | 健康, geom L1 批间噪声正常, 未收敛 |
| 0630 巡检5 | 1A | step928 inst0.095 ema0.226 finite 0err; 4卡100%util | 健康; 后续改每~1h/有事件才记 |

### GPU 编排策略
- **当前**：只有 1A 可跑（1B/1C/2 都依赖 1A 权重）→ 4 卡全给 1A。
  4 卡 9.25s/it vs 2 卡 5.67s/it，但每步 2× 样本，净吞吐 +~18%。
- **1A 完成后**：1B 与 Stage2 **并行**（都只依赖 1A）——
  1B 占 gpu6/7（2卡,DiT,paged8bit），Stage2 占 gpu2/3（2卡,无DiT）。满 4 卡、无 batch/LR 顾虑。
- **1C** 在 1B 之后（依赖 1B），可占其空出的卡对。

### Handoff 命令（待 1A best ckpt 产出后）
```
# best 选择
python3 object_token_teacher_student/ts_monitor.py best \
  /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token \
  /data/gaoya/agent-data/outputs/ts_smoke/stage1a_train.log
# 1B / 2 启动追加：  --init-from <1A_best.pt>
```

## 6. 问题 / 推测原因 / 方案 记录

（按时间追加）

### [已修] 启动脚本 CONFIG 路径漏 `code_vjepa_vggt/`
- 现象：1A 首次启动 `FileNotFoundError: .../Code_vjepa_vggt/object_token_teacher_student/config_...yaml`
- 原因：生成 run_*.sh 时路径少了内层包目录。
- 方案：sed 修正 4 个脚本并校验配置存在。已解决。

### [设计缺口→已补] 跨阶段 handoff
- 现象：`--resume-checkpoint` 会连 step 一起恢复，无法用作下一阶段 init。
- 方案：runner + train() + 6 个入口脚本新增 `--init-from`（strict=False 只载权重、step 归零）。已实现、已编译通过。
