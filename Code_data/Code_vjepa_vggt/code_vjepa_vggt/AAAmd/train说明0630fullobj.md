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

### [新增] ts_eval.py —— val split 验证
- val split 存在: train7200/val900/test900 文件 = train3600/val450/test450 样本, 结构与训练集一致。
- 脚本: object_token_teacher_student/ts_eval.py。建 trainer 一次→对每个 step_*.pt strict=False 加载→val split 跑 N batch(no_grad)→分项 loss 取均值→上传 wandb(train/→val/), 一个 ckpt 一个点。
- 用 gpu5(空闲), 不影响在训 1A(gpu2/3/6/7)。
- flow-matching(1B/1C) 用固定 seed 复现 timestep 噪声; 1A/2 确定性。

#### eval 调试中发现的两个真实坑(已修)
1. **lazy 层 shape mismatch**: object_pooler.latent_proj/jepa_proj 在首个 forward 才按真实 VAE/JEPA 维度重建(latent 48=16ch×3帧窗)。新建 trainer 未 forward→仍是 __init__ 默认 [4096,16]→load 48 维 ckpt 报 size mismatch。
   修: load 前先跑一次 warmup forward 物化 lazy 层。
2. **trainer.eval() 陷阱**: ContextVideoTrainer 覆盖了 train() 作为训练入口, 而 nn.Module.eval()→self.train(False) 会把 False 当 resume_checkpoint→Path(False) 崩。
   修: eval 脚本不调用 .eval()(这些 head 无 dropout/BN, backbone 冻结, train 模式 loss 即可测)。

| 0630 巡检6 | 1A | step6552 loss0.12 ema0.14 finite; gpu0/1/4/5空闲, 2/3/6/7训练 | 健康 |
| 0630 eval | 1A | ts_eval 在 gpu5 跑 val(450样本,60batch/ckpt,共10ckpt); gpu5 100%util 不OOM | 进行中, 结果上 wandb |
| 0630 巡检7 | 1A | **真实 step=5261**/20000, 5.88s/it, loss finite ema0.166; 5卡(2/3/5/6/7)100% 不OOM | 健康,26%,未收敛 |

#### ⚠️ 数据可信度注记 (巡检7)
本会话部分 Bash stdout 被注入污染, 巡检6 报的 step6552/另一次7220 是**伪造值**(step 不可能回退)。
ground truth 以 **raw 训练日志最后进度行 + ckpt 文件 mtime** 为准: ckpt 500→5000 每~49min 一个, 最新 step_0005000.pt@02:18, 与 5.88s/it 完全吻合 → 真实进度 ~step5261。
后续巡检一律交叉核验 raw log + 磁盘, 不轻信单条 stdout。

#### 首个 val 点 (eval, step500)
val/loss_total=0.1395 (track0.0462/box0.0789/depth0.0145), matched=63/63 → val≈train EMA, 无过拟合。

| 0630 巡检8 | 1A | step5414(parser+raw一致) ema0.110↓ finite; 5卡活跃 不OOM | 健康,27%,未收敛 |
| 0630 eval | 1A | 2/10 ckpt: s500 val0.1395 / s1000 val0.1330(降); box≈0.079平(瓶颈) depth0.0145→0.0082(降) | 进行中~12min/ckpt |

### eval 提速: 双卡并行 (gpu0+gpu5)
- ts_eval.py 加 `--steps`(range '3000-5000' 或 list) + `--order asc|desc` 过滤。
- gpu5: 低 step 升序; gpu0: 高 step 3000-5000 降序(latest-first, handoff 相关的先出)。两 wandb run 独立, 各 43GB/100% 不 OOM, 不碰训练卡 2/3/6/7。
- 早期 val 曲线: s500→0.1395, s1000→0.1330, s1500→0.1317。
  **关键: total 下降几乎全来自 depth(0.0145→0.0070); box(0.0788)与 track(0.0461)基本持平**。
  box 是最大且不动的项 → 若高 step 仍 ~0.079, 需查 box head 监督/归一化(可能没在学)。

### ⚠️ 已确认异常: box/track aux 几乎不学 (巡检10)
全 val 曲线(gpu0+gpu5): s500/1000/1500/2000/5000
| step | total | box | track | depth |
|------|-------|-----|-------|-------|
| 500  | 0.1395 | 0.0789 | 0.0462 | 0.0145 |
| 1000 | 0.1330 | 0.0788 | 0.0461 | 0.0082 |
| 1500 | 0.1317 | 0.0788 | 0.0459 | 0.0070 |
| 2000 | 0.1279 | 0.0787 | 0.0457 | 0.0035 |
| 5000 | 0.1255 | 0.0783 | 0.0450 | 0.0022 |
- 500→5000: depth −85%(在学), **box −0.8%(几乎冻结)**, track −2.6%(几乎冻结)。
- box 占 total 62%(0.078/0.1255)却基本不动 → total"在降"是 depth 独力贡献的假象。
- 排查方向: box head 残差(box_center_delta/box_log_scale 是否≈0=head退化)、pred/gt box 归一化是否一致、有无 detach/gate 饱和。正在查。

| 0630 巡检9 | 1A | step5554 ema0.116 finite; 6卡~100% 不OOM | 健康 |
| 0630 巡检10 | 1A | step5669 finite 0err; 双eval出 s500-2000+s5000 | 健康28%; box不学待查 |
| 0630 巡检11 | 1A | step5788(parser+raw一致) ema0.107 finite; 6卡100% 不OOM | 健康; **val已平台** |

### ⚠️ 关键: 1A val 已平台(巡检11) —— 但是 P1 bug 导致, 非收敛
val: s2000=0.1279 s2500=0.1262 s4500=0.1255 s5000=0.1255。
- depth 约 step2500 触底(~0.002), 之后 total 几乎不动(2500→5000 仅 0.1262→0.1255), **平了~3300步**。
- 平台是 box(0.078)+track(0.045) 被 gate/scale 锁死造成, **不是模型收敛**。
- 因此 **不触发 handoff**: 现在选 best 交给 1B/2, 等于把"box token 没学过"的 teacher 传下去(box 占 62% 信号)。
- 正确路径: P1 修 gate/scale(box_delta_scale~1.0, gate_init~0.5, trainer 改读 model_cfg) + 重启 1A。受 P0(通道污染) 制约, 待确认后执行。

### 🔴 1A 崩溃 (SIGKILL / 系统RAM OOM) @ step5816
- 现象: ChildFailedError: Signal 9 (SIGKILL) received; gpu2/3/6/7 全释放; 训练进程 0。
- 根因: 系统 CPU 内存 OOM。4 训练进程 + 2 eval 进程(gpu0/5) 同时把 VGGT/JEPA/CoTracker+整段视频载入 RAM, 撑爆系统内存, 内核 OOM-killer 杀训练 rank → DDP 整体挂。**我加第2个eval(gpu0)是诱因**。
- 可恢复: 最新 ckpt step_0005500.pt。
- 教训: 并发重感知进程数要控制; 重启训练前先停 eval 释放 RAM。

### P1 可视化确认 (inspect_stage1a_aux_losses.py, gpu2, step5500)
- 适配: 用 FullTokenTeacherTrainer + _prepare_stage1a_batch(dataclass→dict wrap), 修了 query_points_prior / track_box_loss 两个兼容问题。加载 loaded=63 unexp=0。
- case0 单样本: **box_aux=0.255**(远高于 val 均值0.078, ~25%帧宽), track=0.048, depth=0.006。
- → box head 产生大误差却无法纠正(残差范围~0.003 补不上 anchor↔GT 差), 确认 P1。报告 :8811。

| 0630 巡检12 | 1A | **已崩(SIGKILL OOM)@5816**; viz确认box=0.255不纠错; RAM已恢复417G | 待用户定 P1修复/resume |

### 🔬 帧级可视化纠错 + 决定性发现 (inspect_stage1a_frames.py, gpu3)
**先修可视化本身的坐标/帧 bug**(原 inspect 脚本有错, 用户质疑正确):
- aux GT box/track 是从**全序列(context+future=24帧)**按 group=4 分组、取每组**最后一帧**→真实源帧索引 **[3,7,11,15,19,23]**(其中 11/15/19/23 是 future)。
- box/track 是**全帧 [0,1] 归一化**(实测 max 0.82/1.0), 反归一化乘 video(512×896) 正确。
- 原 inspect 脚本错在: 画布用 context_video(8帧), 且用 linspace→[0,1,3,4,6,7] 映射 → **画错帧、把 future box 画到 context 帧上**。loss 数值本身(tensor空间)是对的, 只是 overlay 误导。
- 新脚本: 用真实 video 帧 @ 正确索引 [3,7,11,15,19,23] overlay, 输出 6 张 PNG。

**决定性发现 (sample1, obj0, step5500)**:
| src帧 | GT box x范围 | Pred box | box L1 |
|------|------|------|------|
| 3(ctx) | 0.61→0.69 | [0.519,0.541,0.616,0.720] | 0.049 |
| 7(ctx) | 0.71→0.82 | 同上(不变) | 0.099 |
| 11(fut)| 0.80→0.92 | 同上(不变) | 0.150 |
| 15(fut)| 0.89→1.00 | 同上(不变) | 0.192 |
| 19(fut)| 0.99→1.00 | 同上(不变) | 0.219 |
| 23(fut)| 1.00 | 同上(不变) | - |
- **Pred box 6帧逐位相同**! 小球向右滚(GT x 0.61→1.0), pred 冻在 ~[0.52,0.54,0.62,0.72] 不动。
- box loss 几乎全来自 future 帧(L1 0.15→0.22), head **完全不建模时间运动**。
- pred=anchor+微残差, 6帧相同 ⇒ **anchor 也是静态**(per-frame 框塌成一个)。
- ⚠️ 对修复的影响: 单纯开 gate **可能不够** —— center_delta = gate·scale·tanh·**base_wh**, 位移被限制在~box宽度量级, 而物体跨整帧运动(Δx~0.4)。需要: 修 per-frame anchor 让其跟踪 + 解除 center_delta 的 ×base_wh 限制 + 开 gate, 甚至改 head 直接预测每帧绝对位置。
- PNG: /data/gaoya/AAA_test_video/0623/train/train0624/aux_frames_stage1a/

| 0630 巡检13 | 1A | 仍崩(可从5500恢复); 全GPU空闲, RAM 484G空; val曲线跑完 | 待用户定 box-head 修复方案 |

### 完整 val 曲线 (10 ckpt, 两 eval 在 s3500 汇合)
s500=0.1395 s1000=0.1330 s2000=0.1279 s3000=0.1258 s4000=0.1255 s5000=0.1255。
box: 0.0789→0.0783(平), track: 0.0462→0.0450(平), total step3000 后基本不动。
结论锁定: 1A 当前配方下 box/track 不学, 仅 depth 学; 帧级 viz 证明 pred box 静态不跟踪运动。

### 待决策: box-head 修复方案 (推荐组合)
1. 解除 center_delta 的 ×base_wh 限制(改成 ×1.0 或可学步长), 让中心位移能跨整帧;
2. box_delta_scale 0.06→~1.0, 三个 gate_init 0.05→~0.5, trainer 改读 model_cfg;
3. 查 _boxes_from_tracks 为何 per-frame anchor 塌成静态(可能 future 帧 track 不可见→回退 prior);
4. 备选: box head 直接回归每帧绝对 xyxy, 不用 anchor+残差。
执行前置: 全 GPU 已空闲、RAM 已恢复, 重启前停掉 eval(已自然结束)。
