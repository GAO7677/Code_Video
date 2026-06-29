# 0624 object branch 精简版 MoE 训练说明

这份说明对应当前 `train_v_newtrain.py` 的 object-heads-only strict 版本。它不是旧的 4 路融合方案，而是已经收敛成“`track+geometry` / `appearance` 两路融合”的精简版 object branch。

## 0. 2026-06-26 新增：Stage2 冻结 head、只训 object 注入链路

在当前这轮评估里，已经对 5 个已有 checkpoint 做了统一 val 口径评估，结果记录在：

- `/data/gaoya/agent-data/outputs/headonly_val_eval_final/full_val_metrics.json`

其中当前最优 checkpoint 是：

- `step-004000`
- 路径：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_heads_only_gpu67_fresh500_val/checkpoints/step-004000`
- 主要 val 指标：
  - `mean_loss_total = 0.0452403`
  - `mean_track_aux = 0.0635358`
  - `mean_box_aux = 0.3143830`
  - `mean_depth_aux = 0.0744844`

这说明当前最稳定的不是更晚的 `step-007600 / 007800`，而是 `fresh500_val` run 下的 `step-004000`。

基于这个结论，下一阶段不建议继续同时训练 `object_pooler + object_aux_heads + object_adapter + object_dit_branch`。更合理的做法是：

- 冻结已经最稳定的：
  - `object_pooler`
  - `object_aux_heads`
- 只训练：
  - `object_adapter`
  - Wan DiT 的 object 注入分支：
    - `object_embedding`
    - `object_cross_attn`
    - `object_gate`
    - `norm4`

原因：

- 当前 `head-only` 阶段的目标已经基本完成：
  - object token 已经能比较稳定地回归 `track / box / depth`
- 下一步的核心矛盾不再是“怎么把 head 拟合得更好”
- 而是“怎么让主去噪路径真正消费这组已经稳定的 object token”
- 如果继续把 `object_pooler + object_aux_heads` 一起放开训，更容易把已经收敛的 object 表征带漂

当前代码已经原生支持这个切换，不需要改训练主逻辑：

- `--train_object_pooler`
- `--train_object_aux_heads`
- `--train_object_adapter`
- `--train_object_dit_branch`
- `--freeze_non_object_trainables`

对应逻辑在：

- `train_v_newtrain.py`

推荐的 Stage2 训练配置：

- `resume_from = step-004000`
- 不传：
  - `--train_object_pooler`
  - `--train_object_aux_heads`
- 传：
  - `--train_object_adapter`
  - `--train_object_dit_branch`
  - `--freeze_non_object_trainables`
- loss 权重改为：
  - `lambda_main = 1.0`
  - `lambda_track_aux = 0.02`
  - `lambda_box_aux = 0.02`
  - `lambda_depth_aux = 0.02`
- learning rate 建议降到：
  - `5e-5`

解释：

- `lambda_main` 打开后，主去噪路径开始真正回传
- `track / box / depth aux` 不再是主导项，而是“防漂约束”
- 这样更像“固定老师特征，训练学生去消费它”

已新增对应启动脚本：

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_stage2_freeze_heads_gpu67.sh`

该脚本的输出目录固定为：

- `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67`

该脚本的 W&B run name 固定为：

- `pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67`

如果需要“新开一个 run”而不是复用上面的输出目录，也已经新增独立脚本：

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_stage2_freeze_heads_gpu67_freshrun.sh`

对应的新输出目录：

- `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun`

对应的新 W&B run name：

- `pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun`

Stage2 的观测重点不再只是 `train/loss_total`，而应优先关注：

- `headonly val loss`
- `train/loss_main`
- `train/loss_track_aux`
- `train/loss_box_aux`
- `train/loss_depth_aux`
- `train/object_context_abs_max`
- `train/object_latent_tokens_abs_max`
- `train/grad_norm`
- 以及后续基于 checkpoint 的推理视频质量

### 0.1 2026-06-26 Stage2 首次启动失败的真实原因与修复

第一次直接从 `step-004000` 启动 Stage2 时，训练并没有卡在 forward/backward，而是在恢复 optimizer state 时立即失败。

报错现象：

- `ValueError: loaded state dict contains a parameter group that doesn't match the size of optimizer's group`

根因：

- `step-004000` 是 `head-only` 阶段保存的训练状态
- 当时 optimizer 里的 trainable 参数组对应的是：
  - `object_pooler`
  - `object_aux_heads`
- 但 Stage2 切换了训练策略，当前 trainable 参数组改成了：
  - `object_adapter`
  - Wan DiT 的 object 注入分支
- 因此：
  - checkpoint 中保存的 optimizer state 参数组大小
  - 与当前新建 optimizer 的参数组大小
  不再一致
- 这时硬恢复 optimizer state 是不合法的

修复方案：

- 保留：
  - 模型权重恢复
  - `global_step / epoch_id / batch_in_epoch / RNG` 恢复
- 但在检测到 optimizer 参数组不兼容时：
  - 自动跳过 optimizer state restore
  - 同时跳过 scheduler state restore
  - 打印明确日志说明“这是因为 trainable set 改了”

代码修复位置：

- `train_v_newtrain.py`
- `train_loop(...)` 中 `load_training_state(args.resume_from)` 之后的恢复逻辑

修复后的行为语义：

- “从旧 checkpoint 的模型参数继续开始新阶段训练”
- 而不是“强行沿用旧阶段 optimizer 动量状态”

这正是 Stage2 需要的正确语义，因为当前阶段本来就不是同一个优化问题。

### 0.2 2026-06-26 新开 fresh run：已成功跨过恢复阶段并进入真实训练

为避免复用前一个 Stage2 尝试的输出目录和 W&B 名，额外新开了一条 fresh run：

- 启动脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_stage2_freeze_heads_gpu67_freshrun.sh`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun`
- W&B run name：
  - `pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun`
- W&B run id：
  - `4n1dtaoh`

启动后的关键日志已经确认：

- 成功读取：
  - `step-004000/training_state.pt`
- 成功打印：
  - `Skipping optimizer state restore ...`
  - `Skipping scheduler state restore ...`
- 成功恢复：
  - `global_step=4000`
  - `epoch_id=1`
- 成功进入训练循环：
  - `epoch 1 | global_step 4001`
  - `epoch 1 | global_step 4002`
  - `epoch 1 | global_step 4003`
  - `epoch 1 | global_step 4004`

这说明：

- optimizer 参数组不匹配的问题已经被正确绕过
- 新阶段训练不再在恢复阶段崩掉
- forward / backward / optimizer step 已经真实开始推进

额外观察：

- DDP 打印了：
  - `find_unused_parameters=True was specified ... but did not find any unused parameters`
- 这不是错误，只说明当前这版 Stage2 前向里没有真正出现 unused params
- 后续如果确认这一点稳定成立，可以再考虑把启动脚本中的 `--find_unused_parameters` 去掉，以减少每 step 的额外 autograd 遍历开销

### 0.3 2026-06-26 继续监控：fresh run 持续推进到 `global_step 4051`

对 fresh run `4n1dtaoh` 的继续核对结果：

- 当前训练主进程仍健康存活
- GPU 使用仍固定在：
  - `gpu6`
  - `gpu7`
- 最近运行态可见：
  - 两张卡显存都维持在约 `45.9 GiB / 49.1 GiB`
- W&B 本地 run 目录仍在持续写入：
  - `/home/gaoya/wandb/run-20260626_133342-4n1dtaoh`

从训练 stdout / W&B output.log 已确认：

- `global_step` 已从：
  - `4000`
  稳定推进到：
  - `4051`

这进一步说明：

- 不是“只跨过恢复点就停”
- 而是训练主循环已经连续运行了几十个 optimizer step
- `forward / backward / optimizer` 当前仍在稳定工作

当前还未到下一个新 checkpoint 保存点：

- 当前 run 使用：
  - `save_steps = 500`
- 从 `global_step = 4000` 恢复
- 因此下一个预期新 checkpoint 是：
  - `step-004500`
- 再下一个是：
  - `step-005000`

因此当前阶段的重点不再是“能不能启动”，而是继续盯：

- `step-004500` 是否成功落盘
- head-only val loss 是否在 `500 step` 周期正常记录
- 新 checkpoint 是否可直接用于推理脚本产出视频

## 1. 这版方案的目标

当前目标不是训练主去噪路径，而是把 object 表征本身训练稳定，让它能更可靠地回归：

- `track`
- `box`
- `depth`

主去噪 `loss_main` 在这版 strict run 里设为 `0.0`，所以不会参与回传。真正更新的只是一组 object 相关模块。

## 2. 当前可训练模块

对应当前 strict run，真正可训练的是：

- `object_pooler.jepa_proj`
- `object_pooler.latent_proj`
- `object_pooler.vggt_geom_point_proj`
- `object_pooler.motion_point_proj`
- `object_pooler.motion_router_score`
- `object_pooler.geom_router_score`
- `object_pooler.jepa_router_score`
- `object_pooler.latent_router_score`
- `object_pooler.track_geometry_router_score`
- `object_pooler.appearance_router_score`
- `object_pooler.depth_proj`
- `object_pooler.modal_refine`
- `object_pooler.out_norm`
- `object_aux_heads.track_head`
- `object_aux_heads.box_head`
- `object_aux_heads.depth_head`
- `object_aux_heads.track_gate_logit`
- `object_aux_heads.box_center_gate_logit`
- `object_aux_heads.box_size_gate_logit`

当前不参与训练的是：

- `object_adapter`
- Wan DiT 主干
- Wan DiT 的 `object_embedding / object_cross_attn / object_gate / norm4`
- `object_pooler.world_proj`
- 旧的 `object_pooler.track_geom_proj`
- JEPA / CoTracker / VGGT backbone 本身

## 3. 当前 object branch 的完整流程

下面按一次前向把流程写清楚。帧数统一记为 `T`，其余维度保留真实值。

### 3.1 输入数据

一条样本进入训练时，主要输入是：

- `context_video`: `[1, 3, T, 512, 896]`
- `context_boxes`: `[1, T, N_obj, 4]`
- `context_states`: `[1, T, N_obj, state_dim]`

这里数据集里的真实物体数是 `N_obj`，但训练里最多只保留 `4` 个物体槽位。

### 3.2 构造 query prior

训练会先从 GT box 构造每个物体的 query 点。当前设计是：

- 每个物体 `8` 个点
- 最多 `4` 个物体槽位

因此 query 总数是：

- `O = 4`
- `Q = 8`
- `O * Q = 32`

对应张量大致是：

- `query_points_prior`: `[1, 32, 2]`
- `query_frame_ids`: `[1, 32]`
- `object_valid_mask`: `[1, 4]`
- `box_prior_xyxy`: `[1, 4, 4]`

语义是：

- 一个物体对应 `8` 个点
- 最多保留 `4` 个物体槽位
- 无效槽位后面会被 `object_valid_mask` 屏蔽

### 3.3 CoTracker 提轨迹

CoTracker 先对这 32 个点做跟踪，输出点级轨迹：

- `tracks`: `[1, T, 32, 2]`
- `visibility`: `[1, T, 32]`
- `confidence`: `[1, T, 32]`

再按物体分组后得到：

- `tracks_grouped`: `[1, T, 4, 8, 2]`
- `visibility_grouped`: `[1, T, 4, 8]`
- `confidence_grouped`: `[1, T, 4, 8]`

这一步之后，语义变成：

- 第 3 维是物体槽位 `O=4`
- 第 4 维是每个物体的 `8` 个 query 点

### 3.4 JEPA 特征提取

JEPA 对 context video 提取 patch token，进入 object pooler 的是：

- `jepa_patch_tokens`: `[1, T, H_j, W_j, C_jepa]`

这里 `H_j, W_j, C_jepa` 取决于 JEPA backbone。

object pooler 会用 CoTracker 点在 JEPA 特征图上采样并聚合，最终得到：

- `jepa_latent_tokens`: `[1, T_lat, 4, 4096]`

含义是：

- 时间对齐到 VAE latent 的时间长度 `T_lat`
- 每个 latent 时刻、每个物体槽位都有一个 appearance token

### 3.5 Wan VAE latent 特征

Wan VAE 的 context latent 作为另一条 appearance 路径输入：

- `context_latents`: `[1, 16, T_lat, 64, 112]`

这里：

- `16` 是 latent channel
- `64 x 112` 是空间 latent 网格

同样用 CoTracker 点在 latent grid 上采样并聚合，得到：

- `latent_latent_tokens`: `[1, T_lat, 4, 4096]`

这一路表示生成模型自己的低层时空表征。

### 3.6 Motion expert

从 CoTracker 轨迹构造 motion 属性：

- `x, y`
- `dx, dy`
- `vis`
- `conf`

点级 motion 特征是：

- `motion_local`: `[1, T_lat, 4, 8, 6]`

经过 `motion_point_proj` 之后：

- `motion_point_tokens`: `[1, T_lat, 4, 8, 4096]`

再做点内 attention pooling，得到：

- `motion_latent_tokens`: `[1, T_lat, 4, 4096]`

### 3.7 Geometry expert

这条已经接成了你要的形式：

`CoTracker points -> VGGT feature map -> geometry token`

当前支持两种 VGGT 输入来源：

- 在线模式：训练时直接调用 `VGGTTrackAdapter`
- 离线模式：训练时优先读取 `/data/gaoya/AAA_test_video/0623/train/train0624/vggt_cache` 下的缓存文件

当前这套缓存分片和训练启动都已经避开 `gpu4`：

- shard0: `CUDA_VISIBLE_DEVICES=2`
- shard1: `CUDA_VISIBLE_DEVICES=3`
- shard2: `CUDA_VISIBLE_DEVICES=0`
- shard3: `CUDA_VISIBLE_DEVICES=5`
- 正式训练: `CUDA_VISIBLE_DEVICES=6,7`

后续如果要继续重跑缓存或正式训练，不要再把 `gpu4` 放回启动脚本里。

离线缓存脚本会把每个视频的 VGGT dense patch feature 存成 `*.vggt.pt`，训练时按样本 `video_path` 的文件名自动匹配。

当前状态记录：

- `vggt_cache` 已经全量完成，训练集 1200 个样本的 `w000/w001/w002` 三个窗口文件都已生成
- 正式训练脚本仍使用 `CUDA_VISIBLE_DEVICES=6,7`
- 当前活跃的恢复 run:
  - W&B run id: `3utgz1bh`
  - run name: `pybullet0625_diffsynth_object_heads_only_gpu67`
  - 当前训练命令仍来自：
    - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_heads_only_gpu67.sh`
  - validation / benchmark 固定绑定：
    - `gpu5`
- 当前重启后的训练 run id: `fiqxml81`
- 当前训练进度：
  - 已经稳定跑起来，stdout 最新可见进度到 `global_step 343`
  - 当前已保存 checkpoint：
    - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-000200`
    - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-000400`
    - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-000600`
    - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-000800`
    - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-001000`
  - `interrupted-latest` 目录也在同步维护：
    - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/interrupted-latest`
  - 后续 stdout 最新可见进度已推进到 `global_step 1015`
  - 额外排查结论：
    - `step-000200/400/600/training_state.pt` 的 `global_step` 分别为 `200/400/600`，和 checkpoint 目录一致
    - `interrupted-latest/training_state.pt` 仍停在 `global_step 125`
    - 这是正常现象，不代表当前训练没更新；代码里 `interrupted-latest` 只会在 `KeyboardInterrupt` 或 `TrainingInterrupted` 分支里重写
    - 该文件时间戳是 `2026-06-25 19:46:06 UTC`，对应一次更早的中断态快照，不是当前 run 的实时状态镜像
  - `2026-06-25 20:44 UTC` 左右出现一次真实中断，原因不是训练逻辑错误，而是 checkpoint 落盘时磁盘空间耗尽：
    - 报错位置：创建 `step-001000` 目录时抛出 `OSError: [Errno 28] No space left on device`
    - 当时 `/data` 分区 `Avail=0`
    - 当前 run 的每个 `step-*` checkpoint 大约占 `1.7G`
  - 恢复动作：
    - 删除旧的 `interrupted-latest`、`step-000200`、`step-000400`
    - 释放后 `/data` 可用空间恢复到约 `5.1G`
    - 启动脚本已补上：
      - `--resume_from .../checkpoints/step-000800`
      - `--max_checkpoints_keep 2`
    - 新恢复 run 的 W&B run id: `wy4ru3qv`
    - 恢复日志已确认：
      - `Restored training state: global_step=800`
      - 训练循环重新从 `global_step 801` 开始推进
  - 恢复后已确认成功跨过原先失败点：
      - `step-001000` 已成功落盘
      - `--max_checkpoints_keep 2` 已生效，旧的 `step-000600` 已被自动删除
      - 当前 checkpoint 目录只保留：
        - `step-000800`
        - `step-001000`
      - 训练日志已明确打印：
        - `Pruned old checkpoint: .../step-000600`
  - `2026-06-25 21:05:50 UTC` 再次检查当前恢复 run：
    - 训练进程仍健康存活，实际仍运行在 `CUDA_VISIBLE_DEVICES=6,7`
    - stdout 最新可见进度已推进到 `global_step 1097`
    - W&B `wy4ru3qv` 最近 history 已推进到 `_step=1140`
    - 当前 checkpoint 目录仍只保留两份：
      - `step-000800`
      - `step-001000`
    - `/data` 当前可用空间约 `5.1G`
  - `2026-06-25 21:05 UTC` 的最近 loss / 数值观察：
    - `train/loss_total` 最近几步大致在 `0.015 ~ 0.072`
    - `train/loss_track_aux` 最近几步大致在 `0.026 ~ 0.098`
    - `train/loss_box_aux` 最近几步大致在 `0.105 ~ 0.646`
    - `train/loss_depth_aux` 最近几步大致在 `0.0012 ~ 0.094`
    - `train/object_context_abs_max` 最近稳定在 `0.411 ~ 0.413`
    - `train/object_latent_tokens_abs_max` 最近稳定在 `3.86 ~ 3.90`
    - 目前没有看到 `nan/inf`、也没有看到 `object_context_abs_max` 持续上冲，说明 object 分支数值暂时是稳定的
    - 但 `track_box_loss` 的 batch 间波动仍然比较大，最近样本里可见从 `4.65` 到 `43.08` 的抖动，这更像样本难度/匹配差异，不像整体发散
- 当前训练已显式启用 validation:
  - `validation_every_steps=2000`
  - `validation_script_path=run_validation_vbench.py`
  - 额外发现的下一风险点：
    - `run_validation_vbench.py` 会在每次 validation 时基于 `100` 个 meta case 和 `0,1,2,4,6,8` 六组 context 配置生成完整验证产物
    - 这意味着一次 `step 2000` validation 会落很多视频、JSON 和 VBench 结果文件
    - 当前 `/data` 只剩约 `5.1G`，磁盘风险比训练数值风险更高，下一次真正可能中断训练的点更可能是 validation 落盘而不是 loss 爆炸
    - 当前启动脚本已经补上 `--benchmark_cuda_visible_devices 5`，这样后续重启时 validation / benchmark 会固定走 `gpu5`，避免再和主训练 `gpu6,7` 抢卡；这次正在运行的进程不会自动吃到这个修改，只有重启后才会生效
  - `2026-06-25 21:08:04 UTC` 的进一步检查：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1175`
    - stdout 最新可见进度已推进到 `global_step 1176`
    - 最新 summary：
      - `train/loss_total = 0.02865`
      - `train/loss_track_aux = 0.04427`
      - `train/loss_box_aux = 0.21910`
      - `train/loss_depth_aux = 0.02313`
  - `2026-06-25 21:29 UTC` 的继续监控：
    - 训练主进程仍正常存活在 `gpu6,7`
    - stdout 最新可见进度已经推进到 `global_step 1607`
    - `step-001600` 已成功落盘，当前 checkpoint 目录为：
      - `step-001400`
      - `step-001600`
    - retention 继续正常，日志已打印自动清理：
      - `Pruned old checkpoint: .../step-001200`
    - 当前 W&B `wy4ru3qv` 最新：
      - `lastHistoryStep = 1612`
      - `train/loss_total = 0.10717`
      - `train/loss_track_aux = 0.05543`
      - `train/loss_box_aux = 0.42616`
      - `train/loss_depth_aux = 0.59015`
      - `train/object_context_abs_max = 0.41206`
    - 解释：
      - `object_context_abs_max` 仍维持在约 `0.41` 的稳定带，没有出现上下文 token 数值爆炸
      - `loss_depth_aux` 这一时刻比前一个监控点抬高很多，更像 batch 级抖动，不像整体发散；需要继续盯后续几十到几百 step 是否能自然回落
    - 当前仍没有任何 `validation100_vbench` 目录、`summary.json`、`done.json`、`failed.json` 或 validation stdout/stderr 痕迹，说明 `step 2000` validation 还没有开始
    - `/data` 可用空间仍约 `5.1G`，所以当前首要风险依然不是训练 loss，而是即将到来的 validation 落盘体积
  - `2026-06-25 21:31 UTC` 的追加检查：
    - W&B `wy4ru3qv` 最新 `lastHistoryStep = 1637`
    - 最新 summary：
      - `train/loss_total = 0.03885`
      - `train/loss_track_aux = 0.06725`
      - `train/loss_box_aux = 0.25363`
      - `train/loss_depth_aux = 0.06758`
      - `train/object_context_abs_max = 0.41176`
      - `train/object_latent_tokens_abs_max = 3.77130`
    - 解释：
      - 前一个监控点里 `loss_depth_aux = 0.59015` 没有延续，当前已经回落到 `0.06758`
      - 这更像单个 batch 的难样本抖动，而不是 depth head 或 object token 数值已经进入持续发散
      - `object_context_abs_max` 和 `object_latent_tokens_abs_max` 依旧稳定，没有看到上下文注入幅值异常放大
  - `2026-06-25 21:33 UTC` 的继续监控：
    - stdout 最新可见进度已推进到 `global_step 1703`
    - W&B `wy4ru3qv` 最新 `lastHistoryStep = 1704`
    - 最新 summary：
      - `train/loss_total = 0.04488`
      - `train/loss_track_aux = 0.07046`
      - `train/loss_box_aux = 0.34618`
      - `train/loss_depth_aux = 0.03216`
      - `train/object_context_abs_max = 0.39661`
    - 解释：
      - `loss_depth_aux` 继续保持低位，没有延续前面的尖峰
      - `object_context_abs_max` 进一步回落到 `0.39x`，说明 object context 注入幅值仍处在很安全的范围
      - 到目前为止，训练主循环仍然比 validation 更稳定，当前真正需要警惕的还是 `step 2000` 时的验证磁盘/抢卡风险
  - `2026-06-25 21:34 UTC` 的继续监控：
    - stdout 最新可见进度已推进到 `global_step 1725`
    - W&B `wy4ru3qv` 最新 `lastHistoryStep = 1725`
    - 最新 summary：
      - `train/loss_total = 0.03686`
      - `train/loss_track_aux = 0.09745`
      - `train/loss_box_aux = 0.21276`
      - `train/loss_depth_aux = 0.05843`
      - `train/object_context_abs_max = 0.41659`
      - `train/object_latent_tokens_abs_max = 3.79498`
    - 解释：
      - 这几个 loss 都仍然在前面已经出现过的稳定波动带内，没有新的发散迹象
      - `object_context_abs_max` 虽然从 `0.396` 回到 `0.416`，但仍然处在此前反复观察到的安全带内，不构成异常
      - 当前仍未出现任何 validation 子进程、validation 目录或结果文件，说明还没有碰到真正高风险的 `step 2000` 验证阶段
  - `2026-06-25 21:38 UTC` 的关键节点检查：
    - stdout 最新可见进度已推进到 `global_step 1806`
    - `step-001800` 已成功落盘
    - 当前 checkpoint 目录为：
      - `step-001600`
      - `step-001800`
    - retention 继续正常，日志已打印自动清理：
      - `Pruned old checkpoint: .../step-001400`
    - W&B `wy4ru3qv` 最新 `lastHistoryStep = 1810`
    - 最新 summary：
      - `train/loss_total = 0.05023`
      - `train/loss_track_aux = 0.07867`
      - `train/loss_box_aux = 0.36189`
      - `train/loss_depth_aux = 0.06170`
      - `train/object_context_abs_max = 0.40449`
      - `train/object_latent_tokens_abs_max = 3.87282`
    - 解释：
      - `step-001800` 成功保存说明 cache 读盘、前向、反传、优化器和 checkpoint 链路都还在稳定工作
      - 数值仍然在稳定带，没有出现临近 validation 前的异常抬升
      - 当前依旧没有任何 `validation100_vbench`、`summary.json`、`done.json`、`failed.json`、`stdout.log` 或 `stderr.log`，说明 validation 还没开始
      - `/data` 可用空间仍约 `5.1G`，所以接下来真正高风险点还是 `step 2000` 时 validation 的产物落盘
  - `2026-06-25 21:39 UTC` 的继续监控：
    - stdout 最新可见进度已推进到 `global_step 1834`
    - W&B `wy4ru3qv` 最新 `lastHistoryStep = 1834`
    - 最新 summary：
      - `train/loss_total = 0.04287`
      - `train/loss_track_aux = 0.08389`
      - `train/loss_box_aux = 0.32192`
      - `train/loss_depth_aux = 0.02289`
      - `train/object_context_abs_max = 0.41011`
      - `train/object_latent_tokens_abs_max = 3.78715`
    - 解释：
      - `1834` 这一段依旧没有看到任何持续异常，`loss_depth_aux` 甚至进一步回落
      - `object_context_abs_max` 和 `object_latent_tokens_abs_max` 仍在此前稳定带内，没有出现 validation 前夕的数值失控
      - 到当前时刻仍没有任何 validation 子进程或落盘结果，训练还处在 validation 触发前的最后一个区间
  - `2026-06-25 21:41 UTC` 的继续监控：
    - stdout 最新可见进度已推进到 `global_step 1859`
    - W&B `wy4ru3qv` 最新 `lastHistoryStep = 1859`
    - 最新 summary：
      - `train/loss_total = 0.05397`
      - `train/loss_track_aux = 0.13375`
      - `train/loss_box_aux = 0.36748`
      - `train/loss_depth_aux = 0.03848`
      - `train/object_context_abs_max = 0.39999`
      - `train/object_latent_tokens_abs_max = 3.63531`
    - 解释：
      - `loss_track_aux` 这一刻相对前一轮更高，但 `loss_total / box / depth` 和 object token 幅值仍然稳定，因此更像 batch 间正常抖动
      - `object_context_abs_max` 维持在约 `0.40`，没有出现任何注入幅值异常
      - 当前依旧没有任何 validation 子进程或验证结果文件，说明 `step 2000` 还未到达
  - `2026-06-25 21:42 UTC` 的继续监控：
    - stdout 最新可见进度已推进到 `global_step 1882`
    - W&B `wy4ru3qv` 最新 `lastHistoryStep = 1884`
    - 最新 summary：
      - `train/loss_total = 0.05008`
      - `train/loss_track_aux = 0.07097`
      - `train/loss_box_aux = 0.29921`
      - `train/loss_depth_aux = 0.13066`
      - `train/object_context_abs_max = 0.41830`
      - `train/object_latent_tokens_abs_max = 3.82515`
    - 解释：
      - `loss_depth_aux` 较前一刻抬高，但仍然是单项 loss 的正常区间抖动；`loss_total / track / box` 没有同步失控
      - `object_context_abs_max` 和 `object_latent_tokens_abs_max` 仍然处在此前稳定带，没有看到 object 分支数值开始发散
      - 到当前仍没有任何 validation 子进程或验证结果产物，说明第一次 validation 还没有开始
  - `2026-06-25 21:43 UTC` 的继续监控：
    - stdout 最新可见进度已推进到 `global_step 1908`
    - W&B `wy4ru3qv` 最新 `lastHistoryStep = 1913`
    - 最新 summary：
      - `train/loss_total = 0.05779`
      - `train/loss_track_aux = 0.07014`
      - `train/loss_box_aux = 0.39712`
      - `train/loss_depth_aux = 0.11065`
      - `train/object_context_abs_max = 0.41409`
      - `train/object_latent_tokens_abs_max = 3.82051`
    - 解释：
      - `1900+` 这一段依旧没有出现新的持续异常，所有主监控指标都仍在既有波动带内
      - object 分支的两项幅值指标继续稳定，当前没有看到 validation 触发前夕的数值不稳定征兆
      - 直到当前时刻仍没有任何 validation 子进程、日志或结果文件，说明第一次 validation 还未触发
  - `2026-06-25 21:44 UTC` 的继续监控：
    - stdout 最新可见进度已推进到 `global_step 1931`
    - W&B `wy4ru3qv` 最新 `lastHistoryStep = 1933`
    - 最新 summary：
      - `train/loss_total = 0.04454`
      - `train/loss_track_aux = 0.10019`
      - `train/loss_box_aux = 0.30267`
      - `train/loss_depth_aux = 0.04253`
      - `train/object_context_abs_max = 0.41314`
      - `train/object_latent_tokens_abs_max = 3.81417`
    - 解释：
      - `1930+` 这一段继续没有出现新的持续异常，loss 与 object token 幅值都维持在稳定带
      - 当前依旧没有任何 validation 子进程、验证日志或结果文件，说明第一次 validation 还未真正开始
  - `2026-06-25 21:49 UTC` 的关键节点：
    - `step-002000` 已成功落盘：
      - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-002000`
    - 训练在 `step 2000` 后确实触发了 validation
    - 但 validation 首次运行失败，运行时文件已生成：
      - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/test/_benchmark_runtime/validation100_vbench/step-002000/benchmark.failed.json`
      - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/test/_benchmark_runtime/validation100_vbench/step-002000/benchmark.stdout.log`
      - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/test/_benchmark_runtime/validation100_vbench/step-002000/benchmark.stderr.log`
    - 根因不是磁盘，也不是训练数值发散，而是 validation 脚本的 Python 依赖缺失：
      - `benchmark.stderr.log` 报错：
        - `ModuleNotFoundError: No module named 'skimage'`
    - 已处理：
      - 在 `wan-cu128` 环境里补装了 `scikit-image` 和 `torchmetrics`
      - 之后重新从 `step-002000` 恢复训练，新的活跃 run 切到 `3utgz1bh`
  - `2026-06-25 22:06 UTC` 的当前状态刷新：
    - 当前训练进程仍健康存活，实际运行在 `gpu6,7`
    - `nvidia-smi` 观察：
      - `gpu6` 显存约 `42743 MiB / 49140 MiB`
      - `gpu7` 显存约 `42729 MiB / 49140 MiB`
    - stdout 最新可见进度：
      - `global_step 2267`
    - 当前 checkpoint 目录：
      - `step-002000`
      - `step-002200`
    - checkpoint 内部状态核对：
      - `step-002000/training_state.pt -> global_step=2000`
      - `step-002200/training_state.pt -> global_step=2200`
    - retention 继续正常，stdout 已打印：
      - `Pruned old checkpoint: .../step-001800`
    - 当前 W&B `3utgz1bh` 最新：
      - `lastHistoryStep = 2265`
      - `train/loss_total = 0.04265`
      - `train/loss_track_aux = 0.11697`
      - `train/loss_box_aux = 0.24478`
      - `train/loss_depth_aux = 0.06480`
      - `train/object_context_abs_max = 0.41328`
      - `train/object_latent_tokens_abs_max = 3.93634`
    - 从 W&B 最近 10 个点看，当前主要数值带大致为：
      - `train/loss_total`: `0.025 ~ 0.122`
      - `train/loss_track_aux`: `0.019 ~ 0.128`
      - `train/loss_box_aux`: `0.195 ~ 0.600`
      - `train/loss_depth_aux`: `0.010 ~ 0.572`
      - `train/object_context_abs_max`: `0.395 ~ 0.414`
      - `train/object_latent_tokens_abs_max`: `3.78 ~ 3.95`
    - 解释：
      - `loss_box_aux / loss_depth_aux` 仍然有明显 batch 级尖峰，但没有看到持续单调上升
      - `object_context_abs_max` 继续稳定在 `0.40` 左右，暂时没有 object token 幅值发散迹象
      - 当前更像“样本难度导致的批间波动”，不像训练逻辑错误导致的系统性失稳
    - validation 当前状态：
      - 仍只有旧的 `step-002000` 失败记录：
        - `test/_benchmark_runtime/validation100_vbench/step-002000/benchmark.failed.json`
      - 还没有新的 `step-004000` validation 产物，这是正常的，因为当前全局步数还在 `2200+`
    - 磁盘状态：
      - `/data` 当前可用空间仍约 `5.1G`
      - 这仍然是当前最主要的运行风险

## 4. 当前梯度 / loss 监控口径

当前 run 已经能直接监控这些 loss / 数值量：

- `train/loss_total`
- `train/loss_track_aux`
- `train/loss_box_aux`
- `train/loss_depth_aux`
- `train/object_context_abs_max`
- `train/object_latent_tokens_abs_max`
- `train/track_box_loss`
- `train/track_iou_loss`

另外已在代码里补上了显式梯度监控，位置在：

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_v_newtrain.py`

新增指标：

- `train/grad_norm`
- `train/grad_abs_max`
- `train/grad_param_count`
- `train/grad_elem_count`

说明：

- 这次补丁没有中断当前健康运行的进程
- 因此这几个梯度指标会在“下一次重启 / 恢复训练后”开始出现在 W&B
- 当前这轮 `3utgz1bh` 仍只能通过 loss 带、object token 幅值和 checkpoint 连续产出来间接判断反传是否稳定

## 5. 2026-06-25 22:09 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2320`
- 当前 checkpoint 目录暂时仍是：
  - `step-002000`
  - `step-002200`
- 解释：
  - 这说明训练还处在 `2200 -> 2400` 的推进区间内
  - 下一份预期新 checkpoint 仍然是 `step-002400`
- 当前 validation 运行时目录没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
  - 还没有 `step-004000` 对应的新 validation 结果，这符合当前全局步数
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
  - 依旧是当前最主要的运行风险
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2326`
  - `train/loss_total = 0.02332`
  - `train/loss_track_aux = 0.04544`
  - `train/loss_box_aux = 0.17868`
  - `train/loss_depth_aux = 0.00910`
  - `train/object_context_abs_max = 0.41419`
  - `train/object_latent_tokens_abs_max = 3.94974`
- 最近一段 history 观察：
  - `2016 -> 2326` 区间内依旧能看到 batch 级波动，尤其是：
    - `loss_box_aux` 偶尔冲到 `0.4 ~ 0.7`
    - `loss_depth_aux` 偶尔冲到 `0.1 ~ 0.58`
  - 但这些尖峰没有持续保持，后续又会自然回落
  - `object_context_abs_max` 始终维持在大约 `0.39 ~ 0.42`
  - `object_latent_tokens_abs_max` 目前维持在大约 `3.26 ~ 3.98`
- 当前判断：
  - 训练仍表现为“正常批间抖动”，没有看到持续单调上升、`nan/inf` 或 object context 幅值失控
  - 这一阶段没有新的代码错误，也没有新的 validation 失败
- 关于梯度指标：
  - 当前 W&B 上 `train/grad_norm / grad_abs_max / grad_param_count / grad_elem_count` 仍然是 `None`
  - 这不是补丁失效，而是因为当前活跃 run `3utgz1bh` 是在补丁落地之前启动的
  - 需要下一次真正重启 / resume 到新代码后，这些梯度指标才会开始写入 W&B

## 6. 2026-06-25 22:11 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2347`
- 当前 checkpoint 目录仍然是：
  - `step-002000`
  - `step-002200`
- 解释：
  - 当前仍处在 `2200 -> 2400` 区间内推进
  - 下一份预期 checkpoint 仍是 `step-002400`
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
  - 还没有新的 `step-004000` validation 结果，这是符合当前步数的
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
  - 依旧是最主要的运行风险
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2347`
  - `train/loss_total = 0.03090`
  - `train/loss_track_aux = 0.05627`
  - `train/loss_box_aux = 0.20825`
  - `train/loss_depth_aux = 0.04452`
  - `train/object_context_abs_max = 0.41470`
  - `train/object_latent_tokens_abs_max = 3.97446`
- 最近一小段 history 观察：
  - 能看到个别 batch 出现比较高的 `depth` 尖峰，例如：
    - `_step 2006 -> train/loss_depth_aux = 1.41533`
    - `_step 2119 -> train/loss_depth_aux = 0.65558`
  - 但这些点后续都能快速回落，最近 summary 又回到：
    - `train/loss_depth_aux = 0.04452`
  - `loss_box_aux` 也仍然会有 `0.5+` 的批间抖动，但没有形成持续爬升
  - `object_context_abs_max` 继续稳定在大约 `0.39 ~ 0.415`
  - `object_latent_tokens_abs_max` 继续稳定在大约 `3.3 ~ 3.97`
- 当前判断：
  - 训练仍表现为“正常批间波动”，没有看到持续发散、`nan/inf` 或 object context 幅值失控
  - 单次 `depth` 尖峰目前更像难 batch，而不是 depth head 训练逻辑错误
  - 当前没有新的代码报错，也没有新的 validation 故障
- 关于梯度监控的策略判断：
  - 现在这条 run 是健康的，不建议为了让 `grad_norm` 立刻出现在 W&B 而主动中断重启
  - 更合理的做法是继续让它先产出 `step-002400` 及后续权重
  - 等下一次因为自然原因需要 resume，或者到更关键的阶段再吃到梯度监控补丁

## 7. 2026-06-25 22:12 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2371`
- 当前 checkpoint 目录仍然是：
  - `step-002000`
  - `step-002200`
- 解释：
  - 当前还没跨过 `2400`，因此暂时没有新 checkpoint 是符合预期的
  - 下一份预期 checkpoint 仍是 `step-002400`
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
  - 还没有新的 `step-004000` validation 结果
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2367`
  - `train/loss_total = 0.05523`
  - `train/loss_track_aux = 0.04070`
  - `train/loss_box_aux = 0.47345`
  - `train/loss_depth_aux = 0.03818`
  - `train/object_context_abs_max = 0.40036`
  - `train/object_latent_tokens_abs_max = 4.03600`
- 最近一小段 history 观察：
  - 仍然能看到个别 batch 的 `depth` 大尖峰，例如：
    - `_step 2004 -> train/loss_depth_aux = 3.52395`
  - 也能看到 `box_aux` 的高点，例如：
    - `_step 2099 -> train/loss_box_aux = 0.72523`
  - 但这些尖峰都没有持续保持，后续 summary 又会回落
- 当前判断：
  - `object_latent_tokens_abs_max` 从之前约 `3.95` 继续上浮到 `4.036`
  - 但 `object_context_abs_max` 仍然稳定在约 `0.40`
  - 同时 `loss_total / loss_track_aux / loss_depth_aux` 都没有联动失控
  - 所以目前更像 object latent token 幅值的轻微正常漂移，不像已经进入发散
  - 需要继续盯后续几十到几百 step，确认它是否继续单调上升；如果只是在 `4.0` 左右窄幅波动，暂时不构成异常
- 关于梯度指标：
  - 当前 W&B 上 `train/grad_norm / train/grad_abs_max` 仍然是 `None`
  - 原因不变：当前活跃 run 没有吃到补丁后的训练代码

## 8. 2026-06-25 22:13 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2395`
- 当前 checkpoint 目录仍然是：
  - `step-002000`
  - `step-002200`
- 解释：
  - 当前已经非常接近下一次保存点，但还没真正跨过 `2400`
  - 因此还没有新的 `step-002400` 目录是符合预期的
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2393`
  - `train/loss_total = 0.04717`
  - `train/loss_track_aux = 0.02649`
  - `train/loss_box_aux = 0.40062`
  - `train/loss_depth_aux = 0.04463`
  - `train/object_context_abs_max = 0.40191`
  - `train/object_latent_tokens_abs_max = 4.06538`
- 最近一小段 history 观察：
  - `_step 2377` 可见一次比较高的 batch 波动：
    - `train/loss_total = 0.13166`
    - `train/loss_box_aux = 0.78415`
    - `train/loss_depth_aux = 0.51454`
    - `train/object_latent_tokens_abs_max = 4.05306`
  - 但当前最新 summary 又已经回到：
    - `train/loss_total = 0.04717`
    - `train/loss_depth_aux = 0.04463`
  - 说明这仍然更像单 batch 尖峰，而不是已经形成持续失稳
- 当前判断：
  - `object_latent_tokens_abs_max` 继续从 `4.036` 轻微抬升到 `4.065`
  - 但 `object_context_abs_max` 仍然稳定在约 `0.40`
  - `loss_total` 和 `loss_depth_aux` 也没有同步进入持续升高
  - 因此目前仍判断为轻微正常漂移，需要继续跟踪，但还不构成必须干预的异常

## 9. 2026-06-25 22:14 UTC 关键节点快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2411`
- `step-002400` 已成功落盘
- 当前 checkpoint 目录已更新为：
  - `step-002200`
  - `step-002400`
- retention 继续正常，stdout 已打印：
  - `Pruned old checkpoint: .../step-002000`
- 解释：
  - 这说明从 `2200 -> 2400` 这段训练、反传、优化器更新、checkpoint 保存、旧 checkpoint 清理都仍然工作正常
  - 目前已经拿到新的预期权重文件产出：
    - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-002400`
- validation 当前仍没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
  - 还没到下一次 validation 触发点
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2412`
  - `train/loss_total = 0.07579`
  - `train/loss_track_aux = 0.09080`
  - `train/loss_box_aux = 0.64631`
  - `train/loss_depth_aux = 0.02080`
  - `train/object_context_abs_max = 0.40315`
  - `train/object_latent_tokens_abs_max = 4.04863`
- 当前判断：
  - `step-002400` 正常落盘，说明当前 run 仍然是健康的
  - `box_aux` 在最新点偏高，但 `depth_aux` 和 `object_context_abs_max` 仍然稳定
  - `object_latent_tokens_abs_max` 比前一轮略回落到 `4.0486`，没有继续单调上冲
  - 这进一步支持“当前主要是批间波动，而不是已经开始发散”的判断

## 10. 2026-06-25 22:15 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2430`
- 当前 checkpoint 目录仍为：
  - `step-002200`
  - `step-002400`
- retention 维持正常，没有额外异常
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2428`
  - `train/loss_total = 0.04539`
  - `train/loss_track_aux = 0.10161`
  - `train/loss_box_aux = 0.32251`
  - `train/loss_depth_aux = 0.02975`
  - `train/object_context_abs_max = 0.41261`
  - `train/object_latent_tokens_abs_max = 4.10030`
- 当前判断：
  - `2400+` 这一段依旧没有看到新的训练/保存异常
  - `loss_total / box / depth` 仍处在此前已经出现过的波动带内
  - `object_context_abs_max` 继续稳定在约 `0.41`
  - `object_latent_tokens_abs_max` 继续小幅上浮到 `4.10`
  - 这说明它仍然是当前最值得重点盯的单项指标
  - 但由于：
    - `loss_total` 没有联动抬升
    - `depth_aux` 当前反而较低
    - `object_context_abs_max` 没有同步失控
  - 所以当前还不能判定为发散，更像 latent token 幅值的缓慢漂移
  - 后续需要继续观察它是否会持续单调上升到更明显的异常区间

## 11. 2026-06-25 22:16 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2448`
- 当前 checkpoint 目录仍为：
  - `step-002200`
  - `step-002400`
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2448`
  - `train/loss_total = 0.03749`
  - `train/loss_track_aux = 0.12964`
  - `train/loss_box_aux = 0.21021`
  - `train/loss_depth_aux = 0.03503`
  - `train/object_context_abs_max = 0.41193`
  - `train/object_latent_tokens_abs_max = 4.11219`
- 最近一小段 history 观察：
  - `_step 2444`:
    - `train/loss_total = 0.03853`
    - `train/loss_box_aux = 0.24695`
    - `train/loss_depth_aux = 0.07178`
    - `train/object_latent_tokens_abs_max = 4.07734`
  - 当前最新 summary：
    - `train/loss_total = 0.03749`
    - `train/loss_depth_aux = 0.03503`
    - `train/object_latent_tokens_abs_max = 4.11219`
- 当前判断：
  - `object_latent_tokens_abs_max` 仍在缓慢上浮，目前到 `4.112`
  - 但 `loss_total / box / depth` 仍然没有同步进入异常区间
  - `object_context_abs_max` 继续稳定在约 `0.412`
  - 所以当前仍判断为“需要继续重点观察的慢漂移”，而不是必须立即中断训练的异常

## 12. 2026-06-25 22:17 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2466`
- 当前 checkpoint 目录仍为：
  - `step-002200`
  - `step-002400`
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2464`
  - `train/loss_total = 0.03696`
  - `train/loss_track_aux = 0.10973`
  - `train/loss_box_aux = 0.24001`
  - `train/loss_depth_aux = 0.01987`
  - `train/object_context_abs_max = 0.41209`
  - `train/object_latent_tokens_abs_max = 4.11668`
- 最近一小段 history 观察：
  - `_step 2443`:
    - `train/loss_total = 0.04735`
    - `train/loss_box_aux = 0.43351`
    - `train/loss_depth_aux = 0.01080`
    - `train/object_latent_tokens_abs_max = 4.11532`
  - 当前最新 summary：
    - `train/loss_total = 0.03696`
    - `train/loss_box_aux = 0.24001`
    - `train/loss_depth_aux = 0.01987`
    - `train/object_latent_tokens_abs_max = 4.11668`
- 当前判断：
  - `object_latent_tokens_abs_max` 继续缓慢抬升，目前到 `4.1167`
  - 但 `loss_total / box / depth` 仍然保持在稳定波动带
  - `object_context_abs_max` 继续稳定在约 `0.412`
  - 当前仍更像“缓慢漂移但尚未失稳”，所以继续观察优先于立即干预

## 13. 2026-06-25 22:18 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2483`
- 当前 checkpoint 目录仍为：
  - `step-002200`
  - `step-002400`
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2484`
  - `train/loss_total = 0.01413`
  - `train/loss_track_aux = 0.03867`
  - `train/loss_box_aux = 0.09256`
  - `train/loss_depth_aux = 0.01004`
  - `train/object_context_abs_max = 0.41228`
  - `train/object_latent_tokens_abs_max = 4.12628`
- 最近一小段 history 观察：
  - `_step 2443 -> object_latent_tokens_abs_max = 4.11532`
  - `_step 2481 -> object_latent_tokens_abs_max = 4.09337`
  - 当前最新 summary -> `4.12628`
- 当前判断：
  - `object_latent_tokens_abs_max` 的总体趋势仍然偏上行，但最近尾部并不是严格单调上升
  - 既出现了 `4.093` 的回落，也出现了 `4.126` 的新高点
  - 这说明它更像“高位窄幅波动中的慢漂移”，而不是持续失控式上冲
  - 同时：
    - `loss_total / box / depth` 当前都很低
    - `object_context_abs_max` 继续稳定在约 `0.412`
  - 所以当前仍然不支持立即停机干预，继续密切观察更合理

## 14. 2026-06-25 22:19 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2500`
- 当前 checkpoint 目录仍为：
  - `step-002200`
  - `step-002400`
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2499`
  - `train/loss_total = 0.03856`
  - `train/loss_track_aux = 0.05820`
  - `train/loss_box_aux = 0.24668`
  - `train/loss_depth_aux = 0.08074`
  - `train/object_context_abs_max = 0.41214`
  - `train/object_latent_tokens_abs_max = 4.10124`
- 最近一小段 history 观察：
  - `_step 2463 -> object_latent_tokens_abs_max = 4.12345`
  - `_step 2480 -> object_latent_tokens_abs_max = 4.12128`
  - `_step 2488 -> object_latent_tokens_abs_max = 4.12531`
  - 当前 summary -> `4.10124`
- 当前判断：
  - `object_latent_tokens_abs_max` 仍处在 `4.1x` 的高位区间
  - 但最新 summary 已从前面的 `4.126` 回落到 `4.101`
  - 这进一步支持“高位波动中的慢漂移”判断，而不是持续单调上冲
  - 同时 `loss_total / box / depth` 和 `object_context_abs_max` 仍然稳定
  - 现阶段继续观察仍然比主动打断训练更合理

## 15. 2026-06-25 22:20 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2519`
- 当前 checkpoint 目录仍为：
  - `step-002200`
  - `step-002400`
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2520`
  - `train/loss_total = 0.05652`
  - `train/loss_track_aux = 0.12672`
  - `train/loss_box_aux = 0.33862`
  - `train/loss_depth_aux = 0.09989`
  - `train/object_context_abs_max = 0.41191`
  - `train/object_latent_tokens_abs_max = 4.14044`
- 最近一小段 history 观察：
  - `_step 2463 -> object_latent_tokens_abs_max = 4.12345`
  - `_step 2480 -> object_latent_tokens_abs_max = 4.12128`
  - `_step 2488 -> object_latent_tokens_abs_max = 4.12531`
  - `_step 2509 -> object_latent_tokens_abs_max = 4.11295`
  - 当前 summary -> `4.14044`
- 当前判断：
  - `object_latent_tokens_abs_max` 仍然在 `4.11 ~ 4.14` 的高位区间震荡
  - 最近 history 已出现 `4.113` 的回落，但当前 summary 又回到 `4.140`
  - 这仍然更像“高位震荡中的慢漂移”，而不是一条无回撤的单调上冲曲线
  - 同时 `loss_total / box / depth` 和 `object_context_abs_max` 依旧没有联动进入异常区间
  - 因此当前仍然是继续密切观察，而不是主动停训干预

## 16. 2026-06-25 22:21 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2539`
- 当前 checkpoint 目录仍为：
  - `step-002200`
  - `step-002400`
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2540`
  - `train/loss_total = 0.10690`
  - `train/loss_track_aux = 0.06297`
  - `train/loss_box_aux = 0.42868`
  - `train/loss_depth_aux = 0.57733`
  - `train/object_context_abs_max = 0.41198`
  - `train/object_latent_tokens_abs_max = 4.12946`
- 当前判断：
  - `depth_aux` 在最新点再次抬高到 `0.577`
  - 但历史里同量级的 depth 尖峰之前也出现过，并且后续能够自然回落
  - 同时：
    - `object_context_abs_max` 仍然稳定在约 `0.412`
    - `object_latent_tokens_abs_max` 仍然只是 `4.12x` 高位波动
  - 所以当前仍更像单 batch / 小区间波动，而不是已经进入持续失稳
  - 下一轮重点继续看：
    - `loss_depth_aux` 是否像之前那样自然回落
    - `object_latent_tokens_abs_max` 是否继续维持高位震荡而不是明显上冲

## 17. 2026-06-25 22:22 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2556`
- 当前 checkpoint 目录仍为：
  - `step-002200`
  - `step-002400`
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2555`
  - `train/loss_total = 0.01498`
  - `train/loss_track_aux = 0.02258`
  - `train/loss_box_aux = 0.10053`
  - `train/loss_depth_aux = 0.02671`
  - `train/object_context_abs_max = 0.41240`
  - `train/object_latent_tokens_abs_max = 4.12342`
- 当前判断：
  - 上一轮抬高到 `0.577` 的 `loss_depth_aux` 已经回落到 `0.0267`
  - 这进一步说明前面的 depth 抬高更像短时尖峰，而不是持续失稳
  - `object_latent_tokens_abs_max` 当前仍在 `4.12x` 区间，但没有明显突破前面已观察到的高位带
  - `object_context_abs_max` 继续稳定在约 `0.412`
  - 因此当前整体判断仍然是：
    - 训练健康
    - `object_latent_tokens_abs_max` 需要继续重点盯
    - 但现阶段还不支持主动打断训练

## 18. 2026-06-25 22:23 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2574`
- 当前 checkpoint 目录仍为：
  - `step-002200`
  - `step-002400`
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2570`
  - `train/loss_total = 0.05660`
  - `train/loss_track_aux = 0.03805`
  - `train/loss_box_aux = 0.52307`
  - `train/loss_depth_aux = 0.00492`
  - `train/object_context_abs_max = 0.40367`
  - `train/object_latent_tokens_abs_max = 4.15914`
- 当前判断：
  - `loss_depth_aux` 已再次回落到很低，说明最近一轮并没有延续 depth 尖峰
  - `object_context_abs_max` 也维持在安全带内
  - 但 `object_latent_tokens_abs_max` 最新到了 `4.159`
  - 这已经略高于此前主要观察到的 `4.10 ~ 4.14` 区间
  - 目前仍然没有和 `loss_total / depth / context_abs_max` 一起联动失控，所以还不能直接判为发散
  - 但需要提高警惕，后续重点看：
    - 它是否继续上冲到更高区间
    - 是否开始伴随 `loss_total` 或 `object_context_abs_max` 同步抬升

## 19. 2026-06-25 22:24 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2593`
- 当前 checkpoint 目录仍为：
  - `step-002200`
  - `step-002400`
- 解释：
  - 当前已经非常接近 `step-002600`
  - 还没看到新的 `step-002600` 目录，说明还差最后一小段
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2590`
  - `train/loss_total = 0.04694`
  - `train/loss_track_aux = 0.05788`
  - `train/loss_box_aux = 0.39293`
  - `train/loss_depth_aux = 0.01859`
  - `train/object_context_abs_max = 0.41115`
  - `train/object_latent_tokens_abs_max = 4.18935`
- 最近一小段 history 观察：
  - `_step 2520 -> object_latent_tokens_abs_max = 4.14044`
  - `_step 2570 -> object_latent_tokens_abs_max = 4.15914`
  - `_step 2571 -> object_latent_tokens_abs_max = 4.18149`
  - 当前 summary -> `4.18935`
- 当前判断：
  - `object_latent_tokens_abs_max` 这次确实又抬到了新的局部高点，且已经进入 `4.18 ~ 4.19` 区间
  - 但与此同时：
    - `loss_total` 仍然不高
    - `loss_depth_aux` 当前较低
    - `object_context_abs_max` 仍然稳定在约 `0.411`
  - 所以当前还不能直接把它判成“已经发散”
  - 但警戒级别需要再提高一档
  - 下一轮如果继续出现下面任一情况，就需要开始考虑干预而不只是观察：
    - `object_latent_tokens_abs_max` 继续明显上冲，并站上更高平台
    - `object_context_abs_max` 同步抬升
    - `loss_total / loss_depth_aux` 不再回落而是开始持续偏高

## 20. 2026-06-25 22:25 UTC 关键节点快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2613`
- `step-002600` 已成功落盘
- 当前 checkpoint 目录已更新为：
  - `step-002400`
  - `step-002600`
- retention 继续正常，stdout 已打印：
  - `Pruned old checkpoint: .../step-002200`
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2610`
  - `train/loss_total = 0.05851`
  - `train/loss_track_aux = 0.02752`
  - `train/loss_box_aux = 0.54890`
  - `train/loss_depth_aux = 0.00873`
  - `train/object_context_abs_max = 0.40511`
  - `train/object_latent_tokens_abs_max = 4.20417`
- 最近一小段 history 观察：
  - `_step 2529 -> object_latent_tokens_abs_max = 4.12783`
  - `_step 2531 -> object_latent_tokens_abs_max = 4.14606`
  - `_step 2542 -> object_latent_tokens_abs_max = 4.16122`
  - `_step 2604 -> object_latent_tokens_abs_max = 4.21973`
- 当前判断：
  - `step-002600` 正常落盘，说明训练、反传、保存、清理链路仍然工作正常
  - `object_latent_tokens_abs_max` 继续抬升，并已经进入 `4.20+` 区间
  - 但当前：
    - `loss_total` 仍处在可接受波动带
    - `loss_depth_aux` 当前较低
    - `object_context_abs_max` 反而在 `0.405` 左右，没有同步走高
  - 因此现在仍不能直接判成“已经发散”
  - 但这已经是比前几轮更强的预警信号，后续如果继续上冲且伴随其他指标联动恶化，就需要从“继续观察”切换到“主动干预”

## 21. 2026-06-25 22:26 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2632`
- 当前 checkpoint 目录仍为：
  - `step-002400`
  - `step-002600`
- retention 继续正常，没有新的保存链路异常
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2631`
  - `train/loss_total = 0.04405`
  - `train/loss_track_aux = 0.08350`
  - `train/loss_box_aux = 0.24808`
  - `train/loss_depth_aux = 0.10889`
  - `train/object_context_abs_max = 0.41174`
  - `train/object_latent_tokens_abs_max = 4.23473`
- 最近一小段 history 观察：
  - `_step 2604 -> object_latent_tokens_abs_max = 4.21973`
  - `_step 2616 -> object_latent_tokens_abs_max = 4.22385`
  - 当前 summary -> `4.23473`
- 当前判断：
  - `object_latent_tokens_abs_max` 继续缓慢上冲，已经到 `4.23+`
  - 但 `loss_total` 仍然不高，`object_context_abs_max` 也仍稳定在 `0.41` 左右
  - `loss_depth_aux` 在最新点有一定抬升，但还没有形成持续高位滞留
  - 这说明当前更像“更强的预警态”，但还没有充分证据证明已经数值发散
  - 后续如果 `object_latent_tokens_abs_max` 继续上冲，同时 `loss_total / loss_depth_aux / object_context_abs_max` 也开始联动上行，就需要转入主动干预

## 22. 2026-06-25 22:27 UTC 继续监控快照

- 当前训练主进程仍健康存活，实际运行在 `gpu6,7`
- stdout 最新可见进度：
  - `global_step 2651`
- 当前 checkpoint 目录仍为：
  - `step-002400`
  - `step-002600`
- validation 当前没有新增产物：
  - 仍只有旧的 `step-002000` 失败记录
- `/data` 磁盘状态：
  - 可用空间仍约 `5.1G`
- 当前 W&B `3utgz1bh` 最新：
  - `lastHistoryStep = 2652`
  - `train/loss_total = 0.05296`
  - `train/loss_track_aux = 0.09878`
  - `train/loss_box_aux = 0.32387`
  - `train/loss_depth_aux = 0.10694`
  - `train/object_context_abs_max = 0.40776`
  - `train/object_latent_tokens_abs_max = 4.26839`
- 最近一小段 history 观察：
  - `_step 2604 -> object_latent_tokens_abs_max = 4.21973`
  - `_step 2616 -> object_latent_tokens_abs_max = 4.22385`
  - 当前 summary -> `4.26839`
- 当前判断：
  - `object_latent_tokens_abs_max` 继续上冲，已经进入 `4.26+` 区间
  - 但当前还没有看到其他关键指标同步恶化：
    - `object_context_abs_max` 仍在约 `0.408`
    - `loss_total` 仍在此前可接受波动带
    - `loss_depth_aux` 虽有抬升，但还没有形成持续高位平台
  - 所以现在仍然更像“预警进一步增强”，但还没有足够证据证明已经失稳
  - 下一轮如果继续出现：
    - `object_latent_tokens_abs_max` 再明显上冲
    - 且 `loss_total / loss_depth_aux / object_context_abs_max` 开始联动上行
  - 就需要从观察切换到主动干预
        - `ModuleNotFoundError: No module named 'skimage'`
      - 失败位置：
        - `train0419_reference/run_validation_vbench.py`
        - `from skimage.metrics import peak_signal_noise_ratio, structural_similarity`
    - 影响：
      - 训练主进程在 `Validation failed at step 2000` 后退出，需要手动恢复
      - 但 `step-002000` checkpoint 已经完整保存，所以可以直接从 `step-002000` 继续
    - 修复动作：
      - 在 `wan-cu128` 环境内安装缺失依赖：
        - `scikit-image`
        - `torchmetrics`
      - 安装后已验证 import 正常：
        - `skimage_ok 0.25.2`
        - `LearnedPerceptualImagePatchSimilarity` 可导入
      - 启动脚本 `run_train_v_newtrain_object_heads_only_gpu67.sh` 已改成：
        - 自动扫描 `output/checkpoints/step-*`
        - 默认从最新 step 恢复，而不是硬编码回 `step-000800`
    - 恢复状态：
      - 已用修复后的环境和脚本重新启动训练
      - 新恢复 run 的 W&B run id:
        - `3utgz1bh`
      - W&B 链接：
        - `https://wandb.ai/875222004-gy/vjepa_vggt_wan/runs/3utgz1bh`
      - 启动日志已确认：
        - `Resuming from latest checkpoint: .../step-002000`
        - `Loading training state from: .../step-002000/training_state.pt`
      - 当前恢复后的主训练进程已重新存活在 `gpu6,7`
      - 这次恢复后的训练命令也已经显式带上：
        - `--benchmark_cuda_visible_devices 5`
      - 这意味着下一次 validation 若再触发，将不再和主训练 `gpu6,7` 抢卡
  - `2026-06-25 21:54 UTC` 的恢复后首个监控点：
    - 新恢复 run `3utgz1bh` 已经成功继续推进
    - stdout 最新可见进度：
      - `global_step 2006`
    - W&B `3utgz1bh` 最新：
      - `lastHistoryStep = 2003`
      - `train/loss_total = 0.08386`
      - `train/loss_track_aux = 0.08909`
      - `train/loss_box_aux = 0.35208`
      - `train/loss_depth_aux = 0.39740`
      - `train/object_context_abs_max = 0.37182`
      - `train/object_latent_tokens_abs_max = 4.42124`
    - 解释：
      - 这说明从 `step-002000` 恢复后的前几个 step 已经真实跑起来，不是只停留在加载状态
      - `object_context_abs_max` 仍然稳定，没有看到 object context 注入爆炸
      - `object_latent_tokens_abs_max` 比恢复前抬高到 `4.42`，需要后续继续盯是否只是恢复初期 batch 抖动，还是会持续抬升
      - 当前 checkpoint 目录仍保留：
        - `step-001800`
        - `step-002000`
      - `/data` 可用空间仍约 `5.1G`
  - `2026-06-25 21:55 UTC` 的追加检查：
    - 恢复后的训练已经继续推进到：
      - `global_step 2032`
    - W&B `3utgz1bh` 最新：
      - `lastHistoryStep = 2029`
      - `train/loss_total = 0.05646`
      - `train/loss_track_aux = 0.03546`
      - `train/loss_box_aux = 0.52126`
      - `train/loss_depth_aux = 0.00792`
      - `train/object_context_abs_max = 0.38552`
      - `train/object_latent_tokens_abs_max = 3.39228`
    - 解释：
      - `object_latent_tokens_abs_max` 已经从恢复初期观察到的 `4.42` 回落到 `3.39`
      - 这说明之前更像恢复初期 batch 抖动，而不是 object latent token 持续上冲
      - 当前恢复后的训练数值整体仍然稳定，可以继续往下观察下一个 checkpoint 和下一次 validation
  - `2026-06-25 21:56 UTC` 的继续监控：
    - 恢复后的训练已进一步推进到：
      - `global_step 2058`
    - W&B `3utgz1bh` 最新：
      - `lastHistoryStep = 2055`
      - `train/loss_total = 0.03951`
      - `train/loss_track_aux = 0.02438`
      - `train/loss_box_aux = 0.34254`
      - `train/loss_depth_aux = 0.02815`
      - `train/object_context_abs_max = 0.39419`
      - `train/object_latent_tokens_abs_max = 3.52210`
    - 解释：
      - 恢复后的 run 继续健康推进，当前没有新的报错或中断迹象
      - 几个 loss 均回到较平稳区间
      - `object_latent_tokens_abs_max` 维持在 `3.5` 左右，没有再次出现恢复初期的短时抬升
  - `2026-06-25 21:58 UTC` 的继续监控：
    - 恢复后的训练已进一步推进到：
      - `global_step 2083`
    - W&B `3utgz1bh` 最新：
      - `lastHistoryStep = 2081`
      - `train/loss_total = 0.06633`
      - `train/loss_track_aux = 0.04713`
      - `train/loss_box_aux = 0.55621`
      - `train/loss_depth_aux = 0.05993`
      - `train/object_context_abs_max = 0.39355`
      - `train/object_latent_tokens_abs_max = 3.56113`
    - 解释：
      - 当前恢复后的 run 仍在稳定推进，没有新的错误或 validation 相关失败
      - 虽然尚未到新的 `save_steps=2200`，所以 checkpoint 目录还只有 `step-001800` 和 `step-002000`
      - 但从 loss 和 token 幅值来看，目前没有看到坏趋势
  - `2026-06-25 21:59 UTC` 的继续监控：
    - 恢复后的训练已进一步推进到：
      - `global_step 2102`
    - W&B `3utgz1bh` 最新：
      - `lastHistoryStep = 2102`
      - `train/loss_total = 0.03979`
      - `train/loss_track_aux = 0.12495`
      - `train/loss_box_aux = 0.22231`
      - `train/loss_depth_aux = 0.05068`
      - `train/object_context_abs_max = 0.41690`
      - `train/object_latent_tokens_abs_max = 3.60660`
    - 解释：
      - 当前 run 继续稳定推进，没有新的中断或 validation 相关错误
      - `object_context_abs_max` 和 `object_latent_tokens_abs_max` 仍然处在可接受的稳定带
      - 虽然 `loss_track_aux` 这一刻相对前一轮更高，但 `loss_total / box / depth` 没有同步恶化，更像 batch 间正常抖动
  - `2026-06-25 22:00 UTC` 的继续监控：
    - 恢复后的训练已进一步推进到：
      - `global_step 2125`
    - W&B `3utgz1bh` 最新：
      - `lastHistoryStep = 2122`
      - `train/loss_total = 0.06605`
      - `train/loss_track_aux = 0.17756`
      - `train/loss_box_aux = 0.36729`
      - `train/loss_depth_aux = 0.11568`
      - `train/object_context_abs_max = 0.41455`
      - `train/object_latent_tokens_abs_max = 3.64284`
    - 解释：
      - 当前 run 继续健康推进，还未到新的 `save_steps=2200`，所以 checkpoint 目录未变化是正常现象
      - `loss_track_aux` 和 `loss_depth_aux` 这一刻相对更高，但其余监控项没有同步恶化，更像 batch 级正常抖动
      - `object_context_abs_max` / `object_latent_tokens_abs_max` 仍在稳定带，没有看到数值开始失控
  - `2026-06-25 22:01 UTC` 的继续监控：
    - 恢复后的训练已进一步推进到：
      - `global_step 2146`
    - W&B `3utgz1bh` 最新：
      - `lastHistoryStep = 2143`
      - `train/loss_total = 0.06488`
      - `train/loss_track_aux = 0.05052`
      - `train/loss_box_aux = 0.58448`
      - `train/loss_depth_aux = 0.01379`
      - `train/object_context_abs_max = 0.39534`
      - `train/object_latent_tokens_abs_max = 3.81924`
    - 解释：
      - 当前 run 继续稳定推进，仍未到新的 `save_steps=2200`
      - `loss_box_aux` 这一刻偏高，但 `loss_total / track / depth` 与两项幅值指标没有同步异常，更像 batch 间正常抖动
      - 目前仍没有新的 validation 相关错误或主训练中断
  - `2026-06-25 22:02 UTC` 的继续监控：
    - 恢复后的训练已进一步推进到：
      - `global_step 2168`
    - W&B `3utgz1bh` 最新：
      - `lastHistoryStep = 2169`
      - `train/loss_total = 0.04679`
      - `train/loss_track_aux = 0.06847`
      - `train/loss_box_aux = 0.33845`
      - `train/loss_depth_aux = 0.06097`
      - `train/object_context_abs_max = 0.41283`
      - `train/object_latent_tokens_abs_max = 3.89896`
    - 解释：
      - 当前 run 继续健康推进，还未到新的 `save_steps=2200`
      - 各项 loss 和 token 幅值仍然在稳定带，没有看到新的持续恶化趋势
      - 目前仍没有新的 validation 相关错误或主训练中断
  - `2026-06-25 22:03 UTC` 的继续监控：
    - 恢复后的训练已进一步推进到：
      - `global_step 2190`
    - W&B `3utgz1bh` 最新：
      - `lastHistoryStep = 2189`
      - `train/loss_total = 0.04668`
      - `train/loss_track_aux = 0.07689`
      - `train/loss_box_aux = 0.36723`
      - `train/loss_depth_aux = 0.02265`
      - `train/object_context_abs_max = 0.41282`
      - `train/object_latent_tokens_abs_max = 3.92190`
    - 解释：
      - 当前 run 继续稳定推进，离新的 `save_steps=2200` 只差很近
      - 数值仍在稳定带，没有新的持续恶化趋势
      - 当前目录里仍只有上一次失败 validation `step-002000` 的运行时产物，没有出现新的 validation 相关错误
  - `2026-06-25 22:04 UTC` 的关键节点：
    - `step-002200` 已成功落盘：
      - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-002200`
    - retention 继续正常，日志已打印自动清理：
      - `Pruned old checkpoint: .../step-001800`
    - 当前 checkpoint 目录更新为：
      - `step-002000`
      - `step-002200`
    - W&B `3utgz1bh` 最新：
      - `lastHistoryStep = 2209`
      - `train/loss_total = 0.04664`
      - `train/loss_track_aux = 0.10271`
      - `train/loss_box_aux = 0.31990`
      - `train/loss_depth_aux = 0.04382`
      - `train/object_context_abs_max = 0.41262`
      - `train/object_latent_tokens_abs_max = 3.93003`
    - 解释：
      - 这说明从 `step-002000` 恢复后的训练不但主循环稳定，checkpoint 保存链路也已经再次验证通过
      - 当前几项 loss 和两项 token 幅值指标仍处在稳定带，没有看到恢复后逐步恶化的趋势
      - 当前没有新的 validation 相关目录或失败记录出现；下一次 validation 风险点将顺延到后续新的 `validation_every_steps` 触发点
      - `train/object_context_abs_max = 0.41124`
    - 到目前为止仍无 validation 产物生成，说明 `step 2000` 还没触发
    - 当前 checkpoint 目录仍然只有：
      - `step-000800`
      - `step-001000`
    - `/data` 依然只有约 `5.1G` 可用，validation 磁盘风险仍是第一优先级
  - 已追加一个针对 validation 磁盘风险的代码保护：
    - 修改文件：
      - `train0419_reference/run_validation_vbench.py`
    - 修改内容：
      - 每个 context 配置跑完生成、future-GT metrics、VBench 后，立即删除该 context 对应的 `generation_output_root`
      - 保留 `runtime_root` 下的 `summary.json`、GT metrics、manifest、VBench eval json 等轻量结果
    - 作用：
      - 避免 validation 把 6 组 context 的大体积生成视频都长期留在 `/data`
      - 下一次需要重启训练时，这个补丁会直接降低 `step 2000` validation 再次打满磁盘的概率
    - 限制：
      - `run_validation_vbench.py` 是训练过程中后续启动的外部子进程，因此当前这条训练在 `step 2000` 触发 validation 时，会直接用到这版新的“评估后删视频输出”逻辑
      - 但 `benchmark_cuda_visible_devices` 属于训练主进程启动参数，当前这条已经运行中的训练进程命令行里没有该覆盖项，所以它未来触发 validation 时仍会沿用旧默认值 `5,6,7`
      - 也就是说：
        - validation 清理补丁：当前 run 会生效
        - validation 固定只走 `gpu5`：只有后续重启后的 run 才会生效
  - `2026-06-25 21:09:24 UTC` 的最新检查：
    - stdout 最新可见进度已推进到 `global_step 1211`
    - 已成功产出新 checkpoint：
      - `step-001200`
    - retention 继续正常：
      - 旧的 `step-000800` 已被自动删除
      - 当前 checkpoint 目录只保留：
        - `step-001000`
        - `step-001200`
    - `/data` 可用空间仍约 `5.1G`，说明 checkpoint 保留策略暂时稳住了磁盘
    - 最近 12 个采样点观察：
      - `train/object_context_abs_max` 一直稳定在 `0.407 ~ 0.417`
      - `train/loss_depth_aux` 确实偶发性出现较高 batch（例如 `_step=1021` 的 `0.563`、`_step=1058` 的 `0.624`、`_step=1208` summary 的 `0.514`）
      - 但相邻很多 step 又会回落到 `0.008 ~ 0.063`
      - 因此目前更像是数据 batch 差异引起的尖峰，而不是持续性抬升或整体发散
      - `train/track_box_loss` 同样仍有明显 batch 级波动，最大可见到 `46.93`，但并未带动 `object_context_abs_max` 一起失控
  - `2026-06-25 21:10 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1238`
    - stdout 最新可见进度已推进到 `global_step 1243`
    - 当前 summary：
      - `train/loss_total = 0.04740`
      - `train/loss_track_aux = 0.05075`
      - `train/loss_box_aux = 0.40233`
      - `train/loss_depth_aux = 0.02088`
      - `train/object_context_abs_max = 0.41266`
    - 当前 checkpoint 目录仍只保留两份：
      - `step-001000`
      - `step-001200`
    - retention 仍正常，没有再出现 checkpoint 落盘失败
    - 当前还没有任何 `validation100_vbench` 相关目录、`summary.json`、`failed.json` 或 `done.json` 产物，说明 validation 尚未触发，当前离 `step 2000` 还有约 `750+` step
    - `/data` 依旧约 `5.1G` 可用，因此当前第一风险仍然是未来 validation 阶段的落盘和显存竞争，而不是训练主循环本身
  - `2026-06-25 21:11 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1258`
    - stdout 最新可见进度已推进到 `global_step 1258`
    - 最新 summary：
      - `train/loss_total = 0.05063`
      - `train/loss_track_aux = 0.02788`
      - `train/loss_box_aux = 0.44242`
      - `train/loss_depth_aux = 0.03595`
      - `train/object_context_abs_max = 0.41172`
    - 当前 checkpoint 目录仍只保留：
      - `step-001000`
      - `step-001200`
    - 到目前为止仍然没有任何 `validation100_vbench` 目录、`summary.json`、`done.json`、`failed.json` 或 validation stdout/stderr 产物，说明 validation 还完全没有开始
    - 最近抽样的 history 继续支持“loss 有 batch 波动，但整体稳定”的判断：
      - `train/object_context_abs_max` 稳定在 `0.409 ~ 0.415`
      - `train/loss_depth_aux` 仍会偶发冲高到 `0.51 ~ 0.59`
      - `train/loss_box_aux` 也会偶发冲高到 `0.77`
      - 但这些尖峰后续都会回落，没有形成单调抬升趋势
  - `2026-06-25 21:12 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1274`
    - stdout 最新可见进度已推进到 `global_step 1275`
    - 最新 summary：
      - `train/loss_total = 0.03274`
      - `train/loss_track_aux = 0.04565`
      - `train/loss_box_aux = 0.22687`
      - `train/loss_depth_aux = 0.05484`
      - `train/object_context_abs_max = 0.41062`
    - 当前仍未产出 `step-001400`
    - 当前 checkpoint 目录仍只保留：
      - `step-001000`
      - `step-001200`
    - retention 继续正常，磁盘可用空间仍约 `5.1G`
    - 到当前时刻仍没有任何 validation 相关目录或日志文件出现，说明 validation 还完全没有触发
  - `2026-06-25 21:13 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1289`
    - stdout 最新可见进度已推进到 `global_step 1290`
    - 最新 summary：
      - `train/loss_total = 0.04732`
      - `train/loss_track_aux = 0.01677`
      - `train/loss_box_aux = 0.43463`
      - `train/loss_depth_aux = 0.02184`
      - `train/object_context_abs_max = 0.40546`
    - 当前仍未产出 `step-001400`
    - 当前 checkpoint 目录仍只保留：
      - `step-001000`
      - `step-001200`
    - retention 继续正常，磁盘可用空间仍约 `5.1G`
    - 当前依旧没有任何 validation 相关目录、`summary.json`、`done.json`、`failed.json` 或 stdout/stderr 日志出现，说明 validation 还完全没有开始
  - `2026-06-25 21:14 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1304`
    - stdout 最新可见进度已推进到 `global_step 1306`
    - 最新 summary：
      - `train/loss_total = 0.04200`
      - `train/loss_track_aux = 0.00957`
      - `train/loss_box_aux = 0.38641`
      - `train/loss_depth_aux = 0.02400`
      - `train/object_context_abs_max = 0.40440`
    - 当前仍未产出 `step-001400`
    - 当前 checkpoint 目录仍只保留：
      - `step-001000`
      - `step-001200`
    - retention 继续正常，磁盘可用空间仍约 `5.1G`
    - 当前依旧没有任何 validation 相关目录、`summary.json`、`done.json`、`failed.json` 或 stdout/stderr 日志出现，说明 validation 还完全没有开始
  - `2026-06-25 21:14:36 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1320`
    - stdout 最新可见进度已推进到 `global_step 1323`
    - 最新 summary：
      - `train/loss_total = 0.05494`
      - `train/loss_track_aux = 0.12612`
      - `train/loss_box_aux = 0.33420`
      - `train/loss_depth_aux = 0.08905`
      - `train/object_context_abs_max = 0.41097`
    - 当前仍未产出 `step-001400`
    - 当前 checkpoint 目录仍只保留：
      - `step-001000`
      - `step-001200`
    - retention 继续正常，磁盘可用空间仍约 `5.1G`
    - 当前依旧没有任何 validation 相关目录、`summary.json`、`done.json`、`failed.json` 或 stdout/stderr 日志出现，说明 validation 还完全没有开始
  - `2026-06-25 21:15:24 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1335`
    - stdout 最新可见进度已推进到 `global_step 1339`
    - 最新 summary：
      - `train/loss_total = 0.02269`
      - `train/loss_track_aux = 0.04523`
      - `train/loss_box_aux = 0.16488`
      - `train/loss_depth_aux = 0.01681`
      - `train/object_context_abs_max = 0.40845`
    - 当前仍未产出 `step-001400`
    - 当前 checkpoint 目录仍只保留：
      - `step-001000`
      - `step-001200`
    - retention 继续正常，磁盘可用空间仍约 `5.1G`
    - 当前依旧没有任何 validation 相关目录、`summary.json`、`done.json`、`failed.json` 或 stdout/stderr 日志出现，说明 validation 还完全没有开始
  - `2026-06-25 21:16:09 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1350`
    - stdout 最新可见进度已推进到 `global_step 1354`
    - 最新 summary：
      - `train/loss_total = 0.04340`
      - `train/loss_track_aux = 0.02557`
      - `train/loss_box_aux = 0.37059`
      - `train/loss_depth_aux = 0.03781`
      - `train/object_context_abs_max = 0.41266`
    - 当前仍未产出 `step-001400`
    - 当前 checkpoint 目录仍只保留：
      - `step-001000`
      - `step-001200`
    - retention 继续正常，磁盘可用空间仍约 `5.1G`
    - 当前依旧没有任何 validation 相关目录、`summary.json`、`done.json`、`failed.json` 或 stdout/stderr 日志出现，说明 validation 还完全没有开始
  - `2026-06-25 21:17:02 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1371`
    - stdout 最新可见进度已推进到 `global_step 1372`
    - 最新 summary：
      - `train/loss_total = 0.04626`
      - `train/loss_track_aux = 0.09137`
      - `train/loss_box_aux = 0.35330`
      - `train/loss_depth_aux = 0.01797`
      - `train/object_context_abs_max = 0.41322`
    - 当前仍未产出 `step-001400`
    - 当前 checkpoint 目录仍只保留：
      - `step-001000`
      - `step-001200`
    - retention 继续正常，磁盘可用空间仍约 `5.1G`
    - 当前依旧没有任何 validation 相关目录、`summary.json`、`done.json`、`failed.json` 或 stdout/stderr 日志出现，说明 validation 还完全没有开始
  - `2026-06-25 21:17:52 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1386`
    - stdout 最新可见进度已推进到 `global_step 1389`
    - 最新 summary：
      - `train/loss_total = 0.02134`
      - `train/loss_track_aux = 0.05567`
      - `train/loss_box_aux = 0.12520`
      - `train/loss_depth_aux = 0.03254`
      - `train/object_context_abs_max = 0.41088`
    - 当前仍未产出 `step-001400`
    - 当前 checkpoint 目录仍只保留：
      - `step-001000`
      - `step-001200`
    - retention 继续正常，磁盘可用空间仍约 `5.1G`
    - 当前依旧没有任何 validation 相关目录、`summary.json`、`done.json`、`failed.json` 或 stdout/stderr 日志出现，说明 validation 还完全没有开始
  - `2026-06-25 21:18:40 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1405`
    - stdout 最新可见进度已推进到 `global_step 1405`
    - 已成功产出新 checkpoint：
      - `step-001400`
    - retention 继续正常：
      - 旧的 `step-001000` 已被自动删除
      - 当前 checkpoint 目录只保留：
        - `step-001200`
        - `step-001400`
    - 最新 summary：
      - `train/loss_total = 0.04874`
      - `train/loss_track_aux = 0.04236`
      - `train/loss_box_aux = 0.42213`
      - `train/loss_depth_aux = 0.02291`
      - `train/object_context_abs_max = 0.41806`
    - 虽然这一步 `object_context_abs_max` 比前几次略高，但仍处在之前长期观测到的稳定波动带内，当前还看不出发散趋势
    - `/data` 可用空间仍约 `5.1G`
    - 到当前时刻依旧没有任何 validation 相关目录、`summary.json`、`done.json`、`failed.json` 或 stdout/stderr 日志出现，说明 validation 还没有开始
  - `2026-06-25 21:21:15 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1456`
    - stdout 最新可见进度已推进到 `global_step 1458`
    - 当前 checkpoint 目录仍只保留：
      - `step-001200`
      - `step-001400`
    - retention 继续正常，没有新的落盘失败
    - 最新 summary：
      - `train/loss_total = 0.04252`
      - `train/loss_track_aux = 0.10424`
      - `train/loss_box_aux = 0.31365`
      - `train/loss_depth_aux = 0.00727`
      - `train/object_context_abs_max = 0.41569`
    - 当前 `track_aux` 略高，但 `object_context_abs_max` 仍处于已观测到的稳定区间内，暂未看到数值发散证据
    - `/data` 可用空间仍约 `5.1G`
    - 到当前时刻依旧没有任何 validation 相关目录、`summary.json`、`done.json`、`failed.json` 或 stdout/stderr 日志出现，说明 validation 还没有开始
  - `2026-06-25 21:22:17 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1477`
    - stdout 最新可见进度已推进到 `global_step 1478`
    - 当前 checkpoint 目录仍只保留：
      - `step-001200`
      - `step-001400`
    - retention 继续正常，没有新的落盘失败
    - 最新 summary：
      - `train/loss_total = 0.04615`
      - `train/loss_track_aux = 0.07926`
      - `train/loss_box_aux = 0.31722`
      - `train/loss_depth_aux = 0.06502`
      - `train/object_context_abs_max = 0.41147`
    - 当前 `track_aux`、`depth_aux` 有正常 batch 级波动，但 `object_context_abs_max` 仍处于已观测到的稳定区间内，暂未看到数值发散证据
    - `/data` 可用空间仍约 `5.1G`
    - 到当前时刻依旧没有任何 validation 相关目录、`summary.json`、`done.json`、`failed.json` 或 stdout/stderr 日志出现，说明 validation 还没有开始
  - `2026-06-25 21:23:11 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1497`
    - stdout 最新可见进度已推进到 `global_step 1497`
    - 当前 checkpoint 目录仍只保留：
      - `step-001200`
      - `step-001400`
    - retention 继续正常，没有新的落盘失败
    - 最新 summary：
      - `train/loss_total = 0.06025`
      - `train/loss_track_aux = 0.02880`
      - `train/loss_box_aux = 0.48087`
      - `train/loss_depth_aux = 0.09281`
      - `train/object_context_abs_max = 0.40973`
    - 当前 `box_aux`、`depth_aux` 有正常 batch 级波动，但 `object_context_abs_max` 仍处于已观测到的稳定区间内，暂未看到数值发散证据
    - `/data` 可用空间仍约 `5.1G`
    - 到当前时刻依旧没有任何 validation 相关目录、`summary.json`、`done.json`、`failed.json` 或 stdout/stderr 日志出现，说明 validation 还没有开始
  - `2026-06-25 21:24:03 UTC` 的继续跟踪：
    - 当前恢复 run 仍健康，W&B `wy4ru3qv` 最新 `lastHistoryStep=1512`
    - stdout 最新可见进度已推进到 `global_step 1514`
    - 当前 checkpoint 目录仍只保留：
      - `step-001200`
      - `step-001400`
    - retention 继续正常，没有新的落盘失败
    - 最新 summary：
      - `train/loss_total = 0.04943`
      - `train/loss_track_aux = 0.10120`
      - `train/loss_box_aux = 0.31647`
      - `train/loss_depth_aux = 0.07665`
      - `train/object_context_abs_max = 0.41137`
    - 当前 `track_aux`、`depth_aux` 有正常 batch 级波动，但 `object_context_abs_max` 仍处于已观测到的稳定区间内，暂未看到数值发散证据
    - `/data` 可用空间仍约 `5.1G`
    - 到当前时刻依旧没有任何 validation 相关目录、`summary.json`、`done.json`、`failed.json` 或 stdout/stderr 日志出现，说明 validation 还没有开始
- 如果后续训练报错，优先排查：
  - 缓存是否缺文件
  - `vggt_input_hw` 是否和缓存生成时一致
  - object branch / loss 是否存在非 finite 值
  - 离线 cache 读取时，几何坐标系现在优先使用 cache payload 中的 `input_hw`
  - 正式训练脚本已补上 `validation_every_steps=2000`，会定期跑 validation + VBench

这次把 cache 改成 `fp16`，同时把 VGGT 输入从 `420x728` 下调到 `280x504`，原因是原始 dense cache 体积过大，`/data` 分区在全量缓存训练集时会写满并在 `torch.save` 处失败。这个改法不改变 object branch 的读取方式，只是把离线几何特征压缩到能完整落盘的规模。

当前启动脚本 `run_train_v_newtrain_object_heads_only_gpu67.sh` 已经默认加上：

- `--vggt_cache_root /data/gaoya/AAA_test_video/0623/train/train0624/vggt_cache`

因此只要缓存文件存在，就会优先走离线特征，不再在线跑 VGGT。

VGGT 侧输出：

- `dense_patch_tokens`: `[1, T, 20, 36, 2048]`
- `depth`: 对应原图或等价几何网格上的 depth 特征
- `patch_grid_hw = (20, 36)`

object pooler 把 CoTracker 点映射到 VGGT 特征图上，采样得到：

- `geom_local`: `[1, T_lat, 4, 8, 2048]`
- `depth_local`: `[1, T_lat, 4, 8, 1]`
- `motion_local_lat`: `[1, T_lat, 4, 8, 6]`

拼接后：

- `geom_point_features`: `[1, T_lat, 4, 8, 2055]`

这里 `2055 = 2048 + 1 + 6`。

再过 `vggt_geom_point_proj` 和点内 pooling，得到：

- `geom_latent_tokens`: `[1, T_lat, 4, 4096]`

### 3.8 两级融合

现在不是旧的 4 路 MoE 了，而是两级融合：

第一层：

- `track_geometry = fuse(motion_latent_tokens, geom_latent_tokens)`
- shape: `[1, T_lat, 4, 4096]`

第二层：

- `appearance = fuse(jepa_latent_tokens, latent_latent_tokens)`
- shape: `[1, T_lat, 4, 4096]`

最终层：

- `object_latent_tokens = fuse(track_geometry, appearance)`
- shape: `[1, T_lat, 4, 4096]`

时间维再做平均后得到：

- `object_tokens`: `[1, 4, 4096]`

这是每个物体槽位的最终 object token 摘要。

### 3.9 辅助几何基座

除了 `object_latent_tokens`，pooler 还会输出两个几何基座：

1. `active_track_summary`

- shape: `[1, T_lat, 4, 6]`
- 内容是：
  - `center_x, center_y`
  - `delta_x, delta_y`
  - `mean_vis, mean_conf`

2. `active_box_xyxy`

- shape: `[1, T_lat, 4, 4]`
- 这是 box head 的 base anchor
- 优先来自 `box_prior_xyxy`
- 否则从轨迹点包一个框

## 4. object_aux_heads 怎么算

对应 [`object_aux_heads.py`](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/models/object_aux_heads.py)。

输入是：

- `object_latent_tokens`: `[1, T_lat, 4, 4096]`
- `active_track_summary`: `[1, T_lat, 4, 6]`
- `active_box_xyxy`: `[1, T_lat, 4, 4]`

输出是：

1. `pred_track_summary`

- shape: `[1, T_lat, 4, 4]`
- 计算方式：
  - `track_base = active_track_summary[..., :4]`
  - `pred_track_summary = track_base + gated_residual`

2. `pred_box_xyxy`

- shape: `[1, T_lat, 4, 4]`
- 计算方式：
  - 直接以 `active_box_xyxy` 作为 anchor
  - head 只学 center residual 和 size residual

3. `pred_depth`

- shape: `[1, T_lat, 4, 1]`

## 5. 当前 loss 设计

当前 strict run 里：

- `lambda_main = 0.0`
- 主去噪 loss 不参与训练

真正回传的是：

1. `loss_track_aux`

- 用 `pred_track_summary` 和 GT track summary 做 L1
- 主要由：
  - center L1
  - delta L1
 组成

2. `loss_box_aux`

- 用 `pred_box_xyxy` 和 GT box 做 box loss

3. `loss_depth_aux`

- 用 `pred_depth` 和 `context_states` 中选定的 depth target 做 L1

另外还有两个正则项：

- `track_anchor_reg`
- `box_anchor_reg`

是否参与回传取决于它们对应的 lambda。

## 6. 梯度路径

当前 strict run 的梯度路径是：

`track/box/depth GT`
-> `loss_track_aux / loss_box_aux / loss_depth_aux`
-> `object_aux_heads`
-> `object_latent_tokens`
-> `object_pooler`
-> 停止

不会继续回传到：

- `object_adapter`
- Wan DiT object branch
- 主去噪路径

也就是说，这版训练的目标非常明确：

- 先把 object 表征学稳
- 让它能稳定回归 track / box / depth
- 不让主生成路径掺进来干扰这条监督链

## 7. 现在这个方案和旧版的区别

相比旧版，这版收掉了几件事：

- 不再做 4 路大融合
- 不再把 `world_points` 接进主链路
- 不再让旧的 `track_geom_proj` 参与真正计算
- 不训练 `object_adapter`
- 不训练 Wan DiT object branch

保留下来的核心是：

- CoTracker 提供点级运动信息
- VGGT 提供 dense geometry 特征
- JEPA 和 Wan latent 提供 appearance 特征
- 最终用两级 gated fusion 得到 object token

## 8. 一句话总结

当前这版 object branch 的核心设计可以概括成：

`CoTracker 定位点 -> 在 JEPA / Wan latent / VGGT 上取特征 -> 先融合运动和几何，再融合外观 -> 用 fused object token 回归 track / box / depth`

这就是当前 `object_heads_only strict` 方案的完整口径。

## 9. 2026-06-25 22:29 UTC 运行监控快照

当前 goal 对应的运行状态再次核实如下：

- VGGT cache 已经完整存在：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/vggt_cache`
  - 当前缓存文件数约 `3601`
- 正式训练仍然只使用：
  - `gpu6,7`
- validation / benchmark 仍固定预留：
  - `gpu5`
- `gpu4` 仍未参与任何当前训练 / 验证 / cache 任务

当前活跃进程：

- launcher:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_heads_only_gpu67.sh`
- train script:
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_v_newtrain.py`
- 当前恢复点：
  - `--resume_from .../checkpoints/step-002000`

当前 W&B 与 checkpoint 状态：

- W&B run id:
  - `3utgz1bh`
- W&B `lastHistoryStep`:
  - `2709`
- 当前本地 checkpoint 仍只保留两份：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-002400`
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-002600`
- 说明：
  - 训练已经跨过 `2600`，但 `2800` 还未落盘
  - 当前训练没有停住，只是还没到下一次 `save_steps=200`

本轮最新 loss / 数值快照：

- `train/loss_total = 0.04153`
- `train/loss_track_aux = 0.08851`
- `train/loss_box_aux = 0.24840`
- `train/loss_depth_aux = 0.07842`
- `train/object_context_abs_max = 0.40997`
- `train/object_latent_tokens_abs_max = 4.36217`

当前判断：

- `loss_total / track_aux / box_aux / depth_aux` 仍在正常 batch 抖动范围内
- `object_context_abs_max` 仍稳定在 `~0.41`
- 目前还没有看到 `nan/inf`
- 目前最需要持续盯住的仍然是：
  - `train/object_latent_tokens_abs_max`
- 该值此前在 `4.26` 左右，本次又升到 `4.36`
- 但因为：
  - `loss_total` 没有持续同步抬高
  - `object_context_abs_max` 没有跟着飙升
  - `loss_depth_aux / loss_box_aux` 仍会回落
- 所以当前更像“latent token 幅值缓慢上漂”，还不能直接判定为发散

梯度监控现状：

- `train_v_newtrain.py` 已经补入梯度统计：
  - `train/grad_norm`
  - `train/grad_abs_max`
  - `train/grad_param_count`
  - `train/grad_elem_count`
- 但当前正在运行的 `3utgz1bh` 是在补丁生效前启动的
- 因此这次 run 的 W&B summary 里这四项仍是 `None`
- 结论：
  - 代码已经支持梯度监控
  - 但只有下一次真正重启 / resume 后，W&B 才会开始出现这些梯度项
- 当前不为拿到梯度项而主动中断训练，因为现阶段 run 仍健康

validation 现状：

- 目前 benchmark runtime 下仍只有旧的失败产物：
  - `.../validation100_vbench/step-002000/benchmark.failed.json`
  - `.../validation100_vbench/step-002000/benchmark.stdout.log`
  - `.../validation100_vbench/step-002000/benchmark.stderr.log`
- 暂未出现新的 validation 目录
- 需要等后续触发下一轮 validation 时继续核实

磁盘风险：

- `/data` 当前剩余空间仍只有约 `5.1G`
- 这是当前比 loss 更现实的运行风险
- 在 `max_checkpoints_keep=2` 已启用的前提下，checkpoint 自身还可继续转动
- 但一旦 validation 产生较多视频 / 评测文件，仍可能再次撞上磁盘上限

## 10. 2026-06-25 22:35 UTC 继续监控结论

这轮继续监控后，已经确认训练继续向前推进，并且新的 checkpoint 正常产出：

- 新 checkpoint 已成功写出：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-002800`
- 当前 checkpoint 轮转仍正常：
  - 日志已出现：
    - `Pruned old checkpoint: .../step-002400`
- 因此当前保留集按预期继续向前滚动

对应的运行状态快照：

- W&B run id:
  - `3utgz1bh`
- W&B `lastHistoryStep`:
  - `2805`
- stdout 最新可见进度：
  - `global_step 2808`

本轮最新 loss / 数值：

- `train/loss_total = 0.06491`
- `train/loss_track_aux = 0.18023`
- `train/loss_box_aux = 0.35337`
- `train/loss_depth_aux = 0.11555`
- `train/object_context_abs_max = 0.41068`
- `train/object_latent_tokens_abs_max = 4.44181`

当前判断更新：

- `step-002800` 能成功落盘，说明训练主循环、优化器步进、checkpoint 保存链路都还正常
- `loss_total / track / box / depth` 这轮相比前一轮有回升，但仍属于单 batch 波动区间，暂时还不像持续发散
- `object_context_abs_max` 仍然稳定在 `~0.41`
- 但 `object_latent_tokens_abs_max` 又从上一轮的 `4.37` 左右抬到了 `4.44`

因此当前最值得警惕的现象仍然没有变化：

- object latent token 幅值在缓慢持续上漂

暂时仍不立即判定为数值发散，原因是：

- `object_context_abs_max` 没同步变坏
- `loss_total` 没有进入持续高位不回落的状态
- checkpoint 继续稳定产出

后续需要重点观察的升级信号：

- `train/object_latent_tokens_abs_max` 继续明显超过 `4.44`
- 同时伴随：
  - `train/object_context_abs_max` 脱离 `0.40 ~ 0.41`
  - `train/loss_total` 连续维持高位
  - `train/loss_box_aux` / `train/loss_depth_aux` 不再回落

validation 现状本轮无变化：

- 仍只有旧的：
  - `step-002000/benchmark.failed.json`
- 尚未出现新的 validation 目录或新的 val 损失产物

## 11. 2026-06-25 22:39 UTC 继续监控结论

这一轮继续监控时，训练仍在正常前进，但还没有跨过下一次 `save_steps=200` 的保存点：

- W&B `lastHistoryStep`:
  - `2862`
- 当前还未到 `3000`，因此 checkpoint 目录仍保持：
  - `step-002600`
  - `step-002800`

本轮最新指标：

- `train/loss_total = 0.04915`
- `train/loss_track_aux = 0.11531`
- `train/loss_box_aux = 0.33599`
- `train/loss_depth_aux = 0.04015`
- `train/object_context_abs_max = 0.41083`
- `train/object_latent_tokens_abs_max = 4.52872`

本轮判断：

- `loss_total / track / box / depth` 没有出现连续恶化
- `object_context_abs_max` 依然稳定在 `~0.41`
- 训练也没有停住，仍在向 `3000` 推进

但需要更明确地记录一个趋势：

- `object_latent_tokens_abs_max` 从此前的 `4.44` 左右继续升到了 `4.53`

因此当前的结论仍然是：

- 训练暂时没有出现必须立刻停训修代码的硬错误
- 但 object latent token 幅值持续上漂的风险正在累积

目前还不直接介入调整，原因仍是：

- 其他 loss 没同步进入持续坏状态
- `object_context_abs_max` 没出现联动放大
- checkpoint 产出链路仍然健康

validation / 磁盘现状本轮仍无变化：

- validation 目录仍只有旧的 `step-002000` 失败产物
- `/data` 剩余空间仍约 `5.1G`

## 12. 2026-06-25 22:45 UTC 继续监控结论

这一轮继续确认到，训练没有在 `2860~2900` 区间卡住，而是在持续正常推进：

- W&B `lastHistoryStep`:
  - `2898`
- stdout 最新可见进度：
  - `global_step 2901`

因此当前可以明确排除一种担心：

- 不是“W&B 在涨但本地训练已经卡死”

当前仍未到下一次 `save_steps=200` 的保存点，所以 checkpoint 目录暂时仍是：

- `step-002600`
- `step-002800`

本轮最新指标：

- `train/loss_total = 0.04438`
- `train/object_context_abs_max = 0.41137`
- `train/object_latent_tokens_abs_max = 4.55944`

本轮判断：

- `loss_total` 依旧没有显示出持续恶化
- `object_context_abs_max` 仍基本稳定在 `0.41` 附近
- 训练循环本身仍健康

但需要继续强调的风险趋势是：

- `object_latent_tokens_abs_max` 又从 `4.53` 左右升到了 `4.56`

所以截至这一时刻的判断仍是：

- 还没有出现必须马上停训修代码的硬性错误
- 但 latent token 幅值上漂仍是当前最可疑、最需要持续追踪的数值迹象

后续若继续上漂，并开始伴随：

- `object_context_abs_max` 继续上冲
- `loss_total` 不再回落
- `loss_box_aux / loss_depth_aux` 连续维持高位

则需要从“观察”切换到“介入”，优先检查：

- object pooler 输出缩放
- router / gate 饱和
- aux head 残差尺度

## 13. 2026-06-25 22:49 UTC 继续监控结论

这一轮继续监控时，训练已经进一步推进到 `2975~2979`，仍未出现停滞：

- W&B `lastHistoryStep`:
  - `2975`
- stdout 最新可见进度：
  - `global_step 2979`

当前仍未跨过下一次 `save_steps=200` 的边界，因此 `step-003000` 还没有落盘，checkpoint 目录仍是：

- `step-002600`
- `step-002800`

本轮最新指标：

- `train/loss_total = 0.01407`
- `train/loss_track_aux = 0.03316`
- `train/loss_box_aux = 0.10122`
- `train/loss_depth_aux = 0.00628`
- `train/object_context_abs_max = 0.41416`
- `train/object_latent_tokens_abs_max = 4.57401`

本轮判断：

- 从 loss 看，这一小段反而明显回落了
- 说明当前并不是“所有 loss 一起失控”
- 训练主循环和优化步骤仍是健康的

但数值风险上有两个值得继续盯的点：

- `object_latent_tokens_abs_max` 继续缓慢升到 `4.57`
- `object_context_abs_max` 也从之前更常见的 `0.410~0.411` 抬到了 `0.414`

这两个变化现在还不足以单独判定发散，但需要提高警惕，因为它开始表现出：

- latent token 幅值继续上漂
- context 幅值也出现轻微同步上抬

当前仍不立即介入的原因：

- `loss_total / track / box / depth` 这轮都没有持续恶化，反而回落
- checkpoint 保存链路仍正常
- validation 目录仍无新变化，还是只有旧的 `step-002000` 失败产物

## 14. 2026-06-25 22:52 UTC 继续监控结论

这一轮已经确认新的权重文件继续正常产出：

- 新 checkpoint 已成功落盘：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-003000`
- stdout 最新可见进度：
  - `global_step 3006`
- W&B `lastHistoryStep`:
  - `3005`

并且 checkpoint 轮转继续按预期工作：

- 日志已出现：
  - `Pruned old checkpoint: .../step-002600`

因此当前保留的 checkpoint 应前滚为最新两份。

本轮最新指标：

- `train/loss_total = 0.04727`
- `train/object_context_abs_max = 0.41335`
- `train/object_latent_tokens_abs_max = 4.60655`

本轮判断：

- 训练主循环、优化器步进、checkpoint 保存链路都继续正常
- `step-003000` 的成功落盘说明当前 run 已达到新的稳定权重产出节点
- `loss_total` 仍没有显示出持续恶化

当前最需要持续追踪的风险仍然不变，而且比前几轮更明确：

- `object_latent_tokens_abs_max` 已继续上漂到 `4.61`

但之所以当前仍不立即停训介入，是因为：

- `loss_total` 没同步失控
- `object_context_abs_max` 虽有轻微上抬，但还没有明显脱离当前带宽
- 新 checkpoint 仍能稳定产出

截至本轮，validation 仍无新产物：

- `validation100_vbench` 目录下仍只有旧的 `step-002000` 失败记录

## 15. 2026-06-25 22:58 UTC 继续监控结论

这一轮继续监控时，训练仍然在健康推进：

- W&B `lastHistoryStep`:
  - `3051`
- stdout 最新可见进度：
  - `global_step 3033`
- 当前 checkpoint 目录仍是最新两份：
  - `step-002800`
  - `step-003000`

本轮最新 summary 指标：

- `train/loss_total = 0.04661`
- `train/loss_track_aux = 0.08068`
- `train/loss_box_aux = 0.35854`
- `train/loss_depth_aux = 0.02687`
- `train/object_context_abs_max = 0.41419`
- `train/object_latent_tokens_abs_max = 4.67446`

本轮判断：

- 训练主循环仍然健康，`gpu6,7` 上的两张卡依旧在稳定工作
- `loss_total / track / box / depth` 仍没有表现出持续失控
- 但 `object_latent_tokens_abs_max` 已经继续抬到 `4.67`
- `object_context_abs_max` 也维持在 `0.414` 左右，比更早期的 `0.410~0.411` 略高

这说明当前风险画像已经更清楚：

- 还不是“loss 爆炸型”故障
- 更像是 object latent 幅值缓慢上漂，而其他损失暂时还能压住

### 当前监控链路的一个额外发现

本轮额外排查了 W&B API 的 history 读取状态，结论是：

- `run.summary` 可以正常读取，`lastHistoryStep` 也在更新
- 但 `run.scan_history()` 当前始终返回 `0 rows`
- 无论是否指定 `keys=['_step']` 或 loss 相关 key，结果都一样

这意味着当前 run 的在线监控不能依赖 W&B history API 回放曲线，而要优先依赖：

- W&B summary 最新值
- 本地 `output.log` 中的 `global_step`
- checkpoint 是否持续产出

补充证据：

- `wandb` 本地 `debug-internal.log` 显示 filestream 仍在持续上传 `history_lines`
- 所以更像是：
  - 服务器侧 / API 侧 history 回放暂时不可用
  - 而不是本地根本没在发日志

因此后续如果要继续盯数值趋势，当前最稳妥的做法是：

- 用 summary 连续抽样
- 配合 checkpoint 落盘节奏
- 不把 `scan_history()` 为空误判成训练没在记录

## 16. 2026-06-25 23:03 UTC 继续监控结论

这一轮继续监控时，训练已经继续推进到 `3101~3102`：

- W&B `lastHistoryStep`:
  - `3102`
- stdout 最新可见进度：
  - `global_step 3101`

当前仍未到下一个 `save_steps=200` 的保存边界，因此 checkpoint 目录没有变化仍是正常现象：

- `step-002800`
- `step-003000`

validation / 磁盘本轮也没有变化：

- `validation100_vbench` 目录仍然只有旧的 `step-002000` 失败记录
- `/data` 仍只有约 `5.1G` 可用空间

本轮最新指标：

- `train/loss_total = 0.02787`
- `train/loss_track_aux = 0.02132`
- `train/loss_box_aux = 0.25359`
- `train/loss_depth_aux = 0.00383`
- `train/object_context_abs_max = 0.41669`
- `train/object_latent_tokens_abs_max = 4.70212`

本轮判断：

- loss 端依旧没有出现同步恶化，反而整体仍偏低
- 训练主循环还是健康的

但 object 数值的趋势进一步清楚了：

- `object_latent_tokens_abs_max` 已经继续抬到 `4.70`
- `object_context_abs_max` 也升到 `0.4167`

因此当前状态更接近：

- 不是“马上要炸”的训练
- 而是 object latent / context 幅值在缓慢上漂，但 loss 还暂时能压住

这意味着后续需要更严格观察的联动条件是：

- `object_latent_tokens_abs_max` 继续超过 `4.70`
- `object_context_abs_max` 继续往 `0.42+` 漂
- 同时 `loss_total / box_aux / depth_aux` 开始不再回落

只有当这些现象开始联动时，才值得从“继续观察”切到“直接介入调尺度 / 调结构”。

## 17. 2026-06-25 23:08 UTC 继续监控结论

这一轮补充监控时，训练仍在继续健康推进，没有停住：

- W&B `lastHistoryStep`:
  - `3127`
- stdout 最新可见进度：
  - `global_step 3129`

checkpoint 目录本轮仍未变化，这仍然是正常的，因为还没走到 `3200`：

- `step-002800`
- `step-003000`

本轮最新指标：

- `train/loss_total = 0.06109`
- `train/loss_track_aux = 0.02897`
- `train/loss_box_aux = 0.57528`
- `train/loss_depth_aux = 0.00667`
- `train/object_context_abs_max = 0.41693`
- `train/object_latent_tokens_abs_max = 4.75317`

本轮判断：

- `loss_total` 仍未进入持续失控状态
- `track_aux / depth_aux` 仍然较低
- 但 `box_aux` 这一轮出现了更高的一次抬升

结合 object 数值一起看，当前风险已经更接近“需要准备介入”的门槛：

- `object_latent_tokens_abs_max` 已经继续升到 `4.75`
- `object_context_abs_max` 维持在 `0.4169`
- `loss_box_aux` 出现了一次更明显的高点

不过截至这一时刻，仍然还不能直接下结论说训练已经开始失稳，因为：

- `loss_total` 没有持续维持高位
- `track_aux / depth_aux` 没同步变坏
- 训练主循环与 checkpoint 产出节奏仍然健康

当前更准确的判断是：

- 训练仍可继续观察
- 但已经开始接近“如果再继续上漂，就该从观察转为介入”的区间

如果后续出现以下任一组合，建议立即介入排查 object 分支尺度问题：

- `object_latent_tokens_abs_max >= 4.8` 且继续升
- `object_context_abs_max >= 0.42`
- `loss_box_aux` 连续几次维持在当前高位附近，不再回落

## 18. 2026-06-25 23:15 UTC 继续监控结论

这一轮已经确认新的 checkpoint 继续正常产出：

- 新 checkpoint 已成功落盘：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-003200`
- W&B `lastHistoryStep`:
  - `3231`
- stdout 最新可见进度：
  - `global_step 3231`

对应地，checkpoint 保留集继续按预期轮转为最新两份：

- `step-003000`
- `step-003200`

本轮最新指标：

- `train/loss_total = 0.02433`
- `train/loss_track_aux = 0.05131`
- `train/loss_box_aux = 0.17075`
- `train/loss_depth_aux = 0.02119`
- `train/object_context_abs_max = 0.41921`
- `train/object_latent_tokens_abs_max = 4.77119`

本轮判断：

- `step-003200` 能成功落盘，说明训练主循环、优化器步进、checkpoint 保存链路都仍然健康
- 相比上一轮：
  - `loss_box_aux` 已明显回落
  - `object_context_abs_max` 也从刚过 `0.42` 回落到 `0.4192`
- 因此上一轮 `object_context_abs_max >= 0.42` 更像一次短时抖动，而不是已经稳定跨过阈值

### 对当前阈值解释的一个修正

本轮顺着代码链路重新核实后，需要明确一件事：

- 当前这条 `object_heads_only` 训练 run 里：
  - `--lambda_main 0.0`
  - 没有启用 `--train_object_adapter`
  - 也没有启用 `--train_object_dit_branch`
  - `lambda_object_context_reg` 默认也是 `0.0`

这意味着：

- `object_context_abs_max` 目前更多是一个旁路诊断量
- 它不是当前 object-heads-only 训练里直接受主损失强约束的核心链路
- 真正直接承接梯度、决定当前训练稳定性的主量，还是：
  - `object_latent_tokens`
  - `track_aux / box_aux / depth_aux`

所以后续解释要更准确：

- `object_context_abs_max` 可以继续看，但不能单独作为“必须立刻介入”的硬触发器
- 真正更值得优先盯的是：
  - `object_latent_tokens_abs_max`
  - `loss_box_aux`
  - 以及它们是否持续联动恶化

截至这一轮，新的整体判断是：

- 训练仍保持健康
- object latent 幅值仍在慢慢上漂，当前到 `4.77`
- 但因为 `box_aux` 和 `object_context_abs_max` 这轮都回落了，所以还不需要立刻改代码打断当前 run

## 19. 2026-06-25 23:22 UTC 越阈值复核

这一轮继续监控时，出现了第一个需要认真记录的越阈值现象：

- W&B `lastHistoryStep`:
  - `3281`
- stdout 最新可见进度：
  - `global_step 3285`

本轮最新 summary：

- `train/loss_total = 0.10817`
- `train/loss_track_aux = 0.07718`
- `train/loss_box_aux = 0.42478`
- `train/loss_depth_aux = 0.57977`
- `train/object_context_abs_max = 0.41965`
- `train/object_latent_tokens_abs_max = 4.87601`

这里最关键的是：

- `object_latent_tokens_abs_max` 已经第一次明确超过了之前设的 `4.8` 观察阈值
- 同时 `loss_depth_aux` 也出现了一次明显高点

为了判断这是不是单次尖峰，我又做了一次短时复核：

- 复核时 W&B `lastHistoryStep`:
  - `3310`
- 复核最新 summary：
  - `train/loss_total = 0.03848`
  - `train/loss_track_aux = 0.11870`
  - `train/loss_box_aux = 0.21257`
  - `train/loss_depth_aux = 0.05349`
  - `train/object_context_abs_max = 0.41915`
  - `train/object_latent_tokens_abs_max = 5.01381`

复核结论：

- `loss_depth_aux` 已经明显回落
- `loss_total / box_aux` 也没有保持上一轮高位
- 但 `object_latent_tokens_abs_max` 没有回落，反而继续升到了 `5.01`

这说明当前现象不是“全面 loss 爆炸”，而更像：

- object latent 表示幅值本身持续上漂
- loss 端目前还在波动中，有时能压回去

## 20. 2026-06-25 23:24 UTC 参数对比排查

为了判断是不是某个 object 模块参数本身在迅速放大，我直接对比了：

- `step-003000/checkpoint.safetensors`
- `step-003200/checkpoint.safetensors`

排查结论：

- 没有看到某个 `object_pooler / object_aux_heads` 线性层权重出现明显爆炸式增长
- 大部分参数的 `abs_mean` 变化都非常小，基本在稳定微调范围
- 最明显的漂移反而来自几个 gate / logit：
  - `object_aux_heads.box_size_gate_logit`
  - `object_aux_heads.track_gate_logit`
  - `object_aux_heads.box_center_gate_logit`

其中大致趋势是：

- `box_size_gate_logit` 略升
- `track_gate_logit` 略升
- `box_center_gate_logit` 略降

同时：

- `object_pooler.out_norm.weight`
- `object_aux_heads.{track,box,depth}_head.net.0.weight`
- 各 router / proj 层

都没有呈现“某一层参数绝对值突然暴涨”的模式。

因此当前更可能的解释是：

- 不是某个单层权重数值爆炸
- 更像 object 表示在训练中逐步学到更大的激活尺度
- gate / residual 尺度的缓慢漂移，可能在把 latent token 幅值往上带

截至这一轮的判断更新为：

- 当前 run 还没有必要立刻中断
- 但已经从“普通观察”升级到“带定位的重点观察”
- 后续如果 `object_latent_tokens_abs_max` 在接下来若干次抽样里继续稳步高于 `5.0`，并开始再次带动 `box_aux` / `depth_aux` 反复高位，则需要准备实际介入

## 21. 2026-06-25 23:29 UTC 重点观察结论

这一轮已经确认新的 checkpoint 继续正常产出：

- 新 checkpoint 已成功落盘：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-003400`
- W&B `lastHistoryStep`:
  - `3400`
- stdout 最新可见进度：
  - `global_step 3402`

对应的 checkpoint 轮转继续正常：

- 当前保留集应前滚为：
  - `step-003200`
  - `step-003400`

本轮最新指标：

- `train/loss_total = 0.05692`
- `train/loss_track_aux = 0.12784`
- `train/loss_box_aux = 0.34703`
- `train/loss_depth_aux = 0.09431`
- `train/object_context_abs_max = 0.41971`
- `train/object_latent_tokens_abs_max = 4.98673`

本轮最重要的结论不是单个值，而是趋势：

- 前一轮短时复核时，`object_latent_tokens_abs_max` 到了 `5.01381`
- 这一轮又回落到了 `4.98673`

这说明当前更像：

- object latent 幅值已经进入高位区间
- 但还不是单调持续发散
- loss 也仍在高低波动后回落，没有进入持续崩坏状态

因此截至这一轮，判断再次更新为：

- 训练仍可继续跑
- 当前状态属于“高位重点观察”，而不是“必须立即打断”
- 继续重点盯：
  - `object_latent_tokens_abs_max` 是否再次稳定站上 `5.0+`
  - `loss_box_aux / loss_depth_aux` 是否跟着连续高位不回落

validation / 磁盘本轮仍无变化：

- `validation100_vbench` 下仍只有旧的 `step-002000` 失败产物
- `/data` 仍约只剩 `5.1G`

## 22. 2026-06-25 23:34 UTC 高位复现

这一轮继续重点观察时，训练还在继续推进，但尚未走到下一次 `save_steps=200` 的保存边界：

- W&B `lastHistoryStep`:
  - `3458`
- stdout 最新可见进度：
  - `global_step 3459`

因此当前 checkpoint 目录仍然还没新增 `step-003600`，这一点是正常的：

- 当前最新已确认落盘的仍是：
  - `step-003200`
  - `step-003400`

本轮最新指标：

- `train/loss_total = 0.06320`
- `train/loss_track_aux = 0.16123`
- `train/loss_box_aux = 0.36163`
- `train/loss_depth_aux = 0.10914`
- `train/object_context_abs_max = 0.42333`
- `train/object_latent_tokens_abs_max = 5.08760`

这一轮和前几轮相比，关键信号是：

- `object_latent_tokens_abs_max` 再次明确站上了 `5.0+`
- `object_context_abs_max` 也再次超过了 `0.42`
- `box_aux / depth_aux` 虽然没有到之前最极端的高点，但也处在中高位

因此当前状态已经比前一轮更值得警惕：

- 不是一次性的单点尖峰后立刻完全回落
- 而是高位区间在反复出现

不过截至这一轮，仍然没有足够证据说明训练已经进入不可逆失稳，因为：

- `loss_total` 还没有持续拉到极高且不回落
- 训练主循环依然健康
- checkpoint 保存链路没有中断，只是这轮还没到 `3600`

截至本轮的工作判断更新为：

- 当前 run 继续保持“高位重点观察”
- 如果接下来一到两轮抽样里仍持续出现：
  - `object_latent_tokens_abs_max > 5.0`
  - 且 `loss_box_aux / loss_depth_aux` 不明显回落
- 就应当从“重点观察”进一步切到“准备下一次重启时实际修改尺度/约束方案”

## 23. 2026-06-25 23:41 UTC 高位持续但未全面失稳

这一轮继续重点观察时，训练仍在健康推进，但还没有到 `step-003600` 的保存点：

- W&B `lastHistoryStep`:
  - `3509`
- stdout 最新可见进度：
  - `global_step 3511`

因此当前 checkpoint 目录暂时仍保持：

- `step-003200`
- `step-003400`

本轮最新指标：

- `train/loss_total = 0.04248`
- `train/loss_track_aux = 0.07018`
- `train/loss_box_aux = 0.33318`
- `train/loss_depth_aux = 0.02146`
- `train/object_context_abs_max = 0.41610`
- `train/object_latent_tokens_abs_max = 5.36043`

这一轮最关键的事实是：

- `object_latent_tokens_abs_max` 不只是再次高于 `5.0`
- 而且已经抬到了目前观察到的新高：`5.36`

但同时也要如实记录另一面：

- `loss_total` 并没有同步抬到极高
- `loss_box_aux` / `loss_depth_aux` 这轮都比更坏的尖峰时段低
- `object_context_abs_max` 也没有继续跟着上冲，反而低于上一轮的 `0.4233`

因此截至这一轮，更准确的判断是：

- object latent 幅值高位区间已经持续存在
- 但当前还没有演化成“各项 loss 同步崩坏”的失稳态

工作结论更新为：

- 当前 run 仍可继续跑到下一个 checkpoint 节点
- 但已经有足够证据说明：
  - 下次如果需要重启训练，应优先准备 object 分支尺度约束/缩放方案
- 当前不立刻打断，是因为：
  - 训练主循环仍健康
  - checkpoint 仍持续产出
  - loss 端还没有形成持续恶化趋势

## 24. 2026-06-25 23:47 UTC 高位交替模式

这一轮继续守到接近 `step-003600`，训练仍在推进：

- W&B `lastHistoryStep`:
  - `3564`
- stdout 最新可见进度：
  - `global_step 3564`

当前仍未到 `step-003600`，因此 checkpoint 目录本轮没有新增是正常的：

- `step-003200`
- `step-003400`

本轮最新指标：

- `train/loss_total = 0.05995`
- `train/loss_track_aux = 0.02705`
- `train/loss_box_aux = 0.56883`
- `train/loss_depth_aux = 0.00360`
- `train/object_context_abs_max = 0.42768`
- `train/object_latent_tokens_abs_max = 4.94111`

这一轮和上一轮对比，非常值得记录的不是单个高低，而是“高位交替”模式：

- 上一轮：
  - `object_latent_tokens_abs_max` 新高到 `5.36`
  - `object_context_abs_max` 相对较低
- 这一轮：
  - `object_latent_tokens_abs_max` 回落到 `4.94`
  - 但 `object_context_abs_max` 反而冲到 `0.4277`
  - 同时 `loss_box_aux` 再次回到更高位置

这说明当前现象已经不能只用单一阈值来理解，而更像：

- object 分支不同子路径的高位在交替出现
- 有时是 latent token 幅值更高
- 有时是 context 幅值和 box loss 更高

截至这一轮的判断继续更新为：

- 训练仍未进入必须立刻打断的崩坏态
- 但“高位重点观察”已经进一步升级为：
  - 当前 run 可以继续
  - 但下次重启时应优先准备 object 分支尺度/约束修改，而不是直接原样继续

当前不立刻中断的依据仍然是：

- `loss_total` 还没有持续处于极高位
- `track_aux / depth_aux` 至少在这一轮没有同步恶化
- checkpoint / 主循环 / 磁盘状态都还维持住了

## 25. 2026-06-25 23:54 UTC step-003600 落盘与重启预案

这一轮已经确认新的 checkpoint 继续正常产出：

- 新 checkpoint 已成功落盘：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-003600`
- W&B `lastHistoryStep`:
  - `3616`
- stdout 最新可见进度：
  - `global_step 3617`

对应的 checkpoint 保留集继续前滚为：

- `step-003400`
- `step-003600`

本轮最新指标：

- `train/loss_total = 0.03031`
- `train/loss_track_aux = 0.06836`
- `train/loss_box_aux = 0.15809`
- `train/loss_depth_aux = 0.07666`
- `train/object_context_abs_max = 0.42594`
- `train/object_latent_tokens_abs_max = 5.04971`

本轮和前几轮合起来，已经可以给出更明确的工作结论：

- 训练本身仍然健康，checkpoint 继续稳定产出到 `step-003600`
- 但 object 分支高位区间已经不是偶发现象：
  - `object_latent_tokens_abs_max` 多次出现在 `5.0+`
  - `object_context_abs_max` 多次出现在 `0.42+`
  - `box_aux / depth_aux` 也会在部分批次回到较高位置

因此，从“是否继续当前 run”的角度：

- 当前 run 仍然可以继续跑
- 不需要为了这件事立刻中断

但从“下次如果要重启，该怎么更稳”这个角度：

- 已经有足够证据说明，下次重启不应再完全原样启动

### 下次重启的优先修改预案

优先级 1：给 object latent / context 增加显式幅值约束

- 重新启用一个很小的 `lambda_object_context_reg`
  - 当前是 `0.0`
  - 下次建议先尝试一个很小值，例如数量级 `1e-4 ~ 5e-4`
- 目的：
  - 不改主监督目标
  - 先给 object context 一条温和的幅值约束

优先级 2：收缩 object aux residual/gate 尺度

- 当前 object aux 里最明显慢漂的是几个 gate / logit：
  - `object_aux_heads.track_gate_logit`
  - `object_aux_heads.box_center_gate_logit`
  - `object_aux_heads.box_size_gate_logit`
- 下次可优先尝试减小：
  - `object_track_gate_init`
  - `box_center_gate_init`
  - `box_size_gate_init`
- 或者进一步减小：
  - `object_track_delta_scale`
  - `object_box_delta_scale`
  - `object_box_wh_log_scale`

优先级 3：如果仍需更稳，再考虑降低 object 分支学习率

- 不是先动全局训练结构
- 而是优先通过：
  - object residual 尺度
  - object context 正则
  - object 分支有效步长
 这三类手段收缩

当前建议：

- 继续让现在这条 run 往前跑
- 同时把上面这套“下次重启预案”保留好
- 如果后续又连续出现：
  - `object_latent_tokens_abs_max > 5.0`
  - 且 `box_aux / depth_aux` 再次连续高位
- 就可以直接按这套预案准备下一次重启，而不是从头重新判断

## 26. 2026-06-25 23:59 UTC 高位交替继续出现

这一轮继续观察到，训练仍然在健康推进，但还没有到 `step-003800`：

- W&B `lastHistoryStep`:
  - `3676`
- stdout 最新可见进度：
  - `global_step 3677`

因此当前 checkpoint 目录仍保持：

- `step-003400`
- `step-003600`

本轮最新指标：

- `train/loss_total = 0.05942`
- `train/loss_track_aux = 0.02636`
- `train/loss_box_aux = 0.56402`
- `train/loss_depth_aux = 0.00380`
- `train/object_context_abs_max = 0.43219`
- `train/object_latent_tokens_abs_max = 4.85871`

和前几轮对比后，这一轮再次说明当前模式不是单一变量单调发散，而是：

- 有时 `object_latent_tokens_abs_max` 站上 `5.0+`
- 有时 `object_context_abs_max` 站上 `0.43`
- 有时 `box_aux` 抬高
- 但它们并不总是同一时刻一起最坏

这说明当前 object 分支更像处在一个“高幅值、交替波动”的工作区间，而不是立刻崩掉的工作区间。

截至这一轮：

- 当前 run 仍然可以继续跑
- 但“下次重启时收缩 object 分支尺度”已经不是可选项，而是明确建议

### 下次重启的更具体参数建议

如果后续决定基于最新 checkpoint 重启训练，建议按下面顺序做最小干预：

方案 A：最小收缩，优先尝试

- 保持数据、batch、主损失权重不变
- 只加一个很小的 object context 正则：
  - `--lambda_object_context_reg 1e-4`
- 同时减小 object 残差步幅：
  - `--object_track_delta_scale 0.15`
  - `--object_box_delta_scale 0.15`
  - `--object_box_wh_log_scale 0.8`

方案 B：如果 A 还不够稳，再进一步减小 gate 初值

- 在方案 A 基础上再调：
  - `--object_track_gate_init 0.02`
- 如果代码后续暴露了 box gate init 参数，也建议同步调低到同量级

方案 C：如果仍高位摆动，再降低 object 分支有效学习强度

- 不先动全局训练框架
- 优先考虑：
  - object residual 尺度进一步减小
  - 或单独降低 object 相关模块学习率

当前不建议现在立刻打断这次 run 去切方案 A/B/C，原因仍然是：

- 训练主循环稳定
- checkpoint 仍持续产出
- loss 没有形成持续不可回落的坏趋势

## 27. 2026-06-26 00:05 UTC 接近 step-003800 的复核

这一轮继续观察到，训练已经推进到接近 `step-003800`：

- W&B `lastHistoryStep`:
  - `3731`
- stdout 最新可见进度：
  - `global_step 3733`

当前仍未看到 `step-003800` 新目录，这仍然正常，因为还没真正跨过保存步点。

本轮最新指标：

- `train/loss_total = 0.02142`
- `train/loss_track_aux = 0.02261`
- `train/loss_box_aux = 0.16410`
- `train/loss_depth_aux = 0.02753`
- `train/object_context_abs_max = 0.43348`
- `train/object_latent_tokens_abs_max = 4.84047`

这一轮的意义在于进一步确认了当前模式：

- `object_latent_tokens_abs_max` 可以从 `5.0+` 回落到 `4.84`
- 但 `object_context_abs_max` 仍然维持在 `0.433+`
- 同时 loss 端又整体偏低

这再次说明：

- 当前不是简单的“越高越坏”的单调失稳
- 更像 object 分支某些内部量在高位摆动，而主训练目标暂时还压得住

截至这一轮，工作判断保持不变：

- 当前 run 继续跑
- 但如果后续需要重启，不应原样继续

### 下次重启的命令级模板

如果后续要基于最新 checkpoint 做一次更稳的重启，可以在当前启动脚本参数基础上优先改成：

```bash
--lambda_object_context_reg 1e-4 \
--object_track_delta_scale 0.15 \
--object_box_delta_scale 0.15 \
--object_box_wh_log_scale 0.8 \
--object_track_gate_init 0.02
```

如果后续代码把 box gate init 参数暴露出来，建议同步降低到和 `object_track_gate_init` 同量级，再作为方案 B。

## 28. 2026-06-26 00:12 UTC step-003800 落盘

这一轮已经确认新的 checkpoint 继续正常产出：

- 新 checkpoint 已成功落盘：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/checkpoints/step-003800`
- W&B `lastHistoryStep`:
  - `3796`
- stdout 最新可见进度：
  - `global_step 3799`

当前真实保留集已经前滚为：

- `step-003600`
- `step-003800`

本轮最新指标：

- `train/loss_total = 0.03974`
- `train/loss_track_aux = 0.01206`
- `train/loss_box_aux = 0.37521`
- `train/loss_depth_aux = 0.01016`
- `train/object_context_abs_max = 0.43560`
- `train/object_latent_tokens_abs_max = 4.75561`

这轮继续印证当前的整体模式：

- `object_context_abs_max` 仍在 `0.43+`
- `object_latent_tokens_abs_max` 这一拍回落到 `4.76`
- loss 没有形成持续性崩坏

因此到 `step-003800` 为止，判断仍然是：

- 当前 run 可以继续跑
- 但“下次重启不能原样继续”的结论不变

### 额外的新风险：磁盘空间又进一步下降

本轮检查到：

- `/data` 可用空间已经从前面的约 `5.1G` 下降到约 `4.8G`

这意味着：

- 虽然 `max_checkpoints_keep=2` 还在工作
- 但磁盘风险再次变得更现实

所以后续除了继续盯数值，还需要继续盯：

- 新 checkpoint 是否还能稳定落盘
- validation 是否突然产生新产物并进一步压缩磁盘空间

## 29. 2026-06-26 00:18 UTC step-003800 后继续推进

这一轮继续监控时，有一个很关键的新事实：

- 训练不只是正常落盘到了 `step-003800`
- 而且已经顺利跨进了下一轮 epoch

对应证据：

- W&B `lastHistoryStep`:
  - `3858`
- stdout 最新可见进度：
  - `epoch 1 | global_step 3860`

这说明从训练循环、数据加载、优化器状态恢复与继续推进的角度看，当前 run 是非常健康的。

本轮最新指标：

- `train/loss_total = 0.06555`
- `train/loss_track_aux = 0.03853`
- `train/loss_box_aux = 0.60616`
- `train/loss_depth_aux = 0.01077`
- `train/object_context_abs_max = 0.43353`
- `train/object_latent_tokens_abs_max = 5.17737`

这轮的意义在于再次确认了“高位交替”不只是偶发一次：

- `object_latent_tokens_abs_max` 又重新回到 `5.0+`
- `loss_box_aux` 也再次抬到更高位
- 但 `loss_depth_aux` 这轮又保持很低

因此当前更准确的状态描述是：

- 训练循环稳定性很好
- object 分支高位摆动也在持续存在
- 它更像是一个“可以继续训练但建议准备下一次收缩重启”的状态，而不是“当前这次训练已经坏掉”的状态

截至这一轮，工作建议不变：

- 当前 run 继续跑
- 但若后续要从最新 checkpoint 重启，优先使用前面已经给出的缩放/正则收缩方案

## 30. 2026-06-25 23:30 UTC 继续监控：接近 step-004000，但还未落盘

这一轮主要核对了四件事：

- 训练进程是否还活着
- `step-004000` 是否已经生成
- validation 是否在 `4000` 前后重新触发
- `/data` 磁盘空间是否继续恶化

### 运行状态

训练主进程仍然正常运行：

- launcher:
  - `run_train_v_newtrain_object_heads_only_gpu67.sh`
- 两个训练 worker:
  - `train_v_newtrain.py` on `gpu6,7`
- validation 仍然配置为：
  - `benchmark_cuda_visible_devices=5`

并且本轮再次确认：

- 没有使用 `gpu4`

### 本地日志与 W&B 的最新推进位置

本地 `output.log` 最新可见推进到：

- `epoch 1 | global_step 3948`

W&B 最新可见状态：

- run id:
  - `3utgz1bh`
- state:
  - `running`
- `lastHistoryStep`:
  - `3946`

本轮最新指标：

- `train/loss_total = 0.04472`
- `train/loss_track_aux = 0.11408`
- `train/loss_box_aux = 0.28283`
- `train/loss_depth_aux = 0.05024`
- `train/object_context_abs_max = 0.42374`
- `train/object_latent_tokens_abs_max = 5.24247`

### 对这轮数值的判断

这轮仍然不是“整体 loss 直接炸掉”的模式，而是继续符合前面已经反复观察到的模式：

- `loss_total` 仍处在可接受区间
- `loss_box_aux` 与 `loss_depth_aux` 会抖动，但没有同时持续上冲
- `object_context_abs_max` 仍在 `0.42+` 区间
- `object_latent_tokens_abs_max` 再次回到 `5.24` 的高位

因此这轮更准确的结论仍然是：

- 当前 run 还能继续跑
- 但 object 分支内部幅值偏高的问题没有自然消失
- 如果后续因为别的原因需要重启，不应该原样照搬当前超参

### checkpoint 与 validation 状态

截至本轮检查，checkpoint 目录里仍然只有：

- `step-003600`
- `step-003800`

也就是说：

- `step-004000` 还没有落盘

validation 目录也仍然只有旧的失败记录：

- `step-002000/benchmark.failed.json`
- `step-002000/benchmark.stdout.log`
- `step-002000/benchmark.stderr.log`

目前还没有看到：

- `step-004000` 对应的新 validation 运行目录

这说明在本轮检查时点：

- 训练还在正常向 `4000` 推进
- 但 `4000` 的 checkpoint 与后续 validation 都还没真正发生

### 磁盘状态

本轮 `/data` 空间为：

- `Avail = 5.1G`

判断：

- 仍然偏危险
- 但比前一轮观测到的 `4.8G` 略有回升

后续仍需要重点盯：

- `step-004000` 落盘是否成功
- validation 一旦重新触发，是否会再次吃掉更多磁盘空间

### 当前操作建议

本轮不建议中断训练。

优先级最高的下一步观察点仍然是：

- 是否生成 `step-004000`
- `step-004000` 后 validation 是否重新触发
- 触发后是成功、失败，还是因为磁盘/资源问题再次中断

## 31. 2026-06-25 23:39 UTC step-004000 已落盘，validation 失败原因继续收敛

这一轮有三条已经确认的事实：

- `step-004000` checkpoint 已成功落盘
- 训练主进程随后退出，当前 `gpu6,7` 上已经没有活跃训练进程
- `step-004000` 的 validation 再次失败，但失败原因已经从“环境缺包”进一步收敛到“验证清单本身失效”

### 当前最新运行状态

当前 checkpoint 目录：

- `step-003800`
- `step-004000`

`step-004000` 目录中已确认存在：

- `checkpoint.safetensors`
- `training_state.pt`

W&B 当前状态：

- run id:
  - `3utgz1bh`
- state:
  - `running`
- `lastHistoryStep`:
  - `3999`
- summary `_step`:
  - `3999`

但本地进程已退出，W&B service 的 `debug-core.log` 也明确记录了：

- `2026-06-25T23:34:17Z processOutgoingData: finished`
- `2026-06-25T23:34:19Z parent process exited`

因此当前更准确的判断是：

- 这次 run 在 `step-004000` 之后已经停掉
- 不是还在后台继续训练

### validation 失败原因的排查路径

这次 `step-004000` validation 失败不是单一原因，而是按顺序暴露了两层问题：

1. 第一层问题：
   - `run_validation_vbench.py` 在初始化 `ValidationMetricSuite()` 时强依赖本地 DINO checkpoint
   - 缺失文件：
     - `/home/gaoya/.cache/torch/hub/checkpoints/dinov2_vitb14_pretrain.pth`
   - 这会导致 validation 子进程直接退出

2. 第二层问题：
   - 修掉 DINO 强依赖后，validation 继续往下跑
   - 但 `batch_eval_lora.py` 读取的老 validation 清单：
     - `/home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_validation100.txt`
   - 其中所有样本路径都已经失效

### 已做的代码修复

已经完成的修复：

- [run_validation_vbench.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/run_validation_vbench.py)
  - 把 DINO 指标改成“可选”
  - 当本地缺少 `dinov2_vitb14_pretrain.pth` 时：
    - 只 warning
    - 跳过 `future_dino`
    - 保留 `PSNR / SSIM / LPIPS / VBench`

- [batch_eval_lora.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/batch_eval_lora.py)
  - `load_meta_paths()` 已改成：
    - 跳过不存在的 `meta.json`
    - 输出 warning
  - 但这只能解决“部分样本失效”
  - 无法解决“整份 list 100 个路径全部失效”

### 老 validation100 清单为什么不能再用

我做了两层验证：

1. 直接检查旧 list 里的路径：
   - `OpenVid / MOVI-D / Genesis` 三类样本在当前盘上都已经不在原位置

2. 用历史 `step-010000/ctx00` 的 100 个 per-case JSON 反查当前数据盘：
   - 旧 validation100 的 `100` 个 `sample_id`
   - 在当前盘上可匹配到的真实 `meta.json` 数量：
     - `0 / 100`

结论：

- 老的 `benchmark_meta_json_paths_validation100.txt` 已经整体过期
- 不是“修一两个路径”能恢复
- 必须切换到新的、当前真实存在的验证清单

### 当前可用的替代 validation 清单

目前盘上可以直接使用的一组样本来自：

- `/data/gaoya/AAA_test_video/Benchmark/stage0_V2V/tools/visualization/benchmark_compare_portal/assets/samples`

但这里的样本也不是全部有效：

- 目录里共有 `300` 个 `meta.json` 软链接
- 实际当前可读可用的只有 `23` 个
- 当前有效数据集分布是：
  - `physics-iq-benchmark`: `19`
  - `vLAR-PhysInOne`: `4`

基于这批真实可读样本，已生成新的 validation 清单：

- `/home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_validation23_current_assets.txt`

### 已更新的训练启动配置

已修改启动脚本：

- [run_train_v_newtrain_object_heads_only_gpu67.sh](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_heads_only_gpu67.sh)

修改内容：

- `--validation_meta_list_path`
  - 从：
    - `benchmark_meta_json_paths_validation100.txt`
  - 改为：
    - `benchmark_meta_json_paths_validation23_current_assets.txt`

这样后续重启训练时：

- validation 不会再去读已经整体失效的 `validation100` 列表
- 会改成只评估当前盘上真实存在的 23 个 benchmark 样本

### 当前建议

下一步不应该再从 `step-002000` 回退重启。

应该做的是：

- 从 `step-004000/training_state.pt` 继续恢复
- 使用新的 validation23 清单
- 保持：
  - 主训练：`gpu6,7`
  - validation：`gpu5`
  - 不使用 `gpu4`

## 32. 2026-06-25 23:47 UTC 已按新 validation 清单重新续跑

这一步已经执行的动作：

1. 先用新的 validation 清单做 smoke
2. 确认 validation 不再在 metadata 读取阶段因为旧坏路径立即退出
3. 然后正式从 `step-004000` 继续恢复训练

### validation smoke 结果

本轮 smoke 使用：

- `gpu5`
- meta list:
  - `/home/gaoya/Code_Video/Code_data/Code_train/train_0419/benchmark_meta_json_paths_validation23_current_assets.txt`
- context:
  - `ctx08`

已确认现象：

- runtime 目录已经正常创建：
  - `.../validation_smoke_assets23_runtime/ctx08/metadata/step-004000_assets23_smoke_ctx08`
- output 目录已经正常创建：
  - `.../validation_smoke_assets23_outputs/ctx08`
- 前台日志只出现：
  - `DINO metrics disabled because the checkpoint is unavailable`

这说明：

- 新 validation 清单至少已经通过了样本解析和 generation 启动阶段
- 不再像旧 `validation100` 清单那样在 metadata 读取阶段立刻失败

### 正式续跑状态

已重新启动训练：

- 启动脚本：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_heads_only_gpu67.sh`
- 主训练 GPU：
  - `gpu6,7`
- validation / benchmark 预留：
  - `gpu5`

本次启动已经确认：

- `Resuming from latest checkpoint: .../step-004000`
- 实际传入训练脚本的 `--resume_from`：
  - `.../checkpoints/step-004000`
- 实际传入训练脚本的 `--validation_meta_list_path`：
  - `benchmark_meta_json_paths_validation23_current_assets.txt`

当前新的 W&B run：

- run id:
  - `yaxj219k`
- run name:
  - `pybullet0625_diffsynth_object_heads_only_gpu67`

因此当前状态已经从上一轮的：

- “训练停在 `step-004000`”

切换到：

- “训练已重新从 `step-004000` 恢复，且 validation 配置已替换成当前有效清单”

### 当前仍在观察的点

续跑刚刚启动，当前最重要的三个观察点是：

- 是否顺利跨过初始化并进入新的 optimizer step
- 新 run `yaxj219k` 的 `loss / grad` 是否开始正常写入 W&B
- 后续真正到达下一次 validation 触发点时，是否能基于 23-case 新清单继续跑通

### 恢复后的首批训练指标

当前已经确认新的恢复 run 不是只停在初始化：

- 前台训练日志已经推进到：
  - `global_step 4001`
- W&B `yaxj219k` 已经写入到：
  - `lastHistoryStep = 4002`

首批已观测到的训练指标：

- `train/loss_total = 0.60446`
- `train/loss_track_aux = 0.03431`
- `train/loss_box_aux = 0.05670`
- `train/loss_depth_aux = 5.95363`
- `train/object_context_abs_max = 0.38162`
- `train/object_latent_tokens_abs_max = 3.56240`
- `train/grad_norm = 4.82971`
- `train/grad_abs_max = 0.20147`

这组值说明两件事：

1. 梯度已经正常回传
   - `grad_norm` 和 `grad_abs_max` 都是正常有限值
   - 没有出现 `nan/inf`

2. `depth_aux` 在恢复后的头几步出现了明显尖峰
   - 当前最需要盯的是：
     - `train/loss_depth_aux = 5.95`
   - 但与此同时：
     - `object_context_abs_max` 和 `object_latent_tokens_abs_max` 反而没有冲高
   - 所以这一步暂时更像：
     - 单个 batch 的 depth supervision 尖峰
     - 还不能直接判成整体发散

当前判断：

- 训练恢复本身是成功的
- 但接下来必须短周期连续观察 `loss_depth_aux`
- 如果它连续多步都维持在异常高位，再转入代码或数据排查

### 恢复后继续观察：depth 尖峰已回落

进一步连续观察恢复 run `yaxj219k` 后，训练并没有停在最初几步，而是已经稳定推进到：

- 前台可见：
  - `global_step 4057`
- W&B 最新 summary：
  - `_step = 4054`

这轮更有代表性的指标已经变成：

- `train/loss_total = 0.04080`
- `train/loss_track_aux = 0.02708`
- `train/loss_box_aux = 0.36702`
- `train/loss_depth_aux = 0.01388`
- `train/object_context_abs_max = 0.36222`
- `train/object_latent_tokens_abs_max = 3.40371`
- `train/grad_norm = 0.35257`
- `train/grad_abs_max = 0.07945`

和恢复第一拍相比，最关键的变化是：

- `loss_depth_aux`
  - 从 `5.95`
  - 回落到 `0.0139`

因此目前更合理的判断是：

- 之前那次 `depth_aux` 高值更像恢复早期的单步 batch 尖峰
- 不是持续性的 depth 分支发散

同时可以确认：

- 梯度仍然是正常有限值
- `object_context_abs_max` 和 `object_latent_tokens_abs_max` 还在较低且稳定的区间
- 当前 run 已经从“恢复成功”进入“继续稳定推进”阶段

### 2026-06-25 23:53 UTC 继续推进：训练与 validation smoke 同时正常

进一步监控到的状态：

- 训练前台已经继续推进到：
  - `global_step 4080+`
- W&B `yaxj219k` 最新 summary：
  - `_step = 4080`

当前这拍指标：

- `train/loss_total = 0.02360`
- `train/loss_track_aux = 0.05051`
- `train/loss_box_aux = 0.17271`
- `train/loss_depth_aux = 0.01276`
- `train/object_context_abs_max = 0.37882`
- `train/object_latent_tokens_abs_max = 3.51690`
- `train/grad_norm = 0.34071`
- `train/grad_abs_max = 0.07495`

这再次印证：

- `depth_aux` 已经稳定回到低位
- `grad_norm / grad_abs_max` 没有异常放大
- object 分支内部幅值仍然处在温和区间

### validation smoke 的最新结论

这次 smoke 已经不只是“成功启动”，而是已经实际开始生成视频与 sidecar JSON：

- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/test/validation_smoke_assets23_outputs/ctx08`
- 已生成的样例包括：
  - `0005_perspective-center_trimmed-ball-behind-rotating-paper.mp4/.json`
  - `0020_perspective-center_trimmed-ball-ramp.mp4/.json`
  - `0029_perspective-center_trimmed-ball-train.mp4/.json`
  - `0032_perspective-center_trimmed-balls-collide.mp4/.json`
  - `0038_perspective-center_trimmed-blow-balloon.mp4/.json`
  - `0047_perspective-center_trimmed-domino-in-juice.mp4/.json`
  - `0059_perspective-center_trimmed-duck-falls-in-box.mp4/.json`

这说明：

- 新的 `validation23_current_assets` 清单不仅能被读取
- 而且实际 generation 链路已经跑通并开始产出视频

因此当前整体判断进一步更新为：

- 主训练链路：正常
- object 分支 loss / grad：当前稳定
- 新 validation 数据清单：可用
- validation generation 链路：可用

### 2026-06-25 23:54 UTC 继续监控：训练仍稳，等待 step-004200

进一步一轮 W&B 指标：

- `lastHistoryStep = 4111`
- `_step = 4111`

当前这拍指标：

- `train/loss_total = 0.04783`
- `train/loss_track_aux = 0.08397`
- `train/loss_box_aux = 0.32127`
- `train/loss_depth_aux = 0.07309`
- `train/object_context_abs_max = 0.36374`
- `train/object_latent_tokens_abs_max = 3.54063`
- `train/grad_norm = 0.92389`
- `train/grad_abs_max = 0.20339`

判断：

- `depth_aux` 虽然有小幅回升，但仍远低于恢复首拍的 `5.95`
- 当前更像正常 batch 间波动，而不是再次出现异常尖峰
- `object_context_abs_max` / `object_latent_tokens_abs_max` 仍稳定
- 当前最需要继续观察的是：
  - 是否顺利落下下一份 checkpoint `step-004200`

### 2026-06-25 23:55 UTC 持续跟踪：训练仍稳定，validation smoke 继续扩展

继续一轮 W&B summary：

- `lastHistoryStep = 4132`
- `_step = 4132`

当前这拍指标：

- `train/loss_total = 0.05009`
- `train/loss_track_aux = 0.02469`
- `train/loss_box_aux = 0.39342`
- `train/loss_depth_aux = 0.08275`
- `train/object_context_abs_max = 0.36306`
- `train/object_latent_tokens_abs_max = 3.62739`
- `train/grad_norm = 0.33291`
- `train/grad_abs_max = 0.07500`

更新判断：

- `loss_depth_aux` 继续在低位小幅波动
- `grad_norm` 仍稳定，没有放大趋势
- 当前没有新的数值异常证据

### validation smoke 进一步进展

`ctx08` 下已继续新增生成样例：

- `0104_perspective-center_trimmed-marble-run-x.mp4/.json`
- `0107_perspective-center_trimmed-marble-run-y.mp4/.json`

这说明：

- validation smoke 不是卡在前几个样例
- 新 validation 清单上的更多 case 也在持续成功生成

### 2026-06-25 23:56 UTC 新一拍波动：depth 回升，但仍未见发散证据

继续一轮 W&B summary：

- `lastHistoryStep = 4147`
- `_step = 4147`

当前这拍指标：

- `train/loss_total = 0.10734`
- `train/loss_track_aux = 0.07496`
- `train/loss_box_aux = 0.42568`
- `train/loss_depth_aux = 0.57273`
- `train/object_context_abs_max = 0.38473`
- `train/object_latent_tokens_abs_max = 3.69094`
- `train/grad_norm = 0.63075`
- `train/grad_abs_max = 0.14551`

判断更新：

- `loss_depth_aux` 这拍有明显回升
- 但仍明显低于恢复首拍的 `5.95`
- 同时：
  - `grad_norm` 没有失控
  - `object_context_abs_max` / `object_latent_tokens_abs_max` 也没有同步冲高

因此当前更合理的解释仍然是：

- depth supervision 的 batch 级波动
- 还不是持续发散或梯度异常

当前仍继续重点观察：

- `step-004200` 是否按时落盘
- `loss_depth_aux` 后续几拍是否继续向上累积

### GPU 使用约束

- `gpu4` 是坏卡，当前方案不要使用
- 主训练固定使用 `gpu6,7`
- validation / benchmark / smoke 固定使用 `gpu5`
- 如果后续需要重跑 cache，可复用 `gpu0/2/3/5/6/7`，但不要把 `gpu4` 放回任何启动命令

### 2026-06-25 23:59 UTC 持续巡检：`step-004200` 已成功产出，当前 run 仍健康

本轮检查结果：

- 活跃训练进程仍在：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_heads_only_gpu67.sh`
  - 两个 worker 仍在跑 `train_v_newtrain.py`
- 活跃 validation smoke 仍在：
  - `run_validation_vbench.py`
  - `batch_eval_lora.py`
- GPU 占用符合预期：
  - `gpu5` 满载跑 validation smoke
  - `gpu6,7` 被主训练占用
  - `gpu4` 仍未使用

checkpoint 状态：

- 当前 checkpoint 目录已变为：
  - `step-004000`
  - `step-004200`
- 已额外核对：
  - `step-004000/training_state.pt -> global_step = 4000`
  - `step-004200/training_state.pt -> global_step = 4200`

这说明：

- 训练不只是 W&B 前端数字推进
- 实际权重和训练状态文件都已经稳定跨过 `4200`
- `--max_checkpoints_keep 2` 仍然正常生效

### 当前 W&B 运行态

当前 project 下同名 run 有多次历史重启，最新正在运行的是：

- run id: `yaxj219k`
- state: `running`
- display_name: `pybullet0625_diffsynth_object_heads_only_gpu67`

当前 latest summary：

- `_step = 4245`
- `train/loss_total = 0.04827`
- `train/loss_track_aux = 0.07772`
- `train/loss_box_aux = 0.25555`
- `train/loss_depth_aux = 0.14946`
- `train/object_context_abs_max = 0.38993`
- `train/object_latent_tokens_abs_max = 3.88070`
- `train/grad_norm = 1.47919`
- `train/grad_abs_max = 0.34240`

当前判断：

- 训练仍在继续推进，至少已经超过 `step 4245`
- `loss_depth_aux` 比 `3999` 附近的低点有回升，但还处在可接受波动区间
- `object_context_abs_max` / `object_latent_tokens_abs_max` 没有同步异常放大
- `grad_norm` 有抬升，但目前仍是有限值，尚未看到 `nan/inf` 或明显爆炸证据

所以当前结论仍然是：

- 这是可继续观察的 batch 级波动
- 还不是明确的训练发散

### validation smoke 当前进度

当前 smoke 仍在跑 `ctx08`：

- 已产出 `22` 个 `.mp4`
- 已产出 `22` 个 `.json`

最新新增样例包括：

- `0170_perspective-center_trimmed-solid-ball-peakaboo`
- `0173_perspective-center_trimmed-stable-blocks`
- `0185_perspective-center_trimmed-water-in-juice`

说明：

- validation smoke 仍在持续前进
- 不是卡死在前半段 case

额外从 runtime manifest 确认到：

- `requested_output_frames = 24`
- 实际推理参数里 `num_frames = 25`

这和前面排查过的 Wan / DiffSynth 推理帧数约束一致，属于推理实现的 frame packing 行为，不是本轮训练新引入的问题。

### 当前最大风险仍然是磁盘

此刻 `/data`：

- `Avail = 5.1G`

由于当前同时存在：

- 主训练 checkpoint 落盘
- validation smoke 输出视频 / JSON

因此最近期最需要警惕的仍不是 loss，而是：

- 后续 checkpoint 或 validation 产物继续写盘时再次触发 `No space left on device`

### 2026-06-26 00:02 UTC 持续巡检：训练继续推进，validation smoke 已完整收尾

本轮继续核对发现：

- 主训练进程仍存活，两个 `train_v_newtrain.py` worker 仍在高负载运行
- `accelerate launch` 进程本身处于正常等待 / 管理态
- `run_validation_vbench.py` 也仍存活，并在本轮检查时刚好完成本次 smoke 收尾

一开始 `nvidia-smi` 抓到：

- `gpu6,7` 显存仍被占用，但瞬时利用率显示 `0%`
- `gpu5` 也短暂空闲

进一步结合 `ps` 判断后可确认：

- 这不是训练卡死
- 两个训练 worker 实际 CPU 使用率都接近 `100%`
- 很可能只是采样瞬间没有打到 GPU kernel，或者正处于 dataloader / 同步 / host 侧计算阶段

所以当前无需因为那一拍 `0% util` 误判训练停住。

### 当前 W&B 最新状态

继续查询当前 running run：

- run id: `yaxj219k`
- state: `running`

最新 summary 已从 `_step 4245` 推进到：

- `_step = 4280`

对应指标：

- `train/loss_total = 0.06957`
- `train/loss_track_aux = 0.04432`
- `train/loss_box_aux = 0.61284`
- `train/loss_depth_aux = 0.03853`
- `train/object_context_abs_max = 0.36412`
- `train/object_latent_tokens_abs_max = 3.89174`
- `train/grad_norm = 0.92606`
- `train/grad_abs_max = 0.21587`

和上一拍相比：

- `loss_depth_aux` 明显回落
- `grad_norm` 也从 `1.479` 回到 `0.926`
- `object_context_abs_max` / `object_latent_tokens_abs_max` 继续稳定

当前结论：

- 没有看到新的数值异常
- 当前训练状态比 `4245` 那拍更稳一些

### validation smoke 收尾结果

本次 `ctx08` smoke 现已确认完成：

- 生成 `.mp4` 数量：`23`
- 生成 `.json` 数量：`23`

runtime summary:

- `num_cases = 23`
- `num_generated = 23`
- `num_failed = 0`
- `success_rate = 1.0`

对应文件：

- `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67/test/validation_smoke_assets23_runtime/ctx08/summary.json`

这说明前面替换后的当前资产清单是有效的：

- 不再像旧 validation100 清单那样因为 meta path 失效而中途报错
- 当前 smoke 至少在单个 `ctx08` 配置下已经可以稳定全量跑完

### checkpoint 现状

当前 checkpoint 目录仍只有两份：

- `step-004000`
- `step-004200`

说明：

- 训练还没推进到下一次保存点 `step-004400`
- `--max_checkpoints_keep 2` 仍保持正常策略

### 当前最重要的后续观察点

- 下一份 checkpoint `step-004400` 是否顺利落盘
- 磁盘空间是否在下一次 checkpoint / 后续 validation 时再次触发 `No space left on device`

截至本轮：

- 训练数值本身没有暴露出需要立刻改代码或改损失配置的问题
- 真正更接近当前瓶颈的仍是 `/data` 只剩约 `5.1G` 的磁盘风险

### 2026-06-26 00:04 UTC 持续巡检：训练继续向 `step-004400` 推进，当前没有新异常

本轮检查结果：

- checkpoint 目录暂时仍只有：
  - `step-004000`
  - `step-004200`
- 说明训练尚未走到下一次保存点 `step-004400`
- 但训练进程依然正常推进：
  - `gpu6` 利用率约 `71%`
  - `gpu7` 利用率约 `100%`
  - 两个 worker 进程仍保持高负载

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4316`
- `train/loss_total = 0.05321`
- `train/loss_track_aux = 0.01274`
- `train/loss_box_aux = 0.49179`
- `train/loss_depth_aux = 0.02761`
- `train/object_context_abs_max = 0.36356`
- `train/object_latent_tokens_abs_max = 3.97536`
- `train/grad_norm = 0.62036`
- `train/grad_abs_max = 0.14522`

和上一拍 `_step = 4280` 对比：

- `loss_depth_aux` 继续回落
- `grad_norm` 继续回落
- `object_context_abs_max` 继续稳定
- `object_latent_tokens_abs_max` 仍在正常波动范围内

当前判断：

- 训练还在稳定前进
- 没有看到新的梯度尖峰或数值发散迹象
- 当前最合理的操作仍然是继续观察，等待 `step-004400` 落盘

磁盘状态没有改善：

- `/data` 仍只有约 `5.1G` 可用

因此当前第一风险顺位仍然是：

- 下一次 checkpoint 落盘时再次撞到磁盘上限

### 2026-06-26 00:07 UTC 持续巡检：已推进到 `step 4341`，仍未见数值异常

本轮继续检查：

- checkpoint 目录仍只有：
  - `step-004000`
  - `step-004200`
- 说明还没到 `step-004400` 的实际落盘时刻

但训练负载依旧明确正常：

- `gpu6` 利用率约 `96%`
- `gpu7` 利用率约 `100%`
- 两个训练 worker CPU 使用率继续接近 `100%`

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4341`
- `train/loss_total = 0.01586`
- `train/loss_track_aux = 0.02147`
- `train/loss_box_aux = 0.10178`
- `train/loss_depth_aux = 0.03535`
- `train/object_context_abs_max = 0.37446`
- `train/object_latent_tokens_abs_max = 4.00618`
- `train/grad_norm = 0.31626`
- `train/grad_abs_max = 0.07112`

和上一拍 `_step = 4316` 对比：

- `loss_total` 进一步下降
- `loss_box_aux` 明显下降
- `grad_norm` 明显下降
- `depth_aux` 小幅回升但仍处于低位
- `object_context_abs_max` / `object_latent_tokens_abs_max` 仍然稳定

当前判断：

- 训练在接近 `step-004400` 的阶段仍然没有出现新异常
- 当前数值状态甚至比前几拍更平稳
- 目前没有证据支持修改 loss、学习率或 object 分支实现

截至这一轮，最大风险排序仍然不变：

1. `/data` 剩余空间只有约 `5.1G`
2. 下一次 checkpoint 落盘可能再次触发 `No space left on device`
3. 训练数值风险当前反而处于较低优先级

### 2026-06-26 00:09 UTC 持续巡检：已推进到 `step 4366`，仍是正常 batch 波动

本轮检查结果：

- checkpoint 目录仍未出现 `step-004400`
- 当前仍只保留：
  - `step-004000`
  - `step-004200`

进程和负载状态：

- `accelerate` 主进程仍存活
- 两个训练 worker 继续高负载运行
- `gpu6,7` 仍在工作，只是瞬时利用率会有采样波动

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4366`
- `train/loss_total = 0.04870`
- `train/loss_track_aux = 0.13566`
- `train/loss_box_aux = 0.24561`
- `train/loss_depth_aux = 0.10569`
- `train/object_context_abs_max = 0.39376`
- `train/object_latent_tokens_abs_max = 4.01483`
- `train/grad_norm = 0.61683`
- `train/grad_abs_max = 0.14486`

和上一拍 `_step = 4341` 对比：

- `track/box/depth` 都有回升
- 但 `grad_norm` 仍然处于低位
- `object_context_abs_max` / `object_latent_tokens_abs_max` 仍稳定，没有同步放大

因此当前判断仍然是：

- 正常 batch 级波动
- 不是新的发散征兆

当前风险排序仍不变：

1. 等待 `step-004400` 实际落盘
2. `/data` 约 `5.1G` 的空间可能在下次写 checkpoint 时再次成为首要中断原因
3. 当前 loss / grad 风险仍低于磁盘风险

### 2026-06-26 00:11 UTC 关键里程碑：`step-004400` 已成功落盘

这一轮已确认：

- 新 checkpoint：
  - `step-004400`
- 落盘时间：
  - `2026-06-26 00:08:02 UTC` 左右写出 `checkpoint.safetensors`
  - `2026-06-26 00:08:03 UTC` 左右写出 `training_state.pt`

已进一步核对：

- `step-004200/training_state.pt -> global_step = 4200`
- `step-004400/training_state.pt -> global_step = 4400`

这说明：

- checkpoint 真正完整写出
- 训练状态与目录命名严格一致

### checkpoint 保留策略验证

当前 checkpoint 目录只剩：

- `step-004200`
- `step-004400`

这说明 `--max_checkpoints_keep 2` 已继续正常工作：

- 旧的 `step-004000` 已被自动淘汰
- 没有因为 checkpoint 轮换逻辑出错而堆积更多目录

### 当前 W&B 最新状态

当前 running run 仍是：

- run id: `yaxj219k`

latest summary 已进一步推进到：

- `_step = 4417`

当前指标：

- `train/loss_total = 0.03073`
- `train/loss_track_aux = 0.04489`
- `train/loss_box_aux = 0.22114`
- `train/loss_depth_aux = 0.04123`
- `train/object_context_abs_max = 0.39278`
- `train/object_latent_tokens_abs_max = 4.07760`
- `train/grad_norm = 1.18549`
- `train/grad_abs_max = 0.27950`

判断：

- `step-004400` 落盘之后训练没有中断
- 已继续推进到 `4417`
- `grad_norm` 相比前一拍有所回升，但仍是有限值
- `object_context_abs_max` / `object_latent_tokens_abs_max` 没有异常放大

因此当前更合理的解释仍然是：

- 正常 batch 波动
- 不是 checkpoint 落盘后引发的训练异常

### 当前结论更新

到目前为止：

- cache 已完成
- 主训练在 `gpu6,7` 上持续正常推进
- validation smoke 已完成且 `23/23` 成功
- `step-004400` 已成功产出
- 当前没有出现需要立即改代码或改训练方案的数值问题

当前最主要剩余风险仍是：

- `/data` 空间仍只在 `5.1G` 左右
- 下一份 checkpoint `step-004600` 将成为新的磁盘压力点

### 2026-06-26 00:13 UTC 持续巡检：`step-004400` 后继续稳定推进到 `4437`

本轮继续核对：

- checkpoint 目录目前仍只有：
  - `step-004200`
  - `step-004400`
- 说明下一份 `step-004600` 还未实际落盘

训练进程与设备状态：

- `accelerate` 主进程仍正常存活
- 两个训练 worker 继续高负载运行
- `gpu6` 利用率约 `100%`
- `gpu7` 利用率约 `84%`

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4437`
- `train/loss_total = 0.02942`
- `train/loss_track_aux = 0.04773`
- `train/loss_box_aux = 0.21882`
- `train/loss_depth_aux = 0.02762`
- `train/object_context_abs_max = 0.39332`
- `train/object_latent_tokens_abs_max = 4.10287`
- `train/grad_norm = 0.60514`
- `train/grad_abs_max = 0.14242`

和上一拍 `_step = 4417` 对比：

- `loss_total` 小幅下降
- `loss_box_aux` 小幅下降
- `loss_depth_aux` 小幅下降
- `grad_norm` 从 `1.185` 回落到 `0.605`
- `object_context_abs_max` 仍稳定

当前判断：

- `step-004400` 落盘后训练继续正常推进
- 没有出现落盘后才发生的数值异常
- 当前 loss / grad 状态依旧稳定

下一关键观察点保持不变：

- `step-004600` 是否顺利落盘
- `/data` 空间是否会在下一次 checkpoint 写入时再次成为首要中断原因

### 2026-06-26 00:15 UTC 持续巡检：已推进到 `step 4458`，尚未到 `step-004600`

本轮检查结果：

- checkpoint 目录仍只有：
  - `step-004200`
  - `step-004400`
- 说明 `step-004600` 还未落盘

当前训练状态：

- `accelerate` 主进程仍存活
- 两个训练 worker 继续运行
- `nvidia-smi` 这拍抓到的 `gpu6,7` 利用率较低，但这和前面多次观察一致，更像瞬时采样落在 host 侧 / 同步侧阶段，不代表训练停住

最关键的是 W&B 仍在继续增长：

- run id: `yaxj219k`
- state: `running`
- `_step = 4458`

当前指标：

- `train/loss_total = 0.05067`
- `train/loss_track_aux = 0.02788`
- `train/loss_box_aux = 0.44244`
- `train/loss_depth_aux = 0.03632`
- `train/object_context_abs_max = 0.36442`
- `train/object_latent_tokens_abs_max = 4.04849`
- `train/grad_norm = 1.19230`
- `train/grad_abs_max = 0.28244`

和上一拍 `_step = 4437` 对比：

- `loss_box_aux` 与 `grad_norm` 有回升
- 但 `loss_depth_aux` 仍低
- `object_context_abs_max` 反而更低
- `object_latent_tokens_abs_max` 仍稳定

因此当前判断仍然是：

- 正常 batch 级波动
- 还没有出现需要介入的数值异常

当前优先级不变：

1. 继续等待 `step-004600` 落盘
2. 持续警惕 `/data` 约 `5.1G` 可用空间带来的下一次 checkpoint 写盘风险

### 2026-06-26 00:17 UTC 持续巡检：已推进到 `step 4478`，数值再次回稳

本轮继续核对：

- checkpoint 目录仍只有：
  - `step-004200`
  - `step-004400`
- `step-004600` 仍未落盘

运行状态方面：

- `accelerate` 主进程仍存活
- 两个训练 worker 继续高 CPU 运行
- `nvidia-smi` 依旧可能抓到瞬时 `0%` 利用率采样，但结合 W&B step 持续增长，可以继续判断训练没有停住

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4478`
- `train/loss_total = 0.01832`
- `train/loss_track_aux = 0.05339`
- `train/loss_box_aux = 0.09931`
- `train/loss_depth_aux = 0.03053`
- `train/object_context_abs_max = 0.39571`
- `train/object_latent_tokens_abs_max = 4.16685`
- `train/grad_norm = 0.60138`
- `train/grad_abs_max = 0.14071`

和上一拍 `_step = 4458` 对比：

- `loss_total` 明显下降
- `loss_box_aux` 明显下降
- `grad_norm` 明显回落
- `loss_depth_aux` 仍保持低位
- `object_context_abs_max` / `object_latent_tokens_abs_max` 仍处于稳定范围

当前判断：

- 训练仍在稳定推进
- 当前这一拍比上一拍更平稳
- 目前没有证据支持修改代码、调学习率或调整 loss 设计

下一观察点继续保持：

- `step-004600` 是否顺利落盘
- `/data` 剩余空间是否会在下一次 checkpoint 写入时再次先于训练数值成为首要风险

### 2026-06-26 00:18 UTC 持续巡检：已推进到 `step 4509`，离 `step-004600` 更近

本轮继续确认：

- checkpoint 目录仍只有：
  - `step-004200`
  - `step-004400`
- `step-004600` 还未实际写出

训练运行状态：

- 两个训练 worker 继续高负载运行
- `gpu6` / `gpu7` 利用率重新回到明显工作状态
- 说明前几轮偶发的 `0% util` 采样依旧只是瞬时观测，不代表训练停顿

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4509`
- `train/loss_total = 0.04217`
- `train/loss_track_aux = 0.01362`
- `train/loss_box_aux = 0.40048`
- `train/loss_depth_aux = 0.00756`
- `train/object_context_abs_max = 0.36373`
- `train/object_latent_tokens_abs_max = 4.22004`
- `train/grad_norm = 0.31109`
- `train/grad_abs_max = 0.07134`

和上一拍 `_step = 4478` 对比：

- `loss_track_aux` 明显回落
- `loss_depth_aux` 进一步回落到很低
- `grad_norm` 明显回落
- `loss_box_aux` 有正常 batch 级回升
- `object_context_abs_max` 保持稳定

当前判断：

- 训练仍在健康推进
- 当前没有看到接近 `step-004600` 时的特殊异常
- 当前数值风险继续低于磁盘风险

### 2026-06-26 00:20 UTC 持续巡检：已推进到 `step 4535`

本轮继续确认：

- checkpoint 目录仍未出现 `step-004600`
- 当前仍只保留：
  - `step-004200`
  - `step-004400`

训练侧状态：

- 两个训练 worker 继续高负载运行
- `gpu6` / `gpu7` 继续保持工作状态

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4535`
- `train/loss_total = 0.02413`
- `train/loss_track_aux = 0.04523`
- `train/loss_box_aux = 0.16472`
- `train/loss_depth_aux = 0.03136`
- `train/object_context_abs_max = 0.39083`
- `train/object_latent_tokens_abs_max = 4.21725`
- `train/grad_norm = 1.24645`
- `train/grad_abs_max = 0.29672`

和上一拍 `_step = 4509` 对比：

- `loss_total` 继续下降
- `loss_box_aux` 继续下降
- `grad_norm` 有回升
- 但 `object_context_abs_max` 没有同步上冲
- `loss_depth_aux` 仍在低位

因此当前仍更符合：

- 正常 batch 级波动
- 而不是明确的数值发散

当前优先级仍不变：

1. 继续等待 `step-004600` 落盘
2. 继续警惕 `/data` 仅约 `5.1G` 的剩余空间

### 2026-06-26 00:21 UTC 持续巡检：已推进到 `step 4555`

本轮继续确认：

- checkpoint 目录仍只有：
  - `step-004200`
  - `step-004400`
- `step-004600` 还未落盘

训练运行状态：

- 两个训练 worker 继续高负载运行
- `gpu6,7` 的瞬时利用率采样仍可能波动
- 但结合 W&B step 持续增长，可以继续确认训练没有停住

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4555`
- `train/loss_total = 0.01537`
- `train/loss_track_aux = 0.02259`
- `train/loss_box_aux = 0.10407`
- `train/loss_depth_aux = 0.02702`
- `train/object_context_abs_max = 0.39356`
- `train/object_latent_tokens_abs_max = 4.25116`
- `train/grad_norm = 0.61587`
- `train/grad_abs_max = 0.14508`

和上一拍 `_step = 4535` 对比：

- `loss_total` 继续下降
- `loss_box_aux` 明显下降
- `grad_norm` 继续回落
- `loss_depth_aux` 仍保持低位
- `object_context_abs_max` 继续稳定

当前判断：

- 训练仍在健康推进
- 当前这一拍比上一拍更平稳
- 目前仍没有证据表明需要改代码或调整训练方案

### 2026-06-26 00:22 UTC 持续巡检：已推进到 `step 4576`

本轮继续确认：

- checkpoint 目录仍只有：
  - `step-004200`
  - `step-004400`
- `step-004600` 还未落盘

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4576`
- `train/loss_total = 0.02645`
- `train/loss_track_aux = 0.04443`
- `train/loss_box_aux = 0.16929`
- `train/loss_depth_aux = 0.05078`
- `train/object_context_abs_max = 0.39962`
- `train/object_latent_tokens_abs_max = 4.26723`
- `train/grad_norm = 0.07681`
- `train/grad_abs_max = 0.02500`

和上一拍 `_step = 4555` 对比：

- `loss_total` 小幅回升，但仍低
- `loss_box_aux` 小幅回升，但仍处于正常范围
- `grad_norm` 明显变小
- `object_context_abs_max` / `object_latent_tokens_abs_max` 仍稳定

当前判断：

- 这一拍更像一个“容易 batch”导致的低梯度，而不是训练异常
- 当前没有看到 `nan/inf`
- 也没有看到 object token 幅值异常放大

因此目前仍然维持原判断：

- 训练继续正常推进
- 主要风险仍然不是数值，而是下一次 checkpoint 写盘的磁盘空间

### 2026-06-26 00:24 UTC 关键里程碑：`step-004600` 已成功落盘

这一轮已确认：

- 新 checkpoint：
  - `step-004600`
- 落盘时间：
  - `2026-06-26 00:17:49 UTC` 左右写出 `checkpoint.safetensors`
  - `2026-06-26 00:17:50 UTC` 左右写出 `training_state.pt`

已进一步核对：

- `step-004400/training_state.pt -> global_step = 4400`
- `step-004600/training_state.pt -> global_step = 4600`

说明：

- checkpoint 已完整写出
- 训练状态与 checkpoint 目录编号保持一致

### checkpoint 保留策略继续生效

当前 checkpoint 目录只剩：

- `step-004400`
- `step-004600`

这说明：

- `--max_checkpoints_keep 2` 继续正常工作
- 旧的 `step-004200` 已经被自动淘汰

### 当前 W&B 最新状态

当前 running run 仍是：

- run id: `yaxj219k`

latest summary 已继续推进到：

- `_step = 4596`

当前指标：

- `train/loss_total = 0.03514`
- `train/loss_track_aux = 0.09260`
- `train/loss_box_aux = 0.21382`
- `train/loss_depth_aux = 0.04497`
- `train/object_context_abs_max = 0.40027`
- `train/object_latent_tokens_abs_max = 4.28085`
- `train/grad_norm = 0.59035`
- `train/grad_abs_max = 0.14029`

判断：

- `step-004600` 落盘后训练没有中断
- 当前数值仍在正常 batch 波动范围内
- 没有看到 checkpoint 落盘后才触发的异常

### 风险等级更新

相比上一轮，当前最关键的新变化不是 loss，而是磁盘：

- `/data` 可用空间已从约 `5.1G` 下降到约 `3.9G`

这意味着：

- 虽然 `step-004600` 已经成功产出
- 但下一份 `step-004800` 的磁盘风险更高了

当前剩余的首要风险已经进一步集中到：

- `/data` 空间可能在下一次 checkpoint 写入时再次触发 `No space left on device`

### 2026-06-26 00:26 UTC 持续巡检：`step-004600` 后继续推进到 `4646`

本轮继续确认：

- checkpoint 目录当前仍只有：
  - `step-004400`
  - `step-004600`
- 说明 `step-004800` 还未实际落盘

同时注意到一个关键变化：

- `/data` 可用空间当前又回升到了约 `5.1G`

这说明：

- checkpoint 轮换后空间已经部分释放
- `--max_checkpoints_keep 2` 仍在有效缓解磁盘压力

### 当前 W&B 最新状态

当前 running run 仍是：

- run id: `yaxj219k`

summary 最新可见到：

- `_step = 4625`

进一步检查最近 history，可以确认训练已经继续推进到：

- `_step = 4646`

最近几拍的代表性数值：

- `_step = 4589`
  - `loss_total = 0.05429`
  - `loss_box_aux = 0.40540`
  - `loss_depth_aux = 0.08051`
  - `grad_norm = 1.25600`
- `_step = 4596`
  - `loss_total = 0.03514`
  - `loss_box_aux = 0.21382`
  - `loss_depth_aux = 0.04497`
  - `grad_norm = 0.59035`
- `_step = 4634`
  - `loss_total = 0.04845`
  - `loss_track_aux = 0.17016`
  - `loss_depth_aux = 0.10710`
  - `grad_norm = 0.30675`
- `_step = 4639`
  - `loss_total = 0.04782`
  - `loss_box_aux = 0.41931`
  - `loss_depth_aux = 0.03309`
  - `grad_norm = 0.11520`
- `_step = 4646`
  - `loss_total = 0.04360`
  - `loss_box_aux = 0.35205`
  - `loss_depth_aux = 0.02971`
  - `grad_norm = 0.32094`

### 对这几拍波动的判断

最近几十步里可以看到：

- 个别拍会出现 `loss_depth_aux` 或 `loss_box_aux` 的单拍抬升
- 个别拍 `grad_norm` 会抬到 `1.2` 左右

但更关键的是：

- 这些尖峰没有连续累积
- 后续几拍通常会明显回落
- `object_context_abs_max` 一直大致稳定在 `0.36 ~ 0.40`
- `object_latent_tokens_abs_max` 一直大致稳定在 `4.14 ~ 4.41`
- 没有看到 `nan/inf`

因此当前更合理的结论仍然是：

- 正常的 batch 级波动
- 不是持续发散
- 当前仍不需要因为这些波动去修改代码或训练超参

### 当前后续观察点

- 继续等待 `step-004800` 是否顺利落盘
- 继续盯 `/data` 可用空间是否再次明显下探

### 2026-06-26 00:29 UTC 持续巡检：已推进到 `step 4687`

本轮继续确认：

- checkpoint 目录仍只有：
  - `step-004400`
  - `step-004600`
- `step-004800` 仍未落盘

当前磁盘状态：

- `/data` 可用空间仍约 `5.1G`

这说明当前在 `step-004600` 轮换之后，空间还没有再次快速恶化。

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4687`
- `train/loss_total = 0.04494`
- `train/loss_track_aux = 0.06945`
- `train/loss_box_aux = 0.35318`
- `train/loss_depth_aux = 0.02679`
- `train/object_context_abs_max = 0.40947`
- `train/object_latent_tokens_abs_max = 4.46227`
- `train/grad_norm = 0.30632`
- `train/grad_abs_max = 0.07158`

和前几拍对比：

- `loss_track_aux` / `loss_box_aux` 有正常回升
- 但 `loss_depth_aux` 仍低
- `grad_norm` 仍然不高
- `object_context_abs_max` / `object_latent_tokens_abs_max` 虽有小幅抬升，但还在近期正常范围内

当前判断仍然是：

- 正常 batch 级波动
- 训练继续稳定推进
- 当前没有出现需要修改代码或调整方案的异常信号

### 2026-06-26 00:31 UTC 持续巡检：已推进到 `step 4708`

本轮继续确认：

- checkpoint 目录仍只有：
  - `step-004400`
  - `step-004600`
- `step-004800` 仍未落盘

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4708`
- `train/loss_total = 0.03465`
- `train/loss_track_aux = 0.06551`
- `train/loss_box_aux = 0.22019`
- `train/loss_depth_aux = 0.06076`
- `train/object_context_abs_max = 0.41109`
- `train/object_latent_tokens_abs_max = 4.48375`
- `train/grad_norm = 0.59993`
- `train/grad_abs_max = 0.14402`

和上一轮对比：

- `loss_track_aux` / `loss_box_aux` / `loss_depth_aux` 有正常回升
- `grad_norm` 仍然不高
- `object_context_abs_max` / `object_latent_tokens_abs_max` 只是小幅抬升

当前判断：

- 这仍然更像正常 batch 级波动
- 还没有达到需要介入处理的异常区间
- 当前最值得继续盯的仍然是 `step-004800` 落盘和磁盘空间变化

### 2026-06-26 00:33 UTC 持续巡检：已推进到 `step 4728`

本轮继续确认：

- checkpoint 目录仍只有：
  - `step-004400`
  - `step-004600`
- `step-004800` 仍未落盘

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4728`
- `train/loss_total = 0.05859`
- `train/loss_track_aux = 0.13440`
- `train/loss_box_aux = 0.34265`
- `train/loss_depth_aux = 0.10888`
- `train/object_context_abs_max = 0.40913`
- `train/object_latent_tokens_abs_max = 4.44520`
- `train/grad_norm = 0.60646`
- `train/grad_abs_max = 0.14535`

和上一轮对比：

- `track_aux` / `box_aux` / `depth_aux` 都有回升
- 但 `grad_norm` 仍然不高
- `object_context_abs_max` / `object_latent_tokens_abs_max` 没有同步失控

当前判断：

- 这拍波动比上一拍大一些
- 但目前仍然更像正常 batch 级波动
- 还没有形成需要介入的持续异常趋势

### 2026-06-26 00:34 UTC 持续巡检：已推进到 `step 4753`

本轮继续确认：

- checkpoint 目录仍只有：
  - `step-004400`
  - `step-004600`
- `step-004800` 仍未落盘

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4753`
- `train/loss_total = 0.04545`
- `train/loss_track_aux = 0.02242`
- `train/loss_box_aux = 0.38236`
- `train/loss_depth_aux = 0.04969`
- `train/object_context_abs_max = 0.36723`
- `train/object_latent_tokens_abs_max = 4.37915`
- `train/grad_norm = 0.59713`
- `train/grad_abs_max = 0.14256`

和上一轮对比：

- `loss_track_aux` 回落
- `loss_box_aux` 有回升
- `loss_depth_aux` 回到中低位
- `grad_norm` 仍然中等
- `object_context_abs_max` 反而回落

当前判断：

- 训练仍然稳定推进
- 当前数值仍处于正常 batch 波动区间
- 还没有出现需要介入处理的异常趋势

### 2026-06-26 00:36 UTC 持续巡检：已推进到 `step 4773`

本轮继续确认：

- checkpoint 目录仍只有：
  - `step-004400`
  - `step-004600`
- `step-004800` 仍未落盘

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4773`
- `train/loss_total = 0.03102`
- `train/loss_track_aux = 0.04772`
- `train/loss_box_aux = 0.25233`
- `train/loss_depth_aux = 0.01014`
- `train/object_context_abs_max = 0.41214`
- `train/object_latent_tokens_abs_max = 4.49457`
- `train/grad_norm = 0.10305`
- `train/grad_abs_max = 0.06250`

和上一轮对比：

- `loss_total` 回落
- `loss_box_aux` 回落
- `loss_depth_aux` 明显回落到很低
- `grad_norm` 也明显回落

当前判断：

- 这一拍更像容易 batch
- 当前仍然没有看到接近 `step-004800` 时的异常迹象

### 2026-06-26 00:37 UTC 持续巡检：已推进到 `step 4794`

本轮继续确认：

- checkpoint 目录仍只有：
  - `step-004400`
  - `step-004600`
- `step-004800` 仍未落盘

当前 W&B latest summary：

- run id: `yaxj219k`
- state: `running`
- `_step = 4794`
- `train/loss_total = 0.05995`
- `train/loss_track_aux = 0.02133`
- `train/loss_box_aux = 0.54404`
- `train/loss_depth_aux = 0.03414`
- `train/object_context_abs_max = 0.36974`
- `train/object_latent_tokens_abs_max = 4.44015`
- `train/grad_norm = 0.32445`
- `train/grad_abs_max = 0.08750`

和上一轮对比：

- `loss_box_aux` 有明显回升
- 但 `grad_norm` 仍然较低
- `loss_depth_aux` 仍在低位
- `object_context_abs_max` 反而回落

当前判断：

- 这更像单个 batch 的 box supervision 波动
- 目前仍没有证据表明训练失控
- 下一关键点仍然是 `step-004800` 实际落盘

### 2026-06-26 00:42 UTC 关键里程碑：`step-004800` 已成功落盘

这一轮已确认：

- 新 checkpoint：
  - `step-004800`
- 落盘时间：
  - `2026-06-26 00:27:41 UTC` 左右写出 `checkpoint.safetensors`
  - `2026-06-26 00:27:43 UTC` 左右写出 `training_state.pt`

当前 checkpoint 目录只剩：

- `step-004600`
- `step-004800`

这说明：

- `--max_checkpoints_keep 2` 继续正常生效
- 旧的 `step-004400` 已被自动淘汰

### 当前 W&B 最新状态

当前 running run 仍是：

- run id: `yaxj219k`

latest summary 已推进到：

- `_step = 4818`

当前指标：

- `train/loss_total = 0.04618`
- `train/loss_track_aux = 0.03105`
- `train/loss_box_aux = 0.39561`
- `train/loss_depth_aux = 0.03515`
- `train/object_context_abs_max = 0.37030`
- `train/object_latent_tokens_abs_max = 4.41620`
- `train/grad_norm = 0.11457`
- `train/grad_abs_max = 0.07500`

判断：

- `step-004800` 落盘后训练没有中断
- 已继续推进到 `4818`
- 当前数值仍在正常 batch 波动范围内

### 磁盘风险更新

这次 checkpoint 落盘后：

- `/data` 可用空间仍维持在约 `5.1G`

说明：

- checkpoint 轮换释放空间的节奏目前还能跟上
- 磁盘风险仍需要继续盯，但没有像上一次那样立刻下探到 `3.9G`

新的下一个关键观察点变为：

- `step-005000` 是否顺利落盘

### 2026-06-26 00:31 UTC 巡检更新

本轮重新确认训练主进程仍在运行：

- launcher:
  - `run_train_v_newtrain_object_heads_only_gpu67.sh`
- accelerate:
  - `accelerate launch --multi_gpu --num_processes 2 ... train_v_newtrain.py`
- 两个 worker:
  - `train_v_newtrain.py` 仍然都存活
  - `ps` 里 CPU 仍接近 `99%`

GPU 占用状态：

- `gpu6`:
  - 显存约 `42725 / 49140 MiB`
  - util 约 `60%`
- `gpu7`:
  - 显存约 `42725 / 49140 MiB`
  - util 约 `60%`
- `gpu4`:
  - 未被使用

这说明当前训练仍稳定跑在 `gpu6,7`，没有误用坏卡 `gpu4`。

### 当前 checkpoint 落盘状态

重新检查本地目录时，checkpoint 仍只有：

- `step-004600`
- `step-004800`

并且：

- `step-004800/training_state.pt` 中 `global_step = 4800`

因此当前结论是：

- W&B 已继续往前写
- 但本地尚未出现新的 `step-005000`
- 暂时还在等待下一次保存点真正落盘

### 当前 W&B 最新摘要

latest summary 已更新到：

- `_step = 4875`

当前指标：

- `train/loss_total = 0.04916`
- `train/loss_track_aux = 0.07754`
- `train/loss_box_aux = 0.34467`
- `train/loss_depth_aux = 0.06936`
- `train/object_context_abs_max = 0.41952`
- `train/object_latent_tokens_abs_max = 4.57976`
- `train/grad_norm = 0.09256`
- `train/grad_abs_max = 0.05000`

相对上一轮 `_step = 4818`：

- `loss_track_aux` 有一轮 batch 级抬升
- `loss_box_aux` 反而回落
- `loss_depth_aux` 有抬升但仍不算异常
- `grad_norm` 继续处在较低位置
- `object_context_abs_max` / `object_latent_tokens_abs_max` 有轻微上扬

当前判断：

- 这依然更像正常 batch 波动，而不是发散
- 目前没有看到 `nan/inf`
- 也没有看到梯度爆炸迹象

### 磁盘状态

本轮 `/data` 空间仍为：

- `5.1G available`

判断：

- checkpoint 轮换暂时还能维持训练继续前进
- 当前最现实的风险仍然是磁盘，而不是数值稳定性

### 2026-06-26 00:33 UTC 巡检补充：为什么还没有 `step-005000`

这轮再次检查后，训练状态是：

- 训练 launcher 仍存活
- `accelerate launch` 仍存活
- 两个 `train_v_newtrain.py` worker 仍存活
- `gpu6,7` 仍在占用显存
- `gpu4` 仍未被使用

当前 W&B latest summary 已到：

- `_step = 4900`

对应数值：

- `train/loss_total = 0.04340`
- `train/loss_track_aux = 0.06921`
- `train/loss_box_aux = 0.28730`
- `train/loss_depth_aux = 0.07746`
- `train/object_context_abs_max = 0.41961`
- `train/object_latent_tokens_abs_max = 4.56450`
- `train/grad_norm = 1.22304`
- `train/grad_abs_max = 0.29627`

本地 checkpoint 目录仍只有：

- `step-004600`
- `step-004800`

因此这一轮的结论很明确：

- 还没有出现 `step-005000`，不是保存逻辑坏了
- 而是训练目前只推进到 `_step = 4900`
- 按 `save_steps = 200`，下一次真正应该落盘的是 `step-005000`

这一轮指标 interpretation：

- `loss_total` 继续处在低位
- `loss_box_aux` 进一步回落
- `loss_depth_aux` 有所抬升，但还没有形成持续失控的证据
- `grad_norm` 和 `grad_abs_max` 这一轮明显高于上一轮
- 但目前仍更像 batch-level spike，而不是持续发散

当前判断：

- 训练仍在正常推进
- 还没有发现需要立即 patch 代码的问题
- 下一关键观察点依然是：
  - `step-005000` 是否落盘
  - 若落盘，checkpoint 轮换是否仍只保留 2 份
  - 后续 `step-006000` 时验证是否继续正常触发

### 2026-06-26 00:34 UTC 持续巡检

这一轮重新确认：

- 训练主进程仍存活
- 两个 worker 仍存活
- `gpu6,7` 继续承担训练
- `gpu4` 仍未被使用
- `/data` 仍约 `5.1G available`

当前本地 checkpoint 依旧只有：

- `step-004600`
- `step-004800`

W&B latest summary 已更新到：

- `_step = 4931`

当前数值：

- `train/loss_total = 0.05021`
- `train/loss_track_aux = 0.13748`
- `train/loss_box_aux = 0.24279`
- `train/loss_depth_aux = 0.12182`
- `train/object_context_abs_max = 0.42296`
- `train/object_latent_tokens_abs_max = 4.77232`
- `train/grad_norm = 1.45357`
- `train/grad_abs_max = 0.35545`

相对上一轮 `_step = 4900`：

- `loss_box_aux` 继续回落
- `loss_track_aux` 明显抬升
- `loss_depth_aux` 继续抬升
- `grad_norm` / `grad_abs_max` 继续抬升
- `object_context_abs_max` 变化不大
- `object_latent_tokens_abs_max` 略有升高

当前 interpretation：

- 还没到 `step-005000`，因此没有新 checkpoint 落盘仍然正常
- 最新几步在 `track/depth` 辅助项上出现了更明显的 batch-level 波动
- 但当前仍没有 `nan/inf`
- 也还没有证据显示进入持续发散

验证侧状态：

- test 目录里最新产物仍然是上一轮 `validation_smoke_assets23_runtime/ctx08/summary.json`
- 当前没有新的 validation 触发痕迹
- 这与当前尚未到新的验证触发点是一致的

当前结论：

- 继续训练，暂不改代码
- 下一重点仍然是观察：
  - `step-005000` 是否按时落盘
  - 落盘后指标是否回落
  - 若 `grad_norm` / `loss_depth_aux` 后续继续连续上扬，再考虑介入调整 loss 权重或排查 batch 分布

### 2026-06-26 00:36 UTC 回落确认

这一轮再次检查时：

- 本地 checkpoint 仍只有 `step-004600` / `step-004800`
- 训练进程仍然正常
- `gpu6,7` 继续在跑
- `gpu4` 仍未使用
- validation 目录没有新的触发痕迹

W&B latest summary 已推进到：

- `_step = 4962`

当前数值：

- `train/loss_total = 0.04368`
- `train/loss_track_aux = 0.02407`
- `train/loss_box_aux = 0.37501`
- `train/loss_depth_aux = 0.03767`
- `train/object_context_abs_max = 0.38119`
- `train/object_latent_tokens_abs_max = 4.56938`
- `train/grad_norm = 0.31766`
- `train/grad_abs_max = 0.07500`

相对上一轮 `_step = 4931`：

- `loss_track_aux` 大幅回落
- `loss_depth_aux` 大幅回落
- `grad_norm` / `grad_abs_max` 也明显回落
- `loss_box_aux` 有回升，但仍属于正常 batch 波动范围
- `object_context_abs_max` / `object_latent_tokens_abs_max` 也回到了更稳定的带宽

这说明：

- 前一轮看到的 `track/depth/grad` 上扬，更像短时 batch spike
- 当前没有形成持续发散趋势
- 目前依然没有证据需要停训改代码

补充说明：

- `step-004800/training_state.pt` 中仍是 `global_step = 4800`
- `epoch_id = 1`
- `batch_in_epoch = 800`
- 这只是说明最新已落盘 checkpoint 的状态，不能替代 W&B 的 live step

当前判断更新为：

- 数值面暂时重新回到稳定状态
- 下一关键动作仍是等待 `step-005000` 真正落盘

### 2026-06-26 00:38 UTC 关键里程碑：`step-005000` 已成功落盘

这轮已确认：

- 新 checkpoint：
  - `step-005000`
- 落盘时间：
  - `checkpoint.safetensors` 约 `2026-06-26 00:37:31 UTC`
  - `training_state.pt` 约 `2026-06-26 00:37:33 UTC`

`step-005000/training_state.pt` 内状态为：

- `global_step = 5000`
- `epoch_id = 1`
- `batch_in_epoch = 1000`

当前 checkpoint 目录已变为：

- `step-004800`
- `step-005000`

这说明：

- `step-005000` 已完整写出
- `--max_checkpoints_keep 2` 仍然正常工作
- 旧的 `step-004600` 已被自动淘汰

### 保存边界前后的 W&B 观察

保存前一轮 latest summary：

- `_step = 4997`
- `train/loss_total = 0.12330`
- `train/loss_track_aux = 0.04833`
- `train/loss_box_aux = 0.57835`
- `train/loss_depth_aux = 0.60630`
- `train/object_context_abs_max = 0.42873`
- `train/object_latent_tokens_abs_max = 4.73186`
- `train/grad_norm = 0.90399`
- `train/grad_abs_max = 0.22096`

落盘后 latest summary：

- `_step = 5006`
- `train/loss_total = 0.04440`
- `train/loss_track_aux = 0.02841`
- `train/loss_box_aux = 0.38673`
- `train/loss_depth_aux = 0.02884`
- `train/object_context_abs_max = 0.38548`
- `train/object_latent_tokens_abs_max = 4.63390`
- `train/grad_norm = 1.20954`
- `train/grad_abs_max = 0.29616`

interpretation：

- `4997` 处 `loss_depth_aux` 和 `loss_total` 明显偏高
- 但 `5006` 时已经显著回落
- 这再次支持之前的判断：
  - 近期看到的是 batch-level 波动
  - 不是持续发散

补充判断：

- `grad_norm` 在 `5006` 仍不算低，但没有伴随 `loss_total` / `loss_depth_aux` 继续走坏
- `object_context_abs_max` / `object_latent_tokens_abs_max` 也维持在历史稳定区间内
- 当前不需要停训或改代码

### 当前状态结论

- 训练继续正常推进
- `gpu6,7` 正常承担训练
- `gpu4` 仍未被使用
- validation 目录目前还没有新的触发产物
- 下一关键观察点转为：
  - `step-005200` / `step-005400` 附近是否继续稳定
  - 后续更重要的是 `step-006000` 落盘及其验证触发是否正常

### 2026-06-26 00:40 UTC：`step-005000` 之后的延续状态

本轮再查时：

- 本地 checkpoint 仍是：
  - `step-004800`
  - `step-005000`
- 暂时还没有 `step-005200`
- 训练进程仍在
- `gpu6,7` 继续承担训练
- `gpu4` 没有被使用
- `/data` 可用空间仍约 `5.1G`
- validation 目录仍没有新的触发产物

W&B latest summary 已到：

- `_step = 5042`

当前数值：

- `train/loss_total = 0.04832`
- `train/loss_track_aux = 0.10045`
- `train/loss_box_aux = 0.31071`
- `train/loss_depth_aux = 0.07201`
- `train/object_context_abs_max = 0.43210`
- `train/object_latent_tokens_abs_max = 4.76311`
- `train/grad_norm = 1.20705`
- `train/grad_abs_max = 0.29524`

相对 `step-005000` 前的 `_step = 4997`：

- `loss_total` 明显回落并稳定
- `loss_depth_aux` 远低于 `4997` 那次 spike
- `loss_box_aux` 也明显回落
- `track_aux` 有波动，但还在可接受范围

相对落盘后刚过边界的 `_step = 5006`：

- `loss_total` 仍保持在正常低位
- `loss_depth_aux` 虽有回升，但没有回到异常 spike
- `grad_norm` / `grad_abs_max` 基本持平
- `object_context_abs_max` / `object_latent_tokens_abs_max` 继续处于历史稳定带宽

当前判断：

- `step-005000` 之后训练延续正常
- 目前没有看到新的持续性发散证据
- 暂时不需要调整 loss 权重，也不需要 patch 代码

### 2026-06-26 00:41 UTC：`_step = 5072` 继续稳定

本轮检查时：

- 本地 checkpoint 仍是：
  - `step-004800`
  - `step-005000`
- 训练仍在继续推进
- `gpu6,7` 继续承担训练
- `gpu4` 没有被使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary 已到：

- `_step = 5072`

当前数值：

- `train/loss_total = 0.02919`
- `train/loss_track_aux = 0.03204`
- `train/loss_box_aux = 0.22819`
- `train/loss_depth_aux = 0.03170`
- `train/object_context_abs_max = 0.43304`
- `train/object_latent_tokens_abs_max = 4.77889`
- `train/grad_norm = 0.30850`
- `train/grad_abs_max = 0.07403`

相对上一轮 `_step = 5042`：

- `loss_total` 进一步下降
- `loss_track_aux` 明显回落
- `loss_box_aux` 进一步回落
- `loss_depth_aux` 继续回落
- `grad_norm` / `grad_abs_max` 也明显回落

interpretation：

- `step-005000` 后这几轮不是“勉强稳定”，而是确实在重新回到更干净的数值区间
- 当前最强的证据仍然支持：
  - 没有持续发散
  - 近期看到的是正常 batch 波动

当前判断更新：

- 训练状态良好
- 暂时无需代码修补
- 下一观察点继续看：
  - `step-005200` 是否顺利落盘
  - 再往后 `step-005400` / `step-006000` 是否继续稳定并正常触发验证

### 2026-06-26 00:42 UTC：`_step = 5103`

本轮再查时：

- 本地 checkpoint 仍未新增，还是：
  - `step-004800`
  - `step-005000`
- 训练进程继续存活
- `gpu6,7` 使用正常
- `gpu4` 没有被使用
- validation 目录仍没有新触发产物

W&B latest summary 已到：

- `_step = 5103`

当前数值：

- `train/loss_total = 0.03391`
- `train/loss_track_aux = 0.11508`
- `train/loss_box_aux = 0.20439`
- `train/loss_depth_aux = 0.01966`
- `train/object_context_abs_max = 0.43356`
- `train/object_latent_tokens_abs_max = 4.78825`
- `train/grad_norm = 0.59938`
- `train/grad_abs_max = 0.14666`

相对上一轮 `_step = 5072`：

- `loss_total` 仍处于较低区间
- `loss_box_aux` 继续下降
- `loss_depth_aux` 继续下降
- `loss_track_aux` 有一轮上扬
- `grad_norm` / `grad_abs_max` 也随之抬升，但仍明显低于此前异常 spike 段

interpretation：

- 目前更像 `track_aux` 对应 batch 的普通波动
- 因为总 loss、box、depth 没有同步恶化
- object 相关幅值指标也依旧稳定

当前判断：

- 训练仍然健康
- 暂时不需要修改代码或调整 loss 权重
- 继续等待 `step-005200` 实际落盘

### 2026-06-26 00:44 UTC：`_step = 5128`

本轮检查结果：

- 本地 checkpoint 仍未新增，仍为：
  - `step-004800`
  - `step-005000`
- 训练进程继续运行
- `gpu6,7` 正常工作
- `gpu4` 未被使用
- validation 目录仍没有新的触发结果

W&B latest summary 已到：

- `_step = 5128`

当前数值：

- `train/loss_total = 0.02525`
- `train/loss_track_aux = 0.04148`
- `train/loss_box_aux = 0.18977`
- `train/loss_depth_aux = 0.02129`
- `train/object_context_abs_max = 0.43511`
- `train/object_latent_tokens_abs_max = 4.82510`
- `train/grad_norm = 0.92800`
- `train/grad_abs_max = 0.22615`

相对上一轮 `_step = 5103`：

- `loss_total` 继续下降
- `loss_track_aux` 回落
- `loss_box_aux` 继续回落
- `loss_depth_aux` 仍维持低位
- `grad_norm` / `grad_abs_max` 有所抬升，但并未伴随 loss 恶化

interpretation：

- 当前更像健康训练中的正常梯度波动
- 因为总 loss 和主要辅助项没有同步走坏
- `object_context_abs_max` / `object_latent_tokens_abs_max` 仍在稳定带宽

当前判断更新：

- 训练继续保持良好状态
- 暂时仍无需改代码
- 下一关键点继续等待 `step-005200` 落盘

### 2026-06-26 00:46 UTC：`_step = 5164`

本轮再次检查时：

- 本地 checkpoint 仍未新增，仍是：
  - `step-004800`
  - `step-005000`
- 这与当前 live step 尚未达到 `5200` 是一致的
- 训练进程仍存活
- `gpu6,7` 仍用于训练
- `gpu4` 没有被使用

W&B latest summary 已到：

- `_step = 5164`

当前数值：

- `train/loss_total = 0.05940`
- `train/loss_track_aux = 0.02836`
- `train/loss_box_aux = 0.55736`
- `train/loss_depth_aux = 0.00828`
- `train/object_context_abs_max = 0.39834`
- `train/object_latent_tokens_abs_max = 4.84316`
- `train/grad_norm = 0.31168`
- `train/grad_abs_max = 0.07500`

相对上一轮 `_step = 5128`：

- `loss_total` 有回升
- `loss_box_aux` 明显抬升
- `loss_depth_aux` 反而进一步下降到很低
- `grad_norm` / `grad_abs_max` 也处在较低位置
- `track_aux` 继续回落

interpretation：

- 当前更像 `box_aux` 单项 supervision 的 batch-level 波动
- 因为并没有看到：
  - `depth_aux` 同步恶化
  - 梯度同步放大
  - 总体上下文幅值异常

当前判断：

- 还没有证据表明训练失稳
- 暂时不需要调整代码或权重
- 继续等待 `step-005200` 真正落盘

### 2026-06-26 00:47 UTC：`_step = 5194`

本轮检查时：

- 本地 checkpoint 仍然没有 `step-005200`
- 这与当前 live step 还未真正达到 `5200` 是一致的
- 训练进程仍存活
- `gpu6,7` 仍为训练卡
- `gpu4` 未被使用

W&B latest summary 已到：

- `_step = 5194`

当前数值：

- `train/loss_total = 0.04112`
- `train/loss_track_aux = 0.13238`
- `train/loss_box_aux = 0.20828`
- `train/loss_depth_aux = 0.07052`
- `train/object_context_abs_max = 0.43899`
- `train/object_latent_tokens_abs_max = 4.87799`
- `train/grad_norm = 1.20149`
- `train/grad_abs_max = 0.29676`

相对上一轮 `_step = 5164`：

- `loss_total` 回落
- `loss_box_aux` 明显回落
- `loss_track_aux` 抬升
- `loss_depth_aux` 也有抬升
- `grad_norm` / `grad_abs_max` 同步抬升

interpretation：

- 这更像另一轮 batch-level `track/depth` 波动
- 目前还不能判定为持续异常，因为：
  - `loss_total` 没有同步走坏
  - `loss_box_aux` 反而回落
  - object 幅值指标仍在稳定区间

当前判断更新：

- 训练仍在可接受范围内波动
- 继续等待 `step-005200` 实际落盘再做下一轮判断

### 2026-06-26 00:49 UTC 关键里程碑：`step-005200` 已成功落盘

这一轮已确认：

- 新 checkpoint：
  - `step-005200`
- 落盘时间：
  - `checkpoint.safetensors` 约 `2026-06-26 00:47:25 UTC`
  - `training_state.pt` 约 `2026-06-26 00:47:27 UTC`

`step-005200/training_state.pt` 内状态为：

- `global_step = 5200`
- `epoch_id = 1`
- `batch_in_epoch = 1200`

当前 checkpoint 目录现为：

- `step-005000`
- `step-005200`

这说明：

- `step-005200` 已完整写出
- `--max_checkpoints_keep 2` 继续正常工作
- 旧的 `step-004800` 已被自动淘汰

### 保存边界前后的 W&B 对比

保存前最近一轮：

- `_step = 5194`
- `train/loss_total = 0.04112`
- `train/loss_track_aux = 0.13238`
- `train/loss_box_aux = 0.20828`
- `train/loss_depth_aux = 0.07052`
- `train/object_context_abs_max = 0.43899`
- `train/object_latent_tokens_abs_max = 4.87799`
- `train/grad_norm = 1.20149`
- `train/grad_abs_max = 0.29676`

落盘后 latest summary：

- `_step = 5224`
- `train/loss_total = 0.04881`
- `train/loss_track_aux = 0.09368`
- `train/loss_box_aux = 0.36338`
- `train/loss_depth_aux = 0.03099`
- `train/object_context_abs_max = 0.44174`
- `train/object_latent_tokens_abs_max = 4.92085`
- `train/grad_norm = 0.88511`
- `train/grad_abs_max = 0.21883`

interpretation：

- `5194` 时较高的 `track/depth/grad` 在 `5224` 时已经明显回落
- `loss_box_aux` 有回升，但仍属于单项 batch 波动可以解释的范围
- `loss_total` 仍在正常低位

当前判断更新：

- `step-005200` 边界再次证明训练能稳定跨过保存点
- 目前仍没有证据需要停训或 patch 代码
- 下一关键观察点转为：
  - `step-005400` 是否正常落盘
  - 更后面的 `step-006000` 是否落盘并正常触发 validation

### 2026-06-26 00:51 UTC：`_step = 5265`

本轮再次检查时：

- 本地 checkpoint 仍是：
  - `step-005000`
  - `step-005200`
- 暂时还没有 `step-005400`
- 训练进程仍在继续
- `gpu6,7` 继续承担训练
- validation 目录仍未出现新的触发产物

W&B latest summary 已到：

- `_step = 5265`

当前数值：

- `train/loss_total = 0.03725`
- `train/loss_track_aux = 0.06686`
- `train/loss_box_aux = 0.27378`
- `train/loss_depth_aux = 0.03185`
- `train/object_context_abs_max = 0.44322`
- `train/object_latent_tokens_abs_max = 4.98484`
- `train/grad_norm = 0.88337`
- `train/grad_abs_max = 0.21866`

相对上一轮 `_step = 5224`：

- `loss_total` 继续保持在低位
- `loss_track_aux` 小幅回落
- `loss_box_aux` 有一轮回升，但仍在正常波动范围
- `loss_depth_aux` 基本稳定
- `grad_norm` / `grad_abs_max` 基本持平

interpretation：

- `step-005200` 之后训练延续稳定
- 当前没有看到新的异常趋势
- object 幅值指标虽略有上行，但仍未超出此前稳定带宽

当前判断：

- 继续训练
- 暂时无需改代码
- 下一观察重点继续放在 `step-005400` 落盘，以及后续 `step-006000` validation

### 2026-06-26 00:52 UTC：`_step = 5300`

本轮检查时：

- 本地 checkpoint 仍是：
  - `step-005000`
  - `step-005200`
- 暂时还没有 `step-005400`
- validation 目录仍没有新的触发产物

W&B latest summary 已到：

- `_step = 5300`

当前数值：

- `train/loss_total = 0.04361`
- `train/loss_track_aux = 0.08769`
- `train/loss_box_aux = 0.33563`
- `train/loss_depth_aux = 0.01279`
- `train/object_context_abs_max = 0.44492`
- `train/object_latent_tokens_abs_max = 4.97469`
- `train/grad_norm = 0.59540`
- `train/grad_abs_max = 0.14670`

相对上一轮 `_step = 5265`：

- `loss_total` 有小幅回升，但仍在低位
- `loss_track_aux` 轻微上扬
- `loss_box_aux` 也有回升
- `loss_depth_aux` 继续保持很低
- `grad_norm` / `grad_abs_max` 反而回落

interpretation：

- 当前仍是正常 batch 波动
- 没有看到“loss 上升且梯度同步失控”的组合信号
- object 幅值指标继续在可接受区间

当前判断更新：

- 训练继续稳定推进
- 暂时不需要任何代码或配置介入
- 下一关键点继续等待 `step-005400` 落盘

### 2026-06-26 00:54 UTC：`_step = 5336`

本轮再查时：

- 本地 checkpoint 仍是：
  - `step-005000`
  - `step-005200`
- 暂时还没有 `step-005400`
- validation 目录仍未出现新产物

W&B latest summary 已到：

- `_step = 5336`

当前数值：

- `train/loss_total = 0.04998`
- `train/loss_track_aux = 0.05833`
- `train/loss_box_aux = 0.41529`
- `train/loss_depth_aux = 0.02620`
- `train/object_context_abs_max = 0.44947`
- `train/object_latent_tokens_abs_max = 5.04433`
- `train/grad_norm = 0.87646`
- `train/grad_abs_max = 0.21818`

相对上一轮 `_step = 5300`：

- `loss_total` 有小幅回升
- `loss_track_aux` 回落
- `loss_box_aux` 进一步抬升
- `loss_depth_aux` 仍然低位
- `grad_norm` / `grad_abs_max` 有所抬升但并不异常

interpretation：

- 目前仍更像 box supervision 主导的 batch-level 波动
- 没有看到 total loss、depth、grad 一起恶化的异常组合
- object 幅值指标继续缓慢上升，但仍未越过当前经验稳定带宽

当前判断：

- 训练仍在稳定区间内
- 暂时无需改代码
- 继续等待 `step-005400` 实际落盘

### 2026-06-26 00:56 UTC：`_step = 5366`

本轮再查时：

- 本地 checkpoint 仍未出现 `step-005400`
- 这与当前 live step 仍未跨过 `5400` 是一致的
- 训练进程仍在
- `gpu6,7` 仍作为训练卡使用

W&B latest summary 已到：

- `_step = 5366`

当前数值：

- `train/loss_total = 0.04594`
- `train/loss_track_aux = 0.15984`
- `train/loss_box_aux = 0.20398`
- `train/loss_depth_aux = 0.09559`
- `train/object_context_abs_max = 0.44693`
- `train/object_latent_tokens_abs_max = 5.00883`
- `train/grad_norm = 0.07702`
- `train/grad_abs_max = 0.02500`

相对上一轮 `_step = 5336`：

- `loss_track_aux` 明显抬升
- `loss_depth_aux` 也明显抬升
- `loss_box_aux` 反而回落
- `grad_norm` / `grad_abs_max` 显著下降

interpretation：

- 这轮现象值得继续盯，因为它不是典型的“loss 升高伴随梯度放大”
- 更像某个 batch 上监督误差偏大，但实际回传梯度被限制在较低水平
- 目前还不能单凭这一点判断异常，因为：
  - `loss_total` 仍没有明显失控
  - `box_aux` 没有同步恶化
  - object 幅值指标仍在稳定带宽

当前判断更新：

- 继续观察，不急于介入
- 下一关键点仍是 `step-005400` 真正落盘后的数值表现

### 2026-06-26 00:58 UTC 关键里程碑：`step-005400` 已成功落盘

这一轮已确认：

- 新 checkpoint：
  - `step-005400`
- 落盘时间：
  - `checkpoint.safetensors` 约 `2026-06-26 00:57:18 UTC`
  - `training_state.pt` 约 `2026-06-26 00:57:19 UTC`

`step-005400/training_state.pt` 内状态为：

- `global_step = 5400`
- `epoch_id = 1`
- `batch_in_epoch = 1400`

当前 checkpoint 目录现为：

- `step-005200`
- `step-005400`

这说明：

- `step-005400` 已完整写出
- `--max_checkpoints_keep 2` 继续正常工作
- 旧的 `step-005000` 已被自动淘汰

### 保存边界前后的 W&B 对比

保存前最近一轮：

- `_step = 5366`
- `train/loss_total = 0.04594`
- `train/loss_track_aux = 0.15984`
- `train/loss_box_aux = 0.20398`
- `train/loss_depth_aux = 0.09559`
- `train/object_context_abs_max = 0.44693`
- `train/object_latent_tokens_abs_max = 5.00883`
- `train/grad_norm = 0.07702`
- `train/grad_abs_max = 0.02500`

落盘后 latest summary：

- `_step = 5406`
- `train/loss_total = 0.05125`
- `train/loss_track_aux = 0.05912`
- `train/loss_box_aux = 0.42345`
- `train/loss_depth_aux = 0.02993`
- `train/object_context_abs_max = 0.45089`
- `train/object_latent_tokens_abs_max = 5.06606`
- `train/grad_norm = 0.30890`
- `train/grad_abs_max = 0.07394`

interpretation：

- `5366` 时偏高的 `track/depth` 在 `5406` 已明显回落
- `loss_box_aux` 有回升，但仍更像单项 batch 波动
- `loss_total` 仍在可接受低位
- `grad_norm` / `grad_abs_max` 从极低值回到更正常区间

当前判断更新：

- `step-005400` 边界继续证明训练跨保存点稳定
- 目前依然没有需要停训或 patch 代码的证据
- 下一关键观察点转为：
  - `step-005600` / `step-005800`
  - 更重要的是 `step-006000` 是否落盘并触发 validation

### 2026-06-26 01:00 UTC：`_step = 5447`

本轮再查时：

- 本地 checkpoint 仍是：
  - `step-005200`
  - `step-005400`
- 暂时还没有 `step-005600`
- validation 目录仍未出现新产物

W&B latest summary 已到：

- `_step = 5447`

当前数值：

- `train/loss_total = 0.05872`
- `train/loss_track_aux = 0.13579`
- `train/loss_box_aux = 0.34588`
- `train/loss_depth_aux = 0.10550`
- `train/object_context_abs_max = 0.45263`
- `train/object_latent_tokens_abs_max = 5.11363`
- `train/grad_norm = 0.30726`
- `train/grad_abs_max = 0.07500`

相对上一轮 `_step = 5406`：

- `loss_total` 有回升
- `loss_track_aux` 明显抬升
- `loss_depth_aux` 也明显抬升
- `loss_box_aux` 小幅回落
- `grad_norm` / `grad_abs_max` 维持在较低水平

interpretation：

- 这和之前若干次看到的模式一致：
  - 某些辅助项会在单个 batch 上偏高
  - 但没有伴随梯度失控
  - 也没有把总 loss 拉到异常区间

当前判断：

- 仍然更像 batch-level 波动
- 继续观察，不急于调整代码或 loss 权重
- 下一关键点继续看 `step-005600` 和 `step-006000`

### 2026-06-26 01:01 UTC：`_step = 5482`

本轮检查时：

- 本地 checkpoint 仍是：
  - `step-005200`
  - `step-005400`
- 暂时还没有 `step-005600`
- validation 目录仍无新触发结果

W&B latest summary 已到：

- `_step = 5482`

当前数值：

- `train/loss_total = 0.03827`
- `train/loss_track_aux = 0.05595`
- `train/loss_box_aux = 0.24699`
- `train/loss_depth_aux = 0.07974`
- `train/object_context_abs_max = 0.45042`
- `train/object_latent_tokens_abs_max = 5.15155`
- `train/grad_norm = 0.60486`
- `train/grad_abs_max = 0.15103`

相对上一轮 `_step = 5447`：

- `loss_total` 回落
- `loss_track_aux` 明显回落
- `loss_box_aux` 也回落
- `loss_depth_aux` 虽仍偏高，但较上一轮有所下降
- `grad_norm` / `grad_abs_max` 回到更中性的区间

interpretation：

- 这再次支持“高点主要是 batch-level 波动”的判断
- 当前没有证据显示训练进入持续恶化
- object 幅值指标继续缓慢抬升，但仍在当前经验可接受区间

当前判断：

- 训练继续稳定推进
- 暂时不需要任何代码或配置调整
- 下一关键观察点仍是 `step-005600` / `step-005800` 和 `step-006000` validation

### 2026-06-26 01:04 UTC：`_step = 5528`

本轮检查时：

- 本地 checkpoint 仍是：
  - `step-005200`
  - `step-005400`
- 暂时还没有 `step-005600`
- validation 目录仍没有新触发结果

W&B latest summary 已到：

- `_step = 5528`

当前数值：

- `train/loss_total = 0.04933`
- `train/loss_track_aux = 0.08062`
- `train/loss_box_aux = 0.34015`
- `train/loss_depth_aux = 0.07250`
- `train/object_context_abs_max = 0.45916`
- `train/object_latent_tokens_abs_max = 5.17818`
- `train/grad_norm = 1.27948`
- `train/grad_abs_max = 0.31684`

相对上一轮 `_step = 5482`：

- `loss_total` 有回升
- `loss_track_aux` 有回升
- `loss_box_aux` 有回升
- `loss_depth_aux` 也回升
- `grad_norm` / `grad_abs_max` 明显抬升

interpretation：

- 这一轮需要继续盯，因为它出现了“辅助项与梯度同步上扬”
- 但当前还没有把 `loss_total` 推到异常区间
- object 幅值指标虽然继续上行，但仍未明显越界

当前判断更新：

- 训练仍未显示明确失稳
- 但需要继续观察下一轮是否自然回落
- 下一关键点是：
  - `step-005600` 是否正常落盘
  - 落盘后这些指标是否回到稳定区间

### 2026-06-26 01:06 UTC：`_step = 5569`

本轮检查时：

- 本地 checkpoint 仍是：
  - `step-005200`
  - `step-005400`
- 暂时还没有 `step-005600`
- validation 目录仍无新产物

W&B latest summary 已到：

- `_step = 5569`

当前数值：

- `train/loss_total = 0.05963`
- `train/loss_track_aux = 0.02851`
- `train/loss_box_aux = 0.47574`
- `train/loss_depth_aux = 0.09207`
- `train/object_context_abs_max = 0.41463`
- `train/object_latent_tokens_abs_max = 5.23210`
- `train/grad_norm = 0.86443`
- `train/grad_abs_max = 0.22113`

相对上一轮 `_step = 5528`：

- `loss_total` 有回升
- `loss_track_aux` 明显回落
- `loss_box_aux` 明显抬升
- `loss_depth_aux` 仍偏高
- `grad_norm` / `grad_abs_max` 回落

interpretation：

- 这更像 box/depth 主导的 batch-level 波动
- 不像整体失控，因为梯度没有继续放大
- `object_context_abs_max` 甚至回落了一些

当前判断：

- 继续观察，不急于修改代码
- 下一关键点仍是 `step-005600` 实际落盘，以及落盘后数值是否回稳

### 2026-06-26 01:08 UTC 关键里程碑：`step-005600` 已成功落盘

这一轮已确认：

- 新 checkpoint：
  - `step-005600`
- 落盘时间：
  - `checkpoint.safetensors` 约 `2026-06-26 01:07:11 UTC`
  - `training_state.pt` 约 `2026-06-26 01:07:12 UTC`

`step-005600/training_state.pt` 内状态为：

- `global_step = 5600`
- `epoch_id = 1`
- `batch_in_epoch = 1600`

当前 checkpoint 目录现为：

- `step-005400`
- `step-005600`

这说明：

- `step-005600` 已完整写出
- `--max_checkpoints_keep 2` 继续正常工作
- 旧的 `step-005200` 已被自动淘汰

### 保存边界前后的 W&B 对比

保存前最近一轮：

- `_step = 5569`
- `train/loss_total = 0.05963`
- `train/loss_track_aux = 0.02851`
- `train/loss_box_aux = 0.47574`
- `train/loss_depth_aux = 0.09207`
- `train/object_context_abs_max = 0.41463`
- `train/object_latent_tokens_abs_max = 5.23210`
- `train/grad_norm = 0.86443`
- `train/grad_abs_max = 0.22113`

落盘后 latest summary：

- `_step = 5608`
- `train/loss_total = 0.07565`
- `train/loss_track_aux = 0.08889`
- `train/loss_box_aux = 0.65637`
- `train/loss_depth_aux = 0.01126`
- `train/object_context_abs_max = 0.42782`
- `train/object_latent_tokens_abs_max = 5.38435`
- `train/grad_norm = 1.21038`
- `train/grad_abs_max = 0.30849`

interpretation：

- 这次 `step-005600` 边界没有出现统一方向的恶化
- 更像是：
  - `box_aux` 主导的单项偏高 batch
  - `depth_aux` 同时回到很低
  - 梯度有抬升，但仍未表现为全面失控

当前判断更新：

- 训练仍能稳定跨过保存点
- 但 box-related 波动需要继续盯
- 下一关键观察点转为：
  - `step-005800`
  - 更关键的 `step-006000` 落盘与 validation 触发

### 2026-06-26 01:10 UTC：`_step = 5649`

本轮检查时：

- 本地 checkpoint 仍是：
  - `step-005400`
  - `step-005600`
- 暂时还没有 `step-005800`
- validation 目录仍无新触发结果

W&B latest summary 已到：

- `_step = 5649`

当前数值：

- `train/loss_total = 0.01797`
- `train/loss_track_aux = 0.03552`
- `train/loss_box_aux = 0.09713`
- `train/loss_depth_aux = 0.04706`
- `train/object_context_abs_max = 0.46812`
- `train/object_latent_tokens_abs_max = 5.37048`
- `train/grad_norm = 0.09216`
- `train/grad_abs_max = 0.05000`

相对上一轮 `_step = 5608`：

- `loss_total` 明显回落
- `loss_track_aux` 明显回落
- `loss_box_aux` 大幅回落
- `loss_depth_aux` 虽仍有值，但整体更平稳
- `grad_norm` / `grad_abs_max` 也明显回落

interpretation：

- 这再次支持“单个 batch 上的 box/depth 偏高并不意味着训练失稳”
- 当前从数值上看重新回到了很干净的区间
- 目前没有证据需要修改训练代码或 loss 配置

当前判断：

- 训练继续健康推进
- 下一关键点继续看 `step-005800` 是否正常落盘
- 更重要的是 `step-006000` 与 validation 触发

### 2026-06-26 01:12 UTC：`_step = 5695`

本轮检查时：

- 本地 checkpoint 仍是：
  - `step-005400`
  - `step-005600`
- 暂时还没有 `step-005800`
- validation 目录仍无新触发结果

W&B latest summary 已到：

- `_step = 5695`

当前数值：

- `train/loss_total = 0.02323`
- `train/loss_track_aux = 0.05409`
- `train/loss_box_aux = 0.15516`
- `train/loss_depth_aux = 0.02306`
- `train/object_context_abs_max = 0.47265`
- `train/object_latent_tokens_abs_max = 5.43412`
- `train/grad_norm = 0.91976`
- `train/grad_abs_max = 0.23470`

相对上一轮 `_step = 5649`：

- `loss_total` 继续回落
- `loss_track_aux` 小幅回升但仍低位
- `loss_box_aux` 明显回落
- `loss_depth_aux` 明显回落
- `grad_norm` / `grad_abs_max` 有回升，但没有带来 loss 恶化

interpretation：

- 这进一步说明近期看到的高点主要是 batch-level 波动
- 训练总体仍保持在稳定区间
- 当前没有需要介入修改代码的证据

当前判断：

- 继续训练
- 下一关键点继续看 `step-005800`
- 更关键的是 `step-006000` 是否落盘并触发 validation

### 2026-06-26 01:14 UTC：`_step = 5740`

本轮检查时：

- 训练主进程仍存活：
  - `run_train_v_newtrain_object_heads_only_gpu67.sh`
  - `accelerate launch`
  - 2 个 `train_v_newtrain.py` worker
- 两个 worker 仍在高 CPU 占用运行，未见退出迹象
- GPU 使用仍符合预期：
  - `gpu6 = 42725 / 49140 MiB`
  - `gpu7 = 42725 / 49140 MiB`
  - `gpu4 = 1 / 49140 MiB`
- 没有任何证据表明训练误用了 `gpu4`
- 本地 checkpoint 目录此时仍只有：
  - `step-005400`
  - `step-005600`
- `step-005600` 文件时间仍为：
  - `checkpoint.safetensors`: `2026-06-26 01:07:11 UTC`
  - `training_state.pt`: `2026-06-26 01:07:12 UTC`
- validation 目录暂无新一轮产物；目前最新仍是更早前 `step-004000` 对应的 smoke 验证结果
- `/data` 剩余空间仍只有约 `5.1G`，这是当前最明显的运行风险

W&B latest summary：

- `_step = 5740`

当前数值：

- `train/loss_total = 0.04039`
- `train/loss_track_aux = 0.05873`
- `train/loss_box_aux = 0.28470`
- `train/loss_depth_aux = 0.06046`
- `train/object_context_abs_max = 0.47394`
- `train/object_latent_tokens_abs_max = 5.45538`
- `train/grad_norm = 0.61292`
- `train/grad_abs_max = 0.15586`

相对上一轮 `_step = 5695`：

- `loss_total` 有回升
- `loss_track_aux` 小幅回升
- `loss_box_aux` 回升较明显
- `loss_depth_aux` 也有回升
- `grad_norm` / `grad_abs_max` 处于中等水平，没有出现异常爆炸

interpretation：

- 当前依然更像是 batch-level 波动，不是持续性发散
- `object_context_abs_max` 与 `object_latent_tokens_abs_max` 仍在之前缓慢上升但可接受的带宽内
- 目前最需要继续盯的是：
  - `step-005800` 是否正常落盘
  - `/data` 磁盘空间是否导致 checkpoint 落盘延迟或失败
  - `step-006000` 是否触发 validation

当前判断：

- 训练仍在推进，没有 runtime error 迹象
- 暂不需要修改代码
- 继续监控 checkpoint 与 validation 落盘情况

### 2026-06-26 01:16 UTC：`_step = 5791`

本轮检查时：

- 训练进程仍完整存活：
  - `run_train_v_newtrain_object_heads_only_gpu67.sh`
  - `accelerate launch`
  - 2 个 `train_v_newtrain.py` worker
- GPU 使用仍正确：
  - `gpu6 = 42725 / 49140 MiB`
  - `gpu7 = 42725 / 49140 MiB`
  - `gpu4 = 1 / 49140 MiB`
- `gpu4` 依旧没有被训练使用
- checkpoint 目录此时仍只有：
  - `step-005400`
  - `step-005600`
- validation 目录仍没有新增结果，最新仍停留在更早前 `step-004000` 对应的 smoke 验证输出
- `/data` 剩余空间仍约 `5.1G`

W&B latest summary：

- `_step = 5791`

当前数值：

- `train/loss_total = 0.07403`
- `train/loss_track_aux = 0.07806`
- `train/loss_box_aux = 0.65295`
- `train/loss_depth_aux = 0.00933`
- `train/object_context_abs_max = 0.43239`
- `train/object_latent_tokens_abs_max = 5.49015`
- `train/grad_norm = 0.33208`
- `train/grad_abs_max = 0.07923`

相对上一轮 `_step = 5771`：

- `loss_total` 明显回升
- `loss_track_aux` 小幅回升
- `loss_box_aux` 出现一次较明显 spike
- `loss_depth_aux` 反而回落到较低水平
- `grad_norm` / `grad_abs_max` 没有同步爆炸

interpretation：

- 当前更像是 box 分支在单个 batch 上的波动，不像整体训练发散
- 由于 `_step` 还没到 `5800`，checkpoint 尚未新增暂时是正常现象，不是落盘异常
- 仍需继续盯：
  - `step-005800` 是否准时落盘
  - `step-006000` 是否准时落盘并触发 validation
  - `/data` 剩余空间是否在保存时成为瓶颈

当前判断：

- 训练继续推进
- 暂无 runtime error
- 继续短轮询 `5800` checkpoint

### 2026-06-26 01:18 UTC：`step-005800` 已正常落盘

本轮检查结果：

- checkpoint 目录已滚动为：
  - `step-005600`
  - `step-005800`
- `max_checkpoints_keep = 2` 继续正常工作
- `step-005800` 文件时间：
  - `checkpoint.safetensors`: `2026-06-26 01:17:04 UTC`
  - `training_state.pt`: `2026-06-26 01:17:06 UTC`
- `training_state.pt` 已确认：
  - `global_step = 5800`

同时的 W&B latest summary：

- `_step = 5820`

当前数值：

- `train/loss_total = 0.10855`
- `train/loss_track_aux = 0.07898`
- `train/loss_box_aux = 0.41568`
- `train/loss_depth_aux = 0.59080`
- `train/object_context_abs_max = 0.47127`
- `train/object_latent_tokens_abs_max = 5.62501`
- `train/grad_norm = 0.62873`
- `train/grad_abs_max = 0.15899`

相对上一轮 `_step = 5791`：

- `loss_total` 继续上升
- `loss_track_aux` 变化不大
- `loss_box_aux` 从更高 spike 回落了一些
- `loss_depth_aux` 出现一次明显 spike
- `grad_norm` / `grad_abs_max` 仍未表现出失控迹象

interpretation：

- 当前 checkpoint 保存链路是正常的，不存在“到 step 了但写不出来”的问题
- 这轮异常主要体现在 `depth_aux` 的 batch-level spike，需要继续观察后续是否快速回落
- 目前没有证据表明训练发散，也没有 runtime/code error

下一关键点：

- 继续盯 `step-006000`
- 重点确认：
  - `step-006000` 是否准时落盘
  - validation 是否按 `validation_every_steps = 2000` 正常触发

### 2026-06-26 01:20 UTC：`_step = 5845`

本轮检查时：

- 训练进程仍正常存活，`gpu6/7` 占用稳定
- `gpu4` 仍未被使用
- checkpoint 当前仍是：
  - `step-005600`
  - `step-005800`
- validation 目录仍没有新输出，符合“尚未到 `step-006000`”的预期
- `/data` 可用空间仍约 `5.1G`

W&B latest summary：

- `_step = 5845`

当前数值：

- `train/loss_total = 0.05757`
- `train/loss_track_aux = 0.03707`
- `train/loss_box_aux = 0.53372`
- `train/loss_depth_aux = 0.00492`
- `train/object_context_abs_max = 0.43472`
- `train/object_latent_tokens_abs_max = 5.44884`
- `train/grad_norm = 0.31446`
- `train/grad_abs_max = 0.07805`

相对上一轮 `_step = 5820`：

- `loss_total` 回落
- `loss_track_aux` 明显回落
- `loss_box_aux` 仍偏高，但比极端 spike 更可控
- `loss_depth_aux` 从 `0.59080` 快速回落到很低值
- `grad_norm` / `grad_abs_max` 继续保持温和

interpretation：

- 上一轮 `depth_aux` 的异常更像单个 batch 波动，而不是持续性发散
- 当前整体重新回到更稳定的区间
- 仍需继续重点盯 `6000` 这个 validation 触发点

### 2026-06-26 01:30 UTC：定位 `step-006000` 后训练退出原因

关键事实：

- `step-006000` 已正常落盘：
  - `checkpoint.safetensors`: `2026-06-26 01:27:03 UTC`
  - `training_state.pt`: `2026-06-26 01:27:04 UTC`
  - `training_state.pt` 中 `global_step = 6000`
- 当时训练主进程已经退出，`gpu6/7` 也空闲
- validation 不是“没触发”，而是已经触发并失败

证据：

- 失败标记存在：
  - `test/_benchmark_runtime/validation100_vbench/step-006000/benchmark.failed.json`
- 同类失败在更早前已经出现过：
  - `step-002000`
  - `step-004000`
  - `step-006000`

`step-006000` 的直接报错：

- `run_validation_vbench.py` 启动 generation 子进程时，调用了：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/batch_eval_lora.py`
- 但这个路径不存在
- 实际存在的脚本是：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/batch_eval_lora.py`

结论：

- 训练退出的根因不是模型训练发散、不是显存、不是 checkpoint 保存失败
- 根因是 validation 子进程脚本路径配置错误，导致每次 validation 触发都会返回非零退出码

代码修复：

- 修改 `train_v_newtrain.py`
  - 将 `DEFAULT_BENCHMARK_SCRIPT` 从错误的仓库根目录路径
    - `code_vjepa_vggt/batch_eval_lora.py`
  - 改为正确的：
    - `code_vjepa_vggt/train0419_reference/batch_eval_lora.py`
- 同时修改启动脚本 `run_train_v_newtrain_object_heads_only_gpu67.sh`
  - 显式增加：
    - `--benchmark_script_path /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0419_reference/batch_eval_lora.py`

恢复动作：

- 已从最新 checkpoint 重新启动训练：
  - `resume_from = step-006000`
- 重新启动后确认：
  - `accelerate launch` 正常
  - 两个 `train_v_newtrain.py` worker 正常
  - `gpu6/7` 已重新占用
  - `gpu4` 未使用
  - W&B run 状态仍为 `running`

当前判断：

- 问题已定位并修复
- 接下来继续观察恢复后的新 step、loss/梯度走势，以及下一轮 validation 是否真正恢复正常

### 2026-06-26 01:33 UTC：修复后恢复训练状态确认

恢复训练后第一轮检查结果：

- 训练进程已稳定恢复：
  - launcher bash 存活
  - `accelerate launch` 存活
  - 2 个 `train_v_newtrain.py` worker 存活
- 两个 worker CPU 占用重新拉高，说明不是空转壳进程
- GPU 使用恢复到训练态：
  - `gpu6 = 40899 / 49140 MiB`
  - `gpu7 = 40881 / 49140 MiB`
  - `gpu5 = 1 / 49140 MiB`
  - `gpu4 = 1 / 49140 MiB`
- 说明：
  - 训练再次只跑在 `gpu6/7`
  - `gpu4` 仍未使用
  - validation 目前还没有再次启动，占用 `gpu5` 也还没出现，这符合“刚从 `step-006000` 恢复”的预期

checkpoint 当前仍是：

- `step-005800`
- `step-006000`

W&B：

- run 状态仍是 `running`
- summary 暂时还停在恢复前最后一次上报：
  - `_step = 5999`
  - `train/loss_total = 0.05931`
  - `train/loss_track_aux = 0.05853`
  - `train/loss_box_aux = 0.53289`
  - `train/loss_depth_aux = 0.00170`
  - `train/object_context_abs_max = 0.45309`
  - `train/object_latent_tokens_abs_max = 5.70206`
  - `train/grad_norm = 0.32157`
  - `train/grad_abs_max = 0.07853`

interpretation：

- 目前更像是 W&B summary 刷新滞后，而不是训练再次卡死
- 因为进程、CPU、GPU 都已经显示训练真正恢复
- 下一步继续等 W&B 跨过 `6000`，并盯后续 checkpoint / validation

### 2026-06-26 01:36 UTC：确认恢复后是新的 W&B run

后续排查发现：

- 恢复训练时并不是继续沿用旧 run `yaxj219k`
- 实际新启动了一个新的 W&B run：
  - run id: `qberfq1r`
  - run name: `pybullet0625_diffsynth_object_heads_only_gpu67`
- 旧 run `yaxj219k` 因此前一次 validation 崩溃被标记为 `crashed`

证据：

- launcher 输出中明确出现：
  - `wandb: setting up run qberfq1r`
  - `View run at https://wandb.ai/875222004-gy/vjepa_vggt_wan/runs/qberfq1r`
- 当前训练进程日志里已经继续推进：
  - `global_step 6001 ... 6060`

因此：

- 恢复后监控必须切换到新的 W&B run `qberfq1r`
- 不能再用旧 run `yaxj219k` 判断当前训练是否存活

当前新 run 最新摘要：

- `_step = 6060`
- `train/loss_total = 0.13358`
- `train/loss_track_aux = 0.13726`
- `train/loss_box_aux = 0.70569`
- `train/loss_depth_aux = 0.49285`
- `train/object_context_abs_max = 0.40088`
- `train/object_latent_tokens_abs_max = 4.15254`
- `train/grad_norm = 1.05652`
- `train/grad_abs_max = 0.26648`

初步判断：

- 恢复后的训练链路已经重新打通，至少推进到了 `6060`
- 这一轮 `box/depth` 与总 loss 偏高，需要继续看后续 summary 是否回落
- 暂时还不能判断为发散，更像刚恢复后的一段波动窗口

### 2026-06-26 01:38 UTC：恢复后第二轮 summary 已回落

继续监控新 run `qberfq1r`：

- 训练进程仍稳定存活：
  - `accelerate launch` 存活
  - 2 个 `train_v_newtrain.py` worker 持续高 CPU 占用
- 当前训练已经继续推进到：
  - `_step = 6107`
- checkpoint 目录此时仍是：
  - `step-005800`
  - `step-006000`
- 还没到下一次保存点 `6200`，因此没有新 checkpoint 是正常现象

W&B latest summary（新 run `qberfq1r`）：

- `_step = 6107`
- `train/loss_total = 0.03980`
- `train/loss_track_aux = 0.14785`
- `train/loss_box_aux = 0.18612`
- `train/loss_depth_aux = 0.06400`
- `train/object_context_abs_max = 0.39982`
- `train/object_latent_tokens_abs_max = 4.33306`
- `train/grad_norm = 0.69935`
- `train/grad_abs_max = 0.16934`

相对上一轮 `_step = 6060`：

- `loss_total` 明显回落
- `loss_box_aux` 从 `0.70569` 明显回落到 `0.18612`
- `loss_depth_aux` 从 `0.49285` 明显回落到 `0.06400`
- `grad_norm` / `grad_abs_max` 也同步回落
- `loss_track_aux` 仍偏高，但暂未继续恶化

interpretation：

- 这更支持“恢复后最初几步是过渡波动”的判断
- 当前没有看到持续发散证据
- 目前最需要继续盯的是：
  - `step-006200` 是否正常落盘
  - 后续几次 summary 中 `loss_track_aux` 是否也回到更低区间
  - 下一次 validation 触发时是否不再因为脚本路径问题失败

### 2026-06-26 01:39 UTC：新 run 继续回稳

当前训练状态：

- 训练进程继续稳定运行
- `gpu6/7` 维持高显存占用：
  - `gpu6 = 42725 / 49140 MiB`
  - `gpu7 = 42707 / 49140 MiB`
- `gpu4` 依旧未使用
- `/data` 可用空间仍约 `5.1G`

checkpoint：

- 当前仍是：
  - `step-005800`
  - `step-006000`
- 还没到 `step-006200` 的保存点，因此此时没有新 checkpoint 仍属正常

W&B latest summary（新 run `qberfq1r`）：

- `_step = 6133`
- `train/loss_total = 0.04547`
- `train/loss_track_aux = 0.10049`
- `train/loss_box_aux = 0.34437`
- `train/loss_depth_aux = 0.00982`
- `train/object_context_abs_max = 0.40090`
- `train/object_latent_tokens_abs_max = 4.44604`
- `train/grad_norm = 0.44582`
- `train/grad_abs_max = 0.08644`

相对上一轮 `_step = 6107`：

- `loss_total` 小幅回升但仍处低位
- `loss_track_aux` 从 `0.14785` 回落到 `0.10049`
- `loss_box_aux` 有回升，但仍明显低于 `6060` 时的高点
- `loss_depth_aux` 再次回到很低水平
- `grad_norm` / `grad_abs_max` 继续回落

interpretation：

- 当前恢复后的训练整体仍在往稳定区间收敛
- `track_aux` 也开始回落，这是一个积极信号
- 暂时没有新的代码错误或发散证据
- 继续盯 `step-006200` checkpoint，以及后续 validation 触发点

### 2026-06-26 01:41 UTC：`track_aux` 继续回落

当前训练状态：

- 训练进程继续稳定运行
- `gpu6/7` 当前利用率已到 `100%`
- 当前 checkpoint 仍是：
  - `step-005800`
  - `step-006000`
- 距离下一次保存点 `6200` 仍差一点，因此无新 checkpoint 依然正常

W&B latest summary（新 run `qberfq1r`）：

- `_step = 6159`
- `train/loss_total = 0.04303`
- `train/loss_track_aux = 0.06998`
- `train/loss_box_aux = 0.34290`
- `train/loss_depth_aux = 0.01739`
- `train/object_context_abs_max = 0.40451`
- `train/object_latent_tokens_abs_max = 4.46357`
- `train/grad_norm = 0.69853`
- `train/grad_abs_max = 0.17150`

相对上一轮 `_step = 6133`：

- `loss_total` 小幅回落
- `loss_track_aux` 从 `0.10049` 继续回落到 `0.06998`
- `loss_box_aux` 基本持平，仍在中等波动区间
- `loss_depth_aux` 仍保持在很低水平
- `object_context_abs_max` / `object_latent_tokens_abs_max` 变化平稳

interpretation：

- 恢复后的训练稳定性继续得到支持
- 当前最活跃的波动项主要还是 `box_aux`
- 但 `track_aux`、`depth_aux` 和总 loss 都没有恶化
- 继续盯 `step-006200` 落盘和后续 summary

### 2026-06-26 01:42 UTC：`loss_total / track_aux / box_aux` 继续回落

当前训练状态：

- 训练进程继续稳定运行
- `gpu6/7` 持续高利用率：
  - `gpu6` 利用率 `100%`
  - `gpu7` 利用率 `83%`
- checkpoint 仍是：
  - `step-005800`
  - `step-006000`
- 还没到 `6200` 落盘点，因此此时没有新 checkpoint 依然正常

W&B latest summary（新 run `qberfq1r`）：

- `_step = 6180`
- `train/loss_total = 0.02895`
- `train/loss_track_aux = 0.04988`
- `train/loss_box_aux = 0.16259`
- `train/loss_depth_aux = 0.07701`
- `train/object_context_abs_max = 0.39831`
- `train/object_latent_tokens_abs_max = 4.48743`
- `train/grad_norm = 0.73146`
- `train/grad_abs_max = 0.17723`

相对上一轮 `_step = 6159`：

- `loss_total` 继续回落
- `loss_track_aux` 从 `0.06998` 继续回落到 `0.04988`
- `loss_box_aux` 从 `0.34290` 明显回落到 `0.16259`
- `loss_depth_aux` 有小幅回升，但仍处于可接受区间
- `object_context_abs_max` 基本稳定
- `object_latent_tokens_abs_max` 变化平稳

interpretation：

- 恢复后的训练继续朝更稳定的区间收敛
- 当前最需要继续确认的是：
  - `step-006200` 是否正常落盘
  - 下一轮 summary 是否维持这条回落趋势

### 2026-06-26 01:43 UTC：`step-006200` 已正常落盘

checkpoint 检查结果：

- checkpoint 目录已滚动为：
  - `step-006000`
  - `step-006200`
- `max_checkpoints_keep = 2` 继续正常工作
- `step-006200` 文件时间：
  - `checkpoint.safetensors`: `2026-06-26 01:42:57 UTC`
  - `training_state.pt`: `2026-06-26 01:42:58 UTC`
- `training_state.pt` 已确认：
  - `global_step = 6200`
  - `epoch_id = 2`
  - `batch_in_epoch = 200`

同时的 W&B latest summary（新 run `qberfq1r`）：

- `_step = 6200`
- `train/loss_total = 0.05349`
- `train/loss_track_aux = 0.18137`
- `train/loss_box_aux = 0.20895`
- `train/loss_depth_aux = 0.14462`
- `train/object_context_abs_max = 0.40491`
- `train/object_latent_tokens_abs_max = 4.51407`
- `train/grad_norm = 1.36410`
- `train/grad_abs_max = 0.36118`

相对上一轮 `_step = 6180`：

- `loss_total` 有一波回升
- `loss_track_aux` 明显抬升
- `loss_box_aux` 小幅回升
- `loss_depth_aux` 也有明显回升
- `grad_norm` / `grad_abs_max` 同步抬升

interpretation：

- 这更像一次 batch-level 反弹，而不是训练链路故障
- 证据是：
  - checkpoint 落盘完全正常
  - 训练进程仍稳定存活
  - `gpu6/7` 仍持续高利用率
- 但这一轮值得继续跟紧，重点看：
  - `6200` 之后的 1-2 个 summary 是否快速回落
  - 如果 `track_aux` / `depth_aux` 连续维持高位，再考虑进一步排查 batch 特征或 loss 权重

### 2026-06-26 01:45 UTC：`6200` 后反弹已快速回落

继续监控新 run `qberfq1r`：

- 训练进程继续稳定存活
- 两个 worker 持续高 CPU 占用
- checkpoint 目录仍为：
  - `step-006000`
  - `step-006200`
- `6200` 之后尚未到下一次保存点，因此无新 checkpoint 正常

W&B latest summary：

- `_step = 6246`
- `train/loss_total = 0.02806`
- `train/loss_track_aux = 0.02879`
- `train/loss_box_aux = 0.22609`
- `train/loss_depth_aux = 0.02568`
- `train/object_context_abs_max = 0.39465`
- `train/object_latent_tokens_abs_max = 4.48359`
- `train/grad_norm = 0.72095`
- `train/grad_abs_max = 0.17659`

相对上一轮 `_step = 6200`：

- `loss_total` 明显回落
- `loss_track_aux` 从 `0.18137` 快速回落到 `0.02879`
- `loss_box_aux` 小幅回升到 `0.22609`，但仍处于可接受波动范围
- `loss_depth_aux` 从 `0.14462` 快速回落到 `0.02568`
- `grad_norm` / `grad_abs_max` 也同步回落

interpretation：

- 这进一步支持“`6200` 处看到的是 batch-level 反弹，不是持续发散”
- 当前恢复后的训练轨迹仍然健康
- 继续按原计划观察后续 checkpoint 与下一次 validation 触发

### 2026-06-26 01:46 UTC：`6246 -> 6261` 出现温和回弹

当前状态：

- 训练进程继续稳定运行
- `gpu6/7` 仍保持高占用
- checkpoint 目录当前仍为：
  - `step-006000`
  - `step-006200`
- validation 运行目录没有新增文件，当前仍只有更早前：
  - `step-002000/benchmark.failed.json`
  - `step-004000/benchmark.failed.json`
  - `step-006000/benchmark.failed.json`
- 这符合“修复后训练尚未推进到下一次 validation 触发点”的预期

W&B latest summary（新 run `qberfq1r`）：

- `_step = 6261`
- `train/loss_total = 0.03907`
- `train/loss_track_aux = 0.12059`
- `train/loss_box_aux = 0.20990`
- `train/loss_depth_aux = 0.06021`
- `train/object_context_abs_max = 0.40423`
- `train/object_latent_tokens_abs_max = 4.59714`
- `train/grad_norm = 0.99660`
- `train/grad_abs_max = 0.26037`

相对上一轮 `_step = 6246`：

- `loss_total` 有温和回升
- `loss_track_aux` 从 `0.02879` 回升到 `0.12059`
- `loss_box_aux` 仍在中等波动区间，变化不大
- `loss_depth_aux` 从 `0.02568` 回升到 `0.06021`
- `grad_norm` / `grad_abs_max` 也有同步回升

interpretation：

- 当前仍更像 batch-level 波动，而不是新的系统性异常
- 依据是：
  - 训练进程和 GPU 状态正常
  - 没有新的 runtime error
  - checkpoint 链路正常
- 继续重点观察：
  - 下一轮 summary 是否再次回落
  - 下一次 validation 触发时是否真正不再报路径错误

### 2026-06-26 01:47 UTC：`6261 -> 6286` 再次回到低位

当前状态：

- 训练进程继续稳定运行
- `gpu6/7` 仍保持高占用
- checkpoint 当前仍为：
  - `step-006000`
  - `step-006200`
- validation 运行目录仍没有新增文件，说明尚未到新的 validation 触发点

W&B latest summary（新 run `qberfq1r`）：

- `_step = 6286`
- `train/loss_total = 0.02788`
- `train/loss_track_aux = 0.03106`
- `train/loss_box_aux = 0.22863`
- `train/loss_depth_aux = 0.01906`
- `train/object_context_abs_max = 0.39433`
- `train/object_latent_tokens_abs_max = 4.61276`
- `train/grad_norm = 0.43361`
- `train/grad_abs_max = 0.08371`

相对上一轮 `_step = 6261`：

- `loss_total` 明显回落
- `loss_track_aux` 从 `0.12059` 快速回落到 `0.03106`
- `loss_box_aux` 小幅回升到 `0.22863`，但仍处于中等波动范围
- `loss_depth_aux` 从 `0.06021` 回落到 `0.01906`
- `grad_norm` / `grad_abs_max` 也同步回落

interpretation：

- 这进一步支持最近看到的是短时 batch-level 波动，不是持续性异常
- 当前恢复后的训练轨迹仍保持健康
- 继续按原计划盯：
  - 下一次 checkpoint 落盘
  - 下一次 validation 触发是否真正跑通

### 2026-06-26 01:48 UTC：`6306` 处出现一次 `depth_aux / box_aux` 抬升

当前状态：

- 训练进程继续稳定运行
- `gpu6/7` 仍保持占用
- checkpoint 目录当前仍为：
  - `step-006000`
  - `step-006200`
- validation 目录仍没有新增 step，说明还没到下一次 validation 触发点

W&B latest summary（新 run `qberfq1r`）：

- `_step = 6306`
- `train/loss_total = 0.09157`
- `train/loss_track_aux = 0.02752`
- `train/loss_box_aux = 0.41812`
- `train/loss_depth_aux = 0.47009`
- `train/object_context_abs_max = 0.39949`
- `train/object_latent_tokens_abs_max = 4.63728`
- `train/grad_norm = 0.45209`
- `train/grad_abs_max = 0.08990`

相对上一轮 `_step = 6286`：

- `loss_total` 明显回升
- `loss_track_aux` 继续保持低位，甚至略有回落
- `loss_box_aux` 出现明显抬升
- `loss_depth_aux` 出现一次较大的 spike
- `grad_norm` / `grad_abs_max` 没有同步出现异常爆炸

interpretation：

- 当前更像是 `box/depth` 相关 supervision 在单个 batch 上的波动
- 之所以暂时不判断为系统性异常，主要因为：
  - `track_aux` 没有同步恶化
  - 梯度统计没有爆炸
  - 训练进程、GPU、checkpoint 链路都正常
- 下一步继续紧盯后续 1-2 个 summary：
  - 如果快速回落，则仍按 batch-level 波动处理
  - 如果 `depth_aux` / `box_aux` 连续维持高位，再进一步排查具体 batch 或 loss 权重

### 2026-06-26 01:49 UTC：`6306` 的 `depth_aux` spike 已回落

当前状态：

- 训练进程继续稳定运行
- checkpoint 目录仍为：
  - `step-006000`
  - `step-006200`
- validation 运行目录当前仍没有新 step 文件，说明还没推进到下一次 validation 触发点

W&B latest summary（新 run `qberfq1r`）：

- `_step = 6331`
- `train/loss_total = 0.04471`
- `train/loss_track_aux = 0.04650`
- `train/loss_box_aux = 0.37835`
- `train/loss_depth_aux = 0.02227`
- `train/object_context_abs_max = 0.40514`
- `train/object_latent_tokens_abs_max = 4.76440`
- `train/grad_norm = 0.44085`
- `train/grad_abs_max = 0.08697`

相对上一轮 `_step = 6306`：

- `loss_total` 明显回落
- `loss_track_aux` 从 `0.02752` 小幅回升，但仍处低位
- `loss_box_aux` 从 `0.41812` 回落到 `0.37835`
- `loss_depth_aux` 从 `0.47009` 快速回落到 `0.02227`
- `grad_norm` / `grad_abs_max` 继续维持低位

interpretation：

- 这进一步支持 `6306` 看到的是一次短时 `depth_aux` / `box_aux` 波动，而不是系统性异常
- 当前最需要继续盯的是：
  - 下一个 checkpoint 落盘
  - 下一次 validation 触发时是否真正不再失败

### 2026-06-26 01:50 UTC：loss 进一步降低，但梯度出现一次抬升

当前状态：

- 训练进程继续稳定运行
- `gpu6/7` 仍保持高占用
- checkpoint 当前仍为：
  - `step-006000`
  - `step-006200`
- validation 运行目录仍没有新增 step 文件，说明尚未到新的 validation 触发点

W&B latest summary（新 run `qberfq1r`）：

- `_step = 6357`
- `train/loss_total = 0.01700`
- `train/loss_track_aux = 0.03174`
- `train/loss_box_aux = 0.10273`
- `train/loss_depth_aux = 0.03553`
- `train/object_context_abs_max = 0.39600`
- `train/object_latent_tokens_abs_max = 4.68871`
- `train/grad_norm = 1.55033`
- `train/grad_abs_max = 0.42479`

相对上一轮 `_step = 6331`：

- `loss_total` 继续明显回落
- `loss_track_aux` 仍保持低位
- `loss_box_aux` 从 `0.37835` 明显回落到 `0.10273`
- `loss_depth_aux` 仍在较低区间
- 但 `grad_norm` / `grad_abs_max` 出现一次明显抬升

interpretation：

- 当前现象更像“loss 很低，但某个 batch/参数子集上出现了局部梯度尖峰”
- 之所以暂时不判断为系统性异常，主要因为：
  - 总 loss 反而更低
  - `track/box/depth` 三个主要 loss 项没有同步恶化
  - 训练进程、GPU、checkpoint 链路都正常
- 下一步继续重点看：
  - 后续 1-2 个 summary 中 `grad_norm` / `grad_abs_max` 是否回落
  - 如果梯度连续高位，而 loss 仍低，再考虑排查是否有少数参数或稀有 batch 导致的尖峰

### 2026-06-26 01:52 UTC：`box_aux` 与梯度再次出现一轮抬升

当前状态：

- 训练进程继续稳定运行
- `gpu6/7` 仍保持占用
- checkpoint 目录当前仍为：
  - `step-006000`
  - `step-006200`
- validation 目录仍没有新增 step 结果，说明尚未到新的 validation 触发点

W&B latest summary（新 run `qberfq1r`）：

- `_step = 6382`
- `train/loss_total = 0.05414`
- `train/loss_track_aux = 0.01375`
- `train/loss_box_aux = 0.49380`
- `train/loss_depth_aux = 0.03387`
- `train/object_context_abs_max = 0.40697`
- `train/object_latent_tokens_abs_max = 4.79206`
- `train/grad_norm = 1.31052`
- `train/grad_abs_max = 0.35687`

相对上一轮 `_step = 6357`：

- `loss_total` 有回升
- `loss_track_aux` 继续保持低位
- `loss_box_aux` 从 `0.10273` 明显抬升到 `0.49380`
- `loss_depth_aux` 仍在较低区间
- `grad_norm` / `grad_abs_max` 从上一轮的尖峰后再次抬升

interpretation：

- 当前更像 `box_aux` 主导的一次 batch-level 波动
- 暂时不判断为系统性异常，原因是：
  - `track_aux` 仍然很低
  - `depth_aux` 没有同步恶化
  - 训练进程、GPU、checkpoint、validation 链路都正常
- 下一步继续看：
  - 后续 1-2 个 summary 中 `box_aux` 与梯度是否再次回落
  - 如果 `box_aux` 持续高位，再考虑进一步排查 box supervision 对应 batch

### 2026-06-26 01:53 UTC：`step-006400` 已正常落盘，box 波动仍局部存在

checkpoint 检查结果：

- checkpoint 目录已滚动为：
  - `step-006200`
  - `step-006400`
- `max_checkpoints_keep = 2` 继续正常
- `step-006400` 文件时间：
  - `checkpoint.safetensors`: `2026-06-26 01:52:52 UTC`
  - `training_state.pt`: `2026-06-26 01:52:54 UTC`
- `training_state.pt` 已确认：
  - `global_step = 6400`
  - `epoch_id = 2`
  - `batch_in_epoch = 400`

同时的 W&B latest summary（新 run `qberfq1r`）：

- `_step = 6406`
- `train/loss_total = 0.04889`
- `train/loss_track_aux = 0.02709`
- `train/loss_box_aux = 0.45118`
- `train/loss_depth_aux = 0.01060`
- `train/object_context_abs_max = 0.41148`
- `train/object_latent_tokens_abs_max = 4.80630`
- `train/grad_norm = 0.32813`
- `train/grad_abs_max = 0.07500`

相对上一轮 `_step = 6382`：

- `loss_total` 略有回落
- `loss_track_aux` 继续低位
- `loss_box_aux` 仍维持在较高波动区间
- `loss_depth_aux` 已回到很低水平
- `grad_norm` / `grad_abs_max` 明显回落

interpretation：

- 当前 box 分支的波动仍然存在，但它没有拖着整体梯度或其他 loss 一起失控
- 这更支持“主要是局部 box supervision batch 波动，而不是系统性训练异常”
- 继续按原计划盯：
  - 后续 summary 中 `box_aux` 是否再次回落
  - 下一次 validation 触发是否真正跑通

### 2026-06-26 01:55 UTC：`box_aux` 再次回落

当前状态：

- 训练进程继续稳定运行
- checkpoint 当前仍为：
  - `step-006200`
  - `step-006400`
- validation 运行目录仍没有新增 step 结果，说明还没到下一次 validation 触发点

W&B latest summary（新 run `qberfq1r`）：

- `_step = 6442`
- `train/loss_total = 0.03841`
- `train/loss_track_aux = 0.09125`
- `train/loss_box_aux = 0.28407`
- `train/loss_depth_aux = 0.00881`
- `train/object_context_abs_max = 0.40891`
- `train/object_latent_tokens_abs_max = 4.80586`
- `train/grad_norm = 0.43800`
- `train/grad_abs_max = 0.08717`

相对上一轮 `_step = 6406`：

- `loss_total` 回落
- `loss_track_aux` 有回升，但仍处于可接受区间
- `loss_box_aux` 从 `0.45118` 回落到 `0.28407`
- `loss_depth_aux` 继续保持很低
- `grad_norm` / `grad_abs_max` 仍然保持低位

interpretation：

- 这进一步支持当前主要是 box 分支的 batch-level 波动，而不是整体训练异常
- 目前最需要继续盯的是：
  - 下一次 checkpoint 落盘
  - 下一次 validation 触发时是否真正跑通

### 2026-06-26 01:56 UTC：`6468` 出现一次 `depth_aux + box_aux` 联合抬升

当前状态：

- 训练进程继续稳定运行
- checkpoint 当前仍为：
  - `step-006200`
  - `step-006400`
- validation 目录仍没有新增 step 文件，说明还没推进到新的 validation 触发点

W&B latest summary（新 run `qberfq1r`）：

- `_step = 6468`
- `train/loss_total = 0.10559`
- `train/loss_track_aux = 0.04192`
- `train/loss_box_aux = 0.43343`
- `train/loss_depth_aux = 0.58057`
- `train/object_context_abs_max = 0.40474`
- `train/object_latent_tokens_abs_max = 4.79340`
- `train/grad_norm = 0.97318`
- `train/grad_abs_max = 0.25753`

相对上一轮 `_step = 6442`：

- `loss_total` 明显回升
- `loss_track_aux` 仍处低位
- `loss_box_aux` 从 `0.28407` 回升到 `0.43343`
- `loss_depth_aux` 从很低位置抬升到 `0.58057`
- `grad_norm` / `grad_abs_max` 也有同步回升

interpretation：

- 当前更像一次 `box + depth` 共同参与的 batch-level 反弹
- 暂时仍不判断为系统性异常，原因是：
  - `track_aux` 仍然较低
  - 训练进程、GPU、checkpoint 链路都正常
  - 过去几次类似抬升都在后续 summary 里快速回落
- 但这轮值得更紧盯：
  - 如果后续 1-2 个 summary 不回落，就需要进一步排查对应 batch / depth & box supervision

### 2026-06-26 01:57 UTC：`6468` 的 `box + depth` 抬升已回落

当前状态：

- 训练进程继续稳定运行
- checkpoint 当前仍为：
  - `step-006200`
  - `step-006400`
- validation 目录仍没有新增 step 文件，说明还没推进到新的 validation 触发点

W&B latest summary（新 run `qberfq1r`）：

- `_step = 6493`
- `train/loss_total = 0.02702`
- `train/loss_track_aux = 0.05301`
- `train/loss_box_aux = 0.16863`
- `train/loss_depth_aux = 0.04856`
- `train/object_context_abs_max = 0.39215`
- `train/object_latent_tokens_abs_max = 4.66564`
- `train/grad_norm = 1.04839`
- `train/grad_abs_max = 0.25194`

相对上一轮 `_step = 6468`：

- `loss_total` 明显回落
- `loss_track_aux` 小幅回升，但仍处于低位
- `loss_box_aux` 从 `0.43343` 明显回落到 `0.16863`
- `loss_depth_aux` 从 `0.58057` 明显回落到 `0.04856`
- `grad_norm` / `grad_abs_max` 也回落

interpretation：

- 这进一步支持 `6468` 看到的是一次短时 `box + depth` 波动，而不是系统性异常
- 当前训练轨迹仍然健康
- 继续按原计划盯：
  - 下一个 checkpoint 落盘
  - 下一次 validation 触发是否真正跑通

### 2026-06-26 02:03 UTC：确认坏卡 `gpu4` 继续禁用

当前核对结果：

- 当前活跃训练进程仍是：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_heads_only_gpu67.sh`
- 实际训练命令仍固定：
  - `CUDA_VISIBLE_DEVICES=6,7`
- validation / benchmark 仍固定：
  - `--benchmark_cuda_visible_devices 5`
- `nvidia-smi` 当前显存占用显示：
  - `gpu6 = 42725 / 49140 MiB`
  - `gpu7 = 42707 / 49140 MiB`
  - `gpu5 = 1 / 49140 MiB`
  - `gpu4 = 1 / 49140 MiB`

结论：

- 当前训练实际只跑在 `gpu6,7`
- validation 仍预期跑在 `gpu5`
- 没有任何迹象表明当前 run 使用了坏卡 `gpu4`
- 后续继续保持这个约束：
  - 不使用 `gpu4`
  - 训练只用 `gpu6,7`
  - validation 只用 `gpu5`

额外状态：

- checkpoint 目前仍只有：
  - `step-006200`
  - `step-006400`
- validation runtime 目录目前还没有新的 `step-008000` 相关文件
- `/data` 当前剩余空间约 `5.1G`，磁盘空间仍是当前最大的运行风险

### 2026-06-26 02:05 UTC：`step-006600` 已成功落盘，训练继续健康推进

当前核对结果：

- checkpoint 目录已经推进到：
  - `step-006400`
  - `step-006600`
- `step-006600/training_state.pt` 已核对：
  - `global_step = 6600`
  - `epoch_id = 2`
  - `batch_in_epoch = 600`
- 旧的 `step-006200` 已经不在 checkpoint 目录中

这说明：

- `--save_steps 200` 仍正常生效
- `--max_checkpoints_keep 2` 也仍正常生效
- checkpoint 轮转链路目前是健康的

W&B 当前 latest summary（run `qberfq1r`）：

- `_step = 6614`
- `train/loss_total = 0.04405`
- `train/loss_track_aux = 0.10337`
- `train/loss_box_aux = 0.24117`
- `train/loss_depth_aux = 0.09601`
- `train/grad_norm = 0.51208`
- `train/grad_abs_max = 0.05000`
- `train/object_context_abs_max = 0.38902`
- `train/object_latent_tokens_abs_max = 4.67589`

补充看最近一段 history：

- `_step = 6584`
  - `loss_total = 0.02562`
  - `loss_track_aux = 0.05677`
  - `loss_box_aux = 0.13458`
  - `loss_depth_aux = 0.06484`
  - `grad_norm = 1.01489`
- `_step = 6610`
  - `loss_total = 0.05847`
  - `loss_track_aux = 0.02725`
  - `loss_box_aux = 0.55348`
  - `loss_depth_aux = 0.00400`
  - `grad_norm = 0.59089`

interpretation：

- `6600` 前后仍能看到 `box_aux` 的 batch-level 波动
- 但这次波动没有伴随：
  - `track_aux` 同步失控
  - `depth_aux` 同步抬升
  - `grad_norm / grad_abs_max` 明显爆掉
- `object_context_abs_max` 仍维持在 `0.39` 左右，`object_latent_tokens_abs_max` 也仍在之前可接受区间内

当前判断：

- 训练继续健康推进
- 暂时没有看到需要改代码或改 loss 配比的直接证据
- 目前最需要继续盯的仍然是两件事：
  - 下一次 checkpoint `step-006800`
  - 下一次 validation 触发点 `step-008000` 是否真正开始在 `gpu5` 上跑通

额外风险提示：

- `/data` 仍只剩约 `5.1G`
- 虽然当前只保留 2 份 checkpoint，训练本身还能继续
- 但到 `step-008000` 时如果 validation 生成大量产物，磁盘仍可能再次成为第一风险点

### 2026-06-26 02:08 UTC：`6626/6628` 附近出现一次短时高波动，但已快速回落

当前运行状态：

- 训练进程仍健康存活
- GPU 仍是：
  - 训练：`gpu6,7`
  - validation 预留：`gpu5`
  - 坏卡 `gpu4` 未使用
- checkpoint 目录此刻仍是：
  - `step-006400`
  - `step-006600`

W&B latest summary（run `qberfq1r`）：

- `_step = 6645`
- `train/loss_total = 0.04424`
- `train/loss_track_aux = 0.15490`
- `train/loss_box_aux = 0.20925`
- `train/loss_depth_aux = 0.07827`
- `train/grad_norm = 0.59683`
- `train/grad_abs_max = 0.08583`
- `train/object_context_abs_max = 0.39153`
- `train/object_latent_tokens_abs_max = 4.76228`

最近一段 history 里最值得记的一小段是：

- `_step = 6619`
  - `loss_total = 0.04353`
  - `loss_track_aux = 0.06563`
  - `loss_box_aux = 0.34870`
  - `loss_depth_aux = 0.02096`
  - `grad_norm = 1.29736`
- `_step = 6626`
  - `loss_total = 0.13456`
  - `loss_track_aux = 0.07048`
  - `loss_box_aux = 0.72803`
  - `loss_depth_aux = 0.54713`
  - `grad_norm = 0.50653`
  - `grad_abs_max = 0.01250`
- `_step = 6628`
  - `loss_total = 0.13646`
  - `loss_track_aux = 0.36139`
  - `loss_box_aux = 0.42424`
  - `loss_depth_aux = 0.57895`
  - `grad_norm = 0.50834`
- `_step = 6644`
  - `loss_total = 0.03138`
  - `loss_track_aux = 0.05482`
  - `loss_box_aux = 0.21008`
  - `loss_depth_aux = 0.04889`
  - `grad_norm = 0.59162`

interpretation：

- `6626/6628` 这段确实出现了一次明显的多项 loss 联合抬升：
  - `box_aux`
  - `depth_aux`
  - 在 `6628` 上 `track_aux` 也被带起来了
- 但它不是典型的梯度爆炸形态，因为：
  - `grad_norm` 没有同步爆掉
  - `grad_abs_max` 反而很低
  - `object_context_abs_max` / `object_latent_tokens_abs_max` 也没有异常飙升
- 更像是：
  - 某个或某几个 batch 的 supervision 难度较高
  - 导致 loss 短时抬升
  - 但参数更新幅度本身仍受控

当前判断：

- 这次波动更接近 batch-level supervision spike，而不是训练发散
- 因为到 `_step = 6644/6645` 已经明显回落
- 暂时仍不需要为了这次波动改代码或改 loss 权重

接下来继续重点观察：

- 下一份 checkpoint：`step-006800`
- 下一次 validation 触发点：`step-008000`
- 如果后续再次出现类似 `box + depth + track` 联合抬升，并且连续多个 summary 不回落，再进入代码级排查

### 2026-06-26 02:11 UTC：确认没有 stall，训练已继续推进到 `_step = 6686`

这轮额外排查的原因：

- 某一时刻 `nvidia-smi` 里 `gpu6/7` 利用率都短暂掉到 `0`
- checkpoint 目录也还停留在：
  - `step-006400`
  - `step-006600`

进一步核对后确认：

- 训练进程本身没有退出
- W&B 仍在继续前进，当前 latest summary 已到：
  - `_step = 6686`
- 所以这不是训练卡死，更像是：
  - 一次短暂的 batch 间空窗
  - 或数据 / 同步 / 采样带来的瞬时 GPU 利用率下探

当前 latest summary（run `qberfq1r`）：

- `_step = 6686`
- `train/loss_total = 0.02869`
- `train/loss_track_aux = 0.02365`
- `train/loss_box_aux = 0.21145`
- `train/loss_depth_aux = 0.05180`
- `train/grad_norm = 0.51060`
- `train/grad_abs_max = 0.05000`
- `train/object_context_abs_max = 0.39623`
- `train/object_latent_tokens_abs_max = 4.75002`

最近一段 history 补充观察：

- `_step = 6665`
  - `loss_total = 0.06330`
  - `loss_box_aux = 0.55166`
  - `loss_depth_aux = 0.05316`
- `_step = 6677`
  - `loss_total = 0.04979`
  - `loss_track_aux = 0.07926`
  - `loss_box_aux = 0.32490`
  - `loss_depth_aux = 0.09373`
  - `grad_norm = 1.35785`
  - `grad_abs_max = 0.35314`
- `_step = 6686`
  - `loss_total = 0.02869`
  - `loss_track_aux = 0.02365`
  - `loss_box_aux = 0.21145`
  - `loss_depth_aux = 0.05180`
  - `grad_norm = 0.51060`

interpretation：

- `6665` 和 `6677` 一带仍能看到 `box_aux` 主导的波动
- `6677` 的 `grad_norm / grad_abs_max` 也有一次局部抬升
- 但到 `6686` 又明显回落
- 因此目前仍然更像：
  - batch-level fluctuation
  - 而不是持续性不稳定或 silent hang

当前判断：

- 训练仍然健康推进
- 还没到下一份 checkpoint `step-006800`
- 当前没有新证据需要改代码
- 继续重点盯：
  - `step-006800` 是否正常落盘
  - `step-008000` validation 是否真正开始并在 `gpu5` 上跑通

### 2026-06-26 02:13 UTC：训练已推进到 `_step = 6716`，仍未见持续异常

补充核对结果：

- checkpoint 目录仍是：
  - `step-006400`
  - `step-006600`
- 说明这时还没到下一份 `step-006800`
- validation runtime 目录也还没有新的 `step-008000` 相关产物

W&B latest summary（run `qberfq1r`）已经推进到：

- `_step = 6716`
- `train/loss_total = 0.01955`
- `train/loss_track_aux = 0.05073`
- `train/loss_box_aux = 0.12800`
- `train/loss_depth_aux = 0.01678`
- `train/grad_norm = 1.06071`
- `train/grad_abs_max = 0.26078`
- `train/object_context_abs_max = 0.39370`
- `train/object_latent_tokens_abs_max = 4.73892`

最近一段 history 里比较关键的点：

- `_step = 6698`
  - `loss_total = 0.06588`
  - `loss_track_aux = 0.06509`
  - `loss_box_aux = 0.51382`
  - `loss_depth_aux = 0.07989`
  - `grad_norm = 1.29173`
- `_step = 6709`
  - `loss_total = 0.04400`
  - `loss_track_aux = 0.08882`
  - `loss_box_aux = 0.24963`
  - `loss_depth_aux = 0.10151`
  - `grad_norm = 1.29700`
- `_step = 6710`
  - `loss_total = 0.01548`
  - `loss_track_aux = 0.02080`
  - `loss_box_aux = 0.09838`
  - `loss_depth_aux = 0.03560`
  - `grad_norm = 0.58968`

interpretation：

- `6698/6709` 一带仍能看到波动，主要还是：
  - `box_aux`
  - 伴随少量 `depth_aux`
- 但后面很快回落到 `6710/6716` 的较低区间
- `object_context_abs_max` 和 `object_latent_tokens_abs_max` 仍然没有异常漂移

当前判断：

- 训练还在健康推进
- 当前没有出现持续 3 个以上 summary 都高位不回落的异常段
- 暂时仍不需要改代码或调 loss 设计
- 继续重点盯：
  - `step-006800` 落盘
  - `step-008000` validation 触发

### 2026-06-26 02:15 UTC：latest summary 推进到 `_step = 6752`，再次出现短时 `box + depth` 联合抬升

当前运行状态：

- 训练进程仍健康存活
- checkpoint 目录此刻仍是：
  - `step-006400`
  - `step-006600`
- validation runtime 目录仍没有新的 `step-008000` 文件
- 说明当前还没有到：
  - 下一份 checkpoint `step-006800`
  - 下一次 validation 触发点 `step-008000`

W&B latest summary（run `qberfq1r`）：

- `_step = 6752`
- `train/loss_total = 0.12316`
- `train/loss_track_aux = 0.06166`
- `train/loss_box_aux = 0.57975`
- `train/loss_depth_aux = 0.59018`
- `train/grad_norm = 0.58303`
- `train/grad_abs_max = 0.08138`
- `train/object_context_abs_max = 0.39277`
- `train/object_latent_tokens_abs_max = 4.90452`

最近一段 history 里最关键的波动点：

- `_step = 6703`
  - `loss_total = 0.12692`
  - `loss_track_aux = 0.04944`
  - `loss_box_aux = 0.72997`
  - `loss_depth_aux = 0.48983`
  - `grad_norm = 0.59180`
- `_step = 6706`
  - `loss_total = 0.04527`
  - `loss_track_aux = 0.07321`
  - `loss_box_aux = 0.34165`
  - `loss_depth_aux = 0.03782`
  - `grad_norm = 1.64232`
  - `grad_abs_max = 0.43896`
- `_step = 6729`
  - `loss_total = 0.04363`
  - `loss_track_aux = 0.12438`
  - `loss_box_aux = 0.23807`
  - `loss_depth_aux = 0.07384`
  - `grad_norm = 1.29094`
- `_step = 6739`
  - `loss_total = 0.06431`
  - `loss_track_aux = 0.17118`
  - `loss_box_aux = 0.35875`
  - `loss_depth_aux = 0.11313`
  - `grad_norm = 0.51110`

interpretation：

- `6752` 的 latest summary 又回到了典型的：
  - `box_aux`
  - `depth_aux`
 共同抬升的形态
- 但这次仍然没有看到典型梯度爆炸证据：
  - `grad_norm` 没有同步冲高
  - `grad_abs_max` 也仍处于受控区间
  - `object_context_abs_max` 没有异常漂移
- `object_latent_tokens_abs_max = 4.90` 比之前略高，但仍未形成持续单调上冲，需要继续观察而不是立即判异常

当前判断：

- 目前更像重复出现的 batch-level supervision spike
- 还不能据此判断训练发散
- 但需要继续紧盯后续 1-3 个 summary：
  - 如果很快回落，维持现方案
  - 如果连续高位不回落，再进入更细的 batch / supervision 级排查

接下来继续重点盯：

- `step-006800` 是否正常落盘
- `step-008000` validation 是否开始并在 `gpu5` 上跑通

### 2026-06-26 02:17 UTC：`6752` 已回落，但 `6765` 又出现一次同型 `box + depth` spike

补充核对结果：

- W&B latest summary 已推进到：
  - `_step = 6787`
- checkpoint 目录此刻仍是：
  - `step-006400`
  - `step-006600`
- 说明这时还没有走到下一份 `step-006800`

当前 latest summary（run `qberfq1r`）：

- `_step = 6787`
- `train/loss_total = 0.06416`
- `train/loss_track_aux = 0.15831`
- `train/loss_box_aux = 0.38417`
- `train/loss_depth_aux = 0.09909`
- `train/grad_norm = 0.59785`
- `train/grad_abs_max = 0.08765`
- `train/object_context_abs_max = 0.39251`
- `train/object_latent_tokens_abs_max = 4.61771`

`6752` 之后的关键轨迹：

- `_step = 6752`
  - `loss_total = 0.12316`
  - `loss_box_aux = 0.57975`
  - `loss_depth_aux = 0.59018`
  - 属于一次明显 `box + depth` 联合抬升
- `_step = 6754`
  - `loss_total = 0.04180`
  - `loss_box_aux = 0.24222`
  - `loss_depth_aux = 0.11652`
  - 已明显回落
- `_step = 6756`
  - `loss_total = 0.01557`
  - `loss_box_aux = 0.11045`
  - `loss_depth_aux = 0.01185`
  - 回到低位
- `_step = 6765`
  - `loss_total = 0.10644`
  - `loss_track_aux = 0.05637`
  - `loss_box_aux = 0.43046`
  - `loss_depth_aux = 0.57754`
  - `grad_norm = 0.79551`
  - 又出现一次类似的 `box + depth` spike
- `_step = 6773`
  - `loss_total = 0.03012`
  - `loss_box_aux = 0.24393`
  - `loss_depth_aux = 0.00955`
  - 再次回落
- `_step = 6786`
  - `loss_total = 0.02790`
  - `loss_box_aux = 0.21425`
  - `loss_depth_aux = 0.01474`
  - `grad_norm = 1.63866`
  - `grad_abs_max = 0.44054`

interpretation：

- `6752` 不是持续高位，它后面已经快速回落
- 但 `6765` 又重复出现了相同模式，说明：
  - 当前训练里确实反复存在某类 batch / supervision 触发的 `box + depth` 局部抬升
- 好的一面是：
  - 每次 spike 后都能在后续 1-3 个 summary 内回落
  - `object_context_abs_max` 没有异常上漂
  - `object_latent_tokens_abs_max` 反而回落到 `4.62` 左右

当前判断：

- 目前依然更像“重复出现但可恢复的 batch-level supervision spike”
- 还没有形成持续发散
- 暂时仍不需要改代码
- 但如果后面继续重复这类模式，可以考虑下一步做：
  - 记录对应 batch 的 case id
  - 单独回放并检查其 GT track / box / depth supervision 质量

### 2026-06-26 02:20 UTC：`step-006800` 已成功落盘，`6787` 之后仍表现为可恢复波动

当前核对结果：

- checkpoint 已推进到：
  - `step-006600`
  - `step-006800`
- `step-006800/training_state.pt` 已核对：
  - `global_step = 6800`
  - `epoch_id = 2`
  - `batch_in_epoch = 1000`
- checkpoint 轮转仍正常，只保留两份

W&B latest summary（run `qberfq1r`）：

- `_step = 6827`
- `train/loss_total = 0.05509`
- `train/loss_track_aux = 0.02515`
- `train/loss_box_aux = 0.46761`
- `train/loss_depth_aux = 0.05809`
- `train/grad_norm = 1.27262`
- `train/grad_abs_max = 0.33547`
- `train/object_context_abs_max = 0.39536`
- `train/object_latent_tokens_abs_max = 4.74157`

`6787` 之后的关键轨迹：

- `_step = 6803`
  - `loss_total = 0.03396`
  - `loss_box_aux = 0.21499`
  - `loss_depth_aux = 0.03962`
- `_step = 6804`
  - `loss_total = 0.04401`
  - `loss_track_aux = 0.09854`
  - `loss_box_aux = 0.31301`
  - `loss_depth_aux = 0.02854`
  - `grad_norm = 1.87011`
  - `grad_abs_max = 0.51096`
- `_step = 6805`
  - `loss_total = 0.06532`
  - `loss_track_aux = 0.17910`
  - `loss_box_aux = 0.36344`
  - `loss_depth_aux = 0.11062`
- `_step = 6817`
  - `loss_total = 0.01402`
  - `loss_box_aux = 0.09067`
  - `loss_depth_aux = 0.01920`
  - 明显回落
- `_step = 6818`
  - `loss_total = 0.04789`
  - `loss_box_aux = 0.40935`
  - `loss_depth_aux = 0.03846`
- `_step = 6819`
  - `loss_total = 0.01333`
  - `loss_box_aux = 0.09734`
  - `loss_depth_aux = 0.00552`
  - 再次回落

interpretation：

- `6787` 之后没有发展成持续高位异常
- 仍然是：
  - 某些 step 出现局部 `box_aux` 或 `track+box` 抬升
  - 然后很快回到低位
- `6804` 的 `grad_norm / grad_abs_max` 抬升值得记，但没有和持续的高 loss 段绑定在一起
- `object_context_abs_max` 和 `object_latent_tokens_abs_max` 仍稳定，没有支持“数值发散”的证据

当前判断：

- 训练继续健康推进
- `step-006800` 已证明 checkpoint 产出链路正常
- 目前最大的下一个关键点仍然是：
  - `step-008000` validation 是否真正开始并在 `gpu5` 上跑通

额外风险：

- `/data` 仍然只剩约 `5.1G`
- 离 `step-008000` 越近，越需要警惕 validation 产物落盘导致的新一轮磁盘中断

### 2026-06-26 02:23 UTC：latest summary 推进到 `_step = 6873`，`6800` 后仍未见持续异常

当前状态：

- checkpoint 目录仍是：
  - `step-006600`
  - `step-006800`
- 说明当前还没到下一份 `step-007000`
- validation runtime 目录仍未出现新的 `step-008000` 相关文件

W&B latest summary（run `qberfq1r`）：

- `_step = 6873`
- `train/loss_total = 0.01536`
- `train/loss_track_aux = 0.02093`
- `train/loss_box_aux = 0.09762`
- `train/loss_depth_aux = 0.03510`
- `train/grad_norm = 0.59204`
- `train/grad_abs_max = 0.08608`
- `train/object_context_abs_max = 0.40197`
- `train/object_latent_tokens_abs_max = 4.94860`

`6800` 之后这一段的关键轨迹：

- `_step = 6812`
  - `loss_total = 0.10527`
  - `loss_box_aux = 0.42275`
  - `loss_depth_aux = 0.57447`
  - 属于一次明显 `box + depth` 抬升
- `_step = 6814`
  - `loss_total = 0.06007`
  - `loss_box_aux = 0.54432`
  - `loss_depth_aux = 0.01856`
  - 转成更偏 `box_aux` 主导
- `_step = 6820`
  - `loss_total = 0.04741`
  - `loss_box_aux = 0.44308`
  - `loss_depth_aux = 0.00455`
- `_step = 6822`
  - `loss_total = 0.04763`
  - `loss_track_aux = 0.09221`
  - `loss_box_aux = 0.35155`
  - `grad_norm = 1.65559`
  - `grad_abs_max = 0.44616`
- `_step = 6844`
  - `loss_total = 0.05893`
  - `loss_box_aux = 0.55519`
  - `loss_depth_aux = 0.00941`
  - `grad_norm = 1.35270`
- `_step = 6857`
  - `loss_total = 0.06165`
  - `loss_box_aux = 0.54495`
  - `loss_depth_aux = 0.04154`
- `_step = 6870`
  - `loss_total = 0.03286`
  - `loss_box_aux = 0.23264`
  - `loss_depth_aux = 0.03320`
- `_step = 6873`
  - `loss_total = 0.01536`
  - `loss_box_aux = 0.09762`
  - `loss_depth_aux = 0.03510`
  - 已回落到低位

interpretation：

- `6800` 后依然能看到反复出现的：
  - `box_aux` 主导波动
  - 偶尔伴随 `depth_aux` 抬升
- 但这些波动后面仍然可以回落到较低区间
- `object_context_abs_max` 仍稳定在 `0.39 ~ 0.40` 一带
- `object_latent_tokens_abs_max` 到了 `4.95`，这次比之前略高，值得继续盯，但目前仍缺少持续单调上冲证据

当前判断：

- 训练仍在健康推进
- 目前还没有形成“连续多个 summary 高位不回落”的发散模式
- 下一阶段优先关注：
  - `step-007000` 是否正常落盘
  - `step-008000` validation 是否开始并在 `gpu5` 上跑通

### 2026-06-26 02:26 UTC：latest summary 推进到 `_step = 6909`，离 `step-008000` 还差约 1090 step

当前状态：

- checkpoint 目录仍是：
  - `step-006600`
  - `step-006800`
- validation runtime 目录仍无新的 `step-008000` 文件
- 当前离 validation 触发点仍有一段距离，但已经可以开始重点盯磁盘和 benchmark 运行目录

W&B latest summary（run `qberfq1r`）：

- `_step = 6909`
- `train/loss_total = 0.02698`
- `train/loss_track_aux = 0.05188`
- `train/loss_box_aux = 0.17739`
- `train/loss_depth_aux = 0.04051`
- `train/grad_norm = 1.04525`
- `train/grad_abs_max = 0.26055`
- `train/object_context_abs_max = 0.39236`
- `train/object_latent_tokens_abs_max = 4.88165`

最近一段 history 里比较关键的点：

- `_step = 6878`
  - `loss_total = 0.12423`
  - `loss_box_aux = 0.73081`
  - `loss_depth_aux = 0.47037`
  - 典型 `box + depth` spike
- `_step = 6881`
  - `loss_total = 0.13530`
  - `loss_box_aux = 0.72935`
  - `loss_depth_aux = 0.57744`
  - spike 继续了一小段
- `_step = 6888`
  - `loss_total = 0.03155`
  - `loss_box_aux = 0.21486`
  - `loss_depth_aux = 0.03665`
  - 已明显回落
- `_step = 6899`
  - `loss_total = 0.06126`
  - `loss_box_aux = 0.58252`
  - `loss_depth_aux = 0.00188`
  - 更偏 `box_aux` 单项抬升
- `_step = 6902`
  - `loss_total = 0.05649`
  - `loss_track_aux = 0.10912`
  - `loss_box_aux = 0.32756`
  - `loss_depth_aux = 0.12827`
- `_step = 6907`
  - `loss_total = 0.05660`
  - `loss_box_aux = 0.52448`
  - `loss_depth_aux = 0.01157`
- `_step = 6909`
  - `loss_total = 0.02698`
  - `loss_box_aux = 0.17739`
  - `loss_depth_aux = 0.04051`
  - 已再次回落

interpretation：

- `6878/6881` 确实出现了一小段连续的 `box + depth` 高位
- 但后面仍然可以回落到低位，没有扩展成长期高位平台
- `object_context_abs_max` 继续稳定
- `object_latent_tokens_abs_max` 一度接近 `4.99`，当前回落到 `4.88`

当前判断：

- 训练仍然健康推进
- 目前依然更像“重复出现、但可恢复的 supervision spike”
- 离 `step-008000` 越来越近，接下来优先盯：
  - `step-007000` checkpoint
  - `step-008000` validation 实际触发
  - validation 是否在 `gpu5` 启动成功
  - validation 落盘是否把 `/data` 空间压垮

### 2026-06-26 02:29 UTC：latest summary 推进到 `_step = 6949`，尚未到 `step-007000`

当前状态：

- checkpoint 目录仍是：
  - `step-006600`
  - `step-006800`
- 说明还没走到下一份 `step-007000`
- validation runtime 目录仍然没有新的 `step-008000` 文件
- 训练距离 validation 触发点还差约 `1050` step

W&B latest summary（run `qberfq1r`）：

- `_step = 6949`
- `train/loss_total = 0.04859`
- `train/loss_track_aux = 0.07436`
- `train/loss_box_aux = 0.34943`
- `train/loss_depth_aux = 0.06210`
- `train/grad_norm = 0.60160`
- `train/grad_abs_max = 0.08887`
- `train/object_context_abs_max = 0.39795`
- `train/object_latent_tokens_abs_max = 4.94149`

这一段比较值得记录的点：

- `_step = 6923`
  - `loss_total = 0.13770`
  - `loss_track_aux = 0.18157`
  - `loss_box_aux = 0.71787`
  - `loss_depth_aux = 0.47750`
  - 属于一次较强的 `track + box + depth` 联合抬升
- `_step = 6924`
  - `loss_total = 0.01923`
  - `loss_box_aux = 0.10916`
  - `loss_depth_aux = 0.03791`
  - 立即回落
- `_step = 6942`
  - `grad_norm = 2.00586`
  - `grad_abs_max = 0.54924`
  - 是目前这段里最大的梯度尖峰之一
- `_step = 6944`
  - `loss_box_aux = 0.59698`
  - `object_latent_tokens_abs_max = 5.02022`
- `_step = 6948`
  - `loss_box_aux = 0.34076`
  - `object_latent_tokens_abs_max = 5.05005`
  - 是目前看到的局部新高
- `_step = 6949`
  - 各项回到中低位

interpretation：

- 训练主趋势仍然是“spike 后可恢复”
- 但这次需要额外记下一个新的数值信号：
  - `object_latent_tokens_abs_max` 最近几次已摸到 `5.0+`
- 目前它还没有和持续性高 loss / 高 grad 平台绑定在一起，所以还不能判为异常
- 但如果后续继续单调上升，或者和 `box/depth` 高位段同时持续出现，就需要把它升级为重点排查对象

当前判断：

- 训练仍在推进
- 还没有证据表明需要立刻改代码
- 继续重点盯：
  - `step-007000` 是否正常落盘
  - `object_latent_tokens_abs_max` 是否继续走高
  - `step-008000` validation 是否真正开始并在 `gpu5` 上跑通

### 2026-06-26 02:32 UTC：`step-007000` 已成功落盘，latest summary 推进到 `_step = 7009`

当前核对结果：

- checkpoint 已推进到：
  - `step-006800`
  - `step-007000`
- `step-007000/training_state.pt` 已核对：
  - `global_step = 7000`
  - `epoch_id = 2`
  - `batch_in_epoch = 1000`
- checkpoint 轮转依旧正常

W&B latest summary（run `qberfq1r`）：

- `_step = 7009`
- `train/loss_total = 0.02399`
- `train/loss_track_aux = 0.05595`
- `train/loss_box_aux = 0.17179`
- `train/loss_depth_aux = 0.01215`
- `train/grad_norm = 0.60393`
- `train/grad_abs_max = 0.09302`
- `train/object_context_abs_max = 0.39608`
- `train/object_latent_tokens_abs_max = 5.00443`

interpretation：

- `step-007000` 已经证明训练仍在按计划产出权重
- latest summary 回到了较低区间，说明最近的局部 spike 仍然能回落
- `object_latent_tokens_abs_max` 仍在 `5.0` 附近徘徊：
  - 这仍是一个需要盯住的数值信号
  - 但目前还没有和持续性异常绑定

当前判断：

- 训练继续健康推进
- 下一个真正关键点仍然是：
  - `step-008000` validation 是否真正触发
  - validation 是否在 `gpu5` 上启动成功
  - validation 落盘是否触发新的磁盘空间问题

### 2026-06-26 02:35 UTC：latest summary 推进到 `_step = 7060`，`object_latent_tokens_abs_max` 进入需要重点盯的区间

当前状态：

- checkpoint 目录仍是：
  - `step-006800`
  - `step-007000`
- validation runtime 目录仍未出现新的 `step-008000` 文件
- 当前距离 validation 触发点还差约 `940` step

W&B latest summary（run `qberfq1r`）：

- `_step = 7060`
- `train/loss_total = 0.05932`
- `train/loss_track_aux = 0.05911`
- `train/loss_box_aux = 0.53069`
- `train/loss_depth_aux = 0.00340`
- `train/grad_norm = 0.60086`
- `train/grad_abs_max = 0.09179`
- `train/object_context_abs_max = 0.38725`
- `train/object_latent_tokens_abs_max = 4.90108`

这一段里最值得升级关注的是 `object_latent_tokens_abs_max`：

- `_step = 7053`
  - `object_latent_tokens_abs_max = 5.09934`
- `_step = 7054`
  - `object_latent_tokens_abs_max = 5.06346`
- `_step = 7056`
  - `object_latent_tokens_abs_max = 5.07902`
- `_step = 7058`
  - `object_latent_tokens_abs_max = 5.07757`

另外，这一段里也还有典型的 supervision spike：

- `_step = 7013`
  - `loss_total = 0.11011`
  - `loss_box_aux = 0.43444`
  - `loss_depth_aux = 0.62964`
- `_step = 7054`
  - `loss_total = 0.12561`
  - `loss_track_aux = 0.23946`
  - `loss_box_aux = 0.42382`
  - `loss_depth_aux = 0.59278`

但这些 spike 目前仍然满足两个特征：

- 后续还能回落
- 没有直接演化成持续高位平台

新的 interpretation：

- `object_latent_tokens_abs_max` 现在不只是“偶发摸到 5.0”
- 它开始在连续多个 summary 里落在 `5.0 ~ 5.10` 区间
- 这还不足以证明异常，但已经应该从“顺手观察”升级为“重点盯防”

当前判断：

- 训练仍在推进
- 还没有到必须立刻改代码的程度
- 但后续如果出现以下任一情况，就建议进入代码/数据级排查：
  - `object_latent_tokens_abs_max` 持续进一步上升
  - 与 `box/depth` 高位段连续绑定
  - 同时出现更频繁的高 `grad_norm / grad_abs_max`

接下来继续重点盯：

- `step-007200` checkpoint
- `step-008000` validation 实际触发
- `object_latent_tokens_abs_max` 是否继续维持在 `5.0+`

### 2026-06-26 02:38 UTC：latest summary 推进到 `_step = 7100`，`object_latent_tokens_abs_max` 出现 `5.12` 新高

当前状态：

- checkpoint 目录仍是：
  - `step-006800`
  - `step-007000`
- validation runtime 目录仍没有新的 `step-008000` 文件
- 当前距离 validation 触发点约还差 `900` step

W&B latest summary（run `qberfq1r`）：

- `_step = 7100`
- `train/loss_total = 0.04269`
- `train/loss_track_aux = 0.11160`
- `train/loss_box_aux = 0.23615`
- `train/loss_depth_aux = 0.07915`
- `train/grad_norm = 0.60544`
- `train/grad_abs_max = 0.09378`
- `train/object_context_abs_max = 0.39111`
- `train/object_latent_tokens_abs_max = 5.01208`

这段里最重要的新变化：

- `_step = 7066`
  - `object_latent_tokens_abs_max = 5.12243`
- `_step = 7086`
  - `object_latent_tokens_abs_max = 5.08621`
- `_step = 7089`
  - `object_latent_tokens_abs_max = 5.10202`
- `_step = 7097`
  - `object_latent_tokens_abs_max = 5.09962`
- `_step = 7098`
  - `object_latent_tokens_abs_max = 5.08864`

同时，这段里仍然会出现局部 supervision spike：

- `_step = 7062`
  - `loss_total = 0.10869`
  - `loss_box_aux = 0.41639`
  - `loss_depth_aux = 0.59139`
- `_step = 7094`
  - `loss_total = 0.10527`
  - `loss_box_aux = 0.43489`
  - `loss_depth_aux = 0.57985`

但到 latest summary：

- 并没有停留在高位
- `grad_norm / grad_abs_max` 也没有同步爆掉
- `object_context_abs_max` 仍然稳定

新的 interpretation：

- `object_latent_tokens_abs_max` 现在已经不只是“偶尔碰到 5”
- 它在最近一个窗口里多次稳定出现在 `5.0+`，并且已经摸到 `5.12`
- 这说明它确实在朝着需要更严肃监控的方向发展

当前判断：

- 训练仍然在推进，尚未出现必须立刻停训改代码的证据
- 但 `object_latent_tokens_abs_max` 的风险等级再次上升
- 如果后续继续看到：
  - 新高不断刷新
  - 同时伴随更频繁的 `box/depth` 高位段
  - 或 `grad_norm / grad_abs_max` 抬升开始更密集
  那就应该尽快转入针对 object branch 的数值排查

接下来继续重点盯：

- `step-007200` checkpoint
- `object_latent_tokens_abs_max` 是否继续刷新新高
- `step-008000` validation 实际触发与磁盘占用

### 2026-06-26 02:31-02:33 UTC：继续监控，训练进程健康，仍严格避开 `gpu4`

当前进程状态：

- 训练 launcher 仍在：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_heads_only_gpu67.sh`
- `accelerate launch` 仍在正常运行
- 两个 `train_v_newtrain.py` worker 都在正常推进

当前 GPU 状态：

- `gpu6`：训练占用，约 `42725 / 49140 MiB`
- `gpu7`：训练占用，约 `42707 / 49140 MiB`
- `gpu5`：空闲，保留给后续 validation / benchmark
- `gpu4`：未使用

这次检查再次确认：

- 当前训练和验证配置都没有把 `gpu4` 放回去
- 训练主进程仍然只使用 `gpu6,7`
- 启动参数里的 `--benchmark_cuda_visible_devices 5` 仍然保持不变

checkpoint / validation 状态：

- checkpoint 目录当前仍只有：
  - `step-006800`
  - `step-007000`
- 说明自 `step-007000` 之后暂时还没落下新的 200-step checkpoint
- validation runtime 目录里仍然只有修复前遗留的失败记录：
  - `step-002000/benchmark.failed.json`
  - `step-004000/benchmark.failed.json`
  - `step-006000/benchmark.failed.json`
- 目前还没有新的 `step-008000` validation 产物，符合当前还没到触发点的状态

W&B 当前 latest summary（run `qberfq1r`）：

- `_step = 7176`
- `train/loss_total = 0.13182`
- `train/loss_track_aux = 0.05773`
- `train/loss_box_aux = 0.71301`
- `train/loss_depth_aux = 0.54746`
- `train/grad_norm = 0.80492`
- `train/grad_abs_max = 0.17979`
- `train/object_context_abs_max = 0.38938`
- `train/object_latent_tokens_abs_max = 5.11309`

这一时刻的 interpretation：

- 这是一次比较典型的 `box_aux + depth_aux` 联动 spike
- 但从 summary 看，`grad_norm` 还没有进入失控状态
- `object_context_abs_max` 仍然稳定在约 `0.39`
- 因此目前更像是局部 batch supervision spike，而不是全局数值爆炸

需要继续重点盯的量：

- `object_latent_tokens_abs_max`

到这次检查为止，它仍然是当前最需要盯的数值信号：

- 已经不只是偶发触到 `5.0`
- 现在会反复出现在 `5.0+`
- 但还没有证据证明它已经导致训练发散

当前 operational risk 仍然主要是磁盘空间：

- `df -h /data` 仍显示只剩约 `5.1G`
- 这对 `step-008000` 附近的 checkpoint + validation 组合仍然偏紧
- 后续一旦 validation 产物开始写盘，需要优先确认没有再次触发 `No space left on device`

### 2026-06-26 02:33-02:34 UTC：`step-007200` 已成功落盘，latent token 风险继续抬升

最新状态：

- checkpoint 目录已经从：
  - `step-006800`
  - `step-007000`
  推进到：
  - `step-007000`
  - `step-007200`
- `step-007200/training_state.pt` 已确认：
  - `global_step = 7200`
  - `epoch_id = 2`
  - `batch_in_epoch = 1200`

这说明：

- 当前 checkpoint 落盘机制正常
- `--max_checkpoints_keep 2` 仍在生效
- 当前训练还没有在 `7200` 这一段出现卡死或写盘失败

validation 状态仍未变化：

- validation runtime 目录仍只有修复前的：
  - `step-002000`
  - `step-004000`
  - `step-006000`
- 目前还没有新的 `step-008000` runtime 文件
- `gpu5` 仍保持空闲，可用于后续 validation / benchmark

W&B latest summary（run `qberfq1r`）已推进到：

- `_step = 7216`
- `train/loss_total = 0.05025`
- `train/loss_track_aux = 0.07516`
- `train/loss_box_aux = 0.36299`
- `train/loss_depth_aux = 0.06433`
- `train/grad_norm = 0.50888`
- `train/grad_abs_max = 0.03750`
- `train/object_context_abs_max = 0.40026`
- `train/object_latent_tokens_abs_max = 5.17867`

这里最重要的新信息是：

- `object_latent_tokens_abs_max` 再次刷新新高，已经到 `5.17867`

但同一时刻也要注意两点：

- `loss_total` 并不高
- `grad_norm / grad_abs_max` 反而相对平稳

因此当前 interpretation 更新为：

- `object_latent_tokens_abs_max` 的风险等级继续上升
- 但它现在仍然更像“潜在数值风险信号”
- 还没有和一次明确的全局梯度爆炸或 loss 失控绑定起来

也就是说，现阶段还不能仅凭这个新高就判断需要停训改代码；更合理的做法仍然是继续盯下面三件事是否开始同时出现：

- `object_latent_tokens_abs_max` 持续继续创新高
- `box_aux / depth_aux` 更频繁进入高位并停留更久
- `grad_norm / grad_abs_max` 也开始同步变密、变高

当前优先监控目标保持不变：

- 下一个 checkpoint：`step-007400`
- validation 触发点：`step-008000`
- `/data` 剩余空间：仍约 `5.1G`

### 2026-06-26 02:34-02:35 UTC：训练继续推进，`object_latent_tokens_abs_max` 再刷新到 `5.19627`

这一轮检查里，基础运行状态没有新异常：

- checkpoint 目录仍是：
  - `step-007000`
  - `step-007200`
- validation runtime 目录仍没有新的 `step-008000` 产物
- 当前仍未看到 `run_validation_vbench.py` 或 `batch_eval_lora.py` 新进程
- `gpu5` 继续空闲，等待后续 validation / benchmark
- `gpu4` 仍未使用
- `/data` 仍然只剩约 `5.1G`

W&B latest summary（run `qberfq1r`）进一步推进到：

- `_step = 7236`
- `train/loss_total = 0.10754`
- `train/loss_track_aux = 0.02783`
- `train/loss_box_aux = 0.50152`
- `train/loss_depth_aux = 0.54604`
- `train/grad_norm = 0.60415`
- `train/grad_abs_max = 0.09412`
- `train/object_context_abs_max = 0.39980`
- `train/object_latent_tokens_abs_max = 5.19627`

这一步的特征是：

- `box_aux + depth_aux` 再次一起抬高
- 但 `grad_norm / grad_abs_max` 仍然没有同步进入危险区
- `object_context_abs_max` 依然稳定在约 `0.40`

因此当前判断保持为：

- `object_latent_tokens_abs_max` 正在持续刷新新高，已经成为最优先监控的数值信号
- 但到 `_step = 7236` 为止，它仍未和明确的全局梯度失控绑定
- 目前更像“风险持续抬升但尚未证实发散”

后续如果在接近 `step-008000` 的区间里继续出现下面这种组合，就需要更积极地考虑进入 object-branch 数值排查：

- `object_latent_tokens_abs_max` 继续上冲
- `box_aux / depth_aux` 高位变得更频繁
- `grad_norm / grad_abs_max` 也开始一起变密、抬高

### 2026-06-26 02:35-02:36 UTC：`object_latent_tokens_abs_max` 已到 `5.24342`，但仍未与整体 loss / grad 失控绑定

这一轮状态检查仍然没有看到基础设施层面的新问题：

- checkpoint 目录仍是：
  - `step-007000`
  - `step-007200`
- 说明还没推进到下一个 200-step 落盘点
- validation runtime 目录依旧没有新的 `step-008000` 文件
- `run_validation_vbench.py` / `batch_eval_lora.py` 新进程仍未出现
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7257`
- `train/loss_total = 0.01496`
- `train/loss_track_aux = 0.03198`
- `train/loss_box_aux = 0.09386`
- `train/loss_depth_aux = 0.02380`
- `train/grad_norm = 0.81520`
- `train/grad_abs_max = 0.18591`
- `train/object_context_abs_max = 0.40232`
- `train/object_latent_tokens_abs_max = 5.24342`

这一步的关键点比较特殊：

- `object_latent_tokens_abs_max` 再次刷新新高，已经到 `5.24342`
- 但同一时刻：
  - `loss_total` 很低
  - `box_aux / depth_aux` 也不高
  - `grad_norm / grad_abs_max` 没有进入异常高位

这使当前 interpretation 更明确了一点：

- `object_latent_tokens_abs_max` 现在更像一个“单独抬升的内部幅值信号”
- 它确实越来越值得警惕
- 但到 `_step = 7257` 为止，还不能把它直接解释成训练已经在发散

当前更合理的监控策略仍然是：

- 不因为这个单一指标的新高立刻停训
- 继续观察它是否开始和更坏的外部症状绑定：
  - 更频繁的 `box_aux / depth_aux` 高位
  - 更密集的 `grad_norm / grad_abs_max` 上冲
  - 或接近 `step-008000` 时 validation 质量/运行稳定性出现异常

### 2026-06-26 02:36-02:37 UTC：latest summary 回落，暂未看到从内部幅值信号演化为外部失稳

这一轮检查的基础状态仍然没有变化：

- checkpoint 目录仍是：
  - `step-007000`
  - `step-007200`
- validation runtime 目录仍只有旧的：
  - `step-002000`
  - `step-004000`
  - `step-006000`
- 还没有新的 validation 进程出现
- `gpu5` 仍空闲，留给 `step-008000` 附近的 validation / benchmark
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7272`
- `train/loss_total = 0.01472`
- `train/loss_track_aux = 0.04275`
- `train/loss_box_aux = 0.09318`
- `train/loss_depth_aux = 0.01127`
- `train/grad_norm = 0.83008`
- `train/grad_abs_max = 0.19093`
- `train/object_context_abs_max = 0.40684`
- `train/object_latent_tokens_abs_max = 5.16229`

这一步最有价值的信息不是新高，而是：

- `object_latent_tokens_abs_max` 相比前一轮 `5.24342` 出现了回落
- 同时外部可见训练量仍然保持在较低水平：
  - `loss_total` 低
  - `box_aux / depth_aux` 低
  - `grad_norm / grad_abs_max` 没有失控

因此当前判断进一步收敛为：

- 到 `_step = 7272` 为止，`object_latent_tokens_abs_max` 更像“会波动、但总体偏高的内部风险信号”
- 目前还没有证据表明它已经稳定演化成训练发散
- 后续仍然要重点看它是否在接近 `step-008000` 时重新持续创新高，并开始和更坏的外部症状绑定

### 2026-06-26 02:37-02:38 UTC：出现一次 `box_aux` 局部抬高，但整体仍像可恢复尖峰

这一轮基础运行状态仍未变化：

- checkpoint 目录仍是：
  - `step-007000`
  - `step-007200`
- validation runtime 目录仍没有新的 `step-008000` 文件
- 还没有新的 validation / benchmark 子进程
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7298`
- `train/loss_total = 0.06460`
- `train/loss_track_aux = 0.02789`
- `train/loss_box_aux = 0.52066`
- `train/loss_depth_aux = 0.09746`
- `train/grad_norm = 0.60503`
- `train/grad_abs_max = 0.09315`
- `train/object_context_abs_max = 0.39120`
- `train/object_latent_tokens_abs_max = 5.05059`

这一时刻的组合特征是：

- `box_aux` 出现一次比较明显的局部抬高
- 但：
  - `grad_norm / grad_abs_max` 仍然低
  - `depth_aux` 只是轻度抬高
  - `object_latent_tokens_abs_max` 反而回落到 `5.05` 左右

因此当前 interpretation 保持不变：

- 目前更像“可恢复的 supervision spike”
- 还不是持续恶化趋势
- 到这一步为止，最需要等待的仍然是：
  - `step-007400` checkpoint 是否正常落盘
  - `step-008000` validation 是否正常触发到 `gpu5`

### 2026-06-26 02:38-02:39 UTC：latest summary 维持低 loss，等待 `step-007400`

这一轮基础状态仍未变化：

- checkpoint 目录仍是：
  - `step-007000`
  - `step-007200`
- validation runtime 目录仍没有新的 `step-008000` 文件
- 仍未看到新的 validation / benchmark 子进程
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7313`
- `train/loss_total = 0.01453`
- `train/loss_track_aux = 0.01856`
- `train/loss_box_aux = 0.09513`
- `train/loss_depth_aux = 0.03163`
- `train/grad_norm = 1.08227`
- `train/grad_abs_max = 0.27949`
- `train/object_context_abs_max = 0.39910`
- `train/object_latent_tokens_abs_max = 5.07914`

这一时刻更接近“正常低损失批次”：

- `loss_total` 低
- `box_aux / depth_aux` 都不高
- `object_latent_tokens_abs_max` 没有重新冲向前面的局部高点，而是维持在 `5.08` 左右

当前判断保持为：

- 训练暂未出现新的恶化信号
- 继续等待 `step-007400` checkpoint 落盘
- 接近 `step-008000` 时优先检查 validation 是否按预期在 `gpu5` 触发

### 2026-06-26 02:39-02:40 UTC：latest summary 继续稳定，仍未出现 `step-007400`

这一轮基础状态仍然没有变化：

- checkpoint 目录仍是：
  - `step-007000`
  - `step-007200`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `run_validation_vbench.py` / `batch_eval_lora.py` 仍未出现
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7338`
- `train/loss_total = 0.01480`
- `train/loss_track_aux = 0.01951`
- `train/loss_box_aux = 0.09483`
- `train/loss_depth_aux = 0.03365`
- `train/grad_norm = 0.51063`
- `train/grad_abs_max = 0.05000`
- `train/object_context_abs_max = 0.40004`
- `train/object_latent_tokens_abs_max = 5.16130`

这一步继续落在“正常低损失批次”范围内：

- loss 仍低
- 梯度不高
- `object_latent_tokens_abs_max` 维持在 `5.16` 左右，没有明显恶化

当前判断不变：

- 训练仍在稳定推进
- 眼下最重要的仍然是等 `step-007400` 正常落盘
- 然后继续盯 `step-008000` validation 是否会按预期在 `gpu5` 上触发

### 2026-06-26 02:40-02:41 UTC：`object_latent_tokens_abs_max` 再创新高，风险重新升温

基础运行状态在这一轮依然没变：

- checkpoint 目录仍是：
  - `step-007000`
  - `step-007200`
- validation runtime 目录仍没有新的 `step-008000` 文件
- 仍未看到新的 validation / benchmark 子进程
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7364`
- `train/loss_total = 0.04665`
- `train/loss_track_aux = 0.14952`
- `train/loss_box_aux = 0.23635`
- `train/loss_depth_aux = 0.08066`
- `train/grad_norm = 1.39685`
- `train/grad_abs_max = 0.38218`
- `train/object_context_abs_max = 0.40131`
- `train/object_latent_tokens_abs_max = 5.29770`

这一时刻和前面“正常低损失批次”相比，有两点变化：

- `object_latent_tokens_abs_max` 再次刷新新高，已经到 `5.29770`
- `track / box / depth` 三项和梯度都一起抬了一截

但当前还不能直接判成失控，原因是：

- `loss_total` 仍不算高
- `grad_norm / grad_abs_max` 虽然抬升，但还没有进入明显爆炸区
- `object_context_abs_max` 仍然稳定在约 `0.40`

因此当前判断更新为：

- 风险信号重新升温
- 但证据仍更接近“局部抬升”而不是“已经发散”
- 接下来优先看两件事：
  - `step-007400` checkpoint 是否正常落盘
  - 在 `7364` 之后，这组抬升是否会持续，而不是再次回落

### 2026-06-26 02:41-02:42 UTC：`7364` 后已出现回落，暂未看到风险继续扩散

这一轮基础状态仍无变化：

- checkpoint 目录仍是：
  - `step-007000`
  - `step-007200`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7384`
- `train/loss_total = 0.04679`
- `train/loss_track_aux = 0.09494`
- `train/loss_box_aux = 0.34625`
- `train/loss_depth_aux = 0.02667`
- `train/grad_norm = 0.51101`
- `train/grad_abs_max = 0.05000`
- `train/object_context_abs_max = 0.39837`
- `train/object_latent_tokens_abs_max = 5.15315`

## 14. 2026-06-26 新一轮 fresh run 前的配置核查

### 14.1 train / val 是否是单独数据集

已经核查 `PhysStateEpisodeDataset` 的实现：

- 代码位置：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/data/phys_state_dataset.py`
- 数据集实例化时直接读取：
  - `self.root = Path(root) / split`
  - `self.samples = sorted(self.root.glob("*.json"))`

这说明 `train` 和 `val` 不是同一目录里再做随机切分，而是物理上分开的两个子目录。

进一步对磁盘上的当前 phys-state root 做了实际核查：

- root:
  - `/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6_h264_batch1500`
- `train` JSON 数量：
  - `3600`
- `val` JSON 数量：
  - `450`
- `train` / `val` 文件名交集：
  - `0`
- `train` 头几个文件：
  - `sample_000001_w000.json`
  - `sample_000001_w001.json`
  - `sample_000001_w002.json`
- `val` 头几个文件：
  - `sample_000301_w000.json`
  - `sample_000301_w001.json`
  - `sample_000301_w002.json`

结论：

- 当前 `train` 和 `val` 是两套独立样本
- 因此 head-only `val_loss` 可以安全地直接从 `split=val` 单独构建 dataloader，而不需要动现有 benchmark validation 链路

### 14.2 head-only val loss 轻量链路

已经把 head-only `val_loss` 独立抽到：

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/headonly_val_loss.py`

并在：

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_v_newtrain.py`

里通过 import 接入。

这条链路的设计是：

- 只在训练主循环内额外跑少量 `val` batch
- 只统计 object-heads-only 当前已有的 loss / metric
- 以 `val/...` 形式写入 W&B
- 不替换、不修改现有 `run_validation_vbench.py` 外部 benchmark validation 子进程

当前可用的新增参数：

- `--headonly_val_loss_every_steps`
- `--headonly_val_loss_split`
- `--headonly_val_loss_num_batches`

### 14.3 fresh run 启动脚本

已经新增单独的 fresh-run 脚本：

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_heads_only_gpu67_fresh_500_val.sh`

这份脚本的关键点：

- 训练卡固定：
  - `CUDA_VISIBLE_DEVICES=6,7`
- benchmark / validation 子进程继续固定：
  - `--benchmark_cuda_visible_devices 5`
- 严格避免使用 `gpu4`
- 不带 `--resume_from`
- 从头新开 run
- checkpoint 频率改为：
  - `--save_steps 500`
- 轻量 head-only val loss 频率改为：
  - `--headonly_val_loss_every_steps 500`
- val split 固定：
  - `--headonly_val_loss_split val`
- 当前每次 val loss 平均 batch 数：
  - `--headonly_val_loss_num_batches 8`
- W&B name:
  - `pybullet0626_diffsynth_object_heads_only_gpu67_fresh500_val`

### 14.4 当前启动前风险

虽然 `gpu6,7` 当前是空闲的，能够承接新的训练进程，但启动前还有两个客观风险：

- `/data` 剩余空间只有约 `4.8G`
- 旧目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0625_diffsynth_object_heads_only_gpu67`
  当前仍占约 `3.4G`

这意味着：

- 新 run 即使只保留两份 checkpoint，也仍然可能在 `step-500 / 1000` 附近再次遇到磁盘紧张
- 到 `step-2000` 时如果 benchmark validation 产物开始大量落盘，磁盘风险会更高

另外，当前 `gpu5` 已经有一个 `wan-cu128` 进程在占用约 `30.9G` 显存：

- 这不会影响主训练用 `gpu6,7`
- 但会影响后续 `step-2000` 的 benchmark / validation 子进程是否能按预期在 `gpu5` 成功启动

和 `_step = 7364` 相比，这一步最重要的变化是：

- `object_latent_tokens_abs_max` 从 `5.29770` 回落到 `5.15315`
- `grad_norm` 从 `1.39685` 回落到 `0.51101`
- `grad_abs_max` 也回落到 `0.05`

这说明：

- `7364` 那次更像一次短时抬升
- 到 `_step = 7384` 为止，还没有证据表明它正在继续扩散成持续失稳

当前判断进一步收敛为：

- 训练仍然处于“有风险波动，但可以自行回落”的阶段
- 眼下最重要的仍然是确认：
  - `step-007400` checkpoint 是否正常落盘
  - `step-008000` validation 是否能在 `gpu5` 正常触发

### 2026-06-26 02:42-02:43 UTC：`step-007400` 已成功落盘，训练重新回到较低风险区间

这一轮的重要进展：

- checkpoint 目录已经推进到：
  - `step-007200`
  - `step-007400`
- `step-007400/training_state.pt` 已确认：
  - `global_step = 7400`
  - `epoch_id = 2`
  - `batch_in_epoch = 1400`

这说明：

- checkpoint 落盘链路继续正常
- `--max_checkpoints_keep 2` 仍在生效
- 在 `7364` 那次局部抬升之后，训练并没有卡在 checkpoint 写盘上

validation 侧当前仍无变化：

- validation runtime 目录还没有新的 `step-008000` 文件
- `gpu5` 仍然空闲，等待后续 validation / benchmark
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7403`
- `train/loss_total = 0.01961`
- `train/loss_track_aux = 0.01360`
- `train/loss_box_aux = 0.17152`
- `train/loss_depth_aux = 0.01099`
- `train/grad_norm = 0.60130`
- `train/grad_abs_max = 0.09220`
- `train/object_context_abs_max = 0.39844`
- `train/object_latent_tokens_abs_max = 5.18060`

当前 interpretation：

- `step-007400` 已经证明训练可以稳定跨过前面那次局部抬升
- latest summary 也重新回到了较低风险区间
- `object_latent_tokens_abs_max` 仍然偏高，但没有继续沿着 `7364` 的方向加速上冲

接下来最关键的里程碑已经切换为：

- `step-008000` checkpoint
- `step-008000` validation 是否真正触发
- validation 是否确实跑在 `gpu5`
- validation 是否不再产生新的 `benchmark.failed.json`

### 2026-06-26 02:43-02:44 UTC：出现一波中等回弹，但尚未破前高

这一轮外部运行状态没有新变化：

- checkpoint 目录仍是：
  - `step-007200`
  - `step-007400`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7429`
- `train/loss_total = 0.04098`
- `train/loss_track_aux = 0.11500`
- `train/loss_box_aux = 0.23254`
- `train/loss_depth_aux = 0.06221`
- `train/grad_norm = 0.59956`
- `train/grad_abs_max = 0.09232`
- `train/object_context_abs_max = 0.39898`
- `train/object_latent_tokens_abs_max = 5.23899`

这一时刻的特征是：

- `track / box / depth` 都有一波中等抬升
- 但：
  - `grad_norm` 仍在约 `0.60`
  - `grad_abs_max` 也不高
  - `object_latent_tokens_abs_max` 虽然回到 `5.239` 左右，但还没有超过前面的局部高点 `5.29770`

因此当前判断更新为：

- 训练中仍然存在反复波动
- 但到 `_step = 7429` 为止，还没有出现比 `7364` 更坏的新阶段
- 当前更像一次“中等回弹但未破前高”

接下来继续重点盯：

- `step-007600`
- `step-007800`
- `step-008000` validation 是否真正落到 `gpu5`

### 2026-06-26 02:44-02:45 UTC：supervision 侧再次抬升，但 `object_latent_tokens_abs_max` 回到 `5.0` 以下

这一轮外部状态仍没有新变化：

- checkpoint 目录仍是：
  - `step-007200`
  - `step-007400`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7450`
- `train/loss_total = 0.06704`
- `train/loss_track_aux = 0.18592`
- `train/loss_box_aux = 0.36317`
- `train/loss_depth_aux = 0.12135`
- `train/grad_norm = 1.11732`
- `train/grad_abs_max = 0.29105`
- `train/object_context_abs_max = 0.39805`
- `train/object_latent_tokens_abs_max = 4.98859`

这一时刻的关键信息是：

- `track / box / depth` 和梯度又有一波抬升
- 但 `object_latent_tokens_abs_max` 没有继续在 `5.2+` 区间上冲，反而回落到了 `4.99`

这使当前 interpretation 更偏向：

- 这次更像 supervision 侧的局部抬升
- 而不是 object latent token 幅值继续恶化

当前判断更新为：

- 训练仍然存在批次级波动
- 但到 `_step = 7450` 为止，没有出现“supervision 抬升 + latent token 幅值继续创新高”同时发生的更坏组合
- 继续等待 `step-007600`，并保持对 `step-008000` validation 的重点监控

### 2026-06-26 02:46-02:47 UTC：中等波动继续，但 `object_latent_tokens_abs_max` 仍未破前高

这一轮外部运行状态仍未变化：

- checkpoint 目录仍是：
  - `step-007200`
  - `step-007400`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7475`
- `train/loss_total = 0.04691`
- `train/loss_track_aux = 0.07783`
- `train/loss_box_aux = 0.32933`
- `train/loss_depth_aux = 0.06192`
- `train/grad_norm = 0.59620`
- `train/grad_abs_max = 0.08986`
- `train/object_context_abs_max = 0.40085`
- `train/object_latent_tokens_abs_max = 5.22305`

这一时刻依然属于“中等波动”：

- `track / box / depth` 有一波中等抬升
- 但：
  - `grad_norm` 仍约 `0.60`
  - `grad_abs_max` 仍不高
  - `object_latent_tokens_abs_max` 回到 `5.223`，但仍没有超过前面的局部高点 `5.29770`

当前判断维持为：

- 训练有反复波动，但暂未进入更坏的新阶段
- 到 `_step = 7475` 为止，最关键的观察结论仍是“未破前高”
- 接下来继续等 `step-007600` 是否正常落盘，再继续守 `step-008000` validation

### 2026-06-26 02:47-02:48 UTC：latest summary 与前一轮同型，仍未突破 `5.29770`

这一轮外部状态仍无变化：

- checkpoint 目录仍是：
  - `step-007200`
  - `step-007400`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7500`
- `train/loss_total = 0.03659`
- `train/loss_track_aux = 0.12151`
- `train/loss_box_aux = 0.19861`
- `train/loss_depth_aux = 0.04575`
- `train/grad_norm = 1.10763`
- `train/grad_abs_max = 0.28485`
- `train/object_context_abs_max = 0.39591`
- `train/object_latent_tokens_abs_max = 5.23373`

这一时刻和 `_step = 7475` 很接近：

- `track / box / depth` 仍是中等抬升
- `grad_norm` 再次来到约 `1.11`
- 但 `object_latent_tokens_abs_max` 仍然只是在 `5.23` 左右，没有突破前面的局部高点 `5.29770`

当前判断继续保持：

- 训练仍处于“有波动，但未升级成更坏阶段”的状态
- 截至 `_step = 7500`，仍然没有出现新的更坏高点
- 接下来继续等待 `step-007600` checkpoint 和 `step-008000` validation

### 2026-06-26 02:48-02:49 UTC：出现一次低 loss 下的梯度尖峰，先按单次事件观察

这一轮外部状态仍无变化：

- checkpoint 目录仍是：
  - `step-007200`
  - `step-007400`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7526`
- `train/loss_total = 0.01587`
- `train/loss_track_aux = 0.02542`
- `train/loss_box_aux = 0.09272`
- `train/loss_depth_aux = 0.04059`
- `train/grad_norm = 1.77312`
- `train/grad_abs_max = 0.49363`
- `train/object_context_abs_max = 0.40371`
- `train/object_latent_tokens_abs_max = 5.15694`

这一时刻的组合比较特殊：

- `loss_total` 仍然很低
- `track / box / depth` 也不高
- 但 `grad_norm` 和 `grad_abs_max` 比前一轮明显抬高

同时也要注意：

- `object_latent_tokens_abs_max` 只是约 `5.16`
- 并没有伴随新的 latent-token 幅值高点

因此当前 interpretation 是：

- 这更像一次“低 loss 下的梯度尖峰”
- 目前先按单次事件观察
- 只有当后续连续出现类似的高梯度 summary，才值得升级为更严肃的梯度稳定性排查

### 2026-06-26 02:49-02:50 UTC：高梯度尖峰已回落，暂不支持“连续梯度异常”判断

这一轮外部状态仍无变化：

- checkpoint 目录仍是：
  - `step-007200`
  - `step-007400`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7546`
- `train/loss_total = 0.02453`
- `train/loss_track_aux = 0.05444`
- `train/loss_box_aux = 0.17449`
- `train/loss_depth_aux = 0.01635`
- `train/grad_norm = 0.86955`
- `train/grad_abs_max = 0.20701`
- `train/object_context_abs_max = 0.40437`
- `train/object_latent_tokens_abs_max = 5.18171`

和 `_step = 7526` 相比：

- `grad_norm` 从 `1.77312` 回落到 `0.86955`
- `grad_abs_max` 从 `0.49363` 回落到 `0.20701`

这说明：

- 上一轮的高梯度更像一次单次尖峰
- 到 `_step = 7546` 为止，还不支持“连续梯度异常”的判断

当前判断更新为：

- 训练仍然在波动，但当前还没有足够证据说明梯度稳定性开始系统性恶化
- 接下来继续等待 `step-007600` checkpoint，同时继续盯 `step-008000` validation

### 2026-06-26 02:51-02:52 UTC：外部 loss/grad 再次回低，但 `object_latent_tokens_abs_max` 刷新到 `5.37796`

这一轮外部状态仍无变化：

- checkpoint 目录仍是：
  - `step-007200`
  - `step-007400`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7577`
- `train/loss_total = 0.01340`
- `train/loss_track_aux = 0.02061`
- `train/loss_box_aux = 0.09439`
- `train/loss_depth_aux = 0.01903`
- `train/grad_norm = 0.50987`
- `train/grad_abs_max = 0.05000`
- `train/object_context_abs_max = 0.40248`
- `train/object_latent_tokens_abs_max = 5.37796`

这一时刻最值得注意的组合是：

- 外部可见训练量重新回到低位：
  - `loss_total` 低
  - `track / box / depth` 都低
  - `grad_norm / grad_abs_max` 也低
- 但 `object_latent_tokens_abs_max` 单独刷新了新的高点 `5.37796`

因此当前 interpretation 需要再细化一步：

- 目前最突出的风险信号已经更明确地集中在 `object_latent_tokens_abs_max`
- 但到 `_step = 7577` 为止，它仍然更像“内部幅值单独抬升”
- 还没有与外部 loss / 梯度恶化、checkpoint 异常、或 validation 失败形成直接绑定

接下来最关键的观察点变成：

- `step-007600` checkpoint 是否正常落盘
- 接近 `step-008000` 时，这个新的 `5.37796` 高点是否开始和更坏的外部症状绑定

### 2026-06-26 02:52-02:53 UTC：`step-007600` 已成功落盘，当前表现为局部 `box_aux` spike

这一轮的重要进展：

- checkpoint 目录已经推进到：
  - `step-007400`
  - `step-007600`
- `step-007600/training_state.pt` 已确认：
  - `global_step = 7600`
  - `epoch_id = 2`
  - `batch_in_epoch = 1600`

这说明：

- checkpoint 落盘继续正常
- 到 `7600` 为止，训练和保存链路都没有出现异常

validation 侧当前仍无变化：

- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 仍然空闲，等待后续 validation / benchmark
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7601`
- `train/loss_total = 0.06503`
- `train/loss_track_aux = 0.02508`
- `train/loss_box_aux = 0.60516`
- `train/loss_depth_aux = 0.02009`
- `train/grad_norm = 0.83574`
- `train/grad_abs_max = 0.19521`
- `train/object_context_abs_max = 0.39998`
- `train/object_latent_tokens_abs_max = 5.24101`

这一时刻的特征比较清楚：

- `box_aux` 单独抬高得更明显
- 但：
  - `track_aux` 不高
  - `depth_aux` 不高
  - `grad_norm / grad_abs_max` 没有同步进入异常高位
  - `object_latent_tokens_abs_max` 也没有继续逼近前面的 `5.37796`

因此当前判断更新为：

- `step-007600` 已经证明训练可以继续稳定推进
- 当前更像一次局部 `box_aux` spike
- 还没有看到它与更坏的整体失稳绑定

接下来最关键的里程碑继续保持为：

- `step-007800`
- `step-008000`
- `step-008000` validation 是否真正触发到 `gpu5`
- validation 是否还会产生新的 `benchmark.failed.json`

### 2026-06-26 02:53-02:54 UTC：`7601` 的 `box_aux` spike 已回落，训练重新回到低损失区

这一轮外部状态仍无变化：

- checkpoint 目录仍是：
  - `step-007400`
  - `step-007600`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7632`
- `train/loss_total = 0.01506`
- `train/loss_track_aux = 0.02246`
- `train/loss_box_aux = 0.09616`
- `train/loss_depth_aux = 0.03195`
- `train/grad_norm = 1.07295`
- `train/grad_abs_max = 0.28223`
- `train/object_context_abs_max = 0.40041`
- `train/object_latent_tokens_abs_max = 5.24203`

和 `_step = 7601` 相比：

- `box_aux` 从 `0.60516` 明显回落到 `0.09616`
- `loss_total` 也回到较低区间

这说明：

- `7601` 那次更像一次局部 `box_aux` spike
- 到 `_step = 7632` 为止，它并没有继续扩散成持续性异常

当前判断继续保持为：

- 训练仍然有波动，但能自行回落
- 接下来继续重点盯：
  - `step-007800`
  - `step-008000`
  - `step-008000` validation 是否会真正落到 `gpu5`

### 2026-06-26 02:55-02:56 UTC：`object_latent_tokens_abs_max` 再刷新到 `5.38170`，但仍未和整体失稳绑定

这一轮外部状态仍无变化：

- checkpoint 目录仍是：
  - `step-007400`
  - `step-007600`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7657`
- `train/loss_total = 0.04729`
- `train/loss_track_aux = 0.08682`
- `train/loss_box_aux = 0.36048`
- `train/loss_depth_aux = 0.02555`
- `train/grad_norm = 0.61561`
- `train/grad_abs_max = 0.10165`
- `train/object_context_abs_max = 0.40240`
- `train/object_latent_tokens_abs_max = 5.38170`

这一时刻需要区分开的点是：

- `object_latent_tokens_abs_max` 再次刷新前高，已经到 `5.38170`
- 但与此同时：
  - `grad_norm` 只有约 `0.62`
  - `grad_abs_max` 也不高
  - `box_aux` 虽然抬升，但仍属于中等范围
  - `depth_aux` 不高

因此当前判断更新为：

- 内部幅值风险信号仍在继续刷新高点
- 但到 `_step = 7657` 为止，它仍然没有与“整体失稳”形成明确绑定
- 接下来接近 `step-008000` 时，需要特别留意这类内部幅值新高是否开始与：
  - 更高、更持续的 `box/track/depth` 抬升
  - 更明显的 `grad_norm / grad_abs_max` 抬升
  - 或 validation 失败
  同时出现

### 2026-06-26 02:56-02:57 UTC：外部训练量再次回低，但 `object_latent_tokens_abs_max` 继续刷新到 `5.40501`

这一轮外部状态仍无变化：

- checkpoint 目录仍是：
  - `step-007400`
  - `step-007600`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7682`
- `train/loss_total = 0.01391`
- `train/loss_track_aux = 0.02724`
- `train/loss_box_aux = 0.09250`
- `train/loss_depth_aux = 0.01938`
- `train/grad_norm = 0.60943`
- `train/grad_abs_max = 0.09872`
- `train/object_context_abs_max = 0.40608`
- `train/object_latent_tokens_abs_max = 5.40501`

这一时刻的组合进一步强化了前面的判断：

- 外部可见训练量再次回到低位：
  - `loss_total` 低
  - `track / box / depth` 低
  - `grad_norm / grad_abs_max` 也不高
- 但 `object_latent_tokens_abs_max` 单独继续刷新新高，到 `5.40501`

因此当前判断继续收敛为：

- 当前最需要盯防的风险信号，已经高度集中在 `object_latent_tokens_abs_max`
- 但截至 `_step = 7682`，它仍然主要表现为“内部幅值单独抬升”
- 还没有和外部训练失稳、checkpoint 异常或 validation 失败形成直接绑定

接下来临近 `step-008000` 时，需要优先确认两件事：

- `step-007800` checkpoint 是否正常落盘
- `step-008000` validation 是否会把这种内部幅值新高转化成实际 failure signal

### 2026-06-26 02:57-02:58 UTC：`object_latent_tokens_abs_max` 继续小幅刷新，但外部训练量仍未同步恶化

这一轮外部状态仍无变化：

- checkpoint 目录仍是：
  - `step-007400`
  - `step-007600`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7702`
- `train/loss_total = 0.02468`
- `train/loss_track_aux = 0.05761`
- `train/loss_box_aux = 0.17079`
- `train/loss_depth_aux = 0.01843`
- `train/grad_norm = 0.84335`
- `train/grad_abs_max = 0.20016`
- `train/object_context_abs_max = 0.41693`
- `train/object_latent_tokens_abs_max = 5.41086`

这一时刻的核心判断与前一轮一致，但证据更强了一点：

- `object_latent_tokens_abs_max` 又小幅刷新，从 `5.40501` 到 `5.41086`
- 但：
  - `loss_total` 仍低
  - `track / box / depth` 仍不高
  - `grad_norm / grad_abs_max` 也没有同步进入异常高位

因此当前判断继续保持为：

- 风险信号仍主要集中在 `object_latent_tokens_abs_max`
- 到 `_step = 7702` 为止，它依然没有和外部训练恶化形成直接绑定
- 接下来最关键的仍然是：
  - `step-007800` checkpoint 是否正常落盘
  - `step-008000` validation 是否会第一次把这个内部风险转化成外部 failure

### 2026-06-26 02:58-02:59 UTC：出现一波中等回弹，但 `object_latent_tokens_abs_max` 反而从前高回落

这一轮外部状态仍无变化：

- checkpoint 目录仍是：
  - `step-007400`
  - `step-007600`
- validation runtime 目录仍没有新的 `step-008000` 文件
- `gpu5` 继续空闲
- `gpu4` 仍未使用
- `/data` 可用空间仍约 `5.1G`

W&B latest summary（run `qberfq1r`）推进到：

- `_step = 7728`
- `train/loss_total = 0.05790`
- `train/loss_track_aux = 0.12709`
- `train/loss_box_aux = 0.35389`
- `train/loss_depth_aux = 0.09802`
- `train/grad_norm = 0.61088`
- `train/grad_abs_max = 0.09901`
- `train/object_context_abs_max = 0.40322`
- `train/object_latent_tokens_abs_max = 5.35829`

这一时刻更像一次中等回弹：

- `track / box / depth` 和 `loss_total` 都有一波中等抬升
- 但：
  - `grad_norm` 仍然只有约 `0.61`
  - `grad_abs_max` 不高
  - `object_latent_tokens_abs_max` 反而从前一轮的 `5.41086` 回落到 `5.35829`

因此当前判断更新为：

- 到 `_step = 7728` 为止，仍然更像“中等回弹”而不是系统性恶化
- 目前还没有出现“内部幅值继续冲高 + 外部 loss/grad 同步恶化”的更坏组合
- 接下来继续重点守：
  - `step-007800`
  - `step-008000`
  - `step-008000` validation 是否会真正开始暴露 failure signal

### 2026-06-26 03:01 UTC：继续确认 GPU 绑定无误，训练仍在推进到 `step-008000` 前夜

这一轮先重新核对了运行时 GPU 绑定，结果和预期一致：

- 训练主进程仍是：
  - `accelerate launch ... train_v_newtrain.py`
  - 参数里继续固定：
    - 训练：`CUDA_VISIBLE_DEVICES=6,7`
    - 验证：`--benchmark_cuda_visible_devices 5`
- 当前 GPU 占用：
  - `gpu6 = 42725 / 49140 MiB`
  - `gpu7 = 42707 / 49140 MiB`
  - `gpu5 = 1 / 49140 MiB`
  - `gpu4 = 1 / 49140 MiB`
- 这说明：
  - 训练仍稳定跑在 `gpu6,7`
  - `gpu5` 仍预留给 validation
  - 坏卡 `gpu4` 没有被用到

checkpoint / validation 侧仍然还没出现新的外部里程碑：

- checkpoint 目录仍只有：
  - `step-007400`
  - `step-007600`
- 对应 `training_state.pt` 仍是：
  - `step-007400`: `global_step=7400`, `epoch_id=2`, `batch_in_epoch=1400`
  - `step-007600`: `global_step=7600`, `epoch_id=2`, `batch_in_epoch=1600`
- validation runtime 目录仍只有旧失败记录：
  - `step-002000`
  - `step-004000`
  - `step-006000`
- 还没有新的：
  - `step-007800`
  - `step-008000`
  - `run_validation_vbench.py`
  - `batch_eval_lora.py`

磁盘压力仍然是当前最现实的外部风险：

- `/data` 可用空间仍约 `5.1G`
- 下一次 `checkpoint + validation` 同步落盘时，仍要重点防止因为磁盘紧张导致新的外部 failure

W&B latest summary（run `qberfq1r`）这一轮已经推进到 `_step = 7743`，最近几条关键点如下：

- `_step = 7680`
  - `loss_total = 0.10618`
  - `track_aux = 0.03714`
  - `box_aux = 0.42912`
  - `depth_aux = 0.59555`
  - `grad_norm = 1.15932`
  - `object_latent_tokens_abs_max = 5.35991`
- `_step = 7689`
  - `loss_total = 0.12935`
  - `track_aux = 0.27991`
  - `box_aux = 0.42486`
  - `depth_aux = 0.58869`
  - `grad_norm = 1.44077`
  - `object_latent_tokens_abs_max = 5.37060`
- `_step = 7695`
  - `loss_total = 0.02303`
  - `track_aux = 0.05409`
  - `box_aux = 0.15999`
  - `depth_aux = 0.01619`
  - `grad_norm = 0.83921`
  - `object_latent_tokens_abs_max = 5.43711`
- `_step = 7741`
  - `loss_total = 0.03974`
  - `track_aux = 0.15184`
  - `box_aux = 0.18804`
  - `depth_aux = 0.05752`
  - `grad_norm = 0.83588`
  - `object_latent_tokens_abs_max = 5.44224`
- `_step = 7743`
  - `loss_total = 0.04781`
  - `track_aux = 0.05087`
  - `box_aux = 0.40432`
  - `depth_aux = 0.02289`
  - `grad_norm = 0.50865`
  - `grad_abs_max = 0.05000`
  - `object_latent_tokens_abs_max = 5.44839`

这一轮判断继续收敛为：

- `object_latent_tokens_abs_max` 仍在缓慢创新高
- 但到 `_step = 7743` 为止：
  - 还没有和持续高 `grad_norm`
  - 持续高 `loss_total`
  - 或 validation failure
  形成稳定绑定
- 目前最合理的表述仍然是：
  - 内部幅值风险信号在上升
  - 但尚未被外部训练失稳证实

接下来继续重点盯：

- `step-007800` 是否正常落盘
- `step-008000` 是否第一次触发新的 validation
- validation 是否严格使用 `gpu5`
- `step-008000` 是否出现新的 `benchmark.failed.json`

### 2026-06-26 03:03 UTC：`step-007800` 已确认落盘，训练继续健康推进到 `step-008000` 前

这一轮先确认了 cache / 训练 / checkpoint 三件事：

- VGGT cache 已存在并持续可用：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/vggt_cache/sample_000584_w001.vggt.pt`
  - `/data/gaoya/AAA_test_video/0623/train/train0624/vggt_cache/sample_000401_w000.vggt.pt`
  - `/data/gaoya/AAA_test_video/0623/train/train0624/vggt_cache/sample_000208_w002.vggt.pt`
  - 可见 cache 根目录下已有大量 `*.vggt.pt`
- 训练进程仍正常：
  - `accelerate launch ... train_v_newtrain.py`
  - 两个 worker 仍在跑
- GPU 绑定仍正确：
  - 训练：`gpu6,7`
  - validation 预留：`gpu5`
  - 坏卡 `gpu4` 仍未使用

GPU 实际占用这一轮为：

- `gpu6 = 42725 / 49140 MiB`
- `gpu7 = 42707 / 49140 MiB`
- `gpu5 = 1 / 49140 MiB`
- `gpu4 = 1 / 49140 MiB`

checkpoint 侧出现了新的正向进展：

- checkpoint 目录现在是：
  - `step-007600`
  - `step-007800`
- 对应 `training_state.pt`：
  - `step-007600`: `global_step=7600`, `epoch_id=2`, `batch_in_epoch=1600`
  - `step-007800`: `global_step=7800`, `epoch_id=2`, `batch_in_epoch=1800`

这说明：

- `step-007800` 已经正常保存
- `max_checkpoints_keep=2` 仍在生效
- 训练本体没有因为 cache 接入或 object branch 训练而卡住

validation 这一轮仍未开始新的外部执行：

- validation runtime 目录仍只有旧失败：
  - `step-002000`
  - `step-004000`
  - `step-006000`
- 还没有新的：
  - `step-008000`
  - `run_validation_vbench.py`
  - `batch_eval_lora.py`

所以当前最关键的下一跳仍是：

- `step-008000` checkpoint 落盘
- `step-008000` validation 首次触发

W&B latest summary（run `qberfq1r`）已经推进到 `_step = 7820`，最近一段更有代表性的点如下：

- `_step = 7769`
  - `loss_total = 0.07202`
  - `track_aux = 0.05648`
  - `box_aux = 0.65252`
  - `depth_aux = 0.01117`
  - `grad_norm = 1.14845`
  - `grad_abs_max = 0.30674`
  - `object_latent_tokens_abs_max = 5.39778`
- `_step = 7774`
  - `loss_total = 0.04310`
  - `track_aux = 0.09060`
  - `box_aux = 0.25046`
  - `depth_aux = 0.08990`
  - `grad_norm = 1.73016`
  - `grad_abs_max = 0.49604`
  - `object_latent_tokens_abs_max = 5.44789`
- `_step = 7790`
  - `loss_total = 0.03474`
  - `track_aux = 0.09824`
  - `box_aux = 0.20899`
  - `depth_aux = 0.04013`
  - `grad_norm = 1.09267`
  - `grad_abs_max = 0.28882`
  - `object_latent_tokens_abs_max = 5.46029`
- `_step = 7806`
  - `loss_total = 0.13316`
  - `track_aux = 0.12840`
  - `box_aux = 0.70964`
  - `depth_aux = 0.49361`
  - `grad_norm = 0.62580`
  - `grad_abs_max = 0.10817`
  - `object_latent_tokens_abs_max = 5.39935`
- `_step = 7814`
  - `loss_total = 0.01422`
  - `track_aux = 0.01637`
  - `box_aux = 0.09179`
  - `depth_aux = 0.03405`
  - `grad_norm = 0.50981`
  - `grad_abs_max = 0.05000`
  - `object_latent_tokens_abs_max = 5.46156`
- `_step = 7820`
  - `loss_total = 0.11040`
  - `track_aux = 0.07898`
  - `box_aux = 0.41299`
  - `depth_aux = 0.61204`
  - `grad_norm = 0.87664`
  - `grad_abs_max = 0.21266`
  - `object_latent_tokens_abs_max = 5.46312`

这一轮的判断是：

- `box_aux / depth_aux` 仍会出现明显的 batch-level spike
- 但这些 spike 目前仍然更像“单批次监督难样本扰动”，因为：
  - spike 后能很快回落
  - `grad_norm` 没有同步持续抬高
  - `grad_abs_max` 也没有持续失控
- `object_latent_tokens_abs_max` 继续缓慢创新高，到 `_step = 7820` 已到 `5.46312`
- 但截至目前，它依旧没有和外部 failure 形成稳定绑定

当前结论继续维持：

- 训练仍在健康推进
- 还没有看到需要立刻改代码的 runtime 级错误
- 现在最值得防的两件事：
  - `step-008000` validation 触发时再次出现脚本级错误
  - `/data` 只剩约 `5.1G`，checkpoint + validation 同时写盘时可能触发外部失败

### 2026-06-26 03:11 UTC：当前 run 实际已经停止，训练最高推进到 `7898` 左右，但最后可恢复 checkpoint 仍是 `step-007800`

本轮重新核对后，当前运行状态和上一轮相比已经发生变化：

- 训练进程已经全部退出：
  - `accelerate launch` 不在
  - `train_v_newtrain.py` worker 不在
- GPU 当前全空：
  - `gpu6 = 1 / 49140 MiB`
  - `gpu7 = 1 / 49140 MiB`
  - `gpu5 = 1 / 49140 MiB`
  - `gpu4 = 1 / 49140 MiB`

因此当前不是“还在接近 `step-008000` 继续推进”，而是：

- 这次 run 已经停了
- 还没跑到 `step-008000`

本地 checkpoint 当前仍只有：

- `step-007600`
- `step-007800`

并已核对：

- `step-007600/training_state.pt`
  - `global_step = 7600`
  - `epoch_id = 2`
  - `batch_in_epoch = 1600`
- `step-007800/training_state.pt`
  - `global_step = 7800`
  - `epoch_id = 2`
  - `batch_in_epoch = 1800`

但 W&B 本地 `output.log` 说明，这次训练实际并不只停在 `7800`：

- 日志里已确认：
  - `epoch 2 | global_step 7800: 100%`
- 随后又进入下一轮：
  - `epoch 3 | global_step 7898: 5%`

所以当前更准确的训练进度应表述为：

- 最新实际训练步数：
  - 约 `7897 ~ 7898`
- 最新安全可恢复权重：
  - `step-007800`
- 尚未达到：
  - `step-008000`
- 因此也尚未出现新的：
  - `step-008000` validation runtime 文件
  - `run_validation_vbench.py`
  - `batch_eval_lora.py`

W&B 当前状态（run `qberfq1r`）：

- state:
  - `failed`
- url:
  - `https://wandb.ai/875222004-gy/vjepa_vggt_wan/runs/qberfq1r`

尾段 loss / 梯度 / 幅值信号的总体特征：

- `loss_total`
  - 大多数仍在 `0.01 ~ 0.06`
  - 但间歇性 spike 到 `0.11 ~ 0.16`
- `loss_track_aux`
  - 常见在 `0.02 ~ 0.15`
- `loss_box_aux`
  - 仍是波动最大的一项
  - 尾段可见局部高点：
    - `0.73`
    - `0.97`
- `loss_depth_aux`
  - 常见在 `0.01 ~ 0.09`
  - 尾段多次 spike 到：
    - `0.47 ~ 0.62`
- `grad_norm`
  - 常见在 `0.5 ~ 1.2`
  - 尾段几次明显抬升到：
    - `1.50`
    - `1.79`
    - `1.86`
- `object_context_abs_max`
  - 仍基本稳定在：
    - `0.40 ~ 0.42`
- `object_latent_tokens_abs_max`
  - 继续成为最值得警惕的内部信号
  - 尾段进一步抬到：
    - `5.46`
    - `5.49`
    - `5.54`
    - `5.57`

当前 interpretation 更新为：

- 这次 run 停掉前，并没有表现成“整体 loss 持续爆炸”
- 更像：
  - `box_aux / depth_aux` 重复出现 batch-level spike
  - 同时 `object_latent_tokens_abs_max` 继续缓慢上漂
- 其中最重要的风险信号仍然是：
  - `object_latent_tokens_abs_max`
- 因为到尾段为止，它已经不只是偶发触到 `5.0`
  - 而是稳定进入 `5.3 ~ 5.57` 区间

但也需要如实强调：

- 到停掉前，这个内部幅值信号仍没有和“明确不可恢复的外部发散”完全绑定
- 因为尾段仍然能看到：
  - 很低的 `loss_total`
  - 很低的 `box_aux / depth_aux`
  - 与较高 `object_latent_tokens_abs_max` 同时出现

当前 operational 结论：

- 训练目前已经停了
- 还没跑到 `step-008000`
- 所以还没有进入我们最关键的下一观察点：
  - `step-008000` validation 是否会在 `gpu5` 上真正触发并跑通

### 2026-06-26 04:33 UTC：新的 fresh run 在 `step-000500` 后首次 head-only `val_loss` 触发时退出，根因是 `val` split 缺 VGGT cache

新开的 fresh run：

- W&B run id:
  - `m13k8mf2`
- run name:
  - `pybullet0626_diffsynth_object_heads_only_gpu67_fresh500_val`

这轮 fresh run 本体训练先是正常推进，并已成功产生第一份 checkpoint：

- checkpoint:
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_heads_only_gpu67_fresh500_val/checkpoints/step-000500`
- `training_state.pt` 已确认：
  - `global_step = 500`

真正导致进程退出的不是训练主 loop 本身，而是 `step 500` 之后第一次触发轻量 head-only `val_loss` 时，读取 `val` split 样本的 VGGT cache 失败：

- traceback 根因：
  - `FileNotFoundError: VGGT cache not found under /data/gaoya/AAA_test_video/0623/train/train0624/vggt_cache`
- 首个报错缺失文件：
  - `sample_000301_w000.vggt.pt`

解释：

- 当前 `train` 与 `val` 是独立数据集
- 之前 `vggt_cache` 目录里只有 `train` 对应的 `3600` 份 cache
- 但新增的 `headonly_val_loss_split=val` 会在 `step 500` 真正读取 `val` 样本
- 因为 `val` 对应的 `450` 份 VGGT cache 当时还没准备，所以第一次 `val_loss` sweep 直接触发严格模式缺文件退出

### 2026-06-26 04:55 UTC：`val` split 的 `450` 份 VGGT cache 已全部补齐，并从 `step-000500` 成功恢复

为避免再次重写原始 `cache_vggt_dense_features.py` 里那份异常巨大的：

- `/data/gaoya/AAA_test_video/0623/train/train0624/vggt_cache/manifest.jsonl`

这次单独新增了一个只负责回填 `val` split `.vggt.pt` 文件的轻量脚本：

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/cache_vggt_val_split.py`

执行方式：

- 使用 GPU：
  - `gpu2`
  - `gpu3`
- 严格避开：
  - `gpu4`
- 分片：
  - `val [0, 225)`
  - `val [225, 450)`

回填完成后的覆盖率核查：

- `val_json_count = 450`
- `missing_count = 0`

说明当前 `val` split 所有样本都已经具备训练时需要的同名 VGGT cache 文件。

随后新增恢复脚本：

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_heads_only_gpu67_resume_500_val.sh`

从：

- `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_heads_only_gpu67_fresh500_val/checkpoints/step-000500`

继续恢复训练。

恢复 run：

- W&B run id:
  - `lffzw16q`
- run name:
  - `pybullet0626_diffsynth_object_heads_only_gpu67_resume500_val`

恢复日志已明确确认：

- `Loading training state from .../step-000500/training_state.pt`
- `Restored training state: global_step=500, epoch_id=0, batch_in_epoch=500, model_logger_num_steps=500`

随后训练已重新进入主循环并确认继续推进：

- `global_step 501`
- `global_step 502`
- ...
- 已明确看到推进到：
  - `global_step 513`

这说明：

- 之前在 `step 500` 停掉的直接原因已经修复
- 补齐 `val` split VGGT cache 后，head-only `val_loss` 链路不再因为首次触发时缺 cache 而立即退出
- 当前恢复 run 已重新健康运行在：
  - `gpu6,7`

### 2026-06-26 13:xx UTC：Stage2 fresh run 持续推进到 `global_step 4070`，等待 `step-004500` 做推理 smoke

当前活跃的 Stage2 fresh run：

- W&B run id：
  - `4n1dtaoh`
- run name：
  - `pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun`
- 输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun`

本轮运行的关键确认：

- 成功从 `step-004000` 恢复模型权重与训练进度
- optimizer 参数组不兼容时已自动跳过 optimizer / scheduler state restore
- `forward / backward / optimizer step` 已真实开始推进

当前最新训练进度：

- 从 `global_step 4000` 恢复后，已继续推进到：
  - `global_step 4070`

当前还没有新的 fresh checkpoint 目录，原因符合预期：

- 本 run 使用：
  - `save_steps = 500`
- 从：
  - `global_step 4000`
  恢复
- 因此下一份预期新 checkpoint 仍然是：
  - `step-004500`

下一步验证计划已经确定：

1. 等 `step-004500` 真正落盘
2. 立刻使用单 checkpoint 推理脚本做一次 smoke

推荐推理命令模板：

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=5 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/infer_v_newtrain_context_video_wan.py \
  --checkpoint /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun/checkpoints/step-004500 \
  --context-video /data/gaoya/AAA_test_video/0529/vjepa_vggt/test/sample_000339_w000_input_context.mp4 \
  --prompt "industrial rigid body simulation sphere" \
  --output-dir /data/gaoya/AAA_test_video/0623/train/train0624/infer_stage2_freshrun_step004500 \
  --output-video /data/gaoya/AAA_test_video/0623/train/train0624/infer_stage2_freshrun_step004500/prediction.mp4
```

这一步的验证目标：

- fresh checkpoint 是否可被脚本正确解析
- 是否能成功产出 `prediction.mp4`
- 是否会暴露新的 LoRA / object branch / checkpoint 兼容性问题

### 2026-06-26 13:44 UTC：Stage2 fresh run 仍健康推进，尚未到 `step-004500`

对当前 fresh run 的实时复查结果：

- 活跃 run 仍是：
  - W&B run id：
    - `4n1dtaoh`
  - run name：
    - `pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun`
- 当前训练 launcher / workers 仍存活：
  - `accelerate launch ... run_train_v_newtrain_object_stage2_freeze_heads_gpu67_freshrun`
  - 两个 `train_v_newtrain.py` worker 仍在
- 当前 GPU 状态：
  - `gpu6`：
    - `45963 MiB` 已占用
  - `gpu7`：
    - `45963 MiB` 已占用
  - `gpu5`：
    - 约 `6689 MiB`
    - 但当前没有新的 stage2 validation / inference 子进程

stdout 最新可见进度：

- 从：
  - `global_step 4000`
  恢复后
- 当前已推进到至少：
  - `global_step 4137`

当前 checkpoint 目录复查：

- 仍只有 run 根目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun`
- 还没有新的 `checkpoints/step-*` 子目录

这和当前配置是一致的：

- 本 run 使用：
  - `save_steps = 500`
- 且是从：
  - `step-004000`
  恢复
- 因此下一份预期新 checkpoint 仍然是：
  - `step-004500`

当前判断：

- 这条 fresh run 仍在健康推进
- 尚未看到新的 forward / backward / optimizer 级异常
- 目前还不能做 `step-004500` inference smoke，因为对应 checkpoint 还没落盘
- 下一次关键复查点仍然是：
  - `step-004500` 是否成功生成
  - 生成后是否能在 `gpu5` 上跑通单 checkpoint 推理并产出 `prediction.mp4`

### 2026-06-26 13:50-13:53 UTC：按当前代码热重启训练，已从 `interrupted-latest` 成功恢复到 `4220+`

这次为了让新补丁生效：

- 显式给正在运行的 Stage2 fresh run 发送了 `SIGINT`
- 没有强杀
- 目的是让训练主循环自己走 `TrainingInterrupted` 分支，先安全落一份中断恢复点

中断结果：

- stdout 已明确打印：
  - `Training interrupted at step 4220. Saving interrupt checkpoint.`
- 新生成目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun/checkpoints/interrupted-latest`
- 其中已包含：
  - `checkpoint.safetensors`
  - `training_state.pt`

中断态内部状态已核对：

- `global_step = 4220`
- `epoch_id = 1`
- `batch_in_epoch = 220`
- `model_logger_num_steps = 4220`

随后已使用当前最新代码重新启动训练，恢复源改为：

- `--resume_from .../checkpoints/interrupted-latest`

本次重启后新 W&B run：

- run id：
  - `skxps3tw`
- run name 仍是：
  - `pybullet0626_diffsynth_object_stage2_freeze_heads_from004000_gpu67_freshrun`

恢复日志已确认：

- `Restored training state: global_step=4220, epoch_id=1, batch_in_epoch=220, model_logger_num_steps=4220, optimizer_state_restored=True, scheduler_state_restored=True`

恢复后的训练推进已确认：

- `global_step 4221`
- `global_step 4222`
- `global_step 4223`
- `global_step 4224`
- `global_step 4225`
- `global_step 4226`

同时 GPU 状态再次确认：

- `gpu6`：
  - 约 `45963 MiB`
- `gpu7`：
  - 约 `45963 MiB`

这次重启的直接收益：

- 新补丁已经真正被当前活跃训练进程吃到
- 后续 W&B 应开始稳定记录：
  - `train/loss_total`
  - `train/loss_main`
  - `train/loss_track_aux`
  - `train/loss_box_aux`
  - `train/loss_depth_aux`

额外观察：

- 恢复日志里打印了：
  - `LoRA checkpoint loaded: ... total 0 keys`
- 但对中断点文件的离线核对显示：
  - `checkpoint.safetensors` 实际包含 `407` 个 tensor key
- 同时：
  - `training_state.pt` 成功恢复
  - `optimizer_state_restored=True`
  - `scheduler_state_restored=True`
  - 训练已经从 `4220` 推进到 `4226+`

因此当前判断是：

- 这条信息更像现有加载日志口径与 checkpoint 内容类型之间的不一致
- 目前还没有证据表明这阻断了真实恢复或训练推进
- 先继续让训练向 `step-004500` 推进，必要时再单独定位这条 LoRA key 统计日志

### 2026-06-27：depth GT 默认切换为 `Depth Anything pooled GT`

当前 `train_v_newtrain.py` 已新增一条独立的 depth target 分支：

- `depth_target_source=depth_anything_box`

这条分支不会覆盖旧的 `state GT` 代码，而是与旧逻辑并存：

- `depth_target_source=depth_anything_box`
  - 默认值
  - 训练时优先从离线 cache 读取 `Depth Anything` 的 dense depth
  - 再按当前训练已有的 matched GT box 做 box-median pooling
  - 最后仍按原先 `latent_frames` 规则做 `group_last(...)`
  - 得到与旧口径完全兼容的监督 shape：
    - `[B, T_lat, O, 1]`
- `depth_target_source=state`
  - 保留旧逻辑
  - 仍然从：
    - `context_states[..., depth_target_state_index]`
    构造 GT

也就是说：

- `depth_head` 的输出 shape 没变
- `depth_aux_loss = L1(pred_depth, gt_depth)` 没变
- 只是 `gt_depth` 的默认来源从 `state` 切到了 `Depth Anything pooled GT`

新增代码位置：

- 训练侧 cache loader：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/utils/depth_anything_cache.py`
- 训练侧 pooled GT helper：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/utils/depth_target_branch.py`
- 训练主逻辑接入：
  - `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_v_newtrain.py`

当前默认新增参数：

- `--depth_target_source depth_anything_box`
- `--depth_anything_cache_root /data/gaoya/AAA_test_video/0623/train/train0624/depth_anything_cache`

重要说明：

- 这次默认切换不是“在线跑 Depth Anything”
- 训练不会在每个 step 内动态跑 depth model
- 必须先准备好离线 cache，否则训练会显式报错

离线 cache 生成脚本已新增：

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/cache_depth_anything_from_phys_state.py`

当前 cache 文件命名规则：

- `sample_xxx_wyyy.depth_anything.pt`

当前 cache payload 至少包含：

- `depth_frames`
  - shape: `[T_ctx, H, W]`
  - 当前实现中是归一化到 `[0,1]` 的灰度 depth
- `frame_indices`
- `source_video`
- `q_low`
- `q_high`

训练端行为：

- 读取 cache 后，用当前 matched GT boxes 做：
  - `pool_depth_from_boxes_median(...)`
- 然后做：
  - `group_last(...)`
- 得到最终 `gt_depth`

当前已经同步改过的启动脚本：

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_heads_only_gpu67.sh`
- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_heads_only_gpu67_fresh_500_val.sh`
- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_stage2_freeze_heads_gpu67_freshrun.sh`

这些脚本现在都已显式传：

- `--depth_target_source depth_anything_box`
- `--depth_anything_cache_root /data/gaoya/AAA_test_video/0623/train/train0624/depth_anything_cache`

当前风险和限制：

- `Depth Anything` 是相对深度，不是物理绝对深度
- 因此切换后的 `depth_aux` 数值尺度不应再直接按旧 `state GT` 的绝对大小理解
- 更适合关注：
  - 是否更稳定
  - 是否更符合视觉近远关系
  - 是否改善最终生成质量

如果要恢复旧逻辑，可直接显式传：

- `--depth_target_source state`
- `--depth_target_state_index 2`
### 2026-06-27：Depth Anything depth GT 默认切到 compact box cache，停止使用 full-frame cache

这次为了给新的 head-only fresh run 切换默认 depth supervision，先做了一个中间版本：

- 训练默认 `depth_target_source=depth_anything_box`
- 原本第一版 cache 思路是：
  - 先对视频跑 Depth Anything
  - 再把整帧 `depth_frames: [T,H,W]` 保存到 `*.depth_anything.pt`
  - 训练时临时按 `context_boxes` 做 median pooling

但实测后确认，这条链路不适合正式训练，原因有两个：

- 空间完全不可接受
  - 现网样本统计里，单个 full-frame `*.depth_anything.pt` 大约 `138MB`
  - 对应全量 `train 3600 + val 450 = 4050` 个窗口，粗估会到 `~704GB`
  - 这和当前 `/data` 空间约束不兼容
- 对齐方式也不够干净
  - raw `video.mp4` 往往是 `90` 帧
  - 训练实际使用的是 `episodes_v1/*.npz` 里的窗口序列：
    - `context_frames`: `8`
    - `future_frames`: `16`
  - 当前训练配置 `random_context_frames=False`
  - 所以训练真正消费的是窗口前 `8` 帧 `context_frames`，而不是 raw 全视频时间轴

因此已经把默认训练链路改成新的 compact cache 方案：

- 新 cache 脚本：
  - [cache_depth_anything_box_from_npz.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/cache_depth_anything_box_from_npz.py)
- 新 cache 读取模块：
  - [depth_anything_box_cache.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/utils/depth_anything_box_cache.py)
- 训练侧：
  - [train_v_newtrain.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_v_newtrain.py)
  - 现在会优先读取 `*.depth_anything_box.pt`
  - 只有在 compact cache 缺失时，才回退到旧的 `*.depth_anything.pt`

新的 compact cache 具体做法：

- 直接读取 `episodes_v1/*.npz`
- 用 `context_frames: [8,3,144,256]` 写一个临时 `8` 帧视频
- 对这个 `8` 帧 context 视频跑 Depth Anything
- 对 `context_boxes: [8,N_obj,4]` 逐帧做 box 内 median pooling
- 最终只保存训练真正需要的监督量：
  - `depth_boxes_framewise: [8, N_obj, 1]`
- 文件名：
  - `sample_xxxxxx_wyyy.depth_anything_box.pt`

实测结果：

- smoke case：
  - `sample_000001_w000.depth_anything_box.pt`
  - 文件大小约 `2.5KB`
  - shape 为 `(8, 6, 1)`
- 按当前平均文件大小估算：
  - 全量 `4050` 个窗口总大小约 `9.9MB`

所以当前明确结论是：

- 正式训练必须使用 compact `depth_anything_box` cache
- 旧的 full-frame `*.depth_anything.pt` 不能再作为全量训练 cache 方案继续扩展

### 2026-06-27：compact cache 当前运行状态

当前已经清理掉旧的 full-frame cache，并在 `tmux train` 里启动了新的 compact cache 并行任务，严格避开 `gpu4`：

- `train:da_box_train0`
  - `gpu0`
  - `train` split
  - `num_shards=3`
  - `shard_index=0`
- `train:da_box_train1`
  - `gpu2`
  - `train` split
  - `num_shards=3`
  - `shard_index=1`
- `train:da_box_train2`
  - `gpu6`
  - `train` split
  - `num_shards=3`
  - `shard_index=2`
- `train:da_box_val`
  - `gpu3`
  - `val` split
  - `num_shards=1`
  - `shard_index=0`

对应启动脚本：

- [run_cache_depth_anything_box_train_gpu0.sh](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_cache_depth_anything_box_train_gpu0.sh)
- [run_cache_depth_anything_box_train_gpu2.sh](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_cache_depth_anything_box_train_gpu2.sh)
- [run_cache_depth_anything_box_train_gpu6.sh](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_cache_depth_anything_box_train_gpu6.sh)
- [run_cache_depth_anything_box_val_gpu3.sh](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_cache_depth_anything_box_val_gpu3.sh)

当前观察：

- cache 正在持续增长
- 当前已落盘的 `*.depth_anything_box.pt` 数量已经开始增加
- `/data` 可用空间在清理旧 full-frame cache 后恢复到约 `55G`

当前 operational decision：

- 先继续补 compact cache
- cache 足够后，再从新的 fresh run 脚本启动 head-only 训练：
  - [run_train_v_newtrain_object_heads_only_depthanything_gpu67_fresh_500_val.sh](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_object_heads_only_depthanything_gpu67_fresh_500_val.sh)
- 新 run 仍然使用：
  - `gpu6,7`
  - `save_steps=500`
  - `headonly_val_loss_every_steps=500`
  - `W&B name = pybullet0627_diffsynth_object_heads_only_depthanything_gpu67_fresh500_val`

### 2026-06-29：teacher-student Stage 2 predictor 双卡启动排查与修复

新分支位置：

- [object_token_teacher_student](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/object_token_teacher_student)

这次先做了一个 `2 step` 的双卡 sanity run，目的是验证：

- `accelerate` 双卡拉起
- `forward / backward / optimizer.step`
- `wandb` 记录
- `step_*.pt` checkpoint 落盘

sanity 配置：

- 临时配置：
  - `/data/gaoya/agent-data/cache/teacher_student_stage2/config_stage2_predictor_sanity.yaml`
- 输出目录：
  - `/data/gaoya/agent-data/checkpoints/teacher_student_stage2_sanity`
- W&B run：
  - `pybullet0629_teacher_student_stage2_predictor_sanity`
  - run id: `79al3o1j`

第一轮双卡 sanity 暴露的真实代码问题：

- 报错：
  - `RecursionError: maximum recursion depth exceeded while calling a Python object`
- 位置：
  - `runner.py -> model.to(device=accelerator.device)`
- 根因：
  - [oracle_encoder.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/object_token_teacher_student/oracle_encoder.py)
    里 `OracleObjectTokenEncoder` 继承了 `nn.Module`
  - 同时又把整个 `trainer` 作为 `self.trainer` 挂回去
  - 这样形成了 `trainer -> oracle_encoder -> trainer` 的模块引用环
  - `model.to()` 在递归遍历子模块时会无限递归
- 修复：
  - 把 `OracleObjectTokenEncoder` 改成普通 Python helper，不再继承 `nn.Module`

第二轮 sanity 暴露的结构性问题：

- 报错：
  - `CUDA out of memory`
- 位置：
  - [wan_context_model.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/models/wan_context_model.py)
    的 `ensure_dit_loaded()`
- 表现：
  - 双卡 rank 在训练真正开始前，就把每张卡拉到约 `43GB`
- 根因：
  - [runtime.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/object_token_teacher_student/runtime.py)
    里的 `TeacherStudentPredictorTrainer` 虽然只训练 predictor + future heads
  - 但仍沿用了父类默认 `build_optimizer=True` 的初始化路径
  - 这会白白加载完整 Wan DiT 到每个 rank
  - Stage 2 predictor 实际并不跑 DiT forward，也不训练 DiT / LoRA
- 修复：
  - 在 `TeacherStudentPredictorTrainer.__init__` 里固定调用：
    - `super().__init__(..., build_optimizer=False, ...)`
  - 这样保留：
    - VAE
    - text encoder
    - JEPA
    - VGGT
    - CoTracker
    - object pooler / adapter
  - 但不再加载 Wan DiT

修复后的双卡 sanity 结果：

- 在空闲双卡 `gpu0,5` 上实跑通过
- 关键日志已经确认出现：
  - `accelerator.prepare done`
  - `first forward done`
  - `first backward done`
  - `first optimizer.step done`
- W&B 已正常创建并同步：
  - `https://wandb.ai/875222004-gy/vjepa_vggt_wan/runs/79al3o1j`
- checkpoint 已正常落盘：
  - `/data/gaoya/agent-data/checkpoints/teacher_student_stage2_sanity/step_0000001.pt`
  - `/data/gaoya/agent-data/checkpoints/teacher_student_stage2_sanity/step_0000002.pt`
- checkpoint 内容核对：
  - `step_0000001.pt -> step = 1, trainable_keys = 36`
  - `step_0000002.pt -> step = 2, trainable_keys = 36`
- 最近一次 sanity 的 W&B summary：
  - `train/loss_total = 0.7406`
  - `train/loss_future_token = 0.74013`
  - `train/loss_future_track = 0.00238`
  - `train/loss_future_box = 0.0023`
  - `train/pred_future_tokens_abs_max = 2.20312`
  - `train/object_latent_tokens_abs_max = 4.31653`

当前判断：

- `teacher-student Stage 2 predictor` 这条分支本身已经具备：
  - 双卡拉起
  - 正常反传
  - 参数更新
  - W&B 记录
  - checkpoint 落盘
- 当前还没直接在 `gpu6,7` 启动正式 run 的唯一原因不是代码，而是：
  - `gpu6,7` 此刻被另一条 `train_v_newtrain.py` 正式训练占满

已补充的正式启动脚本：

- [run_train_teacher_student_stage2_predictor_gpu67.sh](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_teacher_student_stage2_predictor_gpu67.sh)
- 现在脚本里已经显式包含：
  - `conda activate wan-cu128`
  - `PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt`
  - `CUDA_VISIBLE_DEVICES=6,7`

后续正式启动条件：

- 等 `gpu6,7` 空出来后，直接运行：
  - `bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_teacher_student_stage2_predictor_gpu67.sh`
- 预期输出目录：
  - `/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student_stage2_predictor`
