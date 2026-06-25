# 0624 object branch 精简版 MoE 训练说明

这份说明对应当前 `train_v_newtrain.py` 的 object-heads-only strict 版本。它不是旧的 4 路融合方案，而是已经收敛成“`track+geometry` / `appearance` 两路融合”的精简版 object branch。

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
