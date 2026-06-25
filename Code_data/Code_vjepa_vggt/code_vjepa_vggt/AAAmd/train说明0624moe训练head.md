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
