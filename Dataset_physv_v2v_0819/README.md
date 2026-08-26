# PhysV V2V 0819

面向视频物理动态理解和 V2V 短上下文续写的仿真数据集。共 70 个 case，14 组、每组 5 个控制变量取值。每组只改变表中变量，其余仿真参数、初始外观和相机设置保持一致；F11 不保留方向变体。

> 数据与派生训练资产最后核查：2026-08-26 UTC。本文档以 `/data/gaoya/AAA_test_video/physv_v2v_0819` 中实际存在的 70 个 case、CYCLES 对齐真值和 2026-08-25 生成的训练 cache 为准。

冰球挡板、木箱门框和小球门框组已完成 PyBullet 运动核查，并生成低分辨率 Cycles 预览；前两组为 `640×360`，小球门框组为 `896×512`。

- 冰球挡板组：冰球半径 `0.25 m`、厚度 `0.10 m`，挡板总高 `0.50 m`，各 case 相同；地面摩擦系数 `0.04`。
- 木箱门框组：木箱初速度 `1.80 m/s`、两侧墙总高 `1.80 m`，各 case 相同；地面摩擦系数 `0.18`，仅开口宽度变化。
- 小球门框组：蓝色橡胶球半径 `0.18 m`、初速度 `1.80 m/s`、初始横向偏移 `0.10 m`，沿用同一门框结构，仅开口宽度变化。
- 摆锤撞立柜组：摆长 `1.10 m`、释放角 `18°`、摆球质量 `1.20 kg` 和立柜均固定，仅悬点高度取 `2.00/2.25/2.50/2.75/3.00 m`；5 个 case 的撞击前速度均为 `0.44206 m/s`，并已生成 `896×512` 低分辨率 Cycles 预览。
- SCENE 12 门框由两侧墙体、门洞上方连续墙体和齐平木质门套组成，避免独立几何块的拼接感。

## 关键路径

| 内容 | 路径 |
| --- | --- |
| 本项目脚本 | `/home/gaoya/Code_Video/Dataset_physv_v2v_0819` |
| 生成、导出、物理审计、Cycles | `/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts` |
| 轨迹/运动/VBench 指标 | `/home/gaoya/Code_Video/Dataset_physv_v2v_0819/eval` |
| viewer 服务与页面 | `/home/gaoya/Code_Video/Dataset_physv_v2v_0819/viewer` |
| 正式数据集 | `/data/gaoya/AAA_test_video/physv_v2v_0819` |
| 数据集总 manifest | `/data/gaoya/AAA_test_video/physv_v2v_0819/manifest.json` |
| case 视频 | `/data/gaoya/AAA_test_video/physv_v2v_0819/samples/<case_id>/videos` |
| 短上下文视频 | `/data/gaoya/AAA_test_video/physv_v2v_0819/samples/<case_id>/context` |
| 真值轨迹 | `/data/gaoya/AAA_test_video/physv_v2v_0819/samples/<case_id>/raw/trajectories.npz` |
| 物理监督 | `/data/gaoya/AAA_test_video/physv_v2v_0819/samples/<case_id>/physics_supervision.json` |
| 指标报告 | `/data/gaoya/AAA_test_video/physv_v2v_0819/reports` |
| Cycles 渲染缓存 | `/data/gaoya/agent-data/cache/physv_cycles_previews` |
| CYCLES 对齐真值（独立生成、不覆盖原始真值） | `/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_aligned_truth_v1` |
| CYCLES 训练 overlay（70 条 list-manifest） | `/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_train_v1` |
| CYCLES 动态 mask 源缓存 | `/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_cycles_mask_source_v1` |
| VAE latent cache | `/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_vae_latents` |
| prompt embedding cache | `/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_prompt_cache` |
| latent-mask cache（train split） | `/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_latent_mask_cache` |
| collision supervision cache | `/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_collision_supervision` |
| Utonia scene cache | `/data/gaoya/AAA_test_video/physv_v2v_0819/physv_v2v_0819_utonia_scene_cache` |
| Cycles 轨迹 overlay 与页面 | `/data/gaoya/agent-data/outputs/physv_v2v_0819_trajectory_overlay` |
| 深度/实例 ID 页面视频 | `/data/gaoya/agent-data/outputs/physv_v2v_0819_trajectory_overlay/truth_videos` |
| 纹理/HDRI 资源 | `/data/gaoya/dataset/blender_render_assets/polyhaven_v1`、`/data/gaoya/agent-data/assets/polyhaven_textures_20260820` |

## 数据内容

- `videos/rgb.mp4`: 原始仿真完整视频；`videos/rgb_cycles.mp4`: 与同一套场景、相机和轨迹对齐的 CYCLES 版本；`videos/rgb_cycles.json`: CYCLES 渲染配置。
- `context/context8.mp4`、`context/context16.mp4`: 原始视频的 8 帧和 16 帧短上下文；`context/context8_cycles.mp4`、`context/context16_cycles.mp4`: CYCLES 视频对应的短上下文。
- `videos/depth.mp4`、`videos/masks.mp4`、`videos/trajectory.mp4`、`videos/contacts.mp4`: 深度、实例掩码、轨迹和接触可视化。
- `samples/<case_id>/raw/masks.npz`: 原始仿真相机坐标系的动态物体 GT mask；数据集未保存 SAM/SAM2 预测结果。
- `physv_v2v_0819_cycles_aligned_truth_v1/cases/<case_id>/dynamic_masks.npz`: 使用与 `videos/rgb_cycles.mp4` 相同的 CYCLES 场景构建、相机、轨迹、分辨率和帧序重新渲染 Object Index pass 得到的动态物体像素真值；`masks_thw` 为 `[动态物体数, 帧数, 高, 宽]`，`union_thw` 为所有动态物体并集，背景/静态物体为 0，动态物体从 1 开始编号。
- `physv_v2v_0819_cycles_aligned_truth_v1/cases/<case_id>/trajectory_pixels.npz`: 同一 CYCLES 相机坐标系下的动态物体中心投影，`centers_tnc` 为 `[帧数, 动态物体数, (x_pixel,y_pixel,depth)]`；像素原点在左上角，frame 0 对应 `rgb_cycles.mp4` frame 0。
- `physv_v2v_0819_cycles_aligned_truth_v1/cases/<case_id>/truth_metadata.json`: 记录 CYCLES 配置、分辨率、帧数、动态物体名称与 Object Index 映射，并记录源 `rgb_cycles.mp4`、轨迹真值和渲染脚本。
- 这批 CYCLES 对齐真值不替换 `raw/masks.npz`：后者仍是原始 PyBullet/仿真相机坐标系的 mask。`samples/<case_id>/contacts.json`、`physics_supervision.npz` 和 `raw/trajectories.npz` 仍是同一 90 帧仿真时间轴上的接触、状态和位姿真值；它们不需要因为 CYCLES 像素坐标变化而重算。
- `captions/caption_specific.txt`: 暴露变量值和实际末态的 caption；`captions/caption_abstract.txt`: 隐藏具体变量值但保留实际运动结果的 caption。
- `metadata.json` / `manifest.json` 中的 `caption_observations`: 从 `physics_supervision.npz` 和 `contacts.json` 提取的观测结果，用于区分通过、碰撞、掉落、停止等 case 末态。
- `metadata.json`、`physics_supervision.json`、`raw/trajectories.npz`: 场景参数、物体物理真值和逐帧状态。

## 已生成训练资产

所有派生资产均放在数据集根目录下的独立子目录中，不覆盖 `samples/` 中的原始视频和真值。除特别注明外，cache 的数据集入口都是 `physv_v2v_0819_cycles_train_v1`，训练/推理视频使用 `videos/rgb_cycles.mp4`。

| 资产 | 当前状态与内容 |
| --- | --- |
| `physv_v2v_0819_cycles_train_v1` | 70 条 list-manifest；caption 使用 `caption_abstract.txt`；通过 `SHA1(family_key/case_id)` 规则划分 train/val/test。 |
| `physv_v2v_0819_vae_latents` | complete，70/70；49 帧输入，latent 形状 `[48, 13, 32, 56]`。 |
| `physv_v2v_0819_prompt_cache` | complete，70/70；最大长度 512，embedding 形状 `[512, 4096]`。 |
| `physv_v2v_0819_latent_mask_cache` | complete，当前为 `split=train` 的 60 条；不是 70 条数据缺失，若要对 val/test 或全量训练使用，需另生成对应 split。 |
| `physv_v2v_0819_collision_supervision` | complete，70/70；49 个视频帧映射到 13 个 latent frame，碰撞加权为 `clip(1 + 2 * latent_score, 1, 3)`。 |
| `physv_v2v_0819_utonia_scene_cache` | complete，70/70；读取 context 前 8 帧的第 7 帧，特征形状 `[448, 1386]`，使用官方 VGGT crop、Utonia full-upcast 和 3D-safe 动态过滤。 |

其中 `physv_v2v_0819_cycles_aligned_truth_v1` 提供 CYCLES 像素坐标系的动态 mask/轨迹投影；`raw/masks.npz`、`contacts.json`、`physics_supervision.npz` 和 `raw/trajectories.npz` 仍是原始仿真时间轴真值，两者不可直接混用。

## 常用命令

从项目根目录运行：

```bash
cd /home/gaoya/Code_Video
export PYTHONPATH=/home/gaoya/Code_Video
PYTHON=/data/gaoya/miniconda3/envs/physxnet_mpm_env/bin/python

# 重新导出全部 70 个 case
$PYTHON -m Dataset_physv_v2v_0819.scripts.export_physv_v2v_0819_dataset \
  --output-root /data/gaoya/AAA_test_video/physv_v2v_0819

# 重新生成 caption
$PYTHON -m Dataset_physv_v2v_0819.scripts.refresh_physv_v2v_captions \
  --output-root /data/gaoya/AAA_test_video/physv_v2v_0819

# 补齐每个样本的 Scene/Object/Relation taxonomy 元数据
$PYTHON -m Dataset_physv_v2v_0819.scripts.refresh_taxonomy_0819 \
  --output-root /data/gaoya/AAA_test_video/physv_v2v_0819

# refresh_physv_v2v_captions 同时会同步 testjsons/**/*.json 的三个 caption 字段

# 轨迹指标
$PYTHON Dataset_physv_v2v_0819/eval/compute_motion_amplitude_0819.py \
  --dataset-root /data/gaoya/AAA_test_video/physv_v2v_0819
$PYTHON Dataset_physv_v2v_0819/eval/compute_trajectory_similarity_0819.py \
  --dataset-root /data/gaoya/AAA_test_video/physv_v2v_0819
$PYTHON Dataset_physv_v2v_0819/eval/filter_motion_amplitude_pairs_0819.py \
  --dataset-root /data/gaoya/AAA_test_video/physv_v2v_0819 \
  --threshold 0.30 --min-selected-pairs 4

# Cycles 预览，默认使用 GPU 2
$PYTHON Dataset_physv_v2v_0819/scripts/run_physv_cycles_previews.py \
  v2v_bowl_r080 --dataset-root /data/gaoya/AAA_test_video/physv_v2v_0819 \
  --gpu 2 --width 896 --height 512 --samples 32 --engine CYCLES

# 在 Cycles 原视频上叠加红色 physics_supervision.npz 动态物体轨迹和 GT mask
$PYTHON Dataset_physv_v2v_0819/tools/render_cycles_trajectory_overlay.py \
  --input-list /data/gaoya/AAA_test_video/physv_v2v_0819/testjsons/physv_v2v_0819_all_cycles_test70_ctx8.txt \
  --output-root /data/gaoya/agent-data/outputs/physv_v2v_0819_trajectory_overlay

# 准备深度和全物体实例 ID 的完整/前 8 帧页面视频
$PYTHON Dataset_physv_v2v_0819/tools/prepare_truth_visualizations.py \
  --dataset-root /data/gaoya/AAA_test_video/physv_v2v_0819 \
  --output-root /data/gaoya/agent-data/outputs/physv_v2v_0819_trajectory_overlay
```

overlay 页面中的 `CTX 8` 只包含完整视频的 frame 0–7。页面可切换三种视图：Cycles 红色轨迹 + 动态 GT mask、原始仿真相机的全物体实例 ID、原始仿真相机的深度伪彩；后两者分别来自 `raw/instance_ids.npz` 和 `raw/depth.npz`，不能视为 SAM/SAM2 结果。视频画面不写入文字图例。

viewer 前台启动命令：

```bash
/data/gaoya/miniconda3/envs/vjepa2/bin/python \
  /home/gaoya/Code_Video/Dataset_physv_v2v_0819/viewer/serve_physv_dataset_viewer.py \
  --host 0.0.0.0 --port 8765 \
  --dataset-root /data/gaoya/AAA_test_video/physv_v2v_0819 \
  --viewer-root /home/gaoya/Code_Video/Dataset_physv_v2v_0819/viewer
```

轨迹 overlay 页面前台启动命令：

```bash
/data/gaoya/miniconda3/envs/wan-cu128/bin/python -m http.server 8919 \
  --bind 0.0.0.0 \
  --directory /data/gaoya/agent-data/outputs/physv_v2v_0819_trajectory_overlay
```

## Case 清单与控制变量

## Taxonomy

| Taxonomy | 定义 | 对应组 |
| --- | --- | --- |
| `Scene` | 静态环境几何变化；运动物体几何和物理参数保持一致 | F11 桌高、F12 斜面角度、F12 斜面长度、V2V 碗、V2V 缺口、冰球挡板、木箱门框、小球门框 |
| `Object` | 环境保持一致；只改变运动物体几何或初始状态 | V2V 障碍速度、V2V 障碍尺寸、V2V 摆锤 |
| `Relation` | 物体和环境保持一致；只改变相对位置、方向或支撑关系 | V2V 多米诺、V2V 跷跷板、V2V 摆锤撞立柜 |

每组 5 个 case，控制变量见下表。

| 组 | Case | 场景 | 控制变量 | 取值 |
| --- | --- | --- | --- | --- |
| F11 / 桌高 | `difficulty_l2_f11_h030_sr048` | 小球从桌面滚落 | `table_height_m` 桌面高度 | 0.30 m |
| F11 / 桌高 | `difficulty_l2_f11_h058_sr048` | 小球从桌面滚落 | `table_height_m` 桌面高度 | 0.58 m |
| F11 / 桌高 | `difficulty_l2_f11_h085_sr048` | 小球从桌面滚落 | `table_height_m` 桌面高度 | 0.85 m |
| F11 / 桌高 | `difficulty_l2_f11_h112_sr048` | 小球从桌面滚落 | `table_height_m` 桌面高度 | 1.12 m |
| F11 / 桌高 | `difficulty_l2_f11_h140_sr048` | 小球从桌面滚落 | `table_height_m` 桌面高度 | 1.40 m |
| F12 / 斜面角度 | `difficulty_l2_f12_a008` | 木块从斜面释放 | `ramp_angle_deg` 斜面倾角 | 8° |
| F12 / 斜面角度 | `difficulty_l2_f12_a016` | 木块从斜面释放 | `ramp_angle_deg` 斜面倾角 | 16° |
| F12 / 斜面角度 | `difficulty_l2_f12_a024` | 木块从斜面释放 | `ramp_angle_deg` 斜面倾角 | 24° |
| F12 / 斜面角度 | `difficulty_l2_f12_a033` | 木块从斜面释放 | `ramp_angle_deg` 斜面倾角 | 33° |
| F12 / 斜面角度 | `difficulty_l2_f12_a042` | 木块从斜面释放 | `ramp_angle_deg` 斜面倾角 | 42° |
| F12 / 斜面长度 | `difficulty_l2_f12_length_l080` | 木块从斜面释放 | `ramp_length_m` 斜面长度 | 0.80 m |
| F12 / 斜面长度 | `difficulty_l2_f12_length_l110` | 木块从斜面释放 | `ramp_length_m` 斜面长度 | 1.10 m |
| F12 / 斜面长度 | `difficulty_l2_f12_length_l140` | 木块从斜面释放 | `ramp_length_m` 斜面长度 | 1.40 m |
| F12 / 斜面长度 | `difficulty_l2_f12_length_l170` | 木块从斜面释放 | `ramp_length_m` 斜面长度 | 1.70 m |
| F12 / 斜面长度 | `difficulty_l2_f12_length_l200` | 木块从斜面释放 | `ramp_length_m` 斜面长度 | 2.00 m |
| V2V / 碗 | `v2v_bowl_r080` | 小球沿碗面运动 | `bowl_radius_m` 碗半径 | 0.80 m |
| V2V / 碗 | `v2v_bowl_r130` | 小球沿碗面运动 | `bowl_radius_m` 碗半径 | 1.30 m |
| V2V / 碗 | `v2v_bowl_r180` | 小球沿碗面运动 | `bowl_radius_m` 碗半径 | 1.80 m |
| V2V / 碗 | `v2v_bowl_r229` | 小球沿碗面运动 | `bowl_radius_m` 碗半径 | 2.30 m |
| V2V / 碗 | `v2v_bowl_r280` | 小球沿碗面运动 | `bowl_radius_m` 碗半径 | 2.80 m |
| V2V / 多米诺 | `v2v_domino_g000` | 多米诺骨牌链式碰撞 | `domino_gap_m` 骨牌间距 | 0.00 m |
| V2V / 多米诺 | `v2v_domino_g045` | 多米诺骨牌链式碰撞 | `domino_gap_m` 骨牌间距 | 0.045 m |
| V2V / 多米诺 | `v2v_domino_g090` | 多米诺骨牌链式碰撞 | `domino_gap_m` 骨牌间距 | 0.09 m |
| V2V / 多米诺 | `v2v_domino_g135` | 多米诺骨牌链式碰撞 | `domino_gap_m` 骨牌间距 | 0.135 m |
| V2V / 多米诺 | `v2v_domino_g180` | 多米诺骨牌链式碰撞 | `domino_gap_m` 骨牌间距 | 0.18 m |
| V2V / 缺口 | `v2v_gap_006` | 小球跨越平台缺口 | `gap_width_m` 缺口宽度 | 0.06 m |
| V2V / 缺口 | `v2v_gap_022` | 小球跨越平台缺口 | `gap_width_m` 缺口宽度 | 0.22 m |
| V2V / 缺口 | `v2v_gap_038` | 小球跨越平台缺口 | `gap_width_m` 缺口宽度 | 0.38 m |
| V2V / 缺口 | `v2v_gap_054` | 小球跨越平台缺口 | `gap_width_m` 缺口宽度 | 0.54 m |
| V2V / 缺口 | `v2v_gap_070` | 小球跨越平台缺口 | `gap_width_m` 缺口宽度 | 0.70 m |
| V2V / 障碍尺寸 | `v2v_obstacle_size_r080` | 小球撞击固定障碍 | `ball_radius_m` 小球半径 | 0.080 m |
| V2V / 障碍尺寸 | `v2v_obstacle_size_r110` | 小球撞击固定障碍 | `ball_radius_m` 小球半径 | 0.110 m |
| V2V / 障碍尺寸 | `v2v_obstacle_size_r140` | 小球撞击固定障碍 | `ball_radius_m` 小球半径 | 0.140 m |
| V2V / 障碍尺寸 | `v2v_obstacle_size_r170` | 小球撞击固定障碍 | `ball_radius_m` 小球半径 | 0.170 m |
| V2V / 障碍尺寸 | `v2v_obstacle_size_r200` | 小球撞击固定障碍 | `ball_radius_m` 小球半径 | 0.200 m |
| V2V / 障碍速度 | `v2v_obstacle_v120` | 小球撞击固定障碍 | `initial_speed_mps` 初速度 | 1.2 m/s |
| V2V / 障碍速度 | `v2v_obstacle_v140` | 小球撞击固定障碍 | `initial_speed_mps` 初速度 | 1.4 m/s |
| V2V / 障碍速度 | `v2v_obstacle_v160` | 小球撞击固定障碍 | `initial_speed_mps` 初速度 | 1.6 m/s |
| V2V / 障碍速度 | `v2v_obstacle_v180` | 小球撞击固定障碍 | `initial_speed_mps` 初速度 | 1.8 m/s |
| V2V / 障碍速度 | `v2v_obstacle_v520` | 小球撞击固定障碍 | `initial_speed_mps` 初速度 | 5.2 m/s |
| V2V / 摆锤 | `v2v_pendulum_l055` | 摆锤摆动 | `pendulum_length_m` 绳长 | 0.55 m |
| V2V / 摆锤 | `v2v_pendulum_l083` | 摆锤摆动 | `pendulum_length_m` 绳长 | 0.83 m |
| V2V / 摆锤 | `v2v_pendulum_l110` | 摆锤摆动 | `pendulum_length_m` 绳长 | 1.10 m |
| V2V / 摆锤 | `v2v_pendulum_l138` | 摆锤摆动 | `pendulum_length_m` 绳长 | 1.38 m |
| V2V / 摆锤 | `v2v_pendulum_l165` | 摆锤摆动 | `pendulum_length_m` 绳长 | 1.65 m |
| V2V / 摆锤撞立柜 | `v2v_pendulum_cabinet_h200` | 摆锤撞固定立柜 | `pendulum_anchor_height_m` 悬点高度 | 2.00 m |
| V2V / 摆锤撞立柜 | `v2v_pendulum_cabinet_h225` | 摆锤撞固定立柜 | `pendulum_anchor_height_m` 悬点高度 | 2.25 m |
| V2V / 摆锤撞立柜 | `v2v_pendulum_cabinet_h250` | 摆锤撞固定立柜 | `pendulum_anchor_height_m` 悬点高度 | 2.50 m |
| V2V / 摆锤撞立柜 | `v2v_pendulum_cabinet_h275` | 摆锤撞固定立柜 | `pendulum_anchor_height_m` 悬点高度 | 2.75 m |
| V2V / 摆锤撞立柜 | `v2v_pendulum_cabinet_h300` | 摆锤撞固定立柜 | `pendulum_anchor_height_m` 悬点高度 | 3.00 m |
| V2V / 跷跷板 | `v2v_seesaw_x000` | 2.70 m 跷跷板载荷 | `load_position_x_m` 载荷位置 | 0.00 m |
| V2V / 跷跷板 | `v2v_seesaw_x029` | 2.70 m 跷跷板载荷 | `load_position_x_m` 载荷位置 | 0.2925 m |
| V2V / 跷跷板 | `v2v_seesaw_x058` | 2.70 m 跷跷板载荷 | `load_position_x_m` 载荷位置 | 0.5850 m |
| V2V / 跷跷板 | `v2v_seesaw_x088` | 2.70 m 跷跷板载荷 | `load_position_x_m` 载荷位置 | 0.8775 m |
| V2V / 跷跷板 | `v2v_seesaw_x117` | 2.70 m 跷跷板载荷 | `load_position_x_m` 载荷位置 | 1.17 m |
| Scene / 冰球挡板 | `scene_puck_barrier_n030` | 低摩擦地面冰球撞固定挡板 | `barrier_normal_angle_deg` 挡板平面法线方向 | 30° |
| Scene / 冰球挡板 | `scene_puck_barrier_n045` | 低摩擦地面冰球撞固定挡板 | `barrier_normal_angle_deg` 挡板平面法线方向 | 45° |
| Scene / 冰球挡板 | `scene_puck_barrier_n060` | 低摩擦地面冰球撞固定挡板 | `barrier_normal_angle_deg` 挡板平面法线方向 | 60° |
| Scene / 冰球挡板 | `scene_puck_barrier_n075` | 低摩擦地面冰球撞固定挡板 | `barrier_normal_angle_deg` 挡板平面法线方向 | 75° |
| Scene / 冰球挡板 | `scene_puck_barrier_n090` | 低摩擦地面冰球撞固定挡板 | `barrier_normal_angle_deg` 挡板平面法线方向 | 90° |
| Scene / 木箱门框 | `scene_door_frame_w038` | 木箱穿过固定厚度门框 | `door_opening_width_m` 门框开口宽度 | 0.38 m |
| Scene / 木箱门框 | `scene_door_frame_w046` | 木箱穿过固定厚度门框 | `door_opening_width_m` 门框开口宽度 | 0.46 m |
| Scene / 木箱门框 | `scene_door_frame_w054` | 木箱穿过固定厚度门框 | `door_opening_width_m` 门框开口宽度 | 0.54 m |
| Scene / 木箱门框 | `scene_door_frame_w062` | 木箱穿过固定厚度门框 | `door_opening_width_m` 门框开口宽度 | 0.62 m |
| Scene / 木箱门框 | `scene_door_frame_w074` | 木箱穿过固定厚度门框 | `door_opening_width_m` 门框开口宽度 | 0.74 m |
| Scene / 小球门框 | `scene_door_frame_ball_w038` | 小球穿过固定厚度门框 | `door_opening_width_m` 门框开口宽度 | 0.38 m |
| Scene / 小球门框 | `scene_door_frame_ball_w046` | 小球穿过固定厚度门框 | `door_opening_width_m` 门框开口宽度 | 0.46 m |
| Scene / 小球门框 | `scene_door_frame_ball_w054` | 小球穿过固定厚度门框 | `door_opening_width_m` 门框开口宽度 | 0.54 m |
| Scene / 小球门框 | `scene_door_frame_ball_w062` | 小球穿过固定厚度门框 | `door_opening_width_m` 门框开口宽度 | 0.62 m |
| Scene / 小球门框 | `scene_door_frame_ball_w074` | 小球穿过固定厚度门框 | `door_opening_width_m` 门框开口宽度 | 0.74 m |

F12 斜面长度组保持斜面最高点/支撑高度不变，长度变化会使倾角随之变化；V2V 06 跷跷板保持 2.70 m 板长，载荷从中心向一侧均匀外移。摆锤撞立柜组固定摆长 `1.10 m` 和释放角 `18°`，只改变悬点高度，因此相对摆动能量和撞击速度保持一致，撞击位置随高度变化。

## Cycles 输入列表

全部 70 个 Cycles 视频已整理为：

`/data/gaoya/AAA_test_video/physv_v2v_0819/testjsons/physv_v2v_0819_all_cycles_test70_ctx8.txt`

不带事件时间描述的同规模列表为：

`/data/gaoya/AAA_test_video/physv_v2v_0819/testjsons/physv_v2v_0819_all_cycles_test70_ctx8_description_no_event_timing.txt`

该 txt 每行是一个 JSON 的绝对路径，JSON 位于：

`/data/gaoya/AAA_test_video/physv_v2v_0819/testjsons/v2v_jsons/physv_v2v_0819_all_cycles/`

每个 JSON 主要包含：

- 视频：`input_video` / `input_video_8f` 为 Cycles 前 8 帧，`input_video_16f` 为前 16 帧，`source_video` 为完整 `rgb_cycles.mp4`。
- 文本：`input_caption`、`input_caption_specific` 为具体变量描述，`input_caption_abstract` 为模糊变量描述。
- 控制信息：`taxonomy`、`source_group`、`family_key`、`control`、`title`，其中 `control` 记录控制变量、数值、标签和单位。
- 时序信息：`conditioning` 记录 context 帧选项、首个事件规则、事件帧和事件时间；`frame_counts` 与 `video_spec` 记录帧数、分辨率和帧率。
- 物理监督引用：`metadata_json`、`manifest_json`、`captions_json`、`contacts_json`，以及轨迹、mask、深度和 `physics_supervision` 文件路径。

列表中的 JSON 可直接作为 `batch_infer_from_input_json_lists.py` 的输入；默认使用 8 帧 Cycles context，完整 Cycles 视频作为后续运动参考。
