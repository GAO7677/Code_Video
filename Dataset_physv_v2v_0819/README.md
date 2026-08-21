# PhysV V2V 0819

面向视频物理动态理解和 V2V 短上下文续写的仿真数据集。共 70 个 case，14 组、每组 5 个控制变量取值。每组只改变表中变量，其余仿真参数、初始外观和相机设置保持一致；F11 不保留方向变体。

冰球挡板、木箱门框和小球门框组已完成 PyBullet 运动核查，并生成低分辨率 Cycles 预览；前两组为 `640×360`，小球门框组为 `896×512`。

- 冰球挡板组：冰球半径 `0.25 m`、厚度 `0.10 m`，挡板总高 `0.50 m`，各 case 相同；地面摩擦系数 `0.04`。
- 木箱门框组：木箱初速度 `1.80 m/s`、两侧墙总高 `1.80 m`，各 case 相同；地面摩擦系数 `0.18`，仅开口宽度变化。
- 小球门框组：蓝色橡胶球半径 `0.18 m`、初速度 `1.80 m/s`、初始横向偏移 `0.10 m`，沿用同一门框结构，仅开口宽度变化。
- SCENE 12 门框由两侧墙体、门洞上方连续墙体和齐平木质门套组成，避免独立几何块的拼接感。

## 关键路径

| 内容 | 路径 |
| --- | --- |
| 本项目脚本 | `/home/gaoya/Code_Video/Dataset_physv_v2v_0819` |
| 生成、导出、物理审计、Cycles | `/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts` |
| 轨迹/运动/VBench 指标 | `/home/gaoya/Code_Video/Dataset_physv_v2v_0819/eval` |
| viewer 服务与页面 | `/home/gaoya/Code_Video/Dataset_physv_v2v_0819/viewer` |
| 正式数据集 | `/data/gaoya/AAA_test_video/physv_v2v_0819` |
| case 视频 | `/data/gaoya/AAA_test_video/physv_v2v_0819/samples/<case_id>/videos` |
| 短上下文视频 | `/data/gaoya/AAA_test_video/physv_v2v_0819/samples/<case_id>/context` |
| 真值轨迹 | `/data/gaoya/AAA_test_video/physv_v2v_0819/samples/<case_id>/raw/trajectories.npz` |
| 物理监督 | `/data/gaoya/AAA_test_video/physv_v2v_0819/samples/<case_id>/physics_supervision.json` |
| 指标报告 | `/data/gaoya/AAA_test_video/physv_v2v_0819/reports` |
| Cycles 渲染缓存 | `/data/gaoya/agent-data/cache/physv_cycles_previews` |
| 纹理/HDRI 资源 | `/data/gaoya/dataset/blender_render_assets/polyhaven_v1`、`/data/gaoya/agent-data/assets/polyhaven_textures_20260820` |

## 数据内容

- `videos/rgb.mp4`: 完整视频；`videos/rgb_cycles.mp4`: Cycles 版本。
- `context/context8.mp4`、`context/context16.mp4`: 8 帧和 16 帧短上下文。
- `videos/depth.mp4`、`videos/masks.mp4`、`videos/trajectory.mp4`、`videos/contacts.mp4`: 深度、实例掩码、轨迹和接触可视化。
- `captions/caption_specific.txt`: 暴露变量值的 caption；`captions/caption_abstract.txt`: 隐藏具体变量值的 caption。
- `metadata.json`、`physics_supervision.json`、`raw/trajectories.npz`: 场景参数、物体物理真值和逐帧状态。

## 常用命令

从项目根目录运行：

```bash
cd /home/gaoya/Code_Video
export PYTHONPATH=/home/gaoya/Code_Video
PYTHON=/data/gaoya/miniconda3/envs/physxnet_mpm_env/bin/python

# 重新导出全部 65 个 case
$PYTHON -m Dataset_physv_v2v_0819.scripts.export_physv_v2v_0819_dataset \
  --output-root /data/gaoya/AAA_test_video/physv_v2v_0819

# 重新生成 caption
$PYTHON -m Dataset_physv_v2v_0819.scripts.refresh_physv_v2v_captions \
  --output-root /data/gaoya/AAA_test_video/physv_v2v_0819

# 补齐每个样本的 Scene/Object/Relation taxonomy 元数据
$PYTHON -m Dataset_physv_v2v_0819.scripts.refresh_taxonomy_0819 \
  --output-root /data/gaoya/AAA_test_video/physv_v2v_0819

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
```

viewer 前台启动命令：

```bash
/data/gaoya/miniconda3/envs/vjepa2/bin/python \
  /home/gaoya/Code_Video/Dataset_physv_v2v_0819/viewer/serve_physv_dataset_viewer.py \
  --host 0.0.0.0 --port 8765 \
  --dataset-root /data/gaoya/AAA_test_video/physv_v2v_0819 \
  --viewer-root /home/gaoya/Code_Video/Dataset_physv_v2v_0819/viewer
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
