• 当前训练数据集根目录是：

  /data/gaoya/AAA_test_video/Dataset_physV/0417data/physxnet_try1_rigid_random/train/

  整体结构现在按：

  train/
    rigid/
      <scene_composition>/
        <object_count_bucket>/
          <sample_name>/

  组织。

  当前主要有两类 scene_composition：

  train/
    rigid/
      interaction_pair_plus_dynamic/
        count_02/
          4778__case001_static_left/
          4778__case003_static_highdrop/
          4778__case007_entry_fast_center/

      single_object_preview/
        count_01/
          4778__case001_static_left/
          4778__case003_static_highdrop/
          4778__case007_entry_fast_center/

  其中：

  - interaction_pair_plus_dynamic/count_02：PhysXNet 主物体 + 黄色 striker 小球，共 2 个对象
  - single_object_preview/count_01：只有 PhysXNet 主物体，共 1 个对象

  单个样本目录格式如下：

  sample_name/
    metadata.json
    scene_input.json

    rgb/
      frame_000.png
      frame_001.png
      ...

    depth/
      frame_000.png
      frame_001.png
      ...

    videos/
      rgb.mp4
      depth.mp4

    physics/
      depth_metric.npy
      depth_normalized.npy
      seg.npy
      flow.npy
      contact_graph.npy
      contact_impulse.npy
      frame_phase.npy
      rigid_kinematics.npz
      anchor_targets.npz
      energy.npz
      properties.json
      collision_events.json
      event_windows.json

    visualizations/
      depth_vis.mp4

  核心文件含义

  | 文件 | 含义 |
  |---|---|
  | metadata.json | 样本主元数据，训练时优先读这个 |
  | scene_input.json | 生成该样本的场景输入配置 |
  | rgb/frame_*.png | 每帧 RGB 图像 |
  | depth/frame_*.png | 每帧 8-bit 可视化深度，不建议当真实深度训练 |
  | videos/rgb.mp4 | RGB 视频 |
  | videos/depth.mp4 | 深度可视化视频 |
  | physics/depth_metric.npy | 真实 metric depth，单位米 |
  | physics/depth_normalized.npy | normalized depth，主要用于可视化或调试 |
  | physics/seg.npy | 实例分割 |
  | physics/flow.npy | fallback optical flow |
  | physics/contact_graph.npy | 逐帧物体接触图 |
  | physics/contact_impulse.npy | 逐帧接触冲量，目前多为 fallback/近似 |
  | physics/frame_phase.npy | 每帧运动阶段标签 |
  | physics/rigid_kinematics.npz | 刚体逐帧状态 |
  | physics/anchor_targets.npz | 2D anchor / bbox / 可见性训练目标 |
  | physics/energy.npz | 能量标签 |
  | physics/properties.json | 物理属性，如 restitution |
  | physics/collision_events.json | 碰撞事件 |
  | physics/event_windows.json | 事件窗口 |
  | visualizations/depth_vis.mp4 | 深度可视化视频 |

  metadata.json 主要字段

  {
    "scene_id": "4778__case001_static_left",
    "object_id": "4778",
    "seed": 20270410,
    "split": "train",
    "family": "physxnet_single_object",
    "simulator_type": "rigid",
    "scene_composition": "interaction_pair_plus_dynamic",
    "interaction_pattern": "striker_hits_static_target",
    "object_count_bucket": "count_02",
    "num_objects": 2,
    "frames": 12,
    "resolution": [960, 720],
    "motion_category": "static_left",
    "convention": {...},
    "simulation": {...},
    "camera": {...},
    "camera_intrinsics": {...},
    "objects": [...],
    "environment_entities": [...],
    "outputs": {...}
  }

  关键元数据字段说明

  | 字段 | 含义 |
  |---|---|
  | simulator_type | 当前为 rigid |
  | scene_composition | 场景组成方式，例如 interaction_pair_plus_dynamic / single_object_preview |
  | interaction_pattern | 场景级动力学交互类型 |
  | object_count_bucket | 物体数量 bucket，例如 count_01 / count_02 |
  | motion_category | 当前 case 模板名 |
  | objects | 每个物体的 id、seg id、来源、运动类型、角色 |
  | environment_entities | 地面等特殊环境对象 |
  | outputs | 所有导出文件路径 |

  objects 字段格式示例

  有 striker 的样本：

  "objects": [
    {
      "object_id": 0,
      "seg_id": 1,
      "entity_type": "rigid_assembly",
      "role": "target",
      "object_motion_type": "static_left",
      "object_motion_group": "static_placement",
      "motion_type": "static_left",
      "motion_group": "static_placement",
      "source_tag": "physxnet_main"
    },
    {
      "object_id": 1,
      "seg_id": 2,
      "entity_type": "custom_rigid",
      "role": "initiator",
      "object_motion_type": "striker_hit",
      "object_motion_group": "striker",
      "motion_type": "striker_hit",
      "motion_group": "striker",
      "source_tag": "custom_object"
    }
  ]

  单物体样本：

  "objects": [
    {
      "object_id": 0,
      "seg_id": 1,
      "entity_type": "rigid_assembly",
      "role": "target",
      "object_motion_type": "static_highdrop",
      "object_motion_group": "gravity_drop",
      "motion_type": "static_highdrop",
      "motion_group": "gravity_drop",
      "source_tag": "physxnet_main"
    }
  ]

  物体来源 source_tag

  | source_tag | 来源 |
  |---|---|
  | physxnet_main | PhysXNet 数据集主物体 |
  | custom_object | 脚本内置黄色 striker 小球 |
  | physxnet_soft | PhysXNet 软体部分，当前 rigid-only 通常不会出现 |

  interaction_pattern 类型

  | interaction_pattern | 含义 |
  |---|---|
  | striker_hits_static_target | 黄色球撞击静止 PhysXNet 目标 |
  | striker_hits_falling_target | 黄色球与高处下落 PhysXNet 目标同场 |
  | co_moving_collision | PhysXNet 目标自身运动，同时有 striker |
  | single_object_static_preview | 单 PhysXNet 物体静止预览 |
  | gravity_drop | 单 PhysXNet 物体高处下落 |
  | single_object_entry_motion | 单 PhysXNet 物体带初速度入场 |

  主要 .npy/.npz shape

  假设：

  - T = frames
  - N = num_objects
  - H, W = height, width

  | 文件 | 字段 / shape |
  |---|---|
  | physics/depth_metric.npy | [T, H, W] |
  | physics/depth_normalized.npy | [T, H, W] |
  | physics/seg.npy | [T, H, W] |
  | physics/flow.npy | [T-1, H, W, 2] |
  | physics/contact_graph.npy | [T, N, N] |
  | physics/contact_impulse.npy | [T, N, N] |
  | physics/frame_phase.npy | [T] |
  | physics/rigid_kinematics.npz/object_ids | [N] |
  | physics/rigid_kinematics.npz/seg_ids | [N] |
  | physics/rigid_kinematics.npz/com_pos | [T, N, 3] |
  | physics/rigid_kinematics.npz/orientation_quat | [T, N, 4] |
  | physics/rigid_kinematics.npz/linear_vel | [T, N, 3] |
  | physics/rigid_kinematics.npz/angular_vel | [T, N, 3] |
  | physics/rigid_kinematics.npz/com_uv | [T, N, 2] |
  | physics/rigid_kinematics.npz/bbox_xyxy | [T, N, 4] |
  | physics/rigid_kinematics.npz/visibility_mask | [T, N] |
  | physics/anchor_targets.npz/com_uv | [T, N, 2] |
  | physics/anchor_targets.npz/bbox_xyxy | [T, N, 4] |
  | physics/anchor_targets.npz/visibility_mask | [T, N] |
  | physics/anchor_targets.npz/center_depth | [T, N] |
  | physics/energy.npz/kinetic_trans | [T] |
  | physics/energy.npz/kinetic_rot | [T] |
  | physics/energy.npz/potential_gravity | [T] |
  | physics/energy.npz/mechanical_total | [T] |

  seg id 规则

  - 背景：0
  - 物体 object_id=k：默认 seg_id=k+1
  - 映射写在 metadata.json -> objects[*].seg_id

  depth 规则

  - depth_metric.npy：真实深度，单位米
  - depth_normalized.npy：归一化深度，用于可视化，不建议作为真实物理深度训练

  训练时建议优先读取

  - metadata.json
  - videos/rgb.mp4 或 rgb/frame_*.png
  - physics/depth_metric.npy
  - physics/seg.npy
  - physics/rigid_kinematics.npz
  - physics/contact_graph.npy
  - physics/anchor_targets.npz
  - physics/energy.npz