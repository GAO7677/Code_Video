# `phyco_dataset.py` 说明

## 1. 文件目的

`phyco_dataset.py` 的作用是把原始 PhyCo 数据集目录

- `/data/gaoya/dataset/nnsriram97-phyco_kubric/<scenario>/<date>.tar.gz`

适配成当前 `code_vjepa_vggt` 项目里接近 `PhysStateEpisodeDataset` 的训练输入格式。  
它不是直接读取已经整理好的 `train/*.npz + *.json`，而是：

1. 从原始 `tar.gz` 中索引样本。
2. 解出 `rgba.mp4`、`segmentation.mp4`、`metadata.json`。
3. 从分割视频里恢复 2D box。
4. 用这些 2D box 构造当前项目需要的 `state / appearance / camera` 张量。
5. 把处理结果缓存到：
   - `/data/gaoya/agent-data/cache/phyco_vjepa_dataset/raw`
   - `/data/gaoya/agent-data/cache/phyco_vjepa_dataset/episodes`
   - `/data/gaoya/agent-data/cache/phyco_vjepa_dataset/indices`

它的目标是“先能对接当前训练链路”，不是“严格复原 PhyCo 的全部物理 GT 语义”。

---

## 2. 代码关键位置

| 功能 | 代码位置 |
| --- | --- |
| 主 dataset 类 | [phyco_dataset.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/data/phyco_dataset.py#L370) |
| 伪造 camera 向量 | [phyco_dataset.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/data/phyco_dataset.py#L130) |
| 24 帧时间采样 | [phyco_dataset.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/data/phyco_dataset.py#L183) |
| 从分割视频提 box / area | [phyco_dataset.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/data/phyco_dataset.py#L212) |
| 构造 `state` 和 `boxes` | [phyco_dataset.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/data/phyco_dataset.py#L272) |
| 构造 `appearance` | [phyco_dataset.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/data/phyco_dataset.py#L336) |
| split 哈希划分 | [phyco_dataset.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/data/phyco_dataset.py#L464) |
| 样本缓存与处理主流程 | [phyco_dataset.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/data/phyco_dataset.py#L523) |
| `__getitem__` 输出对齐 | [phyco_dataset.py](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/data/phyco_dataset.py#L637) |

---

## 3. `__getitem__` 返回字段说明

下面是 `PhyCoEpisodeDataset[idx]` 返回的字段。

| 字段 | 类型 / 形状 | 中文说明 | 备注 |
| --- | --- | --- | --- |
| `video` | `torch.Tensor [3, 24, H, W]` | 整段视频，已经 resize，并且数值范围变成 `[-1, 1]` | 对齐现有训练代码 |
| `context_video` | `torch.Tensor [3, K, H, W]` | 从 `video` 里取出的上下文帧子序列 | 当前默认 `K=8` |
| `caption` | `str` | 文本提示词 | 来自 scenario 级 caption 文件 |
| `video_path` | `str` | 样本来源路径 | 格式是 `tar.gz:member_path` |
| `frame_indices` | `torch.Tensor [24]` | 当前 episode 内的帧序号 | 这里只是 0 到 23 |
| `context_frame_indices` | `torch.Tensor [K]` | 当前上下文帧索引 | 默认是前 8 帧，或按内部策略选取 |
| `num_context_frames` | `int` | 实际上下文帧数 | 通常等于 `K` |
| `metadata` | `dict` | 当前缓存后的元信息 | 包含 `sample_key`、对象选择信息、depth/camera 模式等 |
| `context_boxes` | `torch.Tensor [K, N, 4]` | 上下文帧中的目标框 | 归一化 `xyxy`，范围理论上在 `[0, 1]` 附近 |
| `future_boxes` | `torch.Tensor [T, N, 4]` | 未来帧中的目标框 | 当前默认 `T=16` |
| `context_states` | `torch.Tensor [K, N, 10]` | 上下文帧中的状态向量 | 10 维定义见下表 |
| `future_states` | `torch.Tensor [T, N, 10]` | 未来帧中的状态向量 | 10 维定义见下表 |
| `appearance` | `torch.Tensor [N, 16]` | 每个对象的外观 / 粗物理属性向量 | 16 维定义见下表 |
| `camera` | `torch.Tensor [K, 10]` | 相机条件向量 | 当前是近似值，不是严格 GT |

这里：

- `H, W` 通常由 dataset 初始化时的 `resolution` 指定，例如 `(512, 896)`
- `K = num_context_frames`
- `T = num_future_frames`
- `N = max_objects`

---

## 4. `state` 10 维字段解释

`context_states[..., :]` 和 `future_states[..., :]` 的最后一维长度固定为 10。

| 下标 | 字段名 | 中文解释 | 当前来源 |
| --- | --- | --- | --- |
| `0` | `center_x` | 目标框中心点的归一化横坐标 | 由分割框中心计算 |
| `1` | `center_y` | 目标框中心点的归一化纵坐标 | 由分割框中心计算 |
| `2` | `depth` | 深度代理值 | 当前是由 mask 面积推出来的 proxy depth |
| `3` | `log_scale` | 尺度代理值的对数 | 当前是 `log(mask_area / image_area)` |
| `4` | `vel_x` | 横向速度代理 | 相邻帧 `center_x` 差分 |
| `5` | `vel_y` | 纵向速度代理 | 相邻帧 `center_y` 差分 |
| `6` | `depth_vel` | 深度速度代理 | 相邻帧 `depth` 差分 |
| `7` | `visibility` | 可见性标记 | 当前帧能否在分割中找到对应区域 |
| `8` | `existence` | 存在性标记 | 当前实现中只要 slot 被选中就置 1 |
| `9` | `confidence` | 置信度标记 | 当前实现中可见时 1，不可见时 0 或保守值 |

---

## 5. `appearance` 16 维字段解释

`appearance[obj_idx]` 的长度固定为 16，布局尽量对齐现有 PhysState 训练格式。

| 范围 | 字段 | 中文解释 | 当前来源 |
| --- | --- | --- | --- |
| `0:5` | shape one-hot | 物体形状 one-hot | 由 `object_data.type` 字符串启发式映射 |
| `5:8` | role one-hot | 物体角色 one-hot：`dynamic/support/occluder` | 由类型关键词和运动分数推断 |
| `8:11` | color RGB | 物体颜色 | 直接取 `metadata.object_data.color` |
| `11` | max scale | 尺度三维中的最大值 | 取 `metadata.object_data.scale` |
| `12` | min scale | 尺度三维中的最小值 | 取 `metadata.object_data.scale` |
| `13` | scale volume proxy | 尺度体积代理 | `scale_x * scale_y * scale_z` |
| `14` | mass | 质量 | 取 `metadata.object_data.mass` |
| `15` | friction | 摩擦系数 | 取 `metadata.object_data.friction` |

### `shape one-hot` 当前映射

| one-hot 下标 | 形状名 |
| --- | --- |
| `0` | `sphere` |
| `1` | `box` |
| `2` | `cylinder` |
| `3` | `capsule` |
| `4` | `puck` |

### `role one-hot` 当前映射

| one-hot 下标 | 角色名 |
| --- | --- |
| `5` | `dynamic` |
| `6` | `support` |
| `7` | `occluder` |

---

## 6. 当前所有“不是严格 GT，而是占位或启发式”的部分

下面这张表是当前实现里最重要的边界条件。  
如果后续要做严格物理监督，优先从这里开始替换。

| 项目 | 影响字段 | 当前实现 | 为什么不是严格 GT / 风险 |
| --- | --- | --- | --- |
| split 划分 | dataset 属于 `train/val/test` 哪个 split | 用 `scenario/date/sample_id` 做稳定哈希切分 | 这不是 PhyCo 官方 split，只是为了先接当前训练流程 |
| 时间采样 | `video`、`context_video`、所有缓存 episode 张量 | 从原始视频均匀采样 24 帧 | 这是为了贴合当前 `8 + 16` 训练配置，不是原始数据自带的 episode 定义 |
| 相机重建 | `camera`、`camera_full` | 只使用 `camera_position`，再配合固定 FoV 假设构造 10 维向量 | 没有真实完整内参和外参，所以只是近似 camera condition |
| 深度 | `state[..., 2]` | `depth = sqrt(base_area / area)` | 这是 2D box/mask 面积代理，不是 3D 几何深度，也不是 `depth.mp4` 解码结果 |
| 深度速度 | `state[..., 6]` | 相邻帧 proxy depth 差分 | 因为 depth 本身就是代理值，所以 depth velocity 也只是代理 |
| 尺度 | `state[..., 3]` | `log(mask_area / image_area)` | 是投影面积，不是物体真实 3D 尺度状态 |
| 位置 | `state[..., 0:2]` | 用 2D segmentation box 中心 | 这是图像平面中心，不是世界坐标位置 |
| 速度 | `state[..., 4:6]` | 相邻帧中心差分 | 是图像平面速度，不是模拟器里的真实速度 |
| 可见性 | `state[..., 7]` | 当前帧是否检测到对应 segmentation 区域 | 取决于压缩后分割视频的颜色匹配质量 |
| 存在性 | `state[..., 8]` | 只要 slot 被保留就默认置 1 | 不是原始模拟器里单独给出的 object existence GT |
| 置信度 | `state[..., 9]` | 当前实现是启发式 0/1 | 不是原始标注里的单独 confidence |
| 物体 shape | `appearance[:, 0:5]` | 从 `object_data.type` 做字符串匹配 | 类型命名不统一时可能误判 |
| 物体 role | `appearance[:, 5:8]` | 根据类型关键词和运动分数推断 `dynamic/support/occluder` | 这是人为规则，不一定符合原场景语义 |
| 物体排序 | `boxes/states/appearance` 的对象维 | 按 motion、visibility、area 排序，再截断到 `max_objects` | slot 顺序不是原始模拟器顺序，不保证跨样本语义一致 |
| 物体选择 | `boxes/states/appearance` 的对象维 | 仅保留前 `max_objects` 个对象 | 被截断的对象不会进入监督 |
| segmentation 解码 | `context_boxes`、`future_boxes`，以及所有依赖 box/mask 的状态项 | 对压缩后的 `segmentation.mp4` 做“最近颜色 + 容差阈值”匹配 | 压缩边界、颜色漂移、遮挡边缘都可能带来错误 |
| box 提取 | `context_boxes`、`future_boxes`、`full_boxes` | 从分割 mask 取 axis-aligned bbox | 这不是模拟器直接给的 bbox GT，而是二次提取出来的 2D 框 |
| `video_path` 语义 | `video_path` | 保存为 `tar.gz:member_path` | 它不是本地真实 mp4 路径，而是逻辑来源标识 |

---

## 7. 当前更适合哪些训练脚本

当前实现更适合：

- 不严格依赖真实 depth GT 的训练
- 主要需要：
  - `video`
  - `context_video`
  - `context_boxes`
  - `future_boxes`
  - 基本可用的 `context_states / future_states`

比较适合的是你之前提到的这一类脚本：

- [run_train_v_newtrain_gpu67.sh](/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/run_train_v_newtrain_gpu67.sh)

原因是这条脚本里：

- 开了 object branch
- 需要 box / state 结构对齐
- 但 `lambda_depth_aux=0.0`

所以 proxy depth 目前不是第一阻塞项。

---

## 8. 当前不建议直接当严格 GT 使用的场景

下面这些情况不建议直接把当前 `phyco_dataset.py` 当成“严格真值”：

- 需要高精度 depth supervision
- 需要真实 3D camera parameter
- 需要真实 world-space velocity / position
- 需要固定 object slot identity 严格跨样本一致
- 需要全对象完整监督而不是 top-`max_objects` 子集

---

## 9. 如果后续要继续补强，优先级建议

建议的替换顺序如下：

1. 先替换 depth：
   - 优先尝试解码 `depth.mp4`
   - 或从更可靠的 3D 几何信息恢复 depth

2. 再替换 camera：
   - 查 PhyCo 是否能从额外元信息恢复完整 intrinsics / extrinsics

3. 再替换 object state：
   - 如果能从 `animation_data.pkl` 或其他轨迹文件恢复全对象时序，就不要再依赖 2D proxy state

4. 最后替换 slot 组织：
   - 建立稳定 object id 到 slot 的映射
   - 避免按 motion/area 动态排序

---

## 10. 一句话总结

当前 `phyco_dataset.py` 是一个“训练可接入版适配器”：

- 结构上尽量对齐现有 `phys_state_episode`
- 数值上仍有一批 proxy / heuristic 项
- 足够用于先打通训练链路
- 但还不能当成严格物理 GT 版本
