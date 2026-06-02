# 002 PyBullet 动态 Mesh 碰撞能力边界

## 问题现象

用户希望所有物体都直接用真实 mesh 做动态刚体碰撞，不使用代理碰撞。

## 原因

当前仿真链路基于 PyBullet。

PyBullet 对动态刚体更稳定的路径是：

- sphere
- box
- cylinder
- capsule
- convex-like primitive

对于复杂动态 mesh：

- `GEOM_MESH` 不能等价理解为“逐三角真实动态碰撞”
- 凹三角网格更适合静态场景，而不是当前这种动态刚体训练集

## 解决方案

当前阶段收缩到简单刚体：

- `sphere`
- `box`
- `cylinder`
- `capsule`
- `puck`

原则是：

- 碰撞形状和视觉形状尽量一致
- 先保证物理合理和状态监督干净
- 暂不在 PyBullet 上强行做复杂动态 mesh 碰撞

## 对应文件与函数

- 文件：
  - [generate_sim_preview_gallery.py](/home/gaoya/Code_Video/phys_state_video/scripts/generate_sim_preview_gallery.py)
- 关键函数：
  - `_make_mesh()`
  - `_collision_shape()`
  - `build_preview_scenarios()`

## 备注

这不是单纯代码 bug，而是当前物理引擎能力边界和数据集目标之间的折中。
