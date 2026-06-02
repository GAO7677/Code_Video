# 001 Capsule 视觉姿态与物理姿态重复施加

## 问题现象

部分 `capsule` 视频在某些帧里看起来像“部分嵌入地面”，例如：

- `simple_f1_capsule_upright_tumble_slide`

这种现象在翻滚姿态较大、接近贴地接触时更明显。

## 原因

渲染 mesh 和物理刚体都吃到了 `orientation_euler_deg`。

具体表现为：

- `_make_mesh()` 里先对 mesh 施加了一次初始旋转
- `createMultiBody()` 又把同样的初始旋转作为 `baseOrientation` 传给 PyBullet
- 每一帧 `update_pose()` 再根据 PyBullet 返回姿态更新节点

结果是：

- 物理碰撞体姿态是对的
- 渲染 mesh 额外带了一层初始旋转偏置
- 某些视角下会看起来像插进地面

## 解决方案

去掉 `_make_mesh()` 里的初始姿态预旋转。

保留单一姿态来源：

- 初始姿态只交给 PyBullet `baseOrientation`
- 渲染节点每帧只跟随 PyBullet 返回的 `pos + quat`

## 对应文件与函数

- 文件：
  - [generate_sim_preview_gallery.py](/home/gaoya/Code_Video/phys_state_video/scripts/generate_sim_preview_gallery.py)
- 关键函数：
  - `_make_mesh()`
  - `run_scenario()`
  - `PreviewRenderer.update_pose()`

## 备注

这个问题属于“视觉对齐问题”，不一定意味着物理求解本身真的发生了严重穿透。
