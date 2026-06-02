# 仿真数据集说明与可视化入口

## 当前仿真数据方案

当前 `phys_state_video` 的仿真数据方案，优先采用基于 PyBullet 和 Pyrender 的简单刚体方案，先保证物理行为稳定、状态监督完整、视觉外观可控，再逐步扩展规模与复杂度。当前主对象集合是 `sphere / box / cylinder / capsule / puck`，固定使用地球重力 `9.81 m/s²`，显式随机化地面摩擦，并避免中途凭空出生的物体，通过 `pre-roll` 保证入镜连续性。场景主要覆盖单物体运动、双体碰撞、多体连锁、遮挡重现、支撑与跌落五类现象，其中 `capsule` 又额外扩展了多组不同初始角度、线速度和角速度的滚滑/翻滚 case。

视觉主题当前维护两条并行版本：工业训练数据版和日常物体版。两者共用同一套物理参数和场景结构，只在材质、纹理和对象语义上切换，用来同时检查“物理运动是否合理”和“外观变化下的泛化可读性”。默认生成仿真数据时，统一优先走工业训练数据版；日常物体版主要作为外观泛化和对照展示使用。目前工业版和日常版都已经整合进同一个总页面里，后续所有仿真数据集可视化都统一放在这个总页面入口下，不再分散维护多个独立页面。

## 总页面入口

统一总页面目录：

- [sim_objstate_overview_v1](/data/gaoya/AAA_test_video/Dataset_physV/sim_objstate_overview_v1)

统一本地访问地址：

- `http://127.0.0.1:18827`


## 启动可视化页面指令

总页面统一刷新命令：

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/phys_state_video/scripts/generate_sim_overview_page.py \
  --clean \
  --port 18827
```

默认主题，也就是工业训练数据版的单独生成/刷新命令：

```bash
/data/gaoya/miniconda3/envs/wan/bin/python \
  /home/gaoya/Code_Video/phys_state_video/scripts/generate_sim_preview_gallery.py \
  --clean \
  --theme industrial \
  --output-root /data/gaoya/AAA_test_video/Dataset_physV/sim_objstate_rigid_simple_v1_preview \
  --port 18825
```


## 后续维护约定

以后所有新的仿真数据集可视化，都统一并入这个总页面，不再额外新开分散入口。默认维护顺序是：先刷新工业训练数据版，再在需要时刷新日常物体版，最后使用总页面命令更新统一入口，并继续使用本地端口 `18827` 作为固定入口。

## 问题记录

当前仿真数据集相关问题已统一整理到一个文件中，便于后续持续补充和快速检索：

- [仿真数据集问题汇总.md](/home/gaoya/Code_Video/phys_state_video/md/仿真数据集问题汇总.md)
