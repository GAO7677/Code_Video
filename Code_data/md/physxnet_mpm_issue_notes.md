# PhysXNet MPM 问题记录

代码：
- `/home/gaoya/Code_Video/Code_data/try1_physxnet_articulation_mpm.py`

## 1. Genesis 材质/边界报错

问题：
- 部分 MPM 物体运行时会报材质显示模式错误，或粒子超出求解边界。

报错：
```text
genesis.GenesisException: Unsupported `surface.vis_mode` for material <gs.materials.MPM.Sand>: 'visual'. Expected one of: ['particle', 'recon'].
```

原因：
- 某些 MPM 材质不支持 `vis_mode='visual'`。
- 自定义 MPM 物体加入后，原来的 MPM 边界不够大。

解决：
- 给不同 MPM 材质使用合法的显示模式。
- 扩大 `MPMOptions.lower_bound / upper_bound`，把自定义物体也包含进去。

## 2. 19925 首帧桌布消失

问题：
- `19925` 的桌布在首帧直接消失。

原因：
- 桌子随机旋转了 yaw，但桌布没有同步旋转。
- 初始时桌布和桌子 mesh 穿插，导致 cloth 数值不稳定。

解决：
- 让桌布跟随主物体继承相同 yaw。
- 给桌布初始位置留一个小 gap，减少初始穿插。
