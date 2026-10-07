# 场景几何变体仿真：6组 / 18例

已删除最后的「桥板下方支座位置」组；此前的「活动挡板转轴位置」组也不包含在此包中。
保留前6组各3个变体，场景参数、相机与物理解算器沿用已交付的v2源码，没有重新设计运动。

这是自编NumPy简化求解器与CPU渲染器，不是PyBullet，也不是经认证的正式benchmark GT。

## 保留场景

| ID | 场景 | 控制量 | 默认三个值（m） |
|---|---|---|---|
| 1 | 圆柱障碍横向偏移 | `cylinder_y` | 0.65 / 0 / 0.30 |
| 2 | 有限挡板长度 | `barrier_length` | 0.65 / 0.88 / 1.75 |
| 3 | 双轨净间距 | `rail_gap` | 0.14 / 0.30 / 0.56 |
| 4 | 坡峰高度 | `hump_height` | 0.10 / 0.23 / 0.52 |
| 5 | 台面孔洞横向位置 | `hole_y` | 0 / 0.40 / 0.85 |
| 6 | 被撞方块到桌缘的距离 | `edge_x`（桌缘世界x坐标） | 0.65 / 1.05 / 1.90 |

第6组方块初始中心x=0.45m、沿x半长0.16m，因此三个参数不是“净距离”本身。
对应桌缘到方块前表面的初始距离为0.04 / 0.44 / 1.29m。

## 运行

需要Python 3.10+；CPU即可，不需要GPU或外部物理引擎。建议使用独立环境。

```bash
cd geometry_sim_6cases
python -m pip install -r requirements.txt
python simulate.py --families all --out results
```

默认输出90帧、30fps、640×360的单例视频；前8帧另存context，物理积分720Hz。
总时长编码为3秒，采样时刻为0至89/30秒。运行结束后打开 `results/index.html`。
该HTML引用同目录的视频，移动页面时应一起保留整个results目录。

```bash
# 查看场景编号和参数
python simulate.py --list-families

# 只运行双轨与桌缘组，并导出全部PNG帧
python simulate.py --families 3 6 --save-frames --out selected_results

# 896×512单例输出，组内相机不变
python simulate.py --families all --width 896 --height 512 --out results_896

# 只计算物理轨迹，不渲染视频
python simulate.py --families all --physics-only --out physics_results

# 更小步长检查；结局一致不等于已证明高精度收敛
python simulate.py --families all --physics-only --sim-hz 1440 --out fine_check

# 源码快速检查：18例前8帧的有限性、组内一致性及场景注册
python -m unittest discover -s tests -v
```

`--sim-hz`必须为`--fps`的整数倍。`--frames`必须大于`--context`。
修改总帧数不会自动选择关键事件：短片段可能还没有发生接触，输出结局标签仅描述该片段。
每次建议指定新的输出目录，避免与其他配置的旧文件混用。

## 文件结构

```text
simulate.py             # 仿真、渲染、导出与生成对比页面
src/scenes.py           # 六组几何及参数；FAMILIES列表定义变体值
src/physics.py          # 平面刚体 + 三维球/静态盒简化求解器
src/render.py           # CPU正交投影、深度缓冲渲染
requirements.txt        # 依赖范围
requirements-lock.txt   # 本次环境中的固定依赖版本
checks/                 # 源码保留核查及本次运行检查记录
```

每例导出 `rgb.mp4`、`context_8.mp4`、`metadata.json`、`initial_geometry.json`、
`trajectories.npz`、`contacts.json`、`summary.json`及首尾图片。
`--save-frames`额外输出 `frames/00000.png` 等逐帧图片。
每组导出 `comparison.mp4`、`preview.gif` 与对比图片；总目录包含 `all_scenes.mp4` 和HTML。
`--physics-only`只导出数值数据，不输出视频、图片或HTML。

`trajectories.npz`中含 `time_s`，以及`ball__position`、`ball__velocity`等数组。
第6组同时包含`block__position`等。平面xy组位置为[x,y]，平面xz组为[x,z]，
三维组为[x,y,z]；请结合`metadata.json`中的`model`字段读取，不要混用维度。

## 说明与限制

1/2组为水平二维运动；4/6组为竖直平面刚体；3/5组为三维球与静态轴对齐盒接触。
没有统一的全三维多刚体仿真。渲染外观为几何预览，不是之前Test70的写实室内场景。

组内保持目标初始运动和相机一致，仅改变指定几何参数；允许派生接触与运动改变。
第4组坡面仍是48段几何近似，控制峰高会同时改变局部坡度和曲率。
第2组 `summary.json`中的“端点附近接触”与“挡板后转向”子标签还使用长度阈值，
不是从完整接触点和出射方向严格验证的指标；其他结局同样只是粗粒度诊断。

单例 `rgb.mp4`和context不含结果文字；`comparison.mp4`与GIF有控制值和整段结果标签，
仅供人工审查，不能作为模型输入。前8帧历史一致不能替代事件时窗准入或物理精度验证。

源码包不包含字体文件。中文对比文字使用系统字体；缺字时可用环境变量
`GEOM_DEMO_FONT`指向本机已有的中文字体文件。字体缺失不影响无文字RGB和数值轨迹。

依赖固定记录为本次环境版本，不代表在所有操作系统/Python版本均已验证；
无法安装锁定版本时使用requirements.txt。详细执行范围见checks中的记录。

## 本次打包检查

18例均重新完成90帧、720Hz数值仿真，保存的所有状态数组有限；组内前8帧目标位置一致。
与先前交付完整包中这18例的轨迹数组逐项比较，全部完全相同。
另对18例各执行12帧的渲染、MP4/GIF/PNG导出与HTML生成检查；90帧视频本次未重新渲染。
两项unittest通过，已移除的7/8号场景输入会被命令行拒绝。
以上是源码裁剪与运行检查，不是物理精度认证。
