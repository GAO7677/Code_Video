# PhysV Simulation Benchmarking

## Experiment

用 PyBullet 物理引擎生成球撞击木块仿真视频（8 组参数组合），对比原始物理评估指标（PDI-Bench + V-JEPA2）在不同外观下的表现，验证 Benchmark 的外观敏感性。

### Pipeline

```
PyBullet（物理仿真 + 轨迹记录） → Pyrender（PBR 渲染，3 种外观） → PDI-Bench / V-JEPA2 评估
```

## Input

- 8 组物理参数：恢复系数 e∈{0.3,0.5,0.7,0.9}，摩擦 μ∈{0.1,0.5,1.0}，球质量 m∈{0.1,1.0,5.0} kg
- 3 种外观变体：V1（默认橙球灰地板暖白光）、V2（蓝球暗地板冷蓝光）、V3（绿球亮地板暖黄光）
- 球初速 (3.5, 0, 1.8) m/s，重力 9.81 m/s²，木块质量 1.5kg

## Output

```
/data/gaoya/AAA_test_video/Dataset_physV/0526dp/
├── videos/ball_block/           # 原始 8 组 mp4 + json (PDI + JEPA)
├── videos/ball_block_appearance/ # 24 组外观变体 mp4 + json (PDI)
├── eval_report/index.html        # 原始 PDI 结果页 (port 18703)
└── appearance_report/index.html  # 外观敏感性对比页 (port 18705)
```

## 运行指令

```bash
# 生成原始视频
conda run -n wan python physics_sim/simulate_ball_block.py

# 生成外观变体
conda run -n wan python physics_sim/simulate_appearance_variants.py

# 可视化
python3 physics_sim/serve_appearance_compare.py  # → http://127.0.0.1:18705
python3 physics_sim/eval_ball_block.py             # → http://127.0.0.1:18703
```

## Results

### PDI-Bench（几何一致性，↓ 越低越好）

| Scenario | e | μ | m | PDI | Grade |
|----------|---|---|---|------|-------|
| e09 superball | 0.9 | 0.5 | 1.0 | 0.106 | B |
| e07 bouncy | 0.7 | 0.5 | 1.0 | **0.022** | **A** |
| e05 medium | 0.5 | 0.5 | 1.0 | 0.142 | B |
| e03 plastic | 0.3 | 0.5 | 1.0 | 0.273 | B |
| e07 low-fric | 0.7 | 0.1 | 1.0 | 0.191 | B |
| e07 high-fric | 0.7 | 1.0 | 1.0 | 0.138 | B |
| e07 light-ball | 0.7 | 0.5 | 0.1 | **2.064** | **F** |
| e07 heavy-ball | 0.7 | 0.5 | 5.0 | 0.139 | B |

### V-JEPA2（预测合理性，↑ 越高越好）

所有 8 组 JEPA 分数集中在 0.740–0.750，几乎无区分度。合成视频的运动模式高度相似，JEPA 无法有效判别几何质量差异。

### Appearance Sensitivity（同一物理轨迹，3 种外观）

| Scenario | V1 默认 | V2 暗蓝 | V3 暖亮 | max/min |
|----------|---------|---------|---------|---------|
| e05 medium | 0.1542 | **0.0197** | 0.2486 | **12.6×** |
| e07 bouncy | 0.0475 | **0.0218** | 0.2209 | 10.1× |
| e07 heavy | 0.0426 | 0.0469 | **0.2962** | 6.9× |
| e03 plastic | 0.1521 | **0.0449** | 0.2877 | 6.4× |
| e09 superball | 0.1259 | **0.0227** | 0.1505 | 6.6× |

PDI 分数在同物理轨迹下的外观变体间差异达 **4.8×–12.6×**。

## Analysis

1. **PDI-Bench 对外观敏感**。SAM2 掩码质量、DepthAnything 深度估计、CoTracker3 轨迹跟踪均受物体-背景对比度和光照影响。暗背景(V2)始终得分最优，亮背景(V3)始终最差。

2. **PDI 能区分物理参数差异**。轻球(0.1kg) PDI=2.06(F) 远差于其他场景；弹性碰撞(e=0.7) PDI=0.022(A) 最佳。Scale error 是主要区分维度。

3. **V-JEPA2 对合成视频无区分力**。所有场景 JEPA≈0.745，合成视频的运动模式过于相似，JEPA 的 predictive prior 无法捕捉细微几何差异。

4. **使用 PDI-Bench 时需固定渲染管线**。外观变化会引入不可忽略的分数漂移，跨方法/跨模型对比时必须保证一致的渲染环境和视觉条件。

---

## Experiment 2: JEPA Sensitivity to Motion

### 目的

固定外观（V1 默认），系统改变运动参数，测试 V-JEPA2 对运动差异的敏感性阈值。

### 变量设计（控制变量法）

| 实验组 | 变量 | 取值 | 基准 |
|--------|------|------|------|
| Velocity sweep | 球初速 vx | 0.5, 1.5, 3.5, 7.0, 14.0 m/s (28×) | vx=3.5 |
| Mass sweep | 球质量 | 0.01, 0.05, 0.1, 1.0, 5.0, 20, 100 kg (10⁴×) | m=1.0 |
| Gravity sweep | 重力加速度 | 4.9, 9.81, 19.6 m/s² (4×) | g=9.81 |
| Block mass | 木块质量 | 0.5, 1.5, 5.0, 20 kg (40×) | m=1.5 |
| Collision | 碰撞 vs 不碰撞 | 球从上方飞过 | 碰撞 |
| Direction | 正反向 | 球从右侧撞来 | 左侧 |

固定：e=0.7, μ=0.5, 外观=V1

### Results

| 实验组 | JEPA 范围 | max-min | 结论 |
|--------|----------|---------|------|
| Velocity (28×) | 0.7437–0.7509 | 0.0072 | **不敏感** |
| Mass (10⁴×) | 0.7292–0.7471 | 0.0179 | 极低质量(0.01kg)略低 |
| Gravity (4×) | 0.7436–0.7542 | 0.0106 | **不敏感** |
| Block mass (40×) | 0.7361–0.7501 | 0.0140 | **不敏感** |
| No collision | 0.7482 | — | 与碰撞基线 0.7471 无差异 |
| Reverse | 0.7256 | — | 全部最低，但仍极接近 |
| **全部 20 组** | **0.7256–0.7542** | **0.0286** | **几乎无区分力** |

### Analysis

V-JEPA2 对纯刚体仿真视频的运动差异**极其不敏感**。即使将球质量拉满 10⁴ 倍、速度拉满 28 倍、重力从月球到超重、甚至完全取消碰撞，JEPA 分数变化仅 0.029（<4% 相对变化）。原因：

1. V-JEPA2 在自然视频上训练，学会了自然场景中的复杂运动模式（形变、流体、生物运动），simple 刚体碰撞的 latent motion 远在其训练分布之内，不触发 predictive surprise
2. 合成视频缺乏纹理细节、光照变化、遮挡等视觉复杂性，predictor 的 token-level 重建误差始终很低
3. JEPA 的 predictive prior 更适合判别"是否像自然视频"而非"运动参数是否合理"

**结论**：V-JEPA2 不适合作为合成/简单刚体仿真视频的物理质量评估指标。需要几何显式指标（如 PDI）才能区分运动参数差异。

### Output

```
videos/jepa_sensitivity/   # 20 组运动变体 mp4 + json (JEPA)
```

### 可视化

```bash
# 外观敏感性 → http://127.0.0.1:18705
python3 physics_sim/serve_appearance_compare.py

# JEPA 敏感性 → http://127.0.0.1:18703
python3 physics_sim/eval_ball_block.py
```
