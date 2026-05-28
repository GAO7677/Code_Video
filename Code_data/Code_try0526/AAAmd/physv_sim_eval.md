# PhysV Simulation Benchmarking

## 三组测试样本

| 组 | 来源 | 样本数 | 说明 |
|----|------|--------|------|
| A. PDI-Bench 生成视频 | `Output_try0526/PDI-Bench/output/` | 60 (4 method × 15) | GT / Wan / VACE 真实与生成视频 |
| B. 仿真视频 | `Dataset_physV/0526dp/videos/` | 28 (8 ball_block + 20 sensitivity) | PyBullet 刚体仿真，控制变量改变物理参数 |
| C. 帧序打乱 | `Dataset_physV/0526dp/videos/shuffle_test/` | 2 | GT ball 原始 vs 随机打乱帧 |

## 三个评估指标

| 指标 | 含义 | 方向 | 来源 |
|------|------|------|------|
| pdi_score | PDI-Bench 几何一致性（scale + traj + rigidity） | ↓ 越低越好 | SAM2 + CoTracker3 + MegaSaM |
| wmreward_jepa | WMReward 官方 sliding-window V-JEPA2 surprise | ↑ 越高越好（similarity） | Meta WMReward, ViT-G 384 |
| vjepa_proxy | 自定义 V-JEPA2 三路加权分数 | ↑ 越高越好 | Code_try0526 rerank_video/scorers.py |

## A. PDI-Bench 生成视频

### 汇总

| method | N | pdi_score ↓ | scale_error ↓ | traj_error ↓ | rigidity_error ↓ | vp_error ↓ | wmreward_jepa ↑ |
|--------|---|-------------|---------------|---------------|------------------|------------|-----------------|
| GT | 15 | 0.144 | 0.066 | 0.183 | 0.223 | 0.261 | 0.412 |
| VACE_1p3B_TI2V | 15 | 0.466 | 0.862 | 0.234 | 0.139 | 0.240 | 0.431 |
| VACE_1p3B_ctx08 | 15 | 0.569 | 1.111 | 0.232 | 0.158 | 0.339 | 0.425 |
| wan22-5B-TI2V | 15 | 0.878 | 1.883 | 0.292 | 0.040 | 0.169 | 0.435 |

**PDI 与 WMReward JEPA 排序相反**：GT 几何最优（PDI=0.144）但 JEPA 最差（0.412）。Wan 几何最差（PDI=0.878）但 JEPA 最好（0.435）。真实视频视觉复杂不可预测，生成视频视觉单调易于预测。

### 各 case vjepa_proxy

| method | mean vjepa_proxy ↑ |
|--------|--------------------|
| GT | — (需要 context video) |
| wan22-5B-TI2V | — |
| VACE_1p3B_TI2V | — |
| VACE_1p3B_ctx08 | — |

vjepa_proxy 需要 GT context prefix，不适用于直接评估独立视频，此处未计算。

## B. 仿真视频

### B1. Ball-Block 物理参数（8 组）

固定外观，改变恢复系数 e / 摩擦 μ / 球质量 m。

| scenario | e | μ | m | pdi_score ↓ | wmreward_jepa ↑ | vjepa_proxy ↑ |
|----------|---|---|---|-------------|-----------------|---------------|
| e09 superball | 0.9 | 0.5 | 1.0 | 0.106 | 0.454 | 0.750 |
| e07 bouncy | 0.7 | 0.5 | 1.0 | **0.022 (A)** | 0.457 | 0.750 |
| e05 medium | 0.5 | 0.5 | 1.0 | 0.142 | 0.461 | 0.744 |
| e03 plastic | 0.3 | 0.5 | 1.0 | 0.273 | 0.461 | 0.741 |
| e07 low-fric | 0.7 | 0.1 | 1.0 | 0.191 | 0.456 | 0.747 |
| e07 high-fric | 0.7 | 1.0 | 1.0 | 0.138 | **0.450 (最意外)** | 0.740 |
| e07 light-ball | 0.7 | 0.5 | 0.1 | **2.064 (F)** | 0.467 | 0.742 |
| e07 heavy-ball | 0.7 | 0.5 | 5.0 | 0.139 | **0.469 (最优)** | 0.747 |

PDI 有效区分几何质量（轻球 F vs 弹性 A），WMReward JEPA 全量仅 0.019 跨度，vjepa_proxy 全量仅 0.010 跨度。

### B2. JEPA 运动敏感性（20 组）

固定外观(V1)，系统改变速度/质量/重力/碰撞/方向。

| 变量 | 范围 | pdi_score ↓ 范围 | wmreward_jepa ↑ 范围 | vjepa_proxy ↑ 范围 |
|------|------|-----------------|---------------------|--------------------|
| 初速 | 0.5–14.0 m/s (28×) | — | 0.469–0.479 | 0.744–0.751 |
| 球质量 | 0.01–100 kg (10⁴×) | — | 0.474–0.491 | 0.729–0.747 |
| 重力 | 4.9–19.6 m/s² (4×) | — | 0.467–0.478 | 0.744–0.754 |
| 木块质量 | 0.5–20 kg (40×) | — | 0.474–0.482 | 0.736–0.750 |
| 无碰撞 | — | — | 0.495 | 0.748 |
| 反向 | — | — | 0.467 | 0.726 |

WMReward JEPA 全范围 0.045，vjepa_proxy 全范围 0.029。两个 JEPA 变体对运动参数变化均几乎无感。

### B3. 外观敏感性（8 组 × 3 外观）

同一物理轨迹，不同外观（V1 默认/V2 暗蓝冷光/V3 暖亮暖光）。

| scenario | V1 默认 PDI ↓ | V2 暗蓝 PDI ↓ | V3 暖亮 PDI ↓ | max/min |
|----------|--------------|--------------|--------------|---------|
| e05 medium | 0.154 | **0.020** | 0.249 | **12.6×** |
| e07 bouncy | 0.048 | **0.022** | 0.221 | 10.1× |
| e03 plastic | 0.152 | **0.045** | 0.288 | 6.4× |

PDI 对外观敏感（4.8×–12.6×），暗背景始终得分最优，亮背景最差。使用 PDI 时需固定渲染管线。

## C. 帧序打乱 Sanity Check

从 A 组取 5 个 GT 视频，B 组取 5 个仿真视频，随机打乱帧序（seed=42），测试指标对时序破坏的敏感性。

### C1. PDI-Bench GT 视频打乱

| 视频 | PDI 原始 ↓ | PDI 打乱 ↓ | Δ | WMR 原始 ↑ | WMR 打乱 ↑ | Δ |
|------|-----------|-----------|----|-----------|-----------|-----|
| GT ball | 0.175 | **0.102** | -0.073 | 0.446 | 0.439 | -0.007 |
| GT blackswan | 0.380 | **0.010** | -0.370 | 0.380 | 0.418 | +0.038 |
| GT bus | — | **0.019** | — | 0.391 | 0.405 | +0.015 |
| GT car-turn | — | **0.208** | — | 0.410 | 0.419 | +0.009 |
| GT rhino | — | **0.036** | — | 0.413 | 0.421 | +0.008 |

### C2. 仿真视频打乱

| 视频 | PDI 原始 ↓ | PDI 打乱 ↓ | Δ | WMR 原始 ↑ | WMR 打乱 ↑ | Δ |
|------|-----------|-----------|----|-----------|-----------|-----|
| sim baseline (e=0.7) | 0.022 | **0.011** | -0.011 | 0.457 | 0.441 | -0.016 |
| sim light-ball (m=0.1) | 2.064 | **1.974** | -0.090 | 0.467 | 0.451 | -0.016 |
| sim high-fric (μ=1.0) | 0.138 | **0.026** | -0.112 | 0.450 | 0.441 | -0.009 |
| sim vel_140 | — | **0.280** | — | 0.472 | 0.460 | -0.012 |
| sim mass_001 | — | **3.300** | — | 0.483 | 0.466 | -0.017 |

### 分析

**PDI 系统性失效**：打乱帧后 PDI 普遍降低（10 个中 8 个 Δ<0）。GT blackswan 从 0.38 骤降至 0.01——SAM2/CoTracker3 在破坏时序后几乎无法工作，稀疏/错误的 mask 和轨迹意外产生了极低的几何误差。

**WMReward JEPA 方向正确但幅度极小**：GT 组 Δ 平均 -0.009（方向不一致，部分反而上升），sim 组 Δ 平均 -0.014。完全时序破坏的 JEPA 变化仍在指标噪声范围内（<0.02），V-JEPA2 的 16 帧滑动窗口 causal prediction 对全局帧序几乎无感。

**两个指标都未通过时序 sanity check**。使用前必须验证被评估视频的时序完整性。

---

## 汇总结论

1. **PDI 能区分几何质量但对外观和感知前端敏感**：仿真中 PDI 有效（0.022 A vs 2.064 F），但对渲染外观（12× 变化）和帧序破坏（反向改善）不稳定
2. **WMReward JEPA 对运动/外观/帧序均不敏感**：全量跨度 < 0.1，不适合作为简单刚体仿真的独立物理指标
3. **vjepa_proxy 区分力更弱**：全量跨度 < 0.03，且非标准实现
4. **PDI 和 JEPA 互补**：PDI 衡量几何正确性，JEPA 衡量视觉可预测性，排序相反

## 文件结构

```
/data/gaoya/AAA_test_video/Dataset_physV/0526dp/
├── videos/ball_block/              # B1: 8 组物理参数 mp4 + json
├── videos/ball_block_appearance/   # B3: 24 组外观变体 mp4 + json
├── videos/jepa_sensitivity/        # B2: 20 组运动变体 mp4 + json
├── videos/shuffle_test/            # C: 帧序打乱 mp4 + json
├── wmreward_report/index.html      # 可视化 → http://127.0.0.1:18707
└── tmp_eval*/                      # PDI 评估中间文件

/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench/
├── output/GT/ VACE_*/ wan22-*/     # A: 60 个生成视频 mp4 + json (wmreward_jepa)
└── result/metrics.csv              # 汇总表
```

## 可视化

```bash
# 综合报告 → http://127.0.0.1:18707
python3 physics_sim/serve_final_report.py
```
