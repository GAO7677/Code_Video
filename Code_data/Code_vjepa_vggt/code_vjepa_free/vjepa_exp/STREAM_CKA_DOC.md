# Stream CKA：V-JEPA vs Wan DiT-5B 表征相似度分析

**脚本**：`stream_cka.py`

用 **Cosine-kernel CKA**（token 先 L2 归一化再构造 Gram）逐视频比较 V-JEPA 编码器与 Wan DiT-5B 各层在同一时空网格上的表征相似度。注意这与标准 Linear CKA（直接用 X@X.T 作为核矩阵）不同：L2 归一化会丢弃特征范数信息，使核矩阵变为余弦相似度核。

---

## 设计原则

- **流式处理**：每视频跑一次前向，即时计算 Gram 矩阵 → CKA，结果累加到 running sum，无 token/Gram 缓存写盘
- **统一时空网格**：DiT token 通过 trilinear 插值对齐到 V-JEPA 网格，保证两侧 Gram 使用完全相同的 N=T×H×W 个时空坐标点
- **断点续跑**：每视频的 `[9,30,5]` CKA grid 写盘为 `<case_dir>/<sample_name>.npy`，重启时自动跳过已存在的文件

---

## 关键参数（`stream_cka.py` 顶部，L40–L57）

| 参数 | 值 | 说明 |
|------|----|------|
| `VJEPA_H, VJEPA_W` | 160, 240 | V-JEPA 输入分辨率 |
| `PATCH, TUBELET` | 16, 2 | V-JEPA 空间 patch 和时间 tubelet 大小 |
| `ALIGN_H, ALIGN_W` | 10, 15 | V-JEPA 空间 token 网格，固定 |
| `ALIGN_T` | 动态解析 | 首次前向后由 encoder 输出推算（典型值 12） |
| `WAN_PATCH_SIZE` | (1,2,2) | DiT patch 大小 |
| `WAN_VAE_STRIDE` | (4,16,16) | VAE 时空下采样倍率 |
| `VJEPA_LAYERS` | [0,3,5,8,11,14,17,20,23] | 9 层 V-JEPA 中间输出 |
| `DIT_LAYERS` | 0–29 | 全部 30 层 DiT |

---

## 完整计算流程

### Step 1：视频读取（L65–L102）

```
load_video_frames(num_frames=25, first_frames=True)
  → [3, 25, H_orig, W_orig]  float32, [-1,1]
```

`first_frames=True` 取前 N 帧，`False` 则均匀采样。

---

### Step 2：V-JEPA 预处理（L131–L139）

```python
preprocess_vjepa(frames)   # L131
```

```
frames[:, 1:]              → [3, 24, H_orig, W_orig]  # 丢弃第 0 帧（条件帧）
[-1,1] → [0,1] → ImageNet 归一化
bicubic resize → 160×240   → [3, 24, 160, 240]
unsqueeze(0)               → [1, 3, 24, 160, 240]
```

---

### Step 3：V-JEPA 前向与动态解析 ALIGN_T（L143–L157）

```python
vjepa_grams(encoder, frames, device)   # L143
```

```
encoder([1, 3, 24, 160, 240])
  tubelet=2 → T_p = 24/2 = 12
  patch=16  → H_p = 160/16 = 10, W_p = 240/16 = 15
  N = 12×10×15 = 1800 tokens, D = 1024

  输出 9 层: 每层 [1, 1800, 1024]

首次调用时 (L148–L152):
  ALIGN_T = N // (ALIGN_H × ALIGN_W) = 1800 // 150 = 12
  打印: "[V-JEPA] resolved grid: T=12 H=10 W=15 → N=1800"

每层:
  .squeeze(0).reshape(12, 10, 15, 1024)   → [T,H,W,D]
  → gram_aligned()                         → [1800, 1800]
```

返回 `{layer_idx: [1800, 1800]}`，共 9 个。

---

### Step 4：Gram 矩阵计算（L107–L116）

```python
gram_aligned(grid)   # L107
```

```
grid [T, H, W, D]
  reshape(N, D)
  F.normalize(dim=-1)   → [N, D]  每 token L2 norm=1（cosine kernel，非 linear kernel）
  tok @ tok.T           → [N, N]  余弦相似度矩阵，值域 [-1, 1]
```

---

### Step 5：DiT VAE 编码（L199–L223，每视频一次）

```python
dit_grams(pipe, frames, prompt, device, timesteps, height, width)   # L199
```

```
bicubic resize → 480×720    → [3, 25, 480, 720]
vae.encode()
  时间 stride=4 → T_lat = 7
  空间 stride=16 → H_lat=30, W_lat=45
  → z [16, 7, 30, 45]

DiT patch (1,2,2):
  T_p=7, H_p=15, W_p=22  (注：width=720 时)
  seq_len = 7×15×22 = 2310 (或 width=832 时为 2730)
```

---

### Step 6：DiT 前向与 hook 捕获（L229–L247，每 timestep 一次）

```python
capture = BlockOutputCapture(DIT_LAYERS)   # L162
capture.register(pipe.model)               # L168
```

```
对每个 τ ∈ {100, 300, 500, 700, 900}:
  noisy_z = (1-σ)·z + σ·ε,  σ = τ/1000
  DiT forward:
    x = video token 序列 [1, seq_len, 3072]
    context (text) 作为独立参数传入各 block 的 cross-attention，
    不拼入 x — 见 wan/modules/model.py WanModel.forward()
    hook 捕获每层 block(x, ...) 的输出 [1, seq_len, 3072]
  取 raw[0, :seq_len, :]   → [2310, 3072]  （直接是全部 video token，无需截断 ctx）
```

---

### Step 7：DiT Token 对齐到 V-JEPA 网格（L184–L196）

```python
align_to_vjepa_grid(x_flat, T_p, H_p, W_p)   # L184
```

```
[2310, 3072]
  reshape(7, 15, 22, 3072)       → [T_p, H_p, W_p, D]
  permute + unsqueeze            → [1, 3072, 7, 15, 22]
  F.interpolate(trilinear, (12, 10, 15))
                                 → [1, 3072, 12, 10, 15]
  squeeze + permute              → [12, 10, 15, 3072]   ← 与 V-JEPA 同坐标
```

再经 `gram_aligned()` → `[1800, 1800]`。

返回 `{τ: {layer_idx: [1800, 1800]}}`，共 5×30 个。

---

### Step 8：CKA 计算（L119–L126，L288–L296）

```python
cka_from_grams(K, L)   # L119
```

```
对所有 (vi, di, ti) 组合:
  Kv = vj_grams[vi]     [1800, 1800]
  Kd = dt_grams[τ][di]  [1800, 1800]

  H  = I - (1/N)·11ᵀ   # 中心化矩阵
  Kc = H·Kv·H
  Lc = H·Kd·H
  CKA = Σ(Kc⊙Lc) / √(Σ(Kc⊙Kc)·Σ(Lc⊙Lc))   → scalar ∈ [0,1]

每视频输出: grid [9, 30, 5]
写盘: <case_dir>/<sample_name>.npy
```

---

### Step 9：累加与保存（L253–L306，L523–L536）

```python
CKAAccumulator.update()   # L266
```

```
sum[dataset]   += grid    [9, 30, 5]
count[dataset] += 1

全部完成后:
  mean_grid = sum / count

写盘:
  <out_dir>/cka_sums.npz   — sum/count/vj_layers/dt_layers/timesteps
  <case_dir>/timesteps.npy — 本次使用的 timestep 列表
  <out_dir>/cka_per_dataset.png
  <out_dir>/cka_per_layer_curve.png
  <out_dir>/cka_per_timestep.png
  <out_dir>/cka_matrices.npz
```

---

## 关键 Shape 汇总

| 阶段 | tensor | shape |
|------|--------|-------|
| 读入帧 | frames | `[3, 25, H, W]` |
| V-JEPA 输入 | video_tensor | `[1, 3, 24, 160, 240]` |
| V-JEPA 每层输出 | feat | `[1, 1800, 1024]` |
| V-JEPA 时空网格 | grid | `[12, 10, 15, 1024]` |
| DiT latent | z | `[16, 7, 30, 45]` |
| DiT token 序列 | x_flat | `[2310, 3072]` |
| DiT 插值后网格 | grid | `[12, 10, 15, 3072]` |
| 两侧 Gram | K / L | `[1800, 1800]` |
| 每视频 CKA | grid | `[9, 30, 5]` |
| 数据集均值 | mean_grid | `[9, 30, 5]` |

N=1800 = 12×10×15，是 V-JEPA 与 DiT 共用的时空 token 数量。

---

## 运行方式

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main \
CUDA_VISIBLE_DEVICES=4 python stream_cka.py \
  --manifest <manifest.json> \
  --out-dir  <out_dir> \
  --case-dir <case_dir> \
  --num-frames 25 \
  --first-frames \
  --timesteps 100 300 500 700 900 \
  --device cuda
```

- `--manifest`：JSON 列表，每条含 `video_path`、`source`、`caption`、`sample_name`
- `--case-dir`：每视频 `[9,30,5]` 结果写盘路径，断点续跑时自动跳过已存在文件
- `--out-dir`：汇总 npz 和图表输出路径
- `--first-frames`：取前 N 帧（推荐，保持时序一致性）
