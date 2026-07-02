# wan_stage1b_context_only_v2v — 推理流程文档

## 运行指令

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
python3 /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_stage1b_context_only_v2v.py \
  --checkpoint /data/gaoya/agent-data/checkpoints/pybullet0629_teacher_student/stage1b_context_only/<STAGE1B_STEP>.pt \
  --init-from /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name <MODEL_NAME> \
  --sampling-steps 40 \
  --save-raw \
  [--limit N] [--force]
```

默认 config：`object_token_teacher_student/config_stage1b_context_only_template.yaml`

---

## 一、权重加载顺序

推理权重分四层叠加，顺序不能颠倒：

| 顺序 | 参数来源 | 文件路径 | 包含内容 | 加载方式 |
|------|---------|---------|---------|---------|
| ① | Wan DiT base | `config model.wan_ckpt_dir` → `/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B` (3 shards) | Wan 5B DiT 基础权重；新增 object 分支随机初始化 | `from_pretrained`, strict 加载已有 keys |
| ② | Wan LoRA（冻结） | `config model.init_wan_lora_from_checkpoint` → `.../step-000500/checkpoint.safetensors` | rank-32 LoRA 权重，与训练完全对齐 | `trainer constructor` 内 `load_lora_checkpoint(strict=False, zero_missing=True)` |
| ③ | Stage1A pooler | `--init-from step_0005000.pt` | `object_pooler` + `object_aux_heads`（63/63 tensors） | `load_state_dict(strict=False)`，step counter 不恢复 |
| ④ | Stage1B trainables | `--checkpoint step_XXXXXX.pt` | `object_adapter` + Wan DiT `object_embedding` / `object_cross_attn` / `object_gate` / `norm4` | `load_state_dict(strict=False)`，overlay 在③之上 |

> **为什么 ③ 必须先于 ④**：Stage1B checkpoint 只保存可训练参数（adapter + Wan object 分支），不包含 object_pooler；若先加载 ④ 再加载 ③，pooler 会被 ③ 的 strict=False 静默跳过，导致 pooler 仍是随机初始化。正确顺序是先 ③ 填满 pooler，再 ④ 覆盖 adapter/Wan object 分支。

---

## 二、完整前向流程与 shape

以下以单张推理为例（B=1，context_frames=8，resolution=512×896，num_objects=8）。

### Step 1 — text_encoder（Wan T5-XXL）

```
输入:  caption string
模块:  bundle.text_encoder
输出:  text_context  [seq_len=17, D=4096]     # token 数取决于 caption 长度
```

### Step 2 — VAE encode（Wan 3D-VAE）

```
输入:  context_video  [B=1, C=3, T=8, H=512, W=896]
模块:  bundle.vae.encode
公式:  T_lat = (T - 1) // vae_stride_t + 1 = (8-1)//4+1 = 2
       C_lat = 48,  H_lat = H//8 = 64→32,  W_lat = W//8 = 112→56
输出:  context_latents  [C=48, T_lat=2, H=32, W=56]   # 无 batch dim，返回 list
```

实测 shape：`[48, 2, 32, 56]`，std≈0.65。

### Step 3 — JEPA 特征提取

```
输入:  context_video  → resize 到 jepa_input_size=384
模块:  jepa_adapter (ViT-g/16, tubelet=2)
输出:  patch_tokens  [B=1, T×num_patches, D_jepa=1408]
```

### Step 4 — GT bbox → CoTracker query points

```
输入:  context_boxes  [B=1, T_ctx=8, N_obj=8, 4]   # normalized xyxy，来自 .npz GT
模块:  ContextOnlyInjectionTrainer._maybe_build_query_priors
逻辑:  对每个 object，找第一帧有效 box，采样 points_per_object 个点
输出:  query_points_grouped  [B=1, N_obj=8, points_per_object, 2]   (pixel coords)
       object_valid_mask     [B=1, N_obj=8]
```

### Step 5 — CoTracker 跟踪

```
输入:  frames_bthwc  [B=1, T=8, H=384, W=512, C=3]   (cotracker_input_hw)
       query_points  [B=1, total_queries, 2]
模块:  cotracker_adapter (ScaledOfflineCoTracker)
输出:  tracks      [B=1, T=8, N_total, 2]
       visibility  [B=1, T=8, N_total]
       confidence  [B=1, T=8, N_total]
→ _group_tracks_to_objects:
       tracks_grouped      [B=1, T=8, N_obj=8, points_per_object, 2]
       visibility_grouped  [B=1, T=8, N_obj=8, points_per_object]
```

### Step 6 — object_pooler（ObjectTubeProjector）

```
输入:  jepa_patch_tokens      [B, T×patches, 1408]
       context_latents        [C=48, T_lat=2, H=32, W=56]  → stack → [B, C, T_lat, H, W]
       tracks_grouped         [B, T=8, N_obj=8, P, 2]
       visibility_grouped     [B, T=8, N_obj=8, P]
       object_valid_mask      [B, N_obj=8]
模块:  ObjectTubeProjector (Stage1A 训练，推理时冻结)
输出:  object_latent_tokens   [B=1, T_lat=2, N_obj=8, D=4096]
       active_track_summary   [B, T, N, ...]
```

### Step 7 — object_adapter（ObjectConditionAdapter）

```
输入:  object_latent_tokens   [B=1, T_lat=2, N_obj=8, D=4096]
       object_valid_mask      [B=1, N_obj=8]
模块:  ObjectConditionAdapter (Stage1B 训练目标)
内部:
  slot_bias  [N_obj=8, D]         # 区分不同 object slot
  time_bias  [T_lat=2, D]         # 区分不同 latent 时刻
  x = tokens + slot_bias + time_bias
  x = norm(x)
  x = MLP(x)                      # 两层 linear + GELU
  gate = sigmoid(gate_param)      # 可学习标量 gate
  x = gate * x
  → flatten T_lat × N_obj
输出:  object_context             [T_lat×N_obj=16, D=4096]
```

实测：推理时 `context_fraction=0.5`，context 只覆盖前半段，T_lat=1，故 shape 为 `[8, 4096]`。

### Step 8 — 构造 latent_clean 与初始噪声

```
context_latents  [C=48, T_lat=2, H=32, W=56]
→ latent_clean   [B=1, T_total_lat=4, H=32, W=56]    # zero-pad future frames
  T_total_lat = (total_frames - 1) // vae_stride_t + 1 = (16-1)//4+1 = 4
→ noise = randn_like(latent_clean)
→ x_t = context_mask * clean + (1 - context_mask) * (σ₀*noise + (1-σ₀)*clean)
```

实测 shape：`latent_clean [48, 4, 32, 56]`，`x_t_init [48, 4, 32, 56]`。

### Step 9 — DiT denoising loop（num_inference_steps 步）

每步：

```
输入:
  x_t              [B=1, C=48, T_lat=4, H=32, W=56]
                   → patchify → seq_len = T_lat*H*W / (patch_h*patch_w)
  t_tokens         [B=1, seq_len]      (当前 timestep 广播)
  text_context     [17, 4096]          → Wan text cross-attn（每个 DiT block）
  object_context   [[8, 4096]]         → Wan object cross-attn（每个 DiT block，tanh gate）

模块:  bundle.dit (WanModel, 30 blocks)
每个 block 内:
  self-attn → text cross-attn → object cross-attn (tanh(object_gate) residual) → FFN

输出:  pred  [B=1, C=48, T_lat=4, H=32, W=56]   （same shape as x_t）
```

实测 pred shape：`[48, 4, 32, 56]`，std 从步骤 0 的 1.38 收敛到步骤末 0.72。

DPM-Solver 更新 x_t → 重复 N 步。

### Step 10 — VAE decode

```
输入:  pred  [C=48, T_lat=4, H=32, W=56]
模块:  bundle.vae.decode
输出:  video  [T=16, C=3, H=512, W=896]
→ clamp(-1,1) → *127.5+127.5 → uint8 → mp4
```

---

## 三、shape 速查表（实测，单卡 GPU 0）

| 张量 | Shape | dtype | 备注 |
|------|-------|-------|-----|
| `context_latents` | `[48, 2, 32, 56]` | float32 | context 8帧→T_lat=2 |
| `text_context` | `[17, 4096]` | bfloat16 | 17 text tokens |
| `object_context` | `[8, 4096]` | float32 | T_lat×N_obj=1×8（context only） |
| `latent_clean_init` | `[48, 4, 32, 56]` | bfloat16 | 含未来帧（zero-pad） |
| `x_t_init` | `[48, 4, 32, 56]` | bfloat16 | 加噪后初始 latent |
| `pred_step_i` | `[48, 4, 32, 56]` | float32 | 每步 DiT 输出 |
| `output video` | `[16, 3, 512, 896]` | uint8 | VAE decode 后 |

---

## 四、冻结/训练参数对照

| 模块 | 推理阶段来源 | 是否冻结 |
|------|------------|---------|
| Wan DiT backbone (attn/FFN/PE) | ① Wan base ckpt | 冻结 |
| Wan LoRA weights | ② LoRA ckpt | 冻结 |
| Wan object_embedding | ④ Stage1B ckpt | 冻结（推理） |
| Wan object_cross_attn (all 30 blocks) | ④ Stage1B ckpt | 冻结（推理） |
| Wan object_gate (all 30 blocks) | ④ Stage1B ckpt | 冻结（推理） |
| Wan norm4 (all 30 blocks) | ④ Stage1B ckpt | 冻结（推理） |
| object_adapter | ④ Stage1B ckpt | 冻结（推理） |
| object_pooler | ③ Stage1A ckpt | 冻结（推理） |
| object_aux_heads | ③ Stage1A ckpt | 冻结（推理） |
| VAE | ① Wan base ckpt | 冻结 |
| text_encoder | ① Wan base ckpt | 冻结 |
| JEPA / VGGT / CoTracker | 各自 ckpt_dir | 冻结 |
