# Probe-Wan22 数据集生成流程

本文档描述从原始 v2v_jsons 到 probe 训练就绪数据集的完整流程，包括每步的运行指令、幂等性说明和断点续跑策略。

---

## 目录结构

```
probe_wan22/datasets/generated/
├── generations/
│   ├── base/                     # wanti2v 生成的视频 + JSON
│   ├── openvid_lora/             # openvid LoRA 生成的视频 + JSON
│   └── pybullet_lora/            # pybullet LoRA 生成的视频 + JSON
├── manifests/
│   ├── normalized_inputs/        # 路径规范化后的输入 JSON（自动生成）
│   ├── input_jsons.txt           # 当次运行的输入文件列表
│   ├── generation_run_config.json
│   ├── generation_registry_base.csv
│   ├── generation_registry_openvid_lora.csv
│   ├── generation_registry_pybullet_lora.csv
│   ├── generation_registry_all.csv   # 所有模型合并（merge，非覆盖）
│   ├── wmreward_run_config.json
│   ├── generated_probe_pairs.csv
│   ├── generated_probe_pairs.jsonl
│   └── generated_probe_pairs_summary.json
├── wmreward/
│   ├── base/wmreward_scores.csv
│   ├── openvid_lora/wmreward_scores.csv
│   └── pybullet_lora/wmreward_scores.csv
└── wmreward_pending/             # 临时 symlink 目录，每次 backfill 重建
```

---

**GPU 约束：** 不得使用 `gpu4`（硬件故障）。推荐 `gpu0`。

---

## Step 1 — 视频生成

### 1.1 运行全量三模型（首次或补跑）

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/wmreward/probe_wan22/run_generation_pipeline.py \
  --pipeline-root /data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/datasets/generated \
  --models base,openvid_lora,pybullet_lora \
  --cuda-visible-devices 0
```

**幂等性：**
- `base` 模型（wanti2v）：已存在 `{stem}.mp4 + {stem}.json` 时自动 skip，无需 `--force`。
- `openvid_lora` / `pybullet_lora`（batch_eval_lora）：已存在时自动 skip。
- `generation_registry_all.csv` 会合并当次未运行的模型的已有 CSV，**不会**丢失历史数据。

**断点续跑（例如只补跑 base 剩余视频）：**

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/wmreward/probe_wan22/run_generation_pipeline.py \
  --pipeline-root /data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/datasets/generated \
  --models base \
  --cuda-visible-devices 0
```

已生成的视频会被 skip，只生成缺失的。其他模型的 registry CSV 会被合并进 `registry_all.csv`。

### 1.2 smoke 测试（验证单条）

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/wmreward/probe_wan22/run_generation_pipeline.py \
  --smoke-name test_$(date +%m%d) \
  --models base \
  --limit 1 \
  --sampling-steps 5 \
  --cuda-visible-devices 0
```

Smoke 输出写到 `/data/gaoya/AAA_test_video/0626vjepa_free/wmreward/tmp/smoke/pipeline_runs/test_<date>/`，不影响正式数据。

### 1.3 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--sampling-steps` | 40 | 扩散步数，与 probe 对齐 |
| `--cfg-scale` | 5.0 | CFG 系数 |
| `--seed` | 42 | 所有模型统一 seed |
| `--size` | `704*1280` | base 模型分辨率 |
| `--height` / `--width` | 512 / 896 | LoRA 模型分辨率 |
| `--frame-num` | 25 | base 模型帧数 |
| `--num-frames` | 24 | LoRA 模型帧数 |
| `--force` | 未设置 | 设置后强制重新生成已有视频（慎用） |

---

## Step 2 — WMReward 回填

对所有已生成但尚无 wmreward 分数的视频计算 surprise/similarity，写回各自的 JSON。

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/wmreward/probe_wan22/backfill_wmreward_scores.py \
  --pipeline-root /data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/datasets/generated \
  --models base,openvid_lora,pybullet_lora \
  --wmreward-checkpoint-path /data/gaoya/ckpt/Sylvest-vjepa2-vit-g/vitg-384.pt \
  --device cuda:0
```

**幂等性：**
- 只处理 `wmreward_status != ok` 的行（`output_json_exists=True` 且 `output_video_exists=True`）。
- 多次运行安全，已有分数的视频不会重复计算。
- 完成后刷新各模型的 `generation_registry_<model>.csv` 和 `registry_all.csv`。

**断点续跑（只补特定模型）：**

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/wmreward/probe_wan22/backfill_wmreward_scores.py \
  --pipeline-root /data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/datasets/generated \
  --models base \
  --device cuda:0
```

未指定的模型的 registry 会被原样合并入 `registry_all.csv`。

---

## Step 3 — 构建配对 Manifest

从三模型的 registry 中按 `basename` 分组，取最低/最高 surprise 配对。

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/wmreward/probe_wan22/build_generation_manifest.py \
  --pipeline-root /data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/datasets/generated \
  --subset-name generated_probe_pairs \
  --min-group-size 2
```

**前提：** 三个模型均已完成 Step 1 + Step 2，否则每个 basename 只有 1 条 ok 行，`min-group-size=2` 会产生 0 对。

**幂等性：** 每次运行从 `registry_all.csv` 重新计算，输出 CSV/JSONL 直接覆盖（结果确定性）。

**输出：**
- `manifests/generated_probe_pairs.csv` — probe 使用的配对 manifest
- `manifests/generated_probe_pairs_summary.json` — 配对统计

---

## Step 4 — Probe 特征提取

对配对 manifest 中的每个样本重放 Wan2.2 推理（40 步），在指定 step/layer 采集 transformer 激活。

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/gaoya/diffuser_code/src:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/wmreward/probe_wan22/extract_probe_features.py \
  --manifest_csv /data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/datasets/generated/manifests/generated_probe_pairs.csv \
  --output_root /data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/extracted \
  --device cuda:0 \
  --num_inference_steps 40 \
  --capture_steps 8,20,32 \
  --capture_layers 2,8,14,20,29 \
  --seed_mode source
```

**严格对齐原则：**
- `--num_inference_steps 40` 与数据集生成步数一致，确保采集的激活对应相同扩散轨迹
- `--seed_mode source` 从每个 JSON 的 `seed` 字段读取（均为 42），保证完全可复现
- `--capture_steps 8,20,32` 对应 40 步中的 20% / 50% / 80% 位置

**幂等性：** 已存在 `probe_features.pt + meta.json` 的样本自动 skip。强制重跑加 `--overwrite`。

**smoke 测试（单样本验证）：**

```bash
CUDA_VISIBLE_DEVICES=0 \
PYTHONPATH=/home/gaoya/diffuser_code/src:/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/wmreward/probe_wan22/extract_probe_features.py \
  --manifest_csv /data/gaoya/AAA_test_video/0626vjepa_free/wmreward/subsets/subset16_smoke.csv \
  --output_root /data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/extracted_smoke \
  --limit 1 \
  --device cuda:0
```

---

## Step 5 — 构建 Probe Index

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/wmreward/probe_wan22/build_probe_index.py \
  --feature_root /data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/extracted \
  --output_csv /data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/indices/probe_index.csv
```

**幂等性：** 每次重新扫描 extracted/ 目录，完整重建 index。

---

## Step 6 — Probe 训练

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/wmreward/probe_wan22/train_ridge_probe.py \
  --index_csv /data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/indices/probe_index.csv \
  --output_root /data/gaoya/AAA_test_video/0626vjepa_free/wmreward/probe_wan22/probe_results \
  --target_field source_surprise_score \
  --feature_keys h_post_global_mean,delta_h_global_mean,h_post_frame_mean,delta_h_frame_mean \
  --max_splits 4
```

**输出：**
- `probe_results/probe_metrics.csv` — 每个 (step, layer, feature) 组合的 Pearson/Spearman/R²/MAE
- `probe_results/probe_summary.json` — 汇总

---

## 当前进度（2026-07-03）

| Step | 状态 | 备注 |
|------|------|------|
| Step 1 base | 部分完成 | 673/951 视频已生成；277 phyco + 1 physicIQ 待补跑 |
| Step 1 openvid_lora | 未开始 | — |
| Step 1 pybullet_lora | 未开始 | — |
| Step 2 base | 基本完成 | 635/673 有 wmreward；38 待补 |
| Step 2 lora | 未开始 | 待 Step 1 完成 |
| Step 3 | 待执行 | 需三模型均有数据 |
| Step 4 | 待执行 | — |
| Step 5–6 | 待执行 | — |

**下一步：** 在 `gpu0` 上运行 Step 1（`--models base` 补跑 278 个），随后启动两个 LoRA 模型，再按顺序执行 Step 2–6。

---

## 幂等性总结

| 脚本 | 不加 flag 的默认行为 | 强制重跑 |
|------|---------------------|----------|
| `run_generation_pipeline.py` | 已有 mp4+json 的 skip | `--force` |
| `backfill_wmreward_scores.py` | 只处理 wmreward_status≠ok | 无需 flag，再次运行自动幂等 |
| `build_generation_manifest.py` | 覆盖输出（确定性重算） | 直接重跑 |
| `extract_probe_features.py` | 已有 probe_features.pt 的 skip | `--overwrite` |
| `build_probe_index.py` | 全量重扫，覆盖输出 | 直接重跑 |
| `train_ridge_probe.py` | 覆盖输出（确定性） | 直接重跑 |
