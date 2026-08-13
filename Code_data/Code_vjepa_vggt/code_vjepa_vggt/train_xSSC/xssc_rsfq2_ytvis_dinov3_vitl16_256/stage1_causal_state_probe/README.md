# xSSC Stage 1 causal state audit

The experiment contract is in [plan.md](plan.md). All commands below run in the
foreground. They do not implement or train an ODE.

## 1. Causal adaptation

Select a completed noncausal MOVi-C step-050000 checkpoint and two currently
available GPUs. GPU 4 must not be used.

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256
SOURCE_CHECKPOINT=/data/gaoya/agent-data/checkpoints/REPLACE_WITH_SELECTED_STEP_050000.pth \
GPU_IDS=5,6 \
bash run_train_stage1_movic_24f_prefix_causal.sh
```

The launcher intentionally refuses to guess the source checkpoint or GPUs.

## 2. Causality gate

Run this before accepting a trajectory cache:

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256
PYTHONPATH="$PWD:$PWD/upstream:/home/gaoya/Code_Video/vjepa2-main" \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
stage1_causal_state_probe/test_future_perturbation.py \
  --checkpoint /data/gaoya/agent-data/checkpoints/REPLACE_WITH_CAUSAL_BEST.pth \
  --config-file upstream/config-randsfq/rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-24f-slot512-prefix-causal-stage1.py \
  --device cuda:0 \
  --output /data/gaoya/agent-data/outputs/xssc_stage1_causal_state/causality_gate.json
```

## 3. Cache official splits

Repeat for `train`, `validation`, and `test`. Records are restart-safe and RGB is
not duplicated.

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256
PYTHONPATH="$PWD:$PWD/upstream:/home/gaoya/Code_Video/vjepa2-main" \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
stage1_causal_state_probe/cache_causal_slots.py \
  --checkpoint /data/gaoya/agent-data/checkpoints/REPLACE_WITH_CAUSAL_BEST.pth \
  --split validation \
  --device cuda:0
```

Then compute train-only normalization statistics:

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256
PYTHONPATH="$PWD" /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
stage1_causal_state_probe/compute_slot_stats.py \
  --output /data/gaoya/agent-data/cache/xssc_stage1_causal_state/train_slot_stats.pt
```

## 4. Audit and train the factorial matrix

The matrix trains three representations, three histories, two context modes,
and three seeds. It also trains prefix and boundary probes and evaluates both
fixed mappings.

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256
DEVICE=cuda:0 bash stage1_causal_state_probe/run_stage1_matrix.sh
```

Representation stability is independent of predictor training:

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256
PYTHONPATH="$PWD" /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
stage1_causal_state_probe/audit_representation.py \
  --split test \
  --output-dir /data/gaoya/agent-data/outputs/xssc_stage1_causal_state/representation_audit
```

## 5. Case frames and overview

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256
PYTHONPATH="$PWD:$PWD/upstream:/home/gaoya/Code_Video/vjepa2-main" \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
stage1_causal_state_probe/visualize_cases.py \
  --config-file upstream/config-randsfq/rsfq2_c-movi_c-vjepa2_1_vitl16_256-video-24f-slot512-prefix-causal-stage1.py \
  --split test --indices 0 1 2 3 4 --mapping prefix \
  --output-dir /data/gaoya/agent-data/outputs/xssc_stage1_causal_state/case_gallery
```

Build paired history/context contrasts and the overview:

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256
PYTHONPATH="$PWD" /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
stage1_causal_state_probe/analyze_results.py \
  --results-root /data/gaoya/agent-data/outputs/xssc_stage1_causal_state/evaluations/prefix \
  --output-dir /data/gaoya/agent-data/outputs/xssc_stage1_causal_state/paired_analysis

PYTHONPATH="$PWD" /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
stage1_causal_state_probe/build_dashboard.py \
  --results-root /data/gaoya/agent-data/outputs/xssc_stage1_causal_state \
  --case-gallery /data/gaoya/agent-data/outputs/xssc_stage1_causal_state/case_gallery \
  --output-dir /data/gaoya/agent-data/outputs/xssc_stage1_causal_state/dashboard
```

To serve the generated overview locally in the foreground:

```bash
python3 -m http.server 8899 --bind 127.0.0.1 --directory /data/gaoya/agent-data/outputs/xssc_stage1_causal_state/dashboard
```

## Tests

```bash
cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/xssc_rsfq2_ytvis_dinov3_vitl16_256
/home/gaoya/miniconda3/envs/flux/bin/python -m pytest -q stage1_causal_state_probe/tests
```
