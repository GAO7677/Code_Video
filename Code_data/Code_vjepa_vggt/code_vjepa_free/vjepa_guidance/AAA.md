# V-JEPA Guidance Run Index

This file records the main model families that have been run under
`code_vjepa_free/vjepa_guidance`, together with:

- the primary script
- the main output directory
- a representative rerun command

Notes:

- Paths below are the stable locations currently used in this workspace.
- `Wan2.1 1.3B` has smoke and batch outputs, but it was not preserved in the
  same clean 5-family A/B layout as the Wan2.2 families.
- `gpu4` is intentionally not used.

## 1. Wan2.2 Official TI2V-5B

Primary script:

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wanti2v.py`
- Frequency-guided wrapper:
  `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wanti2v_freqguidance.py`
- Official 17-case pilot runner:
  `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_official_freqguide_test5.py`
- Full-metric multicase scorer:
  `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_multicase_allmetrics.py`

Main result directories:

- `/data/gaoya/agent-data/outputs/wanti2v_official_clean_batch_eval`
- `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/wan22_official_ti2v5b`
- `/data/gaoya/agent-data/outputs/vjepa_mask_union_except_first_case`
- `/data/gaoya/agent-data/outputs/vjepa_freqguide_smoke`
- `/data/gaoya/agent-data/outputs/vjepa_guidance_trace/wan22_official_freqguide_000301_gpu21`

Baseline rerun command:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main \
CUDA_VISIBLE_DEVICES=6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wanti2v.py \
  --input-list /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name wan2p2_ti2v5b_baseline \
  --backend official \
  --size 704*1280 \
  --frame-num 49 \
  --sampling-steps 40 \
  --cfg-scale 5.0 \
  --fps 30 \
  --seed 42 \
  --offload-model \
  --vjepa-preset baseline
```

Guided rerun command:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main \
CUDA_VISIBLE_DEVICES=6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wanti2v.py \
  --input-list /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name wan2p2_ti2v5b_target_w24_s15_ratio_0025 \
  --backend official \
  --size 704*1280 \
  --frame-num 49 \
  --sampling-steps 40 \
  --cfg-scale 5.0 \
  --fps 30 \
  --seed 42 \
  --offload-model \
  --vjepa-preset target_w24_s15_ratio_0025 \
  --vjepa-ckpt /data/gaoya/ckpt/VJEPA2/vith.pt
```

Frequency-guided smoke command:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main \
CUDA_VISIBLE_DEVICES=2,1 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wanti2v_freqguidance.py \
  --input-list /data/gaoya/agent-data/outputs/vjepa_freqguide_smoke/case_000301.txt \
  --output-root /data/gaoya/agent-data/outputs/vjepa_freqguide_smoke/guided_gpu21 \
  --model-name wan22_official_freqguide_000301_gpu21 \
  --backend official \
  --size 704*1280 \
  --frame-num 49 \
  --sampling-steps 40 \
  --cfg-scale 5.0 \
  --fps 30 \
  --seed 42 \
  --offload-model \
  --vjepa-preset target_w24_s15_ratio_0025 \
  --vjepa-ckpt /data/gaoya/ckpt/VJEPA2/vith.pt \
  --vjepa-device-id 1 \
  --trace-intermediates \
  --trace-build-html
```

## 2. Wan2.2 Early LoRA Step-000500

Primary script:

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan_openvid_0613pybullet_lorav2v_vjepa.py`

Main result directories:

- `/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results/lora_test5`
- `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/wan22_early_lora_step000500`
- `/data/gaoya/agent-data/outputs/vjepa_phase4_multicase`

Representative guided rerun command:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
CUDA_VISIBLE_DEVICES=0,1 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan_openvid_0613pybullet_lorav2v_vjepa.py \
  --weights-root /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_100.txt \
  --model-name wan_openvid_0613pybullet_lorav2v_step000500_vjepa \
  --device cuda:0 \
  --vjepa-device cuda:1 \
  --num-frames 49 \
  --vjepa-guidance-steps 2 \
  --vjepa-latent-step-size 0.01
```

## 3. Train0705 Custom Wan2.2 Step-001000

Primary batch wrapper:

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_current_modes.py`

Underlying generation script:

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py`

Main result directory:

- `/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes`

Representative current-modes rerun command:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=6,7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_current_modes.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
  --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --model-name-prefix train0705_current \
  --device cuda:0 \
  --vjepa-device cuda:1
```

Related sweep wrappers already used:

- `run_train0705_guard_ablation.py`
- `run_train0705_ratio_cap_sweep.py`
- `run_train0705_s15_local_sweep.py`
- `run_train0705_round7_expansion.py`

Representative guard-ablation rerun command:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=6,7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_guard_ablation.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
  --device cuda:0 \
  --vjepa-device cuda:1 \
  --initialize-model-on-cpu
```

Representative local-sweep rerun command:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
CUDA_VISIBLE_DEVICES=6,7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_train0705_s15_local_sweep.py \
  --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-001000 \
  --device cuda:0 \
  --vjepa-device cuda:1 \
  --initialize-model-on-cpu
```

## 4. Four-Family Frequency-Guidance A/B on test_5

Unified A/B driver:

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_model_weight_ab_test5.py`

Main result directory:

- `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705`

Important subdirectories:

- `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/wan22_official_ti2v5b`
- `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/wan22_early_lora_step000500`
- `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step002500`
- `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/train0705_step007000`
- `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/scores`
- `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/ab_report`
- `/data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705/ab_dashboard`

Current intended family set:

- `wan22_official_ti2v5b`
- `wan22_early_lora_step000500`
- `train0705_step002500`
- `train0705_step007000`

Current guided mode:

- motion mask default: `temporal_union_except_first`
- spectral weighting: `temporal_lowpass_residual`
- `lowpass_ratio = 0.18`
- `spectral_weight_floor = 0.25`
- `spectral_weight_scale = 1.0`
- `spectral_mask_dilation = 5`

Generate command:

```bash
CUDA_VISIBLE_DEVICES=5,6,7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_model_weight_ab_test5.py \
  --stage generate
```

Score command:

```bash
CUDA_VISIBLE_DEVICES=7 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_model_weight_ab_test5.py \
  --stage score
```

## 5. Wan2.1 T2V 1.3B

Primary batch script:

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan21_t2v_1_3b_batch.py`

Underlying single-case script:

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/archive/2026-07-cleanup/wan21_t2v_1_3b_vjepa.py`

Known result directories:

- `/data/gaoya/AAA_test_video/0623/test/v2v_1p3b/train_stage1b_diffsynth_native0706_wan21_13b`
- `/data/gaoya/AAA_test_video/0623/test/v2v/train_stage1b_diffsynth_native0706_wan21_13b_smoke`

Baseline rerun command:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
CUDA_VISIBLE_DEVICES=5 \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan21_t2v_1_3b_batch.py \
  --input-list /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --output-root /data/gaoya/agent-data/outputs/wan21_test5/baseline \
  --model-name wan21_t2v_1p3b_baseline \
  --device-id 0 \
  --disable-vjepa-guidance
```

Guided rerun command:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
CUDA_VISIBLE_DEVICES=5,6 \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wan21_t2v_1_3b_batch.py \
  --input-list /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
  --output-root /data/gaoya/agent-data/outputs/wan21_test5/guided \
  --model-name wan21_t2v_1p3b_guided \
  --device-id 0 \
  --vjepa-device-id 1 \
  --vjepa-guidance-steps 2 \
  --vjepa-min-step-percent 0.35 \
  --vjepa-max-step-percent 0.65 \
  --vjepa-latent-step-size 0.02 \
  --preview-downsample-factor 4 \
  --preview-frame-stride 2 \
  --window-size 8 \
  --context-frames 4 \
  --stride 2
```

## 6. LoRA Phase-4 Multi-Case Pilot

Primary script:

- `/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_phase4_multicase.py`

Main result directory:

- `/data/gaoya/agent-data/outputs/vjepa_phase4_multicase`

Representative rerun command:

```bash
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
CUDA_VISIBLE_DEVICES=7,6 \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_phase4_multicase.py \
  --weights-root /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500 \
  --input-json-list-path /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/phase4_pilot3_cases.txt \
  --model-name-prefix phase4_pilot3 \
  --device cuda:0 \
  --vjepa-device cuda:1
```

## 7. Single-Case Trace / Sweep Utilities

Useful result directories already generated:

- Step-index sweep:
  - `/data/gaoya/agent-data/outputs/vjepa_stepindex_sweep`
- Timestep sweep:
  - `/data/gaoya/agent-data/outputs/vjepa_timestep_sweep_1460`
- Motion-mask `temporal_union_except_first` single case:
  - `/data/gaoya/agent-data/outputs/vjepa_mask_union_except_first_case`

These are not separate model families; they are analysis runs on top of the
Wan2.2 official TI2V-5B pipeline.
