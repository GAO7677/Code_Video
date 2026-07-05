# Wan2.1-1.3B Training Plan

## Canonical Maintenance Root

All future training and inference edits for this 1.3B Wan flow should be made under `code_vjepa_vggt/train0706_wan1p3b/`. Treat the older `train0705` tree as a historical snapshot only.
The files whose names still contain `0705` are kept only for compatibility with the existing call sites; the maintained source of truth is this directory.

## 1. Base LoRA pretraining

Train the OpenVid + MOVI-D + Genesis rigid mixed recipe first.

Script:

```bash
CUDA_VISIBLE_DEVICES=3,5,6,7 sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_train_openvid_mixed_ctx24_384x672_lora_wan21_13b_gpu0235.sh
```

Expected output:

`/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints_wan21_13b/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors`

Smoke:

```bash
GPU=3 sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_smoke_openvid_mixed_ctx24_384x672_lora_wan21_13b.sh
```

Expected smoke output:

`/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints_wan21_13b/smoke/openvid_mixed_ctx24_384x672_lora/checkpoints/step-000002/checkpoint.safetensors`

## 2. Stage 0 phys-state continuation

Continue training from the base LoRA above.

Script:

```bash
CUDA_VISIBLE_DEVICES=3,5,6,7 sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_train_phys_state_lora_continue_wan21_13b.sh
```

Smoke:

```bash
GPU=3 sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_smoke_phys_state_lora_continue_wan21_13b.sh
```

Expected output:

`/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints_wan21_13b/raw_phys_state_wan21_13b_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors`

## 3. Stage 1a

Use the stage 0 step-000500 checkpoint as the initialization for teacher-student stage 1a.

Script:

```bash
sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_train_teacher_student_stage1a_wan21_13b_gpu0235.sh
```

Expected output:

`/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/pybullet0629_teacher_student/stage1a_full_token/step_0003000.pt`

## 4. Stage 1b

Continue from the stage 0 and stage 1a checkpoints.

Script:

```bash
CUDA_VISIBLE_DEVICES=3,5,6,7 sh /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0706_wan1p3b/run_train_stage1b_context_only_no_gt_box_v_newtrain0706_wan21_13b_gpu0235.sh
```
