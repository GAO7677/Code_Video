# DiffTrack Run Commands

This file records the shell commands used under `BBB_my_test`.

## Shared Setup

- Repo: `/home/gaoya/Code_Video/DiffTrack-main`
- Model: `/data/gaoya/ckpt/zai-org-CogVideoX-5b`
- Conda env: `bagel`
- Scripts:
  - `BBB_my_test/run_motion_guidance_5b_local.sh`
  - `BBB_my_test/run_motion_guidance_5b_baseline_same_cases.sh`
  - `BBB_my_test/make_pag_baseline_side_by_side.py`

## 1. Official CAG Prompts

Prompt file:

```text
/home/gaoya/Code_Video/DiffTrack-main/dataset/cag_prompts.txt
```

Outputs:

- PAG: `/data/gaoya/agent-data/outputs/difftrack_motion_guidance_5b`
- Baseline: `/data/gaoya/agent-data/outputs/difftrack_motion_guidance_5b_baseline`
- Compare: `/data/gaoya/agent-data/outputs/difftrack_motion_guidance_5b_compare`

Command:

```bash
set -euo pipefail
REPO_DIR=/home/gaoya/Code_Video/DiffTrack-main
PROMPT_PATH=$REPO_DIR/dataset/cag_prompts.txt
MODEL_DIR=/data/gaoya/ckpt/zai-org-CogVideoX-5b
PAG_OUT=/data/gaoya/agent-data/outputs/difftrack_motion_guidance_5b
BASE_OUT=/data/gaoya/agent-data/outputs/difftrack_motion_guidance_5b_baseline
COMPARE_OUT=/data/gaoya/agent-data/outputs/difftrack_motion_guidance_5b_compare
mkdir -p "$PAG_OUT" "$BASE_OUT" "$COMPARE_OUT"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/diffusers/src:$REPO_DIR"
CUDA_VISIBLE_DEVICES=4 PROMPT_PATH="$PROMPT_PATH" OUTPUT_DIR="$PAG_OUT" MODEL_DIR="$MODEL_DIR" CONDA_ENV=bagel bash "$REPO_DIR/BBB_my_test/run_motion_guidance_5b_local.sh"
CUDA_VISIBLE_DEVICES=5 PROMPT_PATH="$PROMPT_PATH" OUTPUT_DIR="$BASE_OUT" PAG_OUTPUT_DIR="$PAG_OUT" MODEL_DIR="$MODEL_DIR" CONDA_ENV=bagel bash "$REPO_DIR/BBB_my_test/run_motion_guidance_5b_baseline_same_cases.sh"
/home/gaoya/miniconda3/envs/bagel/bin/python "$REPO_DIR/BBB_my_test/make_pag_baseline_side_by_side.py" --pag_dir "$PAG_OUT" --baseline_dir "$BASE_OUT" --output_dir "$COMPARE_OUT"
```

## 2. Rigid-Body Prompt Batch

Prompt file:

```text
/home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/prompt_rigid_body_cases.txt
```

Outputs:

- PAG: `/data/gaoya/agent-data/outputs/difftrack_rigid_body_g56_pag`
- Baseline: `/data/gaoya/agent-data/outputs/difftrack_rigid_body_g56_baseline`
- Compare: `/data/gaoya/agent-data/outputs/difftrack_rigid_body_g56_compare`

Command:

```bash
set -euo pipefail
REPO_DIR=/home/gaoya/Code_Video/DiffTrack-main
PROMPT_PATH=$REPO_DIR/BBB_my_test/prompt_rigid_body_cases.txt
MODEL_DIR=/data/gaoya/ckpt/zai-org-CogVideoX-5b
PAG_OUT=/data/gaoya/agent-data/outputs/difftrack_rigid_body_g56_pag
BASE_OUT=/data/gaoya/agent-data/outputs/difftrack_rigid_body_g56_baseline
COMPARE_OUT=/data/gaoya/agent-data/outputs/difftrack_rigid_body_g56_compare
mkdir -p "$PAG_OUT" "$BASE_OUT" "$COMPARE_OUT"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/diffusers/src:$REPO_DIR"
CUDA_VISIBLE_DEVICES=5 PROMPT_PATH="$PROMPT_PATH" OUTPUT_DIR="$PAG_OUT" MODEL_DIR="$MODEL_DIR" CONDA_ENV=bagel bash "$REPO_DIR/BBB_my_test/run_motion_guidance_5b_local.sh"
CUDA_VISIBLE_DEVICES=6 PROMPT_PATH="$PROMPT_PATH" OUTPUT_DIR="$BASE_OUT" PAG_OUTPUT_DIR="$PAG_OUT" MODEL_DIR="$MODEL_DIR" CONDA_ENV=bagel bash "$REPO_DIR/BBB_my_test/run_motion_guidance_5b_baseline_same_cases.sh"
/home/gaoya/miniconda3/envs/bagel/bin/python "$REPO_DIR/BBB_my_test/make_pag_baseline_side_by_side.py" --pag_dir "$PAG_OUT" --baseline_dir "$BASE_OUT" --output_dir "$COMPARE_OUT"
```

## 3. Single Prompt: a dog is running

Prompt file:

```text
/home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/prompt_dog_running.txt
```

Outputs:

- PAG: `/data/gaoya/agent-data/outputs/difftrack_dogrun_same_script_pag`
- Baseline: `/data/gaoya/agent-data/outputs/difftrack_dogrun_same_script_baseline`
- Compare: `/data/gaoya/agent-data/outputs/difftrack_dogrun_same_script_compare`

Command:

```bash
set -euo pipefail
REPO_DIR=/home/gaoya/Code_Video/DiffTrack-main
PROMPT_PATH=$REPO_DIR/BBB_my_test/prompt_dog_running.txt
MODEL_DIR=/data/gaoya/ckpt/zai-org-CogVideoX-5b
PAG_OUT=/data/gaoya/agent-data/outputs/difftrack_dogrun_same_script_pag
BASE_OUT=/data/gaoya/agent-data/outputs/difftrack_dogrun_same_script_baseline
COMPARE_OUT=/data/gaoya/agent-data/outputs/difftrack_dogrun_same_script_compare
mkdir -p "$PAG_OUT" "$BASE_OUT" "$COMPARE_OUT"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/diffusers/src:$REPO_DIR"
CUDA_VISIBLE_DEVICES=4 PROMPT_PATH="$PROMPT_PATH" OUTPUT_DIR="$PAG_OUT" MODEL_DIR="$MODEL_DIR" CONDA_ENV=bagel bash "$REPO_DIR/BBB_my_test/run_motion_guidance_5b_local.sh"
CUDA_VISIBLE_DEVICES=5 PROMPT_PATH="$PROMPT_PATH" OUTPUT_DIR="$BASE_OUT" PAG_OUTPUT_DIR="$PAG_OUT" MODEL_DIR="$MODEL_DIR" CONDA_ENV=bagel bash "$REPO_DIR/BBB_my_test/run_motion_guidance_5b_baseline_same_cases.sh"
/home/gaoya/miniconda3/envs/bagel/bin/python "$REPO_DIR/BBB_my_test/make_pag_baseline_side_by_side.py" --pag_dir "$PAG_OUT" --baseline_dir "$BASE_OUT" --output_dir "$COMPARE_OUT"
```

## 4. Single Prompt: A ball falls into the box

Prompt file:

```text
/home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/prompt_ball_falls_box.txt
```

Outputs:

- PAG: `/data/gaoya/agent-data/outputs/difftrack_ballfallsbox_same_script_pag`
- Baseline: `/data/gaoya/agent-data/outputs/difftrack_ballfallsbox_same_script_baseline`
- Compare: `/data/gaoya/agent-data/outputs/difftrack_ballfallsbox_same_script_compare`

Command:

```bash
set -euo pipefail
REPO_DIR=/home/gaoya/Code_Video/DiffTrack-main
PROMPT_PATH=$REPO_DIR/BBB_my_test/prompt_ball_falls_box.txt
MODEL_DIR=/data/gaoya/ckpt/zai-org-CogVideoX-5b
PAG_OUT=/data/gaoya/agent-data/outputs/difftrack_ballfallsbox_same_script_pag
BASE_OUT=/data/gaoya/agent-data/outputs/difftrack_ballfallsbox_same_script_baseline
COMPARE_OUT=/data/gaoya/agent-data/outputs/difftrack_ballfallsbox_same_script_compare
mkdir -p "$PAG_OUT" "$BASE_OUT" "$COMPARE_OUT"
cd "$REPO_DIR"
export PYTHONPATH="$REPO_DIR/diffusers/src:$REPO_DIR"
CUDA_VISIBLE_DEVICES=3 PROMPT_PATH="$PROMPT_PATH" OUTPUT_DIR="$PAG_OUT" MODEL_DIR="$MODEL_DIR" CONDA_ENV=bagel bash "$REPO_DIR/BBB_my_test/run_motion_guidance_5b_local.sh"
CUDA_VISIBLE_DEVICES=4 PROMPT_PATH="$PROMPT_PATH" OUTPUT_DIR="$BASE_OUT" PAG_OUTPUT_DIR="$PAG_OUT" MODEL_DIR="$MODEL_DIR" CONDA_ENV=bagel bash "$REPO_DIR/BBB_my_test/run_motion_guidance_5b_baseline_same_cases.sh"
/home/gaoya/miniconda3/envs/bagel/bin/python "$REPO_DIR/BBB_my_test/make_pag_baseline_side_by_side.py" --pag_dir "$PAG_OUT" --baseline_dir "$BASE_OUT" --output_dir "$COMPARE_OUT"
```

## 5. Five-Way Diagnostic Variant Grid

Used files:

- `BBB_my_test/prompt_rigid_case0_original.txt`
- `BBB_my_test/prompt_rigid_case0_simple.txt`
- `BBB_my_test/make_variant_grid_compare.py`

Outputs:

- `/data/gaoya/agent-data/outputs/difftrack_diag_case0_orig_pag151718`
- `/data/gaoya/agent-data/outputs/difftrack_diag_case0_simple_baseline`
- `/data/gaoya/agent-data/outputs/difftrack_diag_case0_simple_pag151718`
- `/data/gaoya/agent-data/outputs/difftrack_diag_case0_compare/case0_variant_grid.mp4`

Generation commands:

```bash
cd /home/gaoya/Code_Video/DiffTrack-main
export PYTHONPATH=/home/gaoya/Code_Video/DiffTrack-main/diffusers/src:/home/gaoya/Code_Video/DiffTrack-main

CUDA_VISIBLE_DEVICES=4 /home/gaoya/miniconda3/bin/conda run -n bagel python /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/motion_guidance_5b_local.py \
  --output_dir /data/gaoya/agent-data/outputs/difftrack_diag_case0_orig_pag151718 \
  --model_version 5b \
  --model_path /data/gaoya/ckpt/zai-org-CogVideoX-5b \
  --txt_path /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/prompt_rigid_case0_original.txt \
  --pag_layers 15 17 18 \
  --pag_scale 1 \
  --cfg_scale 6 \
  --device cuda:0 \
  --max_prompts 1

CUDA_VISIBLE_DEVICES=5 /home/gaoya/miniconda3/bin/conda run -n bagel python /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/motion_guidance_5b_local.py \
  --output_dir /data/gaoya/agent-data/outputs/difftrack_diag_case0_simple_baseline \
  --model_version 5b \
  --model_path /data/gaoya/ckpt/zai-org-CogVideoX-5b \
  --txt_path /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/prompt_rigid_case0_simple.txt \
  --pag_layers 15 17 18 \
  --pag_scale 0 \
  --cfg_scale 6 \
  --device cuda:0 \
  --max_prompts 1

CUDA_VISIBLE_DEVICES=6 /home/gaoya/miniconda3/bin/conda run -n bagel python /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/motion_guidance_5b_local.py \
  --output_dir /data/gaoya/agent-data/outputs/difftrack_diag_case0_simple_pag151718 \
  --model_version 5b \
  --model_path /data/gaoya/ckpt/zai-org-CogVideoX-5b \
  --txt_path /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/prompt_rigid_case0_simple.txt \
  --pag_layers 15 17 18 \
  --pag_scale 1 \
  --cfg_scale 6 \
  --device cuda:0 \
  --max_prompts 1
```

Grid compare command:

```bash
/home/gaoya/miniconda3/envs/bagel/bin/python /home/gaoya/Code_Video/DiffTrack-main/BBB_my_test/make_variant_grid_compare.py \
  --input orig_pag1321=/data/gaoya/agent-data/outputs/difftrack_rigid_body_g56_pag/video_0.mp4 \
  --input orig_base=/data/gaoya/agent-data/outputs/difftrack_rigid_body_g56_baseline/video_0.mp4 \
  --input orig_pag151718=/data/gaoya/agent-data/outputs/difftrack_diag_case0_orig_pag151718/video_0.mp4 \
  --input simple_base=/data/gaoya/agent-data/outputs/difftrack_diag_case0_simple_baseline/video_0.mp4 \
  --input simple_pag151718=/data/gaoya/agent-data/outputs/difftrack_diag_case0_simple_pag151718/video_0.mp4 \
  --output /data/gaoya/agent-data/outputs/difftrack_diag_case0_compare/case0_variant_grid.mp4 \
  --cols 3
```
