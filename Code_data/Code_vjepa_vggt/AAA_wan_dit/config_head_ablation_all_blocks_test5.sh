#!/usr/bin/env bash

# Edit this file for future single-head DiT ablation sweeps.

# Experiment identity and data.
SESSION="wan_head_ablation_all_blocks_test5_gpu56"
SOURCE_LIST="/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"
OUTPUT_BASE="/data/gaoya/AAA_test_video/0623/test/v2v_wan_test5"
RUN_ROOT="${OUTPUT_BASE}/_pipeline"
EXPECTED_CASES=20
DEDUPLICATE_INPUTS=1

# One independent run is generated for every model/block/head tuple.
MODELS="wan_lora xssc physrvg"
BLOCKS="0-29"
HEADS="0-23"
GPUS="5 6"
GEN_WORKERS_PER_GPU=1
GEN_GPU_MAX_USED_MIB=2048
GPU_WAIT_SECONDS=60
COORDINATOR_POLL_SECONDS=30

# Shared inference settings.
HEIGHT=512
WIDTH=896
NUM_FRAMES=49
CONTEXT_FRAMES=8
NUM_INFERENCE_STEPS=40
CFG_SCALE=5.0
GUIDANCE_SCALE=5.0
PHYSRVG_DO_CFG=0
FPS=30
SEED=42
NEGATIVE_PROMPT="模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理"

# Wan+LoRA and xSSC checkpoints.
WAN_ROOT="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"
WAN_LORA_ROOT="/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500"
XSSC_WEIGHTS_ROOT="/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/train_xssc_context_slots/checkpoints/step-001500"
XSSC_ROOT="/home/gaoya/Code_Video/xSSC-main"
XSSC_CONFIG="${XSSC_ROOT}/config-randsfq/rsfq2_r-ytvis.py"
XSSC_CHECKPOINT="/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth"

# Official PhysRVG checkpoints. CFG remains disabled to match official inference.
PHYSRVG_ROOT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/PhysRVG-main"
PHYSRVG_MODEL_ID="/data/gaoya/ckpt/HappyP4nda-PhysRVG/Wan2.2-TI2V-5B-Diffusers"
PHYSRVG_DIT_CHECKPOINT="/data/gaoya/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors"
PHYSRVG_LORA_CHECKPOINT="/data/gaoya/ckpt/HappyP4nda-PhysRVG/lora/checkpoint"

# Full benchmark set.
CPU_METRICS="physics_iq_with_context physics_iq_without_context pmf_with_context pmf_without_context"
GPU_COMMON_METRICS="wmreward vbench_subject_consistency vbench_background_consistency vbench_temporal_flickering vbench_motion_smoothness vbench_dynamic_degree vbench_aesthetic_quality vbench_imaging_quality"
VIDEOPHY2_METRICS="videophy2"
COSMOS_METRICS="cosmos_reason1"

# Metric workers start after all generation configurations validate.
CPU_WORKERS_PER_GPU=8
GPU_COMMON_WORKERS_PER_GPU=3
VIDEOPHY2_WORKERS_PER_GPU=2
COSMOS_WORKERS_PER_GPU=1

# Set RESUME=1 to rebuild the deterministic queue and skip validated task states.
RESUME=0
DRY_RUN=0
