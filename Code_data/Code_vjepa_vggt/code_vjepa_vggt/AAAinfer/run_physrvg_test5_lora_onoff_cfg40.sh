#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
  echo "Usage: $0 DEVICE {full|dit} [MAX_CASES]" >&2
  exit 2
fi

DEVICE="$1"
VARIANT="$2"
MAX_CASES="${3:-}"
if [[ "${DEVICE}" == "4" ]]; then
  echo "GPU 4 is prohibited" >&2
  exit 2
fi
if [[ "${VARIANT}" != "full" && "${VARIANT}" != "dit" ]]; then
  echo "VARIANT must be full or dit" >&2
  exit 2
fi

REPO=/home/gaoya/code_V2V_baselines/PhysRVG-main
PYTHON=/home/gaoya/data/miniconda3/envs/wan22-physicsiq/bin/python
INPUT=/home/gaoya/data/agent-data/cache/physrvg-test5-lora-ablation-20260808
REMOTE_ROOT=/home/gaoya/data/agent-data/outputs/physrvg-test5-lora-onoff-cfg40-20260808
LOCAL_ROOT=/data/gaoya/agent-data/outputs/physrvg_test5_lora_onoff_cfg40_20260808
MODEL=/mnt/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B-Diffusers
DIT=/home/gaoya/data/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors
LORA=/home/gaoya/data/ckpt/HappyP4nda-PhysRVG/lora/checkpoint
NEGATIVE_PROMPT='模糊，低质量，变形，伪影，文字，水印，过曝，欠曝，颜色异常，几何扭曲，物体融化，物理不合理'

if [[ "${VARIANT}" == "full" ]]; then
  NAME=physRVG_test5_LoRA_ON_steps40_512x896_08_49f
else
  NAME=physRVG_test5_LoRA_OFF_steps40_512x896_08_49f
fi

mkdir -p "${REMOTE_ROOT}/${NAME}"
EXTRA_ARGS=()
if [[ -n "${MAX_CASES}" ]]; then
  EXTRA_ARGS+=(--max-cases "${MAX_CASES}")
fi

exec env PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
  "${PYTHON}" "${REPO}/tools_run_v2v_json_ablation.py" \
  --input-dir "${INPUT}" \
  --output-dir "${REMOTE_ROOT}/${NAME}" \
  --metadata-output-root "${LOCAL_ROOT}/${NAME}" \
  --model-id "${MODEL}" \
  --dit-checkpoint "${DIT}" \
  --lora-checkpoint "${LORA}" \
  --model-variant "${VARIANT}" \
  --negative-prompt "${NEGATIVE_PROMPT}" \
  --device "${DEVICE}" \
  --height 512 --width 896 \
  --context-frames 8 --num-frames 49 \
  --num-inference-steps 40 --guidance-scale 5.0 \
  --fps 30 --seed 42 --do-cfg \
  "${EXTRA_ARGS[@]}"
