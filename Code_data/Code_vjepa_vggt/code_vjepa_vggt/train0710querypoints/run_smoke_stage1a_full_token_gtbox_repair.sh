#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-5}"
CONFIG="${CONFIG:-/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0710querypoints/config_stage1a_full_token_gtbox_repair_smoke.yaml}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-none}"
INIT_FROM="${INIT_FROM:-none}"

if [ "${GPU}" = "4" ]; then
  echo "ERROR: gpu4 故障, 禁止使用。" >&2
  exit 1
fi

source /home/gaoya/miniconda3/etc/profile.d/conda.sh
conda activate wan-cu128

cd /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
export CUDA_VISIBLE_DEVICES="${GPU}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CMD=(
  /home/gaoya/miniconda3/envs/wan-cu128/bin/python
  -m code_vjepa_vggt.object_token_teacher_student.train_stage1a_full_token
  --config "${CONFIG}"
)

if [ "${RESUME_CHECKPOINT}" != "none" ]; then
  CMD+=(--resume-checkpoint "${RESUME_CHECKPOINT}")
fi
if [ "${INIT_FROM}" != "none" ]; then
  CMD+=(--init-from "${INIT_FROM}")
fi

echo "[stage1a-smoke] GPU=${GPU}"
echo "[stage1a-smoke] CONFIG=${CONFIG}"
echo "[stage1a-smoke] CMD=${CMD[*]}"
exec "${CMD[@]}"
