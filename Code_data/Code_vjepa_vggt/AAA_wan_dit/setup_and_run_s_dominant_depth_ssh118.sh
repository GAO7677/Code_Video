#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/setup_and_run_s_dominant_depth_ssh118.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
HOST=118
REMOTE_DATA=/mnt/data/gaoya
REMOTE_AGENT="${REMOTE_DATA}/agent-data"
REMOTE_INPUT="${REMOTE_AGENT}/ssh118_s_dominant_depth/input"
REMOTE_OUTPUT="${REMOTE_AGENT}/outputs/wan_dit_s_dominant_depth"
LOCAL_STAGE=/data/gaoya/agent-data/ssh118_staging/s_dominant_depth/input
LOCAL_CONFIG="${SCRIPT_DIR}/head_role_s_dominant_depth_experiment.json"
LOCAL_ROOT=/data/gaoya/agent-data/outputs/wan_dit_s_dominant_depth/seed851

mkdir -p "${LOCAL_STAGE}"
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  "${SCRIPT_DIR}/stage_s_dominant_depth_ssh118_input.py" \
  --input-list /data/gaoya/agent-data/outputs/wan_dit_fulltoken_head_roles_50seeds/input_lists/test5_unique20.txt \
  --output-root "${LOCAL_STAGE}" \
  --remote-root "${REMOTE_INPUT}"

ssh "${HOST}" "mkdir -p \
  '${REMOTE_AGENT}/envs/wan-cu128' \
  '${REMOTE_AGENT}/envs/vjepa2' \
  '${REMOTE_INPUT}' \
  '${REMOTE_OUTPUT}/configs' \
  '/mnt/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis' \
  '/mnt/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/train_xssc_context_slots/checkpoints/step-001500' \
  '/home/gaoya/Code_Video/Code_data/Code_train/train_0419' \
  '/home/gaoya/Code_Video/xSSC-main' \
  '/home/gaoya/code_my_utils' \
  '/home/gaoya/Code_Video/co-tracker-main' \
  '/home/gaoya/Code_Video/vjepa2-main' \
  '/home/gaoya/Code_Video/DreamWorld-main/extract/VGGT'"

echo "[ssh118-setup] syncing wan-cu128 environment"
rsync -a --partial --info=progress2 \
  /home/gaoya/miniconda3/envs/wan-cu128/ \
  "${HOST}:${REMOTE_AGENT}/envs/wan-cu128/"
echo "[ssh118-setup] syncing PhysRVG environment"
rsync -a --partial --info=progress2 \
  /data/gaoya/miniconda3/envs/vjepa2/ \
  "${HOST}:${REMOTE_AGENT}/envs/vjepa2/"

echo "[ssh118-setup] syncing code and inputs"
rsync -a --partial "${SCRIPT_DIR}/" \
  "${HOST}:${PROJECT_ROOT}/AAA_wan_dit/"
rsync -a --partial "${PROJECT_ROOT}/code_vjepa_vggt/" \
  "${HOST}:${PROJECT_ROOT}/code_vjepa_vggt/"
rsync -a --partial "${PROJECT_ROOT}/code_vjepa_free/" \
  "${HOST}:${PROJECT_ROOT}/code_vjepa_free/"
rsync -a --partial /home/gaoya/Code_Video/Code_data/Code_train/train_0419/ \
  "${HOST}:/home/gaoya/Code_Video/Code_data/Code_train/train_0419/"
rsync -a --partial /home/gaoya/Code_Video/xSSC-main/ \
  "${HOST}:/home/gaoya/Code_Video/xSSC-main/"
rsync -a --partial /home/gaoya/code_my_utils/ \
  "${HOST}:/home/gaoya/code_my_utils/"
rsync -a --partial /home/gaoya/Code_Video/co-tracker-main/ \
  "${HOST}:/home/gaoya/Code_Video/co-tracker-main/"
rsync -a --partial /home/gaoya/Code_Video/vjepa2-main/ \
  "${HOST}:/home/gaoya/Code_Video/vjepa2-main/"
rsync -a --partial /home/gaoya/Code_Video/DreamWorld-main/extract/VGGT/ \
  "${HOST}:/home/gaoya/Code_Video/DreamWorld-main/extract/VGGT/"
rsync -a --partial /home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main/ \
  "${HOST}:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main/"
rsync -a --partial \
  "${PROJECT_ROOT}/code_phys_papers_compare/PhysRVG-main/" \
  "${HOST}:${PROJECT_ROOT}/code_phys_papers_compare/PhysRVG-main/"
rsync -a --partial "${LOCAL_STAGE}/" "${HOST}:${REMOTE_INPUT}/"

echo "[ssh118-setup] syncing missing lightweight weights and frozen manifest"
rsync -a --partial \
  /data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/42-0130.pth \
  "${HOST}:/mnt/data/gaoya/ckpt/xSSC/rsfq2_r-ytvis/"
rsync -a --partial \
  /data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/train_xssc_context_slots/checkpoints/step-001500/ \
  "${HOST}:/mnt/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/train_xssc_context_slots/checkpoints/step-001500/"
rsync -a --partial \
  /data/gaoya/agent-data/outputs/wan_dit_s_dominant_depth/configs/s_dominant_depth_subsets.json \
  "${HOST}:${REMOTE_OUTPUT}/configs/"

echo "[ssh118-setup] validating remote environments and frozen task matrix"
ssh "${HOST}" "set -euo pipefail
  /mnt/data/gaoya/agent-data/envs/wan-cu128/bin/python -c 'import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda)'
  PYTHONPATH='${PROJECT_ROOT}/code_phys_papers_compare/PhysRVG-main:${PROJECT_ROOT}/AAA_wan_dit' \
    /mnt/data/gaoya/agent-data/envs/vjepa2/bin/python -c 'from diffusers.models.attention import AttentionMixin; print(\"PhysRVG import OK\")'
  /mnt/data/gaoya/agent-data/envs/wan-cu128/bin/python \
    '${PROJECT_ROOT}/AAA_wan_dit/run_head_role_dose_control_pilot_worker.py' \
    --config '${PROJECT_ROOT}/AAA_wan_dit/head_role_s_dominant_depth_experiment_ssh118.json' \
    --runner '${PROJECT_ROOT}/AAA_wan_dit/run_matched_head_subset_ablation_job_ssh118.sh' \
    --preflight"

if tmux list-sessions -F '#S' 2>/dev/null | rg -Fxq wan_s_dominant_depth; then
  tmux kill-session -t '=wan_s_dominant_depth'
fi
LOCAL_ROOT="${LOCAL_ROOT}" /home/gaoya/miniconda3/envs/wan-cu128/bin/python - <<'PY'
import os
from pathlib import Path

for path in (Path(os.environ["LOCAL_ROOT"]) / "claims").glob("*.json"):
    path.unlink()
PY

echo "[ssh118-setup] launching remote workers"
ssh "${HOST}" "bash '${PROJECT_ROOT}/AAA_wan_dit/run_s_dominant_depth_ssh118_remote.sh'"

if ! tmux has-session -t wan_s_dominant_depth_ssh118_pull 2>/dev/null; then
  tmux new-session -d -s wan_s_dominant_depth_ssh118_pull -n pull \
    "/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
    '${SCRIPT_DIR}/pull_s_dominant_depth_ssh118.py' \
    --host '${HOST}' \
    --remote-root '/mnt/data/gaoya/agent-data/outputs/wan_dit_s_dominant_depth/seed851' \
    --local-config '${LOCAL_CONFIG}'; exec bash"
  tmux new-window -d -t wan_s_dominant_depth_ssh118_pull -n gallery \
    "/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
    '${SCRIPT_DIR}/watch_s_dominant_depth_gallery.py' \
    --config '${LOCAL_CONFIG}'; exec bash"
fi

echo "[ssh118-setup] remote generation and local validated pull are running"
