#!/usr/bin/env bash
set -euo pipefail

# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/AAA_wan_dit/setup_and_run_openvid_lora_head34_ssh118.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt
HOST=118
LOCAL_PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
LOCAL_MANIFEST=/data/gaoya/agent-data/outputs/wan_dit_openvid_lora_head34/configs/openvid_lora_head34_subsets.json
LOCAL_CONFIG="${SCRIPT_DIR}/head_role_openvid_lora_head34_experiment.json"
LOCAL_ROOT=/data/gaoya/agent-data/outputs/wan_dit_openvid_lora_head34/seed851
LOCAL_CHECKPOINT=/data/gaoya/AAA_test_video/Train_test/DiffSynth_wan22_ti2v5B/openvid_mixed_ctx24_384x672_lora/checkpoints/step-010000/checkpoint.safetensors
REMOTE_AGENT=/mnt/data/gaoya/agent-data
REMOTE_ROOT="${REMOTE_AGENT}/outputs/wan_dit_openvid_lora_head34/seed851"
REMOTE_CONFIG="${PROJECT_ROOT}/AAA_wan_dit/head_role_openvid_lora_head34_experiment_ssh118.json"
REMOTE_MANIFEST="${REMOTE_AGENT}/outputs/wan_dit_openvid_lora_head34/configs/openvid_lora_head34_subsets.json"
REMOTE_CHECKPOINT_DIR="${REMOTE_AGENT}/weights/wan_openvid_lora_step10000"
REMOTE_INPUT="${REMOTE_AGENT}/ssh118_s_dominant_depth/input/test5_unique20.txt"
REMOTE_SMOKE_INPUT="${REMOTE_AGENT}/outputs/wan_dit_openvid_lora_head34/smoke/one_case.txt"
REMOTE_SMOKE_ROOT="${REMOTE_AGENT}/outputs/wan_dit_openvid_lora_head34/smoke"

"${LOCAL_PYTHON}" "${SCRIPT_DIR}/build_openvid_lora_head34_manifest.py" \
  --matched-manifest /data/gaoya/agent-data/outputs/wan_dit_s_feature_phased/configs/s_feature_phased_subsets.json \
  --dominance-manifest /data/gaoya/agent-data/outputs/wan_dit_s_dominant_depth/configs/s_dominant_depth_subsets.json \
  --output "${LOCAL_MANIFEST}"

ssh "${HOST}" "mkdir -p \
  '${REMOTE_CHECKPOINT_DIR}' \
  '$(dirname "${REMOTE_MANIFEST}")' \
  '$(dirname "${REMOTE_SMOKE_INPUT}")' \
  '${REMOTE_ROOT}/state' \
  '${REMOTE_ROOT}/logs'"

echo "[openvid-head34] syncing code, manifest, and OpenVid LoRA"
rsync -a --partial "${SCRIPT_DIR}/" "${HOST}:${PROJECT_ROOT}/AAA_wan_dit/"
rsync -a --partial "${PROJECT_ROOT}/code_vjepa_vggt/" \
  "${HOST}:${PROJECT_ROOT}/code_vjepa_vggt/"
rsync -a --partial "${LOCAL_MANIFEST}" "${HOST}:${REMOTE_MANIFEST}"
rsync -a --partial --info=progress2 "${LOCAL_CHECKPOINT}" \
  "${HOST}:${REMOTE_CHECKPOINT_DIR}/checkpoint.safetensors"

ssh "${HOST}" "set -euo pipefail
  test \"\$(sha256sum '${REMOTE_CHECKPOINT_DIR}/checkpoint.safetensors' | cut -d' ' -f1)\" = \
    763a1b00ad370b1af7aeb53304b79d01c53bd8588390e09f2f374dc83f2e54ae
  head -n 1 '${REMOTE_INPUT}' > '${REMOTE_SMOKE_INPUT}'
  /mnt/data/gaoya/agent-data/envs/wan-cu128/bin/python \
    '${PROJECT_ROOT}/AAA_wan_dit/run_head_role_dose_control_pilot_worker.py' \
    --config '${REMOTE_CONFIG}' \
    --runner '${PROJECT_ROOT}/AAA_wan_dit/run_openvid_lora_matched_head_job_ssh118.sh' \
    --preflight"

echo "[openvid-head34] smoke: baseline"
ssh "${HOST}" "env GPU=6 SEED=851 \
  INPUT_LIST='${REMOTE_SMOKE_INPUT}' \
  JOB_ROOT='${REMOTE_SMOKE_ROOT}/baseline' \
  bash '${PROJECT_ROOT}/AAA_wan_dit/run_openvid_lora_baseline_job_ssh118.sh'"

echo "[openvid-head34] smoke: S_local_k32 [0,10)"
ssh "${HOST}" "env MODEL=openvid_lora_step10000 SEED=851 \
  SUBSET_ID=S_local_k32_r00_exactblock GPU=6 \
  STEP_START=0 STEP_END=10 INPUT_LIST='${REMOTE_SMOKE_INPUT}' \
  OUTPUT_ROOT='${REMOTE_SMOKE_ROOT}/grouped' MANIFEST='${REMOTE_MANIFEST}' \
  bash '${PROJECT_ROOT}/AAA_wan_dit/run_openvid_lora_matched_head_job_ssh118.sh'"

ssh "${HOST}" "PYTHONPATH='${PROJECT_ROOT}/AAA_wan_dit' \
  /mnt/data/gaoya/agent-data/envs/wan-cu128/bin/python - <<'PY'
import hashlib
from pathlib import Path
from run_common22_public_head_ablation_worker import _input_cases
from run_head_role_dose_control_pilot_worker import _validate_job
from run_openvid_lora_baseline_worker import _validate

cases = _input_cases(Path('${REMOTE_SMOKE_INPUT}'))
baseline = _validate(Path('${REMOTE_SMOKE_ROOT}/baseline'), cases)
manifest = Path('${REMOTE_MANIFEST}')
grouped = _validate_job(
    Path('${REMOTE_SMOKE_ROOT}/grouped/generation/openvid_lora_step10000/seed-000851/S_local_k32_r00_exactblock_steps00_10'),
    cases=cases,
    subset_id='S_local_k32_r00_exactblock',
    manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest(),
    k=32,
    start=0,
    end=10,
)
print({'baseline_videos': len(baseline), 'grouped_videos': len(grouped)})
PY"

echo "[openvid-head34] launching formal 34-config run on SSH118 GPU6/7"
ssh "${HOST}" "bash '${PROJECT_ROOT}/AAA_wan_dit/run_openvid_lora_head34_ssh118_remote.sh'"

if ! tmux has-session -t wan_openvid_lora_head34_pull 2>/dev/null; then
  tmux new-session -d -s wan_openvid_lora_head34_pull -n pull \
    "${LOCAL_PYTHON} '${SCRIPT_DIR}/pull_s_dominant_depth_ssh118.py' \
      --host '${HOST}' --remote-root '${REMOTE_ROOT}' \
      --local-config '${LOCAL_CONFIG}' && \
     mkdir -p '${LOCAL_ROOT}/generation/openvid_lora_step10000/seed-000851/baseline' \
       '${LOCAL_ROOT}/baseline_state' && \
     rsync -a --partial \
       '${HOST}:${REMOTE_ROOT}/generation/openvid_lora_step10000/seed-000851/baseline/' \
       '${LOCAL_ROOT}/generation/openvid_lora_step10000/seed-000851/baseline/' && \
     rsync -a --partial '${HOST}:${REMOTE_ROOT}/baseline_state/' \
       '${LOCAL_ROOT}/baseline_state/'; exec bash"
fi

echo "[openvid-head34] remote generation and local validated pull are running"
