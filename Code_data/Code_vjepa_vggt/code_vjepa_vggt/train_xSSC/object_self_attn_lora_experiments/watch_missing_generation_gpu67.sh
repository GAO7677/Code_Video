#!/usr/bin/env bash
# Run:
# bash /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train_xSSC/object_self_attn_lora_experiments/watch_missing_generation_gpu67.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
CONFIG="${SCRIPT_DIR}/xssc_lora_three_train_watch_config_with_t_head.json"
STATE=/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_three_run_watch/state
PHYS_ROOT=/data/gaoya/AAA_test_video/0623/test/v2v_wan/xssc
SESSION=xssc_fill_kubric_then_all_phys_gpu67

phys_manifests=(
  "${STATE}/physiciq/inference/full_sa_no_object_kubric100/step-000500.json"
  "${STATE}/physiciq/inference/full_sa_no_object_kubric100/step-001000.json"
  "${STATE}/physiciq/inference/t_head70_slot_dedup_merge_xssc_step050000/step-001000.json"
)

all_phys_done() {
  local path
  for path in "${phys_manifests[@]}"; do
    [[ -f "${path}" ]] || return 1
  done
}

count_mp4() {
  local path="$1"
  find "${path}" -maxdepth 1 -type f -name '*.mp4' 2>/dev/null | wc -l
}

restart_phys_queue() {
  tmux respawn-pane -k -t "${SESSION}:ensure_phys" \
    "cd ${SCRIPT_DIR} && env PYTHONNOUSERSITE=1 ${PYTHON} ./run_missing_checkpoint_generation.py --config ${CONFIG} --gpus 6,7 --methods full_sa_no_object_kubric100,t_head70_slot_dedup_merge_xssc_step050000 --steps 500,1000 --physiciq-only; exec bash"
}

while ! all_phys_done; do
  printf '[%s] K500=%s K1000=%s T1000=%s\n' \
    "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
    "$(count_mp4 "${PHYS_ROOT}/xssc_lora_full_sa_no_object_kubric100_step-000500_steps40_512x896_ctx08_49f_customprompt")" \
    "$(count_mp4 "${PHYS_ROOT}/xssc_lora_full_sa_no_object_kubric100_step-001000_steps40_512x896_ctx08_49f_customprompt")" \
    "$(count_mp4 "${PHYS_ROOT}/xssc_lora_t_head70_slot_dedup_merge_xssc_step050000_step-001000_steps40_512x896_ctx08_49f_customprompt")"
  if ! pgrep -f 'xssc_lora_physiciq_parallel_infer.py --config' >/dev/null; then
    echo '[watchdog] PhysicIQ queue is absent; restarting missing tasks.'
    restart_phys_queue
    sleep 30
  fi
  sleep 60
done

"${PYTHON}" "${SCRIPT_DIR}/build_xssc_lora_checkpoint_dashboard.py" --config "${CONFIG}"
echo '[watchdog] all requested generation manifests are complete.'
