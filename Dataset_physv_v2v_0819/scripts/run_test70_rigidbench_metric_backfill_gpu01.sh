#!/usr/bin/env bash
set -euo pipefail

RUNNER="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_metric_backfill.sh"
BUILDER="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/build_test70_rigidbench_metrics.py"
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/sam/bin/python}"
LOG_ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70/logs/metric_backfill_gpu01"
export PYTHONNOUSERSITE=1
mkdir -p "$LOG_ROOT"

SAFE_TASKS=(
  full_sa_physrvg_dit_gpu56__step-000500
  full_sa_physrvg_latent_mask_loss__step-000500
  full_sa_physrvg_latent_mask_loss__step-001000
  full_sa_physrvg_latent_mask_loss__step-001500
  full_sa_physrvg_no_vjepa_0717_b2g2__step-000500
  full_sa_physrvg_object_xssc_loss__step-000500
  full_sa_physrvg_vjepa_loss_0613_b2g2__step-000500
  full_sa_physrvg_vjepa_rect384x672_0717_w0p3_b4gacc1__step-000500
  full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_enabled__step-000500
  full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_enabled__step-001000
  full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_enabled__step-003000
  full_sa_physrvg_vjepa_utonia_scene_hardmask_v1_enabled__step-003500
)

mapfile -t CASE_IDS < <(
  for metadata in /data/gaoya/AAA_test_video/physv_v2v_0819_strict/truth/cases/*/rigidbench/metadata.json; do
    basename "$(dirname "$(dirname "$metadata")")"
  done | sort
)

TASK_ARGS=()
for task in "${SAFE_TASKS[@]}"; do
  TASK_ARGS+=(--task-id "$task")
done

CASE_ARGS_0=()
CASE_ARGS_1=()
CASE_ARGS_2=()
CASE_ARGS_3=()
for index in "${!CASE_IDS[@]}"; do
  case $((index % 4)) in
    0) CASE_ARGS_0+=(--case-id "${CASE_IDS[$index]}") ;;
    1) CASE_ARGS_1+=(--case-id "${CASE_IDS[$index]}") ;;
    2) CASE_ARGS_2+=(--case-id "${CASE_IDS[$index]}") ;;
    3) CASE_ARGS_3+=(--case-id "${CASE_IDS[$index]}") ;;
  esac
done

run_worker() {
  local gpu="$1"
  local slot="$2"
  shift 2
  local log="$LOG_ROOT/gpu${gpu}_slot${slot}.log"
  {
    echo "[launcher] physical_gpu=$gpu slot=$slot started $(date -Is)"
    CUDA_VISIBLE_DEVICES="$gpu" "$RUNNER" --no-build "${TASK_ARGS[@]}" "$@"
    echo "[launcher] physical_gpu=$gpu slot=$slot finished $(date -Is)"
  } >"$log" 2>&1
}

run_worker 0 0 "${CASE_ARGS_0[@]}" & pid0=$!
run_worker 0 1 "${CASE_ARGS_1[@]}" & pid1=$!
run_worker 1 0 "${CASE_ARGS_2[@]}" & pid2=$!
run_worker 1 1 "${CASE_ARGS_3[@]}" & pid3=$!

status=0
wait "$pid0" || status=1
wait "$pid1" || status=1
wait "$pid2" || status=1
wait "$pid3" || status=1

if [[ "$status" -eq 0 ]]; then
  "$PYTHON" "$BUILDER"
fi
exit "$status"
