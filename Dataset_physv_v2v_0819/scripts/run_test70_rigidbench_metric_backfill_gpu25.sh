#!/usr/bin/env bash
set -euo pipefail

RUNNER="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_metric_backfill.sh"
BUILDER="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/build_test70_rigidbench_metrics.py"
PYTHON="${PYTHON:-/home/gaoya/miniconda3/envs/sam/bin/python}"
LOG_ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70/logs/metric_backfill_gpu2"
export PYTHONNOUSERSITE=1
mkdir -p "$LOG_ROOT"

# Tasks that are not currently being generated.  The runner itself still
# checks every case's generated frames and strict GT prerequisites.
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

TASK_ARGS=()
for task in "${SAFE_TASKS[@]}"; do
  TASK_ARGS+=(--task-id "$task")
done

run_metric() {
  local gpu="$1"
  local metric="$2"
  local log="$LOG_ROOT/gpu${gpu}_${metric}.log"
  {
    echo "[launcher] physical_gpu=$gpu metric=$metric started $(date -Is)"
    CUDA_VISIBLE_DEVICES="$gpu" "$RUNNER" --no-build "${TASK_ARGS[@]}" --metric "$metric"
    echo "[launcher] physical_gpu=$gpu metric=$metric finished $(date -Is)"
  } >"$log" 2>&1
}

# Exactly one process per metric; each process loads its prediction models once.
# Keep all metric workers on GPU 2; GPU 5 is deliberately unused here.
run_metric 2 iou & pid_iou=$!
run_metric 2 ate & pid_ate=$!
run_metric 2 lpips & pid_lpips=$!
run_metric 2 ate3d & pid_ate3d=$!
run_metric 2 bgdrift & pid_bgdrift=$!
run_metric 2 l2 & pid_l2=$!
run_metric 2 chamfer & pid_chamfer=$!
run_metric 2 si_mse & pid_si_mse=$!
run_metric 2 ssim & pid_ssim=$!
run_metric 2 iddrift & pid_iddrift=$!

status=0
for pid in "$pid_iou" "$pid_ate" "$pid_lpips" "$pid_ate3d" "$pid_bgdrift" \
           "$pid_l2" "$pid_chamfer" "$pid_si_mse" "$pid_ssim" "$pid_iddrift"; do
  wait "$pid" || status=1
done

if [[ "$status" -eq 0 ]]; then
  "$PYTHON" "$BUILDER"
fi
exit "$status"
