#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/data/gaoya/home_miniconda3/envs/wan-cu128/bin/python}"
PROJECT_ROOT="${PROJECT_ROOT:-/home/gaoya/Code_Video/phys_state_video}"
WAN_REPO_ROOT="${WAN_REPO_ROOT:-/home/gaoya/Code_Video/Wan2.2-main}"
WAN_CKPT_DIR="${WAN_CKPT_DIR:-/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B}"
EPISODE_ROOT="${EPISODE_ROOT:-/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/episodes_v1/industrial_s1_scale2_256x144_s8_f16_n6}"
BASELINE_CKPT="${BASELINE_CKPT:-/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v2/industrial_s1_scale2_wan_state_v2_ti2vprefix_gpu0123_20260605/checkpoints/predictor_v2_last.joint_finetune.best.pt}"
ABLATION_ROOT="${ABLATION_ROOT:-/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/runs_v2/continuity_ablation_20260605}"
VIS_ROOT="${VIS_ROOT:-/data/gaoya/AAA_test_video/Dataset_physV/phys_state_0601/vis_v2/predictor_continuity_ablation_20260605}"
PORT="${PORT:-18876}"
DEVICE="${DEVICE:-cuda}"
MAX_CASES="${MAX_CASES:-8}"
FPS="${FPS:-6}"
EXPORT_CUDA_VISIBLE_DEVICES="${EXPORT_CUDA_VISIBLE_DEVICES:-${CUDA_VISIBLE_DEVICES:-4}}"
SERVER_PID_FILE="${SERVER_PID_FILE:-$VIS_ROOT/http_${PORT}.pid}"
SERVER_LOG_FILE="${SERVER_LOG_FILE:-$VIS_ROOT/http_${PORT}.log}"

declare -a SCALE_SPECS=(
  "scale0p0_gpu0_20260605_131328|scale0p0|control"
  "scale0p1_gpu1_20260605_131328|scale0p1|boundary0.1"
  "scale0p5_gpu2_20260605_131328|scale0p5|boundary0.5"
  "scale1p0_gpu3_20260605_131328|scale1p0|boundary1.0"
)

wait_for_checkpoint() {
  local ckpt_path="$1"
  while [[ ! -f "$ckpt_path" ]]; do
    echo "[wait] $(date '+%F %T') missing $ckpt_path"
    sleep 60
  done
}

mkdir -p "$VIS_ROOT"

REPORTS=()
for spec in "${SCALE_SPECS[@]}"; do
  IFS="|" read -r run_dir slug label <<<"$spec"
  ckpt_path="$ABLATION_ROOT/$run_dir/checkpoints/predictor_v2_continuity.joint_finetune.best.pt"
  wait_for_checkpoint "$ckpt_path"
  out_dir="$VIS_ROOT/$slug"
  echo "[export] $(date '+%F %T') $label -> $out_dir"
  CUDA_VISIBLE_DEVICES="$EXPORT_CUDA_VISIBLE_DEVICES" "$PYTHON_BIN" \
    "$PROJECT_ROOT/scripts/export_wan_state_v2_predictor_overlay_comparison.py" \
    --episode-root "$EPISODE_ROOT" \
    --predictor-a "$BASELINE_CKPT" \
    --predictor-b "$ckpt_path" \
    --label-a baseline \
    --label-b "$label" \
    --wan-ckpt-dir "$WAN_CKPT_DIR" \
    --output-dir "$out_dir" \
    --splits val test \
    --max-cases "$MAX_CASES" \
    --fps "$FPS" \
    --port 0 \
    --device "$DEVICE" \
    --wan-repo-root "$WAN_REPO_ROOT" \
    --wan-task ti2v-5B \
    --no-serve
  REPORTS+=("$out_dir/report.json")
done

"$PYTHON_BIN" \
  "$PROJECT_ROOT/scripts/generate_predictor_ablation_index.py" \
  --output-dir "$VIS_ROOT" \
  --reports "${REPORTS[@]}"

if [[ -f "$SERVER_PID_FILE" ]]; then
  existing_pid="$(cat "$SERVER_PID_FILE" || true)"
  if [[ -n "$existing_pid" ]] && kill -0 "$existing_pid" 2>/dev/null; then
    echo "[serve] existing server pid=$existing_pid port=$PORT"
    echo "http://127.0.0.1:$PORT"
    exit 0
  fi
fi

(
  cd "$VIS_ROOT"
  nohup python3 -m http.server "$PORT" --bind 127.0.0.1 >"$SERVER_LOG_FILE" 2>&1 &
  echo $! >"$SERVER_PID_FILE"
)

echo "[serve] $(date '+%F %T') http://127.0.0.1:$PORT"
