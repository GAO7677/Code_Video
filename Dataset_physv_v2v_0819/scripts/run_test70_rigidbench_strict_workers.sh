#!/usr/bin/env bash
set -uo pipefail

# One RigidBench evaluator per GPU, with serial model loading within each GPU.
# GPU4 is intentionally excluded by the workspace policy.
GPUS_CSV="${RIGIDBENCH_GPUS:-0,1,2,3,5}"
IFS=',' read -r -a GPUS <<< "$GPUS_CSV"
PYTHON="/home/gaoya/miniconda3/envs/sam/bin/python"
SCRIPT="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/run_test70_rigidbench_strict.py"
BUILDER="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/scripts/build_test70_rigidbench_metrics.py"
DASHBOARD="/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/physv-v2v-0819-test70-no-event-timing-40step/dashboard.json"
LOG_ROOT="/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70/logs"
PYTHONPATH_VALUE="/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/src:/home/gaoya/Code_Video/Dataset_physv_v2v_0819/RigidBench/vendor/Video-Depth-Anything"
mkdir -p "$LOG_ROOT"

mapfile -t TASKS < <(/usr/bin/python3 - "$DASHBOARD" <<'PY'
import json, sys
d=json.load(open(sys.argv[1]))
for m in d.get('models',[]):
    if m.get('status') == 'complete' and int(m.get('generated_cases',0)) == int(m.get('total_cases',70)):
        print(m['task_id'])
PY
)

if ((${#TASKS[@]} == 0)); then
  echo "No complete test70 tasks found in $DASHBOARD" >&2
  exit 2
fi

echo "Starting ${#TASKS[@]} tasks on GPUs: ${GPUS[*]}"
printf '%s\n' "${TASKS[@]}" > "$LOG_ROOT/task_queue.txt"

worker() {
  local gpu="$1"
  local worker_id="$2"
  local index="$worker_id"
  local task
  while ((index < ${#TASKS[@]})); do
    task="${TASKS[$index]}"
    local log="$LOG_ROOT/gpu${gpu}_${task}.log"
    {
      echo "[$(date -u +%FT%TZ)] START gpu=$gpu task=$task"
      CUDA_VISIBLE_DEVICES="$gpu" PYTHONNOUSERSITE=1 PYTHONPATH="$PYTHONPATH_VALUE" \
        "$PYTHON" "$SCRIPT" --task-id "$task"
      rc=$?
      echo "[$(date -u +%FT%TZ)] END gpu=$gpu task=$task rc=$rc"
      /usr/bin/python3 "$BUILDER" || true
      echo "[$(date -u +%FT%TZ)] SNAPSHOT_UPDATED task=$task"
    } >> "$log" 2>&1
    index=$((index + ${#GPUS[@]}))
  done
}

pids=()
for i in "${!GPUS[@]}"; do
  worker "${GPUS[$i]}" "$i" &
  pids+=("$!")
done

rc=0
for pid in "${pids[@]}"; do
  wait "$pid" || rc=1
done
/usr/bin/python3 "$BUILDER" || rc=1
echo "[$(date -u +%FT%TZ)] ALL_WORKERS_DONE rc=$rc"
exit "$rc"
