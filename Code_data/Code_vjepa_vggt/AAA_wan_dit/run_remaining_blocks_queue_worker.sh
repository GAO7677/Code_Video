#!/usr/bin/env bash
set -uo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 GPU_ID KIND WORKER_NAME RUN_ROOT INPUT_ALLOWLIST" >&2
  exit 2
fi

GPU_ID="$1"
KIND="$2"
WORKER_NAME="$3"
RUN_ROOT="$4"
INPUT_ALLOWLIST="$5"
PYTHON_BIN=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
BENCH_PY=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/bench.py
QUEUE="${RUN_ROOT}/queues/${KIND}.tsv"
CURSOR="${RUN_ROOT}/queues/${KIND}.cursor"
LOCK="${RUN_ROOT}/queues/${KIND}.lock"
LOG="${RUN_ROOT}/logs/${WORKER_NAME}.log"
STATE_DIR="${RUN_ROOT}/state"
SUMMARY_DIR="${RUN_ROOT}/task_summaries"
AUX_BASE=/data/gaoya/agent-data/cache/wan_dit_remaining_blocks_metrics

mkdir -p "$(dirname -- "${LOG}")" "${STATE_DIR}" "${SUMMARY_DIR}" "${AUX_BASE}"
exec > >(tee -a "${LOG}") 2>&1

claim_task() {
  local line_number task
  exec 9>"${LOCK}"
  flock 9
  line_number="$(<"${CURSOR}")"
  task="$(sed -n "${line_number}p" "${QUEUE}")"
  if [[ -n "${task}" ]]; then
    printf '%s\n' "$((line_number + 1))" > "${CURSOR}"
  fi
  flock -u 9
  exec 9>&-
  printf '%s' "${task}"
}

echo "[queue-worker] start worker=${WORKER_NAME} kind=${KIND} gpu=${GPU_ID}"
num_done=0
num_failed=0
while true; do
  task="$(claim_task)"
  [[ -z "${task}" ]] && break

  IFS=$'\t' read -r task_id metric result_root <<< "${task}"
  summary_path="${SUMMARY_DIR}/${task_id}.json"
  aux_root="${AUX_BASE}/${task_id}"
  echo "[queue-worker] task=${task_id} metric=${metric} root=${result_root}"

  extra_args=(
    --physics-iq-output-root "${aux_root}/physics_iq"
    --physics-iq-verified-output-root "${aux_root}/physics_iq_verified"
    --pmf-output-root "${aux_root}/pmf"
    --vbench-output-root "${aux_root}/vbench"
  )
  if [[ "${metric}" == "wmreward" ]]; then
    extra_args+=(--wmreward-reset-interval 1000000)
  fi

  set +e
  TOKENIZERS_PARALLELISM=false \
  CUDA_VISIBLE_DEVICES="${GPU_ID}" \
  PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/Code_data/Code_try0526 \
  "${PYTHON_BIN}" "${BENCH_PY}" \
    --metric "${metric}" \
    --result-root "${result_root}" \
    --input-json-allowlist "${INPUT_ALLOWLIST}" \
    --output-summary "${summary_path}" \
    "${extra_args[@]}"
  status=$?
  if [[ "${status}" -eq 0 ]]; then
    "${PYTHON_BIN}" -c '
import json
import sys
payload = json.load(open(sys.argv[1], encoding="utf-8"))
status = payload.get("metric_status", {})
expected = int(sys.argv[2])
ok = (
    int(status.get("num_cases", -1)) == expected
    and int(status.get("num_success", -1)) == expected
    and int(status.get("num_failed", -1)) == 0
)
raise SystemExit(0 if ok else 3)
' "${summary_path}" 67
    status=$?
  fi
  set -e

  if [[ "${status}" -eq 0 ]]; then
    num_done=$((num_done + 1))
    printf '%s\t%s\t%s\t%s\n' \
      "${task_id}" "${metric}" "${result_root}" "${WORKER_NAME}" \
      >> "${RUN_ROOT}/completed_tasks.tsv"
  else
    num_failed=$((num_failed + 1))
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "${task_id}" "${metric}" "${result_root}" "${WORKER_NAME}" "${status}" \
      >> "${RUN_ROOT}/failed_tasks.tsv"
  fi
done

printf 'worker=%s\nkind=%s\ngpu=%s\ndone=%s\nfailed=%s\nfinished_utc=%s\n' \
  "${WORKER_NAME}" "${KIND}" "${GPU_ID}" "${num_done}" "${num_failed}" \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" \
  > "${STATE_DIR}/${WORKER_NAME}.complete"
echo "[queue-worker] finish worker=${WORKER_NAME} done=${num_done} failed=${num_failed}"
