#!/usr/bin/env bash
set -u

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <tmux-train-target> <output-dir> <visible-gpu-ids>" >&2
  exit 2
fi

TRAIN_TARGET="$1"
OUTPUT_DIR="$2"
VISIBLE_GPU_IDS="$3"
POLL_SECONDS="${POLL_SECONDS:-60}"
STALL_POLLS_MAX="${STALL_POLLS_MAX:-10}"
STATUS_LOG="${OUTPUT_DIR}/monitor_status.log"
ALERT_LOG="${OUTPUT_DIR}/monitor_alerts.log"

mkdir -p "${OUTPUT_DIR}"
printf '%s monitor_started target=%s poll_seconds=%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${TRAIN_TARGET}" "${POLL_SECONDS}" >>"${STATUS_LOG}"

last_step=-1
stall_polls=0

while true; do
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if ! tmux list-panes -t "${TRAIN_TARGET}" >/dev/null 2>&1; then
    printf '%s FATAL training_pane_missing target=%s\n' "${now}" "${TRAIN_TARGET}" | tee -a "${ALERT_LOG}" >>"${STATUS_LOG}"
    exit 1
  fi

  train_log="$(find "${OUTPUT_DIR}" -maxdepth 1 -type f -name 'train_*.log' -print 2>/dev/null | sort | tail -1)"
  latest_metric=""
  step=0
  if [ -n "${train_log}" ]; then
    latest_metric="$(rg '\[object-reg\] step=' "${train_log}" 2>/dev/null | tail -1)"
    if [ -n "${latest_metric}" ]; then
      step="$(sed -n 's/.*\[object-reg\] step=\([0-9][0-9]*\).*/\1/p' <<<"${latest_metric}")"
    fi
    if rg -q 'Traceback|CUDA out of memory|ChildFailedError|Training failed|(^|[^[:alpha:]])NaN([^[:alpha:]]|$)' "${train_log}"; then
      printf '%s FATAL fatal_log_marker step=%s log=%s\n' "${now}" "${step}" "${train_log}" | tee -a "${ALERT_LOG}" >>"${STATUS_LOG}"
      exit 1
    fi
  fi

  if [ "${step}" -eq "${last_step}" ]; then
    stall_polls=$((stall_polls + 1))
  else
    stall_polls=0
  fi
  last_step="${step}"

  gpu_state="$(nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | \
    awk -F, -v ids="${VISIBLE_GPU_IDS}" '
      BEGIN { n=split(ids, wanted, ","); for (i=1; i<=n; i++) keep[wanted[i]]=1 }
      { gsub(/ /, ""); if ($1 in keep) printf "%s:%sMiB/%s%% ", $1, $2, $3 }
    ')"
  printf '%s step=%s stall_polls=%s gpu="%s" metric="%s"\n' \
    "${now}" "${step}" "${stall_polls}" "${gpu_state}" "${latest_metric}" >>"${STATUS_LOG}"

  if [ "${stall_polls}" -ge "${STALL_POLLS_MAX}" ]; then
    printf '%s WARN no_step_progress step=%s polls=%s\n' "${now}" "${step}" "${stall_polls}" >>"${ALERT_LOG}"
    stall_polls=0
  fi

  if [ -n "${latest_metric}" ]; then
    awk -v now="${now}" -v line="${latest_metric}" -v alerts="${ALERT_LOG}" '
      BEGIN {
        n=split(line, fields, " ")
        for (i=1; i<=n; i++) {
          split(fields[i], pair, "=")
          if (pair[1] == "adapter_mlp_ratio") {
            split(pair[2], ratios, "/")
            value["adapter_mlp_ratio_mean"]=ratios[1]+0
            value["adapter_mlp_ratio_max"]=ratios[2]+0
          } else if (length(pair[2])) {
            value[pair[1]]=pair[2]+0
          }
        }
        if (value["adapter_mlp_cap"] > 0)
          printf "%s WARN adapter_mlp_cap_applied step=%d fraction=%.6f\n", now, value["step"], value["adapter_mlp_cap"] >> alerts
        if (value["guard_layers"] > 0)
          printf "%s WARN object_branch_guard_applied step=%d layers=%d pre_guard_ratio=%.6f\n", now, value["step"], value["guard_layers"], value["pre_guard_max_ratio"] >> alerts
        if (value["adapter_mlp_ratio_max"] > 2.5)
          printf "%s WARN adapter_mlp_max_above_target step=%d ratio=%.6f\n", now, value["step"], value["adapter_mlp_ratio_max"] >> alerts
      }
    '
  fi

  sleep "${POLL_SECONDS}"
done
