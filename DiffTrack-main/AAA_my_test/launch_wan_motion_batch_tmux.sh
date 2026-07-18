#!/usr/bin/env bash
set -euo pipefail

repo_root="/home/gaoya/Code_Video/DiffTrack-main"
python_bin="/home/gaoya/miniconda3/envs/wan-cu128/bin/python"
output_root="/data/gaoya/agent-data/outputs/wan22_motion_qk"
tracks_dir="${output_root}/tracks_base"
batch_dir="${output_root}/batch_base"
session="wan_motion_batch"
workers="${WAN_WORKERS:-7}"
gpu_list="${WAN_GPUS:-0 1 2 3 4 5 6}"

gate="${output_root}/single_case/case_019_wheel_hits_block_base/gate_report.json"
if [[ ! -f "${gate}" ]]; then
  echo "Missing single-case gate: ${gate}" >&2
  exit 2
fi
"${python_bin}" - "${gate}" <<'PY'
import json, sys
report = json.load(open(sys.argv[1]))
if not report.get("passed"):
    raise SystemExit("Single-case gate did not pass; batch launch is forbidden")
PY

mkdir -p "${output_root}" "${tracks_dir}" "${batch_dir}"
export HF_HOME="/data/gaoya/agent-data/cache/huggingface"
export WAN_CACHE="/data/gaoya/agent-data/cache/wan"

if tmux has-session -t "${session}" 2>/dev/null; then
  echo "tmux session ${session} already exists" >&2
  exit 2
fi

# CoTracker preparation is compact and runs once before parallel model workers.
tmux new-session -d -s "${session}" -n prepare_tracks -c "${repo_root}"
prepare_cmd="${python_bin} -u AAA_my_test/prepare_wan_region_tracks.py --sample-types base --output-dir ${tracks_dir} --device cuda:0 2>&1 | tee ${output_root}/logs_prepare_batch_tracks.log"
tmux send-keys -t "${session}:prepare_tracks" "export HF_HOME=${HF_HOME} WAN_CACHE=${WAN_CACHE}; ${prepare_cmd}" C-m

cat >"${output_root}/launch_batch_workers_after_tracks.sh" <<EOF
#!/usr/bin/env bash
set -euo pipefail
while [[ \$(find "${tracks_dir}" -maxdepth 1 -name 'case_*_base.npz' | wc -l) -lt 50 ]]; do sleep 30; done
EOF

read -r -a gpus <<<"${gpu_list}"
for ((worker=0; worker<workers; worker++)); do
  gpu="${gpus[$((worker % ${#gpus[@]}))]}"
  start=$(( (50 * worker) / workers ))
  end=$(( (50 * (worker + 1)) / workers ))
  worker_dir="${batch_dir}/worker_${worker}"
  log="${output_root}/logs_batch_worker_${worker}.log"
  cat >>"${output_root}/launch_batch_workers_after_tracks.sh" <<EOF
tmux new-window -t "${session}" -n "worker_${worker}" -c "${repo_root}"
tmux send-keys -t "${session}:worker_${worker}" "export HF_HOME=${HF_HOME} WAN_CACHE=${WAN_CACHE}; ${python_bin} -u AAA_my_test/run_wan_motion_scan.py --tracks-dir ${tracks_dir} --output-dir ${worker_dir} --device cuda:${gpu} --start ${start} --end ${end} 2>&1 | tee ${log}" C-m
EOF
done
chmod +x "${output_root}/launch_batch_workers_after_tracks.sh"

tmux new-window -t "${session}" -n launch_workers -c "${repo_root}"
tmux send-keys -t "${session}:launch_workers" "${output_root}/launch_batch_workers_after_tracks.sh" C-m

echo "Started ${session}; attach with: tmux attach -t ${session}"
