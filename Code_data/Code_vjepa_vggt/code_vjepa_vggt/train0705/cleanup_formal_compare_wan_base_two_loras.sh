#!/usr/bin/env bash
set -euo pipefail

SCRIPT="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/run_formal_compare_morpheus_physiciq_wan_base_two_loras_ti2v_t2v_gpu023.sh"
ENTRY="/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/AAAinfer/wan_base_two_loras_ti2v_t2v.py"
BENCH_PAT="bench.sh /data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_morpheus_real_world"

tmux send-keys -t infer:10 C-c || true
sleep 1

for pat in "${SCRIPT}" "${ENTRY}" "${BENCH_PAT}"; do
  mapfile -t pids < <(ps -eo pid,args --no-headers | grep -F "${pat}" | grep -v grep | awk '{print $1}')
  if ((${#pids[@]} > 0)); then
    echo "[cleanup] pattern=${pat}"
    printf '  %s\n' "${pids[@]}"
    kill -9 "${pids[@]}" || true
  fi
done

sleep 1
ps -eo pid,args --no-headers \
  | grep -E 'run_formal_compare_morpheus_physiciq_wan_base_two_loras_ti2v_t2v_gpu023.sh|wan_base_two_loras_ti2v_t2v.py|bench.sh /data/gaoya/AAA_test_video/0623/test/ti2v/train_stage1b_diffsynth_native0705_ctx1dupjepa_0705_morpheus_real_world' \
  | grep -v grep || true
