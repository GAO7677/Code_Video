#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
TRAIN_CONFIG="${ROOT}/configs/formal_full_sa_physrvg_dit_gpu12.json"
WATCH_CONFIG="${ROOT}/xssc_lora_three_train_watch_config_with_t_head.json"
RUN_TAG="${RUN_TAG:-formal_20260808T142258Z_physrvg_dit}"
SESSION="${SESSION:-xssc_full_sa_physrvg_dit_gpu12}"
METHOD=full_sa_physrvg_dit
WATCH_GPU=3
LOG_ROOT=/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_control/physrvg_dit_gpu12

if tmux has-session -t "${SESSION}" 2>/dev/null; then
  echo "tmux session already exists: ${SESSION}" >&2
  exit 1
fi
if [[ ! -s /data/gaoya/ckpt/HappyP4nda-PhysRVG/dit/diffusion_pytorch_model.safetensors ]]; then
  echo "PhysRVG DiT checkpoint is missing" >&2
  exit 2
fi

mkdir -p "${LOG_ROOT}"

tmux new-session -d -s "${SESSION}" -n train -c "${ROOT}"
train_command="set -o pipefail; bash ${ROOT}/run_train_from_config.sh ${TRAIN_CONFIG} --run-tag ${RUN_TAG} 2>&1 | tee ${LOG_ROOT}/train_${RUN_TAG}.log; status=\${PIPESTATUS[0]}; echo TRAIN_EXIT=\${status}; exec bash"
tmux send-keys -t "${SESSION}:train" "${train_command}" C-m

add_window() {
  local name="$1"
  local command="$2"
  tmux new-window -d -t "${SESSION}" -n "${name}" -c "${ROOT}"
  tmux send-keys -t "${SESSION}:${name}" "${command}; status=\$?; echo ${name^^}_EXIT=\${status}; exec bash" C-m
}

common="--config ${WATCH_CONFIG} --methods ${METHOD} --gpus ${WATCH_GPU}"
add_window test5_infer "env PYTHONNOUSERSITE=1 ${PYTHON} ${ROOT}/xssc_lora_checkpoint_watch.py ${common} --mode inference"
add_window test5_cpu "env PYTHONNOUSERSITE=1 ${PYTHON} ${ROOT}/xssc_lora_checkpoint_watch.py ${common} --mode metrics --kind cpu"
add_window test5_gpu "env PYTHONNOUSERSITE=1 ${PYTHON} ${ROOT}/xssc_lora_checkpoint_watch.py ${common} --mode metrics --kind gpu"
add_window phys_infer "env PYTHONNOUSERSITE=1 ${PYTHON} ${ROOT}/xssc_lora_physiciq_watch.py ${common} --mode inference"
add_window phys_cpu "env PYTHONNOUSERSITE=1 ${PYTHON} ${ROOT}/xssc_lora_physiciq_watch.py ${common} --mode metrics --kind cpu"
add_window phys_gpu "env PYTHONNOUSERSITE=1 ${PYTHON} ${ROOT}/xssc_lora_physiciq_watch.py ${common} --mode metrics --kind gpu"
add_window dashboard "env PYTHONNOUSERSITE=1 ${PYTHON} ${ROOT}/xssc_lora_dashboard_refresh_loop.py --config ${WATCH_CONFIG} --interval 60"

tmux select-window -t "${SESSION}:train"
echo "tmux session: ${SESSION}"
echo "training GPUs: 1,2"
echo "watcher GPU: ${WATCH_GPU}"
echo "run tag: ${RUN_TAG}"
echo "attach: tmux attach -t ${SESSION}"
