#!/usr/bin/env bash
# Usage: bash run_v2v_cka_queue.sh <task_list.json> <gpu_id>
TASK_LIST=${1:?need task_list}
GPU=${2:?need gpu_id}
PYTHON=/home/gaoya/miniconda3/envs/wan-cu128/bin/python
SCRIPT=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_exp/stream_cka.py
LOG_ROOT=/data/gaoya/AAA_test_video/0626vjepa_free/GT_check/0623test_100/logs

echo "[queue] task_list=$TASK_LIST  gpu=$GPU"

CUDA_VISIBLE_DEVICES=$GPU \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main \
$PYTHON - "$TASK_LIST" "$LOG_ROOT" "$SCRIPT" "$PYTHON" << 'PYEOF'
import json, subprocess, sys, os
from pathlib import Path

task_list_path, log_root_str, script, python = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
task_list = json.load(open(task_list_path))
log_root = Path(log_root_str)
env = {**os.environ}

for i, task in enumerate(task_list):
    manifest = task['manifest']
    case_dir = task['case_dir']
    n        = task['n']
    label    = Path(case_dir).relative_to(
                   '/data/gaoya/AAA_test_video/0626vjepa_free/GT_check/0623test_100')
    log_file = log_root / str(label).replace('/', '__') / 'run.log'
    log_file.parent.mkdir(parents=True, exist_ok=True)

    done = len(list(Path(case_dir).glob('*.npy'))) if Path(case_dir).exists() else 0
    if done >= n:
        print(f"[{i+1}/{len(task_list)}] SKIP {label}  ({done}/{n})", flush=True)
        continue

    print(f"[{i+1}/{len(task_list)}] START {label}  ({done}/{n})", flush=True)
    Path(case_dir).mkdir(parents=True, exist_ok=True)

    cmd = [python, script,
           '--manifest', manifest,
           '--out-dir', str(Path(case_dir).parent),
           '--case-dir', case_dir,
           '--device', 'cuda']

    with open(log_file, 'a') as lf:
        ret = subprocess.run(cmd, env=env, stdout=lf, stderr=subprocess.STDOUT)

    done_after = len(list(Path(case_dir).glob('*.npy')))
    status = 'OK' if ret.returncode == 0 else f'ERR(rc={ret.returncode})'
    print(f"[{i+1}/{len(task_list)}] {status} {label}  npy={done_after}/{n}", flush=True)

print("[queue] done")
PYEOF
