#!/usr/bin/env python3
"""
Run command:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/DiffSynth-Studio-main:/home/gaoya/Code_Video/Code_data/Code_train/train_0419 \
CUDA_VISIBLE_DEVICES=6,7 /data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/wait_for_phase7_gpus.py \
  --gpu-indices 6 7 \
  --max-memory-mib 8000 \
  --max-utilization 20 \
  --stable-polls 3 \
  --poll-seconds 30 \
  --runner-args --weights-root /data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500 --input-json /data/gaoya/AAA_test_video/0623/testdataset/025_Solid_Mechanics_0002_perspective-center_trimmed/025_Solid_Mechanics_0002_perspective-center_trimmed.json --source-video /data/gaoya/AAA_test_video/0623/testdataset/025_Solid_Mechanics_0002_perspective-center_trimmed/physicIQ_0002_clip_2p5s_3p5s.mp4 --device cuda:0 --vjepa-device cuda:1
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
RUNNER = THIS_DIR / "run_phase7_target_shape.py"
DEFAULT_PYTHON_BIN = Path("/data/gaoya/miniconda3/envs/wan/bin/python")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Foreground helper: wait until the requested physical GPUs look idle, "
            "then launch run_phase7_target_shape.py."
        )
    )
    parser.add_argument("--gpu-indices", type=int, nargs="+", required=True)
    parser.add_argument("--max-memory-mib", type=int, default=8000)
    parser.add_argument("--max-utilization", type=int, default=20)
    parser.add_argument("--stable-polls", type=int, default=3)
    parser.add_argument("--poll-seconds", type=int, default=30)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument(
        "--runner-args",
        nargs=argparse.REMAINDER,
        default=[],
        help="Arguments passed through to run_phase7_target_shape.py after the wait completes.",
    )
    return parser.parse_args()


def query_gpus() -> list[dict[str, int]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used,utilization.gpu",
        "--format=csv,noheader,nounits",
    ]
    result = subprocess.run(cmd, check=True, capture_output=True, text=True)
    rows: list[dict[str, int]] = []
    for line in result.stdout.strip().splitlines():
        if not line.strip():
            continue
        idx_s, mem_s, util_s = [part.strip() for part in line.split(",")]
        rows.append(
            {
                "index": int(idx_s),
                "memory_used_mib": int(mem_s),
                "utilization_gpu": int(util_s),
            }
        )
    return rows


def pick_gpu_rows(all_rows: list[dict[str, int]], gpu_indices: list[int]) -> list[dict[str, int]]:
    row_map = {row["index"]: row for row in all_rows}
    missing = [idx for idx in gpu_indices if idx not in row_map]
    if missing:
        raise RuntimeError(f"Missing GPU indices from nvidia-smi output: {missing}")
    return [row_map[idx] for idx in gpu_indices]


def is_idle(rows: list[dict[str, int]], *, max_memory_mib: int, max_utilization: int) -> bool:
    return all(
        row["memory_used_mib"] <= max_memory_mib and row["utilization_gpu"] <= max_utilization
        for row in rows
    )


def main() -> None:
    args = parse_args()
    stable_count = 0

    while stable_count < args.stable_polls:
        rows = pick_gpu_rows(query_gpus(), list(args.gpu_indices))
        idle = is_idle(
            rows,
            max_memory_mib=int(args.max_memory_mib),
            max_utilization=int(args.max_utilization),
        )
        print(
            json.dumps(
                {
                    "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "gpu_rows": rows,
                    "idle": idle,
                    "stable_count": stable_count,
                    "required_stable_polls": int(args.stable_polls),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        if idle:
            stable_count += 1
        else:
            stable_count = 0
        if stable_count < args.stable_polls:
            time.sleep(max(1, int(args.poll_seconds)))

    env = os.environ.copy()
    cmd = [str(args.python_bin), str(RUNNER), *args.runner_args]
    print("[RUN]", subprocess.list2cmdline(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
