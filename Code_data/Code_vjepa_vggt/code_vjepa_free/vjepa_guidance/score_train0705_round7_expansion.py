#!/usr/bin/env python3
from __future__ import annotations

"""
Run command:
CUDA_VISIBLE_DEVICES=6 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_try0526 \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_train0705_round7_expansion.py
"""

import argparse
import os
import subprocess
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().with_name("score_multicase_methods.py")
DEFAULT_PYTHON_BIN = Path("/data/gaoya/miniconda3/envs/wan/bin/python")
DEFAULT_PYTHONPATH = "/home/gaoya/Code_Video/Code_data/Code_try0526"
DEFAULT_ROUND2_BASE = Path("/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round2_test5")
DEFAULT_ROUND7_BASE = Path("/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round7_test5_expansion")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score the round7 17-case expansion against round2 baseline / ladder / knee references."
    )
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--pythonpath", default=DEFAULT_PYTHONPATH)
    parser.add_argument("--round2-base", type=Path, default=DEFAULT_ROUND2_BASE)
    parser.add_argument("--round7-base", type=Path, default=DEFAULT_ROUND7_BASE)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_ROUND7_BASE / "round7_test5_scores.json",
    )
    parser.add_argument("--videophy2-device", default="cuda:0")
    parser.add_argument("--videophy2-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--videophy2-num-frames", type=int, default=32)
    parser.add_argument(
        "--candidate-label",
        action="append",
        default=[],
        help=(
            "Candidate labels to include from round7-base. Defaults to both "
            "target_w24_s15_ratio_0035 and target_w24_s15_ratio_0025."
        ),
    )
    parser.add_argument(
        "--skip-missing-candidates",
        action="store_true",
        help="Skip missing round7 candidate dirs instead of exiting.",
    )
    parser.add_argument("--skip-wmreward", action="store_true")
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    round2_base = args.round2_base.expanduser().resolve()
    round7_base = args.round7_base.expanduser().resolve()
    requested_candidates = args.candidate_label or [
        "target_w24_s15_ratio_0035",
        "target_w24_s15_ratio_0025",
    ]
    method_dirs = [
        ("baseline", round2_base / "train0705_round2_test5_baseline" / "step-001000"),
        ("ladder_s20", round2_base / "train0705_round2_test5_ladder_s20" / "step-001000"),
        ("knee_mid_s18", round2_base / "train0705_round2_test5_knee_mid_s18" / "step-001000"),
    ]
    for label in requested_candidates:
        method_dirs.append(
            (
                str(label),
                round7_base / f"train0705_round7_test5_{label}" / "step-001000",
            )
        )

    filtered_method_dirs: list[tuple[str, Path]] = []
    for label, path in method_dirs:
        is_reference = label in {"baseline", "ladder_s20", "knee_mid_s18"}
        if path.is_dir():
            filtered_method_dirs.append((label, path))
            continue
        if args.skip_missing_candidates and not is_reference:
            print(f"[skip] missing candidate {label}: {path}", flush=True)
            continue
        if not path.is_dir():
            raise SystemExit(f"Required method directory not found for {label}: {path}")
    method_dirs = filtered_method_dirs

    cmd = [str(args.python_bin), str(SCRIPT_PATH)]
    for label, path in method_dirs:
        cmd.extend(["--method-dir", f"{label}={path}"])
    cmd.extend(
        [
            "--out-json",
            str(args.out_json.expanduser().resolve()),
            "--physics-iq",
            "--videophy2-task",
            "pc",
            "--videophy2-device",
            str(args.videophy2_device),
            "--videophy2-dtype",
            str(args.videophy2_dtype),
            "--videophy2-num-frames",
            str(args.videophy2_num_frames),
            "--cosmos-reason1",
            "--save-every",
            str(args.save_every),
        ]
    )
    if args.skip_wmreward:
        cmd.append("--skip-wmreward")
    if args.dry_run:
        print("PYTHONPATH=" + str(args.pythonpath))
        print(subprocess.list2cmdline(cmd))
        return

    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.pythonpath)
    raise SystemExit(subprocess.run(cmd, check=False, env=env).returncode)


if __name__ == "__main__":
    main()
