#!/usr/bin/env python3
from __future__ import annotations

"""
Run command:
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_try0526 \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_train0705_round2_compare.py \
  --candidate-label target_w24_ratio_005 \
  --candidate-dir /data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round4_test5_ratio_only/train0705_round4_test5_ratio_only_target_w24_ratio_005/step-001000 \
  --out-json /data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round4_test5_ratio_only/round4_test5_ratio_only_scores.json
"""

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve().with_name("score_multicase_methods.py")
DEFAULT_PYTHON_BIN = Path("/data/gaoya/miniconda3/envs/wan/bin/python")
DEFAULT_ROUND2_BASE = Path("/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round2_test5")
DEFAULT_PYTHONPATH = "/home/gaoya/Code_Video/Code_data/Code_try0526"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score a new train0705 method against the established round2_test5 reference methods."
    )
    parser.add_argument("--candidate-label", required=True)
    parser.add_argument("--candidate-dir", type=Path, required=True)
    parser.add_argument(
        "--extra-candidate",
        action="append",
        default=[],
        help="Additional candidate in the form label=/abs/path/to/method_dir.",
    )
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--round2-base", type=Path, default=DEFAULT_ROUND2_BASE)
    parser.add_argument("--pythonpath", default=DEFAULT_PYTHONPATH)
    parser.add_argument("--videophy2-device", default="cuda:0")
    parser.add_argument("--videophy2-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--videophy2-num-frames", type=int, default=32)
    parser.add_argument("--skip-wmreward", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _reference_method_dirs(round2_base: Path) -> list[tuple[str, Path]]:
    return [
        ("baseline", round2_base / "train0705_round2_test5_baseline" / "step-001000"),
        ("ladder_s20", round2_base / "train0705_round2_test5_ladder_s20" / "step-001000"),
        ("knee_mid_s18", round2_base / "train0705_round2_test5_knee_mid_s18" / "step-001000"),
    ]


def main() -> None:
    args = parse_args()
    candidate_dir = args.candidate_dir.expanduser().resolve()
    if not candidate_dir.is_dir():
        raise SystemExit(f"Candidate directory not found: {candidate_dir}")

    method_dirs = _reference_method_dirs(args.round2_base.expanduser().resolve())
    for label, path in method_dirs:
        if not path.is_dir():
            raise SystemExit(f"Reference method directory not found for {label}: {path}")
    method_dirs.append((str(args.candidate_label), candidate_dir))
    for spec in args.extra_candidate:
        if "=" not in spec:
            raise SystemExit(f"Invalid --extra-candidate spec (expected label=path): {spec}")
        label, path_str = spec.split("=", 1)
        path = Path(path_str).expanduser().resolve()
        if not path.is_dir():
            raise SystemExit(f"Extra candidate directory not found for {label}: {path}")
        method_dirs.append((str(label), path))

    cmd = [
        str(args.python_bin),
        str(SCRIPT_PATH),
    ]
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
    if args.limit_cases is not None:
        cmd.extend(["--limit-cases", str(args.limit_cases)])

    if args.dry_run:
        print("PYTHONPATH=" + str(args.pythonpath))
        print(subprocess.list2cmdline(cmd))
        return

    env = dict(**__import__("os").environ)
    env["PYTHONPATH"] = str(args.pythonpath)
    raise SystemExit(subprocess.run(cmd, check=False, env=env).returncode)


if __name__ == "__main__":
    main()
