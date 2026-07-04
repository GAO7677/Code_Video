#!/usr/bin/env python3
from __future__ import annotations

"""
Run command:
CUDA_VISIBLE_DEVICES=1 \
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_try0526 \
/data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_train0705_s15_local_sweep.py

This scores the round6 overlap-5 local sweep against the same overlap-5
baseline / ladder_s20 / knee_mid_s18 references used by round5.
"""

import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Any


SCRIPT_PATH = Path(__file__).resolve().with_name("score_multicase_methods.py")
DEFAULT_PYTHON_BIN = Path("/data/gaoya/miniconda3/envs/wan/bin/python")
DEFAULT_PYTHONPATH = "/home/gaoya/Code_Video/Code_data/Code_try0526"
DEFAULT_ROUND5_BASE = Path(
    "/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round5_ratio_cap_sweep_overlap5"
)
DEFAULT_ROUND6_BASE = Path(
    "/data/gaoya/agent-data/outputs/train0705_vjepa_current_modes/round6_s15_local_sweep_overlap5"
)
ROUND6_MODES = (
    "target_w24_s14_ratio_003",
    "target_w24_s15_ratio_0025",
    "target_w24_s15_ratio_003",
    "target_w24_s15_ratio_0035",
    "target_w24_s16_ratio_003",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Score round6 s15 local-sweep candidates against overlap-5 references."
    )
    parser.add_argument("--round5-base", type=Path, default=DEFAULT_ROUND5_BASE)
    parser.add_argument("--round6-base", type=Path, default=DEFAULT_ROUND6_BASE)
    parser.add_argument(
        "--out-json",
        type=Path,
        default=DEFAULT_ROUND6_BASE / "round6_s15_local_overlap5_scores.json",
    )
    parser.add_argument(
        "--out-md",
        type=Path,
        default=DEFAULT_ROUND6_BASE / "round6_s15_local_overlap5_summary.md",
    )
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--pythonpath", default=DEFAULT_PYTHONPATH)
    parser.add_argument("--videophy2-device", default="cuda:0")
    parser.add_argument("--videophy2-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--videophy2-num-frames", type=int, default=32)
    parser.add_argument("--skip-wmreward", action="store_true")
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _require_dir(path: Path, label: str, *, skip_missing: bool = False) -> bool:
    if path.is_dir():
        return True
    if skip_missing:
        print(f"[skip] missing {label}: {path}", flush=True)
        return False
    raise SystemExit(f"Required directory not found for {label}: {path}")


def _method_dirs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    round5_base = args.round5_base.expanduser().resolve()
    round6_base = args.round6_base.expanduser().resolve()
    method_dirs: list[tuple[str, Path]] = []

    references = (
        ("baseline", round5_base / "overlap5_reference_dirs" / "baseline"),
        ("ladder_s20", round5_base / "overlap5_reference_dirs" / "ladder_s20"),
        ("knee_mid_s18", round5_base / "overlap5_reference_dirs" / "knee_mid_s18"),
    )
    for label, path in references:
        if _require_dir(path, label, skip_missing=bool(args.skip_missing)):
            method_dirs.append((label, path))

    for mode_id in ROUND6_MODES:
        path = round6_base / f"train0705_round6_s15_local_{mode_id}" / "step-001000"
        if _require_dir(path, mode_id, skip_missing=bool(args.skip_missing)):
            method_dirs.append((mode_id, path))
    return method_dirs


def _write_markdown_summary(out_md: Path, score_json: Path) -> None:
    payload: dict[str, Any] = json.loads(score_json.read_text())
    rows = payload.get("ranking_by_mean_delta_surprise", [])
    lines = [
        "# round6 s15 local sweep overlap-5 summary",
        "",
        "Lower `mean_delta_surprise_vs_baseline` is better for the primary WMReward surprise objective.",
        "",
        "| rank | method | cases | Δsurprise | Δphysics_iq | Δvideophy2 | Δcosmos_reason1 | mean_surprise |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            "| {rank} | {method} | {cases} | {ds} | {dp} | {dv} | {dc} | {ms} |".format(
                rank=rank,
                method=row.get("method"),
                cases=row.get("num_cases"),
                ds=row.get("mean_delta_surprise_vs_baseline"),
                dp=row.get("mean_delta_physics_iq_vs_baseline"),
                dv=row.get("mean_delta_videophy2_vs_baseline"),
                dc=row.get("mean_delta_cosmos_reason1_vs_baseline"),
                ms=row.get("mean_surprise"),
            )
        )
    lines.append("")
    lines.append(f"Source JSON: `{score_json}`")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    method_dirs = _method_dirs(args)
    if not method_dirs:
        raise SystemExit("No method directories available to score.")

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
    if args.limit_cases is not None:
        cmd.extend(["--limit-cases", str(args.limit_cases)])

    if args.dry_run:
        print("PYTHONPATH=" + str(args.pythonpath))
        print(subprocess.list2cmdline(cmd))
        return

    env = dict(os.environ)
    env["PYTHONPATH"] = str(args.pythonpath)
    result = subprocess.run(cmd, check=False, env=env)
    if result.returncode != 0:
        raise SystemExit(result.returncode)
    _write_markdown_summary(args.out_md.expanduser().resolve(), args.out_json.expanduser().resolve())
    print(f"Wrote {args.out_md.expanduser().resolve()}", flush=True)


if __name__ == "__main__":
    main()
