#!/usr/bin/env python3
from __future__ import annotations

"""
Official Wan2.2 TI2V-5B frequency-guidance pilot on test_5.

Generate:
CUDA_VISIBLE_DEVICES=2,1 /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_official_freqguide_test5.py \
  --stage generate \
  --main-gpu 2 \
  --vjepa-gpu 1

Score:
CUDA_VISIBLE_DEVICES=7 /home/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/run_official_freqguide_test5.py \
  --stage score \
  --score-gpu 7
"""

import argparse
import json
import os
import subprocess
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
GUIDANCE_DIR = THIS_FILE.parent
ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
WAN22_OFFICIAL_REPO = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main_official")
PY_WAN_CU128 = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
PY_WAN = Path("/data/gaoya/miniconda3/envs/wan/bin/python")

INPUT_LIST = Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt")
WAN22_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
VJEPA_CKPT = Path("/data/gaoya/ckpt/VJEPA2/vith.pt")
OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/official_freqguide_test5")
TRACE_ROOT = Path("/data/gaoya/agent-data/outputs/vjepa_guidance_trace")
DEDUP_LIST = OUTPUT_ROOT / "inputs" / "test5_unique.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run/score official Wan2.2 frequency-guidance pilot on test_5.")
    parser.add_argument("--stage", choices=["generate", "score", "all"], default="all")
    parser.add_argument("--main-gpu", type=int, default=2)
    parser.add_argument("--vjepa-gpu", type=int, default=1)
    parser.add_argument("--score-gpu", type=int, default=7)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def _parse_visible_devices(raw_value: str | None) -> list[str]:
    if raw_value is None:
        return []
    devices: list[str] = []
    for chunk in str(raw_value).split(","):
        value = chunk.strip()
        if value:
            devices.append(value)
    return devices


def resolve_child_cuda_visible_devices(
    parent_env: dict[str, str],
    *,
    main_gpu: int,
    vjepa_gpu: int,
) -> str:
    parent_visible = _parse_visible_devices(parent_env.get("CUDA_VISIBLE_DEVICES"))
    requested = [int(main_gpu), int(vjepa_gpu)]
    if parent_visible and all(0 <= device_idx < len(parent_visible) for device_idx in requested):
        resolved = ",".join(parent_visible[device_idx] for device_idx in requested)
        print(
            "[gpu-map] interpreting requested GPUs as local indices under parent "
            f"CUDA_VISIBLE_DEVICES={','.join(parent_visible)} -> child {resolved}",
            flush=True,
        )
        return resolved
    resolved = ",".join(str(device_idx) for device_idx in requested)
    print(
        "[gpu-map] using requested GPUs as physical ids "
        f"-> child CUDA_VISIBLE_DEVICES={resolved}",
        flush=True,
    )
    return resolved


def run_cmd(cmd: list[str], *, env: dict[str, str], label: str, continue_on_error: bool) -> None:
    print(f"[run] {label}", flush=True)
    print(f"[env] CUDA_VISIBLE_DEVICES={env.get('CUDA_VISIBLE_DEVICES', '<unset>')}", flush=True)
    print(subprocess.list2cmdline(cmd), flush=True)
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        if continue_on_error:
            print(f"[warn] {label} failed with returncode={result.returncode}", flush=True)
            return
        raise SystemExit(result.returncode)


def ensure_unique_input_list() -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    DEDUP_LIST.parent.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    ordered: list[str] = []
    for raw in INPUT_LIST.read_text(encoding="utf-8").splitlines():
        value = raw.strip()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    DEDUP_LIST.write_text("\n".join(ordered) + "\n", encoding="utf-8")
    print(
        f"[info] using {len(ordered)} unique inputs from {INPUT_LIST} -> {DEDUP_LIST}",
        flush=True,
    )
    return DEDUP_LIST


def generate(args: argparse.Namespace) -> None:
    unique_list = ensure_unique_input_list()
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{ROOT}:{WAN22_OFFICIAL_REPO}"
    env["WAN22_OFFICIAL_REPO"] = str(WAN22_OFFICIAL_REPO)
    env["CUDA_VISIBLE_DEVICES"] = resolve_child_cuda_visible_devices(
        env,
        main_gpu=int(args.main_gpu),
        vjepa_gpu=int(args.vjepa_gpu),
    )

    baseline_dir = OUTPUT_ROOT / "baseline"
    guided_dir = OUTPUT_ROOT / "guided"

    baseline_cmd = [
        str(PY_WAN_CU128),
        str(GUIDANCE_DIR / "wanti2v.py"),
        "--input-list",
        str(unique_list),
        "--output-root",
        str(baseline_dir),
        "--model-name",
        "wan22_official_test5_baseline",
        "--backend",
        "official",
        "--wan-root",
        str(WAN22_ROOT),
        "--size",
        "704*1280",
        "--frame-num",
        "49",
        "--sampling-steps",
        "40",
        "--cfg-scale",
        "5.0",
        "--fps",
        "30",
        "--seed",
        "42",
        "--offload-model",
        "--vjepa-preset",
        "baseline",
    ]
    guided_cmd = [
        str(PY_WAN_CU128),
        str(GUIDANCE_DIR / "wanti2v_freqguidance.py"),
        "--input-list",
        str(unique_list),
        "--output-root",
        str(guided_dir),
        "--model-name",
        "wan22_official_test5_freqguide",
        "--backend",
        "official",
        "--wan-root",
        str(WAN22_ROOT),
        "--size",
        "704*1280",
        "--frame-num",
        "49",
        "--sampling-steps",
        "40",
        "--cfg-scale",
        "5.0",
        "--fps",
        "30",
        "--seed",
        "42",
        "--offload-model",
        "--vjepa-preset",
        "target_w24_s15_ratio_0025",
        "--vjepa-ckpt",
        str(VJEPA_CKPT),
        "--vjepa-device-id",
        "1",
        "--trace-intermediates",
        "--trace-build-html",
    ]
    if args.force:
        baseline_cmd.append("--force")
        guided_cmd.append("--force")
    run_cmd(baseline_cmd, env=env, label="official_freqguide_test5/baseline", continue_on_error=args.continue_on_error)
    run_cmd(guided_cmd, env=env, label="official_freqguide_test5/guided", continue_on_error=args.continue_on_error)


def score(args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env["PYTHONPATH"] = f"/home/gaoya/Code_Video/Code_data/Code_try0526:{ROOT}"
    env["CUDA_VISIBLE_DEVICES"] = resolve_child_cuda_visible_devices(
        env,
        main_gpu=int(args.score_gpu),
        vjepa_gpu=int(args.score_gpu),
    ).split(",", maxsplit=1)[0]
    out_json = OUTPUT_ROOT / "scores" / "official_freqguide_test5_summary.json"
    out_md = OUTPUT_ROOT / "scores" / "official_freqguide_test5_summary.md"
    cmd = [
        str(PY_WAN),
        str(GUIDANCE_DIR / "score_multicase_allmetrics.py"),
        "--method-dir",
        f"baseline={OUTPUT_ROOT / 'baseline'}",
        "--method-dir",
        f"guided={OUTPUT_ROOT / 'guided'}",
        "--out-json",
        str(out_json),
        "--out-md",
        str(out_md),
        "--baseline-label",
        "baseline",
        "--videophy2-task",
        "pc",
        "--videophy2-device",
        "cuda:0",
        "--pmf-device",
        "cpu",
        "--phyground-general-only",
    ]
    run_cmd(cmd, env=env, label="score/official_freqguide_test5", continue_on_error=args.continue_on_error)
    if out_json.exists():
        data = json.loads(out_json.read_text(encoding="utf-8"))
        print(
            json.dumps(
                data.get("ranking_by_mean_delta_wmreward_surprise", []),
                ensure_ascii=False,
                indent=2,
            ),
            flush=True,
        )


def main() -> None:
    args = parse_args()
    if args.stage in {"generate", "all"}:
        generate(args)
    if args.stage in {"score", "all"}:
        score(args)


if __name__ == "__main__":
    main()
