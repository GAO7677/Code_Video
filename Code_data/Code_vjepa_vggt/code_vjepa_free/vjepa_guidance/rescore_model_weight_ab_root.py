#!/usr/bin/env python3
from __future__ import annotations

"""
Re-score an existing model-weight A/B output root using the current
full-metric scorer, then regenerate Markdown and HTML summaries.

Example:
  CUDA_VISIBLE_DEVICES=2 /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
  /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/rescore_model_weight_ab_root.py \
    --output-root /data/gaoya/agent-data/outputs/model_weight_ab_test5_20260705 \
    --score-gpu 2
"""

import argparse
import os
import subprocess
from dataclasses import dataclass
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
GUIDANCE_DIR = THIS_FILE.parent
ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
PY_WAN = Path("/data/gaoya/miniconda3/envs/wan/bin/python")
PY_WAN_CU128 = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
DEFAULT_SERVE_ROOT = Path("/data/gaoya")


@dataclass(frozen=True)
class FamilySpec:
    family_id: str
    baseline_dir: Path
    guided_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Re-score an existing model-weight A/B root with current full metrics.")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--families", nargs="*", default=None, help="Optional subset of family ids to rescore.")
    parser.add_argument("--score-gpu", type=int, default=7)
    parser.add_argument("--videophy2-task", choices=["sa", "pc", "rule"], default="pc")
    parser.add_argument("--videophy2-device", default="cuda:0")
    parser.add_argument("--videophy2-num-frames", type=int, default=32)
    parser.add_argument("--proxy-device", default="cuda:0")
    parser.add_argument("--pmf-device", default="cpu")
    parser.add_argument("--pdi-timeout-seconds", type=float, default=None)
    parser.add_argument("--serve-root", type=Path, default=DEFAULT_SERVE_ROOT)
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--skip-pdi", action="store_true")
    parser.add_argument("--skip-wmreward", action="store_true")
    parser.add_argument("--skip-proxy", action="store_true")
    parser.add_argument("--skip-videophy2", action="store_true")
    parser.add_argument("--skip-phyground", action="store_true")
    parser.add_argument("--skip-cosmos", action="store_true")
    parser.add_argument("--skip-physics-iq", action="store_true")
    parser.add_argument("--skip-pmf", action="store_true")
    parser.add_argument("--skip-dashboard", action="store_true")
    parser.add_argument("--skip-markdown", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    return parser.parse_args()


def is_family_dir(path: Path) -> bool:
    if not path.is_dir():
        return False
    return (path / "baseline").is_dir() and (path / "guided").is_dir()


def detect_primary_leaf(directory: Path) -> Path:
    current = directory
    while current.is_dir():
        mp4_count = len([p for p in current.glob("*.mp4") if not p.name.endswith(".browser.mp4")])
        if mp4_count > 0:
            return current
        subdirs = sorted([p for p in current.iterdir() if p.is_dir()])
        if len(subdirs) != 1:
            return current
        current = subdirs[0]
    return directory


def detect_baseline_reuse_leaf(family_id: str) -> Path | None:
    candidate_map = {
        "train0705_step002500": Path("/data/gaoya/AAA_test_video/0623/test/v2v/train_stage1b_diffsynth_native0705_0705/step-002500"),
        "train0705_step005000": Path("/data/gaoya/AAA_test_video/0623/test/v2v/train_stage1b_diffsynth_native0705_0705/step-005000"),
        "train0705_step007000": Path("/data/gaoya/AAA_test_video/0623/test/v2v/train_stage1b_diffsynth_native0705_0705/step-007000"),
    }
    candidate = candidate_map.get(family_id)
    if candidate is not None and candidate.is_dir():
        return candidate
    return None


def discover_families(output_root: Path, selected: set[str] | None) -> list[FamilySpec]:
    families: list[FamilySpec] = []
    for family_root in sorted([p for p in output_root.iterdir() if is_family_dir(p)], key=lambda p: p.name):
        family_id = family_root.name
        if selected and family_id not in selected:
            continue
        baseline_leaf = detect_primary_leaf(family_root / "baseline")
        guided_leaf = detect_primary_leaf(family_root / "guided")
        baseline_reuse = detect_baseline_reuse_leaf(family_id)
        baseline_dir = baseline_reuse if baseline_reuse is not None else baseline_leaf
        families.append(FamilySpec(family_id=family_id, baseline_dir=baseline_dir, guided_dir=guided_leaf))
    return families


def run_cmd(cmd: list[str], *, env: dict[str, str], label: str, continue_on_error: bool) -> None:
    print(f"[run] {label}", flush=True)
    print(subprocess.list2cmdline(cmd), flush=True)
    result = subprocess.run(cmd, env=env, check=False)
    if result.returncode != 0:
        if continue_on_error:
            print(f"[warn] {label} failed with returncode={result.returncode}", flush=True)
            return
        raise SystemExit(result.returncode)


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    if not output_root.is_dir():
        raise SystemExit(f"Output root not found: {output_root}")

    selected = set(args.families) if args.families else None
    families = discover_families(output_root, selected)
    if not families:
        raise SystemExit(f"No baseline/guided family directories found under {output_root}")

    scores_dir = output_root / "scores"
    scores_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"/home/gaoya/Code_Video/Code_data/Code_try0526:{ROOT}"
    env["CUDA_VISIBLE_DEVICES"] = str(args.score_gpu)

    for family in families:
        out_json = scores_dir / f"{family.family_id}_summary.json"
        out_md = scores_dir / f"{family.family_id}_summary.md"
        cmd = [
            str(PY_WAN),
            str(GUIDANCE_DIR / "score_multicase_allmetrics.py"),
            "--method-dir",
            f"baseline={family.baseline_dir}",
            "--method-dir",
            f"guided={family.guided_dir}",
            "--out-json",
            str(out_json),
            "--out-md",
            str(out_md),
            "--baseline-label",
            "baseline",
            "--videophy2-task",
            args.videophy2_task,
            "--videophy2-device",
            args.videophy2_device,
            "--videophy2-num-frames",
            str(args.videophy2_num_frames),
            "--proxy-device",
            args.proxy_device,
            "--pmf-device",
            args.pmf_device,
            "--pdi-timeout-seconds",
            "0" if args.pdi_timeout_seconds is None else str(args.pdi_timeout_seconds),
            "--phyground-general-only",
        ]
        if args.limit_cases is not None:
            cmd.extend(["--limit-cases", str(args.limit_cases)])
        if args.skip_pdi:
            cmd.append("--skip-pdi")
        if args.skip_wmreward:
            cmd.append("--skip-wmreward")
        if args.skip_proxy:
            cmd.append("--skip-proxy")
        if args.skip_videophy2:
            cmd.append("--skip-videophy2")
        if args.skip_phyground:
            cmd.append("--skip-phyground")
        if args.skip_cosmos:
            cmd.append("--skip-cosmos")
        if args.skip_physics_iq:
            cmd.append("--skip-physics-iq")
        if args.skip_pmf:
            cmd.append("--skip-pmf")
        run_cmd(cmd, env=env, label=f"score/{family.family_id}", continue_on_error=args.continue_on_error)

    if not args.skip_markdown:
        report_md = output_root / "ab_report" / "model_weight_ab_report.md"
        cmd = [
            str(PY_WAN_CU128),
            str(GUIDANCE_DIR / "export_model_weight_ab_markdown.py"),
            "--scores-dir",
            str(scores_dir),
            "--output-md",
            str(report_md),
            "--title",
            f"Model-Weight A/B Report: {output_root.name}",
        ]
        run_cmd(cmd, env=os.environ.copy(), label="export/model_weight_ab_markdown", continue_on_error=args.continue_on_error)

    if not args.skip_dashboard:
        dashboard_html = output_root / "ab_dashboard" / "index.html"
        cmd = [
            str(PY_WAN_CU128),
            str(GUIDANCE_DIR / "visualize_model_weight_ab.py"),
            "--scores-dir",
            str(scores_dir),
            "--output-html",
            str(dashboard_html),
            "--serve-root",
            str(args.serve_root),
            "--title",
            f"Model-Weight A/B Dashboard: {output_root.name}",
        ]
        run_cmd(cmd, env=os.environ.copy(), label="build/model_weight_ab_dashboard", continue_on_error=args.continue_on_error)

    print("\nFamilies discovered:", flush=True)
    for family in families:
        print(f"- {family.family_id}: baseline={family.baseline_dir} guided={family.guided_dir}", flush=True)


if __name__ == "__main__":
    main()
