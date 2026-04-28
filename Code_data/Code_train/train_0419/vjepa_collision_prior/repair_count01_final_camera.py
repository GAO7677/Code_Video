#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_COUNT01_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid/single_object_preview/count_01"
)
DEFAULT_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases"
)
DEFAULT_TRY1_SCRIPT = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/try1_physxnet_articulation_mpm0417.py"
)
DEFAULT_PYTHON = Path("/data/gaoya/miniconda3/envs/wan/bin/python")
DEFAULT_REPORT_DIR = Path(
    "/home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/count01_camera_fix"
)


CASE_INDEX_RE = re.compile(r"case(\d+)")


@dataclass
class SampleEval:
    sample_dir: Path
    sample_name: str
    object_id: str
    case_name: str
    case_index: int
    parent_case_index: int
    motion_category: str
    resolution: tuple[int, int]
    camera: dict[str, Any]
    last_bbox: list[float]
    last_visible: bool
    last_inside: bool
    last_safe_margin: bool
    border_margin_min: float
    qa_metrics: dict[str, Any]
    scene_input: dict[str, Any]
    metadata: dict[str, Any]

    def to_json(self) -> dict[str, Any]:
        return {
            "sample_dir": str(self.sample_dir),
            "sample_name": self.sample_name,
            "object_id": self.object_id,
            "case_name": self.case_name,
            "case_index": self.case_index,
            "parent_case_index": self.parent_case_index,
            "motion_category": self.motion_category,
            "resolution": list(self.resolution),
            "camera": self.camera,
            "last_bbox": self.last_bbox,
            "last_visible": self.last_visible,
            "last_inside": self.last_inside,
            "last_safe_margin": self.last_safe_margin,
            "border_margin_min": self.border_margin_min,
            "qa_metrics": self.qa_metrics,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Repair count_01 samples whose last-frame object leaves or clips the frame.")
    parser.add_argument("--count01-root", type=Path, default=DEFAULT_COUNT01_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--try1-script", type=Path, default=DEFAULT_TRY1_SCRIPT)
    parser.add_argument("--python-bin", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--margin", type=float, default=12.0, help="Required minimum border margin in pixels on the final frame.")
    parser.add_argument("--apply", action="store_true", help="Actually regenerate and replace failing samples.")
    parser.add_argument("--limit", type=int, default=0, help="Optional limit on how many failing samples to repair.")
    parser.add_argument("--keep-staging", action="store_true", help="Keep staging outputs after successful replacement.")
    parser.add_argument("--force-all-failing", action="store_true", help="Repair all failing cases regardless of case type.")
    parser.add_argument("--sample-name", action="append", default=None, help="Restrict work to the given sample name. Can be passed multiple times.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def safe_case_index_from_scene(scene_input: dict[str, Any], sample_name: str) -> int:
    if "case_id" in scene_input:
        return int(scene_input["case_id"])
    match = CASE_INDEX_RE.search(sample_name)
    if not match:
        raise ValueError(f"Could not infer case index from {sample_name}")
    return int(match.group(1))


def parent_case_index_from_scene(scene_input: dict[str, Any], case_index: int) -> int:
    counterfactual = dict(scene_input.get("counterfactual") or {})
    if "parent_case_index" in counterfactual:
        return int(counterfactual["parent_case_index"])
    return int(case_index)


def final_border_margin(bbox: np.ndarray, width: int, height: int, visible: bool) -> float:
    if not visible or bbox.shape != (4,) or not np.isfinite(bbox).all():
        return float("-inf")
    x1, y1, x2, y2 = [float(v) for v in bbox.tolist()]
    if x2 <= x1 or y2 <= y1:
        return float("-inf")
    return float(min(x1, y1, (width - 1) - x2, (height - 1) - y2))


def evaluate_sample(sample_dir: Path, margin: float) -> SampleEval:
    scene_input = load_json(sample_dir / "scene_input.json")
    metadata = load_json(sample_dir / "metadata.json")
    qa_metrics_path = sample_dir / "qa_metrics.json"
    qa_metrics = load_json(qa_metrics_path) if qa_metrics_path.exists() else {}
    kin = np.load(sample_dir / "physics" / "rigid_kinematics.npz")
    bbox = np.asarray(kin["bbox_xyxy"], dtype=np.float32)[-1, 0]
    visible = bool(np.asarray(kin["visibility_mask"])[-1, 0])
    width, height = [int(v) for v in metadata.get("resolution", [960, 720])]
    border_margin = final_border_margin(bbox, width, height, visible)
    inside = bool(visible and np.isfinite(border_margin) and border_margin >= 0.0)
    safe_margin = bool(inside and border_margin >= float(margin))
    case_index = safe_case_index_from_scene(scene_input, sample_dir.name)
    parent_case_index = parent_case_index_from_scene(scene_input, case_index)
    return SampleEval(
        sample_dir=sample_dir,
        sample_name=sample_dir.name,
        object_id=str(scene_input.get("object_id", metadata.get("object_id", ""))),
        case_name=str(scene_input.get("case_name", metadata.get("case_name", sample_dir.name))),
        case_index=case_index,
        parent_case_index=parent_case_index,
        motion_category=str(metadata.get("motion_category", scene_input.get("scene_label", ""))),
        resolution=(width, height),
        camera=dict(scene_input.get("camera", metadata.get("camera", {}))),
        last_bbox=[float(v) for v in bbox.tolist()],
        last_visible=visible,
        last_inside=inside,
        last_safe_margin=safe_margin,
        border_margin_min=border_margin,
        qa_metrics=qa_metrics,
        scene_input=scene_input,
        metadata=metadata,
    )


def iter_sample_dirs(root: Path) -> list[Path]:
    return sorted(path for path in root.iterdir() if path.is_dir())


def scan_count01(root: Path, margin: float) -> list[SampleEval]:
    results: list[SampleEval] = []
    for sample_dir in iter_sample_dirs(root):
        try:
            results.append(evaluate_sample(sample_dir, margin))
        except Exception as exc:
            print(f"[WARN] scan_failed sample={sample_dir.name} error={type(exc).__name__}: {exc}", file=sys.stderr)
    return results


def summarize(results: list[SampleEval], margin: float) -> dict[str, Any]:
    failing = [item for item in results if not item.last_safe_margin]
    by_case: dict[str, int] = {}
    by_motion: dict[str, int] = {}
    for item in failing:
        by_case[item.case_name] = by_case.get(item.case_name, 0) + 1
        by_motion[item.motion_category] = by_motion.get(item.motion_category, 0) + 1
    return {
        "total": len(results),
        "passing_last_frame_margin": len(results) - len(failing),
        "failing_last_frame_margin": len(failing),
        "margin_px": float(margin),
        "failing_by_case_name": by_case,
        "failing_by_motion_category": by_motion,
    }


def build_attempt_schedule(sample: SampleEval) -> list[float]:
    case_l = sample.case_name.lower()
    motion_l = sample.motion_category.lower()
    if "highdrop" in case_l or "high_drop" in case_l or "random_parabola" in case_l or "parabola" in motion_l:
        return [1.60, 1.80, 2.00]
    return [1.35, 1.50, 1.70]


def infer_regen_config(sample: SampleEval) -> dict[str, Any]:
    sim_meta = dict(sample.metadata.get("simulation", {}))
    dt = float(sim_meta.get("dt", 0.003))
    substeps = int(sim_meta.get("substeps", 40))
    steps_per_frame = int(sim_meta.get("steps_per_frame", 28))
    fps = max(1, int(round(1.0 / max(dt * steps_per_frame, 1e-8))))
    steps = max(1, int(sample.metadata.get("frames", 13)) - 1)
    return {
        "dt": dt,
        "substeps": substeps,
        "steps": steps,
        "fps": fps,
        "ball_posx": 0.03,
    }


def build_regen_command(
    *,
    python_bin: Path,
    try1_script: Path,
    dataset_root: Path,
    staging_root: Path,
    sample: SampleEval,
    camera_distance_mult: float,
) -> list[str]:
    cfg = infer_regen_config(sample)
    cmd = [
        str(python_bin),
        str(try1_script),
        "--physx_root",
        "/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet",
        "--version",
        "version_1",
        "--object_id",
        sample.object_id,
        "--output_root",
        str(staging_root),
        "--run_genesis",
        "--generate_all_count_motion_cases",
        "--rigid_count_filter",
        "1",
        "--case_index_filter",
        str(sample.parent_case_index),
        "--prefer_existing_runtime_meshes",
        "--dt",
        f"{cfg['dt']}",
        "--substeps",
        str(cfg["substeps"]),
        "--ball_posx",
        f"{cfg['ball_posx']}",
        "--steps",
        str(cfg["steps"]),
        "--fps",
        str(cfg["fps"]),
        "--simulator_mode",
        "rigid",
        "--camera_distance_mult",
        f"{camera_distance_mult:.2f}",
    ]
    if dict(sample.scene_input.get("counterfactual") or {}):
        cmd.extend([
            "--enable_counterfactual_cases",
            "--counterfactual_only",
        ])
    return cmd


def relative_stage_case(staging_root: Path, sample_name: str) -> Path:
    return staging_root / "train" / "rigid" / "single_object_preview" / "count_01" / sample_name


def run_cmd(cmd: list[str], log_path: Path) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.run(cmd, stdout=handle, stderr=subprocess.STDOUT, check=False)
    return int(process.returncode)


def copy_if_exists(src: Path, dst: Path) -> None:
    if not src.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def render_html(report: dict[str, Any]) -> str:
    entries = report.get("repairs", [])
    cards = []
    for item in entries:
        assets = dict(item.get("assets", {}))
        before_video = assets.get("before_video")
        after_video = assets.get("after_video")
        before_eval = dict(item.get("before", {}))
        after_eval = dict(item.get("after", {}))
        status = str(item.get("status", "unknown"))
        cards.append(
            f"""
<article class="card">
  <div class="head">
    <h2>{item.get('sample_name', '')}</h2>
    <span class="status {status}">{status}</span>
  </div>
  <p class="meta">case={item.get('case_name', '')} | motion={item.get('motion_category', '')} | mult={item.get('camera_distance_mult', 'n/a')}</p>
  <p class="meta">before margin={before_eval.get('border_margin_min', 'n/a')} | after margin={after_eval.get('border_margin_min', 'n/a')}</p>
  <div class="videos">
    <div>
      <div class="label">Before</div>
      {'<video controls muted loop src="' + before_video + '"></video>' if before_video else '<div class="missing">missing</div>'}
    </div>
    <div>
      <div class="label">After</div>
      {'<video controls muted loop src="' + after_video + '"></video>' if after_video else '<div class="missing">missing</div>'}
    </div>
  </div>
</article>
"""
        )
    summary = dict(report.get("summary_after_scoped" if report.get("applied") else "summary_before_scoped", {}))
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>count_01 Camera Fix Report</title>
  <style>
    :root {{
      --bg: #f5f2eb;
      --ink: #1d2529;
      --card: #fffdf8;
      --line: #cfbfaa;
      --good: #2d6a4f;
      --bad: #b23a48;
      --muted: #5d6768;
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif; color: var(--ink); background: linear-gradient(180deg, #f8f3e7 0%, var(--bg) 100%); }}
    main {{ max-width: 1480px; margin: 0 auto; padding: 28px 24px 44px; }}
    h1 {{ margin: 0 0 8px; font-size: 34px; letter-spacing: -0.03em; }}
    .lede {{ color: var(--muted); margin: 0 0 18px; }}
    .stats {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin-bottom: 24px; }}
    .stat {{ background: rgba(255,255,255,0.72); border: 1px solid var(--line); border-radius: 16px; padding: 14px 16px; }}
    .stat .label {{ color: var(--muted); font-size: 12px; text-transform: uppercase; letter-spacing: 0.08em; display: block; margin-bottom: 8px; }}
    .stat .value {{ font-size: 30px; font-weight: 700; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }}
    .card {{ background: var(--card); border: 1px solid var(--line); border-radius: 18px; padding: 16px; }}
    .head {{ display: flex; justify-content: space-between; gap: 12px; align-items: center; }}
    h2 {{ margin: 0; font-size: 19px; line-height: 1.2; }}
    .status {{ border-radius: 999px; padding: 4px 10px; font-size: 12px; font-weight: 700; text-transform: uppercase; }}
    .status.repaired {{ background: rgba(45,106,79,0.12); color: var(--good); }}
    .status.failed {{ background: rgba(178,58,72,0.12); color: var(--bad); }}
    .meta {{ margin: 8px 0 0; color: var(--muted); font-size: 13px; }}
    .videos {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; margin-top: 14px; }}
    .label {{ font-size: 12px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 6px; }}
    video {{ width: 100%; border-radius: 12px; border: 1px solid var(--line); background: #000; }}
    .missing {{ min-height: 220px; border-radius: 12px; border: 1px dashed var(--line); display: grid; place-items: center; color: var(--muted); }}
  </style>
</head>
<body>
  <main>
    <h1>count_01 Camera Fix Report</h1>
    <p class="lede">Final-frame safety margin target: {report.get('margin_px', 12.0)} px. Only samples that failed this test are regenerated.</p>
    <section class="stats">
      <div class="stat"><span class="label">Total Samples</span><span class="value">{summary.get('total', 0)}</span></div>
      <div class="stat"><span class="label">Passing</span><span class="value">{summary.get('passing_last_frame_margin', 0)}</span></div>
      <div class="stat"><span class="label">Failing</span><span class="value">{summary.get('failing_last_frame_margin', 0)}</span></div>
      <div class="stat"><span class="label">Repaired Entries</span><span class="value">{len(entries)}</span></div>
    </section>
    <section class="grid">
      {''.join(cards) if cards else '<p>No repairs were applied in this run.</p>'}
    </section>
  </main>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    report_dir = args.report_dir
    assets_dir = report_dir / "assets"
    staging_root = report_dir / "staging"
    logs_dir = report_dir / "logs"
    report_dir.mkdir(parents=True, exist_ok=True)

    before_results = scan_count01(args.count01_root, args.margin)
    before_summary = summarize(before_results, args.margin)
    selected_names = None if not args.sample_name else {str(name) for name in args.sample_name}
    scoped_results = before_results if selected_names is None else [
        item for item in before_results if item.sample_name in selected_names
    ]
    failing = [item for item in scoped_results if not item.last_safe_margin]
    dump_json(report_dir / "scan_before.json", {
        "summary_full": before_summary,
        "summary_scoped": summarize(scoped_results, args.margin),
        "samples": [item.to_json() for item in scoped_results],
    })
    print(json.dumps(summarize(scoped_results, args.margin), ensure_ascii=False, indent=2))

    report: dict[str, Any] = {
        "count01_root": str(args.count01_root),
        "dataset_root": str(args.dataset_root),
        "margin_px": float(args.margin),
        "applied": bool(args.apply),
        "summary_before_full": before_summary,
        "summary_before_scoped": summarize(scoped_results, args.margin),
        "repairs": [],
    }

    if args.apply:
        selected = failing if args.force_all_failing else failing
        if args.limit > 0:
            selected = selected[: int(args.limit)]
        for index, sample in enumerate(selected, start=1):
            print(
                f"[repair] {index}/{len(selected)} sample={sample.sample_name} "
                f"case={sample.case_name} border_margin={sample.border_margin_min:.2f}"
            )
            sample_assets = assets_dir / sample.sample_name
            copy_if_exists(sample.sample_dir / "videos" / "rgb.mp4", sample_assets / "before.mp4")
            copy_if_exists(sample.sample_dir / "scene_input.json", sample_assets / "before.scene_input.json")
            copy_if_exists(sample.sample_dir / "metadata.json", sample_assets / "before.metadata.json")

            repair_entry: dict[str, Any] = {
                "sample_name": sample.sample_name,
                "case_name": sample.case_name,
                "motion_category": sample.motion_category,
                "before": sample.to_json(),
                "status": "failed",
                "assets": {
                    "before_video": str((sample_assets / "before.mp4").relative_to(report_dir)).replace("\\", "/"),
                },
            }

            success = False
            for attempt_idx, mult in enumerate(build_attempt_schedule(sample)):
                staged_sample_dir = relative_stage_case(staging_root, sample.sample_name)
                if staged_sample_dir.exists():
                    shutil.rmtree(staged_sample_dir)
                cmd = build_regen_command(
                    python_bin=args.python_bin,
                    try1_script=args.try1_script,
                    dataset_root=args.dataset_root,
                    staging_root=staging_root,
                    sample=sample,
                    camera_distance_mult=mult,
                )
                log_path = logs_dir / f"{sample.sample_name}__attempt{attempt_idx:02d}.log"
                rc = run_cmd(cmd, log_path)
                if rc != 0:
                    repair_entry["last_returncode"] = rc
                    repair_entry["last_log_path"] = str(log_path)
                    continue
                if not staged_sample_dir.exists():
                    repair_entry["last_returncode"] = rc
                    repair_entry["last_log_path"] = str(log_path)
                    continue
                staged_eval = evaluate_sample(staged_sample_dir, args.margin)
                if not staged_eval.last_safe_margin:
                    repair_entry["last_attempt_eval"] = staged_eval.to_json()
                    repair_entry["last_log_path"] = str(log_path)
                    continue

                backup_dir = report_dir / "replaced_originals" / sample.sample_name
                if backup_dir.exists():
                    shutil.rmtree(backup_dir)
                shutil.copytree(sample.sample_dir, backup_dir)
                if sample.sample_dir.exists():
                    shutil.rmtree(sample.sample_dir)
                shutil.copytree(staged_sample_dir, sample.sample_dir)

                copy_if_exists(sample.sample_dir / "videos" / "rgb.mp4", sample_assets / "after.mp4")
                copy_if_exists(sample.sample_dir / "scene_input.json", sample_assets / "after.scene_input.json")
                copy_if_exists(sample.sample_dir / "metadata.json", sample_assets / "after.metadata.json")
                repair_entry["after"] = staged_eval.to_json()
                repair_entry["status"] = "repaired"
                repair_entry["camera_distance_mult"] = float(mult)
                repair_entry["assets"]["after_video"] = str((sample_assets / "after.mp4").relative_to(report_dir)).replace("\\", "/")
                repair_entry["log_path"] = str(log_path)
                success = True
                if not args.keep_staging and staged_sample_dir.exists():
                    shutil.rmtree(staged_sample_dir)
                break

            report["repairs"].append(repair_entry)
            if not success:
                print(f"[repair] failed sample={sample.sample_name}", file=sys.stderr)

    after_results = scan_count01(args.count01_root, args.margin)
    after_summary = summarize(after_results, args.margin)
    after_scoped = after_results if selected_names is None else [
        item for item in after_results if item.sample_name in selected_names
    ]
    report["summary_after_full"] = after_summary
    report["summary_after_scoped"] = summarize(after_scoped, args.margin)
    dump_json(report_dir / "repair_report.json", report)
    dump_json(report_dir / "scan_after.json", {
        "summary_full": after_summary,
        "summary_scoped": summarize(after_scoped, args.margin),
        "samples": [item.to_json() for item in after_scoped],
    })
    (report_dir / "index.html").write_text(render_html(report), encoding="utf-8")
    print(json.dumps(summarize(after_scoped, args.margin), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
