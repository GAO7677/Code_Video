#!/usr/bin/env python3
# 用途：对指定样本重渲染多视角视频与页面。
from __future__ import annotations

import argparse
import copy
import html
import json
import math
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from generators.try1_physxnet_articulation_mpm0417 import (
    _preview_case_bundles,
    build_preview_case_configs,
    build_argparser,
    ensure_dir,
    prepare_physxnet_object,
    simulate_in_genesis,
)
from core.utils_io import load_json, write_json


FORMAL_DATASET_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases"
)
PORTAL_ROOT = Path("/data/gaoya/AAA_test_video/portal_hub")


def _formal_case_info(case_dir: Path) -> Dict[str, Any]:
    if not case_dir.exists():
        raise FileNotFoundError(case_dir)
    scene_input_path = case_dir / "scene_input.json"
    metadata_path = case_dir / "metadata.json"
    if not scene_input_path.exists():
        raise FileNotFoundError(scene_input_path)
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    scene_input = load_json(scene_input_path)
    metadata = load_json(metadata_path)
    sample_name = str(scene_input.get("sample_name") or case_dir.name)
    object_id = str(scene_input.get("object_id") or sample_name.split("__case", 1)[0])
    case_name = str(scene_input.get("case_name") or sample_name.split("__", 1)[1])
    bucket_name = str(case_dir.parent.name)
    scene_composition = str(case_dir.parent.parent.name)
    target_count = int(metadata.get("num_objects") or int(bucket_name.split("_")[-1]))
    source_camera = dict(scene_input.get("camera") or metadata.get("camera") or {})
    counterfactual = dict(scene_input.get("counterfactual") or metadata.get("counterfactual") or {})
    return {
        "case_dir": case_dir,
        "scene_input": scene_input,
        "metadata": metadata,
        "sample_name": sample_name,
        "object_id": object_id,
        "case_name": case_name,
        "bucket_name": bucket_name,
        "scene_composition": scene_composition,
        "target_count": target_count,
        "source_camera": source_camera,
        "counterfactual": counterfactual,
    }


def _framing_distance_for_fov(camera_cfg: Dict[str, Any], target_fov_deg: float) -> float:
    pos = camera_cfg.get("pos")
    lookat = camera_cfg.get("lookat")
    fov = camera_cfg.get("fov")
    if pos is None or lookat is None or fov is None:
        return 12.0
    pos_v = [float(x) for x in pos]
    lookat_v = [float(x) for x in lookat]
    distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(pos_v, lookat_v)))
    source_half_height = max(1e-4, distance * math.tan(math.radians(float(fov)) / 2.0))
    return max(6.0, source_half_height / max(1e-4, math.tan(math.radians(float(target_fov_deg)) / 2.0)))


def _make_view_specs(camera_cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    lookat = [float(x) for x in camera_cfg.get("lookat", [0.0, 0.0, 0.3])]
    base_pos = [float(x) for x in camera_cfg.get("pos", [2.2, -2.2, 1.1])]
    approx_ortho_fov = 12.0
    # Keep the pseudo-orthographic camera within Genesis' current far plane.
    ortho_distance = min(_framing_distance_for_fov(camera_cfg, approx_ortho_fov), 16.0)
    return [
        {
            "tag": "default_oblique",
            "title": "Default Oblique",
            "note": "original simulation camera",
            "camera_pos": None,
            "camera_lookat": None,
            "camera_up": None,
            "camera_fov": None,
        },
        {
            "tag": "xoy_top_approx_ortho",
            "title": "XOY Top",
            "note": "simulation rerun, approx orthographic top view",
            "camera_pos": [lookat[0], lookat[1], lookat[2] + ortho_distance],
            "camera_lookat": lookat,
            "camera_up": [0.0, 1.0, 0.0],
            "camera_fov": approx_ortho_fov,
        },
        {
            "tag": "yoz_side_approx_ortho",
            "title": "YOZ Side",
            "note": "simulation rerun, approx orthographic side view",
            "camera_pos": [lookat[0] + ortho_distance, lookat[1], lookat[2]],
            "camera_lookat": lookat,
            "camera_up": [0.0, 0.0, 1.0],
            "camera_fov": approx_ortho_fov,
        },
        {
            "tag": "xoz_front_approx_ortho",
            "title": "XOZ Front",
            "note": "simulation rerun, approx orthographic front view",
            "camera_pos": [lookat[0], lookat[1] - ortho_distance, lookat[2]],
            "camera_lookat": lookat,
            "camera_up": [0.0, 0.0, 1.0],
            "camera_fov": approx_ortho_fov,
        },
        {
            "tag": "reverse_oblique",
            "title": "Reverse Oblique",
            "note": "simulation rerun, opposite diagonal view",
            "camera_pos": [-base_pos[0], -base_pos[1], base_pos[2]],
            "camera_lookat": lookat,
            "camera_up": [0.0, 0.0, 1.0],
            "camera_fov": float(camera_cfg.get("fov", 35.0)),
        },
    ]


def _find_case_cfg(args: argparse.Namespace, prepared: Any, case_name: str) -> Dict[str, Any]:
    bundle_cases: List[Dict[str, Any]] = []
    for bundle in _preview_case_bundles(args):
        bundle_args = argparse.Namespace(**vars(args))
        bundle_args.disable_striker = bool(bundle["disable_striker"])
        bundle_args.case_scene_mode = str(bundle["case_scene_mode"])
        bundle_args.num_random_cases = int(bundle["num_random_cases"])
        bundle_args.case_index_filter = list(bundle["case_index_filter"]) if bundle["case_index_filter"] is not None else None
        bundle_args.rigid_target_object_count = bundle.get("rigid_target_object_count", None)
        cases = build_preview_case_configs(
            prepared=prepared,
            output_root=Path(args.output_root),
            object_fixed=bool(bundle_args.object_fixed),
            args=bundle_args,
        )
        bundle_cases.extend(cases)
    for case_cfg in bundle_cases:
        if str(case_cfg.get("case_name")) == case_name:
            return case_cfg
    raise RuntimeError(f"Unable to reconstruct case config for {case_name}")


def _infer_requested_case_indices(case_info: Dict[str, Any]) -> List[int]:
    counterfactual = dict(case_info.get("counterfactual") or {})
    for key in ("parent_case_index", "parent_case_id"):
        value = counterfactual.get(key)
        if isinstance(value, int):
            return [int(value)]
        if isinstance(value, str) and value.strip().isdigit():
            return [int(value.strip())]
    case_name = str(case_info.get("case_name") or "")
    match = re.match(r"case(\d+)_", case_name)
    if match:
        return [int(match.group(1))]
    return []


def _build_runtime_args(case_info: Dict[str, Any], rerender_root: Path) -> argparse.Namespace:
    parser = build_argparser()
    cli_args = [
        "--physx_root",
        "/data/gaoya/dataset/Caoza-PhysX-3D/PhysXNet",
        "--version",
        "version_1",
        "--object_id",
        str(case_info["object_id"]),
        "--output_root",
        str(FORMAL_DATASET_ROOT),
        "--run_genesis",
        "--generate_all_count_motion_cases",
        "--rigid_count_filter",
        str(case_info["target_count"]),
        "--prefer_existing_runtime_meshes",
        "--dt",
        "0.003",
        "--substeps",
        "40",
        "--ball_posx",
        "0.03",
        "--steps",
        "12",
        "--fps",
        "12",
        "--simulator_mode",
        "rigid",
    ]
    requested_case_indices = _infer_requested_case_indices(case_info)
    if requested_case_indices:
        cli_args.extend(["--case_index_filter", *[str(idx) for idx in requested_case_indices]])
    if dict(case_info.get("counterfactual") or {}):
        cli_args.append("--enable_counterfactual_cases")
    args = parser.parse_args(cli_args)
    args.rerender_output_root = str(rerender_root)
    return args


def _write_html(portal_dir: Path, case_info: Dict[str, Any], view_records: List[Dict[str, Any]]) -> None:
    cards = []
    for rec in view_records:
        video_rel = html.escape(rec["video_rel"])
        meta_rel = html.escape(rec["metadata_rel"])
        camera_json = html.escape(json.dumps(rec["camera"], ensure_ascii=False, indent=2))
        cards.append(
            f"""
            <section class="card">
              <div class="card-head">
                <h2>{html.escape(rec['title'])}</h2>
                <p>{html.escape(rec['note'])}</p>
              </div>
              <video controls preload="metadata" playsinline src="{video_rel}"></video>
              <div class="meta">
                <div><strong>tag</strong> {html.escape(rec['tag'])}</div>
                <div><strong>video</strong> <a href="{video_rel}">{video_rel}</a></div>
                <div><strong>metadata</strong> <a href="{meta_rel}">{meta_rel}</a></div>
              </div>
              <pre>{camera_json}</pre>
            </section>
            """
        )

    page = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Simulation Multi-View Rerender</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --ink: #1b1d1f;
      --muted: #5d625c;
      --line: #d4c7b1;
      --card: rgba(255,255,255,0.88);
      --accent: #9e3d22;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(158,61,34,0.12), transparent 30%),
        linear-gradient(180deg, #fbf7f0 0%, var(--bg) 100%);
    }}
    main {{
      max-width: 1500px;
      margin: 0 auto;
      padding: 28px 20px 56px;
    }}
    .hero {{
      margin-bottom: 24px;
      padding: 22px 24px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--card);
      backdrop-filter: blur(8px);
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-family: "IBM Plex Serif", "Noto Serif SC", serif;
      font-size: clamp(28px, 3vw, 40px);
    }}
    .hero p {{
      margin: 8px 0;
      color: var(--muted);
      line-height: 1.5;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(420px, 1fr));
      gap: 18px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--card);
      overflow: hidden;
      box-shadow: 0 18px 50px rgba(47, 38, 27, 0.08);
    }}
    .card-head {{
      padding: 16px 18px 8px;
    }}
    .card-head h2 {{
      margin: 0;
      font-size: 20px;
    }}
    .card-head p {{
      margin: 6px 0 0;
      color: var(--muted);
    }}
    video {{
      display: block;
      width: 100%;
      aspect-ratio: 4 / 3;
      background: #111;
    }}
    .meta {{
      padding: 12px 18px 0;
      display: grid;
      gap: 6px;
      font-size: 14px;
    }}
    pre {{
      margin: 12px 18px 18px;
      padding: 14px;
      border-radius: 14px;
      background: #201f1d;
      color: #f8f3eb;
      overflow: auto;
      font-size: 12px;
      line-height: 1.45;
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    @media (max-width: 640px) {{
      main {{ padding: 16px 12px 36px; }}
      .grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Simulation-time Multi-view Rerender</h1>
      <p><strong>case</strong> {html.escape(case_info['sample_name'])}</p>
      <p><strong>source</strong> {html.escape(str(case_info['case_dir']))}</p>
      <p>These videos are regenerated by rerunning the Genesis simulation with different camera poses, not by redrawing the saved trajectory afterward.</p>
      <p>The top and side views use a narrow-FOV perspective camera to approximate an orthographic framing because the current Genesis camera API only exposes pinhole/thinlens models.</p>
    </section>
    <div class="grid">
      {''.join(cards)}
    </div>
  </main>
</body>
</html>
"""
    (portal_dir / "index.html").write_text(page, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Rerun one formal rigid case with multiple simulation-time camera viewpoints.")
    parser.add_argument("--case_dir", type=str, required=True, help="Formal dataset case directory")
    parser.add_argument("--portal_root", type=str, default=str(PORTAL_ROOT), help="Portal root served by local HTTP server")
    parser.add_argument("--portal_slug", type=str, default="", help="Optional output portal subdirectory name")
    parser.add_argument("--keep_previous", action="store_true", help="Keep previous rerender directory instead of deleting it")
    args = parser.parse_args()

    case_info = _formal_case_info(Path(args.case_dir).resolve())
    portal_root = Path(args.portal_root).resolve()
    portal_slug = args.portal_slug.strip() or f"sim_camera_compare_{case_info['sample_name']}"
    portal_dir = portal_root / portal_slug
    if portal_dir.exists() and not args.keep_previous:
        shutil.rmtree(portal_dir)
    ensure_dir(portal_dir)
    rerender_root = portal_dir / "rerender_outputs"
    ensure_dir(rerender_root)

    runtime_args = _build_runtime_args(case_info, rerender_root)
    prepared = prepare_physxnet_object(
        physx_root=Path(runtime_args.physx_root),
        version=runtime_args.version,
        object_id=str(case_info["object_id"]),
        output_root=Path(runtime_args.output_root),
        voxel_pitch=float(runtime_args.voxel_pitch),
        json_override=Path(runtime_args.json_override) if runtime_args.json_override else None,
        object_scale_mult=float(runtime_args.object_scale_mult),
        fallback_density_kgm3=float(runtime_args.fallback_density_kgm3),
        solver_family_override=runtime_args.solver_family_override,
        all_parts_youngs_threshold_gpa=runtime_args.all_parts_youngs_threshold_gpa,
        rigid_visual_double_sided_shell=True,
        simulator_mode=str(runtime_args.simulator_mode),
    )
    case_cfg = _find_case_cfg(runtime_args, prepared, case_info["case_name"])

    view_records: List[Dict[str, Any]] = []
    for view_spec in _make_view_specs(case_info["source_camera"]):
        view_args = argparse.Namespace(**vars(runtime_args))
        view_args.camera_tag = str(view_spec["tag"])
        view_args.camera_pos_override = copy.deepcopy(view_spec["camera_pos"])
        view_args.camera_lookat_override = copy.deepcopy(view_spec["camera_lookat"])
        view_args.camera_up_override = copy.deepcopy(view_spec["camera_up"])
        view_args.camera_fov_override = view_spec["camera_fov"]
        output_root = rerender_root / str(view_spec["tag"])
        view_args.preview_output_root = str(output_root / "_preview_videos")
        metadata_path = simulate_in_genesis(
            prepared=prepared,
            output_root=output_root,
            steps=int(view_args.steps),
            dt=float(view_args.dt),
            substeps=int(view_args.substeps),
            fps=int(view_args.fps),
            default_friction=float(view_args.default_friction),
            object_fixed=bool(view_args.object_fixed),
            striker_radius=float(view_args.striker_radius),
            striker_speed=float(view_args.striker_speed),
            args=view_args,
            case_cfg=copy.deepcopy(case_cfg),
        )
        case_output_dir = Path(metadata_path).parent
        video_path = case_output_dir / "videos" / "rgb.mp4"
        metadata = load_json(Path(metadata_path))
        view_records.append(
            {
                "tag": str(view_spec["tag"]),
                "title": str(view_spec["title"]),
                "note": str(view_spec["note"]),
                "video_rel": video_path.relative_to(portal_dir).as_posix(),
                "metadata_rel": Path(metadata_path).relative_to(portal_dir).as_posix(),
                "camera": metadata.get("camera", {}),
            }
        )

    _write_html(portal_dir, case_info, view_records)
    manifest = {
        "case": case_info["sample_name"],
        "source_case_dir": str(case_info["case_dir"]),
        "portal_dir": str(portal_dir),
        "views": view_records,
    }
    write_json(portal_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
