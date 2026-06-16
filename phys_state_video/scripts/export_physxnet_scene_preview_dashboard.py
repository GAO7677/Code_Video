#!/usr/bin/env python3
from __future__ import annotations

import argparse
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np


DEFAULT_OBJECT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/physxnet_genesis_mpm0613/19925")
DEFAULT_OUTPUT_DIR = Path("/tmp/physxnet_scene_preview_dashboard_19925")
DEFAULT_PORT = 18891


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export a static dashboard for PhysXNet Genesis scene preview videos and kinematics ground truth."
    )
    parser.add_argument(
        "--object-root",
        default=str(DEFAULT_OBJECT_ROOT),
        help="Object export root that contains meta/, scene_preview/, parts/, rigid/, soft/ ...",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to write report.json and index.html.",
    )
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Display-only local port.")
    parser.add_argument("--clean", action="store_true", help="Delete output dir before export.")
    return parser.parse_args()


def safe_read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_symlink(src: Path, dst: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_symlink() and dst.resolve() == src.resolve():
            return
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=src.is_dir())


def short_name(path: Path) -> str:
    return path.name


def fmt_num(value: float, digits: int = 3) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.{digits}f}"


def svg_polyline(values: np.ndarray, width: int = 520, height: int = 130, stroke: str = "#0d5b54") -> str:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return "<svg viewBox='0 0 520 130' class='plot'><text x='18' y='70'>empty</text></svg>"
    finite = np.isfinite(arr)
    if not np.any(finite):
        return "<svg viewBox='0 0 520 130' class='plot'><text x='18' y='70'>nan</text></svg>"
    valid = arr[finite]
    vmin = float(np.min(valid))
    vmax = float(np.max(valid))
    if abs(vmax - vmin) < 1e-9:
        vmax = vmin + 1.0
    xs = np.linspace(18.0, width - 12.0, num=arr.size)
    pts = []
    for x, y in zip(xs, arr.tolist()):
        if not math.isfinite(y):
            continue
        norm = (float(y) - vmin) / (vmax - vmin)
        py = (height - 18.0) - norm * (height - 34.0)
        pts.append(f"{x:.2f},{py:.2f}")
    baseline = height - 18
    return (
        f"<svg viewBox='0 0 {width} {height}' class='plot' preserveAspectRatio='none'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' rx='14' ry='14' fill='rgba(13,91,84,0.05)'/>"
        f"<line x1='18' y1='{baseline}' x2='{width - 12}' y2='{baseline}' stroke='rgba(32,27,23,0.18)' stroke-width='1'/>"
        f"<polyline fill='none' stroke='{stroke}' stroke-width='3' points='{' '.join(pts)}'/>"
        f"<text x='18' y='18' class='plot-label'>min {html.escape(fmt_num(vmin, 3))}</text>"
        f"<text x='{width - 118}' y='18' class='plot-label'>max {html.escape(fmt_num(vmax, 3))}</text>"
        "</svg>"
    )


def svg_flags(values: np.ndarray, width: int = 520, height: int = 58) -> str:
    arr = np.asarray(values, dtype=np.uint8).reshape(-1)
    if arr.size == 0:
        return "<svg viewBox='0 0 520 58' class='plot'><text x='18' y='32'>empty</text></svg>"
    bar_w = max(1.0, (width - 20.0) / max(1, arr.size))
    rects = []
    for idx, flag in enumerate(arr.tolist()):
        x = 10.0 + idx * bar_w
        color = "#b44d2a" if int(flag) > 0 else "rgba(13,91,84,0.12)"
        rects.append(f"<rect x='{x:.2f}' y='10' width='{bar_w:.2f}' height='38' fill='{color}'/>")
    active = int(arr.sum())
    return (
        f"<svg viewBox='0 0 {width} {height}' class='plot' preserveAspectRatio='none'>"
        f"<rect x='0' y='0' width='{width}' height='{height}' rx='14' ry='14' fill='rgba(185,107,52,0.08)'/>"
        f"{''.join(rects)}"
        f"<text x='18' y='18' class='plot-label'>collision-ish frames: {active}/{arr.size}</text>"
        "</svg>"
    )


def build_case_payload(scene_preview_dir: Path, meta_path: Path, case_plan_index: dict[str, dict[str, Any]]) -> dict[str, Any]:
    meta = safe_read_json(meta_path)
    npz_path = Path(meta["kinematics_npz_path"])
    payload = np.load(npz_path)
    com_pos = np.asarray(payload["com_pos"], dtype=np.float32)
    linear_vel = np.asarray(payload["linear_vel"], dtype=np.float32)
    angular_vel = np.asarray(payload["angular_vel"], dtype=np.float32)
    collision_flags = np.asarray(payload["collision_flags"], dtype=np.uint8)
    kinetic_energy = np.asarray(payload["kinetic_energy"], dtype=np.float32)
    potential_energy = np.asarray(payload["potential_energy"], dtype=np.float32)
    total_energy = np.asarray(payload["total_energy"], dtype=np.float32)

    if linear_vel.size:
        speed = np.linalg.norm(linear_vel, axis=-1)
        max_speed_per_object = np.max(speed, axis=0)
        mean_speed_per_object = np.mean(speed, axis=0)
    else:
        max_speed_per_object = np.zeros((linear_vel.shape[1] if linear_vel.ndim >= 2 else 0,), dtype=np.float32)
        mean_speed_per_object = np.zeros_like(max_speed_per_object)
    if angular_vel.size:
        ang_speed = np.linalg.norm(angular_vel, axis=-1)
        max_ang_speed_per_object = np.max(ang_speed, axis=0)
    else:
        max_ang_speed_per_object = np.zeros((angular_vel.shape[1] if angular_vel.ndim >= 2 else 0,), dtype=np.float32)

    object_rows = []
    object_ids = list(meta.get("object_ids", []))
    object_types = list(meta.get("object_types", []))
    object_sources = list(meta.get("object_sources", []))
    for idx, object_id in enumerate(object_ids):
        object_rows.append(
            {
                "object_id": str(object_id),
                "object_type": str(object_types[idx]) if idx < len(object_types) else "",
                "object_source": str(object_sources[idx]) if idx < len(object_sources) else "",
                "max_speed": float(max_speed_per_object[idx]) if idx < len(max_speed_per_object) else 0.0,
                "mean_speed": float(mean_speed_per_object[idx]) if idx < len(mean_speed_per_object) else 0.0,
                "max_angular_speed": float(max_ang_speed_per_object[idx]) if idx < len(max_ang_speed_per_object) else 0.0,
            }
        )

    case_name = str(meta.get("case_name", meta_path.stem.replace("_kinematics_meta", "")))
    plan_cfg = case_plan_index.get(case_name, {})
    rel_video = Path("artifact") / "scene_preview" / Path(meta["video_path"]).name
    rel_npz = Path("artifact") / "scene_preview" / npz_path.name
    rel_meta = Path("artifact") / "scene_preview" / meta_path.name

    return {
        "case_name": case_name,
        "scene_label": str(meta.get("scene_label", case_name)),
        "video_relpath": str(rel_video),
        "kinematics_relpath": str(rel_npz),
        "kinematics_meta_relpath": str(rel_meta),
        "frames": int(com_pos.shape[0]),
        "tracked_objects": int(com_pos.shape[1]) if com_pos.ndim >= 2 else 0,
        "fps": int(meta.get("fps", 0)),
        "frame_dt_seconds": float(meta.get("frame_dt_seconds", 0.0)),
        "collision_frame_count": int(np.sum(collision_flags)),
        "collision_frame_ratio": float(np.mean(collision_flags)) if collision_flags.size else 0.0,
        "energy_min": float(np.min(total_energy)) if total_energy.size else 0.0,
        "energy_max": float(np.max(total_energy)) if total_energy.size else 0.0,
        "object_rows": object_rows,
        "case_cfg": plan_cfg,
        "plots": {
            "total_energy_svg": svg_polyline(total_energy, stroke="#b96b34"),
            "kinetic_energy_svg": svg_polyline(kinetic_energy, stroke="#0d5b54"),
            "potential_energy_svg": svg_polyline(potential_energy, stroke="#4d7ea8"),
            "collision_svg": svg_flags(collision_flags),
        },
    }


def build_report(object_root: Path, output_dir: Path, port: int) -> dict[str, Any]:
    meta_path = object_root / "meta" / "metadata.json"
    summary_path = object_root / f"{object_root.name}_summary.json"
    scene_preview_dir = object_root / "scene_preview"
    if not meta_path.exists():
        raise FileNotFoundError(f"missing metadata.json under {object_root}")
    if not scene_preview_dir.exists():
        raise FileNotFoundError(f"missing scene_preview under {object_root}")

    metadata = safe_read_json(meta_path)
    summary = safe_read_json(summary_path) if summary_path.exists() else {}
    preview_case_plan_path = scene_preview_dir / "preview_case_plan.json"
    preview_case_plan = safe_read_json(preview_case_plan_path) if preview_case_plan_path.exists() else []
    case_plan_index = {
        str(item.get("case_name", f"case{idx:03d}")): item
        for idx, item in enumerate(preview_case_plan)
        if isinstance(item, dict)
    }
    case_meta_paths = sorted(scene_preview_dir.glob("*_kinematics_meta.json"))
    cases = [build_case_payload(scene_preview_dir, path, case_plan_index) for path in case_meta_paths]

    exported_gt = [
        {
            "name": "preview mp4",
            "shape": "per-case video",
            "meaning": "Rendered Genesis preview video for each case.",
        },
        {
            "name": "com_pos",
            "shape": "[T, N, 3]",
            "meaning": "Tracked object center-of-mass positions in world coordinates.",
        },
        {
            "name": "linear_vel",
            "shape": "[T, N, 3]",
            "meaning": "Tracked object linear velocities in world coordinates.",
        },
        {
            "name": "angular_vel",
            "shape": "[T, N, 3]",
            "meaning": "Tracked object angular velocities in world coordinates.",
        },
        {
            "name": "collision_flags",
            "shape": "[T]",
            "meaning": "Heuristic collision/contact impulse indicator near each saved frame.",
        },
        {
            "name": "kinetic_energy",
            "shape": "[T]",
            "meaning": "Sum of tracked entities kinetic energy.",
        },
        {
            "name": "potential_energy",
            "shape": "[T]",
            "meaning": "Sum of tracked entities gravitational potential energy.",
        },
        {
            "name": "total_energy",
            "shape": "[T]",
            "meaning": "Kinetic plus gravitational potential energy.",
        },
        {
            "name": "preview_case_plan.json",
            "shape": "json",
            "meaning": "Per-case scene config, motion toggles, offsets, and placement settings.",
        },
        {
            "name": "metadata.json",
            "shape": "json",
            "meaning": "Object decomposition, rigid/soft part metadata, solver policy, and runtime application info.",
        },
    ]

    return {
        "title": f"PhysXNet Scene Preview Dashboard · object {object_root.name}",
        "object_root": str(object_root),
        "output_dir": str(output_dir),
        "port": int(port),
        "object_id": str(metadata.get("object_id", object_root.name)),
        "object_name": str(metadata.get("object_name", "")),
        "category": str(metadata.get("category", "")),
        "bbox_min": metadata.get("object_bbox_min", []),
        "bbox_max": metadata.get("object_bbox_max", []),
        "rigid_part_count": int(len(metadata.get("rigid_part_links", []))),
        "soft_part_count": int(len(metadata.get("soft_parts", []))),
        "preview_videos": summary.get("preview_videos", []),
        "exported_gt": exported_gt,
        "preview_case_plan_relpath": "artifact/scene_preview/preview_case_plan.json",
        "metadata_relpath": "artifact/meta/metadata.json",
        "summary_relpath": f"artifact/{object_root.name}_summary.json" if summary_path.exists() else "",
        "case_count": len(cases),
        "cases": cases,
        "mode": "physxnet_scene_preview_dashboard",
    }


def render_gt_table(exported_gt: list[dict[str, Any]]) -> str:
    rows = []
    for item in exported_gt:
        rows.append(
            f"""
            <tr>
              <td>{html.escape(str(item['name']))}</td>
              <td>{html.escape(str(item['shape']))}</td>
              <td>{html.escape(str(item['meaning']))}</td>
            </tr>
            """
        )
    return "".join(rows)


def render_object_rows(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "<p class='empty'>No tracked objects in this case.</p>"
    parts = []
    for item in rows:
        parts.append(
            f"""
            <tr>
              <td>{html.escape(str(item['object_id']))}</td>
              <td>{html.escape(str(item['object_type']))}</td>
              <td>{html.escape(str(item['object_source']))}</td>
              <td>{html.escape(fmt_num(float(item['max_speed']), 3))}</td>
              <td>{html.escape(fmt_num(float(item['mean_speed']), 3))}</td>
              <td>{html.escape(fmt_num(float(item['max_angular_speed']), 3))}</td>
            </tr>
            """
        )
    return (
        "<table><thead><tr><th>object_id</th><th>type</th><th>source</th><th>max |v|</th><th>mean |v|</th><th>max |w|</th></tr></thead>"
        f"<tbody>{''.join(parts)}</tbody></table>"
    )


def render_case_cfg(case_cfg: dict[str, Any]) -> str:
    if not case_cfg:
        return "<p class='empty'>No preview_case_plan entry found.</p>"
    wanted_keys = [
        "scene_label",
        "object_fixed",
        "use_entry_motion",
        "placed_pos_offset",
        "object_euler_deg",
        "entry_linear_velocity",
        "entry_angular_velocity",
        "gravity_z_override",
        "striker_speed_override",
        "case_notes",
    ]
    rows = []
    for key in wanted_keys:
        if key not in case_cfg:
            continue
        rows.append(
            f"<tr><td>{html.escape(str(key))}</td><td><pre>{html.escape(json.dumps(case_cfg[key], ensure_ascii=False, indent=2))}</pre></td></tr>"
        )
    return f"<table><tbody>{''.join(rows)}</tbody></table>"


def build_html(report: dict[str, Any]) -> str:
    case_cards = []
    for case in report["cases"]:
        case_cards.append(
            f"""
            <article class="card case-card" id="{html.escape(case['case_name'])}">
              <div class="card-head">
                <div>
                  <div class="eyebrow">{html.escape(case['scene_label'])}</div>
                  <h2>{html.escape(case['case_name'])}</h2>
                </div>
                <div class="stats">
                  <span>frames={case['frames']}</span>
                  <span>tracked={case['tracked_objects']}</span>
                  <span>fps={case['fps']}</span>
                  <span>dt={fmt_num(case['frame_dt_seconds'], 3)}s</span>
                  <span>collision-ish={case['collision_frame_count']}</span>
                </div>
              </div>
              <div class="video-wrap">
                <video controls playsinline preload="metadata">
                  <source src="{html.escape(case['video_relpath'])}" type="video/mp4">
                </video>
              </div>
              <div class="link-row">
                <a href="{html.escape(case['video_relpath'])}">video.mp4</a>
                <a href="{html.escape(case['kinematics_relpath'])}">kinematics.npz</a>
                <a href="{html.escape(case['kinematics_meta_relpath'])}">kinematics_meta.json</a>
              </div>
              <div class="plot-grid">
                <section class="plot-panel">
                  <h3>Total Energy</h3>
                  {case['plots']['total_energy_svg']}
                </section>
                <section class="plot-panel">
                  <h3>Kinetic Energy</h3>
                  {case['plots']['kinetic_energy_svg']}
                </section>
                <section class="plot-panel">
                  <h3>Potential Energy</h3>
                  {case['plots']['potential_energy_svg']}
                </section>
                <section class="plot-panel">
                  <h3>Collision Flags</h3>
                  {case['plots']['collision_svg']}
                </section>
              </div>
              <div class="detail-grid">
                <section class="detail-panel">
                  <h3>Tracked Objects</h3>
                  {render_object_rows(case['object_rows'])}
                </section>
                <section class="detail-panel">
                  <h3>Case Config</h3>
                  {render_case_cfg(case['case_cfg'])}
                </section>
              </div>
            </article>
            """
        )

    summary_link = ""
    if report.get("summary_relpath"):
        summary_link = f'<a href="{html.escape(str(report["summary_relpath"]))}">summary.json</a>'

    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(report['title'])}</title>
  <style>
    :root {{
      --bg0: #f7f2e9;
      --bg1: #eadfce;
      --panel: rgba(255, 251, 245, 0.96);
      --line: #ddcfbc;
      --ink: #201b17;
      --muted: #6d665d;
      --accent: #0d5b54;
      --accent2: #b96b34;
      --accent3: #4d7ea8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at left top, rgba(185, 107, 52, 0.12), transparent 24%),
        radial-gradient(circle at right top, rgba(13, 91, 84, 0.12), transparent 28%),
        linear-gradient(180deg, var(--bg0) 0%, var(--bg1) 100%);
    }}
    a {{
      color: var(--accent);
      text-decoration: none;
      font-weight: 700;
    }}
    .page {{
      max-width: 1680px;
      margin: 0 auto;
      padding: 28px;
    }}
    .hero, .card, .section {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 20px;
    }}
    .hero {{
      padding: 24px;
      margin-bottom: 20px;
    }}
    .hero h1 {{
      margin: 0 0 10px;
      font-size: 34px;
    }}
    .lead {{
      margin: 0;
      color: var(--muted);
      line-height: 1.75;
      max-width: 1180px;
    }}
    .pill-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 16px;
    }}
    .pill {{
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 8px 12px;
      font-size: 13px;
      background: rgba(255,255,255,0.55);
    }}
    .hero-links {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin-top: 16px;
    }}
    .section {{
      padding: 20px;
      margin-bottom: 20px;
    }}
    .section h2, .card h2, .plot-panel h3, .detail-panel h3 {{
      margin: 0 0 12px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      border-top: 1px solid var(--line);
      padding: 10px 12px;
      vertical-align: top;
      text-align: left;
    }}
    thead th {{
      border-top: 0;
      color: var(--muted);
      font-weight: 700;
    }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: "IBM Plex Mono", "SFMono-Regular", monospace;
      font-size: 12px;
      color: #2b2621;
    }}
    .case-grid {{
      display: grid;
      grid-template-columns: 1fr;
      gap: 20px;
    }}
    .case-card {{
      padding: 20px;
    }}
    .card-head {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 16px;
      margin-bottom: 14px;
    }}
    .eyebrow {{
      color: var(--accent2);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      margin-bottom: 8px;
    }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      justify-content: flex-end;
    }}
    .stats span {{
      padding: 7px 10px;
      border: 1px solid var(--line);
      border-radius: 999px;
      font-size: 13px;
      background: rgba(255,255,255,0.6);
    }}
    .video-wrap {{
      border-radius: 16px;
      overflow: hidden;
      background: #131313;
      border: 1px solid rgba(0,0,0,0.08);
    }}
    video {{
      width: 100%;
      display: block;
      max-height: 760px;
      background: #111;
    }}
    .link-row {{
      display: flex;
      flex-wrap: wrap;
      gap: 16px;
      margin: 12px 0 10px;
    }}
    .plot-grid {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 14px;
      margin-top: 10px;
    }}
    .plot-panel, .detail-panel {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px;
      background: rgba(255,255,255,0.55);
    }}
    .plot {{
      width: 100%;
      height: auto;
      display: block;
    }}
    .plot-label {{
      fill: #6d665d;
      font-size: 11px;
      font-family: "IBM Plex Sans", sans-serif;
    }}
    .detail-grid {{
      display: grid;
      grid-template-columns: 1.15fr 1fr;
      gap: 14px;
      margin-top: 14px;
    }}
    .empty {{
      color: var(--muted);
      margin: 0;
    }}
    @media (max-width: 980px) {{
      .plot-grid {{
        grid-template-columns: 1fr;
      }}
      .detail-grid {{
        grid-template-columns: 1fr;
      }}
      .card-head {{
        flex-direction: column;
      }}
      .stats {{
        justify-content: flex-start;
      }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <h1>{html.escape(report['title'])}</h1>
      <p class="lead">这个页面把当前仿真导出的真值优先按视频组织展示。每个 case 主位是 Genesis 预览 mp4，下面补充对应的运动学真值、碰撞启发标记、能量曲线和生成配置，便于先看现象，再追数值。</p>
      <div class="pill-row">
        <div class="pill">object_id={html.escape(str(report['object_id']))}</div>
        <div class="pill">object_name={html.escape(str(report['object_name']))}</div>
        <div class="pill">category={html.escape(str(report['category']))}</div>
        <div class="pill">rigid_parts={report['rigid_part_count']}</div>
        <div class="pill">soft_parts={report['soft_part_count']}</div>
        <div class="pill">cases={report['case_count']}</div>
        <div class="pill">port={report['port']}</div>
      </div>
      <div class="hero-links">
        <a href="{html.escape(str(report['metadata_relpath']))}">metadata.json</a>
        <a href="{html.escape(str(report['preview_case_plan_relpath']))}">preview_case_plan.json</a>
        {summary_link}
      </div>
    </section>
    <section class="section">
      <h2>导出的真值</h2>
      <table>
        <thead>
          <tr><th>name</th><th>shape</th><th>meaning</th></tr>
        </thead>
        <tbody>
          {render_gt_table(report['exported_gt'])}
        </tbody>
      </table>
    </section>
    <section class="case-grid">
      {''.join(case_cards)}
    </section>
  </div>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    object_root = Path(args.object_root)
    output_dir = Path(args.output_dir)
    if args.clean and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_symlink(object_root, output_dir / "artifact")
    report = build_report(object_root=object_root, output_dir=output_dir, port=int(args.port))
    (output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_dir / "index.html").write_text(build_html(report), encoding="utf-8")
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
