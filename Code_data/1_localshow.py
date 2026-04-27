#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import html
import json
import posixpath
import urllib.parse
from collections import defaultdict
from dataclasses import dataclass
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from fnmatch import fnmatch
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence


@dataclass
class VideoItem:
    category: str
    scene: str
    scene_label: str
    filename: str
    rel_path: str
    count_bucket: str = ""
    case_notes: str = ""
    sim_label: str = ""
    energy_notes: str = ""
    energy_chart_svg: str = ""


def _as_url_path(rel_path: str) -> str:
    return "/" + urllib.parse.quote(rel_path, safe="/")


def _slugify(value: str) -> str:
    out = []
    for ch in value:
        if ch.isalnum():
            out.append(ch.lower())
        elif ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("-")
    return "".join(out).strip("-") or "section"


def _build_energy_svg(drift: Sequence[float], *, width: int = 320, height: int = 120) -> str:
    if not drift:
        return ""
    pad_l = 28
    pad_r = 10
    pad_t = 10
    pad_b = 22
    inner_w = max(1, width - pad_l - pad_r)
    inner_h = max(1, height - pad_t - pad_b)
    ymax = max(max(float(v) for v in drift), 0.05)
    points = []
    for i, val in enumerate(drift):
        x = pad_l if len(drift) == 1 else pad_l + inner_w * i / (len(drift) - 1)
        y = pad_t + inner_h * (1.0 - min(max(float(val) / ymax, 0.0), 1.0))
        points.append(f"{x:.1f},{y:.1f}")
    threshold_y = pad_t + inner_h * (1.0 - min(0.05 / ymax, 1.0))
    return (
        f'<svg class="energy-chart-svg" viewBox="0 0 {width} {height}" preserveAspectRatio="none" aria-label="energy drift chart">'
        f'<rect x="0" y="0" width="{width}" height="{height}" rx="10" ry="10" fill="#fffaf5" stroke="rgba(182, 97, 57, 0.18)"/>'
        f'<line x1="{pad_l}" y1="{pad_t + inner_h}" x2="{width - pad_r}" y2="{pad_t + inner_h}" stroke="#c9b8ab" stroke-width="1"/>'
        f'<line x1="{pad_l}" y1="{pad_t}" x2="{pad_l}" y2="{pad_t + inner_h}" stroke="#c9b8ab" stroke-width="1"/>'
        f'<line x1="{pad_l}" y1="{threshold_y:.1f}" x2="{width - pad_r}" y2="{threshold_y:.1f}" stroke="#d66b4d" stroke-width="1.5" stroke-dasharray="4 3"/>'
        f'<polyline fill="none" stroke="#b66139" stroke-width="2.5" points="{" ".join(points)}"/>'
        f'<text x="{pad_l}" y="{height - 6}" font-size="10" fill="#6e6259">帧 0</text>'
        f'<text x="{width - pad_r}" y="{height - 6}" text-anchor="end" font-size="10" fill="#6e6259">帧 {len(drift) - 1}</text>'
        f'<text x="6" y="{pad_t + 8}" font-size="10" fill="#6e6259">{ymax:.1%}</text>'
        f'<text x="6" y="{threshold_y - 4:.1f}" font-size="10" fill="#d66b4d">5%</text>'
        f'</svg>'
    )


def _video_name_filter(mp4: Path, allowed_names: Sequence[str]) -> bool:
    if not allowed_names:
        return True
    return any(fnmatch(mp4.name, pattern) for pattern in allowed_names)


def _infer_sim_label(scene_root: Path) -> str:
    metadata_path = scene_root / "metadata" / "scene.json"
    if metadata_path.exists():
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            solver_name = str(data.get("solver_name", "")).lower()
            if solver_name in {"sph", "pbd"} or "liquid_preset" in data:
                return "liquid"
            sim_label = str(data.get("sim_label", "")).strip().lower()
            if sim_label:
                return sim_label
        except Exception:
            pass

    scene_text = str(scene_root).lower()
    if "smoke" in scene_text:
        return "smoke"
    if "liquid" in scene_text:
        return "liquid"
    return ""


def scan_genesis_rigid(root: Path, allowed_names: Sequence[str]) -> List[VideoItem]:
    items: List[VideoItem] = []
    for mp4 in sorted(root.rglob("*.mp4")):
        if not mp4.is_file() or not _video_name_filter(mp4, allowed_names):
            continue
        rel_parts = mp4.resolve().relative_to(root.resolve()).parts
        if len(rel_parts) < 6 or rel_parts[0] != "train":
            continue
        if rel_parts[1] == "rigid":
            if len(rel_parts) < 7 or rel_parts[5] != "videos":
                continue
            category, count_bucket, scene = rel_parts[2], rel_parts[3], rel_parts[4]
        else:
            category, count_bucket, scene = rel_parts[1], rel_parts[2], rel_parts[3]
            if rel_parts[4] != "videos":
                continue
        if category == "rigid":
            continue
        items.append(
            VideoItem(
                category=category,
                scene=scene,
                scene_label=scene,
                filename=mp4.name,
                rel_path=mp4.resolve().relative_to(root.resolve()).as_posix(),
                count_bucket=count_bucket,
            )
        )
    return items


def scan_scene_video(root: Path, allowed_names: Sequence[str]) -> List[VideoItem]:
    items: List[VideoItem] = []
    for mp4 in sorted(root.rglob("*.mp4")):
        if not mp4.is_file() or not _video_name_filter(mp4, allowed_names):
            continue
        rel_parts = mp4.resolve().relative_to(root.resolve()).parts
        if len(rel_parts) < 3:
            continue
        scene = rel_parts[0]
        if rel_parts[1] not in ("video", "videos"):
            continue
        items.append(
            VideoItem(
                category="all_scenes",
                scene=scene,
                scene_label=scene,
                filename=mp4.name,
                rel_path=mp4.resolve().relative_to(root.resolve()).as_posix(),
                sim_label=_infer_sim_label(root / scene),
            )
        )
    return items


def _scene_preview_case_label(filename: str) -> str:
    stem = Path(filename).stem
    parts = stem.split("_")
    for part in parts:
        if part.startswith("case") and len(part) >= 7 and part[4:].isdigit():
            return part
    return stem


def scan_physxnet_preview(root: Path, allowed_names: Sequence[str]) -> List[VideoItem]:
    items: List[VideoItem] = []
    root_resolved = root.resolve()
    notes_cache: Dict[str, Dict[str, Dict[str, str]]] = {}
    energy_map: Dict[str, str] = {}
    energy_chart_map: Dict[str, str] = {}

    report_path = root / "energy_report.json"
    if report_path.exists():
        try:
            report = json.loads(report_path.read_text(encoding="utf-8"))
            for row in report.get("rows", []):
                video_path = str(row.get("video_path", ""))
                if not video_path:
                    continue
                rel_video_path = ""
                try:
                    rel_video_path = Path(video_path).resolve().relative_to(root_resolved).as_posix()
                except Exception:
                    continue
                total = [float(x) for x in row.get("total_energy_per_frame", [])]
                if total:
                    e0 = total[0]
                    drift = [abs(x - e0) / max(abs(e0), 1e-8) for x in total]
                    energy_lines = [
                        f"能量守恒检查: {'是' if float(row.get('rel_energy_drift', 1.0)) <= 0.05 else '否'}",
                        f"初始总能量 E0 = {e0:.3f}",
                        f"最终总能量 ET = {float(row.get('final_total_energy', total[-1])):.3f}",
                        f"最大相对漂移 = {max(drift):.3%}",
                        f"全程相对漂移 = {float(row.get('rel_energy_drift', 0.0)):.3%}",
                        "每帧总能量:",
                        ", ".join(f"{x:.3f}" for x in total),
                        "每帧相对漂移:",
                        ", ".join(f"{x:.3%}" for x in drift),
                    ]
                    collision = row.get("collision_flags", [])
                    if collision:
                        energy_lines.extend([
                            "每帧碰撞标志:",
                            ", ".join(str(int(x)) for x in collision),
                        ])
                    energy_map[rel_video_path] = "\n".join(energy_lines)
                    energy_chart_map[rel_video_path] = _build_energy_svg(drift)
        except Exception:
            pass

    def _load_case_notes(obj_root: Path) -> Dict[str, Dict[str, str]]:
        key = str(obj_root.resolve())
        if key in notes_cache:
            return notes_cache[key]
        plan_path = obj_root / "scene_preview" / "preview_case_plan.json"
        mapping: Dict[str, Dict[str, str]] = {}
        if plan_path.exists():
            try:
                data = json.loads(plan_path.read_text(encoding="utf-8"))
                for entry in data:
                    case_name = entry.get("case_name")
                    if not case_name:
                        continue
                    case_name = str(case_name)
                    info = {
                        "scene_label": str(entry.get("scene_label", case_name)),
                        "case_notes": str(entry.get("case_notes", "")),
                    }
                    variants = {case_name}
                    base = case_name.split("_", 1)[0]
                    variants.add(base)
                    for variant in variants:
                        mapping[variant] = info
            except Exception:
                pass
        notes_cache[key] = mapping
        return mapping

    for mp4 in sorted(root.rglob("*.mp4")):
        if not mp4.is_file() or not _video_name_filter(mp4, allowed_names):
            continue
        rel_parts = mp4.resolve().relative_to(root_resolved).parts
        object_id = None
        object_root: Optional[Path] = None
        if len(rel_parts) >= 3 and rel_parts[1] == "scene_preview":
            object_id = rel_parts[0]
            object_root = root_resolved / object_id
        elif len(rel_parts) >= 2 and rel_parts[0] == "scene_preview":
            object_id = root_resolved.name
            object_root = root_resolved
        else:
            continue
        if object_root is None:
            continue
        notes_map = _load_case_notes(object_root)
        scene_label = _scene_preview_case_label(mp4.name)
        note_info = notes_map.get(scene_label, {"scene_label": scene_label, "case_notes": ""})
        items.append(
            VideoItem(
                category=str(object_id),
                scene=note_info.get("scene_label", scene_label),
                scene_label=note_info.get("scene_label", scene_label),
                filename=mp4.name,
                rel_path=mp4.resolve().relative_to(root_resolved).as_posix(),
                case_notes=note_info.get("case_notes", ""),
                energy_notes=energy_map.get(mp4.resolve().relative_to(root_resolved).as_posix(), ""),
                energy_chart_svg=energy_chart_map.get(mp4.resolve().relative_to(root_resolved).as_posix(), ""),
            )
        )
    return items


SCAN_PROFILES: Dict[str, Callable[[Path, Sequence[str]], List[VideoItem]]] = {
    "genesis_rigid": scan_genesis_rigid,
    "scene_video": scan_scene_video,
    "physxnet_preview": scan_physxnet_preview,
}


CATEGORY_ZH_LABELS: Dict[str, str] = {
    "top_drop_only": "下坠",
    "top_toss_only": "上方抛掷",
    "front_slide_only": "前向滑入",
    "diagonal_left_only": "左前对角抛入",
    "diagonal_right_only": "右前对角抛入",
    "side_throw_left_only": "左侧抛入",
    "side_throw_right_only": "右侧抛入",
    "rolling_left_only": "左侧滚动",
    "rolling_right_only": "右侧滚动",
    "projectile_arc_only": "抛物线入场",
    "projectile_cross_left_only": "左侧抛物线横穿",
    "projectile_cross_right_only": "右侧抛物线横穿",
    "swing_drop_left_only": "左侧摆入",
    "swing_drop_right_only": "右侧摆入",
    "ground_static_cluster": "地面静止簇",
    "ground_static_with_intruder": "地面静止加侧向侵入",
    "interaction_pair_multi_motion": "单组交互加多运动",
    "dual_interaction_groups": "双组交互场景",
    "omni_showcase_all_modes": "全模式综合展示",
    "all_scenes": "全部场景",
}


def scan_videos(root: Path, profile: str, allowed_names: Sequence[str]) -> List[VideoItem]:
    if profile == "auto":
        all_items: List[VideoItem] = []
        for name, fn in SCAN_PROFILES.items():
            items = fn(root, allowed_names)
            if items:
                all_items.extend(items)
                if name in ("genesis_rigid", "physxnet_preview"):
                    return all_items
        return sorted(all_items, key=lambda x: (x.category, x.scene, x.filename))
    if profile not in SCAN_PROFILES:
        raise ValueError(f"Unknown profile: {profile}")
    return SCAN_PROFILES[profile](root, allowed_names)


def default_allowed_names_for_profile(profile: str) -> List[str]:
    if profile == "genesis_rigid":
        return ["rgb.mp4"]
    if profile == "physxnet_preview":
        return ["preview*.mp4"]
    return []


def group_items(items: Sequence[VideoItem]) -> Dict[str, List[VideoItem]]:
    grouped: Dict[str, List[VideoItem]] = defaultdict(list)
    for item in items:
        grouped[item.category].append(item)
    for key in grouped:
        grouped[key] = sorted(grouped[key], key=lambda x: (x.count_bucket, x.scene_label, x.filename))
    return dict(sorted(grouped.items(), key=lambda kv: kv[0]))


def _category_display_parts(category: str) -> tuple[str, str]:
    return category, CATEGORY_ZH_LABELS.get(category, "")


def build_index_html(items: Sequence[VideoItem], title: str, profile: str, allowed_names: Sequence[str]) -> str:
    grouped = group_items(items)
    sidebar_sections = []
    main_sections = []

    for category, cat_items in grouped.items():
        category_id = f"cat-{_slugify(category)}"
        category_en, category_zh = _category_display_parts(category)
        sidebar_links = []
        for item in cat_items:
            scene_id = f"scene-{_slugify(category)}-{_slugify(item.scene)}"
            badge = f" <span>{html.escape(item.count_bucket)}</span>" if item.count_bucket else ""
            sim_badge = f' <span class="sim-chip sim-chip-{html.escape(item.sim_label)}">{html.escape(item.sim_label)}</span>' if item.sim_label else ""
            sidebar_links.append(
                f'<a class="scene-link" href="#{scene_id}">{html.escape(item.scene_label)}{badge}{sim_badge}</a>'
            )

        sidebar_sections.append(
            f"""
            <section class="side-group">
              <a class="side-category" href="#{category_id}">
                <span class="side-category-en">{html.escape(category_en)}</span>
                {f'<span class="side-category-zh">{html.escape(category_zh)}</span>' if category_zh else ''}
              </a>
              <div class="side-scenes">
                {''.join(sidebar_links)}
              </div>
            </section>
            """
        )

        cards = []
        for item in cat_items:
            scene_id = f"scene-{_slugify(category)}-{_slugify(item.scene)}"
            cards.append(
                f"""
                <article class="video-card" id="{scene_id}">
                  <div class="card-head">
                    <div class="card-title">{html.escape(item.scene_label)}</div>
                    <div class="card-meta">
                      {f'<span class="count-badge">{html.escape(item.count_bucket)}</span>' if item.count_bucket else ''}
                      {f'<span class="sim-badge sim-badge-{html.escape(item.sim_label)}">{html.escape(item.sim_label)}</span>' if item.sim_label else ''}
                      <code>{html.escape(item.filename)}</code>
                    </div>
                  </div>
                  {f'<div class="case-notes">{html.escape(item.case_notes)}</div>' if item.case_notes else ''}
                  {f'<div class="energy-chart"><div class="energy-chart-title">能量漂移曲线（横轴=帧，纵轴=相对漂移）</div>{item.energy_chart_svg}</div>' if item.energy_chart_svg else ''}
                  {f'<pre class="energy-notes">{html.escape(item.energy_notes)}</pre>' if item.energy_notes else ''}
                  <video controls preload="metadata" playsinline>
                    <source src="{_as_url_path(item.rel_path)}" type="video/mp4">
                    Your browser does not support the video tag.
                  </video>
                </article>
                """
            )

        main_sections.append(
            f"""
            <section class="category-block" id="{category_id}">
              <div class="category-head">
                <h2>
                  <span class="category-en">{html.escape(category_en)}</span>
                  {f'<span class="category-zh">{html.escape(category_zh)}</span>' if category_zh else ''}
                </h2>
                <div class="category-meta">场景 {len(cat_items)} 个</div>
              </div>
              <div class="video-grid">
                {''.join(cards)}
              </div>
            </section>
            """
        )

    if not items:
        main_sections.append('<p class="empty">没有找到符合条件的视频。</p>')

    allowed_names_text = ", ".join(allowed_names) if allowed_names else "*.mp4"

    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(title)}</title>
  <style>
    :root {{
      --bg: #f6f1e8;
      --panel: rgba(255, 251, 247, 0.92);
      --panel-strong: #fffdf9;
      --ink: #231a12;
      --muted: #6e6259;
      --line: rgba(50, 34, 20, 0.10);
      --accent: #b66139;
      --accent-soft: #f4dfcc;
      --shadow: 0 14px 34px rgba(93, 57, 31, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "Avenir Next", "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", sans-serif;
      background:
        radial-gradient(circle at left top, rgba(234, 194, 163, 0.42), transparent 26%),
        linear-gradient(180deg, #fbf7f0 0%, #f2e8dc 100%);
    }}
    .layout {{
      display: grid;
      grid-template-columns: 300px minmax(0, 1fr);
      min-height: 100vh;
    }}
    .sidebar {{
      position: sticky;
      top: 0;
      height: 100vh;
      overflow-y: auto;
      padding: 18px 16px 28px 16px;
      background: rgba(252, 246, 239, 0.92);
      border-right: 1px solid var(--line);
      backdrop-filter: blur(12px);
    }}
    .sidebar h1 {{
      margin: 0 0 10px 0;
      font-size: 24px;
      line-height: 1.05;
      letter-spacing: -0.03em;
    }}
    .sidebar-meta {{
      margin-bottom: 18px;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.5;
    }}
    .side-group + .side-group {{
      margin-top: 14px;
    }}
    .side-category {{
      display: flex;
      flex-direction: column;
      gap: 2px;
      padding: 9px 12px;
      border-radius: 12px;
      text-decoration: none;
      color: var(--ink);
      background: var(--accent-soft);
    }}
    .side-category-en {{
      font-weight: 800;
      line-height: 1.2;
    }}
    .side-category-zh {{
      color: #815138;
      font-size: 12px;
      line-height: 1.2;
    }}
    .side-scenes {{
      display: grid;
      gap: 6px;
      margin-top: 8px;
      padding-left: 4px;
    }}
    .scene-link {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 8px;
      padding: 7px 10px;
      border-radius: 10px;
      color: var(--muted);
      text-decoration: none;
      background: rgba(255,255,255,0.55);
      border: 1px solid transparent;
      font-size: 13px;
    }}
    .scene-link:hover {{
      border-color: rgba(182, 97, 57, 0.18);
      color: var(--ink);
    }}
    .scene-link span {{
      flex: 0 0 auto;
      padding: 2px 7px;
      border-radius: 999px;
      background: #f5ebe2;
      color: #815138;
      font-size: 11px;
      font-weight: 700;
    }}
    .scene-link .sim-chip {{
      background: #e8eefc;
      color: #3559a8;
    }}
    .scene-link .sim-chip-smoke {{
      background: #ececec;
      color: #555;
    }}
    .scene-link .sim-chip-liquid {{
      background: #dff4ea;
      color: #1f7a4d;
    }}
    .main {{
      padding: 22px 24px 40px 24px;
    }}
    .toolbar {{
      position: sticky;
      top: 0;
      z-index: 10;
      margin: -22px -24px 20px -24px;
      padding: 16px 24px;
      background: rgba(246, 241, 232, 0.88);
      backdrop-filter: blur(10px);
      border-bottom: 1px solid var(--line);
    }}
    .toolbar-title {{
      font-size: 15px;
      color: var(--muted);
      line-height: 1.5;
    }}
    .category-block {{
      margin-top: 24px;
      scroll-margin-top: 90px;
    }}
    .case-notes {{
      margin: 6px 0 10px;
      padding: 8px 10px;
      border-left: 3px solid var(--accent);
      background: rgba(182, 97, 57, 0.08);
      color: var(--muted);
      font-size: 13px;
      line-height: 1.4;
      border-radius: 6px;
      white-space: pre-wrap;
    }}
    .energy-notes {{
      margin: 0 10px 10px;
      padding: 10px 12px;
      border: 1px solid rgba(182, 97, 57, 0.22);
      background: rgba(255, 247, 240, 0.96);
      color: #5c3623;
      font-size: 12px;
      line-height: 1.55;
      border-radius: 10px;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 260px;
      overflow: auto;
    }}
    .energy-chart {{
      margin: 0 10px 10px;
      padding: 10px 12px 12px;
      border: 1px solid rgba(182, 97, 57, 0.22);
      background: rgba(255, 250, 245, 0.98);
      border-radius: 10px;
    }}
    .energy-chart-title {{
      margin-bottom: 8px;
      color: #7d492f;
      font-size: 12px;
      font-weight: 800;
    }}
    .energy-chart-svg {{
      display: block;
      width: 100%;
      height: 120px;
    }}
    .category-head {{
      display: flex;
      justify-content: space-between;
      align-items: end;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .category-head h2 {{
      margin: 0;
      display: flex;
      flex-wrap: wrap;
      align-items: baseline;
      gap: 8px 10px;
      font-size: 24px;
      letter-spacing: -0.02em;
    }}
    .category-en {{
      display: inline-block;
    }}
    .category-zh {{
      display: inline-block;
      color: #815138;
      font-size: 15px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .category-meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(260px, 1fr));
      gap: 14px;
    }}
    .video-card {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.75);
      border-radius: 18px;
      overflow: hidden;
      box-shadow: var(--shadow);
      scroll-margin-top: 90px;
    }}
    .card-head {{
      padding: 10px 12px 8px 12px;
      border-bottom: 1px solid var(--line);
      background: rgba(255,255,255,0.70);
    }}
    .card-title {{
      font-size: 14px;
      font-weight: 800;
      word-break: break-word;
    }}
    .card-meta {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
      margin-top: 6px;
    }}
    .count-badge {{
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 999px;
      background: #f5e1d1;
      color: #7d492f;
      font-size: 11px;
      font-weight: 800;
    }}
    .sim-badge {{
      display: inline-flex;
      align-items: center;
      padding: 3px 8px;
      border-radius: 999px;
      font-size: 11px;
      font-weight: 800;
      text-transform: uppercase;
      letter-spacing: 0.03em;
    }}
    .sim-badge-smoke {{
      background: #ececec;
      color: #555;
    }}
    .sim-badge-liquid {{
      background: #dff4ea;
      color: #1f7a4d;
    }}
    code {{
      color: var(--muted);
      font-size: 11px;
    }}
    video {{
      display: block;
      width: 100%;
      aspect-ratio: 1 / 1;
      height: auto;
      background: #000;
    }}
    .empty {{
      color: var(--muted);
      font-size: 15px;
    }}
    @media (max-width: 1100px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      .sidebar {{
        position: relative;
        height: auto;
        border-right: none;
        border-bottom: 1px solid var(--line);
      }}
      .main {{
        padding-top: 14px;
      }}
      .toolbar {{
        position: relative;
        margin-top: 0;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="sidebar">
      <h1>{html.escape(title)}</h1>
      <div class="sidebar-meta">
        共 {len(items)} 个视频。<br/>
        当前 profile: <code>{html.escape(profile)}</code><br/>
        当前文件过滤: <code>{html.escape(allowed_names_text)}</code>
      </div>
      {''.join(sidebar_sections)}
    </aside>
    <main class="main">
      <div class="toolbar">
        <div class="toolbar-title">
          左侧用于快速跳转类别和场景，右侧视频窗口已缩小，方便一行显示更多样本。
          默认建议只看 RGB 视频。
        </div>
      </div>
      {''.join(main_sections)}
    </main>
  </div>
</body>
</html>
"""


class VideoPageHandler(SimpleHTTPRequestHandler):
    server_version = "VideoPageHTTP/0.4"

    def __init__(self, *args, directory=None, videos=None, page_title=None, profile=None, allowed_names=None, **kwargs):
        self._root_dir = Path(directory).resolve()
        self._videos = videos or []
        self._page_title = page_title or "Local Video Viewer"
        self._profile = profile or "auto"
        self._allowed_names = allowed_names or []
        super().__init__(*args, directory=str(self._root_dir), **kwargs)

    def log_message(self, fmt, *args):
        print("[HTTP]", fmt % args)

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path in ("/", "/index.html"):
            content = build_index_html(
                self._videos,
                self._page_title,
                self._profile,
                self._allowed_names,
            ).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            return
        return super().do_GET()

    def translate_path(self, path):
        path = urllib.parse.urlparse(path).path
        path = posixpath.normpath(urllib.parse.unquote(path))
        parts = [p for p in path.split("/") if p and p not in (".", "..")]
        local_path = self._root_dir
        for part in parts:
            local_path = local_path / part
        try:
            local_path.resolve().relative_to(self._root_dir)
        except Exception:
            return str(self._root_dir / "__forbidden__")
        return str(local_path)


def parse_args():
    parser = argparse.ArgumentParser(description="按类别在本地网页展示数据集视频")
    parser.add_argument("--root", type=Path, required=True, help="数据根目录")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="监听地址")
    parser.add_argument("--port", type=int, default=8000, help="端口")
    parser.add_argument("--title", type=str, default="Video Viewer", help="网页标题")
    parser.add_argument(
        "--profile",
        type=str,
        default="genesis_rigid",
        choices=["auto", "genesis_rigid", "scene_video", "physxnet_preview"],
        help="扫描目录结构的 profile。genesis_rigid 适配 train/<category>/<count>/<scene>/videos，scene_video 适配 scene_xxxxx/video，physxnet_preview 适配 <object_id>/scene_preview/*.mp4 或 scene_preview/*.mp4。",
    )
    parser.add_argument(
        "--video-name",
        action="append",
        default=None,
        help="只显示指定文件名或 glob 模式。可重复传入。未传时会按 profile 选择默认值。",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    root = args.root.resolve()
    if not root.exists():
        raise SystemExit(f"[ERROR] root 不存在: {root}")

    allowed_names = list(args.video_name) if args.video_name else default_allowed_names_for_profile(args.profile)
    items = scan_videos(root, args.profile, allowed_names)

    print(f"[INFO] root={root}")
    print(f"[INFO] profile={args.profile}")
    print(f"[INFO] allowed_names={allowed_names}")
    print(f"[INFO] videos_found={len(items)}")
    preview = items[:10]
    for idx, item in enumerate(preview, 1):
        print(f"  {idx:03d}. category={item.category} scene={item.scene} file={item.filename} path=/{item.rel_path}")
    if len(items) > 10:
        print("  ...")

    handler = lambda *a, **kw: VideoPageHandler(
        *a,
        directory=root,
        videos=items,
        page_title=args.title,
        profile=args.profile,
        allowed_names=allowed_names,
        **kw,
    )

    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"[INFO] 页面地址: http://127.0.0.1:{args.port}/")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[INFO] 已停止服务")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
