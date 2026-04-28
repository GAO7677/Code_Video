from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any


DEFAULT_COUNT01_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid/single_object_preview/count_01"
)
DEFAULT_OUTPUT_DIR = Path(
    "/home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/visualization/count01_single_collision_portal"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local HTML portal for count_01 single-collision scenes.")
    parser.add_argument("--count01-root", type=Path, default=DEFAULT_COUNT01_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--future-width", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=2)
    return parser.parse_args()


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def rel_web_path(root: Path, path: Path) -> str:
    return os.path.relpath(path, root).replace(os.sep, "/")


def ensure_symlink(target: Path, link_path: Path) -> Path | None:
    if not target.exists():
        return None
    link_path.parent.mkdir(parents=True, exist_ok=True)
    if link_path.is_symlink() or link_path.exists():
        try:
            if link_path.resolve() == target.resolve():
                return link_path
        except FileNotFoundError:
            pass
        if link_path.is_dir() and not link_path.is_symlink():
            raise RuntimeError(f"Refusing to replace directory {link_path}")
        link_path.unlink()
    link_path.symlink_to(target)
    return link_path


def context_indices(collision_frame: int, context_length: int) -> list[int]:
    start = collision_frame - context_length
    indices = list(range(max(0, start), collision_frame))
    if start < 0:
        indices = ([0] * (-start)) + indices
    return indices[-context_length:]


def build_sample_entry(
    *,
    scene_dir: Path,
    output_dir: Path,
    context_length: int,
    future_width: int,
    horizon: int,
) -> dict[str, Any] | None:
    metadata = read_json(scene_dir / "metadata.json")
    events = read_json(scene_dir / "physics" / "collision_events.json")
    if len(events) != 1:
        return None

    event = events[0]
    collision_frame = int(event["start_frame"])
    total_frames = int(metadata["frames"])
    positive_start = collision_frame + horizon
    future_indices = list(range(positive_start, min(total_frames, positive_start + future_width)))

    asset_dir = output_dir / "assets" / scene_dir.name
    linked = {}
    asset_map = {
        "rgb_video": (scene_dir / "videos" / "rgb.mp4", "rgb.mp4"),
        "contact_timeline": (scene_dir / "visualizations" / "contact_timeline.png", "contact_timeline.png"),
        "summary_state": (scene_dir / "visualizations" / "summary_state.png", "summary_state.png"),
        "metadata_json": (scene_dir / "metadata.json", "metadata.json"),
        "collision_events_json": (scene_dir / "physics" / "collision_events.json", "collision_events.json"),
    }
    for key, (src, dst_name) in asset_map.items():
        linked_path = ensure_symlink(src, asset_dir / dst_name)
        linked[key] = rel_web_path(output_dir, linked_path) if linked_path else None

    object_name = None
    objects = metadata.get("objects", [])
    if objects:
        object_name = objects[0].get("name") or objects[0].get("semantic_name")

    return {
        "scene_id": metadata["scene_id"],
        "object_id": metadata["object_id"],
        "object_name": object_name,
        "interaction_pattern": metadata["interaction_pattern"],
        "motion_category": metadata["motion_category"],
        "frames": total_frames,
        "num_objects": metadata["num_objects"],
        "collision_event_count": len(events),
        "collision_event": event,
        "context_length": context_length,
        "context_indices": context_indices(collision_frame, context_length),
        "future_width": future_width,
        "horizon": horizon,
        "future_indices": future_indices,
        "future_valid": len(future_indices) == future_width,
        "assets": linked,
        "source_dir": str(scene_dir),
    }


def render_html(samples: list[dict[str, Any]], stats: dict[str, Any]) -> str:
    samples_json = json.dumps(samples, ensure_ascii=False)
    stats_json = json.dumps(stats, ensure_ascii=False)
    template = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>count_01 Single Collision Portal</title>
  <style>
    :root {{
      --bg: #f4f1e8;
      --ink: #182024;
      --card: #fffdf7;
      --line: #cdbf9d;
      --accent: #b54d2f;
      --accent-2: #2f6c62;
      --muted: #5e645f;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(181,77,47,0.18), transparent 26%),
        radial-gradient(circle at top right, rgba(47,108,98,0.16), transparent 24%),
        linear-gradient(180deg, #f7f1e5 0%, var(--bg) 100%);
    }}
    .shell {{
      max-width: 1480px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      line-height: 1.1;
      letter-spacing: -0.03em;
    }}
    .lede {{
      max-width: 920px;
      color: var(--muted);
      margin-bottom: 18px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 22px;
    }}
    .stat {{
      background: rgba(255,255,255,0.72);
      border: 1px solid rgba(205,191,157,0.9);
      border-radius: 16px;
      padding: 14px 16px;
      backdrop-filter: blur(8px);
    }}
    .stat .label {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .stat .value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      margin-bottom: 20px;
      background: rgba(255,255,255,0.64);
      border: 1px solid rgba(205,191,157,0.9);
      border-radius: 18px;
      padding: 14px;
    }}
    .controls label {{
      display: flex;
      flex-direction: column;
      gap: 6px;
      font-size: 13px;
      color: var(--muted);
      min-width: 200px;
    }}
    .controls input, .controls select {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      font-size: 14px;
      background: #fffdf8;
      color: var(--ink);
    }}
    #count {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--card);
      border: 1px solid rgba(205,191,157,0.95);
      border-radius: 20px;
      overflow: hidden;
      box-shadow: 0 18px 60px rgba(39, 35, 23, 0.08);
    }}
    .card header {{
      padding: 16px 18px 12px;
      border-bottom: 1px solid rgba(205,191,157,0.7);
      background:
        linear-gradient(135deg, rgba(181,77,47,0.1), transparent),
        linear-gradient(225deg, rgba(47,108,98,0.1), transparent);
    }}
    .scene {{
      font-size: 19px;
      font-weight: 700;
      margin-bottom: 6px;
      word-break: break-word;
    }}
    .subtitle {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.4;
    }}
    .chips {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }}
    .chip {{
      border-radius: 999px;
      padding: 5px 10px;
      font-size: 12px;
      font-weight: 600;
      background: rgba(24,32,36,0.06);
    }}
    .chip.accent {{ background: rgba(181,77,47,0.14); color: #8d341d; }}
    .chip.green {{ background: rgba(47,108,98,0.14); color: #1b5a50; }}
    .body {{
      padding: 16px 18px 18px;
    }}
    .video {{
      width: 100%;
      border-radius: 14px;
      background: #111;
      margin-bottom: 12px;
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 8px 12px;
      margin-bottom: 14px;
      font-size: 13px;
    }}
    .meta div {{
      background: rgba(24,32,36,0.03);
      border-radius: 10px;
      padding: 8px 10px;
    }}
    .meta strong {{
      display: block;
      font-size: 11px;
      letter-spacing: 0.06em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 4px;
    }}
    .thumbs {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-bottom: 12px;
    }}
    .thumbs img {{
      width: 100%;
      display: block;
      border-radius: 12px;
      border: 1px solid rgba(205,191,157,0.7);
    }}
    .event {{
      font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
      font-size: 12px;
      background: #1d2327;
      color: #eff5f4;
      border-radius: 14px;
      padding: 12px;
      overflow-x: auto;
      white-space: pre-wrap;
    }}
    .links {{
      display: flex;
      gap: 10px;
      flex-wrap: wrap;
      margin-top: 12px;
    }}
    .links a {{
      color: var(--accent-2);
      text-decoration: none;
      font-size: 13px;
      font-weight: 600;
    }}
    .empty {{
      padding: 28px;
      border: 1px dashed var(--line);
      border-radius: 16px;
      color: var(--muted);
      text-align: center;
      background: rgba(255,255,255,0.55);
    }}
    @media (max-width: 760px) {{
      .meta, .thumbs {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="shell">
    <h1>count_01 Single-Collision Browser</h1>
    <p class="lede">Only scenes under <code>single_object_preview/count_01</code> with exactly one collision event in the whole video. Environment collisions are counted as valid events, and no main-object assumption is used.</p>
    <div class="stats" id="stats"></div>
    <div class="controls">
      <label>Search scene / object
        <input id="search" type="search" placeholder="scene_id or object id">
      </label>
      <label>Motion category
        <select id="motion"></select>
      </label>
      <label>Interaction pattern
        <select id="pattern"></select>
      </label>
      <label>Collision frame
        <select id="collision"></select>
      </label>
    </div>
    <p id="count"></p>
    <div id="grid" class="grid"></div>
  </div>
  <script>
    const samples = __SAMPLES_JSON__;
    const stats = __STATS_JSON__;

    const searchEl = document.getElementById('search');
    const motionEl = document.getElementById('motion');
    const patternEl = document.getElementById('pattern');
    const collisionEl = document.getElementById('collision');
    const gridEl = document.getElementById('grid');
    const countEl = document.getElementById('count');
    const statsEl = document.getElementById('stats');

    function fillSelect(select, values, label) {{
      select.innerHTML = '';
      const all = document.createElement('option');
      all.value = '';
      all.textContent = `All ${{label}}`;
      select.appendChild(all);
      values.forEach((value) => {{
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = value;
        select.appendChild(opt);
      }});
    }}

    function renderStats() {{
      const blocks = [
        ['Filtered scenes', stats.num_samples],
        ['Distinct objects', stats.num_object_ids],
        ['Motion categories', stats.num_motion_categories],
        ['Collision frames', stats.collision_frame_range],
      ];
      statsEl.innerHTML = blocks.map(([label, value]) => `
        <div class="stat">
          <span class="label">${{label}}</span>
          <span class="value">${{value}}</span>
        </div>
      `).join('');
    }}

    function matches(sample) {{
      const q = searchEl.value.trim().toLowerCase();
      if (q) {{
        const hay = `${{sample.scene_id}} ${{sample.object_id}} ${{sample.object_name || ''}}`.toLowerCase();
        if (!hay.includes(q)) return false;
      }}
      if (motionEl.value && sample.motion_category !== motionEl.value) return false;
      if (patternEl.value && sample.interaction_pattern !== patternEl.value) return false;
      if (collisionEl.value && String(sample.collision_event.start_frame) !== collisionEl.value) return false;
      return true;
    }}

    function sampleCard(sample) {{
      const eventText = JSON.stringify(sample.collision_event, null, 2);
      const safeEvent = eventText
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;');
      const links = [
        sample.assets.metadata_json ? `<a href="${{sample.assets.metadata_json}}" target="_blank">metadata.json</a>` : '',
        sample.assets.collision_events_json ? `<a href="${{sample.assets.collision_events_json}}" target="_blank">collision_events.json</a>` : '',
      ].filter(Boolean).join('');
      return `
        <article class="card">
          <header>
            <div class="scene">${{sample.scene_id}}</div>
            <div class="subtitle">${{sample.object_name || 'unknown object'}} · object_id=${{sample.object_id}}</div>
            <div class="chips">
              <span class="chip accent">${{sample.motion_category}}</span>
              <span class="chip green">${{sample.interaction_pattern}}</span>
              <span class="chip">collision @ frame ${{sample.collision_event.start_frame}}</span>
            </div>
          </header>
          <div class="body">
            ${sample.assets.rgb_video ? `<video class="video" controls preload="metadata" src="${{sample.assets.rgb_video}}"></video>` : ''}
            <div class="meta">
              <div><strong>Frames</strong>${{sample.frames}}</div>
              <div><strong>Event Count</strong>${{sample.collision_event_count}}</div>
              <div><strong>Context</strong>[${{sample.context_indices.join(', ')}}]</div>
              <div><strong>Future h=${{sample.horizon}}</strong>[${{sample.future_indices.join(', ')}}]${{sample.future_valid ? '' : ' (truncated)'}} </div>
            </div>
            <div class="thumbs">
              ${sample.assets.contact_timeline ? `<img src="${{sample.assets.contact_timeline}}" alt="contact timeline">` : '<div></div>'}
              ${sample.assets.summary_state ? `<img src="${{sample.assets.summary_state}}" alt="summary state">` : '<div></div>'}
            </div>
            <div class="event">${safeEvent}</div>
            <div class="links">${links}</div>
          </div>
        </article>
      `;
    }}

    function render() {{
      const filtered = samples.filter(matches);
      countEl.textContent = `${{filtered.length}} / ${{samples.length}} scenes shown`;
      if (!filtered.length) {{
        gridEl.innerHTML = '<div class="empty">No scenes match the current filters.</div>';
        return;
      }}
      gridEl.innerHTML = filtered.map(sampleCard).join('');
    }}

    fillSelect(motionEl, [...new Set(samples.map((s) => s.motion_category))].sort(), 'motions');
    fillSelect(patternEl, [...new Set(samples.map((s) => s.interaction_pattern))].sort(), 'patterns');
    fillSelect(collisionEl, [...new Set(samples.map((s) => String(s.collision_event.start_frame)))].sort((a, b) => Number(a) - Number(b)), 'collision frames');
    renderStats();
    [searchEl, motionEl, patternEl, collisionEl].forEach((el) => el.addEventListener('input', render));
    render();
  </script>
</body>
</html>"""
    return template.replace("__SAMPLES_JSON__", samples_json).replace("__STATS_JSON__", stats_json)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    motion_counter = Counter()
    pattern_counter = Counter()
    collision_frames = []
    object_ids = set()
    for scene_dir in sorted(path for path in args.count01_root.iterdir() if path.is_dir()):
        entry = build_sample_entry(
            scene_dir=scene_dir,
            output_dir=output_dir,
            context_length=args.context_length,
            future_width=args.future_width,
            horizon=args.horizon,
        )
        if entry is None:
            continue
        samples.append(entry)
        motion_counter[entry["motion_category"]] += 1
        pattern_counter[entry["interaction_pattern"]] += 1
        collision_frames.append(int(entry["collision_event"]["start_frame"]))
        object_ids.add(entry["object_id"])

    stats = {
        "num_samples": len(samples),
        "num_object_ids": len(object_ids),
        "num_motion_categories": len(motion_counter),
        "collision_frame_range": f"{min(collision_frames)}-{max(collision_frames)}" if collision_frames else "n/a",
        "motion_counts": dict(motion_counter),
        "pattern_counts": dict(pattern_counter),
    }

    json_path = output_dir / "count01_single_collision_samples.json"
    json_path.write_text(json.dumps({"stats": stats, "samples": samples}, indent=2), encoding="utf-8")
    html_path = output_dir / "index.html"
    html_path.write_text(render_html(samples, stats), encoding="utf-8")

    print(f"Wrote {len(samples)} filtered scenes to {json_path}")
    print(f"Wrote portal to {html_path}")


if __name__ == "__main__":
    main()
