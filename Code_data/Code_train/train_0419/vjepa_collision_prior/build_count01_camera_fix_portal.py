#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_train/train_0419/vjepa_collision_prior/count01_camera_fix_parallel"
)


def rel(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def gather_entries(root: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for report_path in sorted(root.glob("workers/device_*/**/repair_report.json")):
        payload = json.loads(report_path.read_text(encoding="utf-8"))
        repairs = payload.get("repairs", [])
        if not repairs:
            continue
        item = repairs[0]
        before = dict(item.get("before", {}))
        after = dict(item.get("after", {}))
        sample_dir = report_path.parent
        before_video = sample_dir / "assets" / item["sample_name"] / "before.mp4"
        after_video = sample_dir / "assets" / item["sample_name"] / "after.mp4"
        before_gif = root / "gif_assets" / item["sample_name"] / "before.gif"
        after_gif = root / "gif_assets" / item["sample_name"] / "after.gif"
        entries.append(
            {
                "sample_name": item["sample_name"],
                "case_name": item.get("case_name", ""),
                "motion_category": item.get("motion_category", ""),
                "status": item.get("status", "unknown"),
                "device": report_path.parts[-4],
                "camera_distance_mult": item.get("camera_distance_mult"),
                "before_margin": before.get("border_margin_min"),
                "after_margin": after.get("border_margin_min"),
                "before_bbox": before.get("last_bbox"),
                "after_bbox": after.get("last_bbox"),
                "before_camera": before.get("camera"),
                "after_camera": after.get("camera"),
                "before_video": rel(root, before_video) if before_video.exists() else None,
                "after_video": rel(root, after_video) if after_video.exists() else None,
                "before_gif": rel(root, before_gif) if before_gif.exists() else None,
                "after_gif": rel(root, after_gif) if after_gif.exists() else None,
                "backup_dir": rel(root, sample_dir / "replaced_originals" / item["sample_name"]),
                "report_path": rel(root, report_path),
            }
        )
    entries.sort(key=lambda x: (x["case_name"], x["sample_name"]))
    return entries


def render(entries: list[dict[str, Any]]) -> str:
    stats = {
        "total": len(entries),
        "repaired": sum(1 for x in entries if x.get("status") == "repaired"),
        "failed": sum(1 for x in entries if x.get("status") != "repaired"),
    }
    entries_json = json.dumps(entries, ensure_ascii=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>count_01 Camera Fix Portal</title>
  <style>
    :root {{
      --bg: #f3efe6;
      --ink: #162126;
      --muted: #5e6769;
      --card: #fffdf8;
      --line: #cfbfa5;
      --good: #2a6a4a;
      --bad: #b5485f;
      --accent: #9f5a22;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(159,90,34,0.18), transparent 24%),
        radial-gradient(circle at top right, rgba(42,106,74,0.14), transparent 26%),
        linear-gradient(180deg, #f7f2e8 0%, var(--bg) 100%);
    }}
    .shell {{
      max-width: 1540px;
      margin: 0 auto;
      padding: 28px 24px 40px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 34px;
      letter-spacing: -0.03em;
    }}
    .lede {{
      margin: 0 0 18px;
      color: var(--muted);
      max-width: 980px;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .stat {{
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 14px 16px;
      background: rgba(255,255,255,0.72);
    }}
    .label {{
      display: block;
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.08em;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    .value {{
      font-size: 28px;
      font-weight: 700;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 12px;
      padding: 14px;
      border: 1px solid var(--line);
      border-radius: 18px;
      background: rgba(255,255,255,0.66);
      margin-bottom: 20px;
    }}
    .controls label {{
      display: flex;
      flex-direction: column;
      min-width: 220px;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }}
    .controls input, .controls select {{
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      background: #fffdf8;
      color: var(--ink);
      font-size: 14px;
    }}
    #count {{
      margin: 0 0 12px;
      color: var(--muted);
      font-size: 14px;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(440px, 1fr));
      gap: 16px;
    }}
    .card {{
      border: 1px solid var(--line);
      border-radius: 18px;
      background: var(--card);
      padding: 16px;
      box-shadow: 0 16px 36px rgba(22,33,38,0.08);
    }}
    .card h2 {{
      margin: 0 0 8px;
      font-size: 20px;
      line-height: 1.2;
    }}
    .pillrow {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-bottom: 12px;
    }}
    .pill {{
      border-radius: 999px;
      padding: 4px 10px;
      font-size: 12px;
      background: #f4ede0;
      color: var(--ink);
      border: 1px solid rgba(207,191,165,0.9);
    }}
    .pill.good {{
      background: rgba(42,106,74,0.12);
      color: var(--good);
      border-color: rgba(42,106,74,0.25);
    }}
    .video-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-bottom: 12px;
    }}
    .video-box .title {{
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .media {{
      width: 100%;
      aspect-ratio: 4 / 3;
      border-radius: 12px;
      border: 1px solid var(--line);
      background: #000;
      object-fit: contain;
      display: block;
    }}
    .kv {{
      margin: 4px 0;
      color: var(--muted);
      font-size: 13px;
    }}
    .links {{
      margin-top: 10px;
      display: flex;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .links a {{
      color: var(--accent);
      text-decoration: none;
      font-size: 13px;
    }}
    @media (max-width: 900px) {{
      .video-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <h1>count_01 Camera Fix Portal</h1>
    <p class="lede">所有修复样本的前后对比。原数据集目录已经被重生成结果覆盖；旧版本备份保存在各 worker 目录的 <code>replaced_originals/</code> 下。</p>
    <section class="stats">
      <div class="stat"><span class="label">Total Repairs</span><span class="value">{stats["total"]}</span></div>
      <div class="stat"><span class="label">Repaired</span><span class="value">{stats["repaired"]}</span></div>
      <div class="stat"><span class="label">Failed</span><span class="value">{stats["failed"]}</span></div>
    </section>
    <section class="controls">
      <label>
        Search
        <input id="search" type="search" placeholder="sample / case / motion">
      </label>
      <label>
        Motion
        <select id="motion"></select>
      </label>
      <label>
        Case
        <select id="case"></select>
      </label>
    </section>
    <p id="count"></p>
    <section class="grid" id="grid"></section>
  </main>
  <script>
    const entries = {entries_json};
    const motionSel = document.getElementById('motion');
    const caseSel = document.getElementById('case');
    const searchInput = document.getElementById('search');
    const grid = document.getElementById('grid');
    const count = document.getElementById('count');

    function fillSelect(select, values) {{
      select.innerHTML = '';
      const all = document.createElement('option');
      all.value = '';
      all.textContent = 'All';
      select.appendChild(all);
      values.forEach((value) => {{
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = value;
        select.appendChild(opt);
      }});
    }}

    fillSelect(motionSel, [...new Set(entries.map((x) => x.motion_category))].sort());
    fillSelect(caseSel, [...new Set(entries.map((x) => x.case_name))].sort());

    function fmt(value) {{
      if (value === null || value === undefined) return 'n/a';
      if (typeof value === 'number') return Number.isFinite(value) ? value.toFixed(1) : String(value);
      return String(value);
    }}

    function renderCard(item) {{
      const card = document.createElement('article');
      card.className = 'card';
      card.innerHTML = `
        <h2>${{item.sample_name}}</h2>
        <div class="pillrow">
          <span class="pill good">${{item.status}}</span>
          <span class="pill">${{item.case_name}}</span>
          <span class="pill">${{item.motion_category}}</span>
          <span class="pill">${{item.device}}</span>
          <span class="pill">cam x${{fmt(item.camera_distance_mult)}}</span>
        </div>
        <div class="video-grid">
          <div class="video-box">
            <div class="title">Before</div>
            <img class="media" loading="lazy" src="${{item.before_gif || item.before_video || ''}}" alt="before gif">
          </div>
          <div class="video-box">
            <div class="title">After</div>
            <img class="media" loading="lazy" src="${{item.after_gif || item.after_video || ''}}" alt="after gif">
          </div>
        </div>
        <p class="kv">Margin: before=${{fmt(item.before_margin)}} px | after=${{fmt(item.after_margin)}} px</p>
        <p class="kv">BBox: before=${{JSON.stringify(item.before_bbox)}} | after=${{JSON.stringify(item.after_bbox)}}</p>
        <p class="kv">Camera before: pos=${{JSON.stringify(item.before_camera?.pos || null)}}</p>
        <p class="kv">Camera after: pos=${{JSON.stringify(item.after_camera?.pos || null)}}</p>
        <div class="links">
          <a href="${{item.report_path}}" target="_blank" rel="noreferrer">repair report</a>
          <a href="${{item.backup_dir}}" target="_blank" rel="noreferrer">backup dir</a>
        </div>
      `;
      return card;
    }}

    function applyFilters() {{
      const q = searchInput.value.trim().toLowerCase();
      const motion = motionSel.value;
      const c = caseSel.value;
      const filtered = entries.filter((item) => {{
        const hay = `${{item.sample_name}} ${{item.case_name}} ${{item.motion_category}}`.toLowerCase();
        if (q && !hay.includes(q)) return false;
        if (motion && item.motion_category !== motion) return false;
        if (c && item.case_name !== c) return false;
        return true;
      }});
      count.textContent = `${{filtered.length}} / ${{entries.length}} samples`;
      grid.innerHTML = '';
      filtered.forEach((item) => grid.appendChild(renderCard(item)));
    }}

    searchInput.addEventListener('input', applyFilters);
    motionSel.addEventListener('change', applyFilters);
    caseSel.addEventListener('change', applyFilters);
    applyFilters();
  </script>
</body>
</html>
"""


def main() -> None:
    entries = gather_entries(ROOT)
    out_path = ROOT / "index.html"
    out_path.write_text(render(entries), encoding="utf-8")
    print(f"wrote {out_path}")
    print(f"entries={len(entries)}")


if __name__ == "__main__":
    main()
