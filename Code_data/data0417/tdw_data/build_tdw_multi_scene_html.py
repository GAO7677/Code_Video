from pathlib import Path


ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW")
HTML = ROOT / "tdw_multi_scene_cases.html"

DATASETS = [
    {
        "title": "Original Multi Scene Cases",
        "subtitle": "原始多场景动力学 case，保留旧结果，不做覆盖。",
        "directory": "tdw_multi_scene_cases",
        "badge": "Original",
        "mode": "scene_dirs",
    },
    {
        "title": "Real Object Multi Scene Cases",
        "subtitle": "新增的真实物体仿真批次，使用日常物体模型，单独输出，不影响旧 case。",
        "directory": "tdw_multi_scene_real_objects",
        "badge": "Real Object",
        "mode": "scene_dirs",
    },
    {
        "title": "Soft Body / Cloth Demo",
        "subtitle": "TDW 的 Obi 软体布料 demo。这里展示真正的 cloth 求解，而不是软外观刚体。",
        "directory": "tdw_obi_cloth_demo",
        "badge": "Soft Body",
        "mode": "flat_videos",
        "scene_label": "tdw_obi_cloth_demo",
    },
    {
        "title": "Real Scene Gravity Demos",
        "subtitle": "真实室内背景下的重力仿真示例，包含刚体下落和 Obi 布料下落，输出到独立目录并汇总展示。",
        "directory": "tdw_real_scene_gravity_demos",
        "badge": "Real Scene",
        "mode": "flat_videos",
        "scene_label": "tdw_real_scene_gravity_demos",
    },
    {
        "title": "Stable Cloth Demos",
        "subtitle": "针对 cloth 刺穿现象的稳定版批次：提高 Obi solver substeps，改用更规整的静止障碍物，并收敛初始姿态。",
        "directory": "tdw_real_scene_gravity_stable_cloth",
        "badge": "Stable Cloth",
        "mode": "flat_videos",
        "scene_label": "tdw_real_scene_gravity_stable_cloth",
    },
    {
        "title": "Cloth A/B Compare",
        "subtitle": "相同场景、相同障碍物、相同 cloth 初始姿态，只改变稳定化参数。每个 case 的未稳定版和稳定版并排放在同一行。",
        "directory": "tdw_cloth_ab_compare",
        "badge": "A/B Compare",
        "mode": "compare_pairs",
    },
]


def list_scene_videos(dataset_dir: Path):
    if not dataset_dir.exists():
        return []
    scenes = []
    for scene_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        videos = sorted(scene_dir.glob("*.mp4"))
        if videos:
            scenes.append((scene_dir.name, videos))
    return scenes


def list_flat_videos(dataset_dir: Path):
    if not dataset_dir.exists():
        return []
    return sorted(dataset_dir.glob("*.mp4"))


def list_compare_pairs(dataset_dir: Path):
    if not dataset_dir.exists():
        return []
    pairs = []
    for case_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        unstable = case_dir.joinpath("unstable.mp4")
        stable = case_dir.joinpath("stable.mp4")
        if unstable.exists() or stable.exists():
            pairs.append((case_dir.name, unstable if unstable.exists() else None, stable if stable.exists() else None))
    return pairs


cards = []
for dataset in DATASETS:
    dataset_dir = ROOT / dataset["directory"]
    cards.append(f'<section class="section"><h2>{dataset["title"]}</h2><p>{dataset["subtitle"]}</p>')
    if dataset["mode"] == "scene_dirs":
        scenes = list_scene_videos(dataset_dir)
        if not scenes:
            cards.append('<p class="empty">当前还没有生成视频。</p></section>')
            continue
        for scene_name, videos in scenes:
            cards.append(f'<div class="subsection"><h3>{scene_name}</h3><div class="grid">')
            for video in videos:
                rel = str(video.relative_to(ROOT))
                abs_path = str(video)
                case_name = video.stem
                cards.append(
                    f'''<article class="card">
  <video controls preload="metadata" src="{rel}"></video>
  <div class="meta">
    <span class="pill">{dataset["badge"]}</span>
    <span class="pill scene-pill">{scene_name}</span>
    <h4>{case_name}</h4>
    <code>{abs_path}</code>
  </div>
</article>'''
                )
            cards.append("</div></div>")
    elif dataset["mode"] == "flat_videos":
        videos = list_flat_videos(dataset_dir)
        if not videos:
            cards.append('<p class="empty">当前还没有生成视频。</p></section>')
            continue
        scene_label = dataset.get("scene_label", dataset["directory"])
        cards.append('<div class="grid">')
        for video in videos:
            rel = str(video.relative_to(ROOT))
            abs_path = str(video)
            case_name = video.stem
            cards.append(
                f'''<article class="card">
  <video controls preload="metadata" src="{rel}"></video>
  <div class="meta">
    <span class="pill">{dataset["badge"]}</span>
    <span class="pill scene-pill">{scene_label}</span>
    <h4>{case_name}</h4>
    <code>{abs_path}</code>
  </div>
</article>'''
            )
        cards.append("</div>")
    else:
        pairs = list_compare_pairs(dataset_dir)
        if not pairs:
            cards.append('<p class="empty">当前还没有生成视频。</p></section>')
            continue
        for case_name, unstable, stable in pairs:
            cards.append(f'<div class="subsection"><h3>{case_name}</h3><div class="compare-grid">')
            for label, video in [("Unstable", unstable), ("Stable", stable)]:
                if video is None:
                    cards.append(
                        f'''<article class="card">
  <div class="missing-video">Missing {label}</div>
  <div class="meta">
    <span class="pill">{dataset["badge"]}</span>
    <span class="pill scene-pill">{label}</span>
    <h4>{case_name}</h4>
    <code>missing</code>
  </div>
</article>'''
                    )
                    continue
                rel = str(video.relative_to(ROOT))
                abs_path = str(video)
                cards.append(
                    f'''<article class="card">
  <video controls preload="metadata" src="{rel}"></video>
  <div class="meta">
    <span class="pill">{dataset["badge"]}</span>
    <span class="pill scene-pill">{label}</span>
    <h4>{case_name}</h4>
    <code>{abs_path}</code>
  </div>
</article>'''
                )
            cards.append("</div></div>")
    cards.append("</section>")

content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TDW Multi Scene Cases</title>
  <style>
    :root {{
      --bg: #f4f0e8;
      --panel: rgba(255,255,255,0.92);
      --ink: #171714;
      --muted: #6a665e;
      --accent: #8a603b;
      --accent-2: #466b5d;
      --border: rgba(54, 42, 24, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top left, rgba(209, 186, 156, 0.38), transparent 32%),
        radial-gradient(circle at right 18%, rgba(161, 187, 176, 0.30), transparent 28%),
        linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
    }}
    .wrap {{ max-width: 1580px; margin: 0 auto; padding: 36px 20px 56px; }}
    .hero, .section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: 0 18px 40px rgba(44, 35, 19, 0.10);
    }}
    .hero {{ padding: 28px; margin-bottom: 24px; }}
    .section {{ padding: 22px; margin-top: 24px; }}
    .subsection {{ margin-top: 18px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(30px, 5vw, 54px); line-height: 0.96; }}
    h2 {{ margin: 0 0 10px; font-size: 28px; }}
    h3 {{ margin: 8px 0 12px; font-size: 22px; }}
    h4 {{ margin: 0 0 8px; font-size: 20px; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .empty {{ margin-top: 14px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }}
    .compare-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid var(--border); border-radius: 18px; overflow: hidden; background: rgba(255,255,255,0.76); }}
    video {{ width: 100%; display: block; background: #000; aspect-ratio: 16 / 9; }}
    .missing-video {{
      width: 100%;
      aspect-ratio: 16 / 9;
      display: grid;
      place-items: center;
      background: rgba(70, 107, 93, 0.08);
      color: var(--accent-2);
      font-size: 18px;
    }}
    .meta {{ padding: 14px 16px 18px; }}
    .pill {{
      display: inline-block;
      margin-right: 8px;
      margin-bottom: 8px;
      padding: 5px 10px;
      border-radius: 999px;
      background: rgba(138, 96, 59, 0.12);
      color: var(--accent);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    .scene-pill {{
      background: rgba(70, 107, 93, 0.12);
      color: var(--accent-2);
    }}
    code {{ color: var(--accent); font-size: 13px; word-break: break-all; }}
    @media (max-width: 900px) {{
      .compare-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>TDW Multi Scene Cases</h1>
      <p>同一个页面里同时展示旧的多场景 case 和新增的真实物体 case。旧结果保留不动，新结果单独输出到新目录后自动汇总到这里。</p>
    </section>
    {''.join(cards)}
  </div>
</body>
</html>
"""

HTML.write_text(content, encoding="utf-8")
print(HTML)
