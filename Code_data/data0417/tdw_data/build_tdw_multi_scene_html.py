from pathlib import Path


ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW")
HTML = ROOT / "tdw_multi_scene_cases.html"

DATASETS = [
    {
        "title": "Original Multi Scene Cases",
        "subtitle": "原始多场景动力学 case，保留旧结果，不做覆盖。",
        "directory": "tdw_multi_scene_cases",
        "badge": "Original",
    },
    {
        "title": "Real Object Multi Scene Cases",
        "subtitle": "新增的真实物体仿真批次，使用日常物体模型，单独输出，不影响旧 case。",
        "directory": "tdw_multi_scene_real_objects",
        "badge": "Real Object",
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


cards = []
for dataset in DATASETS:
    dataset_dir = ROOT / dataset["directory"]
    scenes = list_scene_videos(dataset_dir)
    cards.append(f'<section class="section"><h2>{dataset["title"]}</h2><p>{dataset["subtitle"]}</p>')
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
    .card {{ border: 1px solid var(--border); border-radius: 18px; overflow: hidden; background: rgba(255,255,255,0.76); }}
    video {{ width: 100%; display: block; background: #000; aspect-ratio: 16 / 9; }}
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
