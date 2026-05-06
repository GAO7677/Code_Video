from pathlib import Path


root = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW")
html = root / "tdw_multi_scene_cases.html"

scenes = ["building_site", "box_room_2018", "tdw_room", "suburb_scene_2023", "suburb_scene_2018", "mm_craftroom_1a", "mm_kitchen_2b"]
cases = ["case000_static_center", "case003_high_drop", "case900_random_parabola", "case005_entry_left"]

cards = []
for scene in scenes:
    cards.append(f'<section class="section"><h2>{scene}</h2><div class="grid">')
    for case in cases:
        rel = f"tdw_multi_scene_cases/{scene}/{case}.mp4"
        abs_path = f"/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/{rel}"
        cards.append(
            f'''<article class="card">
  <video controls preload="metadata" src="{rel}"></video>
  <div class="meta">
    <span class="pill">{scene}</span>
    <h3>{case}</h3>
    <code>{abs_path}</code>
  </div>
</article>'''
        )
    cards.append("</div></section>")

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
      --border: rgba(54, 42, 24, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: Georgia, "Times New Roman", serif;
      color: var(--ink);
      background: linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
    }}
    .wrap {{ max-width: 1500px; margin: 0 auto; padding: 36px 20px 56px; }}
    .hero, .section {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: 0 18px 40px rgba(44, 35, 19, 0.10);
    }}
    .hero {{ padding: 28px; margin-bottom: 24px; }}
    .section {{ padding: 22px; margin-top: 24px; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(30px, 5vw, 54px); line-height: 0.96; }}
    h2 {{ margin: 0 0 14px; font-size: 26px; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.6; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 18px; }}
    .card {{ border: 1px solid var(--border); border-radius: 18px; overflow: hidden; background: rgba(255,255,255,0.76); }}
    video {{ width: 100%; display: block; background: #000; aspect-ratio: 16 / 9; }}
    .meta {{ padding: 14px 16px 18px; }}
    .meta h3 {{ margin: 0 0 8px; font-size: 20px; }}
    .pill {{
      display: inline-block;
      margin-bottom: 8px;
      padding: 5px 10px;
      border-radius: 999px;
      background: rgba(138, 96, 59, 0.12);
      color: var(--accent);
      font-size: 12px;
      text-transform: uppercase;
      letter-spacing: 0.06em;
    }}
    code {{ color: var(--accent); font-size: 13px; word-break: break-all; }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>TDW Multi Scene Cases</h1>
      <p>同一批动力学 case 在不同背景场景下的可视化对比。当前包含：building_site、box_room_2018、tdw_room、suburb_scene_2023、suburb_scene_2018、mm_craftroom_1a、mm_kitchen_2b。</p>
    </section>
    {''.join(cards)}
  </div>
</body>
</html>
"""

html.write_text(content, encoding="utf-8")
print(html)
