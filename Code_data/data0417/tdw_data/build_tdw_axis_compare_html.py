from pathlib import Path


ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW")
HTML = ROOT / "tdw_axis_export_compare.html"
DATASET_DIR = ROOT / "tdw_axis_export_compare"


def main() -> None:
    videos = sorted(DATASET_DIR.glob("*.mp4")) if DATASET_DIR.exists() else []
    cards = []
    if not videos:
        cards.append('<p class="empty">当前还没有生成对比视频。</p>')
    else:
        for video in videos:
            rel = str(video.relative_to(ROOT))
            abs_path = str(video)
            case_name = video.stem
            cards.append(
                f'''<article class="card">
  <video controls preload="metadata" src="{rel}"></video>
  <div class="meta">
    <span class="pill">Axis Compare</span>
    <h2>{case_name}</h2>
    <p>同一段 TDW 仿真分别按 y-up 原生导出和 z-up 离线转换导出后的说明对比。视频画面本身不变，变化发生在坐标、重力轴和后续 3D 真值解释上。</p>
    <code>{abs_path}</code>
  </div>
</article>'''
            )

    content = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Axis Export Compare</title>
  <style>
    :root {{
      --bg: #f4f0e8;
      --panel: rgba(255,255,255,0.94);
      --ink: #171714;
      --muted: #6a665e;
      --accent: #466b5d;
      --border: rgba(54, 42, 24, 0.12);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(209, 186, 156, 0.30), transparent 28%),
        radial-gradient(circle at right 18%, rgba(161, 187, 176, 0.24), transparent 24%),
        linear-gradient(180deg, #f8f4ed 0%, var(--bg) 100%);
    }}
    .wrap {{ max-width: 1680px; margin: 0 auto; padding: 28px 18px 40px; }}
    .hero, .card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 22px;
      box-shadow: 0 18px 40px rgba(44, 35, 19, 0.10);
    }}
    .hero {{ padding: 26px; margin-bottom: 20px; }}
    .card {{ overflow: hidden; }}
    h1 {{ margin: 0 0 8px; font-size: clamp(30px, 5vw, 52px); line-height: 0.96; }}
    h2 {{ margin: 0 0 10px; font-size: 24px; }}
    p {{ margin: 0; color: var(--muted); line-height: 1.65; }}
    .empty {{ padding: 20px; background: var(--panel); border-radius: 18px; border: 1px solid var(--border); }}
    video {{ width: 100%; display: block; background: #000; }}
    .meta {{ padding: 18px 20px 22px; }}
    .pill {{
      display: inline-block;
      margin-bottom: 10px;
      padding: 6px 12px;
      border-radius: 999px;
      background: rgba(70, 107, 93, 0.12);
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    code {{
      display: block;
      margin-top: 12px;
      padding: 12px 14px;
      border-radius: 14px;
      background: rgba(23, 23, 20, 0.05);
      color: #574f45;
      font-size: 13px;
      white-space: pre-wrap;
      word-break: break-all;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <h1>Axis Export Compare</h1>
      <p>只展示 y-up 与 z-up 导出差异说明。核心结论：同一段 TDW 渲染视频不会因为导出坐标系改变而变样，差异体现在位置、速度、重力轴、相机外参与后续 3D 真值解释。</p>
    </section>
    {''.join(cards)}
  </div>
</body>
</html>
"""
    HTML.write_text(content, encoding="utf-8")
    print(HTML)


if __name__ == "__main__":
    main()
