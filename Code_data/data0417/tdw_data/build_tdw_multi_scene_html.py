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
        "title": "Cloth Drop Real Scene From Baseline",
        "subtitle": "从已验证可下坠的 baseline cloth 配置迁移到真实场景后的两组 case：球直接落地摆放，以及球放在更接近现实的矮台面上。",
        "directory": "tdw_cloth_drop_real_scene_from_baseline",
        "badge": "Cloth Real Scene",
        "mode": "flat_videos",
        "scene_label": "tdw_cloth_drop_real_scene_from_baseline",
    },
    {
        "title": "Axis Export Compare",
        "subtitle": "同一段 TDW 仿真分别按 y-up 原生导出和 z-up 离线转换导出后的对比。重点说明：渲染视频本身不变，变化发生在坐标、重力轴和后续 3D 真值解释上。",
        "directory": "tdw_axis_export_compare",
        "badge": "Axis Compare",
        "mode": "flat_videos",
        "scene_label": "tdw_axis_export_compare",
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
        "title": "Stable Cloth Compare Cases",
        "subtitle": "保留修正后的稳定版 cloth 对比 case，仅展示最终采用的稳定结果。",
        "directory": "tdw_cloth_ab_compare",
        "badge": "Stable Compare",
        "mode": "stable_compare",
    },
    {
        "title": "Real Scene Soft Materials",
        "subtitle": "真实室内场景下的非刚体材料示例，包含软体体积、液体和颗粒材料。",
        "directory": "tdw_real_scene_soft_materials",
        "badge": "Soft Material",
        "mode": "flat_videos",
        "scene_label": "tdw_real_scene_soft_materials",
    },
    {
        "title": "Water Parameter Sweep",
        "subtitle": "同一场景同一机位下的多组水体参数对比，用于观察液体从偏稠、偏平滑到偏飞溅的观感差异。",
        "directory": "tdw_water_param_sweep",
        "badge": "Water Sweep",
        "mode": "flat_videos",
        "scene_label": "tdw_water_param_sweep",
    },
    {
        "title": "Continuous Water Streams",
        "subtitle": "针对连续大水流单独调的一组参数，优先保证可见性、体积感和连贯水柱，而不是颗粒飞溅。",
        "directory": "tdw_water_continuous_streams",
        "badge": "Water Stream",
        "mode": "flat_videos",
        "scene_label": "tdw_water_continuous_streams",
    },
    {
        "title": "Non-Fluid Soft Bodies",
        "subtitle": "不包含液体的软体批次，集中展示布料和充气软体的真实场景仿真。",
        "directory": "tdw_nonfluid_soft_bodies",
        "badge": "Soft Body",
        "mode": "flat_videos",
        "scene_label": "tdw_nonfluid_soft_bodies",
    },
    {
        "title": "Non-Fluid Soft Bodies Multi Scene",
        "subtitle": "同一组布料和充气软体 case 切换到不同真实场景背景下运行，用于比较背景、空间和光照变化对软体观感的影响。",
        "directory": "tdw_nonfluid_soft_bodies_multi_scene",
        "badge": "Soft Multi Scene",
        "mode": "scene_dirs",
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


def list_stable_compare_videos(dataset_dir: Path):
    if not dataset_dir.exists():
        return []
    videos = []
    for case_dir in sorted(p for p in dataset_dir.iterdir() if p.is_dir()):
        stable = case_dir.joinpath("stable.mp4")
        if stable.exists():
            videos.append((case_dir.name, stable))
    return videos


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
        videos = list_stable_compare_videos(dataset_dir)
        if not videos:
            cards.append('<p class="empty">当前还没有生成视频。</p></section>')
            continue
        cards.append('<div class="grid">')
        for case_name, video in videos:
            rel = str(video.relative_to(ROOT))
            abs_path = str(video)
            cards.append(
                f'''<article class="card">
  <video controls preload="metadata" src="{rel}"></video>
  <div class="meta">
    <span class="pill">{dataset["badge"]}</span>
    <span class="pill scene-pill">Stable</span>
    <h4>{case_name}</h4>
    <code>{abs_path}</code>
  </div>
</article>'''
            )
        cards.append("</div>")
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
