#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import html
import json
from pathlib import Path


DEFAULT_INPUT_ROOT = Path("/data/gaoya/agent-data/outputs/dataset_new_0705/AAA_check_0710")
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local HTML overview page for dataset_new_0705 batch outputs.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--page-title", default="AAA Check 0710 Batch Overview")
    return parser.parse_args()


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _format_object_line(obj: dict) -> str:
    texture = obj.get("texture_asset") or obj.get("texture_style") or "-"
    mass = obj.get("mass")
    friction = obj.get("friction")
    restitution = obj.get("restitution")
    return (
        f"{obj.get('name', '-')}"
        f" / {obj.get('shape', '-')}"
        f" / role={obj.get('role', '-')}"
        f" / tex={texture}"
        f" / m={mass:.3f}"
        f" / mu={friction:.3f}"
        f" / e={restitution:.3f}"
    )


def load_cases(input_root: Path) -> list[dict]:
    manifest_path = input_root / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"missing manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    cases: list[dict] = []
    for item in manifest:
        meta_path = Path(item["meta"])
        meta = _read_json(meta_path)
        video_rel = _relative_to_root(Path(item["video"]), input_root)
        case = {
            "case_id": item["case_id"],
            "family_key": item["family_key"],
            "seed": item["seed"],
            "video_rel": video_rel,
            "title": meta.get("title", item["case_id"]),
            "description": meta.get("description", ""),
            "family_label": meta.get("family", item["family_key"]),
            "floor_friction": meta.get("floor_friction"),
            "duration_s": meta.get("duration_s"),
            "pre_roll_s": meta.get("pre_roll_s"),
            "surface_key": meta.get("surface_key", ""),
            "lighting_key": meta.get("lighting_key", ""),
            "camera_key": meta.get("blueprint", {}).get("camera_key", ""),
            "tags": meta.get("tags", []),
            "objects": meta.get("objects", []),
            "resolution": meta.get("resolution", []),
        }
        cases.append(case)
    cases.sort(key=lambda item: (item["family_key"], item["case_id"]))
    return cases


def build_page(cases: list[dict], output_root: Path, page_title: str) -> Path:
    family_counts = collections.Counter(case["family_key"] for case in cases)
    family_buttons = ['<button class="chip active" data-family="ALL">ALL</button>']
    for family_key in sorted(family_counts.keys(), key=lambda value: (len(value), value)):
        family_buttons.append(
            f'<button class="chip" data-family="{html.escape(family_key)}">{html.escape(family_key)} · {family_counts[family_key]}</button>'
        )

    cards: list[str] = []
    for case in cases:
        tags_html = "".join(f"<span>{html.escape(str(tag))}</span>" for tag in case["tags"])
        object_lines = "".join(
            f"<li>{html.escape(_format_object_line(obj))}</li>"
            for obj in case["objects"]
        )
        res_text = "x".join(str(v) for v in case["resolution"]) if case["resolution"] else "-"
        cards.append(
            f"""
            <article class="card" data-family="{html.escape(case['family_key'])}">
              <div class="card-top">
                <div>
                  <div class="eyebrow">{html.escape(case['family_label'])} · {html.escape(case['case_id'])}</div>
                  <h2>{html.escape(case['title'])}</h2>
                  <p class="desc">{html.escape(case['description'])}</p>
                </div>
                <div class="metrics">
                  <span>seed {case['seed']}</span>
                  <span>{html.escape(case['surface_key'])}</span>
                  <span>{html.escape(case['lighting_key'])}</span>
                  <span>{html.escape(case['camera_key'])}</span>
                </div>
              </div>
              <video controls preload="metadata" playsinline>
                <source src="{html.escape(case['video_rel'])}" type="video/mp4">
              </video>
              <div class="meta-grid">
                <div><strong>duration</strong><span>{case['duration_s']:.2f}s</span></div>
                <div><strong>pre-roll</strong><span>{case['pre_roll_s']:.3f}s</span></div>
                <div><strong>floor_mu</strong><span>{case['floor_friction']:.3f}</span></div>
                <div><strong>resolution</strong><span>{html.escape(res_text)}</span></div>
              </div>
              <div class="tags">{tags_html}</div>
              <ul class="object-list">{object_lines}</ul>
            </article>
            """
        )

    page = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>{html.escape(page_title)}</title>
  <style>
    :root {{
      --bg: #f3efe7;
      --ink: #1f252a;
      --muted: #66717b;
      --panel: rgba(255, 250, 242, 0.92);
      --line: rgba(31, 37, 42, 0.10);
      --accent: #b55d32;
      --accent-soft: #ead2b8;
      --accent-cool: #5f7c8a;
      --shadow: 0 20px 40px rgba(62, 38, 22, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    html {{ scroll-behavior: smooth; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(181, 93, 50, 0.12), transparent 28%),
        radial-gradient(circle at top right, rgba(95, 124, 138, 0.14), transparent 24%),
        linear-gradient(180deg, #f7f2ea 0%, #f1ebe1 48%, #ece6db 100%);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1500px, calc(100vw - 32px));
      margin: 0 auto;
      padding: 28px 0 56px;
    }}
    .hero {{
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(18px);
      background: rgba(243, 239, 231, 0.78);
      border: 1px solid rgba(255, 255, 255, 0.45);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 24px 24px 18px;
      margin-bottom: 22px;
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.12em;
      text-transform: uppercase;
      color: var(--accent-cool);
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0;
      font-size: clamp(28px, 3.8vw, 50px);
      line-height: 1.02;
      letter-spacing: -0.04em;
    }}
    .subtitle {{
      margin: 10px 0 0;
      max-width: 980px;
      color: var(--muted);
      line-height: 1.6;
      font-size: 15px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .summary-box {{
      background: linear-gradient(180deg, rgba(255,255,255,0.75), rgba(255,248,240,0.86));
      border: 1px solid var(--line);
      border-radius: 18px;
      padding: 14px 16px;
    }}
    .summary-box strong {{
      display: block;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.10em;
      color: var(--accent-cool);
      margin-bottom: 6px;
    }}
    .summary-box span {{
      font-size: 26px;
      font-weight: 700;
      letter-spacing: -0.03em;
    }}
    .filters {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 18px;
    }}
    .chip {{
      border: 0;
      border-radius: 999px;
      padding: 10px 14px;
      background: rgba(255, 255, 255, 0.75);
      color: var(--ink);
      box-shadow: inset 0 0 0 1px var(--line);
      cursor: pointer;
      font: inherit;
      transition: transform 160ms ease, background 160ms ease, color 160ms ease;
    }}
    .chip:hover {{
      transform: translateY(-1px);
      background: #fff;
    }}
    .chip.active {{
      background: var(--accent);
      color: #fffaf3;
      box-shadow: none;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    .card {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.45);
      border-radius: 24px;
      padding: 18px;
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .card.hidden {{
      display: none;
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
    }}
    .card h2 {{
      margin: 4px 0 6px;
      font-size: 22px;
      line-height: 1.1;
      letter-spacing: -0.03em;
    }}
    .desc {{
      margin: 0;
      color: var(--muted);
      line-height: 1.55;
      font-size: 14px;
    }}
    .metrics {{
      display: flex;
      flex-wrap: wrap;
      justify-content: flex-end;
      gap: 8px;
      min-width: 180px;
      align-content: flex-start;
    }}
    .metrics span, .tags span {{
      display: inline-flex;
      align-items: center;
      padding: 6px 9px;
      border-radius: 999px;
      font-size: 12px;
      background: rgba(234, 210, 184, 0.52);
      color: #684329;
    }}
    video {{
      width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 16px;
      background: #000;
      border: 1px solid rgba(0, 0, 0, 0.08);
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }}
    .meta-grid div {{
      background: rgba(255,255,255,0.68);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px 12px;
    }}
    .meta-grid strong {{
      display: block;
      font-size: 11px;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      color: var(--accent-cool);
      margin-bottom: 6px;
    }}
    .meta-grid span {{
      font-size: 14px;
      font-weight: 600;
    }}
    .tags {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
    }}
    .object-list {{
      margin: 0;
      padding-left: 18px;
      color: var(--muted);
      line-height: 1.55;
      font-size: 13px;
    }}
    .footer {{
      margin-top: 20px;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 860px) {{
      .shell {{
        width: min(100vw, calc(100vw - 16px));
        padding-top: 10px;
      }}
      .hero {{
        border-radius: 20px;
        padding: 18px 16px 16px;
      }}
      .card-top {{
        flex-direction: column;
      }}
      .metrics {{
        justify-content: flex-start;
        min-width: 0;
      }}
      .meta-grid {{
        grid-template-columns: repeat(2, minmax(0, 1fr));
      }}
      .grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">dataset_new_0705 · local overview</div>
      <h1>{html.escape(page_title)}</h1>
      <p class="subtitle">
        这个页面汇总了当前 batch 的全部视频结果。页面直接读取同目录下的 `cases/*/*/videos/*.mp4`，
        可以按 family 快速筛选，也能顺手检查标题、材质、摩擦系数、相机和对象配置。
      </p>
      <div class="summary">
        <div class="summary-box"><strong>总 Case 数</strong><span>{len(cases)}</span></div>
        <div class="summary-box"><strong>Family 数</strong><span>{len(family_counts)}</span></div>
        <div class="summary-box"><strong>视频根目录</strong><span style="font-size:14px;line-height:1.5">{html.escape(str(output_root))}</span></div>
      </div>
      <div class="filters">
        {"".join(family_buttons)}
      </div>
    </section>
    <section class="grid" id="case-grid">
      {"".join(cards)}
    </section>
    <p class="footer">生成页入口：index.html。若切换到新的 batch 输出，只需要重新运行本脚本刷新页面即可。</p>
  </main>
  <script>
    const chips = Array.from(document.querySelectorAll('.chip'));
    const cards = Array.from(document.querySelectorAll('.card'));
    function setFamily(family) {{
      chips.forEach((chip) => {{
        chip.classList.toggle('active', chip.dataset.family === family);
      }});
      cards.forEach((card) => {{
        const visible = family === 'ALL' || card.dataset.family === family;
        card.classList.toggle('hidden', !visible);
      }});
    }}
    chips.forEach((chip) => {{
      chip.addEventListener('click', () => setFamily(chip.dataset.family));
    }});
  </script>
</body>
</html>
"""
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / "index.html"
    output_path.write_text(page, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    cases = load_cases(args.input_root)
    output_path = build_page(cases, args.output_root, args.page_title)
    print(json.dumps({"cases": len(cases), "index_html": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
