#!/usr/bin/env python3
from __future__ import annotations

import argparse
import collections
import html
import json
from pathlib import Path

from .scene_generators_0705 import generate_scenario_blueprint


DEFAULT_INPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0713pybullet")
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT
DEFAULT_PORT = 18831


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a local overview page with diversity tables and videos for a rigid dataset batch.")
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--page-title", default="0713 PyBullet Multifamily Diversity Overview")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    return parser.parse_args()


def _read_json(path: Path) -> dict | list:
    return json.loads(path.read_text(encoding="utf-8"))


def _relative_to_root(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def _format_object_line(obj: dict, material_key: str) -> str:
    mass = obj.get("mass")
    friction = obj.get("friction")
    restitution = obj.get("restitution")
    return (
        f"{obj.get('name', '-')}"
        f" / {obj.get('shape', '-')}"
        f" / role={obj.get('role', '-')}"
        f" / mat={material_key or '-'}"
        f" / m={mass:.3f}"
        f" / mu={friction:.3f}"
        f" / e={restitution:.3f}"
    )


def _table_html(title: str, headers: list[str], rows: list[list[str]]) -> str:
    head_html = "".join(f"<th>{html.escape(header)}</th>" for header in headers)
    row_html = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"""
      <section class="stat-card">
        <h3>{html.escape(title)}</h3>
        <div class="table-wrap">
          <table>
            <thead><tr>{head_html}</tr></thead>
            <tbody>{row_html}</tbody>
          </table>
        </div>
      </section>
    """


def load_dataset(input_root: Path) -> tuple[list[dict], dict[str, object]]:
    manifest_path = input_root / "manifest.json"
    manifest = _read_json(manifest_path)

    cases: list[dict] = []
    family_counter: collections.Counter[str] = collections.Counter()
    object_family_counter: collections.Counter[str] = collections.Counter()
    material_counter: collections.Counter[str] = collections.Counter()
    surface_counter: collections.Counter[str] = collections.Counter()
    camera_counter: collections.Counter[str] = collections.Counter()
    motion_counter: collections.Counter[str] = collections.Counter()
    direction_counter: collections.Counter[str] = collections.Counter()
    dynamic_count_counter: collections.Counter[str] = collections.Counter()
    total_count_counter: collections.Counter[str] = collections.Counter()

    family_examples: dict[str, dict] = {}

    for item in manifest:
        meta_path = Path(item["meta"])
        meta = _read_json(meta_path)
        family_key = str(item["family_key"])
        case_id = str(item["case_id"])
        seed = int(item["seed"])
        sample_key = Path(item["output_root"]).name
        blueprint = generate_scenario_blueprint(family_key=family_key, sample_key=sample_key, seed=seed)

        video_rel = _relative_to_root(Path(item["video"]), input_root)
        material_map = meta.get("materials", {})
        object_lines = [
            _format_object_line(obj, str(material_map.get(obj.get("name", ""), "")))
            for obj in meta.get("objects", [])
        ]
        dynamic_objects = [obj for obj in meta.get("objects", []) if obj.get("dynamic")]
        motion_tag = str(meta.get("tags", ["unknown"])[-1]) if meta.get("tags") else "unknown"
        camera_key = str(meta.get("blueprint", {}).get("camera_key", ""))
        surface_key = str(meta.get("surface_key", ""))
        direction_mode = str(
            meta.get("blueprint", {}).get("metadata", {}).get(
                "direction_mode",
                item.get("direction_mode", "legacy_unlabeled"),
            )
        )

        case = {
            "case_id": case_id,
            "family_key": family_key,
            "seed": seed,
            "video_rel": video_rel,
            "title": meta.get("title", case_id),
            "description": meta.get("description", ""),
            "caption": item.get("caption", meta.get("caption", "")),
            "camera_key": camera_key,
            "surface_key": surface_key,
            "lighting_key": str(meta.get("lighting_key", "")),
            "motion_tag": motion_tag,
            "direction_mode": direction_mode,
            "resolution": meta.get("resolution", []),
            "duration_s": float(meta.get("duration_s", 0.0)),
            "pre_roll_s": float(meta.get("pre_roll_s", 0.0)),
            "objects": object_lines,
            "dynamic_count": len(dynamic_objects),
            "total_count": len(meta.get("objects", [])),
            "material_keys": sorted(set(str(v) for v in material_map.values())),
            "object_families": [obj.family_key for obj in blueprint.objects],
            "tags": [str(tag) for tag in meta.get("tags", [])],
        }
        cases.append(case)

        family_counter[family_key] += 1
        surface_counter[surface_key] += 1
        camera_counter[camera_key] += 1
        motion_counter[motion_tag] += 1
        direction_counter[direction_mode] += 1
        dynamic_count_counter[str(len(dynamic_objects))] += 1
        total_count_counter[str(len(meta.get("objects", [])))] += 1
        for material_key in material_map.values():
            material_counter[str(material_key)] += 1
        for obj in blueprint.objects:
            object_family_counter[obj.family_key] += 1

        family_examples.setdefault(family_key, case)

    cases.sort(key=lambda item: (item["family_key"], item["case_id"]))

    stats = {
        "case_count": len(cases),
        "family_counter": dict(family_counter),
        "object_family_counter": dict(object_family_counter),
        "material_counter": dict(material_counter),
        "surface_counter": dict(surface_counter),
        "camera_counter": dict(camera_counter),
        "motion_counter": dict(motion_counter),
        "direction_counter": dict(direction_counter),
        "dynamic_count_counter": dict(dynamic_count_counter),
        "total_count_counter": dict(total_count_counter),
        "family_examples": family_examples,
    }
    return cases, stats


def build_page(cases: list[dict], stats: dict[str, object], output_root: Path, page_title: str, port: int) -> Path:
    family_counts = collections.Counter(case["family_key"] for case in cases)
    family_buttons = ['<button class="chip active" data-family="ALL">ALL</button>']
    for family_key in sorted(family_counts.keys(), key=lambda value: (len(value), value)):
        family_buttons.append(
            f'<button class="chip" data-family="{html.escape(family_key)}">{html.escape(family_key)} · {family_counts[family_key]}</button>'
        )

    summary_tables = [
        _table_html(
            "Family 分布",
            ["Family", "Case 数"],
            [[key, str(value)] for key, value in sorted(stats["family_counter"].items())],
        ),
        _table_html(
            "Object Family 计数",
            ["Object Family", "实例数"],
            [[key, str(value)] for key, value in sorted(stats["object_family_counter"].items(), key=lambda item: (-item[1], item[0]))],
        ),
        _table_html(
            "Material 计数",
            ["Material Key", "实例数"],
            [[key, str(value)] for key, value in sorted(stats["material_counter"].items(), key=lambda item: (-item[1], item[0]))],
        ),
        _table_html(
            "Scene / Camera / Motion",
            ["属性", "计数"],
            [[f"surface:{key}", str(value)] for key, value in sorted(stats["surface_counter"].items())]
            + [[f"camera:{key}", str(value)] for key, value in sorted(stats["camera_counter"].items())]
            + [[f"direction:{key}", str(value)] for key, value in sorted(stats["direction_counter"].items())]
            + [[f"motion:{key}", str(value)] for key, value in sorted(stats["motion_counter"].items(), key=lambda item: (-item[1], item[0]))],
        ),
    ]

    family_examples = [stats["family_examples"][key] for key in sorted(stats["family_examples"].keys(), key=lambda value: (len(value), value))]
    example_cards = []
    for case in family_examples:
        example_cards.append(
            f"""
            <article class="example-card" data-family="{html.escape(case['family_key'])}">
              <div class="example-top">
                <div class="eyebrow">{html.escape(case['family_key'])} · representative</div>
                <h3>{html.escape(case['caption'])}</h3>
              </div>
              <video controls preload="metadata" playsinline>
                <source src="{html.escape(case['video_rel'])}" type="video/mp4">
              </video>
            </article>
            """
        )

    cards = []
    for case in cases:
        tags_html = "".join(f"<span>{html.escape(tag)}</span>" for tag in case["tags"])
        object_lines = "".join(f"<li>{html.escape(line)}</li>" for line in case["objects"])
        material_text = ", ".join(case["material_keys"]) if case["material_keys"] else "-"
        resolution_text = "x".join(str(v) for v in case["resolution"]) if case["resolution"] else "-"
        cards.append(
            f"""
            <article class="card" data-family="{html.escape(case['family_key'])}">
              <div class="card-top">
                <div>
                  <div class="eyebrow">{html.escape(case['family_key'])} · {html.escape(case['case_id'])}</div>
                  <h2>{html.escape(case['caption'])}</h2>
                  <p class="desc">{html.escape(case['description'])}</p>
                </div>
                <div class="metrics">
                  <span>seed {case['seed']}</span>
                  <span>{html.escape(case['surface_key'])}</span>
                  <span>{html.escape(case['camera_key'])}</span>
                  <span>{html.escape(case['direction_mode'])}</span>
                  <span>{html.escape(case['motion_tag'])}</span>
                </div>
              </div>
              <video controls preload="metadata" playsinline>
                <source src="{html.escape(case['video_rel'])}" type="video/mp4">
              </video>
              <div class="meta-grid">
                <div><strong>resolution</strong><span>{html.escape(resolution_text)}</span></div>
                <div><strong>duration</strong><span>{case['duration_s']:.2f}s</span></div>
                <div><strong>dynamic objs</strong><span>{case['dynamic_count']}</span></div>
                <div><strong>total objs</strong><span>{case['total_count']}</span></div>
              </div>
              <div class="meta-line"><strong>materials</strong><span>{html.escape(material_text)}</span></div>
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
      --bg: #ede7dc;
      --ink: #1f252a;
      --muted: #606a73;
      --panel: rgba(255, 251, 245, 0.92);
      --line: rgba(31, 37, 42, 0.10);
      --accent: #915333;
      --accent-soft: #ead8c1;
      --accent-cool: #496d7b;
      --shadow: 0 18px 36px rgba(39, 29, 20, 0.10);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: "IBM Plex Sans", "Source Han Sans SC", "Noto Sans SC", sans-serif;
      background:
        radial-gradient(circle at top left, rgba(145, 83, 51, 0.16), transparent 26%),
        radial-gradient(circle at top right, rgba(73, 109, 123, 0.16), transparent 24%),
        linear-gradient(180deg, #f5efe6 0%, #efe8de 48%, #e9e2d8 100%);
      min-height: 100vh;
    }}
    .shell {{
      width: min(1560px, calc(100vw - 28px));
      margin: 0 auto;
      padding: 20px 0 52px;
    }}
    .hero {{
      position: sticky;
      top: 0;
      z-index: 10;
      backdrop-filter: blur(18px);
      background: rgba(245, 239, 230, 0.82);
      border: 1px solid rgba(255,255,255,0.5);
      border-radius: 28px;
      box-shadow: var(--shadow);
      padding: 24px;
      margin-bottom: 20px;
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
      font-size: clamp(28px, 4vw, 52px);
      line-height: 1.02;
      letter-spacing: -0.04em;
    }}
    .subtitle {{
      margin: 10px 0 0;
      color: var(--muted);
      max-width: 980px;
      line-height: 1.6;
      font-size: 15px;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .summary-box {{
      background: linear-gradient(180deg, rgba(255,255,255,0.76), rgba(255,248,242,0.88));
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
      font-size: 24px;
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
      background: rgba(255,255,255,0.78);
      color: var(--ink);
      box-shadow: inset 0 0 0 1px var(--line);
      cursor: pointer;
      font: inherit;
    }}
    .chip.active {{
      background: var(--accent);
      color: #fffaf2;
      box-shadow: none;
    }}
    .stats-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 20px;
    }}
    .stat-card, .card, .example-card {{
      background: var(--panel);
      border: 1px solid rgba(255,255,255,0.45);
      border-radius: 22px;
      box-shadow: var(--shadow);
    }}
    .stat-card {{
      padding: 16px;
    }}
    .stat-card h3 {{
      margin: 0 0 12px;
      font-size: 18px;
      letter-spacing: -0.02em;
    }}
    .table-wrap {{
      max-height: 360px;
      overflow: auto;
      border-radius: 14px;
      border: 1px solid var(--line);
      background: rgba(255,255,255,0.74);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 13px;
    }}
    th, td {{
      padding: 9px 10px;
      border-bottom: 1px solid rgba(31, 37, 42, 0.08);
      text-align: left;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: rgba(237, 231, 220, 0.95);
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.09em;
      color: var(--accent-cool);
    }}
    .example-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(320px, 1fr));
      gap: 16px;
      margin-bottom: 22px;
    }}
    .example-card {{
      padding: 14px;
    }}
    .example-top h3 {{
      margin: 4px 0 10px;
      font-size: 18px;
      line-height: 1.2;
    }}
    .grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(360px, 1fr));
      gap: 18px;
    }}
    .card {{
      padding: 18px;
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .card.hidden, .example-card.hidden {{
      display: none;
    }}
    .card-top {{
      display: flex;
      justify-content: space-between;
      gap: 14px;
    }}
    .card h2 {{
      margin: 4px 0 6px;
      font-size: 21px;
      line-height: 1.15;
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
      background: rgba(234, 216, 193, 0.52);
      color: #684329;
    }}
    video {{
      width: 100%;
      aspect-ratio: 16 / 9;
      border-radius: 16px;
      background: #000;
      border: 1px solid rgba(0,0,0,0.08);
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 8px;
    }}
    .meta-grid div, .meta-line {{
      background: rgba(255,255,255,0.68);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 10px 12px;
    }}
    .meta-grid strong, .meta-line strong {{
      display: block;
      font-size: 11px;
      letter-spacing: 0.10em;
      text-transform: uppercase;
      color: var(--accent-cool);
      margin-bottom: 6px;
    }}
    .meta-grid span, .meta-line span {{
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
      .grid, .example-grid, .stats-grid {{
        grid-template-columns: 1fr;
      }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <section class="hero">
      <div class="eyebrow">rigid dataset · local overview</div>
      <h1>{html.escape(page_title)}</h1>
      <p class="subtitle">
        这个页面把当前正式数据集的多样性统计和实例视频放到同一个入口里。上半部分是 family / object / material / motion 的计数表，
        下半部分是每个 family 的代表视频和全部 case 卡片，方便一边看分布一边快速抽样检查画面质量。
      </p>
      <div class="summary">
        <div class="summary-box"><strong>总 Case 数</strong><span>{stats["case_count"]}</span></div>
        <div class="summary-box"><strong>Family 数</strong><span>{len(stats["family_counter"])}</span></div>
        <div class="summary-box"><strong>Object Family 覆盖</strong><span>{len(stats["object_family_counter"])}</span></div>
        <div class="summary-box"><strong>Material 覆盖</strong><span>{len(stats["material_counter"])}</span></div>
        <div class="summary-box"><strong>访问端口</strong><span>{port}</span></div>
      </div>
      <div class="filters">
        {"".join(family_buttons)}
      </div>
    </section>
    <section class="stats-grid">
      {"".join(summary_tables)}
    </section>
    <section class="example-grid" id="example-grid">
      {"".join(example_cards)}
    </section>
    <section class="grid" id="case-grid">
      {"".join(cards)}
    </section>
    <p class="footer">刷新页面前如果有新 case 落盘，重新运行本脚本即可更新统计和视频索引。</p>
  </main>
  <script>
    const chips = Array.from(document.querySelectorAll('.chip'));
    const cards = Array.from(document.querySelectorAll('.card'));
    const examples = Array.from(document.querySelectorAll('.example-card'));
    function setFamily(family) {{
      chips.forEach((chip) => {{
        chip.classList.toggle('active', chip.dataset.family === family);
      }});
      cards.forEach((card) => {{
        const visible = family === 'ALL' || card.dataset.family === family;
        card.classList.toggle('hidden', !visible);
      }});
      examples.forEach((card) => {{
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
    cases, stats = load_dataset(args.input_root)
    output_path = build_page(cases, stats, args.output_root, args.page_title, args.port)
    print(json.dumps({"cases": len(cases), "index_html": str(output_path), "port": args.port}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
