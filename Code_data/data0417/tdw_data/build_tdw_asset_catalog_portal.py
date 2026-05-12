#!/usr/bin/env python3
"""Build a browsable TDW asset catalog portal from local metadata libraries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


METADATA_ROOT = Path("/home/gaoya/.venvs/tdw/lib/python3.10/site-packages/tdw/metadata_libraries")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Dataset_physV/0505TDW/tdw_asset_catalog_portal")
PORTAL_TITLE = "TDW Asset Catalog"

MODEL_LIBRARIES = [
    "models_core.json",
    "models_special.json",
    "models_flex.json",
    "models_full.json",
]

REALISTIC_SCENE_PREFIXES = (
    "mm_kitchen_",
    "mm_craftroom_",
    "floorplan_",
)

REALISTIC_CATEGORY_KEYWORDS = {
    "appliance",
    "backpack",
    "bag",
    "basket",
    "bed",
    "book",
    "bottle",
    "bowl",
    "cabinet",
    "can",
    "chair",
    "coffee",
    "cook",
    "cup",
    "dishwasher",
    "fork",
    "furniture",
    "glass",
    "handbag",
    "jar",
    "jug",
    "kettle",
    "kitchen",
    "knife",
    "lamp",
    "luggage",
    "microwave",
    "mug",
    "oven",
    "pan",
    "pillow",
    "plate",
    "pot",
    "purse",
    "refrigerator",
    "shelf",
    "shoe",
    "sofa",
    "spoon",
    "suitcase",
    "table",
    "teakettle",
    "toaster",
    "tool",
    "utensil",
    "vase",
    "wineglass",
}

UNREALISTIC_CATEGORY_KEYWORDS = {
    "animal",
    "toy",
    "plant",
    "flower",
    "tree",
    "weapon",
    "instrument",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a browsable TDW asset catalog portal.")
    parser.add_argument("--metadata_root", type=Path, default=METADATA_ROOT, help="TDW metadata_libraries directory.")
    parser.add_argument("--output_root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="Portal output directory.")
    parser.add_argument("--portal_title", type=str, default=PORTAL_TITLE, help="Portal title.")
    return parser.parse_args()


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def simplify_scene_record(name: str, record: dict[str, Any]) -> dict[str, Any]:
    rooms = record.get("rooms") or []
    return {
        "name": name,
        "location": str(record.get("location", "unknown")),
        "hdri": bool(record.get("hdri", False)),
        "room_count": int(len(rooms)),
        "description": str(record.get("description", "")),
        "recommended_realistic": bool(
            record.get("location") == "interior"
            and bool(record.get("hdri", False))
            and (
                name.startswith(REALISTIC_SCENE_PREFIXES)
                or name in {"archviz_house", "tdw_room", "tdw_room_4x5"}
            )
        ),
    }


def is_public_record(record: dict[str, Any]) -> bool:
    linux_url = str((record.get("urls") or {}).get("Linux", ""))
    return "tdw-public" in linux_url


def is_private_record(record: dict[str, Any]) -> bool:
    linux_url = str((record.get("urls") or {}).get("Linux", ""))
    return "tdw-private" in linux_url


def model_realism_flag(library: str, category: str, name: str, record: dict[str, Any]) -> bool:
    category_norm = category.lower()
    name_norm = name.lower()
    if record.get("do_not_use", False):
        return False
    if library == "models_flex.json":
        return False
    if any(token in category_norm for token in UNREALISTIC_CATEGORY_KEYWORDS):
        return False
    if any(token in category_norm for token in REALISTIC_CATEGORY_KEYWORDS):
        return True
    if any(token in name_norm for token in ("chair", "table", "bottle", "mug", "kettle", "pan", "pot", "bag", "backpack", "cabinet", "fridge", "microwave", "sofa", "pillow", "basket")):
        return True
    return False


def simplify_model_record(library: str, name: str, record: dict[str, Any]) -> dict[str, Any]:
    category = str(record.get("wcategory", "unknown"))
    return {
        "name": name,
        "library": library,
        "category": category,
        "flex": bool(record.get("flex", False)),
        "composite_object": bool(record.get("composite_object", False)),
        "do_not_use": bool(record.get("do_not_use", False)),
        "public_asset": is_public_record(record),
        "private_asset": is_private_record(record),
        "scale_factor": float(record.get("scale_factor", 1.0)),
        "recommended_realistic": model_realism_flag(library=library, category=category, name=name, record=record),
    }


def build_manifest(metadata_root: Path, portal_title: str) -> dict[str, Any]:
    scenes_raw = load_json(metadata_root / "scenes.json")["records"]
    scenes = [simplify_scene_record(name, record) for name, record in sorted(scenes_raw.items())]

    hdri_raw = load_json(metadata_root / "hdri_skyboxes.json")["records"]
    hdri_skyboxes = sorted(
        {
            "name": name,
            "sun_elevation": float(record.get("sun_elevation", 0.0)),
            "initial_skybox_rotation": float(record.get("initial_skybox_rotation", 0.0)),
            "exposure": float(record.get("exposure", 0.0)),
            "location": str(record.get("location", "unknown")),
        }
        for name, record in hdri_raw.items()
    )

    models: list[dict[str, Any]] = []
    library_stats: list[dict[str, Any]] = []
    for library in MODEL_LIBRARIES:
        library_raw = load_json(metadata_root / library)["records"]
        library_models = [simplify_model_record(library, name, record) for name, record in sorted(library_raw.items())]
        models.extend(library_models)
        categories = sorted({item["category"] for item in library_models})
        library_stats.append(
            {
                "library": library,
                "count": len(library_models),
                "public_count": sum(1 for item in library_models if item["public_asset"]),
                "private_count": sum(1 for item in library_models if item["private_asset"]),
                "do_not_use_count": sum(1 for item in library_models if item["do_not_use"]),
                "composite_count": sum(1 for item in library_models if item["composite_object"]),
                "flex_count": sum(1 for item in library_models if item["flex"]),
                "realistic_count": sum(1 for item in library_models if item["recommended_realistic"]),
                "category_count": len(categories),
            }
        )

    category_counts: dict[str, int] = {}
    for item in models:
        category_counts[item["category"]] = category_counts.get(item["category"], 0) + 1

    return {
        "portal_title": portal_title,
        "metadata_root": str(metadata_root),
        "summary": {
            "scene_count": len(scenes),
            "interior_scene_count": sum(1 for item in scenes if item["location"] == "interior"),
            "exterior_scene_count": sum(1 for item in scenes if item["location"] == "exterior"),
            "hdri_scene_count": sum(1 for item in scenes if item["hdri"]),
            "recommended_realistic_scene_count": sum(1 for item in scenes if item["recommended_realistic"]),
            "hdri_skybox_count": len(hdri_skyboxes),
            "model_count": len(models),
            "recommended_realistic_model_count": sum(1 for item in models if item["recommended_realistic"]),
        },
        "library_stats": library_stats,
        "scenes": scenes,
        "hdri_skyboxes": hdri_skyboxes,
        "models": models,
        "category_counts": [
            {"category": category, "count": count}
            for category, count in sorted(category_counts.items(), key=lambda item: (-item[1], item[0]))
        ],
    }


def build_index_html(portal_title: str) -> str:
    return f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{portal_title}</title>
  <style>
    :root {{
      --bg: #efe6d6;
      --panel: rgba(255, 252, 246, 0.96);
      --ink: #16130f;
      --muted: #6c655c;
      --accent: #3f6a56;
      --accent-soft: rgba(63, 106, 86, 0.12);
      --border: rgba(52, 42, 29, 0.14);
      --warn: #8d5a22;
      --bad: #8b3c35;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      color: var(--ink);
      font-family: Georgia, "Times New Roman", serif;
      background:
        radial-gradient(circle at top left, rgba(196, 162, 112, 0.26), transparent 28%),
        radial-gradient(circle at right 12%, rgba(126, 153, 141, 0.22), transparent 22%),
        linear-gradient(180deg, #f8f3ea 0%, var(--bg) 100%);
    }}
    .wrap {{ max-width: 1680px; margin: 0 auto; padding: 24px 18px 40px; }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 24px;
      box-shadow: 0 18px 40px rgba(45, 35, 22, 0.10);
    }}
    .hero {{
      padding: 28px;
      margin-bottom: 18px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: clamp(30px, 5vw, 54px);
      line-height: 0.95;
    }}
    h2 {{
      margin: 0 0 14px;
      font-size: 26px;
    }}
    p {{
      margin: 0;
      color: var(--muted);
      line-height: 1.65;
    }}
    .stats {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
      gap: 12px;
      margin-top: 18px;
    }}
    .stat {{
      padding: 16px 18px;
      border-radius: 18px;
      background: linear-gradient(180deg, rgba(255,255,255,0.92), rgba(243,236,226,0.96));
      border: 1px solid var(--border);
    }}
    .stat strong {{
      display: block;
      font-size: 28px;
      line-height: 1.1;
    }}
    .toolbar {{
      display: grid;
      grid-template-columns: 2fr 1fr 1fr 1fr;
      gap: 12px;
      margin-bottom: 18px;
    }}
    .panel {{
      padding: 20px;
      margin-bottom: 18px;
    }}
    input, select {{
      width: 100%;
      border: 1px solid var(--border);
      border-radius: 14px;
      padding: 12px 14px;
      font: inherit;
      background: rgba(255,255,255,0.92);
      color: var(--ink);
    }}
    .hint {{
      margin-top: 10px;
      font-size: 14px;
    }}
    .pill {{
      display: inline-block;
      margin-right: 8px;
      margin-bottom: 8px;
      padding: 6px 12px;
      border-radius: 999px;
      background: var(--accent-soft);
      color: var(--accent);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .pill.warn {{
      color: var(--warn);
      background: rgba(141, 90, 34, 0.12);
    }}
    .pill.bad {{
      color: var(--bad);
      background: rgba(139, 60, 53, 0.12);
    }}
    .grid {{
      display: grid;
      grid-template-columns: 1.25fr 0.75fr;
      gap: 18px;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    th, td {{
      text-align: left;
      padding: 10px 8px;
      border-bottom: 1px solid var(--border);
      vertical-align: top;
    }}
    th {{
      color: var(--muted);
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
    }}
    .table-wrap {{
      max-height: 880px;
      overflow: auto;
      border-radius: 18px;
      border: 1px solid var(--border);
      background: rgba(255,255,255,0.7);
    }}
    .mini-grid {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 18px;
    }}
    code {{
      font-family: "SFMono-Regular", Consolas, "Liberation Mono", monospace;
      font-size: 12px;
      word-break: break-all;
    }}
    .empty {{
      padding: 22px;
      color: var(--muted);
    }}
    @media (max-width: 1100px) {{
      .toolbar, .grid, .mini-grid {{
        grid-template-columns: 1fr;
      }}
      .table-wrap {{
        max-height: none;
      }}
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <section class="hero">
      <div class="pill">TDW Metadata</div>
      <div class="pill">Asset Catalog</div>
      <h1>{portal_title}</h1>
      <p>从本地 TDW metadata libraries 直接生成的资源总览页。支持查看场景、HDRI skyboxes、模型库规模，以及“真实物体优先”候选。这里的真实优先标记是启发式筛选，用来服务你后续选 scene / object，不代表官方标签。</p>
      <div class="stats" id="summary-stats"></div>
    </section>

    <section class="panel">
      <h2>筛选</h2>
      <div class="toolbar">
        <input id="search-box" type="text" placeholder="搜索 scene / model / category / library">
        <select id="dataset-select">
          <option value="all">全部数据</option>
          <option value="scenes">只看场景</option>
          <option value="models">只看模型</option>
        </select>
        <select id="library-select">
          <option value="all">全部模型库</option>
        </select>
        <select id="realistic-select">
          <option value="all">全部资源</option>
          <option value="recommended">只看真实优先</option>
          <option value="nonrecommended">排除真实优先</option>
        </select>
      </div>
      <p class="hint">建议优先组合：`mm_kitchen_* / mm_craftroom_* / floorplan_*` 场景 + `models_core.json` 里的家具、厨房、容器、电器类模型。</p>
    </section>

    <div class="grid">
      <section class="panel">
        <h2>场景</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Location</th>
                <th>HDRI</th>
                <th>Rooms</th>
                <th>Realistic</th>
                <th>Description</th>
              </tr>
            </thead>
            <tbody id="scenes-body"></tbody>
          </table>
        </div>
      </section>

      <section class="panel">
        <h2>模型库统计</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Library</th>
                <th>Count</th>
                <th>Public</th>
                <th>Private</th>
                <th>Do Not Use</th>
                <th>Composite</th>
                <th>Flex</th>
                <th>Realistic</th>
              </tr>
            </thead>
            <tbody id="library-body"></tbody>
          </table>
        </div>
      </section>
    </div>

    <div class="mini-grid">
      <section class="panel">
        <h2>高频类别</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Category</th>
                <th>Count</th>
              </tr>
            </thead>
            <tbody id="category-body"></tbody>
          </table>
        </div>
      </section>

      <section class="panel">
        <h2>HDRI Skyboxes</h2>
        <div class="table-wrap">
          <table>
            <thead>
              <tr>
                <th>Name</th>
                <th>Location</th>
                <th>Exposure</th>
                <th>Sun Elevation</th>
                <th>Rotation</th>
              </tr>
            </thead>
            <tbody id="hdri-body"></tbody>
          </table>
        </div>
      </section>
    </div>

    <section class="panel">
      <h2>模型</h2>
      <div class="table-wrap">
        <table>
          <thead>
            <tr>
              <th>Name</th>
              <th>Library</th>
              <th>Category</th>
              <th>Public</th>
              <th>Private</th>
              <th>Composite</th>
              <th>Flex</th>
              <th>Do Not Use</th>
              <th>Realistic</th>
            </tr>
          </thead>
          <tbody id="models-body"></tbody>
        </table>
      </div>
    </section>
  </div>

  <script>
    function pill(label, kind) {{
      const cls = kind ? `pill ${{kind}}` : "pill";
      return `<span class="${{cls}}">${{label}}</span>`;
    }}

    function boolLabel(v) {{
      return v ? "yes" : "no";
    }}

    function realismPill(v) {{
      return v ? pill("realistic focus", "") : pill("other", "warn");
    }}

    function recommendationFilter(v, mode) {{
      if (mode === "recommended") return v;
      if (mode === "nonrecommended") return !v;
      return true;
    }}

    async function main() {{
      const response = await fetch("manifest.json");
      const manifest = await response.json();

      const searchBox = document.getElementById("search-box");
      const datasetSelect = document.getElementById("dataset-select");
      const librarySelect = document.getElementById("library-select");
      const realisticSelect = document.getElementById("realistic-select");

      const summary = manifest.summary;
      const summaryStats = document.getElementById("summary-stats");
      summaryStats.innerHTML = [
        ["Scenes", summary.scene_count],
        ["Interior", summary.interior_scene_count],
        ["Exterior", summary.exterior_scene_count],
        ["Scene HDRI", summary.hdri_scene_count],
        ["HDRI Skyboxes", summary.hdri_skybox_count],
        ["Models", summary.model_count],
        ["Realistic Scenes", summary.recommended_realistic_scene_count],
        ["Realistic Models", summary.recommended_realistic_model_count],
      ].map(([label, value]) => `
        <div class="stat">
          <strong>${{value}}</strong>
          <span>${{label}}</span>
        </div>
      `).join("");

      librarySelect.innerHTML += manifest.library_stats
        .map(item => `<option value="${{item.library}}">${{item.library}}</option>`)
        .join("");

      document.getElementById("library-body").innerHTML = manifest.library_stats.map(item => `
        <tr>
          <td><code>${{item.library}}</code></td>
          <td>${{item.count}}</td>
          <td>${{item.public_count}}</td>
          <td>${{item.private_count}}</td>
          <td>${{item.do_not_use_count}}</td>
          <td>${{item.composite_count}}</td>
          <td>${{item.flex_count}}</td>
          <td>${{item.realistic_count}}</td>
        </tr>
      `).join("");

      document.getElementById("category-body").innerHTML = manifest.category_counts
        .slice(0, 80)
        .map(item => `
          <tr>
            <td>${{item.category}}</td>
            <td>${{item.count}}</td>
          </tr>
        `)
        .join("");

      document.getElementById("hdri-body").innerHTML = manifest.hdri_skyboxes
        .map(item => `
          <tr>
            <td><code>${{item.name}}</code></td>
            <td>${{item.location}}</td>
            <td>${{item.exposure.toFixed(2)}}</td>
            <td>${{item.sun_elevation.toFixed(2)}}</td>
            <td>${{item.initial_skybox_rotation.toFixed(2)}}</td>
          </tr>
        `)
        .join("");

      function render() {{
        const query = searchBox.value.trim().toLowerCase();
        const datasetMode = datasetSelect.value;
        const libraryMode = librarySelect.value;
        const realisticMode = realisticSelect.value;

        const filteredScenes = manifest.scenes.filter(item => {{
          if (datasetMode === "models") return false;
          if (!recommendationFilter(item.recommended_realistic, realisticMode)) return false;
          const text = [item.name, item.location, item.description].join(" ").toLowerCase();
          return !query || text.includes(query);
        }});

        document.getElementById("scenes-body").innerHTML = filteredScenes.length ? filteredScenes.map(item => `
          <tr>
            <td><code>${{item.name}}</code></td>
            <td>${{item.location}}</td>
            <td>${{boolLabel(item.hdri)}}</td>
            <td>${{item.room_count}}</td>
            <td>${{realismPill(item.recommended_realistic)}}</td>
            <td>${{item.description}}</td>
          </tr>
        `).join("") : `<tr><td colspan="6" class="empty">没有匹配的场景。</td></tr>`;

        const filteredModels = manifest.models.filter(item => {{
          if (datasetMode === "scenes") return false;
          if (libraryMode !== "all" && item.library !== libraryMode) return false;
          if (!recommendationFilter(item.recommended_realistic, realisticMode)) return false;
          const text = [item.name, item.library, item.category].join(" ").toLowerCase();
          return !query || text.includes(query);
        }});

        document.getElementById("models-body").innerHTML = filteredModels.length ? filteredModels.map(item => `
          <tr>
            <td><code>${{item.name}}</code></td>
            <td><code>${{item.library}}</code></td>
            <td>${{item.category}}</td>
            <td>${{boolLabel(item.public_asset)}}</td>
            <td>${{boolLabel(item.private_asset)}}</td>
            <td>${{boolLabel(item.composite_object)}}</td>
            <td>${{boolLabel(item.flex)}}</td>
            <td>${{item.do_not_use ? pill("do not use", "bad") : pill("ok", "")}}</td>
            <td>${{realismPill(item.recommended_realistic)}}</td>
          </tr>
        `).join("") : `<tr><td colspan="9" class="empty">没有匹配的模型。</td></tr>`;
      }}

      searchBox.addEventListener("input", render);
      datasetSelect.addEventListener("change", render);
      librarySelect.addEventListener("change", render);
      realisticSelect.addEventListener("change", render);
      render();
    }}

    main();
  </script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(metadata_root=Path(args.metadata_root), portal_title=str(args.portal_title))
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    (output_root / "index.html").write_text(build_index_html(str(args.portal_title)), encoding="utf-8")
    print(output_root / "index.html")


if __name__ == "__main__":
    main()
