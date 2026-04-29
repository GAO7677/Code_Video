#!/usr/bin/env python3
"""Build a local HTML portal for adapter-dataset composition proposals."""

from __future__ import annotations

import argparse
import html
import json
import os
import shutil
import socket
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


SCRIPT_DIR = Path(__file__).resolve().parent
STATE_ADAPTER_ROOT = SCRIPT_DIR.parent
TRAIN_ROOT = STATE_ADAPTER_ROOT.parent
GENESIS_CASE_JSON = Path(
    "/home/gaoya/Code_Video/Code_data/data0417/genesis_rigid_data/try1_physxnet_articulation_mpm0417.json"
)
RAW_TRAIN_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/train/rigid"
)
WINDOW_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0417data/version_1_genesis_rigid_data_all_cases/preprocess_v1/oracle_wan_ctx8_fut5_9_13_alltrain"
)
DEFAULT_OUTPUT_DIR = Path("/home/gaoya/portal_hub_sim/adapter_dataset_proposal")


SCENARIO_SPECS = [
    {
        "slug": "core_static_anchor",
        "title": "Core A: 单物体静置锚点",
        "status": "core",
        "goal": "给 adapter 一个最稳定的低难度起点，学习近静止、低位移、低遮挡的未来状态调制。",
        "why": "这组 case 视觉和状态最干净，最适合当第一阶段基线。",
        "case_ids": [0, 1, 2],
    },
    {
        "slug": "core_gravity_drop",
        "title": "Core B: 单物体重力下坠",
        "status": "core",
        "goal": "引入清晰的 z 向重力趋势和落地事件，但仍保持单主物体、低交互复杂度。",
        "why": "相比随机抛射，这组动力学更规则，适合 adapter 先学时间对齐。",
        "case_ids": [3, 901],
    },
    {
        "slug": "core_entry_motion",
        "title": "Core C: 单物体入场平移",
        "status": "core",
        "goal": "让 adapter 学到明显的横向位移和轻微转动，增加速度变化但不一下子拉太复杂。",
        "why": "005/006/007 比随机抛物线更可控，也更贴近简单运动建模。",
        "case_ids": [5, 6, 7],
    },
    {
        "slug": "expand_two_object_nocollision",
        "title": "Expand D: 双物体无互撞独立运动",
        "status": "expand",
        "goal": "在不引入物体间碰撞的前提下，扩展到多物体状态 token 和更丰富的遮挡/可见性模式。",
        "why": "适合作为第二阶段扩展，先用 count=2；count=3/4 先不急着进主训练集。",
        "case_ids": [210, 211],
    },
    {
        "slug": "optional_projectile",
        "title": "Optional E: 更自由的单物体抛射",
        "status": "optional",
        "goal": "补充更随机的速度方向和姿态变化。",
        "why": "这组有价值，但应晚于更规则的 gravity/entry 进入训练，避免 adapter 一开始吃太杂的 future pattern。",
        "case_ids": [900],
    },
    {
        "slug": "defer_hard_or_redundant",
        "title": "Defer F: 暂缓加入的 case",
        "status": "defer",
        "goal": "先不放进第一轮 adapter 数据，避免重复信号或数据尚未成型。",
        "why": "v2 高速撞击、3/4 物体无互撞、counterfactual 都更适合后续阶段再引入。",
        "case_ids": [100, 101, 102, 220, 221, 230, 231],
    },
]


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Adapter Dataset Proposal</title>
  <style>
    :root {{
      --bg:#ece4d8;
      --bg2:#ddd1be;
      --panel:rgba(255,252,247,.94);
      --ink:#1f1811;
      --muted:#685d52;
      --line:rgba(31,24,17,.10);
      --accent:#9e4227;
      --accent2:#215f63;
      --core:#e8f1e2;
      --expand:#eef2db;
      --optional:#fbe8cf;
      --defer:#f3ddd9;
      --shadow:0 18px 42px rgba(58,39,24,.12);
    }}
    * {{ box-sizing:border-box; }}
    body {{
      margin:0;
      color:var(--ink);
      font-family:"Iowan Old Style","Palatino Linotype","Noto Serif SC",serif;
      background:
        radial-gradient(circle at top left, rgba(158,66,39,.15), transparent 24rem),
        radial-gradient(circle at top right, rgba(33,95,99,.14), transparent 28rem),
        linear-gradient(180deg, var(--bg) 0%, var(--bg2) 100%);
    }}
    main {{
      width:min(1880px, calc(100vw - 20px));
      margin:0 auto;
      padding:12px 0 40px;
    }}
    .hero, .scenario {{
      background:var(--panel);
      border:1px solid var(--line);
      border-radius:20px;
      box-shadow:var(--shadow);
    }}
    .hero {{
      padding:16px 18px 14px;
      position:sticky;
      top:8px;
      z-index:5;
      backdrop-filter:blur(8px);
    }}
    h1 {{
      margin:0;
      font-size:clamp(1.6rem, 2.4vw, 2.7rem);
      line-height:1.03;
      letter-spacing:-0.02em;
    }}
    .sub {{
      margin-top:8px;
      color:var(--muted);
      line-height:1.48;
      font-size:.96rem;
    }}
    .stats, .toolbar, .scenario-tags, .case-tags, .links {{
      display:flex;
      flex-wrap:wrap;
      gap:8px;
    }}
    .stats {{
      margin-top:12px;
    }}
    .pill, .tag {{
      border-radius:999px;
      border:1px solid var(--line);
      padding:5px 10px;
      font-size:.82rem;
      background:#fff8ef;
      white-space:nowrap;
    }}
    .toolbar {{
      margin-top:12px;
      align-items:center;
    }}
    .toolbar input, .toolbar select {{
      min-width:180px;
      border:1px solid var(--line);
      border-radius:10px;
      padding:8px 10px;
      background:rgba(255,255,255,.86);
      color:var(--ink);
      font:inherit;
    }}
    .toolbar button {{
      border:1px solid rgba(158,66,39,.18);
      border-radius:10px;
      background:rgba(158,66,39,.08);
      color:var(--accent);
      padding:8px 10px;
      font:inherit;
      cursor:pointer;
    }}
    .scenarios {{
      display:grid;
      gap:12px;
      margin-top:14px;
    }}
    .scenario {{
      padding:14px;
    }}
    .scenario.core {{ border-left:10px solid #7ba05b; }}
    .scenario.expand {{ border-left:10px solid #b59b3c; }}
    .scenario.optional {{ border-left:10px solid #c9842e; }}
    .scenario.defer {{ border-left:10px solid #a95a4a; }}
    .scenario-head {{
      display:flex;
      justify-content:space-between;
      gap:12px;
      align-items:flex-start;
    }}
    .scenario-head h2 {{
      margin:0;
      font-size:1.16rem;
      line-height:1.08;
    }}
    .meta {{
      margin-top:4px;
      color:var(--muted);
      font-size:.84rem;
      line-height:1.42;
      overflow-wrap:anywhere;
    }}
    .case-grid {{
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(350px, 1fr));
      gap:10px;
      margin-top:12px;
    }}
    .case-card {{
      border:1px solid var(--line);
      border-radius:16px;
      background:rgba(255,255,255,.85);
      padding:10px;
    }}
    .case-title {{
      display:flex;
      justify-content:space-between;
      gap:8px;
      align-items:flex-start;
    }}
    .case-title h3 {{
      margin:0;
      font-size:1rem;
      line-height:1.15;
    }}
    .badge {{
      flex:0 0 auto;
      border-radius:999px;
      border:1px solid var(--line);
      padding:4px 8px;
      font-size:.74rem;
      background:#fff;
    }}
    .desc {{
      margin-top:6px;
      color:var(--muted);
      font-size:.86rem;
      line-height:1.45;
    }}
    .preview-grid {{
      display:grid;
      grid-template-columns:repeat(auto-fit, minmax(160px, 1fr));
      gap:8px;
      margin-top:10px;
    }}
    .preview {{
      border:1px solid var(--line);
      border-radius:12px;
      background:rgba(255,255,255,.78);
      padding:8px;
    }}
    .preview video, .preview img {{
      width:100%;
      aspect-ratio:4/3;
      display:block;
      border-radius:10px;
      background:#111;
      object-fit:contain;
    }}
    .caption {{
      margin-top:6px;
      font-size:.76rem;
      color:var(--muted);
      line-height:1.35;
      overflow-wrap:anywhere;
    }}
    .window-box {{
      margin-top:10px;
      border:1px dashed var(--line);
      border-radius:12px;
      padding:8px;
      background:rgba(255,253,248,.9);
    }}
    .window-box h4 {{
      margin:0 0 6px;
      font-size:.86rem;
    }}
    .strip-grid {{
      display:grid;
      grid-template-columns:1fr;
      gap:6px;
    }}
    .links {{
      margin-top:6px;
      justify-content:space-between;
      font-size:.74rem;
    }}
    .links a {{
      color:var(--accent2);
      text-decoration:none;
    }}
    .muted {{
      color:var(--muted);
    }}
    .hidden {{ display:none !important; }}
    @media (max-width: 980px) {{
      main {{ width:min(100vw - 12px, 1880px); }}
      .hero {{ position:static; }}
      .scenario-head {{ display:block; }}
      .scenario-tags {{ margin-top:8px; }}
      .case-grid {{ grid-template-columns:1fr; }}
    }}
  </style>
</head>
<body>
  <main>
    <section class="hero">
      <h1>Adapter Dataset Proposal</h1>
      <div class="sub">上半部分是我建议的 adapter 训练数据构成；下半每个 case 都给了原始样本预览，若当前已有 oracle window，则附带一组 context/future 训练窗口条带图。可视化已按 metadata 过滤 representative sample，避免把双物体碰撞样本误展示成单物体简单运动。</div>
      <div class="stats">
        <span class="pill">raw sample count: {raw_sample_count}</span>
        <span class="pill">current window count: {window_count}</span>
        <span class="pill">proposal scenarios: {scenario_count}</span>
        <span class="pill">generated: {generated_at}</span>
      </div>
      <div class="toolbar">
        <input id="searchBox" type="search" placeholder="搜索 case / scene / motion / recommendation">
        <select id="statusFilter">
          <option value="">全部阶段</option>
          <option value="core">core</option>
          <option value="expand">expand</option>
          <option value="optional">optional</option>
          <option value="defer">defer</option>
        </select>
        <button id="reloadBtn" type="button">刷新页面</button>
      </div>
    </section>
    <section class="scenarios" id="scenarios"></section>
  </main>
  <script id="records" type="application/json">{records_json}</script>
  <script>
    const records = JSON.parse(document.getElementById('records').textContent || '[]');
    const root = document.getElementById('scenarios');
    const searchBox = document.getElementById('searchBox');
    const statusFilter = document.getElementById('statusFilter');
    const reloadBtn = document.getElementById('reloadBtn');
    const esc = (text) => String(text ?? '').replaceAll('&', '&amp;').replaceAll('<', '&lt;').replaceAll('>', '&gt;');
    root.innerHTML = records.map((scenario) => {{
      const scenarioTags = [
        `status ${scenario.status}`,
        `cases ${scenario.cases.length}`,
      ].map((v) => `<span class="tag">${{esc(v)}}</span>`).join('');
      const caseHtml = scenario.cases.map((item) => {{
        const previews = item.raw_previews.map((preview) => `
          <div class="preview">
            <video controls muted preload="metadata" src="${{encodeURI(preview.video_rel)}}"></video>
            <div class="caption">${{esc(preview.sample_name)}}<br>${{esc(preview.group_dir)}}</div>
          </div>
        `).join('');
        const windowHtml = item.window_preview ? `
          <div class="window-box">
            <h4>Current adapter window preview</h4>
            <div class="strip-grid">
              <img loading="lazy" src="${{encodeURI(item.window_preview.context_strip_rel)}}" alt="context strip">
              <img loading="lazy" src="${{encodeURI(item.window_preview.future_strip_rel)}}" alt="future strip">
            </div>
            <div class="caption">
              future_len=${{item.window_preview.future_len}} | complexity=${{esc(item.window_preview.motion_complexity)}} | future_bucket=${{esc(item.window_preview.future_bucket)}}
            </div>
            <div class="links">
              <a href="${{encodeURI(item.window_preview.pair_meta_rel)}}" target="_blank" rel="noreferrer">pair_meta</a>
              <a href="${{encodeURI(item.window_preview.window_dir_rel)}}" target="_blank" rel="noreferrer">window dir</a>
              <a href="${{encodeURI(item.window_preview.source_video_rel)}}" target="_blank" rel="noreferrer">source video</a>
            </div>
          </div>
        ` : `<div class="window-box"><h4>No current adapter window</h4><div class="caption">当前 oracle_wan_ctx8_fut5_9_13_alltrain 里还没有这个 case 的 window；如果你确认收进 adapter 集，需要重建该 case 的 window 根。</div></div>`;
        const tags = [
          `raw ${item.raw_sample_count}/${item.raw_total_count}`,
          `windows ${item.window_count}/${item.window_total_count}`,
          `motion ${item.motion_group}`,
          `objects ${item.object_count}`,
        ].map((v) => `<span class="tag">${{esc(v)}}</span>`).join('');
        return `
          <article class="case-card" data-search="${{esc(item.search_text)}}" data-status="${{esc(scenario.status)}}">
            <div class="case-title">
              <div>
                <h3>${{esc(item.case_label)}}</h3>
                <div class="meta">${{esc(item.scene_label)}}</div>
              </div>
              <div class="badge">case ${item.case_id}</div>
            </div>
            <div class="desc">${{esc(item.motion_cn)}}</div>
            <div class="case-tags" style="margin-top:8px;">${{tags}}</div>
            <div class="preview-grid">${{previews || '<div class="muted">暂无原始样本预览。</div>'}}</div>
            ${windowHtml}
          </article>
        `;
      }}).join('');
      return `
        <article class="scenario ${scenario.status}" data-status="${{esc(scenario.status)}}" data-search="${{esc(scenario.search_text)}}">
          <div class="scenario-head">
            <div>
              <h2>${{esc(scenario.title)}}</h2>
              <div class="meta"><strong>目标</strong>: ${{esc(scenario.goal)}}</div>
              <div class="meta"><strong>理由</strong>: ${{esc(scenario.why)}}</div>
            </div>
            <div class="scenario-tags">${{scenarioTags}}</div>
          </div>
          <div class="case-grid">${{caseHtml}}</div>
        </article>
      `;
    }}).join('');
    const applyFilter = () => {{
      const q = searchBox.value.trim().toLowerCase();
      const status = statusFilter.value;
      for (const scenarioNode of root.querySelectorAll('.scenario')) {{
        const okStatus = !status || scenarioNode.dataset.status === status;
        const okScenarioQuery = !q || scenarioNode.dataset.search.toLowerCase().includes(q);
        let anyCaseVisible = false;
        for (const caseNode of scenarioNode.querySelectorAll('.case-card')) {{
          const okCaseQuery = !q || caseNode.dataset.search.toLowerCase().includes(q) || scenarioNode.dataset.search.toLowerCase().includes(q);
          const visible = okStatus && okCaseQuery;
          caseNode.classList.toggle('hidden', !visible);
          if (visible) anyCaseVisible = true;
        }}
        scenarioNode.classList.toggle('hidden', !(okStatus && (okScenarioQuery || anyCaseVisible)));
      }}
    }};
    searchBox.addEventListener('input', applyFilter);
    statusFilter.addEventListener('change', applyFilter);
    reloadBtn.addEventListener('click', () => window.location.reload());
  </script>
</body>
</html>
"""


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_symlink(dst: Path, src: Path) -> None:
    if dst.is_symlink() or dst.exists():
        if dst.is_dir() and not dst.is_symlink():
            shutil.rmtree(dst)
        else:
            dst.unlink()
    dst.symlink_to(src, target_is_directory=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw_train_root", type=Path, default=RAW_TRAIN_ROOT)
    parser.add_argument("--window_root", type=Path, default=WINDOW_ROOT)
    parser.add_argument("--case_json", type=Path, default=GENESIS_CASE_JSON)
    parser.add_argument("--output_dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--port", type=int, default=8106)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--raw_previews_per_case", type=int, default=2)
    return parser.parse_args()


def load_case_defs(case_json: Path) -> dict[int, dict[str, Any]]:
    payload = json.loads(case_json.read_text(encoding="utf-8"))
    return {int(key): dict(value) for key, value in dict(payload["explicit_cases"]).items()}


def case_name_for_id(case_id: int, case_defs: dict[int, dict[str, Any]]) -> str:
    scene_label = str(case_defs[case_id]["scene_label"])
    return f"case{int(case_id):03d}_{scene_label}"


def collect_raw_samples(raw_root: Path) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for meta_path in sorted(raw_root.rglob("metadata.json")):
        sample_dir = meta_path.parent
        sample_name = sample_dir.name
        if "__cf_" in sample_name:
            continue
        if "__" not in sample_name:
            continue
        case_name = sample_name.split("__", 1)[1]
        if (sample_dir / "videos" / "rgb.mp4").exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            by_case[case_name].append(
                {
                    "sample_dir": sample_dir,
                    "meta": meta,
                }
            )
    return by_case


def collect_window_records(
    window_root: Path,
    raw_sample_lookup: dict[str, dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    by_case: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for meta_path in sorted(window_root.rglob("pair_meta.json")):
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        source_sample_dir = Path(str(meta["source_sample_dir"])).resolve()
        sample_name = source_sample_dir.name
        if "__" not in sample_name:
            continue
        case_name = sample_name.split("__", 1)[1]
        by_case[case_name].append(
            {
                "pair_meta_path": meta_path,
                "meta": meta,
                "source_meta": raw_sample_lookup.get(str(source_sample_dir), {}),
            }
        )
    return by_case


def scenario_sample_matches(
    scenario: dict[str, Any],
    case_def: dict[str, Any],
    meta: dict[str, Any],
) -> bool:
    case_id = int(case_def.get("case_id", -1))
    num_objects = meta.get("num_objects")
    scene_composition = str(meta.get("scene_composition", ""))
    obj_obj_event_count = meta.get("obj_obj_event_count")
    obj_env_event_count = meta.get("obj_env_event_count")
    expected_object_count = case_def.get("object_count")

    if case_id in {0, 1, 2}:
        return (
            num_objects == 1
            and scene_composition == "single_object_preview"
            and obj_obj_event_count == 0
            and obj_env_event_count == 0
        )

    if case_id in {3, 5, 6, 7, 900, 901, 100, 101, 102}:
        return (
            num_objects == 1
            and scene_composition == "single_object_preview"
            and obj_obj_event_count == 0
        )

    if isinstance(expected_object_count, int):
        return (
            num_objects == expected_object_count
            and scene_composition == "multi_object_free_motion"
            and (obj_obj_event_count in (0, None))
        )

    return True


def filter_raw_samples_for_scenario(
    scenario: dict[str, Any],
    case_def: dict[str, Any],
    samples: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matched = [sample for sample in samples if scenario_sample_matches(scenario, case_def, dict(sample["meta"]))]
    return matched if matched else samples


def filter_window_records_for_scenario(
    scenario: dict[str, Any],
    case_def: dict[str, Any],
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    matched = []
    for record in records:
        source_meta = dict(record.get("source_meta") or {})
        if source_meta and scenario_sample_matches(scenario, case_def, source_meta):
            matched.append(record)
    return matched if matched else records


def motion_complexity_summary(records: list[dict[str, Any]]) -> str:
    if not records:
        return "none"
    counter = Counter(
        str((record["meta"].get("motion_complexity") or {}).get("label", "unknown"))
        for record in records
    )
    return ", ".join(f"{label}:{count}" for label, count in counter.most_common())


def build_strip(frame_paths: list[str], dst: Path, thumb_size: tuple[int, int] = (140, 105)) -> None:
    images: list[Image.Image] = []
    for path in frame_paths:
        with Image.open(path) as image:
            thumb = ImageOps.contain(image.convert("RGB"), thumb_size)
            canvas = Image.new("RGB", thumb_size, (248, 245, 238))
            offset = ((thumb_size[0] - thumb.width) // 2, (thumb_size[1] - thumb.height) // 2)
            canvas.paste(thumb, offset)
            images.append(canvas)
    if not images:
        return
    width = sum(image.width for image in images)
    height = max(image.height for image in images)
    strip = Image.new("RGB", (width, height), (250, 246, 240))
    cursor = 0
    for image in images:
        strip.paste(image, (cursor, 0))
        cursor += image.width
    ensure_dir(dst.parent)
    strip.save(dst, quality=90)
    for image in images:
        image.close()
    strip.close()


def make_window_preview_assets(
    case_name: str,
    records: list[dict[str, Any]],
    output_dir: Path,
    window_root: Path,
    raw_root: Path,
) -> dict[str, Any] | None:
    if not records:
        return None
    record = records[0]
    meta = dict(record["meta"])
    pair_meta_path = Path(record["pair_meta_path"])
    rel_window_dir = pair_meta_path.parent.relative_to(window_root)
    asset_dir = output_dir / "assets" / "windows" / case_name
    context_strip = asset_dir / "context_strip.jpg"
    future_strip = asset_dir / "future_strip.jpg"
    build_strip(list(meta.get("x_frame_paths", [])), context_strip)
    build_strip(list(meta.get("y_frame_paths", [])), future_strip)
    source_sample_dir = Path(str(meta.get("source_sample_dir", "")))
    source_video_rel = (
        "raw/" + source_sample_dir.relative_to(raw_root).as_posix() + "/videos/rgb.mp4"
        if source_sample_dir.exists() and raw_root in source_sample_dir.parents
        else ""
    )
    return {
        "context_strip_rel": str(context_strip.relative_to(output_dir).as_posix()),
        "future_strip_rel": str(future_strip.relative_to(output_dir).as_posix()),
        "pair_meta_rel": "windows/" + str(rel_window_dir.as_posix()) + "/pair_meta.json",
        "window_dir_rel": "windows/" + str(rel_window_dir.as_posix()),
        "source_video_rel": source_video_rel,
        "future_len": int(meta.get("future_len", 0)),
        "motion_complexity": str((meta.get("motion_complexity") or {}).get("label", "unknown")),
        "future_bucket": str((meta.get("window_interactions") or {}).get("future_bucket", "")),
    }


def build_records(
    case_defs: dict[int, dict[str, Any]],
    raw_samples: dict[str, list[dict[str, Any]]],
    window_records: dict[str, list[dict[str, Any]]],
    raw_root: Path,
    window_root: Path,
    output_dir: Path,
    raw_previews_per_case: int,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for scenario in SCENARIO_SPECS:
        case_items: list[dict[str, Any]] = []
        scenario_search_tokens = [scenario["title"], scenario["goal"], scenario["why"], scenario["status"]]
        for case_id in scenario["case_ids"]:
            if case_id not in case_defs:
                continue
            case_def = dict(case_defs[case_id])
            case_def["case_id"] = int(case_id)
            case_name = case_name_for_id(case_id, case_defs)
            raw_case_samples_all = raw_samples.get(case_name, [])
            raw_case_samples = filter_raw_samples_for_scenario(scenario, case_def, raw_case_samples_all)
            windows_all = window_records.get(case_name, [])
            windows = filter_window_records_for_scenario(scenario, case_def, windows_all)
            previews = []
            for sample in raw_case_samples[: int(raw_previews_per_case)]:
                sample_dir = Path(sample["sample_dir"])
                rel = sample_dir.relative_to(raw_root)
                previews.append(
                    {
                        "video_rel": "raw/" + rel.as_posix() + "/videos/rgb.mp4",
                        "sample_name": sample_dir.name,
                        "group_dir": "/".join(rel.parts[:-1]),
                    }
                )
            window_preview = make_window_preview_assets(
                case_name=case_name,
                records=windows,
                output_dir=output_dir,
                window_root=window_root,
                raw_root=raw_root,
            )
            item = {
                "case_id": int(case_id),
                "case_label": case_name,
                "scene_label": str(case_def.get("scene_label", "")),
                "motion_cn": str(case_def.get("motion_cn", "")),
                "motion_group": str(case_def.get("motion_group", "")),
                "object_count": str(case_def.get("object_count", "")),
                "raw_sample_count": len(raw_case_samples),
                "raw_total_count": len(raw_case_samples_all),
                "window_count": len(windows),
                "window_total_count": len(windows_all),
                "window_complexity_summary": motion_complexity_summary(windows),
                "raw_previews": previews,
                "window_preview": window_preview,
            }
            item["search_text"] = " ".join(
                [
                    case_name,
                    str(case_id),
                    item["scene_label"],
                    item["motion_cn"],
                    item["motion_group"],
                    scenario["title"],
                    scenario["status"],
                    item["window_complexity_summary"],
                ]
            ).lower()
            case_items.append(item)
            scenario_search_tokens.extend(
                [
                    case_name,
                    item["scene_label"],
                    item["motion_cn"],
                    item["motion_group"],
                ]
            )
        records.append(
            {
                "slug": scenario["slug"],
                "title": scenario["title"],
                "status": scenario["status"],
                "goal": scenario["goal"],
                "why": scenario["why"],
                "search_text": " ".join(scenario_search_tokens).lower(),
                "cases": case_items,
            }
        )
    return records


def start_server(output_dir: Path, host: str, port: int) -> tuple[int, str]:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind((host, port))
    command = [
        sys.executable,
        "-m",
        "http.server",
        str(port),
        "--bind",
        host,
        "--directory",
        str(output_dir),
    ]
    process = subprocess.Popen(
        command,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    return process.pid, f"http://{host}:{port}/index.html"


def main() -> None:
    args = parse_args()
    raw_root = args.raw_train_root.resolve()
    window_root = args.window_root.resolve()
    output_dir = args.output_dir.resolve()
    ensure_dir(output_dir)
    ensure_dir(output_dir / "assets")
    safe_symlink(output_dir / "raw", raw_root)
    if window_root.exists():
        safe_symlink(output_dir / "windows", window_root)

    case_defs = load_case_defs(args.case_json.resolve())
    raw_samples = collect_raw_samples(raw_root)
    raw_sample_lookup = {
        str(Path(sample["sample_dir"]).resolve()): dict(sample["meta"])
        for samples in raw_samples.values()
        for sample in samples
    }
    window_records = collect_window_records(window_root, raw_sample_lookup) if window_root.exists() else {}
    records = build_records(
        case_defs=case_defs,
        raw_samples=raw_samples,
        window_records=window_records,
        raw_root=raw_root,
        window_root=window_root,
        output_dir=output_dir,
        raw_previews_per_case=int(args.raw_previews_per_case),
    )
    html_text = HTML_TEMPLATE
    replacements = {
        "{raw_sample_count}": str(sum(len(values) for values in raw_samples.values())),
        "{window_count}": str(sum(len(values) for values in window_records.values())),
        "{scenario_count}": str(len(records)),
        "{generated_at}": os.popen("date -u '+%Y-%m-%d %H:%M UTC'").read().strip(),
        "{records_json}": json.dumps(records, ensure_ascii=False),
    }
    for needle, value in replacements.items():
        html_text = html_text.replace(needle, value)
    html_text = html_text.replace("{{", "{").replace("}}", "}")
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")
    manifest = {
        "raw_root": str(raw_root),
        "window_root": str(window_root),
        "records": records,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    pid, url = start_server(output_dir, host=str(args.host), port=int(args.port))
    print(f"output_dir={output_dir}")
    print(f"pid={pid}")
    print(f"url={url}")


if __name__ == "__main__":
    main()
