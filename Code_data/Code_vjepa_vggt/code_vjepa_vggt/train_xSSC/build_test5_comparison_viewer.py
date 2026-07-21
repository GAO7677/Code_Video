#!/usr/bin/env python3
"""Build a grouped comparison viewer for xSSC test_5 inference outputs."""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


DEFAULT_ROOT = Path("/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/test_5")


METHOD_LABELS = {
    "formal_mix49_b2_dropout_metrics_20260719T204359Z": "full ctx slots",
    "xssc_randomcrop_pooled_gpu45_mix49_formal_randomcrop_pooled_gpu45_20260720T110031Z": "randomcrop pooled slots",
    "xssc_centercrop_pooled_gpu0123_mix49_formal_centercrop_pooled_gpu0123_20260721T073149Z": "centercrop pooled slots",
    "full_ctx_weight_self_mean_repeat_tokens": "full ctx weight / self mean-repeat tokens",
    "full_ctx_weight_self_mean_repeat_avg_time_tokens": "full ctx weight / self mean-repeat tokens + mean time",
    "full_ctx_weight_self_mean_one_frame_tokens": "full ctx weight / self mean one-frame tokens",
    "full_ctx_weight_self_mean_one_frame_avg_time_tokens": "full ctx weight / self mean one-frame tokens + mean time",
    "full_ctx_weight_random_pooled_tokens_no_time": "full ctx weight / pooled tokens no time",
    "full_ctx_weight_random_pooled_tokens_with_time": "full ctx weight / pooled tokens + time",
    "full_ctx_weight_zero_slot_tokens": "full ctx weight / zero slot tokens",
    "full_ctx_weight_zero_time_embedding_tokens": "full ctx weight / zero time embedding",
}


METHOD_DESCRIPTIONS = {
    "formal_mix49_b2_dropout_metrics_20260719T204359Z": (
        "Frozen xSSC extracts all 8 ctx frames x 7 slots. "
        "Full ctx projector + per-frame time embedding -> 56 object tokens."
    ),
    "xssc_randomcrop_pooled_gpu45_mix49_formal_randomcrop_pooled_gpu45_20260720T110031Z": (
        "Training uses random crop. xSSC slots are averaged over ctx time, "
        "then projected as 7 pooled object tokens."
    ),
    "xssc_centercrop_pooled_gpu0123_mix49_formal_centercrop_pooled_gpu0123_20260721T073149Z": (
        "Training uses center crop. xSSC slots are averaged over ctx time, "
        "then projected as 7 pooled object tokens."
    ),
    "full_ctx_weight_self_mean_repeat_tokens": (
        "Full ctx slots are averaged over 8 ctx frames, repeated to 8 frames, "
        "then projected with full ctx weights and original per-frame time embedding."
    ),
    "full_ctx_weight_self_mean_repeat_avg_time_tokens": (
        "Full ctx slots are averaged over 8 ctx frames and repeated to 8 frames. "
        "time_embedding[0:8] is also averaged and repeated."
    ),
    "full_ctx_weight_self_mean_one_frame_tokens": (
        "Full ctx slots are averaged over 8 ctx frames and kept as one frame. "
        "Full ctx projector + time_embedding[0] -> 7 object tokens."
    ),
    "full_ctx_weight_self_mean_one_frame_avg_time_tokens": (
        "Full ctx slots are averaged over 8 ctx frames and kept as one frame. "
        "Adds mean(time_embedding[0:8]) -> 7 object tokens."
    ),
    "full_ctx_weight_random_pooled_tokens_no_time": (
        "Uses randomcrop pooled checkpoint projector on time-averaged slots, "
        "repeats projected tokens to 8 frames, no full ctx time embedding."
    ),
    "full_ctx_weight_random_pooled_tokens_with_time": (
        "Uses randomcrop pooled checkpoint projector on time-averaged slots, "
        "repeats projected tokens to 8 frames, adds full ctx per-frame time embedding."
    ),
    "full_ctx_weight_zero_slot_tokens": (
        "Slot/projector token is set to zero. Keeps original per-frame "
        "time_embedding[0:8] -> 56 object tokens."
    ),
    "full_ctx_weight_zero_time_embedding_tokens": (
        "Keeps full ctx slot/projector tokens. Sets the time embedding contribution "
        "to zero -> 56 object tokens."
    ),
}


def read_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def rel(path: Path, root: Path) -> str:
    return html.escape(path.relative_to(root).as_posix())


def short_path(value: str | None) -> str:
    if not value:
        return ""
    path = Path(value)
    return path.name or str(path)


def source_key(meta: dict[str, Any], video_path: Path) -> str:
    source = meta.get("source_video") or meta.get("input_video") or meta.get("input_json")
    if source:
        return str(source)
    return video_path.stem


def method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method)


def method_description(method: str) -> str:
    return METHOD_DESCRIPTIONS.get(method, "")


def is_ablation_method(method: str) -> bool:
    return method.startswith("full_ctx_weight_")


def step_sort_key(step: str) -> tuple[int, str]:
    digits = "".join(ch for ch in step if ch.isdigit())
    return (int(digits) if digits else -1, step)


def collect(root: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    records: list[dict[str, Any]] = []
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for video_path in sorted(root.glob("*/*/step-*/*.mp4")):
        try:
            method = video_path.relative_to(root).parts[0]
            outer_step = video_path.relative_to(root).parts[1]
            inner_step = video_path.relative_to(root).parts[2]
        except Exception:
            continue
        json_path = video_path.with_suffix(".json")
        ctx_path = video_path.with_name(f"{video_path.stem}_input_ctx08.jpg")
        meta = read_json(json_path)
        debug = meta.get("object_debug", {}) if isinstance(meta.get("object_debug"), dict) else {}
        temporal = debug.get("xssc_slot_temporal_mode", {}) if isinstance(debug.get("xssc_slot_temporal_mode"), dict) else {}
        preprocess = debug.get("xssc_preprocess", {}) if isinstance(debug.get("xssc_preprocess"), dict) else {}
        record = {
            "method": method,
            "method_label": method_label(method),
            "method_description": method_description(method),
            "step": inner_step or outer_step,
            "case_stem": video_path.stem,
            "video": str(video_path),
            "json": str(json_path) if json_path.exists() else None,
            "input_ctx": str(ctx_path) if ctx_path.exists() else None,
            "input_json": meta.get("input_json"),
            "source_video": meta.get("source_video"),
            "object_context_shape": debug.get("object_context_shape"),
            "xssc_slots_shape": debug.get("xssc_slots_shape"),
            "temporal_mode": temporal.get("mode"),
            "preprocess_mode": preprocess.get("mode"),
            "ckpt": meta.get("ckpt"),
        }
        key = source_key(meta, video_path)
        records.append(record)
        grouped[key].append(record)
    for rows in grouped.values():
        rows.sort(key=lambda item: (item["case_stem"], item["method_label"], step_sort_key(item["step"])))
    return records, dict(sorted(grouped.items(), key=lambda item: short_path(item[0])))


def render_debug(record: dict[str, Any]) -> str:
    bits = []
    if record.get("object_context_shape"):
        bits.append(f"object={record['object_context_shape']}")
    if record.get("xssc_slots_shape"):
        bits.append(f"slots={record['xssc_slots_shape']}")
    if record.get("temporal_mode"):
        bits.append(f"temporal={record['temporal_mode']}")
    if record.get("preprocess_mode"):
        bits.append(f"crop={record['preprocess_mode']}")
    return html.escape(" · ".join(bits))


def render_video_card(record: dict[str, Any], root: Path) -> str:
    return f"""
    <div class="result-card">
      <video src="{rel(Path(record["video"]), root)}" controls loop muted playsinline></video>
      <div class="meta">{render_debug(record)}</div>
      <div class="meta small"><code>{html.escape(record["case_stem"])}</code></div>
    </div>
    """


def render(root: Path, records: list[dict[str, Any]], grouped: dict[str, list[dict[str, Any]]]) -> str:
    method_counts: dict[str, int] = defaultdict(int)
    step_counts: dict[str, int] = defaultdict(int)
    for record in records:
        method_counts[record["method_label"]] += 1
        step_counts[f"{record['method_label']} {record['step']}"] += 1
    preferred_method_keys = [method for method in METHOD_LABELS if any(record["method"] == method for record in records)]
    all_method_keys = sorted({record["method"] for record in records})
    global_method_keys = [method for method in preferred_method_keys if method in all_method_keys]
    global_method_keys.extend(method for method in all_method_keys if method not in global_method_keys)
    main_method_keys = [method for method in global_method_keys if not is_ablation_method(method)]
    ablation_method_keys = [method for method in global_method_keys if is_ablation_method(method)]
    main_steps = sorted(
        {record["step"] for record in records if not is_ablation_method(record["method"])},
        key=step_sort_key,
    )

    sections = []
    for source, rows in grouped.items():
        first = rows[0]
        representative_ctx = next((row.get("input_ctx") for row in rows if row.get("input_ctx")), None)
        matrix: dict[tuple[str, str], dict[str, Any]] = {
            (row["method"], row["step"]): row for row in rows
        }
        ctx_html = ""
        if representative_ctx:
            ctx_html = (
                f"""
                <div class="ctx-strip">
                  <img src="{rel(Path(representative_ctx), root)}" alt="input context">
                  <div><code>{html.escape(str(first.get("input_json") or ""))}</code></div>
                </div>
                """
            )
        header_cells = "".join(f"<th>{html.escape(step)}</th>" for step in main_steps)
        method_rows = []
        for method_key in main_method_keys:
            method_label_text = method_label(method_key)
            description = method_description(method_key)
            cells = []
            for step in main_steps:
                record = matrix.get((method_key, step))
                if record is None:
                    cells.append('<td><div class="missing">missing</div></td>')
                    continue
                cells.append(
                    f"""
                    <td>
                      {render_video_card(record, root)}
                    </td>
                    """
                )
            method_rows.append(
                f"""
                <tr>
                  <th class="method">
                    <div class="method-name">{html.escape(method_label_text)}</div>
                    <div class="method-desc">{html.escape(description)}</div>
                  </th>
                  {''.join(cells)}
                </tr>
                """
            )
        ablation_headers = []
        ablation_cells = []
        for method_key in ablation_method_keys:
            method_records = sorted(
                [row for row in rows if row["method"] == method_key],
                key=lambda item: (step_sort_key(item["step"]), item["case_stem"]),
            )
            if not method_records:
                continue
            method_label_text = method_label(method_key)
            description = method_description(method_key)
            result_cards = "".join(render_video_card(record, root) for record in method_records)
            ablation_headers.append(
                f"""
                <th class="ablation-method">
                  <div class="method-name">{html.escape(method_label_text)}</div>
                  <div class="method-desc">{html.escape(description)}</div>
                </th>
                """
            )
            ablation_cells.append(
                f"""
                <td class="result-cell">
                  <div class="result-strip">{result_cards}</div>
                </td>
                """
            )
        main_table_html = ""
        if method_rows:
            main_table_html = f"""
              <div class="matrix-wrap">
                <table class="matrix">
                  <thead>
                    <tr>
                      <th class="method">method / processing</th>
                      {header_cells}
                    </tr>
                  </thead>
                  <tbody>
                    {''.join(method_rows)}
                  </tbody>
                </table>
              </div>
            """
        ablation_table_html = ""
        if ablation_cells:
            ablation_table_html = f"""
              <div class="subhead">ablation experiments</div>
              <div class="matrix-wrap">
                <table class="matrix ablation">
                  <thead>
                    <tr>
                      {''.join(ablation_headers)}
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      {''.join(ablation_cells)}
                    </tr>
                  </tbody>
                </table>
              </div>
            """
        sections.append(
            f"""
            <section data-case="{html.escape(short_path(source).lower())}">
              <h2>{html.escape(short_path(source))}</h2>
              <div class="source"><code>{html.escape(source)}</code></div>
              {ctx_html}
              {main_table_html}
              {ablation_table_html}
            </section>
            """
        )

    method_summary = " · ".join(
        f"{html.escape(name)}: {count}" for name, count in sorted(method_counts.items())
    )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>xSSC test_5 comparison</title>
  <style>
    :root {{
      color-scheme: dark;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: #101216;
      color: #edf1f7;
    }}
    body {{ margin: 0; background: #101216; }}
    header {{
      position: sticky;
      top: 0;
      z-index: 20;
      padding: 14px 22px;
      background: rgba(16, 18, 22, 0.96);
      border-bottom: 1px solid #2a3038;
      backdrop-filter: blur(8px);
    }}
    h1 {{ margin: 0; font-size: 18px; font-weight: 680; letter-spacing: 0; }}
    .summary {{ margin-top: 6px; color: #bcc5d2; font-size: 12px; line-height: 1.45; }}
    main {{ padding: 18px 22px 36px; display: grid; gap: 28px; }}
    section {{ display: grid; gap: 10px; min-width: 0; }}
    h2 {{ margin: 0; font-size: 15px; font-weight: 660; overflow-wrap: anywhere; }}
    .source {{ color: #aeb8c8; font-size: 12px; overflow-wrap: anywhere; }}
    .ctx-strip {{
      display: grid;
      grid-template-columns: minmax(220px, 420px) 1fr;
      gap: 12px;
      align-items: center;
      border: 1px solid #2a3038;
      border-radius: 8px;
      background: #171b22;
      overflow: hidden;
    }}
    .ctx-strip img {{
      width: 100%;
      border-right: 1px solid #2a3038;
    }}
    .ctx-strip div {{
      min-width: 0;
      padding: 10px 12px;
      color: #c4ccd8;
      font-size: 12px;
      line-height: 1.45;
      overflow-wrap: anywhere;
    }}
    .matrix-wrap {{
      overflow-x: auto;
      border: 1px solid #2a3038;
      border-radius: 8px;
      background: #171b22;
    }}
    table.matrix {{
      width: 100%;
      min-width: {max(1040, 280 + 300 * max(1, len(main_steps)))}px;
      border-collapse: collapse;
      table-layout: fixed;
    }}
    table.matrix.ablation {{
      min-width: {max(1040, 320 * max(1, len(ablation_method_keys)))}px;
    }}
    th, td {{
      border-right: 1px solid #2a3038;
      border-bottom: 1px solid #2a3038;
      vertical-align: top;
    }}
    th {{
      padding: 9px 10px;
      background: #202631;
      color: #edf1f7;
      font-size: 12px;
      font-weight: 650;
      text-align: left;
    }}
    th.method {{
      width: 260px;
      position: sticky;
      left: 0;
      z-index: 2;
    }}
    thead th {{
      position: sticky;
      top: 0;
      z-index: 1;
    }}
    tbody th.method {{
      background: #1c222c;
    }}
    th.ablation-method {{
      min-width: 280px;
      background: #1c222c;
      vertical-align: top;
    }}
    .method-name {{
      font-size: 12px;
      font-weight: 700;
      line-height: 1.35;
      color: #f0f4fb;
      overflow-wrap: anywhere;
    }}
    .method-desc {{
      margin-top: 7px;
      font-size: 11px;
      font-weight: 450;
      line-height: 1.42;
      color: #aeb8c8;
      overflow-wrap: anywhere;
    }}
    .subhead {{
      margin: 4px 0 -2px;
      color: #d8e0ec;
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .result-card {{
      min-width: 0;
    }}
    .result-strip {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 0;
    }}
    .result-cell {{
      padding: 0;
    }}
    video, img {{
      display: block;
      width: 100%;
      height: auto;
      background: #050608;
    }}
    td video {{
      aspect-ratio: 16 / 9;
      object-fit: contain;
    }}
    .meta {{
      padding: 8px 10px;
      color: #c4ccd8;
      font-size: 11px;
      line-height: 1.45;
      overflow-wrap: anywhere;
      border-top: 1px solid #2a3038;
    }}
    .missing {{
      min-height: 190px;
      display: grid;
      place-items: center;
      color: #7f8999;
      font-size: 12px;
      background: #141820;
    }}
    .small {{ color: #9ea8b8; }}
    code {{ color: #dce7ff; overflow-wrap: anywhere; }}
    @media (max-width: 760px) {{
      main {{ padding: 14px; }}
      .ctx-strip {{ grid-template-columns: 1fr; }}
      .ctx-strip img {{ border-right: 0; border-bottom: 1px solid #2a3038; }}
    }}
  </style>
</head>
<body>
  <header>
    <h1>xSSC test_5 comparison grouped by source video</h1>
    <div class="summary">{len(grouped)} source groups · {len(records)} videos · {method_summary}</div>
  </header>
  <main>
    {''.join(sections)}
  </main>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-name", default="index.html")
    args = parser.parse_args()
    records, grouped = collect(args.root)
    manifest = {"root": str(args.root), "num_videos": len(records), "num_source_groups": len(grouped), "records": records}
    (args.root / "viewer_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    (args.root / args.output_name).write_text(render(args.root, records, grouped), encoding="utf-8")
    print(args.root / args.output_name)
    print(args.root / "viewer_manifest.json")
    print(f"videos={len(records)} source_groups={len(grouped)}")


if __name__ == "__main__":
    main()
