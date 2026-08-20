#!/usr/bin/env python3
"""Run VGGT on the available 0819 rgb_cycles videos and group the gallery.

The 0819 manifest contains controlled-variable sweeps.  Rows are grouped by
``(source_group, controlled_variable)`` so cases with the same physical
variable are shown together while unrelated v2v controls stay separate.
"""

from __future__ import annotations

import argparse
import html
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from code_vjepa_vggt.adapters.vggt_adapter import VGGTTrackAdapter
from code_vjepa_vggt.visualize_vggt_0717_manifest_cases import (
    DEFAULT_INPUT_HW,
    DEFAULT_MODEL,
    run_one_case,
    safe_name,
    video_panel,
    write_json,
)


DEFAULT_DATASET_ROOT = Path("/data/gaoya/AAA_test_video/physv_v2v_0819")
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/vggt_0717_train10_context8_prefix49/"
    "vggt_0819_group_compare"
)
VARIABLE_LABELS = {
    "table_height_m": ("table height", "m"),
    "ramp_angle_deg": ("ramp angle", "deg"),
    "ramp_length_m": ("ramp length", "m"),
    "bowl_radius_m": ("bowl radius", "m"),
    "domino_gap_m": ("domino gap", "m"),
    "gap_width_m": ("gap width", "m"),
    "ball_start_x_m": ("ball start x", "m"),
    "ball_radius_m": ("ball radius", "m"),
}


def _value_sort(row: dict[str, Any]) -> tuple[float, str]:
    try:
        value = float(row.get("controlled_value"))
    except (TypeError, ValueError):
        value = float("inf")
    return value, str(row.get("sample_id", ""))


def _group_label(source_group: str, controlled_variable: str) -> str:
    variable, _ = VARIABLE_LABELS.get(controlled_variable, (controlled_variable, ""))
    return f"{source_group} · {variable} sweep"


def _value_label(value: Any, variable: str) -> str:
    variable_label, unit = VARIABLE_LABELS.get(variable, (variable, ""))
    try:
        value_text = f"{float(value):g}"
    except (TypeError, ValueError):
        value_text = str(value)
    return f"{variable_label}={value_text}{unit}"


def load_rows(dataset_root: Path) -> tuple[list[dict[str, Any]], list[str], dict[str, int]]:
    manifest_path = dataset_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    group_counts: dict[str, int] = defaultdict(int)
    for raw in manifest.get("samples", []):
        sample_id = str(raw.get("sample_id", "")).strip()
        sample_dir = Path(str(raw.get("sample_dir", "")))
        video = sample_dir / "videos" / "rgb_cycles.mp4"
        if not sample_id or not video.is_file():
            if sample_id:
                missing.append(sample_id)
            continue
        source_group = str(raw.get("source_group", "unknown")).strip()
        controlled_variable = str(raw.get("controlled_variable", "unknown")).strip()
        group_key = f"{source_group}::{controlled_variable}"
        row = dict(raw)
        row.update(
            {
                "case_id": sample_id,
                "family_key": source_group,
                "video": str(video),
                "caption": str(raw.get("task_type", "0819 PhysV V2V")),
                "group_key": group_key,
                "group_label": _group_label(source_group, controlled_variable),
                "controlled_value_label": _value_label(raw.get("controlled_value"), controlled_variable),
                "stable_split": "0819_available",
                "stable_key": group_key,
            }
        )
        rows.append(row)
        group_counts[group_key] += 1
    rows.sort(key=lambda row: (str(row["group_key"]), _value_sort(row)))
    return rows, missing, dict(sorted(group_counts.items()))


def prepare_selection(dataset_root: Path, output_root: Path) -> list[dict[str, Any]]:
    rows, missing, group_counts = load_rows(dataset_root)
    if not rows:
        raise RuntimeError(f"no rgb_cycles.mp4 rows found under {dataset_root}")
    write_json(
        output_root / "selection.json",
        {
            "dataset_root": str(dataset_root),
            "manifest": str(dataset_root / "manifest.json"),
            "selection": "all manifest samples whose videos/rgb_cycles.mp4 exists",
            "available_case_count": len(rows),
            "manifest_missing_video_case_ids": missing,
            "group_counts": group_counts,
            "rows": rows,
        },
    )
    print(
        f"prepared {len(rows)} rows in {len(group_counts)} groups; "
        f"missing videos={len(missing)}",
        flush=True,
    )
    return rows


def run_mode(args: argparse.Namespace) -> None:
    dataset_root = Path(args.dataset_root).resolve()
    output_root = Path(args.output_root).resolve()
    selection_path = output_root / "selection.json"
    if args.prepare_only:
        prepare_selection(dataset_root, output_root)
        return
    if not selection_path.is_file():
        rows = prepare_selection(dataset_root, output_root)
    else:
        rows = json.loads(selection_path.read_text(encoding="utf-8"))["rows"]
    device = str(args.device)
    adapter = VGGTTrackAdapter(
        model_path=str(Path(args.model_path).resolve()),
        num_queries=int(args.num_queries),
        device=device,
        input_hw=(int(args.input_h), int(args.input_w)),
        trainable=False,
    )
    if adapter.model is None:
        raise RuntimeError(f"failed to load VGGT model: {args.model_path}")
    mode_root = output_root / args.mode
    write_json(
        mode_root / "run_config.json",
        {
            "mode": args.mode,
            "frame_count": int(args.frame_count),
            "context_frames": 8,
            "model_path": str(Path(args.model_path).resolve()),
            "device": device,
            "vggt_input_hw": [int(args.input_h), int(args.input_w)],
            "num_world_points": int(args.num_world_points),
            "seed": int(args.seed),
        },
    )
    print(
        f"loaded VGGT on {device}; mode={args.mode}; rows={len(rows)}; "
        f"frames={args.frame_count}",
        flush=True,
    )
    for index, row in enumerate(rows):
        run_one_case(
            adapter,
            row,
            index=index,
            mode=args.mode,
            frame_count=int(args.frame_count),
            output_root=output_root,
            device=adapter.device_obj,
            autocast_bf16=bool(args.autocast_bf16),
            num_world_points=int(args.num_world_points),
            seed=int(args.seed) + index,
            overwrite=bool(args.overwrite),
            total_cases=len(rows),
        )
    print(f"finished {args.mode}: {len(rows)} rows", flush=True)


def _result_map(output_root: Path, mode: str) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((output_root / mode / "cases").glob("*/result.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        result[str(payload.get("case_id"))] = payload
    return result


def _case_videos(
    result: dict[str, Any],
    prefix: str,
    mode_label: str,
) -> str:
    videos = result.get("videos", {}) if result.get("status") == "ok" else {}
    return "".join(
        (
            f"<div class=\"v0819-mode\"><h5>{html.escape(mode_label)}</h5>",
            video_panel("tracks", "VGGT query trajectories", f"{prefix}{videos.get('vggt_tracks')}" if videos.get("vggt_tracks") else None),
            video_panel("depth", "VGGT depth", f"{prefix}{videos.get('vggt_depth')}" if videos.get("vggt_depth") else None),
            video_panel("world points", "normalized XYZ", f"{prefix}{videos.get('vggt_world_points')}" if videos.get("vggt_world_points") else None),
            video_panel("8,000 点覆盖", "sampled dense world points", f"{prefix}{videos.get('world_points_8000_overlay')}" if videos.get("world_points_8000_overlay") else None),
            "</div>",
        )
    )


def build_group_section(
    output_root: Path,
    rows: list[dict[str, Any]],
    *,
    link_prefix: str = "",
) -> str:
    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_group[str(row["group_key"])].append(row)
    context_results = _result_map(output_root, "context8")
    prefix_results = _result_map(output_root, "prefix49")
    groups: list[str] = []
    for group_key in sorted(by_group):
        group_rows = sorted(by_group[group_key], key=_value_sort)
        group_label = str(group_rows[0].get("group_label", group_key))
        cases: list[str] = []
        for row in group_rows:
            case_id = str(row["case_id"])
            context = context_results.get(case_id, {"status": "pending"})
            prefix = prefix_results.get(case_id, {"status": "pending"})
            statuses = f"8f={context.get('status', 'pending')} · 49f={prefix.get('status', 'pending')}"
            cases.append(
                f"""
                <article class="v0819-case">
                  <header><code>{html.escape(case_id)}</code>
                  <span>{html.escape(str(row.get('controlled_value_label', '')))}</span></header>
                  <p class="v0819-status">{html.escape(statuses)}</p>
                  <div class="v0819-modes">
                    {_case_videos(context, link_prefix, '8 帧 context')}
                    {_case_videos(prefix, link_prefix, '49 帧 prefix')}
                  </div>
                </article>
                """
            )
        groups.append(
            f"""
            <article class="v0819-group">
              <h3>{html.escape(group_label)}</h3>
              <p class="v0819-group-note">同一 controlled variable 的 {len(group_rows)} 个 case；按控制变量值排序。</p>
              <div class="v0819-cases">{''.join(cases)}</div>
            </article>
            """
        )
    return f"""
<!-- VGGT_0819_GROUPS_START -->
<section id="vggt-0819-group-comparison" class="v0819-root">
  <h2>0819 PhysV V2V · VGGT 组内 case 对比</h2>
  <p class="v0819-intro">共 {len(rows)} 个可用 rgb_cycles.mp4，按 controlled variable 分成 {len(by_group)} 组。每个 case 对比 8 帧 context 与 49 帧 prefix 的 tracks、depth、world points 和每帧 8,000 点覆盖；VGGT 输入统一为 420×728（高×宽）。</p>
  {''.join(groups)}
</section>
<!-- VGGT_0819_GROUPS_END -->
"""


def build_gallery(output_root: Path, previous_root: Path | None) -> None:
    selection = json.loads((output_root / "selection.json").read_text(encoding="utf-8"))
    rows = selection["rows"]
    section = build_group_section(output_root, rows)
    document = f"""<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>0819 PhysV V2V × VGGT grouped gallery</title>
<style>
  :root {{ color-scheme:dark; --bg:#0d1117; --panel:#161b22; --line:#30363d; --muted:#8b949e; --accent:#58a6ff; }}
  * {{ box-sizing:border-box; }} body {{ margin:0; padding:20px; background:var(--bg); color:#e6edf3; font:13px/1.4 system-ui,sans-serif; }}
  main {{ max-width:1900px; margin:auto; }} h1,h2,h3,h4,h5 {{ color:var(--accent); }} h1 {{ font-size:24px; }}
  .v0819-root {{ margin:20px 0; }} .v0819-intro,.v0819-group-note,.v0819-status {{ color:var(--muted); }}
  .v0819-group {{ border:1px solid var(--line); border-radius:12px; padding:14px; margin:20px 0; background:var(--panel); }}
  .v0819-cases {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:12px; }}
  .v0819-case {{ min-width:0; border:1px solid var(--line); border-radius:8px; padding:8px; background:#0d1117; }}
  .v0819-case header {{ display:flex; flex-direction:column; gap:3px; border-bottom:1px solid var(--line); padding-bottom:5px; }}
  .v0819-case header span {{ color:#c9d1d9; }} .v0819-status {{ margin:5px 0; font-size:11px; }}
  .v0819-modes {{ display:grid; grid-template-columns:1fr; gap:8px; }} .v0819-mode {{ min-width:0; }}
  .v0819-mode h5 {{ margin:6px 0 3px; font-size:13px; }} figure {{ margin:5px 0; border:1px solid var(--line); border-radius:5px; overflow:hidden; }}
  figcaption {{ display:flex; justify-content:space-between; gap:4px; padding:4px; font-size:11px; }} figcaption span {{ color:var(--muted); text-align:right; }}
  video {{ display:block; width:100%; background:#000; max-height:230px; }} .missing {{ opacity:.5; }}
  @media(max-width:1300px) {{ .v0819-cases {{ grid-template-columns:repeat(3,minmax(0,1fr)); }} }}
  @media(max-width:800px) {{ .v0819-cases {{ grid-template-columns:1fr; }} }}
</style></head><body><main><h1>0819 PhysV V2V · VGGT grouped gallery</h1>{section}</main></body></html>"""
    (output_root / "index.html").write_text(document, encoding="utf-8")
    if previous_root is not None:
        previous_index = previous_root / "index.html"
        previous = previous_index.read_text(encoding="utf-8")
        start = "<!-- VGGT_0819_GROUPS_START -->"
        end = "<!-- VGGT_0819_GROUPS_END -->"
        previous = previous.replace(previous[previous.find(start) : previous.find(end) + len(end)], "") if start in previous and end in previous else previous
        prefix = f"{output_root.name}/"
        integrated = build_group_section(output_root, rows, link_prefix=prefix)
        if "</main>" not in previous:
            raise RuntimeError(f"cannot integrate 0819 section into {previous_index}")
        previous = previous.replace("</main>", integrated + "</main>", 1)
        previous_index.write_text(previous, encoding="utf-8")
        print(f"integrated 0819 grouped section into {previous_index}", flush=True)
    print(f"gallery written to {output_root / 'index.html'}", flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--previous-root", default=None)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--mode", choices=("context8", "prefix49"), default="context8")
    parser.add_argument("--frame-count", type=int, default=8)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--input-h", type=int, default=DEFAULT_INPUT_HW[0])
    parser.add_argument("--input-w", type=int, default=DEFAULT_INPUT_HW[1])
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--num-world-points", type=int, default=8000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--autocast-bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--build-gallery", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root).resolve()
    if args.prepare_only:
        prepare_selection(Path(args.dataset_root).resolve(), output_root)
        return
    if args.build_gallery:
        previous_root = None if args.previous_root is None else Path(args.previous_root).resolve()
        build_gallery(output_root, previous_root)
        return
    if args.mode == "context8" and int(args.frame_count) != 8:
        raise SystemExit("context8 mode requires --frame-count 8")
    if args.mode == "prefix49" and int(args.frame_count) != 49:
        raise SystemExit("prefix49 mode requires --frame-count 49")
    run_mode(args)


if __name__ == "__main__":
    main()
