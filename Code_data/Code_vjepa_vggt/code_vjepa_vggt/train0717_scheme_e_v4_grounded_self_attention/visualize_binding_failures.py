#!/usr/bin/env python3
"""Replay representative binding failures with the production grounding stack."""
from __future__ import annotations

import argparse
import glob
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerAdapter
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    visualize_caption_grounding_track_v2v as base,
)


SELECTIONS = (
    {
        "slug": "pybullet_f2_two_unmatched",
        "video_suffix": "/F2_two_object/sample_000457/video.mp4",
        "category": "exact span unmatched",
        "note": "Both detected slots use malformed phrases; neither phrase maps to a unique caption span.",
    },
    {
        "slug": "pybullet_f4_duplicate_cylinder",
        "video_suffix": "/F4_occlusion/sample_000641/video.mp4",
        "category": "duplicate occurrence conflict",
        "note": "Two cylinders share one caption phrase; the unique-span matcher binds the first and rejects the second.",
    },
    {
        "slug": "kubric_pool_table_unmatched",
        "video_suffix": "/pool_table_force/2025-09-26/674a06/rgba.mp4",
        "category": "exact span unmatched and semantic false matches",
        "note": "The unmatched phrase is 'surface. A force'; other accepted phrases such as 'magnitude' are not object entities.",
    },
    {
        "slug": "pybullet_f3_semantic_false_match",
        "video_suffix": "/F3_chain_reaction/sample_001311/video.mp4",
        "category": "semantic false match",
        "note": "Exact string matching accepts 'interact according to', even though it is not an object noun phrase.",
    },
    {
        "slug": "kubric_friction_semantic_false_match",
        "video_suffix": "/friction_slide_flat_v2/2025-10-08/f1f4d6/rgba.mp4",
        "category": "semantic false match",
        "note": "The only valid slot is bound to 'only thing' instead of the visible brick.",
    },
    {
        "slug": "kubric_ball_drop_zero_slot",
        "video_suffix": "/ball_drop_v3/2025-11-06/48d978/rgba.mp4",
        "category": "zero valid object",
        "note": "The training forward produced no valid object slot for caption 'ball drop multiple'.",
    },
)


_ACTIVE_TRACE: dict[str, Any] | None = None
_ORIGINAL_RENDER_OVERLAY = base._render_overlay


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def _load_trace_records(trace_root: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    pattern = str(trace_root / "grounded_metrics.rank*.jsonl")
    for path in sorted(glob.glob(pattern)):
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                record = json.loads(line)
                records.setdefault(str(record["video_path"]), record)
    return records


def _select_records(
    records: dict[str, dict[str, Any]],
) -> list[tuple[dict[str, str], dict[str, Any]]]:
    selected = []
    for spec in SELECTIONS:
        matches = [
            record
            for path, record in records.items()
            if path.endswith(spec["video_suffix"])
        ]
        if not matches:
            raise RuntimeError(f"trace case not found: {spec['video_suffix']}")
        selected.append((spec, matches[0]))
    return selected


def _trace_status(slot: dict[str, Any]) -> str:
    if not bool(slot.get("valid")):
        return "INVALID"
    return "MATCHED" if bool(slot.get("noun_matched")) else "UNMATCHED"


def _render_overlay_with_trace(*args, **kwargs):
    frames, slot_reports = _ORIGINAL_RENDER_OVERLAY(*args, **kwargs)
    trace = _ACTIVE_TRACE or {}
    trace_slots = list(trace.get("slots", []))
    lines = [
        f"S{slot['slot_id']} {_trace_status(slot)} | phrase={slot.get('phrase', '')}"
        for slot in trace_slots
        if bool(slot.get("valid"))
    ]
    if not lines:
        lines = ["NO VALID OBJECT SLOT IN THE TRAINING FORWARD"]
    header_height = 31 + 22 * len(lines)
    output = np.full(
        (len(frames), frames.shape[1] + header_height, frames.shape[2], 3),
        (25, 27, 28),
        dtype=np.uint8,
    )
    output[:, header_height:] = frames
    for frame in output:
        cv2.putText(
            frame,
            "Scheme-E v4 trace binding status",
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (238, 238, 238),
            1,
            cv2.LINE_AA,
        )
        for line_id, line in enumerate(lines):
            color = (95, 225, 130) if " MATCHED " in line else (245, 105, 95)
            cv2.putText(
                frame,
                line[:170],
                (12, 46 + line_id * 22),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                color,
                1,
                cv2.LINE_AA,
            )
    return output, slot_reports


def _write_gallery(output_root: Path, results: list[dict[str, Any]]) -> None:
    cards = []
    for result in results:
        rel = html.escape(result["case_dir"])
        artifacts = result["artifacts"]
        trace_rows = []
        for slot in result["trace_slots"]:
            status = _trace_status(slot)
            css = "ok" if status == "MATCHED" else "bad" if status == "UNMATCHED" else "muted"
            trace_rows.append(
                "<tr>"
                f"<td>S{int(slot['slot_id'])}</td>"
                f"<td class='{css}'>{status}</td>"
                f"<td>{html.escape(str(slot.get('phrase', '')))}</td>"
                f"<td>{float(slot.get('evidence_confidence', 0.0)):.3f}</td>"
                "</tr>"
            )
        rerun_rows = "".join(
            "<tr>"
            f"<td>S{int(slot['slot_id'])}</td>"
            f"<td>{html.escape(str(slot['source_phrase']))}</td>"
            f"<td>{html.escape(str(slot['detector_query']))}</td>"
            f"<td>{float(slot['score']):.3f}</td>"
            "</tr>"
            for slot in result["slots"]
        )
        cards.append(
            "<section>"
            f"<h2>{html.escape(result['case'])}</h2>"
            f"<p class='category'>{html.escape(result['failure_category'])}</p>"
            f"<p>{html.escape(result['failure_note'])}</p>"
            f"<p><b>Source:</b> <code>{html.escape(result['source_video'])}</code></p>"
            f"<p><b>Caption:</b> {html.escape(result['caption'])}</p>"
            "<div class='media'>"
            f"<figure><video controls preload='metadata' src='{rel}/{html.escape(artifacts['source_video'])}'></video><figcaption>Original source video</figcaption></figure>"
            f"<figure><video controls preload='metadata' src='{rel}/{html.escape(artifacts['model_input_context'])}'></video><figcaption>Actual prefix-8 model input</figcaption></figure>"
            f"<figure><video controls preload='metadata' src='{rel}/{html.escape(artifacts['track_overlay'])}'></video><figcaption>GDINO + SAM2 + CoTracker with trace status</figcaption></figure>"
            f"<figure><img src='{rel}/{html.escape(artifacts['prompt_frame_detection'])}'><figcaption>Prompt-frame detection</figcaption></figure>"
            "</div>"
            "<h3>Training trace binding</h3>"
            "<table><tr><th>Slot</th><th>Status</th><th>Phrase</th><th>Evidence</th></tr>"
            + "".join(trace_rows)
            + "</table>"
            "<h3>Deterministic grounding replay</h3>"
            "<table><tr><th>Slot</th><th>Source phrase</th><th>GDINO query</th><th>Score</th></tr>"
            + rerun_rows
            + "</table>"
            f"<p><a href='{rel}/result.json'>result.json</a> | <a href='{rel}/{html.escape(artifacts['track_grid'])}'>contact sheet</a></p>"
            "</section>"
        )
    page = """<!doctype html><html><head><meta charset='utf-8'>
<title>Scheme-E v4 binding failures</title><style>
:root{--paper:#f1f1ee;--ink:#202322;--line:#c8cbc6;--bad:#a52d27;--ok:#177343}
body{margin:0;background:var(--paper);color:var(--ink);font:15px Georgia,serif}
main{max-width:1560px;margin:auto;padding:24px}h1,h2,h3{font-family:Arial,sans-serif;letter-spacing:0}
section{border-top:2px solid #343937;padding:20px 0 30px;margin:12px 0 30px}.category{font:bold 14px Arial;color:var(--bad);text-transform:uppercase}
.media{display:grid;grid-template-columns:1fr 1fr;gap:14px}figure{margin:0}video,img{display:block;width:100%;background:#000}figcaption{padding:7px 0}
table{width:100%;border-collapse:collapse;margin:8px 0 18px}th,td{border:1px solid var(--line);padding:7px;text-align:left}.bad{color:var(--bad);font-weight:bold}.ok{color:var(--ok);font-weight:bold}.muted{color:#707570}code{overflow-wrap:anywhere}
@media(max-width:900px){.media{grid-template-columns:1fr}}
</style></head><body><main><h1>Scheme-E v4 binding failure replay</h1>""" + "".join(cards) + "</main></body></html>"
    (output_root / "index.html").write_text(page, encoding="utf-8")
    (output_root / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    global _ACTIVE_TRACE
    args = _parse_args()
    args.trace_root = args.trace_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    records = _load_trace_records(args.trace_root)
    selected = _select_records(records)

    base._render_overlay = _render_overlay_with_trace
    provider_args = argparse.Namespace(
        device=args.device,
        num_context_frames=8,
        max_objects=4,
        points_per_object=8,
        caption_max_phrases=4,
        caption_min_score=4.0,
        gdino_box_threshold=0.20,
        gdino_text_threshold=0.15,
    )
    provider = base._build_provider(provider_args)
    cotracker = CoTrackerAdapter(
        checkpoint_path="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth",
        num_queries=32,
        device=args.device,
        input_hw=(384, 512),
        window_len=60,
    )
    run_args = argparse.Namespace(
        output_root=args.output_root,
        num_context_frames=8,
        height=512,
        width=896,
        max_objects=4,
        points_per_object=8,
        fps=args.fps,
    )

    input_root = args.output_root / "_inputs"
    input_root.mkdir(parents=True, exist_ok=True)
    results = []
    for spec, trace in selected:
        input_path = input_root / f"{spec['slug']}.json"
        input_path.write_text(
            json.dumps(
                {
                    "source_video": trace["video_path"],
                    "input_video": trace["video_path"],
                    "input_caption": trace["caption"],
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        _ACTIVE_TRACE = trace
        result = base._run_case(
            json_path=input_path,
            args=run_args,
            provider=provider,
            cotracker=cotracker,
        )
        result.update(
            {
                "failure_category": spec["category"],
                "failure_note": spec["note"],
                "trace_slots": trace["slots"],
                "trace_rank": trace["rank"],
                "trace_forward_index": trace["forward_index"],
            }
        )
        case_result_path = args.output_root / result["case_dir"] / "result.json"
        case_result_path.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        results.append(result)
        print(f"[complete] {spec['slug']}", flush=True)
    _write_gallery(args.output_root, results)
    print(f"[gallery] {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
