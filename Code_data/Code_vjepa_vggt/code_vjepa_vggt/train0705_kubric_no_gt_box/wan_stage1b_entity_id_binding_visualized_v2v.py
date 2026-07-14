"""Entity-ID v2v inference with actual slot/mask/track binding overlays."""
from __future__ import annotations

import html
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from code_vjepa_vggt.train0705 import (
    infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer0705,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    infer_stage1b_context_only_no_gt_box_v_newtrain_kubric as kubric_infer,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    inspect_kubric_train_forward_aux_overlay as inspectmod,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_entity_id_binding_v2v as entity_infer,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as batch,
)


_ACTIVE_CASE_DIR: Path | None = None
_ACTIVE_STEM: str | None = None
_CASE_REPORTS: list[dict[str, Any]] = []
_ORIGINAL_RUN_SINGLE = batch._run_single_case_in_process
_COLORS = ((224, 67, 54), (28, 150, 118), (43, 112, 196), (225, 146, 34))


def _browser_mp4(path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        candidate = Path(sys.executable).resolve().parent / "ffmpeg"
        ffmpeg = str(candidate) if candidate.is_file() else None
    if ffmpeg is None:
        raise RuntimeError("ffmpeg is required for H.264 overlays")
    output = path.with_name(f"{path.stem}.browser.mp4")
    subprocess.run(
        [
            ffmpeg, "-y", "-i", str(path), "-an", "-c:v", "libx264",
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(output),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return output


def _external_bar(frames: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    bar_h = 30 + 25 * min(len(lines), 3)
    output = np.full(
        (len(frames), int(frames.shape[1]) + bar_h, int(frames.shape[2]), 3),
        (246, 243, 236),
        dtype=np.uint8,
    )
    output[:, bar_h:] = frames
    for frame_id, frame in enumerate(output):
        cv2.putText(
            frame, f"{title} | ctx {frame_id:02d}/{len(frames) - 1:02d}",
            (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (30, 30, 30), 1, cv2.LINE_AA,
        )
        for line_id, line in enumerate(lines[:3]):
            cv2.putText(
                frame, line[:150], (12, 49 + line_id * 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (45, 45, 45), 1, cv2.LINE_AA,
            )
    return output


def _binding_by_slot(binding: dict[str, Any]) -> dict[int, dict[str, Any]]:
    records = {}
    for key, matched in (("matched", True), ("unmatched", False)):
        for raw in binding.get(key, []):
            record = dict(raw)
            record["prompt_matched"] = matched
            records[int(record["slot_id"])] = record
    return records


def _render_binding_overlay(
    *,
    context_video: torch.Tensor,
    grounding: Any,
    query_points: torch.Tensor,
    cotracker: Any,
    binding: dict[str, Any],
    output_dir: Path,
    stem: str,
    fps: int = 12,
) -> dict[str, Any]:
    valid_mask = np.asarray(grounding.object_valid_mask) > 0.5
    object_tracks = list(grounding.object_tracks)
    points = query_points[0].detach().float().cpu().numpy()
    tracks = cotracker.tracks[0].detach().float().cpu().numpy()
    visibility = cotracker.visibility[0].detach().float().cpu().numpy()
    points_per_slot = int(points.shape[0] // max(len(valid_mask), 1))
    records = _binding_by_slot(binding)
    slot_reports = []
    frames = []
    for frame_id in range(int(context_video.shape[1])):
        frame = inspectmod.tensor_frame_to_uint8_hwc(context_video[:, frame_id]).copy()
        for slot_id, valid in enumerate(valid_mask.tolist()):
            if not valid or slot_id >= len(object_tracks):
                continue
            track = object_tracks[slot_id]
            color = _COLORS[slot_id % len(_COLORS)]
            masks = np.asarray(track.masks_thw) > 0
            if frame_id < len(masks) and masks[frame_id].shape == frame.shape[:2]:
                mask = masks[frame_id]
                tint = np.zeros_like(frame)
                tint[mask] = color
                frame = np.where(
                    mask[..., None],
                    np.clip(frame.astype(np.float32) * 0.72 + tint.astype(np.float32) * 0.28, 0, 255).astype(np.uint8),
                    frame,
                )
            record = records.get(slot_id, {})
            entity_id = int(record.get("entity_id", -1))
            candidate = str(record.get("matched_candidate", "unmatched"))
            inspectmod.draw_box_rgb(
                frame,
                np.asarray(track.boxes_t4[frame_id], dtype=np.float32),
                color,
                f"S{slot_id}/E{entity_id}:{candidate}",
            )
            base = slot_id * points_per_slot
            for query_id in range(base, min(base + points_per_slot, len(points))):
                if frame_id == 0:
                    inspectmod.draw_point_rgb(frame, points[query_id], color, "", radius=4)
                if query_id < tracks.shape[1] and visibility[frame_id, query_id] >= 0.5:
                    inspectmod.draw_point_rgb(frame, tracks[frame_id, query_id], color, "", radius=4)
        frames.append(frame)

    for slot_id, valid in enumerate(valid_mask.tolist()):
        if not valid or slot_id >= len(object_tracks):
            continue
        track = object_tracks[slot_id]
        record = records.get(slot_id, {})
        slot_reports.append(
            {
                "slot_id": int(slot_id),
                "entity_id": int(record.get("entity_id", -1)),
                "grounding_phrase": str(getattr(track, "phrase", "")),
                "grounding_score": float(getattr(track, "score", 0.0)),
                "prompt_matched": bool(record.get("prompt_matched", False)),
                "matched_candidate": record.get("matched_candidate"),
                "prompt_span_count": int(record.get("prompt_span_count", 0)),
                "prompt_box_xyxy": np.asarray(track.box_prompt_xyxy).astype(float).tolist(),
                "mask_area_pixels": [int(mask.sum()) for mask in (np.asarray(track.masks_thw) > 0)],
            }
        )
    legend = [
        f"S{item['slot_id']}/E{item['entity_id']} -> {item['matched_candidate'] or 'UNMATCHED'} | phrase={item['grounding_phrase']}"
        for item in slot_reports
    ]
    framed = _external_bar(
        np.stack(frames),
        "Actual inference: SAM2 mask + CoTracker + entity binding",
        legend,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = output_dir / f"{stem}_entity_binding_overlay.mp4"
    inspectmod.write_mp4(raw, framed, fps=int(fps))
    browser = _browser_mp4(raw)
    chosen = np.linspace(0, len(framed) - 1, min(8, len(framed))).round().astype(int)
    tiles = [cv2.resize(framed[i], (448, 320), interpolation=cv2.INTER_AREA) for i in chosen]
    columns = 4
    rows = []
    for start in range(0, len(tiles), columns):
        row = tiles[start : start + columns]
        while len(row) < columns:
            row.append(np.full_like(tiles[0], 246))
        rows.append(np.concatenate(row, axis=1))
    grid = np.concatenate(rows, axis=0)
    grid_path = output_dir / f"{stem}_entity_binding_grid.png"
    cv2.imwrite(str(grid_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
    report = {
        "binding": binding,
        "slot_reports": slot_reports,
        "valid_slot_count": len(slot_reports),
        "matched_slot_count": sum(int(item["prompt_matched"]) for item in slot_reports),
        "all_valid_slots_matched": bool(slot_reports) and all(item["prompt_matched"] for item in slot_reports),
        "overlay_video": browser.name,
        "overlay_grid": grid_path.name,
    }
    report_path = output_dir / f"{stem}_slot_binding.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    report["report_json"] = report_path.name
    return report


def _build_visualized_object_context(
    model,
    *,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
):
    original_build_sample = model.viewer_grounding.build_sample
    original_run_cotracker = model._run_cotracker
    captured: dict[str, Any] = {}

    def capture_grounding(*args, **kwargs):
        output = original_build_sample(*args, **kwargs)
        captured["grounding"] = output
        return output

    def capture_cotracker(*args, **kwargs):
        output = original_run_cotracker(*args, **kwargs)
        captured["query_points"] = kwargs["query_points_prior"].detach().clone()
        captured["cotracker"] = output
        return output

    model.viewer_grounding.build_sample = capture_grounding
    model._run_cotracker = capture_cotracker
    try:
        object_context, debug = entity_infer._build_object_context_with_entity_binding(
            model,
            context_video_single=context_video_single,
            prompt=prompt,
            video_path=video_path,
        )
    finally:
        model.viewer_grounding.build_sample = original_build_sample
        model._run_cotracker = original_run_cotracker
    if _ACTIVE_CASE_DIR is not None and {"grounding", "query_points", "cotracker"}.issubset(captured):
        visual = _render_binding_overlay(
            context_video=context_video_single,
            grounding=captured["grounding"],
            query_points=captured["query_points"],
            cotracker=captured["cotracker"],
            binding=debug["entity_id_binding"],
            output_dir=_ACTIVE_CASE_DIR,
            stem=_ACTIVE_STEM or "case",
        )
        debug["entity_binding_visualization"] = visual
    return object_context, debug


def _run_single_visualized(**kwargs):
    global _ACTIVE_CASE_DIR, _ACTIVE_STEM
    output_video = Path(kwargs["output_video"])
    _ACTIVE_CASE_DIR = output_video.parent
    _ACTIVE_STEM = output_video.stem
    try:
        result, logs = _ORIGINAL_RUN_SINGLE(**kwargs)
        visual = result.get("object_debug", {}).get("entity_binding_visualization", {})
        result["entity_binding_validation"] = visual
        _CASE_REPORTS.append(
            {
                "case_stem": output_video.stem,
                "case_dir": str(output_video.parent),
                "prompt": result.get("input_caption", ""),
                "source_video": result.get("source_video", ""),
                "generated_video": output_video.name,
                **visual,
            }
        )
        return result, logs
    finally:
        _ACTIVE_CASE_DIR = None
        _ACTIVE_STEM = None


def _write_gallery() -> Path | None:
    if not _CASE_REPORTS:
        return None
    root = Path(_CASE_REPORTS[0]["case_dir"])
    cards = []
    for report in _CASE_REPORTS:
        rows = "".join(
            "<tr>"
            f"<td>S{item['slot_id']}</td><td>E{item['entity_id']}</td>"
            f"<td>{html.escape(item['grounding_phrase'])}</td>"
            f"<td>{html.escape(str(item['matched_candidate']))}</td>"
            f"<td>{'yes' if item['prompt_matched'] else 'NO'}</td>"
            "</tr>"
            for item in report.get("slot_reports", [])
        )
        cards.append(
            f"<section><h2>{html.escape(report['case_stem'])}</h2>"
            f"<p>{html.escape(report['prompt'])}</p><div class='media'>"
            f"<figure><video controls src='{html.escape(report['overlay_video'])}'></video><figcaption>Actual context slot/entity overlay</figcaption></figure>"
            f"<figure><video controls src='{html.escape(report['generated_video'])}'></video><figcaption>Generated 49-frame result</figcaption></figure></div>"
            f"<table><tr><th>Slot</th><th>Entity</th><th>GDINO phrase</th><th>Prompt candidate</th><th>Matched</th></tr>{rows}</table>"
            f"<p><a href='{html.escape(report['report_json'])}'>binding JSON</a> | <a href='{html.escape(report['overlay_grid'])}'>contact sheet</a></p></section>"
        )
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>Entity binding inference validation</title>
<style>body{{margin:0;background:#f2efe8;color:#20201e;font:15px Georgia,serif}}main{{max-width:1600px;margin:auto;padding:24px}}h1,h2{{font-family:Arial,sans-serif}}section{{background:#fff;border:1px solid #d2ccc0;padding:18px;margin:18px 0}}.media{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}figure{{margin:0}}video{{width:100%;background:#000}}figcaption{{padding:8px 0}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #d8d2c8;padding:8px;text-align:left}}@media(max-width:900px){{.media{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>Current entity-ID inference: slot binding validation</h1><p>Overlay and generated video come from the same in-process inference. A prompt match validates lexical routing; spatial correctness must also be checked against the colored mask/box.</p>{''.join(cards)}</main></body></html>"""
    index = root / "index.html"
    index.write_text(page, encoding="utf-8")
    (root / "binding_summary.json").write_text(
        json.dumps(_CASE_REPORTS, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return index


def _install_hooks() -> None:
    kubric_infer.trainmod.build_model = entity_infer._build_entity_bound_model
    infer0705.t0705 = kubric_infer.trainmod
    infer0705._build_object_context = _build_visualized_object_context
    infer0705._build_model_args = kubric_infer._build_model_args
    batch._run_single_case_in_process = _run_single_visualized


def main() -> None:
    batch._install_kubric_runtime_hooks = _install_hooks
    batch.main()
    index = _write_gallery()
    if index is not None:
        print(f"[entity-binding-gallery] {index}")


if __name__ == "__main__":
    main()
