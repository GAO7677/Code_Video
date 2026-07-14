from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerAdapter
from code_vjepa_vggt.inspect_cotracker_vggt_geometry import (
    OBJECT_COLORS,
    draw_box_rgb,
    draw_point_rgb,
    tensor_frame_to_uint8_hwc,
    write_mp4,
)
from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (
    ViewerGroundingBoxProvider,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    inspect_kubric_train_forward_aux_overlay as inspectmod,
)
from code_vjepa_vggt.utils.video_io import (
    preprocess_video_rgb_uint8,
    read_video_prefix,
)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize caption-grounded GDINO, SAM2, and CoTracker inference."
    )
    parser.add_argument("--input-json", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--num-context-frames", type=int, default=8)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fps", type=int, default=12)
    parser.add_argument("--max-objects", type=int, default=4)
    parser.add_argument("--points-per-object", type=int, default=8)
    parser.add_argument("--caption-max-phrases", type=int, default=4)
    parser.add_argument("--caption-min-score", type=float, default=4.0)
    parser.add_argument("--gdino-box-threshold", type=float, default=0.20)
    parser.add_argument("--gdino-text-threshold", type=float, default=0.15)
    parser.add_argument(
        "--cotracker-checkpoint",
        default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth",
    )
    return parser.parse_args()


def _build_provider(args: argparse.Namespace) -> ViewerGroundingBoxProvider:
    return ViewerGroundingBoxProvider(
        device=str(args.device),
        segment_len=int(args.num_context_frames),
        max_objects=int(args.max_objects),
        points_per_object=int(args.points_per_object),
        proposal_source="gdino_only",
        motion_score_ratio=0.15,
        text_prompt="",
        extra_prompt_terms="",
        include_caption_terms=True,
        gdino_box_threshold=float(args.gdino_box_threshold),
        gdino_text_threshold=float(args.gdino_text_threshold),
        prompt_frame_mode="first",
        track_dedupe_iou_threshold=0.75,
        container_suppress_ratio_threshold=0.95,
        container_suppress_min_contained=2,
        container_suppress_min_area_ratio=1.5,
        container_suppress_small_iou_threshold=0.7,
        caption_prompt_mode="physical_noun_phrases",
        caption_max_phrases=int(args.caption_max_phrases),
        caption_min_score=float(args.caption_min_score),
    )


def _build_query_tensors(
    grounding: Any,
    *,
    max_objects: int,
    points_per_object: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped = torch.from_numpy(grounding.grouped_queries_px).float()
    valid = torch.from_numpy(grounding.object_valid_mask).float()
    boxes = torch.from_numpy(grounding.context_boxes_norm).float()
    query_points = grouped.view(1, int(max_objects) * int(points_per_object), 2)
    frame_ids: list[float] = []
    for object_id in range(int(max_objects)):
        first_valid_frame = int(grounding.prompt_frame_idx)
        if bool(valid[object_id] > 0.5):
            for frame_id in range(int(boxes.shape[0])):
                box = boxes[frame_id, object_id]
                if bool((box[2] > box[0]) and (box[3] > box[1])):
                    first_valid_frame = frame_id
                    break
        frame_ids.extend([float(first_valid_frame)] * int(points_per_object))
    query_frame_ids = torch.tensor(frame_ids, dtype=torch.float32).view(1, -1, 1)
    return query_points, query_frame_ids, valid.view(1, int(max_objects))


def _external_header(frames: np.ndarray, title: str, lines: list[str]) -> np.ndarray:
    header_height = 36 + 23 * len(lines)
    output = np.full(
        (len(frames), frames.shape[1] + header_height, frames.shape[2], 3),
        (247, 245, 239),
        dtype=np.uint8,
    )
    output[:, header_height:] = frames
    for frame_id, output_frame in enumerate(output):
        cv2.putText(
            output_frame,
            f"{title} | context {frame_id:02d}/{len(frames) - 1:02d}",
            (12, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (28, 28, 28),
            1,
            cv2.LINE_AA,
        )
        for line_id, line in enumerate(lines):
            cv2.putText(
                output_frame,
                line[:170],
                (12, 51 + line_id * 23),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (45, 45, 45),
                1,
                cv2.LINE_AA,
            )
    return output


def _candidate_metadata_for_track(track: Any, debug: dict[str, Any]) -> dict[str, Any]:
    source_phrase = str(getattr(track, "source_phrase", None) or track.phrase)
    for record in debug.get("candidate_metadata", []):
        if str(record.get("source_phrase", "")) == source_phrase:
            return dict(record)
    return {}


def _render_overlay(
    *,
    context_video: torch.Tensor,
    grounding: Any,
    cotracker: Any,
    query_points: torch.Tensor,
    valid_query_count: int,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    tracks = cotracker.tracks[0, :, :valid_query_count].detach().float().cpu().numpy()
    visibility = cotracker.visibility[0, :, :valid_query_count].detach().float().cpu().numpy()
    points = query_points[0, :valid_query_count].detach().float().cpu().numpy()
    points_per_object = int(valid_query_count // max(len(grounding.object_tracks), 1))
    slot_reports: list[dict[str, Any]] = []

    for slot_id, track in enumerate(grounding.object_tracks):
        metadata = _candidate_metadata_for_track(track, grounding.debug)
        slot_reports.append(
            {
                "slot_id": int(slot_id),
                "source_phrase": str(getattr(track, "source_phrase", None) or track.phrase),
                "phrase_head": getattr(track, "phrase_head", None),
                "caption_span": list(track.caption_span) if track.caption_span is not None else None,
                "detector_query": metadata.get("detector_query", track.phrase),
                "detected_phrase": metadata.get("detected_phrase", track.phrase),
                "score": float(track.score),
                "prompt_box_xyxy": np.asarray(track.box_prompt_xyxy).astype(float).tolist(),
                "sam_mask_area_pixels": [
                    int(mask.sum()) for mask in (np.asarray(track.masks_thw) > 0)
                ],
            }
        )

    rendered: list[np.ndarray] = []
    for frame_id in range(int(context_video.shape[1])):
        frame = tensor_frame_to_uint8_hwc(context_video[:, frame_id]).copy()
        for slot_id, track in enumerate(grounding.object_tracks):
            color = OBJECT_COLORS[slot_id % len(OBJECT_COLORS)]
            mask = np.asarray(track.masks_thw[frame_id]) > 0
            if mask.shape == frame.shape[:2]:
                tint = np.zeros_like(frame)
                tint[mask] = color
                frame = np.where(
                    mask[..., None],
                    np.clip(
                        frame.astype(np.float32) * 0.72
                        + tint.astype(np.float32) * 0.28,
                        0,
                        255,
                    ).astype(np.uint8),
                    frame,
                )
            phrase = str(getattr(track, "source_phrase", None) or track.phrase)
            draw_box_rgb(
                frame,
                np.asarray(track.boxes_t4[frame_id], dtype=np.float32),
                color,
                f"S{slot_id}:{phrase}",
            )
            if frame_id == int(grounding.prompt_frame_idx):
                draw_box_rgb(
                    frame,
                    np.asarray(track.box_prompt_xyxy, dtype=np.float32),
                    color,
                    f"GDINO{slot_id}",
                )

        for query_id in range(valid_query_count):
            owner = query_id // max(points_per_object, 1)
            color = OBJECT_COLORS[owner % len(OBJECT_COLORS)]
            if frame_id == int(grounding.prompt_frame_idx):
                draw_point_rgb(frame, points[query_id], color, f"q{query_id}", radius=5)
            if visibility[frame_id, query_id] >= 0.5:
                draw_point_rgb(frame, tracks[frame_id, query_id], color, "", radius=4)
        rendered.append(frame)

    lines = [
        f"S{x['slot_id']} source={x['source_phrase']} | GDINO query={x['detector_query']} | score={x['score']:.3f}"
        for x in slot_reports
    ]
    return _external_header(
        np.stack(rendered),
        "Actual forward: GDINO box -> SAM2 mask/track -> CoTracker queries",
        lines,
    ), slot_reports


def _write_contact_sheet(path: Path, frames: np.ndarray) -> None:
    indices = np.linspace(0, len(frames) - 1, min(8, len(frames))).round().astype(int)
    tile_width = 448
    tile_height = max(1, int(round(frames.shape[1] * tile_width / frames.shape[2])))
    tiles = [
        cv2.resize(frames[index], (tile_width, tile_height), interpolation=cv2.INTER_AREA)
        for index in indices
    ]
    rows = [np.concatenate(tiles[start : start + 4], axis=1) for start in range(0, 8, 4)]
    cv2.imwrite(str(path), cv2.cvtColor(np.concatenate(rows, axis=0), cv2.COLOR_RGB2BGR))


def _run_case(
    *,
    json_path: Path,
    args: argparse.Namespace,
    provider: ViewerGroundingBoxProvider,
    cotracker: CoTrackerAdapter,
) -> dict[str, Any]:
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    caption = str(payload["input_caption"])
    source_video = Path(str(payload.get("source_video") or payload["input_video"])).resolve()
    case_dir = args.output_root / json_path.stem
    case_dir.mkdir(parents=True, exist_ok=True)

    raw_frames, frame_indices = read_video_prefix(source_video, int(args.num_context_frames))
    raw_height, raw_width = int(raw_frames.shape[1]), int(raw_frames.shape[2])
    context_video = preprocess_video_rgb_uint8(
        raw_frames,
        (int(args.height), int(args.width)),
        resize_mode="cover_crop",
        cover_crop_hw=(int(args.height), int(args.width)),
    )
    frames_tchw_01 = (
        ((context_video.permute(1, 0, 2, 3).float() + 1.0) / 2.0)
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )

    grounding = provider.build_sample(
        frames_tchw_01=frames_tchw_01,
        caption=caption,
        image_hw=(int(args.height), int(args.width)),
    )
    query_points, query_frame_ids, object_valid = _build_query_tensors(
        grounding,
        max_objects=int(args.max_objects),
        points_per_object=int(args.points_per_object),
    )
    frames_bthwc_01 = torch.from_numpy(frames_tchw_01).permute(0, 2, 3, 1).unsqueeze(0)
    cotracker_output = cotracker(
        frames_bthwc_01,
        query_points_prior=query_points,
        query_frame_ids=query_frame_ids,
        query_image_hw=(int(args.height), int(args.width)),
    )
    valid_object_count = int((object_valid > 0.5).sum().item())
    valid_query_count = valid_object_count * int(args.points_per_object)

    overlay_frames, slot_reports = _render_overlay(
        context_video=context_video,
        grounding=grounding,
        cotracker=cotracker_output,
        query_points=query_points,
        valid_query_count=valid_query_count,
    )
    overlay_raw = case_dir / "detection_sam2_cotracker_overlay.mp4"
    write_mp4(overlay_raw, overlay_frames, fps=int(args.fps))
    overlay_browser = inspectmod._ensure_browser_video(overlay_raw)
    context_browser = inspectmod._write_tensor_video(
        case_dir / "model_input_context.mp4", context_video, fps=int(args.fps)
    )
    source_browser = inspectmod._export_browser_video(
        source_video, case_dir / "source_video.browser.mp4"
    )
    grid_path = case_dir / "detection_sam2_cotracker_grid.png"
    _write_contact_sheet(grid_path, overlay_frames)

    prompt_frame = overlay_frames[int(grounding.prompt_frame_idx)]
    prompt_path = case_dir / "prompt_frame_detection.png"
    cv2.imwrite(str(prompt_path), cv2.cvtColor(prompt_frame, cv2.COLOR_RGB2BGR))

    cot_visibility = cotracker_output.visibility[0, :, :valid_query_count].detach().float().cpu()
    result = {
        "input_json": str(json_path),
        "source_video": str(source_video),
        "caption": caption,
        "source_resolution_hw": [raw_height, raw_width],
        "model_input_resolution_hw": [int(args.height), int(args.width)],
        "context_frame_indices": frame_indices.astype(int).tolist(),
        "prompt_frame_idx": int(grounding.prompt_frame_idx),
        "physical_phrase_extraction": grounding.debug.get("caption_phrase_extraction", {}),
        "candidate_metadata": grounding.debug.get("candidate_metadata", []),
        "slots": slot_reports,
        "valid_object_count": valid_object_count,
        "valid_query_count": valid_query_count,
        "cotracker_visible_fraction": float((cot_visibility >= 0.5).float().mean().item()),
        "artifacts": {
            "source_video": source_browser.name,
            "model_input_context": context_browser.name,
            "prompt_frame_detection": prompt_path.name,
            "track_overlay": overlay_browser.name,
            "track_grid": grid_path.name,
        },
    }
    (case_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"case": json_path.stem, "case_dir": case_dir.name, **result}


def _write_gallery(output_root: Path, results: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for result in results:
        rel = html.escape(result["case_dir"])
        artifacts = result["artifacts"]
        rows = "".join(
            "<tr>"
            f"<td>S{slot['slot_id']}</td>"
            f"<td>{html.escape(slot['source_phrase'])}</td>"
            f"<td>{html.escape(str(slot['detector_query']))}</td>"
            f"<td>{slot['score']:.3f}</td>"
            f"<td>{html.escape(str(slot['prompt_box_xyxy']))}</td>"
            "</tr>"
            for slot in result["slots"]
        )
        cards.append(
            f"<section><h2>{html.escape(result['case'])}</h2>"
            f"<p>{html.escape(result['caption'])}</p>"
            f"<p>Source {result['source_resolution_hw']} -> model input {result['model_input_resolution_hw']}; "
            f"slots={result['valid_object_count']}, queries={result['valid_query_count']}, "
            f"CoTracker visible={result['cotracker_visible_fraction']:.3f}</p>"
            "<div class='media'>"
            f"<figure><video controls src='{rel}/{html.escape(artifacts['source_video'])}'></video><figcaption>Original source video</figcaption></figure>"
            f"<figure><video controls src='{rel}/{html.escape(artifacts['model_input_context'])}'></video><figcaption>Exact prefix-8 model input after cover-crop</figcaption></figure>"
            f"<figure><video controls src='{rel}/{html.escape(artifacts['track_overlay'])}'></video><figcaption>GDINO + SAM2 + CoTracker overlay</figcaption></figure>"
            f"<figure><img src='{rel}/{html.escape(artifacts['prompt_frame_detection'])}'><figcaption>Prompt-frame detection</figcaption></figure>"
            "</div>"
            f"<table><tr><th>Slot</th><th>Caption phrase</th><th>GDINO query</th><th>Score</th><th>Prompt box</th></tr>{rows}</table>"
            f"<p><a href='{rel}/result.json'>result.json</a> | <a href='{rel}/{html.escape(artifacts['track_grid'])}'>contact sheet</a></p>"
            "</section>"
        )
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Caption grounding and tracking</title>
<style>body{{margin:0;background:#f0eee8;color:#20201e;font:15px Georgia,serif}}main{{max-width:1600px;margin:auto;padding:24px}}h1,h2{{font-family:Arial,sans-serif;letter-spacing:0}}section{{background:#fff;border:1px solid #cbc7bd;padding:18px;margin:18px 0}}.media{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}figure{{margin:0}}video,img{{display:block;width:100%;background:#000}}figcaption{{padding:7px 0}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #d5d1c7;padding:7px;text-align:left}}@media(max-width:900px){{.media{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>Caption-grounded detection and tracking forward</h1>{''.join(cards)}</main></body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")
    (output_root / "summary.json").write_text(
        json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    provider = _build_provider(args)
    cotracker = CoTrackerAdapter(
        checkpoint_path=str(args.cotracker_checkpoint),
        num_queries=int(args.max_objects) * int(args.points_per_object),
        device=str(args.device),
        input_hw=(384, 512),
        window_len=60,
    )
    results = []
    for raw_path in args.input_json:
        path = Path(raw_path).expanduser().resolve()
        print(f"[case] {path}", flush=True)
        results.append(
            _run_case(
                json_path=path,
                args=args,
                provider=provider,
                cotracker=cotracker,
            )
        )
    _write_gallery(args.output_root, results)
    print(f"[gallery] {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
