"""Lightweight caption-to-slot entity-ID binding visualization for v2v JSONs.

This runs the inference preprocessing path only: caption noun extraction,
GroundingDINO, SAM2, query sampling, CoTracker, and deterministic entity-ID
routing. It intentionally does not load Wan, JEPA, VGGT, or a Stage1B weight.
"""
from __future__ import annotations

import argparse
import html
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

_PACKAGE_PARENT = str(Path(__file__).resolve().parents[2])
if _PACKAGE_PARENT not in sys.path:
    sys.path.insert(0, _PACKAGE_PARENT)

_DIFFSYNTH_ROOT = os.environ.get(
    "DIFFSYNTH_ROOT",
    "/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main",
)
if _DIFFSYNTH_ROOT not in sys.path:
    sys.path.insert(0, _DIFFSYNTH_ROOT)

from diffsynth.models.wan_video_text_encoder import HuggingfaceTokenizer

from code_vjepa_vggt.inspect_cotracker_vggt_geometry import (
    OBJECT_COLORS,
    draw_box_rgb,
    draw_point_rgb,
    tensor_frame_to_uint8_hwc,
    write_mp4,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_no_gt_box_replay_preserve_entity_id_binding as binding_train,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    visualize_caption_grounding_track_v2v as base,
)
from code_vjepa_vggt.utils.video_io import (
    preprocess_video_rgb_uint8,
    read_video_prefix,
)


DEFAULT_TOKENIZER = "/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B/google/umt5-xxl"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run lightweight GDINO/SAM2/CoTracker preprocessing and visualize "
            "caption-token -> object-slot -> entity-ID routing."
        )
    )
    parser.add_argument("--input-json", action="append", default=[])
    parser.add_argument("--input-json-list-path", type=Path)
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
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER)
    parser.add_argument(
        "--cotracker-checkpoint",
        default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth",
    )
    args = parser.parse_args()
    paths = list(args.input_json)
    if args.input_json_list_path is not None:
        paths.extend(
            line.strip()
            for line in args.input_json_list_path.read_text(encoding="utf-8").splitlines()
            if line.strip() and not line.lstrip().startswith("#")
        )
    resolved = [str(Path(value).expanduser().resolve()) for value in paths]
    if not resolved:
        parser.error("provide --input-json or --input-json-list-path")
    if len(set(resolved)) != len(resolved):
        parser.error("input JSON list contains duplicate paths")
    args.input_json = resolved
    return args


def _metadata_for_tracks(grounding: Any) -> list[dict[str, Any]]:
    records = [dict(item) for item in grounding.debug.get("candidate_metadata", [])]
    used: set[int] = set()
    assigned: list[dict[str, Any]] = []
    for track in grounding.object_tracks:
        source = str(getattr(track, "source_phrase", None) or track.phrase)
        selected = -1
        for index, record in enumerate(records):
            if index in used:
                continue
            if str(record.get("source_phrase", "")) == source:
                selected = index
                break
        if selected < 0:
            for index in range(len(records)):
                if index not in used:
                    selected = index
                    break
        if selected >= 0:
            used.add(selected)
            assigned.append(records[selected])
        else:
            assigned.append({})
    return assigned


def _build_binding_records(
    *,
    caption: str,
    grounding: Any,
    object_valid: torch.Tensor,
    tokenizer: HuggingfaceTokenizer,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    prompt_ids, prompt_mask = tokenizer(
        caption,
        return_mask=True,
        add_special_tokens=True,
    )
    valid_length = int(prompt_mask[0].sum().item())
    prompt_token_ids = [int(value) for value in prompt_ids[0, :valid_length].tolist()]
    # Hidden values do not affect lexical span lookup. The formal forward uses
    # the same IDs with real UMT5 hidden states for semantic pooling.
    dummy_context = torch.zeros((1, valid_length, 1), dtype=torch.float32)
    valid_slots = [
        int(value)
        for value in torch.nonzero(object_valid[0] > 0.5, as_tuple=False)
        .flatten()
        .tolist()
    ]
    metadata_by_slot = _metadata_for_tracks(grounding)
    records: list[dict[str, Any]] = []
    for entity_id, slot_id in enumerate(valid_slots):
        track = grounding.object_tracks[slot_id]
        phrase = str(track.phrase)
        _, span_count, candidate = binding_train._pool_phrase_from_prompt_context(
            prompt_token_ids=prompt_token_ids,
            prompt_context=dummy_context,
            tokenizer=tokenizer,
            phrase=phrase,
        )
        metadata = metadata_by_slot[slot_id] if slot_id < len(metadata_by_slot) else {}
        records.append(
            {
                "slot_id": int(slot_id),
                "entity_id": int(entity_id),
                "grounding_phrase": phrase,
                "source_phrase": str(
                    getattr(track, "source_phrase", None) or track.phrase
                ),
                "phrase_head": getattr(track, "phrase_head", None),
                "caption_char_span": (
                    list(track.caption_span) if track.caption_span is not None else None
                ),
                "detector_query": metadata.get("detector_query", phrase),
                "detected_phrase": metadata.get("detected_phrase", phrase),
                "detector_candidate_rank": metadata.get("candidate_rank"),
                "detector_instance_count": metadata.get("instance_count"),
                "grounding_score": float(track.score),
                "prompt_matched": candidate is not None,
                "matched_candidate": candidate,
                "prompt_span_count": int(span_count),
                "prompt_box_xyxy": np.asarray(track.box_prompt_xyxy)
                .astype(float)
                .tolist(),
                "sam_mask_area_pixels": [
                    int(mask.sum()) for mask in (np.asarray(track.masks_thw) > 0)
                ],
            }
        )
    ids = [record["entity_id"] for record in records]
    candidates = [
        str(record["matched_candidate"])
        for record in records
        if record["matched_candidate"] is not None
    ]
    repeated = {
        candidate: count
        for candidate, count in Counter(candidates).items()
        if count > 1
    }
    diagnostics = {
        "enabled": True,
        "id_policy": "deterministic_valid_slot_order",
        "valid_slot_ids": valid_slots,
        "slot_entity_ids": [
            next(
                (
                    int(record["entity_id"])
                    for record in records
                    if int(record["slot_id"]) == slot_id
                ),
                -1,
            )
            for slot_id in range(int(object_valid.shape[1]))
        ],
        "entity_ids_unique": len(ids) == len(set(ids)),
        "matched_slot_count": sum(int(record["prompt_matched"]) for record in records),
        "unmatched_slot_count": sum(
            int(not record["prompt_matched"]) for record in records
        ),
        "all_valid_slots_matched": bool(records)
        and all(record["prompt_matched"] for record in records),
        "repeated_caption_candidates": repeated,
        "repeated_candidate_policy": (
            "shared_caption_semantics_with_distinct_entity_ids" if repeated else "none"
        ),
        "prompt_valid_token_count": valid_length,
    }
    return records, diagnostics


def _render_overlay(
    *,
    context_video: torch.Tensor,
    grounding: Any,
    cotracker: Any,
    query_points: torch.Tensor,
    records: list[dict[str, Any]],
    points_per_object: int,
) -> np.ndarray:
    valid_query_count = len(records) * int(points_per_object)
    tracks = (
        cotracker.tracks[0, :, :valid_query_count].detach().float().cpu().numpy()
    )
    visibility = (
        cotracker.visibility[0, :, :valid_query_count]
        .detach()
        .float()
        .cpu()
        .numpy()
    )
    points = query_points[0, :valid_query_count].detach().float().cpu().numpy()
    rendered: list[np.ndarray] = []
    for frame_id in range(int(context_video.shape[1])):
        frame = tensor_frame_to_uint8_hwc(context_video[:, frame_id]).copy()
        for record in records:
            slot_id = int(record["slot_id"])
            entity_id = int(record["entity_id"])
            track = grounding.object_tracks[slot_id]
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
            candidate = record["matched_candidate"] or "UNMATCHED"
            draw_box_rgb(
                frame,
                np.asarray(track.boxes_t4[frame_id], dtype=np.float32),
                color,
                f"S{slot_id}/E{entity_id}:{candidate}",
            )
            if frame_id == int(grounding.prompt_frame_idx):
                draw_box_rgb(
                    frame,
                    np.asarray(track.box_prompt_xyxy, dtype=np.float32),
                    color,
                    f"GDINO S{slot_id}",
                )
        for query_id in range(valid_query_count):
            owner = query_id // max(int(points_per_object), 1)
            slot_id = int(records[owner]["slot_id"])
            color = OBJECT_COLORS[slot_id % len(OBJECT_COLORS)]
            if frame_id == int(grounding.prompt_frame_idx):
                draw_point_rgb(frame, points[query_id], color, f"q{query_id}", radius=5)
            if visibility[frame_id, query_id] >= 0.5:
                draw_point_rgb(frame, tracks[frame_id, query_id], color, "", radius=4)
        rendered.append(frame)
    lines = [
        (
            f"S{record['slot_id']}/E{record['entity_id']} "
            f"phrase={record['grounding_phrase']} -> "
            f"caption={record['matched_candidate'] or 'UNMATCHED'} | "
            f"query={record['detector_query']}"
        )
        for record in records
    ]
    if not lines:
        lines = ["NO VALID OBJECT SLOT"]
    return base._external_header(
        np.stack(rendered),
        "Inference preprocessing: caption -> GDINO/SAM2/CoTracker -> slot/entity ID",
        lines,
    )


def _run_case(
    *,
    json_path: Path,
    args: argparse.Namespace,
    provider: Any,
    cotracker: Any,
    tokenizer: HuggingfaceTokenizer,
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
    query_points, query_frame_ids, object_valid = base._build_query_tensors(
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
    records, binding = _build_binding_records(
        caption=caption,
        grounding=grounding,
        object_valid=object_valid,
        tokenizer=tokenizer,
    )
    valid_query_count = len(records) * int(args.points_per_object)
    if valid_query_count:
        visible = (
            cotracker_output.visibility[0, :, :valid_query_count]
            .detach()
            .float()
            .cpu()
        )
        visible_fraction = float((visible >= 0.5).float().mean().item())
    else:
        visible_fraction = 0.0
    for record in records:
        start = int(record["slot_id"]) * int(args.points_per_object)
        end = start + int(args.points_per_object)
        slot_visibility = cotracker_output.visibility[0, :, start:end].detach().float().cpu()
        record["cotracker_visible_fraction"] = float(
            (slot_visibility >= 0.5).float().mean().item()
        )

    overlay_frames = _render_overlay(
        context_video=context_video,
        grounding=grounding,
        cotracker=cotracker_output,
        query_points=query_points,
        records=records,
        points_per_object=int(args.points_per_object),
    )
    overlay_raw = case_dir / "entity_id_binding_overlay.mp4"
    write_mp4(overlay_raw, overlay_frames, fps=int(args.fps))
    overlay_browser = base._ensure_browser_video(overlay_raw)
    context_browser = base._write_context_video(
        case_dir / "model_input_context.mp4", context_video, fps=int(args.fps)
    )
    source_browser = base._export_source_video(
        source_video, case_dir / "source_video.browser.mp4"
    )
    grid_path = case_dir / "entity_id_binding_grid.png"
    base._write_contact_sheet(grid_path, overlay_frames)
    prompt_path = case_dir / "prompt_frame_binding.png"
    cv2.imwrite(
        str(prompt_path),
        cv2.cvtColor(overlay_frames[int(grounding.prompt_frame_idx)], cv2.COLOR_RGB2BGR),
    )
    result = {
        "status": "success",
        "input_json": str(json_path),
        "source_video": str(source_video),
        "caption": caption,
        "source_resolution_hw": [raw_height, raw_width],
        "model_input_resolution_hw": [int(args.height), int(args.width)],
        "context_frame_indices": frame_indices.astype(int).tolist(),
        "prompt_frame_idx": int(grounding.prompt_frame_idx),
        "physical_phrase_extraction": grounding.debug.get(
            "caption_phrase_extraction", {}
        ),
        "grounding_debug": {
            key: value
            for key, value in grounding.debug.items()
            if key not in {"candidate_metadata"}
        },
        "binding": binding,
        "slots": records,
        "valid_object_count": len(records),
        "valid_query_count": valid_query_count,
        "cotracker_visible_fraction": visible_fraction,
        "artifacts": {
            "source_video": source_browser.name,
            "model_input_context": context_browser.name,
            "prompt_frame_binding": prompt_path.name,
            "binding_overlay": overlay_browser.name,
            "binding_grid": grid_path.name,
        },
    }
    (case_dir / "result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"case": json_path.stem, "case_dir": case_dir.name, **result}


def _write_gallery(output_root: Path, results: list[dict[str, Any]]) -> None:
    cards: list[str] = []
    for result in results:
        if result.get("status") != "success":
            cards.append(
                f"<section class='failed'><h2>{html.escape(result['case'])}</h2>"
                f"<p>{html.escape(result.get('error', 'unknown error'))}</p></section>"
            )
            continue
        rel = html.escape(result["case_dir"])
        artifacts = result["artifacts"]
        rows = "".join(
            "<tr>"
            f"<td>S{record['slot_id']}</td><td>E{record['entity_id']}</td>"
            f"<td>{html.escape(record['source_phrase'])}</td>"
            f"<td>{html.escape(str(record['detector_query']))}</td>"
            f"<td>{html.escape(str(record['matched_candidate'] or 'UNMATCHED'))}</td>"
            f"<td class={'ok' if record['prompt_matched'] else 'bad'}>"
            f"{'yes' if record['prompt_matched'] else 'NO'}</td>"
            f"<td>{record['prompt_span_count']}</td>"
            f"<td>{record['cotracker_visible_fraction']:.3f}</td>"
            "</tr>"
            for record in result["slots"]
        )
        binding = result["binding"]
        status_class = "ok" if binding["all_valid_slots_matched"] else "bad"
        cards.append(
            f"<section><h2>{html.escape(result['case'])}</h2>"
            f"<p>{html.escape(result['caption'])}</p>"
            f"<p class='{status_class}'>slots={result['valid_object_count']}; "
            f"matched={binding['matched_slot_count']}; "
            f"unmatched={binding['unmatched_slot_count']}; "
            f"unique IDs={binding['entity_ids_unique']}; "
            f"repeated token candidates={html.escape(str(binding['repeated_caption_candidates']))}</p>"
            "<div class='media'>"
            f"<figure><video controls src='{rel}/{html.escape(artifacts['source_video'])}'></video><figcaption>Source video</figcaption></figure>"
            f"<figure><video controls src='{rel}/{html.escape(artifacts['model_input_context'])}'></video><figcaption>Exact prefix-8 model input</figcaption></figure>"
            f"<figure><video controls src='{rel}/{html.escape(artifacts['binding_overlay'])}'></video><figcaption>SAM2 + CoTracker + slot/entity-ID overlay</figcaption></figure>"
            f"<figure><img src='{rel}/{html.escape(artifacts['prompt_frame_binding'])}'><figcaption>Prompt-frame binding</figcaption></figure>"
            "</div>"
            "<table><tr><th>Slot</th><th>Entity ID</th><th>Caption phrase</th>"
            "<th>Detector query</th><th>Matched token candidate</th><th>Matched</th>"
            f"<th>Span count</th><th>Track visible</th></tr>{rows}</table>"
            f"<p><a href='{rel}/result.json'>result.json</a> | "
            f"<a href='{rel}/{html.escape(artifacts['binding_grid'])}'>contact sheet</a></p>"
            "</section>"
        )
    successes = [item for item in results if item.get("status") == "success"]
    failed = [item for item in results if item.get("status") != "success"]
    cases_with_slots = [item for item in successes if item["valid_object_count"] > 0]
    no_slot_cases = [item for item in successes if item["valid_object_count"] == 0]
    fully_bound_cases = [
        item
        for item in cases_with_slots
        if item["binding"]["all_valid_slots_matched"]
    ]
    partially_bound_cases = [
        item
        for item in cases_with_slots
        if not item["binding"]["all_valid_slots_matched"]
    ]
    total_slots = sum(int(item["valid_object_count"]) for item in successes)
    matched_slots = sum(int(item["binding"]["matched_slot_count"]) for item in successes)
    summary = {
        "requested_case_count": len(results),
        "successful_case_count": len(successes),
        "failed_case_count": len(failed),
        "case_with_valid_slots_count": len(cases_with_slots),
        "no_valid_slot_case_count": len(no_slot_cases),
        "fully_lexically_bound_case_count": len(fully_bound_cases),
        "partially_lexically_bound_case_count": len(partially_bound_cases),
        "no_valid_slot_cases": [item["case"] for item in no_slot_cases],
        "partially_lexically_bound_cases": [
            item["case"] for item in partially_bound_cases
        ],
        "total_valid_slot_count": total_slots,
        "matched_slot_count": matched_slots,
        "unmatched_slot_count": total_slots - matched_slots,
        "all_successful_cases_have_unique_entity_ids": all(
            bool(item["binding"]["entity_ids_unique"]) for item in successes
        ),
        "all_successful_cases_fully_lexically_matched": bool(successes)
        and all(bool(item["binding"]["all_valid_slots_matched"]) for item in successes),
        "cases": results,
    }
    page = f"""<!doctype html><html><head><meta charset="utf-8"><title>Entity-ID binding forward</title>
<style>body{{margin:0;background:#eef0ed;color:#20221f;font:15px Georgia,serif}}main{{max-width:1700px;margin:auto;padding:24px}}h1,h2{{font-family:Verdana,sans-serif;letter-spacing:0}}section{{background:#fff;border:1px solid #c5cac2;padding:18px;margin:18px 0;border-radius:4px}}.media{{display:grid;grid-template-columns:1fr 1fr;gap:14px}}figure{{margin:0}}video,img{{display:block;width:100%;background:#000}}figcaption{{padding:7px 0}}table{{width:100%;border-collapse:collapse}}th,td{{border:1px solid #d0d5cd;padding:7px;text-align:left}}.ok{{color:#17613d;font-weight:bold}}.bad,.failed{{color:#9b2f27;font-weight:bold}}code{{font-family:monospace}}@media(max-width:900px){{.media{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>20-case entity-ID binding forward</h1><p>Pipeline: caption physical noun phrases -> GroundingDINO -> SAM2 -> query points -> CoTracker -> deterministic slot/entity-ID routing. Lexical matches use the exact Scheme-C UMT5 token-span helper; no Wan or Stage1B checkpoint is loaded.</p><p><b>Processed:</b> {len(successes)}/{len(results)}; <b>cases with slots:</b> {len(cases_with_slots)}; <b>no-slot cases:</b> {len(no_slot_cases)}; <b>fully bound cases:</b> {len(fully_bound_cases)}; <b>partial cases:</b> {len(partially_bound_cases)}; <b>slots:</b> {matched_slots}/{total_slots} caption-token matched. Repeated nouns intentionally share semantic text spans while retaining distinct entity IDs.</p>{''.join(cards)}</main></body></html>"""
    (output_root / "index.html").write_text(page, encoding="utf-8")
    (output_root / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = _parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    if str(args.device).startswith("cuda"):
        torch.cuda.set_device(torch.device(args.device))
        torch.cuda.reset_peak_memory_stats(torch.device(args.device))
    tokenizer = HuggingfaceTokenizer(
        name=str(args.tokenizer_path),
        seq_len=512,
        clean="whitespace",
        local_files_only=True,
    )
    provider = base._build_provider(args)
    cotracker = base.CoTrackerAdapter(
        checkpoint_path=str(args.cotracker_checkpoint),
        num_queries=int(args.max_objects) * int(args.points_per_object),
        device=str(args.device),
        input_hw=(384, 512),
        window_len=60,
    )
    results: list[dict[str, Any]] = []
    for case_index, raw_path in enumerate(args.input_json, 1):
        path = Path(raw_path)
        print(f"[case {case_index:02d}/{len(args.input_json):02d}] {path}", flush=True)
        try:
            result = _run_case(
                json_path=path,
                args=args,
                provider=provider,
                cotracker=cotracker,
                tokenizer=tokenizer,
            )
            print(
                f"[binding] slots={result['valid_object_count']} "
                f"matched={result['binding']['matched_slot_count']} "
                f"ids_unique={result['binding']['entity_ids_unique']}",
                flush=True,
            )
            results.append(result)
        except Exception as exc:
            print(f"[case-error] {type(exc).__name__}: {exc}", flush=True)
            results.append(
                {
                    "case": path.stem,
                    "case_dir": path.stem,
                    "status": "failed",
                    "input_json": str(path),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
        _write_gallery(args.output_root, results)
    if str(args.device).startswith("cuda"):
        device = torch.device(args.device)
        peak_allocated = int(torch.cuda.max_memory_allocated(device))
        peak_reserved = int(torch.cuda.max_memory_reserved(device))
        memory = {
            "device": str(args.device),
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_gib": peak_allocated / (1024**3),
            "peak_reserved_gib": peak_reserved / (1024**3),
        }
        (args.output_root / "gpu_memory.json").write_text(
            json.dumps(memory, indent=2), encoding="utf-8"
        )
        print(f"[gpu-memory] {memory}", flush=True)
    print(f"[gallery] {args.output_root / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
