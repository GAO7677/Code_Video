"""Compare the legacy mask query sampler with a no-GT temporal SAM2 repair.

The experiment is deliberately pre-DiT: GDINO/SAM2 tracks are shared by both
schemes and only the CoTracker query selection changes. This keeps the result
attributable to query quality rather than Wan sampling randomness.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerAdapter
from code_vjepa_vggt.inspect_cotracker_vggt_geometry import (
    OBJECT_COLORS,
    render_track_overlay,
    write_mp4,
)
from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (
    DetectedObjectTrack,
    ViewerGroundingBoxProvider,
)
from code_vjepa_vggt.utils.object_priors import _extract_mask_components, sample_points_from_mask
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix


@dataclass
class PointQuality:
    query_x: float
    query_y: float
    visible_ratio: float
    in_mask_ratio: float
    retained_given_visible: float
    mean_mask_margin_px: float
    score: float


def _point_inside(mask: np.ndarray, xy: np.ndarray) -> bool:
    x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
    return 0 <= y < mask.shape[0] and 0 <= x < mask.shape[1] and bool(mask[y, x] > 0)


def _main_component(mask: np.ndarray, prompt_box: np.ndarray) -> np.ndarray | None:
    components = _extract_mask_components(mask)
    if not components:
        return None
    cx = 0.5 * float(prompt_box[0] + prompt_box[2])
    cy = 0.5 * float(prompt_box[1] + prompt_box[3])
    for component in components:
        comp_mask = np.asarray(component["mask"], dtype=np.uint8)
        if _point_inside(comp_mask, np.asarray([cx, cy], dtype=np.float32)):
            return comp_mask
    return np.asarray(components[0]["mask"], dtype=np.uint8)


def _point_qualities(
    *,
    masks_thw: np.ndarray,
    tracks_tk2: np.ndarray,
    visibility_tk: np.ndarray,
) -> list[PointQuality]:
    target_visible = np.asarray([(mask > 0).any() for mask in masks_thw], dtype=bool)
    distance_maps = [
        cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)
        if bool(mask.any())
        else np.zeros_like(mask, dtype=np.float32)
        for mask in masks_thw
    ]
    scale = max(float(np.sqrt(np.median([mask.sum() for mask in masks_thw if mask.any()] or [1]))), 1.0)
    output: list[PointQuality] = []
    for point_idx in range(tracks_tk2.shape[1]):
        visible = visibility_tk[:, point_idx] >= 0.5
        same_mask = 0
        margins: list[float] = []
        for frame_idx in range(tracks_tk2.shape[0]):
            if not target_visible[frame_idx] or not visible[frame_idx]:
                continue
            xy = tracks_tk2[frame_idx, point_idx]
            if _point_inside(masks_thw[frame_idx], xy):
                same_mask += 1
                x, y = int(round(float(xy[0]))), int(round(float(xy[1])))
                margins.append(float(distance_maps[frame_idx][y, x]))
        visible_ratio = float(visible.mean())
        in_mask_ratio = float(same_mask) / max(float(target_visible.sum()), 1.0)
        retained = float(same_mask) / max(float((visible & target_visible).sum()), 1.0)
        mean_margin = float(np.mean(margins)) if margins else 0.0
        score = 4.0 * in_mask_ratio + visible_ratio + min(mean_margin / scale, 1.0)
        output.append(
            PointQuality(
                query_x=float(tracks_tk2[0, point_idx, 0]),
                query_y=float(tracks_tk2[0, point_idx, 1]),
                visible_ratio=visible_ratio,
                in_mask_ratio=in_mask_ratio,
                retained_given_visible=retained,
                mean_mask_margin_px=mean_margin,
                score=score,
            )
        )
    return output


def _select_diverse(qualities: list[PointQuality], *, count: int, min_visible: float, min_in_mask: float) -> list[int]:
    eligible = [
        idx
        for idx, item in enumerate(qualities)
        if item.visible_ratio >= min_visible and item.in_mask_ratio >= min_in_mask
    ]
    if len(eligible) < count:
        return []
    points = np.asarray([[item.query_x, item.query_y] for item in qualities], dtype=np.float32)
    selected = [max(eligible, key=lambda idx: qualities[idx].score)]
    diagonal = max(float(np.linalg.norm(np.ptp(points[eligible], axis=0))), 1.0)
    while len(selected) < count:
        remaining = [idx for idx in eligible if idx not in selected]
        next_idx = max(
            remaining,
            key=lambda idx: qualities[idx].score + 0.75 * min(
                float(np.linalg.norm(points[idx] - points[chosen])) / diagonal
                for chosen in selected
            ),
        )
        selected.append(next_idx)
    return selected


def _run_cotracker(
    cotracker: CoTrackerAdapter,
    frames_bthwc: torch.Tensor,
    points: np.ndarray,
    prompt_frame: int,
    image_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    query = torch.from_numpy(points).unsqueeze(0).float()
    frame_ids = torch.full((1, points.shape[0], 1), float(prompt_frame), dtype=torch.float32)
    out = cotracker(
        frames_bthwc,
        query_points_prior=query,
        query_frame_ids=frame_ids,
        query_image_hw=image_hw,
    )
    return out.tracks[0].detach().float().cpu().numpy(), out.visibility[0].detach().float().cpu().numpy()


def _render_side_by_side(old_frames: np.ndarray, new_frames: np.ndarray, title: str) -> np.ndarray:
    rendered: list[np.ndarray] = []
    for old, new in zip(old_frames, new_frames):
        frame = np.concatenate([old, new], axis=1)
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 30), (0, 0, 0), thickness=-1)
        cv2.putText(frame, f"OLD: frame0 mask sampler                 NEW: temporal SAM2 + CoTracker repair | {title}", (12, 21), cv2.FONT_HERSHEY_SIMPLEX, 0.53, (255, 255, 255), 1, cv2.LINE_AA)
        rendered.append(frame)
    return np.stack(rendered, axis=0)


def _contact_sheet(frames: np.ndarray, path: Path) -> None:
    ids = np.linspace(0, len(frames) - 1, 9).round().astype(int).tolist()
    tiles = [cv2.resize(frames[idx], (448, 256), interpolation=cv2.INTER_AREA) for idx in ids]
    grid = np.concatenate([np.concatenate(tiles[row : row + 3], axis=1) for row in range(0, 9, 3)], axis=0)
    cv2.imwrite(str(path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def _load_case(path: Path) -> tuple[Path, str, str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    video = payload.get("source_video", payload.get("input_video"))
    caption = payload.get("input_caption", payload.get("caption", ""))
    if not isinstance(video, str) or not video:
        raise ValueError(f"missing source_video/input_video: {path}")
    return Path(video).expanduser().resolve(), str(caption), str(payload.get("sample_key", path.stem))


def _make_provider(device: str, points_per_object: int) -> ViewerGroundingBoxProvider:
    return ViewerGroundingBoxProvider(
        device=device,
        segment_len=8,
        max_objects=4,
        points_per_object=points_per_object,
        proposal_source="gdino_only",
        motion_score_ratio=0.15,
        text_prompt="box . cube . block . cylinder . capsule . sphere . ball .",
        extra_prompt_terms="",
        include_caption_terms=False,
        gdino_box_threshold=0.20,
        gdino_text_threshold=0.15,
        prompt_frame_mode="first",
        track_dedupe_iou_threshold=0.75,
        container_suppress_ratio_threshold=0.95,
        container_suppress_min_contained=2,
        container_suppress_min_area_ratio=1.5,
        container_suppress_small_iou_threshold=0.7,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--context-frames", type=int, default=20)
    parser.add_argument("--num-cases", type=int, default=6)
    parser.add_argument("--num-queries", type=int, default=8)
    parser.add_argument("--oversample-factor", type=int, default=4)
    parser.add_argument("--min-visible-ratio", type=float, default=0.60)
    parser.add_argument("--min-in-mask-ratio", type=float, default=0.60)
    parser.add_argument("--fps", type=int, default=12)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    inputs = sorted(path for path in args.input_dir.glob("*.json") if path.name != "manifest.json")
    if not inputs:
        raise RuntimeError(f"no input JSON files under {args.input_dir}")
    provider = _make_provider(args.device, int(args.num_queries))
    cotracker = CoTrackerAdapter(
        checkpoint_path="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth",
        num_queries=max(int(args.num_queries), int(args.num_queries) * int(args.oversample_factor)),
        device=args.device,
        input_hw=(384, 512),
        window_len=60,
    )
    results: list[dict[str, Any]] = []
    for input_path in inputs:
        if len(results) >= int(args.num_cases):
            break
        video_path, caption, key = _load_case(input_path)
        frames, source_indices = read_video_prefix(video_path, int(args.context_frames))
        if len(frames) < 2:
            continue
        context = preprocess_video_rgb_uint8(
            frames, (512, 896), resize_mode="cover_crop", cover_crop_hw=(512, 896)
        )
        frames_tchw = ((context.permute(1, 0, 2, 3).float() + 1.0) / 2.0).cpu().numpy()
        grounding = provider.build_sample(
            frames_tchw_01=frames_tchw,
            caption=caption,
            image_hw=(512, 896),
        )
        valid_slots = [idx for idx, value in enumerate(grounding.object_valid_mask) if value > 0.5]
        if len(valid_slots) < 2:
            continue
        frames_bthwc = ((context.unsqueeze(0).permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
        old_points = grounding.grouped_queries_px[valid_slots].reshape(-1, 2).astype(np.float32)
        old_owner = [slot for slot in valid_slots for _ in range(int(args.num_queries))]
        old_tracks, old_visibility = _run_cotracker(cotracker, frames_bthwc, old_points, grounding.prompt_frame_idx, (512, 896))
        new_points_parts: list[np.ndarray] = []
        new_tracks_parts: list[np.ndarray] = []
        new_visibility_parts: list[np.ndarray] = []
        new_owner: list[int] = []
        slot_reports: list[dict[str, Any]] = []
        for local_idx, slot_idx in enumerate(valid_slots):
            track: DetectedObjectTrack = grounding.object_tracks[slot_idx]
            old_slice = slice(local_idx * int(args.num_queries), (local_idx + 1) * int(args.num_queries))
            old_quality = _point_qualities(
                masks_thw=track.masks_thw,
                tracks_tk2=old_tracks[:, old_slice],
                visibility_tk=old_visibility[:, old_slice],
            )
            anchor = min(max(int(grounding.prompt_frame_idx), 0), int(track.masks_thw.shape[0]) - 1)
            component = _main_component(track.masks_thw[anchor], track.box_prompt_xyxy)
            if component is None:
                slot_reports.append({"slot": slot_idx, "phrase": track.phrase, "new_valid": False, "reason": "empty_anchor_mask", "old": [asdict(item) for item in old_quality]})
                continue
            candidates = sample_points_from_mask(component, int(args.num_queries) * int(args.oversample_factor), avoid_edges=True)
            if candidates.shape[0] < int(args.num_queries):
                slot_reports.append({"slot": slot_idx, "phrase": track.phrase, "new_valid": False, "reason": "insufficient_anchor_candidates", "old": [asdict(item) for item in old_quality]})
                continue
            candidate_tracks, candidate_visibility = _run_cotracker(cotracker, frames_bthwc, candidates, anchor, (512, 896))
            qualities = _point_qualities(
                masks_thw=track.masks_thw,
                tracks_tk2=candidate_tracks,
                visibility_tk=candidate_visibility,
            )
            selected = _select_diverse(
                qualities,
                count=int(args.num_queries),
                min_visible=float(args.min_visible_ratio),
                min_in_mask=float(args.min_in_mask_ratio),
            )
            report: dict[str, Any] = {
                "slot": slot_idx,
                "phrase": track.phrase,
                "old": [asdict(item) for item in old_quality],
                "candidate": [asdict(item) for item in qualities],
                "selected_ids": selected,
                "new_valid": bool(selected),
            }
            slot_reports.append(report)
            if not selected:
                continue
            new_points_parts.append(candidates[selected])
            new_tracks_parts.append(candidate_tracks[:, selected])
            new_visibility_parts.append(candidate_visibility[:, selected])
            new_owner.extend([slot_idx] * len(selected))
        if len(new_points_parts) == 0:
            continue
        new_points = np.concatenate(new_points_parts, axis=0)
        new_tracks = np.concatenate(new_tracks_parts, axis=1)
        new_visibility = np.concatenate(new_visibility_parts, axis=1)
        old_overlay = render_track_overlay(
            context_video=context,
            object_tracks=grounding.object_tracks,
            prompt_frame_idx=int(grounding.prompt_frame_idx),
            query_points_px_k2=old_points,
            query_owner=old_owner,
            tracks_tk2=old_tracks,
            visibility_tk=old_visibility,
            color_rgb=(0, 119, 182),
            prefix="old",
        )
        new_overlay = render_track_overlay(
            context_video=context,
            object_tracks=grounding.object_tracks,
            prompt_frame_idx=int(grounding.prompt_frame_idx),
            query_points_px_k2=new_points,
            query_owner=new_owner,
            tracks_tk2=new_tracks,
            visibility_tk=new_visibility,
            color_rgb=(213, 94, 0),
            prefix="new",
        )
        case_dir = output_dir / f"{len(results):02d}_{input_path.stem}"
        case_dir.mkdir(parents=True, exist_ok=True)
        combined = _render_side_by_side(old_overlay, new_overlay, key)
        write_mp4(case_dir / "old_vs_new_query_overlay.mp4", combined, fps=int(args.fps))
        _contact_sheet(combined, case_dir / "old_vs_new_query_overlay_grid.png")
        payload = {
            "input_json": str(input_path),
            "source_video": str(video_path),
            "caption": caption,
            "sample_key": key,
            "source_frame_indices": source_indices.tolist(),
            "grounding": grounding.debug,
            "old_valid_slots": valid_slots,
            "new_valid_slots": [item["slot"] for item in slot_reports if item["new_valid"]],
            "old_query_count": int(old_points.shape[0]),
            "new_query_count": int(new_points.shape[0]),
            "slot_reports": slot_reports,
            "overlay_video": "old_vs_new_query_overlay.mp4",
            "overlay_grid": "old_vs_new_query_overlay_grid.png",
        }
        (case_dir / "result.json").write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        payload["relative_dir"] = case_dir.name
        results.append(payload)
        print(json.dumps({"case": key, "old_slots": len(valid_slots), "new_slots": len(payload["new_valid_slots"])}, ensure_ascii=False))
    summary = {
        "scheme_old": "legacy SAM-mask frame-0 sampler with box fallback",
        "scheme_new": "main SAM component + 4x candidates + temporal SAM2/CoTracker gate + diversity selection; no box fallback",
        "num_cases": len(results),
        "results": results,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if len(results) < int(args.num_cases):
        raise RuntimeError(f"only found {len(results)} qualifying multi-slot cases")


if __name__ == "__main__":
    main()
