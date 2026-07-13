"""Full Kubric v2v inference with legacy or no-GT temporal SAM2 query sampling."""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from code_vjepa_vggt.inspect_cotracker_vggt_geometry import render_track_overlay, write_mp4
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    infer_stage1b_context_only_no_gt_box_v_newtrain_kubric as kubric_infer,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_context_only_no_gt_box_v_newtrain_kubric as trainmod,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as batch,
)
from code_vjepa_vggt.train0710querypoints.sam2_temporal_query_repair import (
    TemporalQueryRepairConfig,
    repair_grouped_queries_with_sam2_tracks,
)


_SCHEME = "legacy"
_ACTIVE_CASE_DIR: Path | None = None
_ACTIVE_CASE_STEM: str | None = None
_ACTIVE_CASE_DEBUG: dict[str, Any] = {}
_ORIGINAL_RUN_SINGLE = batch._run_single_case_in_process
_SLOT_COLORS_RGB = (
    (214, 40, 40),
    (247, 127, 0),
    (42, 157, 143),
    (39, 125, 161),
)


def _make_browser_compatible_mp4(path: Path) -> None:
    """Replace OpenCV's mp4v output with a broadly playable H.264 MP4."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        conda_ffmpeg = Path(sys.executable).resolve().parent / "ffmpeg"
        if conda_ffmpeg.is_file():
            ffmpeg = str(conda_ffmpeg)
    if ffmpeg is None:
        raise RuntimeError(f"ffmpeg is required to encode browser-compatible overlay: {path}")
    temporary_path = path.with_name(f"{path.stem}.transcoding.mp4")
    try:
        subprocess.run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(path),
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(temporary_path),
            ],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _contact_sheet(frames: np.ndarray, output_path: Path) -> None:
    indices = np.linspace(0, len(frames) - 1, 9).round().astype(int).tolist()
    tiles = [cv2.resize(frames[index], (448, 256), interpolation=cv2.INTER_AREA) for index in indices]
    grid = np.concatenate([np.concatenate(tiles[row : row + 3], axis=1) for row in range(0, 9, 3)], axis=0)
    cv2.imwrite(str(output_path), cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))


def _write_actual_overlay(
    *,
    context_video_single: torch.Tensor,
    grounding_sample: Any,
    final_query_points: torch.Tensor,
    cotracker_out: Any,
) -> dict[str, Any] | None:
    if _ACTIVE_CASE_DIR is None:
        return None
    valid_slots = [
        slot_idx
        for slot_idx, value in enumerate(np.asarray(grounding_sample.object_valid_mask))
        if value > 0.5
    ]
    if not valid_slots:
        return None
    points_per_object = int(final_query_points.shape[1] // len(grounding_sample.object_valid_mask))
    query_ids = np.asarray(
        [slot_idx * points_per_object + query_idx for slot_idx in valid_slots for query_idx in range(points_per_object)],
        dtype=np.int64,
    )
    points = final_query_points[0, query_ids].detach().float().cpu().numpy()
    tracks = cotracker_out.tracks[0, :, query_ids].detach().float().cpu().numpy()
    visibility = cotracker_out.visibility[0, :, query_ids].detach().float().cpu().numpy()
    owners = [slot_idx for slot_idx in valid_slots for _ in range(points_per_object)]
    color = (0, 119, 182) if _SCHEME == "legacy" else (213, 94, 0)
    overlay = render_track_overlay(
        context_video=context_video_single,
        object_tracks=grounding_sample.object_tracks,
        prompt_frame_idx=int(grounding_sample.prompt_frame_idx),
        query_points_px_k2=points,
        query_owner=owners,
        tracks_tk2=tracks,
        visibility_tk=visibility,
        color_rgb=color,
        prefix=_SCHEME,
    )
    slot_labels = []
    for slot_idx in valid_slots:
        track = grounding_sample.object_tracks[slot_idx]
        color_rgb = np.asarray(
            _SLOT_COLORS_RGB[slot_idx % len(_SLOT_COLORS_RGB)], dtype=np.float32
        )
        masks = np.asarray(track.masks_thw) > 0
        for frame_idx in range(min(len(overlay), len(masks))):
            mask = masks[frame_idx]
            if mask.shape != overlay[frame_idx].shape[:2] or not bool(mask.any()):
                continue
            pixels = overlay[frame_idx][mask].astype(np.float32)
            overlay[frame_idx][mask] = np.clip(
                pixels * 0.78 + color_rgb * 0.22, 0, 255
            ).astype(np.uint8)
        phrase = str(getattr(track, "phrase", "object")).strip() or "object"
        slot_labels.append(f"s{slot_idx}:{phrase}")
    slot_summary = ", ".join(slot_labels)
    for frame in overlay:
        cv2.rectangle(frame, (0, 0), (frame.shape[1], 48), (0, 0, 0), thickness=-1)
        cv2.putText(
            frame,
            f"query scheme: {_SCHEME} | valid slots: {len(valid_slots)}",
            (10, 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.54,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            slot_summary[:120],
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    _ACTIVE_CASE_DIR.mkdir(parents=True, exist_ok=True)
    stem = _ACTIVE_CASE_STEM or "case"
    overlay_path = _ACTIVE_CASE_DIR / f"{stem}_input_prepipe_overlay.mp4"
    grid_path = _ACTIVE_CASE_DIR / f"{stem}_input_prepipe_overlay_grid.png"
    write_mp4(overlay_path, overlay, fps=12)
    _make_browser_compatible_mp4(overlay_path)
    _contact_sheet(overlay, grid_path)
    return {
        "overlay_video": str(overlay_path),
        "overlay_grid": str(grid_path),
        "valid_slot_count": len(valid_slots),
        "valid_slot_ids": valid_slots,
    }


def _slot_reports(grounding_sample: Any) -> list[dict[str, Any]]:
    reports = []
    valid_mask = np.asarray(grounding_sample.object_valid_mask)
    for slot_idx, track in enumerate(grounding_sample.object_tracks):
        masks = np.asarray(track.masks_thw) > 0
        reports.append(
            {
                "slot_id": int(slot_idx),
                "valid": bool(slot_idx < len(valid_mask) and valid_mask[slot_idx] > 0.5),
                "phrase": str(getattr(track, "phrase", "")),
                "score": float(getattr(track, "score", 0.0)),
                "prompt_box_xyxy": np.asarray(track.box_prompt_xyxy).astype(float).tolist(),
                "mask_area_pixels_per_frame": [int(mask.sum()) for mask in masks],
            }
        )
    return reports


def _build_object_context_with_query_scheme(
    model,
    *,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: str,
):
    original_build_sample = model.viewer_grounding.build_sample
    original_run_cotracker = model._run_cotracker
    captured: dict[str, Any] = {}

    def build_sample_with_scheme(*args, **kwargs):
        grounding = original_build_sample(*args, **kwargs)
        captured["legacy_queries"] = np.asarray(grounding.grouped_queries_px, dtype=np.float32).copy()
        captured["legacy_valid_mask"] = np.asarray(grounding.object_valid_mask, dtype=np.float32).copy()
        if _SCHEME == "temporal_sam2":
            frames_tchw = np.asarray(kwargs["frames_tchw_01"], dtype=np.float32)
            frames_bthwc = torch.from_numpy(frames_tchw).permute(0, 2, 3, 1).unsqueeze(0).to(
                device=torch.device(model.pipe.device), dtype=torch.float32
            )
            repaired, repaired_valid, repair_debug = repair_grouped_queries_with_sam2_tracks(
                image_hw=tuple(int(value) for value in kwargs["image_hw"]),
                frames_bthwc_01=frames_bthwc,
                grouped_queries_px=grounding.grouped_queries_px,
                object_valid_mask=grounding.object_valid_mask,
                object_tracks=grounding.object_tracks,
                prompt_frame_idx=int(grounding.prompt_frame_idx),
                points_per_object=int(model.object_num_queries),
                run_cotracker=original_run_cotracker,
                config=TemporalQueryRepairConfig(),
            )
            grounding.grouped_queries_px = repaired
            grounding.object_valid_mask = repaired_valid
            grounding.debug["temporal_sam2_query_repair"] = repair_debug
            captured["repair_debug"] = repair_debug
        captured["grounding"] = grounding
        return grounding

    def capture_final_cotracker(*args, **kwargs):
        output = original_run_cotracker(*args, **kwargs)
        captured["final_query_points"] = kwargs["query_points_prior"].detach().clone()
        captured["final_cotracker_out"] = output
        return output

    model.viewer_grounding.build_sample = build_sample_with_scheme
    model._run_cotracker = capture_final_cotracker
    try:
        object_context, object_debug = kubric_infer._build_object_context(
            model=model,
            context_video_single=context_video_single,
            prompt=prompt,
            video_path=video_path,
        )
    finally:
        model.viewer_grounding.build_sample = original_build_sample
        model._run_cotracker = original_run_cotracker

    overlay_paths = None
    if {"grounding", "final_query_points", "final_cotracker_out"}.issubset(captured):
        overlay_paths = _write_actual_overlay(
            context_video_single=context_video_single,
            grounding_sample=captured["grounding"],
            final_query_points=captured["final_query_points"],
            cotracker_out=captured["final_cotracker_out"],
        )
    scheme_debug = {
        "scheme": _SCHEME,
        "legacy_valid_slots": [
            int(index) for index, value in enumerate(captured.get("legacy_valid_mask", [])) if value > 0.5
        ],
        "final_valid_slots": [
            int(index)
            for index, value in enumerate(
                np.asarray(captured["grounding"].object_valid_mask)
                if "grounding" in captured
                else []
            )
            if value > 0.5
        ],
        "slot_reports": (
            _slot_reports(captured["grounding"]) if "grounding" in captured else []
        ),
        "temporal_sam2_query_repair": captured.get("repair_debug"),
        "actual_overlay": overlay_paths,
    }
    object_debug["query_scheme"] = scheme_debug
    _ACTIVE_CASE_DEBUG.clear()
    _ACTIVE_CASE_DEBUG.update(scheme_debug)
    return object_context, object_debug


def _run_single_case_with_query_scheme(**kwargs):
    global _ACTIVE_CASE_DIR, _ACTIVE_CASE_STEM
    output_video = Path(kwargs["output_video"])
    _ACTIVE_CASE_DIR = output_video.parent
    _ACTIVE_CASE_STEM = output_video.stem
    try:
        result, logs = _ORIGINAL_RUN_SINGLE(**kwargs)
        result["stage1a_query_scheme"] = _SCHEME
        result["query_scheme"] = dict(_ACTIVE_CASE_DEBUG)
        return result, logs
    finally:
        _ACTIVE_CASE_DIR = None
        _ACTIVE_CASE_STEM = None
        _ACTIVE_CASE_DEBUG.clear()


def _install_query_scheme_hooks() -> None:
    batch.infer0705.t0705 = trainmod
    batch.infer0705._build_object_context = _build_object_context_with_query_scheme
    batch.infer0705._build_model_args = kubric_infer._build_model_args


def _parse_query_scheme() -> None:
    global _SCHEME
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--query-scheme", choices=["legacy", "temporal_sam2"], default="legacy")
    known, remaining = parser.parse_known_args(sys.argv[1:])
    _SCHEME = str(known.query_scheme)
    sys.argv = [sys.argv[0], *remaining]


def main() -> None:
    _parse_query_scheme()
    batch._install_kubric_runtime_hooks = _install_query_scheme_hooks
    batch._run_single_case_in_process = _run_single_case_with_query_scheme
    batch.main()


if __name__ == "__main__":
    main()
