from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (
    ViewerGroundingBoxProvider,
)
from code_vjepa_vggt.train0705.inspect_stage1b_prepipe_overlay import (
    _build_cotracker_adapter,
    _build_viewer_grounding_provider,
    _context_tensor_to_uint8,
    _ensure_browser_video,
    _render_overlay_video,
    _save_prompt_frame_preview,
    _write_mp4,
)
from code_vjepa_vggt.utils.object_priors import build_vggt_query_prior
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix, read_video_uniform


DEFAULT_INPUT_JSON = Path(
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/"
    "physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper.json"
)
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/query_prior_compare_20260710")
DEFAULT_CASE_NAME = "physicIQ_026_no_box_fallback_check"


class NoBoxFallbackViewerGroundingProvider(ViewerGroundingBoxProvider):
    """Keep only objects with real SAM masks and mask-derived query points."""

    def _make_box_fallback_track(self, box_xyxy, score, phrase, height, width, frames):
        return None

    def _build_grouped_queries(self, tracks):
        grouped_queries = np.zeros((self.max_objects, self.points_per_object, 2), dtype=np.float32)
        object_valid_mask = np.zeros((self.max_objects,), dtype=np.float32)
        for obj_idx, track in enumerate(tracks[: self.max_objects]):
            pts, _ = build_vggt_query_prior(track.masks_thw, track.boxes_t4, num_queries=self.points_per_object)
            if pts.shape[0] <= 0:
                continue
            if pts.shape[0] < self.points_per_object:
                extra = pts[-1:].repeat(self.points_per_object - pts.shape[0], axis=0)
                pts = np.concatenate([pts, extra], axis=0)
            grouped_queries[obj_idx] = pts[: self.points_per_object].astype(np.float32)
            object_valid_mask[obj_idx] = 1.0
        return grouped_queries, object_valid_mask


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Temporary viewer-grounding test: disable box fallback and keep only "
            "objects with real SAM mask success."
        )
    )
    parser.add_argument("--input-json", type=Path, default=DEFAULT_INPUT_JSON)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--case-name", type=str, default=DEFAULT_CASE_NAME)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--context-frames", type=int, default=20)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--grounding-device", default="cuda:0")
    parser.add_argument("--aux-device", default="cuda:0")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--sam2-segment-len", type=int, default=8)
    parser.add_argument("--grounding-proposal-source", default="gdino_only")
    parser.add_argument("--grounding-motion-score-ratio", type=float, default=0.15)
    parser.add_argument(
        "--grounding-text-prompt",
        default="box . cube . block . cylinder . capsule . sphere . ball .",
    )
    parser.add_argument("--grounding-extra-prompt-terms", default="")
    parser.add_argument("--grounding-disable-caption-terms", action="store_true", default=True)
    parser.add_argument("--grounding-gdino-box-threshold", type=float, default=0.20)
    parser.add_argument("--grounding-gdino-text-threshold", type=float, default=0.15)
    parser.add_argument("--grounding-prompt-frame-mode", default="first")
    parser.add_argument("--grounding-track-dedupe-iou-threshold", type=float, default=0.75)
    parser.add_argument("--grounding-container-suppress-ratio-threshold", type=float, default=0.95)
    parser.add_argument("--grounding-container-suppress-min-contained", type=int, default=2)
    parser.add_argument("--grounding-container-suppress-min-area-ratio", type=float, default=1.5)
    parser.add_argument("--grounding-container-suppress-small-iou-threshold", type=float, default=0.7)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument("--object-num-queries", type=int, default=8)
    parser.add_argument("--cotracker-checkpoint", default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
    parser.add_argument("--cotracker-input-h", type=int, default=384)
    parser.add_argument("--cotracker-input-w", type=int, default=512)
    parser.add_argument("--cotracker-window-len", type=int, default=60)
    return parser.parse_args()


def _load_context_frames(args: argparse.Namespace, source_video: Path) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    if args.sampling_mode == "uniform":
        frames, frame_indices = read_video_uniform(source_video, int(args.context_frames))
    else:
        frames, frame_indices = read_video_prefix(source_video, int(args.context_frames))
    context_video_single = preprocess_video_rgb_uint8(
        frames,
        (int(args.height), int(args.width)),
        resize_mode="cover_crop",
        cover_crop_hw=(int(args.height), int(args.width)),
    )
    return frames, frame_indices, context_video_single


def _clone_provider_args(args: argparse.Namespace) -> argparse.Namespace:
    return argparse.Namespace(**vars(args))


def _build_no_fallback_provider(args: argparse.Namespace) -> NoBoxFallbackViewerGroundingProvider:
    include_caption_terms = not bool(args.grounding_disable_caption_terms)
    return NoBoxFallbackViewerGroundingProvider(
        device=str(args.grounding_device or "cpu"),
        segment_len=int(args.sam2_segment_len),
        max_objects=int(args.aux_max_objects),
        points_per_object=int(args.object_num_queries),
        proposal_source=str(args.grounding_proposal_source),
        motion_score_ratio=float(args.grounding_motion_score_ratio),
        text_prompt=str(args.grounding_text_prompt),
        extra_prompt_terms=str(args.grounding_extra_prompt_terms),
        include_caption_terms=include_caption_terms,
        gdino_box_threshold=float(args.grounding_gdino_box_threshold),
        gdino_text_threshold=float(args.grounding_gdino_text_threshold),
        prompt_frame_mode=str(args.grounding_prompt_frame_mode),
        track_dedupe_iou_threshold=float(args.grounding_track_dedupe_iou_threshold),
        container_suppress_ratio_threshold=float(args.grounding_container_suppress_ratio_threshold),
        container_suppress_min_contained=int(args.grounding_container_suppress_min_contained),
        container_suppress_min_area_ratio=float(args.grounding_container_suppress_min_area_ratio),
        container_suppress_small_iou_threshold=float(args.grounding_container_suppress_small_iou_threshold),
    )


def _run_provider(
    *,
    provider: ViewerGroundingBoxProvider,
    cotracker,
    context_video_single: torch.Tensor,
    prompt: str,
    output_dir: Path,
    fps: int,
    name: str,
) -> dict[str, object]:
    image_hw = (int(context_video_single.shape[-2]), int(context_video_single.shape[-1]))
    frames_tchw_01 = (
        ((context_video_single.permute(1, 0, 2, 3).float() + 1.0) / 2.0)
        .clamp(0.0, 1.0)
        .cpu()
        .numpy()
    )
    grounding_sample = provider.build_sample(
        frames_tchw_01=frames_tchw_01,
        caption=prompt,
        image_hw=image_hw,
    )

    grouped_queries = torch.from_numpy(grounding_sample.grouped_queries_px).float()
    query_points_prior = grouped_queries.view(1, int(provider.max_objects) * int(provider.points_per_object), 2)
    query_frame_ids = torch.full(
        (1, int(provider.max_objects) * int(provider.points_per_object), 1),
        float(int(getattr(grounding_sample, "prompt_frame_idx", 0))),
        dtype=torch.float32,
    )
    context_video = context_video_single.unsqueeze(0)
    frames_bthwc_01 = ((context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
    cotracker_device = cotracker.device_obj
    cotracker_out = cotracker(
        frames_bthwc_01.to(cotracker_device),
        query_points_prior=query_points_prior.to(cotracker_device),
        query_frame_ids=query_frame_ids.to(cotracker_device),
        query_image_hw=image_hw,
    )

    overlay_video = _render_overlay_video(
        context_frames=_context_tensor_to_uint8(context_video_single),
        prompt_frame_idx=int(grounding_sample.prompt_frame_idx),
        object_tracks=grounding_sample.object_tracks,
        grouped_queries_px=grounding_sample.grouped_queries_px.astype(np.float32),
        cotracker_tracks=cotracker_out.tracks[0].detach().cpu().numpy().astype(np.float32),
        cotracker_visibility=cotracker_out.visibility[0].detach().cpu().numpy().astype(np.float32),
    )
    overlay_raw = output_dir / f"{name}.mp4"
    _write_mp4(overlay_raw, overlay_video, fps=int(fps))
    overlay_browser = _ensure_browser_video(overlay_raw)

    prompt_preview = output_dir / f"{name}_prompt.png"
    _save_prompt_frame_preview(
        context_frames=_context_tensor_to_uint8(context_video_single),
        prompt_frame_idx=int(grounding_sample.prompt_frame_idx),
        object_tracks=grounding_sample.object_tracks,
        grouped_queries_px=grounding_sample.grouped_queries_px.astype(np.float32),
        output_path=prompt_preview,
    )

    return {
        "name": name,
        "object_count": int(grounding_sample.object_valid_mask.sum()),
        "prompt_frame_idx": int(grounding_sample.prompt_frame_idx),
        "prior_source": str(grounding_sample.prior_source),
        "grounding_debug": grounding_sample.debug,
        "object_phrases": [str(track.phrase) for track in grounding_sample.object_tracks],
        "overlay_video": str(overlay_browser),
        "prompt_preview": str(prompt_preview),
        "query_points_shape": list(grounding_sample.grouped_queries_px.shape),
        "visibility_mean": float(cotracker_out.visibility[0].detach().float().mean().item()),
        "confidence_mean": float(cotracker_out.confidence[0].detach().float().mean().item()),
    }


def main() -> None:
    args = parse_args()
    payload = core._load_input_json(args.input_json)
    source_video = Path(str(payload["source_video"])).expanduser().resolve()
    prompt = core._ensure_str_field(payload, "input_caption", args.input_json)

    output_dir = (args.output_root / args.case_name).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    _, frame_indices, context_video_single = _load_context_frames(args, source_video)
    default_provider = _build_viewer_grounding_provider(_clone_provider_args(args))
    no_fallback_provider = _build_no_fallback_provider(args)
    cotracker = _build_cotracker_adapter(args)

    default_result = _run_provider(
        provider=default_provider,
        cotracker=cotracker,
        context_video_single=context_video_single,
        prompt=prompt,
        output_dir=output_dir,
        fps=int(args.fps),
        name="default_viewer_grounding",
    )
    no_fallback_result = _run_provider(
        provider=no_fallback_provider,
        cotracker=cotracker,
        context_video_single=context_video_single,
        prompt=prompt,
        output_dir=output_dir,
        fps=int(args.fps),
        name="no_box_fallback",
    )

    summary = {
        "input_json": str(args.input_json),
        "source_video": str(source_video),
        "prompt": prompt,
        "frame_indices": frame_indices.tolist(),
        "sampling_mode": str(args.sampling_mode),
        "context_frames": int(args.context_frames),
        "default_viewer_grounding": default_result,
        "no_box_fallback": no_fallback_result,
        "comparison": {
            "default_object_count": int(default_result["object_count"]),
            "no_box_fallback_object_count": int(no_fallback_result["object_count"]),
            "dropped_object_count": int(default_result["object_count"]) - int(no_fallback_result["object_count"]),
        },
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "summary": str(summary_path), **summary["comparison"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
