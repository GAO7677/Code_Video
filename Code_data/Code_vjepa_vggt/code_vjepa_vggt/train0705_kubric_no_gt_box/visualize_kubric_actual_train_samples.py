from __future__ import annotations

import argparse
import html
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from code_vjepa_vggt.adapters.cotracker_adapter import CoTrackerAdapter
from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (
    ViewerGroundingBoxProvider,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    inspect_kubric_train_forward_aux_overlay as inspectmod,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_context_only_no_gt_box_v_newtrain_kubric as trainmod,
)


def _parse_index_list(raw_value: str | None) -> list[int]:
    if raw_value is None or not str(raw_value).strip():
        return []
    return [int(item.strip()) for item in str(raw_value).split(",") if item.strip()]


def _sample_context_length(
    candidates: list[int] | range,
    *,
    context_length_sampling: str,
    rng: random.Random,
) -> int:
    values = [int(value) for value in candidates]
    if not values:
        raise ValueError("Context length candidates must be non-empty.")
    if context_length_sampling == "uniform":
        return rng.choice(values)
    if context_length_sampling == "short_biased":
        max_value = max(values)
        weights = [max_value - value + 1 for value in values]
        return rng.choices(values, weights=weights, k=1)[0]
    raise ValueError(f"Unsupported context_length_sampling={context_length_sampling!r}.")


def _sample_context_spec(
    *,
    total_frames: int,
    ctx_max_length: int,
    min_context_frames: int,
    context_length_sampling: str,
    no_context_ratio: float,
    rng: random.Random,
) -> dict[str, Any]:
    max_context_last_index = min(
        total_frames - 1,
        int(ctx_max_length),
    )
    min_context_last_index = max(int(min_context_frames), 0)
    if max_context_last_index < min_context_last_index:
        raise ValueError(
            "Context sampling range is empty. "
            f"Got total_frames={total_frames}, min_context_frames={min_context_frames}, "
            f"ctx_max_length={ctx_max_length}."
        )

    if rng.random() < no_context_ratio:
        return {
            "mode": "text_only",
            "frame_indices": [],
            "ctx_max_length": int(max_context_last_index),
            "sampled_ctx_last_index": -1,
            "sampled_ctx_num_frames": 0,
        }

    sampled_ctx_last_index = _sample_context_length(
        range(min_context_last_index, max_context_last_index + 1),
        context_length_sampling=context_length_sampling,
        rng=rng,
    )
    frame_indices = list(range(int(sampled_ctx_last_index) + 1))
    return {
        "mode": "prefix",
        "frame_indices": frame_indices,
        "ctx_max_length": int(max_context_last_index),
        "sampled_ctx_last_index": int(sampled_ctx_last_index),
        "sampled_ctx_num_frames": int(len(frame_indices)),
    }


def _build_grounding_provider(args: argparse.Namespace) -> ViewerGroundingBoxProvider:
    include_caption_terms = not bool(args.grounding_disable_caption_terms)
    return ViewerGroundingBoxProvider(
        device=str(args.grounding_device),
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


def _build_cotracker(args: argparse.Namespace) -> CoTrackerAdapter:
    return CoTrackerAdapter(
        checkpoint_path=str(args.cotracker_checkpoint),
        num_queries=int(args.aux_max_objects) * int(args.object_num_queries),
        device=str(args.cotracker_device),
        input_hw=(int(args.cotracker_input_h), int(args.cotracker_input_w)),
        window_len=int(args.cotracker_window_len),
    )


def _build_priors_from_grounding_sample(
    grounding_sample: Any,
    *,
    aux_max_objects: int,
    object_num_queries: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grouped_queries = torch.from_numpy(grounding_sample.grouped_queries_px).float()
    object_valid_mask = torch.from_numpy(grounding_sample.object_valid_mask).float()
    context_boxes_norm = torch.from_numpy(grounding_sample.context_boxes_norm).float()
    prompt_frame_idx = int(getattr(grounding_sample, "prompt_frame_idx", 0))

    flat = grouped_queries.view(1, int(aux_max_objects) * int(object_num_queries), 2)

    frame_ids: list[float] = []
    valid_frames = int(context_boxes_norm.shape[0])
    for object_idx in range(int(aux_max_objects)):
        is_valid = bool(object_valid_mask[object_idx].item() > 0.5)
        first_valid_frame = 0
        if is_valid:
            for frame_idx in range(valid_frames):
                candidate = context_boxes_norm[frame_idx, object_idx]
                if bool(
                    (candidate[2] - candidate[0] > 1.0e-6)
                    and (candidate[3] - candidate[1] > 1.0e-6)
                ):
                    first_valid_frame = frame_idx
                    break
            else:
                first_valid_frame = prompt_frame_idx
        frame_ids.extend([float(first_valid_frame)] * int(object_num_queries))

    frame_ids_tensor = torch.tensor(frame_ids, dtype=torch.float32).view(
        1,
        int(aux_max_objects) * int(object_num_queries),
        1,
    )
    return flat, frame_ids_tensor, object_valid_mask.view(1, int(aux_max_objects))


def _select_actual_context_sample(
    raw_sample: dict[str, Any],
    *,
    ctx_max_length: int,
    min_context_frames: int,
    context_length_sampling: str,
    no_context_ratio: float,
    rng: random.Random,
) -> tuple[dict[str, Any], dict[str, Any]]:
    raw_video = raw_sample["video"]
    total_frames = int(raw_video.shape[1])
    context_spec = _sample_context_spec(
        total_frames=total_frames,
        ctx_max_length=ctx_max_length,
        min_context_frames=min_context_frames,
        context_length_sampling=context_length_sampling,
        no_context_ratio=no_context_ratio,
        rng=rng,
    )
    sampled_indices = torch.tensor(context_spec["frame_indices"], dtype=torch.long)
    sample = dict(raw_sample)
    if int(sampled_indices.numel()) > 0:
        sample["context_video"] = raw_video[:, sampled_indices].contiguous()
    else:
        sample["context_video"] = raw_video[:, :0].contiguous()
    sample["context_frame_indices"] = sampled_indices
    sample["num_context_frames"] = int(sampled_indices.numel())
    sample["ctx_max_length"] = int(context_spec["ctx_max_length"])
    sample["sampled_ctx_last_index"] = int(context_spec["sampled_ctx_last_index"])
    sample["sampled_ctx_num_frames"] = int(context_spec["sampled_ctx_num_frames"])
    return sample, context_spec


def _build_case_page(case_dir: Path, result: dict[str, Any]) -> None:
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Kubric Actual Train Sample</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(360px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; background: #fff; border: 1px solid #ddd; padding: 12px; }}
    video, img {{ width: 100%; border: 1px solid #ccc; background: #000; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 6px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Kubric Actual Train Sample Review</h1>
  <p><b>Sample key:</b> {html.escape(str(result["sample_key"]))}</p>
  <p><b>Caption:</b> {html.escape(str(result["caption"]))}</p>
  <p><b>Video path:</b> {html.escape(str(result["video_path"]))}</p>
  <p><b>Context sampling mode:</b> {html.escape(str(result["context_sampling_mode"]))}</p>
  <p><b>ctx_max_length:</b> {int(result["ctx_max_length"])}</p>
  <p><b>sampled_ctx_last_index:</b> {int(result["sampled_ctx_last_index"])}</p>
  <p><b>sampled_ctx_num_frames:</b> {int(result["sampled_ctx_num_frames"])}</p>
  <p><b>sampled source frame indices:</b> {html.escape(str(result["context_source_frame_indices"]))}</p>
  <div class="grid">
    <figure>
      <video controls preload="none" playsinline src="{html.escape(result['train_clip_video_mp4'])}"></video>
      <figcaption>Sampled train clip full video ({result["num_frames"]} frames)</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(result['source_full_video_mp4'])}"></video>
      <figcaption>Original source rgba.mp4 full video</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(result['context_video_mp4'])}"></video>
      <figcaption>Actual context video used by this emulated train sample</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(result['prompt_preview_png'])}" />
      <figcaption>Viewer-grounding prompt boxes + sampled query points</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(result['input_overlay_video'])}"></video>
      <figcaption>Input pre-pipe overlay: boxes + query points + CoTracker tracks</figcaption>
    </figure>
  </div>
  <h2>Metadata</h2>
  <pre>{html.escape(json.dumps(result["metadata"], ensure_ascii=False, indent=2))}</pre>
</body>
</html>
"""
    (case_dir / "index.html").write_text(html_text, encoding="utf-8")


def _build_summary_page(output_dir: Path, results: list[dict[str, Any]]) -> None:
    sections: list[str] = []
    for result in results:
        case_dir = result["relative_dir"]
        sections.append(
            f"""
<section class="case-card">
  <div class="case-header">
    <div>
      <h2>{html.escape(result['sample_key'])}</h2>
      <p class="meta"><b>dataset_index:</b> {result['inspect_index']}</p>
      <p class="meta"><b>ctx:</b> last={int(result['sampled_ctx_last_index'])}, frames={int(result['sampled_ctx_num_frames'])}, max={int(result['ctx_max_length'])}</p>
      <p class="meta"><b>source_frame_indices:</b> {html.escape(str(result['context_source_frame_indices']))}</p>
      <p class="caption">{html.escape(result['caption'])}</p>
    </div>
    <div class="actions">
      <a href="{html.escape(case_dir)}/index.html">open case report</a>
    </div>
  </div>
  <div class="media-grid">
    <figure>
      <video controls preload="none" playsinline src="{html.escape(case_dir)}/{html.escape(result['train_clip_video_mp4'])}"></video>
      <figcaption>Sampled train clip full video</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(case_dir)}/{html.escape(result['source_full_video_mp4'])}"></video>
      <figcaption>Original source full video</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(case_dir)}/{html.escape(result['context_video_mp4'])}"></video>
      <figcaption>Actual sampled context video</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(case_dir)}/{html.escape(result['input_overlay_video'])}"></video>
      <figcaption>Box/query/track overlay</figcaption>
    </figure>
    <figure>
      <img src="{html.escape(case_dir)}/{html.escape(result['prompt_preview_png'])}" />
      <figcaption>Prompt preview</figcaption>
    </figure>
  </div>
  <details>
    <summary>Metadata</summary>
    <pre>{html.escape(json.dumps(result['metadata'], ensure_ascii=False, indent=2))}</pre>
  </details>
</section>
"""
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Kubric Actual Train Samples</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3efe6;
      --panel: #fffdf8;
      --line: #d9d0c2;
      --text: #1f1f1f;
      --muted: #5f5a53;
      --link: #0b5cad;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      padding: 24px;
      font-family: sans-serif;
      color: var(--text);
      background:
        radial-gradient(circle at top right, #efe6d3 0, transparent 24%),
        linear-gradient(180deg, #f6f1e7 0%, #f2ede3 100%);
    }}
    .page {{ max-width: 1800px; margin: 0 auto; }}
    h1 {{ margin: 0 0 8px; font-size: 32px; }}
    .intro {{ margin: 0 0 24px; color: var(--muted); line-height: 1.5; }}
    .case-list {{ display: grid; gap: 20px; }}
    .case-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 16px;
      padding: 18px;
      box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
    }}
    .case-header {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: flex-start;
      margin-bottom: 16px;
    }}
    .case-header h2 {{ margin: 0 0 8px; font-size: 22px; line-height: 1.25; }}
    .meta {{ margin: 4px 0; color: var(--muted); word-break: break-word; }}
    .caption {{ margin: 12px 0 0; line-height: 1.6; }}
    .actions a {{ color: var(--link); text-decoration: none; font-weight: 600; }}
    .actions a:hover {{ text-decoration: underline; }}
    .media-grid {{ display: grid; grid-template-columns: repeat(3, minmax(320px, 1fr)); gap: 16px; }}
    figure {{ margin: 0; border: 1px solid var(--line); border-radius: 12px; overflow: hidden; background: #ffffff; }}
    img, video {{ display: block; width: 100%; background: #000; }}
    figcaption {{ padding: 10px 12px; font-size: 13px; color: var(--muted); border-top: 1px solid var(--line); }}
    details {{ margin-top: 14px; border-top: 1px solid var(--line); padding-top: 14px; }}
    summary {{ cursor: pointer; font-weight: 600; }}
    pre {{ margin: 12px 0 0; padding: 14px; overflow-x: auto; border-radius: 10px; background: #faf7f0; border: 1px solid var(--line); white-space: pre-wrap; }}
    @media (max-width: 1300px) {{ .media-grid {{ grid-template-columns: 1fr; }} .case-header {{ flex-direction: column; }} }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Kubric Actual Train Sample Gallery</h1>
    <p class="intro">
      These cases are sampled from the current Kubric training dataset configuration.
      Each card shows the sampled 69-frame training clip, the original source video,
      the actual context prefix selected directly from the train clip under the current
      training context policy, and the corresponding box/query/track overlay.
      Total cases: {len(results)}.
    </p>
    <div class="case-list">
      {''.join(sections)}
    </div>
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize several actual Kubric train samples under the current "
            "train0705_kubric_no_gt_box context-sampling configuration."
        )
    )
    parser.add_argument("--kubric_root", type=str, default="/data/gaoya/dataset/nnsriram97-phyco_kubric")
    parser.add_argument("--kubric_split", type=str, default="train")
    parser.add_argument("--kubric_cache_root", type=str, default="/data/gaoya/agent-data/cache/kubric_no_gt_box_dataset")
    parser.add_argument("--kubric_sampling_strategy", type=str, default="prefix")
    parser.add_argument("--kubric_split_train_ratio", type=float, default=0.9)
    parser.add_argument("--kubric_split_val_ratio", type=float, default=0.05)
    parser.add_argument("--kubric_max_retry_samples", type=int, default=8)
    parser.add_argument("--kubric_init_scan_limit", type=int, default=0)
    parser.add_argument("--kubric_scenario", nargs="*", default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num_frames", type=int, default=69)
    parser.add_argument("--fixed_num_context_frames", type=int, default=20)
    parser.add_argument("--ctx_max_length", type=int, default=20)
    parser.add_argument("--min_context_frames", type=int, default=0)
    parser.add_argument(
        "--context_length_sampling",
        type=str,
        default="short_biased",
        choices=["uniform", "short_biased"],
    )
    parser.add_argument("--no_context_ratio", type=float, default=0.0)
    parser.add_argument("--object_num_queries", type=int, default=8)
    parser.add_argument("--aux_max_objects", type=int, default=4)
    parser.add_argument("--cotracker_checkpoint", type=str, default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth")
    parser.add_argument("--cotracker_input_h", type=int, default=384)
    parser.add_argument("--cotracker_input_w", type=int, default=512)
    parser.add_argument("--cotracker_window_len", type=int, default=60)
    parser.add_argument("--cotracker_device", type=str, default="cuda:0")
    parser.add_argument("--grounding_device", type=str, default="cuda:0")
    parser.add_argument("--sam2_segment_len", type=int, default=8)
    parser.add_argument("--grounding_proposal_source", type=str, default="gdino_only")
    parser.add_argument("--grounding_motion_score_ratio", type=float, default=0.15)
    parser.add_argument("--grounding_text_prompt", type=str, default="box . cube . block . cylinder . capsule . sphere . ball .")
    parser.add_argument("--grounding_extra_prompt_terms", type=str, default="")
    parser.add_argument("--grounding_disable_caption_terms", action="store_true", default=True)
    parser.add_argument("--grounding_gdino_box_threshold", type=float, default=0.20)
    parser.add_argument("--grounding_gdino_text_threshold", type=float, default=0.15)
    parser.add_argument("--grounding_prompt_frame_mode", type=str, default="first")
    parser.add_argument("--grounding_track_dedupe_iou_threshold", type=float, default=0.75)
    parser.add_argument("--grounding_container_suppress_ratio_threshold", type=float, default=0.95)
    parser.add_argument("--grounding_container_suppress_min_contained", type=int, default=2)
    parser.add_argument("--grounding_container_suppress_min_area_ratio", type=float, default=1.5)
    parser.add_argument("--grounding_container_suppress_small_iou_threshold", type=float, default=0.7)
    parser.add_argument("--inspect_num_samples", type=int, default=4)
    parser.add_argument("--inspect_indices", type=str, default=None)
    parser.add_argument("--inspect_seed", type=int, default=42)
    parser.add_argument("--inspect_skip_zero_context", action="store_true", default=True)
    parser.add_argument("--inspect_include_zero_context", dest="inspect_skip_zero_context", action="store_false")
    parser.add_argument("--inspect_fps", type=int, default=30)
    parser.add_argument(
        "--inspect_output_dir",
        type=str,
        default="/data/gaoya/agent-data/outputs/kubric_actual_train_samples_20260707",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.inspect_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_args = argparse.Namespace(
        dataset_type="kubric_no_gt_box",
        kubric_root=args.kubric_root,
        kubric_split=args.kubric_split,
        kubric_cache_root=args.kubric_cache_root,
        kubric_sampling_strategy=args.kubric_sampling_strategy,
        kubric_split_train_ratio=args.kubric_split_train_ratio,
        kubric_split_val_ratio=args.kubric_split_val_ratio,
        kubric_max_retry_samples=args.kubric_max_retry_samples,
        kubric_init_scan_limit=(
            None if int(args.kubric_init_scan_limit) <= 0 else int(args.kubric_init_scan_limit)
        ),
        kubric_scenario=args.kubric_scenario,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        fixed_num_context_frames=args.fixed_num_context_frames,
    )
    dataset = trainmod.build_dataset(dataset_args)

    grounding_provider = _build_grounding_provider(args)
    cotracker = _build_cotracker(args)

    explicit_indices = _parse_index_list(args.inspect_indices)
    if explicit_indices:
        candidate_indices = explicit_indices
    else:
        candidate_indices = list(range(len(dataset)))
        random.Random(int(args.inspect_seed)).shuffle(candidate_indices)

    results: list[dict[str, Any]] = []
    skipped_zero_context: list[int] = []
    for dataset_index in candidate_indices:
        if len(results) >= int(args.inspect_num_samples):
            break

        raw_sample = dataset[int(dataset_index)]
        rng = random.Random(int(args.inspect_seed) * 1000003 + int(dataset_index))
        sample, context_spec = _select_actual_context_sample(
            raw_sample,
            ctx_max_length=int(args.ctx_max_length),
            min_context_frames=int(args.min_context_frames),
            context_length_sampling=str(args.context_length_sampling),
            no_context_ratio=float(args.no_context_ratio),
            rng=rng,
        )
        num_context_frames = int(sample["num_context_frames"])
        if num_context_frames <= 0 and bool(args.inspect_skip_zero_context):
            skipped_zero_context.append(int(dataset_index))
            continue
        if num_context_frames <= 0:
            skipped_zero_context.append(int(dataset_index))
            continue

        case_dir = output_dir / f"sample_{int(dataset_index):06d}_ctx{num_context_frames:02d}"
        case_dir.mkdir(parents=True, exist_ok=True)

        context_video = sample["context_video"]
        image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
        frames_tchw_01 = ((context_video.permute(1, 0, 2, 3).float() + 1.0) / 2.0).clamp(0.0, 1.0)
        grounding_sample = grounding_provider.build_sample(
            frames_tchw_01=frames_tchw_01.cpu().numpy(),
            caption=str(sample["caption"]),
            image_hw=image_hw,
        )
        query_points_prior, query_frame_ids, object_valid_mask = _build_priors_from_grounding_sample(
            grounding_sample,
            aux_max_objects=int(args.aux_max_objects),
            object_num_queries=int(args.object_num_queries),
        )
        cotracker_out = cotracker(
            frames_tchw_01.unsqueeze(0).permute(0, 1, 3, 4, 2).contiguous(),
            query_points_prior=query_points_prior,
            query_frame_ids=query_frame_ids,
            query_image_hw=image_hw,
        )

        context_video_browser = inspectmod._write_tensor_video(
            case_dir / "context_video.mp4",
            context_video,
            fps=int(args.inspect_fps),
        )
        train_clip_video_browser = inspectmod._write_tensor_video(
            case_dir / "train_clip_full.mp4",
            raw_sample["video"],
            fps=int(args.inspect_fps),
        )
        source_full_video_browser = inspectmod._export_browser_video(
            Path(str(raw_sample["video_path"])),
            case_dir / "source_full_video.browser.mp4",
        )

        valid_queries = inspectmod._valid_query_count(object_valid_mask, int(args.object_num_queries))
        query_points_np = query_points_prior[0].detach().cpu().numpy()
        tracks_np = cotracker_out.tracks[0].detach().cpu().numpy()
        visibility_np = cotracker_out.visibility[0].detach().cpu().numpy()
        valid_queries_px = query_points_np[:valid_queries]
        valid_tracks = tracks_np[:, :valid_queries]
        valid_visibility = visibility_np[:, :valid_queries]
        query_owner = [
            obj_idx
            for obj_idx in range(int((object_valid_mask[0] > 0.5).sum().item()))
            for _ in range(int(args.object_num_queries))
        ]

        prompt_preview = inspectmod._render_prompt_preview(
            context_video=context_video,
            grounding_sample=grounding_sample,
            valid_queries_px=valid_queries_px,
            query_owner=query_owner,
        )
        prompt_preview_path = case_dir / "prompt_preview.png"
        inspectmod._write_rgb_png(prompt_preview_path, prompt_preview)

        input_overlay_video = inspectmod.render_track_overlay(
            context_video=context_video,
            object_tracks=getattr(grounding_sample, "object_tracks", []),
            prompt_frame_idx=int(getattr(grounding_sample, "prompt_frame_idx", 0)),
            query_points_px_k2=valid_queries_px.astype(np.float32),
            query_owner=query_owner,
            tracks_tk2=valid_tracks.astype(np.float32),
            visibility_tk=valid_visibility.astype(np.float32),
            color_rgb=inspectmod.INPUT_TRACK_COLOR,
            prefix="trk",
        )
        input_overlay_raw = case_dir / "input_prepipe_overlay.mp4"
        inspectmod.write_mp4(input_overlay_raw, input_overlay_video, fps=int(args.inspect_fps))
        input_overlay_browser = inspectmod._ensure_browser_video(input_overlay_raw)

        sampled_source_indices = list(raw_sample.get("metadata", {}).get("sampled_frame_indices", []))
        actual_context_local_indices = sample["context_frame_indices"].tolist()
        context_source_frame_indices = [
            int(sampled_source_indices[idx])
            for idx in actual_context_local_indices
            if 0 <= int(idx) < len(sampled_source_indices)
        ]

        result = {
            "inspect_index": int(dataset_index),
            "sample_key": str(sample.get("metadata", {}).get("sample_key", "")),
            "caption": str(sample["caption"]),
            "video_path": str(sample["video_path"]),
            "context_sampling_mode": str(context_spec["mode"]),
            "context_frame_indices": actual_context_local_indices,
            "context_source_frame_indices": context_source_frame_indices,
            "ctx_max_length": int(sample["ctx_max_length"]),
            "sampled_ctx_last_index": int(sample["sampled_ctx_last_index"]),
            "sampled_ctx_num_frames": int(sample["sampled_ctx_num_frames"]),
            "num_context_frames": int(sample["num_context_frames"]),
            "num_frames": int(raw_sample["video"].shape[1]),
            "train_clip_video_mp4": train_clip_video_browser.name,
            "source_full_video_mp4": source_full_video_browser.name,
            "context_video_mp4": context_video_browser.name,
            "prompt_preview_png": prompt_preview_path.name,
            "input_overlay_video": input_overlay_browser.name,
            "metadata": {
                "sample_key": str(sample.get("metadata", {}).get("sample_key", "")),
                "scenario": str(sample.get("metadata", {}).get("scenario", "")),
                "date": str(sample.get("metadata", {}).get("date", "")),
                "sample_id": str(sample.get("metadata", {}).get("sample_id", "")),
                "source_video_path": str(sample.get("metadata", {}).get("source_video_path", sample["video_path"])),
                "source_frame_count": sample.get("metadata", {}).get("source_frame_count"),
                "sampled_train_frame_indices": sampled_source_indices,
                "ctx_max_length": int(sample["ctx_max_length"]),
                "context_sampling_mode": str(context_spec["mode"]),
                "sampled_ctx_last_index": int(sample["sampled_ctx_last_index"]),
                "sampled_ctx_num_frames": int(sample["sampled_ctx_num_frames"]),
                "sampled_ctx_frame_indices": actual_context_local_indices,
                "sampled_ctx_source_indices": context_source_frame_indices,
                "object_valid_mask": object_valid_mask[0].detach().cpu().tolist(),
                "prompt_frame_idx": int(getattr(grounding_sample, "prompt_frame_idx", 0)),
            },
        }
        result["relative_dir"] = case_dir.name
        (case_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _build_case_page(case_dir, result)
        results.append(result)

    if not results:
        raise RuntimeError("no inspectable samples were generated")

    payload = {
        "output_dir": str(output_dir),
        "case_count": len(results),
        "skipped_zero_context_indices": skipped_zero_context,
        "results": results,
    }
    (output_dir / "results.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _build_summary_page(output_dir, results)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
