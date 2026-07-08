from __future__ import annotations

import argparse
import gc
import html
import json
import random
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    inspect_kubric_train_forward_aux_overlay as inspectmod,
)
from code_vjepa_vggt.train0705 import (
    infer_stage1b_context_only_no_gt_box_v_newtrain0705 as inferbase,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    visualize_kubric_actual_train_samples as actualmod,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_context_only_no_gt_box_v_newtrain_kubric as trainmod,
)


def _parse_index_list(raw_value: str | None) -> list[int]:
    if raw_value is None or not str(raw_value).strip():
        return []
    return [int(item.strip()) for item in str(raw_value).split(",") if item.strip()]


def _build_case_page(case_dir: Path, result: dict[str, Any]) -> None:
    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Kubric Actual Train Forward Aux Overlay</title>
  <style>
    :root {{
      color-scheme: light;
      --bg: #f3efe6;
      --panel: #fffdf8;
      --line: #d9d0c2;
      --text: #1f1f1f;
      --muted: #5f5a53;
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
    .grid {{ display: grid; grid-template-columns: repeat(3, minmax(320px, 1fr)); gap: 16px; }}
    figure {{
      margin: 0;
      border: 1px solid var(--line);
      border-radius: 12px;
      overflow: hidden;
      background: #fff;
    }}
    img, video {{ display: block; width: 100%; background: #000; }}
    figcaption {{ padding: 10px 12px; font-size: 13px; color: var(--muted); border-top: 1px solid var(--line); }}
    pre {{
      margin: 16px 0 0;
      padding: 14px;
      overflow-x: auto;
      border-radius: 10px;
      background: #faf7f0;
      border: 1px solid var(--line);
      white-space: pre-wrap;
    }}
    p {{ line-height: 1.55; }}
    @media (max-width: 1300px) {{ .grid {{ grid-template-columns: 1fr; }} }}
  </style>
</head>
<body>
  <div class="page">
    <h1>Kubric Actual Train Forward Aux Overlay</h1>
    <p><b>Sample key:</b> {html.escape(str(result["sample_key"]))}</p>
    <p><b>Dataset index:</b> {int(result["inspect_index"])}</p>
    <p><b>Caption:</b> {html.escape(str(result["caption"]))}</p>
    <p><b>Video path:</b> {html.escape(str(result["video_path"]))}</p>
    <p><b>Context frame indices:</b> {html.escape(str(result["context_frame_indices"]))}</p>
    <p><b>Mapped source frame indices:</b> {html.escape(str(result["context_source_frame_indices"]))}</p>
    <div class="grid">
      <figure>
        <video controls preload="none" playsinline src="{html.escape(result['train_clip_video_mp4'])}"></video>
        <figcaption>Sampled 69-frame train clip</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(result['source_full_video_mp4'])}"></video>
        <figcaption>Original source full video</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(result['context_pool_video_mp4'])}"></video>
        <figcaption>20-frame context pool from dataset</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(result['context_video_mp4'])}"></video>
        <figcaption>Actual sampled context video used for this forward</figcaption>
      </figure>
      <figure>
        <img src="{html.escape(result['prompt_preview_png'])}" />
        <figcaption>Prompt boxes plus sampled query points</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(result['input_overlay_video'])}"></video>
        <figcaption>Input overlay: boxes, query points, CoTracker tracks</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(result['box_overlay_video'])}"></video>
        <figcaption>Aux predicted boxes vs reference boxes</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(result['track_overlay_video'])}"></video>
        <figcaption>Aux predicted track summaries vs reference track summaries</figcaption>
      </figure>
    </div>
    <h2>Metrics</h2>
    <pre>{html.escape(json.dumps(result["metrics"], ensure_ascii=False, indent=2))}</pre>
    <h2>Metadata</h2>
    <pre>{html.escape(json.dumps(result["metadata"], ensure_ascii=False, indent=2))}</pre>
  </div>
</body>
</html>
"""
    (case_dir / "index.html").write_text(html_text, encoding="utf-8")


def _run_forward_debug(
    model: trainmod.ContextOnlyNoGTBoxWanModule,
    sample: dict[str, Any],
) -> dict[str, Any]:
    context_video = sample["context_video"].unsqueeze(0).to(
        device=model.pipe.device,
        dtype=model.pipe.torch_dtype,
    )
    image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
    query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = model._build_object_query_priors(
        sample,
        image_hw=image_hw,
    )
    query_points_prior = query_points_prior.to(device=model.pipe.device, dtype=model.pipe.torch_dtype)
    query_frame_ids = query_frame_ids.to(device=model.pipe.device, dtype=model.pipe.torch_dtype)
    object_valid_mask = object_valid_mask.to(device=model.pipe.device, dtype=model.pipe.torch_dtype)
    box_prior_xyxy = box_prior_xyxy.to(device=model.pipe.device, dtype=model.pipe.torch_dtype)

    frames_bthwc_01 = (
        (context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0
    ).clamp(0.0, 1.0)
    cotracker_out = model._run_cotracker(
        frames_bthwc_01,
        query_points_prior=query_points_prior,
        query_frame_ids=query_frame_ids,
        query_image_hw=image_hw,
    )

    if model.vggt_cache_root:
        vggt_out = trainmod.load_vggt_cache(sample, model.vggt_cache_root, allow_missing=True)
    elif getattr(model, "vggt_runner", None) is not None or getattr(model, "vggt_adapter", None) is not None:
        vggt_out = model._run_vggt(
            frames_bthwc_01,
            query_points_prior=query_points_prior,
            query_image_hw=image_hw,
        )
    else:
        vggt_out = None

    tracks_grouped, visibility_grouped, confidence_grouped = model._group_tracks_to_objects(
        cotracker_out.tracks,
        cotracker_out.visibility,
        cotracker_out.confidence,
        max_objects=model.aux_max_objects,
        points_per_object=model.object_num_queries,
    )
    context_latents = inferbase._encode_context_latents(model.pipe, sample["context_video"])
    jepa_input_video, _ = trainmod.prepare_jepa_context_video(
        context_video,
        latent_frames=int(context_latents.shape[2]),
        tubelet_size=int(getattr(model, "_jepa_tubelet_size", 2)),
    )
    jepa_out = model._run_jepa(jepa_input_video)
    object_out = model.object_pooler(
        jepa_patch_tokens=jepa_out.patch_tokens,
        context_latents=context_latents,
        tracks=tracks_grouped,
        visibility=visibility_grouped,
        confidence=confidence_grouped,
        track_image_hw=image_hw,
        object_valid_mask=object_valid_mask,
        box_prior_xyxy=box_prior_xyxy,
        vggt_world_points=getattr(vggt_out, "world_points", None),
        vggt_world_points_conf=getattr(vggt_out, "world_points_conf", None),
        vggt_depth=getattr(vggt_out, "depth", None),
        vggt_depth_conf=getattr(vggt_out, "depth_conf", None),
        vggt_dense_patch_tokens=getattr(vggt_out, "dense_patch_tokens", None),
        vggt_patch_grid_hw=getattr(vggt_out, "patch_grid_hw", None),
        vggt_geometry_image_hw=getattr(vggt_out, "input_hw", None)
        if getattr(vggt_out, "input_hw", None) is not None
        else getattr(vggt_out, "image_hw", None),
        frame_valid_mask=None,
    )
    object_aux_out = model.object_aux_heads(
        object_out.object_latent_tokens,
        object_out.active_track_summary,
        object_out.active_box_xyxy,
    )
    object_context = model.object_adapter(
        object_out.object_latent_tokens,
        object_valid_mask=object_valid_mask,
    )
    object_context_reg = object_context.square().mean()

    ref_box_xyxy = object_out.active_box_xyxy[0].detach().float().cpu().numpy()
    object_valid_mask_np = object_valid_mask[0].detach().float().cpu().numpy()
    latent_valid_mask = inspectmod._latent_valid_mask_from_boxes(ref_box_xyxy, object_valid_mask_np)
    valid_mask_t = torch.from_numpy(latent_valid_mask).to(device=object_out.active_box_xyxy.device)

    metrics = {
        "train/loss_object_context_reg": float(object_context_reg.detach().item()),
        "train/object_count": float(object_valid_mask.sum().item()),
        "aux/pred_box_l1_vs_ref": inspectmod._l1_on_mask(
            object_aux_out.pred_box_xyxy[0],
            object_out.active_box_xyxy[0],
            valid_mask_t,
        ),
        "aux/pred_track_l1_vs_ref": inspectmod._l1_on_mask(
            object_aux_out.pred_track_summary[0],
            object_out.active_track_summary[0, ..., :4],
            valid_mask_t,
        ),
    }
    return {
        "metrics": metrics,
        "query_points_prior": query_points_prior,
        "query_frame_ids": query_frame_ids,
        "object_valid_mask": object_valid_mask,
        "box_prior_xyxy": box_prior_xyxy,
        "cotracker_out": cotracker_out,
        "vggt_out": vggt_out,
        "tracks_grouped": tracks_grouped,
        "visibility_grouped": visibility_grouped,
        "confidence_grouped": confidence_grouped,
        "jepa_out": jepa_out,
        "object_out": object_out,
        "object_aux_out": object_aux_out,
        "object_context": object_context,
        "context_latents": context_latents,
        "latent_valid_mask": latent_valid_mask,
    }


def _offload_unused_pipe_modules(model: trainmod.ContextOnlyNoGTBoxWanModule) -> list[str]:
    unloaded: list[str] = []
    pipe = model.pipe
    for module_name in ("dit", "text_encoder"):
        module = getattr(pipe, module_name, None)
        if module is None:
            continue
        module.to("cpu")
        unloaded.append(module_name)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return unloaded


def _inspect_one(
    *,
    model: trainmod.ContextOnlyNoGTBoxWanModule,
    sample: dict[str, Any],
    output_dir: Path,
    inspect_fps: int,
    inspect_index: int,
) -> dict[str, Any]:
    with torch.no_grad():
        debug = _run_forward_debug(
            model,
            sample,
        )

    context_video = sample["context_video"]
    train_clip_video = sample["video"]
    image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
    grounding_sample = model.viewer_grounding.build_sample(
        frames_tchw_01=((context_video.permute(1, 0, 2, 3).float() + 1.0) / 2.0).cpu().numpy(),
        caption=str(sample["caption"]),
        image_hw=image_hw,
    )

    context_video_browser = inspectmod._write_tensor_video(
        output_dir / "context_video.mp4",
        context_video,
        fps=int(inspect_fps),
    )
    train_clip_video_browser = inspectmod._write_tensor_video(
        output_dir / "train_clip_full.mp4",
        train_clip_video,
        fps=int(inspect_fps),
    )
    source_full_video_browser = inspectmod._export_browser_video(
        Path(str(sample["video_path"])),
        output_dir / "source_full_video.browser.mp4",
    )

    valid_queries = inspectmod._valid_query_count(debug["object_valid_mask"], model.object_num_queries)
    valid_queries = max(valid_queries, 0)
    query_points_prior = debug["query_points_prior"][0].detach().float().cpu().numpy()
    cotracker_tracks = debug["cotracker_out"].tracks[0].detach().float().cpu().numpy()
    cotracker_visibility = debug["cotracker_out"].visibility[0].detach().float().cpu().numpy()
    valid_queries_px = query_points_prior[:valid_queries]
    valid_tracks = cotracker_tracks[:, :valid_queries]
    valid_visibility = cotracker_visibility[:, :valid_queries]
    query_owner = [
        obj_idx
        for obj_idx in range(int((debug["object_valid_mask"][0] > 0.5).sum().item()))
        for _ in range(int(model.object_num_queries))
    ]

    prompt_preview = inspectmod._render_prompt_preview(
        context_video=context_video,
        grounding_sample=grounding_sample,
        valid_queries_px=valid_queries_px,
        query_owner=query_owner,
    )
    prompt_preview_path = output_dir / "prompt_preview.png"
    inspectmod._write_rgb_png(prompt_preview_path, prompt_preview)

    input_overlay_video = inspectmod.render_track_overlay(
        context_video=context_video,
        object_tracks=getattr(grounding_sample, "object_tracks", []),
        prompt_frame_idx=int(getattr(grounding_sample, "prompt_frame_idx", 0)),
        query_points_px_k2=valid_queries_px.astype("float32"),
        query_owner=query_owner,
        tracks_tk2=valid_tracks.astype("float32"),
        visibility_tk=valid_visibility.astype("float32"),
        color_rgb=inspectmod.INPUT_TRACK_COLOR,
        prefix="trk",
    )
    input_overlay_raw = output_dir / "input_prepipe_overlay.mp4"
    inspectmod.write_mp4(input_overlay_raw, input_overlay_video, fps=int(inspect_fps))
    input_overlay_browser = inspectmod._ensure_browser_video(input_overlay_raw)

    ref_box_xyxy = debug["object_out"].active_box_xyxy[0].detach().float().cpu().numpy()
    pred_box_xyxy = debug["object_aux_out"].pred_box_xyxy[0].detach().float().cpu().numpy()
    ref_track_summary = debug["object_out"].active_track_summary[0, ..., :4].detach().float().cpu().numpy()
    pred_track_summary = debug["object_aux_out"].pred_track_summary[0].detach().float().cpu().numpy()
    latent_valid_mask = debug["latent_valid_mask"]

    box_overlay_video = inspectmod._render_ref_pred_box_overlay(
        context_video=context_video,
        ref_box_xyxy=ref_box_xyxy,
        pred_box_xyxy=pred_box_xyxy,
        valid_mask=latent_valid_mask,
        image_hw=image_hw,
    )
    box_overlay_raw = output_dir / "aux_pred_box_overlay.mp4"
    inspectmod.write_mp4(box_overlay_raw, box_overlay_video, fps=int(inspect_fps))
    box_overlay_browser = inspectmod._ensure_browser_video(box_overlay_raw)

    track_overlay_video = inspectmod._render_ref_pred_track_overlay(
        context_video=context_video,
        ref_track_summary=ref_track_summary,
        pred_track_summary=pred_track_summary,
        valid_mask=latent_valid_mask,
        image_hw=image_hw,
    )
    track_overlay_raw = output_dir / "aux_pred_track_overlay.mp4"
    inspectmod.write_mp4(track_overlay_raw, track_overlay_video, fps=int(inspect_fps))
    track_overlay_browser = inspectmod._ensure_browser_video(track_overlay_raw)

    return {
        "inspect_index": int(inspect_index),
        "caption": str(sample["caption"]),
        "video_path": str(sample["video_path"]),
        "sample_key": str(sample.get("metadata", {}).get("sample_key", "")),
        "context_frame_indices": sample["context_frame_indices"].tolist(),
        "num_context_frames": int(sample["num_context_frames"]),
        "prompt_preview_png": str(prompt_preview_path.name),
        "context_video_mp4": str(context_video_browser.name),
        "train_clip_video_mp4": str(train_clip_video_browser.name),
        "source_full_video_mp4": str(source_full_video_browser.name),
        "input_overlay_video": str(input_overlay_browser.name),
        "box_overlay_video": str(box_overlay_browser.name),
        "track_overlay_video": str(track_overlay_browser.name),
        "metrics": debug["metrics"],
    }


def _build_summary_page(output_dir: Path, results: list[dict[str, Any]], skipped_zero_context: list[int]) -> None:
    cards: list[str] = []
    for result in results:
        rel_dir = result["relative_dir"]
        cards.append(
            f"""
<section class="case-card">
  <div class="case-header">
    <div>
      <h2>{html.escape(result["sample_key"])}</h2>
      <p class="meta"><b>dataset_index:</b> {int(result["inspect_index"])}</p>
      <p class="meta"><b>context_frames:</b> {int(result["num_context_frames"])} ({html.escape(str(result["context_frame_indices"]))})</p>
      <p class="meta"><b>source_frame_indices:</b> {html.escape(str(result["context_source_frame_indices"]))}</p>
      <p class="caption">{html.escape(result["caption"])}</p>
    </div>
    <div class="actions">
      <a href="{html.escape(rel_dir)}/index.html">open case report</a>
    </div>
  </div>
  <div class="media-grid">
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(result['train_clip_video_mp4'])}"></video>
      <figcaption>Sampled train clip</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(result['source_full_video_mp4'])}"></video>
      <figcaption>Original source full video</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(result['context_video_mp4'])}"></video>
      <figcaption>Actual sampled context video</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(result['input_overlay_video'])}"></video>
      <figcaption>Input overlay</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(result['box_overlay_video'])}"></video>
      <figcaption>Aux box overlay</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(result['track_overlay_video'])}"></video>
      <figcaption>Aux track overlay</figcaption>
    </figure>
  </div>
  <details>
    <summary>Metrics</summary>
    <pre>{html.escape(json.dumps(result["metrics"], ensure_ascii=False, indent=2))}</pre>
  </details>
</section>
"""
        )

    html_text = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Kubric Actual Train Forward Aux Overlay Gallery</title>
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
    .intro {{ margin: 0 0 24px; color: var(--muted); line-height: 1.55; }}
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
    <h1>Kubric Actual Train Forward Aux Overlay</h1>
    <p class="intro">
      These cases use the current Kubric no-GT-box training dataset configuration and emulate the
      actual context-frame sampling policy before running a real object-branch forward.
      Each card includes the sampled train clip, the source full video, the actual sampled context,
      the input box/query/track overlay, and the aux box/track predictions.
      Total cases: {len(results)}.
      Skipped zero-context indices: {html.escape(str(skipped_zero_context))}.
    </p>
    <div class="case-list">
      {''.join(cards)}
    </div>
  </div>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html_text, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = trainmod.build_parser()
    parser.add_argument("--inspect_indices", type=str, default=None)
    parser.add_argument("--inspect_num_samples", type=int, default=4)
    parser.add_argument("--inspect_seed", type=int, default=42)
    parser.add_argument("--inspect_skip_zero_context", action="store_true", default=True)
    parser.add_argument(
        "--inspect_include_zero_context",
        dest="inspect_skip_zero_context",
        action="store_false",
    )
    parser.add_argument(
        "--inspect_output_dir",
        type=str,
        default="/data/gaoya/agent-data/outputs/kubric_actual_train_forward_aux_overlay_20260707",
    )
    parser.add_argument("--inspect_fps", type=int, default=30)
    return parser.parse_args()


def main() -> None:
    args = trainmod.tvn.prepare_args(parse_args())
    output_dir = Path(args.inspect_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    disabled_vggt_cache_root = output_dir / "_disabled_vggt_cache"
    disabled_vggt_cache_root.mkdir(parents=True, exist_ok=True)
    args.vggt_cache_root = str(disabled_vggt_cache_root)

    dataset = trainmod.build_dataset(args)
    accelerator = SimpleNamespace(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = trainmod.build_model(args, accelerator)
    target_device = torch.device(model.pipe.device)
    inspectmod._move_optional_module(model.object_pooler, target_device)
    inspectmod._move_optional_module(model.object_aux_heads, target_device)
    inspectmod._move_optional_module(model.object_adapter, target_device)
    inspectmod._move_optional_module(model.vggt_adapter, target_device)

    if args.stage1a_init_from is not None:
        trainmod.tvn._load_filtered_checkpoint_into_model(
            model,
            args.stage1a_init_from,
            include_prefixes=("object_pooler.", "object_aux_heads."),
        )
    inspectmod._load_optional_stage2_weights(model, args.stage2_resume_from)
    _offload_unused_pipe_modules(model)

    torch.nn.Module.train(model, False)

    explicit_indices = _parse_index_list(args.inspect_indices)
    if explicit_indices:
        candidate_indices = explicit_indices
    else:
        candidate_indices = list(range(len(dataset)))
        random.Random(int(args.inspect_seed)).shuffle(candidate_indices)

    results: list[dict[str, Any]] = []
    skipped_zero_context: list[int] = []
    context_frame_choices = actualmod._parse_context_frame_choices(getattr(args, "context_frame_choices", None))

    for dataset_index in candidate_indices:
        if len(results) >= int(args.inspect_num_samples):
            break
        raw_sample = dataset[int(dataset_index)]
        rng = random.Random(int(args.inspect_seed) * 1000003 + int(dataset_index))
        sample, context_spec = actualmod._select_actual_context_sample(
            raw_sample,
            min_context_frames=int(args.min_context_frames),
            max_context_ratio=float(args.max_context_ratio),
            context_frame_choices=context_frame_choices,
            no_context_ratio=float(args.no_context_ratio),
            rng=rng,
        )
        num_context_frames = int(sample.get("num_context_frames", 0))
        if num_context_frames <= 0 and bool(args.inspect_skip_zero_context):
            skipped_zero_context.append(int(dataset_index))
            continue

        sample_dir = output_dir / f"sample_{int(dataset_index):06d}_ctx{num_context_frames:02d}"
        sample_dir.mkdir(parents=True, exist_ok=True)

        result = _inspect_one(
            model=model,
            sample=sample,
            output_dir=sample_dir,
            inspect_fps=int(args.inspect_fps),
            inspect_index=int(dataset_index),
        )

        context_pool_video_browser = inspectmod._write_tensor_video(
            sample_dir / "context_pool.mp4",
            raw_sample["context_video"],
            fps=int(args.inspect_fps),
        )
        sampled_source_indices = list(raw_sample.get("metadata", {}).get("sampled_frame_indices", []))
        actual_context_local_indices = sample["context_frame_indices"].tolist()
        context_source_frame_indices = [
            int(sampled_source_indices[idx])
            for idx in actual_context_local_indices
            if 0 <= int(idx) < len(sampled_source_indices)
        ]

        result.update(
            {
                "context_sampling_mode": str(context_spec["mode"]),
                "context_pool_video_mp4": str(context_pool_video_browser.name),
                "context_source_frame_indices": context_source_frame_indices,
                "metadata": {
                    "sample_key": str(sample.get("metadata", {}).get("sample_key", "")),
                    "scenario": str(sample.get("metadata", {}).get("scenario", "")),
                    "date": str(sample.get("metadata", {}).get("date", "")),
                    "sample_id": str(sample.get("metadata", {}).get("sample_id", "")),
                    "source_video_path": str(sample.get("metadata", {}).get("source_video_path", sample["video_path"])),
                    "source_frame_count": sample.get("metadata", {}).get("source_frame_count"),
                    "sampled_train_frame_indices": sampled_source_indices,
                    "raw_context_pool_indices": raw_sample["context_frame_indices"].tolist(),
                    "context_sampling_mode": str(context_spec["mode"]),
                    "actual_context_local_indices": actual_context_local_indices,
                    "actual_context_source_indices": context_source_frame_indices,
                    "num_context_frames": num_context_frames,
                },
            }
        )
        result["relative_dir"] = sample_dir.name
        (sample_dir / "result.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        _build_case_page(sample_dir, result)
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
    _build_summary_page(output_dir, results, skipped_zero_context)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
