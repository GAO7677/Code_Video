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
import torch.nn.functional as F

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


def _resize_video_bthwc(
    frames_bthwc_01: torch.Tensor,
    dst_hw: tuple[int, int],
    *,
    align_corners: bool,
) -> torch.Tensor:
    b, t, h, w, c = frames_bthwc_01.shape
    frames_bchw = frames_bthwc_01.permute(0, 1, 4, 2, 3).reshape(b * t, c, h, w)
    resized = F.interpolate(
        frames_bchw,
        size=tuple(int(v) for v in dst_hw),
        mode="bilinear",
        align_corners=align_corners,
    )
    return resized.reshape(b, t, c, int(dst_hw[0]), int(dst_hw[1])).permute(0, 1, 3, 4, 2).contiguous()


def _write_frames_bthwc_video(path: Path, frames_bthwc_01: torch.Tensor, fps: int) -> Path:
    if int(frames_bthwc_01.shape[0]) != 1:
        raise ValueError(f"expected batch size 1 for visualization, got {list(frames_bthwc_01.shape)}")
    video_cthw = (frames_bthwc_01[0].permute(3, 0, 1, 2).contiguous() * 2.0 - 1.0).clamp(-1.0, 1.0)
    return inspectmod._write_tensor_video(path, video_cthw, fps=fps)


def _select_video_frames(video_cthw: torch.Tensor, frame_indices: list[int]) -> torch.Tensor:
    if not frame_indices:
        return video_cthw[:, :0].contiguous()
    return video_cthw[:, torch.tensor(frame_indices, dtype=torch.long)].contiguous()


def _expand_with_last_index(base_indices: list[int], target_len: int) -> list[int]:
    indices = [int(v) for v in base_indices]
    if not indices:
        return []
    if len(indices) >= int(target_len):
        return indices[: int(target_len)]
    last_value = int(indices[-1])
    return indices + [last_value] * (int(target_len) - len(indices))


def _source_indices_from_time_idx(base_indices: list[int], time_idx: list[int]) -> list[int]:
    out: list[int] = []
    for idx in time_idx:
        idx_int = int(idx)
        if 0 <= idx_int < len(base_indices):
            out.append(int(base_indices[idx_int]))
    return out


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
    <p><b>ctx_max_length:</b> {int(result["ctx_max_length"])}</p>
    <p><b>sampled_ctx_last_index:</b> {int(result["sampled_ctx_last_index"])}</p>
    <p><b>sampled_ctx_num_frames:</b> {int(result["sampled_ctx_num_frames"])}</p>
    <p><b>sampled source frame indices:</b> {html.escape(str(result["context_source_frame_indices"]))}</p>
    <p><b>jepa_time_idx:</b> {html.escape(str(result["jepa_time_idx"]))}</p>
    <p><b>latent_time_idx:</b> {html.escape(str(result["latent_time_idx"]))}</p>
    <p><b>jepa_time_source_indices:</b> {html.escape(str(result["jepa_time_source_indices"]))}</p>
    <p><b>latent_input_source_indices:</b> {html.escape(str(result["latent_input_source_indices"]))}</p>
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
        <video controls preload="none" playsinline src="{html.escape(result['context_video_mp4'])}"></video>
        <figcaption>Actual sampled context video used for this forward</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(result['cotracker_input_video_mp4'])}"></video>
        <figcaption>Actual CoTracker input video: resized to 384x512</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(result['vggt_input_video_mp4'])}"></video>
        <figcaption>Actual VGGT input video: resized to 420x728</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(result['jepa_padded_video_mp4'])}"></video>
        <figcaption>JEPA time-padded context video before spatial resize</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(result['jepa_input_video_mp4'])}"></video>
        <figcaption>Actual JEPA input video: time-padded and resized to 384x384</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(result['jepa_time_aligned_context_video_mp4'])}"></video>
        <figcaption>Context frames picked by jepa_time_idx on the native context video</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{html.escape(result['latent_time_aligned_context_video_mp4'])}"></video>
        <figcaption>Context frames picked by latent_time_idx on the native context video</figcaption>
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
    jepa_input_video, jepa_ctx_fix = trainmod.prepare_jepa_context_video(
        context_video,
        latent_frames=int(context_latents.shape[2]),
        tubelet_size=int(getattr(model, "_jepa_tubelet_size", 2)),
    )
    jepa_out = model._run_jepa(jepa_input_video)
    src_frames = int(tracks_grouped.shape[1])
    latent_frames = int(context_latents.shape[2])
    jepa_time_idx = model.object_pooler._time_indices(
        src_frames,
        int(jepa_out.patch_tokens.shape[1]),
        tracks_grouped.device,
    )
    latent_time_idx = model.object_pooler._time_indices(
        src_frames,
        latent_frames,
        tracks_grouped.device,
    )
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
        "frames_bthwc_01": frames_bthwc_01,
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
        "jepa_input_video": jepa_input_video,
        "jepa_ctx_fix": jepa_ctx_fix,
        "jepa_time_idx": jepa_time_idx,
        "latent_time_idx": latent_time_idx,
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
    cotracker_input_video = _resize_video_bthwc(
        debug["frames_bthwc_01"],
        tuple(int(v) for v in debug["cotracker_out"].input_hw),
        align_corners=True,
    )
    cotracker_input_video_browser = _write_frames_bthwc_video(
        output_dir / "cotracker_input_video.mp4",
        cotracker_input_video,
        fps=int(inspect_fps),
    )
    vggt_input_hw = (420, 728)
    if getattr(model, "vggt_adapter", None) is not None and getattr(model.vggt_adapter, "input_hw", None) is not None:
        vggt_input_hw = tuple(int(v) for v in model.vggt_adapter.input_hw)
    vggt_input_video = _resize_video_bthwc(
        debug["frames_bthwc_01"],
        vggt_input_hw,
        align_corners=False,
    )
    vggt_input_video_browser = _write_frames_bthwc_video(
        output_dir / "vggt_input_video.mp4",
        vggt_input_video,
        fps=int(inspect_fps),
    )
    jepa_padded_video_browser = inspectmod._write_tensor_video(
        output_dir / "jepa_padded_context_video.mp4",
        debug["jepa_input_video"][0].detach().float().cpu(),
        fps=int(inspect_fps),
    )
    jepa_crop_size = int(getattr(getattr(model, "jepa_adapter", None), "crop_size", 384))
    jepa_input_video_bthwc_01 = (
        (debug["jepa_input_video"].permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0
    ).clamp(0.0, 1.0)
    jepa_resized_input_video = _resize_video_bthwc(
        jepa_input_video_bthwc_01,
        (jepa_crop_size, jepa_crop_size),
        align_corners=False,
    )
    jepa_input_video_browser = _write_frames_bthwc_video(
        output_dir / "jepa_input_video.mp4",
        jepa_resized_input_video,
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

    jepa_time_idx = [int(v) for v in debug["jepa_time_idx"].detach().cpu().tolist()]
    latent_time_idx = [int(v) for v in debug["latent_time_idx"].detach().cpu().tolist()]
    jepa_time_aligned_context = _select_video_frames(context_video, jepa_time_idx)
    latent_time_aligned_context = _select_video_frames(context_video, latent_time_idx)
    jepa_time_aligned_context_browser = inspectmod._write_tensor_video(
        output_dir / "jepa_time_aligned_context.mp4",
        jepa_time_aligned_context,
        fps=int(inspect_fps),
    )
    latent_time_aligned_context_browser = inspectmod._write_tensor_video(
        output_dir / "latent_time_aligned_context.mp4",
        latent_time_aligned_context,
        fps=int(inspect_fps),
    )

    return {
        "inspect_index": int(inspect_index),
        "caption": str(sample["caption"]),
        "video_path": str(sample["video_path"]),
        "sample_key": str(sample.get("metadata", {}).get("sample_key", "")),
        "context_frame_indices": sample["context_frame_indices"].tolist(),
        "num_context_frames": int(sample["num_context_frames"]),
        "prompt_preview_png": str(prompt_preview_path.name),
        "context_video_mp4": str(context_video_browser.name),
        "cotracker_input_video_mp4": str(cotracker_input_video_browser.name),
        "vggt_input_video_mp4": str(vggt_input_video_browser.name),
        "jepa_padded_video_mp4": str(jepa_padded_video_browser.name),
        "jepa_input_video_mp4": str(jepa_input_video_browser.name),
        "jepa_time_aligned_context_video_mp4": str(jepa_time_aligned_context_browser.name),
        "latent_time_aligned_context_video_mp4": str(latent_time_aligned_context_browser.name),
        "train_clip_video_mp4": str(train_clip_video_browser.name),
        "source_full_video_mp4": str(source_full_video_browser.name),
        "input_overlay_video": str(input_overlay_browser.name),
        "box_overlay_video": str(box_overlay_browser.name),
        "track_overlay_video": str(track_overlay_browser.name),
        "jepa_time_idx": jepa_time_idx,
        "latent_time_idx": latent_time_idx,
        "jepa_input_frames": int(debug["jepa_input_video"].shape[2]),
        "jepa_padding_frames": int(debug["jepa_ctx_fix"].get("padded_context_frames", 0)),
        "latent_frames": int(debug["context_latents"].shape[2]),
        "jepa_token_frames": int(debug["jepa_out"].patch_tokens.shape[1]),
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
          <p class="meta"><b>ctx:</b> last={int(result["sampled_ctx_last_index"])}, frames={int(result["sampled_ctx_num_frames"])}, max={int(result["ctx_max_length"])}</p>
          <p class="meta"><b>source_frame_indices:</b> {html.escape(str(result["context_source_frame_indices"]))}</p>
          <p class="meta"><b>jepa_time_idx:</b> {html.escape(str(result["jepa_time_idx"]))}</p>
          <p class="meta"><b>latent_time_idx:</b> {html.escape(str(result["latent_time_idx"]))}</p>
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
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(result['cotracker_input_video_mp4'])}"></video>
      <figcaption>CoTracker input</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(result['vggt_input_video_mp4'])}"></video>
      <figcaption>VGGT input</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(result['jepa_input_video_mp4'])}"></video>
      <figcaption>JEPA input</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(result['jepa_time_aligned_context_video_mp4'])}"></video>
      <figcaption>jepa_time_idx frames</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(result['latent_time_aligned_context_video_mp4'])}"></video>
      <figcaption>latent_time_idx frames</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(result['input_overlay_video'])}"></video>
      <figcaption>Input overlay</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{html.escape(rel_dir)}/{html.escape(result['jepa_padded_video_mp4'])}"></video>
      <figcaption>JEPA padded context</figcaption>
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
    parser.set_defaults(
        fixed_num_context_frames=20,
        ctx_max_length=20,
        min_context_frames=0,
        max_context_ratio=1.0,
        context_frame_choices=None,
        context_length_sampling="short_biased",
        no_context_ratio=0.0,
    )
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
    for dataset_index in candidate_indices:
        if len(results) >= int(args.inspect_num_samples):
            break
        raw_sample = dataset[int(dataset_index)]
        rng = random.Random(int(args.inspect_seed) * 1000003 + int(dataset_index))
        sample, context_spec = actualmod._select_actual_context_sample(
            raw_sample,
            ctx_max_length=int(args.ctx_max_length),
            min_context_frames=int(args.min_context_frames),
            context_length_sampling=str(args.context_length_sampling),
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
        sampled_source_indices = list(raw_sample.get("metadata", {}).get("sampled_frame_indices", []))
        actual_context_local_indices = sample["context_frame_indices"].tolist()
        context_source_frame_indices = [
            int(sampled_source_indices[idx])
            for idx in actual_context_local_indices
            if 0 <= int(idx) < len(sampled_source_indices)
        ]
        jepa_input_source_indices = _expand_with_last_index(
            context_source_frame_indices,
            int(result["jepa_input_frames"]),
        )
        jepa_time_source_indices = _source_indices_from_time_idx(
            context_source_frame_indices,
            result["jepa_time_idx"],
        )
        latent_input_source_indices = _source_indices_from_time_idx(
            context_source_frame_indices,
            result["latent_time_idx"],
        )

        result.update(
            {
                "context_sampling_mode": str(context_spec["mode"]),
                "ctx_max_length": int(sample["ctx_max_length"]),
                "sampled_ctx_last_index": int(sample["sampled_ctx_last_index"]),
                "sampled_ctx_num_frames": int(sample["sampled_ctx_num_frames"]),
                "context_source_frame_indices": context_source_frame_indices,
                "jepa_input_source_indices": jepa_input_source_indices,
                "jepa_time_source_indices": jepa_time_source_indices,
                "latent_input_source_indices": latent_input_source_indices,
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
                    "jepa_input_frames": int(result["jepa_input_frames"]),
                    "jepa_padding_frames": int(result["jepa_padding_frames"]),
                    "latent_frames": int(result["latent_frames"]),
                    "jepa_token_frames": int(result["jepa_token_frames"]),
                    "jepa_time_idx": result["jepa_time_idx"],
                    "latent_time_idx": result["latent_time_idx"],
                    "jepa_input_source_indices": jepa_input_source_indices,
                    "jepa_time_source_indices": jepa_time_source_indices,
                    "latent_input_source_indices": latent_input_source_indices,
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
