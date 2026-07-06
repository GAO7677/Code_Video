from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any

import cv2
import imageio_ffmpeg
import numpy as np
import torch

from code_vjepa_vggt.context_wan_v_newtrain import flow_match_context_sft_loss
from code_vjepa_vggt.inspect_cotracker_vggt_geometry import (
    draw_box_rgb,
    draw_point_rgb,
    render_track_overlay,
    tensor_frame_to_uint8_hwc,
    write_mp4,
)
from code_vjepa_vggt.train0705_physinone_no_gt_box import (
    train_stage1b_context_only_no_gt_box_v_newtrain_physinone as trainmod,
)


INPUT_TRACK_COLOR = (0, 119, 182)
REF_BOX_COLOR = (214, 40, 40)
PRED_BOX_COLOR = (42, 157, 143)
REF_TRACK_COLOR = (247, 127, 0)
PRED_TRACK_COLOR = (39, 125, 161)


def _write_rgb_png(path: Path, image_hwc: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(image_hwc, cv2.COLOR_RGB2BGR))


def _ensure_browser_video(source_path: Path) -> Path:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        try:
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    if ffmpeg is None:
        return source_path
    out_path = source_path.with_name(f"{source_path.stem}.browser.mp4")
    if (
        out_path.exists()
        and out_path.stat().st_mtime_ns >= source_path.stat().st_mtime_ns
        and out_path.stat().st_size > 0
    ):
        return out_path
    subprocess.run(
        [
            ffmpeg,
            "-y",
            "-i",
            str(source_path),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(out_path),
        ],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    return out_path


def _norm_box_to_px(box_xyxy: np.ndarray, image_hw: tuple[int, int]) -> np.ndarray:
    height, width = image_hw
    scale = np.array([width, height, width, height], dtype=np.float32)
    return np.asarray(box_xyxy, dtype=np.float32) * scale


def _summary_to_px(track_summary_xydxdy: np.ndarray, image_hw: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    height, width = image_hw
    center = np.array(
        [
            float(track_summary_xydxdy[0]) * max(width - 1, 1),
            float(track_summary_xydxdy[1]) * max(height - 1, 1),
        ],
        dtype=np.float32,
    )
    delta = np.array(
        [
            float(track_summary_xydxdy[2]) * max(width - 1, 1),
            float(track_summary_xydxdy[3]) * max(height - 1, 1),
        ],
        dtype=np.float32,
    )
    start = center - delta
    return center, start


def _latent_valid_mask_from_boxes(box_xyxy: np.ndarray, object_valid_mask: np.ndarray) -> np.ndarray:
    box_valid = (
        (box_xyxy[..., 2] - box_xyxy[..., 0]) > 1.0e-6
    ) & (
        (box_xyxy[..., 3] - box_xyxy[..., 1]) > 1.0e-6
    )
    return box_valid & np.broadcast_to(
        object_valid_mask[None, :].astype(bool),
        box_valid.shape,
    )


def _render_prompt_preview(
    *,
    context_video: torch.Tensor,
    grounding_sample: Any,
    valid_queries_px: np.ndarray,
    query_owner: list[int],
) -> np.ndarray:
    prompt_frame_idx = int(getattr(grounding_sample, "prompt_frame_idx", 0))
    prompt_frame_idx = max(0, min(prompt_frame_idx, int(context_video.shape[1]) - 1))
    frame = tensor_frame_to_uint8_hwc(context_video[:, prompt_frame_idx]).copy()
    for obj_idx, track in enumerate(getattr(grounding_sample, "object_tracks", [])):
        color = REF_BOX_COLOR if obj_idx == 0 else PRED_TRACK_COLOR
        draw_box_rgb(frame, track.box_prompt_xyxy.astype(np.float32), color, f"prompt{obj_idx}")
    for q_idx, point in enumerate(valid_queries_px):
        owner = query_owner[q_idx] if q_idx < len(query_owner) else -1
        color = REF_TRACK_COLOR if owner < 0 else ((214, 40, 40), (247, 127, 0), (252, 191, 73), (42, 157, 143))[owner % 4]
        draw_point_rgb(frame, point.astype(np.float32), color, f"q{q_idx}", radius=6)
    return frame


def _render_ref_pred_box_overlay(
    *,
    context_video: torch.Tensor,
    ref_box_xyxy: np.ndarray,
    pred_box_xyxy: np.ndarray,
    valid_mask: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    frames: list[np.ndarray] = []
    latent_frames = int(ref_box_xyxy.shape[0])
    source_frames = int(context_video.shape[1])
    latent_to_source = np.linspace(
        0,
        max(source_frames - 1, 0),
        latent_frames,
    ).round().astype(np.int64)
    for latent_idx in range(latent_frames):
        src_idx = int(latent_to_source[latent_idx])
        frame = tensor_frame_to_uint8_hwc(context_video[:, src_idx]).copy()
        for obj_idx in range(ref_box_xyxy.shape[1]):
            if bool(valid_mask[latent_idx, obj_idx]):
                draw_box_rgb(
                    frame,
                    _norm_box_to_px(ref_box_xyxy[latent_idx, obj_idx], image_hw),
                    REF_BOX_COLOR,
                    f"ref{obj_idx}",
                )
                draw_box_rgb(
                    frame,
                    _norm_box_to_px(pred_box_xyxy[latent_idx, obj_idx], image_hw),
                    PRED_BOX_COLOR,
                    f"pred{obj_idx}",
                )
        frames.append(frame)
    return np.stack(frames, axis=0)


def _render_ref_pred_track_overlay(
    *,
    context_video: torch.Tensor,
    ref_track_summary: np.ndarray,
    pred_track_summary: np.ndarray,
    valid_mask: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    frames: list[np.ndarray] = []
    latent_frames = int(ref_track_summary.shape[0])
    source_frames = int(context_video.shape[1])
    latent_to_source = np.linspace(
        0,
        max(source_frames - 1, 0),
        latent_frames,
    ).round().astype(np.int64)
    for latent_idx in range(latent_frames):
        src_idx = int(latent_to_source[latent_idx])
        frame = tensor_frame_to_uint8_hwc(context_video[:, src_idx]).copy()
        for obj_idx in range(ref_track_summary.shape[1]):
            if not bool(valid_mask[latent_idx, obj_idx]):
                continue
            ref_center, ref_start = _summary_to_px(ref_track_summary[latent_idx, obj_idx], image_hw)
            pred_center, pred_start = _summary_to_px(pred_track_summary[latent_idx, obj_idx], image_hw)
            draw_point_rgb(frame, ref_center, REF_TRACK_COLOR, f"ref{obj_idx}", radius=5)
            draw_point_rgb(frame, ref_start, REF_TRACK_COLOR, f"rs{obj_idx}", radius=3)
            draw_point_rgb(frame, pred_center, PRED_TRACK_COLOR, f"pred{obj_idx}", radius=5)
            draw_point_rgb(frame, pred_start, PRED_TRACK_COLOR, f"ps{obj_idx}", radius=3)
        frames.append(frame)
    return np.stack(frames, axis=0)


def _valid_query_count(object_valid_mask: torch.Tensor, points_per_object: int) -> int:
    return int((object_valid_mask > 0.5).sum().item()) * int(points_per_object)


def _l1_on_mask(pred: torch.Tensor, ref: torch.Tensor, valid_mask: torch.Tensor) -> float:
    if ref.shape[-1] != pred.shape[-1]:
        ref = ref[..., : pred.shape[-1]]
    weights = valid_mask.to(device=pred.device, dtype=pred.dtype).unsqueeze(-1)
    denom = weights.sum().clamp_min(1.0) * pred.shape[-1]
    value = ((pred - ref).abs() * weights).sum() / denom
    return float(value.detach().item())


def _prepare_forward_inputs(model: trainmod.ContextOnlyNoGTBoxWanModule, sample: dict[str, Any]):
    inputs = model.get_pipeline_inputs(sample)
    inputs = model.transfer_data_to_device(inputs, model.pipe.device, model.pipe.torch_dtype)
    for unit in model.pipe.units:
        inputs = model.pipe.unit_runner(unit, model.pipe, *inputs)
    return inputs


def _run_forward_debug(
    model: trainmod.ContextOnlyNoGTBoxWanModule,
    sample: dict[str, Any],
    *,
    inputs_shared: dict[str, Any],
    inputs_posi: dict[str, Any],
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
        vggt_out = trainmod.load_vggt_cache(sample, model.vggt_cache_root, allow_missing=False)
        if vggt_out is None:
            raise RuntimeError(f"VGGT cache missing for sample {sample.get('video_path', '<unknown>')}")
    else:
        vggt_out = model.vggt_adapter(
            frames_bthwc_01,
            query_points_prior=query_points_prior,
            query_image_hw=image_hw,
        )

    tracks_grouped, visibility_grouped, confidence_grouped = model._group_tracks_to_objects(
        cotracker_out.tracks,
        cotracker_out.visibility,
        cotracker_out.confidence,
        max_objects=model.aux_max_objects,
        points_per_object=model.object_num_queries,
    )
    jepa_out = model._run_jepa(context_video)
    context_latents = inputs_shared["clean_prefix_latents"]
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
    loss_main = flow_match_context_sft_loss(
        model.pipe,
        **inputs_shared,
        **inputs_posi,
        object_context=object_context,
    ) if float(model.lambda_main) > 0.0 else object_context.new_zeros(())
    object_context_reg = object_context.square().mean()
    total_loss = (
        model.lambda_main * loss_main
        + model.lambda_object_context_reg * object_context_reg
    )

    ref_box_xyxy = object_out.active_box_xyxy[0].detach().float().cpu().numpy()
    pred_box_xyxy = object_aux_out.pred_box_xyxy[0].detach().float().cpu().numpy()
    ref_track_summary = object_out.active_track_summary[0, ..., :4].detach().float().cpu().numpy()
    pred_track_summary = object_aux_out.pred_track_summary[0].detach().float().cpu().numpy()
    object_valid_mask_np = object_valid_mask[0].detach().float().cpu().numpy()
    latent_valid_mask = _latent_valid_mask_from_boxes(ref_box_xyxy, object_valid_mask_np)

    valid_mask_t = torch.from_numpy(latent_valid_mask).to(device=object_out.active_box_xyxy.device)
    metrics = {
        "train/loss_total": float(total_loss.detach().item()),
        "train/loss_main": float(loss_main.detach().item()),
        "train/loss_object_context_reg": float(object_context_reg.detach().item()),
        "train/object_count": float(object_valid_mask.sum().item()),
        "aux/pred_box_l1_vs_ref": _l1_on_mask(
            object_aux_out.pred_box_xyxy[0],
            object_out.active_box_xyxy[0],
            valid_mask_t,
        ),
        "aux/pred_track_l1_vs_ref": _l1_on_mask(
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
        "latent_valid_mask": latent_valid_mask,
    }


def _load_optional_stage2_weights(
    model: trainmod.ContextOnlyNoGTBoxWanModule,
    stage2_resume_from: str | None,
) -> dict[str, Any] | None:
    if not stage2_resume_from:
        return None
    checkpoint_path = trainmod.tvn.resolve_lora_checkpoint_for_resume(stage2_resume_from)
    return trainmod.tvn._load_filtered_checkpoint_into_model(
        model,
        checkpoint_path,
        include_prefixes=("object_adapter.",),
        include_substrings=(
            "object_embedding",
            ".object_cross_attn.",
            ".object_gate",
            ".norm4.",
        ),
    )


def _move_optional_module(module: torch.nn.Module | None, device: torch.device) -> None:
    if module is not None:
        module.to(device)


def _build_html_report(result: dict[str, Any], output_dir: Path) -> Path:
    prepipe_gallery_block = ""
    if str(result.get("prepipe_gallery_url", "")).strip():
        prepipe_gallery_block = (
            f'<p><b>Input gallery:</b> <a href="{result["prepipe_gallery_url"]}">'
            f'{result["prepipe_gallery_url"]}</a></p>'
        )
    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>PhysInOne Train Forward Aux Overlay</title>
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
  <h1>PhysInOne No-GT-Box: 单样本训练前向</h1>
  <p>这里跑的是 train0705 PhysInOne no-GT-box 分支的一次真实 object branch 前向。主损失仍然是 `loss_main`，aux 可视化这里展示的是 frozen aux head 对 reference active box / active track 的预测输出。</p>
  <p><b>Caption:</b> {result["caption"]}</p>
  <p><b>Video:</b> {result["video_path"]}</p>
  <p><b>Sample key:</b> {result["sample_key"]}</p>
  <p><b>Context frames:</b> {result["context_frame_indices"]}</p>
  {prepipe_gallery_block}
  <div class="grid">
    <figure>
      <img src="{result["prompt_preview_png"]}" />
      <figcaption>Viewer-grounding prompt boxes + sampled query points</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{result["input_overlay_video"]}"></video>
      <figcaption>Input pre-pipe overlay: SAM2 boxes + query points + CoTracker tracks</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{result["box_overlay_video"]}"></video>
      <figcaption>Aux boxes: reference active box(red) vs predicted box(green)</figcaption>
    </figure>
    <figure>
      <video controls preload="none" playsinline src="{result["track_overlay_video"]}"></video>
      <figcaption>Aux tracks: reference summary(orange) vs predicted summary(blue)</figcaption>
    </figure>
  </div>
  <h2>Metrics</h2>
  <pre>{json.dumps(result["metrics"], indent=2, ensure_ascii=False)}</pre>
  <h2>Shapes</h2>
  <pre>{json.dumps(result["shapes"], indent=2, ensure_ascii=False)}</pre>
</body>
</html>
"""
    html_path = output_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def parse_args() -> argparse.Namespace:
    parser = trainmod.build_parser()
    parser.add_argument("--inspect_index", type=int, default=0)
    parser.add_argument(
        "--inspect_output_dir",
        type=str,
        default="/data/gaoya/agent-data/outputs/phisinone_train_forward_aux_overlay",
    )
    parser.add_argument("--inspect_fps", type=int, default=30)
    parser.add_argument(
        "--inspect_prepipe_gallery_url",
        type=str,
        default="",
    )
    return parser.parse_args()


def main() -> None:
    args = trainmod.tvn.prepare_args(parse_args())
    output_dir = Path(args.inspect_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = trainmod.build_dataset(args)
    accelerator = SimpleNamespace(device=torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model = trainmod.build_model(args, accelerator)
    target_device = torch.device(model.pipe.device)
    _move_optional_module(model.object_pooler, target_device)
    _move_optional_module(model.object_aux_heads, target_device)
    _move_optional_module(model.object_adapter, target_device)
    _move_optional_module(model.vggt_adapter, target_device)

    load_info: dict[str, Any] = {
        "stage1a_init_from": None,
        "stage1a_load": None,
        "stage2_resume_from": args.stage2_resume_from,
        "stage2_load": None,
    }
    if args.stage1a_init_from is not None:
        stage1a_info = trainmod.tvn._load_filtered_checkpoint_into_model(
            model,
            args.stage1a_init_from,
            include_prefixes=("object_pooler.", "object_aux_heads."),
        )
        load_info["stage1a_init_from"] = args.stage1a_init_from
        load_info["stage1a_load"] = stage1a_info

    stage2_info = _load_optional_stage2_weights(model, args.stage2_resume_from)
    if stage2_info is not None:
        load_info["stage2_load"] = stage2_info

    torch.nn.Module.train(model, False)
    sample = dataset[int(args.inspect_index)]
    inputs_shared, inputs_posi, _ = _prepare_forward_inputs(model, sample)
    with torch.no_grad():
        debug = _run_forward_debug(
            model,
            sample,
            inputs_shared=inputs_shared,
            inputs_posi=inputs_posi,
        )

    context_video = sample["context_video"]
    image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
    grounding_sample = model.viewer_grounding.build_sample(
        frames_tchw_01=((context_video.permute(1, 0, 2, 3).float() + 1.0) / 2.0).cpu().numpy(),
        caption=str(sample["caption"]),
        image_hw=image_hw,
    )

    valid_queries = _valid_query_count(debug["object_valid_mask"], model.object_num_queries)
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

    prompt_preview = _render_prompt_preview(
        context_video=context_video,
        grounding_sample=grounding_sample,
        valid_queries_px=valid_queries_px,
        query_owner=query_owner,
    )
    prompt_preview_path = output_dir / "prompt_preview.png"
    _write_rgb_png(prompt_preview_path, prompt_preview)

    input_overlay_video = render_track_overlay(
        context_video=context_video,
        object_tracks=getattr(grounding_sample, "object_tracks", []),
        prompt_frame_idx=int(getattr(grounding_sample, "prompt_frame_idx", 0)),
        query_points_px_k2=valid_queries_px.astype(np.float32),
        query_owner=query_owner,
        tracks_tk2=valid_tracks.astype(np.float32),
        visibility_tk=valid_visibility.astype(np.float32),
        color_rgb=INPUT_TRACK_COLOR,
        prefix="trk",
    )
    input_overlay_raw = output_dir / "input_prepipe_overlay.mp4"
    write_mp4(input_overlay_raw, input_overlay_video, fps=int(args.inspect_fps))
    input_overlay_browser = _ensure_browser_video(input_overlay_raw)

    ref_box_xyxy = debug["object_out"].active_box_xyxy[0].detach().float().cpu().numpy()
    pred_box_xyxy = debug["object_aux_out"].pred_box_xyxy[0].detach().float().cpu().numpy()
    ref_track_summary = debug["object_out"].active_track_summary[0, ..., :4].detach().float().cpu().numpy()
    pred_track_summary = debug["object_aux_out"].pred_track_summary[0].detach().float().cpu().numpy()
    latent_valid_mask = debug["latent_valid_mask"]

    box_overlay_video = _render_ref_pred_box_overlay(
        context_video=context_video,
        ref_box_xyxy=ref_box_xyxy,
        pred_box_xyxy=pred_box_xyxy,
        valid_mask=latent_valid_mask,
        image_hw=image_hw,
    )
    box_overlay_raw = output_dir / "aux_pred_box_overlay.mp4"
    write_mp4(box_overlay_raw, box_overlay_video, fps=int(args.inspect_fps))
    box_overlay_browser = _ensure_browser_video(box_overlay_raw)

    track_overlay_video = _render_ref_pred_track_overlay(
        context_video=context_video,
        ref_track_summary=ref_track_summary,
        pred_track_summary=pred_track_summary,
        valid_mask=latent_valid_mask,
        image_hw=image_hw,
    )
    track_overlay_raw = output_dir / "aux_pred_track_overlay.mp4"
    write_mp4(track_overlay_raw, track_overlay_video, fps=int(args.inspect_fps))
    track_overlay_browser = _ensure_browser_video(track_overlay_raw)

    result = {
        "caption": str(sample["caption"]),
        "video_path": str(sample["video_path"]),
        "sample_key": str(sample.get("metadata", {}).get("sample_key", "")),
        "context_frame_indices": sample["context_frame_indices"].tolist(),
        "prepipe_gallery_url": str(args.inspect_prepipe_gallery_url),
        "prompt_preview_png": str(prompt_preview_path.name),
        "input_overlay_video": str(input_overlay_browser.name),
        "box_overlay_video": str(box_overlay_browser.name),
        "track_overlay_video": str(track_overlay_browser.name),
        "metrics": debug["metrics"],
        "shapes": {
            "context_video": list(context_video.shape),
            "query_points_prior": list(debug["query_points_prior"].shape),
            "query_frame_ids": list(debug["query_frame_ids"].shape),
            "box_prior_xyxy": list(debug["box_prior_xyxy"].shape),
            "cotracker_tracks": list(debug["cotracker_out"].tracks.shape),
            "tracks_grouped": list(debug["tracks_grouped"].shape),
            "object_latent_tokens": list(debug["object_out"].object_latent_tokens.shape),
            "object_context": list(debug["object_context"].shape),
            "active_box_xyxy": list(debug["object_out"].active_box_xyxy.shape),
            "active_track_summary": list(debug["object_out"].active_track_summary.shape),
            "pred_box_xyxy": list(debug["object_aux_out"].pred_box_xyxy.shape),
            "pred_track_summary": list(debug["object_aux_out"].pred_track_summary.shape),
            "pred_depth": list(debug["object_aux_out"].pred_depth.shape),
        },
        "load_info": load_info,
    }
    (output_dir / "result.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    html_path = _build_html_report(result, output_dir)
    print(json.dumps({"output_dir": str(output_dir), "html": str(html_path), "result": result}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
