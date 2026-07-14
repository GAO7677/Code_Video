"""Run and visualize one actual Scheme-C entity-binding training forward."""
from __future__ import annotations

import html
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
import torch

from code_vjepa_vggt.data.pybullet_raw_no_gt_box_dataset import PyBulletRawNoGTBoxDataset
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    inspect_kubric_actual_train_forward_aux_overlay as implementation,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    inspect_kubric_train_forward_aux_overlay as inspectmod,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_context_only_no_gt_box_v_newtrain_kubric as base_train,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_no_gt_box_replay_preserve as replay_train,
)
from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    train_stage1b_no_gt_box_replay_preserve_entity_id_binding as entity_train,
)


_LAST_DEBUG: dict[str, Any] = {}
_LAST_RESULT: dict[str, Any] = {}
_COLORS = ((230, 74, 69), (46, 139, 87), (42, 111, 199), (218, 139, 42))


def _shape(value: Any) -> list[int] | None:
    if isinstance(value, torch.Tensor):
        return [int(item) for item in value.shape]
    if isinstance(value, np.ndarray):
        return [int(item) for item in value.shape]
    return None


def _record(trace: list[dict[str, Any]], stage: str, value: Any, meaning: str) -> None:
    item = {"stage": stage, "shape": _shape(value), "meaning": meaning}
    if isinstance(value, torch.Tensor):
        item.update({"dtype": str(value.dtype), "device": str(value.device)})
    elif isinstance(value, np.ndarray):
        item["dtype"] = str(value.dtype)
    elif value is not None:
        item["value"] = value
    trace.append(item)


class _SourceTaggedDataset:
    def __init__(self, dataset: PyBulletRawNoGTBoxDataset) -> None:
        self.dataset = dataset
        self.samples = dataset.samples

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = dict(self.dataset[index])
        metadata = dict(sample.get("metadata", {}))
        metadata.update({"dataset_source": "pybullet", "dataset_source_id": 0})
        sample["metadata"] = metadata
        return sample


def _build_inspection_dataset(args) -> _SourceTaggedDataset:
    dataset = PyBulletRawNoGTBoxDataset(
        root=args.pybullet_raw_root,
        split=args.pybullet_raw_split,
        resolution=(args.height, args.width),
        num_frames=args.num_frames,
        num_context_frames=args.fixed_num_context_frames,
        sampling_strategy=args.pybullet_raw_sampling_strategy,
        window_starts=tuple(
            int(value.strip())
            for value in args.pybullet_raw_window_starts.split(",")
            if value.strip()
        ),
        init_scan_limit=args.pybullet_raw_init_scan_limit,
    )
    return _SourceTaggedDataset(dataset)


def _external_title_bar(
    frames_thwc_u8: np.ndarray,
    *,
    title: str,
    primary_indices: list[int] | None = None,
    primary_label: str | None = None,
    secondary_indices: list[int] | None = None,
    secondary_label: str | None = None,
    tertiary_indices: list[int] | None = None,
    tertiary_label: str | None = None,
) -> np.ndarray:
    frames = np.asarray(frames_thwc_u8, dtype=np.uint8)
    bar_h = 70
    output = np.full(
        (int(frames.shape[0]), int(frames.shape[1]) + bar_h, int(frames.shape[2]), 3),
        (246, 243, 236),
        dtype=np.uint8,
    )
    output[:, bar_h:] = frames
    for frame_id in range(int(frames.shape[0])):
        labels = [f"{title} | frame {frame_id:02d}/{max(int(frames.shape[0]) - 1, 0):02d}"]
        timeline = []
        for label, values in (
            (primary_label, primary_indices),
            (secondary_label, secondary_indices),
            (tertiary_label, tertiary_indices),
        ):
            if label:
                value = "n/a" if values is None or frame_id >= len(values) else str(int(values[frame_id]))
                timeline.append(f"{label}={value}")
        labels.append(" | ".join(timeline))
        for line_id, line in enumerate(labels):
            cv2.putText(
                output[frame_id], line, (14, 25 + line_id * 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.58, (34, 34, 34), 1, cv2.LINE_AA,
            )
    return output


def _full_forward_debug(model, inputs_cpu) -> dict[str, Any]:
    trace: list[dict[str, Any]] = []
    inputs = model.transfer_data_to_device(inputs_cpu, model.pipe.device, model.pipe.torch_dtype)
    for unit in model.pipe.units:
        inputs = model.pipe.unit_runner(unit, model.pipe, *inputs)
    inputs_shared, inputs_posi, _inputs_nega = inputs
    sample = inputs_shared["raw_sample"]
    _record(trace, "01_raw_training_video", sample["video"], "[C,T,H,W], decoded and resized training clip")
    _record(trace, "02_context_video", sample["context_video"], "[C,Tctx,H,W], fixed clean prefix shown to Stage1A")
    _record(trace, "03_prompt_t5_context", inputs_posi.get("context"), "[B,L,D], UMT5 prompt token embeddings")
    _record(trace, "04_input_latents", inputs_shared.get("input_latents"), "[B,Cz,Tz,Hz,Wz], VAE latents for all 49 frames")
    _record(trace, "05_clean_prefix_latents", inputs_shared.get("clean_prefix_latents"), "[B,Cz,Tctx_z,Hz,Wz], context latents kept clean")

    context_video = sample["context_video"].unsqueeze(0).to(
        device=model.pipe.device, dtype=model.pipe.torch_dtype
    )
    image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
    grounding_capture: dict[str, Any] = {}
    original_build = model.viewer_grounding.build_sample

    def capture_grounding(*args, **kwargs):
        result = original_build(*args, **kwargs)
        grounding_capture["sample"] = result
        return result

    model.viewer_grounding.build_sample = capture_grounding
    model.entity_bound_adapter.clear_entity_binding_context()
    model.entity_bound_adapter.pop_entity_binding_metrics()
    model._entity_binding_runtime = {
        "prompt_context": inputs_posi["context"],
        "prompt": str(sample.get("caption", "")),
        "tokenizer": model.pipe.tokenizer,
    }
    try:
        query_points, query_frame_ids, object_valid_mask_before, box_prior = (
            model._build_object_query_priors(sample, image_hw=image_hw)
        )
    finally:
        model.viewer_grounding.build_sample = original_build
    grounding_sample = grounding_capture["sample"]
    query_points = query_points.to(device=model.pipe.device, dtype=model.pipe.torch_dtype)
    query_frame_ids = query_frame_ids.to(device=model.pipe.device, dtype=model.pipe.torch_dtype)
    object_valid_mask_before = object_valid_mask_before.to(device=model.pipe.device, dtype=model.pipe.torch_dtype)
    box_prior = box_prior.to(device=model.pipe.device, dtype=model.pipe.torch_dtype)
    object_valid_mask, slot_metrics = model._apply_object_slot_dropout(object_valid_mask_before)
    _record(trace, "06_grounding_context_boxes", grounding_sample.context_boxes_norm, "[Tctx,O,4], GDINO+SAM2 normalized xyxy tracks")
    _record(trace, "07_query_points", query_points, "[B,O*Q,2], mask-sampled CoTracker/VGGT queries in pixels")
    _record(trace, "08_query_frame_ids", query_frame_ids, "[B,O*Q,1], query anchor frame IDs")
    _record(trace, "09_object_valid_before_dropout", object_valid_mask_before, "[B,O], valid grounded slots")
    _record(trace, "10_object_valid_after_dropout", object_valid_mask, "[B,O], actual training slot-dropout mask")
    _record(trace, "11_box_prior", box_prior, "[B,O,4], first valid normalized box per slot")

    adapter = model.entity_bound_adapter
    entity_text = adapter._entity_text_by_id
    entity_match = adapter._entity_text_match_mask
    slot_entity_ids = adapter._slot_entity_ids
    _record(trace, "12_entity_text_by_id", entity_text, "[B,E,D], noun-span pooled T5 vectors indexed by video-local entity ID")
    _record(trace, "13_entity_match_mask", entity_match, "[B,E], entity IDs that matched a prompt noun span")
    _record(trace, "14_slot_entity_ids", slot_entity_ids, "[B,O], hard route from tracked slot to video-local entity ID")
    phrases = [str(getattr(track, "phrase", "")) for track in grounding_sample.object_tracks]
    route_summary = []
    for slot_id, phrase in enumerate(phrases):
        entity_id = int(slot_entity_ids[0, slot_id].item()) if slot_id < int(slot_entity_ids.shape[1]) else -1
        matched = bool(entity_match[0, entity_id].item()) if 0 <= entity_id < int(entity_match.shape[1]) else False
        route_summary.append(f"slot{slot_id}->entity{entity_id}: phrase={phrase!r}, prompt_match={matched}")
    _record(trace, "14b_entity_route_summary", route_summary, "; ".join(route_summary))

    frames_bthwc = ((context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
    _record(trace, "15_frames_bthwc_01", frames_bthwc, "[B,Tctx,H,W,C], object feature-extractor input in [0,1]")
    cotracker = model._run_cotracker(
        frames_bthwc,
        query_points_prior=query_points,
        query_frame_ids=query_frame_ids,
        query_image_hw=image_hw,
    )
    _record(trace, "16_cotracker_tracks", cotracker.tracks, "[B,Tctx,O*Q,2], tracked point coordinates")
    _record(trace, "17_cotracker_visibility", cotracker.visibility, "[B,Tctx,O*Q], visibility probabilities/mask")
    _record(trace, "18_cotracker_confidence", cotracker.confidence, "[B,Tctx,O*Q], tracker confidence")
    vggt = model._run_vggt(frames_bthwc, query_points_prior=query_points, query_image_hw=image_hw)
    for name in ("dense_patch_tokens", "depth", "depth_conf", "world_points", "world_points_conf"):
        _record(trace, f"19_vggt_{name}", getattr(vggt, name, None), f"VGGT {name}")
    tracks, visibility, confidence = model._group_tracks_to_objects(
        cotracker.tracks, cotracker.visibility, cotracker.confidence,
        max_objects=model.aux_max_objects, points_per_object=model.object_num_queries,
    )
    _record(trace, "20_grouped_tracks", tracks, "[B,Tctx,O,Q,2], CoTracker points grouped into object slots")

    context_latents = inputs_shared["clean_prefix_latents"]
    jepa_input, jepa_fix = base_train.prepare_jepa_context_video(
        context_video,
        latent_frames=int(context_latents.shape[2]),
        tubelet_size=int(model._jepa_tubelet_size),
    )
    jepa = model._run_jepa(jepa_input)
    _record(trace, "21_jepa_padded_input", jepa_input, "[B,C,Tpad,H,W], temporal padding before JEPA resize/encode")
    _record(trace, "22_jepa_patch_tokens", jepa.patch_tokens, "[B,Tj,Nj,Dj], JEPA patch tokens")
    object_out = model.object_pooler(
        jepa_patch_tokens=jepa.patch_tokens,
        context_latents=context_latents,
        tracks=tracks,
        visibility=visibility,
        confidence=confidence,
        track_image_hw=image_hw,
        object_valid_mask=object_valid_mask,
        box_prior_xyxy=box_prior,
        vggt_world_points=getattr(vggt, "world_points", None),
        vggt_world_points_conf=getattr(vggt, "world_points_conf", None),
        vggt_depth=getattr(vggt, "depth", None),
        vggt_depth_conf=getattr(vggt, "depth_conf", None),
        vggt_dense_patch_tokens=getattr(vggt, "dense_patch_tokens", None),
        vggt_patch_grid_hw=getattr(vggt, "patch_grid_hw", None),
        vggt_geometry_image_hw=getattr(vggt, "input_hw", None) or getattr(vggt, "image_hw", None),
        frame_valid_mask=None,
    )
    _record(trace, "23_object_pooler_tokens", object_out.object_latent_tokens, "[B,Tz_ctx,O,D], fused JEPA/latent/track/VGGT object tokens")
    _record(trace, "24_active_box_xyxy", object_out.active_box_xyxy, "[B,Tz_ctx,O,4], latent-time object boxes")

    captured: dict[str, Any] = {}
    original_apply = adapter.apply_entity_binding

    def capture_binding(tokens, *, object_valid_mask=None):
        bound = original_apply(tokens, object_valid_mask=object_valid_mask)
        captured["bound_tokens"] = bound
        return bound

    adapter.apply_entity_binding = capture_binding
    try:
        object_context = adapter(
            object_out.object_latent_tokens,
            object_valid_mask=object_valid_mask,
            bbox_xyxy=object_out.active_box_xyxy,
        )
    finally:
        adapter.apply_entity_binding = original_apply
    bound_tokens = captured["bound_tokens"]
    object_context_for_dit = (
        base_train.compact_object_context_valid_slots(object_context, object_valid_mask)
        if model.compact_object_context_slots else object_context
    )
    _record(trace, "25_entity_bound_object_tokens", bound_tokens, "[B,Tz_ctx,O,D], pooler tokens plus gated text+ID residual")
    _record(trace, "26_object_adapter_context", object_context, "[B,Tz_ctx*O,D], slot/time/box encoded object context")
    _record(trace, "27_compacted_object_context", object_context_for_dit, "[B,Tz_ctx*Ovalid,D], tokens passed to every Wan object cross-attention block")

    flow_capture: dict[str, Any] = {}
    scheduler = model.pipe.scheduler
    original_target = scheduler.training_target
    original_model_fn = model.pipe.model_fn

    def capture_target(input_latents, noise, timestep):
        target = original_target(input_latents, noise, timestep)
        flow_capture.update(input_latents=input_latents, noise=noise, timestep=timestep, target=target)
        return target

    def capture_model_fn(*args, **kwargs):
        flow_capture["noisy_latents"] = kwargs.get("latents")
        prediction = original_model_fn(*args, **kwargs)
        flow_capture["prediction"] = prediction
        return prediction

    scheduler.training_target = capture_target
    model.pipe.model_fn = capture_model_fn
    try:
        loss_main, dit_trace = model._run_main_loss_with_trace(
            model.pipe, inputs_shared, inputs_posi, object_context_for_dit
        )
    finally:
        scheduler.training_target = original_target
        model.pipe.model_fn = original_model_fn
    _record(trace, "28_sampled_noise", flow_capture.get("noise"), "[B,Cz,Tz,Hz,Wz], one actual training noise draw")
    _record(trace, "29_noisy_latents_with_clean_context", flow_capture.get("noisy_latents"), "[B,Cz,Tz,Hz,Wz], DiT input with context latents restored clean")
    _record(trace, "30_dit_prediction", flow_capture.get("prediction"), "[B,Cz,Tz,Hz,Wz], student flow prediction")
    _record(trace, "31_training_target", flow_capture.get("target"), "[B,Cz,Tz,Hz,Wz], scheduler flow-matching target")
    num_prefix = int(inputs_shared.get("num_clean_prefix_latents", context_latents.shape[2]))
    _record(trace, "32_supervised_future_prediction", flow_capture["prediction"][:, :, num_prefix:], "future-only loss slice")
    _record(trace, "33_supervised_future_target", flow_capture["target"][:, :, num_prefix:], "future-only target slice")

    object_context_reg = object_context.square().mean()
    object_gate_reg, gate_metrics = model._compute_object_gate_regularizer(model.pipe)
    adapter_mlp_reg, adapter_mlp_metrics = model._consume_object_adapter_mlp_regularizer(
        model.pipe, object_valid_mask
    )
    main_loss_weight = float(slot_metrics["train/object_main_loss_weight"])
    total_loss = (
        model.lambda_main * main_loss_weight * loss_main
        + model.lambda_object_context_reg * object_context_reg
        + model.lambda_object_gate_reg * object_gate_reg
        + model.lambda_object_adapter_mlp_reg * adapter_mlp_reg
    )
    loss_components = {
        "loss_total": float(total_loss.detach().item()),
        "loss_main": float(loss_main.detach().item()),
        "loss_object_context_reg": float(object_context_reg.detach().item()),
        "loss_object_gate_reg": float(object_gate_reg.detach().item()),
        "loss_object_adapter_mlp_reg": float(adapter_mlp_reg.detach().item()),
    }
    _record(
        trace,
        "34_actual_training_loss_components",
        loss_components,
        "Scheme-C weighted total uses main + object-context + gate + adapter-MLP regularization",
    )

    # These heads are diagnostic only in this no-GT-box configuration; their loss weights are zero.
    object_aux = model.object_aux_heads(
        object_out.object_latent_tokens,
        object_out.active_track_summary,
        object_out.active_box_xyxy,
    )
    ref_box = object_out.active_box_xyxy[0].detach().float().cpu().numpy()
    valid_np = object_valid_mask[0].detach().float().cpu().numpy()
    latent_valid = inspectmod._latent_valid_mask_from_boxes(ref_box, valid_np)
    metrics = {
        "train/loss_main": float(loss_main.detach().item()),
        "train/loss_total": float(total_loss.detach().item()),
        "train/loss_object_context_reg": float(object_context_reg.detach().item()),
        "train/loss_object_gate_reg": float(object_gate_reg.detach().item()),
        "train/loss_object_adapter_mlp_reg": float(adapter_mlp_reg.detach().item()),
        "train/object_count_before_dropout": float(object_valid_mask_before.sum().item()),
        "train/object_count": float(object_valid_mask.sum().item()),
        "train/object_context_abs_mean": float(object_context.detach().abs().mean().item()),
        "train/object_context_abs_max": float(object_context.detach().abs().max().item()),
        **slot_metrics,
        **gate_metrics,
        **adapter_mlp_metrics,
        **model._last_preservation_metrics,
        **adapter.pop_entity_binding_metrics(),
        **model._entity_binding_prepare_metrics,
    }
    jepa_time_idx = model.object_pooler._time_indices(
        int(tracks.shape[1]), int(jepa.patch_tokens.shape[1]), tracks.device
    )
    latent_time_idx = model.object_pooler._time_indices(
        int(tracks.shape[1]), int(context_latents.shape[2]), tracks.device
    )
    debug = {
        "metrics": metrics,
        "shape_trace": trace,
        "frames_bthwc_01": frames_bthwc,
        "query_points_prior": query_points,
        "query_frame_ids": query_frame_ids,
        "object_valid_mask": object_valid_mask,
        "object_valid_mask_before_dropout": object_valid_mask_before,
        "box_prior_xyxy": box_prior,
        "cotracker_out": cotracker,
        "vggt_out": vggt,
        "tracks_grouped": tracks,
        "visibility_grouped": visibility,
        "confidence_grouped": confidence,
        "jepa_out": jepa,
        "object_out": object_out,
        "object_aux_out": object_aux,
        "object_context": object_context,
        "object_context_for_dit": object_context_for_dit,
        "bound_tokens": bound_tokens,
        "context_latents": context_latents,
        "jepa_input_video": jepa_input,
        "jepa_ctx_fix": jepa_fix,
        "jepa_time_idx": jepa_time_idx,
        "latent_time_idx": latent_time_idx,
        "latent_valid_mask": latent_valid,
        "grounding_sample": grounding_sample,
        "inputs_shared": inputs_shared,
        "slot_entity_ids": slot_entity_ids,
        "entity_match_mask": entity_match,
        "dit_trace": dit_trace,
    }
    _LAST_DEBUG.clear()
    _LAST_DEBUG.update(debug)
    model._entity_binding_runtime = None
    adapter.clear_entity_binding_context()
    return debug


def _render_entity_overlay(sample: dict[str, Any], debug: dict[str, Any], output_dir: Path, fps: int) -> Path:
    context = sample["context_video"]
    grounding = debug["grounding_sample"]
    tracks = debug["cotracker_out"].tracks[0].detach().float().cpu().numpy()
    visibility = debug["cotracker_out"].visibility[0].detach().float().cpu().numpy()
    slot_ids = debug["slot_entity_ids"][0].detach().cpu().tolist()
    match = debug["entity_match_mask"][0].detach().cpu().tolist()
    valid = debug["object_valid_mask_before_dropout"][0].detach().cpu().tolist()
    object_tracks = list(getattr(grounding, "object_tracks", []))
    frames = []
    for frame_id in range(int(context.shape[1])):
        frame = inspectmod.tensor_frame_to_uint8_hwc(context[:, frame_id]).copy()
        for slot_id, track in enumerate(object_tracks):
            if slot_id >= len(valid) or float(valid[slot_id]) <= 0.5:
                continue
            color = _COLORS[slot_id % len(_COLORS)]
            mask = np.asarray(track.masks_thw[frame_id]) > 0
            tint = np.zeros_like(frame)
            tint[mask] = color
            frame = np.where(mask[..., None], (0.68 * frame + 0.32 * tint).astype(np.uint8), frame)
            entity_id = int(slot_ids[slot_id])
            matched = bool(match[entity_id]) if 0 <= entity_id < len(match) else False
            keep = float(debug["object_valid_mask"][0, slot_id].item()) > 0.5
            label = f"S{slot_id}/E{entity_id} match={int(matched)} keep={int(keep)}"
            inspectmod.draw_box_rgb(frame, np.asarray(track.boxes_t4[frame_id], dtype=np.float32), color, label)
            base = slot_id * int(len(tracks[frame_id]) // max(len(valid), 1))
            per_slot = int(len(tracks[frame_id]) // max(len(valid), 1))
            for query_id in range(base, min(base + per_slot, len(tracks[frame_id]))):
                if float(visibility[frame_id, query_id]) >= 0.5:
                    inspectmod.draw_point_rgb(frame, tracks[frame_id, query_id], color, "", radius=4)
        frames.append(frame)
    framed = _external_title_bar(
        np.stack(frames),
        title="Actual train: SAM2 masks + CoTracker + slot/entity hard routing",
        primary_indices=sample["context_frame_indices"].tolist(),
        primary_label="train_local",
    )
    raw = output_dir / "entity_id_binding_overlay.mp4"
    inspectmod.write_mp4(raw, framed, fps=fps)
    return inspectmod._ensure_browser_video(raw)


def _write_report(output_dir: Path, result: dict[str, Any], debug: dict[str, Any]) -> None:
    case_dir = output_dir / result["relative_dir"]
    trace_path = case_dir / "shape_trace.json"
    trace_path.write_text(json.dumps(debug["shape_trace"], indent=2), encoding="utf-8")
    metrics_path = case_dir / "smoke_metrics.json"
    metrics_path.write_text(json.dumps(debug["metrics"], indent=2), encoding="utf-8")
    dit_trace_path = case_dir / "dit_object_branch_trace.json"
    dit_trace_path.write_text(json.dumps(debug["dit_trace"] or [], indent=2), encoding="utf-8")
    rows = "".join(
        f"<tr><td>{html.escape(item['stage'])}</td><td><code>{html.escape(str(item.get('shape')))}</code></td>"
        f"<td>{html.escape(str(item.get('dtype', '')))}</td><td>{html.escape(item['meaning'])}</td></tr>"
        for item in debug["shape_trace"]
    )
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>Entity-ID train forward smoke</title>
<style>body{{margin:0;background:#f4f1ea;color:#20201e;font:15px Georgia,serif}}main{{max-width:1500px;margin:auto;padding:24px}}h1,h2{{font-family:Arial,sans-serif}}video{{width:100%;background:#000}}.grid{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}}figure{{margin:0;background:#fff;border:1px solid #cbc5b9}}figcaption{{padding:10px}}table{{width:100%;border-collapse:collapse;background:#fff}}th,td{{padding:9px;border:1px solid #d8d2c7;text-align:left;vertical-align:top}}code,pre{{font-family:monospace}}pre{{background:#fff;padding:14px;overflow:auto}}@media(max-width:900px){{.grid{{grid-template-columns:1fr}}}}</style></head>
<body><main><h1>Scheme-C entity-ID binding: actual training forward smoke</h1>
<p><b>Sample:</b> {html.escape(result['video_path'])}<br><b>Caption:</b> {html.escape(result['caption'])}</p>
<p>This is a forward-only smoke in <code>train()</code> mode. It executes the real PyBullet branch through flow-matching DiT loss; no optimizer step is taken. Object aux overlays are diagnostic only because current no-GT-box training sets their loss weights to zero.</p>
<div class='grid'><figure><video controls src='{html.escape(result['relative_dir'])}/entity_id_binding_overlay.browser.mp4'></video><figcaption>SAM2 masks, CoTracker points, GroundingDINO phrase, slot ID and video-local entity ID. The title bar is outside the source image.</figcaption></figure>
<figure><video controls src='{html.escape(result['relative_dir'])}/{html.escape(result['input_overlay_video'])}'></video><figcaption>Baseline Stage1A input overlay from the same forward.</figcaption></figure>
<figure><video controls src='{html.escape(result['relative_dir'])}/{html.escape(result['jepa_input_video_mp4'])}'></video><figcaption>Actual JEPA resized input.</figcaption></figure>
<figure><video controls src='{html.escape(result['relative_dir'])}/{html.escape(result['box_overlay_video'])}'></video><figcaption>Aux head diagnostic, not an active training loss.</figcaption></figure></div>
<h2>Shape transitions</h2><table><thead><tr><th>Stage</th><th>Shape</th><th>Dtype</th><th>Meaning</th></tr></thead><tbody>{rows}</tbody></table>
<h2>Smoke metrics</h2><pre>{html.escape(json.dumps(debug['metrics'], indent=2))}</pre>
<p><a href='{html.escape(result['relative_dir'])}/shape_trace.json'>shape_trace.json</a> | <a href='{html.escape(result['relative_dir'])}/smoke_metrics.json'>smoke_metrics.json</a> | <a href='{html.escape(result['relative_dir'])}/dit_object_branch_trace.json'>DiT trace</a> | <a href='{html.escape(result['relative_dir'])}/index.html'>all middleware views</a></p>
</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")


def _install() -> None:
    implementation.trainmod = SimpleNamespace(
        ContextOnlyNoGTBoxWanModule=entity_train.EntityIDBindingReplayPreserveWanModule,
        build_parser=entity_train.build_parser,
        build_dataset=_build_inspection_dataset,
        build_model=entity_train.build_model,
        load_vggt_cache=base_train.load_vggt_cache,
        prepare_jepa_context_video=base_train.prepare_jepa_context_video,
        tvn=base_train.tvn,
    )
    implementation._run_forward_debug = _full_forward_debug
    implementation._offload_unused_pipe_modules = lambda model: []
    implementation._overlay_timeline_labels = _external_title_bar

    original_inspect_one = implementation._inspect_one

    def inspect_one(*, model, raw_sample, sample, inputs_cpu, output_dir, inspect_fps, inspect_index):
        result = original_inspect_one(
            model=model, raw_sample=raw_sample, sample=sample, inputs_cpu=inputs_cpu,
            output_dir=output_dir, inspect_fps=inspect_fps, inspect_index=inspect_index,
        )
        entity_video = _render_entity_overlay(sample, _LAST_DEBUG, output_dir, int(inspect_fps))
        result["entity_id_binding_overlay"] = entity_video.name
        _LAST_RESULT.clear()
        _LAST_RESULT.update(result)
        return result

    implementation._inspect_one = inspect_one


def _find_f2_index(args) -> int:
    root = Path(args.pybullet_raw_root) / args.pybullet_raw_split
    videos = sorted(root.glob("*/sample_*/video.mp4"))
    for index, path in enumerate(videos):
        if path.parent.parent.name == "F2_two_object":
            return index
    raise RuntimeError(f"no F2_two_object sample found under {root}")


def main() -> None:
    _install()
    # Preserve an explicit user choice; otherwise deterministically select the first F2 sample.
    if "--inspect_indices" not in sys.argv:
        parser = entity_train.build_parser()
        known, _ = parser.parse_known_args()
        sys.argv.extend(["--inspect_indices", str(_find_f2_index(known))])
    random.seed(42)
    torch.manual_seed(42)
    implementation.main()
    output_flag = sys.argv.index("--inspect_output_dir")
    output_dir = Path(sys.argv[output_flag + 1]).expanduser().resolve()
    if not _LAST_RESULT:
        raise RuntimeError("smoke visualizer produced no sample")
    result = dict(_LAST_RESULT)
    result["relative_dir"] = str(Path(f"sample_{int(result['inspect_index']):06d}_ctx{int(result['num_context_frames']):02d}"))
    _write_report(output_dir, result, _LAST_DEBUG)


if __name__ == "__main__":
    main()
