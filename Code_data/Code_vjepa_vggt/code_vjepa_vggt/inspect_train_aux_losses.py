from __future__ import annotations

import argparse
import http.server
import json
import socketserver
from pathlib import Path
from typing import Any

import numpy as np
import torch

from code_vjepa_vggt.infer_context_video_wan import _load_trainable_state
from code_vjepa_vggt.inspect_cotracker_vggt_geometry import (
    colorize_scalar_video,
    draw_box_rgb,
    draw_point_rgb,
    ensure_browser_video,
    write_mp4,
)
from code_vjepa_vggt.trainers.context_video_trainer import ContextVideoTrainer
from code_vjepa_vggt.utils.config import load_yaml_config
from code_vjepa_vggt.utils.masks import collate_video_batch


BOX_GT_COLOR = (214, 40, 40)
BOX_PRED_COLOR = (42, 157, 143)
TRACK_GT_COLOR = (247, 127, 0)
TRACK_PRED_COLOR = (39, 125, 161)


def tensor_frame_to_uint8_hwc(frame_chw: torch.Tensor) -> np.ndarray:
    x = frame_chw.detach().cpu().clamp(-1.0, 1.0)
    x = ((x + 1.0) * 127.5).to(torch.uint8).permute(1, 2, 0).contiguous()
    return x.numpy()


def _resolve_checkpoint_path(args: argparse.Namespace) -> Path:
    if args.checkpoint:
        path = Path(args.checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"checkpoint not found: {path}")
        return path
    if not args.checkpoint_dir:
        raise ValueError("either --checkpoint or --checkpoint-dir must be provided")
    candidates = sorted(Path(args.checkpoint_dir).glob("step_*.pt"))
    if not candidates:
        raise FileNotFoundError(f"no step_*.pt found in {args.checkpoint_dir}")
    return candidates[-1]


def _load_aux_state_into_trainer(trainer: ContextVideoTrainer, checkpoint_path: Path) -> dict[str, Any]:
    state_dict = _load_trainable_state(checkpoint_path)
    latent_key = "object_pooler.latent_proj.weight"
    if latent_key in state_dict and len(state_dict[latent_key].shape) == 2:
        trainer.object_pooler._ensure_latent_proj(int(state_dict[latent_key].shape[1]), trainer.device_obj)
    selected_prefixes = ("object_pooler.", "object_aux_heads.", "object_adapter.")
    current_state = trainer.state_dict()
    filtered_state = {}
    skipped_shape_mismatch: dict[str, dict[str, Any]] = {}
    for key, value in state_dict.items():
        if not key.startswith(selected_prefixes):
            continue
        if key not in current_state:
            continue
        if tuple(current_state[key].shape) != tuple(value.shape):
            skipped_shape_mismatch[key] = {
                "checkpoint_shape": list(value.shape),
                "model_shape": list(current_state[key].shape),
            }
            continue
        filtered_state[key] = value
    missing = trainer.load_state_dict(filtered_state, strict=False)
    return {
        "loaded_keys": len(filtered_state),
        "missing_keys": list(missing.missing_keys),
        "unexpected_keys": list(missing.unexpected_keys),
        "skipped_shape_mismatch": skipped_shape_mismatch,
    }


def _compute_aux_metrics(
    prepared: dict[str, Any],
) -> dict[str, float]:
    object_aux_out = prepared["object_aux_out"]
    gt_track_summary = prepared["gt_track_summary"]
    gt_track_valid = prepared["gt_track_valid"]
    gt_box_xyxy = prepared["gt_box_xyxy"]
    gt_box_valid = prepared["gt_box_valid"]
    gt_depth = prepared["gt_depth"]
    gt_depth_valid = prepared["gt_depth_valid"]

    dtype_device_tensor = object_aux_out.pred_track_summary
    track_aux_loss = dtype_device_tensor.new_zeros(())
    box_aux_loss = dtype_device_tensor.new_zeros(())
    depth_aux_loss = dtype_device_tensor.new_zeros(())

    if gt_track_summary is not None and gt_track_valid is not None:
        pred_track_summary = torch.nan_to_num(object_aux_out.pred_track_summary, nan=0.0, posinf=0.0, neginf=0.0)
        weights = gt_track_valid.unsqueeze(-1).to(dtype=pred_track_summary.dtype, device=pred_track_summary.device)
        denom = weights.sum().clamp_min(1.0) * pred_track_summary.shape[-1]
        track_aux_loss = ((pred_track_summary - gt_track_summary).abs() * weights).sum() / denom
    if gt_box_xyxy is not None and gt_box_valid is not None:
        pred_box_xyxy = torch.nan_to_num(object_aux_out.pred_box_xyxy, nan=0.0, posinf=0.0, neginf=0.0)
        weights = gt_box_valid.unsqueeze(-1).to(dtype=pred_box_xyxy.dtype, device=pred_box_xyxy.device)
        denom = weights.sum().clamp_min(1.0) * pred_box_xyxy.shape[-1]
        box_aux_loss = ((pred_box_xyxy - gt_box_xyxy).abs() * weights).sum() / denom
    if gt_depth is not None and gt_depth_valid is not None:
        pred_depth = torch.nan_to_num(object_aux_out.pred_depth, nan=0.0, posinf=0.0, neginf=0.0)
        weights = gt_depth_valid.unsqueeze(-1).to(dtype=pred_depth.dtype, device=pred_depth.device)
        denom = weights.sum().clamp_min(1.0) * pred_depth.shape[-1]
        depth_aux_loss = ((pred_depth - gt_depth).abs() * weights).sum() / denom

    return {
        "train/loss_track_aux": float(track_aux_loss.item()),
        "train/loss_box_aux": float(box_aux_loss.item()),
        "train/loss_depth_aux": float(depth_aux_loss.item()),
        "train/track_box_loss": 0.0
        if prepared["track_box_loss"] is None
        else float(torch.nan_to_num(prepared["track_box_loss"]).item()),
        "train/track_iou_loss": 0.0
        if prepared["track_iou_loss"] is None
        else float(torch.nan_to_num(prepared["track_iou_loss"]).item()),
    }


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


def _render_box_overlay(
    context_video: torch.Tensor,
    gt_box_xyxy: np.ndarray,
    gt_box_valid: np.ndarray,
    pred_box_xyxy: np.ndarray,
    pred_box_valid: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    frames: list[np.ndarray] = []
    latent_frames = int(gt_box_xyxy.shape[0])
    source_frames = int(context_video.shape[1])
    latent_to_source = np.linspace(0, max(source_frames - 1, 0), latent_frames).round().astype(np.int64)
    for latent_idx in range(latent_frames):
        src_idx = int(latent_to_source[latent_idx])
        frame = tensor_frame_to_uint8_hwc(context_video[:, src_idx]).copy()
        for obj_idx in range(gt_box_xyxy.shape[1]):
            if bool(gt_box_valid[latent_idx, obj_idx]):
                draw_box_rgb(
                    frame,
                    _norm_box_to_px(gt_box_xyxy[latent_idx, obj_idx], image_hw),
                    BOX_GT_COLOR,
                    f"gt{obj_idx}",
                )
            if bool(pred_box_valid[latent_idx, obj_idx]):
                draw_box_rgb(
                    frame,
                    _norm_box_to_px(pred_box_xyxy[latent_idx, obj_idx], image_hw),
                    BOX_PRED_COLOR,
                    f"pred{obj_idx}",
                )
        frames.append(frame)
    return np.stack(frames, axis=0)


def _render_track_overlay(
    context_video: torch.Tensor,
    gt_track_summary: np.ndarray,
    gt_track_valid: np.ndarray,
    pred_track_summary: np.ndarray,
    pred_track_valid: np.ndarray,
    image_hw: tuple[int, int],
) -> np.ndarray:
    frames: list[np.ndarray] = []
    latent_frames = int(gt_track_summary.shape[0])
    source_frames = int(context_video.shape[1])
    latent_to_source = np.linspace(0, max(source_frames - 1, 0), latent_frames).round().astype(np.int64)
    for latent_idx in range(latent_frames):
        src_idx = int(latent_to_source[latent_idx])
        frame = tensor_frame_to_uint8_hwc(context_video[:, src_idx]).copy()
        for obj_idx in range(gt_track_summary.shape[1]):
            if bool(pred_track_valid[latent_idx, obj_idx]):
                pred_center, pred_start = _summary_to_px(pred_track_summary[latent_idx, obj_idx], image_hw)
                draw_point_rgb(frame, pred_center, TRACK_PRED_COLOR, f"pred{obj_idx}", radius=5)
                draw_point_rgb(frame, pred_start, TRACK_PRED_COLOR, f"s{obj_idx}", radius=3)
            if bool(gt_track_valid[latent_idx, obj_idx]):
                gt_center, gt_start = _summary_to_px(gt_track_summary[latent_idx, obj_idx], image_hw)
                draw_point_rgb(frame, gt_center, TRACK_GT_COLOR, f"gt{obj_idx}", radius=5)
                draw_point_rgb(frame, gt_start, TRACK_GT_COLOR, f"gs{obj_idx}", radius=3)
        frames.append(frame)
    return np.stack(frames, axis=0)


def _render_depth_panel(
    gt_depth: np.ndarray,
    gt_depth_valid: np.ndarray,
    pred_depth: np.ndarray,
) -> np.ndarray:
    valid_values = []
    if gt_depth_valid.any():
        valid_values.append(gt_depth[gt_depth_valid])
    if gt_depth_valid.any():
        valid_values.append(pred_depth[gt_depth_valid])
    if valid_values:
        concat = np.concatenate(valid_values, axis=0)
        lo = float(np.min(concat))
        hi = float(np.max(concat))
        if hi - lo < 1.0e-6:
            hi = lo + 1.0
    else:
        lo, hi = 0.0, 1.0
    gt_map = np.where(gt_depth_valid, gt_depth, lo)
    pred_map = np.where(gt_depth_valid, pred_depth, lo)
    gt_vis = colorize_scalar_video(((gt_map - lo) / (hi - lo + 1.0e-6)).astype(np.float32))
    pred_vis = colorize_scalar_video(((pred_map - lo) / (hi - lo + 1.0e-6)).astype(np.float32))
    frames: list[np.ndarray] = []
    for t in range(gt_vis.shape[0]):
        left = gt_vis[t]
        right = pred_vis[t]
        pad = np.full((left.shape[0], 24, 3), 245, dtype=np.uint8)
        panel = np.concatenate([left, pad, right], axis=1)
        frames.append(panel)
    return np.stack(frames, axis=0)


def _prepare_case(
    trainer: ContextVideoTrainer,
    sample_index: int,
    checkpoint_path: Path,
    output_dir: Path,
    fps: int,
) -> dict[str, Any]:
    sample = trainer.dataset[int(sample_index)]
    batch = collate_video_batch([sample])
    with torch.no_grad():
        prepared = trainer._prepare_batch(batch)
    object_aux_out = prepared["object_aux_out"]
    context_video = batch["context_video"][0]
    image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))

    gt_track_summary = prepared["gt_track_summary"][0].detach().cpu().numpy()
    gt_track_valid = prepared["gt_track_valid"][0].detach().cpu().numpy() > 0.5
    gt_box_xyxy = prepared["gt_box_xyxy"][0].detach().cpu().numpy()
    gt_box_valid = prepared["gt_box_valid"][0].detach().cpu().numpy() > 0.5
    pred_track_summary = object_aux_out.pred_track_summary[0].detach().cpu().numpy()
    pred_box_xyxy = object_aux_out.pred_box_xyxy[0].detach().cpu().numpy()
    pred_depth = object_aux_out.pred_depth[0, ..., 0].detach().cpu().numpy()
    object_valid_mask = prepared.get("object_valid_mask")
    if object_valid_mask is None:
        pred_slot_valid = np.ones((pred_box_xyxy.shape[1],), dtype=bool)
    else:
        pred_slot_valid = object_valid_mask[0].detach().cpu().numpy() > 0.5
    pred_box_valid_mask = np.broadcast_to(pred_slot_valid[None, :], (pred_box_xyxy.shape[0], pred_box_xyxy.shape[1]))
    pred_track_valid_mask = np.broadcast_to(pred_slot_valid[None, :], (pred_track_summary.shape[0], pred_track_summary.shape[1]))

    gt_depth = None
    gt_depth_valid = None
    depth_video_rel = None
    if prepared["gt_depth"] is not None and prepared["gt_depth_valid"] is not None:
        gt_depth = prepared["gt_depth"][0, ..., 0].detach().cpu().numpy()
        gt_depth_valid = prepared["gt_depth_valid"][0].detach().cpu().numpy() > 0.5

    case_stem = f"case_{sample_index:05d}"
    assets_dir = output_dir / "assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    box_video = _render_box_overlay(context_video, gt_box_xyxy, gt_box_valid, pred_box_xyxy, pred_box_valid_mask, image_hw)
    box_raw = assets_dir / f"{case_stem}__box_overlay.mp4"
    write_mp4(box_raw, box_video, fps=fps)
    box_browser = ensure_browser_video(box_raw)

    track_video = _render_track_overlay(context_video, gt_track_summary, gt_track_valid, pred_track_summary, pred_track_valid_mask, image_hw)
    track_raw = assets_dir / f"{case_stem}__track_overlay.mp4"
    write_mp4(track_raw, track_video, fps=fps)
    track_browser = ensure_browser_video(track_raw)

    if gt_depth is not None and gt_depth_valid is not None:
        depth_video = _render_depth_panel(gt_depth, gt_depth_valid, pred_depth)
        depth_raw = assets_dir / f"{case_stem}__depth_panel.mp4"
        write_mp4(depth_raw, depth_video, fps=max(1, min(fps, 8)))
        depth_browser = ensure_browser_video(depth_raw)
        depth_video_rel = str(depth_browser.relative_to(output_dir))

    metrics = _compute_aux_metrics(prepared)
    result = {
        "sample_index": int(sample_index),
        "video_path": sample["video_path"],
        "caption": sample["caption"],
        "context_frame_indices": sample["context_frame_indices"].tolist(),
        "checkpoint": str(checkpoint_path),
        "box_overlay_video": str(box_browser.relative_to(output_dir)),
        "track_overlay_video": str(track_browser.relative_to(output_dir)),
        "depth_panel_video": depth_video_rel,
        "metrics": {
            "train/loss_track_aux": float(metrics.get("train/loss_track_aux", 0.0)),
            "train/loss_box_aux": float(metrics.get("train/loss_box_aux", 0.0)),
            "train/loss_depth_aux": float(metrics.get("train/loss_depth_aux", 0.0)),
            "train/track_box_loss": float(metrics.get("train/track_box_loss", 0.0)),
            "train/track_iou_loss": float(metrics.get("train/track_iou_loss", 0.0)),
        },
        "shapes": {
            "gt_track_summary": list(prepared["gt_track_summary"].shape) if prepared["gt_track_summary"] is not None else None,
            "gt_box_xyxy": list(prepared["gt_box_xyxy"].shape) if prepared["gt_box_xyxy"] is not None else None,
            "gt_depth": list(prepared["gt_depth"].shape) if prepared["gt_depth"] is not None else None,
            "pred_track_summary": list(object_aux_out.pred_track_summary.shape),
            "pred_box_xyxy": list(object_aux_out.pred_box_xyxy.shape),
            "pred_depth": list(object_aux_out.pred_depth.shape),
            "object_valid_mask": list(object_valid_mask.shape) if object_valid_mask is not None else None,
        },
    }
    return result


def _build_report(results: list[dict[str, Any]], output_dir: Path, checkpoint_path: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = output_dir / "metrics.json"
    summary = {
        "checkpoint": str(checkpoint_path),
        "num_cases": len(results),
        "cases": results,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    blocks: list[str] = []
    for result in results:
        depth_block = ""
        if result["depth_panel_video"] is not None:
            depth_block = f"""
      <figure>
        <video controls preload="none" playsinline src="{result['depth_panel_video']}"></video>
        <figcaption>Depth aux: GT depth vs Pred depth</figcaption>
      </figure>
"""
        blocks.append(
            f"""
  <section class="case">
    <h2>Case {result['sample_index']}</h2>
    <p><b>Caption:</b> {result['caption']}</p>
    <p><b>Context frames:</b> {result['context_frame_indices']}</p>
    <p><b>Video:</b> {result['video_path']}</p>
    <p><b>Losses:</b> track_aux={result['metrics']['train/loss_track_aux']:.6f}, box_aux={result['metrics']['train/loss_box_aux']:.6f}, depth_aux={result['metrics']['train/loss_depth_aux']:.6f}</p>
    <div class="video-grid">
      <figure>
        <video controls preload="none" playsinline src="{result['track_overlay_video']}"></video>
        <figcaption>Track aux: GT summary vs Pred summary on source frame</figcaption>
      </figure>
      <figure>
        <video controls preload="none" playsinline src="{result['box_overlay_video']}"></video>
        <figcaption>Box aux: GT box(red) vs Pred box(green)</figcaption>
      </figure>
      {depth_block}
    </div>
    <pre>{json.dumps({'metrics': result['metrics'], 'shapes': result['shapes']}, indent=2, ensure_ascii=False)}</pre>
  </section>
"""
        )

    html = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8">
  <title>Train Aux Loss Inspection</title>
  <style>
    body {{ font-family: sans-serif; margin: 20px; background: #f6f4ee; color: #222; }}
    .case {{ margin-bottom: 40px; padding-bottom: 20px; border-bottom: 1px solid #ddd; }}
    .video-grid {{ display: grid; grid-template-columns: repeat(2, minmax(320px, 1fr)); gap: 16px; align-items: start; }}
    .video-grid video {{ width: 100%; border: 1px solid #ccc; background: #000; }}
    figure {{ margin: 0; }}
    figcaption {{ font-size: 12px; color: #444; margin-top: 4px; }}
    pre {{ background: #fff; border: 1px solid #ddd; padding: 16px; white-space: pre-wrap; }}
  </style>
</head>
<body>
  <h1>Train Aux Loss Inspection</h1>
  <p>这页只展示当前训练里真实参与 `train/loss_track_aux`、`train/loss_box_aux`、`train/loss_depth_aux` 计算的量。不是额外定义的可视化代理，而是直接从 `ContextVideoTrainer._prepare_batch()` 和 `object_aux_heads` 的输出取值。</p>
  <p><b>Checkpoint:</b> {checkpoint_path}</p>
  {''.join(blocks)}
</body>
</html>
"""
    html_path = output_dir / "index.html"
    html_path.write_text(html, encoding="utf-8")
    return html_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint")
    parser.add_argument("--checkpoint-dir")
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--port", type=int, default=8810)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    checkpoint_path = _resolve_checkpoint_path(args)
    cfg = load_yaml_config(args.config)
    cfg["data"]["num_workers"] = 0
    cfg["data"]["batch_size"] = 1
    cfg["model"]["init_wan_lora_from_checkpoint"] = None
    trainer = ContextVideoTrainer(cfg, build_optimizer=False, device=args.device)
    torch.nn.Module.train(trainer, False)
    load_info = _load_aux_state_into_trainer(trainer, checkpoint_path)
    if load_info["missing_keys"] or load_info["unexpected_keys"]:
        print(json.dumps(load_info, indent=2, ensure_ascii=False))

    output_dir = Path(args.output_dir)
    results = []
    for sample_index in args.indices:
        results.append(
            _prepare_case(
                trainer=trainer,
                sample_index=int(sample_index),
                checkpoint_path=checkpoint_path,
                output_dir=output_dir,
                fps=int(args.fps),
            )
        )

    html_path = _build_report(results, output_dir, checkpoint_path)
    print(f"aux loss report: {html_path}")

    class Handler(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *handler_args, **handler_kwargs):
            super().__init__(*handler_args, directory=str(output_dir), **handler_kwargs)

    class ReusableTCPServer(socketserver.TCPServer):
        allow_reuse_address = True

    with ReusableTCPServer(("0.0.0.0", int(args.port)), Handler) as httpd:
        print(f"serving report at http://0.0.0.0:{int(args.port)}/index.html")
        httpd.serve_forever()


if __name__ == "__main__":
    main()
