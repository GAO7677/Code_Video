from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from code_vjepa_vggt.train_v_newtrain import build_accelerator, build_dataset, build_model, prepare_args, wan_parser


def _grad_report(model: torch.nn.Module) -> dict[str, Any]:
    num_trainable = 0
    with_grad = 0
    nonfinite_grad = []
    by_prefix: dict[str, dict[str, Any]] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        num_trainable += 1
        grad = param.grad
        prefix = name.split(".", 1)[0]
        bucket = by_prefix.setdefault(prefix, {"params": 0, "with_grad": 0, "grad_abs_max": 0.0})
        bucket["params"] += 1
        if grad is None:
            continue
        with_grad += 1
        grad_f = grad.detach().float()
        if not torch.isfinite(grad_f).all():
            nonfinite_grad.append(name)
        bucket["with_grad"] += 1
        bucket["grad_abs_max"] = max(bucket["grad_abs_max"], float(grad_f.abs().max().item()))
    return {
        "num_trainable": num_trainable,
        "with_grad": with_grad,
        "nonfinite_grad": nonfinite_grad,
        "by_prefix": by_prefix,
    }


def _mean_box_stats(box_xyxy: torch.Tensor, valid: torch.Tensor | None = None) -> dict[str, float]:
    if box_xyxy is None:
        return {}
    if valid is not None and tuple(valid.shape) == tuple(box_xyxy.shape[:-1]):
        valid_mask = valid > 0.5
    else:
        valid_mask = (box_xyxy[..., 2] > box_xyxy[..., 0]) & (box_xyxy[..., 3] > box_xyxy[..., 1])
    if not bool(valid_mask.any().item()):
        return {}
    boxes = box_xyxy[valid_mask]
    wh = (boxes[..., 2:] - boxes[..., :2]).clamp_min(0.0)
    center = 0.5 * (boxes[..., :2] + boxes[..., 2:])
    return {
        "mean_w": float(wh[..., 0].mean().item()),
        "mean_h": float(wh[..., 1].mean().item()),
        "mean_cx": float(center[..., 0].mean().item()),
        "mean_cy": float(center[..., 1].mean().item()),
    }


@torch.no_grad()
def _collect_object_debug(model_unwrapped: torch.nn.Module, sample: dict[str, Any]) -> dict[str, Any]:
    context_video = sample["context_video"].unsqueeze(0).to(device=model_unwrapped.pipe.device, dtype=model_unwrapped.pipe.torch_dtype)
    image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
    query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = model_unwrapped._build_object_query_priors(sample, image_hw=image_hw)
    query_points_prior = query_points_prior.to(device=model_unwrapped.pipe.device, dtype=model_unwrapped.pipe.torch_dtype)
    query_frame_ids = query_frame_ids.to(device=model_unwrapped.pipe.device, dtype=model_unwrapped.pipe.torch_dtype)
    object_valid_mask = object_valid_mask.to(device=model_unwrapped.pipe.device, dtype=model_unwrapped.pipe.torch_dtype)
    box_prior_xyxy = box_prior_xyxy.to(device=model_unwrapped.pipe.device, dtype=model_unwrapped.pipe.torch_dtype)
    frames_bthwc_01 = ((context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
    cotracker_out = model_unwrapped.cotracker_adapter(
        frames_bthwc_01,
        query_points_prior=query_points_prior,
        query_frame_ids=query_frame_ids,
        query_image_hw=image_hw,
    )
    tracks_grouped, visibility_grouped, confidence_grouped = model_unwrapped._group_tracks_to_objects(
        cotracker_out.tracks,
        cotracker_out.visibility,
        cotracker_out.confidence,
        max_objects=model_unwrapped.aux_max_objects,
        points_per_object=model_unwrapped.object_num_queries,
    )
    jepa_dtype = next(model_unwrapped.jepa_adapter.parameters()).dtype
    jepa_out = model_unwrapped.jepa_adapter(context_video.to(dtype=jepa_dtype))
    preprocessed_context_video = model_unwrapped.pipe.preprocess_video(
        [img for img in model_unwrapped.get_pipeline_inputs(sample)[0]["context_video"]]
    )
    clean_prefix_latents = model_unwrapped.pipe.vae.encode(
        preprocessed_context_video,
        device=model_unwrapped.pipe.device,
        tiled=True,
        tile_size=(30, 52),
        tile_stride=(15, 26),
    ).to(dtype=model_unwrapped.pipe.torch_dtype, device=model_unwrapped.pipe.device)
    object_out = model_unwrapped.object_pooler(
        jepa_patch_tokens=jepa_out.patch_tokens,
        context_latents=clean_prefix_latents,
        tracks=tracks_grouped,
        visibility=visibility_grouped,
        confidence=confidence_grouped,
        track_image_hw=image_hw,
        object_valid_mask=object_valid_mask,
        box_prior_xyxy=box_prior_xyxy,
        frame_valid_mask=None,
    )
    object_aux_out = model_unwrapped.object_aux_heads(
        object_out.object_latent_tokens,
        object_out.active_track_summary,
        object_out.active_box_xyxy,
    )
    object_context = model_unwrapped.object_adapter(
        object_out.object_latent_tokens,
        object_valid_mask=object_valid_mask,
    )
    gt_boxes = sample["context_boxes"].unsqueeze(0).to(device=model_unwrapped.pipe.device, dtype=model_unwrapped.pipe.torch_dtype)
    center_tracks_native, center_track_valid = model_unwrapped._object_center_tracks_from_grouped(
        tracks_grouped,
        visibility_grouped,
        confidence_grouped,
        object_valid_mask=object_valid_mask,
    )
    from code_vjepa_vggt.utils.track_supervision import align_tracks_to_boxes

    track_alignment = align_tracks_to_boxes(center_tracks_native, gt_boxes, image_hw=image_hw)
    latent_frames = int(object_out.object_latent_tokens.shape[1])
    gt_valid_full = (track_alignment.matched_gt_valid > 0.5) & center_track_valid
    gt_track_summary, gt_track_valid = model_unwrapped._group_track_summary(
        track_alignment.matched_gt_centers,
        gt_valid_full,
        image_hw=image_hw,
        latent_frames=latent_frames,
    )
    matched_gt_boxes = model_unwrapped._gather_matched_gt_features(gt_boxes, track_alignment.matched_gt_indices)
    matched_gt_box_valid = ((matched_gt_boxes[..., 2] - matched_gt_boxes[..., 0]) > 1.0e-6) & (
        (matched_gt_boxes[..., 3] - matched_gt_boxes[..., 1]) > 1.0e-6
    )
    gt_box_xyxy, gt_box_valid = model_unwrapped._group_box_targets(
        matched_gt_boxes,
        matched_gt_box_valid,
        latent_frames,
    )
    pred_box_valid = object_valid_mask[:, None, :].expand_as(object_aux_out.pred_box_xyxy[..., 0])
    pred_track_valid = object_valid_mask[:, None, :].expand_as(object_aux_out.pred_track_summary[..., 0]) > 0.5

    pred_track = object_aux_out.pred_track_summary.detach().float().cpu()
    gt_track = gt_track_summary.detach().float().cpu()
    active_track = object_out.active_track_summary.detach().float().cpu()
    gt_track_valid_mask = gt_track_valid.detach().cpu()
    return {
        "box_stats": {
            "pred": _mean_box_stats(object_aux_out.pred_box_xyxy.detach().float().cpu(), pred_box_valid.detach().cpu()),
            "gt": _mean_box_stats(gt_box_xyxy.detach().float().cpu(), gt_box_valid.detach().cpu()),
            "active_prior": _mean_box_stats(object_out.active_box_xyxy.detach().float().cpu(), pred_box_valid.detach().cpu()),
            "box_prior": _mean_box_stats(box_prior_xyxy.detach().float().cpu(), object_valid_mask.detach().cpu()),
        },
        "track_stats": {
            "pred_center_mean": [
                float(pred_track[..., 0][pred_track_valid.cpu()].mean().item()) if bool(pred_track_valid.any().item()) else 0.0,
                float(pred_track[..., 1][pred_track_valid.cpu()].mean().item()) if bool(pred_track_valid.any().item()) else 0.0,
            ],
            "gt_center_mean": [
                float(gt_track[..., 0][gt_track_valid_mask].mean().item()) if bool(gt_track_valid_mask.any().item()) else 0.0,
                float(gt_track[..., 1][gt_track_valid_mask].mean().item()) if bool(gt_track_valid_mask.any().item()) else 0.0,
            ],
            "active_center_mean": [
                float(active_track[..., 0][pred_track_valid.cpu()].mean().item()) if bool(pred_track_valid.any().item()) else 0.0,
                float(active_track[..., 1][pred_track_valid.cpu()].mean().item()) if bool(pred_track_valid.any().item()) else 0.0,
            ],
        },
        "object_context_abs_max": float(object_context.detach().abs().max().item()),
        "object_context_abs_mean": float(object_context.detach().abs().mean().item()),
        "box_prior_xyxy": box_prior_xyxy.detach().float().cpu().tolist(),
    }


def main() -> None:
    base_parser = wan_parser()
    parser = argparse.ArgumentParser(add_help=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--indices", type=int, nargs="+", default=[0])
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--log-every", type=int, default=1)
    args, remaining = parser.parse_known_args()

    train_args = base_parser.parse_args(remaining)
    train_args = prepare_args(train_args)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "history.json"

    accelerator = build_accelerator(train_args)
    dataset = build_dataset(train_args)
    model = build_model(train_args, accelerator)
    model.to(accelerator.device)
    model.train()

    params = list(model.trainable_modules())
    optimizer = torch.optim.AdamW(params, lr=float(train_args.learning_rate), weight_decay=float(train_args.weight_decay))

    history: list[dict[str, Any]] = []
    for step in range(int(args.num_steps)):
        sample_index = int(args.indices[step % len(args.indices)])
        sample = dataset[sample_index]
        seed = int(args.seed) + step
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        optimizer.zero_grad(set_to_none=True)
        loss = model.forward(sample)
        loss.backward()
        optimizer.step()

        model_unwrapped = accelerator.unwrap_model(model)
        metrics = dict(model_unwrapped.last_train_metrics)
        row: dict[str, Any] = {
            "step": step + 1,
            "sample_index": sample_index,
            "seed": seed,
            "loss": float(loss.detach().item()),
            "metrics": metrics,
        }
        if (step + 1) % int(args.log_every) == 0 or step == 0 or step + 1 == int(args.num_steps):
            row["grad_summary"] = _grad_report(model_unwrapped)
            row.update(_collect_object_debug(model_unwrapped, sample))
        history.append(row)
        print(json.dumps(row, ensure_ascii=False))

    payload = {
        "config_args": vars(train_args),
        "indices": [int(v) for v in args.indices],
        "num_steps": int(args.num_steps),
        "seed": int(args.seed),
        "history": history,
    }
    report_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"saved history to {report_path}")


if __name__ == "__main__":
    main()
