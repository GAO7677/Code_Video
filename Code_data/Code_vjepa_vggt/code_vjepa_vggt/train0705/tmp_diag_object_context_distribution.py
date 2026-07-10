from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import numpy as np
import torch

from code_vjepa_vggt import batch_infer_v_newtrain_from_jsonl as core
from code_vjepa_vggt.train0705 import infer_stage1b_context_only_no_gt_box_v_newtrain0705 as infer0705
from code_vjepa_vggt.train0705.tmp_cross_object_context_between_ckpts import (
    DEFAULT_BASE_LORA,
    DEFAULT_DIFFSYNTH_ROOT,
    DEFAULT_STAGE1A,
    DEFAULT_WAN_ROOT,
    _make_runtime_args,
    _resolve_runtime_device,
)
from code_vjepa_vggt.utils.video_io import preprocess_video_rgb_uint8, read_video_prefix, read_video_uniform


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose native object_latent_tokens/object_context distributions for two checkpoints "
            "on the same case."
        )
    )
    parser.add_argument("--checkpoint-a", type=Path, required=True)
    parser.add_argument("--checkpoint-b", type=Path, required=True)
    parser.add_argument("--input-json", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--lora-checkpoint", type=Path, default=DEFAULT_BASE_LORA)
    parser.add_argument("--stage1a-init-from", type=Path, default=DEFAULT_STAGE1A)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--input-cover-crop-height", type=int, default=480)
    parser.add_argument("--input-cover-crop-width", type=int, default=832)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--disable-object-branch", action="store_true")
    parser.add_argument("--object-num-queries", type=int, default=8)
    parser.add_argument("--aux-max-objects", type=int, default=4)
    parser.add_argument("--object-pooler-latent-dim", type=int, default=16)
    parser.add_argument("--cond-proj-dim", type=int, default=4096)
    parser.add_argument("--jepa-window-radius", type=int, default=1)
    parser.add_argument("--latent-window-radius", type=int, default=1)
    parser.add_argument("--object-gate-init", type=float, default=0.1)
    parser.add_argument(
        "--jepa-ckpt-path",
        default="/data/gaoya/ckpt/facebook-vjepa2-vitg-fpc64-384/original/model.pth",
    )
    parser.add_argument("--jepa-input-size", type=int, default=384)
    parser.add_argument("--jepa-patch-size", type=int, default=16)
    parser.add_argument("--jepa-tubelet-size", type=int, default=2)
    parser.add_argument(
        "--cotracker-checkpoint",
        default="/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth",
    )
    parser.add_argument("--cotracker-input-h", type=int, default=384)
    parser.add_argument("--cotracker-input-w", type=int, default=512)
    parser.add_argument("--cotracker-window-len", type=int, default=60)
    parser.add_argument("--vggt-model-path", default="/data/gaoya/ckpt/facebook-VGGT-1B")
    parser.add_argument("--vggt-input-h", type=int, default=420)
    parser.add_argument("--vggt-input-w", type=int, default=728)
    parser.add_argument("--vggt-cache-root", default=None)
    parser.add_argument("--aux-device", default=None)
    parser.add_argument("--grounding-device", default=None)
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
    parser.add_argument("--initialize-model-on-cpu", action="store_true")
    infer0705.add_vjepa_cli_args(parser)
    return parser.parse_args()


def _resolve_source_video(payload: dict[str, object], json_path: Path) -> Path:
    source_video = payload.get("source_video")
    if isinstance(source_video, str) and source_video.strip():
        return Path(source_video.strip()).expanduser().resolve()
    return Path(core._resolve_input_video(payload, json_path)).expanduser().resolve()


def _load_context(args: argparse.Namespace, source_video: Path) -> tuple[np.ndarray, np.ndarray, torch.Tensor]:
    if str(args.sampling_mode) == "uniform":
        frames, frame_indices = read_video_uniform(source_video, int(args.context_frames))
    else:
        frames, frame_indices = read_video_prefix(source_video, int(args.context_frames))
    if int(frames.shape[0]) > int(args.context_frames):
        frames = frames[: int(args.context_frames)]
        frame_indices = frame_indices[: int(args.context_frames)]
    context_video_single = preprocess_video_rgb_uint8(
        frames,
        (int(args.height), int(args.width)),
        resize_mode="cover_crop",
        cover_crop_hw=(int(args.input_cover_crop_height), int(args.input_cover_crop_width)),
    )
    return frames, frame_indices, context_video_single


def _tensor_stats(tensor: torch.Tensor) -> dict[str, object]:
    x = tensor.detach().float()
    return {
        "shape": list(tensor.shape),
        "mean": float(x.mean().item()),
        "std": float(x.std(unbiased=False).item()),
        "abs_mean": float(x.abs().mean().item()),
        "abs_max": float(x.abs().max().item()),
    }


def _norm_stats(tensor: torch.Tensor) -> dict[str, object]:
    x = tensor.detach().float()
    norms = torch.linalg.norm(x, dim=-1)
    return {
        "shape": list(norms.shape),
        "mean": float(norms.mean().item()),
        "std": float(norms.std(unbiased=False).item()),
        "min": float(norms.min().item()),
        "max": float(norms.max().item()),
        "values": norms.cpu().tolist(),
    }


def _reshape_context_by_slots(object_context: torch.Tensor, slot_count: int) -> torch.Tensor:
    seq_len = int(object_context.shape[1])
    if seq_len % int(slot_count) != 0:
        raise ValueError(f"object_context seq_len={seq_len} not divisible by slot_count={slot_count}")
    time_steps = seq_len // int(slot_count)
    return object_context.view(object_context.shape[0], time_steps, int(slot_count), object_context.shape[-1])


def _pairwise_slot_cosine(tokens_tsd: torch.Tensor, valid_mask_s: torch.Tensor) -> dict[str, object]:
    x = tokens_tsd.detach().float().cpu()
    valid = valid_mask_s.detach().float().cpu() > 0.5
    per_time = []
    all_off_diag: list[float] = []
    for t in range(x.shape[0]):
        xt = x[t]
        xt = xt / torch.clamp(torch.linalg.norm(xt, dim=-1, keepdim=True), min=1.0e-12)
        sim = xt @ xt.transpose(0, 1)
        sim_np = sim.numpy()
        valid_ids = [idx for idx, ok in enumerate(valid.tolist()) if ok]
        off_diag: list[float] = []
        for i in valid_ids:
            for j in valid_ids:
                if i >= j:
                    continue
                off_diag.append(float(sim_np[i, j]))
                all_off_diag.append(float(sim_np[i, j]))
        per_time.append(
            {
                "time_index": int(t),
                "valid_slot_ids": valid_ids,
                "matrix": sim_np.tolist(),
                "off_diag_mean": None if not off_diag else float(np.mean(off_diag)),
                "off_diag_min": None if not off_diag else float(np.min(off_diag)),
                "off_diag_max": None if not off_diag else float(np.max(off_diag)),
            }
        )
    return {
        "per_time": per_time,
        "all_off_diag_mean": None if not all_off_diag else float(np.mean(all_off_diag)),
        "all_off_diag_min": None if not all_off_diag else float(np.min(all_off_diag)),
        "all_off_diag_max": None if not all_off_diag else float(np.max(all_off_diag)),
    }


def _temporal_stats(tokens_tsd: torch.Tensor, valid_mask_s: torch.Tensor) -> dict[str, object]:
    x = tokens_tsd.detach().float().cpu()
    valid = valid_mask_s.detach().float().cpu() > 0.5
    if x.shape[0] <= 1:
        return {"time_steps": int(x.shape[0]), "per_slot": [], "mean_pairwise_time_cosine": None}
    per_slot = []
    all_cos: list[float] = []
    all_l2: list[float] = []
    var_per_slot = torch.var(x, dim=0, unbiased=False)
    for slot_id in range(x.shape[1]):
        if not bool(valid[slot_id].item()):
            continue
        slot_ts = x[:, slot_id, :]
        cos_vals: list[float] = []
        l2_vals: list[float] = []
        for t0 in range(slot_ts.shape[0]):
            for t1 in range(t0 + 1, slot_ts.shape[0]):
                a = slot_ts[t0]
                b = slot_ts[t1]
                cos = torch.nn.functional.cosine_similarity(a.view(1, -1), b.view(1, -1), dim=1).item()
                l2 = torch.linalg.norm(a - b).item()
                cos_vals.append(float(cos))
                l2_vals.append(float(l2))
                all_cos.append(float(cos))
                all_l2.append(float(l2))
        per_slot.append(
            {
                "slot_id": int(slot_id),
                "time_var_mean": float(var_per_slot[slot_id].mean().item()),
                "time_var_abs_mean": float(var_per_slot[slot_id].abs().mean().item()),
                "pairwise_time_cosine_mean": None if not cos_vals else float(np.mean(cos_vals)),
                "pairwise_time_cosine_min": None if not cos_vals else float(np.min(cos_vals)),
                "pairwise_time_l2_mean": None if not l2_vals else float(np.mean(l2_vals)),
                "pairwise_time_l2_max": None if not l2_vals else float(np.max(l2_vals)),
            }
        )
    return {
        "time_steps": int(x.shape[0]),
        "per_slot": per_slot,
        "mean_pairwise_time_cosine": None if not all_cos else float(np.mean(all_cos)),
        "min_pairwise_time_cosine": None if not all_cos else float(np.min(all_cos)),
        "mean_pairwise_time_l2": None if not all_l2 else float(np.mean(all_l2)),
        "max_pairwise_time_l2": None if not all_l2 else float(np.max(all_l2)),
    }


def _compare_tensors(a: torch.Tensor, b: torch.Tensor) -> dict[str, object]:
    af = a.detach().float().reshape(1, -1)
    bf = b.detach().float().reshape(1, -1)
    cosine = torch.nn.functional.cosine_similarity(af, bf, dim=1).item()
    mse = torch.mean((af - bf) ** 2).item()
    return {
        "cosine": float(cosine),
        "mse": float(mse),
    }


def _compare_aligned_tokens(a: torch.Tensor, b: torch.Tensor, valid_mask: torch.Tensor) -> dict[str, object]:
    af = a.detach().float().cpu()
    bf = b.detach().float().cpu()
    valid = valid_mask.detach().float().cpu() > 0.5
    per_slot = []
    all_cos: list[float] = []
    all_mse: list[float] = []
    for t in range(af.shape[0]):
        for s in range(af.shape[1]):
            if not bool(valid[s].item()):
                continue
            av = af[t, s]
            bv = bf[t, s]
            cos = torch.nn.functional.cosine_similarity(av.view(1, -1), bv.view(1, -1), dim=1).item()
            mse = torch.mean((av - bv) ** 2).item()
            per_slot.append(
                {
                    "time_index": int(t),
                    "slot_id": int(s),
                    "cosine": float(cos),
                    "mse": float(mse),
                }
            )
            all_cos.append(float(cos))
            all_mse.append(float(mse))
    return {
        "entries": per_slot,
        "mean_cosine": None if not all_cos else float(np.mean(all_cos)),
        "min_cosine": None if not all_cos else float(np.min(all_cos)),
        "mean_mse": None if not all_mse else float(np.mean(all_mse)),
        "max_mse": None if not all_mse else float(np.max(all_mse)),
    }


def _build_native_outputs(
    *,
    args: argparse.Namespace,
    checkpoint_dir: Path,
    context_video_single: torch.Tensor,
    prompt: str,
    video_path: Path,
) -> dict[str, object]:
    runtime_args = _make_runtime_args(args, checkpoint_dir, args.output_root / "_tmp_runtime")
    model, _, load_info = infer0705._build_runtime_model(runtime_args)
    try:
        pipe = model.pipe
        device = torch.device(pipe.device)
        context_video = context_video_single.unsqueeze(0).to(device=device, dtype=pipe.torch_dtype)
        image_hw = (int(context_video.shape[-2]), int(context_video.shape[-1]))
        sample = {
            "context_video": context_video_single,
            "num_context_frames": int(context_video_single.shape[1]),
            "caption": prompt,
            "video_path": str(video_path),
        }
        with torch.no_grad():
            query_points_prior, query_frame_ids, object_valid_mask, box_prior_xyxy = (
                model._build_object_query_priors(sample, image_hw=image_hw)
            )
            query_points_prior = query_points_prior.to(device=device, dtype=pipe.torch_dtype)
            query_frame_ids = query_frame_ids.to(device=device, dtype=pipe.torch_dtype)
            object_valid_mask = object_valid_mask.to(device=device, dtype=pipe.torch_dtype)
            box_prior_xyxy = box_prior_xyxy.to(device=device, dtype=pipe.torch_dtype)

            frames_bthwc_01 = ((context_video.permute(0, 2, 3, 4, 1).float() + 1.0) / 2.0).clamp(0.0, 1.0)
            cotracker_out = model._run_cotracker(
                frames_bthwc_01,
                query_points_prior=query_points_prior,
                query_frame_ids=query_frame_ids,
                query_image_hw=image_hw,
            )

            if model.vggt_cache_root:
                vggt_out = infer0705.load_vggt_cache(sample, model.vggt_cache_root, allow_missing=False)
                if vggt_out is None:
                    raise RuntimeError(f"VGGT cache missing for {video_path}")
            else:
                vggt_device = getattr(model.vggt_adapter, "device_obj", device)
                vggt_out = model.vggt_adapter(
                    frames_bthwc_01.to(vggt_device),
                    query_points_prior=query_points_prior.to(vggt_device),
                    query_image_hw=image_hw,
                )
                for attr_name in (
                    "query_points",
                    "tracks",
                    "visibility",
                    "confidence",
                    "pose_enc",
                    "depth",
                    "depth_conf",
                    "world_points",
                    "world_points_conf",
                    "dense_patch_tokens",
                ):
                    attr_value = getattr(vggt_out, attr_name, None)
                    if isinstance(attr_value, torch.Tensor):
                        setattr(vggt_out, attr_name, attr_value.to(device))

            tracks_grouped, visibility_grouped, confidence_grouped = model._group_tracks_to_objects(
                cotracker_out.tracks,
                cotracker_out.visibility,
                cotracker_out.confidence,
                max_objects=model.aux_max_objects,
                points_per_object=model.object_num_queries,
            )
            jepa_out = model._run_jepa(context_video)
            clean_prefix_latents = infer0705._encode_context_latents(pipe, context_video_single)
            object_out = model.object_pooler(
                jepa_patch_tokens=jepa_out.patch_tokens,
                context_latents=clean_prefix_latents,
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
            object_context = model.object_adapter(
                object_out.object_latent_tokens,
                object_valid_mask=object_valid_mask,
            )

        object_latent_tokens = object_out.object_latent_tokens.detach().cpu()
        object_context = object_context.detach().cpu()
        object_valid_mask = object_valid_mask.detach().cpu()
        slot_count = int(object_latent_tokens.shape[2])
        context_reshaped = _reshape_context_by_slots(object_context, slot_count=slot_count).squeeze(0)
        latent_reshaped = object_latent_tokens.squeeze(0)
        valid_slots = object_valid_mask.squeeze(0)

        result = {
            "checkpoint": str(checkpoint_dir),
            "load_info": infer0705._summarize_load_info(load_info),
            "slot_count": int(slot_count),
            "valid_slot_ids": [int(idx) for idx, ok in enumerate((valid_slots > 0.5).tolist()) if ok],
            "object_valid_mask": object_valid_mask.tolist(),
            "object_latent_tokens_stats": _tensor_stats(object_latent_tokens),
            "object_context_stats": _tensor_stats(object_context),
            "object_latent_tokens_norm": _norm_stats(latent_reshaped),
            "object_context_norm": _norm_stats(context_reshaped),
            "object_latent_tokens_temporal": _temporal_stats(latent_reshaped, valid_slots),
            "object_context_temporal": _temporal_stats(context_reshaped, valid_slots),
            "object_latent_tokens_slot_similarity": _pairwise_slot_cosine(latent_reshaped, valid_slots),
            "object_context_slot_similarity": _pairwise_slot_cosine(context_reshaped, valid_slots),
            "raw_tensors": {
                "object_latent_tokens": object_latent_tokens,
                "object_context": object_context,
                "object_context_reshaped": context_reshaped,
                "object_valid_mask": valid_slots,
            },
        }
        return result
    finally:
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    infer0705.apply_vjepa_preset_if_requested(args)
    args.device = _resolve_runtime_device(args.device)

    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    ckpt_a = args.checkpoint_a.expanduser().resolve()
    ckpt_b = args.checkpoint_b.expanduser().resolve()
    payload = core._load_input_json(args.input_json)
    source_video = _resolve_source_video(payload, args.input_json)
    input_caption = core._ensure_str_field(payload, "input_caption", args.input_json)
    _, frame_indices, context_video_single = _load_context(args, source_video)

    a = _build_native_outputs(
        args=args,
        checkpoint_dir=ckpt_a,
        context_video_single=context_video_single,
        prompt=input_caption,
        video_path=source_video,
    )
    b = _build_native_outputs(
        args=args,
        checkpoint_dir=ckpt_b,
        context_video_single=context_video_single,
        prompt=input_caption,
        video_path=source_video,
    )

    a_lat = a["raw_tensors"]["object_latent_tokens"]
    b_lat = b["raw_tensors"]["object_latent_tokens"]
    a_ctx = a["raw_tensors"]["object_context"]
    b_ctx = b["raw_tensors"]["object_context"]
    a_ctx_rs = a["raw_tensors"]["object_context_reshaped"]
    b_ctx_rs = b["raw_tensors"]["object_context_reshaped"]
    valid_slots = a["raw_tensors"]["object_valid_mask"]

    comparison = {
        "object_latent_tokens_global": _compare_tensors(a_lat, b_lat),
        "object_context_global": _compare_tensors(a_ctx, b_ctx),
        "object_latent_tokens_aligned": _compare_aligned_tokens(a_lat.squeeze(0), b_lat.squeeze(0), valid_slots),
        "object_context_aligned": _compare_aligned_tokens(a_ctx_rs, b_ctx_rs, valid_slots),
    }

    summary = {
        "input_json": str(args.input_json),
        "source_video": str(source_video),
        "input_caption": str(input_caption),
        "frame_indices": frame_indices.tolist(),
        "checkpoint_a": str(ckpt_a),
        "checkpoint_b": str(ckpt_b),
        "native_a": {k: v for k, v in a.items() if k != "raw_tensors"},
        "native_b": {k: v for k, v in b.items() if k != "raw_tensors"},
        "comparison": comparison,
    }
    summary_path = output_root / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "comparison": comparison}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
