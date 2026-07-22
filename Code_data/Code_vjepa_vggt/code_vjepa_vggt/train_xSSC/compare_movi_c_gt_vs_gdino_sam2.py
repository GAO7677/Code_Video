#!/usr/bin/env python3
"""Compare GT-box and GDINO+SAM2-box conditioning on fixed MOVi-C cases."""

from __future__ import annotations

import argparse
import csv
import gc
import html
import json
from pathlib import Path
import random
import sys
import time

import cv2
import imageio_ffmpeg
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn.functional as torch_f


TRAIN_XSSC_ROOT = Path(__file__).resolve().parent
REPO_ROOT = TRAIN_XSSC_ROOT.parent
PACKAGE_PARENT = REPO_ROOT.parent
EXPERIMENT = TRAIN_XSSC_ROOT / "xssc_rsfq2_ytvis_dinov3_vitl16_256"
sys.path.insert(0, str(PACKAGE_PARENT))
sys.path.insert(0, str(EXPERIMENT / "third_party/dinov3"))
sys.path.insert(0, str(EXPERIMENT / "upstream"))

DEFAULT_CONFIG = EXPERIMENT / (
    "upstream/config-randsfq/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000.py"
)
DEFAULT_SUBSET = Path(
    "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/"
    "dinov3_xSSC/restart_save1000_20260720T140029Z/"
    "movi_c_transfer15000_b64_acc3_20260721T134713Z/"
    "rsfq2_c-movi_c-dinov3_vitl16_256-slot512-transfer15000/42/"
    "val_subset.json"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/"
    "movi_c_gt_vs_gdino_sam2_fixed5_20260722"
)
PALETTE = np.asarray(
    [
        [239, 68, 68],
        [59, 130, 246],
        [34, 197, 94],
        [250, 204, 21],
        [168, 85, 247],
        [6, 182, 212],
        [249, 115, 22],
        [236, 72, 153],
        [132, 204, 22],
        [20, 184, 166],
        [244, 114, 182],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--subset-file", type=Path, default=DEFAULT_SUBSET)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--num-cases", type=int, default=5)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--amp-dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--fps", type=float, default=12.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--text-prompt", default="object . item .")
    parser.add_argument("--gdino-box-threshold", type=float, default=0.20)
    parser.add_argument("--gdino-text-threshold", type=float, default=0.15)
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_lightweight_checkpoint(model, checkpoint):
    state = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
    incompatible = model.load_state_dict(state, strict=False)
    missing = [
        key
        for key in incompatible.missing_keys
        if not key.startswith("m.encode_backbone.")
    ]
    if missing or incompatible.unexpected_keys:
        raise RuntimeError(
            f"checkpoint mismatch: missing={missing}, "
            f"unexpected={incompatible.unexpected_keys}"
        )
    report = {
        "loaded_keys": len(state),
        "missing_frozen_backbone_keys": len(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
    del state
    gc.collect()
    return report


def metric_values(metric_output):
    result = {}
    for key, (value, valid) in metric_output.items():
        selected = value[valid]
        if selected.numel() == 0:
            raise RuntimeError(f"metric {key} has no valid values")
        result[key] = float(selected.float().mean().item())
    return result


def condition_box_stats(gt_boxes, pseudo_boxes):
    gt = gt_boxes[0]
    pseudo = pseudo_boxes[0]
    gt = gt[(gt[:, 2] > gt[:, 0]) & (gt[:, 3] > gt[:, 1])]
    pseudo = pseudo[(pseudo[:, 2] > pseudo[:, 0]) & (pseudo[:, 3] > pseudo[:, 1])]
    if len(gt) == 0:
        return {
            "gt_count": 0,
            "pseudo_count": int(len(pseudo)),
            "gt_covered_mean_iou": 1.0,
            "gt_recall_at_50": 1.0,
        }
    ious = np.zeros((len(gt), len(pseudo)), dtype=np.float32)
    for gt_index, gt_box in enumerate(gt):
        for pseudo_index, pseudo_box in enumerate(pseudo):
            top_left = np.maximum(gt_box[:2], pseudo_box[:2])
            bottom_right = np.minimum(gt_box[2:], pseudo_box[2:])
            intersection = np.prod(np.maximum(bottom_right - top_left, 0.0))
            gt_area = np.prod(np.maximum(gt_box[2:] - gt_box[:2], 0.0))
            pseudo_area = np.prod(np.maximum(pseudo_box[2:] - pseudo_box[:2], 0.0))
            ious[gt_index, pseudo_index] = intersection / max(
                gt_area + pseudo_area - intersection, 1.0e-8
            )
    covered = np.zeros(len(gt), dtype=np.float32)
    if len(pseudo):
        gt_ids, pseudo_ids = linear_sum_assignment(-ious)
        covered[gt_ids] = ious[gt_ids, pseudo_ids]
    return {
        "gt_count": int(len(gt)),
        "pseudo_count": int(len(pseudo)),
        "gt_covered_mean_iou": float(covered.mean()),
        "gt_recall_at_50": float((covered >= 0.5).mean()),
    }


def align_slot_labels(anchor, target, num_slots):
    pair_iou = np.zeros((num_slots, num_slots), dtype=np.float64)
    for anchor_slot in range(num_slots):
        anchor_mask = anchor == anchor_slot
        for target_slot in range(num_slots):
            target_mask = target == target_slot
            union = np.logical_or(anchor_mask, target_mask).sum()
            if union:
                pair_iou[anchor_slot, target_slot] = (
                    np.logical_and(anchor_mask, target_mask).sum() / union
                )
    anchor_ids, target_ids = linear_sum_assignment(-pair_iou)
    aligned = np.empty_like(target)
    mapping = {}
    for anchor_id, target_id in zip(anchor_ids, target_ids):
        aligned[target == target_id] = anchor_id
        mapping[int(target_id)] = int(anchor_id)
    return aligned, mapping


def add_patch_grid(frames, patch_size=16):
    result = frames.copy()
    result[:, patch_size::patch_size, :, :] = (
        result[:, patch_size::patch_size, :, :].astype(np.float32) * 0.70
    ).astype(np.uint8)
    result[:, :, patch_size::patch_size, :] = (
        result[:, :, patch_size::patch_size, :].astype(np.float32) * 0.70
    ).astype(np.uint8)
    return result


def slot_overlay(video, labels):
    labels_full = labels.repeat(16, axis=1).repeat(16, axis=2)
    colors = PALETTE[labels_full % len(PALETTE)]
    output = (
        video.astype(np.float32) * 0.50 + colors.astype(np.float32) * 0.50
    ).round().clip(0, 255).astype(np.uint8)
    return add_patch_grid(output)


def gt_mask_overlay(video, segment):
    labels = segment.argmax(axis=-1)
    colors = np.zeros_like(video)
    foreground = labels > 0
    colors[foreground] = PALETTE[(labels[foreground] - 1) % len(PALETTE)]
    output = video.copy().astype(np.float32)
    output[foreground] = (
        output[foreground] * 0.45 + colors[foreground].astype(np.float32) * 0.55
    )
    return output.round().clip(0, 255).astype(np.uint8)


def normalized_box_to_pixels(box, height, width):
    x0, y0, x1, y1 = [float(value) for value in box]
    if x1 <= x0 or y1 <= y0:
        return None
    return (
        int(np.clip(round(x0 * width), 0, width - 1)),
        int(np.clip(round(y0 * height), 0, height - 1)),
        int(np.clip(round(x1 * width), 0, width - 1)),
        int(np.clip(round(y1 * height), 0, height - 1)),
    )


def draw_boxes(video, boxes, prefix):
    rendered = []
    for frame, frame_boxes in zip(video, boxes):
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        height, width = bgr.shape[:2]
        for slot_id, box in enumerate(frame_boxes):
            pixels = normalized_box_to_pixels(box, height, width)
            if pixels is None:
                continue
            color = tuple(int(value) for value in PALETTE[slot_id % len(PALETTE)][::-1])
            cv2.rectangle(bgr, pixels[:2], pixels[2:], color, 2, cv2.LINE_AA)
            cv2.putText(
                bgr,
                f"{prefix}{slot_id + 1}",
                (pixels[0] + 2, max(pixels[1] + 13, 13)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                color,
                1,
                cv2.LINE_AA,
            )
        rendered.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
    return np.stack(rendered)


def add_title(frame, title):
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 21), (18, 18, 18), -1)
    cv2.putText(
        output,
        title,
        (5, 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.40,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    return output


def write_video(path, frames, fps):
    height, width = frames.shape[1:3]
    writer = imageio_ffmpeg.write_frames(
        str(path),
        (width, height),
        fps=fps,
        codec="libx264",
        pix_fmt_in="rgb24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "18", "-movflags", "+faststart"],
    )
    writer.send(None)
    try:
        for frame in frames:
            writer.send(np.ascontiguousarray(frame))
    finally:
        writer.close()


def make_contact_sheet(frames):
    frame_ids = [0, len(frames) // 2, len(frames) - 1]
    return np.concatenate([frames[index] for index in frame_ids], axis=0)


def infer_branch(model, metric, video, segment, boxes, cfg, device, amp_dtype):
    batch = {
        "video": video.to(device, non_blocking=True),
        "segment": segment.to(device, non_blocking=True),
        "bbox": torch.from_numpy(boxes).to(device),
    }
    with torch.inference_mode(), torch.autocast("cuda", dtype=amp_dtype):
        output = model(batch=batch)
        output["segment"] = torch_f.one_hot(
            cfg.interpolat_argmax_attent(
                output["attentd"].detach(), size=cfg.resolut0
            ).long()
        ).bool()
        values = metric_values(metric(batch=batch, output=output))
    attention = output["attentd"][0].detach().float().cpu().numpy()
    labels = attention.argmax(axis=1).astype(np.uint8)
    return values, labels, list(attention.shape)


def build_html(payload):
    rows = []
    for case in payload["cases"]:
        gt = case["metrics"]["gt_box"]
        pseudo = case["metrics"]["pseudo_box"]
        rows.append(
            "<tr>"
            f"<td>{case['dataset_index']}</td>"
            f"<td>{gt['ari_fg']:.4f}</td><td>{pseudo['ari_fg']:.4f}</td>"
            f"<td>{gt['mbo']:.4f}</td><td>{pseudo['mbo']:.4f}</td>"
            f"<td>{gt['miou']:.4f}</td><td>{pseudo['miou']:.4f}</td>"
            f"<td>{case['box_stats']['gt_count']}</td>"
            f"<td>{case['box_stats']['pseudo_count']}</td>"
            "</tr>"
        )
    cards = []
    for case in payload["cases"]:
        cards.append(
            f"<article><h2>test index {case['dataset_index']:03d}</h2>"
            f"<p>Frame-0 GT boxes {case['box_stats']['gt_count']}; pseudo boxes "
            f"{case['box_stats']['pseudo_count']}; pseudo recall@0.5 "
            f"{case['box_stats']['gt_recall_at_50']:.3f}</p>"
            f"<video controls muted loop playsinline preload='metadata' "
            f"src='{html.escape(case['video'])}'></video>"
            f"<img loading='lazy' src='{html.escape(case['contact_sheet'])}' "
            "alt='GT and pseudo box-conditioned slot comparison'></article>"
        )
    summary = payload["summary"]
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MOVi-C GT vs pseudo boxes</title><style>
body{{margin:0;background:#101214;color:#f3f4f6;font:14px Arial,sans-serif}}
main{{max-width:1320px;margin:auto;padding:24px}}h1{{font-size:24px}}h2{{font-size:17px}}
p{{color:#b7bec8}}table{{border-collapse:collapse;width:100%;margin:18px 0}}
th,td{{border:1px solid #3a4048;padding:7px;text-align:right}}th:first-child,td:first-child{{text-align:left}}
article{{border-top:1px solid #343a40;padding:18px 0}}video,img{{display:block;width:100%;height:auto;margin-top:10px;background:#000}}
</style></head><body><main><h1>MOVi-C GT box vs GDINO+SAM2 pseudo box</h1>
<p>Checkpoint: {html.escape(Path(payload['checkpoint']).name)}. All metrics are higher-is-better and use the official xSSC MOVi-C validation implementation.</p>
<p>Mean: GT ARI-FG {summary['gt_box']['ari_fg']:.4f}, mBO {summary['gt_box']['mbo']:.4f}, mIoU {summary['gt_box']['miou']:.4f}; pseudo ARI-FG {summary['pseudo_box']['ari_fg']:.4f}, mBO {summary['pseudo_box']['mbo']:.4f}, mIoU {summary['pseudo_box']['miou']:.4f}.</p>
<table><thead><tr><th>case</th><th>GT ARI-FG</th><th>Pseudo ARI-FG</th><th>GT mBO</th><th>Pseudo mBO</th><th>GT mIoU</th><th>Pseudo mIoU</th><th>GT boxes</th><th>Pseudo boxes</th></tr></thead><tbody>{''.join(rows)}</tbody></table>
{''.join(cards)}</main></body></html>"""


def main():
    args = parse_args()
    if args.num_cases <= 0:
        raise ValueError("num-cases must be positive")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    set_seed(args.seed)

    from code_vjepa_vggt.object_token_teacher_student.viewer_grounding_box_provider import (
        ViewerGroundingBoxProvider,
    )
    from object_centric_bench.datum import MOViTFRecord
    from object_centric_bench.learn import MetricWrap
    from object_centric_bench.model import ModelWrap
    from object_centric_bench.util import Config, build_from_config
    from object_centric_bench.util_model import interpolat_argmax_attent

    config_file = args.config_file.resolve()
    checkpoint = args.checkpoint.resolve()
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    cfg = Config.fromfile(config_file)
    cfg.dataset_v.base_dir = args.data_dir.resolve()
    cfg.interpolat_argmax_attent = interpolat_argmax_attent
    dataset = build_from_config(cfg.dataset_v)
    raw_dataset = MOViTFRecord(
        data_file="kubric-movi/movi-c",
        split="test",
        extra_keys=["segment", "bbox"],
        base_dir=args.data_dir.resolve(),
    )
    collate_fn = build_from_config(cfg.collate_fn_v)
    subset = json.loads(args.subset_file.read_text())
    indices = [int(index) for index in subset["indices"][: args.num_cases]]

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    model = ModelWrap(
        build_from_config(cfg.model), cfg.model_imap, cfg.model_omap
    ).to(device).eval()
    model.freez(cfg.freez, verbose=False)
    checkpoint_report = load_lightweight_checkpoint(model, checkpoint)
    metric = MetricWrap(detach=True, **build_from_config(cfg.acc_fn_v)).to(device)
    amp_dtype = getattr(torch, args.amp_dtype)

    provider = ViewerGroundingBoxProvider(
        device=str(device),
        segment_len=8,
        max_objects=cfg.num_slots,
        points_per_object=8,
        proposal_source="gdino_only",
        motion_score_ratio=0.15,
        text_prompt=args.text_prompt,
        extra_prompt_terms="",
        include_caption_terms=False,
        gdino_box_threshold=args.gdino_box_threshold,
        gdino_text_threshold=args.gdino_text_threshold,
        prompt_frame_mode="first",
        track_dedupe_iou_threshold=0.75,
        container_suppress_ratio_threshold=0.95,
        container_suppress_min_contained=2,
        container_suppress_min_area_ratio=1.5,
        container_suppress_small_iou_threshold=0.7,
    )

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    metric_names = ("ari", "ari_fg", "mbo", "miou")
    totals = {
        branch: {metric_name: 0.0 for metric_name in metric_names}
        for branch in ("gt_box", "pseudo_box")
    }

    for position, dataset_index in enumerate(indices, start=1):
        started = time.time()
        batch = collate_fn([dataset[dataset_index]])
        video = batch["video"]
        segment = batch["segment"]
        gt_boxes = batch["bbox"].numpy()
        gt_metrics, gt_labels, attention_shape = infer_branch(
            model, metric, video, segment, gt_boxes, cfg, device, amp_dtype
        )

        raw_sample = raw_dataset[dataset_index]
        raw_video = raw_sample["video"].float().numpy() / 255.0
        pseudo_started = time.time()
        pseudo_sample = provider.build_sample(
            frames_tchw_01=raw_video,
            caption="",
            image_hw=(raw_video.shape[-2], raw_video.shape[-1]),
        )
        pseudo_seconds = time.time() - pseudo_started

        pseudo_boxes = pseudo_sample.context_boxes_norm[None]
        pseudo_metrics, pseudo_labels_raw, _ = infer_branch(
            model, metric, video, segment, pseudo_boxes, cfg, device, amp_dtype
        )
        pseudo_labels, slot_mapping = align_slot_labels(
            gt_labels, pseudo_labels_raw, cfg.num_slots
        )
        for metric_name in metric_names:
            totals["gt_box"][metric_name] += gt_metrics[metric_name]
            totals["pseudo_box"][metric_name] += pseudo_metrics[metric_name]

        video_rgb = (
            raw_sample["video"].permute(0, 2, 3, 1).contiguous().numpy()
        )
        segment_np = raw_sample["segment"].numpy()
        gt_box_panel = draw_boxes(video_rgb, gt_boxes[0], "G")
        pseudo_box_panel = draw_boxes(video_rgb, pseudo_boxes[0], "P")
        gt_slot_panel = slot_overlay(video_rgb, gt_labels)
        pseudo_slot_panel = slot_overlay(video_rgb, pseudo_labels)
        gt_mask_panel = gt_mask_overlay(video_rgb, segment_np)
        rendered = []
        for frame_index in range(len(video_rgb)):
            panels = [
                add_title(gt_mask_panel[frame_index], f"GT mask | f{frame_index:02d}"),
                add_title(gt_box_panel[frame_index], f"GT boxes | f{frame_index:02d}"),
                add_title(gt_slot_panel[frame_index], "slots from GT boxes"),
                add_title(pseudo_box_panel[frame_index], f"pseudo boxes | f{frame_index:02d}"),
                add_title(pseudo_slot_panel[frame_index], "slots from pseudo boxes"),
            ]
            rendered.append(np.concatenate(panels, axis=1))
        rendered = np.stack(rendered)

        stem = f"case_{position:02d}_test_index_{dataset_index:03d}"
        video_path = output_dir / f"{stem}.mp4"
        contact_path = output_dir / f"{stem}_contact.png"
        arrays_path = output_dir / f"{stem}.npz"
        write_video(video_path, rendered, args.fps)
        cv2.imwrite(
            str(contact_path),
            cv2.cvtColor(make_contact_sheet(rendered), cv2.COLOR_RGB2BGR),
        )
        np.savez_compressed(
            arrays_path,
            gt_boxes=gt_boxes[0],
            pseudo_boxes=pseudo_boxes[0],
            gt_condition_slot_labels=gt_labels,
            pseudo_condition_slot_labels_raw=pseudo_labels_raw,
            pseudo_condition_slot_labels_aligned=pseudo_labels,
        )

        box_stats = condition_box_stats(gt_boxes[0], pseudo_boxes[0])
        record = {
            "dataset_index": dataset_index,
            "metrics": {"gt_box": gt_metrics, "pseudo_box": pseudo_metrics},
            "delta_pseudo_minus_gt": {
                key: pseudo_metrics[key] - gt_metrics[key] for key in metric_names
            },
            "box_stats": box_stats,
            "pseudo_debug": pseudo_sample.debug,
            "pseudo_seconds": pseudo_seconds,
            "total_seconds": time.time() - started,
            "attention_shape": attention_shape,
            "pseudo_to_gt_slot_color_mapping": slot_mapping,
            "video": video_path.name,
            "contact_sheet": contact_path.name,
            "arrays": arrays_path.name,
        }
        cases.append(record)
        print(
            f"[case {position}/{len(indices)}] index={dataset_index} "
            f"GT={gt_metrics} pseudo={pseudo_metrics} boxes={box_stats}",
            flush=True,
        )

    summary = {
        branch: {
            metric_name: totals[branch][metric_name] / len(cases)
            for metric_name in metric_names
        }
        for branch in ("gt_box", "pseudo_box")
    }
    summary["delta_pseudo_minus_gt"] = {
        metric_name: summary["pseudo_box"][metric_name]
        - summary["gt_box"][metric_name]
        for metric_name in metric_names
    }
    payload = {
        "checkpoint": str(checkpoint),
        "checkpoint_report": checkpoint_report,
        "config": str(config_file),
        "dataset": str(raw_dataset.data_dir),
        "split": "test",
        "fixed_subset_file": str(args.subset_file.resolve()),
        "indices": indices,
        "num_cases": len(indices),
        "metric_protocol": (
            "official xSSC MOVi-C validation metrics over all 24 frames; "
            "ARI-FG skips GT channel 0; mBO and mIoU include background"
        ),
        "controlled_variable": (
            "same model, checkpoint, video, GT masks and eval mode; only bbox "
            "conditioning changes"
        ),
        "model_condition_note": "RandSFQ2 consumes bbox[:, 0] only",
        "pseudo_box_config": {
            "pipeline": "first-frame GroundingDINO plus SAM2 video tracking",
            "proposal_source": "gdino_only",
            "text_prompt": args.text_prompt,
            "include_caption_terms": False,
            "box_threshold": args.gdino_box_threshold,
            "text_threshold": args.gdino_text_threshold,
            "max_objects": cfg.num_slots,
            "track_dedupe_iou_threshold": 0.75,
            "container_suppression": [0.95, 2, 1.5, 0.7],
        },
        "amp_dtype": args.amp_dtype,
        "summary": summary,
        "cases": cases,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    with (output_dir / "metrics.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            [
                "dataset_index",
                "branch",
                *metric_names,
                "gt_box_count",
                "pseudo_box_count",
                "pseudo_gt_covered_mean_iou",
                "pseudo_gt_recall_at_50",
            ]
        )
        for case in cases:
            for branch in ("gt_box", "pseudo_box"):
                writer.writerow(
                    [
                        case["dataset_index"],
                        branch,
                        *[case["metrics"][branch][key] for key in metric_names],
                        case["box_stats"]["gt_count"],
                        case["box_stats"]["pseudo_count"],
                        case["box_stats"]["gt_covered_mean_iou"],
                        case["box_stats"]["gt_recall_at_50"],
                    ]
                )
    (output_dir / "index.html").write_text(build_html(payload))
    (output_dir / "README.md").write_text(
        "# MOVi-C GT box vs GDINO+SAM2 pseudo box\n\n"
        "This directory compares two inference branches on the same fixed "
        "MOVi-C validation cases. The only changed model input is `bbox`; "
        "all videos, masks, weights, precision, and evaluation code are shared.\n\n"
        "RandSFQ2 initializes slots from `bbox[:, 0]`, so only frame-0 boxes "
        "affect xSSC inference; later SAM2 boxes are retained for diagnostics.\n\n"
        "`metrics.json` is the complete machine-readable record. `metrics.csv` "
        "contains the flat metric table. Each MP4 shows GT masks, GT boxes, "
        "GT-conditioned slots, pseudo boxes, and pseudo-conditioned slots. "
        "Pseudo slot colors are Hungarian-aligned to the GT-conditioned branch "
        "for visualization only; metric values use the raw predictions.\n"
    )
    print(json.dumps(summary, indent=2), flush=True)
    print(f"output_dir={output_dir}", flush=True)


if __name__ == "__main__":
    main()
