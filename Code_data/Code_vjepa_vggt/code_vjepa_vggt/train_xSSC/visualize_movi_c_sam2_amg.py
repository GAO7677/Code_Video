#!/usr/bin/env python3
"""Run prompt-free SAM2 AMG on fixed MOVi-C first frames and extend the report."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import re
import sys
import time

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
import torch


TRAIN_XSSC_ROOT = Path(__file__).resolve().parent
EXPERIMENT = TRAIN_XSSC_ROOT / "xssc_rsfq2_ytvis_dinov3_vitl16_256"
sys.path.insert(0, str(EXPERIMENT / "upstream"))
sys.path.insert(0, "/home/gaoya/Grounded-SAM-2-main")

DEFAULT_REPORT = Path(
    "/data/gaoya/agent-data/outputs/"
    "movi_c_gt_vs_gdino_sam2_fixed5_20260722"
)
DEFAULT_CONFIG = Path(
    "/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml"
)
DEFAULT_CHECKPOINT = Path(
    "/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt"
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
        [14, 165, 233],
        [245, 158, 11],
        [16, 185, 129],
        [139, 92, 246],
        [225, 29, 72],
    ],
    dtype=np.uint8,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT)
    parser.add_argument("--data-dir", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--sam2-config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--sam2-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-selected", type=int, default=11)
    parser.add_argument("--min-area-ratio", type=float, default=0.004)
    parser.add_argument("--max-area-ratio", type=float, default=0.35)
    parser.add_argument("--min-bbox-side", type=float, default=7.0)
    parser.add_argument("--background-area-ratio", type=float, default=0.06)
    parser.add_argument("--background-span-ratio", type=float, default=0.75)
    parser.add_argument("--border-area-ratio", type=float, default=0.025)
    parser.add_argument("--border-occupancy-ratio", type=float, default=0.18)
    parser.add_argument("--opposite-edge-area-ratio", type=float, default=0.04)
    parser.add_argument("--shadow-min-area-ratio", type=float, default=0.03)
    parser.add_argument("--shadow-max-luminance-ratio", type=float, default=0.55)
    parser.add_argument(
        "--shadow-max-chromaticity-distance", type=float, default=0.10
    )
    parser.add_argument("--shadow-max-gradient-mean", type=float, default=20.0)
    parser.add_argument("--duplicate-iou", type=float, default=0.70)
    parser.add_argument("--duplicate-containment", type=float, default=0.85)
    return parser.parse_args()


def resolve_sam2_config_name(config_path):
    path = Path(config_path)
    if path.name.startswith("sam2.1_"):
        return f"configs/sam2.1/{path.name}"
    if path.name.startswith("sam2_"):
        return f"configs/sam2/{path.name}"
    return str(config_path)


def mask_iou(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 0.0


def mask_containment(mask_a, mask_b):
    intersection = np.logical_and(mask_a, mask_b).sum()
    smaller = min(mask_a.sum(), mask_b.sum())
    return float(intersection / smaller) if smaller else 0.0


def select_xssc_candidates(annotations, image_area, args, image=None):
    image_side = image_area**0.5
    if image is not None:
        image_float = image.astype(np.float32) / 255.0
        chromaticity = image_float / (
            image_float.sum(axis=2, keepdims=True) + 1.0e-5
        )
        luminance = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)[..., 0].astype(
            np.float32
        )
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float32)
        gradient = cv2.magnitude(
            cv2.Sobel(gray, cv2.CV_32F, 1, 0),
            cv2.Sobel(gray, cv2.CV_32F, 0, 1),
        )
    else:
        chromaticity = luminance = gradient = None
    candidates = []
    for annotation in annotations:
        area_ratio = float(annotation["area"] / image_area)
        if not args.min_area_ratio <= area_ratio <= args.max_area_ratio:
            continue
        _, _, box_width, box_height = annotation["bbox"]
        if min(box_width, box_height) < args.min_bbox_side:
            continue
        spans_background = (
            box_width / image_side >= args.background_span_ratio
            or box_height / image_side >= args.background_span_ratio
        )
        if area_ratio >= args.background_area_ratio and spans_background:
            continue
        mask = annotation["segmentation"].astype(bool)
        border = np.concatenate(
            [mask[0, :], mask[-1, :], mask[:, 0], mask[:, -1]]
        )
        if (
            area_ratio >= args.border_area_ratio
            and float(border.mean()) >= args.border_occupancy_ratio
        ):
            continue
        touches_opposite_edges = (
            mask[:, 0].any() and mask[:, -1].any()
        ) or (mask[0, :].any() and mask[-1, :].any())
        if (
            area_ratio >= args.opposite_edge_area_ratio
            and touches_opposite_edges
        ):
            continue
        if image is not None and area_ratio >= args.shadow_min_area_ratio:
            mask_u8 = mask.astype(np.uint8)
            ring = (
                cv2.dilate(mask_u8, np.ones((11, 11), dtype=np.uint8))
                - mask_u8
            ).astype(bool)
            if ring.any():
                luminance_ratio = float(
                    luminance[mask].mean() / max(luminance[ring].mean(), 1.0e-5)
                )
                chromaticity_distance = float(
                    np.linalg.norm(
                        chromaticity[mask].mean(axis=0)
                        - chromaticity[ring].mean(axis=0)
                    )
                )
                gradient_mean = float(gradient[mask].mean())
                shadow_like = (
                    luminance_ratio <= args.shadow_max_luminance_ratio
                    and chromaticity_distance
                    <= args.shadow_max_chromaticity_distance
                    and gradient_mean <= args.shadow_max_gradient_mean
                )
                if shadow_like:
                    continue
        quality = float(annotation["predicted_iou"] * annotation["stability_score"])
        rank_score = quality * area_ratio**0.10
        candidates.append((rank_score, annotation))
    candidates.sort(key=lambda item: item[0], reverse=True)

    selected = []
    for _, candidate in candidates:
        mask = candidate["segmentation"]
        duplicate = any(
            mask_iou(mask, kept["segmentation"]) >= args.duplicate_iou
            or mask_containment(mask, kept["segmentation"])
            >= args.duplicate_containment
            for kept in selected
        )
        if duplicate:
            continue
        selected.append(candidate)
        if len(selected) >= args.max_selected:
            break
    return selected


def proposal_metrics(annotations, gt_masks):
    proposals = [annotation["segmentation"].astype(bool) for annotation in annotations]
    if not len(gt_masks):
        return {"mean_best_iou": 1.0, "recall_at_50": 1.0, "hungarian_miou": 1.0}
    pairwise = np.zeros((len(gt_masks), len(proposals)), dtype=np.float32)
    for gt_index, gt_mask in enumerate(gt_masks):
        for proposal_index, proposal in enumerate(proposals):
            pairwise[gt_index, proposal_index] = mask_iou(gt_mask, proposal)
    best = pairwise.max(axis=1) if len(proposals) else np.zeros(len(gt_masks))
    matched = np.zeros(len(gt_masks), dtype=np.float32)
    if len(proposals):
        gt_ids, proposal_ids = linear_sum_assignment(-pairwise)
        matched[gt_ids] = pairwise[gt_ids, proposal_ids]
    return {
        "mean_best_iou": float(best.mean()),
        "recall_at_50": float((best >= 0.5).mean()),
        "hungarian_miou": float(matched.mean()),
    }


def overlay_masks(image, annotations, alpha=0.52):
    output = image.astype(np.float32).copy()
    ordered = sorted(
        annotations,
        key=lambda item: float(item["predicted_iou"] * item["stability_score"]),
    )
    for mask_index, annotation in enumerate(ordered):
        mask = annotation["segmentation"].astype(bool)
        color = PALETTE[mask_index % len(PALETTE)].astype(np.float32)
        output[mask] = output[mask] * (1.0 - alpha) + color * alpha
    return output.round().clip(0, 255).astype(np.uint8)


def overlay_gt(image, segment):
    output = image.astype(np.float32).copy()
    labels = segment.argmax(axis=-1)
    for label in range(1, int(labels.max()) + 1):
        mask = labels == label
        color = PALETTE[(label - 1) % len(PALETTE)].astype(np.float32)
        output[mask] = output[mask] * 0.45 + color * 0.55
    return output.round().clip(0, 255).astype(np.uint8)


def draw_selected_boxes(image, annotations):
    output = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    for index, annotation in enumerate(annotations):
        x, y, width, height = annotation["bbox"]
        x0, y0 = int(round(x)), int(round(y))
        x1, y1 = int(round(x + width)), int(round(y + height))
        color = tuple(int(value) for value in PALETTE[index % len(PALETTE)][::-1])
        cv2.rectangle(output, (x0, y0), (x1, y1), color, 2, cv2.LINE_AA)
        cv2.putText(
            output,
            f"A{index + 1}",
            (x0 + 2, max(y0 + 13, 13)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    return cv2.cvtColor(output, cv2.COLOR_BGR2RGB)


def add_title(image, title):
    output = image.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 22), (18, 18, 18), -1)
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


def annotation_metadata(annotation):
    return {
        "area": int(annotation["area"]),
        "bbox_xywh": [float(value) for value in annotation["bbox"]],
        "predicted_iou": float(annotation["predicted_iou"]),
        "stability_score": float(annotation["stability_score"]),
        "point_coords": annotation["point_coords"],
        "crop_box": [float(value) for value in annotation["crop_box"]],
    }


def build_html_section(payload):
    cards = []
    for case in payload["cases"]:
        cards.append(
            f"<article><h2>test index {case['dataset_index']:03d}</h2>"
            f"<p>GT instances {case['gt_instance_count']}; raw AMG masks "
            f"{case['raw_mask_count']}; selected {case['selected_mask_count']}. "
            f"Selected recall@0.5 {case['selected_metrics']['recall_at_50']:.3f}, "
            f"Hungarian mIoU {case['selected_metrics']['hungarian_miou']:.3f}.</p>"
            f"<img loading='lazy' src='{html.escape(case['image'])}' "
            "alt='Prompt-free SAM2 AMG segmentation comparison'></article>"
        )
    return (
        "<!-- SAM2_AMG_START -->"
        "<section id='sam2-amg'><h1>SAM2 Automatic Mask Generator</h1>"
        "<p class='note'>First frame only. No caption, detector, GT prompt, point, "
        "box, or mask is supplied. Raw AMG uses official Hiera-L defaults; "
        "top-11 removes masks below 0.4% or above 35% area, edge-spanning "
        "backgrounds, low-texture shadow-like regions, short-side boxes below "
        "7 px, and duplicate masks.</p>"
        f"<img loading='lazy' src='{html.escape(payload['overview'])}' "
        "alt='SAM2 AMG overview'>"
        f"{''.join(cards)}</section>"
        "<!-- SAM2_AMG_END -->"
    )


def update_report_html(report_dir, payload):
    index_path = report_dir / "index.html"
    page = index_path.read_text()
    page = re.sub(
        r"<!-- SAM2_AMG_START -->.*?<!-- SAM2_AMG_END -->",
        "",
        page,
        flags=re.DOTALL,
    )
    section = build_html_section(payload)
    if "</main>" not in page:
        raise RuntimeError(f"Cannot find </main> in {index_path}")
    index_path.write_text(page.replace("</main>", section + "</main>"))


def main():
    args = parse_args()
    report_dir = args.report_dir.resolve()
    metrics_file = report_dir / "metrics.json"
    if not metrics_file.is_file():
        raise FileNotFoundError(metrics_file)
    if not args.sam2_config.is_file() or not args.sam2_checkpoint.is_file():
        raise FileNotFoundError("SAM2 config or checkpoint is missing")

    from object_centric_bench.datum import MOViTFRecord
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    from sam2.build_sam import build_sam2

    comparison = json.loads(metrics_file.read_text())
    indices = [int(index) for index in comparison["indices"]]
    dataset = MOViTFRecord(
        data_file="kubric-movi/movi-c",
        split="test",
        extra_keys=["segment", "bbox"],
        base_dir=args.data_dir.resolve(),
    )

    device = torch.device(args.device)
    torch.cuda.set_device(device)
    resolved_config_name = resolve_sam2_config_name(args.sam2_config)
    sam2 = build_sam2(
        resolved_config_name,
        str(args.sam2_checkpoint.resolve()),
        device=str(device),
        mode="eval",
    )
    generator = SAM2AutomaticMaskGenerator(sam2)

    output_dir = report_dir / "sam2_amg"
    output_dir.mkdir(parents=True, exist_ok=True)
    cases = []
    overview_rows = []
    for position, dataset_index in enumerate(indices, start=1):
        sample = dataset[dataset_index]
        image = sample["video"][0].permute(1, 2, 0).contiguous().numpy()
        segment = sample["segment"][0].numpy()
        gt_masks = segment[..., 1:].transpose(2, 0, 1)
        gt_masks = gt_masks[gt_masks.reshape(len(gt_masks), -1).any(axis=1)]

        started = time.time()
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            annotations = generator.generate(image)
        seconds = time.time() - started
        selected = select_xssc_candidates(
            annotations, image.shape[0] * image.shape[1], args, image=image
        )

        gt_panel = overlay_gt(image, segment)
        raw_panel = overlay_masks(image, annotations)
        selected_panel = draw_selected_boxes(overlay_masks(image, selected), selected)
        comparison_image = np.concatenate(
            [
                add_title(image, "raw frame 0"),
                add_title(gt_panel, f"GT instances | n={len(gt_masks)}"),
                add_title(raw_panel, f"SAM2 AMG raw | n={len(annotations)}"),
                add_title(selected_panel, f"AMG filtered top-11 | n={len(selected)}"),
            ],
            axis=1,
        )
        stem = f"case_{position:02d}_test_index_{dataset_index:03d}"
        image_path = output_dir / f"{stem}.png"
        arrays_path = output_dir / f"{stem}.npz"
        cv2.imwrite(str(image_path), cv2.cvtColor(comparison_image, cv2.COLOR_RGB2BGR))
        np.savez_compressed(
            arrays_path,
            all_masks=np.stack(
                [annotation["segmentation"] for annotation in annotations], axis=0
            ),
            selected_masks=np.stack(
                [annotation["segmentation"] for annotation in selected], axis=0
            ),
            gt_masks=gt_masks,
        )
        raw_metrics = proposal_metrics(annotations, gt_masks)
        selected_metrics = proposal_metrics(selected, gt_masks)
        record = {
            "dataset_index": dataset_index,
            "gt_instance_count": int(len(gt_masks)),
            "raw_mask_count": int(len(annotations)),
            "selected_mask_count": int(len(selected)),
            "raw_metrics": raw_metrics,
            "selected_metrics": selected_metrics,
            "seconds": seconds,
            "image": str(image_path.relative_to(report_dir)),
            "arrays": str(arrays_path.relative_to(report_dir)),
            "raw_annotations": [annotation_metadata(item) for item in annotations],
            "selected_annotations": [annotation_metadata(item) for item in selected],
        }
        cases.append(record)
        overview_rows.append(comparison_image)
        print(
            f"[AMG {position}/{len(indices)}] index={dataset_index} "
            f"raw={len(annotations)} selected={len(selected)} "
            f"selected_metrics={selected_metrics} seconds={seconds:.2f}",
            flush=True,
        )

    overview = np.concatenate(overview_rows, axis=0)
    overview_path = output_dir / "overview.png"
    cv2.imwrite(str(overview_path), cv2.cvtColor(overview, cv2.COLOR_RGB2BGR))
    payload = {
        "method": "SAM2 Automatic Mask Generator on MOVi-C frame 0",
        "uses_caption": False,
        "uses_external_detector": False,
        "uses_gt_prompt": False,
        "sam2_config": str(args.sam2_config.resolve()),
        "sam2_package_config_name": resolved_config_name,
        "sam2_checkpoint": str(args.sam2_checkpoint.resolve()),
        "indices": indices,
        "amg_config": {
            "points_per_side": 32,
            "points_per_batch": 64,
            "pred_iou_thresh": 0.8,
            "stability_score_thresh": 0.95,
            "box_nms_thresh": 0.7,
            "crop_n_layers": 0,
        },
        "selection_config": {
            "max_selected": args.max_selected,
            "min_area_ratio": args.min_area_ratio,
            "max_area_ratio": args.max_area_ratio,
            "min_bbox_side": args.min_bbox_side,
            "background_area_ratio": args.background_area_ratio,
            "background_span_ratio": args.background_span_ratio,
            "border_area_ratio": args.border_area_ratio,
            "border_occupancy_ratio": args.border_occupancy_ratio,
            "opposite_edge_area_ratio": args.opposite_edge_area_ratio,
            "shadow_min_area_ratio": args.shadow_min_area_ratio,
            "shadow_max_luminance_ratio": args.shadow_max_luminance_ratio,
            "shadow_max_chromaticity_distance": (
                args.shadow_max_chromaticity_distance
            ),
            "shadow_max_gradient_mean": args.shadow_max_gradient_mean,
            "duplicate_iou": args.duplicate_iou,
            "duplicate_containment": args.duplicate_containment,
            "uses_gt": False,
        },
        "overview": str(overview_path.relative_to(report_dir)),
        "cases": cases,
    }
    (output_dir / "metrics.json").write_text(json.dumps(payload, indent=2) + "\n")
    update_report_html(report_dir, payload)
    print(f"report={report_dir / 'index.html'}", flush=True)


if __name__ == "__main__":
    main()
