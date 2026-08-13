#!/usr/bin/env python3
"""Render fixed-identity causal slot overlays as frame contact sheets."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as functional


STAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = STAGE_ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "upstream"))
sys.path.insert(0, "/home/gaoya/Code_Video/vjepa2-main")

from stage1_causal_state_probe import LABEL_FRAME_INDICES, STATIC_DIM  # noqa: E402
from stage1_causal_state_probe.alignment import hard_slot_masks  # noqa: E402
from stage1_causal_state_probe.cache_causal_slots import build_dataset  # noqa: E402
from stage1_causal_state_probe.data import TrajectoryDataset  # noqa: E402
from stage1_causal_state_probe.io_utils import atomic_write_json  # noqa: E402


PALETTE = np.asarray(
    [
        [239, 68, 68], [59, 130, 246], [34, 197, 94], [250, 204, 21],
        [168, 85, 247], [6, 182, 212], [249, 115, 22], [236, 72, 153],
        [132, 204, 22], [20, 184, 166],
    ],
    dtype=np.uint8,
)
IMAGENET_MEAN = torch.tensor([123.675, 116.28, 103.53]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([58.395, 57.12, 57.375]).view(1, 3, 1, 1)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=Path, required=True)
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=Path("/data/gaoya/agent-data/cache/xssc_stage1_causal_state"),
    )
    parser.add_argument("--data-root", type=Path, default=Path("/data/gaoya/dataset"))
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument("--indices", type=int, nargs="+", required=True)
    parser.add_argument("--mapping", choices=("prefix", "boundary"), default="prefix")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def decode_video(value):
    rgb = value.cpu() * IMAGENET_STD + IMAGENET_MEAN
    return rgb.clamp(0, 255).round().byte().permute(0, 2, 3, 1).numpy()


def overlay(rgb, labels, valid, alpha=0.55):
    color = np.zeros_like(rgb)
    color[valid] = PALETTE[labels[valid] % len(PALETTE)]
    output = rgb.copy().astype(np.float32)
    output[valid] = (1 - alpha) * output[valid] + alpha * color[valid]
    return output.round().clip(0, 255).astype(np.uint8)


def label_image(image, text):
    canvas = Image.new("RGB", (image.shape[1], image.shape[0] + 22), (18, 21, 26))
    canvas.paste(Image.fromarray(image), (0, 22))
    ImageDraw.Draw(canvas).text((6, 6), text, fill=(238, 241, 245), font=ImageFont.load_default())
    return np.asarray(canvas)


def contact_sheet(rgb, record, mapping_key):
    state_rgb = rgb[np.asarray(LABEL_FRAME_INDICES)]
    slot_masks = hard_slot_masks(record["slot_attention"].float())
    slot_labels = slot_masks.float().argmax(dim=1)
    slot_labels = functional.interpolate(
        slot_labels[:, None].float(), size=state_rgb.shape[1:3], mode="nearest"
    )[:, 0].long()
    mapping = record[mapping_key].long()
    mapped = mapping[slot_labels]
    slot_valid = mapped >= 0
    slot_object = mapped.clamp_min(0).numpy()
    slot_valid = slot_valid.numpy()

    gt_masks = record["gt_mask"].bool()
    gt_labels = gt_masks.float().argmax(dim=1)
    gt_any = gt_masks.any(dim=1)
    gt_labels = functional.interpolate(
        gt_labels[:, None].float(), size=state_rgb.shape[1:3], mode="nearest"
    )[:, 0].long().numpy()
    gt_any = functional.interpolate(
        gt_any[:, None].float(), size=state_rgb.shape[1:3], mode="nearest"
    )[:, 0].bool().numpy()

    tiles = []
    for time_index, frame in enumerate(state_rgb):
        panels = np.concatenate(
            [
                frame,
                overlay(frame, slot_object[time_index], slot_valid[time_index]),
                overlay(frame, gt_labels[time_index], gt_any[time_index]),
            ],
            axis=1,
        )
        tiles.append(
            label_image(
                panels,
                f"state {time_index:02d} / raw frame {LABEL_FRAME_INDICES[time_index]:02d}"
                "    RGB | fixed slot identity | GT identity",
            )
        )
    rows = [np.concatenate(tiles[start : start + 3], axis=1) for start in range(0, 12, 3)]
    return np.concatenate(rows, axis=0)


def plot_drift(record, path):
    slots = record["slots"][:, record["slot_valid"].bool()].float()
    static = (slots[1:, :, :STATIC_DIM] - slots[:-1, :, :STATIC_DIM]).square().mean(-1).sqrt()
    dynamic = (slots[1:, :, STATIC_DIM:] - slots[:-1, :, STATIC_DIM:]).square().mean(-1).sqrt()
    x = np.arange(1, slots.shape[0])
    fig, axis = plt.subplots(figsize=(8.5, 3.6))
    axis.plot(x, static.mean(1), label="static drift", color="#2563eb")
    axis.plot(x, dynamic.mean(1), label="dynamic drift", color="#ef4444")
    axis.set_xlabel("causal xSSC state")
    axis.set_ylabel("RMS latent drift")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def build_html(cases, mapping):
    cards = []
    for case in cases:
        cards.append(
            f"<article><h2>Case {case['index']} · {html.escape(case['video_name'])}</h2>"
            f"<img src=\"{case['contact_sheet']}\" alt=\"slot contact sheet\">"
            f"<img class=\"plot\" src=\"{case['drift_plot']}\" alt=\"latent drift\"></article>"
        )
    return f"""<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>xSSC Stage 1 causal slots</title><style>body{{margin:0;background:#101318;color:#edf1f5;font:14px system-ui}}header,main{{max-width:1500px;margin:auto;padding:18px}}article{{border-top:1px solid #343a43;padding:18px 0}}img{{display:block;max-width:100%;height:auto;background:#050607}}.plot{{width:min(850px,100%);margin-top:10px}}h1{{font-size:22px}}h2{{font-size:16px}}</style></head><body><header><h1>xSSC Stage 1 · causal fixed-identity slots</h1><p>Alignment: {mapping}. Identity is calibrated once in states 0–3 and frozen afterwards.</p></header><main>{''.join(cards)}</main></body></html>"""


def main():
    args = parse_args()
    from object_centric_bench.util import Config

    cfg = Config.fromfile(args.config_file.resolve())
    source_dataset = build_dataset(cfg, args.split, args.data_root.resolve())
    cache_dataset = TrajectoryDataset(args.cache_root.resolve(), args.split)
    cache_by_index = {
        int(entry["index"]): position for position, entry in enumerate(cache_dataset.entries)
    }
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_key = f"{args.mapping}_slot_to_object"
    cases = []
    for index in args.indices:
        if index not in cache_by_index:
            raise KeyError(f"Case {index} is not present in the cache")
        sample = source_dataset[index]
        record = cache_dataset[cache_by_index[index]]
        case_dir = output_dir / f"case_{index:06d}"
        case_dir.mkdir(parents=True, exist_ok=True)
        sheet_path = case_dir / "causal_slot_frames.jpg"
        drift_path = case_dir / "latent_drift.png"
        Image.fromarray(contact_sheet(decode_video(sample["video"]), record, mapping_key)).save(
            sheet_path, quality=92
        )
        plot_drift(record, drift_path)
        cases.append(
            {
                "index": index,
                "video_name": record["source"]["video_name"],
                "contact_sheet": sheet_path.relative_to(output_dir).as_posix(),
                "drift_plot": drift_path.relative_to(output_dir).as_posix(),
            }
        )
    (output_dir / "index.html").write_text(build_html(cases, args.mapping))
    atomic_write_json({"mapping": args.mapping, "cases": cases}, output_dir / "metadata.json")
    print(json.dumps({"index": str(output_dir / 'index.html'), "cases": len(cases)}, indent=2))


if __name__ == "__main__":
    main()
