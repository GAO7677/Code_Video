#!/usr/bin/env python3
"""Extract frozen xSSC slots from multiple source videos and visualize PCA."""
from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from code_vjepa_vggt.train_xSSC import train_xssc_context_slots as train
from code_vjepa_vggt.train_xSSC.visualize_xssc_slot_attention import (
    _cover_crop_to_tensor,
    _extract_slots_and_attention,
    _resolve_video_path,
)
from code_vjepa_vggt.utils.video_io import read_video_prefix


DEFAULT_CASES = [
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px.json",
]


def _safe_stem(path: Path) -> str:
    text = path.stem.replace("physicIQ_", "")
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in text)


def _save_contact_sheet(frames: np.ndarray, path: Path, *, max_width: int = 180) -> None:
    images = []
    for frame in frames.astype(np.uint8):
        image = Image.fromarray(frame)
        scale = float(max_width) / max(float(image.width), 1.0)
        image = image.resize((max_width, max(1, int(round(image.height * scale)))))
        images.append(image)
    canvas = Image.new("RGB", (sum(image.width for image in images), max(image.height for image in images)))
    offset = 0
    for image in images:
        canvas.paste(image, (offset, 0))
        offset += image.width
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=95)


def _pca(x: np.ndarray, components: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = x.astype(np.float64) - x.astype(np.float64).mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:components].T
    denom = max(float(np.sum(singular_values**2)), 1.0e-12)
    explained = (singular_values[:components] ** 2) / denom
    return coords.astype(np.float32), vt[:components].astype(np.float32), explained.astype(np.float32)


def _plot_explained(explained: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.4, 3.8), dpi=170)
    xs = np.arange(1, len(explained) + 1)
    ax.bar(xs, explained)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"PC{i}" for i in xs])
    ax.set_ylim(0, max(1.0, float(explained.max()) * 1.15))
    ax.set_ylabel("explained variance ratio")
    ax.set_title("Frozen xSSC slot PCA variance")
    for x, y in zip(xs, explained):
        ax.text(x, float(y), f"{float(y):.3f}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_global_by_case(
    coords: np.ndarray,
    records: list[dict[str, object]],
    output_path: Path,
) -> None:
    case_ids = sorted({int(record["case_id"]) for record in records})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(case_ids), 1)))
    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=170)
    for idx, case_id in enumerate(case_ids):
        indices = [i for i, record in enumerate(records) if int(record["case_id"]) == case_id]
        xy = coords[indices]
        label = str(records[indices[0]]["case_label"])
        ax.scatter(xy[:, 0], xy[:, 1], s=24, alpha=0.72, color=colors[idx], label=label)
        centroid = xy[:, :2].mean(axis=0)
        ax.text(centroid[0], centroid[1], f"C{case_id}", fontsize=10, weight="bold", color=colors[idx])
    ax.axhline(0, color="#bbb", linewidth=0.8)
    ax.axvline(0, color="#bbb", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Frozen xSSC slots PCA: colored by case")
    ax.legend(fontsize=7, loc="best")
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_global_by_slot(
    coords: np.ndarray,
    records: list[dict[str, object]],
    output_path: Path,
) -> None:
    slot_ids = sorted({int(record["slot_id"]) for record in records})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(slot_ids), 1)))
    fig, ax = plt.subplots(figsize=(8.2, 6.2), dpi=170)
    for idx, slot_id in enumerate(slot_ids):
        indices = [i for i, record in enumerate(records) if int(record["slot_id"]) == slot_id]
        xy = coords[indices]
        ax.scatter(xy[:, 0], xy[:, 1], s=22, alpha=0.65, color=colors[idx], label=f"slot{slot_id:02d}")
    ax.axhline(0, color="#bbb", linewidth=0.8)
    ax.axvline(0, color="#bbb", linewidth=0.8)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Frozen xSSC slots PCA: colored by slot id")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def _plot_case_tracks(
    coords: np.ndarray,
    records: list[dict[str, object]],
    output_path: Path,
) -> None:
    case_ids = sorted({int(record["case_id"]) for record in records})
    slot_ids = sorted({int(record["slot_id"]) for record in records})
    fig, axes = plt.subplots(2, 2, figsize=(12.2, 9.8), dpi=160)
    axes = axes.flatten()
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(slot_ids), 1)))
    for ax, case_id in zip(axes, case_ids):
        case_records = [record for record in records if int(record["case_id"]) == case_id]
        label = str(case_records[0]["case_label"])
        for color, slot_id in zip(colors, slot_ids):
            indices = [
                i
                for i, record in enumerate(records)
                if int(record["case_id"]) == case_id and int(record["slot_id"]) == slot_id
            ]
            indices.sort(key=lambda i: int(records[i]["frame_id"]))
            xy = coords[indices]
            ax.plot(xy[:, 0], xy[:, 1], "-o", color=color, linewidth=1.2, markersize=3.5, label=f"s{slot_id}")
            for i in indices:
                ax.text(coords[i, 0], coords[i, 1], str(records[i]["frame_id"]), fontsize=6)
        ax.axhline(0, color="#ddd", linewidth=0.7)
        ax.axvline(0, color="#ddd", linewidth=0.7)
        ax.set_title(f"C{case_id}: {label}", fontsize=9)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, ncol=7, loc="lower center", fontsize=8)
    fig.suptitle("Frozen xSSC slot temporal tracks in shared PCA space", y=0.98)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    fig.savefig(output_path)
    plt.close(fig)


def _plot_case_centroids(coords: np.ndarray, records: list[dict[str, object]], output_path: Path) -> None:
    case_ids = sorted({int(record["case_id"]) for record in records})
    slot_ids = sorted({int(record["slot_id"]) for record in records})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(slot_ids), 1)))
    fig, ax = plt.subplots(figsize=(8.0, 6.0), dpi=170)
    for slot_color, slot_id in zip(colors, slot_ids):
        points = []
        labels = []
        for case_id in case_ids:
            indices = [
                i
                for i, record in enumerate(records)
                if int(record["case_id"]) == case_id and int(record["slot_id"]) == slot_id
            ]
            xy = coords[indices, :2].mean(axis=0)
            points.append(xy)
            labels.append(f"C{case_id}")
        points_arr = np.stack(points, axis=0)
        ax.plot(points_arr[:, 0], points_arr[:, 1], "-o", color=slot_color, label=f"slot{slot_id:02d}")
        for xy, label in zip(points_arr, labels):
            ax.text(xy[0], xy[1], label, fontsize=6, color=slot_color)
    ax.axhline(0, color="#ddd", linewidth=0.7)
    ax.axvline(0, color="#ddd", linewidth=0.7)
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Per-case slot centroids in shared PCA space")
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--input-cover-crop-height", type=int, default=512)
    parser.add_argument("--input-cover-crop-width", type=int, default=896)
    parser.add_argument("--context-frames", type=int, default=train.XSSC_NUM_CONTEXT_FRAMES)
    args = parser.parse_args()

    case_jsons = args.case_json or [Path(path) for path in DEFAULT_CASES]
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(args.device)
    model = SimpleNamespace()
    xssc, slot_dim, num_slots = train._load_xssc_model(
        xssc_root=train.DEFAULT_XSSC_ROOT,
        config_path=train.DEFAULT_XSSC_CONFIG,
        checkpoint_path=train.DEFAULT_XSSC_CHECKPOINT,
        device=device,
    )
    model.xssc = xssc
    model.xssc_slot_dim = slot_dim
    model.xssc_num_slots = num_slots
    model.xssc_input_size = 256

    all_slots = []
    records: list[dict[str, object]] = []
    case_summaries = []
    for case_id, raw_path in enumerate(case_jsons):
        case_json = raw_path.expanduser().resolve()
        payload = json.loads(case_json.read_text(encoding="utf-8"))
        source_video = _resolve_video_path(payload, case_json)
        frames, frame_indices = read_video_prefix(source_video, int(args.context_frames))
        context_video_single, preprocess_metadata = _cover_crop_to_tensor(
            frames,
            target_hw=(int(args.height), int(args.width)),
            cover_crop_hw=(int(args.input_cover_crop_height), int(args.input_cover_crop_width)),
        )
        context_video = context_video_single.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
        with torch.no_grad():
            slots, attention = _extract_slots_and_attention(model, context_video)
        slots_tsd = slots[0].float().cpu().numpy()
        attention_tshw = attention[0].float().cpu().numpy()
        label = _safe_stem(case_json)
        contact_name = f"case{case_id}_{label}_ctx_sheet.jpg"
        _save_contact_sheet(frames, output_dir / contact_name)
        np.savez_compressed(
            output_dir / f"case{case_id}_{label}_frozen_xssc_slots.npz",
            slots_tsd=slots_tsd.astype(np.float32),
            attention_tshw=attention_tshw.astype(np.float16),
            frame_indices=np.asarray(frame_indices, dtype=np.int64),
        )
        for frame_id in range(int(slots_tsd.shape[0])):
            for slot_id in range(int(slots_tsd.shape[1])):
                all_slots.append(slots_tsd[frame_id, slot_id])
                records.append(
                    {
                        "case_id": int(case_id),
                        "case_label": label,
                        "case_json": str(case_json),
                        "source_video": str(source_video),
                        "frame_id": int(frame_id),
                        "source_frame_index": int(frame_indices[frame_id]),
                        "slot_id": int(slot_id),
                        "slot_norm": float(np.linalg.norm(slots_tsd[frame_id, slot_id])),
                    }
                )
        case_summaries.append(
            {
                "case_id": int(case_id),
                "case_label": label,
                "case_json": str(case_json),
                "source_video": str(source_video),
                "input_caption": str(payload.get("input_caption", "")),
                "frame_indices": [int(value) for value in frame_indices.tolist()],
                "slots_shape": list(slots_tsd.shape),
                "attention_shape": list(attention_tshw.shape),
                "preprocess": preprocess_metadata,
                "context_sheet": contact_name,
            }
        )

    slot_matrix = np.stack(all_slots, axis=0).astype(np.float32)
    coords, components, explained = _pca(slot_matrix, components=3)
    normed = slot_matrix / np.maximum(np.linalg.norm(slot_matrix, axis=1, keepdims=True), 1.0e-12)
    normed_coords, normed_components, normed_explained = _pca(normed, components=3)

    for record, coord, normed_coord in zip(records, coords, normed_coords):
        record["pc1"] = float(coord[0])
        record["pc2"] = float(coord[1])
        record["pc3"] = float(coord[2])
        record["normed_pc1"] = float(normed_coord[0])
        record["normed_pc2"] = float(normed_coord[1])
        record["normed_pc3"] = float(normed_coord[2])

    coords_path = output_dir / "frozen_xssc_slot_pca_coords.csv"
    with coords_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    np.savez_compressed(
        output_dir / "frozen_xssc_slot_pca_data.npz",
        slot_matrix=slot_matrix,
        coords=coords,
        components=components,
        explained_variance=explained,
        normed_coords=normed_coords,
        normed_components=normed_components,
        normed_explained_variance=normed_explained,
    )

    plots = {
        "explained_variance": "pca_explained_variance.png",
        "global_by_case": "pca_global_by_case.png",
        "global_by_slot": "pca_global_by_slot.png",
        "case_tracks": "pca_case_slot_temporal_tracks.png",
        "case_centroids": "pca_case_slot_centroids.png",
        "normed_global_by_case": "normed_pca_global_by_case.png",
        "normed_case_tracks": "normed_pca_case_slot_temporal_tracks.png",
    }
    _plot_explained(explained, output_dir / plots["explained_variance"])
    _plot_global_by_case(coords, records, output_dir / plots["global_by_case"])
    _plot_global_by_slot(coords, records, output_dir / plots["global_by_slot"])
    _plot_case_tracks(coords, records, output_dir / plots["case_tracks"])
    _plot_case_centroids(coords, records, output_dir / plots["case_centroids"])
    _plot_global_by_case(normed_coords, records, output_dir / plots["normed_global_by_case"])
    _plot_case_tracks(normed_coords, records, output_dir / plots["normed_case_tracks"])

    summary = {
        "method": "frozen official xSSC slot extraction on source_video, shared PCA over all case/frame/slot embeddings",
        "xssc_config": train.DEFAULT_XSSC_CONFIG,
        "xssc_checkpoint": train.DEFAULT_XSSC_CHECKPOINT,
        "context_frames": int(args.context_frames),
        "slot_dim": int(slot_dim),
        "num_slots": int(num_slots),
        "slot_matrix_shape": list(slot_matrix.shape),
        "explained_variance": [float(value) for value in explained.tolist()],
        "normed_explained_variance": [float(value) for value in normed_explained.tolist()],
        "coords_csv": coords_path.name,
        "data_npz": "frozen_xssc_slot_pca_data.npz",
        "plots": plots,
        "cases": case_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    case_cards = "".join(
        "<article>"
        f"<h3>C{case['case_id']} {html.escape(str(case['case_label']))}</h3>"
        f"<img src='{html.escape(str(case['context_sheet']))}'>"
        f"<div class='meta'>{html.escape(str(case['source_video']))}</div>"
        "</article>"
        for case in case_summaries
    )
    plot_cards = "".join(
        "<article>"
        f"<h3>{html.escape(name.replace('_', ' '))}</h3>"
        f"<img src='{html.escape(path)}'>"
        "</article>"
        for name, path in plots.items()
    )
    page = f"""<!doctype html><html><head><meta charset='utf-8'><title>Frozen xSSC Slot PCA</title>
<style>
body{{margin:0;background:#101114;color:#eceff4;font:14px Arial,sans-serif}}
main{{max-width:1700px;margin:auto;padding:22px}}
h1,h2,h3{{letter-spacing:0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(360px,1fr));gap:14px;align-items:start}}
article{{background:#171a20;border:1px solid #2b2f36;border-radius:8px;padding:12px;overflow:hidden}}
img{{display:block;width:100%;height:auto;background:#050608}}
.meta{{font-size:12px;color:#b9c0ca;line-height:1.45;margin-top:8px;word-break:break-word}}
a{{color:#9cc8ff}}
</style></head><body><main>
<h1>Frozen xSSC Slot PCA · Source Videos</h1>
<p>Each point is one frozen xSSC slot embedding from one source-video context frame. PCA is fit jointly over all four cases, all 8 ctx frames, and all 7 slots: {summary['slot_matrix_shape']}.</p>
<p>Explained variance PC1/PC2/PC3: {', '.join(f'{float(v):.4f}' for v in explained)}. <a href='summary.json'>summary JSON</a> | <a href='{coords_path.name}'>coords CSV</a> | <a href='frozen_xssc_slot_pca_data.npz'>data NPZ</a></p>
<h2>PCA Plots</h2><div class='grid'>{plot_cards}</div>
<h2>Source Context Frames</h2><div class='grid'>{case_cards}</div>
</main></body></html>"""
    (output_dir / "index.html").write_text(page, encoding="utf-8")
    print(json.dumps({"summary": str(summary_path), "index": str(output_dir / "index.html")}, indent=2))


if __name__ == "__main__":
    main()
