#!/usr/bin/env python3
"""Analyze Scheme A latent xSSC slot temporal similarity and PCA.

This script mirrors the Scheme A inference conditioning path up to the object
tokens: source video prefix frames -> frozen xSSC slots -> Wan latent-time mean
alignment -> LayerNorm/Linear/time embedding loaded from a Scheme A checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import html
import json
from pathlib import Path
import types
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from safetensors.torch import load_file
import torch
import torch.nn.functional as F

from code_vjepa_vggt.train_xSSC import infer_xssc_allframe_oracle_slots as infer
from code_vjepa_vggt.train_xSSC import train_xssc_allframe_oracle_slots as train
from code_vjepa_vggt.utils.video_io import read_video_prefix
from code_vjepa_vggt.train_xSSC.batch_infer_xssc_allframe_oracle_slots import (
    _resolve_source_video,
    preprocess_video_rgb_uint8,
)
from code_vjepa_vggt.train_xSSC.visualize_xssc_slot_attention import _resolve_video_path


DEFAULT_CASES = [
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed-ball-and-block-fall_motion_to_end.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper.json",
    "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons/physicIQ_025_Solid_Mechanics_0002_perspective-center_trimmed_crop_top60px.json",
]


def _safe_stem(path: Path) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in path.stem)


def _cosine_matrix(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1.0e-12)
    return (x @ x.T).astype(np.float32)


def _offdiag_mean(mat: np.ndarray) -> float:
    if mat.shape[0] <= 1:
        return float("nan")
    mask = ~np.eye(mat.shape[0], dtype=bool)
    return float(mat[mask].mean())


def _adjacent_mean(mat: np.ndarray) -> float:
    if mat.shape[0] <= 1:
        return float("nan")
    return float(np.diag(mat, k=1).mean())


def _pca(x: np.ndarray, components: int = 3) -> tuple[np.ndarray, np.ndarray]:
    centered = x.astype(np.float64) - x.astype(np.float64).mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    coords = centered @ vt[:components].T
    denom = max(float(np.sum(singular_values**2)), 1.0e-12)
    explained = (singular_values[:components] ** 2) / denom
    return coords.astype(np.float32), explained.astype(np.float32)


def _plot_heatmap(mat: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(5.2, 4.5), dpi=170)
    im = ax.imshow(mat, vmin=-1.0, vmax=1.0, cmap="coolwarm")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xlabel("latent time")
    ax.set_ylabel("latent time")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _plot_pca(
    coords: np.ndarray,
    records: list[dict[str, object]],
    path: Path,
    title: str,
    explained: np.ndarray,
) -> None:
    slot_ids = sorted({int(record["slot_id"]) for record in records})
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(slot_ids), 1)))
    fig, ax = plt.subplots(figsize=(8.4, 6.4), dpi=170)
    for color, slot_id in zip(colors, slot_ids):
        idxs = [i for i, record in enumerate(records) if int(record["slot_id"]) == slot_id]
        idxs.sort(key=lambda i: int(records[i]["latent_id"]))
        xy = coords[idxs, :2]
        ax.plot(xy[:, 0], xy[:, 1], "-o", color=color, linewidth=1.2, markersize=3.8, label=f"slot{slot_id:02d}")
        for i in idxs:
            ax.text(coords[i, 0], coords[i, 1], str(records[i]["latent_id"]), fontsize=6, color=color)
    ax.axhline(0, color="#ddd", linewidth=0.7)
    ax.axvline(0, color="#ddd", linewidth=0.7)
    ax.set_xlabel(f"PC1 ({float(explained[0]):.3f})")
    ax.set_ylabel(f"PC2 ({float(explained[1]):.3f})")
    ax.set_title(title)
    ax.legend(ncol=2, fontsize=8)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _load_projector(checkpoint_dir: Path, device: torch.device):
    ckpt_path = checkpoint_dir / "checkpoint.safetensors"
    if not ckpt_path.is_file():
        raise FileNotFoundError(ckpt_path)
    state = load_file(str(ckpt_path), device=str(device))
    slot_norm = torch.nn.LayerNorm(256).to(device=device, dtype=torch.bfloat16)
    slot_projector = torch.nn.Linear(256, 3072).to(device=device, dtype=torch.bfloat16)
    time_embedding = torch.nn.Embedding(64, 3072).to(device=device, dtype=torch.bfloat16)
    slot_norm.load_state_dict(
        {
            "weight": state["slot_norm.weight"].to(device=device),
            "bias": state["slot_norm.bias"].to(device=device),
        }
    )
    slot_projector.load_state_dict(
        {
            "weight": state["slot_projector.weight"].to(device=device),
            "bias": state["slot_projector.bias"].to(device=device),
        }
    )
    time_embedding.load_state_dict({"weight": state["time_embedding.weight"].to(device=device)})
    slot_norm.eval()
    slot_projector.eval()
    time_embedding.eval()
    return slot_norm, slot_projector, time_embedding


def _build_minimal_xssc_model(device: torch.device):
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
    model.xssc_vae_temporal_stride = train.DEFAULT_WAN_VAE_TEMPORAL_STRIDE
    model.xssc_oracle_video_frames = train.DEFAULT_XSSC_ORACLE_VIDEO_FRAMES
    model.xssc_max_time_steps = 64
    model._preprocess_xssc = types.MethodType(
        train.XSSCAllFrameOracleSlotsWanModule._preprocess_xssc, model
    )
    model._extract_xssc_slots = types.MethodType(
        train.XSSCAllFrameOracleSlotsWanModule._extract_xssc_slots, model
    )
    return model


def _extract_latent_slots(model, source_video: Path, args, device: torch.device):
    frames, frame_indices = read_video_prefix(source_video, int(args.max_frames))
    actual_frames = int(frames.shape[0])
    if actual_frames <= 0:
        raise RuntimeError(f"no readable frames: {source_video}")
    video_single = preprocess_video_rgb_uint8(
        frames,
        (int(args.height), int(args.width)),
        resize_mode="cover_crop",
        cover_crop_hw=(int(args.input_cover_crop_height), int(args.input_cover_crop_width)),
    )
    video = video_single.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    xssc_video, preprocess_debug = infer._preprocess_xssc_with_mode(
        model,
        video,
        mode="center_crop",
    )
    slots = model._extract_xssc_slots(xssc_video)
    latent_slots, latent_debug = infer._make_latent_slots(
        slots,
        stride=int(model.xssc_vae_temporal_stride),
        mode="mean_latent_align",
    )
    return slots, latent_slots, {
        "actual_frames": actual_frames,
        "frame_indices": [int(v) for v in frame_indices.tolist()],
        "preprocess": preprocess_debug,
        "latent_slot_mode": latent_debug,
    }


@torch.no_grad()
def _project_object_tokens(latent_slots: torch.Tensor, projector) -> torch.Tensor:
    slot_norm, slot_projector, time_embedding = projector
    latent_slots = latent_slots.to(device=slot_norm.weight.device, dtype=slot_norm.weight.dtype)
    tokens = slot_projector(slot_norm(latent_slots))
    time_ids = torch.arange(tokens.shape[1], device=tokens.device)
    tokens = tokens + time_embedding(time_ids).view(1, tokens.shape[1], 1, -1).to(dtype=tokens.dtype)
    return tokens


def _analyze_embedding(case_dir: Path, case_label: str, embedding_tsd: np.ndarray, prefix: str) -> dict[str, object]:
    time_steps, num_slots, dim = embedding_tsd.shape
    rows = []
    records = []
    flat = []
    for slot_id in range(num_slots):
        mat = _cosine_matrix(embedding_tsd[:, slot_id])
        _plot_heatmap(mat, case_dir / f"{prefix}_slot{slot_id:02d}_latent_cosine.png", f"{case_label} {prefix} slot{slot_id:02d}")
        rows.append(
            {
                "case": case_label,
                "embedding": prefix,
                "slot_id": slot_id,
                "time_steps": time_steps,
                "dim": dim,
                "adjacent_cosine_mean": _adjacent_mean(mat),
                "offdiag_cosine_mean": _offdiag_mean(mat),
                "cosine_min": float(mat[~np.eye(time_steps, dtype=bool)].min()) if time_steps > 1 else float("nan"),
                "cosine_max_offdiag": float(mat[~np.eye(time_steps, dtype=bool)].max()) if time_steps > 1 else float("nan"),
                "temporal_l2_step_mean": float(np.linalg.norm(np.diff(embedding_tsd[:, slot_id], axis=0), axis=1).mean())
                if time_steps > 1
                else float("nan"),
            }
        )
        for latent_id in range(time_steps):
            flat.append(embedding_tsd[latent_id, slot_id])
            records.append({"slot_id": slot_id, "latent_id": latent_id})
    coords, explained = _pca(np.asarray(flat), components=3)
    _plot_pca(
        coords,
        records,
        case_dir / f"{prefix}_pca_latent_tracks.png",
        f"{case_label}: PCA of {prefix} latent tracks",
        explained,
    )
    np.savez_compressed(
        case_dir / f"{prefix}_pca_data.npz",
        coords=coords,
        explained=explained,
        embedding_tsd=embedding_tsd.astype(np.float32),
    )
    return {
        "rows": rows,
        "pca_explained": [float(v) for v in explained.tolist()],
        "pca_png": str(case_dir / f"{prefix}_pca_latent_tracks.png"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-json", action="append", type=Path, default=[])
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/0623/train/train0624/train_xSSC/offcial_xSSC/"
            "train_xssc_allframe_oracle_slots/checkpoints/step-002000"
        ),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--input-cover-crop-height", type=int, default=512)
    parser.add_argument("--input-cover-crop-width", type=int, default=896)
    parser.add_argument("--max-frames", type=int, default=49)
    args = parser.parse_args()

    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    model = _build_minimal_xssc_model(device)
    projector = _load_projector(args.checkpoint_dir.expanduser().resolve(), device)
    case_jsons = args.case_json or [Path(path) for path in DEFAULT_CASES]

    all_rows = []
    summaries = []
    for case_json in case_jsons:
        case_json = case_json.expanduser().resolve()
        payload = json.loads(case_json.read_text(encoding="utf-8"))
        source_video = Path(_resolve_source_video(payload, case_json)).expanduser().resolve()
        case_label = _safe_stem(case_json)
        case_dir = output_dir / case_label
        case_dir.mkdir(parents=True, exist_ok=True)
        slots, latent_slots, debug = _extract_latent_slots(model, source_video, args, device)
        object_tokens = _project_object_tokens(latent_slots, projector)
        latent_np = latent_slots[0].detach().float().cpu().numpy()
        token_np = object_tokens[0].detach().float().cpu().numpy()
        np.savez_compressed(
            case_dir / "scheme_a_latent_embeddings.npz",
            raw_slots_tsd=slots[0].detach().float().cpu().numpy().astype(np.float32),
            latent_slots_tsd=latent_np.astype(np.float32),
            object_tokens_tsd=token_np.astype(np.float32),
            frame_indices=np.asarray(debug["frame_indices"], dtype=np.int64),
        )
        latent_summary = _analyze_embedding(case_dir, case_label, latent_np, "latent_slot256")
        token_summary = _analyze_embedding(case_dir, case_label, token_np, "object_token3072")
        rows = latent_summary["rows"] + token_summary["rows"]
        all_rows.extend(rows)
        summaries.append(
            {
                "case": case_label,
                "input_json": str(case_json),
                "source_video": str(source_video),
                "debug": debug,
                "raw_slots_shape": list(slots.shape),
                "latent_slots_shape": list(latent_slots.shape),
                "object_tokens_shape": list(object_tokens.shape),
                "latent_slot256_pca_explained": latent_summary["pca_explained"],
                "object_token3072_pca_explained": token_summary["pca_explained"],
            }
        )

    csv_path = output_dir / "scheme_a_latent_slot_similarity.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0].keys()))
        writer.writeheader()
        writer.writerows(all_rows)

    summary = {
        "checkpoint_dir": str(args.checkpoint_dir.expanduser().resolve()),
        "num_cases": len(summaries),
        "metric_note": (
            "latent_slot256 compares frozen xSSC slots after 49->13 Wan latent-time mean alignment; "
            "object_token3072 compares the final K/V conditioning tokens after LayerNorm+Linear+time embedding."
        ),
        "cases": summaries,
        "csv": str(csv_path),
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    rows_html = "\n".join(
        "<tr>"
        f"<td>{html.escape(row['case'])}</td><td>{html.escape(row['embedding'])}</td>"
        f"<td>{row['slot_id']}</td><td>{row['adjacent_cosine_mean']:.4f}</td>"
        f"<td>{row['offdiag_cosine_mean']:.4f}</td><td>{row['cosine_min']:.4f}</td>"
        f"<td>{row['temporal_l2_step_mean']:.4f}</td>"
        "</tr>"
        for row in all_rows
    )
    figures = []
    for case in summaries:
        case_name = str(case["case"])
        figures.append(
            f"<section><h2>{html.escape(case_name)}</h2>"
            f"<p><code>{html.escape(case['source_video'])}</code></p>"
            f"<img src='{case_name}/latent_slot256_pca_latent_tracks.png'>"
            f"<img src='{case_name}/object_token3072_pca_latent_tracks.png'>"
            "</section>"
        )
    index_html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Scheme A latent slot similarity</title>
<style>
body {{ font-family: system-ui, sans-serif; margin: 24px; background: #111; color: #eee; }}
table {{ border-collapse: collapse; width: 100%; font-size: 12px; }}
th, td {{ border: 1px solid #444; padding: 6px 8px; text-align: left; }}
th {{ background: #222; position: sticky; top: 0; }}
img {{ width: min(780px, 100%); margin: 10px 16px 18px 0; background: white; }}
code {{ color: #cde; }}
</style></head><body>
<h1>Scheme A latent slot temporal similarity</h1>
<p>{html.escape(summary['metric_note'])}</p>
<p><a href="summary.json">summary.json</a> | <a href="scheme_a_latent_slot_similarity.csv">CSV</a></p>
<table><thead><tr><th>case</th><th>embedding</th><th>slot</th><th>adjacent cosine</th><th>offdiag cosine</th><th>min cosine</th><th>adjacent L2</th></tr></thead>
<tbody>{rows_html}</tbody></table>
{''.join(figures)}
</body></html>
"""
    (output_dir / "index.html").write_text(index_html, encoding="utf-8")
    print(summary_path)
    print(csv_path)
    print(output_dir / "index.html")


if __name__ == "__main__":
    main()
