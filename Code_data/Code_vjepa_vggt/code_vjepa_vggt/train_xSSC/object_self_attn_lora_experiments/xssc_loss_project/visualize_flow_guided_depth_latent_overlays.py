#!/usr/bin/env python3
"""Build latent-axis overlays for flow-guided Depth Anything loss weighting."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import av
import cv2
import numpy as np


DEFAULT_FLOW_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_dual_loss_heatmaps/"
    "dinov3_movic_step50000_sample000_actual_timelines_sigma6_512x896_49f"
)
DEFAULT_DEPTH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/full_sa_no_object_depth_loss_sigma_demo"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/"
    "full_sa_no_object_flow_guided_depth_latent_overlays"
)
HUB_LINK = Path(
    "/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub/"
    "flow-guided-depth-latent-overlays"
)
QUANTILES = ((0.80, "Top-20%"), (0.90, "Top-10%"), (0.95, "Top-5%"))
MASK_ALPHA = 2.0
PANEL_WIDTH = 384
PANEL_HEIGHT = 216


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flow-root", type=Path, default=DEFAULT_FLOW_ROOT)
    parser.add_argument("--depth-root", type=Path, default=DEFAULT_DEPTH_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--rebuild", action="store_true")
    return parser.parse_args()


def read_rgb_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode any frames from {path}")
    return np.stack(frames)


def write_mp4(path: Path, frames: np.ndarray, fps: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=fps)
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    for image in frames:
        frame = av.VideoFrame.from_ndarray(image, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()


def resize_maps(maps: np.ndarray, height: int, width: int, interpolation: int) -> np.ndarray:
    return np.stack(
        [cv2.resize(item, (width, height), interpolation=interpolation) for item in maps]
    ).astype(np.float32)


def heatmap_overlay(base: np.ndarray, values: np.ndarray, scale: float) -> np.ndarray:
    normalized = np.clip(values.astype(np.float32) / max(scale, 1e-12), 0.0, 1.0)
    colors = []
    for item in normalized:
        color = cv2.applyColorMap(
            np.rint(item * 255.0).astype(np.uint8),
            cv2.COLORMAP_TURBO,
        )
        colors.append(cv2.cvtColor(color, cv2.COLOR_BGR2RGB))
    colors_array = np.stack(colors).astype(np.float32)
    strength = normalized[..., None]
    mixed = base.astype(np.float32) * (1.0 - 0.62 * strength)
    mixed += colors_array * (0.62 * strength)
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def mask_overlay(base: np.ndarray, mask: np.ndarray) -> np.ndarray:
    color = np.zeros_like(base, dtype=np.float32)
    color[..., 0] = 255.0
    color[..., 1] = 53.0
    color[..., 2] = 92.0
    strength = mask.astype(np.float32)[..., None] * 0.62
    mixed = base.astype(np.float32) * (1.0 - strength) + color * strength
    return np.clip(np.rint(mixed), 0, 255).astype(np.uint8)


def labeled_panel(frames: np.ndarray, label: str) -> np.ndarray:
    output = []
    for frame in frames:
        panel = cv2.resize(
            frame,
            (PANEL_WIDTH, PANEL_HEIGHT),
            interpolation=cv2.INTER_AREA,
        )
        panel = panel.copy()
        cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 31), (18, 31, 43), -1)
        cv2.putText(
            panel,
            label,
            (9, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        output.append(panel)
    return np.stack(output)


def information_panel(
    *,
    latent_ids: np.ndarray,
    representative_frames: np.ndarray,
    valid_latents: np.ndarray,
    title: str,
    lines: list[str],
) -> np.ndarray:
    output = []
    for latent_id, frame_id, valid in zip(
        latent_ids, representative_frames, valid_latents
    ):
        panel = np.full((PANEL_HEIGHT, PANEL_WIDTH, 3), (230, 238, 242), np.uint8)
        cv2.rectangle(panel, (0, 0), (PANEL_WIDTH, 31), (18, 31, 43), -1)
        cv2.putText(
            panel,
            title,
            (9, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        status = "FUTURE LOSS" if bool(valid) else "CONTEXT / EXCLUDED"
        cv2.putText(
            panel,
            f"latent {int(latent_id):02d}/12  representative frame {int(frame_id):02d}/48",
            (12, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (26, 53, 72),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            panel,
            status,
            (12, 87),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.50,
            (36, 113, 142) if valid else (170, 87, 51),
            1,
            cv2.LINE_AA,
        )
        for line_id, line in enumerate(lines):
            cv2.putText(
                panel,
                line,
                (12, 122 + 24 * line_id),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.39,
                (53, 75, 89),
                1,
                cv2.LINE_AA,
            )
        output.append(panel)
    return np.stack(output)


def pool_depth_to_latents(
    depth_49f: np.ndarray,
    groups: list[list[int]],
    valid_latents: np.ndarray,
    future_start_frame: int,
) -> np.ndarray:
    pooled = np.zeros((len(groups), *depth_49f.shape[1:]), dtype=np.float32)
    for latent_id, group in enumerate(groups):
        frame_ids = [int(item) for item in group if int(item) >= future_start_frame]
        if bool(valid_latents[latent_id]) and frame_ids:
            pooled[latent_id] = depth_49f[frame_ids].mean(axis=0)
    return pooled


def build_page(output_dir: Path, metadata: dict) -> None:
    sections = []
    for level in metadata["levels"]:
        scalar = level["scalar_losses_latent_axis"]
        metrics = [
            f"Flow {level['loss_main']:.5f}",
            f"Depth {scalar['depth_unweighted']:.5f}",
        ]
        metrics.extend(
            f"{item['label']} {item['depth_weighted']:.5f}"
            for item in level["masks"]
        )
        metric_html = "".join(f"<b>{html.escape(item)}</b>" for item in metrics)
        sections.append(
            f'''<section><header><div><span>LATENT AXIS · 13 FRAMES</span>
<h2>σ={level['sigma']:.4f} · timestep={level['timestep']:.1f}</h2></div>
<div class="metrics">{metric_html}</div></header>
<video controls muted loop playsinline preload="metadata" src="{html.escape(level['folder'])}/latent_overlay_grid.mp4"></video></section>'''
        )
    prompt = html.escape(metadata["sample"].get("caption", ""))
    document = f'''<!doctype html><html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Flow-guided depth loss · latent overlays</title><style>
:root{{--paper:#edf1f2;--ink:#172f41;--muted:#607483;--line:#b7c7cf;--accent:#d85e3a;--blue:#28708d}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--paper);color:var(--ink);font:15px/1.55 system-ui,sans-serif}}.mast{{padding:27px max(24px,4vw);background:#dce8ed;border-bottom:1px solid var(--line)}}.eyebrow,section span{{font:800 10px ui-monospace,monospace;letter-spacing:.14em;color:var(--accent)}}h1{{font:600 clamp(29px,4vw,47px)/1.04 Georgia,serif;margin:7px 0 12px}}.mast p{{max-width:1250px;margin:6px 0;color:var(--muted)}}.prompt{{margin-top:12px;padding:9px 12px;background:#f5f9fa;border-left:5px solid var(--blue)}}main{{padding:25px max(24px,4vw) 90px}}section{{background:#ffffffa8;border:1px solid var(--line);padding:18px;box-shadow:0 7px 20px #24475d12}}section+section{{margin-top:25px}}section header{{display:flex;justify-content:space-between;align-items:end;gap:18px;padding-bottom:11px;margin-bottom:12px;border-bottom:1px solid var(--line)}}h2{{font:600 21px Georgia,serif;margin:4px 0 0}}.metrics{{display:flex;gap:6px;flex-wrap:wrap;justify-content:end}}.metrics b{{background:var(--ink);color:white;padding:5px 7px;font:700 9px ui-monospace,monospace}}video{{display:block;width:100%;aspect-ratio:1536/648;background:#122431}}code{{font-size:12px}}#replay{{position:fixed;right:21px;bottom:20px;border:0;background:var(--accent);color:white;font-weight:800;padding:12px 16px;box-shadow:0 7px 20px #172f4160;cursor:pointer}}@media(max-width:720px){{section header{{align-items:start;flex-direction:column}}.metrics{{justify-content:start}}}}
</style></head><body><header class="mast"><div class="eyebrow">FLOW-MATCHING RESIDUAL → HARD MASK → DEPTH LOSS WEIGHT</div><h1>Flow-guided Depth Loss<br>原生 Latent 时间轴</h1><p>所有视频严格为 DiT 原生 13-step 时间轴；背景帧采用每个 TinyVAE causal group 的代表视频帧，不进行时间插值。前三个 latent 是条件区间，loss 与 mask 置零。</p><p>Flow mask 在原生 32×56 flow residual 上按每个样本、每个 σ 分别取 Top-20% / Top-10% / Top-5%，随后 nearest-neighbor 上采样。加权图使用 <code>w=1+2×mask</code>，再除以有效区域平均权重，使 depth loss 总尺度可比较。Flow 使用自己的全局 P99；原始和加权 Depth 共用同一个全局 P99。</p><div class="prompt"><b>Prompt：</b>{prompt}</div></header><main>{''.join(sections)}</main><button id="replay">全部重新播放</button><script>document.getElementById('replay').onclick=()=>document.querySelectorAll('video').forEach(v=>{{v.currentTime=0;v.play().catch(()=>{{}})}});</script></body></html>'''
    (output_dir / "index.html").write_text(document, encoding="utf-8")


def main() -> None:
    args = parse_args()
    flow_root = args.flow_root.expanduser().resolve()
    depth_root = args.depth_root.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    if output_dir.exists() and not args.rebuild:
        raise FileExistsError(f"Output already exists: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=args.rebuild)

    flow_metadata = json.loads((flow_root / "metadata.json").read_text(encoding="utf-8"))
    depth_metadata = json.loads((depth_root / "metadata.json").read_text(encoding="utf-8"))
    for key in ("config", "sample_index", "seed", "same_noise_across_levels"):
        if flow_metadata[key] != depth_metadata[key]:
            raise ValueError(f"Source metadata mismatch for {key}: {flow_metadata[key]!r} vs {depth_metadata[key]!r}")
    if flow_metadata["sample"] != depth_metadata["sample"]:
        raise ValueError("Flow and depth sources use different samples")

    timeline = flow_metadata["timeline"]
    groups = timeline["flow_latent_to_video_frame_groups"]
    representative_frames = np.asarray(
        timeline["flow_native_background_frames"], dtype=np.int64
    )
    latent_ids = np.arange(len(groups), dtype=np.int64)
    output_fps = int(timeline["flow_native_fps"])
    future_start = int(depth_metadata["future_frames"][0])
    flow_scale = float(flow_metadata["flow_global_p99"])
    depth_scale = float(depth_metadata["loss_global_p99"])
    gt_rgb = read_rgb_video(depth_root / "gt.mp4")[representative_frames]

    depth_by_folder = {item["folder"]: item for item in depth_metadata["levels"]}
    output_levels = []
    for flow_level in flow_metadata["levels"]:
        folder = flow_level["folder"]
        depth_level = depth_by_folder.get(folder)
        if depth_level is None or not np.isclose(flow_level["sigma"], depth_level["sigma"]):
            raise ValueError(f"Missing or mismatched depth level for {folder}")
        with np.load(flow_root / folder / "native_loss_maps.npz") as data:
            flow_native = data["flow"].astype(np.float32)
        with np.load(depth_root / folder / "depth_loss_maps.npz") as data:
            depth_49f = data["depth_loss"].astype(np.float32)
        valid_latents = flow_native.reshape(flow_native.shape[0], -1).max(axis=1) > 0
        depth_latent = pool_depth_to_latents(
            depth_49f,
            groups,
            valid_latents,
            future_start,
        )
        pred_rgb = read_rgb_video(depth_root / folder / "pred_x0.mp4")[
            representative_frames
        ]
        height, width = pred_rgb.shape[1:3]
        flow_up = resize_maps(flow_native, height, width, cv2.INTER_LINEAR)
        valid_pixels = np.broadcast_to(
            valid_latents[:, None, None], depth_latent.shape
        )
        valid_flow_values = flow_native[valid_latents].reshape(-1)
        if not valid_flow_values.size:
            raise RuntimeError(f"No valid future flow values in {folder}")

        raw_depth_overlay = heatmap_overlay(pred_rgb, depth_latent, depth_scale)
        flow_overlay = heatmap_overlay(pred_rgb, flow_up, flow_scale)
        masks = []
        mask_panels = []
        weighted_panels = []
        mask_maps = {}
        weighted_maps = {}
        for quantile, label in QUANTILES:
            threshold = float(np.quantile(valid_flow_values, quantile))
            mask_native = (flow_native >= threshold) & valid_latents[:, None, None]
            mask_up = resize_maps(
                mask_native.astype(np.float32),
                height,
                width,
                cv2.INTER_NEAREST,
            )
            mask_up = mask_up > 0.5
            mask_maps[label] = mask_up
            weights = 1.0 + MASK_ALPHA * mask_up.astype(np.float32)
            mean_weight = float(weights[valid_pixels].mean())
            weighted_depth = depth_latent * weights / max(mean_weight, 1e-12)
            weighted_depth[~valid_pixels] = 0.0
            weighted_maps[label] = weighted_depth
            mask_fraction = float(mask_native[valid_latents].mean())
            weighted_scalar = float(weighted_depth[valid_pixels].mean())
            masks.append(
                {
                    "label": label,
                    "quantile": quantile,
                    "threshold": threshold,
                    "native_mask_fraction": mask_fraction,
                    "depth_weighted": weighted_scalar,
                    "mean_weight_before_normalization": mean_weight,
                }
            )
            mask_panels.append(mask_overlay(pred_rgb, mask_up))
            weighted_panels.append(
                heatmap_overlay(pred_rgb, weighted_depth, depth_scale)
            )

        base_scalar = float(depth_latent[valid_pixels].mean())
        metrics_lines = [
            f"Depth base {base_scalar:.5f}",
            f"Top-20 {masks[0]['depth_weighted']:.5f}  Top-10 {masks[1]['depth_weighted']:.5f}",
            f"Top-5 {masks[2]['depth_weighted']:.5f}  hard mask alpha=2",
        ]
        panel_list = [
            labeled_panel(gt_rgb, "GT RGB | representative frame"),
            labeled_panel(pred_rgb, "Pred x0 RGB | representative frame"),
            labeled_panel(flow_overlay, "Flow-matching loss overlay"),
            labeled_panel(raw_depth_overlay, "Depth loss overlay | unweighted"),
            labeled_panel(mask_panels[0], "Flow mask | Top-20%"),
            labeled_panel(weighted_panels[0], "Depth loss | Top-20% weighted"),
            labeled_panel(mask_panels[1], "Flow mask | Top-10%"),
            labeled_panel(weighted_panels[1], "Depth loss | Top-10% weighted"),
            labeled_panel(mask_panels[2], "Flow mask | Top-5%"),
            labeled_panel(weighted_panels[2], "Depth loss | Top-5% weighted"),
            information_panel(
                latent_ids=latent_ids,
                representative_frames=representative_frames,
                valid_latents=valid_latents,
                title="Native latent alignment",
                lines=["13-step axis; no temporal interpolation", "Flow mask at 32x56, nearest resize"],
            ),
            information_panel(
                latent_ids=latent_ids,
                representative_frames=representative_frames,
                valid_latents=valid_latents,
                title="Scalar loss on shared future latents",
                lines=metrics_lines,
            ),
        ]
        rows = [np.concatenate(panel_list[start : start + 4], axis=2) for start in (0, 4, 8)]
        composite = np.concatenate(rows, axis=1)
        level_dir = output_dir / folder
        level_dir.mkdir(parents=True, exist_ok=args.rebuild)
        write_mp4(level_dir / "latent_overlay_grid.mp4", composite, output_fps)
        np.savez_compressed(
            level_dir / "latent_flow_guided_depth_maps.npz",
            flow_native=flow_native,
            depth_unweighted=depth_latent,
            mask_top20=mask_maps["Top-20%"].astype(np.uint8),
            mask_top10=mask_maps["Top-10%"].astype(np.uint8),
            mask_top05=mask_maps["Top-5%"].astype(np.uint8),
            depth_weighted_top20=weighted_maps["Top-20%"],
            depth_weighted_top10=weighted_maps["Top-10%"],
            depth_weighted_top05=weighted_maps["Top-5%"],
            representative_frames=representative_frames,
            valid_latents=valid_latents,
        )
        output_levels.append(
            {
                "folder": folder,
                "scheduler_index": flow_level["scheduler_index"],
                "sigma_target": flow_level["sigma_target"],
                "sigma": flow_level["sigma"],
                "timestep": flow_level["timestep"],
                "loss_main": flow_level["loss_main"],
                "loss_depth_video_axis": depth_level["loss_depth"],
                "scalar_losses_latent_axis": {"depth_unweighted": base_scalar},
                "masks": masks,
            }
        )
        print(f"[flow-guided-depth] rendered {folder}", flush=True)

    metadata = {
        "flow_source": str(flow_root),
        "depth_source": str(depth_root),
        "sample": flow_metadata["sample"],
        "sample_index": flow_metadata["sample_index"],
        "seed": flow_metadata["seed"],
        "same_noise_across_levels": True,
        "time_axis": "DiT native latent",
        "frames": len(groups),
        "fps": output_fps,
        "resolution": [PANEL_HEIGHT * 3, PANEL_WIDTH * 4],
        "representative_video_frames": representative_frames.tolist(),
        "temporal_interpolation": False,
        "flow_native_shape": [13, 32, 56],
        "flow_scale_global_p99": flow_scale,
        "depth_scale_global_p99": depth_scale,
        "mask_quantiles": [item[0] for item in QUANTILES],
        "mask_alpha": MASK_ALPHA,
        "weight_normalization": "divide by mean weight over shared valid future latents",
        "levels": output_levels,
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    build_page(output_dir, metadata)
    HUB_LINK.parent.mkdir(parents=True, exist_ok=True)
    if HUB_LINK.is_symlink() or HUB_LINK.is_file():
        HUB_LINK.unlink()
    HUB_LINK.symlink_to(output_dir)
    print(f"[flow-guided-depth] page: {HUB_LINK}", flush=True)


if __name__ == "__main__":
    main()
