#!/usr/bin/env python3
"""Post-hoc PyBullet trajectory-loss evaluation and overlay report.

The evaluator reuses the fixed prepared inputs from the train-subset validation
page.  This keeps timestep, noise, context conditioning, and case selection
identical to the existing ``loss_main`` scores while adding frozen CoTracker
object-trajectory metrics.
"""

from __future__ import annotations

import argparse
import html
import json
from fractions import Fraction
from pathlib import Path
import sys
from typing import Any

import av
import cv2
import numpy as np
import torch
import torch.nn.functional as F


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
PACKAGE_ROOT = EXPERIMENT_ROOT.parents[2]
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
COTRACKER_ROOT = Path("/home/gaoya/Code_Video/co-tracker-main")
TRAJECTORY_ROOT = EXPERIMENT_ROOT / "object_cotracker_trajectory_project"
for _path in (
    HERE,
    EXPERIMENT_ROOT,
    TRAIN_XSSC_ROOT,
    PACKAGE_ROOT,
    DIFFSYNTH_ROOT,
    COTRACKER_ROOT,
    TRAJECTORY_ROOT,
):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import code_vjepa_vggt.context_wan_v_newtrain as context_wan  # noqa: E402
import evaluate_train_subset_val_loss as evaluator  # noqa: E402
from object_trajectory_loss import (  # noqa: E402
    object_equal_visibility_aware_trajectory_loss,
)
from trajectory_cache import PyBulletTrajectoryCache  # noqa: E402
from train_xssc_object_self_attn_lora_trajectory_loss import (  # noqa: E402
    TRACK_HEIGHT,
    TRACK_WIDTH,
    TrajectoryLossWanModule,
    differentiable_track_video_with_scores,
)
from vjepa_loss_project.train_xssc_object_self_attn_lora_vjepa_loss import (  # noqa: E402
    _load_tiny_vae,
)


DEFAULT_CONFIG = HERE / "validation_30cases_config.json"
DEFAULT_INPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_train_validation_30cases"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/xssc_trajectory_validation_preview"
)
DEFAULT_TRAJECTORY_CACHE = Path(
    "/data/gaoya/agent-data/cache/pybullet0713_object_cotracker_trajectory_v1"
)
DEFAULT_COTRACKER = Path(
    "/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"
)
DEFAULT_TINY_VAE_ROOT = Path("/home/gaoya/Code_Video/taehv")
DEFAULT_TINY_VAE_CHECKPOINT = Path("/data/gaoya/ckpt/taew2_2.pth")
DEFAULT_ENTRY = "full_sa_no_object_inventory_latest_step3000"
FPS = 8.0
ANCHOR_FRAME = 4
FUTURE_START_FRAME = 8
POINTS_PER_OBJECT = 24
HUBER_DELTA = 0.01
VISIBILITY_THRESHOLD = 0.9
VISIBILITY_LOSS_WEIGHT = 0.05
OBJECT_COLORS = np.asarray(
    [
        (238, 75, 71),
        (17, 150, 141),
        (45, 107, 185),
        (226, 167, 36),
        (154, 83, 170),
        (80, 170, 80),
    ],
    dtype=np.uint8,
)


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def read_video(path: Path, expected_frames: int = 49) -> np.ndarray:
    container = av.open(str(path))
    try:
        frames = [frame.to_ndarray(format="rgb24") for frame in container.decode(video=0)]
    finally:
        container.close()
    result = np.stack(frames).astype(np.uint8)
    if result.shape[0] != expected_frames:
        raise RuntimeError(f"expected {expected_frames} frames in {path}, got {result.shape}")
    return result


def decode_tiny_vae(tiny_vae, tiny_vae_apply, latents: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    latent_ntchw = latents.permute(0, 2, 1, 3, 4).contiguous()
    with torch.autocast(
        device_type=latent_ntchw.device.type,
        dtype=dtype,
        enabled=latent_ntchw.device.type == "cuda",
    ):
        video = tiny_vae_apply(tiny_vae.decoder, latent_ntchw, False, False)
        if tiny_vae.patch_size > 1:
            video = F.pixel_shuffle(video, tiny_vae.patch_size)
    if not (tiny_vae.is_cogvideox and latent_ntchw.shape[1] % 2 == 0):
        video = video[:, tiny_vae.frames_to_trim :]
    return video


def video_to_uint8(video: torch.Tensor) -> np.ndarray:
    item = video[0].detach().float().clamp(0.0, 1.0)
    return (item * 255.0).round().to(torch.uint8).permute(0, 2, 3, 1).cpu().numpy()


def write_mp4(path: Path, frames: np.ndarray, fps: float = FPS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    container = av.open(str(path), mode="w")
    stream = container.add_stream("libx264", rate=Fraction(str(float(fps))))
    stream.width = int(frames.shape[2])
    stream.height = int(frames.shape[1])
    stream.pix_fmt = "yuv420p"
    stream.options = {"crf": "18", "preset": "medium"}
    try:
        for image in frames:
            frame = av.VideoFrame.from_ndarray(image, format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    finally:
        container.close()


def load_cache() -> PyBulletTrajectoryCache:
    return PyBulletTrajectoryCache(
        DEFAULT_TRAJECTORY_CACHE,
        num_frames=49,
        anchor_frame=ANCHOR_FRAME,
        points_per_object=POINTS_PER_OBJECT,
        track_height=TRACK_HEIGHT,
        track_width=TRACK_WIDTH,
    )


def predict_x0(
    model: torch.nn.Module,
    prepared: tuple[dict, dict, dict],
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Run the page's one-step loss path while retaining its model output."""
    shared, positive, _negative = evaluator.transfer_prepared(model, prepared)
    captured: list[dict[str, Any]] = []
    pipe = model.pipe
    original_model_fn = pipe.model_fn

    def capture_model_fn(*args, **kwargs):
        output = original_model_fn(*args, **kwargs)
        captured.append(
            {
                "model_output": output,
                "latents": kwargs.get("latents"),
                "timestep": kwargs.get("timestep"),
                "inputs": kwargs,
            }
        )
        return output

    pipe.model_fn = capture_model_fn
    try:
        with torch.inference_mode():
            _loss, metrics = model._compute_object_losses(pipe, shared, positive)
    finally:
        pipe.model_fn = original_model_fn
    if len(captured) != 1:
        raise RuntimeError(f"expected exactly one DiT forward, captured {len(captured)}")
    record = captured[0]
    sigma = context_wan._diffsynth_sigma_for_timestep(
        pipe.scheduler, record["timestep"]
    ).to(device=record["latents"].device, dtype=record["latents"].dtype)
    while sigma.ndim < record["latents"].ndim:
        sigma = sigma.unsqueeze(-1)
    pred_x0 = record["latents"] - sigma * record["model_output"]
    pred_x0 = TrajectoryLossWanModule._restore_condition_latents(
        pred_x0, shared["input_latents"], record["inputs"]
    )
    numeric = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and np.isfinite(float(value))
    }
    return pred_x0.detach(), numeric


def evaluate_trajectory(
    predictor,
    pred_raw: torch.Tensor,
    cache_record: dict[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    object_count = int(cache_record["object_count"])
    points_per_object = int(cache_record["points_per_object"])
    if points_per_object != POINTS_PER_OBJECT:
        raise RuntimeError(f"unexpected points_per_object={points_per_object}")
    query_points = cache_record["query_points"].to(device=device, dtype=torch.float32).reshape(-1, 2)
    frame_ids = torch.full(
        (query_points.shape[0], 1),
        float(ANCHOR_FRAME),
        device=device,
        dtype=query_points.dtype,
    )
    queries = torch.cat((frame_ids, query_points), dim=-1).unsqueeze(0)
    # The project CoTracker wrapper follows the training path and accepts
    # [B, C, T, H, W], while the tracker itself flattens C*T internally.
    tracker_video = pred_raw.float().mul(255.0)
    with torch.inference_mode():
        pred_tracks, pred_visibility_probability, pred_confidence_probability = (
            differentiable_track_video_with_scores(predictor, tracker_video, queries)
        )
    gt_tracks = cache_record["gt_tracks"].to(device=device, dtype=torch.float32).reshape(
        1, 49, object_count * points_per_object, 2
    )
    gt_visibility_probability = cache_record["gt_visibility_probability"].to(
        device=device, dtype=torch.float32
    ).reshape(1, 49, object_count * points_per_object)
    gt_confidence_probability = cache_record["gt_confidence_probability"].to(
        device=device, dtype=torch.float32
    ).reshape(1, 49, object_count * points_per_object)
    gt_geometric_visibility = cache_record["gt_geometric_visibility"].to(
        device=device
    ).bool().reshape(1, 49, object_count * points_per_object)
    loss, diagnostics = object_equal_visibility_aware_trajectory_loss(
        pred_tracks.float(),
        gt_tracks,
        gt_visibility_probability,
        gt_confidence_probability,
        pred_visibility_probability.float(),
        gt_geometric_visibility,
        object_count=object_count,
        points_per_object=points_per_object,
        height=TRACK_HEIGHT,
        width=TRACK_WIDTH,
        anchor_frame=ANCHOR_FRAME,
        future_start_frame=FUTURE_START_FRAME,
        huber_delta=HUBER_DELTA,
        visibility_threshold=VISIBILITY_THRESHOLD,
        visibility_loss_weight=VISIBILITY_LOSS_WEIGHT,
    )
    metrics = {
        "trajectory_loss": float(loss.item()),
        "trajectory_coordinate_loss": float(diagnostics["coordinate_loss"].item()),
        "trajectory_visibility_loss": float(diagnostics["visibility_loss"].item()),
        "trajectory_visibility_penalty": float(
            VISIBILITY_LOSS_WEIGHT * diagnostics["visibility_loss"].item()
        ),
        "trajectory_normalized_ade": float(diagnostics["normalized_ade"].item()),
        "trajectory_normalized_rmse": float(diagnostics["normalized_rmse"].item()),
        "trajectory_gt_motion": float(diagnostics["normalized_gt_motion"].item()),
        "trajectory_valid_fraction": float(diagnostics["valid_fraction"].item()),
        "trajectory_effective_weight_fraction": float(
            diagnostics["effective_weight_fraction"].item()
        ),
        "trajectory_valid_object_fraction": float(
            diagnostics["valid_object_fraction"].item()
        ),
        "trajectory_object_count": object_count,
        "trajectory_points_per_object": points_per_object,
        "trajectory_definition": "F04-relative displacement, F08-F48, equal-object visibility-aware SmoothL1",
        "trajectory_cache": str(DEFAULT_TRAJECTORY_CACHE),
        "trajectory_cotracker_checkpoint": str(DEFAULT_COTRACKER),
        "trajectory_huber_delta": HUBER_DELTA,
        "trajectory_visibility_loss_weight": VISIBILITY_LOSS_WEIGHT,
    }
    arrays = {
        "query_points": query_points.detach().cpu().numpy().astype(np.float32),
        "gt_tracks": gt_tracks[0].detach().cpu().numpy().astype(np.float32),
        "pred_tracks": pred_tracks[0].detach().cpu().numpy().astype(np.float32),
        "gt_geometric_visibility": gt_geometric_visibility[0].cpu().numpy().astype(np.uint8),
        "pred_visibility_probability": pred_visibility_probability[0].float().cpu().numpy(),
        "pred_visibility": (pred_visibility_probability[0] > VISIBILITY_THRESHOLD).cpu().numpy().astype(np.uint8),
        "object_ids_per_point": np.repeat(np.arange(object_count, dtype=np.int32), points_per_object),
    }
    return metrics, arrays


def trackres_to_native(tracks: np.ndarray, height: int, width: int) -> np.ndarray:
    result = tracks.astype(np.float32).copy()
    result[..., 0] *= float(width - 1) / float(TRACK_WIDTH - 1)
    result[..., 1] *= float(height - 1) / float(TRACK_HEIGHT - 1)
    return result


def draw_track_history(
    frame: np.ndarray,
    tracks: np.ndarray,
    visibility: np.ndarray,
    frame_id: int,
    colors: np.ndarray,
    ring: bool = False,
) -> None:
    start = min(ANCHOR_FRAME, frame_id)
    stop = max(ANCHOR_FRAME, frame_id)
    for point_id in range(tracks.shape[1]):
        color = tuple(int(v) for v in colors[point_id % len(colors)])
        history: list[tuple[int, int]] = []
        for index in range(start, stop + 1):
            if bool(visibility[index, point_id]):
                history.append(tuple(np.rint(tracks[index, point_id]).astype(int)))
            elif len(history) >= 2:
                cv2.polylines(frame, [np.asarray(history)], False, color, 2, cv2.LINE_AA)
                history = []
        if len(history) >= 2:
            cv2.polylines(frame, [np.asarray(history)], False, color, 2, cv2.LINE_AA)
        if bool(visibility[frame_id, point_id]):
            point = tuple(np.rint(tracks[frame_id, point_id]).astype(int))
            if ring:
                cv2.circle(frame, point, 7, (255, 255, 255), 2, cv2.LINE_AA)
            else:
                cv2.circle(frame, point, 5, color, -1, cv2.LINE_AA)
                cv2.circle(frame, point, 7, (10, 10, 10), 1, cv2.LINE_AA)


def add_panel_label(frame: np.ndarray, title: str, detail: str) -> None:
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], 56), (8, 12, 16), -1)
    cv2.addWeighted(overlay, 0.82, frame, 0.18, 0.0, frame)
    cv2.putText(frame, title, (14, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(frame, detail, (14, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (190, 207, 218), 1, cv2.LINE_AA)


def render_overlay(
    output_dir: Path,
    source_frames: np.ndarray,
    pred_frames: np.ndarray,
    arrays: dict[str, np.ndarray],
    metrics: dict[str, Any],
) -> None:
    height, width = source_frames.shape[1:3]
    gt_native = trackres_to_native(arrays["gt_tracks"], height, width)
    pred_native = trackres_to_native(arrays["pred_tracks"], height, width)
    point_colors = OBJECT_COLORS[
        arrays["object_ids_per_point"] % len(OBJECT_COLORS)
    ]
    gt_visible = arrays["gt_geometric_visibility"].astype(bool)
    pred_visible = arrays["pred_visibility"].astype(bool)
    frames = []
    for frame_id in range(source_frames.shape[0]):
        gt_panel = source_frames[frame_id].copy()
        pred_panel = pred_frames[frame_id].copy()
        compare_panel = pred_frames[frame_id].copy()
        draw_track_history(gt_panel, gt_native, gt_visible, frame_id, point_colors)
        draw_track_history(pred_panel, pred_native, pred_visible, frame_id, point_colors)
        draw_track_history(compare_panel, gt_native, gt_visible, frame_id, point_colors, ring=True)
        draw_track_history(compare_panel, pred_native, pred_visible, frame_id, point_colors)
        if frame_id >= FUTURE_START_FRAME:
            for point_id in range(gt_native.shape[1]):
                if bool(gt_visible[frame_id, point_id]):
                    start = tuple(np.rint(gt_native[frame_id, point_id]).astype(int))
                    stop = tuple(np.rint(pred_native[frame_id, point_id]).astype(int))
                    cv2.line(compare_panel, start, stop, (255, 72, 72), 1, cv2.LINE_AA)
        detail = f"F{frame_id:02d} | trajectory={metrics['trajectory_loss']:.5f}"
        add_panel_label(gt_panel, "GT RGB + GT geometric tracks", detail)
        add_panel_label(pred_panel, "Tiny-VAE x0_pred + predicted tracks", detail)
        add_panel_label(compare_panel, "white=GT, color=pred, red=error", detail)
        frames.append(np.concatenate((gt_panel, pred_panel, compare_panel), axis=1))
    rendered = np.stack(frames).astype(np.uint8)
    write_mp4(output_dir / "trajectory_overlay.mp4", rendered)
    cv2.imwrite(
        str(output_dir / "trajectory_preview.jpg"),
        cv2.cvtColor(rendered[min(32, rendered.shape[0] - 1)], cv2.COLOR_RGB2BGR),
        [cv2.IMWRITE_JPEG_QUALITY, 94],
    )


def build_html(output_root: Path, entry: dict[str, Any], cases: list[dict[str, Any]]) -> Path:
    cards = []
    for index, case in enumerate(cases, start=1):
        rel = Path("cases") / case["case_id"]
        cards.append(
            f"""<section class=case><div class=head><div><small>CASE {index:02d}</small><h2>{html.escape(case['case_id'])}</h2><p>{html.escape(case.get('prompt',''))}</p></div><div class=loss><b>{case['metrics']['trajectory_loss']:.6f}</b><span>trajectory loss</span></div></div>
<div class=metrics><span><b>{case['metrics']['trajectory_coordinate_loss']:.6f}</b>coordinate</span><span><b>{case['metrics']['trajectory_visibility_penalty']:.6f}</b>visibility penalty</span><span><b>{case['metrics']['trajectory_normalized_ade']:.5f}</b>normalized ADE</span><span><b>{case['metrics']['trajectory_valid_fraction']:.1%}</b>GT valid fraction</span><span><b>{case['metrics']['trajectory_object_count']}</b>objects</span></div>
<video controls muted loop playsinline preload=metadata poster="{rel}/trajectory_preview.jpg" src="{rel}/trajectory_overlay.mp4"></video><p><a href="{rel}/metrics.json">metrics.json</a> · <a href="{rel}/trajectories.npz">tracks</a></p></section>"""
        )
    document = f"""<!doctype html><html lang=zh-CN><head><meta charset=utf-8><meta name=viewport content=width=device-width,initial-scale=1><title>Trajectory loss preview</title><style>
:root{{--bg:#101719;--panel:#182326;--ink:#ecf4f1;--muted:#9aada8;--line:#30413e;--accent:#62d0b3;--gold:#f3bd58}}*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#091014,#16231f);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}main{{max-width:1500px;margin:auto;padding:28px 24px 70px}}h1{{font:500 42px Georgia,serif;margin:0 0 8px}}p{{color:var(--muted)}}.summary{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}.pill{{padding:9px 13px;border:1px solid var(--line);background:var(--panel);border-radius:8px}}.case{{padding:24px 0 32px;border-top:1px solid var(--line)}}.head{{display:flex;justify-content:space-between;gap:18px;align-items:end}}h2{{margin:3px 0;font-size:22px}}small{{color:var(--accent);font-weight:700;letter-spacing:0}}.loss{{border-left:4px solid var(--gold);padding:4px 12px;text-align:right}}.loss b{{display:block;color:var(--gold);font-size:25px}}.loss span{{color:var(--muted)}}.metrics{{display:grid;grid-template-columns:repeat(5,1fr);border:1px solid var(--line);background:var(--panel);margin:14px 0}}.metrics span{{padding:10px 12px;border-right:1px solid var(--line);color:var(--muted)}}.metrics span:last-child{{border:0}}.metrics b{{display:block;color:var(--ink);font-size:17px}}video{{display:block;width:100%;aspect-ratio:21/4;object-fit:contain;background:#050708;border:1px solid var(--line)}}a{{color:var(--accent)}}@media(max-width:850px){{h1{{font-size:34px}}.head{{display:block}}.loss{{text-align:left;margin-top:10px}}.metrics{{grid-template-columns:repeat(2,1fr)}}.metrics span{{border-bottom:1px solid var(--line)}}video{{aspect-ratio:16/9}}}}</style></head><body><main><h1>Trajectory loss · overlay preview</h1><p>Current validation-page inputs · frozen CoTracker3 scaled_offline · F04-relative displacement · F08–F48 · equal-object visibility-aware SmoothL1.</p><div class=summary><span class=pill>checkpoint <b>{html.escape(entry['method_label'])}</b></span><span class=pill>step <b>{entry['step']}</b></span><span class=pill>cases <b>{len(cases)}</b></span><span class=pill>ranking direction <b>lower is better</b></span></div>{''.join(cards)}</main></body></html>"""
    path = output_root / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--entry-id", default=DEFAULT_ENTRY)
    parser.add_argument("--case-count", type=int, default=3)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trajectory-cache", type=Path, default=DEFAULT_TRAJECTORY_CACHE)
    parser.add_argument("--cotracker-checkpoint", type=Path, default=DEFAULT_COTRACKER)
    parser.add_argument("--tiny-vae-root", type=Path, default=DEFAULT_TINY_VAE_ROOT)
    parser.add_argument("--tiny-vae-checkpoint", type=Path, default=DEFAULT_TINY_VAE_CHECKPOINT)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if str(args.device).startswith("cuda:4"):
        raise ValueError("GPU4 is prohibited by workspace rules")
    args.config = args.config.expanduser().resolve()
    args.input_root = args.input_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.trajectory_cache = args.trajectory_cache.expanduser().resolve()
    args.cotracker_checkpoint = args.cotracker_checkpoint.expanduser().resolve()
    args.tiny_vae_root = args.tiny_vae_root.expanduser().resolve()
    args.tiny_vae_checkpoint = args.tiny_vae_checkpoint.expanduser().resolve()
    payload = json.loads(args.config.read_text(encoding="utf-8"))
    entries = evaluator.inventory_entries(args.config)
    matching = [entry for entry in entries if entry["entry_id"] == args.entry_id]
    if len(matching) != 1:
        raise ValueError(f"entry-id not found or duplicated: {args.entry_id}")
    entry = matching[0]
    manifest = json.loads(Path(payload["cases_manifest"]).read_text(encoding="utf-8"))
    cases = [case for case in manifest["cases"] if case["source"] == "pybullet"]
    if args.case_id:
        wanted = set(args.case_id)
        cases = [case for case in cases if case["case_id"] in wanted]
    else:
        cases = cases[: int(args.case_count)]
    if not cases:
        raise ValueError("no cases selected")
    args.output_root.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    checkpoint = Path(entry["checkpoint"]).expanduser().resolve()
    manifest_path = evaluator.find_manifest(checkpoint)
    print(f"[model] loading {entry['entry_id']} on {device}", flush=True)
    model, _model_args, model_config, model_kind = evaluator.build_model(manifest_path, device)
    load_info = evaluator.load_checkpoint(model, checkpoint)
    datasets = evaluator.build_source_datasets(
        evaluator.load_resolved(Path(entry["config"]).expanduser().resolve())
    )
    cache = PyBulletTrajectoryCache(
        args.trajectory_cache,
        num_frames=49,
        anchor_frame=ANCHOR_FRAME,
        points_per_object=POINTS_PER_OBJECT,
        track_height=TRACK_HEIGHT,
        track_width=TRACK_WIDTH,
    )
    print("[tracker] loading frozen CoTracker3", flush=True)
    from cotracker.predictor import CoTrackerPredictor

    predictor = CoTrackerPredictor(
        checkpoint=str(args.cotracker_checkpoint), offline=True, v2=False, window_len=60
    ).to(device).eval().requires_grad_(False)
    tiny_vae, tiny_vae_apply = _load_tiny_vae(
        args.tiny_vae_root, args.tiny_vae_checkpoint, device, model.pipe.torch_dtype
    )
    tiny_vae.eval().requires_grad_(False)
    case_rows: list[dict[str, Any]] = []
    try:
        for position, case in enumerate(cases, start=1):
            case_dir = args.output_root / "cases" / case["case_id"]
            metrics_path = case_dir / "metrics.json"
            if metrics_path.is_file() and not args.overwrite:
                case_rows.append(json.loads(metrics_path.read_text(encoding="utf-8")))
                continue
            print(f"[case {position}/{len(cases)}] {case['case_id']}", flush=True)
            prepared = evaluator.prepare_inputs(
                model, datasets["pybullet"], case, args.input_root
            )
            pred_x0, main_metrics = predict_x0(model, prepared)
            with torch.inference_mode():
                pred_raw = decode_tiny_vae(
                    tiny_vae, tiny_vae_apply, pred_x0, model.pipe.torch_dtype
                )
                pred_frames = video_to_uint8(pred_raw)
            if pred_frames.shape != (49, 512, 896, 3):
                raise RuntimeError(f"unexpected decoded video shape: {pred_frames.shape}")
            trajectory_metrics, arrays = evaluate_trajectory(
                predictor, pred_raw, cache.load(case["sample_key"]), device
            )
            source_frames = read_video(Path(case["gt_video"]))
            case_dir.mkdir(parents=True, exist_ok=True)
            write_mp4(case_dir / "pred_x0.mp4", pred_frames)
            write_mp4(case_dir / "gt.mp4", source_frames)
            np.savez_compressed(case_dir / "trajectories.npz", **arrays)
            render_overlay(case_dir, source_frames, pred_frames, arrays, trajectory_metrics)
            full_metrics = {
                **case,
                **trajectory_metrics,
                "loss_main": main_metrics.get("train/loss_main"),
                "main_metrics": main_metrics,
                "checkpoint": str(checkpoint),
                "method_label": entry["method_label"],
                "entry_id": entry["entry_id"],
                "model_kind": model_kind,
                "model_manifest": str(manifest_path),
                "load_info": load_info,
            }
            atomic_json(metrics_path, full_metrics)
            case_rows.append(full_metrics)
            del prepared, pred_x0, pred_raw, pred_frames, arrays
            torch.cuda.empty_cache()
    finally:
        del tiny_vae, tiny_vae_apply, predictor, model
        torch.cuda.empty_cache()
    report = {
        "schema_version": 1,
        "state": "complete",
        "entry": entry,
        "checkpoint": str(checkpoint),
        "model_kind": model_kind,
        "model_manifest": str(manifest_path),
        "trajectory_cache": str(args.trajectory_cache),
        "cases": case_rows,
        "mean_trajectory_loss": float(np.mean([row["trajectory_loss"] for row in case_rows])),
        "mean_trajectory_coordinate_loss": float(np.mean([row["trajectory_coordinate_loss"] for row in case_rows])),
        "mean_trajectory_visibility_penalty": float(np.mean([row["trajectory_visibility_penalty"] for row in case_rows])),
    }
    atomic_json(args.output_root / "report.json", report)
    atomic_json(args.output_root / "run_manifest.json", {"entry": entry, "cases": cases, "load_info": load_info})
    index = build_html(args.output_root, entry, case_rows)
    print(json.dumps({"index": str(index), "mean_trajectory_loss": report["mean_trajectory_loss"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
