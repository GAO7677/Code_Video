from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
import torch

from .config import BenchConfig
from .manifest import BenchSample


def _load_last_context_frame(sample: BenchSample) -> np.ndarray | None:
    if sample.image_path:
        return np.array(Image.open(sample.image_path).convert("RGB"))

    if sample.context_frame_paths:
        return np.array(Image.open(sample.context_frame_paths[-1]).convert("RGB"))

    if sample.context_frames_dir:
        frame_dir = Path(sample.context_frames_dir)
        image_paths = sorted([p for p in frame_dir.iterdir() if p.is_file()])
        image_paths = [p for p in image_paths if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}]
        if image_paths:
            return np.array(Image.open(image_paths[-1]).convert("RGB"))
    return None


def _load_video_frames(video_path: str, max_frames: int = -1) -> list[np.ndarray]:
    import decord

    decord.bridge.set_bridge("native")
    vr = decord.VideoReader(video_path, num_threads=1)
    num_frames = len(vr) if max_frames < 0 else min(len(vr), max_frames)
    frames = vr.get_batch(range(num_frames)).asnumpy()
    return [frame.astype(np.uint8) for frame in frames]


def _to_float01(frame: np.ndarray) -> np.ndarray:
    return frame.astype(np.float32) / 255.0


def _mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2))


def _align(a: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    h = min(a.shape[0], b.shape[0])
    w = min(a.shape[1], b.shape[1])
    return a[:h, :w], b[:h, :w]


class LPIPSScorer:
    def __init__(self, net: str = "alex") -> None:
        try:
            import lpips  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "LPIPS is not installed. Run `pip install lpips` or disable it in config."
            ) from exc
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = lpips.LPIPS(net=net).to(self.device).eval()

    def score(self, a: np.ndarray, b: np.ndarray) -> float:
        a, b = _align(a, b)
        ta = torch.from_numpy(a).permute(2, 0, 1).float() / 127.5 - 1.0
        tb = torch.from_numpy(b).permute(2, 0, 1).float() / 127.5 - 1.0
        with torch.no_grad():
            value = self.model(ta.unsqueeze(0).to(self.device), tb.unsqueeze(0).to(self.device))
        return float(value.item())


def _boundary_metrics(context_last: np.ndarray, generated_first: np.ndarray, lpips_scorer: LPIPSScorer | None) -> dict[str, float]:
    context_last, generated_first = _align(context_last, generated_first)
    context_f = _to_float01(context_last)
    generated_f = _to_float01(generated_first)

    metrics = {
        "boundary_mse": _mse(context_last, generated_first),
        "boundary_psnr": float(peak_signal_noise_ratio(context_f, generated_f, data_range=1.0)),
        "boundary_ssim": float(structural_similarity(context_f, generated_f, channel_axis=2, data_range=1.0)),
    }
    if lpips_scorer is not None:
        metrics["boundary_lpips"] = lpips_scorer.score(context_last, generated_first)
    return metrics


def _future_metrics(
    gen_frames: list[np.ndarray],
    gt_frames: list[np.ndarray],
    lpips_scorer: LPIPSScorer | None,
    gen_start_frame: int,
    gt_start_frame: int,
) -> dict[str, float]:
    gen_eval = gen_frames[gen_start_frame:]
    gt_eval = gt_frames[gt_start_frame:]
    pair_count = min(len(gen_eval), len(gt_eval))
    if pair_count <= 0:
        return {}

    psnr_values = []
    ssim_values = []
    lpips_values = []
    mse_values = []
    for idx in range(pair_count):
        gen, gt = _align(gen_eval[idx], gt_eval[idx])
        gen_f = _to_float01(gen)
        gt_f = _to_float01(gt)
        mse_values.append(_mse(gen, gt))
        psnr_values.append(float(peak_signal_noise_ratio(gt_f, gen_f, data_range=1.0)))
        ssim_values.append(float(structural_similarity(gt_f, gen_f, channel_axis=2, data_range=1.0)))
        if lpips_scorer is not None:
            lpips_values.append(lpips_scorer.score(gen, gt))

    metrics = {
        "future_pair_count": float(pair_count),
        "future_mse": float(np.mean(mse_values)),
        "future_psnr": float(np.mean(psnr_values)),
        "future_ssim": float(np.mean(ssim_values)),
    }
    if lpips_values:
        metrics["future_lpips"] = float(np.mean(lpips_values))
    return metrics


def run_continuation_metrics(
    config: BenchConfig,
    samples: list[BenchSample],
    output_dir: str,
    run_name: str = "continuation_metrics",
) -> str:
    output_root = Path(output_dir).expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    lpips_scorer = None
    if config.continuation.enable_lpips:
        lpips_scorer = LPIPSScorer(net=config.continuation.lpips_net)

    per_sample: list[dict[str, Any]] = []
    aggregate_buckets: dict[str, list[float]] = {}

    for sample in samples:
        generated_frames = _load_video_frames(sample.video_path, max_frames=config.continuation.max_video_frames)
        if not generated_frames:
            raise ValueError(f"No frames decoded from generated video: {sample.video_path}")

        result: dict[str, Any] = {
            "sample_id": sample.sample_id,
            "prompt": sample.prompt,
            "video_path": sample.video_path,
        }

        context_last = _load_last_context_frame(sample)
        gen_start_frame = (
            sample.generated_start_frame
            if sample.generated_start_frame is not None
            else config.continuation.generated_start_frame
        )
        gt_start_frame = sample.gt_start_frame if sample.gt_start_frame is not None else config.continuation.gt_start_frame

        if context_last is not None and gen_start_frame < len(generated_frames):
            boundary = _boundary_metrics(context_last, generated_frames[gen_start_frame], lpips_scorer)
            result.update(boundary)

        if sample.gt_video_path:
            gt_frames = _load_video_frames(sample.gt_video_path, max_frames=config.continuation.max_video_frames)
            future = _future_metrics(
                gen_frames=generated_frames,
                gt_frames=gt_frames,
                lpips_scorer=lpips_scorer,
                gen_start_frame=gen_start_frame,
                gt_start_frame=gt_start_frame,
            )
            result.update(future)

        per_sample.append(result)
        for key, value in result.items():
            if isinstance(value, (int, float)) and key not in {"sample_id", "video_path"}:
                aggregate_buckets.setdefault(key, []).append(float(value))

    aggregate = {key: float(np.mean(values)) for key, values in aggregate_buckets.items() if values}
    payload = {
        "suite": "continuation_metrics",
        "aggregate": aggregate,
        "per_sample": per_sample,
    }

    output_path = output_root / f"{run_name}.json"
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    return str(output_path)

