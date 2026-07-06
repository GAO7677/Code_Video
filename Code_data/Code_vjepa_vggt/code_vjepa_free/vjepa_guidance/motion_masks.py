from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MotionMaskResult:
    name: str
    heat: np.ndarray  # [T,H,W], float32 in [0, 1]
    mask: np.ndarray  # [T,H,W], float32 in {0,1}
    coverage: float
    threshold: float


def extract_motion_mask_thw(
    video_thwc_u8: np.ndarray,
    *,
    method: str = "background_residual",
    diff_quantile: float = 0.97,
    flow_quantile: float = 0.96,
    dilate_px: int = 10,
    blur_ksize: int = 5,
) -> np.ndarray:
    """
    Standard project API:
      input:  video_thwc_u8  [T,H,W,3]
      output: motion_mask_thw [T,H,W] float32 in {0,1}
    """
    results = compute_all_motion_masks(
        video_thwc_u8,
        diff_quantile=diff_quantile,
        flow_quantile=flow_quantile,
        dilate_px=dilate_px,
        blur_ksize=blur_ksize,
    )
    if method not in results:
        raise KeyError(f"unknown motion mask method: {method}")
    return results[method].mask.astype(np.float32)


def _normalize_heat(heat: np.ndarray) -> np.ndarray:
    heat = np.nan_to_num(heat.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    max_value = float(heat.max(initial=0.0))
    if max_value <= 1.0e-6:
        return np.zeros_like(heat, dtype=np.float32)
    return np.clip(heat / max_value, 0.0, 1.0).astype(np.float32)


def _quantile_threshold(heat: np.ndarray, quantile: float) -> float:
    flat = heat.reshape(-1)
    flat = flat[flat > 1.0e-8]
    if flat.size == 0:
        return 1.0
    quantile = float(np.clip(quantile, 0.0, 1.0))
    return float(np.quantile(flat, quantile))


def _dilate_mask(mask: np.ndarray, dilate_px: int) -> np.ndarray:
    if dilate_px <= 0:
        return mask.astype(np.float32)
    kernel_size = int(max(1, dilate_px) * 2 + 1)
    kernel = np.ones((kernel_size, kernel_size), dtype=np.uint8)
    out = np.zeros_like(mask, dtype=np.float32)
    for index in range(mask.shape[0]):
        out[index] = cv2.dilate(mask[index].astype(np.uint8), kernel, iterations=1).astype(np.float32)
    return np.clip(out, 0.0, 1.0).astype(np.float32)


def _filter_connected_components(
    mask: np.ndarray,
    *,
    keep_top_k: int = 4,
    min_area_ratio: float = 0.0008,
) -> np.ndarray:
    out = np.zeros_like(mask, dtype=np.float32)
    frame_area = float(mask.shape[1] * mask.shape[2])
    min_area = max(1.0, frame_area * float(min_area_ratio))
    for index in range(mask.shape[0]):
        binary = mask[index].astype(np.uint8)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
        if num_labels <= 1:
            continue
        components: list[tuple[int, int]] = []
        for label in range(1, num_labels):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if float(area) >= min_area:
                components.append((area, label))
        components.sort(reverse=True)
        keep = {label for _, label in components[: int(max(1, keep_top_k))]}
        if not keep:
            continue
        filtered = np.isin(labels, list(keep)).astype(np.float32)
        out[index] = filtered
    return out


def _blur_gray_frames(video_thwc_u8: np.ndarray, blur_ksize: int) -> np.ndarray:
    gray = np.stack([cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY) for frame in video_thwc_u8], axis=0).astype(np.float32)
    if blur_ksize > 1:
        ksize = int(blur_ksize)
        if ksize % 2 == 0:
            ksize += 1
        gray = np.stack(
            [cv2.GaussianBlur(frame, (ksize, ksize), 0.0) for frame in gray],
            axis=0,
        )
    return gray


def compute_frame_diff_mask(
    video_thwc_u8: np.ndarray,
    *,
    quantile: float = 0.97,
    dilate_px: int = 8,
    blur_ksize: int = 5,
) -> MotionMaskResult:
    gray = _blur_gray_frames(video_thwc_u8, blur_ksize=blur_ksize)
    frames = gray.shape[0]
    heat = np.zeros_like(gray, dtype=np.float32)
    if frames > 1:
        diff = np.abs(gray[1:] - gray[:-1])
        heat[1:] += diff
        heat[:-1] += diff
        heat /= 2.0
    heat = _normalize_heat(heat)
    threshold = _quantile_threshold(heat, quantile=quantile)
    mask = (heat >= threshold).astype(np.float32)
    mask = _filter_connected_components(mask, keep_top_k=4, min_area_ratio=0.0008)
    mask = _dilate_mask(mask, dilate_px=dilate_px)
    return MotionMaskResult(
        name="frame_diff",
        heat=heat,
        mask=mask,
        coverage=float(mask.mean()),
        threshold=float(threshold),
    )


def compute_flow_mask(
    video_thwc_u8: np.ndarray,
    *,
    quantile: float = 0.96,
    dilate_px: int = 8,
    blur_ksize: int = 5,
) -> MotionMaskResult:
    gray = _blur_gray_frames(video_thwc_u8, blur_ksize=blur_ksize)
    frames = gray.shape[0]
    heat = np.zeros_like(gray, dtype=np.float32)
    if frames > 1:
        for index in range(frames - 1):
            flow = cv2.calcOpticalFlowFarneback(
                gray[index].astype(np.uint8),
                gray[index + 1].astype(np.uint8),
                None,
                pyr_scale=0.5,
                levels=3,
                winsize=21,
                iterations=3,
                poly_n=5,
                poly_sigma=1.2,
                flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2).astype(np.float32)
            heat[index] += mag
            heat[index + 1] += mag
        heat /= 2.0
    heat = _normalize_heat(heat)
    threshold = _quantile_threshold(heat, quantile=quantile)
    mask = (heat >= threshold).astype(np.float32)
    mask = _filter_connected_components(mask, keep_top_k=4, min_area_ratio=0.0008)
    mask = _dilate_mask(mask, dilate_px=dilate_px)
    return MotionMaskResult(
        name="optical_flow",
        heat=heat,
        mask=mask,
        coverage=float(mask.mean()),
        threshold=float(threshold),
    )


def compute_background_residual_mask(
    video_thwc_u8: np.ndarray,
    *,
    quantile: float = 0.96,
    dilate_px: int = 10,
    blur_ksize: int = 5,
) -> MotionMaskResult:
    gray = _blur_gray_frames(video_thwc_u8, blur_ksize=blur_ksize)
    background = np.median(gray, axis=0, keepdims=True)
    heat = np.abs(gray - background)
    heat = _normalize_heat(heat)
    threshold = _quantile_threshold(heat, quantile=quantile)
    mask = (heat >= threshold).astype(np.float32)
    mask = _filter_connected_components(mask, keep_top_k=4, min_area_ratio=0.0008)
    mask = _dilate_mask(mask, dilate_px=dilate_px)
    return MotionMaskResult(
        name="background_residual",
        heat=heat,
        mask=mask,
        coverage=float(mask.mean()),
        threshold=float(threshold),
    )


def compute_hybrid_mask(
    video_thwc_u8: np.ndarray,
    *,
    diff_quantile: float = 0.97,
    flow_quantile: float = 0.96,
    bg_quantile: float = 0.96,
    dilate_px: int = 12,
    blur_ksize: int = 5,
) -> MotionMaskResult:
    diff_result = compute_frame_diff_mask(
        video_thwc_u8,
        quantile=diff_quantile,
        dilate_px=max(1, dilate_px - 4),
        blur_ksize=blur_ksize,
    )
    flow_result = compute_flow_mask(
        video_thwc_u8,
        quantile=flow_quantile,
        dilate_px=max(1, dilate_px - 4),
        blur_ksize=blur_ksize,
    )
    bg_result = compute_background_residual_mask(
        video_thwc_u8,
        quantile=bg_quantile,
        dilate_px=max(1, dilate_px - 2),
        blur_ksize=blur_ksize,
    )
    heat = np.maximum(np.maximum(diff_result.heat, flow_result.heat), bg_result.heat)
    motion_seed = np.clip(diff_result.mask + bg_result.mask, 0.0, 1.0)
    flow_gate = _dilate_mask(flow_result.mask, dilate_px=max(1, dilate_px - 5))
    mask = np.clip(motion_seed * np.maximum(flow_gate, 0.5), 0.0, 1.0)
    mask = _filter_connected_components(mask, keep_top_k=4, min_area_ratio=0.0008)
    mask = _dilate_mask(mask, dilate_px=dilate_px)
    return MotionMaskResult(
        name="hybrid",
        heat=heat.astype(np.float32),
        mask=mask.astype(np.float32),
        coverage=float(mask.mean()),
        threshold=max(float(diff_result.threshold), float(flow_result.threshold), float(bg_result.threshold)),
    )


def compute_all_motion_masks(
    video_thwc_u8: np.ndarray,
    *,
    diff_quantile: float = 0.97,
    flow_quantile: float = 0.96,
    dilate_px: int = 10,
    blur_ksize: int = 5,
) -> dict[str, MotionMaskResult]:
    return {
        "frame_diff": compute_frame_diff_mask(
            video_thwc_u8,
            quantile=diff_quantile,
            dilate_px=max(1, dilate_px - 2),
            blur_ksize=blur_ksize,
        ),
        "background_residual": compute_background_residual_mask(
            video_thwc_u8,
            quantile=flow_quantile,
            dilate_px=dilate_px,
            blur_ksize=blur_ksize,
        ),
        "hybrid": compute_hybrid_mask(
            video_thwc_u8,
            diff_quantile=diff_quantile,
            flow_quantile=flow_quantile,
            bg_quantile=flow_quantile,
            dilate_px=dilate_px + 2,
            blur_ksize=blur_ksize,
        ),
        "optical_flow": compute_flow_mask(
            video_thwc_u8,
            quantile=flow_quantile,
            dilate_px=max(1, dilate_px - 1),
            blur_ksize=blur_ksize,
        ),
    }
