from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def load_npz_array(path: str | Path, *keys: str) -> np.ndarray:
    """Load one named array from an NPZ file; no task/case metadata is needed."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as data:
        for key in keys:
            if key in data.files:
                return data[key]
    raise KeyError(f"None of {keys!r} exists in {path}")


def load_video_rgb(path: str | Path) -> np.ndarray:
    """Decode a video/frame directory as uint8 RGB ``(T,H,W,3)`` frames."""
    path = Path(path)
    if path.is_dir():
        files = sorted(path.glob("*.jpg")) or sorted(path.glob("*.png"))
        if not files:
            raise ValueError(f"No jpg/png frames found under {path}")
        from PIL import Image

        return np.stack([np.asarray(Image.open(file).convert("RGB"), dtype=np.uint8) for file in files])
    capture = cv2.VideoCapture(str(path))
    frames: list[np.ndarray] = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    finally:
        capture.release()
    if not frames:
        raise ValueError(f"Could not decode any frames from {path}")
    return np.stack(frames, axis=0)


def as_masks(value: np.ndarray, name: str) -> np.ndarray:
    """Normalize a mask array to bool (T, N, H, W).

    Accepted input values are bool, integer 0/1, or integer 0/255. Float masks
    are accepted only when all values are in [0, 1] and are thresholded at 0.5.
    """
    arr = np.asarray(value)
    if arr.ndim == 2:
        arr = arr[None, None, :, :]
    elif arr.ndim == 3:
        arr = arr[:, None, :, :]
    if arr.ndim != 4:
        raise ValueError(f"{name} must have shape (T,N,H,W) or (T,H,W), got {arr.shape}")
    if arr.dtype == np.bool_:
        return arr
    if np.issubdtype(arr.dtype, np.integer):
        values = np.unique(arr)
        if not np.all(np.isin(values, [0, 1, 255])):
            raise ValueError(f"{name} integer values must be 0/1 or 0/255, got {values[:10]}")
        return arr > 0
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all() or arr.min() < 0 or arr.max() > 1:
            raise ValueError(f"{name} float values must be finite and normalized to [0,1]")
        return arr >= 0.5
    raise TypeError(f"Unsupported {name} dtype: {arr.dtype}")


def check_same_mask_shape(gt: np.ndarray, pred: np.ndarray) -> None:
    if gt.shape != pred.shape:
        raise ValueError(f"GT and prediction masks must have identical shape, got {gt.shape} vs {pred.shape}")


def as_tracks(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 3 or arr.shape[-1] != 2:
        raise ValueError(f"{name} must have shape (N,T,2), got {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain finite pixel coordinates")
    return arr.astype(np.float32, copy=False)


def as_visibility(value: np.ndarray | None, shape: tuple[int, int]) -> np.ndarray | None:
    if value is None:
        return None
    arr = np.asarray(value)
    if arr.shape != shape:
        raise ValueError(f"visibility must have shape {shape}, got {arr.shape}")
    return arr.astype(bool, copy=False)


def as_frames(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 4 or arr.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (T,H,W,3), got {arr.shape}")
    if arr.dtype == np.uint8:
        return arr
    if np.issubdtype(arr.dtype, np.floating):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")
        if 0 <= arr.min() and arr.max() <= 1:
            return np.rint(arr * 255).astype(np.uint8)
        if 0 <= arr.min() and arr.max() <= 255:
            return np.rint(arr).astype(np.uint8)
    raise ValueError(f"{name} must be uint8 [0,255] or float [0,1]/[0,255]")


def as_depth(value: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(value)
    if arr.ndim != 3:
        raise ValueError(f"{name} must have shape (T,H,W), got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number) or not np.isfinite(arr).all() or np.any(arr <= 0):
        raise ValueError(f"{name} must be finite positive numeric data")
    return arr.astype(np.float32, copy=False)


def check_same_length(*arrays: np.ndarray) -> int:
    lengths = {int(len(array)) for array in arrays}
    if len(lengths) != 1:
        raise ValueError(f"All temporal inputs must have the same T, got {sorted(lengths)}")
    return lengths.pop()


def cli_print(result: dict) -> None:
    import json

    print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=True))
