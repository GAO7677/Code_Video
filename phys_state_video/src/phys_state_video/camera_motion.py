from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(slots=True)
class CameraMotionResult:
    camera_features: np.ndarray
    pairwise_affines: np.ndarray
    global_affines: np.ndarray
    valid: np.ndarray


def _to_gray_uint8(frame_chw: np.ndarray) -> np.ndarray:
    image = np.transpose(np.clip(frame_chw, 0.0, 1.0), (1, 2, 0))
    image = (image * 255.0).round().astype(np.uint8)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _estimate_pairwise_affine(
    prev_gray: np.ndarray,
    curr_gray: np.ndarray,
    *,
    number_of_iterations: int = 50,
    termination_eps: float = 1e-5,
) -> tuple[np.ndarray, bool]:
    warp = np.eye(2, 3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        int(number_of_iterations),
        float(termination_eps),
    )
    try:
        _, warp = cv2.findTransformECC(
            prev_gray,
            curr_gray,
            warp,
            motionType=cv2.MOTION_AFFINE,
            criteria=criteria,
            inputMask=None,
            gaussFiltSize=5,
        )
        return warp.astype(np.float32), True
    except cv2.error:
        shift, _ = cv2.phaseCorrelate(
            np.asarray(prev_gray, dtype=np.float32),
            np.asarray(curr_gray, dtype=np.float32),
        )
        fallback = np.asarray(
            [[1.0, 0.0, float(shift[0])], [0.0, 1.0, float(shift[1])]],
            dtype=np.float32,
        )
        return fallback, False


def _affine_2x3_to_3x3(affine: np.ndarray) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float32)
    matrix[:2] = affine.astype(np.float32)
    return matrix


def _extract_camera_features(
    global_affines: np.ndarray,
    pairwise_affines: np.ndarray,
    valid: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    num_frames = int(global_affines.shape[0])
    camera = np.zeros((num_frames, 10), dtype=np.float32)
    prev_tx = 0.0
    prev_ty = 0.0
    prev_scale = 1.0
    prev_rot = 0.0
    for idx in range(num_frames):
        matrix = global_affines[idx]
        a, b, tx = float(matrix[0, 0]), float(matrix[0, 1]), float(matrix[0, 2])
        c, d, ty = float(matrix[1, 0]), float(matrix[1, 1]), float(matrix[1, 2])
        scale_x = max(np.sqrt(max(a * a + c * c, 1e-8)), 1e-4)
        scale_y = max(np.sqrt(max(b * b + d * d, 1e-8)), 1e-4)
        scale = float(np.sqrt(scale_x * scale_y))
        rotation = float(np.arctan2(c, a))
        shear = float(np.arctan2(-b, d) - rotation)
        tx_norm = tx / max(width, 1)
        ty_norm = ty / max(height, 1)
        if idx == 0:
            dtx_norm = 0.0
            dty_norm = 0.0
            dscale = 0.0
            drot = 0.0
        else:
            dtx_norm = tx_norm - prev_tx
            dty_norm = ty_norm - prev_ty
            dscale = float(np.log(max(scale, 1e-6)) - np.log(max(prev_scale, 1e-6)))
            drot = rotation - prev_rot
        prev_tx = tx_norm
        prev_ty = ty_norm
        prev_scale = scale
        prev_rot = rotation
        camera[idx] = np.asarray(
            [
                tx_norm,
                ty_norm,
                float(np.log(max(scale_x, 1e-6))),
                float(np.log(max(scale_y, 1e-6))),
                rotation / np.pi,
                shear / np.pi,
                dtx_norm,
                dty_norm,
                dscale,
                drot / np.pi,
            ],
            dtype=np.float32,
        )
        if not bool(valid[idx]):
            camera[idx, -1] = 0.0
    return camera


def estimate_global_camera_motion(frames_tchw: np.ndarray) -> CameraMotionResult:
    if frames_tchw.ndim != 4:
        raise ValueError(f"expected frames with shape [T, 3, H, W], got {tuple(frames_tchw.shape)}")
    num_frames, _, height, width = frames_tchw.shape
    if num_frames <= 0:
        raise ValueError("frames_tchw must be non-empty")
    pairwise_affines = np.repeat(np.eye(2, 3, dtype=np.float32)[None], repeats=num_frames, axis=0)
    global_affines = np.repeat(np.eye(3, dtype=np.float32)[None], repeats=num_frames, axis=0)
    valid = np.ones((num_frames,), dtype=np.bool_)
    if num_frames == 1:
        camera = np.zeros((1, 10), dtype=np.float32)
        return CameraMotionResult(
            camera_features=camera,
            pairwise_affines=pairwise_affines,
            global_affines=global_affines,
            valid=valid,
        )

    grays = [_to_gray_uint8(frame) for frame in frames_tchw]
    for idx in range(1, num_frames):
        pairwise, ok = _estimate_pairwise_affine(grays[idx - 1], grays[idx])
        pairwise_affines[idx] = pairwise
        valid[idx] = bool(ok)
        global_affines[idx] = _affine_2x3_to_3x3(pairwise) @ global_affines[idx - 1]
    camera = _extract_camera_features(
        global_affines,
        pairwise_affines,
        valid,
        width=int(width),
        height=int(height),
    )
    return CameraMotionResult(
        camera_features=camera.astype(np.float32),
        pairwise_affines=pairwise_affines.astype(np.float32),
        global_affines=global_affines.astype(np.float32),
        valid=valid,
    )
