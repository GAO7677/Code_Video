from __future__ import annotations

import argparse
import numpy as np

from rigidbench.eval.score.depth import affine_align_disparity
from rigidbench.eval.score.trajectory import compute_ate3d
from rigidbench.eval.score.trajectory import quat_wxyz_to_rotmat, reconstruct_centroids

from .common import as_depth, as_tracks, as_visibility, cli_print, load_video_rgb
from .prediction import extract_disparity, extract_tracks


def score_case(
    pred_video,
    gt_tracks,
    gt_visibility,
    gt_depth,
    gt_trajectories: dict,
    actors: list[str],
    camera: dict,
    actor_offsets,
    vda_model,
    cotracker_model,
    device: str = "cuda",
    frames: np.ndarray | None = None,
) -> dict:
    """Return ATE-3D from GT inputs and a generated video.

    VDA predicts disparity and CoTracker predicts 2D tracks internally.  The
    camera and GT trajectory data are used only for the 3D reconstruction and
    GT comparison.
    """
    gt_tr = as_tracks(gt_tracks, "gt_tracks")
    gt_vis = as_visibility(gt_visibility, gt_tr.shape[:2])
    if gt_vis is None:
        raise ValueError("gt_visibility is required for official RigidBench ATE-3D semantics")
    gt_d = as_depth(gt_depth, "gt_depth")
    pred_tr, pred_vis = extract_tracks(pred_video, gt_tr, cotracker_model, frames=frames)
    pred_disp = extract_disparity(pred_video, vda_model, device, frames=frames)
    return score_from_predictions(
        gt_tracks,
        gt_visibility,
        gt_depth,
        gt_trajectories,
        actors,
        camera,
        actor_offsets,
        pred_tr,
        pred_vis,
        pred_disp,
    )


def score_from_predictions(
    gt_tracks,
    gt_visibility,
    gt_depth,
    gt_trajectories: dict,
    actors: list[str],
    camera: dict,
    actor_offsets,
    pred_tracks,
    pred_visibility,
    pred_disparity,
) -> dict:
    """Compute ATE-3D from already extracted prediction tracks/disparity."""
    gt_tr = as_tracks(gt_tracks, "gt_tracks")
    gt_vis = as_visibility(gt_visibility, gt_tr.shape[:2])
    if gt_vis is None:
        raise ValueError("gt_visibility is required for official RigidBench ATE-3D semantics")
    gt_d = as_depth(gt_depth, "gt_depth")
    pred_tr = as_tracks(pred_tracks, "pred_tracks")
    pred_vis = as_visibility(pred_visibility, pred_tr.shape[:2])
    if pred_vis is None:
        raise ValueError("pred_visibility is required for ATE-3D")
    pred_disp = as_depth(pred_disparity, "pred_disparity")
    T = min(gt_tr.shape[1], gt_d.shape[0], pred_tr.shape[1], pred_disp.shape[0])
    gt_tr = gt_tr[:, :T]
    gt_d = gt_d[:T]
    pred_tr = pred_tr[:, :T]
    pred_vis = pred_vis[:, :T]
    gt_vis = gt_vis[:, :T]
    visibility = pred_vis & gt_vis
    aligned, _, _ = affine_align_disparity(pred_disp[:T], gt_d)
    intrinsics = camera["intrinsics"]
    extrinsics = camera["extrinsics"]
    pred_centroids = reconstruct_centroids(
        pred_tr,
        visibility,
        aligned,
        intrinsics,
        np.asarray(extrinsics["location"], dtype=np.float64),
        quat_wxyz_to_rotmat(np.asarray(extrinsics["rotation"], dtype=np.float64)),
        np.asarray(actor_offsets, dtype=np.int64),
    )
    if len(pred_centroids) != len(actors):
        raise ValueError("actor_offsets/gt_tracks and actors disagree")
    return compute_ate3d(pred_centroids, gt_trajectories, actors)


if __name__ == "__main__":
    raise SystemExit("Use score_case(...) from Python; this metric requires GT camera/trajectory inputs and a generated video.")
