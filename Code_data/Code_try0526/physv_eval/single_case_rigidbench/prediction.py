"""Prediction-side extraction used by the single-case RigidBench metrics.

The public metric APIs receive GT supervision plus a generated video.  This
module owns the prediction-side SAM2, CoTracker3 and Video-Depth-Anything
calls.  Models are passed in by the metric worker and are therefore loaded
once per metric process, not once per case.
"""

from __future__ import annotations

import contextlib
import tempfile
from pathlib import Path
from typing import Iterator

import numpy as np

from .common import as_frames, as_masks, as_tracks, as_visibility, load_video_rgb


def load_sam2_model(device: str):
    local_checkpoint = Path("/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt")
    local_config = Path("/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_l.yaml")
    if local_checkpoint.is_file() and local_config.is_file():
        from sam2.build_sam import build_sam2_video_predictor

        config_name = f"configs/sam2.1/{local_config.name}"
        return build_sam2_video_predictor(
            config_name,
            str(local_checkpoint),
            device=device,
            mode="eval",
        )

    from sam2.sam2_video_predictor import SAM2VideoPredictor
    from rigidbench.core.constants import SEGMENTATION_MODEL

    return SAM2VideoPredictor.from_pretrained(SEGMENTATION_MODEL).to(device)


def load_cotracker_model(device: str):
    import sys

    import torch

    cotracker_root = Path("/home/gaoya/Code_Video/co-tracker-main")
    if str(cotracker_root) not in sys.path:
        sys.path.insert(0, str(cotracker_root))
    from cotracker.predictor import CoTrackerPredictor

    candidates = (
        Path("/data/gaoya/ckpt/facebook-cotracker3/scaled_offline.pth"),
        Path("/home/gaoya/.cache/torch/hub/checkpoints/cotracker3_scaled_offline.pth"),
    )
    checkpoint = next((path for path in candidates if path.is_file()), None)
    if checkpoint is None:
        raise FileNotFoundError("No local CoTracker checkpoint found")
    return CoTrackerPredictor(
        checkpoint=str(checkpoint), offline=True, v2=False, window_len=60,
    ).to(device).eval().requires_grad_(False)


def load_vda_model(device: str):
    from video_depth_anything.video_depth import VideoDepthAnything

    local_checkpoint = Path("/data/gaoya/ckpt/Video-Depth-Anything-Large/video_depth_anything_vitl.pth")
    if local_checkpoint.is_file():
        checkpoint = local_checkpoint
    else:
        from huggingface_hub import hf_hub_download
        from rigidbench.core.constants import DEPTH_MODEL

        checkpoint = Path(hf_hub_download(repo_id=DEPTH_MODEL, filename="video_depth_anything_vitl.pth"))
    model = VideoDepthAnything(encoder="vitl", features=256, out_channels=[256, 512, 1024, 1024])
    import torch

    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=False))
    return model.to(device).eval()


def load_dinov2_model(device: str):
    import torch

    return torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14").to(device).eval()


def _frame_files(path: Path) -> list[Path]:
    files = sorted(path.glob("*.jpg")) or sorted(path.glob("*.png"))
    if not files:
        raise FileNotFoundError(f"No video frames found under {path}")
    return files


@contextlib.contextmanager
def frame_directory(pred_video: str | Path, frames: np.ndarray | None = None) -> Iterator[Path]:
    """Yield a directory suitable for SAM2's video predictor.

    Existing frame directories are used directly.  A real video is decoded
    once and materialized into a temporary directory only for SAM2, whose
    official API accepts a frame directory rather than an in-memory tensor.
    """
    path = Path(pred_video)
    if path.is_dir():
        _frame_files(path)
        yield path
        return

    if frames is None:
        frames = load_video_rgb(path)
    frames = as_frames(frames, "pred_frames")
    with tempfile.TemporaryDirectory(prefix="rigidbench_pred_frames_") as name:
        directory = Path(name)
        import cv2

        for index, frame in enumerate(frames):
            output = directory / f"{index:05d}.jpg"
            cv2.imwrite(str(output), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        yield directory


def _active_mask(gt_mask: np.ndarray, active_actor_indices: list[int] | tuple[int, ...] | None) -> np.ndarray:
    masks = as_masks(gt_mask, "gt_mask")
    if active_actor_indices is None:
        return masks
    indices = np.asarray(active_actor_indices, dtype=np.int64)
    if indices.ndim != 1 or np.any(indices < 0) or np.any(indices >= masks.shape[1]):
        raise ValueError(f"active_actor_indices are outside GT mask actor dimension {masks.shape[1]}")
    return masks[:, indices]


def extract_masks(
    pred_video: str | Path,
    gt_mask: np.ndarray,
    sam2_model,
    active_actor_indices: list[int] | tuple[int, ...] | None = None,
) -> np.ndarray:
    """Propagate GT first-frame active masks through the generated video."""
    first_frame_masks = _active_mask(gt_mask, active_actor_indices)[0]
    frames = load_video_rgb(pred_video)
    with frame_directory(pred_video, frames) as frames_dir:
        state = sam2_model.init_state(video_path=str(frames_dir))
        sam2_model.reset_state(state)
        for object_id, mask in enumerate(first_frame_masks):
            sam2_model.add_new_mask(
                inference_state=state,
                frame_idx=0,
                obj_id=object_id,
                mask=mask,
            )
        frame_masks: dict[int, dict[int, np.ndarray]] = {}
        for frame_idx, object_ids, mask_logits in sam2_model.propagate_in_video(state):
            frame_masks[frame_idx] = {
                object_id: (mask_logits[i].squeeze() > 0.0).cpu().numpy()
                for i, object_id in enumerate(object_ids)
            }

    height, width = first_frame_masks.shape[-2:]
    output = np.zeros((len(frames), first_frame_masks.shape[0], height, width), dtype=bool)
    for frame_idx, object_masks in frame_masks.items():
        if frame_idx >= len(output):
            continue
        for object_id, mask in object_masks.items():
            output[frame_idx, object_id] = mask
    return output


def extract_tracks(
    pred_video: str | Path,
    gt_tracks: np.ndarray,
    cotracker_model,
) -> tuple[np.ndarray, np.ndarray]:
    """Track GT first-frame query points through the generated video."""
    import torch

    frames = as_frames(load_video_rgb(pred_video), "pred_frames")
    gt = as_tracks(gt_tracks, "gt_tracks")
    queries = np.zeros((1, gt.shape[0], 3), dtype=np.float32)
    queries[0, :, 1:] = gt[:, 0]
    video = torch.from_numpy(frames).permute(0, 3, 1, 2).unsqueeze(0).float().to(next(cotracker_model.parameters()).device)
    query_tensor = torch.from_numpy(queries).to(video.device)
    with torch.no_grad():
        pred_tracks, pred_visibility = cotracker_model(video, queries=query_tensor)
    tracks = pred_tracks[0].detach().cpu().numpy().transpose(1, 0, 2)
    visibility = pred_visibility[0].detach().cpu().numpy().astype(bool).T
    return tracks, visibility


def extract_disparity(pred_video: str | Path, vda_model, device: str) -> np.ndarray:
    """Run the official Video-Depth-Anything configuration on the video."""
    import torch
    from rigidbench.core.constants import DEPTH_INPUT_SIZE

    frames = as_frames(load_video_rgb(pred_video), "pred_frames")
    with torch.no_grad():
        disparity, _ = vda_model.infer_video_depth(
            frames,
            target_fps=24,
            input_size=DEPTH_INPUT_SIZE,
            device=device,
        )
    return np.asarray(disparity)


def active_actor_indices(gt_mask_path: str | Path, metadata: dict) -> list[int]:
    """Return active-object indices in the strict GT mask array."""
    with np.load(gt_mask_path, allow_pickle=False) as data:
        names = [str(value) for value in data["object_names"]]
    actors = metadata.get("actors", {})
    active = [index for index, name in enumerate(names) if actors.get(name, {}).get("role") == "active"]
    return active or list(range(len(names)))


def concatenate_gt_tracks(case_dir: str | Path, metadata: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """Build GT tracks from strict CYCLES truth using the official RigidBench projector."""
    from rigidbench.eval.track.gt import compute_gt_trajectories

    case_dir = Path(case_dir)
    with np.load(case_dir / "masks.npz", allow_pickle=False) as data:
        names = [str(value) for value in data["object_names"]]
    actors = [name for name, info in metadata.get("actors", {}).items() if info.get("role") == "active"]
    if not actors:
        actors = names
    bundles = [compute_gt_trajectories(case_dir, actor) for actor in actors]
    offsets = np.cumsum([0] + [len(bundle["query_points"]) for bundle in bundles]).astype(np.int64)
    tracks = np.concatenate([bundle["trajectories"] for bundle in bundles], axis=0)
    visibility = np.concatenate([bundle["visibility"] for bundle in bundles], axis=0)
    return tracks, visibility, offsets, actors
