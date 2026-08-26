#!/usr/bin/env python3
"""Run the RigidBench tracker/scorer on one native CYCLES GT video.

The reference video is also used as the prediction video, but masks, tracks,
and depth are re-estimated by SAM2, CoTracker3, and Video Depth Anything.
This makes the result a learned-pipeline sanity check rather than the
oracle GT-vs-GT result.  CYCLES' native 896x512/30 FPS protocol is preserved.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import torch


FPS = 30
SAM2_CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
SAM2_CHECKPOINT = Path("/data/gaoya/ckpt/facebook-sam2.1-hiera-large/sam2.1_hiera_large.pt")
COTRACKER_CHECKPOINT = Path("/home/gaoya/.cache/torch/hub/checkpoints/cotracker3_scaled_offline.pth")
VDA_CHECKPOINT = Path("/data/gaoya/ckpt/Video-Depth-Anything-Large/video_depth_anything_vitl.pth")


def extract_native_frames(video: Path, destination: Path) -> int:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video}")
    count = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            out = destination / f"{count:05d}.jpg"
            if not cv2.imwrite(str(out), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
                raise RuntimeError(f"Could not write {out}")
            count += 1
    finally:
        capture.release()
    if count == 0:
        raise RuntimeError(f"No frames decoded from {video}")
    return count


def patch_local_trackers() -> None:
    """Use explicit local checkpoints instead of HF/torch.hub model lookup."""
    from sam2.build_sam import build_sam2_video_predictor
    from cotracker.predictor import CoTrackerPredictor
    from rigidbench.core.io import save_npz
    from rigidbench.core.paths import OutputPaths
    from rigidbench.core.constants import DEPTH_KEY
    from rigidbench.eval.track.cotracker3 import CoTracker3Tracker
    from rigidbench.eval.track.sam2 import SAM2Tracker
    from rigidbench.eval.track.vda import VDATracker
    from video_depth_anything.video_depth import VideoDepthAnything
    from PIL import Image

    if not SAM2_CHECKPOINT.is_file():
        raise FileNotFoundError(SAM2_CHECKPOINT)
    if not COTRACKER_CHECKPOINT.is_file():
        raise FileNotFoundError(COTRACKER_CHECKPOINT)
    if not VDA_CHECKPOINT.is_file():
        raise FileNotFoundError(VDA_CHECKPOINT)

    def sam_enter(self):
        print(f"[setup] loading local SAM2 checkpoint: {SAM2_CHECKPOINT}", flush=True)
        self._predictor = build_sam2_video_predictor(
            SAM2_CONFIG,
            ckpt_path=str(SAM2_CHECKPOINT),
            device=self.device,
        )
        return self

    def cotracker_enter(self):
        print(f"[setup] loading local CoTracker3 checkpoint: {COTRACKER_CHECKPOINT}", flush=True)
        self._model = CoTrackerPredictor(
            checkpoint=str(COTRACKER_CHECKPOINT),
            offline=True,
            v2=False,
            window_len=60,
        ).to(self.device).eval()
        return self

    def vda_enter(self):
        print(f"[setup] loading local VDA checkpoint: {VDA_CHECKPOINT}", flush=True)
        self._model = VideoDepthAnything(
            encoder="vitl",
            features=256,
            out_channels=[256, 512, 1024, 1024],
        )
        state = torch.load(str(VDA_CHECKPOINT), map_location="cpu")
        self._model.load_state_dict(state, strict=True)
        self._model = self._model.to(self.device).eval()
        return self

    def vda_track(self, sample, paths: OutputPaths) -> None:
        output_path = paths.depth(sample.id)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame_files = sorted(paths.generated_dir(sample.id).glob("*.jpg"))
        frames = np.stack([np.array(Image.open(f).convert("RGB")) for f in frame_files])
        with torch.no_grad():
            disparity, _ = self._model.infer_video_depth(
                frames,
                target_fps=FPS,
                input_size=384,
                device=self.device,
            )
        save_npz(output_path, **{DEPTH_KEY: disparity})

    SAM2Tracker.__enter__ = sam_enter
    CoTracker3Tracker.__enter__ = cotracker_enter
    VDATracker.__enter__ = vda_enter
    VDATracker.track = vda_track

    # The scoring stage uses a second hard-coded torch.hub lookup for the
    # BGDrift metric. Reuse the same local checkpoint there as well.
    original_hub_load = torch.hub.load

    def local_hub_load(repo_or_dir, model, *args, **kwargs):
        if repo_or_dir == "facebookresearch/co-tracker" and model == "cotracker3_offline":
            print(f"[setup] reusing local CoTracker3 for scoring: {COTRACKER_CHECKPOINT}", flush=True)
            return CoTrackerPredictor(
                checkpoint=str(COTRACKER_CHECKPOINT),
                offline=True,
                v2=False,
                window_len=60,
            ).to("cuda").eval()
        return original_hub_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = local_hub_load


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", default="cycles_gt_native_sam2_cotracker3_vda")
    args = parser.parse_args()

    for path in (SAM2_CHECKPOINT, COTRACKER_CHECKPOINT, VDA_CHECKPOINT):
        if not path.is_file():
            raise FileNotFoundError(path)

    adapter = args.case_dir / "rigidbench"
    video = adapter / "video.mp4"
    sample_id = json.loads((adapter / "metadata.json").read_text())["sample_id"]
    from rigidbench.core.paths import OutputPaths

    model_output = OutputPaths(args.output_dir / args.model_name)
    frame_count = extract_native_frames(video, model_output.generated_dir(sample_id))

    # ScoreContext imports GT_FPS as a module-level value.  Set it to the
    # native CYCLES rate so no artificial 30->24 temporal resampling occurs.
    import rigidbench.eval.score.context as score_context

    score_context.GT_FPS = FPS
    patch_local_trackers()

    from rigidbench.eval.run import run_eval

    print(
        json.dumps(
            {
                "protocol": "rigidbench-code-native-cycles",
                "sample_id": sample_id,
                "video": str(video),
                "fps": FPS,
                "frame_count": frame_count,
                "resolution": [896, 512],
                "sam2_checkpoint": str(SAM2_CHECKPOINT),
                "cotracker_checkpoint": str(COTRACKER_CHECKPOINT),
                "vda_checkpoint": str(VDA_CHECKPOINT),
            },
            indent=2,
        ),
        flush=True,
    )
    aggregate = run_eval(
        args.model_name,
        data_dir=str(adapter.parent.parent.parent / "rigidbench_dataset"),
        output_dir=str(args.output_dir),
        split="eval",
        sample_ids=[sample_id],
        force=True,
        generated_fps=FPS,
    )

    report_dir = args.output_dir / args.model_name
    (report_dir / "native_protocol.json").write_text(
        json.dumps(
            {
                "protocol": "rigidbench-code-native-cycles",
                "sample_id": sample_id,
                "video": str(video),
                "fps": FPS,
                "resolution": [896, 512],
                "trackers": ["SAM2.1 Hiera-Large", "CoTracker3 offline", "Video Depth Anything Large"],
                "checkpoints": {
                    "sam2": str(SAM2_CHECKPOINT),
                    "cotracker3": str(COTRACKER_CHECKPOINT),
                    "vda": str(VDA_CHECKPOINT),
                },
                "aggregated": aggregate,
            },
            indent=2,
        )
        + "\n"
    )
    print("FINAL_METRICS=" + json.dumps(aggregate, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
