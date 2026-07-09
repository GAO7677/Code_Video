from __future__ import annotations

"""
Full-conditioning variant of the Kubric batch V2V runner.

This wrapper reuses the existing Kubric batch runner implementation but swaps
the context-video loader so each sample feeds the full conditioning video into
the pipeline instead of truncating to the first `context_frames` frames.
"""

import numpy as np
from decord import VideoReader, cpu

from code_vjepa_vggt.train0705_kubric_no_gt_box import (
    wan_stage1b_context_only_no_gt_box_vnewtrain_kubric_v2v as base,
)


def _load_full_context_video_for_mode(
    *,
    video_path,
    target_context_frames: int,
    sampling_mode: str,
):
    del target_context_frames, sampling_mode
    vr = VideoReader(str(video_path), ctx=cpu(0))
    if len(vr) <= 0:
        raise RuntimeError(f"context video {video_path} does not provide any readable frames")
    frame_indices = np.arange(len(vr), dtype=np.int64)
    frames = vr.get_batch(frame_indices).asnumpy()
    return frames, frame_indices


def main() -> None:
    base._load_context_video_for_mode = _load_full_context_video_for_mode
    base.main()


if __name__ == "__main__":
    main()
