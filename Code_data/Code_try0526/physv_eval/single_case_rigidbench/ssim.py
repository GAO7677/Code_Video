from __future__ import annotations

import argparse

from rigidbench.eval.score.frame import ssim_per_frame

from .common import as_frames, cli_print, load_video_rgb


def score_case(gt_frames, pred_frames, device: str = "cuda") -> dict:
    """Return SSIM from uint8 RGB frames shaped (T,H,W,3), range [0,255]."""
    gt = as_frames(gt_frames, "gt_frames")
    pred = as_frames(pred_frames, "pred_frames")
    result = ssim_per_frame(gt, pred, device)
    return {"ssim": float(result.mean()), "per_frame": result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score one case's frame SSIM")
    parser.add_argument("--gt-video", required=True)
    parser.add_argument("--pred-video", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cli_print(score_case(load_video_rgb(args.gt_video), load_video_rgb(args.pred_video), args.device))
