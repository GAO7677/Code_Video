from __future__ import annotations

import argparse

from rigidbench.eval.score.frame import lpips_per_frame

from .common import as_frames, cli_print, load_video_rgb


def score_case(gt_frames, pred_frames, model, device: str = "cuda") -> dict:
    """Return LPIPS from uint8 RGB frames shaped (T,H,W,3), range [0,255]."""
    gt = as_frames(gt_frames, "gt_frames")
    pred = as_frames(pred_frames, "pred_frames")
    result = lpips_per_frame(gt, pred, model, device)
    return {"lpips": float(result.mean()), "per_frame": result}


if __name__ == "__main__":
    import lpips as lpips_pkg

    parser = argparse.ArgumentParser(description="Score one case's LPIPS")
    parser.add_argument("--gt-video", required=True)
    parser.add_argument("--pred-video", required=True)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    model = lpips_pkg.LPIPS(net="alex").to(args.device).eval()
    cli_print(score_case(load_video_rgb(args.gt_video), load_video_rgb(args.pred_video), model, args.device))
