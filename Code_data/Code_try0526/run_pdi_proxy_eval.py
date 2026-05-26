#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from rerank_video.pdi_proxy_eval import default_cases, run_eval


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Wan/VACE on a few PDI-style cases with geometry-proxy scoring.")
    parser.add_argument("--run_name", default="pdi_proxy_eval")
    parser.add_argument("--output_root", type=Path, default=Path("/data/gaoya/AAA_test_video/Output_try0526"))
    parser.add_argument("--wan_root", type=Path, default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"))
    parser.add_argument("--vace_root", type=Path, default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--fps", type=int, default=16)
    parser.add_argument("--num_frames", type=int, default=49)
    parser.add_argument("--num_inference_steps", type=int, default=30)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--negative_prompt", default="")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_root = args.output_root / "runs" / args.run_name
    tmp_root = args.output_root / "tmp" / args.run_name
    summary = run_eval(
        output_root=run_root,
        tmp_root=tmp_root,
        cases=default_cases(),
        wan_root=args.wan_root.expanduser().resolve(),
        vace_root=args.vace_root.expanduser().resolve(),
        device=args.device,
        width=args.width,
        height=args.height,
        fps=args.fps,
        num_frames=args.num_frames,
        num_inference_steps=args.num_inference_steps,
        cfg_scale=args.cfg_scale,
        quality=args.quality,
        negative_prompt=args.negative_prompt,
        seed=args.seed,
    )
    print(summary)


if __name__ == "__main__":
    main()
