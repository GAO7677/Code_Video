from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.utils import require_torch
from phys_state_video.wan_bridge import WanTextImageToVideoBackend

torch = require_torch()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Wan TI2V bundle inference from an exported phys_state_video state-condition bundle."
    )
    parser.add_argument("--bundle-dir", required=True, help="Directory containing input_image.png, state_condition.npz, prompt.txt.")
    parser.add_argument("--wan-ckpt-dir", default="/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
    parser.add_argument("--wan-repo-root", default="/home/gaoya/Code_Video/Wan2.2-main")
    parser.add_argument("--task", default="ti2v-5B", choices=["ti2v-5B"])
    parser.add_argument("--size", default="704*1280")
    parser.add_argument("--frame-num", type=int, default=9)
    parser.add_argument("--sample-solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sampling-steps", type=int, default=1)
    parser.add_argument("--guide-scale", type=float, default=5.0)
    parser.add_argument("--shift", type=float, default=5.0)
    parser.add_argument("--state-scale", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--state-adapter-ckpt", default=None)
    parser.add_argument("--output", required=True, help="Output directory.")
    return parser.parse_args()


def load_state_condition(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        return {key: payload[key] for key in payload.files}


def write_mp4(path: Path, frames_cthw: np.ndarray, fps: int) -> None:
    import cv2

    frames = np.clip((frames_cthw.transpose(1, 2, 3, 0) + 1.0) * 0.5, 0.0, 1.0)
    frames = (frames * 255.0).round().astype(np.uint8)
    height, width = frames.shape[1:3]
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 8, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"failed to open video writer for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def main():
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    bundle_dir = Path(args.bundle_dir)
    image_path = bundle_dir / "input_image.png"
    state_condition_path = bundle_dir / "state_condition.npz"
    prompt_path = bundle_dir / "prompt.txt"
    meta_path = bundle_dir / "meta.json"

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Torch CUDA is unavailable in the active environment. "
            f"torch={torch.__version__}, cuda_device_count={torch.cuda.device_count()}. "
            "This bundle runner requires a working CUDA Wan runtime."
        )

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    state_condition = load_state_condition(state_condition_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    backend = WanTextImageToVideoBackend(
        ckpt_dir=args.wan_ckpt_dir,
        wan_repo_root=args.wan_repo_root,
        task=args.task,
        device="cuda:0",
        state_adapter_ckpt=args.state_adapter_ckpt,
    )
    video = backend.generate(
        prompt=prompt,
        first_frame=image_path,
        size=args.size,
        frame_num=args.frame_num,
        memory_tokens=state_condition.get("memory_tokens"),
        condition_maps=state_condition.get("condition_maps"),
        sample_solver=args.sample_solver,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        shift=args.shift,
        negative_prompt="",
        seed=args.seed,
        state_scale=args.state_scale,
    )

    if video is None:
        raise RuntimeError("WanTI2V returned None.")

    video_np = video.detach().cpu().numpy()
    np.savez_compressed(output_dir / "wan_state_condition_bundle_outputs.npz", video=video_np)
    write_mp4(output_dir / "wan_state_condition_bundle.mp4", video_np, fps=8)
    (output_dir / "meta.json").write_text(
        json.dumps(
            {
                "bundle_dir": str(bundle_dir),
                "wan_ckpt_dir": args.wan_ckpt_dir,
                "task": args.task,
                "size": args.size,
                "frame_num": args.frame_num,
                "sampling_steps": args.sampling_steps,
                "guide_scale": args.guide_scale,
                "shift": args.shift,
                "state_scale": args.state_scale,
                "state_adapter_ckpt": args.state_adapter_ckpt,
                "bundle_meta": meta,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"saved Wan state-condition bundle outputs to {output_dir}")


if __name__ == "__main__":
    main()
