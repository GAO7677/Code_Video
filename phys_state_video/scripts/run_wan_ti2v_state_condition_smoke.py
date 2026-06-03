from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from phys_state_video.utils import require_torch

torch = require_torch()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run a TI2V Wan smoke test from an exported phys_state_video state-condition bundle."
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
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
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
            "This machine currently reports a driver/runtime mismatch for the `wan` environment, "
            "so the TI2V smoke test cannot proceed to model sampling."
        )

    wan_repo_root = Path(args.wan_repo_root)
    if str(wan_repo_root) not in sys.path:
        sys.path.insert(0, str(wan_repo_root))

    from wan_.configs import MAX_AREA_CONFIGS, SIZE_CONFIGS, SUPPORTED_SIZES, WAN_CONFIGS
    from wan_.textimage2video import WanTI2V

    if args.task not in WAN_CONFIGS:
        raise ValueError(f"unsupported task: {args.task}")
    if args.size not in SUPPORTED_SIZES[args.task]:
        raise ValueError(f"unsupported size {args.size} for task {args.task}")

    prompt = prompt_path.read_text(encoding="utf-8").strip()
    image = Image.open(image_path).convert("RGB")
    state_condition = load_state_condition(state_condition_path)
    meta = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.exists() else {}

    pipeline = WanTI2V(
        config=WAN_CONFIGS[args.task],
        checkpoint_dir=str(args.wan_ckpt_dir),
        device_id=0,
        rank=0,
    )
    if args.state_adapter_ckpt is not None:
        pipeline.load_state_adapter(args.state_adapter_ckpt, state_condition=state_condition)

    video = pipeline.generate(
        prompt,
        img=image,
        size=SIZE_CONFIGS[args.size],
        max_area=MAX_AREA_CONFIGS[args.size],
        frame_num=args.frame_num,
        shift=args.shift,
        sample_solver=args.sample_solver,
        sampling_steps=args.sampling_steps,
        guide_scale=args.guide_scale,
        seed=args.seed,
        offload_model=True,
        state_condition=state_condition,
        state_scale=args.state_scale,
    )

    if video is None:
        raise RuntimeError("WanTI2V returned None.")

    video_np = video.detach().cpu().numpy()
    np.savez_compressed(output_dir / "wan_ti2v_smoke_outputs.npz", video=video_np)
    write_mp4(output_dir / "wan_ti2v_smoke.mp4", video_np, fps=8)
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
    print(f"saved Wan TI2V smoke outputs to {output_dir}")


if __name__ == "__main__":
    main()
