from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/0626vjepa_free/test")
DEFAULT_IMAGE_PATH = Path(
    "/data/gaoya/AAA_test_video/0623/testdataset/"
    "025_Solid_Mechanics_0002_perspective-center_trimmed/"
    "physicIQ_0002_clip_2p5s_3p5s_firstframe.png"
)
DEFAULT_WAN_CKPT_DIR = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")

DEFAULT_PROMPTS = [
    "A red ball rolls down a wooden ramp and collides with a blue cube.",
    "A glass falls from a table and shatters on the floor.",
    "A basketball bounces on the ground several times.",
    "A toy car drives behind a box and reappears on the other side.",
    "A stack of blocks is pushed and topples over.",
    "Water is poured from a cup into a bowl.",
    "A pendulum swings back and forth.",
    "A ball rolls off a table and falls to the ground.",
]


@dataclass
class BaselineCase:
    case_id: str
    prompt: str
    seed: int
    image_path: str
    run_dir: str
    video_path: str
    prompt_path: str
    seed_path: str
    image_path_txt: str


def build_cases(
    run_name: str,
    seeds: list[int],
    prompts: list[str],
    image_paths: list[Path],
    output_root: Path,
) -> list[BaselineCase]:
    run_dir = output_root / run_name
    cases: list[BaselineCase] = []
    if not image_paths:
        raise ValueError("At least one image path is required for TI2V baseline generation.")

    for prompt_idx, prompt in enumerate(prompts):
        image_path = image_paths[prompt_idx % len(image_paths)].expanduser().resolve()
        for seed in seeds:
            case_id = f"p{prompt_idx:02d}_s{seed}"
            case_dir = run_dir / case_id
            cases.append(
                BaselineCase(
                    case_id=case_id,
                    prompt=prompt,
                    seed=seed,
                    image_path=str(image_path),
                    run_dir=str(case_dir),
                    video_path=str(case_dir / "video.mp4"),
                    prompt_path=str(case_dir / "prompt.txt"),
                    seed_path=str(case_dir / "seed.txt"),
                    image_path_txt=str(case_dir / "image_path.txt"),
                )
            )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a Wan TI2V baseline experiment manifest.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3])
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--image-path", dest="image_paths", action="append", type=Path, default=None)
    parser.add_argument("--image-list", type=Path, default=None)
    parser.add_argument("--ckpt-dir", type=Path, default=DEFAULT_WAN_CKPT_DIR)
    parser.add_argument("--size", default="1280*704")
    parser.add_argument("--frame-num", type=int, default=25)
    parser.add_argument("--sample-steps", type=int, default=40)
    parser.add_argument("--sample-guide-scale", type=float, default=5.0)
    parser.add_argument("--sample-shift", type=float, default=5.0)
    parser.add_argument("--offload-model", action="store_true", default=True)
    parser.add_argument("--t5-cpu", action="store_true", default=True)
    parser.add_argument("--convert-model-dtype", action="store_true", default=True)
    return parser.parse_args()


def resolve_image_paths(args: argparse.Namespace) -> list[Path]:
    image_paths = list(args.image_paths or [])
    if args.image_list is not None:
        lines = args.image_list.expanduser().resolve().read_text().splitlines()
        image_paths.extend(Path(line.strip()) for line in lines if line.strip())
    if not image_paths:
        image_paths = [DEFAULT_IMAGE_PATH]
    resolved = [path.expanduser().resolve() for path in image_paths]
    for path in resolved:
        if not path.exists():
            raise FileNotFoundError(f"Image path does not exist: {path}")
    return resolved


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    run_root = output_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    image_paths = resolve_image_paths(args)
    cases = build_cases(
        run_name=args.run_name,
        seeds=args.seeds,
        prompts=DEFAULT_PROMPTS,
        image_paths=image_paths,
        output_root=output_root,
    )
    for case in cases:
        case_dir = Path(case.run_dir)
        case_dir.mkdir(parents=True, exist_ok=True)
        Path(case.prompt_path).write_text(case.prompt + "\n")
        Path(case.seed_path).write_text(str(case.seed) + "\n")
        Path(case.image_path_txt).write_text(case.image_path + "\n")

    payload = {
        "run_name": args.run_name,
        "task": "ti2v-5B",
        "output_root": str(output_root),
        "ckpt_dir": str(args.ckpt_dir.expanduser().resolve()),
        "wan_args": {
            "size": args.size,
            "frame_num": args.frame_num,
            "sample_steps": args.sample_steps,
            "sample_guide_scale": args.sample_guide_scale,
            "sample_shift": args.sample_shift,
            "offload_model": args.offload_model,
            "t5_cpu": args.t5_cpu,
            "convert_model_dtype": args.convert_model_dtype,
        },
        "num_cases": len(cases),
        "cases": [asdict(case) for case in cases],
    }
    manifest_path = run_root / args.manifest_name
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2))
    print(str(manifest_path))


if __name__ == "__main__":
    main()
