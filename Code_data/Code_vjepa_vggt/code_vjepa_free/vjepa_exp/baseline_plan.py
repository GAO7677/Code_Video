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
DEFAULT_SOURCE_VIDEO = Path(
    "/data/gaoya/AAA_test_video/0623/testdataset/"
    "025_Solid_Mechanics_0002_perspective-center_trimmed/"
    "physicIQ_0002_clip_2p5s_3p5s.mp4"
)
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")

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
    source_video: str
    input_video: str
    input_image: str
    input_caption: str
    run_dir: str
    input_json: str
    output_video: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create Wan TI2V baseline inputs compatible with AAAinfer/wanti2v.py.")
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--manifest-name", default="manifest.json")
    parser.add_argument("--input-list-name", default="input_list.txt")
    parser.add_argument("--image-path", dest="image_paths", action="append", type=Path, default=None)
    parser.add_argument("--image-list", type=Path, default=None)
    parser.add_argument("--source-video", dest="source_videos", action="append", type=Path, default=None)
    parser.add_argument("--source-video-list", type=Path, default=None)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--backend", default="official", choices=["official", "legacy"])
    parser.add_argument("--size", default="704*1280", choices=["704*1280", "1280*704"])
    parser.add_argument("--frame-num", type=int, default=25)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--sample-shift", type=float, default=5.0)
    parser.add_argument("--sample-solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--offload-model", action="store_true", default=True)
    parser.add_argument("--t5-cpu", action="store_true", default=True)
    parser.add_argument("--convert-model-dtype", action="store_true", default=True)
    return parser.parse_args()


def _resolve_paths(cli_paths: list[Path] | None, list_path: Path | None, default_path: Path) -> list[Path]:
    paths = list(cli_paths or [])
    if list_path is not None:
        lines = list_path.expanduser().resolve().read_text().splitlines()
        paths.extend(Path(line.strip()) for line in lines if line.strip())
    if not paths:
        paths = [default_path]
    resolved = [path.expanduser().resolve() for path in paths]
    for path in resolved:
        if not path.exists():
            raise FileNotFoundError(f"Path does not exist: {path}")
    return resolved


def build_cases(
    *,
    run_name: str,
    seeds: list[int],
    prompts: list[str],
    image_paths: list[Path],
    source_videos: list[Path],
    output_root: Path,
) -> list[BaselineCase]:
    run_root = output_root / run_name
    cases: list[BaselineCase] = []
    for prompt_idx, prompt in enumerate(prompts):
        image_path = image_paths[prompt_idx % len(image_paths)].expanduser().resolve()
        source_video = source_videos[prompt_idx % len(source_videos)].expanduser().resolve()
        for seed in seeds:
            case_id = f"p{prompt_idx:02d}_s{seed}"
            case_dir = run_root / case_id
            input_json = case_dir / f"{case_id}.json"
            output_video = case_dir / "video.mp4"
            cases.append(
                BaselineCase(
                    case_id=case_id,
                    prompt=prompt,
                    seed=seed,
                    source_video=str(source_video),
                    input_video=str(source_video),
                    input_image=str(image_path),
                    input_caption=prompt,
                    run_dir=str(case_dir),
                    input_json=str(input_json),
                    output_video=str(output_video),
                )
            )
    return cases


def main() -> None:
    args = parse_args()
    output_root = args.output_root.expanduser().resolve()
    run_root = output_root / args.run_name
    run_root.mkdir(parents=True, exist_ok=True)

    image_paths = _resolve_paths(args.image_paths, args.image_list, DEFAULT_IMAGE_PATH)
    source_videos = _resolve_paths(args.source_videos, args.source_video_list, DEFAULT_SOURCE_VIDEO)
    cases = build_cases(
        run_name=args.run_name,
        seeds=args.seeds,
        prompts=DEFAULT_PROMPTS,
        image_paths=image_paths,
        source_videos=source_videos,
        output_root=output_root,
    )

    input_list_path = run_root / args.input_list_name
    with input_list_path.open("w", encoding="utf-8") as list_handle:
        for case in cases:
            case_dir = Path(case.run_dir)
            case_dir.mkdir(parents=True, exist_ok=True)
            input_json_path = Path(case.input_json)
            payload = {
                "source_video": case.source_video,
                "input_video": case.input_video,
                "input_image": case.input_image,
                "input_caption": case.input_caption,
                "seed": case.seed,
                "case_id": case.case_id,
                "expected_output_video": case.output_video,
            }
            input_json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
            list_handle.write(str(input_json_path) + "\n")

    payload = {
        "run_name": args.run_name,
        "wan_root": str(args.wan_root.expanduser().resolve()),
        "backend": args.backend,
        "input_list": str(input_list_path),
        "output_root": str(run_root),
        "wan_args": {
            "size": args.size,
            "frame_num": args.frame_num,
            "fps": args.fps,
            "sampling_steps": args.sampling_steps,
            "sample_shift": args.sample_shift,
            "sample_solver": args.sample_solver,
            "cfg_scale": args.cfg_scale,
            "negative_prompt": args.negative_prompt,
            "offload_model": args.offload_model,
            "t5_cpu": args.t5_cpu,
            "convert_model_dtype": args.convert_model_dtype,
        },
        "num_cases": len(cases),
        "cases": [asdict(case) for case in cases],
    }
    manifest_path = run_root / args.manifest_name
    manifest_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(str(manifest_path))


if __name__ == "__main__":
    main()
