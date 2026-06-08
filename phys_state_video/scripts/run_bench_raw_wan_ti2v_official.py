from __future__ import annotations

import argparse
import importlib
import json
import logging
import re
import sys
import types
from pathlib import Path
from typing import Any

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WAN_REPO_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run raw Wan-AI-Wan2.2-TI2V-5B on benchmark json entries with official-style TI2V calls."
    )
    parser.add_argument("--bench-json", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--ckpt-dir", required=True)
    parser.add_argument("--wan-repo-root", default=str(DEFAULT_WAN_REPO_ROOT))
    parser.add_argument("--task", default="ti2v-5B")
    parser.add_argument("--size", default="1280*704")
    parser.add_argument("--frame-num", type=int, default=None)
    parser.add_argument("--sample-solver", default="unipc", choices=["unipc", "dpm++"])
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--sample-shift", type=float, default=None)
    parser.add_argument("--sample-guide-scale", type=float, default=None)
    parser.add_argument("--base-seed", type=int, default=-1)
    parser.add_argument("--offload-model", action="store_true", default=False)
    parser.add_argument("--convert-model-dtype", action="store_true", default=False)
    parser.add_argument("--t5-cpu", action="store_true", default=False)
    parser.add_argument("--device-id", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true", default=False)
    return parser.parse_args()


def _slugify(text: str) -> str:
    normalized = re.sub(r"\s+", "_", text.strip())
    normalized = re.sub(r"[^0-9A-Za-z_\\-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "case"


def _load_cases(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if not isinstance(payload, list):
        raise ValueError(f"expected list json at {path}, got {type(payload).__name__}")
    return payload


def _case_stem(index: int, category: str, caption: str) -> str:
    return f"{index:03d}_{_slugify(category)}_{_slugify(caption)}_gen"


def _load_official_wan_modules(repo_root: str | Path) -> dict[str, Any]:
    repo_path = Path(repo_root)
    if not repo_path.exists():
        raise FileNotFoundError(f"Wan repo root does not exist: {repo_path}")
    if str(repo_path) not in sys.path:
        sys.path.insert(0, str(repo_path))

    # The official repo's `wan/__init__.py` eagerly imports every task,
    # including S2V, which pulls in optional audio deps like `librosa`.
    # For TI2V bench runs we only need a package shell plus the TI2V-related
    # submodules, so bootstrap `wan` as a namespace-like package and import
    # only the required pieces.
    wan_pkg_root = repo_path / "wan"
    if not wan_pkg_root.exists():
        raise FileNotFoundError(f"Official wan package directory does not exist: {wan_pkg_root}")

    for module_name in list(sys.modules.keys()):
        if module_name == "wan" or module_name.startswith("wan."):
            del sys.modules[module_name]

    wan_pkg = types.ModuleType("wan")
    wan_pkg.__path__ = [str(wan_pkg_root)]
    wan_pkg.__file__ = str(wan_pkg_root / "__init__.py")
    sys.modules["wan"] = wan_pkg

    return {
        "WAN_CONFIGS": importlib.import_module("wan.configs").WAN_CONFIGS,
        "SIZE_CONFIGS": importlib.import_module("wan.configs").SIZE_CONFIGS,
        "MAX_AREA_CONFIGS": importlib.import_module("wan.configs").MAX_AREA_CONFIGS,
        "SUPPORTED_SIZES": importlib.import_module("wan.configs").SUPPORTED_SIZES,
        "WanTI2V": importlib.import_module("wan.textimage2video").WanTI2V,
        "save_video": importlib.import_module("wan.utils.utils").save_video,
    }


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    modules = _load_official_wan_modules(args.wan_repo_root)
    cfg = modules["WAN_CONFIGS"][args.task]
    if args.size not in modules["SUPPORTED_SIZES"][args.task]:
        raise ValueError(
            f"unsupported size '{args.size}' for task '{args.task}', "
            f"expected one of {modules['SUPPORTED_SIZES'][args.task]}"
        )

    if args.frame_num is None:
        args.frame_num = cfg.frame_num
    if args.sample_steps is None:
        args.sample_steps = cfg.sample_steps
    if args.sample_shift is None:
        args.sample_shift = cfg.sample_shift
    if args.sample_guide_scale is None:
        args.sample_guide_scale = cfg.sample_guide_scale

    bench_json = Path(args.bench_json)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.info("Creating raw WanTI2V pipeline from official repo without state adapter.")
    pipeline = modules["WanTI2V"](
        config=cfg,
        checkpoint_dir=str(args.ckpt_dir),
        device_id=args.device_id,
        rank=0,
        t5_cpu=args.t5_cpu,
        convert_model_dtype=args.convert_model_dtype,
    )

    cases = _load_cases(bench_json)
    end_index = len(cases) if args.end_index is None else min(args.end_index, len(cases))

    for case_index in range(args.start_index, end_index):
        case = cases[case_index]
        category = str(case["category"])
        caption = str(case["caption"])
        input_image_path = Path(case["first_frame"])
        source_video_path = str(case["source_video"])
        stem = _case_stem(case_index, category, caption)
        output_video_path = output_dir / f"{stem}.mp4"
        output_json_path = output_dir / f"{stem}.json"

        if output_video_path.exists() and output_json_path.exists() and not args.overwrite:
            logging.info("Skipping existing case %s", stem)
            continue

        logging.info("Running case %s", stem)
        image = Image.open(input_image_path).convert("RGB")
        video = pipeline.generate(
            caption,
            img=image,
            size=modules["SIZE_CONFIGS"][args.size],
            max_area=modules["MAX_AREA_CONFIGS"][args.size],
            frame_num=args.frame_num,
            shift=args.sample_shift,
            sample_solver=args.sample_solver,
            sampling_steps=args.sample_steps,
            guide_scale=args.sample_guide_scale,
            seed=args.base_seed,
            offload_model=args.offload_model,
        )
        if video is None:
            raise RuntimeError(f"WanTI2V returned None for {stem}")

        modules["save_video"](
            tensor=video[None],
            save_file=str(output_video_path),
            fps=cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )
        output_json_path.write_text(
            json.dumps(
                {
                    "input_prompt": caption,
                    "input_image": str(input_image_path),
                    "source_video": source_video_path,
                    "model_name": "Wan-AI-Wan2.2-TI2V-5B_base",
                    "model_ckpt_dir": str(args.ckpt_dir),
                    "size": args.size,
                    "frame_num": args.frame_num,
                    "sample_solver": args.sample_solver,
                    "sample_steps": args.sample_steps,
                    "sample_shift": args.sample_shift,
                    "sample_guide_scale": args.sample_guide_scale,
                    "base_seed": args.base_seed,
                    "output_video": str(output_video_path),
                },
                ensure_ascii=False,
                indent=2,
            )
        )
        logging.info("Finished case %s", stem)


if __name__ == "__main__":
    main()
