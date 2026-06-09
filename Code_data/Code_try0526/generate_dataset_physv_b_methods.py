#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import importlib
import json
import re
import shutil
import sys
import types
from pathlib import Path
from typing import Any

import torch

from rerank_video.generators import VaceGenerator
from rerank_video.pdi_proxy_eval import VaceTI2VRunner, WanTI2VRunner
from rerank_video.schemas import GeneratorConfig, InputSpec
from rerank_video.video_utils import ensure_dir, extract_first_frame, write_json


BENCH_JSON = Path("/data/gaoya/AAA_test_video/Output_try0526/bench_jsons_mer/B.json")
OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/Dataset_physV_B_benchmark")
WAN_REPO_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/Wan2.2-main")
WAN_CKPT_DIR = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
VACE_CKPT_DIR = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B")

METHODS = ["wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]

SEED = 42
FPS = 16
NUM_FRAMES = 49
CONTEXT_FRAMES = 8
NUM_INFERENCE_STEPS = 30
CFG_SCALE = 5.0
QUALITY = 5
WIDTH = 672
HEIGHT = 384
NEGATIVE_PROMPT = ""
DEVICE = "cuda"
WAN_TASK = "ti2v-5B"
WAN_SIZE = "1280*704"
WAN_SAMPLE_SOLVER = "unipc"
WAN_BACKEND = "legacy"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Dataset_physV B benchmark videos with Wan/VACE baselines.")
    parser.add_argument("--bench-json", type=Path, default=BENCH_JSON)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=METHODS)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--device", type=str, default=DEVICE)
    parser.add_argument("--wan-repo-root", type=Path, default=WAN_REPO_ROOT)
    parser.add_argument("--wan-ckpt-dir", type=Path, default=WAN_CKPT_DIR)
    parser.add_argument("--vace-ckpt-dir", type=Path, default=VACE_CKPT_DIR)
    parser.add_argument("--wan-task", type=str, default=WAN_TASK)
    parser.add_argument("--wan-size", type=str, default=WAN_SIZE)
    parser.add_argument("--wan-sample-solver", type=str, choices=["unipc", "dpm++"], default=WAN_SAMPLE_SOLVER)
    parser.add_argument("--wan-backend", type=str, choices=["legacy", "official"], default=WAN_BACKEND)
    return parser.parse_args()


def _slugify(text: str) -> str:
    normalized = re.sub(r"\s+", "_", text.strip())
    normalized = re.sub(r"[^0-9A-Za-z_\-]+", "_", normalized)
    normalized = re.sub(r"_+", "_", normalized).strip("_")
    return normalized or "case"


def _case_key(index: int, category: str, clip_name: str) -> str:
    return f"{index:03d}_{_slugify(category)}_{_slugify(clip_name)}"


def load_rows(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"Expected list json at {path}, got {type(payload).__name__}")
    return payload


def slice_rows(rows: list[dict[str, Any]], args: argparse.Namespace) -> list[tuple[int, dict[str, Any]]]:
    indexed = list(enumerate(rows))
    if args.start_index > 0:
        indexed = indexed[args.start_index:]
    if args.end_index is not None:
        indexed = indexed[: max(args.end_index - args.start_index, 0)]
    if args.limit is not None:
        indexed = indexed[: args.limit]
    return indexed


def first_frame_path(output_root: Path, case_key: str) -> Path:
    return output_root / "_inputs" / f"{case_key}.first_frame.png"


def output_paths(output_root: Path, method: str, case_key: str) -> tuple[Path, Path]:
    base = output_root / "output" / method / f"{case_key}"
    return base.with_suffix(".mp4"), base.with_suffix(".json")


def manifest_path(output_root: Path) -> Path:
    return output_root / "manifest.json"


def is_complete(video_path: Path, json_path: Path, overwrite: bool) -> bool:
    return not overwrite and video_path.is_file() and json_path.is_file()


def _load_official_wan_modules(repo_root: Path) -> dict[str, Any]:
    if not repo_root.exists():
        raise FileNotFoundError(f"Wan repo root does not exist: {repo_root}")
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    wan_pkg_root = repo_root / "wan"
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


class OfficialWanTI2VRunner:
    def __init__(
        self,
        *,
        repo_root: Path,
        ckpt_dir: Path,
        task: str,
        size: str,
        sample_solver: str,
        device_id: int,
    ) -> None:
        self.modules = _load_official_wan_modules(repo_root)
        if size not in self.modules["SUPPORTED_SIZES"][task]:
            raise ValueError(
                f"Unsupported size '{size}' for task '{task}', expected one of {self.modules['SUPPORTED_SIZES'][task]}"
            )
        self.cfg = self.modules["WAN_CONFIGS"][task]
        self.task = task
        self.size = size
        self.sample_solver = sample_solver
        self.pipeline = self.modules["WanTI2V"](
            config=self.cfg,
            checkpoint_dir=str(ckpt_dir),
            device_id=device_id,
            rank=0,
            t5_cpu=False,
            convert_model_dtype=False,
        )

    def generate(self, *, prompt: str, first_frame: Path, output_path: Path) -> None:
        from PIL import Image

        image = Image.open(first_frame).convert("RGB")
        video = self.pipeline.generate(
            prompt,
            img=image,
            size=self.modules["SIZE_CONFIGS"][self.size],
            max_area=self.modules["MAX_AREA_CONFIGS"][self.size],
            frame_num=self.cfg.frame_num,
            shift=self.cfg.sample_shift,
            sample_solver=self.sample_solver,
            sampling_steps=self.cfg.sample_steps,
            guide_scale=self.cfg.sample_guide_scale,
            seed=SEED,
            offload_model=False,
        )
        if video is None:
            raise RuntimeError(f"WanTI2V returned None for output {output_path}")
        ensure_dir(output_path.parent)
        self.modules["save_video"](
            tensor=video[None],
            save_file=str(output_path),
            fps=self.cfg.sample_fps,
            nrow=1,
            normalize=True,
            value_range=(-1, 1),
        )


def ensure_first_frame(output_root: Path, case_key: str, source_video: Path) -> Path:
    path = first_frame_path(output_root, case_key)
    if path.is_file():
        return path
    frame = extract_first_frame(source_video)
    ensure_dir(path.parent)
    frame.save(path)
    return path


def build_payload(
    *,
    benchmark_name: str,
    method_name: str,
    category: str,
    case_key: str,
    source_video: Path,
    output_video: Path,
    first_frame: Path,
    prompt: str,
    context_frames: int,
    conditioning_mode: str,
) -> dict[str, Any]:
    return {
        "benchmark": benchmark_name,
        "method_name": method_name,
        "category": category,
        "case_key": case_key,
        "input_prompt": prompt,
        "input_image": str(first_frame) if context_frames == 1 else None,
        "input_context_video": str(source_video) if context_frames > 1 else None,
        "source_video": str(source_video),
        "output_video": str(output_video),
        "conditioning_mode": conditioning_mode,
        "context_frames": context_frames,
        "seed": SEED,
        "fps": FPS,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "cfg_scale": CFG_SCALE,
        "width": WIDTH,
        "height": HEIGHT,
        "negative_prompt": NEGATIVE_PROMPT,
    }


def run_wan_ti2v(
    output_root: Path,
    rows: list[tuple[int, dict[str, Any]]],
    args: argparse.Namespace,
) -> None:
    if args.wan_backend == "official":
        runner: Any = OfficialWanTI2VRunner(
            repo_root=args.wan_repo_root,
            ckpt_dir=args.wan_ckpt_dir,
            task=args.wan_task,
            size=args.wan_size,
            sample_solver=args.wan_sample_solver,
            device_id=0,
        )
    else:
        runner = WanTI2VRunner(model_root=args.wan_ckpt_dir, device=args.device)
    for index, row in rows:
        category = str(row["category"])
        source_video = Path(str(row["source_video"]))
        prompt = str(row["caption"])
        case_key = _case_key(index, category, source_video.stem)
        image_path = ensure_first_frame(output_root, case_key, source_video)
        output_video_path, output_json_path = output_paths(output_root, "wan22-5B-TI2V", case_key)
        if is_complete(output_video_path, output_json_path, args.overwrite):
            print(f"[skip] wan22-5B-TI2V {case_key}", flush=True)
            continue
        print(f"[run] wan22-5B-TI2V {case_key}", flush=True)
        if args.wan_backend == "official":
            runner.generate(prompt=prompt, first_frame=image_path, output_path=output_video_path)
        else:
            runner.generate(
                first_frame_path=image_path,
                prompt=prompt,
                output_path=output_video_path,
                seed=SEED,
                negative_prompt=NEGATIVE_PROMPT,
                width=WIDTH,
                height=HEIGHT,
                num_frames=NUM_FRAMES,
                fps=FPS,
                num_inference_steps=NUM_INFERENCE_STEPS,
                cfg_scale=CFG_SCALE,
                quality=QUALITY,
            )
        write_json(
            output_json_path,
            build_payload(
                benchmark_name="Dataset_physV_B_benchmark",
                method_name="wan22-5B-TI2V",
                category=category,
                case_key=case_key,
                source_video=source_video,
                output_video=output_video_path,
                first_frame=image_path,
                prompt=prompt,
                context_frames=1,
                conditioning_mode="TI2V_first_frame",
            ),
        )
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_vace_ti2v(output_root: Path, rows: list[tuple[int, dict[str, Any]]], args: argparse.Namespace) -> None:
    runner = VaceTI2VRunner(model_root=args.vace_ckpt_dir, device=args.device)
    for index, row in rows:
        category = str(row["category"])
        source_video = Path(str(row["source_video"]))
        prompt = str(row["caption"])
        case_key = _case_key(index, category, source_video.stem)
        image_path = ensure_first_frame(output_root, case_key, source_video)
        output_video_path, output_json_path = output_paths(output_root, "VACE_1p3B_TI2V", case_key)
        if is_complete(output_video_path, output_json_path, args.overwrite):
            print(f"[skip] VACE_1p3B_TI2V {case_key}", flush=True)
            continue
        print(f"[run] VACE_1p3B_TI2V {case_key}", flush=True)
        runner.generate(
            first_frame_path=image_path,
            prompt=prompt,
            output_path=output_video_path,
            seed=SEED,
            negative_prompt=NEGATIVE_PROMPT,
            width=WIDTH,
            height=HEIGHT,
            num_frames=NUM_FRAMES,
            fps=FPS,
            num_inference_steps=NUM_INFERENCE_STEPS,
            cfg_scale=CFG_SCALE,
            quality=QUALITY,
        )
        write_json(
            output_json_path,
            build_payload(
                benchmark_name="Dataset_physV_B_benchmark",
                method_name="VACE_1p3B_TI2V",
                category=category,
                case_key=case_key,
                source_video=source_video,
                output_video=output_video_path,
                first_frame=image_path,
                prompt=prompt,
                context_frames=1,
                conditioning_mode="TI2V_first_frame",
            ),
        )
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_vace_ctx08(output_root: Path, rows: list[tuple[int, dict[str, Any]]], args: argparse.Namespace) -> None:
    config = GeneratorConfig(
        key="vace_ctx08",
        type="vace",
        enabled=True,
        device=args.device,
        model_root=args.vace_ckpt_dir,
        num_candidates=1,
        base_seed=SEED,
        height=HEIGHT,
        width=WIDTH,
        fps=FPS,
        num_frames=NUM_FRAMES,
        context_frames=CONTEXT_FRAMES,
        num_inference_steps=NUM_INFERENCE_STEPS,
        cfg_scale=CFG_SCALE,
        quality=QUALITY,
        negative_prompt=NEGATIVE_PROMPT,
    )
    runner = VaceGenerator(config)
    for index, row in rows:
        category = str(row["category"])
        source_video = Path(str(row["source_video"]))
        prompt = str(row["caption"])
        case_key = _case_key(index, category, source_video.stem)
        image_path = ensure_first_frame(output_root, case_key, source_video)
        output_video_path, output_json_path = output_paths(output_root, "VACE_1p3B_ctx08", case_key)
        if is_complete(output_video_path, output_json_path, args.overwrite):
            print(f"[skip] VACE_1p3B_ctx08 {case_key}", flush=True)
            continue
        print(f"[run] VACE_1p3B_ctx08 {case_key}", flush=True)
        tmp_dir = ensure_dir(output_video_path.parent / "_tmp")
        records = runner.generate(
            input_spec=InputSpec(prompt=prompt, context_video_path=source_video),
            config=config,
            output_dir=tmp_dir,
        )
        if len(records) != 1:
            raise RuntimeError(f"Expected exactly one ctx08 record for {case_key}, got {len(records)}")
        ensure_dir(output_video_path.parent)
        shutil.copy2(records[0].video_path, output_video_path)
        records[0].video_path.unlink()
        if not any(tmp_dir.iterdir()):
            tmp_dir.rmdir()
        write_json(
            output_json_path,
            build_payload(
                benchmark_name="Dataset_physV_B_benchmark",
                method_name="VACE_1p3B_ctx08",
                category=category,
                case_key=case_key,
                source_video=source_video,
                output_video=output_video_path,
                first_frame=image_path,
                prompt=prompt,
                context_frames=CONTEXT_FRAMES,
                conditioning_mode="V2V_ctx08",
            ),
        )
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def write_manifest(output_root: Path, rows: list[tuple[int, dict[str, Any]]], methods: list[str]) -> None:
    payload = {
        "benchmark": "Dataset_physV_B_benchmark",
        "source_manifest": str(BENCH_JSON),
        "output_root": str(output_root),
        "num_cases": len(rows),
        "methods": methods,
        "case_keys": [
            _case_key(index, str(row["category"]), Path(str(row["source_video"])).stem)
            for index, row in rows
        ],
    }
    write_json(manifest_path(output_root), payload)


def main() -> None:
    args = parse_args()
    rows = load_rows(args.bench_json)
    sliced_rows = slice_rows(rows, args)
    write_manifest(args.output_root, sliced_rows, args.methods)
    if "wan22-5B-TI2V" in args.methods:
        run_wan_ti2v(args.output_root, sliced_rows, args)
    if "VACE_1p3B_TI2V" in args.methods:
        run_vace_ti2v(args.output_root, sliced_rows, args)
    if "VACE_1p3B_ctx08" in args.methods:
        run_vace_ctx08(args.output_root, sliced_rows, args)
    print(f"Generated methods: {', '.join(args.methods)}", flush=True)


if __name__ == "__main__":
    main()
