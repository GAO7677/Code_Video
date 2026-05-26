#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import shutil
from pathlib import Path
from typing import Any

import cv2
import torch
from PIL import Image

from rerank_video.generators import VaceGenerator
from rerank_video.pdi_proxy_eval import VaceTI2VRunner, WanTI2VRunner
from rerank_video.schemas import GeneratorConfig, InputSpec
from rerank_video.video_utils import ensure_dir, write_json


DATASET_ROOT = Path("/data/gaoya/dataset/AnteaWu-PDI-Dataset")
OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/PDI-Bench")
GT_ROOT = OUTPUT_ROOT / "output" / "GT"
WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
VACE_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B")
METHODS = ["wan22-5B-TI2V", "VACE_1p3B_TI2V", "VACE_1p3B_ctx08"]
SEED = 42
FPS = 16
NUM_FRAMES = 49
NUM_INFERENCE_STEPS = 30
CFG_SCALE = 5.0
QUALITY = 5
WIDTH = 672
HEIGHT = 384
NEGATIVE_PROMPT = ""
DEVICE = "cuda"


def normalize_task(task: str) -> str:
    return "partial_occlusion" if task == "Partial_Occlusion" else task


def load_gt_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with (DATASET_ROOT / "metadata.csv").open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["provider"] != "GT":
                continue
            row = dict(row)
            row["task"] = normalize_task(row["task"])
            rows.append(row)
    return rows


def gt_video_path(row: dict[str, str]) -> Path:
    path = GT_ROOT / row["task"] / f"{row['prompt']}.mp4"
    if not path.is_file():
        raise FileNotFoundError(f"Missing copied GT video: {path}")
    return path


def gt_json_path(row: dict[str, str]) -> Path:
    return GT_ROOT / row["task"] / f"{row['prompt']}.json"


def ensure_gt_json_fields(row: dict[str, str]) -> dict[str, Any]:
    json_path = gt_json_path(row)
    payload = {}
    if json_path.is_file():
        import json

        payload = json.loads(json_path.read_text(encoding="utf-8"))
    video_path = gt_video_path(row)
    first_frame_path = GT_ROOT / row["task"] / f"{row['prompt']}.first_frame.png"
    if not first_frame_path.is_file():
        cap = cv2.VideoCapture(str(video_path))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read first frame from {video_path}")
        image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        image.save(first_frame_path)
    payload["source"] = str(video_path)
    payload["prompt"] = row["prompt"]
    payload["first_frame"] = str(first_frame_path)
    write_json(json_path, payload)
    return payload


def _copy_or_replace(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        dst.unlink()
    shutil.copy2(src, dst)


def _is_complete(output_video: Path, output_json: Path) -> bool:
    return output_video.is_file() and output_json.is_file()


def _method_json_payload(
    *,
    method: str,
    row: dict[str, str],
    output_video_path: Path,
    gt_payload: dict[str, Any],
    conditioning_mode: str,
    context_video_path: Path | None,
    context_frames: int,
) -> dict[str, Any]:
    payload = {
        "benchmark": "PDI-Bench",
        "method": method,
        "task": row["task"],
        "clip_name": row["prompt"],
        "prompt": row["prompt"],
        "source": gt_payload["source"],
        "first_frame": gt_payload["first_frame"],
        "video_path": str(output_video_path),
        "seed": SEED,
        "fps": FPS,
        "num_frames": NUM_FRAMES,
        "num_inference_steps": NUM_INFERENCE_STEPS,
        "cfg_scale": CFG_SCALE,
        "width": WIDTH,
        "height": HEIGHT,
        "conditioning_mode": conditioning_mode,
        "negative_prompt": NEGATIVE_PROMPT,
    }
    if context_video_path is not None:
        payload["context_video"] = str(context_video_path)
        payload["context_frames"] = context_frames
    return payload


def run_wan_ti2v(rows: list[dict[str, str]]) -> None:
    runner = WanTI2VRunner(model_root=WAN_ROOT, device=DEVICE)
    method_root = OUTPUT_ROOT / "output" / "wan22-5B-TI2V"
    for row in rows:
        gt_payload = ensure_gt_json_fields(row)
        task_dir = ensure_dir(method_root / row["task"])
        output_video = task_dir / f"{row['prompt']}.mp4"
        output_json = task_dir / f"{row['prompt']}.json"
        if _is_complete(output_video, output_json):
            print(f"[skip] wan22-5B-TI2V {row['task']}/{row['prompt']}")
            continue
        print(f"[run] wan22-5B-TI2V {row['task']}/{row['prompt']} seed={SEED}")
        runner.generate(
            first_frame_path=Path(gt_payload["first_frame"]),
            prompt=row["prompt"],
            output_path=output_video,
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
            output_json,
            _method_json_payload(
                method="wan22-5B-TI2V",
                row=row,
                output_video_path=output_video,
                gt_payload=gt_payload,
                conditioning_mode="TI2V_first_frame",
                context_video_path=None,
                context_frames=0,
            ),
        )
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_vace_ti2v(rows: list[dict[str, str]]) -> None:
    runner = VaceTI2VRunner(model_root=VACE_ROOT, device=DEVICE)
    method_root = OUTPUT_ROOT / "output" / "VACE_1p3B_TI2V"
    for row in rows:
        gt_payload = ensure_gt_json_fields(row)
        task_dir = ensure_dir(method_root / row["task"])
        output_video = task_dir / f"{row['prompt']}.mp4"
        output_json = task_dir / f"{row['prompt']}.json"
        if _is_complete(output_video, output_json):
            print(f"[skip] VACE_1p3B_TI2V {row['task']}/{row['prompt']}")
            continue
        print(f"[run] VACE_1p3B_TI2V {row['task']}/{row['prompt']} seed={SEED}")
        runner.generate(
            first_frame_path=Path(gt_payload["first_frame"]),
            prompt=row["prompt"],
            output_path=output_video,
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
            output_json,
            _method_json_payload(
                method="VACE_1p3B_TI2V",
                row=row,
                output_video_path=output_video,
                gt_payload=gt_payload,
                conditioning_mode="TI2V_first_frame",
                context_video_path=None,
                context_frames=0,
            ),
        )
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_vace_ctx08(rows: list[dict[str, str]]) -> None:
    config = GeneratorConfig(
        key="vace_ctx08",
        type="vace",
        enabled=True,
        device=DEVICE,
        model_root=VACE_ROOT,
        num_candidates=1,
        base_seed=SEED,
        height=HEIGHT,
        width=WIDTH,
        fps=FPS,
        num_frames=NUM_FRAMES,
        context_frames=8,
        num_inference_steps=NUM_INFERENCE_STEPS,
        cfg_scale=CFG_SCALE,
        quality=QUALITY,
        negative_prompt=NEGATIVE_PROMPT,
    )
    runner = VaceGenerator(config)
    method_root = OUTPUT_ROOT / "output" / "VACE_1p3B_ctx08"
    for row in rows:
        gt_payload = ensure_gt_json_fields(row)
        task_dir = ensure_dir(method_root / row["task"])
        output_video = task_dir / f"{row['prompt']}.mp4"
        output_json = task_dir / f"{row['prompt']}.json"
        if _is_complete(output_video, output_json):
            print(f"[skip] VACE_1p3B_ctx08 {row['task']}/{row['prompt']}")
            continue
        print(f"[run] VACE_1p3B_ctx08 {row['task']}/{row['prompt']} seed={SEED}")
        tmp_dir = ensure_dir(task_dir / "_tmp")
        records = runner.generate(
            input_spec=InputSpec(prompt=row["prompt"], context_video_path=gt_video_path(row)),
            config=config,
            output_dir=tmp_dir,
        )
        if len(records) != 1:
            raise RuntimeError(f"Expected exactly one VACE ctx record for {row['prompt']}, got {len(records)}")
        _copy_or_replace(records[0].video_path, output_video)
        for tmp_file in tmp_dir.glob("*"):
            tmp_file.unlink()
        tmp_dir.rmdir()
        write_json(
            output_json,
            _method_json_payload(
                method="VACE_1p3B_ctx08",
                row=row,
                output_video_path=output_video,
                gt_payload=gt_payload,
                conditioning_mode="V2V_ctx08",
                context_video_path=gt_video_path(row),
                context_frames=8,
            ),
        )
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=METHODS,
        default=METHODS,
        help="Subset of methods to generate.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_gt_rows()
    if "wan22-5B-TI2V" in args.methods:
        run_wan_ti2v(rows)
    if "VACE_1p3B_TI2V" in args.methods:
        run_vace_ti2v(rows)
    if "VACE_1p3B_ctx08" in args.methods:
        run_vace_ctx08(rows)
    print(f"Generated methods: {', '.join(args.methods)}")


if __name__ == "__main__":
    main()
