#!/usr/bin/env python3
"""Generate paired Wan videos with five high-PCK self-attention heads zeroed by stage."""

from __future__ import annotations

import argparse
import gc
import json
from pathlib import Path

import torch

from AAA_my_test import run_lorav2v_toy_analysis_worker as source


TOP5 = (
    {"global_rank": 190, "step": 39, "block": 9, "head": 13, "macro_pck32": 87.769478},
    {"global_rank": 210, "step": 39, "block": 7, "head": 8, "macro_pck32": 87.058082},
    {"global_rank": 226, "step": 39, "block": 6, "head": 2, "macro_pck32": 86.660507},
    {"global_rank": 231, "step": 39, "block": 21, "head": 7, "macro_pck32": 86.509190},
    {"global_rank": 405, "step": 39, "block": 13, "head": 5, "macro_pck32": 84.399165},
)
CASES = (
    "case_001_ball_roll",
    "case_002_puck_slide",
    "case_003_capsule_slide",
    "case_004_cylinder_topple",
    "case_005_box_slide",
)
STAGES = (
    ("original", ()),
    ("steps_00_10", tuple(range(0, 10))),
    ("steps_10_20", tuple(range(10, 20))),
    ("steps_20_30", tuple(range(20, 30))),
    ("steps_30_40", tuple(range(30, 40))),
    ("steps_00_40", tuple(range(0, 40))),
)
OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/top5_pck_head_zero_ablation_5case")
WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")


class HeadOutputZeroer:
    def __init__(self, pipe) -> None:
        self.pipe = pipe
        self.active_steps: set[int] = set()
        self.current_step = -1
        self.call_count = 0
        self.handles = []
        self.original_model_fn = pipe.model_fn
        grouped: dict[int, list[int]] = {}
        for item in TOP5:
            grouped.setdefault(int(item["block"]), []).append(int(item["head"]))
        models = [pipe.dit]
        if getattr(pipe, "dit2", None) is not None and pipe.dit2 is not pipe.dit:
            models.append(pipe.dit2)
        for model in models:
            for block, heads in grouped.items():
                if not 0 <= block < len(model.blocks):
                    raise ValueError(f"block {block} outside [0, {len(model.blocks) - 1}]")
                module = model.blocks[block].self_attn.attn
                for head in heads:
                    if not 0 <= head < int(module.num_heads):
                        raise ValueError(f"head {head} outside [0, {int(module.num_heads) - 1}]")
                self.handles.append(
                    module.register_forward_hook(self._make_head_hook(tuple(sorted(heads))))
                )
        pipe.model_fn = self._wrapped_model_fn

    def set_stage(self, steps: tuple[int, ...]) -> None:
        self.active_steps = set(steps)
        self.current_step = -1

    def _scheduler_step(self, timestep: torch.Tensor) -> int:
        value = float(timestep.detach().flatten()[0].float().cpu().item())
        timesteps = self.pipe.scheduler.timesteps.detach().float().cpu()
        return int(torch.argmin((timesteps - value).abs()).item())

    def _wrapped_model_fn(self, *args, **kwargs):
        timestep = kwargs.get("timestep")
        if timestep is None:
            return self.original_model_fn(*args, **kwargs)
        self.current_step = self._scheduler_step(timestep)
        try:
            return self.original_model_fn(*args, **kwargs)
        finally:
            self.current_step = -1

    def _make_head_hook(self, heads: tuple[int, ...]):
        def hook(module, _inputs, output):
            if self.current_step not in self.active_steps:
                return output
            if not isinstance(output, torch.Tensor) or output.ndim != 3:
                raise RuntimeError(f"unexpected attention output: {type(output)}")
            num_heads = int(module.num_heads)
            if output.shape[-1] % num_heads:
                raise RuntimeError(f"attention width {output.shape[-1]} is not head-aligned")
            head_dim = output.shape[-1] // num_heads
            masked = output.clone()
            for head in heads:
                masked[..., head * head_dim : (head + 1) * head_dim] = 0
            self.call_count += 1
            return masked

        return hook

    def remove(self) -> None:
        self.pipe.model_fn = self.original_model_fn
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("baseline", "lora"), required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--input-json-list", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_json_cases(list_path: Path) -> list[dict]:
    seen = set()
    cases = []
    for raw_line in list_path.read_text(encoding="utf-8").splitlines():
        raw_line = raw_line.strip()
        if not raw_line:
            continue
        json_path = Path(raw_line).expanduser().resolve()
        if json_path in seen:
            continue
        seen.add(json_path)
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        video = Path(str(payload.get("input_video") or "")).expanduser().resolve()
        caption = str(payload.get("input_caption") or "").strip()
        if not video.is_file() or not caption:
            raise RuntimeError(f"invalid input JSON: {json_path}")
        cases.append(
            {
                "case_key": json_path.stem,
                "manifest": str(json_path),
                "input_json": str(json_path),
                "video": str(video),
                "caption": caption,
                "dataset_group": "test_5.txt",
            }
        )
    if not cases:
        raise RuntimeError(f"no unique cases in {list_path}")
    return cases


def generate(pipe, zeroer: HeadOutputZeroer, context, prompt: str, steps: tuple[int, ...]):
    source.probe.seed_everything(42)
    zeroer.set_stage(steps)
    with torch.inference_mode():
        return pipe(
            prompt=prompt,
            negative_prompt="",
            input_image=context[0],
            context_video=context,
            height=512,
            width=896,
            num_frames=source.target.core.align_generation_num_frames(24),
            seed=42,
            cfg_scale=5.0,
            num_inference_steps=40,
            tiled=True,
        )


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    model_root = output_root / args.model
    model_root.mkdir(parents=True, exist_ok=True)
    lora_path = None if args.model == "baseline" else source.target._resolve_lora_path(source.DEFAULT_WEIGHTS_ROOT)
    pipe = source.target.core.build_pipeline(WAN_ROOT, str(args.device), lora_path)
    zeroer = HeadOutputZeroer(pipe)
    cases = (
        load_json_cases(args.input_json_list.expanduser().resolve())
        if args.input_json_list is not None
        else source.load_cases(source.DEFAULT_DATASET_ROOT, list(CASES))
    )
    run_label = "test5" if args.input_json_list is not None else "toy5"
    completed = []
    try:
        for case_index, case in enumerate(cases, start=1):
            case_root = model_root / "cases" / case["case_key"]
            case_root.mkdir(parents=True, exist_ok=True)
            context = source.target.core.load_context_frames(
                context_path=Path(case["video"]), context_frames=8,
                height=512, width=896, resize_mode="crop",
            )
            files = {}
            for variant, steps in STAGES:
                video_path = case_root / f"{variant}.mp4"
                if video_path.is_file() and video_path.stat().st_size > 0 and not args.overwrite:
                    print(f"[{case_index}/{len(cases)}] skip {args.model} {case['case_key']} {variant}", flush=True)
                else:
                    before = zeroer.call_count
                    print(f"[{case_index}/{len(cases)}] start {args.model} {case['case_key']} {variant}", flush=True)
                    video = generate(pipe, zeroer, context, str(case["caption"]), steps)
                    source.probe.save_video(video, str(video_path), fps=30, quality=5)
                    del video
                    gc.collect()
                    torch.cuda.empty_cache()
                    if steps and zeroer.call_count == before:
                        raise RuntimeError(f"head-zero hooks were not invoked for {variant}")
                    print(f"[{case_index}/{len(cases)}] complete {args.model} {case['case_key']} {variant}", flush=True)
                files[variant] = str(video_path)
                (model_root / f"progress_{run_label}.json").write_text(
                    json.dumps(
                        {
                            "model": args.model, "run": run_label,
                            "case_index": case_index, "case_count": len(cases),
                            "case_key": case["case_key"], "variant": variant,
                        },
                        ensure_ascii=False, indent=2,
                    ) + "\n",
                    encoding="utf-8",
                )
            manifest = {
                "case_key": case["case_key"], "model": args.model,
                "weights_root": None if args.model == "baseline" else str(source.DEFAULT_WEIGHTS_ROOT),
                "checkpoint": None if lora_path is None else str(lora_path),
                "context_video": case["video"], "prompt": case["caption"],
                "seed": 42, "sampling_steps": 40, "height": 512, "width": 896,
                "requested_num_frames": 24,
                "input_json": case.get("input_json"),
                "dataset_group": case.get("dataset_group", "ToyDataset"),
                "ablation": "zero five selected self-attention head outputs before o projection",
                "stage_semantics": "left-closed right-open denoising step indices",
                "top5": TOP5, "stages": {name: list(steps) for name, steps in STAGES},
                "files": files,
            }
            (case_root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            completed.append(case["case_key"])
    finally:
        zeroer.remove()
    (model_root / f"complete_{run_label}.json").write_text(
        json.dumps({"model": args.model, "run": run_label, "completed": completed, "top5": TOP5}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
