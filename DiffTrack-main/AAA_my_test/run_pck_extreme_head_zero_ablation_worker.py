#!/usr/bin/env python3
"""Stage-wise joint ablation of PCK Top30 and Bottom30 attention heads."""

from __future__ import annotations

import argparse
import csv
import gc
import json
import os
from pathlib import Path

import torch

from AAA_my_test import run_top5_head_zero_ablation_worker as top5


HEAD_CONFIG = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/"
    "train_xSSC/object_self_attn_lora_experiments/configs/common_t_heads_full70.json"
)
PCK_TABLE = Path(
    "/data/gaoya/agent-data/outputs/three_model_allblocks_allsteps_headwise_50case/"
    "three_model_combined_summary.csv"
)
OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_top30_bottom30_head_zero_ablation_test5"
)
LEGACY_ROOT = top5.OUTPUT_ROOT
STAGE_RANGES = (
    ("steps_00_10", tuple(range(0, 10))),
    ("steps_10_20", tuple(range(10, 20))),
    ("steps_20_30", tuple(range(20, 30))),
    ("steps_30_40", tuple(range(30, 40))),
    ("steps_00_40", tuple(range(0, 40))),
)


def select_heads(ranking_pool: str) -> dict[str, list[dict]]:
    allowed = None
    if ranking_pool == "common70":
        with HEAD_CONFIG.open(encoding="utf-8") as stream:
            allowed = {(int(x["block"]), int(x["head"])) for x in json.load(stream)["targets"]}
    best: dict[tuple[int, int], dict] = {}
    with PCK_TABLE.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            key = (int(row["block"]), int(row["head"]))
            if row["scope"] != "objects" or (allowed is not None and key not in allowed):
                continue
            record = {
                "global_rank": int(row["rank_macro_pck32"]),
                "step": int(row["step"]), "block": key[0], "head": key[1],
                "macro_pck32": float(row["macro_pck32"]),
                "gt_pck32": float(row["gt_macro_pck32"]),
                "lora_pck32": float(row["lora_macro_pck32"]),
                "baseline_pck32": float(row["baseline_macro_pck32"]),
            }
            if key not in best or record["global_rank"] < best[key]["global_rank"]:
                best[key] = record
    ranked = sorted(best.values(), key=lambda x: x["global_rank"])
    expected = 70 if ranking_pool == "common70" else 720
    if len(ranked) != expected:
        raise RuntimeError(f"expected {expected} ranked heads, got {len(ranked)}")
    return {"top30": ranked[:30], "bottom30": ranked[-30:]}


class ExtremeHeadZeroer:
    def __init__(self, pipe, groups: dict[str, list[dict]]) -> None:
        self.pipe = pipe
        self.group: str | None = None
        self.active_steps: set[int] = set()
        self.current_step = -1
        self.call_count = 0
        self.handles = []
        self.original_model_fn = pipe.model_fn
        grouped: dict[int, dict[str, list[int]]] = {}
        for group, rows in groups.items():
            for row in rows:
                grouped.setdefault(int(row["block"]), {}).setdefault(group, []).append(int(row["head"]))
        models = [pipe.dit]
        if getattr(pipe, "dit2", None) is not None and pipe.dit2 is not pipe.dit:
            models.append(pipe.dit2)
        for model in models:
            for block, by_group in grouped.items():
                module = model.blocks[block].self_attn.attn
                self.handles.append(module.register_forward_hook(self._make_hook(by_group)))
        pipe.model_fn = self._wrapped_model_fn

    def set_variant(self, group: str | None, steps: tuple[int, ...]) -> None:
        self.group = group
        self.active_steps = set(steps)

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

    def _make_hook(self, by_group: dict[str, list[int]]):
        def hook(module, _inputs, output):
            heads = by_group.get(self.group or "", ())
            if not heads or self.current_step not in self.active_steps:
                return output
            num_heads = int(module.num_heads)
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=("baseline", "lora"), required=True)
    parser.add_argument("--input-json-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--num-shards", type=int, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--ranking-pool", choices=("common70", "all720"), default="common70")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def generate(pipe, zeroer, context, prompt: str, group: str | None, steps: tuple[int, ...]):
    top5.source.probe.seed_everything(42)
    zeroer.set_variant(group, steps)
    with torch.inference_mode():
        return pipe(
            prompt=prompt, negative_prompt="", input_image=context[0], context_video=context,
            height=512, width=896, num_frames=top5.source.target.core.align_generation_num_frames(24),
            seed=42, cfg_scale=5.0, num_inference_steps=40, tiled=True,
        )


def main() -> None:
    args = parse_args()
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    groups = select_heads(args.ranking_pool)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    selection_path = output_root / "selection.json"
    if not selection_path.exists():
        selection_path.write_text(json.dumps(groups, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cases = top5.load_json_cases(args.input_json_list.expanduser().resolve())
    assigned = cases[args.shard_index :: args.num_shards]
    model_root = output_root / args.model
    model_root.mkdir(parents=True, exist_ok=True)
    lora_path = None if args.model == "baseline" else top5.source.target._resolve_lora_path(top5.source.DEFAULT_WEIGHTS_ROOT)
    pipe = top5.source.target.core.build_pipeline(top5.WAN_ROOT, str(args.device), lora_path)
    zeroer = ExtremeHeadZeroer(pipe, groups)
    variants = [("original", None, ())]
    variants += [(f"top30_{name}", "top30", steps) for name, steps in STAGE_RANGES]
    variants += [(f"bottom30_{name}", "bottom30", steps) for name, steps in STAGE_RANGES]
    completed = []
    try:
        for case_index, case in enumerate(assigned, start=1):
            case_root = model_root / "cases" / case["case_key"]
            case_root.mkdir(parents=True, exist_ok=True)
            context = top5.source.target.core.load_context_frames(
                context_path=Path(case["video"]), context_frames=8,
                height=512, width=896, resize_mode="crop",
            )
            files = {}
            for variant, group, steps in variants:
                video_path = case_root / f"{variant}.mp4"
                if variant == "original" and not video_path.exists():
                    legacy = LEGACY_ROOT / args.model / "cases" / case["case_key"] / "original.mp4"
                    if legacy.is_file() and legacy.stat().st_size:
                        os.link(legacy, video_path)
                if video_path.is_file() and video_path.stat().st_size and not args.overwrite:
                    print(f"[{case_index}/{len(assigned)}] skip {args.model} {case['case_key']} {variant}", flush=True)
                else:
                    before = zeroer.call_count
                    print(f"[{case_index}/{len(assigned)}] start {args.model} {case['case_key']} {variant}", flush=True)
                    video = generate(pipe, zeroer, context, str(case["caption"]), group, steps)
                    top5.source.probe.save_video(video, str(video_path), fps=30, quality=5)
                    del video
                    gc.collect()
                    torch.cuda.empty_cache()
                    if group is not None and zeroer.call_count == before:
                        raise RuntimeError(f"head-zero hooks not invoked for {variant}")
                files[variant] = str(video_path)
                progress = {
                    "model": args.model, "shard_index": args.shard_index,
                    "num_shards": args.num_shards, "case_key": case["case_key"],
                    "case_index": case_index, "case_count": len(assigned), "variant": variant,
                }
                (model_root / f"progress_shard_{args.shard_index:02d}.json").write_text(
                    json.dumps(progress, indent=2) + "\n", encoding="utf-8"
                )
            manifest = {
                "case_key": case["case_key"], "model": args.model,
                "input_json": case["input_json"], "context_video": case["video"],
                "prompt": case["caption"], "seed": 42, "sampling_steps": 40,
                "ranking_pool": args.ranking_pool, "selection": str(selection_path), "files": files,
            }
            (case_root / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            completed.append(case["case_key"])
    finally:
        zeroer.remove()
    (model_root / f"complete_shard_{args.shard_index:02d}.json").write_text(
        json.dumps({"model": args.model, "shard": args.shard_index, "completed": completed}, indent=2) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
