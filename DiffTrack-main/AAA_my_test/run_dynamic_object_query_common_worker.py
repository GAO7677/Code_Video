#!/usr/bin/env python3
"""Capture no-op Top100 attention using CoTracker-derived query tokens per latent frame."""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

from AAA_my_test import run_attention_lora_seed_sweep_worker as launcher


LATENT_FRAMES = 13
LATENT_HEIGHT = 16
LATENT_WIDTH = 28
SPATIAL_TOKENS = LATENT_HEIGHT * LATENT_WIDTH


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--inference-steps", type=int, choices=(10, 40), required=True)
    parser.add_argument("--track-file", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--input-json-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def load_dynamic_regions(path: Path) -> list[dict]:
    with np.load(path, allow_pickle=False) as data:
        names = data["region_names"].astype(str)
        phrases = data["region_phrases"].astype(str)
        starts = data["point_starts"].astype(np.int64)
        ends = data["point_ends"].astype(np.int64)
        tokens = data["anchor_token_indices"].astype(np.int64)
        tracks = data["tracks"].astype(np.float32)
        visibility = data["visibility"].astype(bool)
        anchors = data["anchor_frames"].astype(np.int64)
    regions = []
    for name, phrase, start, end in zip(names, phrases, starts, ends):
        frame_rows = []
        for latent_index in range(LATENT_FRAMES):
            rows = tokens[latent_index, start:end]
            frame_rows.append(sorted({int(row) for row in rows if int(row) >= 0}))
        regions.append(
            {
                "name": str(name),
                "phrase": str(phrase),
                "frame_rows": frame_rows,
                "tracks": tracks[:, start:end],
                "visibility": visibility[:, start:end],
                "anchors": anchors,
            }
        )
    return regions


def build_dynamic_capture(parent, args: argparse.Namespace):
    dynamic_regions = load_dynamic_regions(args.track_file)

    class DynamicObjectQueryCapture(parent):
        def __init__(self, pipe, groups):
            super().__init__(pipe, groups)
            self.dynamic_regions = dynamic_regions
            self.dynamic_capture_root = args.capture_root
            self.dynamic_entries = {}
            self.current_timestep_value = float("nan")
            self.current_sigma = float("nan")

        def _wrapped_model_fn(self, *model_args, **kwargs):
            timestep = kwargs.get("timestep")
            self.current_step = self._scheduler_step(timestep) if timestep is not None else -1
            self.current_timestep_value = (
                float(timestep.detach().flatten()[0].cpu().item())
                if timestep is not None
                else float("nan")
            )
            sigmas = getattr(self.pipe.scheduler, "sigmas", None)
            self.current_sigma = (
                float(sigmas[self.current_step].detach().cpu().item())
                if self.current_step >= 0 and sigmas is not None
                else float("nan")
            )
            if self.current_step != self._cfg_step:
                self._cfg_step = self.current_step
                self._cfg_call_index = 0
            else:
                self._cfg_call_index += 1
            self.current_cfg_branch = (
                "conditional" if self._cfg_call_index % 2 == 0 else "unconditional"
            )
            self.group = (
                f"{self.adaptive_prefix}_step_{self.current_step:02d}"
                if self.adaptive_prefix is not None and self.current_step >= 0
                else None
            )
            self.dynamic_entries = {
                region["name"]: {
                    "sum": torch.zeros(
                        (LATENT_FRAMES, LATENT_HEIGHT, LATENT_WIDTH), dtype=torch.float64
                    ),
                    "head_counts": torch.zeros(LATENT_FRAMES, dtype=torch.int32),
                }
                for region in self.dynamic_regions
            }
            result = self.original_model_fn(*model_args, **kwargs)
            if self.group is not None:
                self._write_dynamic_capture()
            return result

        @staticmethod
        def _reshape_qk(q, k):
            num_heads = int(q.shape[-1] // 128)
            head_dim = int(q.shape[-1] // num_heads)
            qh = q.reshape(q.shape[0], q.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
            kh = k.reshape(k.shape[0], k.shape[1], num_heads, head_dim).permute(0, 2, 1, 3)
            return qh, kh, head_dim

        def _attention(self, q, k, v, original, groups, block):
            output = original(q, k, v)
            heads = sorted(set(groups.get(self.group or "", ())))
            if not heads or self.current_step not in self.active_steps:
                return output
            qh, kh, head_dim = self._reshape_qk(q, k)
            if qh.shape[2] != LATENT_FRAMES * SPATIAL_TOKENS:
                raise RuntimeError(f"Expected 5824 tokens, got {qh.shape[2]}")
            all_rows = sorted(
                {
                    row
                    for region in self.dynamic_regions
                    for frame_rows in region["frame_rows"]
                    for row in frame_rows
                }
            )
            if not all_rows:
                return output
            row_positions = {row: index for index, row in enumerate(all_rows)}
            row_tensor = torch.as_tensor(all_rows, device=q.device, dtype=torch.long)
            selected_q = qh[:, heads][:, :, row_tensor]
            selected_k = kh[:, heads]
            probabilities = torch.softmax(
                torch.matmul(selected_q, selected_k.transpose(-1, -2)).float()
                * (1.0 / math.sqrt(head_dim)),
                dim=-1,
            )
            for region in self.dynamic_regions:
                entry = self.dynamic_entries[region["name"]]
                for latent_index, rows in enumerate(region["frame_rows"]):
                    if not rows:
                        continue
                    positions = torch.as_tensor(
                        [row_positions[row] for row in rows], device=q.device, dtype=torch.long
                    )
                    key_start = latent_index * SPATIAL_TOKENS
                    key_end = key_start + SPATIAL_TOKENS
                    same_frame = probabilities[:, :, positions, key_start:key_end]
                    per_head = same_frame.mean(dim=(0, 2)).reshape(
                        len(heads), LATENT_HEIGHT, LATENT_WIDTH
                    )
                    entry["sum"][latent_index] += per_head.sum(dim=0).double().cpu()
                    entry["head_counts"][latent_index] += len(heads)
            return output

        def _write_dynamic_capture(self):
            names, phrases, means, counts, token_counts = [], [], [], [], []
            for region in self.dynamic_regions:
                name = region["name"]
                entry = self.dynamic_entries[name]
                head_counts = entry["head_counts"].numpy().astype(np.int32)
                visible = np.asarray([bool(rows) for rows in region["frame_rows"]])
                if np.any(head_counts[visible] != 100):
                    raise RuntimeError(
                        f"Expected 100 heads for {name}, got {head_counts.tolist()}"
                    )
                denominator = np.maximum(head_counts, 1)[:, None, None]
                means.append((entry["sum"].numpy() / denominator).astype(np.float32))
                counts.append(head_counts)
                token_counts.append(
                    np.asarray([len(rows) for rows in region["frame_rows"]], dtype=np.int32)
                )
                names.append(name)
                phrases.append(region["phrase"])
            self.dynamic_capture_root.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(
                self.dynamic_capture_root
                / f"step_{self.current_step:02d}__{self.current_cfg_branch}.npz",
                mean=np.stack(means),
                head_counts=np.stack(counts),
                query_token_counts=np.stack(token_counts),
                region_names=np.asarray(names),
                region_phrases=np.asarray(phrases),
                step=np.int32(self.current_step),
                cfg_branch=np.asarray(self.current_cfg_branch),
                seed=np.int32(args.seed),
                inference_steps=np.int32(args.inference_steps),
                timestep=np.float32(self.current_timestep_value),
                sigma=np.float32(self.current_sigma),
                query_protocol=np.asarray("cotracker_dynamic_Qt_to_same_frame_Kt"),
                attention_normalization=np.asarray("softmax_over_all_5824_keys_then_select_Kt"),
                selected_heads=np.int32(100),
                qkv_modified=np.bool_(False),
            )

    return DynamicObjectQueryCapture


def main() -> None:
    args = parse_args()
    os.environ["ATTENTION_NOISE_MODE"] = "probability_object_query_identity"
    os.environ["ATTENTION_NUM_INFERENCE_STEPS"] = str(args.inference_steps)
    os.environ["ATTENTION_EXTREME_COUNT"] = "100"
    os.environ["ATTENTION_GROUP_FILTER"] = "top"
    os.environ["ATTENTION_CFG_BRANCH_MODE"] = "both"
    os.environ["ATTENTION_MASK_LATENT_FRAMES"] = "13"
    os.environ["ATTENTION_MASK_CONTEXT_LATENT_FRAMES"] = "2"

    stage = launcher.import_path(
        "dynamic_object_query_common_stage",
        Path(__file__).with_name("run_pck_step_adaptive_attention_replacement_49f_worker.py"),
    )
    worker = stage.worker
    worker.base.select_heads = lambda ranking_pool, extreme_count: launcher.neighbor_heads(
        "pck32", ranking_pool, extreme_count
    )
    capture_class = build_dynamic_capture(worker.AdaptiveQKLogitNoise, args)
    worker.AdaptiveQKLogitNoise = capture_class
    worker.base.ExtremeHeadZeroer = capture_class
    worker.original_generate = launcher.seeded_generate(worker.base, args.seed)
    worker.base.LEGACY_ROOTS = ()
    run_args = argparse.Namespace(
        model="lora", input_json_list=args.input_json_list, output_root=args.output_root
    )
    sys.argv = launcher.base_argv(run_args)
    worker.load_capture_prompt_cases()
    worker.write_experiment_metadata()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "DYNAMIC_OBJECT_QUERY_COMMON.json").write_text(
        json.dumps(
            {
                "seed": args.seed,
                "inference_steps": args.inference_steps,
                "selected_heads": 100,
                "ranking": "Wan+LoRA PCK@32 Top100",
                "query": "CoTracker-derived object tokens at every latent frame",
                "key": "same latent frame spatial tokens",
                "attention": "post-softmax no intervention",
                "track_file": str(args.track_file),
                "capture_root": str(args.capture_root),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    worker.base.main()


if __name__ == "__main__":
    main()
