#!/usr/bin/env python3
"""Capture 10-step object-query attention rows or transplant them into 40-step inference."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

from AAA_my_test import run_attention_lora_seed_sweep_worker as launcher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("donor", "replacement"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--donor-root", type=Path, required=True)
    parser.add_argument("--mapping-csv", type=Path, required=True)
    parser.add_argument("--capture-root", type=Path, required=True)
    parser.add_argument("--input-json-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def build_transplanter(parent, mode: str, seed: int, donor_root: Path, mapping_csv: Path, capture_root: Path):
    class ReverseMatchedAttentionTransplanter(parent):
        def __init__(self, pipe, groups):
            super().__init__(pipe, groups)
            self.reverse_mode = mode
            self.reverse_seed = seed
            self.reverse_donor_root = donor_root
            self.reverse_capture_root = capture_root
            self.reverse_entries: dict[tuple[int, int, str], torch.Tensor] = {}
            self.reverse_capture_entries: dict[tuple[int, int, str], dict[str, object]] = {}
            self.reverse_donor_cache: dict[tuple[str, int], dict[tuple[int, int, str], torch.Tensor]] = {}
            self.reverse_mapping = {}
            if mode == "replacement":
                with mapping_csv.open(newline="") as handle:
                    for row in csv.DictReader(handle):
                        if int(row["seed"]) != seed:
                            continue
                        key = (
                            row["branch"],
                            row["object"],
                            int(row["step40"]),
                            int(row["block"]),
                            int(row["head"]),
                        )
                        self.reverse_mapping[key] = (int(row["best_step10"]), float(row["cosine"]))
                if len(self.reverse_mapping) != 2 * 2 * 40 * 100:
                    raise RuntimeError(f"Expected 16000 reverse mappings, got {len(self.reverse_mapping)}")

        def _wrapped_model_fn(self, *args, **kwargs):
            timestep = kwargs.get("timestep")
            self.current_step = self._scheduler_step(timestep) if timestep is not None else -1
            if self.current_step != self._cfg_step:
                self._cfg_step = self.current_step
                self._cfg_call_index = 0
            else:
                self._cfg_call_index += 1
            self.current_cfg_branch = "conditional" if self._cfg_call_index % 2 == 0 else "unconditional"
            self.group = (
                f"{self.adaptive_prefix}_step_{self.current_step:02d}"
                if self.adaptive_prefix is not None and self.current_step >= 0
                else None
            )
            self.reverse_entries = {}
            self.reverse_capture_entries = {}
            result = self.original_model_fn(*args, **kwargs)
            if self.group is not None:
                if self.reverse_mode == "donor":
                    self._write_reverse_donor()
                else:
                    self._write_reverse_capture()
            return result

        @staticmethod
        def _reshape_qkv(q, k, v):
            num_heads = int(q.shape[-1] // 128)
            head_dim = int(q.shape[-1] // num_heads)
            batch, sequence, _ = q.shape
            qh = q.reshape(batch, sequence, num_heads, head_dim).permute(0, 2, 1, 3)
            kh = k.reshape(batch, sequence, num_heads, head_dim).permute(0, 2, 1, 3)
            vh = v.reshape(batch, sequence, num_heads, head_dim).permute(0, 2, 1, 3)
            return qh, kh, vh, head_dim

        def _load_donor(self, branch: str, step10: int):
            cache_key = (branch, step10)
            if cache_key in self.reverse_donor_cache:
                return self.reverse_donor_cache[cache_key]
            path = self.reverse_donor_root / f"step_{step10:02d}__{branch}.npz"
            if not path.is_file():
                raise FileNotFoundError(f"Missing donor cache: {path}")
            loaded = {}
            with np.load(path, allow_pickle=False) as data:
                blocks = data["blocks"].astype(np.int64)
                heads = data["heads"].astype(np.int64)
                names = data["region_names"].astype(str)
                for region_index, name in enumerate(names):
                    values = torch.from_numpy(data[f"attention_{region_index}"].astype(np.float32))
                    for rank, (block, head) in enumerate(zip(blocks, heads)):
                        loaded[(int(block), int(head), str(name))] = values[rank]
            self.reverse_donor_cache[cache_key] = loaded
            return loaded

        def _attention(self, q, k, v, original, groups, block):
            heads = sorted(set(groups.get(self.group or "", ())))
            if not heads or self.current_step not in self.active_steps:
                return original(q, k, v)
            self.call_count += 1
            qh, kh, vh, head_dim = self._reshape_qkv(q, k, v)
            selected_q = qh[:, heads]
            selected_k = kh[:, heads]
            selected_v = vh[:, heads]
            key_t = selected_k.transpose(-1, -2)
            scale = 1.0 / math.sqrt(head_dim)

            if self.reverse_mode == "donor":
                output = original(q, k, v)
                for region in self.object_continuity_regions:
                    name = str(region["name"])
                    indices = torch.as_tensor(region["token_indices"], device=q.device, dtype=torch.long)
                    probabilities = torch.softmax(
                        torch.matmul(selected_q[:, :, indices], key_t).float() * scale,
                        dim=-1,
                    ).mean(dim=0).to(torch.float16).cpu()
                    for local_index, head in enumerate(heads):
                        self.reverse_entries[(int(block), int(head), name)] = probabilities[local_index]
                return output

            output = original(q, k, v)
            output_h = output.reshape(output.shape[0], output.shape[1], qh.shape[1], head_dim)
            branch = str(self.current_cfg_branch)
            for region in self.object_continuity_regions:
                name = str(region["name"])
                indices = torch.as_tensor(region["token_indices"], device=q.device, dtype=torch.long)
                target = torch.softmax(
                    torch.matmul(selected_q[:, :, indices], key_t).float() * scale,
                    dim=-1,
                )
                donor_rows = []
                matched_steps = []
                cosine_scores = []
                for head in heads:
                    mapping_key = (branch, name, int(self.current_step), int(block), int(head))
                    if mapping_key not in self.reverse_mapping:
                        raise KeyError(f"Missing reverse mapping {mapping_key}")
                    step10, cosine = self.reverse_mapping[mapping_key]
                    donor = self._load_donor(branch, step10).get((int(block), int(head), name))
                    if donor is None:
                        raise KeyError(f"Missing donor L{block:02d}/H{head:02d} {name} S{step10:02d} {branch}")
                    donor_rows.append(donor)
                    matched_steps.append(step10)
                    cosine_scores.append(cosine)
                donor = torch.stack(donor_rows, dim=0).to(device=q.device, dtype=torch.float32)
                if donor.shape[-2:] != target.shape[-2:]:
                    raise RuntimeError(f"Donor shape {tuple(donor.shape)} != target {tuple(target.shape)}")
                donor = donor.clamp_min(0)
                donor = donor / donor.sum(dim=-1, keepdim=True).clamp_min(1e-12)
                donor_batch = donor.unsqueeze(0).expand(target.shape[0], -1, -1, -1)
                replacement = torch.matmul(donor_batch.to(selected_v.dtype), selected_v)
                for local_index, head in enumerate(heads):
                    output_h[:, indices, head, :] = replacement[:, local_index].to(output_h.dtype)
                    self.reverse_capture_entries[(int(block), int(head), name)] = {
                        "before": target[:, local_index].mean(dim=(0, 1)).to(torch.float16).cpu(),
                        "donor": donor[local_index].mean(dim=0).to(torch.float16).cpu(),
                        "step10": matched_steps[local_index],
                        "cosine": cosine_scores[local_index],
                    }
            return output_h.reshape_as(output)

        def _complete_head_ids(self, entries):
            names = [str(region["name"]) for region in self.object_continuity_regions]
            ids = sorted({(key[0], key[1]) for key in entries})
            complete = [head_id for head_id in ids if all((head_id[0], head_id[1], name) in entries for name in names)]
            if len(complete) != 100:
                raise RuntimeError(f"Expected 100 complete heads, got {len(complete)}")
            return names, complete

        def _write_reverse_donor(self):
            names, head_ids = self._complete_head_ids(self.reverse_entries)
            self.reverse_donor_root.mkdir(parents=True, exist_ok=True)
            payload = {
                "blocks": np.asarray([item[0] for item in head_ids], dtype=np.int16),
                "heads": np.asarray([item[1] for item in head_ids], dtype=np.int16),
                "region_names": np.asarray(names),
                "step": np.asarray(self.current_step, dtype=np.int16),
                "branch": np.asarray(self.current_cfg_branch),
                "protocol": np.asarray("per_query_row_post_softmax_frozen_donor"),
            }
            for region_index, name in enumerate(names):
                payload[f"attention_{region_index}"] = np.stack(
                    [self.reverse_entries[(block, head, name)].numpy() for block, head in head_ids]
                )
            np.savez(
                self.reverse_donor_root / f"step_{self.current_step:02d}__{self.current_cfg_branch}.npz",
                **payload,
            )

        def _write_reverse_capture(self):
            names, head_ids = self._complete_head_ids(self.reverse_capture_entries)
            self.reverse_capture_root.mkdir(parents=True, exist_ok=True)
            before = []
            donor = []
            matched = []
            cosine = []
            for block, head in head_ids:
                before.append(np.stack([self.reverse_capture_entries[(block, head, name)]["before"].numpy() for name in names]))
                donor.append(np.stack([self.reverse_capture_entries[(block, head, name)]["donor"].numpy() for name in names]))
                matched.append([self.reverse_capture_entries[(block, head, name)]["step10"] for name in names])
                cosine.append([self.reverse_capture_entries[(block, head, name)]["cosine"] for name in names])
            before_array = np.stack(before)
            donor_array = np.stack(donor)
            np.savez_compressed(
                self.reverse_capture_root / f"step_{self.current_step:02d}__{self.current_cfg_branch}.npz",
                before=before_array,
                donor=donor_array,
                after=donor_array,
                delta=donor_array - before_array,
                blocks=np.asarray([item[0] for item in head_ids], dtype=np.int16),
                heads=np.asarray([item[1] for item in head_ids], dtype=np.int16),
                region_names=np.asarray(names),
                matched_step10=np.asarray(matched, dtype=np.int16),
                cosine=np.asarray(cosine, dtype=np.float32),
                step40=np.asarray(self.current_step, dtype=np.int16),
                branch=np.asarray(self.current_cfg_branch),
                qkv_modified=np.asarray(False),
                intervention=np.asarray("post_softmax_object_query_row_transplant_A10_at_V40"),
            )

    return ReverseMatchedAttentionTransplanter


def main() -> None:
    args = parse_args()
    os.environ["ATTENTION_NOISE_MODE"] = "probability_object_query_identity"
    os.environ["ATTENTION_NUM_INFERENCE_STEPS"] = "10" if args.mode == "donor" else "40"
    os.environ["ATTENTION_EXTREME_COUNT"] = "100"
    os.environ["ATTENTION_GROUP_FILTER"] = "top"
    os.environ["ATTENTION_CFG_BRANCH_MODE"] = "both"
    os.environ["ATTENTION_MASK_LATENT_FRAMES"] = "13"
    os.environ["ATTENTION_MASK_CONTEXT_LATENT_FRAMES"] = "2"

    stage = launcher.import_path(
        "reverse_attention_transplant_stage",
        Path(__file__).with_name("run_pck_step_adaptive_attention_replacement_49f_worker.py"),
    )
    worker = stage.worker
    worker.base.select_heads = lambda ranking_pool, extreme_count: launcher.neighbor_heads(
        "pck32", ranking_pool, extreme_count
    )
    transplanter = build_transplanter(
        worker.AdaptiveQKLogitNoise,
        args.mode,
        args.seed,
        args.donor_root,
        args.mapping_csv,
        args.capture_root,
    )
    worker.AdaptiveQKLogitNoise = transplanter
    worker.base.ExtremeHeadZeroer = transplanter
    worker.original_generate = launcher.seeded_generate(worker.base, args.seed)
    worker.base.LEGACY_ROOTS = ()
    run_args = argparse.Namespace(
        model="lora",
        input_json_list=args.input_json_list,
        output_root=args.output_root,
    )
    sys.argv = launcher.base_argv(run_args)
    worker.load_capture_prompt_cases()
    worker.write_experiment_metadata()
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "REVERSE_ATTENTION_TRANSPLANT.json").write_text(
        json.dumps(
            {
                "mode": args.mode,
                "seed": args.seed,
                "inference_steps": 10 if args.mode == "donor" else 40,
                "selected_heads": 100,
                "cfg_branches": ["conditional", "unconditional"],
                "objects": ["object_A", "object_B"],
                "qkv_modified": False,
                "replacement": "A40[object_query_rows] <- matched A10 donor; output = A_donor @ V40",
                "mapping_csv": str(args.mapping_csv),
                "donor_root": str(args.donor_root),
                "capture_root": str(args.capture_root),
            },
            indent=2,
        )
        + "\n"
    )
    worker.base.main()


if __name__ == "__main__":
    main()
