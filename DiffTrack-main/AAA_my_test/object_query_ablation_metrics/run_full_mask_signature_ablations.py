#!/usr/bin/env python3
"""Run full-SAM2-mask signature M1/M2/M3 ablations on frozen baselines."""

from __future__ import annotations

import argparse
import gc
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[2]
CODE_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
for path in (ROOT, CODE_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from code_vjepa_vggt.AAAinfer.utils.wanti2v_runtime import (  # noqa: E402
    build_wan_ti2v_pipeline,
    save_video_np,
)
from AAA_my_test.legacy_ti2v_firstlatent_physiciq67_common import CASES  # noqa: E402
from AAA_my_test.object_query_ablation_metrics.full_mask_signature_regions import (  # noqa: E402
    LATENT_GRID,
    SignaturePartition,
    build_signature_partition,
    torch_signature_groups,
    unpack_mask_cache,
)
from AAA_my_test.object_query_ablation_metrics.training_free_m1_control.run_multi_object_guidance_search import (  # noqa: E402
    apply_grouped_m1_ablation,
)
from AAA_my_test.run_legacy_ti2v_firstlatent_physiciq67_attention_zero_ablations import (  # noqa: E402
    AttentionMatrixAblator,
    build_args,
    generate_video,
    generation_inputs,
)
from AAA_my_test.run_legacy_ti2v_temporal_object_tube_ablations import (  # noqa: E402
    atomic_npz,
    head_scope_counts,
    selected_head_entries,
    sha256_file,
)


EXPERIMENT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1"
)
DEFAULT_OUTPUT_ROOT = EXPERIMENT_ROOT / "full_mask_signature_pilot_v1"
DEFAULT_RANKING = EXPERIMENT_ROOT / "head_scopes_latest3350_with_random100.json"
SOURCE_MANIFESTS = (
    EXPERIMENT_ROOT / "stage4_runtime/stage4_manifest_001460.json",
    Path(
        "/data/gaoya/agent-data/outputs/"
        "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
        "attention_zero_seed47326/cases_other10_6seeds_latest3350.json"
    ),
)
DEFAULT_CASES = (
    "0613pybullet_sample_001460_w002",
    "physicIQ_009_Fluid_Dynamics_0131_perspective-center_trimmed-paint-on-glass",
    "physicIQ_026_Solid_Mechanics_0005_perspective-center_trimmed-ball-behind-rotating-paper",
)
MASK_MODES = ("self_only", "incoming_only", "outgoing_only")
HEAD_SCOPES = ("top100", "bottom100")
MODE_IDS = {"self_only": "M1", "incoming_only": "M2", "outgoing_only": "M3"}
PROTOCOL = "full_sam2_mask_signature_attention_ablation_v1"


class FullMaskSignatureAblator(AttentionMatrixAblator):
    """Use a frozen full-mask partition instead of sparse point/tube tokens."""

    def __init__(
        self,
        pipe: Any,
        entries: list[dict],
        partition: SignaturePartition,
        mask_mode: str,
        *,
        record_dose: bool,
        group_batch_size: int = 1,
    ) -> None:
        if mask_mode not in MASK_MODES:
            raise ValueError(f"unsupported mode: {mask_mode}")
        super().__init__(
            pipe,
            entries,
            np.zeros((1, 2), dtype=np.float32),
            {},
            (704, 1280),
            "all_objects",
            mask_mode,
            None,
            record_dose=record_dose,
        )
        self.partition = partition
        self.group_batch_size = int(group_batch_size)
        self.deleted_pair_count_per_head: int | None = None

    def _rows(self, device: torch.device) -> torch.Tensor:
        if self.current_grid != LATENT_GRID:
            raise RuntimeError(
                f"runtime latent grid {self.current_grid} != frozen mask grid {LATENT_GRID}"
            )
        rows = torch.as_tensor(
            self.partition.union_rows, device=device, dtype=torch.long
        )
        values = list(self.partition.union_rows)
        if self.query_token_indices is None:
            self.query_token_indices = values
        elif self.query_token_indices != values:
            raise RuntimeError("full-mask token mapping changed during generation")
        return rows

    def _attention(self, q, k, v, original, block: int):
        heads = self.by_block.get(block, ())
        if not self.active or not heads or self.mask_mode != "self_only":
            return super()._attention(q, k, v, original, block)
        num_heads = int(q.shape[-1] // 128)
        if num_heads <= 0 or q.shape[-1] % num_heads:
            raise RuntimeError(f"query width {q.shape[-1]} is not head-aligned")
        # Validate the runtime grid and register the frozen union for the base
        # audit.  The grouped M1 path otherwise has no need to call _rows().
        self._rows(q.device)
        groups, _ = torch_signature_groups(self.partition, q.device)
        output = original(q, k, v)
        recorder = None
        if self.record_dose:
            recorder = lambda target, removed, original_output, mass: self._record_removed_dose(
                block, heads, target, removed, original_output, mass
            )
        calls, affected_rows, deleted_pairs = apply_grouped_m1_ablation(
            output=output,
            q=q,
            k=k,
            v=v,
            original=original,
            heads=heads,
            num_heads=num_heads,
            groups=groups,
            group_batch_size=self.group_batch_size,
            dose_recorder=recorder,
        )
        if self.deleted_pair_count_per_head is None:
            self.deleted_pair_count_per_head = deleted_pairs
        elif self.deleted_pair_count_per_head != deleted_pairs:
            raise RuntimeError("signature M1 deleted-pair count changed")
        self.auxiliary_attention_calls += calls
        self.modified_forward_calls += 1
        self.modified_head_events += len(heads)
        self.affected_query_vectors += int(output.shape[0]) * affected_rows * len(heads)
        return output

    def signature_audit(self) -> dict[str, Any]:
        return {
            **self.partition.audit(),
            "M1_deleted_pair_count_per_head": self.deleted_pair_count_per_head,
            "M1_preserved_cross_signature_pairs": True,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", nargs="+", default=list(DEFAULT_CASES))
    parser.add_argument("--seed", type=int, default=47326)
    parser.add_argument("--head-scopes", nargs="+", choices=HEAD_SCOPES, default=list(HEAD_SCOPES))
    parser.add_argument("--mask-modes", nargs="+", choices=MASK_MODES, default=list(MASK_MODES))
    parser.add_argument("--head-ranking-path", type=Path, default=DEFAULT_RANKING)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--task-index", type=int, default=None)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--record-dose", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def samples_by_key() -> dict[tuple[str, int], dict]:
    result: dict[tuple[str, int], dict] = {}
    for path in SOURCE_MANIFESTS:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for sample in manifest["samples"]:
            result[(str(sample["case"]), int(sample["seed"]))] = sample
    return result


def mask_cache_for(case: str, seed: int) -> Path:
    candidates = [
        EXPERIMENT_ROOT / "stage4_metrics/head_scope_trajectory" / case / f"seed_{seed:05d}/object_survival/masks/baseline.npz",
        EXPERIMENT_ROOT / "stage3_metrics/head_scope_trajectory" / case / f"seed_{seed:05d}/object_survival/masks/baseline.npz",
    ]
    found = [path for path in candidates if path.is_file()]
    if len(found) != 1:
        raise RuntimeError(f"expected exactly one frozen baseline mask cache for {case}/{seed}: {found}")
    return found[0]


def task_directory(root: Path, task: dict) -> Path:
    return (
        root
        / task["case"]
        / f"seed_{int(task['seed']):05d}"
        / f"fullmask_signature__{task['mask_mode']}__{task['head_scope']}_s039r3350"
    )


def build_tasks(args: argparse.Namespace, samples: dict[tuple[str, int], dict]) -> list[tuple[dict, dict]]:
    tasks = []
    for case in args.cases:
        sample = samples.get((case, args.seed))
        if sample is None:
            raise KeyError(f"sample absent from source manifests: {case}/seed_{args.seed:05d}")
        for scope in args.head_scopes:
            for mode in args.mask_modes:
                tasks.append((sample, {"case": case, "seed": args.seed, "head_scope": scope, "mask_mode": mode}))
    if args.task_index is not None:
        if not 0 <= args.task_index < len(tasks):
            raise ValueError(f"task-index must be in [0,{len(tasks)})")
        return [tasks[args.task_index]]
    return tasks[args.worker_id :: args.num_workers]


def process(
    pipe: Any,
    sample: dict,
    task: dict,
    ranking: dict,
    args: argparse.Namespace,
) -> None:
    output = task_directory(args.output_root, task)
    required = ("complete.json", "manifest.json", "generated.mp4")
    if args.record_dose:
        required += ("dose_metrics.npz",)
    if not args.overwrite and all((output / name).is_file() for name in required):
        print(f"skip {output.relative_to(args.output_root)}", flush=True)
        return
    output.mkdir(parents=True, exist_ok=True)
    for name in ("complete.json", "error.txt"):
        (output / name).unlink(missing_ok=True)

    baseline = Path(str(sample["baseline_video"]))
    if not baseline.is_file():
        raise FileNotFoundError(baseline)
    mask_path = mask_cache_for(str(task["case"]), int(task["seed"]))
    masks = unpack_mask_cache(mask_path, expected_video=baseline)
    names = tuple(
        str(row["region_name"])
        for row in sample["regions"]
        if row.get("region_type") == "object"
    )
    partition = build_signature_partition(masks, names)
    entries = selected_head_entries(
        list(ranking["entries"]), str(task["head_scope"]), ranking.get("head_scopes")
    )
    case_lookup = {case.key: case for case in CASES}
    json_path, _, payload, generation_args, image = generation_inputs(
        sample, case_lookup, int(task["seed"])
    )
    ablator = FullMaskSignatureAblator(
        pipe.pipe,
        entries,
        partition,
        str(task["mask_mode"]),
        record_dose=bool(args.record_dose),
    )
    ablator.install()
    try:
        video = generate_video(pipe, payload, generation_args, image, int(task["seed"]))
    finally:
        ablator.remove()
    audit = ablator.audit()
    if args.record_dose:
        atomic_npz(output / "dose_metrics.npz", **ablator.dose_arrays())
    temporary = output / "generated.tmp.mp4"
    save_video_np(video, temporary, fps=30)
    temporary.replace(output / "generated.mp4")

    mode = str(task["mask_mode"])
    exact = {
        "self_only": "sum over nonzero signatures S of A[R_S,R_S]V[R_S]",
        "incoming_only": "A[R_all,C]V[C], where C=Omega\\R_all",
        "outgoing_only": "A[C,R_all]V[R_all], where C=Omega\\R_all",
    }[mode]
    metadata = {
        **task,
        "variant_id": output.name,
        "protocol": PROTOCOL,
        "flow_id": MODE_IDS[mode],
        "attention_definition": "A=softmax(QK^T/sqrt(d)); Y=A@V",
        "exact_deleted_contribution": exact,
        "post_mask_renormalization": False,
        "selected_token_definition": "full frozen baseline SAM2 masks at F00,F04,...,F48; a latent cell belongs to every intersecting object",
        "M1_shared_token_policy": "each exact object-membership signature is a separate self block; cross-signature communication is preserved",
        "input_json": str(json_path),
        "baseline_video": str(baseline),
        "mask_cache": str(mask_path),
        "mask_cache_sha256": sha256_file(mask_path),
        "head_ranking": str(args.head_ranking_path),
        "head_ranking_sha256": sha256_file(args.head_ranking_path),
        "selected_head_count": len(entries),
        "selected_entries": entries,
        "signature_partition": ablator.signature_audit(),
        "runtime_audit": audit,
        "dose_metrics": str(output / "dose_metrics.npz") if args.record_dose else None,
    }
    (output / "manifest.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "complete.json").write_text(
        json.dumps(
            {
                "case": task["case"],
                "seed": task["seed"],
                "variant_id": output.name,
                "modified_head_events": audit["modified_head_events"],
                "signature_count": len(partition.signature_rows),
                "union_token_count": len(partition.union_rows),
            },
            ensure_ascii=False,
            indent=2,
        ) + "\n",
        encoding="utf-8",
    )
    del video


def main() -> None:
    args = parse_args()
    if not 0 <= args.worker_id < args.num_workers:
        raise ValueError("worker-id must be in [0,num-workers)")
    ranking = json.loads(args.head_ranking_path.read_text(encoding="utf-8"))
    entries = list(ranking.get("entries") or [])
    pairs = {(int(row["block"]), int(row["head"])) for row in entries}
    if len(entries) != 720 or pairs != {(b, h) for b in range(30) for h in range(24)}:
        raise RuntimeError("ranking is not a complete unique 30x24 layer-head ranking")
    unknown = set(args.head_scopes) - set(head_scope_counts(ranking))
    if unknown:
        raise ValueError(f"unknown head scopes: {sorted(unknown)}")
    tasks = build_tasks(args, samples_by_key())
    preview = [
        {**task, "output": str(task_directory(args.output_root, task))}
        for _, task in tasks
    ]
    if args.dry_run:
        print(json.dumps({"task_count": len(tasks), "tasks": preview}, ensure_ascii=False, indent=2))
        return
    if not tasks:
        return
    pipe = build_wan_ti2v_pipeline(build_args(int(tasks[0][1]["seed"])))
    for index, (sample, task) in enumerate(tasks, 1):
        output = task_directory(args.output_root, task)
        print(f"[{index}/{len(tasks)}] start {output.relative_to(args.output_root)}", flush=True)
        try:
            process(pipe, sample, task, ranking, args)
        except Exception:
            output.mkdir(parents=True, exist_ok=True)
            (output / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
            raise
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[{index}/{len(tasks)}] complete {output.relative_to(args.output_root)}", flush=True)


if __name__ == "__main__":
    main()
