#!/usr/bin/env python3
"""Deterministic train-subset flow-loss evaluation for every experiment checkpoint.

The cross-method score is the common, timestep-weighted flow-matching MSE
(`train/loss_main`). Auxiliary objectives such as xSSC/V-JEPA are deliberately
excluded from the cross-method ranking because they are not shared by all runs.
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import random
import shlex
import statistics
import sys
import time
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch


HERE = Path(__file__).resolve().parent
EXPERIMENT_ROOT = HERE.parent
TRAIN_XSSC_ROOT = EXPERIMENT_ROOT.parent
PACKAGE_ROOT = EXPERIMENT_ROOT.parents[2]
DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
for path in (HERE, EXPERIMENT_ROOT, TRAIN_XSSC_ROOT, PACKAGE_ROOT, DIFFSYNTH_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import train_xssc_object_self_attn_lora as core
import train_xssc_object_self_attn_lora_physrvg_dit as physrvg_train
import train_xssc_object_self_attn_lora_slot_dedup as slot_dedup_train
import xssc_loss_project.train_full_sa_object_slot_dedup_xssc_loss as slot_dedup_xssc_train
import xssc_loss_project.train_xssc_object_self_attn_lora_xssc_loss as xssc_train
import vjepa_loss_project.train_xssc_object_self_attn_lora_vjepa_loss as vjepa_train
from code_vjepa_vggt.data.mixed_replay_no_gt_box_dataset import (
    KubricReplayNoGTBoxDataset,
    OpenVidNoGTBoxDataset,
)
from code_vjepa_vggt.data.pybullet0713_no_gt_box_dataset import (
    PyBullet0713NoGTBoxDataset,
)


DEFAULT_INVENTORY = Path(
    "/data/gaoya/agent-data/outputs/test5_step500_all_methods_train_cases/"
    "all_checkpoint_inventory.json"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/train_subset_val_loss_seed42"
)
DEFAULT_REFERENCE_CONFIG = Path(
    "/data/gaoya/agent-data/checkpoints/xssc_feature_loss/"
    "full_sa_no_object_xssc_loss_dinov3_movic_step50000/formal_gpu01/"
    "resolved_experiment_config.json"
)
SOURCE_ORDER = ("pybullet", "kubric", "openvid")
SOURCE_LABELS = {
    "pybullet": "PyBullet train",
    "kubric": "Kubric train",
    "openvid": "OpenVid train",
}


class AcceleratorStub:
    def __init__(self, device: torch.device) -> None:
        self.device = device
        self.is_main_process = True

    @staticmethod
    def print(*args, **kwargs) -> None:
        print(*args, **kwargs)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def seed_all(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def stable_case_seed(global_seed: int, source: str, source_index: int) -> int:
    digest = hashlib.sha256(
        f"{global_seed}|{source}|{source_index}".encode("utf-8")
    ).digest()
    return int(global_seed) + int.from_bytes(digest[:4], "big") % 1_000_000_000


def find_manifest(checkpoint: Path) -> Path:
    for parent in checkpoint.resolve().parents:
        candidate = parent / "resolved_experiment_config.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No resolved_experiment_config.json above {checkpoint}")


def load_resolved(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    config = payload.get("resolved_config")
    if not isinstance(config, dict):
        raise ValueError(f"Missing resolved_config in {path}")
    return config


def launch_argv(manifest_path: Path) -> list[str]:
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    summary = payload.get("launch_summary", {})
    command = summary.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"Missing launch_summary.command in {manifest_path}")
    argv = shlex.split(command)
    script_index = next(
        (index for index, value in enumerate(argv) if value.endswith(".py")),
        None,
    )
    if script_index is None:
        raise ValueError(f"Training script not found in launch command: {manifest_path}")
    return argv[script_index + 1 :]


def model_kind(config: dict[str, Any]) -> str:
    script = str(config.get("experiment", {}).get("script", ""))
    if "vjepa_loss" in script or "vjepa" in str(config.get("experiment", {}).get("name", "")).lower():
        return "vjepa"
    initialization = str(config.get("initialization", {}).get("type", "openvid_lora"))
    if initialization == "physrvg_dit":
        return "physrvg"
    xssc_enabled = bool(config.get("xssc_loss_enabled", False)) or bool(
        config.get("xssc_loss")
    )
    dedup = str(config.get("conditioning", {}).get("slot_dedup", {}).get("mode", "none"))
    if dedup != "none":
        return "slot_dedup_xssc" if xssc_enabled else "slot_dedup"
    # Resolved xSSC configs record the objective in the top-level flags even
    # when the experiment name does not contain a script suffix. Recognize it
    # after slot-dedup so the specialized combined implementation is selected.
    if xssc_enabled:
        return "xssc"
    return "core"


def parse_model_args(manifest_path: Path) -> tuple[argparse.Namespace, dict[str, Any], str]:
    config = load_resolved(manifest_path)
    kind = model_kind(config)
    parser = {
        "core": core.build_parser,
        "physrvg": physrvg_train.build_parser,
        "slot_dedup": slot_dedup_train.build_parser,
        "slot_dedup_xssc": slot_dedup_xssc_train.build_parser,
        "xssc": xssc_train.build_parser,
        "vjepa": vjepa_train.build_parser,
    }[kind]()
    args, _unknown = parser.parse_known_args(launch_argv(manifest_path))
    if kind == "physrvg" and not getattr(args, "physrvg_dit_checkpoint", None):
        args.physrvg_dit_checkpoint = config["paths"]["physrvg_dit_checkpoint"]
    args.train_batch_size = 1
    args.no_context_ratio = 0.0
    args.xssc_slot_track_dropout = 0.0
    args.xssc_filter_empty_amg = False
    args.xssc_empty_amg_max_resample_attempts = 0
    args.use_gradient_checkpointing = False
    args.use_gradient_checkpointing_offload = False
    args.experiment_seed = 42
    return core.tvn.prepare_args(args), config, kind


def build_source_datasets(config: dict[str, Any]) -> dict[str, Any]:
    paths = config["paths"]
    data = config["data"]
    model = config["model"]
    resolution = (int(model["height"]), int(model["width"]))
    num_frames = int(model["num_frames"])
    context_frames = int(model["fixed_num_context_frames"])
    return {
        "pybullet": PyBullet0713NoGTBoxDataset(
            root=paths["pybullet_root"],
            split="train",
            resolution=resolution,
            num_frames=num_frames,
            num_context_frames=context_frames,
            sampling_strategy=str(data["pybullet_sampling_strategy"]),
        ),
        "kubric": KubricReplayNoGTBoxDataset(
            root=paths["kubric_root"],
            split="train",
            resolution=resolution,
            num_frames=num_frames,
            num_context_frames=context_frames,
            index_num_frames=int(data["kubric_replay_index_num_frames"]),
            index_num_context_frames=int(data["kubric_replay_index_num_context_frames"]),
            sampling_strategy=str(data["kubric_sampling_strategy"]),
            seed=42,
            cache_root=data["kubric_cache_root"],
        ),
        "openvid": OpenVidNoGTBoxDataset(
            root=paths["openvid_root"],
            resolution=resolution,
            num_frames=num_frames,
            num_context_frames=context_frames,
        ),
    }


def source_record_key(dataset: Any, source: str, index: int) -> str:
    if source == "openvid":
        return f"openvid/row_{index:06d}"
    record = dataset.samples[index]
    return str(record.key)


def ensure_case_manifest(
    output_root: Path,
    reference_config: Path,
    *,
    seed: int,
    cases_per_source: int,
) -> dict[str, Any]:
    path = output_root / "case_manifest.json"
    if path.is_file():
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload["seed"] != seed or payload["cases_per_source"] != cases_per_source:
            raise ValueError(
                f"Existing manifest settings differ: {path}; refusing to change the fixed sample set"
            )
        return payload
    config = load_resolved(reference_config)
    datasets = build_source_datasets(config)
    cases: list[dict[str, Any]] = []
    lengths: dict[str, int] = {}
    for source_id, source in enumerate(SOURCE_ORDER):
        dataset = datasets[source]
        lengths[source] = len(dataset)
        if len(dataset) < cases_per_source:
            raise ValueError(f"{source} has only {len(dataset)} cases, need {cases_per_source}")
        rng = random.Random(seed + 10_000 * (source_id + 1))
        indices = sorted(rng.sample(range(len(dataset)), cases_per_source))
        for index in indices:
            case_seed = stable_case_seed(seed, source, index)
            cases.append(
                {
                    "case_id": f"{source}_{index:07d}",
                    "source": source,
                    "source_index": index,
                    "sample_key": source_record_key(dataset, source, index),
                    "case_seed": case_seed,
                }
            )
    payload = {
        "schema_version": 1,
        "created_utc": utc_now(),
        "seed": seed,
        "sampling": "random_without_replacement_per_source",
        "cases_per_source": cases_per_source,
        "total_cases": len(cases),
        "source_lengths": lengths,
        "reference_config": str(reference_config),
        "cases": cases,
    }
    atomic_json(path, payload)
    return payload


def inventory_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"No entries in {path}")
    return entries


def build_model(
    manifest_path: Path,
    device: torch.device,
) -> tuple[torch.nn.Module, argparse.Namespace, dict[str, Any], str]:
    args, config, kind = parse_model_args(manifest_path)
    accelerator = AcceleratorStub(device)
    implementation = {
        "core": core,
        "physrvg": physrvg_train,
        "slot_dedup": slot_dedup_train,
        "slot_dedup_xssc": slot_dedup_xssc_train,
        "xssc": xssc_train,
        "vjepa": vjepa_train,
    }[kind]
    model = implementation.build_model(args, accelerator)
    model.to(device)
    model.pipe.to(device=device, dtype=model.pipe.torch_dtype)
    model.eval()
    return model, args, config, kind


def load_checkpoint(model: torch.nn.Module, checkpoint: Path) -> dict[str, Any]:
    checkpoint_file = core.tvn._resolve_checkpoint_file(checkpoint)
    identity = None
    if model.self_attn_adaptation_mode in core.HEAD_SELECTIVE_ADAPTATION_MODES:
        identity = core.validate_head_selection_resume_checkpoint(model, checkpoint_file)
    info = core.tvn._load_filtered_checkpoint_into_model(
        model,
        checkpoint_file,
        include_prefixes=("slot_norm.", "slot_projector.", "time_embedding."),
        include_substrings=(".object_cross_attn.", ".object_gate", ".self_attn."),
    )
    expected = sum(1 for _, parameter in model.named_parameters() if parameter.requires_grad)
    if info["loaded_count"] != expected or info["skipped_shape_mismatch"]:
        raise RuntimeError(
            f"Incomplete checkpoint load {checkpoint}: "
            f"loaded={info['loaded_count']}/{expected}, "
            f"shape_mismatch={len(info['skipped_shape_mismatch'])}"
        )
    info["head_identity"] = identity
    return info


def recursively_cpu(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, dict):
        return {key: recursively_cpu(child) for key, child in value.items()}
    if isinstance(value, list):
        return [recursively_cpu(child) for child in value]
    if isinstance(value, tuple):
        return tuple(recursively_cpu(child) for child in value)
    return value


def prune_prepared_inputs(prepared: tuple[dict, dict, dict]) -> tuple[dict, dict, dict]:
    shared, positive, negative = recursively_cpu(prepared)
    shared.pop("input_video", None)
    shared.pop("context_video", None)
    raw = shared.get("raw_sample")
    if isinstance(raw, dict):
        raw.pop("video", None)
    return shared, positive, negative


def cache_path(output_root: Path, case: dict[str, Any]) -> Path:
    return output_root / "prepared_inputs" / case["source"] / f"{case['case_id']}.pt"


def prepare_inputs(
    model: torch.nn.Module,
    dataset: Any,
    case: dict[str, Any],
    output_root: Path,
) -> tuple[dict, dict, dict]:
    path = cache_path(output_root, case)
    if path.is_file() and path.stat().st_size > 0:
        return torch.load(path, map_location="cpu", weights_only=False)
    seed_all(int(case["case_seed"]))
    sample = dataset[int(case["source_index"])]
    actual_key = str(sample.get("metadata", {}).get("sample_key", case["sample_key"]))
    if actual_key != case["sample_key"]:
        raise RuntimeError(
            f"Dataset retry changed case identity: expected {case['sample_key']}, got {actual_key}"
        )
    with torch.inference_mode():
        prepared = model._prepare_pipeline_sample(sample)
    prepared = prune_prepared_inputs(prepared)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    torch.save(prepared, temporary)
    os.replace(temporary, path)
    return prepared


def transfer_prepared(
    model: torch.nn.Module,
    prepared: tuple[dict, dict, dict],
) -> tuple[dict, dict, dict]:
    return model.transfer_data_to_device(
        prepared, model.pipe.device, model.pipe.torch_dtype
    )


def evaluate_prepared(
    model: torch.nn.Module,
    prepared_cpu: tuple[dict, dict, dict],
    case_seed: int,
) -> tuple[float, dict[str, float]]:
    seed_all(case_seed)
    shared, positive, negative = transfer_prepared(model, prepared_cpu)
    with torch.inference_mode():
        # Auxiliary-loss subclasses override this path even when the object
        # branch is disabled. Calling task_to_loss directly would silently
        # drop their xSSC/V-JEPA validation terms.
        loss, metrics = model._compute_object_losses(model.pipe, shared, positive)
    loss_main = float(metrics.get("train/loss_main", loss.detach().item()))
    numeric = {
        key: float(value)
        for key, value in metrics.items()
        if isinstance(value, (int, float)) and np.isfinite(float(value))
    }
    if not np.isfinite(loss_main):
        raise FloatingPointError(f"Non-finite loss_main={loss_main}")
    del shared, positive, negative, loss
    return loss_main, numeric


def result_path(output_root: Path, entry: dict[str, Any]) -> Path:
    return (
        output_root
        / "results"
        / entry["method_key"]
        / f"step-{int(entry['step']):06d}.json"
    )


def evaluate_entry(
    *,
    model: torch.nn.Module,
    datasets: dict[str, Any],
    case_manifest: dict[str, Any],
    entry: dict[str, Any],
    output_root: Path,
    repeat_check: bool,
    load_info: dict[str, Any],
    manifest_path: Path,
    kind: str,
) -> dict[str, Any]:
    path = result_path(output_root, entry)
    if path.is_file():
        result = json.loads(path.read_text(encoding="utf-8"))
    else:
        result = {
            "schema_version": 1,
            "entry_id": entry["entry_id"],
            "method_key": entry["method_key"],
            "method_label": entry["method_label"],
            "step": int(entry["step"]),
            "checkpoint": entry["checkpoint"],
            "experiment_manifest": str(manifest_path),
            "model_kind": kind,
            "metric": "common_timestep_weighted_flow_matching_mse",
            "seed": int(case_manifest["seed"]),
            "load_info": load_info,
            "cases": [],
            "state": "running",
            "started_utc": utc_now(),
        }
    completed = {record["case_id"] for record in result["cases"]}
    for position, case in enumerate(case_manifest["cases"], start=1):
        if case["case_id"] in completed:
            continue
        start = time.monotonic()
        prepared = prepare_inputs(
            model,
            datasets[case["source"]],
            case,
            output_root,
        )
        loss_main, metrics = evaluate_prepared(model, prepared, int(case["case_seed"]))
        if repeat_check and not result["cases"]:
            repeated, _ = evaluate_prepared(model, prepared, int(case["case_seed"]))
            if abs(loss_main - repeated) > max(1e-7, abs(loss_main) * 1e-6):
                raise RuntimeError(
                    f"Determinism check failed: first={loss_main}, repeat={repeated}"
                )
            result["determinism_check"] = {
                "case_id": case["case_id"],
                "first": loss_main,
                "repeat": repeated,
                "absolute_difference": abs(loss_main - repeated),
                "passed": True,
            }
        result["cases"].append(
            {
                **case,
                "loss_main": loss_main,
                "metrics": metrics,
                "seconds": time.monotonic() - start,
            }
        )
        result["updated_utc"] = utc_now()
        atomic_json(path, result)
        print(
            f"[{entry['entry_id']}] {position:03d}/{len(case_manifest['cases'])} "
            f"{case['case_id']} loss={loss_main:.8f}",
            flush=True,
        )
    result["state"] = "complete"
    result["completed_utc"] = utc_now()
    atomic_json(path, result)
    return result


def selected_methods(
    entries: list[dict[str, Any]],
    worker_id: int,
    num_workers: int,
) -> list[str]:
    methods: list[str] = []
    for entry in entries:
        if entry["method_key"] not in methods:
            methods.append(entry["method_key"])
    counts = {
        method: sum(entry["method_key"] == method for entry in entries)
        for method in methods
    }
    assignments: list[list[str]] = [[] for _ in range(num_workers)]
    loads = [0 for _ in range(num_workers)]
    for method in sorted(methods, key=lambda item: (-counts[item], methods.index(item))):
        target = min(range(num_workers), key=lambda index: (loads[index], index))
        assignments[target].append(method)
        loads[target] += counts[method]
    return assignments[worker_id]


def run_worker(args: argparse.Namespace) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if "4" in [item.strip() for item in visible.split(",")]:
        raise ValueError("GPU4 is prohibited by workspace rules")
    device = torch.device("cuda:0")
    entries = inventory_entries(args.inventory)
    if args.entry_id:
        entries = [entry for entry in entries if entry["entry_id"] == args.entry_id]
        if not entries:
            raise ValueError(f"Unknown entry id: {args.entry_id}")
        methods = [entries[0]["method_key"]]
    else:
        methods = selected_methods(entries, args.worker_id, args.num_workers)
        entries = [entry for entry in entries if entry["method_key"] in methods]
    case_manifest = ensure_case_manifest(
        args.output_root,
        args.reference_config,
        seed=args.seed,
        cases_per_source=args.cases_per_source,
    )
    reference = load_resolved(args.reference_config)
    datasets = build_source_datasets(reference)
    worker_status = args.output_root / "status" / f"worker-{args.worker_id}.json"
    atomic_json(
        worker_status,
        {
            "state": "running",
            "worker_id": args.worker_id,
            "num_workers": args.num_workers,
            "visible_gpu": visible,
            "methods": methods,
            "entries": [entry["entry_id"] for entry in entries],
            "updated_utc": utc_now(),
        },
    )
    try:
        for method in methods:
            method_entries = [entry for entry in entries if entry["method_key"] == method]
            if not method_entries:
                continue
            manifest_path = find_manifest(Path(method_entries[0]["checkpoint"]))
            model, _model_args, _config, kind = build_model(manifest_path, device)
            for entry in method_entries:
                info = load_checkpoint(model, Path(entry["checkpoint"]))
                evaluate_entry(
                    model=model,
                    datasets=datasets,
                    case_manifest=case_manifest,
                    entry=entry,
                    output_root=args.output_root,
                    repeat_check=args.repeat_check,
                    load_info=info,
                    manifest_path=manifest_path,
                    kind=kind,
                )
                torch.cuda.empty_cache()
            del model
            torch.cuda.empty_cache()
        atomic_json(
            worker_status,
            {
                "state": "complete",
                "worker_id": args.worker_id,
                "num_workers": args.num_workers,
                "visible_gpu": visible,
                "methods": methods,
                "updated_utc": utc_now(),
            },
        )
    except Exception as exc:
        atomic_json(
            worker_status,
            {
                "state": "failed",
                "worker_id": args.worker_id,
                "num_workers": args.num_workers,
                "visible_gpu": visible,
                "methods": methods,
                "error": repr(exc),
                "updated_utc": utc_now(),
            },
        )
        raise


def summarize_result(result: dict[str, Any], total_expected: int) -> dict[str, Any]:
    by_source: dict[str, list[float]] = {source: [] for source in SOURCE_ORDER}
    for case in result.get("cases", []):
        by_source[case["source"]].append(float(case["loss_main"]))
    means = {
        source: statistics.fmean(values) if values else None
        for source, values in by_source.items()
    }
    is_complete = (
        result.get("state") == "complete"
        and len(result.get("cases", [])) == total_expected
    )
    macro = (
        statistics.fmean([means[source] for source in SOURCE_ORDER])
        if is_complete and all(means[source] is not None for source in SOURCE_ORDER)
        else None
    )
    return {
        "entry_id": result["entry_id"],
        "method_key": result["method_key"],
        "method_label": result["method_label"],
        "step": int(result["step"]),
        "checkpoint": result["checkpoint"],
        "state": result.get("state", "running"),
        "completed_cases": len(result.get("cases", [])),
        "total_cases": total_expected,
        "macro_mean": macro,
        "source_means": means,
        "source_counts": {source: len(values) for source, values in by_source.items()},
    }


def build_report(args: argparse.Namespace) -> Path:
    entries = inventory_entries(args.inventory)
    case_manifest = ensure_case_manifest(
        args.output_root,
        args.reference_config,
        seed=args.seed,
        cases_per_source=args.cases_per_source,
    )
    rows: list[dict[str, Any]] = []
    for entry in entries:
        path = result_path(args.output_root, entry)
        if path.is_file():
            rows.append(
                summarize_result(
                    json.loads(path.read_text(encoding="utf-8")),
                    case_manifest["total_cases"],
                )
            )
        else:
            rows.append(
                {
                    **entry,
                    "state": "pending",
                    "completed_cases": 0,
                    "total_cases": case_manifest["total_cases"],
                    "macro_mean": None,
                    "source_means": {source: None for source in SOURCE_ORDER},
                    "source_counts": {source: 0 for source in SOURCE_ORDER},
                }
            )
    complete = [row for row in rows if row["macro_mean"] is not None]
    overall_ranking = sorted(complete, key=lambda row: row["macro_mean"])
    for rank, row in enumerate(overall_ranking, start=1):
        row["overall_rank"] = rank
    source_rankings: dict[str, list[dict[str, Any]]] = {}
    for source in SOURCE_ORDER:
        ranked = sorted(
            [row for row in complete if row["source_means"][source] is not None],
            key=lambda row: row["source_means"][source],
        )
        source_rankings[source] = [
            {
                "rank": rank,
                "entry_id": row["entry_id"],
                "method_label": row["method_label"],
                "step": row["step"],
                "mean_loss": row["source_means"][source],
            }
            for rank, row in enumerate(ranked, start=1)
        ]
    payload = {
        "schema_version": 1,
        "updated_utc": utc_now(),
        "metric": "common_timestep_weighted_flow_matching_mse",
        "ranking_direction": "ascending",
        "seed": args.seed,
        "cases_per_source": args.cases_per_source,
        "total_entries": len(entries),
        "complete_entries": len(complete),
        "rows": rows,
        "overall_ranking": [row["entry_id"] for row in overall_ranking],
        "source_rankings": source_rankings,
    }
    atomic_json(args.output_root / "rankings.json", payload)
    csv_path = args.output_root / "rankings.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "overall_rank",
                "method",
                "step",
                "macro_mean",
                "pybullet_mean",
                "kubric_mean",
                "openvid_mean",
                "completed_cases",
                "entry_id",
                "checkpoint",
            ]
        )
        for row in sorted(
            rows,
            key=lambda item: (
                item["macro_mean"] is None,
                item["macro_mean"] if item["macro_mean"] is not None else float("inf"),
            ),
        ):
            writer.writerow(
                [
                    row.get("overall_rank", ""),
                    row["method_label"],
                    row["step"],
                    row["macro_mean"],
                    row["source_means"]["pybullet"],
                    row["source_means"]["kubric"],
                    row["source_means"]["openvid"],
                    row["completed_cases"],
                    row["entry_id"],
                    row["checkpoint"],
                ]
            )
    return build_html(args.output_root, payload)


def format_loss(value: float | None) -> str:
    return "—" if value is None else f"{value:.8f}"


def build_html(output_root: Path, payload: dict[str, Any]) -> Path:
    rows = sorted(
        payload["rows"],
        key=lambda item: (
            item["macro_mean"] is None,
            item["macro_mean"] if item["macro_mean"] is not None else float("inf"),
        ),
    )
    table_rows = []
    for row in rows:
        progress = f"{row['completed_cases']}/{row['total_cases']}"
        table_rows.append(
            "<tr>"
            f"<td>{row.get('overall_rank', '—')}</td>"
            f"<td><strong>{html.escape(row['method_label'])}</strong><small>{html.escape(row['entry_id'])}</small></td>"
            f"<td>{row['step']}</td>"
            f"<td>{format_loss(row['macro_mean'])}</td>"
            f"<td>{format_loss(row['source_means']['pybullet'])}</td>"
            f"<td>{format_loss(row['source_means']['kubric'])}</td>"
            f"<td>{format_loss(row['source_means']['openvid'])}</td>"
            f"<td>{progress}</td>"
            "</tr>"
        )
    ranking_sections = []
    for source in SOURCE_ORDER:
        ranking = payload["source_rankings"][source]
        items = "".join(
            f"<li><b>#{item['rank']}</b><span>{html.escape(item['method_label'])} · step-{item['step']}</span><code>{item['mean_loss']:.8f}</code></li>"
            for item in ranking
        ) or "<li>等待完整结果</li>"
        ranking_sections.append(
            f"<section><h2>{SOURCE_LABELS[source]}</h2><ol>{items}</ol></section>"
        )
    document = f"""<!doctype html><html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><meta http-equiv="refresh" content="120">
<title>Train-subset deterministic validation loss</title><style>
:root{{--bg:#0e1417;--panel:#172126;--ink:#ecf4f1;--muted:#97aaa5;--line:#30413e;--accent:#62d0b3;--gold:#f3bd58}}
*{{box-sizing:border-box}}body{{margin:0;background:linear-gradient(145deg,#091014,#16231f);color:var(--ink);font:14px/1.5 system-ui,sans-serif}}
main{{max-width:1600px;margin:auto;padding:28px 24px 80px}}h1{{font:500 clamp(30px,4vw,52px) Georgia,serif;margin:0 0 8px}}p{{color:var(--muted)}}
.summary{{display:flex;gap:12px;flex-wrap:wrap;margin:20px 0}}.pill{{padding:9px 13px;border:1px solid var(--line);border-radius:999px;background:var(--panel)}}
.table-wrap{{overflow:auto;border:1px solid var(--line);border-radius:13px;background:var(--panel)}}table{{border-collapse:collapse;width:100%;min-width:1100px}}th,td{{padding:10px 12px;border-bottom:1px solid var(--line);text-align:right}}th{{position:sticky;top:0;background:#20302e;color:var(--accent)}}th:nth-child(2),td:nth-child(2){{text-align:left}}tbody tr:hover{{background:#20302e}}small{{display:block;color:var(--muted);font:11px monospace}}code{{color:var(--gold)}}
.rankings{{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:14px;margin-top:22px}}section{{background:var(--panel);border:1px solid var(--line);border-radius:13px;padding:15px}}h2{{font-size:17px;margin:0 0 10px;color:var(--accent)}}ol{{list-style:none;padding:0;margin:0;max-height:620px;overflow:auto}}li{{display:grid;grid-template-columns:38px 1fr auto;gap:8px;padding:7px 0;border-bottom:1px solid #30413e88}}
a{{color:var(--accent)}}@media(max-width:1000px){{.rankings{{grid-template-columns:1fr}}}}
</style></head><body><main><h1>固定训练子集 · Validation Flow Loss 排名</h1>
<p>PyBullet / Kubric / OpenVid 训练集各随机无放回抽取 60 条，固定 seed=42；每条 case 的 timestep 与 Gaussian noise 对所有 checkpoint 完全一致。跨方案只排名共有的 timestep-weighted flow-matching MSE（越低越好），不把 xSSC/V-JEPA 私有辅助 loss 混入。</p>
<div class="summary"><span class="pill">完成 <b>{payload['complete_entries']}/{payload['total_entries']}</b> 权重</span><span class="pill">总前向 <b>{sum(row['completed_cases'] for row in payload['rows'])}/{payload['total_entries'] * 180}</b></span><span class="pill">两分钟自动刷新</span><a class="pill" href="rankings.csv">下载 CSV</a><a class="pill" href="rankings.json">查看 JSON</a><a class="pill" href="case_manifest.json">固定 case manifest</a></div>
<div class="table-wrap"><table><thead><tr><th>总排名</th><th>方案 / 权重</th><th>step</th><th>三数据集宏平均</th><th>PyBullet</th><th>Kubric</th><th>OpenVid</th><th>进度</th></tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<div class="rankings">{''.join(ranking_sections)}</div></main></body></html>"""
    path = output_root / "index.html"
    path.write_text(document, encoding="utf-8")
    return path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, default=DEFAULT_INVENTORY)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--reference-config", type=Path, default=DEFAULT_REFERENCE_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cases-per-source", type=int, default=60)
    parser.add_argument("--worker-id", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--entry-id", default=None)
    parser.add_argument("--repeat-check", action="store_true")
    parser.add_argument("--build-report-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.inventory = args.inventory.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.reference_config = args.reference_config.expanduser().resolve()
    if args.worker_id < 0 or args.worker_id >= args.num_workers:
        raise ValueError("worker-id must be in [0, num-workers)")
    args.output_root.mkdir(parents=True, exist_ok=True)
    if args.build_report_only:
        print(build_report(args))
        return
    run_worker(args)
    print(build_report(args))


if __name__ == "__main__":
    main()
