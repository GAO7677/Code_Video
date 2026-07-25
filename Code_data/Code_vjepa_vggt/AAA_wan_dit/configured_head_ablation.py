#!/usr/bin/env python3
"""Shared configuration helpers for the all-block/all-head ablation sweep."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


MODELS = ("wan_lora", "xssc", "physrvg")
METRIC_KINDS = ("cpu", "gpu_common", "videophy2", "cosmos")


def load_config(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("config schema_version must be 1")

    for key in ("experiment", "input", "output", "sweep", "inference", "metrics"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"config.{key} must be an object")

    models = payload["sweep"].get("models")
    if (
        not isinstance(models, list)
        or not models
        or len(models) != len(set(models))
        or any(model not in MODELS for model in models)
    ):
        raise ValueError(f"sweep.models must be unique values from {MODELS}")

    blocks = expand_int_selection(payload["sweep"].get("blocks"), "blocks")
    heads = expand_int_selection(payload["sweep"].get("heads"), "heads")
    if any(value < 0 or value > 29 for value in blocks):
        raise ValueError("all block ids must be in [0, 29]")
    if any(value < 0 or value > 23 for value in heads):
        raise ValueError("all head ids must be in [0, 23]")
    if payload["sweep"].get("mode") != "self_attn_head_zero":
        raise ValueError("this runner requires sweep.mode=self_attn_head_zero")

    gpus = payload["experiment"].get("gpus")
    if (
        not isinstance(gpus, list)
        or not gpus
        or len(gpus) != len(set(gpus))
        or any(not isinstance(gpu, int) or gpu < 0 for gpu in gpus)
    ):
        raise ValueError("experiment.gpus must be a non-empty unique integer list")

    expected_cases = payload["input"].get("expected_unique_cases")
    if not isinstance(expected_cases, int) or expected_cases <= 0:
        raise ValueError("input.expected_unique_cases must be positive")

    metric_groups = payload["metrics"].get("groups")
    workers = payload["metrics"].get("workers_per_gpu")
    if not isinstance(metric_groups, dict) or not isinstance(workers, dict):
        raise ValueError("metrics.groups/workers_per_gpu must be objects")
    flattened: list[str] = []
    for kind in METRIC_KINDS:
        values = metric_groups.get(kind)
        count = workers.get(kind)
        if not isinstance(values, list) or not values:
            raise ValueError(f"metrics.groups.{kind} must be a non-empty list")
        if not isinstance(count, int) or count <= 0:
            raise ValueError(f"metrics.workers_per_gpu.{kind} must be positive")
        flattened.extend(values)
    if len(flattened) != len(set(flattened)):
        raise ValueError("each metric may appear in only one group")

    for key in (
        "height",
        "width",
        "num_frames",
        "context_frames",
        "num_inference_steps",
        "fps",
        "seed",
    ):
        if not isinstance(payload["inference"].get(key), int):
            raise ValueError(f"inference.{key} must be an integer")
    for key in ("cfg_scale", "guidance_scale"):
        if not isinstance(payload["inference"].get(key), (int, float)):
            raise ValueError(f"inference.{key} must be numeric")
    return payload


def expand_int_selection(value: Any, name: str) -> list[int]:
    if isinstance(value, list):
        if not value or any(not isinstance(item, int) for item in value):
            raise ValueError(f"sweep.{name} list must contain integers")
        result = value
    elif isinstance(value, dict):
        start = value.get("start")
        stop = value.get("stop")
        step = value.get("step", 1)
        if (
            not isinstance(start, int)
            or not isinstance(stop, int)
            or not isinstance(step, int)
            or step <= 0
            or stop < start
        ):
            raise ValueError(f"invalid sweep.{name} range")
        result = list(range(start, stop + 1, step))
    else:
        raise ValueError(f"sweep.{name} must be a list or range object")
    if len(result) != len(set(result)):
        raise ValueError(f"sweep.{name} contains duplicates")
    return result


def read_unique_inputs(config: dict[str, Any]) -> list[Path]:
    source = Path(config["input"]["list_path"]).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(source)
    deduplicate = bool(config["input"].get("deduplicate", True))
    entries: list[Path] = []
    seen: set[Path] = set()
    for line in source.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        path = Path(stripped).expanduser().resolve()
        if deduplicate and path in seen:
            continue
        seen.add(path)
        entries.append(path)
    limit = config["input"].get("max_cases")
    if limit is not None:
        if not isinstance(limit, int) or limit <= 0:
            raise ValueError("input.max_cases must be null or positive")
        entries = entries[:limit]
    expected = int(config["input"]["expected_unique_cases"])
    if len(entries) != expected:
        raise ValueError(f"expected {expected} input cases, found {len(entries)}")
    stems = [path.stem for path in entries]
    if len(stems) != len(set(stems)):
        raise ValueError("input JSON stems must be unique after deduplication")
    missing = [str(path) for path in entries if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing input JSONs: {missing}")
    return entries


def config_fingerprint(path: Path) -> str:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


def output_base(config: dict[str, Any]) -> Path:
    return Path(config["output"]["base_dir"]).expanduser().resolve()


def run_root(config: dict[str, Any]) -> Path:
    configured = config["output"].get("run_root")
    if configured:
        return Path(configured).expanduser().resolve()
    return output_base(config) / "_pipeline"


def selected_blocks(config: dict[str, Any]) -> list[int]:
    return expand_int_selection(config["sweep"]["blocks"], "blocks")


def selected_heads(config: dict[str, Any]) -> list[int]:
    return expand_int_selection(config["sweep"]["heads"], "heads")


def generation_jobs(config: dict[str, Any]) -> list[tuple[str, str, int, int]]:
    jobs = []
    index = 0
    for model in config["sweep"]["models"]:
        for block in selected_blocks(config):
            for head in selected_heads(config):
                jobs.append((f"gen-{index:04d}", model, block, head))
                index += 1
    return jobs


def ablation_tag(block: int, head: int) -> str:
    return f"self_attn_head_zero_block{block:02d}_head{head:02d}"


def config_root(
    config: dict[str, Any], model: str, block: int, head: int
) -> Path:
    model_dir = "PhyRVG" if model == "physrvg" else model
    return output_base(config) / model_dir / ablation_tag(block, head)


def result_config_count(config: dict[str, Any]) -> int:
    return (
        len(config["sweep"]["models"])
        * len(selected_blocks(config))
        * len(selected_heads(config))
    )


def metric_count(config: dict[str, Any]) -> int:
    return sum(
        len(config["metrics"]["groups"][kind]) for kind in METRIC_KINDS
    )


def metric_worker_count(config: dict[str, Any]) -> int:
    per_gpu = sum(
        int(config["metrics"]["workers_per_gpu"][kind])
        for kind in METRIC_KINDS
    )
    return len(config["experiment"]["gpus"]) * per_gpu

