#!/usr/bin/env python3
"""Prepare and evaluate GT clips for the 8844 benchmark tables.

The evaluator intentionally uses the same ``AAAinfer/bench.py`` entry point
as generated videos.  GT clips are kept in a separate data root and are
never placed below a model result directory.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


DATA_ROOT = Path("/data/gaoya/agent-data/outputs/gt_metrics_all_8844")
RESULT_ROOT = DATA_ROOT / "results"
VIDEO_CACHE_ROOT = DATA_ROOT / "video_cache"
SUMMARY_ROOT = DATA_ROOT / "summaries"
INPUT_JSON_ROOT = DATA_ROOT / "input_jsons"
MANIFEST_PATH = DATA_ROOT / "manifest.json"
ALLOWLIST_PATH = DATA_ROOT / "input_allowlist.txt"

BENCH_PYTHON = Path(os.environ.get("BENCH_PYTHON", str(sys.executable)))
BENCH_SCRIPT = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/"
    "code_vjepa_vggt/AAAinfer/bench.py"
)
PROJECT_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt")
TRY0526_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
FFMPEG = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")

DATASETS: dict[str, dict[str, Any]] = {
    "test5": {
        "label": "test_5",
        "input_list": Path("/data/gaoya/AAA_test_video/0623/testjsons/test_5.txt"),
        "page_roots": ["test5-average-metrics"],
    },
    "physiciq": {
        "label": "PhysicIQ",
        "input_list": Path(
            "/data/gaoya/AAA_test_video/0623/testjsons/v2v_jsons_physicIQ.txt"
        ),
        "page_roots": [
            "physiciq-average-metrics",
            "physiciq-average-metrics/solid-mechanics",
        ],
    },
    "physv40": {
        "label": "PhysV V2V 0819 · 40-case",
        # The original 40-case list was consumed by the running evaluator;
        # its immutable input inventory is the authoritative copy.
        "input_list": Path(
            "/data/gaoya/agent-data/outputs/physv_v2v_0819_physrvg/input_list.txt"
        ),
        "page_roots": ["physv-v2v-0819-physrvg"],
    },
    "test70": {
        "label": "PhysV V2V 0819 · all-cycles test70",
        "input_list": Path(
            "/data/gaoya/AAA_test_video/physv_v2v_0819/testjsons/"
            "physv_v2v_0819_all_cycles_test70_ctx8.txt"
        ),
        "page_roots": [
            "physv-v2v-0819-full-sa-test70",
            "physv-v2v-0819-test70-no-event-timing-40step",
        ],
    },
}

# These are the fields actually produced by bench.py.  The main 8844 table
# derives four additional VideoPhy2 columns from the nested result below.
METRICS = (
    "physics_iq_with_context",
    "physics_iq_without_context",
    "pmf_with_context",
    "pmf_without_context",
    "wmreward",
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
    "videophy2",
    "cosmos_reason1",
)
DISPLAY_METRICS = (
    "videophy2_pc_raw",
    "cosmos_reason1",
    "physics_iq_with_context",
    "physics_iq_without_context",
    "videophy2",
    "videophy2_sa",
    "videophy2_pc",
    "videophy2_joint_rate",
    "pmf_with_context",
    "pmf_without_context",
    "wmreward",
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
)
GPU_METRICS = {
    "wmreward",
    "vbench_subject_consistency",
    "vbench_background_consistency",
    "vbench_temporal_flickering",
    "vbench_motion_smoothness",
    "vbench_dynamic_degree",
    "vbench_aesthetic_quality",
    "vbench_imaging_quality",
    "videophy2",
    "cosmos_reason1",
}

HUB_ROOT = Path("/data/gaoya/agent-data/outputs/xssc_object_self_attn_lora_hub")

# The active 40-case inventory still points at the original dataset tree, but
# ten source folders were moved during dataset cleanup.  This backup is an
# immutable copy of those exact cases (including their context8 clips).  Use
# it only when the path recorded in the input JSON is absent; all other cases
# continue to resolve directly from the JSON.
PHYSV40_BACKUP_ROOT = Path(
    "/data/gaoya/AAA_test_video/physv_v2v_0819_backup_20260820/samples"
)


def atomic_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def require_file(path: Path, description: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")
    return path


def safe_name(value: str) -> str:
    return "".join(char if char.isalnum() or char in "._-" else "_" for char in value)


def link_once(source: Path, target: Path) -> None:
    source = source.resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_symlink():
        if target.resolve() == source:
            return
        raise FileExistsError(f"Refusing to replace symlink: {target}")
    if target.exists():
        if target.resolve() == source:
            return
        raise FileExistsError(f"Refusing to replace existing file: {target}")
    target.symlink_to(source)


def render_gt_clip(source: Path) -> Path:
    source = require_file(source, "GT source_video")
    digest = hashlib.sha1(str(source).encode("utf-8")).hexdigest()[:20]
    destination = VIDEO_CACHE_ROOT / f"{digest}_gt_49f_30fps.mp4"
    if destination.is_file() and destination.stat().st_size > 0:
        return destination.resolve()
    ffmpeg = FFMPEG if FFMPEG.is_file() else Path(shutil.which("ffmpeg") or "")
    if not ffmpeg.is_file():
        raise FileNotFoundError("ffmpeg is required to materialize the GT clip")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Keep the .mp4 suffix so ffmpeg can select its muxer for the temporary
    # output as well.
    temporary = destination.with_name(
        f".{destination.stem}.tmp.{os.getpid()}.mp4"
    )
    subprocess.run(
        [
            str(ffmpeg),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(source),
            "-vf",
            "fps=30",
            "-frames:v",
            "49",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "fast",
            "-crf",
            "18",
            "-pix_fmt",
            "yuv420p",
            str(temporary),
        ],
        check=True,
    )
    os.replace(temporary, destination)
    return destination.resolve()


def resolve_case_media(
    dataset_key: str,
    stem: str,
    source_value: str,
    context_value: str,
) -> tuple[Path, Path, dict[str, str]]:
    """Resolve the declared GT/context paths, with an audited 40-case fallback."""

    declared_source = Path(source_value).expanduser()
    declared_context = Path(context_value).expanduser()
    source = declared_source if declared_source.is_file() else None
    context = declared_context if declared_context.is_file() else None
    resolution: dict[str, str] = {}
    if dataset_key == "physv40" and (source is None or context is None):
        backup_case = PHYSV40_BACKUP_ROOT / stem
        backup_source = backup_case / "videos" / "rgb.mp4"
        backup_context = backup_case / "context" / "context8.mp4"
        if source is None and backup_source.is_file():
            source = backup_source
            resolution["source_video"] = "physv_v2v_0819_backup_20260820"
        if context is None and backup_context.is_file():
            context = backup_context
            resolution["input_video"] = "physv_v2v_0819_backup_20260820"
    if source is None:
        raise FileNotFoundError(
            f"source_video not found for {dataset_key}/{stem}: {declared_source}"
        )
    if context is None:
        raise FileNotFoundError(
            f"input_video context not found for {dataset_key}/{stem}: {declared_context}"
        )
    return source.resolve(), context.resolve(), resolution


def write_evaluation_input_json(
    dataset_key: str,
    stem: str,
    payload: dict[str, Any],
    source: Path,
    context: Path,
    original_input_json: Path,
) -> Path:
    """Create an isolated input JSON so bench.py can resolve GT reliably."""

    shadow = dict(payload)
    shadow["source_video"] = str(source)
    shadow["input_video"] = str(context)
    shadow["context_video"] = str(context)
    if "input_video_8f" in shadow:
        shadow["input_video_8f"] = str(context)
    shadow["original_input_json"] = str(original_input_json)
    destination = INPUT_JSON_ROOT / dataset_key / f"{safe_name(stem)}.json"
    atomic_write(destination, shadow)
    return destination


def load_cases(dataset_key: str, spec: dict[str, Any]) -> list[dict[str, Any]]:
    input_list = require_file(Path(spec["input_list"]), f"{dataset_key} input list")
    cases: list[dict[str, Any]] = []
    seen: set[Path] = set()
    for line in input_list.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        input_json = require_file(Path(line.strip()), f"{dataset_key} input JSON")
        if input_json in seen:
            continue
        seen.add(input_json)
        payload = read_json(input_json)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected JSON object: {input_json}")
        context_value = payload.get("input_video") or payload.get("input_video_8f")
        stem = input_json.stem
        declared_source = str(payload.get("source_video", ""))
        declared_context = str(context_value or "")
        source, context, media_resolution = resolve_case_media(
            dataset_key,
            stem,
            declared_source,
            declared_context,
        )
        evaluation_input_json = write_evaluation_input_json(
            dataset_key,
            stem,
            payload,
            source,
            context,
            input_json,
        )
        cases.append(
            {
                "dataset": dataset_key,
                "dataset_label": str(spec["label"]),
                "stem": stem,
                "record_key": f"{dataset_key}__{safe_name(stem)}",
                "input_json": str(input_json),
                "evaluation_input_json": str(evaluation_input_json),
                "source_video": str(source),
                "context_video": str(context),
                "declared_source_video": declared_source,
                "declared_context_video": declared_context,
                "media_resolution": media_resolution,
                "caption": str(payload.get("input_caption") or payload.get("caption") or ""),
            }
        )
    if not cases:
        raise ValueError(f"No cases found for {dataset_key}")
    return cases


def prepare() -> dict[str, Any]:
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    RESULT_ROOT.mkdir(parents=True, exist_ok=True)
    VIDEO_CACHE_ROOT.mkdir(parents=True, exist_ok=True)
    datasets: dict[str, Any] = {}
    all_input_jsons: list[str] = []
    seen_inputs: set[str] = set()
    for dataset_key, spec in DATASETS.items():
        cases = load_cases(dataset_key, spec)
        for case in cases:
            candidate = render_gt_clip(Path(case["source_video"]))
            result_dir = RESULT_ROOT / dataset_key
            result_json = result_dir / f"{case['record_key']}.json"
            result_video = result_json.with_suffix(".mp4")
            link_once(candidate, result_video)
            payload = {
                "schema_version": 1,
                "method": "GT · Reference",
                "model": "ground_truth",
                "step": -1,
                "dataset": dataset_key,
                "input_json": case["evaluation_input_json"],
                "input_caption": case["caption"],
                "input_video": case["context_video"],
                "context_video": case["context_video"],
                "source_video": case["source_video"],
                "output_video": str(candidate),
                "inference": {
                    "height": 512,
                    "width": 896,
                    "num_frames": 49,
                    "fps": 30,
                    "context_frames": 8,
                    "num_inference_steps": 40,
                    "seed": 42,
                    "candidate_protocol": "source_video -> fps=30, first 49 frames",
                },
            }
            # ``prepare`` is intentionally resumable: re-running it must not
            # erase metric fields already backfilled by bench.py.
            if result_json.is_file():
                existing = read_json(result_json)
                if isinstance(existing, dict):
                    for metric in METRICS:
                        if existing.get(metric) is not None:
                            payload[metric] = existing[metric]
            atomic_write(result_json, payload)
            case["result_json"] = str(result_json)
            case["candidate_video"] = str(candidate)
            evaluation_input_json = case["evaluation_input_json"]
            if evaluation_input_json not in seen_inputs:
                all_input_jsons.append(evaluation_input_json)
                seen_inputs.add(evaluation_input_json)
        datasets[dataset_key] = {
            "label": spec["label"],
            "input_list": str(spec["input_list"]),
            "case_count": len(cases),
            "cases": cases,
        }
    manifest = {
        "schema_version": 1,
        "created_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "data_root": str(DATA_ROOT),
        "result_root": str(RESULT_ROOT),
        "input_allowlist": str(ALLOWLIST_PATH),
        "datasets": datasets,
        "all_input_jsons": all_input_jsons,
        "metrics": list(METRICS),
        "display_metrics": list(DISPLAY_METRICS),
        "protocol": {
            "candidate": "source_video rendered with the existing 49-frame/30-FPS GT renderer",
            "context": "input_video from each input JSON",
            "height": 512,
            "width": 896,
            "num_frames": 49,
            "fps": 30,
            "context_frames": 8,
            "seed": 42,
        },
    }
    ALLOWLIST_PATH.write_text("\n".join(all_input_jsons) + "\n", encoding="utf-8")
    atomic_write(MANIFEST_PATH, manifest)
    print(
        json.dumps(
            {
                "datasets": {key: value["case_count"] for key, value in datasets.items()},
                "unique_input_jsons": len(all_input_jsons),
                "result_root": str(RESULT_ROOT),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return manifest


def nested_number(payload: dict[str, Any], *path: str) -> float | None:
    value: Any = payload
    for key in path:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def extract_display_metrics(payload: dict[str, Any], *, wmreward_mode: str) -> dict[str, float]:
    values: dict[str, float] = {}
    paths = {
        "videophy2_pc_raw": ("videophy2", "pc_raw_score"),
        "cosmos_reason1": ("cosmos_reason1", "score"),
        "physics_iq_with_context": ("physics_iq_with_context", "score"),
        "physics_iq_without_context": ("physics_iq_without_context", "score"),
        "videophy2": ("videophy2", "score"),
        "videophy2_sa": ("videophy2", "sa_score"),
        "videophy2_pc": ("videophy2", "pc_score"),
        "videophy2_joint_rate": ("videophy2", "joint_rate"),
        "pmf_with_context": ("pmf_with_context", "score"),
        "pmf_without_context": ("pmf_without_context", "score"),
        "vbench_subject_consistency": ("vbench_subject_consistency", "score"),
        "vbench_background_consistency": ("vbench_background_consistency", "score"),
        "vbench_temporal_flickering": ("vbench_temporal_flickering", "score"),
        "vbench_motion_smoothness": ("vbench_motion_smoothness", "score"),
        "vbench_dynamic_degree": ("vbench_dynamic_degree", "score"),
        "vbench_aesthetic_quality": ("vbench_aesthetic_quality", "score"),
        "vbench_imaging_quality": ("vbench_imaging_quality", "score"),
    }
    for key, path in paths.items():
        number = nested_number(payload, *path)
        if number is not None:
            values[key] = number
    wmreward = payload.get("wmreward")
    if isinstance(wmreward, dict):
        field = "similarity" if wmreward_mode == "similarity" else "surprise"
        number = nested_number(wmreward, field)
        if number is not None:
            values["wmreward"] = number
        surprise = nested_number(wmreward, "surprise")
        similarity = nested_number(wmreward, "similarity")
        if surprise is not None:
            values["wmreward_surprise"] = surprise
        if similarity is not None:
            values["wmreward_similarity"] = similarity
    return values


def build_summary(dataset_key: str, cases: list[dict[str, Any]], *, wmreward_mode: str) -> dict[str, Any]:
    case_metrics: dict[str, dict[str, float]] = {}
    for case in cases:
        payload = read_json(Path(case["result_json"]))
        if isinstance(payload, dict):
            case_metrics[case["stem"]] = extract_display_metrics(
                payload, wmreward_mode=wmreward_mode
            )
    counts: dict[str, int] = {}
    averages: dict[str, float] = {}
    for metric in (*DISPLAY_METRICS, "wmreward_surprise", "wmreward_similarity"):
        numbers = [
            values[metric]
            for values in case_metrics.values()
            if metric in values
        ]
        if numbers:
            counts[metric] = len(numbers)
            averages[metric] = sum(numbers) / len(numbers)
    return {
        "schema_version": 1,
        "dataset": dataset_key,
        "label": "GT · Reference",
        "color": "#C7852C",
        "step": -1,
        "case_count": len(cases),
        "metric_counts": counts,
        "metric_averages": averages,
        "case_metrics": case_metrics,
        "wmreward_mode": wmreward_mode,
        "protocol": {
            "candidate": "source_video rendered with fps=30 and first 49 frames",
            "context": "input_video from the source JSON",
            "same_as_generated_evaluator": True,
        },
    }


def summarize() -> dict[str, Any]:
    manifest = read_json(MANIFEST_PATH)
    summaries: dict[str, Any] = {}
    for dataset_key, dataset in manifest["datasets"].items():
        summaries[dataset_key] = build_summary(
            dataset_key,
            list(dataset["cases"]),
            wmreward_mode="surprise",
        )
        atomic_write(SUMMARY_ROOT / f"{dataset_key}.json", summaries[dataset_key])
    atomic_write(SUMMARY_ROOT / "all.json", {"datasets": summaries})
    publish_page_sidecars(summaries)
    print(json.dumps({key: value["metric_counts"] for key, value in summaries.items()}, ensure_ascii=False, indent=2))
    return summaries


def publish_page_sidecars(summaries: dict[str, Any]) -> None:
    # The PhysV pages use the same underlying dashboard convention as their
    # generated rows: WMReward is represented by similarity there.  The main
    # xSSC tables retain their established surprise convention.
    for dataset_key in ("physv40", "test70"):
        source = summaries[dataset_key]
        page_summary = dict(source)
        page_summary["metric_averages"] = dict(source["metric_averages"])
        page_summary["metric_counts"] = dict(source["metric_counts"])
        page_summary["wmreward_mode"] = "similarity"
        for values in page_summary["case_metrics"].values():
            if "wmreward_similarity" in values:
                values["wmreward"] = values["wmreward_similarity"]
        for metric in DISPLAY_METRICS:
            numbers = [
                values[metric]
                for values in page_summary["case_metrics"].values()
                if metric in values
            ]
            if numbers:
                page_summary["metric_counts"][metric] = len(numbers)
                page_summary["metric_averages"][metric] = sum(numbers) / len(numbers)
        for page_root in DATASETS[dataset_key]["page_roots"]:
            atomic_write(HUB_ROOT / page_root / "gt_metrics.json", page_summary)


def metric_command(metric: str, *, gpu: str | None, overwrite: bool) -> tuple[list[str], dict[str, str]]:
    kind = "gpu" if metric in GPU_METRICS else "cpu"
    artifact_root = DATA_ROOT / "metric_artifacts" / metric
    summary_path = SUMMARY_ROOT / "bench" / f"{metric}.json"
    command = [
        str(BENCH_PYTHON),
        str(BENCH_SCRIPT),
        "--metric",
        metric,
        "--result-root",
        str(RESULT_ROOT),
        "--input-json-allowlist",
        str(ALLOWLIST_PATH),
        "--output-summary",
        str(summary_path),
        "--physics-iq-output-root",
        str(artifact_root / "physics_iq"),
        "--physics-iq-verified-output-root",
        str(artifact_root / "physics_iq_verified"),
        "--pmf-output-root",
        str(artifact_root / "pmf"),
        "--vbench-output-root",
        str(artifact_root / "vbench"),
        "--pmf-device",
        "cpu",
        "--vbench-device",
        "cuda" if kind == "gpu" else "cpu",
        "--wmreward-reset-interval",
        "1000000",
    ]
    if overwrite:
        command.append("--overwrite")
    env = dict(os.environ)
    env.update(
        {
            "PYTHONNOUSERSITE": "1",
            "PYTHONPATH": f"{PROJECT_ROOT}:{TRY0526_ROOT}",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    else:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return command, env


def gpu_memory_mib(gpu: int) -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={gpu}",
                "--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        return int(result.stdout.strip().splitlines()[0])
    except (OSError, ValueError, IndexError, subprocess.CalledProcessError):
        return 10**9


def run_one_metric(metric: str, gpu: str | None, *, overwrite: bool) -> int:
    command, env = metric_command(metric, gpu=gpu, overwrite=overwrite)
    log_path = DATA_ROOT / "logs" / f"{metric}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(f"\n[start] metric={metric} gpu={gpu} command={' '.join(command)}\n")
        handle.flush()
        process = subprocess.run(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        handle.write(f"[finish] metric={metric} gpu={gpu} rc={process.returncode}\n")
    return process.returncode


def run_all(*, once: bool, overwrite: bool) -> int:
    if not BENCH_PYTHON.is_file() or not BENCH_SCRIPT.is_file():
        raise FileNotFoundError("bench runtime or script is missing")
    if not MANIFEST_PATH.is_file():
        prepare()
    DATA_ROOT.joinpath("logs").mkdir(parents=True, exist_ok=True)
    # Publish a pending GT row before the long-running workers start; each
    # completed metric below refreshes the sidecars again so the dashboards
    # expose live counts instead of appearing absent until the end.
    summarize()
    # CPU metrics are independent, but keep one process per metric so each
    # runner can backfill incrementally and be resumed without a monolithic job.
    cpu_jobs = [metric for metric in METRICS if metric not in GPU_METRICS]
    gpu_jobs = [metric for metric in METRICS if metric in GPU_METRICS]
    processes: list[tuple[str, str | None, subprocess.Popen[Any]]] = []
    for metric in cpu_jobs:
        command, env = metric_command(metric, gpu=None, overwrite=overwrite)
        log_path = DATA_ROOT / "logs" / f"{metric}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        handle = log_path.open("a", encoding="utf-8")
        handle.write(f"\n[start] metric={metric} gpu=cpu command={' '.join(command)}\n")
        handle.flush()
        process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
        processes.append((metric, None, process))
    # Keep all GPU-side GT metrics on the single approved queue GPU.  CPU
    # metrics remain parallel and unchanged.  The environment override keeps
    # this evaluator easy to reuse, while the default follows the PhysRVG
    # watcher policy.
    gpu_spec = os.environ.get("GT_METRIC_GPUS", "6")
    available_gpus = [int(item.strip()) for item in gpu_spec.split(",") if item.strip()]
    if not available_gpus or 4 in available_gpus:
        raise ValueError("GT_METRIC_GPUS must contain at least one non-GPU4 device")
    running: list[tuple[str, str, subprocess.Popen[Any]]] = []
    pending = list(gpu_jobs)
    last_summary = 0.0
    while pending or running:
        for gpu in available_gpus:
            if not pending or any(item[1] == str(gpu) for item in running):
                continue
            if gpu_memory_mib(gpu) > 8000:
                continue
            metric = pending.pop(0)
            command, env = metric_command(metric, gpu=str(gpu), overwrite=overwrite)
            log_path = DATA_ROOT / "logs" / f"{metric}.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            handle = log_path.open("a", encoding="utf-8")
            handle.write(f"\n[start] metric={metric} gpu={gpu} command={' '.join(command)}\n")
            handle.flush()
            process = subprocess.Popen(command, env=env, stdout=handle, stderr=subprocess.STDOUT)
            running.append((metric, str(gpu), process))
        still_running: list[tuple[str, str, subprocess.Popen[Any]]] = []
        for metric, gpu, process in running:
            return_code = process.poll()
            if return_code is None:
                still_running.append((metric, gpu, process))
                continue
            (DATA_ROOT / "logs" / f"{metric}.log").open("a", encoding="utf-8").write(
                f"[finish] metric={metric} gpu={gpu} rc={return_code}\n"
            )
            if return_code != 0:
                print(f"[metric:error] {metric} gpu={gpu} rc={return_code}", flush=True)
            summarize()
        running = still_running
        if time.monotonic() - last_summary >= 30:
            try:
                summarize()
            except (OSError, json.JSONDecodeError, KeyError, TypeError) as exc:
                print(f"[summary:warning] {exc}", flush=True)
            last_summary = time.monotonic()
        if pending or running:
            time.sleep(30)
    for metric, gpu, process in processes:
        return_code = process.wait()
        if return_code != 0:
            print(f"[metric:error] {metric} cpu rc={return_code}", flush=True)
        summarize()
    summarize()
    return 0


def status() -> None:
    if not MANIFEST_PATH.is_file():
        print("GT manifest: not prepared")
        return
    manifest = read_json(MANIFEST_PATH)
    rows: dict[str, Any] = {}
    for dataset_key, dataset in manifest["datasets"].items():
        done = defaultdict(int)
        total = len(dataset["cases"])
        for case in dataset["cases"]:
            payload = read_json(Path(case["result_json"]))
            for metric in METRICS:
                if payload.get(metric) is not None:
                    done[metric] += 1
        rows[dataset_key] = {metric: f"{done[metric]}/{total}" for metric in METRICS}
    print(json.dumps(rows, ensure_ascii=False, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("prepare", "run", "summarize", "status"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--once", action="store_true", help="reserved for resume-compatible callers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mode == "prepare":
        prepare()
    elif args.mode == "run":
        raise SystemExit(run_all(once=args.once, overwrite=args.overwrite))
    elif args.mode == "summarize":
        summarize()
    else:
        status()


if __name__ == "__main__":
    main()
