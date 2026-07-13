from __future__ import annotations

import argparse
import csv
import gc
import json
import math
import os
import random
import subprocess
import sys
import traceback
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import cv2

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = THIS_FILE.parents[2]
TRY0526_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_try0526")
for _path in (PROJECT_ROOT, TRY0526_ROOT):
    _path_str = str(_path)
    if _path_str not in sys.path:
        sys.path.insert(0, _path_str)

from physv_eval.official_pdi import OfficialPDIRunner
from physv_eval.phyground_official import OfficialPhyGroundRunner
from physv_eval.paths import VPHY_PYTHON
from physv_eval.proxy_runner import ProxyRunner
from physv_eval.single_case import pdi, phyground, physics_iq, pmf, proxy, videophy2, wmreward
from physv_eval.vbench_official import OfficialVBenchRunner
from physv_eval.videophy2_auto import VideoPhy2Runner
from physv_eval.wmreward_official import WMRewardRunner


DEFAULT_OLD_ROOT = Path(
    "/data/gaoya/AAA_test_video/Dataset_physV/0613pybullet/raw_v1/industrial_s1_scale2_merged_h264_batch1500"
)
DEFAULT_NEW_ROOT = Path("/data/gaoya/agent-data/outputs/dataset_new_0705/AAA_check_0710")
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/rigid_dataset_single_case_metrics_compare_20260713"
)
DEFAULT_METRICS = (
    "pdi",
    "wmreward",
    "proxy",
    "videophy2",
    "phyground",
    "cosmos_reason1",
    "physics_iq_with_context",
    "physics_iq_without_context",
    "pmf_with_context",
    "pmf_without_context",
)
DEFAULT_VBENCH_DIMENSIONS = (
    "subject_consistency",
    "background_consistency",
    "temporal_flickering",
    "motion_smoothness",
    "dynamic_degree",
    "aesthetic_quality",
    "imaging_quality",
)
OLD_FAMILY_LABELS = {
    "F1_single_object": "F1",
    "F2_two_object": "F2",
    "F3_chain_reaction": "F3",
    "F4_occlusion": "F4",
    "F5_drop_support": "F5",
}


@dataclass(frozen=True)
class CaseSpec:
    dataset: str
    case_id: str
    family_key: str
    video_path: str
    source_video_path: str
    caption: str
    context_frames: int
    meta_json_path: str
    sample_root: str
    context_video_path: str | None = None
    notes: str | None = None


@dataclass(frozen=True)
class MetricRunSpec:
    name: str
    setup: Callable[[argparse.Namespace], tuple[dict[str, Any], Callable[[CaseSpec, dict[str, Any], Path], dict[str, Any] | None]]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare old/new rigid simulation datasets with physv_eval.single_case metrics. "
            "The new dataset uses the current 60 cases; the old dataset is sampled to 60 cases."
        )
    )
    parser.add_argument("--old-root", type=Path, default=DEFAULT_OLD_ROOT)
    parser.add_argument("--new-root", type=Path, default=DEFAULT_NEW_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--old-split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--old-total-cases", type=int, default=60)
    parser.add_argument("--new-total-cases", type=int, default=60)
    parser.add_argument("--old-sampling-mode", default="balanced", choices=["balanced", "random"])
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260713)
    parser.add_argument("--metrics", default=",".join(DEFAULT_METRICS))
    parser.add_argument("--vbench-dimensions", default=",".join(DEFAULT_VBENCH_DIMENSIONS))
    parser.add_argument("--vbench2-dimensions", default="")
    parser.add_argument("--vbench-device", default="cuda")
    parser.add_argument("--vbench-load-ckpt-from-local", action="store_true")
    parser.add_argument("--vbench-read-frame", action="store_true")
    parser.add_argument(
        "--vbench-imaging-quality-preprocessing-mode",
        default="longer",
        choices=["shorter", "longer", "shorter_centercrop", "None"],
    )
    parser.add_argument("--proxy-device", default="cuda")
    parser.add_argument("--videophy2-device", default="cuda")
    parser.add_argument("--videophy2-task", default="pc", choices=["sa", "pc", "rule"])
    parser.add_argument("--phyground-general-only", action="store_true", default=True)
    parser.add_argument("--phyground-fps", type=float, default=2.0)
    parser.add_argument("--cosmos-fps", type=int, default=16)
    parser.add_argument("--cosmos-python", type=Path, default=VPHY_PYTHON)
    parser.add_argument("--cosmos-cuda-visible-devices", default=None)
    parser.add_argument("--wmreward-cuda-visible-devices", default=None)
    parser.add_argument("--pdi-cuda-visible-devices", default=None)
    parser.add_argument("--phyground-cuda-visible-devices", default=None)
    parser.add_argument("--limit-per-dataset", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _ensure_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")


def _safe_caption(meta: dict[str, Any], *, fallback_case_id: str) -> str:
    for key in ("input_prompt", "description", "title", "family", "key"):
        value = meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return fallback_case_id


def _resolve_old_case_spec(meta_path: Path, context_frames: int) -> CaseSpec:
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    sample_root = meta_path.parent
    family_slug = str(payload.get("family_slug") or sample_root.parent.name)
    family_key = OLD_FAMILY_LABELS.get(family_slug, family_slug.split("_", 1)[0])
    case_id = str(payload.get("sample_id") or sample_root.name)
    video_path = sample_root / "video.mp4"
    source_video_path = sample_root / "source_video.mp4"
    if not source_video_path.is_file():
        source_video_path = video_path
    _ensure_file(video_path, "old video")
    _ensure_file(source_video_path, "old source_video")
    return CaseSpec(
        dataset="old",
        case_id=case_id,
        family_key=family_key,
        video_path=str(video_path),
        source_video_path=str(source_video_path),
        caption=_safe_caption(payload, fallback_case_id=case_id),
        context_frames=context_frames,
        meta_json_path=str(meta_path),
        sample_root=str(sample_root),
        context_video_path=None,
        notes="old dataset sample; source_video.mp4 preferred when available",
    )


def _load_old_cases(args: argparse.Namespace) -> list[CaseSpec]:
    split_root = args.old_root / args.old_split
    if not split_root.is_dir():
        raise FileNotFoundError(f"old split root not found: {split_root}")
    meta_paths = sorted(split_root.glob("*/*/meta.json"))
    if not meta_paths:
        raise RuntimeError(f"no old meta.json files found under {split_root}")

    if args.old_sampling_mode == "random":
        rng = random.Random(args.seed)
        picked = sorted(rng.sample(meta_paths, k=min(args.old_total_cases, len(meta_paths))))
        return [_resolve_old_case_spec(path, args.context_frames) for path in picked]

    family_to_paths: dict[str, list[Path]] = {}
    for path in meta_paths:
        family_slug = path.parent.parent.name
        family_key = OLD_FAMILY_LABELS.get(family_slug, family_slug.split("_", 1)[0])
        family_to_paths.setdefault(family_key, []).append(path)

    family_keys = sorted(family_to_paths)
    if not family_keys:
        raise RuntimeError("old dataset family grouping is empty")

    base = args.old_total_cases // len(family_keys)
    remainder = args.old_total_cases % len(family_keys)
    rng = random.Random(args.seed)
    picked: list[Path] = []
    shortage = 0
    leftovers: list[Path] = []
    for idx, family_key in enumerate(family_keys):
        requested = base + (1 if idx < remainder else 0)
        candidates = list(family_to_paths[family_key])
        if len(candidates) <= requested:
            picked.extend(candidates)
            shortage += requested - len(candidates)
            continue
        selected = rng.sample(candidates, k=requested)
        picked.extend(selected)
        selected_set = {str(path) for path in selected}
        leftovers.extend(path for path in candidates if str(path) not in selected_set)

    if shortage > 0:
        if len(leftovers) < shortage:
            raise RuntimeError(
                f"old dataset does not have enough samples to fill {args.old_total_cases} balanced cases"
            )
        picked.extend(rng.sample(leftovers, k=shortage))

    picked = sorted(picked)[: args.old_total_cases]
    return [_resolve_old_case_spec(path, args.context_frames) for path in picked]


def _resolve_new_case_spec(item: dict[str, Any], context_frames: int) -> CaseSpec:
    meta_path = Path(str(item["meta"]))
    payload = json.loads(meta_path.read_text(encoding="utf-8"))
    case_id = str(item["case_id"])
    family_key = str(item["family_key"])
    video_path = Path(str(item["video"]))
    _ensure_file(meta_path, "new meta")
    _ensure_file(video_path, "new video")
    return CaseSpec(
        dataset="new",
        case_id=case_id,
        family_key=family_key,
        video_path=str(video_path),
        source_video_path=str(video_path),
        caption=_safe_caption(payload, fallback_case_id=case_id),
        context_frames=context_frames,
        meta_json_path=str(meta_path),
        sample_root=str(Path(str(item["output_root"]))),
        context_video_path=None,
        notes="new dataset sample; self-reference source_video uses the case video itself",
    )


def _load_new_cases(args: argparse.Namespace) -> list[CaseSpec]:
    manifest_path = args.new_root / "manifest.json"
    _ensure_file(manifest_path, "new manifest")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise TypeError(f"expected list manifest in {manifest_path}, got {type(payload).__name__}")
    cases = [_resolve_new_case_spec(item, args.context_frames) for item in payload]
    cases = sorted(cases, key=lambda item: (item.family_key, item.case_id))
    return cases[: args.new_total_cases]


def _read_prefix_frames(video_path: Path, max_frames: int) -> tuple[list[Any], float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"failed to open video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = 30.0
    frames: list[Any] = []
    while len(frames) < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"video has no readable frames: {video_path}")
    return frames, fps


def _write_video(path: Path, frames: list[Any], fps: float) -> None:
    if not frames:
        raise ValueError("cannot write empty frame list")
    path.parent.mkdir(parents=True, exist_ok=True)
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"failed to open VideoWriter for {path}")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()


def _materialize_context_video(case: CaseSpec, cache_root: Path) -> Path:
    if case.context_video_path:
        candidate = Path(case.context_video_path)
        if candidate.is_file():
            return candidate
    output_path = cache_root / case.dataset / f"{case.case_id}_ctx{case.context_frames:02d}.mp4"
    if output_path.is_file():
        return output_path
    frames, fps = _read_prefix_frames(Path(case.source_video_path), case.context_frames)
    if len(frames) < case.context_frames:
        raise RuntimeError(
            f"{case.dataset}:{case.case_id} only has {len(frames)} readable frames, "
            f"but context_frames={case.context_frames}"
        )
    _write_video(output_path, frames[: case.context_frames], fps)
    return output_path


def _build_eval_case(case: CaseSpec, context_cache_root: Path) -> dict[str, Any]:
    context_video_path = _materialize_context_video(case, context_cache_root)
    return {
        "video": case.video_path,
        "source_video": case.source_video_path,
        "context_video": str(context_video_path),
        "caption": case.caption,
        "context_frames": int(case.context_frames),
        "dataset": case.dataset,
        "case_id": case.case_id,
        "family_key": case.family_key,
        "json_path": case.meta_json_path,
    }


def _flatten_numeric_leaves(payload: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(payload, bool):
        return out
    if isinstance(payload, (int, float)):
        value = float(payload)
        if math.isfinite(value):
            key = prefix if prefix else "value"
            out[key] = value
        return out
    if isinstance(payload, dict):
        for key, value in payload.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            out.update(_flatten_numeric_leaves(value, child_prefix))
    return out


def _stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return {
        "mean": mean,
        "std": math.sqrt(max(variance, 0.0)),
        "min": min(values),
        "max": max(values),
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _parse_csv_tokens(raw: str) -> list[str]:
    return [token.strip() for token in raw.split(",") if token.strip()]


def _build_metric_specs(requested_metrics: list[str]) -> list[MetricRunSpec]:
    specs = {
        "pdi": MetricRunSpec(name="pdi", setup=_setup_pdi),
        "wmreward": MetricRunSpec(name="wmreward", setup=_setup_wmreward),
        "proxy": MetricRunSpec(name="proxy", setup=_setup_proxy),
        "videophy2": MetricRunSpec(name="videophy2", setup=_setup_videophy2),
        "phyground": MetricRunSpec(name="phyground", setup=_setup_phyground),
        "cosmos_reason1": MetricRunSpec(name="cosmos_reason1", setup=_setup_cosmos_reason1),
        "physics_iq_with_context": MetricRunSpec(name="physics_iq_with_context", setup=_setup_physics_iq_with_context),
        "physics_iq_without_context": MetricRunSpec(name="physics_iq_without_context", setup=_setup_physics_iq_without_context),
        "pmf_with_context": MetricRunSpec(name="pmf_with_context", setup=_setup_pmf_with_context),
        "pmf_without_context": MetricRunSpec(name="pmf_without_context", setup=_setup_pmf_without_context),
    }
    expanded_specs: list[MetricRunSpec] = []
    unknown: list[str] = []
    for name in requested_metrics:
        if name in specs:
            expanded_specs.append(specs[name])
        elif name.startswith("vbench_"):
            dimension = name[len("vbench_") :]
            expanded_specs.append(MetricRunSpec(name=name, setup=_make_vbench_setup(dimension)))
        elif name == "vbench":
            expanded_specs.append(MetricRunSpec(name="__expand_vbench__", setup=lambda _: ({}, lambda *_: None)))
        else:
            unknown.append(name)
    if unknown:
        raise ValueError(f"unsupported metrics: {unknown}")
    return expanded_specs


def _expand_requested_metric_specs(args: argparse.Namespace, requested_metrics: list[str]) -> list[MetricRunSpec]:
    raw_specs = _build_metric_specs(requested_metrics)
    final_specs: list[MetricRunSpec] = []
    for spec in raw_specs:
        if spec.name != "__expand_vbench__":
            final_specs.append(spec)
            continue
        dimensions = _parse_csv_tokens(args.vbench_dimensions)
        if not dimensions:
            raise ValueError("metrics includes 'vbench' but --vbench-dimensions is empty")
        for dimension in dimensions:
            final_specs.append(MetricRunSpec(name=f"vbench_{dimension}", setup=_make_vbench_setup(dimension)))
    return final_specs


def _setup_pdi(args: argparse.Namespace) -> tuple[dict[str, Any], Callable[[CaseSpec, dict[str, Any], Path], dict[str, Any] | None]]:
    runner = OfficialPDIRunner(cuda_visible_devices=args.pdi_cuda_visible_devices)

    def run(case: CaseSpec, eval_case: dict[str, Any], _: Path) -> dict[str, Any] | None:
        return pdi.score_case(eval_case, text_query=case.caption, runner=runner)

    return {"runner": runner}, run


def _setup_wmreward(args: argparse.Namespace) -> tuple[dict[str, Any], Callable[[CaseSpec, dict[str, Any], Path], dict[str, Any] | None]]:
    runner = WMRewardRunner(cuda_visible_devices=args.wmreward_cuda_visible_devices)

    def run(_: CaseSpec, eval_case: dict[str, Any], __: Path) -> dict[str, Any] | None:
        return wmreward.score_case(eval_case, runner=runner)

    return {"runner": runner}, run


def _setup_proxy(args: argparse.Namespace) -> tuple[dict[str, Any], Callable[[CaseSpec, dict[str, Any], Path], dict[str, Any] | None]]:
    runner = ProxyRunner(device=args.proxy_device)

    def run(case: CaseSpec, eval_case: dict[str, Any], __: Path) -> dict[str, Any] | None:
        return proxy.score_case(
            eval_case,
            context_video_path=eval_case["context_video"],
            runner=runner,
        )

    return {"runner": runner}, run


def _setup_videophy2(args: argparse.Namespace) -> tuple[dict[str, Any], Callable[[CaseSpec, dict[str, Any], Path], dict[str, Any] | None]]:
    runner = VideoPhy2Runner(device=args.videophy2_device)

    def run(case: CaseSpec, eval_case: dict[str, Any], __: Path) -> dict[str, Any] | None:
        return videophy2.score_case(
            eval_case,
            task=args.videophy2_task,
            caption=case.caption,
            runner=runner,
        )

    return {"runner": runner}, run


def _setup_phyground(args: argparse.Namespace) -> tuple[dict[str, Any], Callable[[CaseSpec, dict[str, Any], Path], dict[str, Any] | None]]:
    runner = OfficialPhyGroundRunner(
        cuda_visible_devices=args.phyground_cuda_visible_devices,
        fps=args.phyground_fps,
    )

    def run(case: CaseSpec, eval_case: dict[str, Any], __: Path) -> dict[str, Any] | None:
        laws = [] if args.phyground_general_only else None
        return phyground.score_case(
            eval_case,
            caption=case.caption,
            laws=laws,
            runner=runner,
        )

    return {"runner": runner}, run


def _setup_cosmos_reason1(args: argparse.Namespace) -> tuple[dict[str, Any], Callable[[CaseSpec, dict[str, Any], Path], dict[str, Any] | None]]:
    cosmos_python = Path(args.cosmos_python).expanduser().resolve()
    if not cosmos_python.is_file():
        raise FileNotFoundError(f"cosmos python not found: {cosmos_python}")

    def run(case: CaseSpec, _: dict[str, Any], work_dir: Path) -> dict[str, Any] | None:
        output_json = work_dir / "cosmos_reason1_result.json"
        output_json.parent.mkdir(parents=True, exist_ok=True)
        command = [
            str(cosmos_python),
            "-m",
            "physv_eval.single_case.cosmos_reason1",
            "--video",
            case.video_path,
            "--fps",
            str(args.cosmos_fps),
            "--output-json",
            str(output_json),
        ]
        env = os.environ.copy()
        existing_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{TRY0526_ROOT}{os.pathsep}{existing_pythonpath}" if existing_pythonpath else str(TRY0526_ROOT)
        )
        if args.cosmos_cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(args.cosmos_cuda_visible_devices)
        completed = subprocess.run(
            command,
            cwd=str(TRY0526_ROOT),
            env=env,
            text=True,
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "cosmos_reason1 subprocess failed\n"
                f"command: {' '.join(command)}\n"
                f"returncode: {completed.returncode}\n"
                f"stdout:\n{completed.stdout[-4000:]}\n"
                f"stderr:\n{completed.stderr[-4000:]}"
            )
        if not output_json.is_file():
            raise FileNotFoundError(f"cosmos_reason1 output json not found: {output_json}")
        return json.loads(output_json.read_text(encoding='utf-8'))

    return {"runner": None, "python": str(cosmos_python)}, run


def _setup_physics_iq_with_context(_: argparse.Namespace) -> tuple[dict[str, Any], Callable[[CaseSpec, dict[str, Any], Path], dict[str, Any] | None]]:
    def run(case: CaseSpec, eval_case: dict[str, Any], aligned_dir: Path) -> dict[str, Any] | None:
        return physics_iq.score_case(
            eval_case,
            source_video_path=Path(case.source_video_path),
            context_mode="with_context",
            context_frames=case.context_frames,
            aligned_video_dir=aligned_dir,
        )

    return {}, run


def _setup_physics_iq_without_context(_: argparse.Namespace) -> tuple[dict[str, Any], Callable[[CaseSpec, dict[str, Any], Path], dict[str, Any] | None]]:
    def run(case: CaseSpec, eval_case: dict[str, Any], aligned_dir: Path) -> dict[str, Any] | None:
        return physics_iq.score_case(
            eval_case,
            source_video_path=Path(case.source_video_path),
            context_mode="without_context",
            context_frames=case.context_frames,
            aligned_video_dir=aligned_dir,
        )

    return {}, run


def _setup_pmf_with_context(args: argparse.Namespace) -> tuple[dict[str, Any], Callable[[CaseSpec, dict[str, Any], Path], dict[str, Any] | None]]:
    def run(case: CaseSpec, eval_case: dict[str, Any], aligned_dir: Path) -> dict[str, Any] | None:
        return pmf.score_case(
            eval_case,
            source_video_path=Path(case.source_video_path),
            context_mode="with_context",
            context_frames=case.context_frames,
            device=args.proxy_device,
            aligned_video_dir=aligned_dir,
        )

    return {}, run


def _setup_pmf_without_context(args: argparse.Namespace) -> tuple[dict[str, Any], Callable[[CaseSpec, dict[str, Any], Path], dict[str, Any] | None]]:
    def run(case: CaseSpec, eval_case: dict[str, Any], aligned_dir: Path) -> dict[str, Any] | None:
        return pmf.score_case(
            eval_case,
            source_video_path=Path(case.source_video_path),
            context_mode="without_context",
            context_frames=case.context_frames,
            device=args.proxy_device,
            aligned_video_dir=aligned_dir,
        )

    return {}, run


def _make_vbench_setup(
    dimension: str,
) -> Callable[[argparse.Namespace], tuple[dict[str, Any], Callable[[CaseSpec, dict[str, Any], Path], dict[str, Any] | None]]]:
    def _setup(args: argparse.Namespace) -> tuple[dict[str, Any], Callable[[CaseSpec, dict[str, Any], Path], dict[str, Any] | None]]:
        runner = OfficialVBenchRunner(
            device=args.vbench_device,
            load_ckpt_from_local=args.vbench_load_ckpt_from_local,
            read_frame=args.vbench_read_frame,
            imaging_quality_preprocessing_mode=args.vbench_imaging_quality_preprocessing_mode,
        )

        def run(_: CaseSpec, eval_case: dict[str, Any], out_dir: Path) -> dict[str, Any] | None:
            return runner.score_case(
                eval_case,
                dimension=dimension,
                caption=eval_case["caption"],
                output_path=out_dir,
            )

        return {"runner": runner, "dimension": dimension}, run

    return _setup


def _cleanup_runtime(runtime: dict[str, Any]) -> None:
    runner = runtime.get("runner")
    if runner is not None:
        del runner
    gc.collect()


def _prepare_dataset_map(args: argparse.Namespace) -> dict[str, list[CaseSpec]]:
    old_cases = _load_old_cases(args)
    new_cases = _load_new_cases(args)
    if args.limit_per_dataset is not None:
        old_cases = old_cases[: max(0, int(args.limit_per_dataset))]
        new_cases = new_cases[: max(0, int(args.limit_per_dataset))]
    return {"old": old_cases, "new": new_cases}


def main() -> None:
    args = parse_args()
    metric_names = [token.strip() for token in args.metrics.split(",") if token.strip()]
    if not metric_names:
        raise ValueError("metrics list is empty")

    dataset_map = _prepare_dataset_map(args)
    output_root = args.output_root.expanduser().resolve()
    context_cache_root = output_root / "context_cache"
    per_case_root = output_root / "per_case"
    aligned_root = output_root / "aligned_pairs"
    summary_root = output_root / "summary"
    metadata_root = output_root / "metadata"
    output_root.mkdir(parents=True, exist_ok=True)

    _write_json(
        metadata_root / "run_config.json",
        {
            "old_root": str(args.old_root),
            "new_root": str(args.new_root),
            "old_split": args.old_split,
            "old_sampling_mode": args.old_sampling_mode,
            "seed": args.seed,
            "context_frames": args.context_frames,
            "metrics": metric_names,
            "cosmos_python": str(args.cosmos_python),
            "cosmos_cuda_visible_devices": args.cosmos_cuda_visible_devices,
            "note": (
                "Reference-based metrics in this script are run in self-reference mode for dataset clips. "
                "For new cases without an explicit source_video, source_video falls back to the case video itself."
            ),
        },
    )
    _write_json(metadata_root / "sampled_old_cases.json", [asdict(item) for item in dataset_map["old"]])
    _write_json(metadata_root / "sampled_new_cases.json", [asdict(item) for item in dataset_map["new"]])

    if args.dry_run:
        print(json.dumps({key: len(value) for key, value in dataset_map.items()}, ensure_ascii=False, indent=2))
        return

    metric_specs = _expand_requested_metric_specs(args, metric_names)
    case_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []

    for metric_spec in metric_specs:
        runtime, scorer = metric_spec.setup(args)
        try:
            per_metric_numeric: dict[tuple[str, str], list[float]] = {}
            per_metric_counts: dict[str, dict[str, int]] = {}
            for dataset_name, cases in dataset_map.items():
                per_metric_counts[dataset_name] = {"success": 0, "failed": 0}
                for case in cases:
                    eval_case = _build_eval_case(case, context_cache_root)
                    result_path = per_case_root / metric_spec.name / dataset_name / f"{case.case_id}.json"
                    aligned_dir = aligned_root / metric_spec.name / dataset_name / case.case_id
                    record = {
                        "metric": metric_spec.name,
                        "dataset": dataset_name,
                        "case_id": case.case_id,
                        "family_key": case.family_key,
                        "video": case.video_path,
                        "source_video": case.source_video_path,
                        "context_video": eval_case["context_video"],
                        "caption": case.caption,
                    }
                    try:
                        result = scorer(case, eval_case, aligned_dir)
                        if result is None:
                            raise RuntimeError("metric returned None")
                        numeric = _flatten_numeric_leaves(result)
                        record["status"] = "ok"
                        record["result"] = result
                        record["numeric"] = numeric
                        per_metric_counts[dataset_name]["success"] += 1
                        for field, value in numeric.items():
                            per_metric_numeric.setdefault((dataset_name, field), []).append(value)
                    except Exception as exc:
                        record["status"] = "error"
                        record["error"] = str(exc)
                        record["traceback"] = traceback.format_exc()
                        per_metric_counts[dataset_name]["failed"] += 1
                    _write_json(result_path, record)
                    case_rows.append(record)

            for dataset_name in sorted(dataset_map):
                counters = per_metric_counts[dataset_name]
                fields = sorted(field for (row_dataset, field) in per_metric_numeric if row_dataset == dataset_name)
                if not fields:
                    summary_rows.append(
                        {
                            "metric": metric_spec.name,
                            "dataset": dataset_name,
                            "field": "",
                            "count": 0,
                            "mean": 0.0,
                            "std": 0.0,
                            "min": 0.0,
                            "max": 0.0,
                            "success": counters["success"],
                            "failed": counters["failed"],
                        }
                    )
                    continue
                for field in fields:
                    values = per_metric_numeric[(dataset_name, field)]
                    stats = _stats(values)
                    summary_rows.append(
                        {
                            "metric": metric_spec.name,
                            "dataset": dataset_name,
                            "field": field,
                            "count": len(values),
                            "mean": round(stats["mean"], 6),
                            "std": round(stats["std"], 6),
                            "min": round(stats["min"], 6),
                            "max": round(stats["max"], 6),
                            "success": counters["success"],
                            "failed": counters["failed"],
                        }
                    )
        finally:
            _cleanup_runtime(runtime)

    _write_json(summary_root / "case_results.json", case_rows)
    _write_json(summary_root / "metric_summary_long.json", summary_rows)
    _write_csv(
        summary_root / "metric_summary_long.csv",
        summary_rows,
        ["metric", "dataset", "field", "count", "mean", "std", "min", "max", "success", "failed"],
    )

    wide_rows: dict[str, dict[str, Any]] = {}
    for row in summary_rows:
        dataset_name = str(row["dataset"])
        wide = wide_rows.setdefault(dataset_name, {"dataset": dataset_name})
        key = f"{row['metric']}__{row['field'] or 'no_numeric_field'}"
        wide[key] = row["mean"]
    wide_fieldnames = ["dataset"]
    for row in summary_rows:
        key = f"{row['metric']}__{row['field'] or 'no_numeric_field'}"
        if key not in wide_fieldnames:
            wide_fieldnames.append(key)
    _write_csv(summary_root / "metric_summary_wide.csv", list(wide_rows.values()), wide_fieldnames)

    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "old_cases": len(dataset_map["old"]),
                "new_cases": len(dataset_map["new"]),
                "metrics": metric_names,
                "summary_csv": str(summary_root / "metric_summary_long.csv"),
                "wide_csv": str(summary_root / "metric_summary_wide.csv"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
