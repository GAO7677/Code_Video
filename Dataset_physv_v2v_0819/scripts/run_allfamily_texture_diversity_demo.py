"""Render a few high-diversity PBR variants for every original 0717 family.

This is an all-family extension of the existing F2/F3 texture-diversity
pipeline.  It selects a small deterministic subset from the original
``pybullet0717`` manifest, stages each variant through the same PyBullet
reconstruction, and then renders with the existing fast Eevee/PBR Blender
pass.  The original dataset is read-only.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

from generate_texture_diversity_demo import _stage_and_render_job


SOURCE_ROOT_DEFAULT = Path(
    "/data/gaoya/agent-data/datasets/pybullet0717_prompt_physics_consistency_v1"
)
OUTPUT_ROOT_DEFAULT = SOURCE_ROOT_DEFAULT / (
    "external_allfamily_texture_realism_demo_20260825"
)
BLENDER_DEFAULT = Path("/data/gaoya/agent-data/tools/blender-3.6.23-linux-x64/blender")
FFMPEG_DEFAULT = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/ffmpeg")
BACKGROUND_PROFILES = (
    "warehouse_cobalt",
    "machine_shop_amber",
    "color_studio",
    "glasshouse_mint",
    "courtyard_terracotta",
    "foundry_safety",
    "garage_teal",
    "neon_studio",
)


def _load_jobs(
    source_root: Path,
    *,
    cases_per_family: int,
    variants_per_case: int,
) -> list[dict[str, Any]]:
    if cases_per_family <= 0 or variants_per_case <= 0:
        raise ValueError("cases-per-family and variants-per-case must be positive")
    manifest = json.loads((source_root / "manifest.json").read_text(encoding="utf-8"))
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in manifest:
        family = str(item.get("family_key", ""))
        if family:
            by_family[family].append(item)

    selected: list[dict[str, Any]] = []
    job_index = 0
    for family in sorted(by_family):
        family_items = sorted(
            by_family[family],
            key=lambda item: (int(item.get("attempt_index", 0)), str(item.get("case_id", ""))),
        )[:cases_per_family]
        if len(family_items) < cases_per_family:
            raise RuntimeError(f"{family}: only {len(family_items)} source cases available")
        for item in family_items:
            case_id = str(item["case_id"])
            relative_case = f"{family}/{case_id}"
            for variant_index in range(variants_per_case):
                selected.append(
                    {
                        "job_index": job_index,
                        "family_key": family,
                        "source_case_id": case_id,
                        "relative_case": relative_case,
                        "variant_index": variant_index,
                        "background_profile": BACKGROUND_PROFILES[
                            job_index % len(BACKGROUND_PROFILES)
                        ],
                    }
                )
                job_index += 1
    return selected


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _worker(args: argparse.Namespace) -> int:
    output_root = args.output_root.resolve()
    jobs = json.loads((output_root / "job_plan.json").read_text(encoding="utf-8"))
    selected = [
        item for item in jobs if int(item["job_index"]) % args.worker_count == args.worker_index
    ]
    completed: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    print(
        f"worker={args.worker_index}/{args.worker_count} gpu={args.gpu} "
        f"jobs={[item['job_index'] for item in selected]}",
        flush=True,
    )
    for item in selected:
        job_index = int(item["job_index"])
        print(
            f"START job={job_index} family={item['family_key']} "
            f"source={item['source_case_id']} background={item['background_profile']} "
            f"gpu={args.gpu}",
            flush=True,
        )
        try:
            result = _stage_and_render_job(
                job_index=job_index,
                relative_case=str(item["relative_case"]),
                background_profile=str(item["background_profile"]),
                source_root=args.source_root.resolve(),
                output_root=output_root,
                blender=args.blender.resolve(),
                ffmpeg=args.ffmpeg.resolve(),
                gpu=str(args.gpu),
                exposure=float(args.exposure),
                samples=int(args.samples),
                camera_distance_scale=float(args.camera_distance_scale),
                stage_width=int(args.stage_width),
                stage_height=int(args.stage_height),
                seed=int(args.seed),
                worker_index=int(args.worker_index),
            )
            result.update(
                {
                    "variant_index": int(item["variant_index"]),
                    "source_family_key": str(item["family_key"]),
                }
            )
            completed.append(result)
            print(f"DONE job={job_index} video={result['video']}", flush=True)
        except Exception as exc:  # keep independent family jobs running
            error = {
                "job_index": job_index,
                "family_key": item["family_key"],
                "source_case_id": item["source_case_id"],
                "error": repr(exc),
            }
            errors.append(error)
            print(f"ERROR job={job_index}: {exc!r}", flush=True)
    _write_json(
        output_root / "_worker_jobs" / f"worker_{args.worker_index:02d}.json",
        {"worker_index": args.worker_index, "jobs": completed, "errors": errors},
    )
    print(
        f"worker_done={args.worker_index} completed={len(completed)} errors={len(errors)}",
        flush=True,
    )
    return 0 if not errors else 1


def _finalize(output_root: Path, source_root: Path, planned_jobs: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for path in sorted((output_root / "_worker_jobs").glob("worker_*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload.get("jobs", []))
        errors.extend(payload.get("errors", []))
    records.sort(key=lambda item: str(item.get("sample_key", "")))
    _write_json(output_root / "manifest.json", records)
    family_counts: dict[str, int] = defaultdict(int)
    for record in records:
        family_counts[str(record.get("family_key", ""))] += 1
    readme = {
        "purpose": "high-diversity PBR/Eevee renders for every original 0717 family",
        "source_root": str(source_root),
        "cases_per_family": int(args.cases_per_family),
        "variants_per_case": int(args.variants_per_case),
        "planned_jobs": len(planned_jobs),
        "completed_jobs": len(records),
        "errors": errors,
        "family_counts": dict(sorted(family_counts.items())),
        "render_engine": "BLENDER_EEVEE",
        "cycles_used": False,
        "changed_variables": [
            "true_geometry_size",
            "mass_constant_density",
            "object_display_palette",
            "object_pbr_texture_asset_and_mapping",
            "background_tint",
            "background_pbr_texture_mapping",
            "background_hdri_rotation",
        ],
        "unchanged_physics_coefficients": [
            "gravity",
            "friction",
            "restitution",
            "linear_damping",
            "angular_damping",
        ],
    }
    _write_json(output_root / "README.json", readme)
    _write_json(output_root / "reports" / "failure_report.json", errors)
    _write_json(output_root / "reports" / "summary.json", readme)
    print(f"finalized={len(records)} errors={len(errors)} output_root={output_root}", flush=True)
    return readme


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=SOURCE_ROOT_DEFAULT)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT_DEFAULT)
    parser.add_argument("--blender", type=Path, default=BLENDER_DEFAULT)
    parser.add_argument("--ffmpeg", type=Path, default=FFMPEG_DEFAULT)
    parser.add_argument("--seed", type=int, default=20260825)
    parser.add_argument("--gpu-list", default="0,1,2")
    parser.add_argument("--workers-per-gpu", type=int, default=2)
    parser.add_argument("--cases-per-family", type=int, default=2)
    parser.add_argument("--variants-per-case", type=int, default=2)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--exposure", type=float, default=-0.15)
    parser.add_argument("--camera-distance-scale", type=float, default=1.0)
    parser.add_argument("--stage-width", type=int, default=320)
    parser.add_argument("--stage-height", type=int, default=180)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--worker-index", type=int)
    parser.add_argument("--worker-count", type=int)
    parser.add_argument("--gpu")
    args = parser.parse_args()

    args.source_root = args.source_root.resolve()
    args.output_root = args.output_root.resolve()
    if args.worker_index is not None:
        if args.worker_count is None or args.gpu is None:
            raise ValueError("worker mode requires --worker-count and --gpu")
        raise SystemExit(_worker(args))

    gpus = [item.strip() for item in args.gpu_list.split(",") if item.strip()]
    if not gpus or any(item == "4" for item in gpus):
        raise ValueError("gpu-list must be non-empty and must not include GPU 4")
    if args.workers_per_gpu <= 0:
        raise ValueError("workers-per-gpu must be positive")
    if args.overwrite and args.output_root.exists():
        shutil.rmtree(args.output_root)
    args.output_root.mkdir(parents=True, exist_ok=True)
    jobs = _load_jobs(
        args.source_root,
        cases_per_family=args.cases_per_family,
        variants_per_case=args.variants_per_case,
    )
    _write_json(args.output_root / "job_plan.json", jobs)
    worker_count = len(gpus) * args.workers_per_gpu
    processes: list[tuple[int, str, subprocess.Popen[Any], Any]] = []
    for worker_index in range(worker_count):
        gpu = gpus[worker_index % len(gpus)]
        log_path = args.output_root / "logs" / f"supervisor_worker_{worker_index:02d}_gpu{gpu}.log"
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log_handle = log_path.open("w", encoding="utf-8")
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--source-root", str(args.source_root),
            "--output-root", str(args.output_root),
            "--blender", str(args.blender),
            "--ffmpeg", str(args.ffmpeg),
            "--seed", str(args.seed),
            "--samples", str(args.samples),
            "--exposure", str(args.exposure),
            "--camera-distance-scale", str(args.camera_distance_scale),
            "--stage-width", str(args.stage_width),
            "--stage-height", str(args.stage_height),
            "--cases-per-family", str(args.cases_per_family),
            "--variants-per-case", str(args.variants_per_case),
            "--worker-index", str(worker_index),
            "--worker-count", str(worker_count),
            "--gpu", gpu,
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu
        process = subprocess.Popen(command, env=env, stdout=log_handle, stderr=subprocess.STDOUT)
        processes.append((worker_index, gpu, process, log_handle))
        print(f"launch worker={worker_index}/{worker_count} gpu={gpu} log={log_path}", flush=True)

    failed = False
    for worker_index, gpu, process, log_handle in processes:
        return_code = process.wait()
        log_handle.close()
        print(f"worker_done worker={worker_index} gpu={gpu} returncode={return_code}", flush=True)
        failed = failed or return_code != 0
    summary = _finalize(args.output_root, args.source_root, jobs, args)
    if failed or summary["errors"]:
        raise SystemExit("one or more all-family jobs failed; inspect reports/failure_report.json")


if __name__ == "__main__":
    main()
