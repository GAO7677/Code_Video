#!/usr/bin/env python3
"""Build the 27-method benchmark tree after attention-noise inference finishes."""

from __future__ import annotations

import json
import os
from pathlib import Path


BENCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_probability_noise_metrics_test5"
)
OLD_BENCH_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme_benchmark_test5_ready"
)
BASELINE_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_additive_noise_baseline_test5"
)
LORA_ROOT = Path(
    "/data/gaoya/agent-data/outputs/attention_additive_noise_lora_001460"
)
FULL_SA_ROOT = Path(
    "/data/gaoya/agent-data/outputs/full_sa_no_object_step2500_attention_probability_noise"
)

MODEL_SPECS = (
    ("wan22_baseline", "baseline", BASELINE_ROOT),
    ("wan_lora", "lora", LORA_ROOT),
    ("full_sa_no_object_step2500", "full_sa", FULL_SA_ROOT),
)
PERTURBATIONS = tuple(
    (rank, count, alpha)
    for count in (30, 100)
    for rank in ("top", "bottom")
    for alpha in (0.9, 1.5)
)


def alpha_dir(alpha: float) -> str:
    return f"alpha{int(round(alpha * 100)):03d}"


def alpha_token(alpha: float) -> str:
    return str(alpha).replace(".", "p")


def method_specs() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for model_slug, model_kind, source_root in MODEL_SPECS:
        specs.append(
            {
                "name": f"{model_slug}_original",
                "model_slug": model_slug,
                "model_kind": model_kind,
                "source_root": source_root,
                "rank": "original",
                "count": None,
                "alpha": None,
            }
        )
        for rank, count, alpha in PERTURBATIONS:
            specs.append(
                {
                    "name": f"{model_slug}_{rank}{count}_alpha{alpha_token(alpha)}",
                    "model_slug": model_slug,
                    "model_kind": model_kind,
                    "source_root": source_root,
                    "rank": rank,
                    "count": count,
                    "alpha": alpha,
                }
            )
    return specs


def resolve_source(spec: dict[str, object], case: str) -> Path:
    model_kind = str(spec["model_kind"])
    source_root = Path(spec["source_root"])
    rank = str(spec["rank"])
    count = spec["count"]
    alpha = spec["alpha"]

    if model_kind in {"baseline", "lora"}:
        if rank == "original":
            experiment_dir = source_root / "alpha090_count30"
            filename = "original.mp4"
        else:
            experiment_dir = source_root / f"{alpha_dir(float(alpha))}_count{count}"
            filename = f"{rank}{count}_steps_00_40.mp4"
        return (
            experiment_dir
            / "videos"
            / model_kind
            / "cases"
            / case
            / filename
        )

    if rank == "original":
        candidates = sorted(source_root.glob("*step-002500*baseline"))
    else:
        suffix = f"_{rank}{count}_alpha{alpha_token(float(alpha))}"
        candidates = sorted(source_root.glob(f"*step-002500*{suffix}"))
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected one Full-SA directory for {spec['name']}, found {len(candidates)}: "
            f"{[str(path) for path in candidates]}"
        )
    return candidates[0] / f"{case}.mp4"


def load_case_metadata(case: str, input_json: Path) -> dict[str, object]:
    old_json = OLD_BENCH_ROOT / "methods" / "wan22_baseline_original" / f"{case}.json"
    old_data: dict[str, object] = {}
    if old_json.is_file():
        old_data = json.loads(old_json.read_text())

    prompt = old_data.get("prompt") or old_data.get("caption") or case
    return {
        "input_json": str(input_json),
        "case_json": str(input_json),
        "context_frames": int(old_data.get("context_frames", 8)),
        "effective_context_frames": int(old_data.get("effective_context_frames", 8)),
        "prompt": prompt,
        "caption": old_data.get("caption") or prompt,
    }


def atomic_json(path: Path, payload: object) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n")
    os.replace(temp, path)


def main() -> None:
    allowlist_source = OLD_BENCH_ROOT / "all_conditions_input_json_allowlist.txt"
    input_jsons = [
        Path(line.strip())
        for line in allowlist_source.read_text().splitlines()
        if line.strip()
    ]
    cases = [(path.stem, path) for path in input_jsons]
    specs = method_specs()

    missing: list[str] = []
    source_map: dict[tuple[str, str], Path] = {}
    for spec in specs:
        for case, _ in cases:
            source = resolve_source(spec, case)
            source_map[(str(spec["name"]), case)] = source
            if not source.is_file():
                missing.append(f"{spec['name']} | {case} | {source}")
    if missing:
        preview = "\n".join(missing[:40])
        raise RuntimeError(
            f"Benchmark preparation stopped: {len(missing)} expected videos are missing.\n{preview}"
        )

    methods_root = BENCH_ROOT / "methods"
    methods_root.mkdir(parents=True, exist_ok=True)
    allowlist_target = BENCH_ROOT / "input_json_allowlist.txt"
    allowlist_target.write_text("\n".join(str(path) for _, path in cases) + "\n")

    method_paths: list[Path] = []
    for spec in specs:
        method_name = str(spec["name"])
        method_dir = methods_root / method_name
        method_dir.mkdir(parents=True, exist_ok=True)
        method_paths.append(method_dir)

        for case, input_json in cases:
            source = source_map[(method_name, case)]
            video_link = method_dir / f"{case}.mp4"
            if video_link.is_symlink() or video_link.exists():
                if video_link.resolve() != source.resolve():
                    video_link.unlink()
            if not video_link.exists():
                video_link.symlink_to(source)

            result_json = method_dir / f"{case}.json"
            existing: dict[str, object] = {}
            if result_json.is_file():
                existing = json.loads(result_json.read_text())
            metadata = load_case_metadata(case, input_json)
            metadata.update(
                {
                    "output_video": str(video_link),
                    "source_experiment_video": str(source),
                    "method": method_name,
                    "model": spec["model_slug"],
                    "ablation_variant": spec["rank"],
                    "head_count": spec["count"],
                    "attention_noise_alpha": spec["alpha"],
                    "attention_noise_mode": "probability_additive_renormalized",
                }
            )
            existing.update(metadata)
            atomic_json(result_json, existing)

        atomic_json(
            method_dir / "batch_manifest.json",
            {
                "input_json_list_path": str(allowlist_target),
                "num_cases": len(cases),
                "comparison_policy": "shared_20_case_test5_matrix",
                "method": method_name,
            },
        )

    methods_file = BENCH_ROOT / "bench_methods.txt"
    methods_file.write_text("\n".join(str(path) for path in method_paths) + "\n")
    atomic_json(
        BENCH_ROOT / "prepared_manifest.json",
        {
            "status": "ready",
            "methods_file": str(methods_file),
            "input_json_allowlist": str(allowlist_target),
            "num_methods": len(specs),
            "num_cases": len(cases),
            "num_method_cases": len(specs) * len(cases),
            "cases": [case for case, _ in cases],
            "methods": specs,
        },
    )
    (BENCH_ROOT / "PREPARED").write_text("ready\n")
    print(f"Prepared {len(specs)} methods x {len(cases)} cases at {BENCH_ROOT}")


if __name__ == "__main__":
    main()
