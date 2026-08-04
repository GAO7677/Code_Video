#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


STATIC30_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme30_all720_head_zero_ablation_test5"
)
STATIC100_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme100_all720_head_zero_ablation_test5"
)
ADAPTIVE30_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "pck_step_adaptive_top30_bottom30_all720_head_zero_ablation_test5"
)
OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme_benchmark_test5_ready"
)

MODELS = ("baseline", "lora")
NEW_VARIANTS = {
    "top30_steps_00_10": (STATIC30_ROOT, "top30_steps_00_10.mp4"),
    "top30_steps_10_20": (STATIC30_ROOT, "top30_steps_10_20.mp4"),
    "top30_steps_20_30": (STATIC30_ROOT, "top30_steps_20_30.mp4"),
    "top30_steps_30_40": (STATIC30_ROOT, "top30_steps_30_40.mp4"),
    "bottom30_steps_00_10": (STATIC30_ROOT, "bottom30_steps_00_10.mp4"),
    "bottom30_steps_10_20": (STATIC30_ROOT, "bottom30_steps_10_20.mp4"),
    "bottom30_steps_20_30": (STATIC30_ROOT, "bottom30_steps_20_30.mp4"),
    "bottom30_steps_30_40": (STATIC30_ROOT, "bottom30_steps_30_40.mp4"),
    "top100_steps_00_10": (STATIC100_ROOT, "top100_steps_00_10.mp4"),
    "top100_steps_10_20": (STATIC100_ROOT, "top100_steps_10_20.mp4"),
    "top100_steps_20_30": (STATIC100_ROOT, "top100_steps_20_30.mp4"),
    "top100_steps_30_40": (STATIC100_ROOT, "top100_steps_30_40.mp4"),
    "bottom100_steps_00_10": (STATIC100_ROOT, "bottom100_steps_00_10.mp4"),
    "bottom100_steps_10_20": (STATIC100_ROOT, "bottom100_steps_10_20.mp4"),
    "bottom100_steps_20_30": (STATIC100_ROOT, "bottom100_steps_20_30.mp4"),
    "bottom100_steps_30_40": (STATIC100_ROOT, "bottom100_steps_30_40.mp4"),
    "adaptive_top30_steps_00_40": (ADAPTIVE30_ROOT, "top30_steps_00_40.mp4"),
    "adaptive_bottom30_steps_00_40": (ADAPTIVE30_ROOT, "bottom30_steps_00_40.mp4"),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return payload


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def atomic_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def method_name(model: str, variant: str) -> str:
    prefix = "wan22_baseline" if model == "baseline" else "wan_lora"
    return f"{prefix}_{variant}"


def ensure_link(link: Path, source: Path) -> None:
    expected = source.resolve()
    if link.is_symlink() and link.resolve() == expected:
        return
    if link.exists() or link.is_symlink():
        link.unlink()
    temporary = link.with_name(f".{link.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    os.symlink(expected, temporary)
    temporary.replace(link)


def canonical_cases(model: str) -> list[str]:
    root = STATIC100_ROOT / model / "cases"
    return sorted(path.name for path in root.iterdir() if path.is_dir())


def prepare_case(model: str, variant: str, case_key: str) -> Path | None:
    source_root, filename = NEW_VARIANTS[variant]
    source = source_root / model / "cases" / case_key / filename
    if not source.is_file() or source.stat().st_size == 0:
        return None
    manifest_path = STATIC30_ROOT / model / "cases" / case_key / "manifest.json"
    manifest = load_json(manifest_path)
    input_json = Path(str(manifest["input_json"])).expanduser().resolve()
    method = method_name(model, variant)
    result_root = OUTPUT_ROOT / "methods" / method
    result_root.mkdir(parents=True, exist_ok=True)
    result_json = result_root / f"{case_key}.json"
    result_video = result_root / f"{case_key}.mp4"
    ensure_link(result_video, source)
    payload = load_json(result_json) if result_json.is_file() else {}
    payload.update(
        {
            "input_json": str(input_json),
            "case_json": str(input_json),
            "output_video": str(result_video.resolve()),
            "method": method,
            "model": model,
            "ablation_variant": variant,
            "context_frames": 8,
            "effective_context_frames": 8,
            "source_experiment_video": str(source.resolve()),
        }
    )
    prompt = manifest.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        payload["prompt"] = prompt.strip()
        payload["caption"] = prompt.strip()
    atomic_json(result_json, payload)
    return input_json


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    methods: list[Path] = []
    input_jsons: set[Path] = set()
    counts: dict[str, int] = {}
    for model in MODELS:
        cases = canonical_cases(model)
        for variant in NEW_VARIANTS:
            method = method_name(model, variant)
            result_root = OUTPUT_ROOT / "methods" / method
            result_root.mkdir(parents=True, exist_ok=True)
            methods.append(result_root)
            prepared = 0
            for case_key in cases:
                input_json = prepare_case(model, variant, case_key)
                if input_json is not None:
                    input_jsons.add(input_json)
                    prepared += 1
            counts[method] = prepared

    allowlist = OUTPUT_ROOT / "all_conditions_input_json_allowlist.txt"
    methods_file = OUTPUT_ROOT / "bench_new_conditions_methods.txt"
    atomic_text(allowlist, "".join(f"{path}\n" for path in sorted(input_jsons)))
    atomic_text(methods_file, "".join(f"{path.resolve()}\n" for path in methods))
    for result_root in methods:
        atomic_json(
            result_root / "batch_manifest.json",
            {
                "input_json_list_path": str(allowlist.resolve()),
                "expected_cases": 20,
                "comparison_policy": "all_test5_cases_when_source_video_is_ready",
            },
        )
    atomic_json(
        OUTPUT_ROOT / "all_conditions_prepared_manifest.json",
        {
            "status": "ready",
            "methods_file": str(methods_file.resolve()),
            "input_json_allowlist": str(allowlist.resolve()),
            "num_new_methods": len(methods),
            "prepared_method_cases": sum(counts.values()),
            "method_case_counts": counts,
        },
    )
    print(
        json.dumps(
            {
                "methods_file": str(methods_file),
                "num_new_methods": len(methods),
                "prepared_method_cases": sum(counts.values()),
                "adaptive_ready": sum(
                    count for method, count in counts.items() if "adaptive_" in method
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
