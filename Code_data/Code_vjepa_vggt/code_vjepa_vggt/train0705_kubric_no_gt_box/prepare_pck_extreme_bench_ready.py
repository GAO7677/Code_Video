#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any


DEFAULT_TOP30_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme30_all720_head_zero_ablation_test5"
)
DEFAULT_TOP100_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme100_all720_head_zero_ablation_test5"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/pck_extreme_benchmark_test5_ready"
)

MODELS = ("baseline", "lora")
VARIANTS = {
    "original": ("top30", "original.mp4"),
    "top30_steps_00_40": ("top30", "top30_steps_00_40.mp4"),
    "bottom30_steps_00_40": ("top30", "bottom30_steps_00_40.mp4"),
    "top100_steps_00_40": ("top100", "top100_steps_00_40.mp4"),
    "bottom100_steps_00_40": ("top100", "bottom100_steps_00_40.mp4"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build bench.sh-compatible result roots for all currently complete "
            "baseline/LoRA Top30, Bottom30, Top100 and Bottom100 cases."
        )
    )
    parser.add_argument("--top30-root", type=Path, default=DEFAULT_TOP30_ROOT)
    parser.add_argument("--top100-root", type=Path, default=DEFAULT_TOP100_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    temporary.replace(path)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def method_name(model: str, variant: str) -> str:
    model_label = "wan22_baseline" if model == "baseline" else "wan_lora"
    return f"{model_label}_{variant}"


def source_path(
    roots: dict[str, Path], model: str, case_key: str, variant: str
) -> Path:
    root_name, filename = VARIANTS[variant]
    return roots[root_name] / model / "cases" / case_key / filename


def discover_complete_cases(roots: dict[str, Path]) -> dict[str, list[str]]:
    cases_by_model: dict[str, list[str]] = {}
    for model in MODELS:
        cases_root = roots["top100"] / model / "cases"
        candidates = {
            path.name for path in cases_root.iterdir() if path.is_dir()
        } if cases_root.is_dir() else set()
        complete = {
            case_key
            for case_key in candidates
            if all(
                source_path(roots, model, case_key, variant).is_file()
                for variant in VARIANTS
            )
        }
        cases_by_model[model] = sorted(complete)
    return cases_by_model


def ensure_video_link(link_path: Path, source: Path) -> None:
    expected = source.resolve()
    if link_path.is_symlink() and link_path.resolve() == expected:
        return
    if link_path.exists() or link_path.is_symlink():
        link_path.unlink()
    temporary = link_path.with_name(f".{link_path.name}.tmp")
    if temporary.exists() or temporary.is_symlink():
        temporary.unlink()
    os.symlink(expected, temporary)
    temporary.replace(link_path)


def prepare_result(
    *,
    roots: dict[str, Path],
    output_root: Path,
    model: str,
    variant: str,
    case_key: str,
) -> Path:
    source = source_path(roots, model, case_key, variant)
    manifest_path = roots["top30"] / model / "cases" / case_key / "manifest.json"
    if not manifest_path.is_file():
        manifest_path = roots["top100"] / model / "cases" / case_key / "manifest.json"
    manifest = load_json(manifest_path)
    input_json = Path(str(manifest["input_json"])).expanduser().resolve()
    if not input_json.is_file():
        raise FileNotFoundError(f"Missing input JSON: {input_json}")

    method = method_name(model, variant)
    result_root = output_root / "methods" / method
    result_root.mkdir(parents=True, exist_ok=True)
    result_json = result_root / f"{case_key}.json"
    result_video = result_root / f"{case_key}.mp4"
    ensure_video_link(result_video, source)

    # Preserve metric fields from previous incremental benchmark runs.
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
    atomic_write_json(result_json, payload)
    return input_json


def main() -> None:
    args = parse_args()
    roots = {
        "top30": args.top30_root.expanduser().resolve(),
        "top100": args.top100_root.expanduser().resolve(),
    }
    output_root = args.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    cases_by_model = discover_complete_cases(roots)
    if not all(cases_by_model.values()):
        raise RuntimeError("No case is complete across both models and all five conditions")

    method_roots: list[Path] = []
    input_jsons: set[Path] = set()
    for model in MODELS:
        for variant in VARIANTS:
            method = method_name(model, variant)
            result_root = output_root / "methods" / method
            method_roots.append(result_root)
            for case_key in cases_by_model[model]:
                input_jsons.add(
                    prepare_result(
                        roots=roots,
                        output_root=output_root,
                        model=model,
                        variant=variant,
                        case_key=case_key,
                    )
                )

    allowlist_path = output_root / "input_json_allowlist.txt"
    methods_path = output_root / "bench_methods.txt"
    atomic_write_text(
        allowlist_path,
        "".join(f"{path}\n" for path in sorted(input_jsons)),
    )
    atomic_write_text(
        methods_path,
        "".join(f"{path.resolve()}\n" for path in method_roots),
    )
    for result_root in method_roots:
        model = "baseline" if result_root.name.startswith("wan22_baseline_") else "lora"
        atomic_write_json(
            result_root / "batch_manifest.json",
            {
                "input_json_list_path": str(allowlist_path.resolve()),
                "num_cases": len(cases_by_model[model]),
                "comparison_policy": "per_model_intersection_across_five_conditions",
            },
        )

    unique_cases = sorted(set().union(*map(set, cases_by_model.values())))
    num_method_cases = sum(len(cases) * len(VARIANTS) for cases in cases_by_model.values())

    atomic_write_json(
        output_root / "prepared_manifest.json",
        {
            "status": "ready",
            "top30_root": str(roots["top30"]),
            "top100_root": str(roots["top100"]),
            "methods_file": str(methods_path.resolve()),
            "input_json_allowlist": str(allowlist_path.resolve()),
            "num_methods": len(method_roots),
            "num_cases": len(unique_cases),
            "num_method_cases": num_method_cases,
            "cases": unique_cases,
            "cases_by_model": cases_by_model,
            "methods": [path.name for path in method_roots],
        },
    )
    print(
        json.dumps(
            {
                "output_root": str(output_root),
                "methods_file": str(methods_path),
                "input_json_allowlist": str(allowlist_path),
                "num_methods": len(method_roots),
                "num_cases": len(unique_cases),
                "cases_by_model": {model: len(cases) for model, cases in cases_by_model.items()},
                "num_method_cases": num_method_cases,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
