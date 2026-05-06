#!/usr/bin/env python3
"""Re-run the currently displayed geometry-diagnostics cases with empty caption."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
DEFAULT_DISPLAY_ROOT = TRAIN0419_ROOT / "geometry_diagnostics" / "_debug_run_tracking"
DEFAULT_SOURCE_BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V")
DEFAULT_TARGET_BENCHMARK_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/stage0_V2V_nullcaption")
DEFAULT_PYTHON_BIN = Path("/data/gaoya/miniconda3/envs/wan/bin/python")
MODEL_NAME = "vace_v2v_ctx08f"
MODEL_RELATIVE_DIR = Path("output") / "VACE_1_3B_V2V" / "context_08f"
RUNTIME_RELATIVE_DIR = Path("tools") / "runtime" / MODEL_NAME
DEFAULT_VACE_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-VACE-1.3B")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--display_root", type=Path, default=DEFAULT_DISPLAY_ROOT)
    parser.add_argument("--source_benchmark_root", type=Path, default=DEFAULT_SOURCE_BENCHMARK_ROOT)
    parser.add_argument("--target_benchmark_root", type=Path, default=DEFAULT_TARGET_BENCHMARK_ROOT)
    parser.add_argument("--python_bin", type=Path, default=DEFAULT_PYTHON_BIN)
    parser.add_argument("--vace_root", type=Path, default=DEFAULT_VACE_ROOT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--only-sample", action="append", default=[])
    parser.add_argument("--prep-tag", default="nullcaption_rerun")
    parser.add_argument("--runtime-tag", default="")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def collect_selected_cases(display_root: Path) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for diagnostics_path in sorted(display_root.glob("*/diagnostics.json")):
        payload = read_json(diagnostics_path)
        dataset = payload.get("dataset")
        sample_id = payload.get("sample_id")
        if not isinstance(dataset, str) or not isinstance(sample_id, str):
            raise ValueError(f"diagnostics missing dataset/sample_id: {diagnostics_path}")
        selected.append((dataset, sample_id))
    if not selected:
        raise ValueError(f"no displayed cases found under {display_root}")
    return selected


def build_case_bundle(
    *,
    selected: list[tuple[str, str]],
    source_benchmark_root: Path,
    prep_root: Path,
    limit: int | None,
    only_samples: set[str],
) -> tuple[list[Path], list[dict[str, Any]], dict[str, Any]]:
    source_sidecar_root = source_benchmark_root / MODEL_RELATIVE_DIR
    temp_meta_root = prep_root / "temp_meta"

    selected_meta_paths: list[Path] = []
    manifest_rows: list[dict[str, Any]] = []

    for dataset, sample_id in selected:
        if only_samples and sample_id not in only_samples:
            continue
        sidecar_path = source_sidecar_root / f"{dataset}__{sample_id}.json"
        if not sidecar_path.is_file():
            raise FileNotFoundError(f"source sidecar not found: {sidecar_path}")
        sidecar = read_json(sidecar_path)
        paths = sidecar.get("paths", {})
        if not isinstance(paths, dict):
            raise ValueError(f"invalid paths payload in {sidecar_path}")
        meta_json_path = paths.get("meta_json_path")
        if not isinstance(meta_json_path, str) or not meta_json_path:
            raise ValueError(f"missing meta_json_path in {sidecar_path}")

        source_meta_path = Path(meta_json_path)
        source_meta = read_json(source_meta_path)
        source_caption = str(source_meta.get("caption") or source_meta.get("description") or "")
        source_meta["caption"] = ""
        if "description" in source_meta:
            source_meta["description"] = ""

        target_meta_path = temp_meta_root / dataset / f"{sample_id}.json"
        write_json(target_meta_path, source_meta)
        selected_meta_paths.append(target_meta_path)
        manifest_rows.append(
            {
                "dataset": dataset,
                "sample_id": sample_id,
                "source_sidecar_path": str(sidecar_path),
                "source_meta_json_path": str(source_meta_path),
                "temp_meta_json_path": str(target_meta_path),
                "source_caption": source_caption,
                "null_caption": "",
            }
        )
        if limit is not None and len(selected_meta_paths) >= limit:
            break

    if not selected_meta_paths:
        raise ValueError("selection became empty after applying filters")

    manifest = {
        "model_name": MODEL_NAME,
        "num_cases": len(selected_meta_paths),
        "source_benchmark_root": str(source_benchmark_root),
        "model_relative_dir": str(MODEL_RELATIVE_DIR),
        "cases": manifest_rows,
    }
    return selected_meta_paths, manifest_rows, manifest


def build_batch_eval_command(
    *,
    args: argparse.Namespace,
    meta_list_path: Path,
) -> list[str]:
    output_root = args.target_benchmark_root / MODEL_RELATIVE_DIR
    runtime_dir_name = MODEL_NAME if not args.runtime_tag else f"{MODEL_NAME}__{args.runtime_tag}"
    runtime_root = args.target_benchmark_root / "tools" / "runtime" / runtime_dir_name
    return [
        str(args.python_bin),
        str(TRAIN0419_ROOT / "batch_eval_vace.py"),
        "--vace_root",
        str(args.vace_root),
        "--meta_list_path",
        str(meta_list_path),
        "--output_root",
        str(output_root),
        "--runtime_root",
        str(runtime_root),
        "--model_name",
        MODEL_NAME,
        "--mode",
        "v2v_clipref",
        "--device",
        "cuda:0",
        "--height",
        "544",
        "--width",
        "720",
        "--fps",
        "16",
        "--num_frames",
        "49",
        "--context_frames",
        "8",
        "--num_inference_steps",
        "50",
        "--cfg_scale",
        "5.0",
        "--seed",
        "42",
    ] + (["--overwrite"] if args.overwrite else [])


def main() -> None:
    args = parse_args()
    args.display_root = args.display_root.expanduser().resolve()
    args.source_benchmark_root = args.source_benchmark_root.expanduser().resolve()
    args.target_benchmark_root = args.target_benchmark_root.expanduser().resolve()
    args.python_bin = args.python_bin.expanduser().resolve()
    args.vace_root = args.vace_root.expanduser().resolve()

    prep_root = args.target_benchmark_root / "tools" / args.prep_tag
    selected = collect_selected_cases(args.display_root)
    meta_paths, _, manifest = build_case_bundle(
        selected=selected,
        source_benchmark_root=args.source_benchmark_root,
        prep_root=prep_root,
        limit=args.limit,
        only_samples={sample_id for sample_id in args.only_sample},
    )

    meta_list_path = prep_root / "displayed_cases_nullcaption_meta_list.txt"
    write_text(meta_list_path, "\n".join(str(path) for path in meta_paths) + "\n")
    manifest["meta_list_path"] = str(meta_list_path)
    write_json(prep_root / "selection_manifest.json", manifest)

    runtime_dir_name = MODEL_NAME if not args.runtime_tag else f"{MODEL_NAME}__{args.runtime_tag}"
    runtime_root = args.target_benchmark_root / "tools" / "runtime" / runtime_dir_name
    command = build_batch_eval_command(args=args, meta_list_path=meta_list_path)
    write_text(prep_root / "last_command.sh", " ".join(command) + "\n")

    print(f"prepared_cases={len(meta_paths)}")
    print(f"meta_list_path={meta_list_path}")
    print(f"output_root={args.target_benchmark_root / MODEL_RELATIVE_DIR}")
    print(f"runtime_root={runtime_root}")
    if args.dry_run:
        print("dry_run=1")
        return

    subprocess.run(command, check=True)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"command failed with exit code {exc.returncode}", file=sys.stderr)
        raise
