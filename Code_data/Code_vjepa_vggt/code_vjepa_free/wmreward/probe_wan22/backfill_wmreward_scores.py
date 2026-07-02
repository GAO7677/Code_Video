#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from pipeline_common import (
    DEFAULT_PIPELINE_ROOT,
    DEFAULT_WAN_PYTHON,
    DEFAULT_WMREWARD_CHECKPOINT,
    DEFAULT_WMREWARD_MODEL_NAME,
    MODEL_SPECS,
    WMREWARD_ROOT,
    generation_registry_fieldnames,
    load_json,
    parse_model_keys,
    read_csv_rows,
    resolve_python_bin,
    scan_generated_records,
    write_csv_rows,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run WMReward scoring on generated outputs, then write the surprise/similarity fields back into each result JSON."
        )
    )
    parser.add_argument("--pipeline-root", type=Path, default=DEFAULT_PIPELINE_ROOT)
    parser.add_argument("--models", default="base,openvid_lora,pybullet_lora")
    parser.add_argument("--python-bin", default=str(DEFAULT_WAN_PYTHON))
    parser.add_argument("--wmreward-checkpoint-path", type=Path, default=DEFAULT_WMREWARD_CHECKPOINT)
    parser.add_argument("--wmreward-model-name", default=DEFAULT_WMREWARD_MODEL_NAME)
    parser.add_argument("--window-size", type=int, default=16)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=49)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def run_wmreward_subprocess(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def update_result_json(
    *,
    result_json_path: Path,
    row: dict[str, str],
    pipeline_root: Path,
    seed: int,
) -> None:
    payload = load_json(result_json_path)
    surprise_value = row.get("surprise_score", "")
    similarity_value = row.get("similarity_score", "")
    wmreward_payload = {
        "surprise": float(surprise_value) if surprise_value else None,
        "similarity": float(similarity_value) if similarity_value else None,
        "method": "batch_compute_wmreward.py",
        "model": row["model_name"],
        "img_size": None,
        "window_size": int(row["window_size"]) if row.get("window_size") else None,
        "context_frames": int(row["context_frames"]) if row.get("context_frames") else None,
        "stride": int(row["stride"]) if row.get("stride") else None,
        "seed": seed,
        "checkpoint_path": row["checkpoint_path"],
        "status": row["status"],
        "error": row["error"],
        "relative_path": row["relative_path"],
        "json_path": row["json_path"],
        "video_path": row["video_path"],
        "pipeline_root": str(pipeline_root),
    }
    payload["wmreward"] = wmreward_payload
    payload["surprise_score"] = float(surprise_value) if surprise_value else None
    payload["similarity_score"] = float(similarity_value) if similarity_value else None
    write_json(result_json_path, payload)


def main() -> None:
    args = parse_args()
    pipeline_root = args.pipeline_root.expanduser().resolve()
    python_bin = resolve_python_bin(args.python_bin)
    selected_model_keys = parse_model_keys(args.models)
    all_rows: list[dict[str, object]] = []

    for model_key in selected_model_keys:
        spec = MODEL_SPECS[model_key]
        model_output_root = (pipeline_root / spec.output_subdir).resolve()
        score_output_dir = pipeline_root / "wmreward" / model_key
        output_name = "wmreward_scores.csv"
        cmd = [
            str(python_bin),
            str(WMREWARD_ROOT / "batch_compute_wmreward.py"),
            "--input_root",
            str(model_output_root),
            "--output_dir",
            str(score_output_dir),
            "--checkpoint_path",
            str(args.wmreward_checkpoint_path.expanduser().resolve()),
            "--model_name",
            args.wmreward_model_name,
            "--window_size",
            str(args.window_size),
            "--context_frames",
            str(args.context_frames),
            "--stride",
            str(args.stride),
            "--max_frames",
            str(args.max_frames),
            "--device",
            args.device,
            "--seed",
            str(args.seed),
            "--output_name",
            output_name,
        ]
        if args.limit is not None:
            cmd.extend(["--limit", str(args.limit)])
        print(f"[wmreward] model={model_key} input_root={model_output_root}")
        print(" ".join(cmd), flush=True)
        run_wmreward_subprocess(cmd)

        score_csv_path = score_output_dir / output_name
        score_rows = read_csv_rows(score_csv_path)
        for row in score_rows:
            result_json_path = Path(row["json_path"]).expanduser().resolve()
            if result_json_path.is_file():
                update_result_json(
                    result_json_path=result_json_path,
                    row=row,
                    pipeline_root=pipeline_root,
                    seed=args.seed,
                )

        registry_input_rows = read_csv_rows(pipeline_root / "manifests" / f"generation_registry_{model_key}.csv")
        input_json_paths = [Path(row["input_json_path"]).expanduser().resolve() for row in registry_input_rows]
        refreshed_rows = scan_generated_records(
            model_spec=spec,
            input_json_paths=input_json_paths,
            pipeline_root=pipeline_root,
        )
        write_csv_rows(
            pipeline_root / "manifests" / f"generation_registry_{model_key}.csv",
            refreshed_rows,
            generation_registry_fieldnames(),
        )
        all_rows.extend(refreshed_rows)

    write_csv_rows(
        pipeline_root / "manifests" / "generation_registry_all.csv",
        all_rows,
        generation_registry_fieldnames(),
    )
    write_json(
        pipeline_root / "manifests" / "wmreward_run_config.json",
        {
            "pipeline_root": str(pipeline_root),
            "selected_models": selected_model_keys,
            "wmreward_checkpoint_path": str(args.wmreward_checkpoint_path.expanduser().resolve()),
            "wmreward_model_name": args.wmreward_model_name,
            "window_size": args.window_size,
            "context_frames": args.context_frames,
            "stride": args.stride,
            "max_frames": args.max_frames,
            "seed": args.seed,
            "device": args.device,
        },
    )
    print(pipeline_root / "manifests" / "generation_registry_all.csv")


if __name__ == "__main__":
    main()
