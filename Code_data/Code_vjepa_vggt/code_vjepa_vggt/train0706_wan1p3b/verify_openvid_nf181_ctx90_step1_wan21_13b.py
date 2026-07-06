#!/usr/bin/env python3
"""
Hard-verify Wan2.1-1.3B OpenVid smoke training with:
- 90 context frames
- 181 total frames
- 1 optimizer step

The script:
1. Extracts one accepted OpenVid parquet row into a dedicated 1-row train root.
2. Exports the initialization-time trainable state using the exact training module code path.
3. Runs a 1-step training smoke job with the current training constraints unchanged.
4. Compares the exported init state against step-000001/checkpoint.safetensors.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file as load_safetensors_file
from safetensors.torch import save_file as save_safetensors_file


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from code_vjepa_vggt.train0706_wan1p3b.train_v_newtrain import (  # noqa: E402
    build_model,
    format_step_tag,
    prepare_args,
    wan_parser,
)


DEFAULT_SOURCE_PARQUET = Path(
    "/data/gaoya/dataset/mvp-lab-OpenVidHD-0.4M-720p-48fps/train/"
    "rank0_1760955446.4554715_720x1280.parquet"
)
DEFAULT_OUTPUT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/train0706_verify_nf181_ctx90_step1"
)
DEFAULT_ACCELERATE_BIN = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate")
DEFAULT_TRAIN_SCRIPT = THIS_DIR / "train_v_newtrain.py"
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.1-T2V-1.3B")
DEFAULT_DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Hard-verify 181-frame / 90-context Wan2.1-1.3B smoke training."
    )
    parser.add_argument("--source-parquet", type=Path, default=DEFAULT_SOURCE_PARQUET)
    parser.add_argument("--row-index", type=int, default=0)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--accelerate-bin", type=Path, default=DEFAULT_ACCELERATE_BIN)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--gpu-set", type=str, default="3,6")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=672)
    parser.add_argument("--num-frames", type=int, default=181)
    parser.add_argument("--context-frames", type=int, default=90)
    parser.add_argument("--dataset-repeat", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--timeout-seconds", type=int, default=4 * 60 * 60)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--keep-train-run", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def decode_info_blob(blob: bytes) -> dict[str, Any]:
    payload = torch.load(io.BytesIO(blob), map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected dict in parquet info blob, got {type(payload).__name__}.")
    return payload


def extract_single_row_parquet(source_parquet: Path, row_index: int, output_parquet: Path) -> dict[str, Any]:
    parquet_file = pq.ParquetFile(str(source_parquet))
    total_rows = int(parquet_file.metadata.num_rows)
    if row_index < 0 or row_index >= total_rows:
        raise IndexError(
            f"row_index out of range for {source_parquet}: {row_index} not in [0, {total_rows})"
        )

    table = parquet_file.read_row_group(int(row_index), columns=["info", "raw_video"])
    if table.num_rows != 1:
        raise RuntimeError(
            f"Expected exactly 1 row from row group {row_index}, got {table.num_rows} rows."
        )

    output_parquet.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(table, str(output_parquet))

    info = decode_info_blob(table.column("info")[0].as_py())
    metadata_frames = info.get("num_frames", info.get("frame", -1))
    return {
        "source_parquet": str(source_parquet),
        "row_index": int(row_index),
        "caption": str(info.get("caption", "")),
        "metadata_num_frames": int(metadata_frames) if metadata_frames is not None else -1,
        "metadata_fps": float(info.get("fps", -1.0)) if info.get("fps") is not None else -1.0,
        "metadata_seconds": float(info.get("seconds", -1.0)) if info.get("seconds") is not None else -1.0,
        "keys": sorted(info.keys()),
        "output_parquet": str(output_parquet),
        "raw_video_bytes": int(len(table.column("raw_video")[0].as_py())),
    }


def build_openvid_only_config(train_root: Path, dataset_repeat: int) -> list[dict[str, Any]]:
    return [
        {
            "type": "openvid",
            "path": str(train_root.resolve()),
            "repeat": int(dataset_repeat),
        }
    ]


def build_train_args_list(args: argparse.Namespace, dataset_config_path: Path, train_output_root: Path) -> list[str]:
    return [
        "--diffsynth_root",
        str(args.diffsynth_root),
        "--wan_root",
        str(args.wan_root),
        "--dataset_base_path",
        str(dataset_config_path),
        "--dataset_metadata_path",
        "",
        "--height",
        str(int(args.height)),
        "--width",
        str(int(args.width)),
        "--num_frames",
        str(int(args.num_frames)),
        "--max_train_steps",
        "1",
        "--context_sampling_profile",
        "legacy_prefix",
        "--min_context_frames",
        str(int(args.context_frames)),
        "--max_context_ratio",
        "0.5",
        "--dataset_repeat",
        str(int(args.dataset_repeat)),
        "--dataset_num_workers",
        "0",
        "--learning_rate",
        str(float(args.learning_rate)),
        "--weight_decay",
        str(float(args.weight_decay)),
        "--num_epochs",
        "1",
        "--gradient_accumulation_steps",
        "1",
        "--save_steps",
        "1",
        "--max_checkpoints_keep",
        "2",
        "--remove_prefix_in_ckpt",
        "pipe.dit.",
        "--output_path",
        str(train_output_root),
        "--lora_base_model",
        "dit",
        "--lora_target_modules",
        "q,k,v,o,ffn.0,ffn.2",
        "--lora_rank",
        str(int(args.lora_rank)),
        "--report_to",
        "none",
    ]


def export_initial_trainable_state(train_args_list: list[str], output_path: Path) -> dict[str, Any]:
    parser = wan_parser()
    parsed_args = prepare_args(
        parser.parse_args(train_args_list + ["--initialize_model_on_cpu"])
    )
    model = build_model(
        parsed_args,
        SimpleNamespace(device=torch.device("cpu")),
    )
    trainable_state = model.export_trainable_state_dict(
        model.state_dict(),
        remove_prefix=parsed_args.remove_prefix_in_ckpt,
    )
    cpu_state = {name: tensor.detach().cpu().contiguous() for name, tensor in trainable_state.items()}
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_safetensors_file(cpu_state, str(output_path))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "path": str(output_path),
        "num_tensors": len(cpu_state),
        "num_elements": int(sum(t.numel() for t in cpu_state.values())),
    }


def run_training_job(args: argparse.Namespace, train_args_list: list[str], log_path: Path) -> dict[str, Any]:
    log_path.parent.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.project_root}:{args.diffsynth_root}"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_set
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    gpu_ids = [item.strip() for item in str(args.gpu_set).split(",") if item.strip()]
    command = [str(args.accelerate_bin), "launch"]
    if len(gpu_ids) > 1:
        command.extend(
            [
                "--multi_gpu",
                "--num_processes",
                str(int(args.num_processes)),
                "--num_machines",
                "1",
                "--mixed_precision",
                "bf16",
            ]
        )
    else:
        command.extend(["--num_processes", "1", "--num_machines", "1"])
    command.append(str(args.train_script))
    command.extend(train_args_list)

    started_at = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n\n")
        handle.flush()
        result = subprocess.run(
            command,
            cwd=str(THIS_DIR),
            env=env,
            stdout=handle,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=int(args.timeout_seconds),
        )
    elapsed = round(time.time() - started_at, 3)

    tail_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-120:]
    return {
        "command": command,
        "log_path": str(log_path),
        "returncode": int(result.returncode),
        "elapsed_seconds": elapsed,
        "tail_lines": tail_lines,
    }


def compare_states(initial_state_path: Path, trained_state_path: Path, diff_json_path: Path) -> dict[str, Any]:
    initial_state = load_safetensors_file(str(initial_state_path), device="cpu")
    trained_state = load_safetensors_file(str(trained_state_path), device="cpu")

    initial_keys = set(initial_state.keys())
    trained_keys = set(trained_state.keys())
    missing_in_trained = sorted(initial_keys - trained_keys)
    extra_in_trained = sorted(trained_keys - initial_keys)
    common_keys = sorted(initial_keys & trained_keys)

    changed_tensors = 0
    unchanged_tensors = 0
    total_elements = 0
    changed_elements = 0
    max_abs_diff = 0.0
    mean_abs_diff_sum = 0.0
    max_abs_diff_tensor = None
    first_changed = []

    for name in common_keys:
        init_tensor = initial_state[name].float()
        trained_tensor = trained_state[name].float()
        if init_tensor.shape != trained_tensor.shape:
            raise ValueError(
                f"Shape mismatch for {name}: init={tuple(init_tensor.shape)} trained={tuple(trained_tensor.shape)}"
            )
        diff = (trained_tensor - init_tensor).abs()
        tensor_max = float(diff.max().item()) if diff.numel() > 0 else 0.0
        tensor_mean = float(diff.mean().item()) if diff.numel() > 0 else 0.0
        tensor_changed_elements = int((diff > 0).sum().item())
        total_elements += int(diff.numel())
        changed_elements += tensor_changed_elements
        mean_abs_diff_sum += tensor_mean * int(diff.numel())
        if tensor_changed_elements > 0:
            changed_tensors += 1
            if len(first_changed) < 20:
                first_changed.append(
                    {
                        "name": name,
                        "max_abs_diff": tensor_max,
                        "mean_abs_diff": tensor_mean,
                        "changed_elements": tensor_changed_elements,
                        "num_elements": int(diff.numel()),
                    }
                )
        else:
            unchanged_tensors += 1
        if tensor_max > max_abs_diff:
            max_abs_diff = tensor_max
            max_abs_diff_tensor = name

    mean_abs_diff = mean_abs_diff_sum / float(total_elements) if total_elements > 0 else 0.0
    summary = {
        "initial_state_path": str(initial_state_path),
        "trained_state_path": str(trained_state_path),
        "num_initial_tensors": len(initial_keys),
        "num_trained_tensors": len(trained_keys),
        "num_common_tensors": len(common_keys),
        "missing_in_trained": missing_in_trained,
        "extra_in_trained": extra_in_trained,
        "changed_tensors": int(changed_tensors),
        "unchanged_tensors": int(unchanged_tensors),
        "total_elements": int(total_elements),
        "changed_elements": int(changed_elements),
        "max_abs_diff": float(max_abs_diff),
        "max_abs_diff_tensor": max_abs_diff_tensor,
        "mean_abs_diff": float(mean_abs_diff),
        "first_changed_tensors": first_changed,
    }
    write_json(diff_json_path, summary)
    return summary


def main() -> None:
    args = parse_args()
    if (int(args.num_frames) - 1) % 4 != 0:
        raise ValueError(f"num_frames must satisfy 4n+1, got {args.num_frames}")
    if int(args.context_frames) != min(int(args.num_frames) - 1, int(int(args.num_frames) * 0.5)):
        raise ValueError(
            "context_frames must match the exact max context implied by max_context_ratio=0.5 "
            f"for this hard verification, got context_frames={args.context_frames}, num_frames={args.num_frames}"
        )
    if ",4," in f",{args.gpu_set},":
        raise ValueError("gpu4 is faulty and cannot be used.")

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists():
        if not args.force:
            raise FileExistsError(f"Output root already exists: {output_root}. Pass --force to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_root = output_root / "dataset"
    train_root = dataset_root / "train"
    config_root = dataset_root / "configs"
    reports_root = output_root / "reports"
    train_run_root = output_root / "train_run"
    dataset_parquet_path = train_root / "sample_row.parquet"
    config_path = config_root / "openvid_only_config.json"
    init_state_path = reports_root / "init_trainable.safetensors"
    train_log_path = reports_root / "train_step1.log"

    sample_info = extract_single_row_parquet(
        source_parquet=args.source_parquet.expanduser().resolve(),
        row_index=int(args.row_index),
        output_parquet=dataset_parquet_path,
    )
    openvid_only_config = build_openvid_only_config(train_root=train_root, dataset_repeat=args.dataset_repeat)
    write_json(config_path, openvid_only_config)

    train_args_list = build_train_args_list(
        args=args,
        dataset_config_path=config_path,
        train_output_root=train_run_root,
    )
    init_export_info = export_initial_trainable_state(train_args_list=train_args_list, output_path=init_state_path)
    train_result = run_training_job(args=args, train_args_list=train_args_list, log_path=train_log_path)
    if train_result["returncode"] != 0:
        write_json(
            reports_root / "failed_summary.json",
            {
                "sample_info": sample_info,
                "train_result": train_result,
                "init_export_info": init_export_info,
            },
        )
        raise RuntimeError(
            f"Training job failed with return code {train_result['returncode']}. See {train_log_path}"
        )

    step_tag = format_step_tag(1)
    trained_state_path = train_run_root / "checkpoints" / step_tag / "checkpoint.safetensors"
    if not trained_state_path.is_file():
        raise FileNotFoundError(f"Expected trained checkpoint not found: {trained_state_path}")

    diff_summary = compare_states(
        initial_state_path=init_state_path,
        trained_state_path=trained_state_path,
        diff_json_path=reports_root / "weight_diff_summary.json",
    )
    summary = {
        "sample_info": sample_info,
        "openvid_only_config": str(config_path),
        "train_args_list": train_args_list,
        "init_export_info": init_export_info,
        "train_result": train_result,
        "trained_state_path": str(trained_state_path),
        "diff_summary": diff_summary,
    }
    write_json(reports_root / "summary.json", summary)

    if not args.keep_train_run:
        # Keep the step-000001 checkpoint, logs, and reports; discard any non-essential leftovers.
        pass

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
