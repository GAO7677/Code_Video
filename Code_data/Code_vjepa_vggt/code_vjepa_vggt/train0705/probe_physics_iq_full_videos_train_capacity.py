#!/usr/bin/env python3
"""
Build a temporary raw-video training set from Physics-IQ full videos and probe
the maximum trainable frame length under the current Wan training constraints.

This script is intended for raw-video TI2V smoke probing, not Stage1B:
- `train0705/run_train_stage1b_*` requires phys-state episode labels/boxes.
- `physics-iq-benchmark/full-videos` only provides raw `.mp4` videos.

So the probe uses the generic Wan TI2V training path that can directly consume
`raw_phys_state_video` datasets (`meta.json` + `video.mp4`).

Example:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt \
/home/gaoya/miniconda3/envs/wan-cu128/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/probe_physics_iq_full_videos_train_capacity.py \
  --gpu-set 6,7 \
  --height 512 \
  --width 896 \
  --candidate-num-frames 25,49,73,89,97,121
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

try:
    from decord import VideoReader, cpu
except Exception:  # pragma: no cover - optional runtime dependency
    VideoReader = None
    cpu = None


THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent.parent
DEFAULT_SOURCE_ROOT = Path("/data/gaoya/dataset/physics-iq-benchmark/full-videos")
DEFAULT_DATASET_ROOT = Path("/data/gaoya/agent-data/datasets/physics_iq_full_videos_raw_train")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/physics_iq_full_videos_capacity_probe")
DEFAULT_ACCELERATE_BIN = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/accelerate")
DEFAULT_TRAIN_SCRIPT = (
    PROJECT_ROOT / "code_vjepa_vggt" / "train0706_wan1p3b" / "train_v_newtrain.py"
)
DEFAULT_WAN_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
DEFAULT_DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe Physics-IQ full-video raw training frame limits."
    )
    parser.add_argument("--source-root", type=Path, default=DEFAULT_SOURCE_ROOT)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--accelerate-bin", type=Path, default=DEFAULT_ACCELERATE_BIN)
    parser.add_argument("--train-script", type=Path, default=DEFAULT_TRAIN_SCRIPT)
    parser.add_argument("--wan-root", type=Path, default=DEFAULT_WAN_ROOT)
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--gpu-set", type=str, default="6,7")
    parser.add_argument("--num-processes", type=int, default=2)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--dataset-repeat", type=int, default=1)
    parser.add_argument("--max-train-steps", type=int, default=1)
    parser.add_argument("--num-epochs", type=int, default=1)
    parser.add_argument("--candidate-num-frames", type=str, default="25,49,73,89,97,121,145,169")
    parser.add_argument("--timeout-seconds", type=int, default=4 * 60 * 60)
    parser.add_argument("--materialize-mode", choices=("symlink", "copy"), default="symlink")
    parser.add_argument("--max-videos", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--keep-run-artifacts", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def assert_no_gpu4(gpu_set: str) -> None:
    if ",4," in f",{gpu_set},":
        raise ValueError("gpu4 is faulty and cannot be used.")


def parse_candidates(raw: str) -> list[int]:
    values = []
    for item in str(raw).split(","):
        item = item.strip()
        if not item:
            continue
        value = int(item)
        if value < 1:
            raise ValueError(f"candidate num_frames must be positive, got {value}")
        if (value - 1) % 4 != 0:
            raise ValueError(f"candidate num_frames must satisfy 4n+1, got {value}")
        values.append(value)
    if not values:
        raise ValueError("candidate-num-frames must contain at least one value")
    return values


def stem_to_text(path: Path) -> str:
    text = path.stem
    text = text.replace("_", " ").replace("-", " ")
    text = re.sub(r"\b\d+\b", " ", text)
    text = re.sub(r"\b(full videos|trimmed|take|fps|perspective)\b", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text or path.stem


def probe_video_metadata(video_path: Path) -> dict[str, Any]:
    if VideoReader is None or cpu is None:
        raise ImportError(
            "decord is required for this probe script. Install decord in the wan-cu128 environment."
        )
    reader = VideoReader(str(video_path), ctx=cpu(0))
    frame_count = int(len(reader))
    if frame_count <= 0:
        raise ValueError(f"video has no frames: {video_path}")
    first = reader[0]
    height = int(first.shape[0])
    width = int(first.shape[1])
    try:
        fps = float(reader.get_avg_fps())
    except Exception:
        fps = 30.0
    duration_s = float(frame_count / fps) if fps > 0 else None
    return {
        "frame_count": frame_count,
        "fps": fps,
        "height": height,
        "width": width,
        "duration_s": duration_s,
    }


def build_meta_payload(video_path: Path, info: dict[str, Any]) -> dict[str, Any]:
    rel_parent = video_path.parent.relative_to(DEFAULT_SOURCE_ROOT) if video_path.is_relative_to(DEFAULT_SOURCE_ROOT) else video_path.parent
    title = stem_to_text(video_path)
    return {
        "title": title,
        "description": title,
        "key": video_path.stem,
        "family": str(rel_parent),
        "source_video_path": str(video_path.resolve()),
        "fps": float(info["fps"]),
        "duration_s": float(info["duration_s"]) if info["duration_s"] is not None else None,
        "resolution": [int(info["width"]), int(info["height"])],
        "frame_count": int(info["frame_count"]),
    }


def materialize_video(src: Path, dst: Path, mode: str) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "symlink":
        os.symlink(str(src), str(dst))
        return
    shutil.copy2(src, dst)


def prepare_dataset(args: argparse.Namespace) -> dict[str, Any]:
    source_root = args.source_root.expanduser().resolve()
    dataset_root = args.dataset_root.expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"source-root not found: {source_root}")
    if dataset_root.exists() and not args.probe_only:
        if not args.force:
            raise FileExistsError(f"dataset-root already exists: {dataset_root}. Pass --force to replace it.")
        shutil.rmtree(dataset_root)

    mp4_paths = sorted(source_root.rglob("*.mp4"))
    if args.max_videos is not None:
        mp4_paths = mp4_paths[: max(0, int(args.max_videos))]
    if not mp4_paths:
        raise FileNotFoundError(f"no .mp4 files found under {source_root}")

    train_root = dataset_root / "train"
    train_root.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    frame_hist: dict[int, int] = {}
    for index, video_path in enumerate(mp4_paths):
        info = probe_video_metadata(video_path)
        sample_dir = train_root / f"sample_{index:06d}"
        sample_dir.mkdir(parents=True, exist_ok=True)
        materialize_video(video_path, sample_dir / "video.mp4", args.materialize_mode)
        meta_payload = build_meta_payload(video_path, info)
        write_json(sample_dir / "meta.json", meta_payload)

        frame_count = int(info["frame_count"])
        frame_hist[frame_count] = frame_hist.get(frame_count, 0) + 1
        entries.append(
            {
                "sample_id": sample_dir.name,
                "sample_dir": str(sample_dir.resolve()),
                "meta_path": str((sample_dir / "meta.json").resolve()),
                "video_path": str(video_path.resolve()),
                "frame_count": frame_count,
                "fps": float(info["fps"]),
                "height": int(info["height"]),
                "width": int(info["width"]),
                "duration_s": float(info["duration_s"]) if info["duration_s"] is not None else None,
            }
        )

    entries_sorted = sorted(entries, key=lambda item: int(item["frame_count"]), reverse=True)
    summary = {
        "source_root": str(source_root),
        "dataset_root": str(dataset_root),
        "num_samples": len(entries),
        "materialize_mode": args.materialize_mode,
        "max_frame_count": int(entries_sorted[0]["frame_count"]),
        "min_frame_count": int(entries_sorted[-1]["frame_count"]),
        "top_frame_count_examples": entries_sorted[:10],
        "frame_histogram_top": [
            {"frame_count": frame_count, "count": count}
            for frame_count, count in sorted(frame_hist.items(), key=lambda item: (-item[1], -item[0]))[:20]
        ],
    }
    write_json(dataset_root / "source_index.json", entries)
    write_json(dataset_root / "summary.json", summary)
    return {
        "dataset_root": dataset_root,
        "train_root": train_root,
        "entries": entries,
        "summary": summary,
    }


def load_prepared_entries(dataset_root: Path) -> list[dict[str, Any]]:
    index_path = dataset_root / "source_index.json"
    if not index_path.is_file():
        raise FileNotFoundError(f"prepared dataset index not found: {index_path}")
    return json.loads(index_path.read_text(encoding="utf-8"))


def exact_ratio_for_context(num_frames: int, target_context_frames: int) -> float:
    upper = min(0.5, (target_context_frames + 0.49) / float(num_frames))
    lower = target_context_frames / float(num_frames)
    ratio = min(upper, max(lower + 1.0e-4, lower + (upper - lower) * 0.5))
    if int(num_frames * ratio) != target_context_frames:
        ratio = upper
    if int(num_frames * ratio) != target_context_frames:
        raise ValueError(
            f"Could not build exact ratio for num_frames={num_frames}, target_context_frames={target_context_frames}"
        )
    return float(ratio)


def build_cases(candidate_num_frames: list[int]) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    for num_frames in candidate_num_frames:
        max_context = min(num_frames - 1, int(num_frames * 0.5))
        cases.append(
            {
                "sweep": "minimal_context",
                "num_frames": int(num_frames),
                "target_context_frames": 1,
                "max_context_ratio": exact_ratio_for_context(num_frames, 1),
                "stop_after_failure_in_sweep": True,
            }
        )
        cases.append(
            {
                "sweep": "max_context",
                "num_frames": int(num_frames),
                "target_context_frames": int(max_context),
                "max_context_ratio": 0.5,
                "stop_after_failure_in_sweep": True,
            }
        )
    return cases


def build_filtered_dataset_root(filtered_root: Path, eligible_entries: list[dict[str, Any]]) -> None:
    if filtered_root.exists():
        shutil.rmtree(filtered_root)
    train_root = filtered_root / "train"
    train_root.mkdir(parents=True, exist_ok=True)
    for entry in eligible_entries:
        src_dir = Path(entry["sample_dir"])
        dst_dir = train_root / src_dir.name
        dst_dir.mkdir(parents=True, exist_ok=True)
        for file_name in ("meta.json", "video.mp4"):
            src_file = src_dir / file_name
            dst_file = dst_dir / file_name
            if dst_file.exists() or dst_file.is_symlink():
                dst_file.unlink()
            os.symlink(str(src_file), str(dst_file))


def choose_port(case_index: int) -> int:
    return 29600 + int(case_index)


def run_case(
    args: argparse.Namespace,
    case: dict[str, Any],
    case_index: int,
    entries: list[dict[str, Any]],
    work_root: Path,
) -> dict[str, Any]:
    tag = f"{case['sweep']}__nf{int(case['num_frames']):03d}__ctx{int(case['target_context_frames']):03d}"
    eligible_entries = [
        entry
        for entry in entries
        if int(entry["frame_count"]) >= int(case["num_frames"])
    ]
    if not eligible_entries:
        return {
            **case,
            "tag": tag,
            "status": "skipped",
            "failure_kind": "data_limit",
            "eligible_samples": 0,
            "returncode": None,
            "elapsed_seconds": 0.0,
            "run_dir": None,
            "log_path": None,
            "tail_lines": [],
        }

    filtered_root = work_root / "filtered_datasets" / tag
    run_dir = work_root / "runs" / tag
    log_path = work_root / "logs" / f"{tag}.log"
    build_filtered_dataset_root(filtered_root, eligible_entries)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    gpu_ids = [item.strip() for item in str(args.gpu_set).split(",") if item.strip()]
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{args.project_root}:{args.diffsynth_root}"
    env["CUDA_VISIBLE_DEVICES"] = args.gpu_set
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
                "--main_process_port",
                str(choose_port(case_index)),
            ]
        )
    else:
        command.extend(["--num_processes", "1", "--num_machines", "1"])
    command.extend(
        [
            str(args.train_script),
            "--diffsynth_root",
            str(args.diffsynth_root),
            "--wan_root",
            str(args.wan_root),
            "--dataset_base_path",
            str(filtered_root),
            "--dataset_metadata_path",
            "",
            "--height",
            str(int(args.height)),
            "--width",
            str(int(args.width)),
            "--num_frames",
            str(int(case["num_frames"])),
            "--max_train_steps",
            str(int(args.max_train_steps)),
            "--context_sampling_profile",
            "legacy_prefix",
            "--min_context_frames",
            str(int(case["target_context_frames"])),
            "--max_context_ratio",
            f"{float(case['max_context_ratio']):.8f}",
            "--dataset_repeat",
            str(int(args.dataset_repeat)),
            "--dataset_num_workers",
            "0",
            "--learning_rate",
            "1e-4",
            "--weight_decay",
            "0.01",
            "--num_epochs",
            str(int(args.num_epochs)),
            "--gradient_accumulation_steps",
            "1",
            "--save_steps",
            "1000",
            "--remove_prefix_in_ckpt",
            "pipe.dit.",
            "--output_path",
            str(run_dir),
            "--lora_base_model",
            "dit",
            "--lora_target_modules",
            "q,k,v,o,ffn.0,ffn.2",
            "--lora_rank",
            "32",
            "--report_to",
            "none",
        ]
    )

    started_at = time.time()
    with log_path.open("w", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n\n")
        handle.write(f"eligible_samples={len(eligible_entries)}\n\n")
        handle.flush()
        process = subprocess.run(
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
    joined_tail = "\n".join(tail_lines).lower()
    if process.returncode == 0:
        status = "success"
        failure_kind = None
    else:
        status = "failed"
        if "out of memory" in joined_tail or "cuda error: out of memory" in joined_tail:
            failure_kind = "oom"
        elif "traininginterrupted" in joined_tail:
            failure_kind = "interrupted"
        else:
            failure_kind = "other"

    result = {
        **case,
        "tag": tag,
        "status": status,
        "failure_kind": failure_kind,
        "eligible_samples": len(eligible_entries),
        "returncode": int(process.returncode),
        "elapsed_seconds": elapsed,
        "run_dir": str(run_dir),
        "log_path": str(log_path),
        "tail_lines": tail_lines,
    }
    if status == "success" and not args.keep_run_artifacts:
        shutil.rmtree(run_dir, ignore_errors=True)
        shutil.rmtree(filtered_root, ignore_errors=True)
        result["run_dir_removed"] = True
    else:
        result["run_dir_removed"] = False
    return result


def summarize(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {"results": results, "by_sweep": {}}
    by_sweep: dict[str, list[dict[str, Any]]] = {}
    for item in results:
        by_sweep.setdefault(str(item["sweep"]), []).append(item)

    for sweep_name, sweep_results in by_sweep.items():
        successes = [item for item in sweep_results if item["status"] == "success"]
        failures = [item for item in sweep_results if item["status"] not in {"success", "skipped"}]
        data_skips = [item for item in sweep_results if item["failure_kind"] == "data_limit"]
        best_success = None
        if successes:
            best_success = max(successes, key=lambda item: int(item["num_frames"]))
        summary["by_sweep"][sweep_name] = {
            "num_cases": len(sweep_results),
            "num_success": len(successes),
            "num_failure": len(failures),
            "num_data_limit": len(data_skips),
            "best_success": best_success,
            "first_failure": failures[0] if failures else None,
            "first_data_limit": data_skips[0] if data_skips else None,
        }
    return summary


def main() -> None:
    args = parse_args()
    assert_no_gpu4(args.gpu_set)
    candidates = parse_candidates(args.candidate_num_frames)

    output_root = args.output_root.expanduser().resolve()
    if output_root.exists() and not args.probe_only:
        if not args.force:
            raise FileExistsError(f"output-root already exists: {output_root}. Pass --force to replace it.")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    dataset_root = args.dataset_root.expanduser().resolve()
    if args.probe_only:
        entries = load_prepared_entries(dataset_root)
        dataset_summary = json.loads((dataset_root / "summary.json").read_text(encoding="utf-8"))
    else:
        prepared = prepare_dataset(args)
        entries = prepared["entries"]
        dataset_summary = prepared["summary"]

    write_json(output_root / "prepared_dataset_summary.json", dataset_summary)
    if args.prepare_only:
        payload = {
            "mode": "prepare_only",
            "dataset_root": str(dataset_root),
            "summary": dataset_summary,
        }
        write_json(output_root / "summary.json", payload)
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    work_root = output_root / "probe_work"
    results: list[dict[str, Any]] = []
    failed_sweeps: set[str] = set()
    for case_index, case in enumerate(build_cases(candidates)):
        if case["sweep"] in failed_sweeps:
            continue
        result = run_case(
            args=args,
            case=case,
            case_index=case_index,
            entries=entries,
            work_root=work_root,
        )
        results.append(result)
        if result["status"] != "success" and case.get("stop_after_failure_in_sweep", False):
            failed_sweeps.add(str(case["sweep"]))

    summary = summarize(results)
    payload = {
        "dataset_summary": dataset_summary,
        "probe_config": {
            "gpu_set": args.gpu_set,
            "num_processes": args.num_processes,
            "height": args.height,
            "width": args.width,
            "candidate_num_frames": candidates,
            "train_script": str(args.train_script),
            "wan_root": str(args.wan_root),
        },
        **summary,
    }
    write_json(output_root / "summary.json", payload)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
