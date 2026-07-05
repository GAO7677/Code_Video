from __future__ import annotations

# Run command example:
# PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt:/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main \
# CUDA_VISIBLE_DEVICES=7 \
# /home/gaoya/miniconda3/envs/wan-cu128/bin/python \
# /home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_vggt/train0705/sweep_source_video_context_frames_train0705.py \
#   --weights-root /data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/train_stage1b_diffsynth_native0705/run_gpu0235_20260703/checkpoints/step-007000 \
#   --input-json-list-path /data/gaoya/AAA_test_video/0623/testjsons/test_5.txt \
#   --model-name train_stage1b_diffsynth_native0705_0705_sourcectx_sweep \
#   --context-frames-list 4 16 24 \
#   --num-inference-steps 40

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


THIS_FILE = Path(__file__).resolve()
TRAIN0705_DIR = THIS_FILE.parent
PACKAGE_DIR = TRAIN0705_DIR.parent
REPO_ROOT = PACKAGE_DIR.parent
DEFAULT_BATCH_SCRIPT = TRAIN0705_DIR / "wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v.py"
DEFAULT_DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/v2v")


def _read_list_file(list_path: Path, *, deduplicate: bool) -> list[Path]:
    items: list[Path] = []
    seen: set[Path] = set()
    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            item = Path(line).expanduser().resolve()
            if deduplicate and item in seen:
                continue
            seen.add(item)
            items.append(item)
    return items


def _load_input_json(json_path: Path) -> dict[str, object]:
    with json_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"input json must be an object: {json_path}")
    return payload


def _resolve_source_video(
    payload: dict[str, object],
    *,
    json_path: Path,
    source_video_field: str,
    fallback_to_input_video: bool,
) -> str:
    value = payload.get(source_video_field)
    if isinstance(value, str) and value.strip():
        return value.strip()
    if fallback_to_input_video:
        value = payload.get("input_video")
        if isinstance(value, str) and value.strip():
            return value.strip()
    raise KeyError(
        f"missing usable source video in {json_path}; checked {source_video_field!r}"
    )


def _build_patched_jsons(
    *,
    input_json_paths: list[Path],
    context_frames: int,
    scratch_root: Path,
    source_video_field: str,
    fallback_to_input_video: bool,
) -> tuple[list[Path], list[dict[str, object]]]:
    ctx_dir = scratch_root / f"context_frames_{int(context_frames):02d}"
    ctx_dir.mkdir(parents=True, exist_ok=True)

    patched_jsons: list[Path] = []
    manifest_items: list[dict[str, object]] = []
    for input_json_path in input_json_paths:
        payload = _load_input_json(input_json_path)
        source_video = _resolve_source_video(
            payload,
            json_path=input_json_path,
            source_video_field=source_video_field,
            fallback_to_input_video=fallback_to_input_video,
        )
        original_input_video = payload.get("input_video")
        payload["input_video"] = str(source_video)
        payload["sweep_source_video_field"] = str(source_video_field)
        payload["sweep_context_frames"] = int(context_frames)
        payload["original_input_video"] = original_input_video
        patched_json_path = ctx_dir / input_json_path.name
        patched_json_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        patched_jsons.append(patched_json_path)
        manifest_items.append(
            {
                "input_json": str(input_json_path),
                "patched_json": str(patched_json_path),
                "source_video": str(source_video),
                "original_input_video": original_input_video,
            }
        )
    return patched_jsons, manifest_items


def _write_list_file(list_path: Path, items: list[Path]) -> None:
    list_path.parent.mkdir(parents=True, exist_ok=True)
    with list_path.open("w", encoding="utf-8") as handle:
        for item in items:
            handle.write(str(item))
            handle.write("\n")


def _build_command(
    *,
    args: argparse.Namespace,
    context_frames: int,
    patched_list_path: Path,
    context_output_root: Path,
) -> list[str]:
    cmd = [
        str(args.python_executable),
        str(args.batch_script),
        "--weights-root",
        str(args.weights_root),
        "--input-json-list-path",
        str(patched_list_path),
        "--model-name",
        str(args.model_name),
        "--output-root",
        str(context_output_root),
        "--device",
        str(args.device),
        "--wan-root",
        str(args.wan_root),
        "--diffsynth-root",
        str(args.diffsynth_root),
        "--lora-checkpoint",
        str(args.lora_checkpoint),
        "--stage1a-init-from",
        str(args.stage1a_init_from),
        "--height",
        str(int(args.height)),
        "--width",
        str(int(args.width)),
        "--num-frames",
        str(int(args.num_frames)),
        "--context-frames",
        str(int(context_frames)),
        "--fps",
        str(int(args.fps)),
        "--sampling-mode",
        str(args.sampling_mode),
        "--num-inference-steps",
        str(int(args.num_inference_steps)),
        "--cfg-scale",
        str(float(args.cfg_scale)),
        "--seed",
        str(int(args.seed)),
        "--quality",
        str(int(args.quality)),
    ]
    if args.limit is not None:
        cmd.extend(["--limit", str(int(args.limit))])
    if args.force:
        cmd.append("--force")
    if args.overwrite:
        cmd.append("--overwrite")
    if args.initialize_model_on_cpu:
        cmd.append("--initialize-model-on-cpu")
    return cmd


def _build_env(args: argparse.Namespace) -> dict[str, str]:
    env = dict(os.environ)
    pythonpath_items = [
        str(REPO_ROOT),
        str(args.diffsynth_root),
    ]
    existing = env.get("PYTHONPATH", "").strip()
    if existing:
        pythonpath_items.append(existing)
    env["PYTHONPATH"] = ":".join(item for item in pythonpath_items if item)
    return env


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep train0705 inference over multiple context lengths by replacing "
            "input_video in each json with source_video and reusing the existing "
            "batch inference script."
        )
    )
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--input-json-list-path", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--scratch-root", type=Path, default=None)
    parser.add_argument("--context-frames-list", type=int, nargs="+", required=True)
    parser.add_argument("--source-video-field", type=str, default="source_video")
    parser.add_argument("--fallback-to-input-video", action="store_true")
    parser.add_argument("--no-deduplicate", action="store_true")
    parser.add_argument("--python-executable", type=Path, default=Path(sys.executable))
    parser.add_argument("--batch-script", type=Path, default=DEFAULT_BATCH_SCRIPT)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--wan-root", type=Path, default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B"))
    parser.add_argument("--diffsynth-root", type=Path, default=DEFAULT_DIFFSYNTH_ROOT)
    parser.add_argument(
        "--lora-checkpoint",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/0529/vjepa_vggt/train/checkpoints/"
            "raw_phys_state_wan_lora_continue_576x1024_f24/checkpoints/step-000500/checkpoint.safetensors"
        ),
    )
    parser.add_argument(
        "--stage1a-init-from",
        type=Path,
        default=Path(
            "/data/gaoya/AAA_test_video/0623/train/train0624/checkpoints/"
            "pybullet0629_teacher_student/stage1a_full_token_old/step_0005000.pt"
        ),
    )
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--initialize-model-on-cpu", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.weights_root = args.weights_root.expanduser().resolve()
    args.input_json_list_path = args.input_json_list_path.expanduser().resolve()
    args.python_executable = args.python_executable.expanduser().resolve()
    args.batch_script = args.batch_script.expanduser().resolve()
    args.wan_root = args.wan_root.expanduser().resolve()
    args.diffsynth_root = args.diffsynth_root.expanduser().resolve()
    args.lora_checkpoint = args.lora_checkpoint.expanduser().resolve()
    args.stage1a_init_from = args.stage1a_init_from.expanduser().resolve()

    if args.output_root is None:
        args.output_root = DEFAULT_OUTPUT_ROOT / str(args.model_name).strip()
    else:
        args.output_root = args.output_root.expanduser().resolve()
    if args.scratch_root is None:
        args.scratch_root = args.output_root / "_source_video_patched_jsons"
    else:
        args.scratch_root = args.scratch_root.expanduser().resolve()

    input_json_paths = _read_list_file(
        args.input_json_list_path,
        deduplicate=not bool(args.no_deduplicate),
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.scratch_root.mkdir(parents=True, exist_ok=True)

    summary_entries: list[dict[str, object]] = []
    for context_frames in args.context_frames_list:
        patched_jsons, manifest_items = _build_patched_jsons(
            input_json_paths=input_json_paths,
            context_frames=int(context_frames),
            scratch_root=args.scratch_root,
            source_video_field=str(args.source_video_field),
            fallback_to_input_video=bool(args.fallback_to_input_video),
        )
        patched_list_path = args.scratch_root / f"context_frames_{int(context_frames):02d}.txt"
        _write_list_file(patched_list_path, patched_jsons)

        context_output_root = args.output_root / f"context_frames_{int(context_frames):02d}"
        context_output_root.mkdir(parents=True, exist_ok=True)

        manifest = {
            "weights_root": str(args.weights_root),
            "input_json_list_path": str(args.input_json_list_path),
            "deduplicated": not bool(args.no_deduplicate),
            "context_frames": int(context_frames),
            "num_items": len(patched_jsons),
            "patched_items": manifest_items,
        }
        (context_output_root / "source_video_manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

        cmd = _build_command(
            args=args,
            context_frames=int(context_frames),
            patched_list_path=patched_list_path,
            context_output_root=context_output_root,
        )
        print(f"[run] context_frames={int(context_frames)}")
        print(" ".join(cmd))
        subprocess.run(
            cmd,
            check=True,
            cwd=str(REPO_ROOT),
            env=_build_env(args),
        )
        summary_entries.append(
            {
                "context_frames": int(context_frames),
                "output_root": str(context_output_root),
                "patched_list_path": str(patched_list_path),
                "num_items": len(patched_jsons),
            }
        )

    summary = {
        "weights_root": str(args.weights_root),
        "input_json_list_path": str(args.input_json_list_path),
        "model_name": str(args.model_name),
        "output_root": str(args.output_root),
        "context_frames_list": [int(item) for item in args.context_frames_list],
        "entries": summary_entries,
    }
    (args.output_root / "context_sweep_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(f"[done] outputs under {args.output_root}")


if __name__ == "__main__":
    main()
