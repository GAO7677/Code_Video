from __future__ import annotations

"""
Prepare and run Physics-IQ Verified V2V inference through the train0705
native DiffSynth runner.

This wrapper follows the official Physics-IQ Verified workflow:
1. Read the official descriptions CSV.
2. Use only the 198 `take-1` cases.
3. Use the official conditioning videos for V2V.
4. Generate benchmark-named videos matching `generated_video_name`.
5. Respect the model's `num_frames % 4 == 1` requirement during generation,
   then trim outputs back to exact 5.0 seconds for official evaluation.
"""

import argparse
import csv
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/WAN_2p2/DiffSynth-Studio-main")
if "DIFFSYNTH_ROOT" not in os.environ:
    os.environ["DIFFSYNTH_ROOT"] = str(DEFAULT_DIFFSYNTH_ROOT)
if str(DEFAULT_DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DEFAULT_DIFFSYNTH_ROOT))

from code_vjepa_vggt.train0705 import (  # noqa: E402
    wan_stage1b_context_only_no_gt_box_vnewtrain0705_v2v as batchmod,
)


DEFAULT_VERIFIED_ROOT = Path("/data/gaoya/dataset/Anates-Labs-Research-Physics-IQ-Verified")
DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/0623/test/physicsiq")
DEFAULT_PHYSICS_IQ_REPO = Path(
    "/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_phys_papers_compare/physics-IQ-benchmark-main"
)
DEFAULT_DESCRIPTIONS = (
    DEFAULT_PHYSICS_IQ_REPO / "descriptions" / "best_practice" / "descriptions_base.csv"
)


def _infer_prompt_setting(descriptions_file: Path) -> str:
    name = descriptions_file.name.lower()
    if "original" in name:
        return "op"
    return "bpp"


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(
        description="Adapt the native train0705 V2V runner to Physics-IQ Verified."
    )
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--verified-root", type=Path, default=DEFAULT_VERIFIED_ROOT)
    parser.add_argument("--benchmark-repo-root", type=Path, default=DEFAULT_PHYSICS_IQ_REPO)
    parser.add_argument("--descriptions-file", type=Path, default=DEFAULT_DESCRIPTIONS)
    parser.add_argument(
        "--run-name",
        type=str,
        default=None,
        help=(
            "Prepared-input bundle name. Defaults to <model-name>-<prompt_setting>-run_01. "
            "Final generated videos stay in the native step-* directory."
        ),
    )
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--num-frames",
        type=int,
        default=150,
        help="Target final official frames. Wrapper aligns generation frames and trims back to 5.0s.",
    )
    parser.add_argument("--context-frames", type=int, default=20)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--num-inference-steps", type=int, default=40)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--keep-prepared-inputs", action="store_true")
    parser.add_argument("--force-reprepare", action="store_true")
    args, passthrough = parser.parse_known_args()
    return args, passthrough


def _validate_cli_args(args: argparse.Namespace) -> None:
    if int(args.fps) <= 0:
        raise ValueError("--fps must be positive")
    if int(args.num_frames) <= 0:
        raise ValueError("--num-frames must be positive")
    if int(args.num_frames) != int(args.fps) * 5:
        raise ValueError(
            "this wrapper targets strict 5-second Physics-IQ Verified outputs; "
            f"got num_frames={args.num_frames} and fps={args.fps}, "
            f"but expected num_frames == fps * 5 == {int(args.fps) * 5}"
        )


def _align_generation_num_frames(target_num_frames: int) -> int:
    target = int(target_num_frames)
    remainder = target % 4
    if remainder == 1:
        return target
    return target + ((1 - remainder) % 4)


def _resolve_conditioning_dir(verified_root: Path, fps: int) -> Path:
    candidates = [
        verified_root / "split-videos" / "conditioning" / f"{int(fps)}FPS",
        verified_root / "split-videos" / "conditioning-videos" / f"{int(fps)}FPS",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(
        "could not find Physics-IQ Verified conditioning videos under any known path: "
        + ", ".join(str(path) for path in candidates)
    )


def _load_take1_rows(descriptions_file: Path) -> list[dict[str, str]]:
    with descriptions_file.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    take1_rows = [row for row in rows if "take-1" in str(row.get("scenario", ""))]
    if len(take1_rows) != 198:
        raise ValueError(
            f"expected exactly 198 take-1 rows in {descriptions_file}, found {len(take1_rows)}"
        )
    return take1_rows


def _conditioning_name_from_scenario(scenario: str, fps: int) -> str:
    parts = scenario.split("_")
    if len(parts) < 4:
        raise ValueError(f"unexpected scenario format: {scenario}")
    file_id, perspective, take = parts[:3]
    scenario_suffix = "_".join(parts[3:])
    return (
        f"{file_id}_conditioning-videos_{int(fps)}FPS_"
        f"{perspective}_{take}_{scenario_suffix}"
    )


def _prepare_inputs(
    *,
    rows: list[dict[str, str]],
    conditioning_dir: Path,
    fps: int,
    prepared_root: Path,
    force_reprepare: bool,
    limit: int | None,
) -> tuple[Path, list[Path]]:
    if prepared_root.exists() and force_reprepare:
        shutil.rmtree(prepared_root)
    items_dir = prepared_root / "items"
    items_dir.mkdir(parents=True, exist_ok=True)

    selected_rows = rows if limit is None else rows[: max(0, int(limit))]
    json_paths: list[Path] = []
    for row in selected_rows:
        scenario = str(row["scenario"]).strip()
        caption = str(row["description"]).strip()
        generated_video_name = str(row["generated_video_name"]).strip()
        conditioning_name = _conditioning_name_from_scenario(scenario, fps)
        conditioning_path = conditioning_dir / conditioning_name
        if not conditioning_path.exists():
            raise FileNotFoundError(
                f"missing conditioning video for scenario {scenario}: {conditioning_path}"
            )
        payload = {
            "input_video": str(conditioning_path),
            "source_video": str(conditioning_path),
            "input_caption": caption,
            "benchmark_scenario": scenario,
            "generated_video_name": generated_video_name,
            "conditioning_video": str(conditioning_path),
        }
        json_path = items_dir / f"{Path(generated_video_name).stem}.json"
        with json_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
            handle.write("\n")
        json_paths.append(json_path)

    list_path = prepared_root / "input_jsons.txt"
    with list_path.open("w", encoding="utf-8") as handle:
        for path in json_paths:
            handle.write(str(path))
            handle.write("\n")
    return list_path, json_paths


def _build_run_name(args: argparse.Namespace) -> str:
    if args.run_name and str(args.run_name).strip():
        return str(args.run_name).strip()
    prompt_setting = _infer_prompt_setting(args.descriptions_file)
    return f"{str(args.model_name).strip()}-{prompt_setting}-run_01"


def _build_batch_command(
    *,
    args: argparse.Namespace,
    passthrough: list[str],
    input_json_list_path: Path,
    generation_num_frames: int,
) -> list[str]:
    command = [
        sys.executable,
        str(Path(batchmod.__file__).resolve()),
        "--weights-root",
        str(args.weights_root),
        "--input-json-list-path",
        str(input_json_list_path),
        "--model-name",
        str(args.model_name),
        "--output-root",
        str(args.output_root),
        "--fps",
        str(int(args.fps)),
        "--num-frames",
        str(int(generation_num_frames)),
        "--context-frames",
        str(int(args.context_frames)),
        "--sampling-mode",
        str(args.sampling_mode),
        "--num-inference-steps",
        str(int(args.num_inference_steps)),
        "--cfg-scale",
        str(float(args.cfg_scale)),
        "--seed",
        str(int(args.seed)),
        "--device",
        str(args.device),
    ]
    if args.limit is not None:
        command.extend(["--limit", str(int(args.limit))])
    command.extend(passthrough)
    return command


def _trim_video_inplace_to_official_duration(
    *,
    video_path: Path,
    fps: int,
    duration_seconds: float = 5.0,
) -> None:
    temp_path = video_path.with_suffix(".trimmed.mp4")
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-t",
        str(float(duration_seconds)),
        "-r",
        str(int(fps)),
        str(temp_path),
    ]
    subprocess.run(command, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    temp_path.replace(video_path)


def _finalize_outputs(
    *,
    step_output_dir: Path,
    fps: int,
) -> tuple[int, list[Path]]:
    if not step_output_dir.exists():
        raise FileNotFoundError(f"native step output dir not found: {step_output_dir}")

    trimmed_count = 0
    missing_json_paths: list[Path] = []
    for video_path in sorted(step_output_dir.glob("*.mp4")):
        output_json = video_path.with_suffix(".json")
        if not output_json.exists():
            missing_json_paths.append(output_json)
            continue
        _trim_video_inplace_to_official_duration(video_path=video_path, fps=int(fps))
        trimmed_count += 1
    return trimmed_count, missing_json_paths


def main() -> None:
    args, passthrough = _parse_args()
    _validate_cli_args(args)

    args.weights_root = args.weights_root.expanduser().resolve()
    args.output_root = args.output_root.expanduser().resolve()
    args.verified_root = args.verified_root.expanduser().resolve()
    args.benchmark_repo_root = args.benchmark_repo_root.expanduser().resolve()
    args.descriptions_file = args.descriptions_file.expanduser().resolve()

    if not args.weights_root.exists():
        raise FileNotFoundError(f"--weights-root not found: {args.weights_root}")
    if not args.verified_root.exists():
        raise FileNotFoundError(f"--verified-root not found: {args.verified_root}")
    if not args.descriptions_file.exists():
        raise FileNotFoundError(f"--descriptions-file not found: {args.descriptions_file}")

    run_name = _build_run_name(args)
    conditioning_dir = _resolve_conditioning_dir(args.verified_root, int(args.fps))
    rows = _load_take1_rows(args.descriptions_file)
    generation_num_frames = _align_generation_num_frames(int(args.num_frames))

    prepared_root = args.output_root / "_physics_iq_inputs" / run_name
    input_json_list_path, json_paths = _prepare_inputs(
        rows=rows,
        conditioning_dir=conditioning_dir,
        fps=int(args.fps),
        prepared_root=prepared_root,
        force_reprepare=bool(args.force_reprepare),
        limit=args.limit,
    )

    print(f"[prepared] run_name={run_name}")
    print(f"[prepared] input_json_list_path={input_json_list_path}")
    print(f"[prepared] num_cases={len(json_paths)}")
    if generation_num_frames != int(args.num_frames):
        print(
            "[note] underlying Wan generation requires num_frames % 4 == 1; "
            f"using generation_num_frames={generation_num_frames} and trimming outputs back to 5.0s"
        )

    command = _build_batch_command(
        args=args,
        passthrough=passthrough,
        input_json_list_path=input_json_list_path,
        generation_num_frames=generation_num_frames,
    )
    print("[command]")
    print(" ".join(subprocess.list2cmdline([part]) for part in command))

    if args.prepare_only:
        return

    subprocess.run(command, check=True, env=os.environ.copy())

    step_output_dir = args.output_root / args.weights_root.name
    trimmed_count, missing_json_paths = _finalize_outputs(
        step_output_dir=step_output_dir,
        fps=int(args.fps),
    )
    print(
        "[trim] normalized completed generated outputs to exact 5.0s "
        f"at {int(args.fps)} FPS under {step_output_dir} (trimmed={trimmed_count})"
    )
    if missing_json_paths:
        print(
            "[trim] skipped mp4 files without matching json metadata: "
            + ", ".join(str(path.with_suffix(".mp4").name) for path in missing_json_paths)
        )

    if not args.keep_prepared_inputs:
        shutil.rmtree(prepared_root, ignore_errors=True)
        print(f"[cleanup] removed prepared inputs: {prepared_root}")

    print(f"[done] generated Physics-IQ Verified run at: {step_output_dir}")


if __name__ == "__main__":
    main()
