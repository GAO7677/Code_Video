from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def _read_list_file(list_path: Path) -> list[Path]:
    items: list[Path] = []
    with list_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            items.append(Path(line).expanduser().resolve())
    return items


def _load_input_json(json_path: Path) -> dict[str, object]:
    with json_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"input json must be an object: {json_path}")
    return data


def _ensure_str_field(payload: dict[str, object], key: str, json_path: Path) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing or empty {key!r} in {json_path}")
    return value.strip()


def _resolve_input_video(payload: dict[str, object], json_path: Path) -> tuple[str, str]:
    for key in ("input_video8f", "input_video_randomf", "input_video"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return key, value.strip()
    raise ValueError(
        f"missing input video field in {json_path}; expected one of input_video8f, input_video_randomf, input_video"
    )


def _run_single_case(
    *,
    python_exe: Path,
    infer_script: Path,
    checkpoint_dir: Path,
    config_path: Path,
    input_json_path: Path,
    input_video: str,
    input_caption: str,
    output_dir: Path,
    output_video: Path,
    num_frames: int,
    context_frames: int,
    sampling_mode: str,
    sampling_steps: int,
    fps: int,
    seed: int,
    cfg_scale: float,
    env: dict[str, str],
) -> subprocess.CompletedProcess[str]:
    cmd = [
        str(python_exe),
        str(infer_script),
        "--checkpoint",
        str(checkpoint_dir),
        "--config",
        str(config_path),
        "--context-video",
        str(input_video),
        "--prompt",
        str(input_caption),
        "--output-dir",
        str(output_dir),
        "--output-video",
        str(output_video),
        "--num-frames",
        str(int(num_frames)),
        "--context-frames",
        str(int(context_frames)),
        "--sampling-mode",
        str(sampling_mode),
        "--sampling-steps",
        str(int(sampling_steps)),
        "--fps",
        str(int(fps)),
        "--seed",
        str(int(seed)),
        "--cfg-scale",
        str(float(cfg_scale)),
    ]
    return subprocess.run(cmd, text=True, capture_output=True, env=env, check=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Batch-run v_newtrain inference from a list of input json files."
    )
    parser.add_argument("--input-list", required=True, help="text file containing one input json path per line")
    parser.add_argument("--checkpoint-root", required=True, help="root dir containing step-xxxx checkpoint folders")
    parser.add_argument("--steps", nargs="+", required=True, help="checkpoint step names such as step-000400")
    parser.add_argument("--config", required=True)
    parser.add_argument("--infer-script", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--python-exe", default=sys.executable)
    parser.add_argument("--num-frames", type=int, default=24)
    parser.add_argument("--context-frames", type=int, default=8)
    parser.add_argument("--sampling-mode", choices=["prefix", "uniform"], default="prefix")
    parser.add_argument("--sampling-steps", type=int, default=40)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cfg-scale", type=float, default=5.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    input_list = Path(args.input_list).expanduser().resolve()
    checkpoint_root = Path(args.checkpoint_root).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    infer_script = Path(args.infer_script).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    python_exe = Path(args.python_exe).expanduser().resolve()

    json_paths = _read_list_file(input_list)
    output_root.mkdir(parents=True, exist_ok=True)

    env = dict(**__import__("os").environ)

    run_manifest: dict[str, object] = {
        "input_list": str(input_list),
        "checkpoint_root": str(checkpoint_root),
        "steps": [str(step) for step in args.steps],
        "num_items": len(json_paths),
        "sampling_steps": int(args.sampling_steps),
        "cfg_scale": float(args.cfg_scale),
        "seed": int(args.seed),
    }
    with (output_root / "batch_manifest.json").open("w", encoding="utf-8") as handle:
        json.dump(run_manifest, handle, indent=2, ensure_ascii=False)

    for step_name in args.steps:
        checkpoint_dir = checkpoint_root / str(step_name)
        if not checkpoint_dir.exists():
            raise FileNotFoundError(f"checkpoint dir not found: {checkpoint_dir}")

        step_output_dir = output_root / str(step_name)
        step_output_dir.mkdir(parents=True, exist_ok=True)

        for input_json_path in json_paths:
            payload = _load_input_json(input_json_path)
            input_video_key, input_video = _resolve_input_video(payload, input_json_path)
            input_caption = _ensure_str_field(payload, "input_caption", input_json_path)
            sample_stem = input_json_path.stem
            output_video = step_output_dir / f"{sample_stem}.mp4"
            output_json = step_output_dir / f"{sample_stem}.json"
            output_log = step_output_dir / f"{sample_stem}.log"

            if output_video.exists() and output_json.exists() and not args.force:
                print(f"[skip] {step_name} {sample_stem}")
                continue

            process = _run_single_case(
                python_exe=python_exe,
                infer_script=infer_script,
                checkpoint_dir=checkpoint_dir,
                config_path=config_path,
                input_json_path=input_json_path,
                input_video=input_video,
                input_caption=input_caption,
                output_dir=step_output_dir,
                output_video=output_video,
                num_frames=int(args.num_frames),
                context_frames=int(args.context_frames),
                sampling_mode=str(args.sampling_mode),
                sampling_steps=int(args.sampling_steps),
                fps=int(args.fps),
                seed=int(args.seed),
                cfg_scale=float(args.cfg_scale),
                env=env,
            )

            with output_log.open("w", encoding="utf-8") as handle:
                if process.stdout:
                    handle.write(process.stdout)
                if process.stderr:
                    if process.stdout:
                        handle.write("\n")
                    handle.write(process.stderr)

            if process.returncode != 0:
                raise RuntimeError(
                    f"inference failed for {sample_stem} @ {step_name}, see log: {output_log}"
                )
            if not output_video.exists():
                raise FileNotFoundError(f"missing output video after inference: {output_video}")

            sidecar = {
                "input_json": str(input_json_path),
                "input_video_key": str(input_video_key),
                "input_video": str(input_video),
                "input_caption": str(input_caption),
                "output_video": str(output_video),
                "seed": int(args.seed),
                "step": int(args.sampling_steps),
                "guidance": float(args.cfg_scale),
                "ckpt": str(checkpoint_dir),
            }
            with output_json.open("w", encoding="utf-8") as handle:
                json.dump(sidecar, handle, indent=2, ensure_ascii=False)
                handle.write("\n")

            result_json = step_output_dir / "result.json"
            if result_json.exists():
                result_json.unlink()

            print(f"[done] {step_name} {sample_stem}")


if __name__ == "__main__":
    main()
