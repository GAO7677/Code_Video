#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

DIFFSYNTH_ROOT = Path("/home/gaoya/Code_Video/DiffSynth-Studio-main")
TRAIN0419_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")
BENCHMARK_ROOT = Path("/home/gaoya/Code_Video/physics-IQ-benchmark-main")
DATASET_ROOT = Path("/data/gaoya/dataset/physics-iq-benchmark")
MODEL_ROOT = Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B")
OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Benchmark/physics_IQ")
DESCRIPTIONS_CSV = BENCHMARK_ROOT / "descriptions" / "descriptions.csv"

if str(DIFFSYNTH_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFSYNTH_ROOT))
if str(TRAIN0419_ROOT) not in sys.path:
    sys.path.insert(0, str(TRAIN0419_ROOT))

from diffsynth import ModelConfig  # noqa: E402
from diffsynth.utils.data import VideoData, save_video  # noqa: E402
from context_wan import ContextAwareWanVideoPipeline  # noqa: E402

sys.path.append("/home/gaoya/code_my_utils")
from tools.seed import seed_everything  # noqa: E402


DEFAULT_NEGATIVE_PROMPT = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run pre-train context-aware Wan TV2V on Physics-IQ with optional multi-GPU sharding."
    )
    parser.add_argument("--benchmark_root", type=Path, default=BENCHMARK_ROOT)
    parser.add_argument("--dataset_root", type=Path, default=DATASET_ROOT)
    parser.add_argument("--wan_root", type=Path, default=MODEL_ROOT)
    parser.add_argument("--output_root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--descriptions_csv", type=Path, default=DESCRIPTIONS_CSV)

    parser.add_argument("--device", default="cuda")
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--num_frames", type=int, default=151)
    parser.add_argument("--context_frames", type=int, default=8)

    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--cfg_scale", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quality", type=int, default=5)
    parser.add_argument("--model_name", default="wan22_tv2v_pretrain_ctx8")
    parser.add_argument("--negative_prompt", default=DEFAULT_NEGATIVE_PROMPT)

    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument("--limit", type=int, default=None)

    parser.add_argument("--multi_gpu", action="store_true")
    parser.add_argument("--num_shards", type=int, default=1)
    parser.add_argument("--shard_id", type=int, default=0)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def assert_exists(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} not found: {path}")


def required_output_frames(fps: int) -> int:
    return fps * 5


def validate_args(args: argparse.Namespace) -> None:
    if args.limit is not None and not args.skip_evaluation:
        raise ValueError("--limit is only for generation smoke tests. Use it with --skip_evaluation.")
    if args.num_shards < 1:
        raise ValueError(f"--num_shards must be >= 1, got {args.num_shards}")
    if not (0 <= args.shard_id < args.num_shards):
        raise ValueError(f"Invalid shard setting: shard_id={args.shard_id}, num_shards={args.num_shards}")
    if args.context_frames < 1:
        raise ValueError(f"context_frames must be >= 1, got {args.context_frames}")
    if args.context_frames >= args.num_frames:
        raise ValueError(
            f"context_frames must be smaller than num_frames, got {args.context_frames} >= {args.num_frames}."
        )
    if args.height % 16 != 0 or args.width % 16 != 0:
        raise ValueError(f"height and width must be divisible by 16, got {(args.height, args.width)}.")
    if (args.num_frames - 1) % 4 != 0:
        raise ValueError(f"num_frames must satisfy 4n+1, got {args.num_frames}.")
    if not args.skip_generation and args.num_frames < required_output_frames(args.fps):
        raise ValueError(
            f"num_frames={args.num_frames} is too small for fps={args.fps}. Need at least {required_output_frames(args.fps)}."
        )


def load_cases(descriptions_csv: Path, limit: int | None) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with descriptions_csv.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if "take-1" not in row["scenario"]:
                continue
            rows.append(row)
    rows.sort(key=lambda row: row["generated_video_name"])
    if limit is not None:
        rows = rows[:limit]
    return rows


def conditioning_video_path(dataset_root: Path, scenario_filename: str, fps: int) -> Path:
    stem = Path(scenario_filename).stem
    file_id, perspective, take, scenario = stem.split("_", 3)
    filename = f"{file_id}_conditioning-videos_{fps}FPS_{perspective}_{take}_{scenario}.mp4"
    return dataset_root / "split-videos" / "conditioning" / f"{fps}FPS" / filename


def sha256_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def maybe_git_commit(path: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=path,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return None


def collect_environment_snapshot(args: argparse.Namespace) -> dict[str, Any]:
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "torch_cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "device_count": torch.cuda.device_count(),
        "device": args.device,
        "multi_gpu": args.multi_gpu,
        "num_shards": args.num_shards,
        "shard_id": args.shard_id,
        "context_frames": args.context_frames,
        "repos": {
            "benchmark_root": str(args.benchmark_root),
            "benchmark_commit": maybe_git_commit(args.benchmark_root),
            "train0419_root": str(TRAIN0419_ROOT),
            "train0419_commit": maybe_git_commit(TRAIN0419_ROOT),
        },
        "model_files": {},
    }
    for rel in [
        "diffusion_pytorch_model-00001-of-00003.safetensors",
        "diffusion_pytorch_model-00002-of-00003.safetensors",
        "diffusion_pytorch_model-00003-of-00003.safetensors",
        "models_t5_umt5-xxl-enc-bf16.pth",
        "Wan2.2_VAE.pth",
    ]:
        path = args.wan_root / rel
        info["model_files"][rel] = {
            "path": str(path),
            "exists": path.exists(),
            "sha256": sha256_file(path) if path.exists() else None,
        }
    return info


def write_json(path: Path, payload: dict[str, Any] | list[dict[str, Any]]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")


def find_tokenizer_path(wan_root: Path) -> Path:
    candidates = [
        wan_root / "google" / "umt5-xxl",
        wan_root / "google",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(
        f"Tokenizer directory not found. Checked: {', '.join(str(path) for path in candidates)}"
    )


def build_model_configs(wan_root: Path) -> list[ModelConfig]:
    dit_shards = [
        wan_root / "diffusion_pytorch_model-00001-of-00003.safetensors",
        wan_root / "diffusion_pytorch_model-00002-of-00003.safetensors",
        wan_root / "diffusion_pytorch_model-00003-of-00003.safetensors",
    ]
    t5_path = wan_root / "models_t5_umt5-xxl-enc-bf16.pth"
    vae_path = wan_root / "Wan2.2_VAE.pth"
    for path in dit_shards + [t5_path, vae_path]:
        if not path.is_file():
            raise FileNotFoundError(f"Required model file not found: {path}")
    return [
        ModelConfig(path=[str(path) for path in dit_shards]),
        ModelConfig(path=str(t5_path)),
        ModelConfig(path=str(vae_path)),
    ]


def build_pipeline(wan_root: Path, device: str) -> ContextAwareWanVideoPipeline:
    return ContextAwareWanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=build_model_configs(wan_root),
        tokenizer_config=ModelConfig(path=str(find_tokenizer_path(wan_root))),
    )


def load_context_frames(context_path: Path, context_frames: int, height: int, width: int):
    if context_path.is_file():
        data = VideoData(video_file=str(context_path), height=height, width=width)
    elif context_path.is_dir():
        data = VideoData(image_folder=str(context_path), height=height, width=width)
    else:
        raise FileNotFoundError(f"context_path not found: {context_path}")

    if len(data) < context_frames:
        raise ValueError(f"context source only has {len(data)} frames, but context_frames={context_frames}.")
    return [data[index] for index in range(context_frames)]


def generate_one_video(
    pipe: ContextAwareWanVideoPipeline,
    context_path: Path,
    prompt: str,
    negative_prompt: str,
    seed: int,
    height: int,
    width: int,
    num_frames: int,
    fps: int,
    cfg_scale: float,
    num_inference_steps: int,
    context_frames: int,
):
    seed_everything(seed)
    context = load_context_frames(
        context_path=context_path,
        context_frames=context_frames,
        height=height,
        width=width,
    )
    with torch.no_grad():
        video = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            input_image=context[0],
            context_video=context,
            height=height,
            width=width,
            num_frames=num_frames,
            seed=seed,
            cfg_scale=cfg_scale,
            num_inference_steps=num_inference_steps,
            tiled=True,
        )
    keep = required_output_frames(fps)
    if len(video) < keep:
        raise ValueError(
            f"Generated only {len(video)} frames for {context_path.name}, need at least {keep}."
        )
    return video[:keep]


def build_case_metadata(
    *,
    args: argparse.Namespace,
    row: dict[str, str],
    index: int,
    seed: int,
    context_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    sidecar_path = output_path.with_suffix(".json")
    return {
        "model_name": args.model_name,
        "generated_video_name": row["generated_video_name"],
        "sample_json_name": sidecar_path.name,
        "index_in_sorted_list": index,
        "scenario": row.get("scenario"),
        "perspective": row.get("perspective"),
        "seed": seed,
        "prompt": row["description"],
        "negative_prompt": args.negative_prompt,
        "context_path": str(context_path),
        "output_path": str(output_path),
        "context_frames": args.context_frames,
        "shard_id": args.shard_id,
        "num_shards": args.num_shards,
        "generation_params": {
            "height": args.height,
            "width": args.width,
            "fps": args.fps,
            "num_inference_steps": args.num_inference_steps,
            "cfg_scale": args.cfg_scale,
            "num_frames": args.num_frames,
            "task": "tv2v_pretrain_context",
        },
    }


def per_case_jsonl_path(metadata_dir: Path, model_name: str, num_shards: int, shard_id: int) -> Path:
    if num_shards <= 1:
        return metadata_dir / f"{model_name}_per_case.jsonl"
    return metadata_dir / f"{model_name}_per_case_shard{shard_id:02d}of{num_shards:02d}.jsonl"


def write_run_manifest(args: argparse.Namespace, metadata_dir: Path) -> None:
    manifest = {
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "environment": collect_environment_snapshot(args),
    }
    suffix = "" if args.num_shards <= 1 else f"_shard{args.shard_id:02d}of{args.num_shards:02d}"
    manifest_path = metadata_dir / f"{args.model_name}_run_manifest{suffix}.json"
    write_json(manifest_path, manifest)


def run_generation(args: argparse.Namespace, generated_dir: Path, metadata_dir: Path) -> None:
    cases = load_cases(args.descriptions_csv, args.limit)
    expected_count = 198 if args.limit is None else args.limit
    if len(cases) != expected_count:
        raise ValueError(
            f"Expected {expected_count} take-1 cases, but found {len(cases)} in {args.descriptions_csv}."
        )

    generated_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)
    write_run_manifest(args, metadata_dir)

    per_case_jsonl = per_case_jsonl_path(metadata_dir, args.model_name, args.num_shards, args.shard_id)
    if args.overwrite and per_case_jsonl.exists():
        per_case_jsonl.unlink()

    indexed_cases = list(enumerate(cases))
    shard_cases = [(idx, row) for idx, row in indexed_cases if idx % args.num_shards == args.shard_id]
    print(
        f"[worker] shard_id={args.shard_id}/{args.num_shards}, "
        f"num_cases={len(shard_cases)}, device={args.device}, seed={args.seed}, context_frames={args.context_frames}"
    )

    pipe = build_pipeline(args.wan_root, args.device)

    for index, row in shard_cases:
        output_name = row["generated_video_name"]
        output_path = generated_dir / output_name
        sidecar_path = output_path.with_suffix(".json")
        context_path = conditioning_video_path(args.dataset_root, row["scenario"], args.fps)
        assert_exists(context_path, "Conditioning video")

        if output_path.exists() and not args.overwrite:
            print(f"[skip][shard {args.shard_id}] {output_name} | seed={args.seed}")
            case_payload = build_case_metadata(
                args=args,
                row=row,
                index=index,
                seed=args.seed,
                context_path=context_path,
                output_path=output_path,
            )
            if not sidecar_path.exists():
                write_json(sidecar_path, case_payload)
            append_jsonl(per_case_jsonl, case_payload)
            continue

        print(f"[generate][shard {args.shard_id}] {output_name} | seed={args.seed}")
        video = generate_one_video(
            pipe=pipe,
            context_path=context_path,
            prompt=row["description"],
            negative_prompt=args.negative_prompt,
            seed=args.seed,
            height=args.height,
            width=args.width,
            num_frames=args.num_frames,
            fps=args.fps,
            cfg_scale=args.cfg_scale,
            num_inference_steps=args.num_inference_steps,
            context_frames=args.context_frames,
        )
        save_video(video, str(output_path), fps=args.fps, quality=args.quality)

        case_payload = build_case_metadata(
            args=args,
            row=row,
            index=index,
            seed=args.seed,
            context_path=context_path,
            output_path=output_path,
        )
        write_json(sidecar_path, case_payload)
        append_jsonl(per_case_jsonl, case_payload)


def merge_shard_jsonl_files(metadata_dir: Path, model_name: str, num_shards: int) -> Path | None:
    if num_shards <= 1:
        path = metadata_dir / f"{model_name}_per_case.jsonl"
        return path if path.exists() else None

    merged_path = metadata_dir / f"{model_name}_per_case.jsonl"
    shard_paths = [
        metadata_dir / f"{model_name}_per_case_shard{shard_id:02d}of{num_shards:02d}.jsonl"
        for shard_id in range(num_shards)
    ]
    merged_entries: list[dict[str, Any]] = []
    for path in shard_paths:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                merged_entries.append(json.loads(line))
    if not merged_entries:
        return None

    merged_entries.sort(key=lambda x: x.get("index_in_sorted_list", 10**18))
    with merged_path.open("w", encoding="utf-8") as f:
        for entry in merged_entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return merged_path


def run_evaluation(
    benchmark_root: Path,
    descriptions_csv: Path,
    generated_dir: Path,
    eval_output_dir: Path,
) -> None:
    eval_output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "code/run_physics_iq.py",
        "--input_folders",
        str(generated_dir),
        "--output_folder",
        str(eval_output_dir),
        "--descriptions_file",
        str(descriptions_csv),
    ]
    print("[eval] " + " ".join(cmd))
    subprocess.run(cmd, cwd=benchmark_root, check=True)


def detect_visible_gpu_tokens() -> list[str]:
    env = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if env:
        tokens = [token.strip() for token in env.split(",") if token.strip()]
        if tokens:
            return tokens
    if not torch.cuda.is_available():
        return []
    return [str(i) for i in range(torch.cuda.device_count())]


def launch_multi_gpu_workers(args: argparse.Namespace, generated_dir: Path, metadata_dir: Path, eval_output_dir: Path) -> None:
    visible_gpu_tokens = detect_visible_gpu_tokens()
    if len(visible_gpu_tokens) <= 1:
        print("[multi_gpu] Only one visible GPU detected. Falling back to normal single-worker execution.")
        run_generation(args, generated_dir, metadata_dir)
        if not args.skip_evaluation:
            run_evaluation(
                benchmark_root=args.benchmark_root,
                descriptions_csv=args.descriptions_csv,
                generated_dir=generated_dir,
                eval_output_dir=eval_output_dir,
            )
        return

    num_shards = len(visible_gpu_tokens)
    script_path = Path(__file__).resolve()
    base_argv = sys.argv[1:]

    print(f"[multi_gpu] Launching {num_shards} workers on visible GPUs: {visible_gpu_tokens}")
    procs: list[tuple[int, str, subprocess.Popen[str]]] = []
    for shard_id, gpu_token in enumerate(visible_gpu_tokens):
        cmd = [
            sys.executable,
            str(script_path),
            *base_argv,
            "--worker",
            "--num_shards", str(num_shards),
            "--shard_id", str(shard_id),
            "--device", "cuda:0",
            "--skip_evaluation",
        ]
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = gpu_token
        print(f"[multi_gpu] shard {shard_id}: CUDA_VISIBLE_DEVICES={gpu_token}")
        proc = subprocess.Popen(cmd, env=env, text=True)
        procs.append((shard_id, gpu_token, proc))

    failed = False
    failed_info: list[tuple[int, str, int]] = []
    for shard_id, gpu_token, proc in procs:
        ret = proc.wait()
        if ret != 0:
            failed = True
            failed_info.append((shard_id, gpu_token, ret))

    if failed:
        raise RuntimeError(f"One or more worker processes failed: {failed_info}")

    merged_jsonl = merge_shard_jsonl_files(metadata_dir, args.model_name, num_shards)
    if merged_jsonl is not None:
        print(f"[multi_gpu] Merged shard metadata into: {merged_jsonl}")

    if not args.skip_evaluation:
        run_evaluation(
            benchmark_root=args.benchmark_root,
            descriptions_csv=args.descriptions_csv,
            generated_dir=generated_dir,
            eval_output_dir=eval_output_dir,
        )


def main() -> None:
    args = parse_args()
    assert_exists(args.benchmark_root, "Benchmark root")
    assert_exists(args.dataset_root, "Dataset root")
    assert_exists(args.wan_root, "Wan root")
    assert_exists(args.descriptions_csv, "Descriptions CSV")
    validate_args(args)

    generated_dir = args.output_root / "generated_videos" / args.model_name
    eval_output_dir = args.output_root / "eval_outputs"
    metadata_dir = args.output_root / "metadata" / args.model_name
    args.output_root.mkdir(parents=True, exist_ok=True)

    if args.multi_gpu and not args.worker and not args.skip_generation:
        launch_multi_gpu_workers(args, generated_dir, metadata_dir, eval_output_dir)
        print(f"Generated videos: {generated_dir}")
        print(f"Evaluation outputs: {eval_output_dir}")
        print(f"Metadata outputs: {metadata_dir}")
        return

    if not args.skip_generation:
        run_generation(args, generated_dir, metadata_dir)

    if not args.skip_evaluation:
        if args.num_shards > 1 and not args.worker:
            merge_shard_jsonl_files(metadata_dir, args.model_name, args.num_shards)
        run_evaluation(
            benchmark_root=args.benchmark_root,
            descriptions_csv=args.descriptions_csv,
            generated_dir=generated_dir,
            eval_output_dir=eval_output_dir,
        )

    print(f"Generated videos: {generated_dir}")
    print(f"Evaluation outputs: {eval_output_dir}")
    print(f"Metadata outputs: {metadata_dir}")


if __name__ == "__main__":
    main()
