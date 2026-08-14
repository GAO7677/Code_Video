from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import torch
import torch.distributed as dist
from PIL import Image
from safetensors import safe_open
from safetensors.torch import save_file

from code_vjepa_vggt.data.pybullet0713_no_gt_box_dataset import (
    PyBullet0713NoGTBoxDataset,
)
from code_vjepa_vggt.data.pybullet_vae_cache import (
    CACHE_SCHEMA_VERSION,
    LATENT_TENSOR_KEY,
    CacheIndexEntry,
    PyBulletVaeCacheError,
    PyBulletVaeLatentCache,
    canonical_json,
    encoding_id,
    latent_relative_path,
    sample_uid,
    sha256_bytes,
    sha256_file,
    torch_dtype_name,
)


DEFAULT_CACHE_NAME = "vae_latents_wan22_512x896_49f_prefix_bf16"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _distributed_context() -> tuple[int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    visible_device_count = torch.cuda.device_count()
    if visible_device_count == 0:
        raise RuntimeError("Wan VAE cache generation requires at least one visible CUDA device")
    device_index = local_rank % visible_device_count
    torch.cuda.set_device(device_index)
    if world_size > 1 and not dist.is_initialized():
        dist.init_process_group(backend="gloo")
    return rank, world_size, device_index


def _broadcast_text(value: str | None, rank: int, world_size: int) -> str:
    if world_size == 1:
        if value is None:
            raise RuntimeError("rank 0 did not provide a value")
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=0)
    return str(payload[0])


def _tensor_video_to_pil_list(video: torch.Tensor) -> list[Image.Image]:
    frames = video.detach().cpu().permute(1, 2, 3, 0)
    frames = ((frames + 1.0) * 127.5).clamp(0, 255).to(torch.uint8).numpy()
    return [Image.fromarray(frame) for frame in frames]


def _load_vae(vae_path: Path, device: torch.device, dtype: torch.dtype):
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=dtype,
        device=device,
        model_configs=[ModelConfig(path=str(vae_path))],
        tokenizer_config=None,
        redirect_common_files=False,
    )
    if pipe.vae is None:
        raise RuntimeError(f"DiffSynth did not load a VAE from {vae_path}")
    pipe.vae.eval().requires_grad_(False)
    return pipe


@torch.inference_mode()
def _encode_sample(pipe, video: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    input_video = pipe.preprocess_video(_tensor_video_to_pil_list(video))
    latent = pipe.vae.encode(input_video, device=pipe.device, tiled=False)
    if latent.ndim != 5 or int(latent.shape[0]) != 1:
        raise RuntimeError(f"Unexpected Wan VAE latent shape: {tuple(latent.shape)}")
    return latent[0].to(device="cpu", dtype=dtype).contiguous()


def _build_dataset(args: argparse.Namespace) -> PyBullet0713NoGTBoxDataset:
    return PyBullet0713NoGTBoxDataset(
        root=args.pybullet_root,
        split="all",
        resolution=(args.height, args.width),
        num_frames=args.num_frames,
        num_context_frames=min(args.num_context_frames, args.num_frames),
        sampling_strategy=args.sampling_strategy,
    )


def _read_existing_entry(
    path: Path,
    *,
    logical_key: str,
    uid: str,
    expected_encoding_id: str,
    source_sha256: str,
) -> CacheIndexEntry | None:
    if not path.is_file():
        return None
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if (
                metadata.get("logical_key") != logical_key
                or metadata.get("sample_uid") != uid
                or metadata.get("encoding_id") != expected_encoding_id
                or metadata.get("source_sha256") != source_sha256
            ):
                return None
            latent = handle.get_tensor(LATENT_TENSOR_KEY)
        return CacheIndexEntry(
            logical_key=logical_key,
            sample_uid=uid,
            source_relpath=str(metadata["source_relpath"]),
            source_size=int(metadata["source_size"]),
            source_mtime_ns=int(metadata["source_mtime_ns"]),
            source_sha256=source_sha256,
            sampled_frame_indices=tuple(json.loads(metadata["sampled_frame_indices"])),
            latent_file=path.relative_to(path.parents[2]).as_posix(),
            latent_shape=tuple(int(value) for value in latent.shape),
            latent_dtype=torch_dtype_name(latent.dtype),
        )
    except Exception:  # noqa: BLE001
        return None


def _entry_from_file(cache_dir: Path, record, expected_encoding_id: str) -> CacheIndexEntry:
    uid = sample_uid(record.key)
    path = cache_dir / latent_relative_path(uid)
    if not path.is_file():
        raise PyBulletVaeCacheError(f"Missing latent after build: {record.key} -> {path}")
    with safe_open(path, framework="pt", device="cpu") as handle:
        metadata = handle.metadata() or {}
        latent = handle.get_tensor(LATENT_TENSOR_KEY)
    if (
        metadata.get("logical_key") != record.key
        or metadata.get("sample_uid") != uid
        or metadata.get("encoding_id") != expected_encoding_id
    ):
        raise PyBulletVaeCacheError(f"Invalid latent metadata after build: {path}")
    return CacheIndexEntry(
        logical_key=record.key,
        sample_uid=uid,
        source_relpath=str(metadata["source_relpath"]),
        source_size=int(metadata["source_size"]),
        source_mtime_ns=int(metadata["source_mtime_ns"]),
        source_sha256=str(metadata["source_sha256"]),
        sampled_frame_indices=tuple(json.loads(metadata["sampled_frame_indices"])),
        latent_file=latent_relative_path(uid),
        latent_shape=tuple(int(value) for value in latent.shape),
        latent_dtype=torch_dtype_name(latent.dtype),
    )


def build_cache(args: argparse.Namespace) -> None:
    rank, world_size, device_index = _distributed_context()
    if not torch.cuda.is_available():
        raise RuntimeError("Wan VAE cache generation requires CUDA")
    device = torch.device("cuda", device_index)
    dtype = torch.bfloat16
    root = Path(args.pybullet_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir or root / DEFAULT_CACHE_NAME).expanduser().resolve()
    vae_path = Path(args.wan_root).expanduser().resolve() / "Wan2.2_VAE.pth"
    if not vae_path.is_file():
        raise FileNotFoundError(f"Wan2.2 VAE checkpoint not found: {vae_path}")
    if cache_dir.parent != root:
        raise ValueError(
            f"Cache directory must be directly under the PyBullet root: root={root}, "
            f"cache_dir={cache_dir}"
        )

    dataset = _build_dataset(args)
    keys = [record.key for record in dataset.samples]
    uids = [sample_uid(key) for key in keys]
    if len(keys) != len(set(keys)) or len(uids) != len(set(uids)):
        raise RuntimeError("PyBullet logical keys or stable sample UIDs are not unique")

    vae_sha256 = _broadcast_text(sha256_file(vae_path) if rank == 0 else None, rank, world_size)
    vae_stat = vae_path.stat()
    encoding_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "vae": {
            "class": "WanVideoVAE38",
            "checkpoint_name": vae_path.name,
            "checkpoint_size": vae_stat.st_size,
            "checkpoint_sha256": vae_sha256,
            "z_dim": 48,
            "upsampling_factor": 16,
        },
        "preprocess": {
            "height": args.height,
            "width": args.width,
            "num_frames": args.num_frames,
            "sampling_strategy": args.sampling_strategy,
            "value_range": "minus_one_to_one",
            "pil_uint8_roundtrip": True,
            "tiled": False,
            "framewise_encoding": False,
            "dtype": "bfloat16",
            "version": 1,
        },
    }
    current_encoding_id = encoding_id(encoding_payload)
    config_path = cache_dir / "cache_config.json"
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        existing = None
        if config_path.is_file():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing is not None and existing.get("encoding_id") != current_encoding_id:
            raise RuntimeError(
                "Existing cache directory uses a different encoding configuration: "
                f"{cache_dir}"
            )
        config = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_kind": "pybullet0713_wan_vae_latents",
            "status": "building",
            "dataset_name": "pybullet0713",
            "dataset_root": str(root),
            "num_samples": len(dataset.samples),
            "encoding_id": current_encoding_id,
            **encoding_payload,
        }
        _atomic_write_text(config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")
    if world_size > 1:
        dist.barrier()

    pipe = _load_vae(vae_path, device=device, dtype=dtype)
    assigned = [
        record
        for record in dataset.samples
        if int(sample_uid(record.key), 16) % world_size == rank
    ]
    encoded_count = 0
    skipped_count = 0
    started = time.monotonic()
    for local_index, record in enumerate(assigned, start=1):
        uid = sample_uid(record.key)
        output_path = cache_dir / latent_relative_path(uid)
        video_path = Path(record.video_path).resolve()
        video_stat = video_path.stat()
        source_sha256 = sha256_file(video_path)
        existing = _read_existing_entry(
            output_path,
            logical_key=record.key,
            uid=uid,
            expected_encoding_id=current_encoding_id,
            source_sha256=source_sha256,
        )
        if existing is not None:
            skipped_count += 1
        else:
            sample = dataset._load_sample(record)
            latent = _encode_sample(pipe, sample["video"], dtype)
            source_relpath = video_path.relative_to(root).as_posix()
            metadata = {
                "logical_key": record.key,
                "sample_uid": uid,
                "encoding_id": current_encoding_id,
                "source_relpath": source_relpath,
                "source_size": str(video_stat.st_size),
                "source_mtime_ns": str(video_stat.st_mtime_ns),
                "source_sha256": source_sha256,
                "sampled_frame_indices": canonical_json(
                    sample["metadata"]["sampled_frame_indices"]
                ),
            }
            output_path.parent.mkdir(parents=True, exist_ok=True)
            temporary = output_path.with_name(f".{output_path.name}.{os.getpid()}.tmp")
            save_file({LATENT_TENSOR_KEY: latent}, temporary, metadata=metadata)
            os.replace(temporary, output_path)
            encoded_count += 1
        if local_index == 1 or local_index % args.log_every == 0 or local_index == len(assigned):
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"[vae-cache][rank {rank}] {local_index}/{len(assigned)} "
                f"encoded={encoded_count} skipped={skipped_count} "
                f"rate={local_index / elapsed:.3f} samples/s",
                flush=True,
            )

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        entries = [
            _entry_from_file(cache_dir, record, current_encoding_id)
            for record in dataset.samples
        ]
        shapes = {entry.latent_shape for entry in entries}
        dtypes = {entry.latent_dtype for entry in entries}
        if len(shapes) != 1 or dtypes != {"bfloat16"}:
            raise RuntimeError(f"Inconsistent cached latents: shapes={shapes}, dtypes={dtypes}")
        index_text = "".join(canonical_json(entry.to_json()) + "\n" for entry in entries)
        _atomic_write_text(cache_dir / "index.jsonl", index_text)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(
            {
                "status": "complete",
                "latent": {
                    "tensor_key": LATENT_TENSOR_KEY,
                    "shape": list(next(iter(shapes))),
                    "dtype": "bfloat16",
                },
                "index_file": "index.jsonl",
                "index_sha256": sha256_bytes(index_text.encode("utf-8")),
            }
        )
        _atomic_write_text(config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")
        print(
            f"[vae-cache][complete] cache_dir={cache_dir} samples={len(entries)} "
            f"shape={next(iter(shapes))} dtype=bfloat16",
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def verify_cache(args: argparse.Namespace) -> None:
    root = Path(args.pybullet_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir or root / DEFAULT_CACHE_NAME).expanduser().resolve()
    dataset = _build_dataset(args)
    cache = PyBulletVaeLatentCache(
        cache_dir,
        resolution=(args.height, args.width),
        num_frames=args.num_frames,
        sampling_strategy=args.sampling_strategy,
    )
    cache.validate_records(dataset.samples, root)
    checked = 0
    for record in dataset.samples:
        cache.load(record.key)
        checked += 1
        if checked % args.log_every == 0 or checked == len(dataset.samples):
            print(f"[vae-cache][verify] {checked}/{len(dataset.samples)}", flush=True)

    comparisons: list[dict[str, Any]] = []
    if args.online_compare_samples > 0:
        if not torch.cuda.is_available():
            raise RuntimeError("Online comparison requires CUDA")
        device = torch.device("cuda", 0)
        vae_path = Path(args.wan_root).expanduser().resolve() / "Wan2.2_VAE.pth"
        pipe = _load_vae(vae_path, device=device, dtype=torch.bfloat16)
        for record in dataset.samples[: args.online_compare_samples]:
            sample = dataset._load_sample(record)
            online = _encode_sample(pipe, sample["video"], torch.bfloat16)
            cached = cache.load(record.key)
            delta = (online.float() - cached.float()).abs()
            row = {
                "logical_key": record.key,
                "max_abs_error": float(delta.max().item()),
                "mean_abs_error": float(delta.mean().item()),
                "equal": bool(torch.equal(online, cached)),
            }
            comparisons.append(row)
            print(f"[vae-cache][compare] {row}", flush=True)
            if not torch.equal(online, cached):
                raise RuntimeError(f"Online/cache latent mismatch: {row}")

    report = {
        "status": "passed",
        "num_index_entries": len(cache.entries),
        "num_tensor_files_checked": checked,
        "online_comparisons": comparisons,
    }
    reports_dir = cache_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    _atomic_write_text(
        reports_dir / "verification.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(f"[vae-cache][verify-complete] {canonical_json(report)}", flush=True)


def inspect_cache(args: argparse.Namespace) -> None:
    root = Path(args.pybullet_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir or root / DEFAULT_CACHE_NAME).expanduser().resolve()
    config = json.loads((cache_dir / "cache_config.json").read_text(encoding="utf-8"))
    print(json.dumps(config, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and verify PyBullet Wan2.2 VAE latents")
    parser.add_argument("command", choices=("build", "verify", "inspect"))
    parser.add_argument("--pybullet-root", required=True)
    parser.add_argument("--wan-root", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-context-frames", type=int, default=8)
    parser.add_argument("--sampling-strategy", choices=("prefix", "uniform"), default="prefix")
    parser.add_argument("--log-every", type=int, default=25)
    parser.add_argument("--online-compare-samples", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        build_cache(args)
    elif args.command == "verify":
        verify_cache(args)
    else:
        inspect_cache(args)


if __name__ == "__main__":
    main()
