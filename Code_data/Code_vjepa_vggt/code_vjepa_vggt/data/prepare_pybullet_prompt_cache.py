from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from typing import Any

import torch
import torch.distributed as dist
from safetensors import safe_open
from safetensors.torch import save_file

from code_vjepa_vggt.data.prepare_pybullet_vae_cache import (
    _build_dataset as _build_cache_dataset,
    _dataset_identity,
    _distributed_context,
)
from code_vjepa_vggt.data.pybullet_prompt_cache import (
    ATTENTION_MASK_TENSOR_KEY,
    CACHE_SCHEMA_VERSION,
    PROMPT_TENSOR_KEY,
    PromptIndexEntry,
    PyBulletPromptCacheError,
    PyBulletPromptEmbeddingCache,
    SamplePromptIndexEntry,
    embedding_relative_path,
    prompt_sha256,
    tokenizer_fingerprint,
)
from code_vjepa_vggt.data.pybullet_vae_cache import (
    canonical_json,
    encoding_id,
    sample_uid,
    sha256_bytes,
    sha256_file,
    torch_dtype_name,
)


DEFAULT_CACHE_NAME = "prompt_embeddings_wan22_umt5_bf16"


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _broadcast_text(value: str | None, world_size: int) -> str:
    if world_size == 1:
        if value is None:
            raise RuntimeError("rank 0 did not provide a value")
        return value
    payload = [value]
    dist.broadcast_object_list(payload, src=0)
    return str(payload[0])


def _clean_text_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return list(dict.fromkeys(str(item).strip() for item in value if str(item).strip()))


def _sample_prompt_roles(record) -> dict[str, str | list[str]]:
    prompt_roles = getattr(record, "prompt_roles", None)
    if prompt_roles is not None:
        return {
            "positive_prompt": str(prompt_roles["positive_prompt"]),
            "negative_prompt": str(prompt_roles["negative_prompt"]),
            "object_phrases": _clean_text_list(prompt_roles["object_phrases"]),
            "dynamic_object_phrases": _clean_text_list(
                prompt_roles["dynamic_object_phrases"]
            ),
            "static_object_phrases": _clean_text_list(
                prompt_roles["static_object_phrases"]
            ),
        }
    manifest = json.loads(Path(record.manifest_path).read_text(encoding="utf-8"))
    return {
        "positive_prompt": str(record.caption),
        "negative_prompt": str(record.negative_prompt),
        "object_phrases": _clean_text_list(manifest.get("object_phrases")),
        "dynamic_object_phrases": _clean_text_list(
            manifest.get("dynamic_object_phrases")
        ),
        "static_object_phrases": _clean_text_list(manifest.get("static_object_phrases")),
    }


def _append_prompt_suffix(prompt: str, suffix: str) -> str:
    prompt = str(prompt).strip()
    suffix = str(suffix).strip()
    if not suffix or prompt.endswith(suffix):
        return prompt
    return f"{prompt} {suffix}"


def _collect_prompts(
    dataset,
    positive_prompt_suffix: str = "",
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    prompts: dict[str, str] = {}
    sample_roles: dict[str, dict[str, Any]] = {}
    for record in dataset.samples:
        roles = _sample_prompt_roles(record)
        roles["positive_prompt"] = _append_prompt_suffix(
            str(roles["positive_prompt"]), positive_prompt_suffix
        )
        hashed_roles: dict[str, Any] = {}
        for role, value in roles.items():
            texts = [value] if isinstance(value, str) else value
            hashes = []
            for text in texts:
                prompt_hash = prompt_sha256(text)
                existing = prompts.setdefault(prompt_hash, text)
                if existing != text:
                    raise RuntimeError(f"SHA256 collision for prompt: {prompt_hash}")
                hashes.append(prompt_hash)
            hashed_roles[role] = hashes[0] if isinstance(value, str) else hashes
        sample_roles[record.key] = hashed_roles
    return prompts, sample_roles


def _load_text_encoder(
    checkpoint_path: Path,
    tokenizer_path: Path,
    device: torch.device,
    diffsynth_root: str | Path,
):
    diffsynth_root = Path(diffsynth_root).expanduser().resolve()
    if not (diffsynth_root / "diffsynth").is_dir():
        raise FileNotFoundError(f"DiffSynth package not found under: {diffsynth_root}")
    if str(diffsynth_root) not in sys.path:
        sys.path.insert(0, str(diffsynth_root))
    from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline

    pipe = WanVideoPipeline.from_pretrained(
        torch_dtype=torch.bfloat16,
        device=device,
        model_configs=[ModelConfig(path=str(checkpoint_path))],
        tokenizer_config=ModelConfig(path=str(tokenizer_path)),
        redirect_common_files=False,
    )
    if pipe.text_encoder is None or pipe.tokenizer is None:
        raise RuntimeError("DiffSynth did not load the Wan text encoder and tokenizer")
    pipe.text_encoder.eval().requires_grad_(False)
    return pipe


@torch.inference_mode()
def _encode_prompts(pipe, prompts: list[str]) -> list[tuple[torch.Tensor, torch.Tensor]]:
    ids, mask = pipe.tokenizer(prompts, return_mask=True, add_special_tokens=True)
    ids = ids.to(pipe.device)
    mask = mask.to(pipe.device)
    embeddings = pipe.text_encoder(ids, mask)
    seq_lens = mask.gt(0).sum(dim=1).long()
    outputs = []
    for index, valid_length in enumerate(seq_lens.tolist()):
        embedding = embeddings[index].clone()
        embedding[int(valid_length) :] = 0
        outputs.append(
            (
                embedding.to(device="cpu", dtype=torch.bfloat16).contiguous(),
                mask[index].to(device="cpu", dtype=torch.int64).contiguous(),
            )
        )
    return outputs


def _read_existing_entry(
    path: Path,
    *,
    expected_prompt: str,
    expected_encoding_id: str,
) -> PromptIndexEntry | None:
    if not path.is_file():
        return None
    prompt_hash = prompt_sha256(expected_prompt)
    try:
        with safe_open(path, framework="pt", device="cpu") as handle:
            metadata = handle.metadata() or {}
            if (
                metadata.get("prompt_sha256") != prompt_hash
                or metadata.get("encoding_id") != expected_encoding_id
            ):
                return None
            embedding = handle.get_tensor(PROMPT_TENSOR_KEY)
            mask = handle.get_tensor(ATTENTION_MASK_TENSOR_KEY)
        return PromptIndexEntry(
            prompt_sha256=prompt_hash,
            prompt=expected_prompt,
            embedding_file=embedding_relative_path(prompt_hash),
            embedding_shape=tuple(int(value) for value in embedding.shape),
            embedding_dtype=torch_dtype_name(embedding.dtype),
            valid_token_count=int(mask.sum().item()),
        )
    except Exception:  # noqa: BLE001
        return None


def _entry_from_file(
    cache_dir: Path,
    prompt_hash: str,
    prompt: str,
    expected_encoding_id: str,
) -> PromptIndexEntry:
    path = cache_dir / embedding_relative_path(prompt_hash)
    entry = _read_existing_entry(
        path,
        expected_prompt=prompt,
        expected_encoding_id=expected_encoding_id,
    )
    if entry is None:
        raise PyBulletPromptCacheError(f"Missing or invalid prompt embedding: {path}")
    return entry


def _build_sample_entry(root: Path, record, roles: dict[str, Any]) -> SamplePromptIndexEntry:
    manifest_path = Path(record.manifest_path).resolve()
    stat = manifest_path.stat()
    return SamplePromptIndexEntry(
        logical_key=record.key,
        sample_uid=sample_uid(record.key),
        source_relpath=manifest_path.relative_to(root).as_posix(),
        source_size=stat.st_size,
        source_mtime_ns=stat.st_mtime_ns,
        source_sha256=sha256_file(manifest_path),
        positive_prompt=str(roles["positive_prompt"]),
        negative_prompt=str(roles["negative_prompt"]),
        object_phrases=tuple(roles["object_phrases"]),
        dynamic_object_phrases=tuple(roles["dynamic_object_phrases"]),
        static_object_phrases=tuple(roles["static_object_phrases"]),
    )


def _write_documentation(cache_dir: Path) -> None:
    readme = """# PyBullet Wan2.2 Prompt Embedding Cache

This cache stores deduplicated Wan2.2 UMT5 embeddings for every positive prompt,
negative prompt, object phrase, dynamic object phrase, and static object phrase in
the PyBullet dataset. `index.jsonl` maps stable dataset sample UIDs to prompt
hashes; `prompt_index.jsonl` maps each unique prompt hash to a safetensors file.

Training uses the positive prompt embedding and bypasses repeated tokenizer/UMT5
execution. All tensors preserve the online
512-token padded representation, including zeroed invalid-token embeddings.
"""
    reader = """from __future__ import annotations

import argparse
from pathlib import Path

from code_vjepa_vggt.data.pybullet_prompt_cache import PyBulletPromptEmbeddingCache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--logical-key", default="F1/0717_f1_attempt000000")
    args = parser.parse_args()
    cache = PyBulletPromptEmbeddingCache(Path(__file__).resolve().parent)
    embedding = cache.load(args.logical_key)
    print(f"logical_key: {args.logical_key}")
    print(f"embedding: shape={tuple(embedding.shape)} dtype={embedding.dtype}")


if __name__ == "__main__":
    main()
"""
    _atomic_write_text(cache_dir / "README.md", readme)
    _atomic_write_text(cache_dir / "read_cache_example.py", reader)


def build_cache(args: argparse.Namespace) -> None:
    rank, world_size, device_index = _distributed_context()
    device = torch.device("cuda", device_index)
    root = Path(args.pybullet_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir or root / DEFAULT_CACHE_NAME).expanduser().resolve()
    wan_root = Path(args.wan_root).expanduser().resolve()
    checkpoint_path = wan_root / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer_path = wan_root / "google" / "umt5-xxl"
    if not checkpoint_path.is_file() or not tokenizer_path.is_dir():
        raise FileNotFoundError("Wan text encoder checkpoint or tokenizer is missing")

    dataset = _build_cache_dataset(args)
    dataset_name, vae_cache_kind = _dataset_identity(args)
    cache_kind = vae_cache_kind.replace("vae_latents", "prompt_embeddings")
    prompts, sample_roles = _collect_prompts(
        dataset,
        positive_prompt_suffix=args.positive_prompt_suffix,
    )
    checkpoint_hash = _broadcast_text(
        sha256_file(checkpoint_path) if rank == 0 else None,
        world_size,
    )
    tokenizer_hash = _broadcast_text(
        tokenizer_fingerprint(tokenizer_path) if rank == 0 else None,
        world_size,
    )
    checkpoint_stat = checkpoint_path.stat()
    encoding_payload = {
        "cache_schema_version": CACHE_SCHEMA_VERSION,
        "text_encoder": {
            "class": "WanTextEncoder",
            "checkpoint_name": checkpoint_path.name,
            "checkpoint_size": checkpoint_stat.st_size,
            "checkpoint_sha256": checkpoint_hash,
            "output_dim": 4096,
        },
        "tokenizer": {
            "class": "HuggingfaceTokenizer",
            "fingerprint": tokenizer_hash,
            "sequence_length": 512,
            "clean": "whitespace",
            "add_special_tokens": True,
            "padding": "max_length",
            "truncation": True,
        },
        "preprocess": {
            "dtype": "bfloat16",
            "invalid_token_embeddings_zeroed": True,
            "deduplicate_by": "raw_prompt_sha256",
            "version": 1,
        },
    }
    current_encoding_id = encoding_id(encoding_payload)
    config_path = cache_dir / "cache_config.json"
    if rank == 0:
        cache_dir.mkdir(parents=True, exist_ok=True)
        if config_path.is_file():
            existing = json.loads(config_path.read_text(encoding="utf-8"))
            if existing.get("encoding_id") != current_encoding_id:
                raise RuntimeError(f"Existing prompt cache has incompatible encoding: {cache_dir}")
        config = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "cache_kind": cache_kind,
            "status": "building",
            "dataset_name": dataset_name,
            "dataset_root": str(root),
            "num_samples": len(dataset.samples),
                "num_unique_prompts": len(prompts),
                "positive_prompt_suffix": args.positive_prompt_suffix,
                "encoding_id": current_encoding_id,
            **encoding_payload,
        }
        _atomic_write_text(config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")
        _write_documentation(cache_dir)
    if world_size > 1:
        dist.barrier()

    pipe = _load_text_encoder(
        checkpoint_path,
        tokenizer_path,
        device,
        args.diffsynth_root,
    )
    assigned = [
        (prompt_hash, prompt)
        for prompt_hash, prompt in sorted(prompts.items())
        if int(prompt_hash, 16) % world_size == rank
    ]
    encoded_count = 0
    skipped_count = 0
    started = time.monotonic()
    for start in range(0, len(assigned), args.batch_size):
        batch = assigned[start : start + args.batch_size]
        pending = []
        for prompt_hash, prompt in batch:
            path = cache_dir / embedding_relative_path(prompt_hash)
            existing = _read_existing_entry(
                path,
                expected_prompt=prompt,
                expected_encoding_id=current_encoding_id,
            )
            if existing is None:
                pending.append((prompt_hash, prompt, path))
            else:
                skipped_count += 1
        if pending:
            encoded = _encode_prompts(pipe, [prompt for _, prompt, _ in pending])
            for (prompt_hash, prompt, path), (embedding, mask) in zip(pending, encoded):
                path.parent.mkdir(parents=True, exist_ok=True)
                temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
                save_file(
                    {
                        PROMPT_TENSOR_KEY: embedding,
                        ATTENTION_MASK_TENSOR_KEY: mask,
                    },
                    temporary,
                    metadata={
                        "prompt_sha256": prompt_hash,
                        "encoding_id": current_encoding_id,
                        "prompt": prompt,
                    },
                )
                os.replace(temporary, path)
                encoded_count += 1
        processed = min(start + args.batch_size, len(assigned))
        if processed == len(assigned) or processed % args.log_every < args.batch_size:
            elapsed = max(time.monotonic() - started, 1e-6)
            print(
                f"[prompt-cache][rank {rank}] {processed}/{len(assigned)} "
                f"encoded={encoded_count} skipped={skipped_count} "
                f"rate={processed / elapsed:.3f} prompts/s",
                flush=True,
            )

    if world_size > 1:
        dist.barrier()
    if rank == 0:
        prompt_entries = [
            _entry_from_file(cache_dir, prompt_hash, prompt, current_encoding_id)
            for prompt_hash, prompt in sorted(prompts.items())
        ]
        sample_entries = [
            _build_sample_entry(root, record, sample_roles[record.key])
            for record in dataset.samples
        ]
        shapes = {entry.embedding_shape for entry in prompt_entries}
        dtypes = {entry.embedding_dtype for entry in prompt_entries}
        if shapes != {(512, 4096)} or dtypes != {"bfloat16"}:
            raise RuntimeError(f"Unexpected prompt cache tensors: shapes={shapes}, dtypes={dtypes}")
        prompt_index_text = "".join(
            canonical_json(entry.to_json()) + "\n" for entry in prompt_entries
        )
        sample_index_text = "".join(
            canonical_json(entry.to_json()) + "\n" for entry in sample_entries
        )
        _atomic_write_text(cache_dir / "prompt_index.jsonl", prompt_index_text)
        _atomic_write_text(cache_dir / "index.jsonl", sample_index_text)
        config = json.loads(config_path.read_text(encoding="utf-8"))
        config.update(
            {
                "status": "complete",
                "embedding": {
                    "tensor_key": PROMPT_TENSOR_KEY,
                    "attention_mask_tensor_key": ATTENTION_MASK_TENSOR_KEY,
                    "shape": [512, 4096],
                    "dtype": "bfloat16",
                },
                "prompt_index_file": "prompt_index.jsonl",
                "prompt_index_sha256": sha256_bytes(prompt_index_text.encode("utf-8")),
                "index_file": "index.jsonl",
                "index_sha256": sha256_bytes(sample_index_text.encode("utf-8")),
            }
        )
        _atomic_write_text(config_path, json.dumps(config, indent=2, sort_keys=True) + "\n")
        print(
            f"[prompt-cache][complete] samples={len(sample_entries)} "
            f"unique_prompts={len(prompt_entries)} cache_dir={cache_dir}",
            flush=True,
        )
    if world_size > 1:
        dist.barrier()
        dist.destroy_process_group()


def _comparison_metrics(online: torch.Tensor, cached: torch.Tensor) -> dict[str, Any]:
    delta = (online.float() - cached.float()).abs()
    return {
        "exact_equal": bool(torch.equal(online, cached)),
        "max_abs_error": float(delta.max().item()),
        "mean_abs_error": float(delta.mean().item()),
    }


def verify_cache(args: argparse.Namespace) -> None:
    root = Path(args.pybullet_root).expanduser().resolve()
    wan_root = Path(args.wan_root).expanduser().resolve()
    cache_dir = Path(args.cache_dir or root / DEFAULT_CACHE_NAME).expanduser().resolve()
    checkpoint_path = wan_root / "models_t5_umt5-xxl-enc-bf16.pth"
    tokenizer_path = wan_root / "google" / "umt5-xxl"
    dataset = _build_cache_dataset(args)
    cache = PyBulletPromptEmbeddingCache(
        cache_dir,
        text_encoder_checkpoint_path=checkpoint_path,
        tokenizer_path=tokenizer_path,
    )
    cache.validate_records(dataset.samples, root)
    checked = 0
    for prompt_hash in cache.prompts:
        cache.load_prompt_hash(prompt_hash)
        checked += 1
        if checked % args.log_every == 0 or checked == len(cache.prompts):
            print(f"[prompt-cache][verify] {checked}/{len(cache.prompts)}", flush=True)

    comparisons = []
    if args.online_compare_samples > 0:
        if not torch.cuda.is_available():
            raise RuntimeError("Online prompt comparison requires CUDA")
        pipe = _load_text_encoder(
            checkpoint_path,
            tokenizer_path,
            torch.device("cuda", 0),
            args.diffsynth_root,
        )
        selected = list(cache.prompts.values())[: args.online_compare_samples]
        for entry in selected:
            online, _ = _encode_prompts(pipe, [entry.prompt])[0]
            cached, _ = cache.load_prompt_hash(entry.prompt_sha256)
            row = {"prompt_sha256": entry.prompt_sha256, **_comparison_metrics(online, cached)}
            comparisons.append(row)
            print(f"[prompt-cache][compare] {row}", flush=True)
            if not row["exact_equal"]:
                raise RuntimeError(f"Online/cache prompt embedding mismatch: {row}")

    report = {
        "status": "passed",
        "num_sample_index_entries": len(cache.entries),
        "num_unique_prompt_entries": len(cache.prompts),
        "num_tensor_files_checked": checked,
        "online_comparisons": comparisons,
    }
    _atomic_write_text(
        cache_dir / "reports" / "verification.json",
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(f"[prompt-cache][verify-complete] {canonical_json(report)}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build and verify PyBullet Wan prompt cache")
    parser.add_argument("command", choices=("build", "verify", "inspect"))
    parser.add_argument("--pybullet-root", required=True)
    parser.add_argument(
        "--dataset-format",
        choices=("pybullet0713", "pybullet0613_raw"),
        default="pybullet0713",
    )
    parser.add_argument("--wan-root", required=True)
    parser.add_argument("--diffsynth-root", required=True)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--width", type=int, default=896)
    parser.add_argument("--num-frames", type=int, default=49)
    parser.add_argument("--num-context-frames", type=int, default=8)
    parser.add_argument("--sampling-strategy", choices=("prefix", "uniform"), default="prefix")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument("--online-compare-samples", type=int, default=0)
    parser.add_argument("--positive-prompt-suffix", default="")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "build":
        build_cache(args)
    elif args.command == "verify":
        verify_cache(args)
    else:
        root = Path(args.pybullet_root).expanduser().resolve()
        cache_dir = Path(args.cache_dir or root / DEFAULT_CACHE_NAME).expanduser().resolve()
        print((cache_dir / "cache_config.json").read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
