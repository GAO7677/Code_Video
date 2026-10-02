#!/usr/bin/env python3
"""Build or online-verify native Pusa/Wan14B caches on an explicit CUDA device.

Examples:
  python audit_cache.py --config configs/base.json --mode inspect
  python audit_cache.py --config configs/base.json --mode build --component vae --device cuda:0
  python audit_cache.py --config configs/base.json --mode build --component text --device cuda:0

Components are loaded separately; no CPU model loading/offload is used.  Existing
legacy tensors are never reused on shape alone.  Optional legacy text reuse needs
an individual numerical comparison against the native encoder for that caption.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import gc
import html
import importlib.util
import json
import os
from pathlib import Path
import re
import time

from data import (SCHEMA_VERSION, atomic_json, cache_path, checked_entry, fingerprint,
                  load_records, make_cache_contract, read_rgb_prefix, sha256_file,
                  stable_sha, validate_config)


def _module(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _state_cuda(path, device):
    import torch
    if Path(path).suffix == ".safetensors":
        from safetensors.torch import load_file
        state = load_file(str(path), device=str(device))
    else:
        state = torch.load(path, map_location=device, weights_only=True, mmap=True)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if not isinstance(state, dict) or not all(isinstance(v, torch.Tensor) for v in state.values()):
        raise ValueError("Expected a flat native tensor state dict")
    if not all(v.device.type == "cuda" for v in state.values()):
        raise RuntimeError("All checkpoint tensors must load directly on CUDA")
    return state


class NativeEncoder:
    """Thin wrapper around the pinned official source, no pipeline reimplementation."""
    def __init__(self, config, component, device):
        import torch
        self.config, self.component = config, component
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("Model components must load on CUDA, never CPU")
        # Resolve visible-index -> physical-index for the permanent GPU4 ban.
        visible = os.environ.get("CUDA_VISIBLE_DEVICES")
        physical = str(self.device.index or 0)
        if visible:
            choices = [v.strip() for v in visible.split(",")]
            physical = choices[self.device.index or 0]
        if physical == "4":
            raise ValueError("Physical GPU4 is forbidden")
        cfg = config["cache"][component]
        source = cfg.get("source_file") or str(Path(__file__).parent / "vendor" /
                      ("wan_video_vae.py" if component == "vae" else "wan_video_text_encoder.py"))
        module = _module(source, "wan14b_cache_" + component)
        if component == "vae":
            # VAE is small; construct directly on CUDA so its unregistered mean /
            # std tensors are CUDA tensors as well. No temporary CPU model.
            with torch.device(self.device):
                self.model = module.WanVideoVAE(z_dim=16)
            state = _state_cuda(cfg["checkpoint"], self.device)
            if not all(k.startswith("model.") for k in state):
                state = {"model." + k: v for k, v in state.items()}
            state = {k: v.to(dtype=torch.float32) for k, v in state.items()}
            self.model.load_state_dict(state, strict=True, assign=True)
            self.model = self.model.to(device=self.device, dtype=torch.float32).eval().requires_grad_(False)
        else:
            from transformers import AutoTokenizer
            with torch.device("meta"):
                self.model = module.WanTextEncoder()
            state = _state_cuda(cfg["checkpoint"], self.device)
            state = {k: v.to(dtype=torch.float32) for k, v in state.items()}
            self.model.load_state_dict(state, strict=True, assign=True)
            self.model = self.model.eval().requires_grad_(False)
            self.tokenizer = AutoTokenizer.from_pretrained(cfg["tokenizer"], local_files_only=True)
        del state
        if any(p.device.type != "cuda" or p.dtype != torch.float32 for p in self.model.parameters()):
            raise RuntimeError("Native encoder must contain only CUDA FP32 parameters")

    def encode_video(self, video):
        import torch
        if self.component != "vae":
            raise RuntimeError("Load the VAE component first")
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            # Calling single_encode avoids official encode()'s .to('cpu') staging.
            result = self.model.single_encode(video.unsqueeze(0).to(self.device, dtype=torch.float32), self.device)[0]
            self.model.model.clear_cache()
            return result

    def encode_text(self, caption):
        import ftfy
        import torch
        if self.component != "text":
            raise RuntimeError("Load the text component first")
        cleaned = re.sub(r"\s+", " ", html.unescape(html.unescape(ftfy.fix_text(caption))).strip()).strip()
        tokens = self.tokenizer([cleaned], return_tensors="pt", padding="max_length",
                                truncation=True, max_length=512, add_special_tokens=True)
        ids, mask = tokens.input_ids.to(self.device), tokens.attention_mask.to(self.device)
        with torch.inference_mode(), torch.autocast("cuda", enabled=False):
            output = self.model(ids, mask)
            length = int(mask[0].gt(0).sum().item())
            output[:, length:] = 0
            return output[0]


def encoder_factory(config, component, device):
    return NativeEncoder(config, component, device)


def compare_tensors(reference, candidate, *, atol, rtol):
    import torch
    a = reference.detach().float()
    b = candidate.detach().to(device=a.device, dtype=torch.float32)
    if tuple(a.shape) != tuple(b.shape):
        return {"passed": False, "reason": "shape_mismatch", "reference_shape": list(a.shape),
                "candidate_shape": list(b.shape)}
    finite = bool(torch.isfinite(a).all().item() and torch.isfinite(b).all().item())
    delta = (a - b).abs()
    return {"passed": finite and bool(torch.allclose(a, b, atol=atol, rtol=rtol)),
            "finite": finite, "atol": atol, "rtol": rtol, "shape": list(a.shape),
            "max_abs": float(delta.max().item()), "mean_abs": float(delta.mean().item()),
            "rmse": float(delta.square().mean().sqrt().item())}


@contextmanager
def cache_lock(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        yield


def commit_entry(path, tensor_name, value, metadata):
    from safetensors.torch import save_file, load_file
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    save_file({tensor_name: value.detach().cpu().contiguous()}, str(temp),
              metadata={"contract_sha256": metadata["contract_sha256"], "schema_version": str(SCHEMA_VERSION)})
    with temp.open("rb") as f:
        os.fsync(f.fileno())
    if not torch.equal(load_file(str(temp), device="cpu")[tensor_name], value.detach().cpu()):
        temp.unlink(missing_ok=True)
        raise RuntimeError("Serialization roundtrip mismatch")
    metadata["tensor_file_sha256"] = sha256_file(temp)
    os.replace(temp, path)
    atomic_json(path.with_suffix(".json"), metadata)


def legacy_text_index(config):
    root = config["cache"].get("legacy_text_root")
    if not root:
        return {}
    root = Path(root)
    cfg = json.loads((root / "cache_config.json").read_text())
    if cfg.get("caption_field") != config["data"]["caption_field"]:
        raise ValueError("Legacy text cache caption field mismatch")
    result = {}
    with (root / cfg.get("index_file", "index.jsonl")).open() as f:
        for line in f:
            item = json.loads(line)
            sha = item.get("caption_sha256")
            if sha and item.get("positive_prompt") == sha:
                result[sha] = (root / item["embedding_file"], cfg["embedding"].get("tensor_key", "prompt_embedding"))
    return result


def inspect(config):
    records = load_records(config)
    unique = len({r["caption_sha256"] for r in records})
    d = config["data"]
    latent_bytes = len(records) * 16 * ((d["num_frames"] - 1) // 4 + 1) * (d["height"] // 8) * (d["width"] // 8) * 2
    text_bytes = unique * 512 * 4096 * 4
    return {"mode": "inspect", "samples": len(records), "unique_captions": unique,
            "sample_list_sha256": sha256_file(d["sample_list"]),
            "latent_shape": [16, 13, 64, 112], "text_shape": [512, 4096],
            "estimated_cache_bytes": {"vae": latent_bytes, "text": text_bytes, "total": latent_bytes + text_bytes},
            "legacy_vae_reuse": False, "legacy_vae_reason": "48-channel/16x caches are incompatible with native16-channel/8xVAE",
            "context_protocol": {"supplied_rgb": 8, "latent_frames": 2,
                                 "causal_complete_rgb": 5, "unused_incomplete_suffix_rgb": 3,
                                 "future_rgb_read_for_context": False,
                                 "note": "Native encoder iterates 1+(T-1)//4, consuming RGB0 then RGB1..4 for T=8; no RGB8 leakage."}}


def run(config, args):
    import torch
    from safetensors.torch import load_file
    records = load_records(config)
    component = args.component
    if component == "text":
        records = list({r["caption_sha256"]: r for r in reversed(records)}.values())[::-1]
    records = [r for i, r in enumerate(records) if i % args.num_shards == args.shard_index]
    if args.limit is not None:
        records = records[:args.limit]
    root = Path(config["cache"]["root"])
    contract = make_cache_contract(config, component)
    envelope = {"contract": contract, "sha256": stable_sha(contract)}
    contract_path = root / "contracts" / (component + ".json")
    with cache_lock(contract_path.with_suffix(".lock")):
        if contract_path.exists():
            if json.loads(contract_path.read_text()) != envelope:
                raise ValueError("Cache contract changed; use a fresh cache root rather than overwrite audited results")
        elif args.mode == "verify":
            raise FileNotFoundError(contract_path)
        else:
            atomic_json(contract_path, envelope)
    validation = config["cache"]["validation"]
    legacy = legacy_text_index(config) if component == "text" else {}
    encoder = None
    report = {"component": component, "mode": args.mode, "device": args.device,
              "contract_sha256": envelope["sha256"], "total": len(records), "verified": 0,
              "reused_current": 0, "reused_legacy": 0, "errors": [], "started": time.time()}
    report_path = root / "audit" / f"{args.mode}-{component}-shard{args.shard_index:03d}.json"
    for record in records:
        key = record["logical_key"] if component == "vae" else record["caption_sha256"]
        path = cache_path(root, component, key)
        with cache_lock(path.with_suffix(".lock")):
            if args.mode == "build" and path.exists() and path.with_suffix(".json").exists():
                checked_path, meta = checked_entry(config, record, component, envelope["sha256"])
                if sha256_file(checked_path) != meta["tensor_file_sha256"]:
                    raise ValueError(f"Existing cache file SHA mismatch: {checked_path}")
                report["reused_current"] += 1
                report["verified"] += 1
                atomic_json(report_path, report)
                continue
            if encoder is None:
                encoder = encoder_factory(config, component, args.device)
            meta = {"schema_version": SCHEMA_VERSION, "component": component,
                    "logical_key": record["logical_key"], "caption_sha256": record["caption_sha256"],
                    "contract_sha256": envelope["sha256"], "status": "verified", "time": time.time()}
            if component == "vae":
                video, preprocess = read_rgb_prefix(record, config)
                online = encoder.encode_video(video)
                if tuple(online.shape) != (16, 13, 64, 112):
                    raise ValueError(f"Unexpected native latent shape: {online.shape}")
                prefix = encoder.encode_video(video[:, :config["data"]["context_frames"]])
                audit = compare_tensors(online[:, :2], prefix,
                                        atol=validation["context_atol"], rtol=validation["context_rtol"])
                audit.update({"supplied_rgb": 8, "complete_causal_rgb": 5,
                              "ignored_incomplete_rgb": 3, "no_future_rgb": True})
                if not audit["passed"]:
                    raise RuntimeError(f"Causal prefix differs from full-video context: {record['logical_key']}: {audit}")
                source_fingerprint = fingerprint(record["video"])
                old_provenance = record.get("legacy_video_provenance")
                if old_provenance and source_fingerprint["sha256"] != old_provenance["source_sha256"]:
                    raise ValueError(f"Source RGB differs from original Wan5B cache provenance: {record['logical_key']}")
                meta.update({"source": source_fingerprint, "preprocess": preprocess, "context_causality": audit})
                candidate = online.to(dtype=torch.bfloat16)
                name = "latents"
            else:
                online = encoder.encode_text(record["caption"])
                if tuple(online.shape) != (512, 4096):
                    raise ValueError(f"Unexpected native text shape: {online.shape}")
                candidate = online.float()
                name = "text"
                if record["caption_sha256"] in legacy:
                    old_path, old_key = legacy[record["caption_sha256"]]
                    old = load_file(str(old_path), device="cpu")[old_key]
                    audit = compare_tensors(online, old, atol=validation["atol"], rtol=validation["rtol"])
                    meta["legacy_comparison"] = {**audit, "source": fingerprint(old_path)}
                    if config["cache"].get("reuse_legacy_text", False) and audit["passed"]:
                        candidate = old.float().to(online.device)
                        meta["reused_legacy"] = True
                        report["reused_legacy"] += 1
            if args.mode == "verify":
                _, prior_meta = checked_entry(config, record, component, envelope["sha256"])
                if sha256_file(path) != prior_meta["tensor_file_sha256"]:
                    raise ValueError(f"Cached tensor checksum mismatch: {path}")
                candidate = load_file(str(path), device="cpu")[name]
            reference = online.to(dtype=torch.bfloat16) if component == "vae" else online
            if component == "vae":
                meta["fp32_to_bf16_quantization"] = compare_tensors(online, reference, atol=0.0, rtol=0.0)
                meta["fp32_to_bf16_quantization"].pop("passed")
            comparison = compare_tensors(reference, candidate, atol=validation["atol"], rtol=validation["rtol"])
            comparison["reference_stage"] = "online_native_encoder_then_declared_storage_dtype"
            if not comparison["passed"]:
                raise RuntimeError(f"Cache differs from online encoder: {record['logical_key']}: {comparison}")
            meta["online_comparison"] = comparison
            if args.mode == "build":
                commit_entry(path, name, candidate, meta)
                readback = load_file(str(path), device="cpu")[name]
                if not torch.equal(readback, candidate.detach().cpu()):
                    path.with_suffix(".json").unlink(missing_ok=True)
                    raise RuntimeError("Serialization roundtrip mismatch; cache was invalidated")
            else:
                meta["tensor_file_sha256"] = sha256_file(path)
                atomic_json(path.with_suffix(".json"), meta)
            report["verified"] += 1
            report["updated"] = time.time()
            atomic_json(report_path, report)
            print(json.dumps({"component": component, "completed": report["verified"], "total": report["total"],
                              "logical_key": record["logical_key"], "max_abs": comparison["max_abs"]}), flush=True)
            del online, candidate
    report["finished"] = time.time()
    report["status"] = "complete"
    atomic_json(report_path, report)
    del encoder
    gc.collect()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=("inspect", "build", "verify"), required=True)
    parser.add_argument("--component", choices=("vae", "text"))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    args = parser.parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        parser.error("Invalid shard selection")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    config = json.loads(Path(args.config).read_text())
    validate_config(config)
    if args.mode == "inspect":
        result = inspect(config)
    else:
        if not args.component:
            parser.error("build/verify require --component (load VAE and text separately)")
        result = run(config, args)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
