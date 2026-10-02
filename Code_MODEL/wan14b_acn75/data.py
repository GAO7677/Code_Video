"""Strict, CPU-side Generic2175 records and audited native Wan14B cache reader.

Only generated, online-verified caches with this contract are consumed.  Legacy
48-channel Wan5B VAE caches are intentionally not a fallback.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def stable_sha(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False,
                                     separators=(",", ":")).encode()).hexdigest()


def file_stamp(path: str | Path) -> dict:
    path = Path(path).expanduser().resolve()
    st = path.stat()
    return {"path": str(path), "size": st.st_size, "mtime_ns": st.st_mtime_ns}


def fingerprint(path: str | Path) -> dict:
    item = file_stamp(path)
    item["sha256"] = sha256_file(path)
    return item


def atomic_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temp.open("w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(temp, path)


def validate_config(config: dict) -> None:
    d, c = config["data"], config["cache"]
    if d.get("caption_field") != "caption_initial0907":
        raise ValueError("This experiment requires caption_initial0907")
    if (d["num_frames"], d["height"], d["width"]) != (49, 512, 896):
        raise ValueError("Expected the agreed 49-frame 512x896 protocol")
    if (d["context_frames"], d["context_latent_frames"]) != (8, 2):
        raise ValueError("Expected 8 supplied RGB frames and 2 causal context latents")
    if d.get("resize_mode") != "stretch":
        raise ValueError("Training cache requires explicit stretch resize")
    if c["latent_dtype"] != "bfloat16" or c["text_dtype"] != "float32":
        raise ValueError("Expected BF16 latents and FP32 text; changes require a new audited contract")
    root = Path(c["root"]).expanduser().resolve()
    if root.is_relative_to(Path("/home")):
        raise ValueError("Large caches cannot be stored under /home; use the configured data volume")
    for key in ("atol", "rtol", "context_atol", "context_rtol"):
        if key not in c["validation"] or c["validation"][key] < 0:
            raise ValueError(f"Explicit nonnegative cache.validation.{key} is required")


def load_records(config: dict) -> list[dict]:
    validate_config(config)
    d = config["data"]
    selection = Path(d["sample_list"]).expanduser().resolve()
    seen, records = set(), []
    legacy = {}
    legacy_root = config["cache"].get("legacy_latent_root")
    if legacy_root:
        legacy_root = Path(legacy_root)
        legacy_cfg = json.loads((legacy_root / "cache_config.json").read_text())
        with (legacy_root / legacy_cfg.get("index_file", "index.jsonl")).open() as f:
            for line in f:
                item = json.loads(line)
                legacy[item["logical_key"]] = item
    for raw in selection.read_text().splitlines():
        raw = raw.strip()
        if not raw or raw.startswith("#"):
            continue
        sidecar = Path(raw).expanduser()
        if not sidecar.is_absolute():
            sidecar = selection.parent / sidecar
        sidecar = sidecar.resolve()
        item = json.loads(sidecar.read_text())
        key = item["logical_key"]
        if key in seen:
            raise ValueError(f"Duplicate logical key: {key}")
        seen.add(key)
        source = item["source"]
        def resolve_source(name):
            value = source.get(name + "_resolved") or source[name]
            path = Path(value).expanduser()
            return (sidecar.parent / path if not path.is_absolute() else path).resolve()
        video = resolve_source("video")
        metadata = resolve_source("metadata_json")
        caption = json.loads(metadata.read_text())
        for part in d["caption_field"].split("."):
            caption = caption[part]
        if not isinstance(caption, str) or not caption.strip():
            raise ValueError(f"Missing caption: {metadata}")
        caption = caption.strip()
        records.append({"logical_key": key, "sidecar": str(sidecar),
                        "video": str(video), "metadata": str(metadata),
                        "caption": caption,
                        "caption_sha256": hashlib.sha256(caption.encode()).hexdigest(),
                        "legacy_video_provenance": legacy.get(key)})
        if legacy_root and key not in legacy:
            raise ValueError(f"Missing original training frame provenance: {key}")
    if len(records) != d["expected_samples"]:
        raise ValueError(f"Expected {d['expected_samples']} records, found {len(records)}")
    return records


def read_rgb_prefix(record: dict, config: dict):
    """Decode prefix with the existing cache's 60fps stride-2 policy.

    Resize matches torch bilinear(align_corners=False), performed on float32
    RGB [0,255], followed by /255*2-1, as in the prior training cache.
    """
    import numpy as np
    import torch
    import torch.nn.functional as F
    from decord import VideoReader, cpu
    d = config["data"]
    vr = VideoReader(record["video"], ctx=cpu(0), num_threads=1)
    fps = float(vr.get_avg_fps())
    is60 = abs(fps - 60.0) <= float(d.get("fps60_tolerance", 0.1))
    stride = int(d["fps60_sample_step"] if is60 else d["fps_other_sample_step"])
    if stride < 1:
        raise ValueError("Sampling stride must be positive")
    indices = np.arange(d["num_frames"], dtype=np.int64) * stride
    if len(vr) <= indices[-1]:
        raise ValueError(f"Not enough frames: {record['video']}; need index {indices[-1]}")
    old = record.get("legacy_video_provenance")
    if old and old.get("sampled_frame_indices") != indices.tolist():
        raise ValueError(f"Frame sampling differs from Wan5B training cache: {record['logical_key']}")
    rgb = vr.get_batch(indices).asnumpy()
    x = torch.from_numpy(rgb).permute(0, 3, 1, 2).float()
    x = F.interpolate(x, size=(d["height"], d["width"]), mode="bilinear", align_corners=False)
    x = ((x / 255.0) * 2.0 - 1.0).permute(1, 0, 2, 3).contiguous()
    return x, {"source_fps": fps, "frame_indices": indices.tolist(),
               "source_hw": list(rgb.shape[1:3]), "preprocess": "torch_bilinear_align_corners_false_stretch"}


def cache_path(root: str | Path, component: str, key: str) -> Path:
    digest = hashlib.sha256(key.encode()).hexdigest()
    return Path(root) / component / digest[:2] / (digest + ".safetensors")


def component_files(config: dict, component: str) -> list[Path]:
    part = config["cache"][component]
    paths = [Path(part["checkpoint"]).expanduser().resolve(),
             Path(part.get("source_file") or Path(__file__).parent / "vendor" /
                  ("wan_video_vae.py" if component == "vae" else "wan_video_text_encoder.py")).resolve(),
             Path(__file__).resolve(), Path(__file__).with_name("audit_cache.py").resolve()]
    if component == "vae" and config["cache"].get("legacy_latent_root"):
        legacy = Path(config["cache"]["legacy_latent_root"])
        old_config = legacy / "cache_config.json"
        paths.extend([old_config, legacy / json.loads(old_config.read_text()).get("index_file", "index.jsonl")])
    if component == "text":
        tokenizer = Path(part["tokenizer"]).expanduser().resolve()
        paths.extend(sorted(p for p in tokenizer.iterdir() if p.is_file()
                            and p.suffix in {".json", ".model", ".txt"}))
        if len(paths) == 4:
            raise ValueError(f"No tokenizer files: {tokenizer}")
    return paths


def make_cache_contract(config: dict, component: str) -> dict:
    """Hash real component files once during build, never instantiate models."""
    validate_config(config)
    d = config["data"]
    settings = {k: d[k] for k in ("caption_field", "height", "width", "num_frames",
                "context_frames", "context_latent_frames", "resize_mode",
                "fps60_sample_step", "fps_other_sample_step")}
    settings["fps60_tolerance"] = d.get("fps60_tolerance", 0.1)
    return {"schema_version": SCHEMA_VERSION, "component": component,
            "settings": settings, "validation": config["cache"]["validation"],
            "encoder": config["cache"][component],
            "storage_dtype": config["cache"]["latent_dtype" if component == "vae" else "text_dtype"],
            "files": [fingerprint(p) for p in component_files(config, component)],
            "normalization": "native_wan16_mean_inverse_std" if component == "vae" else
                             "official_pusa_whitespace_clean_umt5_512_zero_padding",
            "posterior": "mean_deterministic" if component == "vae" else None}


def load_verified_contract(config: dict, component: str) -> dict:
    path = Path(config["cache"]["root"]) / "contracts" / (component + ".json")
    envelope = json.loads(path.read_text())
    contract = envelope["contract"]
    if stable_sha(contract) != envelope["sha256"]:
        raise ValueError(f"Contract hash mismatch: {path}")
    if contract["encoder"] != config["cache"][component]:
        raise ValueError(f"Encoder settings changed: {component}")
    if contract["validation"] != config["cache"]["validation"]:
        raise ValueError("Validation threshold change requires a new cache audit")
    for key, value in contract["settings"].items():
        if config["data"].get(key, 0.1 if key == "fps60_tolerance" else None) != value:
            raise ValueError(f"Preprocessing changed: {key}")
    for file in contract["files"]:
        if file_stamp(file["path"]) != {k: file[k] for k in ("path", "size", "mtime_ns")}:
            raise ValueError(f"Component file changed; rebuild audit: {file['path']}")
    return envelope


def checked_entry(config: dict, record: dict, component: str, contract_sha: str):
    key = record["logical_key"] if component == "vae" else record["caption_sha256"]
    path = cache_path(config["cache"]["root"], component, key)
    meta = json.loads(path.with_suffix(".json").read_text())
    if (meta.get("contract_sha256") != contract_sha or meta.get("status") != "verified"
            or not meta.get("online_comparison", {}).get("passed")):
        raise ValueError(f"Unverified/incompatible cache: {path}")
    if component == "vae":
        if meta["logical_key"] != record["logical_key"]:
            raise ValueError(f"Cache identity mismatch: {path}")
        source = meta["source"]
        if file_stamp(record["video"]) != {k: source[k] for k in ("path", "size", "mtime_ns")}:
            raise ValueError(f"Source video changed: {record['video']}")
        if not meta.get("context_causality", {}).get("passed"):
            raise ValueError(f"Missing causal context verification: {path}")
    elif meta.get("caption_sha256") != record["caption_sha256"]:
        raise ValueError(f"Caption mismatch: {path}")
    return path, meta


class Wan14BCachedDataset:
    """Map-style dataset; compatible with torch's default dict collate."""
    def __init__(self, config: dict, require_verified: bool = True):
        if not require_verified:
            raise ValueError("Unverified caches cannot be used by the training dataset")
        self.config = config
        self.records = load_records(config)
        self.contracts = {part: load_verified_contract(config, part) for part in ("vae", "text")}
        self.entries = []
        for record in self.records:
            self.entries.append({part: checked_entry(config, record, part,
                                  self.contracts[part]["sha256"]) for part in ("vae", "text")})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        from safetensors.torch import load_file
        import torch
        tensors = {}
        for part, name in (("vae", "latents"), ("text", "text")):
            path, meta = self.entries[index][part]
            if sha256_file(path) != meta["tensor_file_sha256"]:
                raise ValueError(f"Cache tensor content changed: {path}")
            tensors[name] = load_file(str(path), device="cpu")[name]
        expected = (16, 13, 64, 112)
        if tuple(tensors["latents"].shape) != expected or tensors["latents"].dtype != torch.bfloat16:
            raise ValueError(f"Expected BF16 latent {expected}")
        if tuple(tensors["text"].shape) != (512, 4096) or tensors["text"].dtype != torch.float32:
            raise ValueError("Expected FP32 text [512,4096]")
        record = self.records[index]
        return {**tensors, "logical_key": record["logical_key"], "caption": record["caption"]}
