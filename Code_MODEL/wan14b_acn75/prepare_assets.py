"""Download pinned Pusa/Wan assets with disk and integrity checks; never load a model."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import time


ROOT = Path(__file__).resolve().parent


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(16 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/data/gaoya/agent-data/weights/wan14b_acn75")
    parser.add_argument("--manifest", type=Path, default=ROOT / "configs/assets.json")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--reserve-gib", type=float, default=60.0,
                        help="Space left for cache, checkpoints and concurrently running jobs")
    parser.add_argument("--reuse-t5", type=Path,
                        default=Path("/data/gaoya/ckpt/Wan-AI-Wan2.2-TI2V-5B/models_t5_umt5-xxl-enc-bf16.pth"))
    args = parser.parse_args()
    root = Path(args.root).resolve()
    if root.is_relative_to(Path("/home/gaoya")):
        raise SystemExit("Large model files must not be stored under /home/gaoya")
    manifest = json.loads(args.manifest.read_text())
    items = [(repo, item, root / repo["folder"] / item["path"])
             for repo in manifest["repositories"] for item in repo["files"]]
    missing = [(repo, item, dest) for repo, item, dest in items
               if not dest.is_file() or dest.stat().st_size != item["size"]]
    reuse = {}
    if args.reuse_t5.is_file():
        for repo, item, dest in missing:
            if item["path"] == args.reuse_t5.name and args.reuse_t5.stat().st_size == item["size"]:
                print("Checking existing UMT5 against the pinned 14B SHA256", flush=True)
                if sha256(args.reuse_t5) == item["sha256"]:
                    reuse[str(dest)] = args.reuse_t5.resolve()
                    print("Existing UMT5 checkpoint is byte-identical", flush=True)
    needed = sum(item["size"] for _, item, dest in missing if str(dest) not in reuse)
    ancestor = root
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    report = {"root": str(root), "download_bytes": needed, "free_bytes": free,
              "reserve_bytes": int(args.reserve_gib * 1024 ** 3),
              "reused_checkpoints": {k: str(v) for k, v in reuse.items()},
              "space_ok": free >= needed + args.reserve_gib * 1024 ** 3}
    print(json.dumps(report, indent=2), flush=True)
    if not args.download:
        return
    if not report["space_ok"]:
        raise SystemExit("Insufficient disk space; download was not started")
    from huggingface_hub import hf_hub_download
    root.mkdir(parents=True, exist_ok=True)
    audit = {"manifest_sha256": sha256(args.manifest), "files": [], "complete": False}
    for repo, item, dest in items:
        if str(dest) in reuse and not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.symlink_to(reuse[str(dest)])
        if not dest.is_file() or dest.stat().st_size != item["size"]:
            hf_hub_download(repo_id=repo["repo"], revision=repo["revision"],
                            filename=item["path"], local_dir=root / repo["folder"])
        if dest.stat().st_size != item["size"]:
            raise RuntimeError(f"Wrong file size: {dest}")
        actual_hash = sha256(dest)
        if item["sha256"] and actual_hash != item["sha256"]:
            raise RuntimeError(f"SHA256 mismatch: {dest}")
        audit["files"].append({"path": str(dest), "size": dest.stat().st_size,
                               "sha256": actual_hash, "revision": repo["revision"]})
        print(f"verified {repo['folder']}/{item['path']}", flush=True)
    audit.update(complete=True, completed_at=time.time())
    temporary = root / "assets_verified.json.tmp"
    temporary.write_text(json.dumps(audit, indent=2) + "\n")
    os.replace(temporary, root / "assets_verified.json")


if __name__ == "__main__":
    main()
