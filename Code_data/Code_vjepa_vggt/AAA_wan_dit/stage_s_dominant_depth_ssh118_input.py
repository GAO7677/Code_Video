#!/usr/bin/env python3
"""Stage the 20-case input list with paths rewritten for SSH host 118."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--remote-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_list = args.input_list.expanduser().resolve()
    output_root = args.output_root.expanduser().resolve()
    remote_root = args.remote_root
    files_root = output_root / "files"
    json_root = output_root / "jsons"
    json_root.mkdir(parents=True, exist_ok=True)
    copied: set[Path] = set()

    def rewrite(value: Any) -> Any:
        if isinstance(value, dict):
            return {key: rewrite(child) for key, child in value.items()}
        if isinstance(value, list):
            return [rewrite(child) for child in value]
        if not isinstance(value, str) or not value.startswith("/"):
            return value
        source = Path(value)
        if source.is_file():
            destination = files_root / source.as_posix().lstrip("/")
            if source not in copied:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)
                copied.add(source)
            return str(remote_root / "files" / source.as_posix().lstrip("/"))
        return value

    remote_jsons = []
    for line in input_list.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        source_json = Path(line.strip()).expanduser().resolve()
        payload = rewrite(json.loads(source_json.read_text(encoding="utf-8")))
        destination = json_root / source_json.name
        destination.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        remote_jsons.append(str(remote_root / "jsons" / source_json.name))

    (output_root / "test5_unique20.txt").write_text(
        "\n".join(remote_jsons) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "cases": len(remote_jsons),
                "copied_files": len(copied),
                "copied_bytes": sum(path.stat().st_size for path in copied),
                "output_root": str(output_root),
                "remote_root": str(remote_root),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
