#!/usr/bin/env python3
"""Validate path-list files under data_summary/by_raw_window.

该脚本用于检查 by_raw_window 目录下各类样本清单是否有效；
输入为 summary 根目录，输出为每个列表文件的条目数、缺失路径数，以及
json/txt 同名文件是否一致。发现问题时返回非零退出码。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_SUMMARY_ROOT = Path("/home/gaoya/Code_Video/Code_data/data0417/data_summary/by_raw_window")
SKIP_FILENAMES = {"README.md", "summary.json"}


def load_entries(path: Path) -> list[str]:
    if path.suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError(f"{path} is not a JSON list")
        return [str(item).strip() for item in data if str(item).strip()]
    if path.suffix == ".txt":
        return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    raise ValueError(f"Unsupported file type: {path}")


def iter_list_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.name in SKIP_FILENAMES:
            continue
        if path.suffix not in {".json", ".txt"}:
            continue
        yield path


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate data_summary/by_raw_window path-list files")
    parser.add_argument("--root", type=Path, default=DEFAULT_SUMMARY_ROOT, help="by_raw_window root to scan")
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.exists():
        print(f"[error] root does not exist: {root}")
        return 2

    problems: list[str] = []
    checked_files = 0
    total_entries = 0

    for path in iter_list_files(root):
        checked_files += 1
        try:
            entries = load_entries(path)
        except Exception as exc:
            problems.append(f"{path}: failed to parse ({type(exc).__name__}: {exc})")
            continue

        total_entries += len(entries)
        missing = [entry for entry in entries if not Path(entry).exists()]
        print(f"[check] {path} entries={len(entries)} missing={len(missing)}")
        if missing:
            preview = ", ".join(missing[:5])
            problems.append(f"{path}: missing {len(missing)} path(s); examples: {preview}")

        if path.suffix == ".json":
            txt_path = path.with_suffix(".txt")
            if txt_path.exists():
                try:
                    txt_entries = load_entries(txt_path)
                except Exception as exc:
                    problems.append(f"{txt_path}: failed to parse ({type(exc).__name__}: {exc})")
                    continue
                if entries != txt_entries:
                    problems.append(f"{path}: content mismatch with {txt_path}")
            else:
                problems.append(f"{path}: missing sibling txt file {txt_path}")

    print(f"[summary] checked_files={checked_files} total_entries={total_entries} problems={len(problems)}")
    if problems:
        for problem in problems:
            print(f"[problem] {problem}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
