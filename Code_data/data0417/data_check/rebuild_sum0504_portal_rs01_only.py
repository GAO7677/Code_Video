#!/usr/bin/env python3
from __future__ import annotations

import argparse
import types
from pathlib import Path


SOURCE = Path("/home/gaoya/Code_Video/Code_data/data0417/data_check/rebuild_sum0504_portal_with_sample_pages.py")
TARGET_ROOT = Path("/home/gaoya/portal_hub_sim/sum0504_portal_rs01_only")


def load_module() -> types.ModuleType:
    module = types.ModuleType("sum0504_rs01_only_builder")
    module.__file__ = str(SOURCE)
    code = SOURCE.read_text(encoding="utf-8")
    exec(compile(code, str(SOURCE), "exec"), module.__dict__)
    module.ROOT = TARGET_ROOT
    module.MANIFEST_PATH = TARGET_ROOT / "manifest.json"
    return module


def main() -> None:
    parser = argparse.ArgumentParser(description="Rebuild an rs01-only sum0504 portal.")
    parser.add_argument("--sample_substring", type=str, default="__rs01")
    args = parser.parse_args()

    module = load_module()
    groups = module.load_groups_from_sum0504(sample_substring=args.sample_substring)
    groups = module.build_group_cards(groups)
    TARGET_ROOT.mkdir(parents=True, exist_ok=True)
    (TARGET_ROOT / "index.html").write_text(module.build_index(groups), encoding="utf-8")
    (TARGET_ROOT / "manifest.json").write_text(module.json.dumps(groups, ensure_ascii=False, indent=2), encoding="utf-8")
    print(TARGET_ROOT / "index.html")


if __name__ == "__main__":
    main()
