#!/usr/bin/env python3
"""Build S/T/P/C/G top-1 QK capture selection from classification output."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--classification", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.classification.read_text(encoding="utf-8"))
    samples: dict[str, dict[str, dict]] = {}
    for sample in report["samples"]:
        roles = {}
        for role in ("S", "T", "P", "C", "G"):
            top = sample["top_heads"][role][0]
            roles[role] = {
                "block": int(top["block"]),
                "head": int(top["head"]),
            }
        samples.setdefault(sample["model"], {})[sample["case"]] = {
            "roles": roles
        }
    payload = {
        "source": str(args.classification.expanduser().resolve()),
        "policy": "top-1 stable candidate for each S/T/P/C/G role",
        "samples": samples,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
