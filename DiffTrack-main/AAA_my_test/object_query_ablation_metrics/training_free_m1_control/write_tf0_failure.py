#!/usr/bin/env python3
"""Write a machine-readable TF-0 failure marker for shell-level failures."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/training_free_m1_control_v1"
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--exit-code", type=int, required=True)
    args = parser.parse_args()
    output = args.root / "tf0"
    output.mkdir(parents=True, exist_ok=True)
    (output / "PASS.json").unlink(missing_ok=True)
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "FAIL",
        "exit_code": int(args.exit_code),
        "reason": "TF-0 shell queue terminated before a PASS marker was written",
    }
    (output / "FAIL.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
