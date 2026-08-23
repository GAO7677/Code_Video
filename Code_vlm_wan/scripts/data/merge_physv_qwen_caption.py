#!/usr/bin/env python3
"""Write generated captions into PhysV V2V case JSON files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--case-dir", type=Path, required=True)
    args = parser.parse_args()

    manifest = json.loads((args.dataset / "manifest.json").read_text(encoding="utf-8"))
    samples = {s["sample_id"]: s for s in manifest["samples"]}
    rows = [json.loads(line) for line in args.results.read_text(encoding="utf-8").splitlines() if line.strip()]
    errors = [r for r in rows if r.get("status") != "ok" or not r.get("caption")]
    if errors:
        raise RuntimeError(f"Refusing partial merge; failed rows: {[r.get('case_id') for r in errors]}")
    missing = sorted(set(samples) - {r["case_id"] for r in rows})
    if missing:
        raise RuntimeError(f"Refusing partial merge; missing samples: {missing}")

    args.case_dir.mkdir(parents=True, exist_ok=True)
    created = updated = 0
    for row in rows:
        sample_id = row["case_id"]
        sample = samples[sample_id]
        path = args.case_dir / f"{sample_id}.json"
        if path.is_file():
            payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
            updated += 1
        else:
            sample_dir = Path(sample["sample_dir"])
            metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
            payload = {
                "source_video": str(sample_dir / "videos/rgb.mp4"),
                "input_caption": metadata.get("caption", metadata.get("title", sample_id)),
                "sample_id": sample_id,
                "dataset": manifest.get("dataset"),
                "schema_version": manifest.get("schema_version"),
                "task_type": sample.get("task_type"),
                "source_group": sample.get("source_group"),
                "family_key": sample.get("family_key"),
                "control": {
                    "variable": sample.get("controlled_variable"),
                    "value": sample.get("controlled_value"),
                },
            }
            created += 1
        payload["qwen_caption"] = row["caption"]
        payload["qwen_caption_prompt"] = row.get("prompt", "Describe the physical phenomena in the video.")
        payload["qwen_caption_model"] = row.get("model")
        payload["qwen_caption_video"] = row.get("video")
        payload["qwen_caption_status"] = row.get("status")
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"rows={len(rows)} updated={updated} created={created}")


if __name__ == "__main__":
    main()
