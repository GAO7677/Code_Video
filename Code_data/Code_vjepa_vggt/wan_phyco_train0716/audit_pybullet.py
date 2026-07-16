from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    files = sorted(args.root.glob("*/*/meta.json"))
    report = Counter()
    families = Counter()
    for path in files:
        metadata = json.loads(path.read_text(encoding="utf-8"))
        families[str(metadata.get("family_slug", path.parent.parent.name))] += 1
        for item in metadata.get("objects", []):
            report["objects"] += 1
            for key in ("friction", "restitution", "mass"):
                report[f"objects_with_{key}"] += int(item.get(key) is not None)
            velocity = item.get("linear_velocity", [0.0, 0.0, 0.0])
            report["objects_with_nonzero_linear_velocity"] += int(
                math.sqrt(sum(float(value) ** 2 for value in velocity)) > 1.0e-8
            )
            report["objects_with_deformation"] += int(
                any("deform" in str(key).lower() or "elastic" in str(key).lower() for key in item)
            )
            report["objects_with_force"] += int(
                any("force" in str(key).lower() or "impulse" in str(key).lower() for key in item)
            )
    payload = {
        "samples": len(files),
        "families": dict(families),
        "coverage": dict(report),
        "conclusion": {
            "rigid_branch": "direct friction/restitution supervision",
            "deformation_branch": "unsupported; null control",
            "force_motion_branch": "initial velocity direction proxy only",
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

