#!/usr/bin/env python3
"""Build a compact completion inventory after the TF-1 GPU queue."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CODE_DIR = Path(__file__).resolve().parent
MATRIX_PATH = CODE_DIR / "tf1_matrix.json"
ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/training_free_m1_control_v1"
)
EXPERIMENT_ROOT = ROOT.parent
POSITIVE_ROOTS = (
    EXPERIMENT_ROOT / "training_free_top100_m1_guidance_v1",
    EXPERIMENT_ROOT / "training_free_top100_m23_guidance_v1",
)


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def find_positive(case: str, seed: int, value: float) -> str | None:
    for root in POSITIVE_ROOTS:
        seed_root = root / case / f"seed_{seed:05d}"
        if not seed_root.is_dir():
            continue
        for complete in seed_root.glob("*/complete.json"):
            try:
                row = load(complete)
            except (OSError, ValueError, TypeError):
                continue
            if (
                str(row.get("flow", "m1")) == "m1"
                and str(row.get("time_scope", "all_time")) == "all_time"
                and float(row.get("pag_scale", float("nan"))) == value
                and (complete.parent / "generated.mp4").is_file()
            ):
                return str(complete.parent)
    return None


def main() -> None:
    matrix = load(MATRIX_PATH)
    source = load(Path(matrix["source_manifest"]))
    source_lookup = {
        (str(row["case"]), int(row["seed"])): row for row in source["samples"]
    }
    records: list[dict[str, Any]] = []
    for case in matrix["cases"]:
        for seed in matrix["seeds"]:
            sample = source_lookup[(case, int(seed))]
            baseline = Path(str(sample["baseline_video"]))
            records.append(
                {
                    "case": case,
                    "seed": int(seed),
                    "family": "baseline",
                    "value": 0.0,
                    "path": str(baseline.parent),
                    "complete": baseline.is_file(),
                }
            )
            for alpha in matrix["soft_scaling"]["generated_alphas"]:
                tag = f"{float(alpha):g}".replace("-", "m").replace(".", "p")
                path = (
                    ROOT
                    / "soft_scaling"
                    / case
                    / f"seed_{int(seed):05d}"
                    / f"single_object__object_A__m1_all_time__top100__alpha_{tag}"
                )
                records.append(
                    {
                        "case": case,
                        "seed": int(seed),
                        "family": "soft_scaling",
                        "value": float(alpha),
                        "path": str(path),
                        "complete": (path / "complete.json").is_file()
                        and (path / "generated.mp4").is_file(),
                    }
                )
            for value in matrix["contrast_guidance"]["generated_lambdas"]:
                if float(value) > 0:
                    found = find_positive(case, int(seed), float(value))
                    records.append(
                        {
                            "case": case,
                            "seed": int(seed),
                            "family": "contrast_guidance_reused",
                            "value": float(value),
                            "path": found,
                            "complete": found is not None,
                        }
                    )
                else:
                    tag = f"{float(value):g}".replace("-", "m").replace(".", "p")
                    path = (
                        ROOT
                        / "contrast_raw"
                        / case
                        / f"seed_{int(seed):05d}"
                        / f"single_object__object_A__m1_all_time__top100__pag{tag}"
                    )
                    records.append(
                        {
                            "case": case,
                            "seed": int(seed),
                            "family": "contrast_guidance_new",
                            "value": float(value),
                            "path": str(path),
                            "complete": (path / "complete.json").is_file()
                            and (path / "generated.mp4").is_file(),
                        }
                    )
    counts: dict[str, dict[str, int]] = {}
    for row in records:
        family = str(row["family"])
        bucket = counts.setdefault(family, {"complete": 0, "total": 0})
        bucket["total"] += 1
        bucket["complete"] += int(bool(row["complete"]))
    report = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_id": matrix["experiment_id"],
        "counts": counts,
        "all_complete": all(bool(row["complete"]) for row in records),
        "records": records,
    }
    ROOT.mkdir(parents=True, exist_ok=True)
    (ROOT / "tf1_inventory.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report["counts"], ensure_ascii=False, indent=2))
    if not report["all_complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
