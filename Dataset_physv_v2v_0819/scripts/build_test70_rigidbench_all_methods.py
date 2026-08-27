#!/usr/bin/env python3
"""Build the standalone all-methods RigidBench test70 visualization data."""

from __future__ import annotations

import json
import fcntl
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np


OUTPUT_ROOT = Path("/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_all_methods")
PAGE_ROOT = Path(
    "/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/"
    "physv-v2v-0819-test70-rigidbench-all-methods"
)
METRICS = ("iou", "l2", "chamfer", "ate", "si_mse", "lpips", "ssim", "ate3d", "iddrift", "bgdrift")
MAIN_PAGE_REL = "../physv-v2v-0819-test70-no-event-timing-40step/"
METRIC_LABELS = {
    "iou": "IoU ↑",
    "l2": "L2 ↓",
    "chamfer": "Chamfer ↓",
    "ate": "ATE ↓",
    "si_mse": "SI-MSE ↓",
    "lpips": "LPIPS ↓",
    "ssim": "SSIM ↑",
    "ate3d": "ATE-3D ↓",
    "iddrift": "IdDrift ↓",
    "bgdrift": "BG-Drift ↓",
}
DIRECTIONS = {"iou": "higher", "ssim": "higher"}


def read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else {}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=True) + "\n", encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def finite(value: Any) -> bool:
    try:
        return bool(np.isfinite(float(value)))
    except (TypeError, ValueError):
        return False


def build_snapshot() -> int:
    registry = read_json(OUTPUT_ROOT / "registry.json")
    models_out = []
    total_metric_cells = 0
    completed_metric_cells = 0
    for model in registry.get("models", []):
        case_rows = []
        aggregate_values: dict[str, list[float]] = {metric: [] for metric in METRICS}
        metric_counts = {metric: 0 for metric in METRICS}
        complete_cases = 0
        generated_case_count = 0
        for case in model.get("cases", []):
            payload = read_json(OUTPUT_ROOT / "methods" / model["task_id"] / "metrics" / f"{case['case_id']}.json")
            values = {}
            for metric in METRICS:
                value = payload.get(metric)
                if finite(value):
                    value = float(value)
                    values[metric] = value
                    aggregate_values[metric].append(value)
                    metric_counts[metric] += 1
            if all(metric in values for metric in METRICS):
                complete_cases += 1
            # The registry is initialized before inference/metric workers run,
            # so its cached existence bit can be stale.  The snapshot must
            # reflect the filesystem at build time.
            if Path(case["video_path"]).is_file():
                generated_case_count += 1
            case_rows.append(
                {
                    "case_id": case["case_id"],
                    "family_key": case.get("family_key"),
                    "video_url": MAIN_PAGE_REL + case["video_url"],
                    "prediction_exists": bool(case.get("prediction_exists", False)),
                    "metrics": values,
                }
            )
        total_metric_cells += len(case_rows) * len(METRICS)
        completed_metric_cells += sum(metric_counts.values())
        if complete_cases == len(case_rows) and case_rows:
            status = "complete"
        elif any(metric_counts.values()) or generated_case_count:
            status = "partial"
        else:
            status = "pending"
        models_out.append(
            {
                "task_id": model["task_id"],
                "model_key": model.get("model_key"),
                "label": model.get("label", model["task_id"]),
                "color": model.get("color"),
                "step": model.get("step"),
                "checkpoint_format": model.get("checkpoint_format"),
                "source_checkpoint": model.get("source_checkpoint"),
                "inference_steps": model.get("inference_steps"),
                "status": status,
                "generated_case_count": generated_case_count,
                "case_count": len(case_rows),
                "complete_case_count": complete_cases,
                "metric_counts": metric_counts,
                "metrics": {
                    metric: (float(np.mean(values)) if values else None)
                    for metric, values in aggregate_values.items()
                },
                "cases": case_rows,
            }
        )
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "title": "RigidBench 风格指标 · PHYRVG test70 全方法",
        "source_dashboard": registry.get("source_dashboard"),
        "strict_root": registry.get("strict_root"),
        "protocol": registry.get("protocol"),
        "official": False,
        "protocol_note": "本页面复用 test70 已生成的 49 帧 / 30 FPS / 896×512 视频，使用 strict CYCLES GT；不是官方 RigidBench 完整协议分数。",
        "fps": registry.get("fps", 30),
        "resolution": registry.get("resolution", [896, 512]),
        "window_frames": registry.get("window_frames", 49),
        "metrics": list(METRICS),
        "metric_labels": METRIC_LABELS,
        "directions": DIRECTIONS,
        "trackers": ["SAM2.1 Hiera-Large", "CoTracker3 offline", "Video Depth Anything Large", "DINOv2 ViT-L/14", "LPIPS Alex"],
        "model_count": len(models_out),
        "case_count": len(registry.get("case_ids", [])),
        "total_metric_cells": total_metric_cells,
        "completed_metric_cells": completed_metric_cells,
        "models": models_out,
        "case_ids": registry.get("case_ids", []),
    }
    atomic_json(PAGE_ROOT / "data.json", payload)
    print(json.dumps({"output": str(PAGE_ROOT / "data.json"), "models": len(models_out), "cases": len(payload["case_ids"]), "completed_metric_cells": completed_metric_cells, "total_metric_cells": total_metric_cells}, ensure_ascii=False))
    return 0


def main() -> int:
    """Build one snapshot while serializing refresh/background builders.

    Both the HTTP refresh endpoint and the long-running RigidBench launcher
    may rebuild this file.  Serializing the complete read/aggregate/replace
    cycle prevents an older, slower snapshot from replacing a newer one.
    """
    lock_path = OUTPUT_ROOT / ".builder.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        return build_snapshot()


if __name__ == "__main__":
    raise SystemExit(main())
