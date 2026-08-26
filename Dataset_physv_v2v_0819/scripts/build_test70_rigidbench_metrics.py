#!/usr/bin/env python3
"""Build the RigidBench-style strict-CYCLES metric snapshot for the test70 page."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path


DASHBOARD = Path(
    "/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/"
    "physv-v2v-0819-test70-no-event-timing-40step/dashboard.json"
)
RUNS = Path("/data/gaoya/agent-data/outputs/physv_v2v_0819_rigidbench_strict_test70/runs")
DEST = Path(
    "/data/gaoya/agent-data/physv_v2v_0819/visualization/hub/"
    "physv-v2v-0819-test70-no-event-timing-40step/metrics/rigidbench_strict_metrics.json"
)
METRICS = ("iou", "l2", "chamfer", "ate", "si_mse", "lpips", "ssim", "ate3d", "iddrift", "bgdrift")


def read_json(path: Path) -> dict | None:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def main() -> int:
    dashboard = read_json(DASHBOARD)
    if not dashboard:
        raise FileNotFoundError(DASHBOARD)
    rows = []
    for model in dashboard.get("models", []):
        task_id = str(model.get("task_id", ""))
        report = read_json(RUNS / task_id / "strict_cycles_test70.json") if task_id else None
        aggregate = (report or {}).get("aggregated", {})
        values = {key: aggregate.get(key) for key in METRICS if key in aggregate}
        evaluated = int((report or {}).get("evaluated_case_count", 0))
        missing = int((report or {}).get("missing_case_count", dashboard.get("case_count", 70)))
        rows.append({
            "task_id": task_id,
            "model_key": model.get("model_key"),
            "label": model.get("label", task_id),
            "color": model.get("color"),
            "step": model.get("step"),
            "source_checkpoint": model.get("source_checkpoint"),
            "status": "complete" if report and evaluated == int(dashboard.get("case_count", 70)) else ("partial" if report else "pending"),
            "evaluated_case_count": evaluated,
            "missing_case_count": missing,
            "metric_case_count": int(aggregate.get("n_samples", 0)) if aggregate else 0,
            "metrics": values,
            "strict_reference_exact_count": len((report or {}).get("strict_reference_exact_cases", [])),
            "strict_reference_rerendered_count": len((report or {}).get("strict_reference_rerendered_cases", [])),
            "report": str(RUNS / task_id / "strict_cycles_test70.json") if report else None,
        })
    payload = {
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "source_dashboard": str(DASHBOARD),
        "protocol": "rigidbench-style-local-test70-strict-cycles",
        "official": False,
        "protocol_note": "本表不是官方 RigidBench 完整分数：复用 test70 已生成的 49 帧/30 FPS 视频（约 1.6 s），使用 strict CYCLES 896×512/30 FPS GT；官方协议为 2.0 s、24 FPS。",
        "gt_root": "/data/gaoya/AAA_test_video/physv_v2v_0819_strict",
        "window_frames": 49,
        "window_seconds": 49 / 30,
        "fps": 30,
        "resolution": [896, 512],
        "metrics": list(METRICS),
        "metric_labels": {
            "iou": "IoU ↑", "l2": "L2 ↓", "chamfer": "Chamfer ↓", "ate": "ATE ↓",
            "si_mse": "SI-MSE ↓", "lpips": "LPIPS ↓", "ssim": "SSIM ↑", "ate3d": "ATE-3D ↓",
            "iddrift": "IdDrift ↓", "bgdrift": "BGDrift ↓",
        },
        "trackers": ["SAM2.1 Hiera-Large", "CoTracker3 offline", "Video Depth Anything Large"],
        "models": rows,
    }
    atomic_write(DEST, payload)
    complete = sum(row["status"] == "complete" for row in rows)
    partial = sum(row["status"] == "partial" for row in rows)
    print(json.dumps({"output": str(DEST), "models": len(rows), "complete": complete, "partial": partial}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
