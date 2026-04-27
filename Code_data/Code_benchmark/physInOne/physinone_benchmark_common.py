#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List


MY_BENCH_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_benchmark/my_bench")
WAN_TRAIN_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_train/train_0419")

if str(MY_BENCH_ROOT) not in sys.path:
    sys.path.insert(0, str(MY_BENCH_ROOT))
if str(WAN_TRAIN_ROOT) not in sys.path:
    sys.path.insert(0, str(WAN_TRAIN_ROOT))

from benchlib.config import load_config  # noqa: E402
from benchlib.continuation import run_continuation_metrics  # noqa: E402
from benchlib.manifest import load_manifest  # noqa: E402
from benchlib.vbench_wrappers import run_vbench_i2v, run_vbench_short  # noqa: E402


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_jsonl(path: Path) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def humanize_physics_type(name: str) -> str:
    text = re.sub(r"(?<!^)(?=[A-Z])", " ", str(name or "")).strip()
    return re.sub(r"\s+", " ", text).lower()


def join_physics_types(physics_types: List[str]) -> str:
    parts = [humanize_physics_type(item) for item in physics_types]
    if not parts:
        return "physical motion"
    if len(parts) == 1:
        return parts[0]
    if len(parts) == 2:
        return f"{parts[0]} and {parts[1]}"
    return ", ".join(parts[:-1]) + f", and {parts[-1]}"


def build_prompt(group_name: str, physics_types: List[str], camera_name: str) -> str:
    motion_text = join_physics_types(physics_types)
    camera_label = str(camera_name or "").replace("_", " ")
    return (
        f"A realistic physics video from {camera_label} showing {motion_text}. "
        f"The motion should stay physically plausible and temporally smooth."
    )


def find_tokenizer_path(wan_root: Path) -> Path:
    candidates = [
        wan_root / "google" / "umt5-xxl",
        wan_root / "google",
    ]
    for path in candidates:
        if path.is_dir():
            return path
    raise FileNotFoundError(f"Tokenizer directory not found under {wan_root}")


def parse_vbench_eval(path: Path) -> Dict[str, float]:
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics: Dict[str, float] = {}
    for key, value in data.items():
        if isinstance(value, list) and value:
            metrics[key] = float(value[0])
    return metrics


def run_all_benchmarks(
    *,
    bench_config_path: Path,
    benchmark_manifest_path: Path,
    output_root: Path,
    run_prefix: str,
) -> Dict[str, Any]:
    config = load_config(str(bench_config_path))
    samples = load_manifest(str(benchmark_manifest_path))

    short_dir = output_root / "vbench_short"
    i2v_dir = output_root / "vbench_i2v"
    continuation_dir = output_root / "continuation"

    short_path = Path(
        run_vbench_short(
            config=config,
            samples=samples,
            output_dir=str(short_dir),
            run_name=f"{run_prefix}_short",
        )
    )
    i2v_path = Path(
        run_vbench_i2v(
            config=config,
            samples=samples,
            output_dir=str(i2v_dir),
            run_name=f"{run_prefix}_i2v",
            resolution="1-1",
        )
    )
    continuation_path = Path(
        run_continuation_metrics(
            config=config,
            samples=samples,
            output_dir=str(continuation_dir),
            run_name=f"{run_prefix}_continuation",
        )
    )

    continuation_payload = json.loads(continuation_path.read_text(encoding="utf-8"))
    summary = {
        "short_metrics": parse_vbench_eval(short_path),
        "i2v_metrics": parse_vbench_eval(i2v_path),
        "continuation_metrics": continuation_payload.get("aggregate", {}),
        "artifacts": {
            "short_eval_json": str(short_path),
            "i2v_eval_json": str(i2v_path),
            "continuation_json": str(continuation_path),
        },
    }
    write_json(output_root / "summary.json", summary)
    return summary
