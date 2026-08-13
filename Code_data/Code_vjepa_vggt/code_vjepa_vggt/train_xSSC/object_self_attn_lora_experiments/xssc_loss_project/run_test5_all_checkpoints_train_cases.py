#!/usr/bin/env python3
"""Run the nine fixed training cases for every available method checkpoint."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import time


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "test5_step500_all_methods_train_cases.json"
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
RUN_INFER = ROOT.parent / "run_infer_from_experiment.sh"
BUILD_PAGE = ROOT / "build_test5_all_checkpoints_train_case_gallery.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=None)
    parser.add_argument("--inventory-only", action="store_true")
    return parser.parse_args()


def timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp.{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def result_name(step: int) -> str:
    return f"step-{step:06d}_steps40_512x896_ctx08_49f"


def case_ids(config: dict) -> list[str]:
    manifest = json.loads(
        Path(config["cases_manifest"]).read_text(encoding="utf-8")
    )
    result = [str(case["case_id"]) for case in manifest["cases"]]
    if len(result) != 9 or len(result) != len(set(result)):
        raise ValueError(f"Expected nine unique cases, found {len(result)}")
    return result


def completed_cases(root: Path, cases: list[str]) -> list[str]:
    return [
        case_id
        for case_id in cases
        if (root / f"{case_id}.mp4").is_file()
        and (root / f"{case_id}.mp4").stat().st_size > 0
        and (root / f"{case_id}.json").is_file()
        and (root / f"{case_id}.json").stat().st_size > 0
    ]


def checkpoint_steps(method: dict) -> list[tuple[int, Path]]:
    checkpoint_parent = Path(method["checkpoint"]).expanduser().resolve().parent
    records: list[tuple[int, Path]] = []
    for path in checkpoint_parent.glob("step-*"):
        match = re.fullmatch(r"step-(\d+)", path.name)
        if match and (path / "checkpoint.safetensors").is_file():
            records.append((int(match.group(1)), path.resolve()))
    return sorted(records)


def result_root(output_root: Path, method: dict, step: int) -> Path:
    reuse = method.get("reuse_result_root")
    if reuse:
        reuse_candidate = Path(reuse).expanduser().resolve().parent / result_name(step)
        if reuse_candidate.is_dir():
            return reuse_candidate
    return output_root / "inference" / method["key"] / result_name(step)


def discover_inventory(config: dict) -> dict:
    output_root = Path(config["output_root"]).expanduser().resolve()
    records = []
    for method in config["methods"]:
        steps = checkpoint_steps(method)
        if not steps:
            raise FileNotFoundError(
                f"No checkpoints found beside {method['checkpoint']}"
            )
        for step, checkpoint in steps:
            records.append(
                {
                    "entry_id": f"{method['key']}@{step:06d}",
                    "method_key": method["key"],
                    "method_label": method["label"],
                    "color": method["color"],
                    "step": step,
                    "checkpoint": str(checkpoint),
                    "result_root": str(result_root(output_root, method, step)),
                }
            )
    return {
        "schema_version": 1,
        "created_utc": timestamp(),
        "num_methods": len(config["methods"]),
        "num_checkpoints": len(records),
        "num_cases": 9,
        "entries": records,
    }


def build_page(config_path: Path) -> None:
    subprocess.run(
        [str(PYTHON), str(BUILD_PAGE), "--config", str(config_path)],
        check=True,
    )


def main() -> None:
    args = parse_args()
    config_path = args.config.expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    gpu = int(config["gpu"] if args.gpu is None else args.gpu)
    if gpu == 4:
        raise ValueError("GPU4 is prohibited by workspace rules")
    output_root = Path(config["output_root"]).expanduser().resolve()
    cases = case_ids(config)
    inventory = discover_inventory(config)
    inventory_path = output_root / "all_checkpoint_inventory.json"
    atomic_json(inventory_path, inventory)
    if args.inventory_only:
        build_page(config_path)
        print(inventory_path)
        return

    status_path = output_root / "all_checkpoint_runtime_status.json"
    logs = output_root / "logs" / "all_checkpoints_inference"
    logs.mkdir(parents=True, exist_ok=True)
    status = {
        "schema_version": 1,
        "updated_utc": timestamp(),
        "gpu": gpu,
        "state": "running",
        "num_checkpoints": inventory["num_checkpoints"],
        "num_cases": len(cases),
        "entries": {},
    }
    for entry in inventory["entries"]:
        done = completed_cases(Path(entry["result_root"]), cases)
        status["entries"][entry["entry_id"]] = {
            "state": "complete" if len(done) == len(cases) else "pending",
            "completed_cases": len(done),
            "total_cases": len(cases),
        }
    atomic_json(status_path, status)
    build_page(config_path)

    failures: list[str] = []
    for index, entry in enumerate(inventory["entries"], start=1):
        entry_id = entry["entry_id"]
        target = Path(entry["result_root"])
        done = completed_cases(target, cases)
        if len(done) == len(cases):
            print(
                f"[{timestamp()}] skip complete {index:02d}/{inventory['num_checkpoints']} "
                f"{entry_id} (9/9)",
                flush=True,
            )
            continue

        record = status["entries"][entry_id]
        record["state"] = "running"
        status["current_entry"] = entry_id
        status["updated_utc"] = timestamp()
        atomic_json(status_path, status)
        build_page(config_path)
        print(
            f"[{timestamp()}] start {index:02d}/{inventory['num_checkpoints']} "
            f"{entry_id} on GPU{gpu}",
            flush=True,
        )

        method_root = output_root / "inference" / entry["method_key"]
        trace_root = (
            output_root
            / "numeric_traces"
            / entry["method_key"]
            / result_name(int(entry["step"]))
        )
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "TEST_LIST": config["input_list"],
                "NUM_INFERENCE_STEPS": str(
                    config["inference"]["num_inference_steps"]
                ),
                "NEGATIVE_PROMPT": config["inference"]["negative_prompt"],
                "STEP_OUTPUT_DIR_NAME": result_name(int(entry["step"])),
                "TRACE_ROOT": str(trace_root),
                "SHARD_TAG": entry_id.replace("@", "_step"),
            }
        )
        log_path = logs / f"{entry_id.replace('@', '_step')}.log"
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.run(
                [
                    "bash",
                    str(RUN_INFER),
                    entry["checkpoint"],
                    str(gpu),
                    str(method_root),
                ],
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        done = completed_cases(target, cases)
        record["completed_cases"] = len(done)
        record["return_code"] = process.returncode
        if process.returncode == 0 and len(done) == len(cases):
            record["state"] = "complete"
            print(f"[{timestamp()}] finish {entry_id} (9/9)", flush=True)
        else:
            record["state"] = "failed"
            failures.append(entry_id)
            print(
                f"[{timestamp()}] failed {entry_id}: rc={process.returncode} "
                f"complete={len(done)}/9 log={log_path}",
                flush=True,
            )
        status["updated_utc"] = timestamp()
        atomic_json(status_path, status)
        build_page(config_path)

    status.pop("current_entry", None)
    status["updated_utc"] = timestamp()
    status["state"] = "failed" if failures else "complete"
    status["failed_entries"] = failures
    atomic_json(status_path, status)
    build_page(config_path)
    if failures:
        raise RuntimeError(f"Failed checkpoints: {', '.join(failures)}")
    print(
        f"[{timestamp()}] all {inventory['num_checkpoints']} checkpoints complete",
        flush=True,
    )


if __name__ == "__main__":
    main()
