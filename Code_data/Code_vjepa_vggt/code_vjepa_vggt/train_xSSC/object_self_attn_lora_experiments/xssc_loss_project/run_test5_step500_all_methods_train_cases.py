#!/usr/bin/env python3
"""Run the nine fixed training cases for every test5-table step-500 method."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = ROOT / "test5_step500_all_methods_train_cases.json"
PYTHON = Path("/home/gaoya/miniconda3/envs/wan-cu128/bin/python")
RUN_INFER = ROOT.parent / "run_infer_from_experiment.sh"
BUILD_PAGE = ROOT / "build_test5_step500_all_methods_train_case_gallery.py"
RESULT_NAME = "step-000500_steps40_512x896_ctx08_49f"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--gpu", type=int, default=None)
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


def case_ids(config: dict) -> list[str]:
    manifest = json.loads(
        Path(config["cases_manifest"]).read_text(encoding="utf-8")
    )
    result = [str(case["case_id"]) for case in manifest["cases"]]
    if len(result) != 9 or len(result) != len(set(result)):
        raise ValueError(f"Expected nine unique cases, found {len(result)}")
    return result


def result_root(output_root: Path, method: dict) -> Path:
    reuse = method.get("reuse_result_root")
    if reuse:
        return Path(reuse).expanduser().resolve()
    return output_root / "inference" / method["key"] / RESULT_NAME


def completed_cases(root: Path, cases: list[str]) -> list[str]:
    return [
        case_id
        for case_id in cases
        if (root / f"{case_id}.mp4").is_file()
        and (root / f"{case_id}.mp4").stat().st_size > 0
        and (root / f"{case_id}.json").is_file()
        and (root / f"{case_id}.json").stat().st_size > 0
    ]


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
    methods = config["methods"]
    if len(methods) != 18:
        raise ValueError(f"Expected 18 methods, found {len(methods)}")
    cases = case_ids(config)
    output_root = Path(config["output_root"]).expanduser().resolve()
    logs = output_root / "logs" / "inference"
    status_path = output_root / "runtime_status.json"
    logs.mkdir(parents=True, exist_ok=True)

    status = {
        "schema_version": 1,
        "updated_utc": timestamp(),
        "gpu": gpu,
        "state": "running",
        "num_methods": len(methods),
        "num_cases": len(cases),
        "methods": {},
    }
    for method in methods:
        done = completed_cases(result_root(output_root, method), cases)
        status["methods"][method["key"]] = {
            "state": "complete" if len(done) == len(cases) else "pending",
            "completed_cases": len(done),
            "total_cases": len(cases),
        }
    atomic_json(status_path, status)
    build_page(config_path)

    failures: list[str] = []
    for index, method in enumerate(methods, start=1):
        key = str(method["key"])
        target = result_root(output_root, method)
        done = completed_cases(target, cases)
        if len(done) == len(cases):
            print(
                f"[{timestamp()}] skip complete {index:02d}/{len(methods)} "
                f"{key} ({len(done)}/{len(cases)})",
                flush=True,
            )
            continue

        checkpoint = Path(method["checkpoint"]).expanduser().resolve()
        if not (checkpoint / "checkpoint.safetensors").is_file():
            raise FileNotFoundError(checkpoint / "checkpoint.safetensors")
        status["updated_utc"] = timestamp()
        status["current_method"] = key
        status["methods"][key]["state"] = "running"
        atomic_json(status_path, status)
        build_page(config_path)
        print(
            f"[{timestamp()}] start {index:02d}/{len(methods)} {key} on GPU{gpu}",
            flush=True,
        )

        method_root = output_root / "inference" / key
        trace_root = output_root / "numeric_traces" / key / RESULT_NAME
        environment = os.environ.copy()
        environment.update(
            {
                "PYTHONNOUSERSITE": "1",
                "TEST_LIST": config["input_list"],
                "NUM_INFERENCE_STEPS": str(
                    config["inference"]["num_inference_steps"]
                ),
                "NEGATIVE_PROMPT": config["inference"]["negative_prompt"],
                "STEP_OUTPUT_DIR_NAME": RESULT_NAME,
                "TRACE_ROOT": str(trace_root),
                "SHARD_TAG": key,
            }
        )
        log_path = logs / f"{key}.log"
        with log_path.open("a", encoding="utf-8") as log_handle:
            process = subprocess.run(
                [
                    "bash",
                    str(RUN_INFER),
                    str(checkpoint),
                    str(gpu),
                    str(method_root),
                ],
                env=environment,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                check=False,
            )
        done = completed_cases(target, cases)
        record = status["methods"][key]
        record["completed_cases"] = len(done)
        record["return_code"] = process.returncode
        if process.returncode == 0 and len(done) == len(cases):
            record["state"] = "complete"
            print(f"[{timestamp()}] finish {key} (9/9)", flush=True)
        else:
            record["state"] = "failed"
            failures.append(key)
            print(
                f"[{timestamp()}] failed {key}: rc={process.returncode} "
                f"complete={len(done)}/{len(cases)} log={log_path}",
                flush=True,
            )
        status["updated_utc"] = timestamp()
        atomic_json(status_path, status)
        build_page(config_path)

    status.pop("current_method", None)
    status["updated_utc"] = timestamp()
    status["state"] = "failed" if failures else "complete"
    status["failed_methods"] = failures
    atomic_json(status_path, status)
    build_page(config_path)
    if failures:
        raise RuntimeError(f"Failed methods: {', '.join(failures)}")
    print(f"[{timestamp()}] all 18 step-500 methods complete", flush=True)


if __name__ == "__main__":
    main()
