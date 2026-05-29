from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import yaml

from .paths import A_OUTPUT, PDI_FLORENCE_MODEL, PDI_ROOT, RUN_ROOT, SAM_PYTHON
from .records import load_payload, stable_path_id


_GT_PROMPT_INDEX: dict[str, str] | None = None


def parse_report(report_path: Path) -> dict[str, Any]:
    text = report_path.read_text(encoding="utf-8")

    def extract(pattern: str, cast: type | None = None) -> Any:
        match = re.search(pattern, text, re.MULTILINE)
        if not match:
            return None
        value = match.group(1).strip()
        return cast(value) if cast else value

    return {
        "pdi_score": extract(r"FINAL PDI SCORE:\s*([0-9.]+)", float),
        "grade": extract(r"OVERALL GRADE:\s*(.+)"),
        "scale_component": extract(r"Scale Component .*?:\s*([0-9.]+)", float),
        "traj_component": extract(r"Trajectory Component .*?:\s*([0-9.]+)", float),
        "epsilon_rigidity": extract(r"Epsilon Rigidity:\s*([0-9.]+)", float),
        "rigidity_strategy": extract(r"Rigidity Strategy:\s*(.+)"),
        "vp_component": extract(r"VP Component .*?:\s*([0-9.]+)", float),
        "ra_math_pass": extract(r"RA Math Pass:\s*(True|False)"),
        "ra_ground_rmse": extract(r"RA Ground RMSE:\s*([0-9.eE+-]+)", float),
        "ra_scale_jump": extract(r"RA Scale Jump:\s*([0-9.eE+-]+)", float),
        "ra_reproj_err": extract(r"RA Reproj Err:\s*([0-9.eE+-]+)", float),
        "ra_overall_pass": extract(r"RA Overall Pass:\s*(True|False)"),
        "raw_report_path": str(report_path),
    }


def _build_gt_prompt_index() -> dict[str, str]:
    global _GT_PROMPT_INDEX
    if _GT_PROMPT_INDEX is not None:
        return _GT_PROMPT_INDEX
    index: dict[str, str] = {}
    for json_path in sorted((A_OUTPUT / "GT").rglob("*.json")):
        payload = load_payload(json_path)
        clip_name = str(payload.get("clip_name") or json_path.stem)
        prompt = str(payload.get("prompt") or "")
        if prompt:
            index[clip_name] = prompt
    _GT_PROMPT_INDEX = index
    return index


def resolve_text_query(video_path: Path, payload: dict[str, Any]) -> str:
    for key in ("target_object", "prompt"):
        value = payload.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    stem = video_path.stem
    if "shuffle_test" in video_path.parts and stem.startswith("gt_"):
        base_name = stem[len("gt_") :]
        for suffix in ("_original", "_shuffled"):
            if base_name.endswith(suffix):
                base_name = base_name[: -len(suffix)]
        prompt = _build_gt_prompt_index().get(base_name)
        if prompt:
            return prompt
    return "ball"


def build_temp_config(cache_dir: Path) -> Path:
    with (PDI_ROOT / "configs" / "default.yaml").open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    config["cache_dir"] = str(cache_dir)
    config_path = cache_dir / "default_eval.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config_path


class OfficialPDIRunner:
    def __init__(
        self,
        *,
        python_bin: str | None = None,
        cuda_visible_devices: str | None = None,
        max_retries: int = 3,
    ) -> None:
        default_python = str(SAM_PYTHON) if SAM_PYTHON.is_file() else sys.executable
        self.python_bin = python_bin or default_python
        self.cuda_visible_devices = cuda_visible_devices
        self.max_retries = max(int(max_retries), 1)

    def run(self, video_path: Path, text_query: str, refresh: bool = False) -> dict[str, Any]:
        sample_id = stable_path_id(video_path)
        output_dir = RUN_ROOT / "pdi" / sample_id
        report_dir = output_dir / video_path.stem
        report_path = report_dir / f"{video_path.stem}_pdi_report.txt"
        cache_dir = RUN_ROOT / "pdi_cache" / sample_id
        if report_path.exists() and not refresh:
            return parse_report(report_path)
        if refresh and cache_dir.exists():
            shutil.rmtree(cache_dir)

        config_path = build_temp_config(cache_dir)

        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = str(PDI_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["PDI_FLORENCE_MODEL_ID"] = str(PDI_FLORENCE_MODEL)
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)

        cmd = [
            self.python_bin,
            "evaluation/main.py",
            "--input",
            str(video_path),
            "--config",
            str(config_path),
            "--text",
            text_query,
            "--output_dir",
            str(output_dir),
        ]
        last_error = ""
        for attempt in range(1, self.max_retries + 1):
            completed = subprocess.run(
                cmd,
                cwd=PDI_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            if completed.returncode == 0:
                return parse_report(report_path)

            stderr_tail = completed.stderr[-2000:] if completed.stderr else ""
            last_error = stderr_tail
            transient_cuda = "illegal memory access" in stderr_tail.lower() or "cublas" in stderr_tail.lower()
            if attempt >= self.max_retries or not transient_cuda:
                break
            time.sleep(2.0 * attempt)

        raise RuntimeError(f"Official PDI failed for {video_path.name}\n{last_error}")
