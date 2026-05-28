from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from .paths import A_OUTPUT, PDI_FLORENCE_MODEL, PDI_ROOT, RUN_ROOT
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


class OfficialPDIRunner:
    def __init__(
        self,
        *,
        python_bin: str | None = None,
        cuda_visible_devices: str | None = None,
    ) -> None:
        self.python_bin = python_bin or sys.executable
        self.cuda_visible_devices = cuda_visible_devices

    def run(self, video_path: Path, text_query: str, refresh: bool = False) -> dict[str, Any]:
        sample_id = stable_path_id(video_path)
        output_dir = RUN_ROOT / "pdi" / sample_id
        report_dir = output_dir / video_path.stem
        report_path = report_dir / f"{video_path.stem}_pdi_report.txt"
        if report_path.exists() and not refresh:
            return parse_report(report_path)

        env = os.environ.copy()
        env["PYTHONNOUSERSITE"] = "1"
        env["PYTHONPATH"] = str(PDI_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["PDI_FLORENCE_MODEL_ID"] = str(PDI_FLORENCE_MODEL)
        if self.cuda_visible_devices is not None:
            env["CUDA_VISIBLE_DEVICES"] = str(self.cuda_visible_devices)

        tracker_ckpt = PDI_ROOT / "checkpoints" / "tracker" / "scaled_offline.pth"
        tracker_bak = tracker_ckpt.with_suffix(".pth.bak")
        renamed_tracker = False
        if tracker_ckpt.exists() and not tracker_bak.exists():
            tracker_ckpt.rename(tracker_bak)
            renamed_tracker = True

        cmd = [
            self.python_bin,
            "evaluation/main.py",
            "--input",
            str(video_path),
            "--text",
            text_query,
            "--output_dir",
            str(output_dir),
        ]
        try:
            completed = subprocess.run(
                cmd,
                cwd=PDI_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
        finally:
            if renamed_tracker and tracker_bak.exists() and not tracker_ckpt.exists():
                tracker_bak.rename(tracker_ckpt)

        if completed.returncode != 0:
            stderr_tail = completed.stderr[-2000:] if completed.stderr else ""
            raise RuntimeError(f"Official PDI failed for {video_path.name}\n{stderr_tail}")
        return parse_report(report_path)
