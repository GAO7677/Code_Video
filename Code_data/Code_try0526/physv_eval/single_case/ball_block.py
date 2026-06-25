from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..case_inputs import coerce_eval_case
from .common import emit_result, load_eval_case, result_record

DATA_DIR = Path("/data/gaoya/AAA_test_video/Dataset_physV/0526dp")
PDI_ROOT = Path("/home/gaoya/Code_Video/Code_data/Code_benchmark/PDI-Bench-main")
TMP_DIR = DATA_DIR / "tmp_eval"


def run_pdi(video_path: Path, output_dir: Path, *, caption: str = "ball") -> dict[str, Any] | None:
    stem = video_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir = output_dir / stem
    report_path = report_dir / f"{stem}_pdi_report.txt"

    if not report_path.exists():
        wrapper = TMP_DIR / "_pdi_wrapper.py"
        wrapper.parent.mkdir(parents=True, exist_ok=True)
        wrapper.write_text(
            "import sys, os\n"
            "try:\n"
            "    import flash_attn\n"
            "    from flash_attn.bert_padding import index_first_axis, pad_input, unpad_input\n"
            "    from flash_attn.layers.rotary import apply_rotary_emb\n"
            "    from flash_attn.flash_attn_interface import flash_attn_func\n"
            "except Exception:\n"
            "    pass\n"
            "sys.argv = [sys.argv[0], '--input', sys.argv[1], '--text', sys.argv[2], '--output_dir', sys.argv[3]]\n"
            "import runpy\n"
            "runpy.run_path('evaluation/main.py', run_name='__main__')\n",
            encoding="utf-8",
        )
        cmd = [sys.executable, "-u", str(wrapper), str(video_path), caption, str(output_dir)]
        tracker_ckpt = PDI_ROOT / "checkpoints/tracker/scaled_offline.pth"
        tracker_bak = PDI_ROOT / "checkpoints/tracker/scaled_offline.pth.bak"
        renamed_tracker_ckpt = False
        if tracker_ckpt.exists() and not tracker_bak.exists():
            tracker_ckpt.rename(tracker_bak)
            renamed_tracker_ckpt = True
        env = os.environ.copy()
        env["PYTHONPATH"] = str(PDI_ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        env["PDI_FLORENCE_MODEL_ID"] = "/data/gaoya/ckpt/microsoft-Florence-2-base"
        try:
            completed = subprocess.run(cmd, cwd=PDI_ROOT, env=env, capture_output=True, text=True)
        finally:
            if renamed_tracker_ckpt and tracker_bak.exists() and not tracker_ckpt.exists():
                tracker_bak.rename(tracker_ckpt)
        if completed.returncode != 0:
            return None

    text = report_path.read_text(encoding="utf-8")

    def extract(pattern: str, cast: type | None = None) -> Any:
        match = re.search(pattern, text)
        if not match:
            return None
        value = match.group(1)
        return cast(value) if cast else value

    return {
        "pdi_score": extract(r"FINAL PDI SCORE:\s*([0-9.]+)", float),
        "grade": extract(r"OVERALL GRADE:\s*([A-Z+-]+)"),
        "scale_error": extract(r"Scale Component .*?:\s*([0-9.]+)", float),
        "traj_error": extract(r"Trajectory Component .*?:\s*([0-9.]+)", float),
        "rigidity_error": extract(r"Epsilon Rigidity:\s*([0-9.]+)", float),
        "vp_error": extract(r"VP Component .*?:\s*([0-9.]+)", float),
    }


def run_jepa(video_path: Path) -> dict[str, Any] | None:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from rerank_video.scorers import JEPAPredictiveScorer
    from rerank_video.schemas import JEPAScoreConfig
    from rerank_video.video_utils import ensure_dir, load_video_frames, uniform_subsample_frames
    import cv2

    frames = load_video_frames(video_path)
    total = len(frames)
    if total < 30:
        return None
    split = min(60, total // 2)
    ctx = uniform_subsample_frames(frames[:split], 8)
    fut = uniform_subsample_frames(frames[split:], 16)

    tmp = ensure_dir(TMP_DIR / "jepa" / video_path.stem)
    ctx_p = tmp / "context.mp4"
    fut_p = tmp / "future.mp4"

    def write_video(path: Path, frs: list[Any]) -> None:
        height, width = frs[0].shape[:2]
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 16, (width, height))
        try:
            for frame in frs:
                writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        finally:
            writer.release()

    write_video(ctx_p, ctx)
    write_video(fut_p, fut)

    scorer = JEPAPredictiveScorer(
        JEPAScoreConfig(
            backend="vjepa2",
            device="cuda",
            max_frames=32,
            context_frames=8,
            future_frames=16,
            context_repeat_frames=8,
            crop_size=384,
            vjepa_checkpoint=Path("/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt"),
            vjepa_repo_root=Path("/home/gaoya/Code_Video/vjepa2-main"),
            vjepa_model_name="vjepa2_1_vit_large_384",
        )
    )
    score, _details = scorer.score(context_video_path=ctx_p, candidate_video_path=fut_p)
    return {"jepa_score": float(score)}


def score_case(
    case: Path | str | dict[str, Any],
    *,
    caption: str = "ball",
    output_dir: Path | None = None,
    skip_pdi: bool = False,
    skip_jepa: bool = False,
) -> dict[str, Any]:
    normalized = coerce_eval_case(case, caption=caption)
    result: dict[str, Any] = {"caption": normalized.caption or caption}
    if not skip_pdi:
        result["pdi"] = run_pdi(normalized.video_path, output_dir or (TMP_DIR / "pdi" / normalized.video_path.stem), caption=normalized.caption or caption)
    if not skip_jepa:
        result["jepa"] = run_jepa(normalized.video_path)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a single-case ball-block evaluation.")
    parser.add_argument("--input-json", type=Path, default=None)
    parser.add_argument("--video", type=Path, default=None)
    parser.add_argument("--caption", default="ball")
    parser.add_argument("--output-json", type=Path, default=None)
    parser.add_argument("--skip-pdi", action="store_true")
    parser.add_argument("--skip-jepa", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized = load_eval_case(input_json=args.input_json, video=args.video, caption=args.caption)
    case: Path | str | dict[str, Any] = args.input_json if args.input_json is not None else args.video
    if case is None:
        raise ValueError("--input-json or --video is required")
    result = score_case(case, caption=args.caption, skip_pdi=args.skip_pdi, skip_jepa=args.skip_jepa)
    emit_result(result_record(normalized, result), output_json=args.output_json)


if __name__ == "__main__":
    main()
