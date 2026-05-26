from __future__ import annotations

import cv2
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .pdi_proxy_eval import (
    PDICase,
    default_cases,
    ensure_gt_video,
    extract_first_frame_image,
)
from .scorers import JEPAPredictiveScorer
from .schemas import JEPAScoreConfig
from .video_utils import detect_video_fps, ensure_dir, load_video_frames, write_json


DEFAULT_OUTPUT_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/runs/pdi_jepa_eval_demo")
DEFAULT_TMP_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/tmp/pdi_proxy_eval_demo")
DEFAULT_GENERATED_ROOT = Path("/data/gaoya/AAA_test_video/Output_try0526/runs/pdi_proxy_eval_demo/generated")
DEFAULT_OFFICIAL_SUMMARY = Path("/data/gaoya/AAA_test_video/Output_try0526/runs/pdi_official_eval_demo/report/summary.json")
DEFAULT_VJEPA_REPO_ROOT = Path("/home/gaoya/Code_Video/vjepa2-main")
DEFAULT_VJEPA_CKPT = Path("/data/gaoya/ckpt/VJEPA2/vjepa2_1_vitl_dist_vitG_384.pt")


@dataclass
class EvalResult:
    case_id: str
    provider: str
    prompt: str
    target_object: str
    gt_video_path: Path
    first_frame_path: Path
    video_path: Path
    context_video_path: Path
    scored_future_video_path: Path
    jepa_score: float
    jepa_details: dict[str, Any]


def _rankdata(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_vals = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_vals[end] == sorted_vals[start]:
            end += 1
        avg_rank = (start + end - 1) / 2.0 + 1.0
        ranks[order[start:end]] = avg_rank
        start = end
    return ranks


def _spearmanr(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 2 or len(xs) != len(ys):
        return None
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    rx = _rankdata(x)
    ry = _rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt(np.sum(rx ** 2) * np.sum(ry ** 2)))
    if denom <= 1e-12:
        return None
    return float(np.sum(rx * ry) / denom)


def _load_official_index(summary_path: Path) -> dict[tuple[str, str], dict[str, Any]]:
    if not summary_path.is_file():
        return {}
    rows = json.loads(summary_path.read_text(encoding="utf-8"))
    return {(row["case_id"], row["provider"]): row for row in rows}


def _build_relations(
    results: list[EvalResult],
    official_index: dict[tuple[str, str], dict[str, Any]],
) -> dict[str, Any]:
    xs: list[float] = []
    ys: list[float] = []
    by_case: dict[str, dict[str, list[float]]] = {}
    for item in results:
        official = official_index.get((item.case_id, item.provider))
        if not official or official.get("pdi_score") is None:
            continue
        xs.append(float(item.jepa_score))
        ys.append(float(-official["pdi_score"]))
        slot = by_case.setdefault(item.case_id, {"jepa": [], "neg_pdi": []})
        slot["jepa"].append(float(item.jepa_score))
        slot["neg_pdi"].append(float(-official["pdi_score"]))
    return {
        "spearman_jepa_vs_neg_official_pdi": _spearmanr(xs, ys),
        "num_pairs": len(xs),
        "by_case": {
            case_id: {
                "spearman_jepa_vs_neg_official_pdi": _spearmanr(payload["jepa"], payload["neg_pdi"]),
                "num_pairs": len(payload["jepa"]),
            }
            for case_id, payload in by_case.items()
        },
    }


def _render_html(results: list[EvalResult], relations: dict[str, Any], output_path: Path) -> None:
    rows = []
    for item in results:
        rows.append(
            f"""
            <tr>
              <td>{item.case_id}</td>
              <td>{item.provider.upper()}</td>
              <td>{item.jepa_score:.4f}</td>
              <td>{item.jepa_details.get('predictive_alignment', 0.0):.4f}</td>
              <td>{item.jepa_details.get('temporal_relation_error', 0.0):.4f}</td>
              <td>{item.jepa_details.get('delta_l2', 0.0):.4f}</td>
              <td>{item.jepa_details.get('context_mode', '-')}</td>
            </tr>
            """
        )
    corr = relations.get("spearman_jepa_vs_neg_official_pdi")
    html = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>JEPA Case Eval</title>
  <style>
    body {{ font-family: Arial, sans-serif; padding: 24px; background: #f6f3ee; color: #1f1c18; }}
    .note {{ margin-bottom: 20px; line-height: 1.6; }}
    table {{ width: 100%; border-collapse: collapse; background: white; }}
    th, td {{ border: 1px solid #d8cdbd; padding: 8px 10px; text-align: center; }}
    th {{ background: #f2eadf; }}
  </style>
</head>
<body>
  <h1>JEPA Case Eval</h1>
  <div class="note">
    当前 JEPA demo 使用 GT 前缀 clip 作为共享 context，再比较各方法对应的 future clip。<br />
    与官方 PDI 的 Spearman 相关： {corr if corr is not None else "-"}
  </div>
  <table>
    <thead>
      <tr>
        <th>Case</th>
        <th>方法</th>
        <th>JEPA 分数 ↑</th>
        <th>Tok Cos ↑</th>
        <th>Rel Err ↓</th>
        <th>Delta Err ↓</th>
        <th>Context</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
    output_path.write_text(html, encoding="utf-8")


def _write_video_cv2(path: Path, frames: list[np.ndarray], fps: int) -> None:
    ensure_dir(path.parent)
    if not frames:
        raise ValueError(f"No frames to write for {path}")
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        max(int(fps), 1),
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open cv2.VideoWriter for {path}")
    try:
        for frame in frames:
            writer.write(cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR))
    finally:
        writer.release()


def _stage_context_and_future_clips(
    *,
    case_id: str,
    provider: str,
    gt_video_path: Path,
    candidate_video_path: Path,
    clip_root: Path,
    context_frames: int,
) -> tuple[Path, Path, dict[str, Any]]:
    gt_frames = load_video_frames(gt_video_path)
    candidate_frames = load_video_frames(candidate_video_path)
    if len(gt_frames) < context_frames + 2:
        raise ValueError(f"GT video too short for context split: {gt_video_path}")
    if len(candidate_frames) < context_frames + 2:
        raise ValueError(f"Candidate video too short for context split: {candidate_video_path}")

    context_clip_frames = gt_frames[:context_frames]
    future_clip_frames = candidate_frames[context_frames:]
    if len(future_clip_frames) < 2:
        raise ValueError(f"Future clip too short after trimming context prefix: {candidate_video_path}")

    case_root = ensure_dir(clip_root / case_id)
    context_clip_path = case_root / "context_gt_prefix.mp4"
    future_clip_path = case_root / f"{provider}_future.mp4"
    fps = detect_video_fps(candidate_video_path, fallback=16)
    if not context_clip_path.is_file():
        _write_video_cv2(context_clip_path, context_clip_frames, fps=fps)
    _write_video_cv2(future_clip_path, future_clip_frames, fps=fps)
    return context_clip_path, future_clip_path, {
        "context_mode": "gt_prefix_video",
        "context_source": "gt_prefix",
        "context_prefix_frames": int(context_frames),
        "future_trim_start_frame": int(context_frames),
    }


def run_eval(
    *,
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    tmp_root: Path = DEFAULT_TMP_ROOT,
    generated_root: Path = DEFAULT_GENERATED_ROOT,
    official_summary_path: Path = DEFAULT_OFFICIAL_SUMMARY,
    cases: list[PDICase] | None = None,
    device: str = "cuda",
) -> dict[str, Any]:
    cases = cases or default_cases()
    run_dir = ensure_dir(output_root)
    gt_root = ensure_dir(tmp_root / "gt_videos")
    ref_root = ensure_dir(run_dir / "reference")
    report_root = ensure_dir(run_dir / "report")
    clip_root = ensure_dir(tmp_root / "jepa_case_eval_clips")

    scorer = JEPAPredictiveScorer(
        JEPAScoreConfig(
            backend="vjepa2",
            device=device,
            max_frames=32,
            context_frames=8,
            future_frames=16,
            context_repeat_frames=8,
            crop_size=384,
            vjepa_checkpoint=DEFAULT_VJEPA_CKPT,
            vjepa_repo_root=DEFAULT_VJEPA_REPO_ROOT,
            vjepa_model_name="vjepa2_1_vit_large_384",
        )
    )

    results: list[EvalResult] = []
    for case in cases:
        gt_video_path = ensure_gt_video(case, gt_root)
        first_frame_path = extract_first_frame_image(gt_video_path, ref_root / case.case_id / "first_frame.png")
        providers = {
            "gt": gt_video_path,
            "wan": generated_root / case.case_id / "wan" / "wan.mp4",
            "vace": generated_root / case.case_id / "vace" / "vace.mp4",
        }
        for provider, video_path in providers.items():
            if not video_path.is_file():
                continue
            context_video_path, future_video_path, split_details = _stage_context_and_future_clips(
                case_id=case.case_id,
                provider=provider,
                gt_video_path=gt_video_path,
                candidate_video_path=video_path,
                clip_root=clip_root,
                context_frames=scorer.config.context_frames,
            )
            score, details = scorer.score(
                context_video_path=context_video_path,
                candidate_video_path=future_video_path,
            )
            details = {**split_details, **details}
            results.append(
                EvalResult(
                    case_id=case.case_id,
                    provider=provider,
                    prompt=case.prompt,
                    target_object=case.target_object,
                    gt_video_path=gt_video_path,
                    first_frame_path=first_frame_path,
                    video_path=video_path,
                    context_video_path=context_video_path,
                    scored_future_video_path=future_video_path,
                    jepa_score=float(score),
                    jepa_details=details,
                )
            )

    official_index = _load_official_index(official_summary_path)
    relations = _build_relations(results, official_index)
    payload = {
        "cases": [asdict(case) for case in cases],
        "results": [
            {
                **asdict(item),
                "gt_video_path": str(item.gt_video_path),
                "first_frame_path": str(item.first_frame_path),
                "video_path": str(item.video_path),
                "context_video_path": str(item.context_video_path),
                "scored_future_video_path": str(item.scored_future_video_path),
            }
            for item in results
        ],
        "relations": relations,
    }
    write_json(report_root / "summary.json", payload)
    _render_html(results, relations, report_root / "index.html")
    return {
        "run_dir": str(run_dir),
        "summary_path": str(report_root / "summary.json"),
        "html_path": str(report_root / "index.html"),
        "result_count": len(results),
        "relations": relations,
    }


if __name__ == "__main__":
    print(json.dumps(run_eval(), ensure_ascii=False, indent=2))
