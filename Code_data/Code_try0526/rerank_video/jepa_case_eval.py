from __future__ import annotations

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
from .video_utils import ensure_dir, write_json


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
              <td>{item.jepa_details.get('predictive_l2', 0.0):.4f}</td>
              <td>{item.jepa_details.get('continuity', 0.0):.4f}</td>
              <td>{item.jepa_details.get('temporal_smoothness', 0.0):.4f}</td>
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
    当前 JEPA context 使用重复首帧构造，因此这里更适合做相对重排实验，不适合作为最终绝对评测。<br />
    与官方 PDI 的 Spearman 相关： {corr if corr is not None else "-"}
  </div>
  <table>
    <thead>
      <tr>
        <th>Case</th>
        <th>方法</th>
        <th>JEPA 分数 ↑</th>
        <th>Predictive Align ↑</th>
        <th>Predictive L2 ↓</th>
        <th>Continuity ↑</th>
        <th>Temporal Smoothness ↑</th>
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
            score, details = scorer.score_from_anchor_image(
                anchor_image_path=first_frame_path,
                candidate_video_path=video_path,
            )
            results.append(
                EvalResult(
                    case_id=case.case_id,
                    provider=provider,
                    prompt=case.prompt,
                    target_object=case.target_object,
                    gt_video_path=gt_video_path,
                    first_frame_path=first_frame_path,
                    video_path=video_path,
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
