from __future__ import annotations

import csv
from typing import Any

from .generators import build_generator
from .schemas import CandidateRecord, CandidateScore, RunConfig
from .scorers import GeometryProxyScorer, JEPAPredictiveScorer, LatentMotionScorer
from .video_utils import ensure_dir, symlink_or_copy, to_jsonable, write_json, write_jsonl


def _normalize_score_table(rows: list[dict[str, Any]], score_names: list[str]) -> list[dict[str, float]]:
    normalized_rows: list[dict[str, float]] = []
    score_ranges: dict[str, tuple[float, float]] = {}
    for score_name in score_names:
        values = [float(row["raw_scores"][score_name]) for row in rows]
        score_ranges[score_name] = (min(values), max(values))
    for row in rows:
        normalized: dict[str, float] = {}
        for score_name in score_names:
            low, high = score_ranges[score_name]
            value = float(row["raw_scores"][score_name])
            if high <= low + 1e-8:
                normalized[score_name] = 0.5
            else:
                normalized[score_name] = (value - low) / (high - low)
        normalized_rows.append(normalized)
    return normalized_rows


def _write_ranking_csv(path: Path, ranking: list[CandidateScore]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "rank",
                "candidate_id",
                "weighted_total",
                "latent_motion_raw",
                "geometry_raw",
                "jepa_raw",
                "latent_motion_norm",
                "geometry_norm",
                "jepa_norm",
            ],
        )
        writer.writeheader()
        for rank, item in enumerate(ranking, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "candidate_id": item.candidate_id,
                    "weighted_total": f"{item.weighted_total:.6f}",
                    "latent_motion_raw": f"{item.raw_scores['latent_motion']:.6f}",
                    "geometry_raw": f"{item.raw_scores['geometry']:.6f}",
                    "jepa_raw": f"{item.raw_scores['jepa']:.6f}",
                    "latent_motion_norm": f"{item.normalized_scores['latent_motion']:.6f}",
                    "geometry_norm": f"{item.normalized_scores['geometry']:.6f}",
                    "jepa_norm": f"{item.normalized_scores['jepa']:.6f}",
                }
            )


def run_pipeline(config: RunConfig) -> dict[str, Any]:
    run_dir = ensure_dir(config.output_root / "runs" / config.run_name)
    tmp_dir = ensure_dir(config.tmp_root / config.run_name)
    candidates_dir = ensure_dir(run_dir / "candidates")
    scoring_dir = ensure_dir(run_dir / "scores")
    selected_dir = ensure_dir(run_dir / "selected")
    manifest_dir = ensure_dir(run_dir / "meta")
    ensure_dir(tmp_dir)

    write_json(
        manifest_dir / "run_config_resolved.json",
        {
            "run_name": config.run_name,
            "output_root": str(config.output_root),
            "tmp_root": str(config.tmp_root),
            "input": {
                "prompt": config.input.prompt,
                "context_video_path": str(config.input.context_video_path),
            },
            "generators": [to_jsonable(item) for item in config.generators],
            "scoring": {
                "weights": config.scoring.weights,
                "latent_motion": to_jsonable(config.scoring.latent_motion),
                "geometry": to_jsonable(config.scoring.geometry),
                "jepa": to_jsonable(config.scoring.jepa),
            },
        },
    )

    candidate_records: list[CandidateRecord] = []
    for generator_config in config.generators:
        if not generator_config.enabled:
            continue
        generator = build_generator(generator_config)
        generator_output_dir = ensure_dir(candidates_dir / generator_config.key)
        generated = generator.generate(
            input_spec=config.input,
            config=generator_config,
            output_dir=generator_output_dir,
        )
        candidate_records.extend(generated)

    if not candidate_records:
        raise RuntimeError("No candidate videos were generated.")

    write_jsonl(
        manifest_dir / "candidate_manifest.jsonl",
        [
            {
                "candidate_id": item.candidate_id,
                "generator_key": item.generator_key,
                "generator_type": item.generator_type,
                "seed": item.seed,
                "video_path": str(item.video_path),
                "used_context_frames": item.used_context_frames,
                "metadata": item.metadata,
            }
            for item in candidate_records
        ],
    )

    latent_motion_scorer = LatentMotionScorer(config.scoring.latent_motion)
    geometry_scorer = GeometryProxyScorer(config.scoring.geometry)
    jepa_scorer = JEPAPredictiveScorer(config.scoring.jepa)

    score_rows: list[dict[str, Any]] = []
    for candidate in candidate_records:
        latent_motion_score, latent_motion_details = latent_motion_scorer.score(candidate.video_path)
        geometry_score, geometry_details = geometry_scorer.score(
            context_video_path=config.input.context_video_path,
            candidate_video_path=candidate.video_path,
        )
        jepa_score, jepa_details = jepa_scorer.score(
            context_video_path=config.input.context_video_path,
            candidate_video_path=candidate.video_path,
        )
        score_rows.append(
            {
                "candidate": candidate,
                "raw_scores": {
                    "latent_motion": float(latent_motion_score),
                    "geometry": float(geometry_score),
                    "jepa": float(jepa_score),
                },
                "details": {
                    "latent_motion": latent_motion_details,
                    "geometry": geometry_details,
                    "jepa": jepa_details,
                },
            }
        )

    normalized_rows = _normalize_score_table(score_rows, ["latent_motion", "geometry", "jepa"])
    ranking: list[CandidateScore] = []
    for row, normalized in zip(score_rows, normalized_rows):
        weighted_total = 0.0
        for score_name, score_value in normalized.items():
            weighted_total += float(config.scoring.weights.get(score_name, 0.0)) * float(score_value)
        ranking.append(
            CandidateScore(
                candidate_id=row["candidate"].candidate_id,
                raw_scores=row["raw_scores"],
                normalized_scores=normalized,
                weighted_total=float(weighted_total),
                details=row["details"],
            )
        )

    ranking.sort(key=lambda item: item.weighted_total, reverse=True)
    best = ranking[0]
    best_candidate = next(item for item in candidate_records if item.candidate_id == best.candidate_id)
    symlink_or_copy(best_candidate.video_path, selected_dir / "best.mp4")

    write_json(
        selected_dir / "best.json",
        {
            "candidate_id": best.candidate_id,
            "source_video_path": str(best_candidate.video_path),
            "weighted_total": best.weighted_total,
            "raw_scores": best.raw_scores,
            "normalized_scores": best.normalized_scores,
            "details": best.details,
        },
    )
    write_jsonl(
        scoring_dir / "per_candidate_scores.jsonl",
        [
            {
                "candidate_id": item.candidate_id,
                "weighted_total": item.weighted_total,
                "raw_scores": item.raw_scores,
                "normalized_scores": item.normalized_scores,
                "details": item.details,
            }
            for item in ranking
        ],
    )
    _write_ranking_csv(scoring_dir / "ranking.csv", ranking)

    summary = {
        "run_name": config.run_name,
        "run_dir": str(run_dir),
        "tmp_dir": str(tmp_dir),
        "num_candidates": len(candidate_records),
        "best_candidate_id": best.candidate_id,
        "best_video_path": str(best_candidate.video_path),
        "selected_video_path": str(selected_dir / "best.mp4"),
        "weights": config.scoring.weights,
        "ranking": [
            {
                "candidate_id": item.candidate_id,
                "weighted_total": item.weighted_total,
                "raw_scores": item.raw_scores,
                "normalized_scores": item.normalized_scores,
            }
            for item in ranking
        ],
    }
    write_json(run_dir / "summary.json", summary)
    return summary
