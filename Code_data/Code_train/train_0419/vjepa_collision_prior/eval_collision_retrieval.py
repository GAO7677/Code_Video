from __future__ import annotations

import argparse
from pathlib import Path

from lib.backends import build_backend
from lib.data_utils import read_jsonl, write_json, write_jsonl
from lib.metrics import summarize_ranking_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run collision future retrieval evaluation.")
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--backend",
        required=True,
        choices=["vjepa_predictor", "vjepa_context", "videomae", "dino", "clip", "state_extrap", "random"],
    )
    parser.add_argument("--pooling", default="global", choices=["global", "object", "object_pair"])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--crop-size", type=int, default=384)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--vjepa-checkpoint", default="/data/gaoya/ckpt/Sylvest-vjepa2-vit-g/vitg-384.pt")
    parser.add_argument("--videomae-model-id", default="")
    parser.add_argument("--dino-model-id", default="")
    parser.add_argument("--clip-model-id", default="")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest_rows = read_jsonl(args.manifest)
    if args.max_queries:
        manifest_rows = manifest_rows[: args.max_queries]

    backend = build_backend(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidate_cache = {}
    results = []
    for query_row in manifest_rows:
        query_state = backend.encode_query(query_row)
        scored = []
        for candidate in query_row["candidates"]:
            cache_key = (args.backend, candidate["candidate_id"])
            if cache_key not in candidate_cache:
                candidate_cache[cache_key] = backend.encode_candidate(candidate)
            candidate_state = candidate_cache[cache_key]
            score = backend.score(
                query_row=query_row,
                candidate=candidate,
                query_state=query_state,
                candidate_state=candidate_state,
                pooling=args.pooling,
            )
            scored.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "role": candidate["role"],
                    "negative_type": candidate["negative_type"],
                    "scene_id": candidate["scene_id"],
                    "score": score,
                }
            )

        scored = sorted(scored, key=lambda item: item["score"], reverse=True)
        gt_rank = next(idx for idx, item in enumerate(scored, start=1) if item["role"] == "positive")
        positive_score = next(item["score"] for item in scored if item["role"] == "positive")
        best_negative_score = max(item["score"] for item in scored if item["role"] != "positive")
        results.append(
            {
                "query_id": query_row["query_id"],
                "scene_id": query_row["scene_id"],
                "object_id": query_row["object_id"],
                "case_id": query_row["case_id"],
                "horizon": query_row["horizon"],
                "future_width": query_row["future_width"],
                "pooling": args.pooling,
                "backend": args.backend,
                "gt_rank": gt_rank,
                "positive_negative_margin": positive_score - best_negative_score,
                "ranked_candidates": scored,
            }
        )

    summary = summarize_ranking_results(results)
    summary["backend"] = args.backend
    summary["pooling"] = args.pooling
    summary["manifest"] = args.manifest

    write_json(output_dir / "summary.json", summary)
    write_jsonl(output_dir / "per_query.jsonl", results)
    print(f"Wrote summary to {output_dir / 'summary.json'}")
    print(f"Wrote per-query results to {output_dir / 'per_query.jsonl'}")


if __name__ == "__main__":
    main()
