from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from lib.data_utils import (
    SampleMeta,
    candidate_frame_range,
    collision_scene_ids,
    discover_source_catalog,
    grouped_scene_ids,
    is_valid_window,
    parse_scene_id,
    primary_collision_event,
    read_jsonl,
    rng_for_key,
    write_jsonl,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build collision-future retrieval manifest.")
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--eval-manifest", required=True, help="Genesis heldout manifest.jsonl to evaluate from.")
    parser.add_argument("--output", required=True)
    parser.add_argument("--context-length", type=int, default=8)
    parser.add_argument("--future-width", type=int, default=4)
    parser.add_argument("--horizons", type=str, default="2,4,8,12")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-queries", type=int, default=0)
    parser.add_argument("--no-context-left-pad", action="store_true")
    parser.add_argument("--include-environment-collisions", action="store_true")
    parser.add_argument("--num-random-negatives", type=int, default=4)
    parser.add_argument("--max-counterfactual-per-query", type=int, default=4)
    parser.add_argument("--max-no-collision-per-query", type=int, default=2)
    parser.add_argument("--counterfactual-caseids", type=str, default="")
    parser.add_argument(
        "--counterfactual-scene-pattern",
        type=str,
        default="{object_id}__case{caseid}*",
        help="Pattern kept explicit so future Genesis runs can be plugged in directly.",
    )
    return parser.parse_args()


def make_clip_record(
    *,
    candidate_id: str,
    role: str,
    negative_type: str | None,
    scene_id: str,
    sample_dir: str,
    frame_start: int,
    frame_width: int,
    participants: list[int],
    object_id: str,
    case_id: str,
    notes: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "role": role,
        "negative_type": negative_type,
        "scene_id": scene_id,
        "sample_dir": sample_dir,
        "frame_start": frame_start,
        "frame_width": frame_width,
        "frame_indices": candidate_frame_range(frame_start, frame_width),
        "participants": participants,
        "object_id": object_id,
        "case_id": case_id,
        "notes": notes or {},
    }


def counterfactual_scene_ids_for_query(
    query_meta: SampleMeta,
    grouped_ids: dict[str, dict[str, list[str]]],
    requested_caseids: list[str],
) -> tuple[list[str], list[str]]:
    scene_ids = grouped_ids.get(query_meta.object_id, {}).get("interaction_pair_plus_dynamic", [])
    found = []
    missing: list[str] = []
    if requested_caseids:
        for caseid in requested_caseids:
            matched = [
                scene_id
                for scene_id in scene_ids
                if parse_scene_id(scene_id)[1] == caseid and scene_id != query_meta.scene_id
            ]
            if matched:
                found.extend(matched)
            else:
                missing.append(caseid)
    else:
        found = [scene_id for scene_id in scene_ids if scene_id != query_meta.scene_id]
    return sorted(set(found)), missing


def main() -> None:
    args = parse_args()
    horizons = [int(item) for item in args.horizons.split(",") if item]
    if args.context_length % 2 != 0 or args.future_width % 2 != 0:
        raise ValueError("context length and future width should be even to align with tubelet_size=2 backbones.")

    catalog = discover_source_catalog(args.dataset_root)
    grouped_ids = grouped_scene_ids(catalog)
    requested_caseids = [item.strip() for item in args.counterfactual_caseids.split(",") if item.strip()]
    eval_rows = read_jsonl(args.eval_manifest)
    collision_scene_pool = collision_scene_ids(catalog, include_environment=args.include_environment_collisions)

    rows: list[dict[str, Any]] = []
    num_queries = 0
    for eval_row in eval_rows:
        source_dir = eval_row["source_paths"]["source_sample_dir"]
        scene_id = Path(source_dir).name
        query_meta = catalog.get(scene_id)
        if query_meta is None:
            continue

        primary_event = primary_collision_event(source_dir, include_environment=args.include_environment_collisions)
        if primary_event is None:
            continue

        participants = list(primary_event.get("object_indices", primary_event.get("participants", [])))[:2]
        collision_frame = int(primary_event["start_frame"])
        context_start = collision_frame - args.context_length
        if context_start < 0 and args.no_context_left_pad:
            continue
        context_indices = list(range(max(0, context_start), collision_frame))
        if context_start < 0:
            context_indices = ([0] * (-context_start)) + context_indices
        context_indices = context_indices[-args.context_length :]
        if len(context_indices) != args.context_length:
            continue

        counterfactual_ids, missing_caseids = counterfactual_scene_ids_for_query(
            query_meta,
            grouped_ids,
            requested_caseids=requested_caseids,
        )
        no_collision_ids = grouped_ids.get(query_meta.object_id, {}).get("single_object_preview", [])

        for horizon in horizons:
            positive_start = collision_frame + horizon
            if not is_valid_window(query_meta.frames, positive_start, args.future_width):
                continue

            query_id = f"{eval_row['sample_id']}__h{horizon:02d}"
            rng = rng_for_key(args.seed, query_id)
            candidates: list[dict[str, Any]] = []

            candidates.append(
                make_clip_record(
                    candidate_id=f"{query_id}__positive",
                    role="positive",
                    negative_type=None,
                    scene_id=query_meta.scene_id,
                    sample_dir=query_meta.source_dir,
                    frame_start=positive_start,
                    frame_width=args.future_width,
                    participants=participants,
                    object_id=query_meta.object_id,
                    case_id=query_meta.case_id,
                )
            )

            pre_start = collision_frame - args.future_width
            if is_valid_window(query_meta.frames, pre_start, args.future_width):
                candidates.append(
                    make_clip_record(
                        candidate_id=f"{query_id}__neg_same_video_pre",
                        role="negative",
                        negative_type="same_video_wrong_time",
                        scene_id=query_meta.scene_id,
                        sample_dir=query_meta.source_dir,
                        frame_start=pre_start,
                        frame_width=args.future_width,
                        participants=participants,
                        object_id=query_meta.object_id,
                        case_id=query_meta.case_id,
                    )
                )

            for alt_horizon in horizons:
                if alt_horizon == horizon:
                    continue
                alt_start = collision_frame + alt_horizon
                if is_valid_window(query_meta.frames, alt_start, args.future_width):
                    candidates.append(
                        make_clip_record(
                            candidate_id=f"{query_id}__neg_same_video_h{alt_horizon:02d}",
                            role="negative",
                            negative_type="same_video_wrong_time",
                            scene_id=query_meta.scene_id,
                            sample_dir=query_meta.source_dir,
                            frame_start=alt_start,
                            frame_width=args.future_width,
                            participants=participants,
                            object_id=query_meta.object_id,
                            case_id=query_meta.case_id,
                        )
                    )

            for idx, scene_id_cf in enumerate(counterfactual_ids[: args.max_counterfactual_per_query]):
                meta_cf = catalog[scene_id_cf]
                event_cf = primary_collision_event(meta_cf.source_dir, include_environment=args.include_environment_collisions)
                if event_cf is None:
                    continue
                cf_start = int(event_cf["start_frame"]) + horizon
                if not is_valid_window(meta_cf.frames, cf_start, args.future_width):
                    continue
                cf_participants = list(event_cf.get("object_indices", event_cf.get("participants", [])))[:2]
                candidates.append(
                    make_clip_record(
                        candidate_id=f"{query_id}__neg_counterfactual_{idx:02d}",
                        role="negative",
                        negative_type="same_scene_counterfactual",
                        scene_id=meta_cf.scene_id,
                        sample_dir=meta_cf.source_dir,
                        frame_start=cf_start,
                        frame_width=args.future_width,
                        participants=cf_participants,
                        object_id=meta_cf.object_id,
                        case_id=meta_cf.case_id,
                        notes={"missing_requested_caseids": missing_caseids},
                    )
                )

            for idx, scene_id_nc in enumerate(no_collision_ids[: args.max_no_collision_per_query]):
                meta_nc = catalog[scene_id_nc]
                nc_start = min(max(0, positive_start), meta_nc.frames - args.future_width)
                if not is_valid_window(meta_nc.frames, nc_start, args.future_width):
                    continue
                candidates.append(
                    make_clip_record(
                        candidate_id=f"{query_id}__neg_no_collision_{idx:02d}",
                        role="negative",
                        negative_type="no_collision",
                        scene_id=meta_nc.scene_id,
                        sample_dir=meta_nc.source_dir,
                        frame_start=nc_start,
                        frame_width=args.future_width,
                        participants=[0, 1] if meta_nc.num_objects > 1 else [0],
                        object_id=meta_nc.object_id,
                        case_id=meta_nc.case_id,
                    )
                )

            random_pool = [
                scene_id_rand
                for scene_id_rand in collision_scene_pool
                if scene_id_rand != query_meta.scene_id and scene_id_rand not in {item["scene_id"] for item in candidates}
            ]
            rng.shuffle(random_pool)
            for idx, scene_id_rand in enumerate(random_pool[: args.num_random_negatives]):
                meta_rand = catalog[scene_id_rand]
                event_rand = primary_collision_event(meta_rand.source_dir, include_environment=args.include_environment_collisions)
                if event_rand is None:
                    continue
                rand_start = int(event_rand["start_frame"]) + horizon
                if not is_valid_window(meta_rand.frames, rand_start, args.future_width):
                    continue
                rand_participants = list(event_rand.get("object_indices", event_rand.get("participants", [])))[:2]
                candidates.append(
                    make_clip_record(
                        candidate_id=f"{query_id}__neg_random_{idx:02d}",
                        role="negative",
                        negative_type="random",
                        scene_id=meta_rand.scene_id,
                        sample_dir=meta_rand.source_dir,
                        frame_start=rand_start,
                        frame_width=args.future_width,
                        participants=rand_participants,
                        object_id=meta_rand.object_id,
                        case_id=meta_rand.case_id,
                    )
                )

            row = {
                "query_id": query_id,
                "sample_id": eval_row["sample_id"],
                "scene_id": query_meta.scene_id,
                "source_sample_dir": query_meta.source_dir,
                "object_id": query_meta.object_id,
                "case_id": query_meta.case_id,
                "fps": query_meta.fps,
                "collision_frame": collision_frame,
                "participants": participants,
                "context": {
                    "sample_dir": query_meta.source_dir,
                    "frame_start": context_start,
                    "frame_width": args.context_length,
                    "frame_indices": context_indices,
                    "pad_left": max(0, -context_start),
                },
                "positive_start": positive_start,
                "horizon": horizon,
                "future_width": args.future_width,
                "counterfactual_missing_caseids": missing_caseids,
                "candidates": candidates,
            }
            rows.append(row)
            num_queries += 1
            if args.max_queries and num_queries >= args.max_queries:
                break
        if args.max_queries and num_queries >= args.max_queries:
            break

    output_path = Path(args.output)
    write_jsonl(output_path, rows)
    print(f"Wrote {len(rows)} query rows to {output_path}")


if __name__ == "__main__":
    main()
