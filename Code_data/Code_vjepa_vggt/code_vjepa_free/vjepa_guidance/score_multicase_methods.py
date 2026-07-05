#!/usr/bin/env python3
from __future__ import annotations

"""
Run command:
PYTHONPATH=/home/gaoya/Code_Video/Code_data/Code_try0526 \
CUDA_VISIBLE_DEVICES=6 /data/gaoya/miniconda3/envs/wan/bin/python \
/home/gaoya/Code_Video/Code_data/Code_vjepa_vggt/code_vjepa_free/vjepa_guidance/score_multicase_methods.py \
  --method-dir baseline=/data/gaoya/AAA_test_video/0626vjepa_free/vjepa_guidance/test/results/lora_test5/wan_openvid_0613pybullet_lorav2v_step000500_test5_vjepa_baseline \
  --method-dir ladder_s20=/data/gaoya/agent-data/outputs/vjepa_phase4_multicase/phase4_pilot3_ladder_s20 \
  --out-json /data/gaoya/agent-data/outputs/vjepa_phase4_multicase/phase4_pilot3_scores.json \
  --physics-iq --videophy2-task pc --videophy2-device cuda:0 --cosmos-reason1
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score multiple method directories across multiple cases using read-only physv_eval entry points.")
    parser.add_argument("--method-dir", action="append", required=True,
                        help="Method spec in the form label=/abs/path/to/generated_videos_dir.")
    parser.add_argument("--out-json", type=Path, required=True)
    parser.add_argument("--baseline-label", type=str, default="baseline")
    parser.add_argument("--skip-wmreward", action="store_true")
    parser.add_argument("--physics-iq", action="store_true")
    parser.add_argument("--videophy2-task", default=None, choices=["sa", "pc", "rule"])
    parser.add_argument("--videophy2-device", default="cuda")
    parser.add_argument("--videophy2-dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    parser.add_argument("--videophy2-num-frames", type=int, default=32)
    parser.add_argument("--cosmos-reason1", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=None)
    parser.add_argument("--save-every", type=int, default=1,
                        help="Write an incremental JSON snapshot every N scored rows. Set <=0 to only write once at the end.")
    return parser.parse_args()


def parse_method_dirs(specs: list[str]) -> list[tuple[str, Path]]:
    parsed: list[tuple[str, Path]] = []
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"Invalid --method-dir spec (expected label=path): {spec}")
        label, path_str = spec.split("=", 1)
        label = label.strip()
        path = Path(path_str).expanduser().resolve()
        if not label:
            raise ValueError(f"Empty method label in spec: {spec}")
        if not path.is_dir():
            raise FileNotFoundError(f"Method directory not found: {path}")
        parsed.append((label, path))
    return parsed


def load_case_sidecar(sidecar_path: Path) -> tuple[dict[str, Any], dict[str, Any] | None]:
    sidecar = json.loads(sidecar_path.read_text())
    input_json_path = sidecar.get("input_json")
    input_payload = None
    if isinstance(input_json_path, str) and input_json_path:
        input_json = Path(input_json_path).expanduser().resolve()
        if input_json.is_file():
            input_payload = json.loads(input_json.read_text())
            input_payload["_json_path"] = str(input_json)
    return sidecar, input_payload


def collect_records(method_dirs: list[tuple[str, Path]], limit_cases: int | None = None) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for method_label, method_dir in method_dirs:
        local_videos = {path.stem: path for path in sorted(method_dir.glob("*.mp4"))}
        sidecars = {
            path.stem: path
            for path in sorted(method_dir.glob("*.json"))
            if path.name != "batch_manifest.json"
        }
        case_ids = sorted(set(local_videos) | set(sidecars))
        if limit_cases is not None:
            case_ids = case_ids[:limit_cases]
        for case_id in case_ids:
            local_video_path = local_videos.get(case_id)
            sidecar_path = sidecars.get(case_id)
            sidecar, input_payload = ({}, None) if sidecar_path is None else load_case_sidecar(sidecar_path)
            resolved_video_path = None
            referenced_output = sidecar.get("output_video")
            if isinstance(referenced_output, str) and referenced_output:
                candidate = Path(referenced_output).expanduser().resolve()
                if candidate.is_file():
                    resolved_video_path = candidate
            if resolved_video_path is None:
                resolved_video_path = local_video_path
            if resolved_video_path is None:
                print(f"[warn] skip {method_label}/{case_id}: no local mp4 and output_video is missing or invalid", flush=True)
                continue
            source_video = None
            prompt = None
            input_json = sidecar.get("input_json")
            if input_payload is not None:
                source_video = input_payload.get("source_video") or input_payload.get("source_video_path")
                prompt = input_payload.get("input_caption")
            if prompt is None:
                prompt = sidecar.get("input_caption")
            records.append(
                {
                    "method": method_label,
                    "method_dir": str(method_dir),
                    "case_id": case_id,
                    "video_path": str(resolved_video_path),
                    "local_video_path": str(local_video_path) if local_video_path is not None else None,
                    "sidecar_path": str(sidecar_path) if sidecar_path is not None else None,
                    "input_json": input_json,
                    "source_video": source_video,
                    "prompt": prompt,
                    "referenced_output_video": sidecar.get("output_video"),
                }
            )
    return records


def mean_or_none(values: list[float | None]) -> float | None:
    filtered = [float(value) for value in values if value is not None]
    if not filtered:
        return None
    return sum(filtered) / len(filtered)


def build_output_payload(
    *,
    scored_rows: list[dict[str, Any]],
    method_dirs: list[tuple[str, Path]],
    baseline_label: str,
) -> dict[str, Any]:
    by_case: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    by_method: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in scored_rows:
        by_case[row["case_id"]][row["method"]] = row
        by_method[row["method"]].append(row)

    summary_by_method: dict[str, dict[str, Any]] = {}
    baseline_by_case = by_case
    for method, rows in sorted(by_method.items()):
        overlap_rows: list[dict[str, Any]] = []
        for row in rows:
            base = baseline_by_case.get(row["case_id"], {}).get(baseline_label)
            if base is not None and method != baseline_label:
                row["delta_surprise_vs_baseline"] = (
                    row.get("surprise") - base.get("surprise")
                    if row.get("surprise") is not None and base.get("surprise") is not None
                    else None
                )
                row["delta_similarity_vs_baseline"] = (
                    row.get("similarity") - base.get("similarity")
                    if row.get("similarity") is not None and base.get("similarity") is not None
                    else None
                )
                row["delta_physics_iq_vs_baseline"] = (
                    row.get("physics_iq_score") - base.get("physics_iq_score")
                    if row.get("physics_iq_score") is not None and base.get("physics_iq_score") is not None
                    else None
                )
                row["delta_videophy2_vs_baseline"] = (
                    row.get("videophy2_score") - base.get("videophy2_score")
                    if row.get("videophy2_score") is not None and base.get("videophy2_score") is not None
                    else None
                )
                row["delta_cosmos_reason1_vs_baseline"] = (
                    row.get("cosmos_reason1_score") - base.get("cosmos_reason1_score")
                    if row.get("cosmos_reason1_score") is not None and base.get("cosmos_reason1_score") is not None
                    else None
                )
                overlap_rows.append(row)
            elif method == baseline_label:
                row["delta_surprise_vs_baseline"] = 0.0 if row.get("surprise") is not None else None
                row["delta_similarity_vs_baseline"] = 0.0 if row.get("similarity") is not None else None
                row["delta_physics_iq_vs_baseline"] = 0.0 if row.get("physics_iq_score") is not None else None
                row["delta_videophy2_vs_baseline"] = 0.0 if row.get("videophy2_score") is not None else None
                row["delta_cosmos_reason1_vs_baseline"] = 0.0 if row.get("cosmos_reason1_score") is not None else None

        summary_by_method[method] = {
            "num_cases": len(rows),
            "num_overlap_with_baseline": len(overlap_rows) if method != baseline_label else len(rows),
            "mean_surprise": mean_or_none([row.get("surprise") for row in rows]),
            "mean_similarity": mean_or_none([row.get("similarity") for row in rows]),
            "mean_physics_iq": mean_or_none([row.get("physics_iq_score") for row in rows]),
            "mean_videophy2": mean_or_none([row.get("videophy2_score") for row in rows]),
            "mean_cosmos_reason1": mean_or_none([row.get("cosmos_reason1_score") for row in rows]),
            "mean_delta_surprise_vs_baseline": mean_or_none([row.get("delta_surprise_vs_baseline") for row in overlap_rows]),
            "mean_delta_similarity_vs_baseline": mean_or_none([row.get("delta_similarity_vs_baseline") for row in overlap_rows]),
            "mean_delta_physics_iq_vs_baseline": mean_or_none([row.get("delta_physics_iq_vs_baseline") for row in overlap_rows]),
            "mean_delta_videophy2_vs_baseline": mean_or_none([row.get("delta_videophy2_vs_baseline") for row in overlap_rows]),
            "mean_delta_cosmos_reason1_vs_baseline": mean_or_none([row.get("delta_cosmos_reason1_vs_baseline") for row in overlap_rows]),
        }

    ranking = sorted(
        summary_by_method.items(),
        key=lambda item: (
            item[1].get("mean_delta_surprise_vs_baseline")
            if item[1].get("mean_delta_surprise_vs_baseline") is not None
            else float("inf")
        ),
    )
    return {
        "baseline_label": baseline_label,
        "method_dirs": {label: str(path) for label, path in method_dirs},
        "rows": scored_rows,
        "summary_by_method": summary_by_method,
        "ranking_by_mean_delta_surprise": [
            {"method": method, **summary} for method, summary in ranking
        ],
    }


def write_output_snapshot(out_json: Path, payload: dict[str, Any]) -> None:
    out_json.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_json.with_suffix(out_json.suffix + ".tmp")
    tmp_path.write_text(json.dumps(payload, indent=2) + "\n")
    tmp_path.replace(out_json)


def main() -> None:
    args = parse_args()
    method_dirs = parse_method_dirs(args.method_dir)
    records = collect_records(method_dirs, limit_cases=args.limit_cases)
    if not records:
        raise SystemExit("No videos found in method directories")

    wm_runner = None
    physics_iq_score_case = None
    videophy2_score_case = None
    videophy2_runner = None
    cosmos_score_case = None
    cosmos_runner = None

    if not args.skip_wmreward:
        from physv_eval.wmreward_official import WMRewardRunner

        wm_runner = WMRewardRunner()
    if args.physics_iq:
        from physv_eval.single_case.physics_iq import score_case as physics_iq_score_case
    if args.videophy2_task is not None:
        from physv_eval.single_case.videophy2 import score_case as videophy2_score_case
        from physv_eval.videophy2_auto import VideoPhy2Runner

        videophy2_runner = VideoPhy2Runner(
            device=args.videophy2_device,
            dtype=args.videophy2_dtype,
            num_frames=args.videophy2_num_frames,
        )
    if args.cosmos_reason1:
        from physv_eval.single_case.cosmos_reason1 import score_case as cosmos_score_case
        from physv_eval.cosmos_reason1_official import OfficialCosmosReason1Runner

        cosmos_runner = OfficialCosmosReason1Runner()

    scored_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(records, start=1):
        video_path = Path(row["video_path"])
        out = dict(row)
        print(f"[{idx}/{len(records)}] {row['method']} :: {row['case_id']}", flush=True)
        if wm_runner is not None:
            wm = wm_runner.score(video_path)
            out["surprise"] = wm.get("surprise")
            out["similarity"] = wm.get("similarity")
        if physics_iq_score_case is not None and isinstance(row.get("source_video"), str):
            try:
                piq = physics_iq_score_case(str(video_path), source_video_path=str(row["source_video"]))
                out["physics_iq_score"] = piq.get("score") if piq else None
            except Exception as exc:
                out["physics_iq_score"] = None
                out["physics_iq_error"] = repr(exc)
        if videophy2_score_case is not None and videophy2_runner is not None:
            try:
                vp2 = videophy2_score_case(str(video_path), task=args.videophy2_task, runner=videophy2_runner)
                out["videophy2_task"] = args.videophy2_task
                out["videophy2_score"] = vp2.get("score") if vp2 else None
            except Exception as exc:
                out["videophy2_score"] = None
                out["videophy2_error"] = repr(exc)
        if cosmos_score_case is not None and cosmos_runner is not None:
            try:
                cosmos = cosmos_score_case(str(video_path), runner=cosmos_runner)
                out["cosmos_reason1_score"] = cosmos.get("score") if cosmos else None
            except Exception as exc:
                out["cosmos_reason1_score"] = None
                out["cosmos_reason1_error"] = repr(exc)
        scored_rows.append(out)
        if args.save_every > 0 and (idx % args.save_every == 0 or idx == len(records)):
            payload = build_output_payload(
                scored_rows=scored_rows,
                method_dirs=method_dirs,
                baseline_label=args.baseline_label,
            )
            write_output_snapshot(args.out_json, payload)

    output = build_output_payload(
        scored_rows=scored_rows,
        method_dirs=method_dirs,
        baseline_label=args.baseline_label,
    )
    write_output_snapshot(args.out_json, output)

    print("\nMethod summary:", flush=True)
    for summary_row in output["ranking_by_mean_delta_surprise"]:
        method = summary_row["method"]
        print(
            f"{method:18s} mean_surprise={summary_row.get('mean_surprise')} "
            f"mean_d_surprise={summary_row.get('mean_delta_surprise_vs_baseline')} "
            f"mean_physics_iq={summary_row.get('mean_physics_iq')} "
            f"mean_videophy2={summary_row.get('mean_videophy2')}",
            flush=True,
        )
    print(f"\nWrote {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
