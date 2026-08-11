#!/usr/bin/env python3
"""Audit existing temporal object-tube ablations before reusing them.

The audit is deliberately read-only with respect to experiment outputs.  It
produces a JSON inventory, a flat CSV, and a concise Markdown report under a
new output directory.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_ABLATION_ROOT = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326/attention_matrix_ablations_temporal_tube_v1"
)
DEFAULT_HEAD_SCOPES = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50/visual_samples/"
    "attention_zero_seed47326/pck_head_scopes_s039_latest3350.json"
)
DEFAULT_RANKING_SOURCE = Path(
    "/data/gaoya/agent-data/outputs/"
    "wan22_ti2v_legacy_firstlatent_physiciq67_pck50"
)
DEFAULT_METRICS_ROOT = Path(
    "/data/gaoya/agent-data/outputs/object_query_ablation_metrics"
)
DEFAULT_OUTPUT = Path(
    "/data/gaoya/agent-data/outputs/object_query_information_flow_redesign/"
    "latest3350_v1/stage0_inventory"
)
SPEC_PATH = Path(__file__).with_name("experiment_spec_latest3350.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ablation-root", type=Path, default=DEFAULT_ABLATION_ROOT)
    parser.add_argument("--head-scopes", type=Path, default=DEFAULT_HEAD_SCOPES)
    parser.add_argument("--ranking-source-root", type=Path, default=DEFAULT_RANKING_SOURCE)
    parser.add_argument("--metrics-root", type=Path, default=DEFAULT_METRICS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json_hash(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def atomic_json(path: Path, payload: Any) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def selected_pairs(manifest: dict[str, Any]) -> list[list[int]]:
    pairs = {
        (int(item["block"]), int(item["head"]))
        for item in manifest.get("selected_entries", [])
        if "block" in item and "head" in item
    }
    return [[block, head] for block, head in sorted(pairs)]


def expected_scope_pairs(head_payload: dict[str, Any]) -> dict[str, list[list[int]]]:
    scopes = head_payload.get("head_scopes", {})
    ranked_entries = head_payload.get("entries", [])
    result: dict[str, list[list[int]]] = {}
    for name, definition in scopes.items():
        if isinstance(definition, dict) and "rank_start" in definition:
            start = int(definition["rank_start"]) - 1
            end = int(definition["rank_end"])
            entries = ranked_entries[start:end]
        else:
            entries = definition
        pairs = []
        for item in entries:
            if isinstance(item, dict):
                pairs.append([int(item["block"]), int(item["head"])])
            else:
                pairs.append([int(item[0]), int(item[1])])
        result[str(name)] = sorted(pairs)
    return result


def ranking_cases(root: Path) -> set[str]:
    runs = root / "runs"
    if not runs.is_dir():
        return set()
    return {path.name for path in runs.iterdir() if path.is_dir()}


def metric_record_ids(root: Path) -> tuple[set[tuple[str, int, str]], list[str]]:
    ids: set[tuple[str, int, str]] = set()
    bad_reports: list[str] = []
    for path in root.rglob("report.json"):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            bad_reports.append(str(path))
            continue
        case = str(payload.get("case") or "")
        seed = int(payload.get("seed") or -1)
        for record in payload.get("records", []):
            record_id = record.get("id")
            if record_id:
                ids.add((case, seed, str(record_id)))
    return ids, bad_reports


def flow_time(mask_mode: str) -> tuple[str | None, str | None]:
    prefix = {
        "self": "M1",
        "incoming": "M2",
        "outgoing": "M3",
    }
    flow = next((value for key, value in prefix.items() if mask_mode.startswith(key)), None)
    if mask_mode.endswith("_only"):
        mode = "all_time"
    elif mask_mode.endswith("_same"):
        mode = "same"
    elif mask_mode.endswith("_future"):
        mode = "future"
    elif mask_mode.endswith("_past"):
        mode = "past"
    else:
        mode = None
    return flow, mode


def main() -> None:
    args = parse_args()
    if not args.ablation_root.is_dir():
        raise FileNotFoundError(args.ablation_root)
    for path in (args.head_scopes, SPEC_PATH):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    head_payload = json.loads(args.head_scopes.read_text(encoding="utf-8"))
    expected_pairs = expected_scope_pairs(head_payload)
    expected_hashes = {
        scope: stable_json_hash(pairs) for scope, pairs in expected_pairs.items()
    }
    source_cases = ranking_cases(args.ranking_source_root)
    metric_ids, bad_metric_reports = metric_record_ids(args.metrics_root)

    records: list[dict[str, Any]] = []
    parse_errors: list[dict[str, str]] = []
    for manifest_path in sorted(args.ablation_root.rglob("manifest.json")):
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            parse_errors.append({"path": str(manifest_path), "error": repr(exc)})
            continue
        variant_dir = manifest_path.parent
        complete_path = variant_dir / "complete.json"
        video_path = variant_dir / "generated.mp4"
        pairs = selected_pairs(manifest)
        pair_hash = stable_json_hash(pairs) if pairs else None
        head_scope = str(manifest.get("head_scope") or "")
        ranking_tag = str(manifest.get("ranking_tag") or "")
        mask_mode = str(manifest.get("mask_mode") or "")
        flow, time_mode = flow_time(mask_mode)
        case = str(manifest.get("case") or "")
        variant_id = str(manifest.get("variant_id") or variant_dir.name)
        expected = expected_pairs.get(head_scope)
        head_match = bool(expected is not None and pairs == expected)
        if head_scope == "all720" and len(pairs) == 720:
            head_match = True
        audit = manifest.get("audit") or {}
        call_counts = audit.get("model_call_counts") or {}
        full_40x2 = (
            len(manifest.get("denoising_steps") or []) == 40
            and set(manifest.get("cfg_branches") or [])
            == {"conditional", "unconditional"}
            and len(call_counts) == 40
            and all(int(value) == 2 for value in call_counts.values())
        )
        complete = complete_path.is_file() and video_path.is_file() and video_path.stat().st_size > 0
        rank_current = ranking_tag == "s039r3350" or head_scope == "all720"
        reasons = []
        if not complete:
            reasons.append("missing_complete_or_video")
        if not rank_current:
            reasons.append("not_latest3350")
        if not head_match:
            reasons.append("head_scope_mismatch_or_unverifiable")
        if not full_40x2:
            reasons.append("missing_full_40step_2cfg_audit")
        if "git_commit" not in manifest and "code_hash" not in manifest:
            reasons.append("missing_code_provenance")
        reusable = complete and rank_current and head_match and full_40x2
        records.append(
            {
                "case": case,
                "seed": manifest.get("seed"),
                "target_scope": manifest.get("target_scope"),
                "region": manifest.get("region"),
                "variant_id": variant_id,
                "flow": flow,
                "time_mode": time_mode,
                "mask_mode": mask_mode,
                "head_scope": head_scope,
                "ranking_tag": ranking_tag,
                "selected_head_count": len(pairs),
                "selected_head_hash": pair_hash,
                "expected_head_hash": expected_hashes.get(head_scope),
                "head_scope_matches_latest3350": head_match,
                "full_40steps_2cfg_audited": full_40x2,
                "complete": complete,
                "metric_record_found": (
                    case, int(manifest.get("seed") or -1), variant_id
                ) in metric_ids,
                "ranking_source_case_overlap": case in source_cases,
                "reusable_for_latest3350_pilot": reusable,
                "reusability_notes": reasons,
                "manifest_path": str(manifest_path),
                "complete_path": str(complete_path),
                "video_path": str(video_path),
            }
        )

    key_to_paths: dict[tuple[Any, ...], list[str]] = defaultdict(list)
    for record in records:
        key = (
            record["case"],
            record["seed"],
            record["target_scope"],
            record["region"],
            record["mask_mode"],
            record["flow"],
            record["time_mode"],
            record["selected_head_hash"],
        )
        key_to_paths[key].append(record["manifest_path"])
    duplicate_keys = [
        {"key": list(key), "paths": paths}
        for key, paths in key_to_paths.items()
        if len(paths) > 1
    ]

    summary = {
        "manifest_count": len(records),
        "complete_count": sum(row["complete"] for row in records),
        "latest3350_tag_count": sum(row["ranking_tag"] == "s039r3350" for row in records),
        "latest3350_reusable_count": sum(row["reusable_for_latest3350_pilot"] for row in records),
        "metric_record_count": sum(row["metric_record_found"] for row in records),
        "ranking_source_overlap_count": sum(row["ranking_source_case_overlap"] for row in records),
        "independent_of_ranking_source_count": sum(
            not row["ranking_source_case_overlap"] for row in records
        ),
        "case_count": len({row["case"] for row in records if row["case"]}),
        "seed_count": len({row["seed"] for row in records if row["seed"] is not None}),
        "variant_counts_by_ranking_tag": dict(Counter(row["ranking_tag"] for row in records)),
        "variant_counts_by_head_scope": dict(Counter(row["head_scope"] for row in records)),
        "variant_counts_by_flow": dict(Counter(str(row["flow"]) for row in records)),
        "variant_counts_by_time_mode": dict(Counter(str(row["time_mode"]) for row in records)),
    }
    payload = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_spec": str(SPEC_PATH),
        "experiment_spec_sha256": sha256_file(SPEC_PATH),
        "head_scopes": str(args.head_scopes),
        "head_scopes_sha256": sha256_file(args.head_scopes),
        "ablation_root": str(args.ablation_root),
        "metrics_root": str(args.metrics_root),
        "ranking_source_case_count": len(source_cases),
        "summary": summary,
        "parse_errors": parse_errors,
        "bad_metric_reports": bad_metric_reports,
        "duplicate_keys": duplicate_keys,
        "records": records,
    }
    atomic_json(args.output_dir / "inventory.json", payload)

    columns = list(records[0]) if records else []
    with (args.output_dir / "inventory.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in records:
            writer.writerow(
                {
                    key: json.dumps(value, ensure_ascii=False)
                    if isinstance(value, (list, dict))
                    else value
                    for key, value in row.items()
                }
            )

    lines = [
        "# Stage 0 Existing-output Inventory",
        "",
        f"Generated: `{payload['generated_at_utc']}`",
        "",
        "| item | count |",
        "|---|---:|",
        *[f"| {key} | {value} |" for key, value in summary.items() if not isinstance(value, dict)],
        "",
        "## Important audit conclusions",
        "",
        f"- Ranking construction contains `{len(source_cases)}` PhysicIQ cases.",
        f"- `{summary['ranking_source_overlap_count']}` manifests overlap ranking-source cases; "
        "these are exploratory and cannot be blind confirmation.",
        f"- `{summary['latest3350_reusable_count']}` manifests satisfy current head-scope, "
        "40-step/two-CFG, complete-video checks for pilot reuse.",
        "- Reuse still lacks exact code provenance whenever the manifest has no git/code hash; "
        "those records must not be mixed into a strict confirmatory run.",
        f"- `{len(duplicate_keys)}` duplicate experimental keys require review.",
        f"- `{len(parse_errors)}` manifests and `{len(bad_metric_reports)}` metric reports failed JSON parsing.",
        "",
        "See `inventory.json` for paths and per-variant reasons.",
    ]
    (args.output_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
