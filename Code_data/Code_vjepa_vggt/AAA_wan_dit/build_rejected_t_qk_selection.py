#!/usr/bin/env python3
"""Select representative aggregate-T heads rejected by confidence filters."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


MODELS = ("wan_lora", "xssc", "physrvg")
ROLES = ("S", "T", "P", "C", "G")
MARGIN_THRESHOLD = 0.08
SUPPORT_THRESHOLD = 0.50


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=851)
    parser.add_argument("--per-group", type=int, default=2)
    return parser.parse_args()


def load_case(input_list: Path) -> tuple[str, str]:
    paths = [
        Path(line.strip()).expanduser().resolve()
        for line in input_list.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if len(paths) != 1:
        raise ValueError(f"expected exactly one input JSON, got {len(paths)}")
    if not paths[0].is_file():
        raise FileNotFoundError(paths[0])
    return paths[0].stem, str(paths[0])


def candidate_record(row: dict[str, str]) -> dict:
    scores = {role: float(row[f"score_{role}"]) for role in ROLES}
    ordered_roles = sorted(ROLES, key=lambda role: scores[role], reverse=True)
    margin = float(row["margin"])
    support = float(row["support"])
    if margin >= MARGIN_THRESHOLD and support >= SUPPORT_THRESHOLD:
        failure = "pass_reference"
    elif margin < MARGIN_THRESHOLD and support >= SUPPORT_THRESHOLD:
        failure = "margin_only"
    elif margin >= MARGIN_THRESHOLD and support < SUPPORT_THRESHOLD:
        failure = "support_only"
    else:
        failure = "margin_and_support"
    return {
        "model": row["model"],
        "block": int(row["block"]),
        "head": int(row["head"]),
        "aggregate_role": row["role"],
        "candidate_role": ordered_roles[0],
        "runner_up_role": ordered_roles[1],
        "score_T": scores["T"],
        "runner_up_score": scores[ordered_roles[1]],
        "margin": margin,
        "support": support,
        "support_ci95_low": float(row["support_ci95_low"]),
        "support_ci95_high": float(row["support_ci95_high"]),
        "valid_trajectory_samples": int(row["valid_trajectory_samples"]),
        "total_samples": int(row["total_samples"]),
        "failure": failure,
        **{f"score_{role}": scores[role] for role in ROLES},
    }


def select_group(
    records: list[dict],
    *,
    group: str,
    count: int,
) -> list[dict]:
    candidates = [record for record in records if record["failure"] == group]
    if group == "margin_only":
        candidates.sort(
            key=lambda record: (
                MARGIN_THRESHOLD - record["margin"],
                -record["support"],
                -record["score_T"],
            )
        )
    elif group == "support_only":
        candidates.sort(
            key=lambda record: (
                SUPPORT_THRESHOLD - record["support"],
                -record["margin"],
                -record["score_T"],
            )
        )
    elif group == "margin_and_support":
        candidates.sort(
            key=lambda record: (
                (MARGIN_THRESHOLD - record["margin"]) / MARGIN_THRESHOLD
                + (SUPPORT_THRESHOLD - record["support"]) / SUPPORT_THRESHOLD,
                -record["score_T"],
            )
        )
    else:
        raise ValueError(group)
    return candidates[:count]


def main() -> None:
    args = parse_args()
    report = args.report.expanduser().resolve()
    input_list = args.input_list.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    case, input_json = load_case(input_list)

    by_model: dict[str, list[dict]] = defaultdict(list)
    with report.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            record = candidate_record(row)
            if record["candidate_role"] == "T":
                by_model[record["model"]].append(record)

    selected_by_model: dict[str, list[dict]] = {}
    for model in MODELS:
        records = by_model[model]
        chosen: list[dict] = []
        for group in ("margin_only", "support_only", "margin_and_support"):
            for record in select_group(
                records, group=group, count=int(args.per_group)
            ):
                record = dict(record)
                record["selection_reason"] = f"{group}_boundary"
                chosen.append(record)

        used = {(record["block"], record["head"]) for record in chosen}
        high_score_rejected = sorted(
            (
                record
                for record in records
                if record["failure"] != "pass_reference"
                and (record["block"], record["head"]) not in used
            ),
            key=lambda record: (
                -record["score_T"],
                -record["margin"],
                -record["support"],
            ),
        )
        for record in high_score_rejected[: int(args.per_group)]:
            record = dict(record)
            record["selection_reason"] = "high_score_rejected"
            chosen.append(record)

        passed = sorted(
            (
                record
                for record in records
                if record["failure"] == "pass_reference"
            ),
            key=lambda record: (
                -record["score_T"],
                -record["margin"],
                -record["support"],
            ),
        )
        if not passed:
            raise RuntimeError(f"{model}: no stable T reference")
        reference = dict(passed[0])
        reference["selection_reason"] = "stable_reference"
        chosen.append(reference)
        selected_by_model[model] = chosen

    selection_samples = {}
    flat_rows = []
    for model, records in selected_by_model.items():
        roles = {}
        for index, record in enumerate(records, start=1):
            key = (
                f"{record['selection_reason']}_{index:02d}_"
                f"B{record['block']:02d}H{record['head']:02d}"
            )
            roles[key] = {
                **record,
                "representative_seed": int(args.seed),
            }
            flat_rows.append({"selection_key": key, **record})
        selection_samples[model] = {case: {"roles": roles}}

    selection = {
        "schema_version": 1,
        "source": str(report),
        "policy": (
            "Per model, select aggregate T candidates near each failed confidence "
            "boundary, additional highest-score rejected candidates, and one stable "
            "T reference. No cross-model intersection is applied."
        ),
        "thresholds": {
            "aggregate_margin": MARGIN_THRESHOLD,
            "aggregate_support": SUPPORT_THRESHOLD,
        },
        "representative_seed": int(args.seed),
        "case": case,
        "input_json": input_json,
        "samples": selection_samples,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "model",
        "selection_key",
        "selection_reason",
        "failure",
        "block",
        "head",
        "candidate_role",
        "aggregate_role",
        "runner_up_role",
        "score_T",
        "runner_up_score",
        "margin",
        "support",
        "support_ci95_low",
        "support_ci95_high",
        "valid_trajectory_samples",
        "total_samples",
        *(f"score_{role}" for role in ROLES),
    ]
    with (output / "heads.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(flat_rows)

    summary = {
        model: {
            "selected": len(records),
            "groups": {
                group: sum(
                    record["selection_reason"] == group for record in records
                )
                for group in (
                    "margin_only_boundary",
                    "support_only_boundary",
                    "margin_and_support_boundary",
                    "high_score_rejected",
                    "stable_reference",
                )
            },
        }
        for model, records in selected_by_model.items()
    }
    (output / "selection_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
