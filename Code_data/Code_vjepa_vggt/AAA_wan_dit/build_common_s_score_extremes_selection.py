#!/usr/bin/env python3
"""Select cross-model common S heads at the extremes of mean score_S."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path


MODELS = ("wan_lora", "xssc", "physrvg")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--input-list", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--seed", type=int, default=851)
    return parser.parse_args()


def case_key(input_list: Path) -> str:
    lines = [
        line.strip()
        for line in input_list.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    if len(lines) != 1:
        raise ValueError(f"expected exactly one input case, found {len(lines)}")
    return Path(lines[0]).stem


def main() -> None:
    args = parse_args()
    if args.count <= 0:
        raise ValueError("--count must be positive")
    report = args.report.expanduser().resolve()
    input_list = args.input_list.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows_by_head: dict[tuple[int, int], dict[str, dict[str, str]]] = defaultdict(dict)
    with report.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["block"]), int(row["head"]))
            rows_by_head[key][row["model"]] = row

    common_s = []
    for (block, head), model_rows in rows_by_head.items():
        if set(model_rows) != set(MODELS):
            continue
        if any(model_rows[model]["role"] != "S" for model in MODELS):
            continue
        scores = [float(model_rows[model]["score_S"]) for model in MODELS]
        common_s.append(
            {
                "block": block,
                "head": head,
                "mean_score_S": sum(scores) / len(scores),
                "min_score_S": min(scores),
                "max_score_S": max(scores),
                "models": model_rows,
            }
        )
    common_s.sort(key=lambda item: (-item["mean_score_S"], item["block"], item["head"]))
    if len(common_s) < 2 * args.count:
        raise RuntimeError(
            f"found only {len(common_s)} common S heads for {2 * args.count} selections"
        )

    selected = [
        ("top", rank, item)
        for rank, item in enumerate(common_s[: args.count], start=1)
    ]
    selected.extend(
        ("bottom", rank, item)
        for rank, item in enumerate(reversed(common_s[-args.count:]), start=1)
    )
    case = case_key(input_list)
    samples: dict[str, dict] = {}
    for model in MODELS:
        roles = {}
        for group, rank, item in selected:
            row = item["models"][model]
            label = f"S_{group}{rank:02d}"
            roles[label] = {
                "block": item["block"],
                "head": item["head"],
                "aggregate_score": float(row["score_S"]),
                "cross_model_mean_score_S": item["mean_score_S"],
                "score_group": group,
                "group_rank": rank,
                "representative_seed": args.seed,
            }
        samples[model] = {case: {"roles": roles}}

    selection = {
        "source": str(report),
        "policy": (
            "same aggregate S role in wan_lora/xssc/physrvg; rank by cross-model "
            "mean score_S; select top and bottom extremes"
        ),
        "representative_seed": args.seed,
        "case": case,
        "common_s_count": len(common_s),
        "count_per_extreme": args.count,
        "samples": samples,
    }
    (output / "selection.json").write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    fieldnames = [
        "score_group",
        "group_rank",
        "block",
        "head",
        "mean_score_S",
        "min_score_S",
        "max_score_S",
    ]
    for model in MODELS:
        fieldnames.extend(
            (
                f"{model}_score_S",
                f"{model}_margin",
                f"{model}_support",
                f"{model}_support_ci95_low",
            )
        )
    with (output / "ranking.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for group, rank, item in selected:
            row = {
                "score_group": group,
                "group_rank": rank,
                "block": item["block"],
                "head": item["head"],
                "mean_score_S": item["mean_score_S"],
                "min_score_S": item["min_score_S"],
                "max_score_S": item["max_score_S"],
            }
            for model in MODELS:
                source = item["models"][model]
                for field in ("score_S", "margin", "support", "support_ci95_low"):
                    row[f"{model}_{field}"] = source[field]
            writer.writerow(row)

    print(f"selected {args.count} top + {args.count} bottom from {len(common_s)} common S heads")
    print(output / "selection.json")
    print(output / "ranking.csv")


if __name__ == "__main__":
    main()
